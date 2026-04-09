import math

import torch as th

import omnigibson as og
from omnigibson.object_states.object_state_base import AbsoluteObjectState
from omnigibson.utils.python_utils import classproperty


class TensorizedValueState(AbsoluteObjectState):
    """
    A state-mixin that implements optimized global value updates across all object state instances
    of this type, i.e.: all values across all object state instances are updated at once, rather than per
    individual instance update() call.

    Multi-scene layout
    ------------------
    VALUES      (S, N, *value_shape)  — S = number of scenes, N = number of unique object types
    OBJ_IDXS    {relative_prim_path: int}  — shared N index; same for the same object type in every scene
    IDX_OBJS    list[list[obj|None]]  — IDX_OBJS[scene_idx][obj_idx] = object instance

    Registration and tensor allocation are handled exclusively by ``initialize_view()``,
    which is called from ``simulator.py`` after scene changes.
    """

    # Tensor of raw internally tracked values — GPU-resident during computation
    # Shape is (S, N, ...), where S = scenes, N = object types, ... = value_shape
    VALUES = None

    # Pinned CPU mirror of VALUES — updated via async DMA after each global_update pass.
    # _get_value() always reads from here to avoid GPU stalls for Python callers.
    VALUES_CPU = None

    # Dictionary mapping relative prim path to index in the N dimension of VALUES
    OBJ_IDXS = None

    # 2-D list: IDX_OBJS[scene_idx][obj_idx] = object instance (or None if absent in that scene)
    IDX_OBJS = None

    # Int representing per-object state size
    STATE_SIZE = None

    @classmethod
    def global_initialize(cls):
        """Initialize the global class-level tensors and indices."""
        # Shape (0, 0, *value_shape) — zero scenes, zero object types; GPU-resident
        cls.VALUES = th.empty(0, dtype=cls.value_type, device="cuda").reshape(0, 0, *cls.value_shape)
        cls.VALUES_CPU = th.empty(0, dtype=cls.value_type).pin_memory().reshape(0, 0, *cls.value_shape)
        cls.OBJ_IDXS = {}  # {relative_prim_path: int}
        cls.IDX_OBJS = []  # list[list[obj|None]]

        # Compute and cache state size
        # This is the flattened size of @self.value_shape
        cls.STATE_SIZE = math.prod(cls.value_shape)

    @classmethod
    def initialize_view(cls):
        """
        Rebuild all class-level tensors by scanning current objects across all scenes.
        Called from ``simulator.py`` after scene changes. Child classes should call
        ``super().initialize_view()`` first, then extend with their own initialization.

        Carry-over: values for objects whose relative_prim_path still exists are preserved;
        new slots are initialized to zero (subclasses override to set correct defaults).
        """
        # Snapshot for carry-over (OBJ_IDXS / VALUES are None on the very first call)
        prev_obj_idxs = dict(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else {}
        prev_values = cls.VALUES.clone() if cls.VALUES is not None and cls.VALUES.numel() > 0 else None

        # Reset
        cls.global_initialize()

        # Scan all scenes and register objects that have this state
        for scene_idx, scene in enumerate(og.sim.scenes):
            for obj in scene.objects:
                if cls not in obj.states:
                    continue
                rel_path = obj.relative_prim_path

                # Extend scene dimension if this is a new scene index
                while len(cls.IDX_OBJS) <= scene_idx:
                    obj_idx = len(cls.OBJ_IDXS)
                    cls.IDX_OBJS.append([None] * obj_idx)
                    cls.VALUES = th.cat(
                        [cls.VALUES, th.zeros((1, obj_idx, *cls.value_shape), dtype=cls.value_type, device="cuda")],
                        dim=0,
                    )

                # Register new relative path if first seen across all scenes
                if rel_path not in cls.OBJ_IDXS:
                    obj_idx = len(cls.OBJ_IDXS)
                    cls.OBJ_IDXS[rel_path] = obj_idx
                    for s_row in cls.IDX_OBJS:
                        s_row.append(None)
                    cls.VALUES = th.cat(
                        [
                            cls.VALUES,
                            th.zeros((len(cls.IDX_OBJS), 1, *cls.value_shape), dtype=cls.value_type, device="cuda"),
                        ],
                        dim=1,
                    )

                cls.IDX_OBJS[scene_idx][cls.OBJ_IDXS[rel_path]] = obj

        # Carry over values for surviving objects (same relative path present before and after)
        if prev_values is not None and cls.VALUES.numel() > 0:
            for rel_path, obj_idx_old in prev_obj_idxs.items():
                if rel_path not in cls.OBJ_IDXS:
                    continue
                obj_idx_new = cls.OBJ_IDXS[rel_path]
                for s_idx in range(min(prev_values.shape[0], len(cls.IDX_OBJS))):
                    if cls.IDX_OBJS[s_idx][obj_idx_new] is not None:
                        cls.VALUES[s_idx, obj_idx_new] = prev_values[s_idx, obj_idx_old]

        # Rebuild pinned CPU mirror — synchronous copy so _get_value() is valid before first async copy
        cls.VALUES_CPU = th.zeros(cls.VALUES.shape, dtype=cls.value_type).pin_memory()
        if cls.VALUES.numel() > 0:
            cls.VALUES_CPU.copy_(cls.VALUES)

    @classmethod
    def global_update(cls):
        """
        Globally update all values. Calls _update_values(), detects changed entries, and fires
        state_updated() for affected objects. Skips if there are no tracked objects.
        """
        if cls.VALUES is None or cls.VALUES.numel() == 0:
            return
        S, N = cls.VALUES.shape[:2]
        if S == 0 or N == 0:
            return

        new_values = cls._update_values(values=cls.VALUES)

        # Compare with previous values, and add any changed objects to the scene-tracked set.
        # changed_mask shape: (S, N) — collapse any trailing value_shape dimensions if present.
        diff = new_values != cls.VALUES
        changed_mask = th.any(diff, dim=tuple(range(2, diff.ndim))) if diff.ndim > 2 else diff
        for s_idx in range(S):
            for obj_idx in th.where(changed_mask[s_idx])[0].tolist():
                obj = cls.IDX_OBJS[s_idx][obj_idx]
                if obj is not None:
                    obj.state_updated()

        cls.VALUES = new_values

        # Async GPU → CPU copy on the shared stream; simulator calls GPU_TO_CPU.synchronize()
        # after the Pass 1 loop before any VALUES_CPU reads in Pass 2.
        with th.cuda.stream(og.sim.GPU_TO_CPU):
            cls.VALUES_CPU.copy_(cls.VALUES, non_blocking=True)

    @classmethod
    def _update_values(cls, values):
        """
        Updates all internally tracked @values for this object state. Should be implemented by subclass.

        Args:
            values (th.tensor): Tensorized value array of shape (S, N, *value_shape)

        Returns:
            th.tensor: Updated tensorized value array of same shape
        """
        raise NotImplementedError

    @classproperty
    def value_shape(cls):
        """
        Returns:
            tuple: Expected shape of the per-object state instance value. Default is () (scalar).
        """
        return ()

    @classproperty
    def value_type(cls):
        """
        Returns:
            type: Type of the internal value array, e.g., bool, th.uint, th.float32, etc. Default is th.float32
        """
        return th.float32

    @classproperty
    def value_name(cls):
        """
        Returns:
            str: Name of the value key to assign when dumping / loading the state. Should be implemented by subclass
        """
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        # Run super first; registration in OBJ_IDXS / IDX_OBJS / VALUES is done by initialize_view()
        super().__init__(*args, **kwargs)

    def remove(self):
        # No-op for tensor cleanup — initialize_view() will rebuild from scratch on next call
        pass

    def _get_value(self):
        # Read from the pinned CPU mirror — no GPU stall for Python callers
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        val = self.VALUES_CPU[s, obj_idx].to(self.value_type)
        if isinstance(val, th.Tensor) and val.numel() == 1:
            val = val.item()
        return val

    def _set_value(self, new_value):
        # Write to both GPU (VALUES) and CPU mirror (VALUES_CPU) synchronously.
        # Safe: setters are never called from within _update_values.
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        self.VALUES[s, obj_idx] = new_value
        self.VALUES_CPU[s, obj_idx] = new_value
        return True

    @property
    def state_size(self):
        # This is merely the class state size
        return self.STATE_SIZE

    # For this state, we simply store its value.
    def _dump_state(self):
        return {self.value_name: self._get_value()}

    def _load_state(self, state):
        self._set_value(state[self.value_name])

    def serialize(self, state):
        # If the state value is not an iterable, wrap it in a numpy array
        val = (
            state[self.value_name]
            if isinstance(state[self.value_name], th.Tensor)
            else th.tensor([state[self.value_name]])
        ).float()
        return val.flatten()

    def deserialize(self, state):
        value_length = int(math.prod(self.value_shape))
        value = state[:value_length].reshape(self.value_shape)
        return {self.value_name: value}, value_length

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("TensorizedValueState")
        return classes
