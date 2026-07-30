import math

import torch as th

import omnigibson as og
from omnigibson.object_states.object_state_base import RelativeObjectState
from omnigibson.object_states.tensorized_state import TensorizedState, _wp_from_torch
from omnigibson.utils.python_utils import classproperty


class TensorizedObjectSystemState(TensorizedState, RelativeObjectState):
    """
    Tensorized states indexed by (scene, object, particle-system).

    Sibling of ``TensorizedRelativeState``, but the second index is a particle system rather than a
    second object.

    VALUES      (S, N_obj, N_sys, *value_shape)  — [scene, object, system, ...]
    OBJ_IDXS    {relative_prim_path: int}         — object index (objects that have this state)
    IDX_OBJS    list[list[obj|None]]              — IDX_OBJS[scene_idx][obj_idx] = object instance
    SYS_IDXS    {system_name: int}                — system index (physical + visual particle systems)
    """

    # {system_name: int} index into the N_sys dimension.
    SYS_IDXS = None

    @classmethod
    def global_initialize(cls):
        """Initialize the class-level tensors and indices for the (S, N_obj, N_sys, ...) shape."""
        cls.VALUES = th.empty(0, dtype=cls.value_type, device="cuda").reshape(0, 0, 0, *cls.value_shape)
        cls.VALUES_CPU = th.empty(0, dtype=cls.value_type).pin_memory().reshape(0, 0, 0, *cls.value_shape)
        cls.PREV_VALUES = th.empty(0, dtype=cls.value_type).reshape(0, 0, 0, *cls.value_shape)
        cls.OBJ_IDXS = {}  # {relative_prim_path: int}
        cls.IDX_OBJS = []  # list[list[obj|None]]
        cls.SYS_IDXS = {}  # {system_name: int}
        cls.STATE_SIZE = math.prod(cls.value_shape)

    @classmethod
    def initialize_view(cls):
        """
        Rebuild all class-level tensors by scanning current objects/systems across all scenes.
        Called from ``simulator.py`` after scene changes.
        """
        cls.global_initialize()

        for scene_idx, scene in enumerate(og.sim.scenes):
            if scene is None:
                continue
            while len(cls.IDX_OBJS) <= scene_idx:
                cls.IDX_OBJS.append([None] * len(cls.OBJ_IDXS))
            for obj in scene.objects:
                if cls not in obj.states:
                    continue
                rel_path = obj.relative_prim_path
                if rel_path not in cls.OBJ_IDXS:
                    cls.OBJ_IDXS[rel_path] = len(cls.OBJ_IDXS)
                    for s_row in cls.IDX_OBJS:
                        s_row.append(None)
                cls.IDX_OBJS[scene_idx][cls.OBJ_IDXS[rel_path]] = obj
            # scene.active_systems already excludes the cloth system, so every entry here is a
            # physical or visual particle system (the families ParticleViewAPI tracks).
            for system_name in scene.active_systems.keys():
                if system_name not in cls.SYS_IDXS:
                    cls.SYS_IDXS[system_name] = len(cls.SYS_IDXS)

        # Zero-initialize VALUES
        S = len(cls.IDX_OBJS)
        N_obj = len(cls.OBJ_IDXS)
        N_sys = len(cls.SYS_IDXS)
        if S > 0 and N_obj > 0 and N_sys > 0:
            cls.VALUES = th.zeros((S, N_obj, N_sys, *cls.value_shape), dtype=cls.value_type, device="cuda")

        # Pinned CPU mirror + previous-step snapshot for change detection
        cls.VALUES_CPU = th.zeros(cls.VALUES.shape, dtype=cls.value_type).pin_memory()
        cls.PREV_VALUES = cls.VALUES_CPU.clone()

        # Wrap as wp.array for kernel consumption
        if cls.VALUES.numel() > 0:
            cls.VALUES_WP = _wp_from_torch(cls.VALUES)
            cls.VALUES_CPU_WP = _wp_from_torch(cls.VALUES_CPU)
        else:
            cls.VALUES_WP = None
            cls.VALUES_CPU_WP = None

        # The captured wp.graph holds stale pointers/shapes — force a re-capture before next step.
        TensorizedState.graph_dirty = True

    def _get_value(self, system):
        # Read the (scene, self-object, system) cell from the pinned CPU mirror (no GPU stall).
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        sys_idx = self.SYS_IDXS[system.name]
        val = self.VALUES_CPU[s, obj_idx, sys_idx].to(self.value_type)
        if isinstance(val, th.Tensor) and val.numel() == 1:
            val = val.item()
        return val

    def _set_value(self, system, new_value):
        """Default: not implemented. Subclasses override if a setter is needed (typically by
        reusing the original non-tensorized setter logic)."""
        raise NotImplementedError(
            f"_set_value not implemented for {self.__class__.__name__}. "
            "Override in the subclass if an (object, system) setter is needed."
        )

    # Derived state; persistence is a no-op by default (recomputed each step).
    def _dump_state(self):
        return {}

    def _load_state(self, state):
        pass

    def serialize(self, state):
        return th.empty(0)

    def deserialize(self, state):
        return {}, 0

    @property
    def state_size(self):
        return 0

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("TensorizedObjectSystemState")
        return classes
