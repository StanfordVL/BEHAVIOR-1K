import math

import torch as th

import omnigibson as og
from omnigibson.object_states.object_state_base import RelativeObjectState
from omnigibson.object_states.tensorized_state import TensorizedState, _wp_from_torch
from omnigibson.utils.python_utils import classproperty


class TensorizedRelativeState(TensorizedState, RelativeObjectState):
    """
    Tensorized state-mixin for RELATIVE (pairwise) values.

    All pairwise values across all object state instances of this type are updated at once
    via _update_values(), rather than per individual instance update() call.

    Multi-scene layout
    ------------------
    VALUES      (S, N, N, *value_shape)  — [scene, self_obj, other_obj, ...]
    OBJ_IDXS    {relative_prim_path: int}  — shared N index across both pairwise dims
    IDX_OBJS    list[list[obj|None]]  — IDX_OBJS[scene_idx][obj_idx] = object instance

    Conventions
    -----------
    - Diagonal cells `VALUES[s, i, i, ...]` should be False / zero by convention; subclasses
      enforce this in their `_update_values` kernel.
    - Cross-scene cells (object `a` exists in scene s but `b` does not) should be False / zero;
      subclasses enforce via existence-mask checks in their kernel.

    Subclasses are derived states (computed from simulator state each step). The default
    `_dump_state` / `_load_state` / `serialize` / `deserialize` / `state_size` are no-ops
    because pairwise values are recomputed and don't need to round-trip through saves.
    Subclasses may override if persistence is needed.

    `_set_value(self, other, new_value)` raises NotImplementedError by default. Subclasses
    that need a setter should override it (typically by copying the original non-tensorized
    setter logic and writing into the (s, idx_self, idx_other) cell).
    """

    @classmethod
    def global_initialize(cls):
        """Initialize the global class-level tensors and indices for the (S, N, N, ...) shape."""
        cls.VALUES = th.empty(0, dtype=cls.value_type, device="cuda").reshape(0, 0, 0, *cls.value_shape)
        cls.VALUES_CPU = th.empty(0, dtype=cls.value_type).pin_memory().reshape(0, 0, 0, *cls.value_shape)
        cls.PREV_VALUES = th.empty(0, dtype=cls.value_type).reshape(0, 0, 0, *cls.value_shape)
        cls.OBJ_IDXS = {}  # {relative_prim_path: int}
        cls.IDX_OBJS = []  # list[list[obj|None]]

        # Per-pair state size (flattened size of value_shape). Full per-instance state size
        # depends on N; see `state_size` property below.
        cls.STATE_SIZE = math.prod(cls.value_shape)

    @classmethod
    def initialize_view(cls):
        """
        Rebuild all class-level tensors by scanning current objects across all scenes.
        Called from ``simulator.py`` after scene changes.

        Two-pass to avoid the O(N^3) grow-by-cat pattern the naive implementation has:
        pass 1 only touches Python dicts/lists to settle OBJ_IDXS / IDX_OBJS at their
        final size; pass 2 allocates VALUES once at (S, N, N, *value_shape) and fills
        carry-over values directly. For a scene with N objects across S scenes this is
        O(S*N^2) instead of O(N^3).

        Carry-over: for each (a, b) pair where both relative_prim_paths still exist, the
        previous VALUES[s, a, b] is preserved. New rows/columns are zero-initialized.
        """
        # Snapshot for carry-over (OBJ_IDXS / VALUES are None on the very first call)
        prev_obj_idxs = dict(cls.OBJ_IDXS) if cls.OBJ_IDXS is not None else {}
        prev_values = cls.VALUES.clone() if cls.VALUES is not None and cls.VALUES.numel() > 0 else None

        # Reset
        cls.global_initialize()

        # Pass 1: scan all scenes, settle OBJ_IDXS / IDX_OBJS at final shape (Python only).
        for scene_idx, scene in enumerate(og.sim.scenes):
            for obj in scene.objects:
                if cls not in obj.states:
                    continue
                rel_path = obj.relative_prim_path

                # Extend scene dimension if this is a new scene index
                while len(cls.IDX_OBJS) <= scene_idx:
                    cls.IDX_OBJS.append([None] * len(cls.OBJ_IDXS))

                # Register new relative path if first seen across all scenes
                if rel_path not in cls.OBJ_IDXS:
                    cls.OBJ_IDXS[rel_path] = len(cls.OBJ_IDXS)
                    for s_row in cls.IDX_OBJS:
                        s_row.append(None)

                cls.IDX_OBJS[scene_idx][cls.OBJ_IDXS[rel_path]] = obj

        # Pass 2: allocate VALUES once at the final shape, then fill carry-over values.
        S = len(cls.IDX_OBJS)
        N = len(cls.OBJ_IDXS)
        if S > 0 and N > 0:
            cls.VALUES = th.zeros((S, N, N, *cls.value_shape), dtype=cls.value_type, device="cuda")

            if prev_values is not None and prev_values.numel() > 0:
                for rel_path_a, idx_old_a in prev_obj_idxs.items():
                    if rel_path_a not in cls.OBJ_IDXS:
                        continue
                    idx_new_a = cls.OBJ_IDXS[rel_path_a]
                    for rel_path_b, idx_old_b in prev_obj_idxs.items():
                        if rel_path_b not in cls.OBJ_IDXS:
                            continue
                        idx_new_b = cls.OBJ_IDXS[rel_path_b]
                        for s_idx in range(min(prev_values.shape[0], S)):
                            if (
                                cls.IDX_OBJS[s_idx][idx_new_a] is not None
                                and cls.IDX_OBJS[s_idx][idx_new_b] is not None
                            ):
                                cls.VALUES[s_idx, idx_new_a, idx_new_b] = prev_values[s_idx, idx_old_a, idx_old_b]

        # Rebuild pinned CPU mirror — synchronous copy so _get_value() is valid before first async copy
        cls.VALUES_CPU = th.zeros(cls.VALUES.shape, dtype=cls.value_type).pin_memory()
        if cls.VALUES.numel() > 0:
            cls.VALUES_CPU.copy_(cls.VALUES)

        cls.PREV_VALUES = cls.VALUES_CPU.clone()

        # Wrap as wp.array for kernel consumption
        if cls.VALUES.numel() > 0:
            cls.VALUES_WP = _wp_from_torch(cls.VALUES)
            cls.VALUES_CPU_WP = _wp_from_torch(cls.VALUES_CPU)
        else:
            cls.VALUES_WP = None
            cls.VALUES_CPU_WP = None

        # Mark the captured wp.graph as stale — the simulator will re-capture before the next step.
        TensorizedState.graph_dirty = True

        super().initialize_view()

    def _get_value(self, other):
        # Read from the pinned CPU mirror — no GPU stall for Python callers.
        s = self.obj.scene.idx
        idx_self = self.OBJ_IDXS[self.obj.relative_prim_path]
        idx_other = self.OBJ_IDXS[other.relative_prim_path]
        val = self.VALUES_CPU[s, idx_self, idx_other].to(self.value_type)
        if isinstance(val, th.Tensor) and val.numel() == 1:
            val = val.item()
        return val

    def _set_value(self, other, new_value):
        """Default: not implemented for tensorized relative states.

        Subclasses may override to support pairwise setters by copying the original
        non-tensorized setter logic — typically writing the value into both
        ``self.VALUES[s, idx_self, idx_other]`` and ``self.VALUES_CPU[s, idx_self, idx_other]``.
        Most tensorized relative states (e.g. Adjacency, Touching) are derived from
        simulator state and don't expose setters.
        """
        raise NotImplementedError(
            f"_set_value not implemented for {self.__class__.__name__}. "
            "Override in the subclass if a pairwise setter is needed."
        )

    # Pairwise values are derived state; persistence is a no-op by default.
    # Subclasses override if round-tripping through saves is required.

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
        classes.add("TensorizedRelativeState")
        return classes
