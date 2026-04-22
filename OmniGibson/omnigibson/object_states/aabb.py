import torch as th

from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI


class AABB(TensorizedValueState):
    """
    Axis-aligned bounding box state, computed in bulk across all objects and scenes.

    For rigid objects, batched AABB computation is delegated to RigidBodyViewAPI.get_aabb(),
    which uses pre-stored LOCAL_POINTS and _POSE_MATRICES. Cloth objects fall back to
    per-object EntityPrim.aabb.

    VALUES shape: (S, O, 6) — [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]
    """

    # (N_links_aabb_tracked,) int64 — index into RigidBodyViewAPI's pose data
    PRIM_BODY_IDX = None

    # (N_links_aabb_tracked,) int64 — pre-computed s*O + obj_idx for scatter in get_aabb()
    LINK_IDX = None

    # list[int] — O indices of cloth objects (per-object fallback)
    CLOTH_OBJ_IDXS = None

    # (S, O) bool CPU — True where cached AABB may be stale since last global_update()
    # Often caused by sample_kinematics()
    STALE = None

    @classproperty
    def value_shape(cls):
        return (6,)

    @classproperty
    def value_name(cls):
        return "aabb"

    @classmethod
    def initialize_view(cls):
        # Snapshot before super() calls global_initialize() which resets OBJ_IDXS
        prev_rel_paths = set(cls.OBJ_IDXS.keys()) if cls.OBJ_IDXS is not None else set()

        # Base class rebuilds OBJ_IDXS, IDX_OBJS, VALUES (S, O, 6)
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        O = len(cls.OBJ_IDXS)

        if S == 0 or O == 0:
            cls.PRIM_BODY_IDX = th.zeros((0,), dtype=th.long, device="cuda")
            cls.LINK_IDX = th.zeros((0,), dtype=th.long, device="cuda")
            cls.CLOTH_OBJ_IDXS = []
            cls.STALE = th.zeros((0, 0), dtype=th.bool)
            return

        prim_body_idx = []
        link_idx = []

        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for obj_index, obj in enumerate(scene_row):
                if obj is None:
                    continue
                for link in obj.links.values():
                    flat_idx = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                    if flat_idx is None:
                        continue  # articulated/cloth link not in RigidBodyViewAPI
                    if not RigidBodyViewAPI.POINTS_MASK[flat_idx].any():
                        continue  # no collision geometry for this link
                    prim_body_idx.append(flat_idx)
                    link_idx.append(scene_idx * O + obj_index)

        covered_obj_idxs = {li % O for li in link_idx}
        cls.CLOTH_OBJ_IDXS = [i for i in range(O) if i not in covered_obj_idxs]

        cls.PRIM_BODY_IDX = th.tensor(prim_body_idx, dtype=th.long, device="cuda")
        cls.LINK_IDX = th.tensor(link_idx, dtype=th.long, device="cuda")
        cls.STALE = th.zeros((S, O), dtype=th.bool)

        # Initialize new VALUE slots for objects that just appeared
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(S):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = th.nan

    @classmethod
    def _update_values(cls, values):
        S = values.shape[0]
        O = values.shape[1]

        # Fill with infinite values for the lower bound and -infinite values for the upper bound
        values.fill_(float("inf"))
        values[:, :, 3:] = float("-inf")

        # Batched AABB for all rigid links (physx_tracked + physx_untracked kinematic)
        if cls.PRIM_BODY_IDX is not None and cls.PRIM_BODY_IDX.numel() > 0:
            RigidBodyViewAPI.get_aabb(cls.PRIM_BODY_IDX, cls.LINK_IDX, values.view(S * O, 6))

        # Cloth fallback — per-object, uses existing obj.aabb
        for obj_idx in cls.CLOTH_OBJ_IDXS:
            for s_idx, s_row in enumerate(cls.IDX_OBJS):
                obj = s_row[obj_idx]
                lo, hi = obj.aabb
                values[s_idx, obj_idx, :3] = lo
                values[s_idx, obj_idx, 3:] = hi

    @classmethod
    def mark_stale(cls, obj):
        """Mark obj's cached AABB as stale; _get_value() will recompute fresh until next global_update()."""
        if obj.relative_prim_path not in cls.OBJ_IDXS:
            return
        cls.STALE[obj.scene.idx, cls.OBJ_IDXS[obj.relative_prim_path]] = True

    @classmethod
    def post_update(cls):
        super().post_update()
        cls.STALE.fill_(False)

    def _get_value(self):
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        if self.STALE[s, obj_idx].item():
            return self.obj.aabb  # fresh recompute via EntityPrim property
        v = self.VALUES_CPU[s, obj_idx]  # (6,) — CPU mirror, no GPU stall
        return v[:3], v[3:]  # (lo, hi) — matches EntityPrim.aabb return type

    def _set_value(self, new_value):
        raise NotImplementedError("AABB is read-only; it is derived from pose and cannot be set directly.")

    def _dump_state(self):
        # Return the raw flat (6,) VALUES_CPU row so serialize() receives a CPU tensor
        s = self.obj.scene.idx
        if self.obj.relative_prim_path in self.OBJ_IDXS:
            obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
            return {self.value_name: self.VALUES_CPU[s, obj_idx].clone()}
        else:
            # obj not initialized yet. return a empty (6,) tensor
            return {self.value_name: th.zeros(self.value_shape)}

    def _load_state(self, state):
        # AABB is fully derived from pose; it will be recomputed on the next step.
        pass
