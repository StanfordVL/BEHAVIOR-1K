import torch as th

import omnigibson as og
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

    # (S*O, 3) scratch buffers — pre-allocated in initialize_view(), reused each step via fill_()
    _AABB_LO = None
    _AABB_HI = None

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
            cls._AABB_LO = th.zeros((0, 3), device="cuda")
            cls._AABB_HI = th.zeros((0, 3), device="cuda")
            return

        prim_body_idx = []
        link_idx = []

        for scene_idx, scene in enumerate(og.sim.scenes):
            for obj in scene.objects:
                obj_i = cls.OBJ_IDXS.get(obj.relative_prim_path)
                if obj_i is None:
                    continue  # not tracked by AABB (robots, etc.)
                for link in obj.links.values():
                    flat_idx = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                    if flat_idx is None:
                        continue  # articulated/cloth link not in RigidBodyViewAPI
                    if not RigidBodyViewAPI.POINTS_MASK[flat_idx].any():
                        continue  # no collision geometry for this link
                    prim_body_idx.append(flat_idx)
                    link_idx.append(scene_idx * O + obj_i)

        covered_obj_idxs = {li % O for li in link_idx}
        cls.CLOTH_OBJ_IDXS = [i for i in range(O) if i not in covered_obj_idxs]

        cls.PRIM_BODY_IDX = th.tensor(prim_body_idx, dtype=th.long, device="cuda")
        cls.LINK_IDX = th.tensor(link_idx, dtype=th.long, device="cuda")

        # Pre-allocated scratch buffers — reused each step via fill_(), never recreated
        cls._AABB_LO = th.full((S * O, 3), float("inf"), device="cuda")
        cls._AABB_HI = th.full((S * O, 3), float("-inf"), device="cuda")

        # Initialize new VALUE slots for objects that just appeared
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(S):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = 0.0  # will be correct after first _update_values

    @classmethod
    def _update_values(cls, values):
        S = values.shape[0]
        O = values.shape[1]

        cls._AABB_LO.fill_(float("inf"))
        cls._AABB_HI.fill_(float("-inf"))

        # Batched AABB for all rigid links (physx_tracked + physx_untracked kinematic)
        if cls.PRIM_BODY_IDX is not None and cls.PRIM_BODY_IDX.numel() > 0:
            RigidBodyViewAPI.get_aabb(cls.PRIM_BODY_IDX, cls.LINK_IDX, cls._AABB_LO, cls._AABB_HI)

        # Cloth fallback — per-object, uses existing obj.aabb
        for obj_idx in cls.CLOTH_OBJ_IDXS:
            for s_idx, s_row in enumerate(cls.IDX_OBJS):
                obj = s_row[obj_idx]
                if obj is not None:
                    lo, hi = obj.aabb
                    cls._AABB_LO[s_idx * O + obj_idx] = lo
                    cls._AABB_HI[s_idx * O + obj_idx] = hi

        # Write into values in-place — two slice-writes, no th.cat, no new tensor
        values[..., :3] = cls._AABB_LO.view(S, O, 3)
        values[..., 3:] = cls._AABB_HI.view(S, O, 3)
        return values

    def _get_value(self):
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        v = self.VALUES_CPU[s, obj_idx]  # (6,) — CPU mirror, no GPU stall
        return v[:3], v[3:]  # (lo, hi) — matches EntityPrim.aabb return type

    def _set_value(self, new_value):
        raise NotImplementedError("AABB is read-only; it is derived from pose and cannot be set directly.")

    def _dump_state(self):
        # Return the raw flat (6,) VALUES_CPU row so serialize() receives a CPU tensor
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return {self.value_name: self.VALUES_CPU[s, obj_idx].clone()}

    def _load_state(self, state):
        # AABB is fully derived from pose; it will be recomputed on the next step.
        pass
