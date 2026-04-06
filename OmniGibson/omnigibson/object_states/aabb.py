import torch as th

from omnigibson.object_states.tensorized_value_state import TensorizedValueState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI


class AABB(TensorizedValueState):
    """
    Axis-aligned bounding box state, computed in bulk across all objects and scenes.

    For rigid objects, boundary points are pre-scaled into link-local homogeneous coords
    (P_obj, V_max, 4) and transformed each step via batched pose matrices from RigidBodyViewAPI.
    Cloth objects fall back to per-object EntityPrim.aabb.

    VALUES shape: (S, O, 6) — [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]
    """

    # (P_obj, V_max, 4) — padded homogeneous local boundary points (local * link.scale, w=1)
    LOCAL_POINTS = None

    # (P_obj, V_max) bool — True for valid (non-padded) point slots
    POINTS_MASK = None

    # (P_obj,) int64 — O index (object index) for each rigid link
    PRIM_TO_OBJ_IDX = None

    # (P_obj,) int64 — P index into RigidBodyViewAPI for each rigid link
    PRIM_BODY_IDX = None

    # list[int] — O indices of cloth objects (per-object fallback)
    CLOTH_OBJ_IDXS = None

    # (S, O, 3) scratch buffers — allocated in initialize_view(), reused each step
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
            cls.LOCAL_POINTS = th.zeros((0, 0, 4))
            cls.POINTS_MASK = th.zeros((0, 0), dtype=th.bool)
            cls.PRIM_TO_OBJ_IDX = th.zeros((0,), dtype=th.long)
            cls.PRIM_BODY_IDX = th.zeros((0,), dtype=th.long)
            cls.CLOTH_OBJ_IDXS = []
            cls._AABB_LO = th.full((S, O, 3), float("inf"))
            cls._AABB_HI = th.full((S, O, 3), float("-inf"))
            return

        all_local_points = []  # list of (V_i, 4) per rigid link
        prim_to_obj = []  # O index for each rigid link
        prim_links = []  # link objects for RigidBodyViewAPI.get_body_indices()
        cls.CLOTH_OBJ_IDXS = []

        for obj_idx in range(O):
            # Use scene-0 representative (same structure across all scenes)
            obj = next(row[obj_idx] for row in cls.IDX_OBJS if row[obj_idx] is not None)

            if obj.prim_type == PrimType.CLOTH or obj.kinematic_only:
                cls.CLOTH_OBJ_IDXS.append(obj_idx)
                continue

            for link in obj.links.values():
                local_pts = link.collision_boundary_points_local  # (V_i, 3) or None
                if local_pts is None:
                    continue
                world_scale = link.get_world_scale()  # (3,) — full world-accumulated scale, no PoseAPI
                scaled_pts = local_pts * world_scale  # apply full world-accumulated scale
                homog = th.cat([scaled_pts, th.ones(len(scaled_pts), 1)], dim=1)  # (V_i, 4)
                all_local_points.append(homog)
                prim_to_obj.append(obj_idx)
                prim_links.append(link)

        P_obj = len(all_local_points)
        V_max = max(p.shape[0] for p in all_local_points) if P_obj > 0 else 0

        cls.LOCAL_POINTS = th.zeros((P_obj, V_max, 4))
        cls.POINTS_MASK = th.zeros((P_obj, V_max), dtype=th.bool)
        for i, pts in enumerate(all_local_points):
            V_i = pts.shape[0]
            cls.LOCAL_POINTS[i, :V_i] = pts
            cls.POINTS_MASK[i, :V_i] = True

        cls.PRIM_TO_OBJ_IDX = th.tensor(prim_to_obj, dtype=th.long)
        cls.PRIM_BODY_IDX = RigidBodyViewAPI.get_body_indices(prim_links)

        # Allocate scratch buffers — reused each step, never recreated
        cls._AABB_LO = th.full((S, O, 3), float("inf"))
        cls._AABB_HI = th.full((S, O, 3), float("-inf"))

        # Initialize new VALUE slots for objects that just appeared
        for rel_path, obj_idx in cls.OBJ_IDXS.items():
            if rel_path not in prev_rel_paths:
                for s_idx in range(S):
                    if cls.IDX_OBJS[s_idx][obj_idx] is not None:
                        cls.VALUES[s_idx, obj_idx] = 0.0  # will be correct after first _update_values

    @classmethod
    def _update_values(cls, values):
        S = values.shape[0]
        P_obj = cls.PRIM_BODY_IDX.shape[0] if cls.PRIM_BODY_IDX is not None else 0

        if P_obj > 0:
            # 1. Gather pose matrices for object rigid links only
            poses = RigidBodyViewAPI.get_pose_matrices()[:, cls.PRIM_BODY_IDX]  # (S, P_obj, 4, 4)

            # 2. Transform local points to world frame
            #    einsum 'spij,pvj->spvi': M[s,p] @ pts[p,v] for each (s,p,v)
            #    poses: (S, P_obj, 4, 4),  LOCAL_POINTS: (P_obj, V_max, 4)
            world_pts = th.einsum("spij,pvj->spvi", poses, cls.LOCAL_POINTS)  # (S, P_obj, V_max, 4)
            world_pts = world_pts[..., :3]  # (S, P_obj, V_max, 3)

            # 3. Mask padding slots (invalid points get ±inf so min/max ignores them)
            mask = cls.POINTS_MASK.unsqueeze(0).expand(S, -1, -1)  # (S, P_obj, V_max)
            world_pts_min = world_pts.clone()
            world_pts_max = world_pts.clone()
            world_pts_min[~mask] = float("inf")
            world_pts_max[~mask] = float("-inf")

            # 4. Per-prim min/max over V_max dim
            min_p = world_pts_min.min(dim=2).values  # (S, P_obj, 3)
            max_p = world_pts_max.max(dim=2).values  # (S, P_obj, 3)

            # 5. Scatter per-prim min/max into per-object AABB using pre-allocated scratch buffers
            #    Each rigid link maps to exactly one object; scatter_reduce_ aggregates across links
            idx = cls.PRIM_TO_OBJ_IDX.view(1, -1, 1).expand(S, -1, 3)  # (S, P_obj, 3)
            cls._AABB_LO.fill_(float("inf"))
            cls._AABB_HI.fill_(float("-inf"))
            cls._AABB_LO.scatter_reduce_(1, idx, min_p, reduce="amin", include_self=True)
            cls._AABB_HI.scatter_reduce_(1, idx, max_p, reduce="amax", include_self=True)

        # 6. Cloth / kinematic_only fallback — per-object, uses existing EntityPrim.aabb
        for obj_idx in cls.CLOTH_OBJ_IDXS:
            for s_idx, s_row in enumerate(cls.IDX_OBJS):
                obj = s_row[obj_idx]
                if obj is not None:
                    lo, hi = obj.aabb
                    cls._AABB_LO[s_idx, obj_idx] = lo
                    cls._AABB_HI[s_idx, obj_idx] = hi

        # 7. Write into values in-place — no new tensor allocation
        values[..., :3] = cls._AABB_LO
        values[..., 3:] = cls._AABB_HI
        return values

    def _get_value(self):
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        v = self.VALUES[s, obj_idx]  # (6,)
        return v[:3], v[3:]  # (lo, hi) — matches EntityPrim.aabb return type

    def _set_value(self, new_value):
        raise NotImplementedError("AABB is read-only; it is derived from pose and cannot be set directly.")

    def _dump_state(self):
        # Return the raw flat (6,) VALUES row so serialize() receives a tensor, not the (lo, hi) tuple
        s = self.obj.scene.idx
        obj_idx = self.OBJ_IDXS[self.obj.relative_prim_path]
        return {self.value_name: self.VALUES[s, obj_idx].clone()}

    def _load_state(self, state):
        # AABB is fully derived from pose; it will be recomputed on the next step.
        pass
