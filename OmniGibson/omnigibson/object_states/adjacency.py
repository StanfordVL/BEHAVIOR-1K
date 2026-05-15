import math

import torch as th
import warp as wp

from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI, rigid_inverse_mat44

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.MAX_DISTANCE_VERTICAL = 5.0
m.MAX_DISTANCE_HORIZONTAL = 5.0

# Number of horizontal directions, evenly spaced around the XY plane at angles k * 360/N.
m.HORIZONTAL_DIRECTION_COUNT = 10


# Tensorized Adjacency state
#
# VALUE is (S, N, N, 12) bool tensor populated by Warp ray casts against per-link wp.Mesh.
#
# Axis layout:
#   k=0       : +Z   (other above self)
#   k=1       : -Z   (other below self)
#   k=2..11   : 10 horizontal directions evenly spaced on the XY plane,
#               direction k has angle (k-2) * 2π / 10 (so k=2 is +X, k=7 is -X).
#
# VALUES[s, a, b, k] = True iff a ray from object a's AABB center in direction k
# hits any collision-link of object b within max_distances[k]. Self pairs and
# cross-scene pairs are False.
#
# Cloth is skipped via is_compatible — cloth has no collision_mesh_cpu_data. TODO(andi) verrify this

# Total number of axis directions (2 vertical + 10 horizontal)
_ADJ_AXIS_COUNT = 2 + m.HORIZONTAL_DIRECTION_COUNT  # = 12

# Horizontal-direction slice into the K axis: range(_HORIZONTAL_K_START, _HORIZONTAL_K_END).
_HORIZONTAL_K_START = 2
_HORIZONTAL_K_END = _ADJ_AXIS_COUNT  # exclusive


@wp.kernel
def _adjacency_ray_cast_kernel(
    aabb_values: wp.array3d(dtype=wp.float32),  # (S, N_aabb, 6) — [lo_x, lo_y, lo_z, hi_x, hi_y, hi_z]
    aabb_obj_idxs: wp.array(dtype=wp.int32),  # (N_adj,) — adj_idx → aabb_idx, -1 if no AABB
    directions: wp.array2d(dtype=wp.float32),  # (K, 3) — world-frame ray directions
    max_distances: wp.array(dtype=wp.float32),  # (K,) — per-axis max ray distance
    link_mesh_ids: wp.array(dtype=wp.uint64),  # (L,) — wp.Mesh ids; 0 if link has no collision mesh
    link_pose_matrices: wp.array(dtype=wp.mat44),  # (L,) — link world transforms
    link_to_obj_idx: wp.array(dtype=wp.int32),  # (L,) — link → adj_idx of parent obj, -1 if untracked
    link_to_scene_idx: wp.array(dtype=wp.int32),  # (L,) — link → scene_idx of parent obj
    output: wp.array4d(dtype=wp.int32),  # (S, N_adj, N_adj, K) — atomic_max target
):
    """
    One thread per (scene, origin_obj, target_link, axis). On hit, atomic-OR sets scratch[s, a, b, k].
    """
    s, a, l, k = wp.tid()

    # Skip if origin object has no tracked AABB (e.g. cloth, which is_compatible filters out
    # of Adjacency but might still have an Adjacency.OBJ_IDXS entry from a cross-scene partner).
    aabb_idx = aabb_obj_idxs[a]
    if aabb_idx < 0:
        return

    # Skip if target link has no collision mesh
    mesh_id = link_mesh_ids[l]
    if mesh_id == wp.uint64(0):
        return

    # Skip if target link's parent isn't in this scene
    if link_to_scene_idx[l] != s:
        return

    # Skip if target link is untracked, or belongs to the same object (self-pair)
    b = link_to_obj_idx[l]
    if b < 0 or b == a:
        return

    # Compute origin (AABB center in world frame) — read 6 floats inline
    lo_x = aabb_values[s, aabb_idx, 0]
    lo_y = aabb_values[s, aabb_idx, 1]
    lo_z = aabb_values[s, aabb_idx, 2]
    hi_x = aabb_values[s, aabb_idx, 3]
    hi_y = aabb_values[s, aabb_idx, 4]
    hi_z = aabb_values[s, aabb_idx, 5]
    origin_w = wp.vec3((lo_x + hi_x) * 0.5, (lo_y + hi_y) * 0.5, (lo_z + hi_z) * 0.5)
    dir_w = wp.vec3(directions[k, 0], directions[k, 1], directions[k, 2])
    t_max = max_distances[k]

    # Transform ray into target link's local frame.
    # transform_point applies translation; transform_vector is rotation-only (correct for direction).
    inv = rigid_inverse_mat44(link_pose_matrices[l])
    origin_local = wp.transform_point(inv, origin_w)
    dir_local = wp.transform_vector(inv, dir_w)

    if wp.mesh_query_ray_anyhit(mesh_id, origin_local, dir_local, t_max):
        wp.atomic_max(output, s, a, b, k, wp.int32(1))


@wp.kernel
def _adjacency_finalize_kernel(
    output: wp.array4d(dtype=wp.int32),  # (S, N, N, K) int32
    values: wp.array4d(dtype=wp.uint8),  # (S, N, N, K) uint8 — backed by th.bool storage
):
    """Convert int32 scratch to uint8 bool VALUES."""
    s, a, b, k = wp.tid()
    if output[s, a, b, k] > wp.int32(0):
        values[s, a, b, k] = wp.uint8(1)
    else:
        values[s, a, b, k] = wp.uint8(0)


def _build_adjacency_axis_tables():
    """Build the (12, 3) directions table and (12,) max-distance table.

    Layout matches the kernel's k axis:
      [+Z, -Z, h_0, h_1, ..., h_(N-1)]
    where N = m.HORIZONTAL_DIRECTION_COUNT and h_k = (cos(k·2π/N), sin(k·2π/N), 0).
    """
    n_horizontal = m.HORIZONTAL_DIRECTION_COUNT
    angles = th.arange(n_horizontal, dtype=th.float32) * (2.0 * math.pi / n_horizontal)
    horizontal_dirs = th.stack([th.cos(angles), th.sin(angles), th.zeros_like(angles)], dim=1)  # (N, 3)

    directions = th.zeros((_ADJ_AXIS_COUNT, 3), dtype=th.float32)
    directions[0] = th.tensor([0.0, 0.0, 1.0])
    directions[1] = th.tensor([0.0, 0.0, -1.0])
    directions[2:] = horizontal_dirs

    max_distances = th.full((_ADJ_AXIS_COUNT,), m.MAX_DISTANCE_HORIZONTAL, dtype=th.float32)
    max_distances[0] = m.MAX_DISTANCE_VERTICAL
    max_distances[1] = m.MAX_DISTANCE_VERTICAL
    return directions, max_distances


class Adjacency(TensorizedRelativeState):
    """
    Pairwise adjacency state.

    S = number of scenes
    N = number of objects with Adjacency state
    VALUES has shape (S, N, N, 22) bool — VALUES[s, a, b, k] is True iff object b is
    adjacent to object a in direction k (from a's AABB center).

    Diagonal and cross-scene cells are always False.
    Cloth is excluded via is_compatible (no collision mesh to ray-cast against).
    """

    # wp kernel input
    AABB_OBJ_IDXS_WP = None  # (N_adj,) int32
    LINK_TO_OBJ_IDX_WP = None  # (L_total,) int32
    LINK_TO_SCENE_IDX_WP = None  # (L_total,) int32
    DIRECTIONS_WP = None  # (22, 3) float32
    MAX_DISTANCES_WP = None  # (22,) float32
    OUTPUT_WP = None  # (S, N_adj, N_adj, 22) int32 — atomic_max target

    # Underlying torch tensors
    _aabb_obj_idxs = None
    _link_to_obj_idx = None
    _link_to_scene_idx = None
    _directions = None
    _max_distances = None
    _output = None

    @classproperty
    def value_shape(cls):
        return (_ADJ_AXIS_COUNT,)

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "adjacency"

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    @classmethod
    def is_compatible(cls, obj, **kwargs):
        compatible, reason = super().is_compatible(obj, **kwargs)
        if not compatible:
            return compatible, reason
        # Cloth has no collision_mesh_cpu_data — exclude as both origin and target.
        # TODO: revisit when cloth gains a queryable mesh proxy.
        if obj.prim_type == PrimType.CLOTH:
            return False, "Adjacency does not support cloth objects"
        return True, None

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        # Build the constant axis tables (directions + per-axis max distance).
        directions_cpu, max_distances_cpu = _build_adjacency_axis_tables()
        cls._directions = directions_cpu.cuda()
        cls._max_distances = max_distances_cpu.cuda()
        cls.DIRECTIONS_WP = wp.from_torch(cls._directions)
        cls.MAX_DISTANCES_WP = wp.from_torch(cls._max_distances)

    @classmethod
    def initialize_view(cls):
        super().initialize_view()

        S = len(cls.IDX_OBJS)
        N = len(cls.OBJ_IDXS)

        # Build aabb_obj_idxs: maps Adjacency-N → AABB-N (or -1 if unknown to AABB).
        if N > 0 and AABB.OBJ_IDXS is not None:
            aabb_obj_idxs = th.full((N,), -1, dtype=th.int32)
            for rel_path, adj_idx in cls.OBJ_IDXS.items():
                aabb_obj_idxs[adj_idx] = AABB.OBJ_IDXS.get(rel_path, -1)
        else:
            aabb_obj_idxs = th.empty(0, dtype=th.int32)
        cls._aabb_obj_idxs = aabb_obj_idxs.cuda()
        cls.AABB_OBJ_IDXS_WP = wp.from_torch(cls._aabb_obj_idxs) if N > 0 else None

        # Build link_to_obj_idx / link_to_scene_idx tables, length = N_links_total in
        # RigidBodyViewAPI (whether or not those links belong to Adjacency-tracked objects).
        # Untracked link slots stay at -1; the kernel skips them.
        if RigidBodyViewAPI._PATH_TO_IDX:
            L_total = len(RigidBodyViewAPI._PATH_TO_IDX)
            link_to_obj = th.full((L_total,), -1, dtype=th.int32)
            link_to_scene = th.full((L_total,), -1, dtype=th.int32)
            for s_idx, scene_row in enumerate(cls.IDX_OBJS):
                for adj_idx, obj in enumerate(scene_row):
                    if obj is None:
                        continue
                    for link in obj.links.values():
                        flat_idx = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                        if flat_idx is None:
                            continue
                        link_to_obj[flat_idx] = adj_idx
                        link_to_scene[flat_idx] = s_idx
            cls._link_to_obj_idx = link_to_obj.cuda()
            cls._link_to_scene_idx = link_to_scene.cuda()
            cls.LINK_TO_OBJ_IDX_WP = wp.from_torch(cls._link_to_obj_idx)
            cls.LINK_TO_SCENE_IDX_WP = wp.from_torch(cls._link_to_scene_idx)
        else:
            cls._link_to_obj_idx = None
            cls._link_to_scene_idx = None
            cls.LINK_TO_OBJ_IDX_WP = None
            cls.LINK_TO_SCENE_IDX_WP = None

        # Allocate the int32 scratch (S, N, N, 22) used as the atomic_max target.
        if S > 0 and N > 0:
            cls._output = th.zeros((S, N, N, _ADJ_AXIS_COUNT), dtype=th.int32, device="cuda")
            cls.OUTPUT_WP = wp.from_torch(cls._output)
        else:
            cls._output = None
            cls.OUTPUT_WP = None

    @classmethod
    def _update_values(cls, values):
        # All required handles must be live; otherwise nothing to do this step.
        if (
            cls.OUTPUT_WP is None
            or cls.VALUES_WP is None
            or AABB.VALUES_WP is None
            or cls.AABB_OBJ_IDXS_WP is None
            or RigidBodyViewAPI.LINK_MESH_IDS is None
            or RigidBodyViewAPI.POSE_MATRICES is None
            or cls.LINK_TO_OBJ_IDX_WP is None
            or cls.LINK_TO_SCENE_IDX_WP is None
        ):
            return

        S = values.shape[0]
        N = values.shape[1]
        K = _ADJ_AXIS_COUNT
        L = RigidBodyViewAPI.LINK_MESH_IDS.shape[0]
        if S == 0 or N == 0 or L == 0:
            return

        # 1. Zero the int32 scratch via CUDA memset (graph-capturable for contiguous wp.array).
        cls.OUTPUT_WP.zero_()

        # 2. Ray-cast against each link's mesh; atomic_max into scratch on hit.
        wp.launch(
            kernel=_adjacency_ray_cast_kernel,
            dim=(S, N, L, K),
            inputs=[
                AABB.VALUES_WP,
                cls.AABB_OBJ_IDXS_WP,
                cls.DIRECTIONS_WP,
                cls.MAX_DISTANCES_WP,
                RigidBodyViewAPI.LINK_MESH_IDS,
                RigidBodyViewAPI.POSE_MATRICES,
                cls.LINK_TO_OBJ_IDX_WP,
                cls.LINK_TO_SCENE_IDX_WP,
                cls.OUTPUT_WP,
            ],
            device="cuda",
        )

        # 3. Convert int32 scratch → uint8 bool VALUES.
        wp.launch(
            kernel=_adjacency_finalize_kernel,
            dim=(S, N, N, K),
            inputs=[cls.OUTPUT_WP, cls.VALUES_WP],
            device="cuda",
        )
