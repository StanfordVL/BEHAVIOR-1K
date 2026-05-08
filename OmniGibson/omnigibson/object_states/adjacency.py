import math
from collections import namedtuple

import torch as th
import warp as wp

from omnigibson.macros import create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sampling_utils import raytest, raytest_batch
from omnigibson.utils.usd_utils import RigidBodyViewAPI, rigid_inverse_mat44

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.MAX_DISTANCE_VERTICAL = 5.0
m.MAX_DISTANCE_HORIZONTAL = 5.0

# Number of horizontal directions, evenly spaced around the XY plane at angles k * 360/N.
m.HORIZONTAL_DIRECTION_COUNT = 10

# Per-direction adjacency neighbors found during a ray cast (used by compute_adjacencies, which
# remains alive as the stale-AABB fallback path inside Adjacency._get_value).
AxisAdjacencyList = namedtuple("AxisAdjacencyList", ("positive_neighbors", "negative_neighbors"))


def compute_adjacencies(obj, axes, max_distance, use_aabb_center=True):
    """
    Given an object and a list of axes, find the adjacent objects in the axes'
    positive and negative directions.

    If @obj is of PrimType.CLOTH, then adjacent objects are found with respect to the
    @obj's centroid particle position

    Args:
        obj (StatefulObject): The object to check adjacencies of.
        axes (2D-array): (n_axes, 3) array defining the axes to check in.
            Note that each axis will be checked in both its positive and negative direction.
        use_aabb_center (bool): If True and @obj is not of PrimType.CLOTH, will shoot rays from @obj's aabb center.
            Otherwise, will dynamically compute starting points based on the requested @axes

    Returns:
        list of AxisAdjacencyList: List of length len(axes) containing the adjacencies.
    """
    # Get vectors for each of the axes' directions.
    # The ordering is axes1+, axis1-, axis2+, axis2- etc.
    directions = th.empty((len(axes) * 2, 3))
    directions[0::2] = axes
    directions[1::2] = -axes

    # Prepare this object's info for ray casting.
    if obj.prim_type == PrimType.CLOTH:
        ray_starts = th.tile(obj.root_link.centroid_particle_position, (len(directions), 1))

    else:
        aabb_lower, aabb_higher = obj.states[AABB].get_value()
        object_position = (aabb_lower + aabb_higher) / 2.0
        ray_starts = th.tile(object_position, (len(directions), 1))

        if not use_aabb_center:
            # Dynamically compute start points by iterating over the directions and pre-shooting rays from
            # which to shoot back from
            # For a given direction, we go in the negative (opposite) direction to the edge of the object extent,
            # and then proceed with an additional offset before shooting rays
            shooting_offset = 0.01

            direction_half_extent = directions * (aabb_higher - aabb_lower).reshape(1, 3) / 2.0
            pre_start = object_position.reshape(1, 3) + (direction_half_extent + directions * shooting_offset)
            pre_end = object_position.reshape(1, 3) - direction_half_extent

            idx = 0
            obj_link_paths = {link.prim_path for link in obj.links.values()}

            def _ray_callback(hit):
                # Check for self-hit -- if so, record the position and terminate early
                should_continue = True
                if hit.rigid_body in obj_link_paths:
                    ray_starts[idx] = th.tensor(hit.position)
                    should_continue = False
                return should_continue

            for ray_start, ray_end in zip(pre_start, pre_end):
                raytest(
                    start_point=ray_start,
                    end_point=ray_end,
                    only_closest=False,
                    callback=_ray_callback,
                )
                idx += 1

    # Prepare the rays to cast.
    ray_endpoints = ray_starts + (directions * max_distance)

    # Cast time.
    prim_paths = obj.link_prim_paths
    ray_results = raytest_batch(
        ray_starts, ray_endpoints, only_closest=False, ignore_bodies=prim_paths, ignore_collisions=prim_paths
    )

    # Add the results to the appropriate lists
    # For now, we keep our result in the dimensionality of (direction, hit_object_order).
    # We convert the hit link into unique objects encountered
    objs_by_direction = []
    for results in ray_results:
        unique_objs = set()
        for result in results:
            # Check if the inferred hit object is not None, we add it to our set
            obj_prim_path = "/".join(result["rigidBody"].split("/")[:-1])
            hit_obj = obj.scene.object_registry("prim_path", obj_prim_path, None)
            if hit_obj is not None:
                unique_objs.add(hit_obj)
        objs_by_direction.append(unique_objs)

    # Reshape so that these have the following indices:
    # (axis_idx, direction-one-or-zero, hit_idx)
    objs_by_axis = [
        AxisAdjacencyList(positive_neighbors, negative_neighbors)
        for positive_neighbors, negative_neighbors in zip(objs_by_direction[::2], objs_by_direction[1::2])
    ]
    return objs_by_axis


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


@wp.func
def _ray_aabb_hit(origin: wp.vec3, dir: wp.vec3, t_max: wp.float32, lo: wp.vec3, hi: wp.vec3) -> wp.bool:
    """Slab test: returns True iff the ray (origin, dir, t in [0, t_max]) intersects the AABB."""
    eps = wp.float32(1.0e-9)
    tmin = wp.float32(0.0)
    tmax = t_max
    for i in range(3):
        di = dir[i]
        oi = origin[i]
        loi = lo[i]
        hii = hi[i]
        if wp.abs(di) < eps:
            # Ray parallel to slab i — miss if origin lies outside [lo_i, hi_i]
            if oi < loi or oi > hii:
                return False
        else:
            inv_d = wp.float32(1.0) / di
            t1 = (loi - oi) * inv_d
            t2 = (hii - oi) * inv_d
            if t1 > t2:
                tmp = t1
                t1 = t2
                t2 = tmp
            if t1 > tmin:
                tmin = t1
            if t2 < tmax:
                tmax = t2
            if tmin > tmax:
                return False
    return True


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

    # Broad-phase cull: skip the full BVH traversal if the target object's AABB doesn't
    # intersect the ray segment. Only applies when the target has a tracked AABB; otherwise
    # we fall through to the per-link BVH query.
    b_aabb_idx = aabb_obj_idxs[b]
    if b_aabb_idx >= 0:
        b_lo = wp.vec3(
            aabb_values[s, b_aabb_idx, 0],
            aabb_values[s, b_aabb_idx, 1],
            aabb_values[s, b_aabb_idx, 2],
        )
        b_hi = wp.vec3(
            aabb_values[s, b_aabb_idx, 3],
            aabb_values[s, b_aabb_idx, 4],
            aabb_values[s, b_aabb_idx, 5],
        )
        if not _ray_aabb_hit(origin_w, dir_w, t_max, b_lo, b_hi):
            return

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

    def _get_value(self, other):
        """Read the (22,) bool adjacency vector from self to other.

        If AABB is stale for either side (typically right after sample_kinematics
        between full sim.step() calls), fall back to a per-object on-demand ray cast
        via the legacy compute_adjacencies. This mirrors the legacy
        VerticalAdjacency / HorizontalAdjacency behavior, which was always on-demand.
        """
        s = self.obj.scene.idx

        # TODO(andi) change caching mechanism
        # Staleness check via AABB.STALE — same trigger the legacy code used.
        self_aabb_idx = AABB.OBJ_IDXS.get(self.obj.relative_prim_path, -1) if AABB.OBJ_IDXS else -1
        other_aabb_idx = AABB.OBJ_IDXS.get(other.relative_prim_path, -1) if AABB.OBJ_IDXS else -1
        self_stale = AABB.STALE is not None and self_aabb_idx >= 0 and bool(AABB.STALE[s, self_aabb_idx].item())
        other_stale = AABB.STALE is not None and other_aabb_idx >= 0 and bool(AABB.STALE[s, other_aabb_idx].item())
        if self_stale or other_stale:
            return self._compute_on_demand(other)

        return super()._get_value(other)

    def _compute_on_demand(self, other):
        """Fallback: legacy on-demand ray cast for self in all 12 directions; returns bool vector for `other`.

        Used when cached VALUES are known stale (post-sample_kinematics, pre-next-sim.step).
        Reuses compute_adjacencies which reads fresh AABB via AABB.STALE bypass.
        """
        # Vertical: k=0 (+Z), k=1 (-Z). Use the surface-finding pre-step (use_aabb_center=False)
        # to match legacy VerticalAdjacency.
        vertical_axis_lists = compute_adjacencies(
            self.obj, th.tensor([[0.0, 0.0, 1.0]]), m.MAX_DISTANCE_VERTICAL, use_aabb_center=False
        )
        vertical_pair = vertical_axis_lists[0]

        # Horizontal: 10 evenly-spaced directions at angles k·2π/10 (k=0..9), populating k=2..11.
        # compute_adjacencies probes each axis in both +/- directions, so we pass the first half
        # (5 axes at 0°..144°) and read positive_neighbors → k=2..6, negative_neighbors → k=7..11.
        n_horizontal = m.HORIZONTAL_DIRECTION_COUNT
        half = n_horizontal // 2
        angles = th.arange(half, dtype=th.float32) * (2.0 * math.pi / n_horizontal)
        horizontal_axes = th.stack([th.cos(angles), th.sin(angles), th.zeros_like(angles)], dim=1)
        horizontal_axis_lists = compute_adjacencies(
            self.obj, horizontal_axes, m.MAX_DISTANCE_HORIZONTAL, use_aabb_center=True
        )

        out = th.zeros(_ADJ_AXIS_COUNT, dtype=th.bool)
        out[0] = other in vertical_pair.positive_neighbors
        out[1] = other in vertical_pair.negative_neighbors
        for axis_idx, pair in enumerate(horizontal_axis_lists):
            out[2 + axis_idx] = other in pair.positive_neighbors
            out[2 + half + axis_idx] = other in pair.negative_neighbors
        return out

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
