import math

import torch as th
import warp as wp

import omnigibson as og
from omnigibson.macros import macros, create_module_macros
from omnigibson.object_states.aabb import AABB
from omnigibson.object_states.contains import m as contains_m
from omnigibson.object_states.kinematics_mixin import KinematicsMixin
from omnigibson.object_states.object_state_base import BooleanStateMixin
from omnigibson.object_states.tensorized_relative_state import TensorizedRelativeState
from omnigibson.utils.constants import PrimType
from omnigibson.utils.object_state_utils import (
    m as os_m,
    get_reachability_sampling_context,
    is_pose_reachable_for_predicate,
)
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import RigidBodyViewAPI, RigidContactAPI, rigid_inverse_mat44
import omnigibson.utils.transform_utils as T


m = create_module_macros(module_path=__file__)

m.CONTAINER_POSITION_CHANGE_THRESHOLD = 0.05  # 5cm
m.CONTAINER_ORIENTATION_CHANGE_THRESHOLD = th.deg2rad(th.tensor([10.0])).item()  # 10 degrees
m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_TRANSLATION = 1e-2  # 1cm
m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_ROTATION = math.radians(1)  # 1 degree


# Tensorized Inside state.
#
# VALUES shape: (S, N, N) bool — VALUES[s, inner, container] is True iff
#   1. inner's AABB center lies inside container's AABB, AND
#   2. that point lies inside the convex hull of one of container's container-meta-link
#      visual meshes.
#
# Inside is NOT symmetric (VALUES[s, a, b] != VALUES[s, b, a] in general).
# Diagonal and cross-scene cells are always False.
#
# Only USD Mesh-typed container visual meshes are supported. Primitive-typed visual meshes
# (Sphere/Cube/Cylinder/Cone) are skipped at initialize_view; container hulls are expected
# to be authored as a merged convex Mesh per asset_pipeline/guide/fillable.md.
#
# Volume check is a convex-hull halfspace test per visual mesh — point is inside iff
# `(p - face_centroid) · face_normal < 0` for every face. The halfspace tests are
# parallelized across (scene, inner, face) and reduced per (scene, inner, mesh) — see
# _inside_halfspace_test_kernel / _inside_mesh_reduce_kernel.
#
# All meshes across all containers are flattened into a single global table (length M),
# all faces into a single global table (length F_total). Per step we recompute each mesh's
# world to local-w-scale inverse from its parent link's current world pose; everything else
# is precomputed once in initialize_view.


@wp.kernel
def _inside_inv_world_kernel(
    pose_matrices: wp.array(dtype=wp.mat44),  # (L,) link world transforms (rigid)
    mesh_parent_link: wp.array(dtype=wp.int32),  # (M,) flat link idx
    mesh_inv_local_w_scale: wp.array(dtype=wp.mat44),  # (M,) static inv of mesh→link-local-w-scale
    inv_world: wp.array(dtype=wp.mat44),  # (M,) output: world → mesh-local-unscaled
):
    """Per mesh: compose static inv-local with the link's current rigid world inverse."""
    m = wp.tid()
    parent = mesh_parent_link[m]
    if parent < 0:
        return
    inv_world[m] = wp.mul(mesh_inv_local_w_scale[m], rigid_inverse_mat44(pose_matrices[parent]))


@wp.kernel
def _inside_aabb_prefilter_kernel(
    aabb_values: wp.array3d(dtype=wp.float32),  # (S, N_aabb, 6)
    aabb_idx: wp.array(dtype=wp.int32),  # (N,) Inside-N → AABB-N, -1 if missing
    prefilter: wp.array3d(dtype=wp.int32),  # (S, N, N)
):
    """Check whether this obj's AABB center inside container's AABB."""
    s, i, j = wp.tid()
    if i == j:
        prefilter[s, i, j] = wp.int32(0)
        return
    a_i = aabb_idx[i]
    a_j = aabb_idx[j]
    if a_i < 0 or a_j < 0:
        prefilter[s, i, j] = wp.int32(0)
        return

    cx = (aabb_values[s, a_i, 0] + aabb_values[s, a_i, 3]) * 0.5
    cy = (aabb_values[s, a_i, 1] + aabb_values[s, a_i, 4]) * 0.5
    cz = (aabb_values[s, a_i, 2] + aabb_values[s, a_i, 5]) * 0.5

    lo_x = aabb_values[s, a_j, 0]
    lo_y = aabb_values[s, a_j, 1]
    lo_z = aabb_values[s, a_j, 2]
    hi_x = aabb_values[s, a_j, 3]
    hi_y = aabb_values[s, a_j, 4]
    hi_z = aabb_values[s, a_j, 5]

    inside = cx >= lo_x and cx <= hi_x and cy >= lo_y and cy <= hi_y and cz >= lo_z and cz <= hi_z
    if inside:
        prefilter[s, i, j] = wp.int32(1)
    else:
        prefilter[s, i, j] = wp.int32(0)


@wp.kernel
def _inside_halfspace_test_kernel(
    aabb_values: wp.array3d(dtype=wp.float32),  # (S, N_aabb, 6)
    aabb_idx: wp.array(dtype=wp.int32),  # (N,) Inside-N → AABB-N
    prefilter: wp.array3d(dtype=wp.int32),  # (S, N, N)
    inv_world: wp.array(dtype=wp.mat44),  # (M,) per-mesh world → mesh-local-unscaled
    face_to_mesh: wp.array(dtype=wp.int32),  # (F_total,) face → owning mesh index
    mesh_container_idx: wp.array(dtype=wp.int32),  # (M,) Inside-N container idx, -1 if untracked
    mesh_scene_idx: wp.array(dtype=wp.int32),  # (M,)
    face_centroid: wp.array(dtype=wp.vec3),  # (F_total,)
    face_normal: wp.array(dtype=wp.vec3),  # (F_total,)
    outside_flag: wp.array3d(dtype=wp.int32),  # (S, N, M) atomic_max target — 1 means "saw a failing face"
):
    """
    Per (scene, inner_i, face f):
    test inner_i's AABB center against face f's halfspace; on fail (`>= 0`),
    atomic_max outside_flag[s, i, mesh(f)] to 1. The atomic is idempotent so contention
    is bounded — at most one transition per mesh per (s, i).
    """
    s, i, f = wp.tid()

    m = face_to_mesh[f]
    container_j = mesh_container_idx[m]
    if container_j < 0:
        return
    if mesh_scene_idx[m] != s:
        return
    if i == container_j:
        return

    a_i = aabb_idx[i]
    if a_i < 0:
        return

    if prefilter[s, i, container_j] == wp.int32(0):
        return

    # Inner_i's AABB center, world frame.
    cx = (aabb_values[s, a_i, 0] + aabb_values[s, a_i, 3]) * 0.5
    cy = (aabb_values[s, a_i, 1] + aabb_values[s, a_i, 4]) * 0.5
    cz = (aabb_values[s, a_i, 2] + aabb_values[s, a_i, 5]) * 0.5
    p_world = wp.vec3(cx, cy, cz)

    # change from world to mesh-local-unscaled
    p_local = wp.transform_point(inv_world[m], p_world)

    if wp.dot(p_local - face_centroid[f], face_normal[f]) >= wp.float32(0.0):
        wp.atomic_max(outside_flag, s, i, m, wp.int32(1))


@wp.kernel
def _inside_mesh_reduce_kernel(
    aabb_idx: wp.array(dtype=wp.int32),  # (N,) Inside-N → AABB-N
    prefilter: wp.array3d(dtype=wp.int32),  # (S, N, N)
    mesh_container_idx: wp.array(dtype=wp.int32),  # (M,)
    mesh_scene_idx: wp.array(dtype=wp.int32),  # (M,)
    outside_flag: wp.array3d(dtype=wp.int32),  # (S, N, M) 1 iff some face said outside
    pair_scratch: wp.array3d(dtype=wp.int32),  # (S, N, N) atomic_max target
):
    """
    Per (scene, inner_i, mesh m):
    if all the halfspace tests passed (outside_flag == 0) and the same gates as the test
    kernel hold, atomic_max pair_scratch[s, i, container(m)] to 1.
    """
    s, i, m = wp.tid()

    container_j = mesh_container_idx[m]
    if container_j < 0:
        return
    if mesh_scene_idx[m] != s:
        return
    if i == container_j:
        return

    a_i = aabb_idx[i]
    if a_i < 0:
        return

    if prefilter[s, i, container_j] == wp.int32(0):
        return

    if outside_flag[s, i, m] == wp.int32(0):
        wp.atomic_max(pair_scratch, s, i, container_j, wp.int32(1))


@wp.kernel
def _inside_finalize_kernel(
    pair: wp.array3d(dtype=wp.int32),  # (S, N, N)
    values: wp.array3d(dtype=wp.uint8),  # (S, N, N) uint8 view of bool VALUES
):
    s, i, j = wp.tid()
    if i == j:
        values[s, i, j] = wp.uint8(0)
        return
    if pair[s, i, j] > wp.int32(0):
        values[s, i, j] = wp.uint8(1)
    else:
        values[s, i, j] = wp.uint8(0)


class Inside(TensorizedRelativeState, KinematicsMixin, BooleanStateMixin):
    # Used by _inside_aabb_prefilter_kernel and the halfspace_test/mesh_reduce kernels to read
    # inner AABB centers and container AABBs from AABB.VALUES_WP.
    _aabb_idx = None  # (N,) int32 torch  — keep-alive owner of the GPU storage
    _aabb_idx_wp = None  # (N,) int32 wp.array view — kernel input

    # All container-meta-link visual meshes across every scene are flattened into
    # one global table of length M. M = total number of containers scross scenes

    # Which *container* object that owns this mesh, or -1 if
    # the parent object isn't Inside-tracked.
    # halfspace_test/mesh_reduce kernels use this to pick the column
    # (s, _, container_j) into which mesh_reduce atomically writes.
    _mesh_container_idx = None  # (M,) int32 torch
    _mesh_container_idx_wp = None

    # Scene index of the container that owns this mesh.
    # halfspace_test/mesh_reduce kernels skip meshes in other scenes.
    _mesh_scene_idx = None  # (M,) int32 torch
    _mesh_scene_idx_wp = None

    # RigidBodyViewAPI flat index of the parent link. inv_world_kernel indexes POSE_MATRICES
    # with this to fetch the current world pose for inv-world composition.
    _mesh_parent_link = None  # (M,) int32 torch
    _mesh_parent_link_wp = None

    # Static matrix that can transform a point in parent-link-frame into mesh-local-unscaled frame
    # inv_world_kernel then multiply it with rigid_inverse_mat44(parent_link_world) each step to get
    # the full world to mesh-local inverse.
    _mesh_inv_local_w_scale = None  # (M, 4, 4) float32 torch
    _mesh_inv_local_w_scale_wp = None

    # Reverse lookup: which mesh does each face belong to. halfspace_test_kernel uses this to
    # resolve face f → mesh m → all per-mesh state (inv_world, container, scene).
    _face_to_mesh = None  # (F_total,) int32 torch
    _face_to_mesh_wp = None

    # Flat per-face data for all container meshes, concatenated end-to-end. Both are in
    # mesh-local-unscaled frame (the frame halfspace_test_kernel transforms its query point
    # into via inv_world[m]). The halfspace test is (p_local - centroid) · normal < 0.
    _face_centroid = None  # (F_total, 3) float32 torch (wrapped as wp.vec3)
    _face_centroid_wp = None
    _face_normal = None  # (F_total, 3) float32 torch (wrapped as wp.vec3)
    _face_normal_wp = None

    # Below are scratches used by kernels

    # Scratch used by inv_world_kernel
    # Per-mesh inverse "world → mesh-local-unscaled" transform.
    # Written by inv_world_kernel each step; consumed by halfspace_test_kernel.
    _inv_world = None  # (M, 4, 4) float32 torch (wrapped as wp.mat44)
    _inv_world_wp = None

    # Scratch written by aabb_prefilter_kernel (1 if inner_i's AABB center
    # lies inside container_j's AABB, else 0). Consumed by halfspace_test/mesh_reduce kernels
    # as an early-exit gate. Meshes whose container failed the AABB check are skipped entirely.
    _prefilter = None  # (S, N, N) int32 torch
    _prefilter_wp = None

    # Per-mesh "saw at least one failing halfspace" flag, atomic_max target for halfspace_test_kernel.
    # Zeroed each step; mesh_reduce_kernel reads `outside_flag == 0` as "inner_i is inside mesh m".
    _outside_flag = None  # (S, N, M) int32 torch
    _outside_flag_wp = None

    # atomic_max target for "any mesh of container_j contains inner_i's center".
    # Zeroed at the top of _update_values, written by mesh_reduce_kernel, read by finalize_kernel
    # which converts it (int32 → uint8) into VALUES.
    _pair_scratch = None  # (S, N, N) int32 torch
    _pair_scratch_wp = None

    @classproperty
    def value_shape(cls):
        return ()

    @classproperty
    def value_type(cls):
        return th.bool

    @classproperty
    def value_name(cls):
        return "inside"

    @classmethod
    def get_dependencies(cls):
        deps = super().get_dependencies()
        deps.add(AABB)
        return deps

    @classmethod
    def global_initialize(cls):
        super().global_initialize()
        cls._aabb_idx = None
        cls._aabb_idx_wp = None
        cls._mesh_container_idx = None
        cls._mesh_container_idx_wp = None
        cls._mesh_scene_idx = None
        cls._mesh_scene_idx_wp = None
        cls._mesh_parent_link = None
        cls._mesh_parent_link_wp = None
        cls._mesh_inv_local_w_scale = None
        cls._mesh_inv_local_w_scale_wp = None
        cls._face_to_mesh = None
        cls._face_to_mesh_wp = None
        cls._face_centroid = None
        cls._face_centroid_wp = None
        cls._face_normal = None
        cls._face_normal_wp = None
        cls._inv_world = None
        cls._inv_world_wp = None
        cls._prefilter = None
        cls._prefilter_wp = None
        cls._outside_flag = None
        cls._outside_flag_wp = None
        cls._pair_scratch = None
        cls._pair_scratch_wp = None

    @classmethod
    def initialize_view(cls):
        super().initialize_view()
        S = len(cls.IDX_OBJS)
        N = len(cls.OBJ_IDXS)

        if S == 0 or N == 0:
            cls._aabb_idx_wp = None
            cls._inv_world_wp = None
            cls._prefilter_wp = None
            cls._pair_scratch_wp = None
            return

        # Build Inside-N → AABB-N
        aabb_idx = th.full((N,), -1, dtype=th.int32)
        aabb_map = AABB.OBJ_IDXS or {}
        for rel_path, idx in cls.OBJ_IDXS.items():
            aabb_idx[idx] = aabb_map.get(rel_path, -1)
        cls._aabb_idx = aabb_idx.cuda()
        cls._aabb_idx_wp = wp.from_torch(cls._aabb_idx)

        # Walk every Inside-tracked object's container meta-links and collect each visual mesh.
        # Only USD Mesh-typed visual meshes are supported; primitive types are skipped.
        mesh_records = []  # list of dicts; rolled into the flat tables below.
        face_centroids_list = []  # CPU torch tensors, concatenated at the end
        face_normals_list = []
        face_to_mesh_list = []  # CPU ints; for each face f appends the index of its owning mesh in mesh_records.

        for scene_idx, scene_row in enumerate(cls.IDX_OBJS):
            for container_idx, container_obj in enumerate(scene_row):
                if container_obj is None:
                    continue
                if container_obj.prim_type == PrimType.CLOTH:
                    continue
                for link in container_obj.links.values():
                    if not link.is_meta_link:
                        continue
                    if link.meta_link_type not in contains_m.CONTAINER_META_LINK_TYPES:
                        continue
                    parent_flat = RigidBodyViewAPI.get_flat_idx(link.prim_path)
                    if parent_flat is None:
                        continue
                    # Link's rigid world transform at init time, used to derive the static
                    # mesh→link-local-w-scale piece.
                    link_world_init = T.pose2mat(link.get_position_orientation())
                    link_world_inv_init = th.linalg.inv(link_world_init)

                    for mesh in link.visual_meshes.values():
                        if mesh._mesh_type != "Mesh":
                            continue

                        centroids = mesh.mesh_face_centroids  # (F, 3) local-unscaled
                        normals = mesh.mesh_face_normals  # (F, 3) local-unscaled
                        face_count = centroids.shape[0]
                        if face_count == 0:
                            continue

                        mesh_scaled_world_init = mesh.scaled_transform
                        local_w_scale = link_world_inv_init @ mesh_scaled_world_init
                        inv_local_w_scale = th.linalg.inv(local_w_scale)

                        mesh_idx = len(mesh_records)
                        face_centroids_list.append(centroids.to(th.float32))
                        face_normals_list.append(normals.to(th.float32))
                        face_to_mesh_list.extend([mesh_idx] * face_count)

                        mesh_records.append(
                            {
                                "container": container_idx,
                                "scene": scene_idx,
                                "parent_link": parent_flat,
                                "inv_local_w_scale": inv_local_w_scale.to(th.float32),
                            }
                        )

        M = len(mesh_records)
        if M == 0:
            cls._mesh_container_idx_wp = None
            cls._mesh_scene_idx_wp = None
            cls._mesh_parent_link_wp = None
            cls._mesh_inv_local_w_scale_wp = None
            cls._face_to_mesh_wp = None
            cls._face_centroid_wp = None
            cls._face_normal_wp = None
            cls._inv_world_wp = None
            cls._outside_flag_wp = None
        else:
            cls._mesh_container_idx = th.tensor([r["container"] for r in mesh_records], dtype=th.int32, device="cuda")
            cls._mesh_scene_idx = th.tensor([r["scene"] for r in mesh_records], dtype=th.int32, device="cuda")
            cls._mesh_parent_link = th.tensor([r["parent_link"] for r in mesh_records], dtype=th.int32, device="cuda")
            inv_local_stack = th.stack([r["inv_local_w_scale"] for r in mesh_records]).cuda()
            cls._mesh_inv_local_w_scale = inv_local_stack

            cls._mesh_container_idx_wp = wp.from_torch(cls._mesh_container_idx)
            cls._mesh_scene_idx_wp = wp.from_torch(cls._mesh_scene_idx)
            cls._mesh_parent_link_wp = wp.from_torch(cls._mesh_parent_link)
            cls._mesh_inv_local_w_scale_wp = wp.from_torch(cls._mesh_inv_local_w_scale, dtype=wp.mat44)

            # Flat face arrays. M > 0 here and every retained mesh contributed at least one
            # face (zero-face meshes are filtered above), so F_total > 0.
            face_centroids_flat = th.cat(face_centroids_list, dim=0).cuda().contiguous()
            face_normals_flat = th.cat(face_normals_list, dim=0).cuda().contiguous()
            cls._face_centroid = face_centroids_flat
            cls._face_normal = face_normals_flat
            cls._face_centroid_wp = wp.from_torch(face_centroids_flat, dtype=wp.vec3)
            cls._face_normal_wp = wp.from_torch(face_normals_flat, dtype=wp.vec3)

            cls._face_to_mesh = th.tensor(face_to_mesh_list, dtype=th.int32, device="cuda")
            cls._face_to_mesh_wp = wp.from_torch(cls._face_to_mesh)

            # Per-step scratch
            cls._inv_world = th.zeros((M, 4, 4), dtype=th.float32, device="cuda")
            cls._inv_world_wp = wp.from_torch(cls._inv_world, dtype=wp.mat44)

            cls._outside_flag = th.zeros((S, N, M), dtype=th.int32, device="cuda")
            cls._outside_flag_wp = wp.from_torch(cls._outside_flag)

        cls._prefilter = th.zeros((S, N, N), dtype=th.int32, device="cuda")
        cls._prefilter_wp = wp.from_torch(cls._prefilter)
        cls._pair_scratch = th.zeros((S, N, N), dtype=th.int32, device="cuda")
        cls._pair_scratch_wp = wp.from_torch(cls._pair_scratch)

    @classmethod
    def _update_values(cls, values):
        if (
            cls.VALUES_WP is None
            or cls._aabb_idx_wp is None
            or cls._prefilter_wp is None
            or cls._pair_scratch_wp is None
            or AABB.VALUES_WP is None
        ):
            return
        S, N, _ = values.shape
        if S == 0 or N == 0:
            return

        cls._pair_scratch_wp.zero_()

        # refresh per-mesh world→local inverse from current link poses.
        if cls._mesh_parent_link_wp is not None and RigidBodyViewAPI.POSE_MATRICES is not None:
            M = cls._mesh_parent_link.shape[0]
            wp.launch(
                kernel=_inside_inv_world_kernel,
                dim=M,
                inputs=[
                    RigidBodyViewAPI.POSE_MATRICES,
                    cls._mesh_parent_link_wp,
                    cls._mesh_inv_local_w_scale_wp,
                    cls._inv_world_wp,
                ],
                device="cuda",
            )

        wp.launch(
            kernel=_inside_aabb_prefilter_kernel,
            dim=(S, N, N),
            inputs=[AABB.VALUES_WP, cls._aabb_idx_wp, cls._prefilter_wp],
            device="cuda",
        )

        if cls._mesh_container_idx_wp is not None:
            M = cls._mesh_parent_link.shape[0]
            F_total = cls._face_centroid.shape[0]

            # Each face independently votes "outside" via atomic_max into outside_flag.
            # Mesh "contains" iff no face voted outside → reduce kernel writes pair_scratch.
            cls._outside_flag_wp.zero_()
            wp.launch(
                kernel=_inside_halfspace_test_kernel,
                dim=(S, N, F_total),
                inputs=[
                    AABB.VALUES_WP,
                    cls._aabb_idx_wp,
                    cls._prefilter_wp,
                    cls._inv_world_wp,
                    cls._face_to_mesh_wp,
                    cls._mesh_container_idx_wp,
                    cls._mesh_scene_idx_wp,
                    cls._face_centroid_wp,
                    cls._face_normal_wp,
                    cls._outside_flag_wp,
                ],
                device="cuda",
            )
            wp.launch(
                kernel=_inside_mesh_reduce_kernel,
                dim=(S, N, M),
                inputs=[
                    cls._aabb_idx_wp,
                    cls._prefilter_wp,
                    cls._mesh_container_idx_wp,
                    cls._mesh_scene_idx_wp,
                    cls._outside_flag_wp,
                    cls._pair_scratch_wp,
                ],
                device="cuda",
            )

        # int32 pair to uint8 VALUES, zero diagonal
        wp.launch(
            kernel=_inside_finalize_kernel,
            dim=(S, N, N),
            inputs=[cls._pair_scratch_wp, cls.VALUES_WP],
            device="cuda",
        )

    def _get_value(self, other):
        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot detect if an object is inside a cloth object.")

        return super()._get_value(other)

    def _set_value(self, other, new_value, reset_before_sampling=False, use_trav_map=False):
        """
        Set the Inside state for this object with respect to another object (container).

        This samples a random position inside the container's fillable volume, places the object,
        lets it settle via physics, and verifies it's still inside.

        The sampling strategy uses a two-phase approach:
        1. First half of attempts: Sample from inset AABB (container bounds minus object extent).
           This ensures the sampled position is at the object's centroid, not near the edges,
           which makes it more likely to fit inside.
        2. Second half of attempts: Sample from full container AABB. This is a fallback for
           cases where the object is too large to fit entirely within the inset bounds.

        Each candidate pose passes through several rejection-sampling stages:
        1. The sampled point must lie inside the container's fillable volume.
        2. After placement and a single physics step, the object must not already be
           intersecting anything (catches interpenetration with container walls).
        3. The object must come into contact with something after placement and half a second of
           physics steps.
        4. After settling, the container's root pose must not have moved
           beyond CONTAINER_POSITION_CHANGE_THRESHOLD / CONTAINER_ORIENTATION_CHANGE_THRESHOLD.
        5. The container's articulated joints must not have moved beyond the per-DOF-type
           CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_{TRANSLATION,ROTATION} thresholds
           (catches cases where the placed object swings a door/lid or pushes a drawer).
        6. The object must still register as Inside the container after settling, and
           reachable via the traversability map if use_trav_map is enabled.

        Args:
            other: The container object to place this object inside.
            new_value: True to set Inside state (only True is supported).
            reset_before_sampling: If True, reset this object before sampling.
            use_trav_map: Whether to use traversability-based reachability checks.
        Returns:
            True if successfully placed inside, False otherwise.
        """
        if not new_value:
            raise NotImplementedError("Inside does not support set_value(False)")

        if other.prim_type == PrimType.CLOTH:
            raise ValueError("Cannot set an object inside a cloth object.")

        # Save the initial position and orientation of the container
        container_pos_initial, container_orn_initial = other.get_position_orientation()
        container_joint_positions_initial = other.get_joint_positions() if other.n_joints > 0 else None

        # Find the container's fillable meta link (fillable or openfillable)
        container_link = None
        for link in other.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                container_link = link
                break

        assert container_link is not None, f"Container object {other.name} must have a fillable meta link"

        # Save simulator state for restoration on failed attempts
        state = og.sim.dump_state(serialized=False)

        if reset_before_sampling:
            self.obj.reset()

        # Get container's fillable volume bounds in world frame
        aabb_low, aabb_high = container_link.visual_aabb
        # Get the object extent to compute inset bounds
        obj_extent = self.obj.aabb_extent

        # Inset the container AABB by half the object extent in each dimension.
        # This ensures the sampled position (used as object centroid) won't place
        # any part of the object outside the container bounds.
        inset_aabb_low = aabb_low + obj_extent / 2.0
        inset_aabb_high = aabb_high - obj_extent / 2.0

        # Calculate the total attempt count. Here we don't have a sense of high/low-level attempts,
        # so to make the same numbr of attempts as the original implementation, we just multiply
        # the two sampling parameters.
        total_attempts = os_m.DEFAULT_HIGH_LEVEL_SAMPLING_ATTEMPTS * os_m.DEFAULT_LOW_LEVEL_SAMPLING_ATTEMPTS

        if use_trav_map:
            reachability_context = get_reachability_sampling_context(
                objB=other,
                predicate="inside",
                use_trav_map=use_trav_map,
                warn_on_scene_mismatch=False,
            )
            use_trav_map = reachability_context is not None

        for attempt_idx in range(total_attempts):
            # Sample orientation if the object supports random orientations, otherwise use default
            orientation = (
                self.obj.sample_orientation()
                if (hasattr(self.obj, "orientations") and self.obj.orientations is not None)
                else th.tensor([0, 0, 0, 1.0])
            )

            # Also add a random world Z-axis offset to the orientation
            random_z_orientation = T.axisangle2quat(th.as_tensor([0, 0, th.rand(1) * 2 * th.pi]))
            orientation = T.quat_multiply(orientation, random_z_orientation)

            # First half: use inset bounds (smarter sampling)
            # Second half: use full bounds (fallback for large objects)
            # Also fallback if inset bounds are invalid (object too large for container)
            if attempt_idx < total_attempts // 2 and th.all(inset_aabb_low < inset_aabb_high):
                pos = inset_aabb_low + th.rand(3) * (inset_aabb_high - inset_aabb_low)
            else:
                pos = aabb_low + th.rand(3) * (aabb_high - aabb_low)

            # Rejection sampling #1: Verify the sampled point is actually inside the container volume
            if not container_link.check_points_in_volume(pos.unsqueeze(0)).item():
                og.sim.load_state(state, serialized=False)
                continue

            # Add small z-offset to avoid spawning inside the container floor
            pos[2] += 0.01
            self.obj.set_position_orientation(position=pos, orientation=orientation)
            self.obj.keep_still()
            # Step physics once so the contact buffer gets populated for the newly-placed object
            og.sim.step_physics()

            # Rejection sampling #2: Reject if the object is already intersecting anything
            # immediately after placement (e.g. interpenetration with a container wall).
            if RigidContactAPI.is_in_contact(
                scene_idx=self.obj.scene.idx,
                query_set=[self.obj],
                with_set=None,
                ignore_set=None,
                current_only=True,
            ):
                og.sim.load_state(state, serialized=False)
                continue

            # Rejection sampling #3: step until contact is made or max steps reached
            # (0.5 seconds of sim time) to let the object settle onto a resting surface.
            # If it can't get into contact by then, we reject the placement.
            n_steps_max = int(0.5 / og.sim.get_physics_dt())
            for _ in range(n_steps_max):
                og.sim.step_physics()
                if RigidContactAPI.is_in_contact(
                    scene_idx=self.obj.scene.idx,
                    query_set=[self.obj],
                    with_set=None,
                    ignore_set=None,
                    current_only=True,
                ):
                    break
            else:
                og.sim.load_state(state, serialized=False)
                continue
            self.obj.keep_still()
            other.keep_still()

            # Step a few more times to let velocity stabilize
            for _ in range(5):
                og.sim.step_physics()
            settle_step_idx = 0
            while th.norm(self.obj.get_linear_velocity()) > 1e-3 and settle_step_idx < n_steps_max:
                og.sim.step_physics()
                settle_step_idx += 1

            # Rejection sampling #4: Reject if the container's root pose drifted past the
            # position/orientation thresholds (i.e. the placed object pushed the container).
            container_pos, container_orn = other.get_position_orientation()
            position_difference = th.norm(container_pos - container_pos_initial)
            orientation_difference = T.get_orientation_diff_in_radian(container_orn, container_orn_initial)
            if (
                position_difference > m.CONTAINER_POSITION_CHANGE_THRESHOLD
                or orientation_difference > m.CONTAINER_ORIENTATION_CHANGE_THRESHOLD
            ):
                og.sim.load_state(state, serialized=False)
                continue

            # Rejection sampling #5: Reject if any of the container's articulated joints moved
            # past the per-DOF-type delta thresholds (e.g. placed object swung a lid or pushed
            # a drawer). Thresholds are applied separately for rotational and translational DOFs.
            if container_joint_positions_initial is not None:
                container_joint_positions_final = other.get_joint_positions()
                joint_thresholds = th.where(
                    other.get_joint_dof_types(),
                    m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_ROTATION,
                    m.CONTAINER_JOINT_POSITION_DELTA_THRESHOLD_TRANSLATION,
                )
                container_joint_positions_delta = th.abs(
                    container_joint_positions_final - container_joint_positions_initial
                )
                if th.any(container_joint_positions_delta > joint_thresholds):
                    og.sim.load_state(state, serialized=False)
                    continue

            # Rejection sampling #6: Verify object is still inside after settling and within reach if using trav map.
            # The lazy-refresh gate in TensorizedState._get_value brings VALUES_CPU back in sync before this read.
            if self.get_value(other):
                if use_trav_map:
                    settled_pos, _ = self.obj.get_position_orientation()
                    if not is_pose_reachable_for_predicate(
                        pos=settled_pos,
                        objB=other,
                        predicate="inside",
                        reachability_context=reachability_context,
                    ):
                        og.sim.load_state(state, serialized=False)
                        continue
                return True

        # Reset the simulator state to the initial state
        og.sim.load_state(state, serialized=False)
        return False
