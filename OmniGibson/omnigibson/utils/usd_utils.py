import collections
import os
import re
from collections.abc import Iterable
from typing import Tuple

import numpy as np
import torch as th
import trimesh
import warp as wp
from numba import jit, prange

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.macros import gm
from omnigibson.utils.constants import PRIMITIVE_MESH_TYPES, JointType, PrimType
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.python_utils import assert_valid_key, torch_compile
from omnigibson.utils.ui_utils import create_module_logger, suppress_omni_log

# Create module logger
log = create_module_logger(module_name=__name__)


@wp.func
def rigid_inverse_mat44(T: wp.mat44) -> wp.mat44:
    """
    Inverse of a rigid transform `T = [R | t; 0 | 1]` with R orthogonal.
    Result: `[R^T | -R^T t; 0 | 1]`. Cheap to compute (transpose + dot products),
    avoids a general 4x4 inverse and works correctly under wp.graph capture.

    Used by _toggle_overlap_kernel (and any future kernel that needs to transform a world-space point into a link's
    local frame).
    """
    R = wp.mat33(
        T[0, 0],
        T[0, 1],
        T[0, 2],
        T[1, 0],
        T[1, 1],
        T[1, 2],
        T[2, 0],
        T[2, 1],
        T[2, 2],
    )
    t = wp.vec3(T[0, 3], T[1, 3], T[2, 3])
    Rt = wp.transpose(R)
    nt = -wp.mul(Rt, t)
    return wp.mat44(
        Rt[0, 0],
        Rt[0, 1],
        Rt[0, 2],
        nt[0],
        Rt[1, 0],
        Rt[1, 1],
        Rt[1, 2],
        nt[1],
        Rt[2, 0],
        Rt[2, 1],
        Rt[2, 2],
        nt[2],
        0.0,
        0.0,
        0.0,
        1.0,
    )


@wp.kernel
def _is_in_contact_batch_kernel(
    query_masks: wp.array2d(dtype=wp.uint8),  # (N, R)
    contact_matrix: wp.array2d(dtype=wp.uint8),  # (R, C)
    col_filter: wp.array2d(dtype=wp.uint8),  # (N, C) — with-mask if mode=0, ignore-mask if mode=1, ignored if mode=2
    mode: wp.int32,  # 0 = with_masks, 1 = ignore_masks, 2 = no filter
    out: wp.array(dtype=wp.int32),  # (N,) — atomic-OR'd, int32 because uint8 atomics aren't supported
):
    """
    Per (query, row, col) thread: full N*R*C parallelism. Each thread checks one cell:
      if query_masks[i, r] AND contact_matrix[r, c] AND col_filter passes:
          atomic-set out[i] to 1.

    Output is int32 because Warp's atomic_max requires [u]int32, [u]int64, float32, or
    float64 (CUDA doesn't expose 8-bit atomics natively). Same-value atomic_max writes
    short-circuit on Pascal+ hardware, so contention on out[i] is bounded — only the
    first hitting thread per query actually does meaningful work; subsequent writes are
    NOPs at the memory subsystem.

    Caller must zero `out` before launch (see is_in_contact_batch_warp). The atomic only
    sets-to-1 on hits; cells with no hits remain at their initial value.
    """
    i, r, c = wp.tid()

    # Column filter — early exit before touching query_masks/contact_matrix.
    if mode == wp.int32(0):  # with-mask: only count if filter is True
        if col_filter[i, c] == wp.uint8(0):
            return
    elif mode == wp.int32(1):  # ignore-mask: only count if filter is False
        if col_filter[i, c] != wp.uint8(0):
            return
    # mode == 2: no filter, fall through

    # Row mask — does query i include row r?
    if query_masks[i, r] == wp.uint8(0):
        return

    # Contact lookup.
    if contact_matrix[r, c] == wp.uint8(0):
        return

    # Hit. Set out[i] = 1. atomic_max because multiple (i, *, *) threads may converge
    # on the same out[i]. Same-value short-circuit keeps contention cheap.
    wp.atomic_max(out, i, wp.int32(1))


@wp.func
def _body_awake_at_step(
    i: wp.int32,
    b: wp.int32,
    all_transforms: wp.array3d(dtype=wp.float32),
    prev_transforms: wp.array2d(dtype=wp.float32),
    body_to_row: wp.array(dtype=wp.int32),
    net_forces: wp.array3d(dtype=wp.float32),
    pos_eps: wp.float32,
    ori_eps: wp.float32,
) -> wp.bool:
    """
    Returns True iff body ``b`` was awake at physics-step ``i``.

    A body is awake if either its position/orientation changed since the previous step,
    or — for bodies that map to a contact-matrix row — its net contact force is nonzero.
    For step ``i == 0`` the predecessor is the cached ``prev_transforms`` (the
    ``_BODY_TRANSFORMS`` snapshot from the previous update), otherwise it's
    ``all_transforms[i-1, b]``.

    Shared by ``_update_contact_matrices_kernel`` and ``_update_body_transforms_kernel``
    so both kernels see the exact same awakeness predicate.
    """
    if i == 0:
        px0 = prev_transforms[b, 0]
        py0 = prev_transforms[b, 1]
        pz0 = prev_transforms[b, 2]
        qx0 = prev_transforms[b, 3]
        qy0 = prev_transforms[b, 4]
        qz0 = prev_transforms[b, 5]
        qw0 = prev_transforms[b, 6]
    else:
        px0 = all_transforms[i - 1, b, 0]
        py0 = all_transforms[i - 1, b, 1]
        pz0 = all_transforms[i - 1, b, 2]
        qx0 = all_transforms[i - 1, b, 3]
        qy0 = all_transforms[i - 1, b, 4]
        qz0 = all_transforms[i - 1, b, 5]
        qw0 = all_transforms[i - 1, b, 6]
    px1 = all_transforms[i, b, 0]
    py1 = all_transforms[i, b, 1]
    pz1 = all_transforms[i, b, 2]
    qx1 = all_transforms[i, b, 3]
    qy1 = all_transforms[i, b, 4]
    qz1 = all_transforms[i, b, 5]
    qw1 = all_transforms[i, b, 6]

    pos_changed = wp.abs(px1 - px0) > pos_eps or wp.abs(py1 - py0) > pos_eps or wp.abs(pz1 - pz0) > pos_eps
    qdot = qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1
    ori_changed = wp.abs(qdot) < (wp.float32(1.0) - ori_eps)
    awake = pos_changed or ori_changed

    r = body_to_row[b]
    if r >= 0:
        fx = net_forces[i, r, 0]
        fy = net_forces[i, r, 1]
        fz = net_forces[i, r, 2]
        if fx != 0.0 or fy != 0.0 or fz != 0.0:
            awake = True
    return awake


@wp.kernel
def _update_contact_matrices_kernel(
    all_transforms: wp.array3d(dtype=wp.float32),  # (N, B, 7)
    prev_transforms: wp.array2d(dtype=wp.float32),  # (B, 7) — cached BODY_TRANSFORMS
    net_forces: wp.array3d(dtype=wp.float32),  # (N, R, 3)
    body_to_row: wp.array(dtype=wp.int32),  # (B,) -1 if body is not a row body
    impulses: wp.array4d(dtype=wp.float32),  # (N, R, C, 3)
    row_to_rigid: wp.array(dtype=wp.int32),  # (R,) row index → body index
    col_to_rigid: wp.array(dtype=wp.int32),  # (C,) col index → body index, -1 = kinematic
    n_steps_arr: wp.array(dtype=wp.int32),  # (1,) read at runtime, NOT a captured constant
    pos_eps: wp.float32,
    ori_eps: wp.float32,
    contact_matrix: wp.array2d(dtype=wp.uint8),  # (R, C) in/out — "any contact during recent steps"
    current_contact_matrix: wp.array2d(dtype=wp.uint8),  # (R, C) in/out — "contact at most recent awake step"
):
    """
    Per (r, c) thread: walks ``n_steps_arr[0]`` physics sub-steps, evaluating pair awakeness
    inline via ``_body_awake_at_step``. Replaces the entire torch pipeline in the old
    ``RigidContactAPIImpl.update()``:

      - ``th.cat`` of prev+all_transforms → folded into the i==0 base case in _body_awake_at_step.
      - ``th.where(per_step_awake, idx, -1).max(dim=0)`` → tracked as a scalar ``last_awake`` per thread.
      - ``th.any(impulses != 0, dim=-1)`` and the masked indexed writes →
        scalar OR over the awake sub-steps, single write at the end.

    ``n_steps`` comes from ``n_steps_arr[0]`` (a 1-element GPU array refreshed from Python
    before each graph launch) rather than a captured constant. This matches the old code's
    ``all_impulses[:self._PENDING_STEPS]`` slicing — when ``og.sim.step_physics()`` is used
    (e.g. inside ``sample_kinematics``), only 1 < n_physics_timesteps_per_render sub-step
    has actually run, and the remaining pending slots hold stale data from previous frames.

    Pairs that were never awake retain ``current_contact_matrix[r, c]`` and copy it into
    ``contact_matrix[r, c]``, matching the torch behavior at the end of the original update().
    """
    r, c = wp.tid()
    n_steps = n_steps_arr[0]
    row_b = row_to_rigid[r]
    col_b = col_to_rigid[c]
    has_col_b = col_b >= 0

    last_awake = wp.int32(-1)
    any_contact = wp.uint8(0)
    for i in range(n_steps):
        row_awake = _body_awake_at_step(
            i, row_b, all_transforms, prev_transforms, body_to_row, net_forces, pos_eps, ori_eps
        )
        col_awake = False
        if has_col_b:
            col_awake = _body_awake_at_step(
                i, col_b, all_transforms, prev_transforms, body_to_row, net_forces, pos_eps, ori_eps
            )
        if row_awake or col_awake:
            last_awake = i
            if impulses[i, r, c, 0] != 0.0 or impulses[i, r, c, 1] != 0.0 or impulses[i, r, c, 2] != 0.0:
                any_contact = wp.uint8(1)

    if last_awake >= 0:
        cur = wp.uint8(0)
        if (
            impulses[last_awake, r, c, 0] != 0.0
            or impulses[last_awake, r, c, 1] != 0.0
            or impulses[last_awake, r, c, 2] != 0.0
        ):
            cur = wp.uint8(1)
        current_contact_matrix[r, c] = cur
        contact_matrix[r, c] = any_contact
    else:
        # Pair was never awake — both matrices collapse to the (carried-over) current value.
        contact_matrix[r, c] = current_contact_matrix[r, c]


@wp.kernel
def _update_body_transforms_kernel(
    all_transforms: wp.array3d(dtype=wp.float32),  # (N, B, 7)
    prev_transforms: wp.array2d(dtype=wp.float32),  # (B, 7) — same buffer as body_transforms below
    net_forces: wp.array3d(dtype=wp.float32),  # (N, R, 3)
    body_to_row: wp.array(dtype=wp.int32),  # (B,)
    n_steps_arr: wp.array(dtype=wp.int32),  # (1,) read at runtime, shared with the contact kernel
    pos_eps: wp.float32,
    ori_eps: wp.float32,
    body_transforms: wp.array2d(dtype=wp.float32),  # (B, 7) in/out — snapshot of last awake step per body
):
    """
    Per (b,) thread: snapshots each body's transform from its last awake step.

    MUST be launched *after* ``_update_contact_matrices_kernel`` because both kernels
    use ``body_transforms`` as the ``prev_transforms`` ``_body_awake_at_step`` reads for
    ``i == 0``, and this kernel writes to it in-place. Same-stream ordering inside the
    captured graph guarantees this.
    """
    b = wp.tid()
    n_steps = n_steps_arr[0]
    last_awake = wp.int32(-1)
    for i in range(n_steps):
        if _body_awake_at_step(i, b, all_transforms, prev_transforms, body_to_row, net_forces, pos_eps, ori_eps):
            last_awake = i
    if last_awake >= 0:
        for k in range(7):
            body_transforms[b, k] = all_transforms[last_awake, b, k]


def ensure_usd_api(prim, api):
    """
    Ensures that a USD API schema is applied to a prim. If the prim already has the API,
    returns the existing wrapper. Otherwise, applies the API inside an editing_usd() context
    (triggering a USD-to-Fabric sync) and returns the newly applied wrapper.

    Args:
        prim (Usd.Prim): The prim to check / apply the API on
        api: The USD API class (e.g. lazy.pxr.UsdPhysics.RigidBodyAPI)

    Returns:
        The API wrapper for the prim
    """
    if prim.HasAPI(api):
        return api(prim)
    with og.sim.editing_usd():
        return api.Apply(prim)


def array_to_vtarray(arr, element_type):
    """
    Converts array @arr into a Vt-typed array, where each individual element of type @element_type.

    Args:
        arr (n-array): An array of values. Can be, e.g., a list, or numpy array
        element_type (type): Per-element type to convert the elements from @arr into.
            Valid options are keys of GF_TO_VT_MAPPING

    Returns:
        Vt.Array: Vt-typed array, of specified type corresponding to @element_type
    """
    GF_TO_VT_MAPPING = {
        lazy.pxr.Gf.Vec3d: lazy.pxr.Vt.Vec3dArray,
        lazy.pxr.Gf.Vec3f: lazy.pxr.Vt.Vec3fArray,
        lazy.pxr.Gf.Vec3h: lazy.pxr.Vt.Vec3hArray,
        lazy.pxr.Gf.Quatd: lazy.pxr.Vt.QuatdArray,
        lazy.pxr.Gf.Quatf: lazy.pxr.Vt.QuatfArray,
        lazy.pxr.Gf.Quath: lazy.pxr.Vt.QuathArray,
        int: lazy.pxr.Vt.IntArray,
        float: lazy.pxr.Vt.FloatArray,
        bool: lazy.pxr.Vt.BoolArray,
        str: lazy.pxr.Vt.StringArray,
        chr: lazy.pxr.Vt.CharArray,
    }

    # Make sure array type is valid
    assert_valid_key(key=element_type, valid_keys=GF_TO_VT_MAPPING, name="array element type")

    # Construct list of values
    arr_list = []

    # Check first to see if elements are vectors or not. If this is an iterable value that is not a string,
    # then this is a vector and we have to map it to the correct type via *
    is_vec_element = (isinstance(arr[0], Iterable)) and (not isinstance(arr[0], str))

    # Loop over array and set values
    for ele in arr:
        arr_list.append(element_type(*ele) if is_vec_element else ele)

    return GF_TO_VT_MAPPING[element_type](arr_list)


def get_prim_nested_children(prim):
    """
    Grabs all nested prims starting from root @prim via depth-first-search

    Args:
        prim (Usd.Prim): root prim from which to search for nested children prims

    Returns:
        list of Usd.Prim: nested prims
    """
    prims = []
    for child in lazy.isaacsim.core.utils.prims.get_prim_children(prim):
        prims.append(child)
        prims += get_prim_nested_children(prim=child)

    return prims


def create_joint(
    prim_path,
    joint_type,
    body0=None,
    body1=None,
    enabled=True,
    exclude_from_articulation=False,
    joint_frame_in_parent_frame_pos=None,
    joint_frame_in_parent_frame_quat=None,
    joint_frame_in_child_frame_pos=None,
    joint_frame_in_child_frame_quat=None,
    break_force=None,
    break_torque=None,
    stage=None,
):
    """
    Creates a joint between @body0 and @body1 of specified type @joint_type

    Args:
        prim_path (str): absolute path to where the joint will be created
        joint_type (str or JointType): type of joint to create. Valid options are:
            "FixedJoint", "Joint", "PrismaticJoint", "RevoluteJoint", "SphericalJoint"
                        (equivalently, one of JointType)
        body0 (str or None): absolute path to the first body's prim. At least @body0 or @body1 must be specified.
        body1 (str or None): absolute path to the second body's prim. At least @body0 or @body1 must be specified.
        enabled (bool): whether to enable this joint or not.
        exclude_from_articulation (bool): whether to exclude this joint from the articulation or not.
        joint_frame_in_parent_frame_pos (th.tensor or None): relative position of the joint frame to the parent frame (body0).
        joint_frame_in_parent_frame_quat (th.tensor or None): relative orientation of the joint frame to the parent frame (body0).
        joint_frame_in_child_frame_pos (th.tensor or None): relative position of the joint frame to the child frame (body1).
        joint_frame_in_child_frame_quat (th.tensor or None): relative orientation of the joint frame to the child frame (body1).
        break_force (float or None): break force for linear dofs, unit is Newton.
        break_torque (float or None): break torque for angular dofs, unit is Newton-meter.
        stage (None or Usd.Stage): If specified, stage on which the joint should be created. If None, will use og.sim.stage

    Returns:
        Usd.Prim: Created joint prim
    """
    with og.sim.editing_usd(stage=stage):
        current_stage = stage or og.sim.stage
        # Make sure we have valid joint_type
        assert JointType.is_valid(joint_type=joint_type), f"Invalid joint specified for creation: {joint_type}"

        # Make sure at least body0 or body1 is specified
        assert (
            body0 is not None or body1 is not None
        ), "At least either body0 or body1 must be specified when creating a joint!"

        # Create the joint
        joint = getattr(lazy.pxr.UsdPhysics, joint_type).Define(current_stage, prim_path)

        # Possibly add body0, body1 targets
        if body0 is not None:
            assert current_stage.GetPrimAtPath(body0).IsValid(), f"Invalid body0 path specified: {body0}"
            joint.GetBody0Rel().SetTargets([lazy.pxr.Sdf.Path(body0)])
        if body1 is not None:
            assert current_stage.GetPrimAtPath(body1).IsValid(), f"Invalid body1 path specified: {body1}"
            joint.GetBody1Rel().SetTargets([lazy.pxr.Sdf.Path(body1)])

        # Get the prim pointed to at this path
        joint_prim = current_stage.GetPrimAtPath(prim_path)

    # Apply joint API interface
    ensure_usd_api(joint_prim, lazy.pxr.PhysxSchema.PhysxJointAPI)

    # We need to step rendering once to auto-fill the local pose before overwriting it.
    # Note that for some reason, if multi_gpu is used, this line will crash if create_joint is called during on_contact
    # callback, e.g. when an attachment joint is being created due to contacts.
    # TODO(#2082): Is this necessary? Can it be removed altogether or replaced with a refresh?
    if stage is None:
        og.sim.render()

    with og.sim.editing_usd(stage=stage):
        if joint_frame_in_parent_frame_pos is not None:
            joint_prim.GetAttribute("physics:localPos0").Set(
                lazy.pxr.Gf.Vec3f(*joint_frame_in_parent_frame_pos.tolist())
            )
        if joint_frame_in_parent_frame_quat is not None:
            joint_prim.GetAttribute("physics:localRot0").Set(
                lazy.pxr.Gf.Quatf(*joint_frame_in_parent_frame_quat[[3, 0, 1, 2]].tolist())
            )
        if joint_frame_in_child_frame_pos is not None:
            joint_prim.GetAttribute("physics:localPos1").Set(
                lazy.pxr.Gf.Vec3f(*joint_frame_in_child_frame_pos.tolist())
            )
        if joint_frame_in_child_frame_quat is not None:
            joint_prim.GetAttribute("physics:localRot1").Set(
                lazy.pxr.Gf.Quatf(*joint_frame_in_child_frame_quat[[3, 0, 1, 2]].tolist())
            )

        if break_force is not None:
            joint_prim.GetAttribute("physics:breakForce").Set(break_force)
        if break_torque is not None:
            joint_prim.GetAttribute("physics:breakTorque").Set(break_torque)

        # Possibly (un-/)enable this joint
        joint_prim.GetAttribute("physics:jointEnabled").Set(enabled)

        # Possibly exclude this joint from the articulation
        joint_prim.GetAttribute("physics:excludeFromArticulation").Set(exclude_from_articulation)

    # Update handles to include the new joint
    og.sim.update_handles()

    # Return this joint
    return joint_prim


@wp.kernel
def _aabb_reduce_kernel(
    pose_matrices: wp.array(dtype=wp.mat44),  # (N_links_total,)
    mesh_ids: wp.array(dtype=wp.uint64),  # (N_links_total,) — 0 if link has no mesh
    aabb_links: wp.array(dtype=wp.int32),  # (K,) k → which link this thread belongs to
    aabb_vertices: wp.array(dtype=wp.int32),  # (K,) k → which vertex within that link's mesh
    aabb_objs: wp.array(dtype=wp.int32),  # (K,) k → output row (= s*O + obj_idx)
    out_values: wp.array2d(dtype=wp.float32),  # (S*O, 6)
):
    # K = total number of link vertices that have AABB state = total number of threads.
    # Each thread processes ONE (link, vertex) pair. Three K-length lookup tables tell
    # thread k where to read the point from and where to write its contribution.
    #
    # Example with K = 5, two tracked links A (3 verts → object 5) and B (2 verts → object 7):
    #
    #   thread index k:    0    1    2    3    4
    #                     ─────────────────────────
    #   aabb_links:      [ A ,  A ,  A ,  B ,  B ]   ← which link I belong to
    #   aabb_vertices:   [ 0 ,  1 ,  2 ,  0 ,  1 ]   ← which vertex within that link
    #   aabb_objs:       [ 5 ,  5 ,  5 ,  7 ,  7 ]   ← which object's AABB I write to
    k = wp.tid()
    body = aabb_links[k]
    # NOTE: wp.mesh_get_point(id, i) is a *face-vertex* lookup (returns mesh.points[mesh.indices[i]]),
    # not a vertex lookup. We want vertex i directly, so we read mesh.points[i] via wp.mesh_get(id).
    mesh = wp.mesh_get(mesh_ids[body])
    pt3 = mesh.points[aabb_vertices[k]]
    pt4 = wp.vec4(pt3[0], pt3[1], pt3[2], 1.0)
    world = wp.mul(pose_matrices[body], pt4)
    obj = aabb_objs[k]
    wp.atomic_min(out_values, obj, 0, world[0])
    wp.atomic_min(out_values, obj, 1, world[1])
    wp.atomic_min(out_values, obj, 2, world[2])
    wp.atomic_max(out_values, obj, 3, world[0])
    wp.atomic_max(out_values, obj, 4, world[1])
    wp.atomic_max(out_values, obj, 5, world[2])


@wp.kernel
def _aabb_baselink_fallback_kernel(
    poses: wp.array2d(dtype=wp.float32),  # POSES_GPU as (N_links_total, 7)
    base_link_links: wp.array(dtype=wp.int32),  # (N_rigid,) flat body idx of each rigid obj's base link
    base_link_objs: wp.array(dtype=wp.int32),  # (N_rigid,) output row in out_values
    out_values: wp.array2d(dtype=wp.float32),  # (S*O, 6) — populated by _aabb_reduce_kernel
):
    # Each thread = one rigid object's base link. If the main kernel never touched this
    # obj's row (still +inf at lo_x), no link had collision geometry — write a point AABB
    # at the base link's world position. Each rigid object has exactly one base link, so
    # threads never share an obj — non-atomic writes are safe.
    n = wp.tid()
    obj = base_link_objs[n]
    if wp.isinf(out_values[obj, 0]):
        body = base_link_links[n]
        x = poses[body, 0]
        y = poses[body, 1]
        z = poses[body, 2]
        out_values[obj, 0] = x
        out_values[obj, 1] = y
        out_values[obj, 2] = z
        out_values[obj, 3] = x
        out_values[obj, 4] = y
        out_values[obj, 5] = z


@wp.kernel
def _poses_to_matrices_kernel(
    poses: wp.array2d(dtype=wp.float32),  # (N, 7) — [px, py, pz, qx, qy, qz, qw]
    matrices: wp.array(dtype=wp.mat44),  # (N,)
):
    """Convert (N, 7) pose tensor (xyz + xyzw quat) into (N,) wp.mat44 rigid transforms."""
    i = wp.tid()
    px = poses[i, 0]
    py = poses[i, 1]
    pz = poses[i, 2]
    qx = poses[i, 3]
    qy = poses[i, 4]
    qz = poses[i, 5]
    qw = poses[i, 6]

    # Normalize the quaternion defensively (PhysX usually returns unit quats, but matching
    # transform_utils.quat2mat behavior keeps results bit-for-bit consistent with the legacy CPU path).
    n = wp.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    inv_n = wp.float32(1.0) / n
    qx = qx * inv_n
    qy = qy * inv_n
    qz = qz * inv_n
    qw = qw * inv_n

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    matrices[i] = wp.mat44(
        wp.float32(1.0) - wp.float32(2.0) * (yy + zz),
        wp.float32(2.0) * (xy - zw),
        wp.float32(2.0) * (xz + yw),
        px,
        wp.float32(2.0) * (xy + zw),
        wp.float32(1.0) - wp.float32(2.0) * (xx + zz),
        wp.float32(2.0) * (yz - xw),
        py,
        wp.float32(2.0) * (xz - yw),
        wp.float32(2.0) * (yz + xw),
        wp.float32(1.0) - wp.float32(2.0) * (xx + yy),
        pz,
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(1.0),
    )


class RigidBodyViewAPI:
    """
    Batched rigid-body pose cache for all rigid bodies across all scenes.

    Poses are stored in a flat layout keyed by absolute prim path, so each scene can
    have an independent link count (different object models are supported).

    Two categories of links are tracked:
      - physx_tracked:   rigid body links included in create_rigid_body_view.
                         Poses updated every step from PhysX.
      - physx_untracked: kinematic-only articulated objects' links not tracked by physx's rigid body view.
                         Poses seeded once at initialize_view() and refreshed only by invalidate_kinematic().

    Flat index layout in POSE_MATRICES (N_links_total, 4, 4):
      [scene_0 physx_tracked] [scene_1 physx_tracked] ... [physx_untracked links across all scenes]

    Also exposes LINK_MESH_IDS / LINK_VERTEX_COUNTS so the _aabb_reduce_kernel can read each
    link's collision mesh via wp.mesh_get_point. Each link's wp.Mesh is owned by this view
    (built once in initialize_view from link.collision_mesh_cpu_data with batched H2D copies),
    and stored in cls._link_meshes alongside the GPU-resident points/faces backing tensors.
    """

    # Rigid body view for batched pose reads (one per scene)
    _RIGID_BODY_VIEW = None

    # Flat path index across all scenes keyed by absolute prim path.
    _PATH_TO_IDX = {}  # {link_absolute_prim_path: int}
    _IDX_TO_PATH = []  # list[link_absolute_prim_path]

    # Position + quaternion pose of all links in the scene, indexed by flat index.
    # POSES (CPU pinned): where read_from_physx() copies PhysX output;
    # POSES_GPU: GPU mirror;
    # POSE_MATRICES: the derived (N,) wp.mat44 transform table.
    # All three are populated inside the captured wp.graph each step via update().
    POSES = None  # wp.array, device="cpu", pinned=True, shape=(N_links_total, 7), float32
    POSES_GPU = None  # wp.array, device="cuda", shape=(N_links_total, 7), float32
    POSE_MATRICES = None  # wp.array, device="cuda", shape=(N_links_total,), wp.mat44

    # Per-link wp.Mesh ids (0 if link has no collision geometry), kernel input for
    # wp.mesh_get_point(mesh_ids[body], v). Built in initialize_view via batched H2D copies of
    # link.collision_mesh_cpu_data, with each per-link wp.Mesh viewing a slice of the shared
    # GPU points/faces tensors.
    LINK_MESH_IDS = None  # wp.array(dtype=wp.uint64) on CUDA, shape (N_links_total,)

    # Per-link vertex count. Used by prepare_aabb_kernel_inputs to vectorize the K-length
    # thread-table expansion. CPU is fine (small, no transfer needed during expansion).
    LINK_VERTEX_COUNTS = None  # th.Tensor on CPU, shape (N_links_total,) int32

    # Per-link wp.Mesh cache, keyed by absolute prim path. Keeps wp.Mesh objects (and their
    # backing GPU points/faces tensors) alive across initialize_view() calls so re-init only
    # rebuilds meshes for newly-registered links. Reuse is gated by Python id(link), so a
    # rebuild also happens if a path is registered to a different RigidPrim instance.
    # Each entry: {"link_id": int, "mesh": wp.Mesh, "n_pts": int,
    #              "pts_buffer": th.Tensor, "faces_buffer": th.Tensor}.
    _link_mesh_cache = {}

    # K = total number links' vertices of objects with AABB states
    # Cached K-length kernel inputs for the main AABB kernel. Built by
    # prepare_aabb_kernel_inputs from the (PRIM_BODY_IDX, LINK_IDX) pair AABB.initialize_view
    # provides. Reset to None on every initialize_view().
    _aabb_links = None  # wp.array(dtype=wp.int32) on CUDA, shape (K,)
    _aabb_vertices = None  # wp.array(dtype=wp.int32) on CUDA, shape (K,)
    _aabb_objs = None  # wp.array(dtype=wp.int32) on CUDA, shape (K,)

    # Cached fallback-kernel inputs for meshes without boundry points (one entry per rigid object's base link).
    _aabb_base_link_links = None  # wp.array(dtype=wp.int32) on CUDA, shape (N_rigid,)
    _aabb_base_link_objs = None  # wp.array(dtype=wp.int32) on CUDA, shape (N_rigid,)

    # Tolerances for change detection
    _POS_EPS = 1e-4
    _ORI_EPS = 1e-4

    @classmethod
    def initialize_view(cls):
        """
        Initializes the rigid body view. Note: Can only be done when sim is playing!

        Builds a flat layout keyed by absolute prim path so each scene can have an
        independent link count. Layout of POSE_MATRICES (N_links_total, 4, 4):
          [scene_0 physx_tracked] [scene_1 physx_tracked] ... [physx_untracked across all scenes]

        Also collects each registered link's wp.Mesh id and vertex count into
        LINK_MESH_IDS / LINK_VERTEX_COUNTS for the AABB Warp kernel.
        """
        assert og.sim.is_playing(), "Cannot create rigid body view while sim is not playing!"

        # Ensure Warp's runtime is up before any kernel launches in get_aabb. Idempotent —
        # subsequent calls are no-ops once the runtime has been constructed.
        wp.init()

        # Snapshot existing kinematic-link poses ton reuse them below.
        # cls.POSES is now a wp.array, convert to a torch view for the per-row
        # indexing/unsqueeze below.
        prev_path_to_idx = dict(cls._PATH_TO_IDX)  # snapshot before clear()
        prev_poses = wp.to_torch(cls.POSES) if cls.POSES is not None else None

        # Reset
        cls.clear()

        if len(og.sim.scenes) == 0:
            return

        # If there are no rigid bodies in any scene, return early (view creation would fail)
        has_rigid_bodies = any(
            obj.prim_type == PrimType.RIGID and len(obj.links) > 0 for scene in og.sim.scenes for obj in scene.objects
        )
        if not has_rigid_bodies:
            cls.clear()
            return

        # Create a PhysX view for all scenes, build _PATH_TO_IDX, collect poses
        poses_list = []

        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            cls._RIGID_BODY_VIEW = og.sim.physics_sim_view.create_rigid_body_view(pattern="/World/scene_*/*/*")
            for abs_path in list(cls._RIGID_BODY_VIEW.prim_paths):
                cls._PATH_TO_IDX[abs_path] = len(cls._PATH_TO_IDX)
                cls._IDX_TO_PATH.append(abs_path)
            poses_list.append(cls._RIGID_BODY_VIEW.get_transforms().clone())

        # Add physx_untracked kinematic links and remember each registered link object so
        # we can pull its wp.Mesh after _PATH_TO_IDX is finalized below.
        link_by_idx = {}  # {flat_idx: link object}

        for _, scene in enumerate(og.sim.scenes):
            for obj in scene.objects:
                is_untracked = obj.kinematic_only and obj.prim_type != PrimType.CLOTH
                for link in obj.links.values():
                    abs_path = link.prim_path

                    # Kinematic objects' child links (e.g. fillable meta links) may be dynamic
                    # and already tracked by PhysX even though the parent is kinematic_only=True.
                    # Only manually register links that PhysX did not track.
                    if is_untracked and abs_path not in cls._PATH_TO_IDX:
                        idx = len(cls._PATH_TO_IDX)
                        cls._PATH_TO_IDX[abs_path] = idx
                        cls._IDX_TO_PATH.append(abs_path)

                        if abs_path in prev_path_to_idx and prev_poses is not None:
                            # Reuse the cached pose
                            pose = prev_poses[prev_path_to_idx[abs_path]].unsqueeze(0)
                        elif og.sim.currently_stepping:
                            # New kinematic link appearing mid-step, cannot read Fabric.
                            # Use zero placeholder.
                            # Corrected by the next out-of-step initialize_view() or update_handles().
                            pose = th.zeros(1, 7)
                        else:
                            pos, quat_xyzw = link.get_position_orientation()
                            pose = th.cat([pos, quat_xyzw]).unsqueeze(0)
                        poses_list.append(pose)

                    if abs_path in cls._PATH_TO_IDX:
                        link_by_idx[cls._PATH_TO_IDX[abs_path]] = link

        initial_poses = th.cat(poses_list, dim=0).contiguous()  # (N, 7) torch CPU, used only to seed
        N = initial_poses.shape[0]
        cls.POSES = wp.empty(shape=(N, 7), dtype=wp.float32, device="cpu", pinned=True)
        cls.POSES_GPU = wp.zeros(shape=(N, 7), dtype=wp.float32, device="cuda")
        cls.POSE_MATRICES = wp.zeros(shape=(N,), dtype=wp.mat44, device="cuda")
        # Seed POSES via the zero-copy torch view (same memory as the wp.array).
        wp.to_torch(cls.POSES)[:] = initial_poses

        # Build / reuse per-link wp.Mesh objects. Cached entries (same prim path + same RigidPrim
        # instance) are reused as-is; only newly-registered links flow through a single batched
        # H2D copy + BVH-build pass. Drop cache entries for paths no longer registered.
        N_total = len(cls._PATH_TO_IDX)
        mesh_id_list = [0] * N_total
        vertex_count_list = [0] * N_total

        new_cache = {}
        new_links = []  # (idx, abs_path, link, pts_cpu, faces_cpu) for paths needing a rebuild
        for idx, link in link_by_idx.items():
            abs_path = link.prim_path
            cached = cls._link_mesh_cache.get(abs_path)
            if cached is not None and cached["link_id"] == id(link):
                mesh_id_list[idx] = cached["mesh"].id
                vertex_count_list[idx] = cached["n_pts"]
                new_cache[abs_path] = cached
                continue
            data = link.collision_mesh_cpu_data()
            if data is None:
                continue
            pts_cpu, faces_cpu = data
            new_links.append((idx, abs_path, link, pts_cpu, faces_cpu))

        if new_links:
            pts_segments = [item[3] for item in new_links]
            faces_segments = [item[4] for item in new_links]
            pts_offsets = [0]
            faces_offsets = [0]
            for p, f in zip(pts_segments, faces_segments):
                pts_offsets.append(pts_offsets[-1] + p.shape[0])
                faces_offsets.append(faces_offsets[-1] + f.shape[0])

            pts_gpu_buf = th.cat(pts_segments, dim=0).cuda()
            faces_gpu_buf = th.cat(faces_segments, dim=0).cuda()

            for i, (idx, abs_path, link, pts_cpu, _) in enumerate(new_links):
                p0, p1 = pts_offsets[i], pts_offsets[i + 1]
                f0, f1 = faces_offsets[i], faces_offsets[i + 1]
                pts_view = pts_gpu_buf[p0:p1]
                faces_view = faces_gpu_buf[f0:f1]
                mesh = wp.Mesh(
                    points=wp.from_torch(pts_view, dtype=wp.vec3),
                    indices=wp.from_torch(faces_view),
                )
                mesh_id_list[idx] = mesh.id
                vertex_count_list[idx] = pts_cpu.shape[0]
                # Hold strong refs to the batch buffers in the cache entry so the wp.array
                # views into them stay valid for the lifetime of `mesh`.
                new_cache[abs_path] = {
                    "link_id": id(link),
                    "mesh": mesh,
                    "n_pts": pts_cpu.shape[0],
                    "pts_buffer": pts_gpu_buf,
                    "faces_buffer": faces_gpu_buf,
                }

        cls._link_mesh_cache = new_cache
        cls.LINK_MESH_IDS = wp.array(mesh_id_list, dtype=wp.uint64, device="cuda")
        cls.LINK_VERTEX_COUNTS = th.tensor(vertex_count_list, dtype=th.int32)

        # Invalidate any previously cached AABB kernel inputs — AABB.initialize_view() must
        # call prepare_aabb_kernel_inputs() again before the next get_aabb() call.
        cls._aabb_links = None
        cls._aabb_vertices = None
        cls._aabb_objs = None
        cls._aabb_base_link_links = None
        cls._aabb_base_link_objs = None

    @classmethod
    def read_from_physx(cls):
        """
        Read the latest PhysX transforms into the pinned-CPU staging buffer (cls.POSES).
        Stays on the host: PhysX returns a CPU torch tensor, and graph capture freezes the
        call (see scripts/test_physx_in_warp_graph.py). The H2D copy + pose2mat kernel run
        inside the captured wp.graph via update().

        physx_untracked links are never touched here; they are updated only by invalidate_kinematic().
        """
        if not cls._RIGID_BODY_VIEW:
            return
        try:
            transforms = cls._RIGID_BODY_VIEW.get_transforms()  # torch CPU tensor
        except Exception:
            log.warning(
                "RigidBodyViewAPI cannot fetch transforms because the physics sim view is invalid. "
                "This is expected during initial scene loading."
            )
            return

        # Copy PhysX output into the fixed pinned wp.array via a torch view
        N_physx_tracked = transforms.shape[0]
        wp.to_torch(cls.POSES)[:N_physx_tracked] = transforms
        wp.copy(cls.POSES_GPU, cls.POSES)

    @classmethod
    def get_flat_idx(cls, abs_prim_path):
        """
        Return the flat index into POSE_MATRICES for a link's absolute prim path, or None.

        Args:
            abs_prim_path (str): Absolute prim path of the link (e.g. /World/scene_0/obj/link).

        Returns:
            int or None
        """
        return cls._PATH_TO_IDX.get(abs_prim_path)

    @classmethod
    def prepare_aabb_kernel_inputs(cls, prim_body_idx, link_idx, base_link_body_idx, base_link_values_idx):
        """
        Build and cache the K-length kernel input tables for the AABB kernel and the
        N_rigid-length tables for the base-link fallback kernel. Called once by
        AABB.initialize_view() after it builds its (PRIM_BODY_IDX, LINK_IDX,
        BASE_LINK_BODY_IDX, BASE_LINK_VALUES_IDX) tensors.

        Args:
            prim_body_idx (th.Tensor):       (n_tracked_links,) int32 CPU — flat idx per AABB-tracked link
            link_idx (th.Tensor):            (n_tracked_links,) int32 CPU — output row (s*O + obj_idx) per tracked link
            base_link_body_idx (th.Tensor):  (n_rigid_objs,) int32 CPU — flat idx of each rigid object's base link
            base_link_values_idx (th.Tensor):(n_rigid_objs,) int32 CPU — output row per rigid object
        """
        # Always reset the cache; we rebuild what we can below.
        cls._aabb_links = None
        cls._aabb_vertices = None
        cls._aabb_objs = None
        cls._aabb_base_link_links = None
        cls._aabb_base_link_objs = None

        # Nothing to prepare if no rigid bodies are registered.
        if cls.LINK_VERTEX_COUNTS is None:
            return

        if prim_body_idx.numel() > 0:
            # Goal: expand from "one entry per tracked link" (length = n_tracked_links) to
            # "one entry per (link, vertex) pair" (length K = sum of vertex counts). The
            # K-length tables are what the AABB kernel reads, one per thread.
            #
            # All ops on CPU; n_tracked_links is small.

            verts_per_link = cls.LINK_VERTEX_COUNTS[prim_body_idx]  # (n_tracked_links,)
            K = int(verts_per_link.sum().item())

            # Repeat each link's body / obj index by its vertex count so each thread that
            # belongs to that link sees the same body and obj.
            aabb_links = th.repeat_interleave(prim_body_idx, verts_per_link)  # (K,)
            aabb_objs = th.repeat_interleave(link_idx, verts_per_link)  # (K,)

            # Compute for each link, the index of the first thread that run the link's vertex
            link_thread_starts = th.cat([th.zeros(1, dtype=verts_per_link.dtype), verts_per_link.cumsum(0)])
            thread_idx = th.arange(K, dtype=verts_per_link.dtype)
            # for thread k, find the smallest i such that k < link_thread_starts[1:][i]
            tracked_link_per_thread = th.searchsorted(link_thread_starts[1:], thread_idx, right=True)
            aabb_vertices = (thread_idx - link_thread_starts[tracked_link_per_thread]).to(th.int32)

            cls._aabb_links = wp.from_torch(aabb_links.cuda())
            cls._aabb_vertices = wp.from_torch(aabb_vertices.cuda())
            cls._aabb_objs = wp.from_torch(aabb_objs.cuda())

        if base_link_body_idx.numel() > 0:
            cls._aabb_base_link_links = wp.from_torch(base_link_body_idx.cuda())
            cls._aabb_base_link_objs = wp.from_torch(base_link_values_idx.cuda())

    @classmethod
    def get_aabb(cls, out_values):
        """
        Compute per-object world-space AABBs by launching two Warp kernels back-to-back on
        torch's current CUDA stream:
          1. _aabb_reduce_kernel: per-(link, vertex) thread, transforms one local point and
             atomically narrows the owning object's (lo, hi) row of out_values.
          2. _aabb_baselink_fallback_kernel: per-rigid-object thread, writes base-link
             position as a point AABB for any object whose row was untouched (still +inf).

        Args:
            out_values (th.Tensor): (S*O, 6) float32 CUDA — caller pre-fills with +/-inf,
                                    written in-place.
        """
        if cls._aabb_links is None or cls._aabb_links.shape[0] == 0:
            return
        out_arr = wp.from_torch(out_values)

        wp.launch(
            kernel=_aabb_reduce_kernel,
            dim=cls._aabb_links.shape[0],
            inputs=[
                cls.POSE_MATRICES,
                cls.LINK_MESH_IDS,
                cls._aabb_links,
                cls._aabb_vertices,
                cls._aabb_objs,
                out_arr,
            ],
            device="cuda",
        )
        # use base-link pose for meshes without boundry points
        if cls._aabb_base_link_links is not None and cls._aabb_base_link_links.shape[0] > 0:
            wp.launch(
                kernel=_aabb_baselink_fallback_kernel,
                dim=cls._aabb_base_link_links.shape[0],
                inputs=[
                    cls.POSES_GPU,
                    cls._aabb_base_link_links,
                    cls._aabb_base_link_objs,
                    out_arr,
                ],
                device="cuda",
            )

    @classmethod
    def invalidate_kinematic(cls, links):
        """
        Refresh the CPU pose staging buffer for kinematic links after an explicit move.

        For physx_untracked links (no physics:RigidBodyAPI) this is the only update path.
        For physx_tracked kinematic links this write is harmless — read_from_physx() will
        re-read from PhysX on the next step.

        Only POSES (pinned CPU) is touched here. POSES_GPU and POSE_MATRICES are
        refreshed by the next captured-graph replay — every consumer reads them inside the
        graph (or post-synchronize via UpdateStateMixin), so no eager H2D / kernel is needed.

        Args:
            links: iterable of RigidPrim link objects whose poses should be refreshed.
        """
        if cls.POSES is None:
            return
        # Zero-copy torch view over the pinned wp.array — same memory.
        poses_torch = wp.to_torch(cls.POSES)
        for link in links:
            idx = cls._PATH_TO_IDX.get(link.prim_path)
            if idx is None:
                continue  # not registered (e.g. particle templates not in scene.objects)
            pos, quat_xyzw = link.get_position_orientation()
            poses_torch[idx, :3] = pos
            poses_torch[idx, 3:] = quat_xyzw

    @classmethod
    def update(cls):
        """
        Enqueue the per-step H2D + pose-to-mat work onto the current Warp stream. Called
        inside wp.ScopedCapture so the captured graph replays the H2D + kernel together with
        the tensorized-state global_updates each step.
        """
        if cls.POSES is None or cls.POSES_GPU is None or cls.POSE_MATRICES is None:
            return
        N = cls.POSES_GPU.shape[0]
        if N == 0:
            return
        wp.launch(
            _poses_to_matrices_kernel,
            dim=N,
            inputs=[cls.POSES_GPU, cls.POSE_MATRICES],
            device="cuda",
        )

    @classmethod
    def clear(cls):
        """Reset all cached state."""
        cls._RIGID_BODY_VIEW = None
        cls._PATH_TO_IDX = {}
        cls._IDX_TO_PATH = []
        cls.POSES = None
        cls.POSES_GPU = None
        cls.POSE_MATRICES = None
        cls.LINK_MESH_IDS = None
        cls.LINK_VERTEX_COUNTS = None
        cls._link_mesh_cache = {}
        cls._aabb_links = None
        cls._aabb_vertices = None
        cls._aabb_objs = None
        cls._aabb_base_link_links = None
        cls._aabb_base_link_objs = None


class ArticulatedObjectViewAPI:
    """
    Batched DOF position cache for non-robot articulated objects across all scenes.

    We exclude robots because they have a lot more DOFs than other objects, which would inflate the
    size of this view drastically. However, this view is mainly used for Open state checking, which
    does not apply to robots. As a result, robots are excluded for optimization purposes.

    Creates a single ArticulationView covering all non-robot articulated objects across all scenes
    using the pattern "/World/scene_\\d+/articulated_.*" (objects are placed under articulated_<name>
    by _preapply_articulation_root in usd_object.py).

    Data layout:
        _VIEW        ArticulationView          single view for all non-robot articulated objects
        _OBJ_TO_VIEW_IDX  {abs_art_root_path: int}  row index in _JOINT_POSITIONS
        _JOINT_POSITIONS   (N_objects_all_scenes, max_dof)  one row per object per scene, PhysX row order
    """

    _VIEW = None
    _OBJ_TO_VIEW_IDX = {}
    _JOINT_POSITIONS = None  # wp.array, device="cuda", shape=(N_art, max_dof), float32

    @classmethod
    def initialize_view(cls):
        cls.clear()

        if len(og.sim.scenes) == 0:
            return

        from omnigibson.robots import Robot

        articulation_objs = [
            obj
            for scene in og.sim.scenes
            for obj in scene.objects
            if obj.relative_prim_path.startswith("/articulated__") and not isinstance(obj, Robot)
        ]
        if not articulation_objs:
            return

        pattern = "/World/scene_*/articulated__*/*"
        cls._VIEW = og.sim.physics_sim_view.create_articulation_view(pattern)
        assert set(cls._VIEW.prim_paths) == set(
            obj.articulation_root_path for obj in articulation_objs
        ), "Articulation view prim paths mismatch!"
        cls._OBJ_TO_VIEW_IDX = {abs_path: row for row, abs_path in enumerate(cls._VIEW.prim_paths)}
        # Allocate fixed pinned-CPU and CUDA wp.arrays of the right shape.
        seed_positions = cls._VIEW.get_dof_positions().contiguous()  # torch CPU, used only to seed
        N, max_dof = seed_positions.shape
        cls._JOINT_POSITIONS = wp.zeros(shape=(N, max_dof), dtype=wp.float32, device="cuda")
        wp.copy(cls._JOINT_POSITIONS, wp.from_torch(seed_positions))

    @classmethod
    def read_from_physx(cls):
        """Pull latest DOF positions from PhysX into the pinned CPU staging buffer.
        Stays on the host (PhysX returns CPU torch tensor). H2D happens inside the captured
        wp.graph via update()."""
        if cls._VIEW is None or cls._JOINT_POSITIONS is None:
            return
        positions = cls._VIEW.get_dof_positions()  # torch CPU
        wp.copy(cls._JOINT_POSITIONS, wp.from_torch(positions.contiguous()))

    @classmethod
    def get_view_row(cls, abs_prim_path):
        """Return row index in _JOINT_POSITIONS for a given absolute articulation root path, or None."""
        return cls._OBJ_TO_VIEW_IDX.get(abs_prim_path)

    @classmethod
    def get_max_dof(cls):
        """Return the padded DOF width of _JOINT_POSITIONS; 0 if not initialized."""
        return cls._JOINT_POSITIONS.shape[1] if cls._JOINT_POSITIONS is not None else 0

    @classmethod
    def clear(cls):
        cls._VIEW = None
        cls._OBJ_TO_VIEW_IDX = {}
        cls._JOINT_POSITIONS = None


class RigidContactAPIImpl:
    """
    Class containing class methods to aggregate rigid body contacts across all rigid bodies in the simulator.

    This API checks for contacts on every physics step, and then aggregates this into a boolean contact matrix
    on every non-physics step. Callers can then use this API to query either for contacts that are still ongoing,
    or contacts that occurred at any point since the last non-physics step (e.g. for checking for contact events).
    Contact information is cached per-physics-step and only updated for body pairs who have at least one side
    not asleep, which allows this API to bypass the limitations of the view (which returns contacts only for awake bodies).

    Since there is no direct tensorized way to check for object sleep state, this API approximates this by checking for
    the net contact force on an object (only reported for awake bodies) and also the position change since the last step.
    """

    def __init__(self):
        # Dictionary mapping rigid body prim path to corresponding row / col index in the contact view matrix
        self._PATH_TO_ROW_IDX = dict()
        self._PATH_TO_COL_IDX = dict()

        # Arrays of rigid body prim paths where each array index maps directly to the contact matrix row / col
        self._ROW_IDX_TO_PATH = dict()
        self._COL_IDX_TO_PATH = dict()

        # Contact view for generating contact matrices at each timestep
        self._CONTACT_VIEW = dict()

        # Rigid body view for batched body transform reads used by persistence logic
        self._RIGID_BODY_VIEW = dict()

        # Precomputed tensors mapping row/col indices to rigid body view indices
        self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY = dict()

        # int32 wp.array mirrors of the row/col→rigid-body index maps, consumed by
        # the warp kernels in update(). Kept alongside the torch versions to avoid
        # type-casting inside the captured graph.
        self._ROW_TO_RIGID_WP = dict()
        self._COL_TO_RIGID_WP = dict()

        # Body → row index map (B,), -1 if body is not a contact-matrix row body.
        # Inverse of _CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS. Used by _body_awake_at_step
        # to apply net-force-awake inline, eliminating the separate per-step OR pass.
        self._BODY_TO_ROW = dict()
        self._BODY_TO_ROW_WP = dict()

        # Contact matrix tracking contacts that occurred at any point during the last N physics steps
        # (between consecutive update calls). Shape: (R, C)
        self._CONTACT_MATRIX = dict()
        self._CONTACT_MATRIX_GPU = dict()  # GPU primary — updated directly in update()
        # wp.array views over _CONTACT_MATRIX_GPU per scene; wrapped once after each (re-)allocation
        # for use inside Warp kernels and graph capture.
        self._CONTACT_MATRIX_GPU_WP = dict()

        # Contact matrix tracking contacts at only the most recent physics step. Shape: (R, C)
        self._CURRENT_CONTACT_MATRIX = dict()
        self._CURRENT_CONTACT_MATRIX_GPU = dict()  # GPU primary — updated directly in update()
        self._CURRENT_CONTACT_MATRIX_GPU_WP = dict()

        # A matrix of indices for the contact matrix. This can be indexed the same way as the contact matrix
        # to obtain row and column indices to map back to prim paths. Shape: (R, C, 2)
        self._INDEX_MATRIX = dict()

        # Cached body transforms used for change detection. Shape: (N, 7) [pos(3), quat(4)]
        self._BODY_TRANSFORMS = dict()
        # wp.array view over _BODY_TRANSFORMS for the contact-matrix and body-transform kernels.
        self._BODY_TRANSFORMS_WP = dict()

        # Accumulated impulse matrices and transforms from individual physics steps,
        # collected between consecutive update calls.
        self._PENDING_STEPS = 0
        self._PENDING_IMPULSES = dict()
        self._PENDING_TRANSFORMS = dict()
        self._PENDING_NET_FORCES = dict()

        # wp.array views over _PENDING_* for use inside the captured warp graph.
        # Re-wrapped on every initialize_view because the underlying torch storage is
        # (re-)allocated there.
        self._PENDING_IMPULSES_WP = dict()
        self._PENDING_TRANSFORMS_WP = dict()
        self._PENDING_NET_FORCES_WP = dict()

        # Pinned-CPU wp.array views over the pinned torch CPU mirrors. The captured
        # update() does `wp.copy(host_wp, gpu_wp)` after the kernels to refresh them
        # each replay — replaces the old sync_cpu_mirrors method that ran outside the
        # captured region.
        self._CONTACT_MATRIX_HOST_WP = dict()
        self._CURRENT_CONTACT_MATRIX_HOST_WP = dict()

        # 1-element GPU buffer holding _PENDING_STEPS — the number of physics sub-steps
        # the captured contact kernels should iterate via `n_steps_arr[0]`.
        # Must be a wp.array (not a scalar kernel arg) because scalar args get baked into
        # the captured graph at capture time, but this count varies between replays
        # (``og.sim.step()`` = N sub-steps, ``og.sim.step_physics()`` = 1).
        self._N_STEPS_TORCH = None  # torch.Tensor on cuda, shape (1,), int32
        self._N_STEPS_GPU = None  # wp.array view over _N_STEPS_TORCH

        # Position / orientation tolerances for deciding whether a pair should be updated
        self._POS_EPS = 1e-6
        self._ORI_EPS = 1e-4

    @classmethod
    def get_body_filters(cls):
        filters = dict()
        for scene_idx, scene in enumerate(og.sim.scenes):
            filters[scene_idx] = []

            # Add the (global) floor plane if there is one
            if og.sim.floor_plane is not None:
                filters[scene_idx].append(og.sim.floor_plane.prim_path + "/collisionPlane")

            for obj in scene.objects:
                if obj.prim_type == PrimType.RIGID:
                    for link in obj.links.values():
                        from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
                        from omnigibson.prims.rigid_kinematic_prim import RigidKinematicPrim

                        if isinstance(link, (RigidDynamicPrim, RigidKinematicPrim)) and link.contact_reporting_enabled:
                            filters[scene_idx].append(link.prim_path)

        return filters

    @classmethod
    def get_max_contact_data_count(cls, n_bodies):
        return 256

    def initialize_view(self):
        """
        Initializes the rigid contact view. Note: Can only be done when sim is playing!
        """
        assert og.sim.is_playing(), "Cannot create rigid contact view while sim is not playing!"

        # Snapshot the old contact matrices and path mappings so we can carry over
        # cached contact state for pairs of bodies that already existed.
        prev_contact_matrix = dict(self._CONTACT_MATRIX)
        prev_current_contact_matrix = dict(self._CURRENT_CONTACT_MATRIX)
        prev_path_to_row_idx = dict(self._PATH_TO_ROW_IDX)
        prev_path_to_col_idx = dict(self._PATH_TO_COL_IDX)

        # Rebuild views from scratch to pick up any new/removed bodies.
        self.clear()

        self._N_STEPS_TORCH = th.zeros(1, dtype=th.int32, device="cuda")
        self._N_STEPS_GPU = wp.from_torch(self._N_STEPS_TORCH, dtype=wp.int32)

        body_filters = self.get_body_filters()

        # If there are no valid bodies, clear all views / mappings and terminate early
        if not any(len(filters) > 0 for filters in body_filters.values()):
            self.clear()
            return

        # Generate views, making sure to update simulation first so the physx backend is synchronized.
        with suppress_omni_log(channels=["omni.physx.tensors.plugin"]):
            for scene_idx, _ in enumerate(og.sim.scenes):
                scene_body_filters = body_filters[scene_idx]
                if len(scene_body_filters) == 0:
                    continue

                # Rows correspond to dynamic rigid prims only, while columns correspond to all rigid prims.
                scene_dynamic_body_filters = []
                for obj in og.sim.scenes[scene_idx].objects:
                    if obj.prim_type == PrimType.RIGID:
                        for link in obj.links.values():
                            from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim

                            if isinstance(link, RigidDynamicPrim) and link.contact_reporting_enabled:
                                scene_dynamic_body_filters.append(link.prim_path)

                # If there are only kinematic/static bodies, skip view creation for this scene.
                if len(scene_dynamic_body_filters) == 0:
                    continue

                self._CONTACT_VIEW[scene_idx] = og.sim.physics_sim_view.create_rigid_contact_view(
                    pattern=f"/World/scene_{scene_idx}/*/*",
                    filter_patterns=scene_body_filters,
                    max_contact_data_count=self.get_max_contact_data_count(len(scene_body_filters)),
                )
                row_paths = list(self._CONTACT_VIEW[scene_idx].sensor_paths)
                col_paths = list(getattr(self._CONTACT_VIEW[scene_idx], "filter_patterns", scene_body_filters))

                if set(row_paths) != set(scene_dynamic_body_filters):
                    missing_rows = sorted(set(scene_dynamic_body_filters) - set(row_paths))
                    extra_rows = sorted(set(row_paths) - set(scene_dynamic_body_filters))
                    raise AssertionError(
                        "RigidContactAPI contact-view row mismatch. "
                        f"Expected {len(scene_dynamic_body_filters)} dynamic rows, got {len(row_paths)} rows. "
                        f"Missing rows ({len(missing_rows)}): {missing_rows}. "
                        f"Extra rows ({len(extra_rows)}): {extra_rows}."
                    )

                if set(col_paths) != set(scene_body_filters):
                    missing_cols = sorted(set(scene_body_filters) - set(col_paths))
                    extra_cols = sorted(set(col_paths) - set(scene_body_filters))
                    raise AssertionError(
                        "RigidContactAPI contact-view column mismatch. "
                        f"Expected {len(scene_body_filters)} rigid columns, got {len(col_paths)} columns. "
                        f"Missing columns ({len(missing_cols)}): {missing_cols}. "
                        f"Extra columns ({len(extra_cols)}): {extra_cols}."
                    )

                # Create the lookup tables
                self._ROW_IDX_TO_PATH[scene_idx] = row_paths
                self._COL_IDX_TO_PATH[scene_idx] = col_paths
                self._PATH_TO_ROW_IDX[scene_idx] = {path: i for i, path in enumerate(row_paths)}
                self._PATH_TO_COL_IDX[scene_idx] = {path: i for i, path in enumerate(col_paths)}

                # Create the rigid body view, and create some indexing tensors that allow for fast lookups
                # between the contact matrix rows and the rigid body view indices.
                self._RIGID_BODY_VIEW[scene_idx] = og.sim.physics_sim_view.create_rigid_body_view(
                    pattern=f"/World/scene_{scene_idx}/*/*"
                )
                path_to_view_idx = {path: i for i, path in enumerate(list(self._RIGID_BODY_VIEW[scene_idx].prim_paths))}
                self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS[scene_idx] = th.tensor(
                    [path_to_view_idx[path] for path in row_paths], dtype=th.long, device="cuda"
                )

                # Some contact-matrix columns can correspond to kinematic-only links that do not appear
                # in the rigid-body view. We encode those as -1 and track a validity mask.
                col_to_rigid_rows = [path_to_view_idx.get(path, -1) for path in col_paths]
                self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS[scene_idx] = th.tensor(
                    col_to_rigid_rows, dtype=th.long, device="cuda"
                )
                self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY[scene_idx] = (
                    self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS[scene_idx] >= 0
                )
                ii, jj = th.meshgrid(th.arange(len(row_paths)), th.arange(len(col_paths)), indexing="ij")
                self._INDEX_MATRIX[scene_idx] = th.stack([ii, jj], dim=-1)
                self._BODY_TRANSFORMS[scene_idx] = self._RIGID_BODY_VIEW[scene_idx].get_transforms().cuda()

                # int32 mirrors of the row/col→rigid-body maps for the warp kernels (wp.tid()
                # returns int32; keeping these as int32 avoids per-launch casts inside the captured graph).
                row_to_rigid_int32 = self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS[scene_idx].to(th.int32)
                col_to_rigid_int32 = self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS[scene_idx].to(th.int32)
                self._ROW_TO_RIGID_WP[scene_idx] = wp.from_torch(row_to_rigid_int32, dtype=wp.int32)
                self._COL_TO_RIGID_WP[scene_idx] = wp.from_torch(col_to_rigid_int32, dtype=wp.int32)

                # Body → row map (B,), -1 if body is not a row body. Inverse scatter is safe
                # without atomics because rows correspond to unique bodies (see initialize_view's
                # dynamic-body iteration above).
                num_bodies = self._BODY_TRANSFORMS[scene_idx].shape[0]
                body_to_row = th.full((num_bodies,), -1, dtype=th.int32, device="cuda")
                row_idxs = th.arange(len(row_paths), dtype=th.int32, device="cuda")
                body_to_row[self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS[scene_idx]] = row_idxs
                self._BODY_TO_ROW[scene_idx] = body_to_row
                self._BODY_TO_ROW_WP[scene_idx] = wp.from_torch(body_to_row, dtype=wp.int32)

                # wp.array view over _BODY_TRANSFORMS for the contact kernels.
                self._BODY_TRANSFORMS_WP[scene_idx] = wp.from_torch(self._BODY_TRANSFORMS[scene_idx], dtype=wp.float32)

                # Build the new contact matrices. Start from current impulses (captures contacts
                # for newly added bodies), then overwrite with previously cached values for
                # every pair of bodies that already existed before the rebuild.
                initial_impulses = self._CONTACT_VIEW[scene_idx].get_contact_force_matrix(dt=1.0)
                initial_contacts = th.any(initial_impulses != 0, dim=-1)
                self._CONTACT_MATRIX[scene_idx] = initial_contacts.clone().pin_memory()
                self._CURRENT_CONTACT_MATRIX[scene_idx] = initial_contacts.clone().pin_memory()
                self._CONTACT_MATRIX_GPU[scene_idx] = initial_contacts.cuda()
                self._CURRENT_CONTACT_MATRIX_GPU[scene_idx] = initial_contacts.cuda()

                # Finally, remap data from the old matrices into the new ones. This lets us avoid losing our
                # cached data when new bodies are added or removed.
                old_matrix = prev_contact_matrix.get(scene_idx)
                old_current_matrix = prev_current_contact_matrix.get(scene_idx)
                old_row_map = prev_path_to_row_idx.get(scene_idx)
                old_col_map = prev_path_to_col_idx.get(scene_idx)
                if old_matrix is not None and old_row_map is not None and old_col_map is not None:
                    # Find rows and columns that existed in both the old and new views
                    surviving_row_paths = [p for p in row_paths if p in old_row_map]
                    surviving_col_paths = [p for p in col_paths if p in old_col_map]
                    if surviving_row_paths and surviving_col_paths:
                        old_row_idxs = th.tensor([old_row_map[p] for p in surviving_row_paths], dtype=th.long)
                        old_col_idxs = th.tensor([old_col_map[p] for p in surviving_col_paths], dtype=th.long)
                        new_row_idxs = th.tensor(
                            [self._PATH_TO_ROW_IDX[scene_idx][p] for p in surviving_row_paths], dtype=th.long
                        )
                        new_col_idxs = th.tensor(
                            [self._PATH_TO_COL_IDX[scene_idx][p] for p in surviving_col_paths], dtype=th.long
                        )
                        self._CONTACT_MATRIX[scene_idx][new_row_idxs[:, None], new_col_idxs[None, :]] = old_matrix[
                            old_row_idxs[:, None], old_col_idxs[None, :]
                        ]
                        self._CURRENT_CONTACT_MATRIX[scene_idx][new_row_idxs[:, None], new_col_idxs[None, :]] = (
                            old_current_matrix[old_row_idxs[:, None], old_col_idxs[None, :]]
                        )

                # Sync GPU mirrors from CPU matrices — carry-over may have updated the CPU tensors above.
                self._CONTACT_MATRIX_GPU[scene_idx].copy_(self._CONTACT_MATRIX[scene_idx])
                self._CURRENT_CONTACT_MATRIX_GPU[scene_idx].copy_(self._CURRENT_CONTACT_MATRIX[scene_idx])

                # Wrap the GPU contact matrices as wp.array views for use inside Warp kernels and
                # graph capture. Re-wrap on every initialize_view because the underlying torch
                # storage was just (re-)allocated above. Bool tensors → wp.uint8 (byte-per-bool).
                self._CONTACT_MATRIX_GPU_WP[scene_idx] = wp.from_torch(
                    self._CONTACT_MATRIX_GPU[scene_idx].view(th.uint8), dtype=wp.uint8
                )
                self._CURRENT_CONTACT_MATRIX_GPU_WP[scene_idx] = wp.from_torch(
                    self._CURRENT_CONTACT_MATRIX_GPU[scene_idx].view(th.uint8), dtype=wp.uint8
                )
                # Pinned-host wp.array views over the pinned torch CPU mirrors. The captured
                # ``wp.copy`` in update() refreshes these from the GPU each replay, so callers
                # of is_in_contact / get_contact_pairs (which read the torch CPU tensors) see
                # fresh data without a separate sync method.
                self._CONTACT_MATRIX_HOST_WP[scene_idx] = wp.from_torch(
                    self._CONTACT_MATRIX[scene_idx].view(th.uint8), dtype=wp.uint8
                )
                self._CURRENT_CONTACT_MATRIX_HOST_WP[scene_idx] = wp.from_torch(
                    self._CURRENT_CONTACT_MATRIX[scene_idx].view(th.uint8), dtype=wp.uint8
                )

                # Initialize pending accumulation lists for this scene
                # Note that existing data in these lists will be lost when the view is rebuilt.
                # TODO: Assert here that this is not happening during a physics step, and that these buffers are empty.
                # This TODO can be accomplished after the follow-up PR removes RigidContactAPI use in assisted grasping.
                n_physics_steps = og.sim.n_physics_timesteps_per_render
                self._PENDING_STEPS = 0
                self._PENDING_IMPULSES[scene_idx] = th.zeros(
                    n_physics_steps,
                    self._CONTACT_MATRIX[scene_idx].shape[0],
                    self._CONTACT_MATRIX[scene_idx].shape[1],
                    3,
                    device="cuda",
                )
                self._PENDING_TRANSFORMS[scene_idx] = th.zeros(
                    n_physics_steps, self._BODY_TRANSFORMS[scene_idx].shape[0], 7, device="cuda"
                )
                self._PENDING_NET_FORCES[scene_idx] = th.zeros(
                    n_physics_steps,
                    self._CONTACT_MATRIX[scene_idx].shape[0],
                    3,
                    device="cuda",
                )

                # wp.array views over the pending GPU buffers, consumed by the captured
                # update() kernels. Re-wrap on every initialize_view because the underlying
                # torch storage was just (re-)allocated above.
                self._PENDING_IMPULSES_WP[scene_idx] = wp.from_torch(
                    self._PENDING_IMPULSES[scene_idx], dtype=wp.float32
                )
                self._PENDING_TRANSFORMS_WP[scene_idx] = wp.from_torch(
                    self._PENDING_TRANSFORMS[scene_idx], dtype=wp.float32
                )
                self._PENDING_NET_FORCES_WP[scene_idx] = wp.from_torch(
                    self._PENDING_NET_FORCES[scene_idx], dtype=wp.float32
                )

    def read_from_physx(self):
        """
        Fetches contact impulse matrices and body transforms from the current physics step
        and appends them to pending lists. Should be called by the simulator after every
        individual physics step. The accumulated data is later processed in bulk by
        update().
        """
        assert og.sim.currently_stepping, "read_from_physx must be called during a physics step"
        assert self._PENDING_STEPS < og.sim.n_physics_timesteps_per_render, "Pending steps buffer is full"

        scene_idx_list = list(self._CONTACT_VIEW.keys())
        for scene_idx in scene_idx_list:
            try:
                # Get the contact impulse and net force matrices for this scene
                self._PENDING_IMPULSES[scene_idx][self._PENDING_STEPS].copy_(
                    self._CONTACT_VIEW[scene_idx].get_contact_force_matrix(dt=og.sim.get_physics_dt()),
                    non_blocking=True,
                )
                self._PENDING_NET_FORCES[scene_idx][self._PENDING_STEPS].copy_(
                    self._CONTACT_VIEW[scene_idx].get_net_contact_forces(dt=og.sim.get_physics_dt()),
                    non_blocking=True,
                )
            except Exception:
                log.warning(
                    "RigidContactAPI cannot compute contacts because the physics sim view is invalid. "
                    "This is expected if the physics sim view is not yet initialized, e.g. you are loading "
                    "a scene for the first time."
                )
                continue

            # Get the body transforms for this scene
            # TODO(vector): Replace this with a wp.copy.
            self._PENDING_TRANSFORMS[scene_idx][self._PENDING_STEPS].copy_(
                self._RIGID_BODY_VIEW[scene_idx].get_transforms(), non_blocking=True
            )

        # Increment once per physics step and push the new count to GPU so the captured
        # contact kernels see the right `n_steps` at replay time.
        if scene_idx_list:
            self._PENDING_STEPS += 1
            self._N_STEPS_TORCH.fill_(self._PENDING_STEPS)

    def update(self):
        """
        Issue the per-step warp kernels that bring the GPU contact matrices and body-transform
        snapshot up to date from the pending physics-step data ``read_from_physx`` staged.
        Also issues the D2H ``wp.copy``s that mirror the GPU matrices into the pinned CPU
        tensors ``is_in_contact`` / ``get_contact_pairs`` read — captured into the graph so
        they replay each frame without a separate sync hook.

        Algorithm:
          1. Per (r, c): walk the N physics sub-steps, marking each step's pair-awakeness
             via ``_body_awake_at_step`` for both row and column bodies. Track the last awake
             step + whether any awake step had a nonzero impulse. Write
             ``current_contact_matrix`` from the last-awake-step impulse and
             ``contact_matrix`` from the OR over all awake-step impulses. Pairs that were
             never awake collapse both matrices to the carried-over current value.
          2. Per (b,): snapshot each body's transform from its own last awake step into
             ``_BODY_TRANSFORMS`` (also the ``prev_transforms`` next-frame baseline).
          3. ``wp.copy`` GPU → pinned CPU for both contact matrices.
        """
        for scene_idx in self._CONTACT_VIEW.keys():
            R, C = self._CONTACT_MATRIX_GPU[scene_idx].shape
            B = self._BODY_TRANSFORMS[scene_idx].shape[0]

            # need to launch first because _update_body_transforms_kernel
            # overwrites _BODY_TRANSFORMS, which read by both kernels
            wp.launch(
                kernel=_update_contact_matrices_kernel,
                dim=(R, C),
                inputs=[
                    self._PENDING_TRANSFORMS_WP[scene_idx],
                    self._BODY_TRANSFORMS_WP[scene_idx],
                    self._PENDING_NET_FORCES_WP[scene_idx],
                    self._BODY_TO_ROW_WP[scene_idx],
                    self._PENDING_IMPULSES_WP[scene_idx],
                    self._ROW_TO_RIGID_WP[scene_idx],
                    self._COL_TO_RIGID_WP[scene_idx],
                    self._N_STEPS_GPU,
                    wp.float32(self._POS_EPS),
                    wp.float32(self._ORI_EPS),
                    self._CONTACT_MATRIX_GPU_WP[scene_idx],
                    self._CURRENT_CONTACT_MATRIX_GPU_WP[scene_idx],
                ],
                device="cuda",
            )

            wp.launch(
                kernel=_update_body_transforms_kernel,
                dim=B,
                inputs=[
                    self._PENDING_TRANSFORMS_WP[scene_idx],
                    self._BODY_TRANSFORMS_WP[scene_idx],
                    self._PENDING_NET_FORCES_WP[scene_idx],
                    self._BODY_TO_ROW_WP[scene_idx],
                    self._N_STEPS_GPU,
                    wp.float32(self._POS_EPS),
                    wp.float32(self._ORI_EPS),
                    self._BODY_TRANSFORMS_WP[scene_idx],
                ],
                device="cuda",
            )

            # so `is_in_contact` reads stay fresh each replay.
            wp.copy(self._CONTACT_MATRIX_HOST_WP[scene_idx], self._CONTACT_MATRIX_GPU_WP[scene_idx])
            wp.copy(self._CURRENT_CONTACT_MATRIX_HOST_WP[scene_idx], self._CURRENT_CONTACT_MATRIX_GPU_WP[scene_idx])

    def _get_prim_paths(self, objects_links_or_prim_paths):
        """
        Converts a set of objects, links, or prim paths to a list of prim paths for contact matrix lookups.

        Args:
            objects_links_or_prim_paths (set of EntityPrim, RigidPrim, str, or USDObject): Objects, links, or prim paths to convert to prim paths.

        Returns:
            list[str]: List of prim paths.
        """
        # Avoid circular imports
        from omnigibson.prims.entity_prim import EntityPrim
        from omnigibson.prims.rigid_prim import RigidPrim

        outputs = []
        for inp in objects_links_or_prim_paths:
            if isinstance(inp, EntityPrim):
                outputs.extend([link.prim_path for link in inp.links.values()])
            elif isinstance(inp, RigidPrim):
                outputs.append(inp.prim_path)
            elif isinstance(inp, str):
                outputs.append(inp)
            else:
                raise ValueError(f"Input set must be a set of EntityPrim, RigidPrim, or str, found {type(inp)}")
        return outputs

    def get_contact_row_indices(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets the row indices of the contact matrix for a given set of objects, links, or prim paths.
        This is the index of the rigid body in the contact matrix. This can be used by external callers to
        pre-cache the indices they care about for faster lookups later (e.g. avoiding a lookup on every call to is_in_contact).

        Args:
            scene_idx (int): Scene index to get the contact row indices for.
            objects_links_or_prim_paths (set of EntityPrim, RigidPrim, str, or USDObject): Objects, links, or prim paths to get the contact row indices for.

        Returns:
            th.Tensor: Tensor of row indices.
        """
        # If the input is already a tensor just return it
        if isinstance(objects_links_or_prim_paths, th.Tensor):
            return objects_links_or_prim_paths

        # Otherwise, convert to prim paths, filtering out kinematic-only bodies that are not rows.
        # dtype=long so an empty result (all paths kinematic-only) is still safe to use as an index.
        prim_paths = self._get_prim_paths(objects_links_or_prim_paths)
        row_map = self._PATH_TO_ROW_IDX.get(scene_idx, {})
        return th.tensor([row_map[path] for path in prim_paths if path in row_map], dtype=th.long)

    def get_contact_col_indices(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets the column indices of the contact matrix for a given set of objects, links, or prim paths.
        This is the index of the rigid body in the contact matrix. This can be used by external callers to
        pre-cache the indices they care about for faster lookups later (e.g. avoiding a lookup on every call to is_in_contact).

        Args:
            scene_idx (int): Scene index to get the contact column indices for.
            objects_links_or_prim_paths (set of EntityPrim, RigidPrim, str, or USDObject): Objects, links, or prim paths to get the contact column indices for.

        Returns:
            th.Tensor: Tensor of column indices.
        """
        # If the input is already a tensor just return it
        if isinstance(objects_links_or_prim_paths, th.Tensor):
            return objects_links_or_prim_paths

        # Otherwise, convert to prim paths, filtering out bodies without contact reporting
        # (e.g. visual-only links) that are not columns. dtype=long so an empty result is safe
        # to use as an index.
        prim_paths = self._get_prim_paths(objects_links_or_prim_paths)
        col_map = self._PATH_TO_COL_IDX.get(scene_idx, {})
        return th.tensor([col_map[path] for path in prim_paths if path in col_map], dtype=th.long)

    def get_contact_pairs(self, scene_idx, query_set, with_set, current_only):
        """
        Get pairs of prim paths that are in contact.

        Args:
            scene_idx (int): Scene index to get the contact pairs for.
            query_set (set of RigidPrim, str, or USDObject): Prims, prim paths, or objects for contact sensor objects to check. Must be specified.
            with_set (set of RigidPrim, str, or USDObject): Prims, prim paths, or objects to filter the contact pairs by. Only these objects will be considered for contact. Can be None to check for contact with any object.
            current_only (bool): If True, only checks the most recent physics step. If False, checks whether contact occurred at any physics step since the last non-physics step.
                The True mode is recommended for use cases like Touching state etc. where a contact at the current position of the object is important.
                The False mode is recommended for use cases like transition rules etc. where a contact at any point during the last N physics steps is enough (e.g. as a trigger event).

        Returns:
            set of tuples: Set of tuples of (query_prim_path, filter_prim_path) pairs that are in contact.
        """
        if scene_idx not in self._CONTACT_MATRIX or scene_idx not in self._PATH_TO_COL_IDX:
            return set()
        contact_matrix = self._CURRENT_CONTACT_MATRIX[scene_idx] if current_only else self._CONTACT_MATRIX[scene_idx]
        assert contact_matrix.ndim == 2, f"Contact matrix should be 2D, found shape {contact_matrix.shape}"

        # Get the row indices corresponding to the sensor prim paths
        row_idxs = self.get_contact_row_indices(scene_idx, query_set)

        # Slice the contact matrix and the index matrix with the same indexing so that
        # nonzero positions in the submatrix can be mapped back to original row/col indices.
        idx_matrix = self._INDEX_MATRIX[scene_idx]
        if with_set is not None:
            col_idxs = self.get_contact_col_indices(scene_idx, with_set)
            in_contact = contact_matrix[row_idxs[:, None], col_idxs[None, :]]
            idx_matrix = idx_matrix[row_idxs[:, None], col_idxs[None, :]]
        else:
            in_contact = contact_matrix[row_idxs, :]
            idx_matrix = idx_matrix[row_idxs, :]

        # Early return if not in contact.
        if not th.any(in_contact).item():
            return set()

        original_indices = idx_matrix[in_contact].cpu().tolist()

        return {
            (self._ROW_IDX_TO_PATH[scene_idx][row], self._COL_IDX_TO_PATH[scene_idx][col])
            for row, col in original_indices
        }

    def is_in_contact(self, scene_idx, query_set, with_set, ignore_set, current_only):
        """
        Check if any of the prims in @query_set are in contact with any of the prims in @with_set, or not in contact with any of the prims in @ignore_set.

        Returns CPU tensors.

        Args:
            scene_idx (int): Scene index to check for contact.
            query_set (set of RigidPrim, str, or USDObject): Prims, prim paths, or objects to check for contact.
            with_set (set of RigidPrim, str, or USDObject): Prims, prim paths, or objects to check for contact with. Can be None to check for contact with any object.
            ignore_set (set of RigidPrim, str, or USDObject): Prims, prim paths, or objects to ignore contact with. Can be None to not ignore any objects.
            current_only (bool): If True, only checks the most recent physics step. If False, checks whether contact occurred at any physics step since the last non-physics step.
                The True mode is recommended for use cases like Touching state etc. where a contact at the current position of the object is important.
                The False mode is recommended for use cases like transition rules etc. where a contact at any point during the last N physics steps is enough (e.g. as a trigger event).

        Returns:
            bool: True if any of the prims in @query_set are in contact with any of the prims in @with_set, or not in contact with any of the prims in @ignore_set, else False.
        """
        if with_set is not None and ignore_set is not None:
            raise ValueError("At most one of with_set or ignore_set may be specified.")

        if scene_idx not in self._CONTACT_MATRIX or scene_idx not in self._PATH_TO_COL_IDX:
            return False

        contact_matrix = self._CURRENT_CONTACT_MATRIX[scene_idx] if current_only else self._CONTACT_MATRIX[scene_idx]
        rows = self.get_contact_row_indices(scene_idx, query_set)
        if rows.numel() == 0:
            return False
        if with_set is not None:
            cols = self.get_contact_col_indices(scene_idx, with_set)
            return th.any(contact_matrix[rows, :][:, cols]).item()
        elif ignore_set is not None:
            ignore_mask = th.ones(contact_matrix.shape[1], dtype=th.bool)
            ignore_mask[self.get_contact_col_indices(scene_idx, ignore_set)] = False
            return th.any(contact_matrix[rows, :][:, ignore_mask]).item()

        # Base case, return any collisions with any other prim
        return th.any(contact_matrix[rows]).item()

    def get_contact_row_mask(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets a boolean mask over contact matrix rows for a given set of objects, links, or prim paths.
        Useful for building batch query masks for :meth:`is_in_contact_batch`.

        Args:
            scene_idx (int): Scene index.
            objects_links_or_prim_paths: Objects, links, or prim paths (or a pre-cached index tensor).

        Returns:
            th.Tensor: (R,) boolean tensor where R is the number of contact matrix rows.
        """
        R = self._CONTACT_MATRIX[scene_idx].shape[0]
        idxs = self.get_contact_row_indices(scene_idx, objects_links_or_prim_paths)
        mask = th.zeros(R, dtype=th.bool)
        mask[idxs] = True
        return mask

    def get_contact_col_mask(self, scene_idx, objects_links_or_prim_paths):
        """
        Gets a boolean mask over contact matrix columns for a given set of objects, links, or prim paths.
        Useful for building batch with/ignore masks for :meth:`is_in_contact_batch`.

        Args:
            scene_idx (int): Scene index.
            objects_links_or_prim_paths: Objects, links, or prim paths (or a pre-cached index tensor).

        Returns:
            th.Tensor: (C,) boolean tensor where C is the number of contact matrix columns.
        """
        C = self._CONTACT_MATRIX[scene_idx].shape[1]
        idxs = self.get_contact_col_indices(scene_idx, objects_links_or_prim_paths)
        mask = th.zeros(C, dtype=th.bool)
        mask[idxs] = True
        return mask

    def is_in_contact_batch(self, scene_idx, query_masks, with_masks, ignore_masks, current_only):
        """
        Batch contact check for N queries, fully tensorized. Returns GPU tensors.

        Each row ``i`` of the input masks defines one independent contact query. The method
        returns an ``(N,)`` boolean tensor where entry ``i`` is ``True`` iff any row selected
        by ``query_masks[i]`` is in contact with any column selected by ``with_masks[i]``
        (or any column *not* in ``ignore_masks[i]``).

        Provide either ``with_masks`` for all N queries or ``ignore_masks`` for all N queries,
        but not both (and not a per-query mix).

        Use :meth:`get_contact_row_mask` / :meth:`get_contact_col_mask` to build individual
        masks, then ``torch.stack`` them into the ``(N, R)`` / ``(N, C)`` tensors expected here.

        Args:
            scene_idx (int): Scene index to check for contact.
            query_masks (th.Tensor): ``(N, R)`` boolean tensor. ``query_masks[i, j]`` is True
                if contact-matrix row ``j`` belongs to query set ``i``.
            with_masks (th.Tensor or None): ``(N, C)`` boolean tensor. ``with_masks[i, j]`` is
                True if contact-matrix column ``j`` belongs to the with-set for query ``i``. Can be None to check for contact with any object.
            ignore_masks (th.Tensor or None): ``(N, C)`` boolean tensor. ``ignore_masks[i, j]``
                is True if contact-matrix column ``j`` should be *ignored* for query ``i``. Can be None to not ignore any objects.
            current_only (bool): If True, only checks the most recent physics step. If False, checks whether contact occurred at any physics step since the last non-physics step.
                The True mode is recommended for use cases like Touching state etc. where a contact at the current position of the object is important.
                The False mode is recommended for use cases like transition rules etc. where a contact at any point during the last N physics steps is enough (e.g. as a trigger event).

        Returns:
            th.Tensor: ``(N,)`` boolean tensor of contact results.
        """
        assert with_masks is None or ignore_masks is None, "Provide either with_masks or ignore_masks, not both."

        if scene_idx not in self._CONTACT_MATRIX_GPU or scene_idx not in self._PATH_TO_COL_IDX:
            return th.zeros(query_masks.shape[0], dtype=th.bool, device="cuda")

        contact_matrix = (
            self._CURRENT_CONTACT_MATRIX_GPU[scene_idx] if current_only else self._CONTACT_MATRIX_GPU[scene_idx]
        )

        # query_contacts[i, c] = True iff any row in query set i is in contact with column c.
        # We use float matmul for speed: (N, R) @ (R, C) -> (N, C), then threshold.
        query_contacts = (query_masks.float() @ contact_matrix.float()) > 0

        if with_masks is not None:
            return (query_contacts & with_masks).any(dim=1)
        elif ignore_masks is not None:
            return (query_contacts & ~ignore_masks).any(dim=1)

        return query_contacts.any(dim=1)

    def is_in_contact_batch_warp(self, scene_idx, query_masks_wp, with_masks_wp, ignore_masks_wp, current_only, out_wp):
        """
        Warp-kernel variant of is_in_contact_batch. Same semantics, but takes pre-cached
        wp.array inputs (uint8 masks) and writes to a caller-allocated output wp.array.
        Designed to be safe to call inside wp.graph capture.

        out_wp must be int32, NOT uint8.

        Args:
            scene_idx (int): Scene index.
            query_masks_wp (wp.array): (N, R) uint8 — wp.from_torch wrapper of bool query masks.
            with_masks_wp (wp.array | None): (N, C) uint8 — with-mask, mutually exclusive with ignore_masks_wp.
            ignore_masks_wp (wp.array | None): (N, C) uint8 — ignore-mask.
            current_only (bool): True → use _CURRENT_CONTACT_MATRIX_GPU_WP, else _CONTACT_MATRIX_GPU_WP.
            out_wp (wp.array): (N,) **int32** — caller-allocated, must be pre-zeroed.

        If the scene has no contact view yet (e.g. only kinematic bodies), no kernel is
        launched and out_wp is left as-is.
        """
        assert (
            with_masks_wp is None or ignore_masks_wp is None
        ), "Provide either with_masks_wp or ignore_masks_wp, not both."
        # Scene without a contact view: nothing to do; caller's out_wp is left as-is.
        if scene_idx not in self._CONTACT_MATRIX_GPU_WP:
            return

        contact_matrix_wp = (
            self._CURRENT_CONTACT_MATRIX_GPU_WP[scene_idx] if current_only else self._CONTACT_MATRIX_GPU_WP[scene_idx]
        )

        # Decide kernel mode + col_filter argument. The kernel always reads from col_filter,
        # so when there's no real filter we substitute query_masks_wp as a dummy of compatible
        # shape — mode=2 means the kernel ignores it.
        if with_masks_wp is not None:
            col_filter = with_masks_wp
            mode = 0
        elif ignore_masks_wp is not None:
            col_filter = ignore_masks_wp
            mode = 1
        else:
            # Dummy placeholder; kernel ignores when mode=2. Use query_masks_wp for matching device.
            col_filter = query_masks_wp
            mode = 2

        N = query_masks_wp.shape[0]
        R = contact_matrix_wp.shape[0]
        C = contact_matrix_wp.shape[1]

        # 3D launch: one thread per (query, row, col) cell. Each thread does at most a few
        # reads + 1 atomic_max on hit. Maximum parallelism, no serial loops.
        wp.launch(
            kernel=_is_in_contact_batch_kernel,
            dim=(N, R, C),
            inputs=[
                query_masks_wp,
                contact_matrix_wp,
                col_filter,
                wp.int32(mode),
                out_wp,
            ],
            device="cuda",
        )

    def has_contact_view(self, scene_idx):
        """Returns True if a valid contact view has been initialized for @scene_idx."""
        return scene_idx in self._CONTACT_MATRIX

    def get_contact_matrix_shape(self, scene_idx):
        """
        Returns:
            tuple(int, int) | None: (R, C) shape of the contact matrix for @scene_idx, or None
            if no contact view exists for that scene.
        """
        if scene_idx not in self._CONTACT_MATRIX:
            return None
        return tuple(self._CONTACT_MATRIX[scene_idx].shape)

    def get_contact_matrix_wp(self, scene_idx, current_only):
        """
        Get the wp.array view of the GPU contact matrix for use inside Warp kernels (e.g. inside
        a captured wp.graph). Same view that ``is_in_contact_batch_warp`` consumes internally.

        Args:
            scene_idx (int): Scene index.
            current_only (bool): True → "current step only" matrix, False → "any recent step" matrix.

        Returns:
            wp.array(uint8) | None: (R, C) uint8 view, or None if no contact view exists for @scene_idx.
        """
        if scene_idx not in self._CONTACT_MATRIX_GPU_WP:
            return None
        return (
            self._CURRENT_CONTACT_MATRIX_GPU_WP[scene_idx] if current_only else self._CONTACT_MATRIX_GPU_WP[scene_idx]
        )

    def clear(self):
        """
        Clears internal contact views, mappings, and caches.
        """
        self._PATH_TO_ROW_IDX = dict()
        self._PATH_TO_COL_IDX = dict()
        self._ROW_IDX_TO_PATH = dict()
        self._COL_IDX_TO_PATH = dict()
        self._CONTACT_VIEW = dict()
        self._RIGID_BODY_VIEW = dict()
        self._CONTACT_MATRIX_ROWS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_TO_RIGID_BODY_ROWS = dict()
        self._CONTACT_MATRIX_COLS_HAS_RIGID_BODY = dict()
        self._ROW_TO_RIGID_WP = dict()
        self._COL_TO_RIGID_WP = dict()
        self._BODY_TO_ROW = dict()
        self._BODY_TO_ROW_WP = dict()
        self._CONTACT_MATRIX = dict()
        self._CONTACT_MATRIX_GPU = dict()
        self._CONTACT_MATRIX_GPU_WP = dict()
        self._CONTACT_MATRIX_HOST_WP = dict()
        self._CURRENT_CONTACT_MATRIX = dict()
        self._CURRENT_CONTACT_MATRIX_GPU = dict()
        self._CURRENT_CONTACT_MATRIX_GPU_WP = dict()
        self._CURRENT_CONTACT_MATRIX_HOST_WP = dict()
        self._INDEX_MATRIX = dict()
        self._BODY_TRANSFORMS = dict()
        self._BODY_TRANSFORMS_WP = dict()
        self._PENDING_STEPS = 0
        self._PENDING_IMPULSES = dict()
        self._PENDING_TRANSFORMS = dict()
        self._PENDING_NET_FORCES = dict()
        self._PENDING_IMPULSES_WP = dict()
        self._PENDING_TRANSFORMS_WP = dict()
        self._PENDING_NET_FORCES_WP = dict()


# Instantiate the RigidContactAPI
RigidContactAPI = RigidContactAPIImpl()


class CollisionAPI:
    """
    Class containing class methods to facilitate collision handling, e.g. collision groups
    """

    ACTIVE_COLLISION_GROUPS = dict()

    @classmethod
    def create_collision_group(cls, col_group, filter_self_collisions=False):
        """
        Creates a new collision group with name @col_group

        Args:
            col_group (str): Name of the collision group to create
            filter_self_collisions (bool): Whether to ignore self-collisions within the group. Default is False
        """
        with og.sim.editing_usd():
            # Can only be done when sim is stopped
            assert og.sim is None or og.sim.is_stopped(), "Cannot create a collision group unless og.sim is stopped!"

            # Make sure the group doesn't already exist
            assert (
                col_group not in cls.ACTIVE_COLLISION_GROUPS
            ), f"Cannot create collision group {col_group} because it already exists!"

            # Create the group
            col_group_prim_path = f"/World/collision_groups/{col_group}"
            group = lazy.pxr.UsdPhysics.CollisionGroup.Define(og.sim.stage, col_group_prim_path)
            if filter_self_collisions:
                # Do not collide with self
                group.GetFilteredGroupsRel().AddTarget(col_group_prim_path)
            cls.ACTIVE_COLLISION_GROUPS[col_group] = group

    @classmethod
    def add_to_collision_group(cls, col_group, prim_path):
        """
        Adds the prim and all nested prims specified by @prim_path to the global collision group @col_group. If @col_group
        does not exist, then it will either be created if @create_if_not_exist is True, otherwise will raise an Error.
        Args:
            col_group (str): Name of the collision group to assign the prim at @prim_path to
            prim_path (str): Prim (and all nested prims) to assign to this @col_group
        """
        with og.sim.editing_usd():
            # Make sure collision group exists
            assert (
                col_group in cls.ACTIVE_COLLISION_GROUPS
            ), f"Cannot add to collision group {col_group} because it does not exist!"

            # Add this prim to the collision group
            cls.ACTIVE_COLLISION_GROUPS[col_group].GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim_path)

    @classmethod
    def add_group_filter(cls, col_group, filter_group):
        """
        Adds a new group filter for group @col_group, filtering all collision with group @filter_group
        Args:
            col_group (str): Name of the collision group which will have a new filter group added
            filter_group (str): Name of the group that should be filtered
        """
        with og.sim.editing_usd():
            # Make sure the group doesn't already exist
            for group_name in (col_group, filter_group):
                assert group_name in cls.ACTIVE_COLLISION_GROUPS, (
                    f"Cannot add group filter {filter_group} to collision group {col_group} because at least one group "
                    f"does not exist!"
                )

            # Grab the group, and add the filter
            filter_group_prim_path = f"/World/collision_groups/{filter_group}"
            group = cls.ACTIVE_COLLISION_GROUPS[col_group]
            group.GetFilteredGroupsRel().AddTarget(filter_group_prim_path)

    @classmethod
    def clear(cls):
        """
        Clears the internal state of this CollisionAPI
        """
        # Remove all the collision group prims
        for col_group_prim in cls.ACTIVE_COLLISION_GROUPS.values():
            delete_or_deactivate_prim(col_group_prim.GetPath().pathString)

        # Remove the collision groups tree
        delete_or_deactivate_prim("/World/collision_groups")

        # Clear the dictionary
        cls.ACTIVE_COLLISION_GROUPS = {}


def setup_collision_apis(prim):
    """
    Apply collision-related physics APIs to a USD prim. This should be called for prims
    that are identified as collision meshes (e.g. those appearing under a "collisions" scope prim).

    This applies the CollisionAPI, PhysxCollisionAPI, and (for meshes) MeshCollisionAPI to the prim,
    sets a default convex hull collision approximation for mesh types, and enables/disables collisions
    based on the global VISUAL_ONLY setting.

    Note: This does NOT set the prim's purpose. The caller should set purpose as appropriate
    (e.g. "guide" for collision-only meshes, "default" for collision+visual meshes).

    Args:
        prim: The USD prim to set up collision APIs on.

    Returns:
        tuple: (collision_api, physx_collision_api, mesh_collision_api) where mesh_collision_api
            may be None for non-mesh prims.
    """
    # Create / get CollisionAPI reference
    collision_api = ensure_usd_api(prim, lazy.pxr.UsdPhysics.CollisionAPI)
    physx_collision_api = ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxCollisionAPI)
    mesh_collision_api = (
        ensure_usd_api(prim, lazy.pxr.UsdPhysics.MeshCollisionAPI)
        if prim.GetPrimTypeInfo().GetTypeName() == "Mesh"
        else None
    )

    # Optionally add mesh collision API if this is a mesh
    if mesh_collision_api is not None:
        # Set the approximation to be convex hull by default
        apply_collision_approximation(prim, mesh_collision_api, "convexHull")

    with og.sim.editing_usd():
        # Set collision enabled based on global setting
        collision_api.GetCollisionEnabledAttr().Set(not gm.VISUAL_ONLY)

    return collision_api, physx_collision_api, mesh_collision_api


def apply_collision_approximation(prim, mesh_collision_api, approximation_type):
    """
    Apply a collision approximation type to a single collision mesh prim.

    Args:
        prim: The USD prim to apply the collision approximation to.
        mesh_collision_api: The UsdPhysics.MeshCollisionAPI for this prim.
        approximation_type (str): Approximation type to use. One of:
            {"none", "convexHull", "convexDecomposition", "meshSimplification", "sdf",
             "boundingSphere", "boundingCube"}
    """
    assert mesh_collision_api is not None, "collision_approximation only applicable for meshes!"
    assert_valid_key(
        key=approximation_type,
        valid_keys={
            "none",
            "convexHull",
            "convexDecomposition",
            "meshSimplification",
            "sdf",
            "boundingSphere",
            "boundingCube",
        },
        name="collision approximation type",
    )

    # Make sure to add the appropriate API if we're setting certain values
    if approximation_type == "convexHull":
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI)
    elif approximation_type == "convexDecomposition":
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxConvexDecompositionCollisionAPI)
    elif approximation_type == "meshSimplification":
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI)
    elif approximation_type == "sdf":
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxSDFMeshCollisionAPI)
    elif approximation_type == "none":
        ensure_usd_api(prim, lazy.pxr.PhysxSchema.PhysxTriangleMeshCollisionAPI)

    with og.sim.editing_usd():
        if approximation_type == "convexHull":
            pch_api = lazy.pxr.PhysxSchema.PhysxConvexHullCollisionAPI(prim)
            # Also make sure the maximum vertex count is 60 (max number compatible with GPU)
            # https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html#collision-settings
            if pch_api.GetHullVertexLimitAttr().Get() is None:
                pch_api.CreateHullVertexLimitAttr()
            pch_api.GetHullVertexLimitAttr().Set(60)

        mesh_collision_api.GetApproximationAttr().Set(approximation_type)


def get_world_pose(prim_path):
    """
    Gets pose of the prim object with respect to the world frame
    Args:
        Prim_path: the path of the prim object
    Returns:
        2-tuple:
            - torch.Tensor: (x,y,z) position in the world frame
            - torch.Tensor: (x,y,z,w) quaternion orientation in the world frame
    """
    matrix = _get_world_pose_with_scale_from_fabric_hierarchy(prim_path)
    quaternion = matrix.RemoveScaleShear().ExtractRotationQuat()
    position = th.tensor(matrix.ExtractTranslation(), dtype=th.float32)
    orientation = th.tensor([*quaternion.GetImaginary(), quaternion.GetReal()], dtype=th.float32)
    return position, orientation


def _get_world_pose_with_scale_from_fabric_hierarchy(prim_path):
    # Check that no reads from Fabric are happening during a physics step.
    assert not og.sim.currently_stepping, "Do not read poses from Fabric during a physics step, this is quite slow!"

    return og.sim.fabric_hierarchy.get_world_xform(lazy.usdrt.Sdf.Path(prim_path))


def get_world_pose_with_scale(prim_path):
    """
    This is used when information about the prim's global scale is needed,
    e.g. when converting points in the prim frame to the world frame.
    """

    return th.tensor(_get_world_pose_with_scale_from_fabric_hierarchy(prim_path), dtype=th.float32).T


def get_local_pose(prim_path):
    """
    Gets pose of the prim with respect to its parent prim's frame (local / parent-relative transform).

    Args:
        prim_path: the path of the prim object

    Returns:
        2-tuple:
            - torch.Tensor: (x,y,z) position in the parent frame
            - torch.Tensor: (x,y,z,w) quaternion orientation in the parent frame
    """
    matrix = _get_local_pose_with_scale_from_fabric_hierarchy(prim_path)
    quaternion = matrix.RemoveScaleShear().ExtractRotationQuat()
    position = th.tensor(matrix.ExtractTranslation(), dtype=th.float32)
    orientation = th.tensor([*quaternion.GetImaginary(), quaternion.GetReal()], dtype=th.float32)
    return position, orientation


def _get_local_pose_with_scale_from_fabric_hierarchy(prim_path):
    assert not og.sim.currently_stepping, "Do not read poses from Fabric during a physics step, this is quite slow!"

    return og.sim.fabric_hierarchy.get_local_xform(lazy.usdrt.Sdf.Path(prim_path))


def get_local_pose_with_scale(prim_path):
    """
    Like get_local_pose, but returns the full 4x4 local transform matrix (with scale),
    for converting points between the prim frame and the parent frame.
    """

    return th.tensor(_get_local_pose_with_scale_from_fabric_hierarchy(prim_path), dtype=th.float32).T


class BatchControlViewAPIImpl:
    """
    A centralized view that allows for reading and writing to an ArticulationView that covers multiple
    controllable objects in the scene. This is used to avoid the overhead of reading from many views
    for each robot in each physics step, a source of significant overhead.

    **Compute backend:** Isaac's physics sim articulation view APIs return **torch** tensors. This layer
    caches **compute-backend arrays** (``cb.arr_type``). All public getters on this class therefore
    return ``cb`` arrays (positions, quaternions, Jacobians, etc.). Batched writes from controllers
    expect ``cb`` arrays as well. ``flush_control`` converts cached targets back to torch for the PhysX backend.
    """

    def __init__(self, pattern):
        # The prim path pattern that will be passed into the view
        self._pattern = pattern

        # The unified ArticulationView used to access all of the controllable objects in the scene.
        self._view = None

        # Cache for all of the view functions' return values within the same simulation step.
        # Keyed by function name without get_, the value is the return value of the function.
        self._read_cache = {}

        # Cache for all of the view functions' write values within the same simulation step.
        # Keyed by the function name without set_, the value is the set of indices that need to be updated.
        self._write_idx_cache = collections.defaultdict(set)

        # Mapping from prim path to index in the view.
        self._idx = {}

        # Mapping from prim idx to a dict that maps link name to link index in the view.
        self._link_idx = {}

        # Mapping from prim path to base footprint link name if one exists, None if the root is the base link.
        self._base_footprint_link_names = {}

        # Prior link transforms / dof positions for estimating velocities since Omni gives inaccurate values
        self._last_state = None

    def post_physics_step(self):
        # Should be called every sim physics step, right after a new physics step occurs
        # The current poses (if it exists) are now the former poses from the previous timestep
        # These values are needed to compute velocity estimates
        if (
            "root_transforms" in self._read_cache
            and "link_transforms" in self._read_cache
            and "dof_positions" in self._read_cache
        ):
            self._last_state = {
                "root_transforms": cb.copy(self._read_cache["root_transforms"]),
                "link_transforms": cb.copy(self._read_cache["link_transforms"]),
                "dof_positions": cb.copy(self._read_cache["dof_positions"]),
            }
        else:
            # We don't have enough info to populate the history, so simply clear it instead
            self._last_state = None

        # Clear the internal data since everything is outdated
        self.clear(keep_last_pose=True)

    def clear(self, keep_last_pose=False):
        self._read_cache = {}
        self._write_idx_cache = collections.defaultdict(set)

        # Clear our last timestep's cached values by default
        if not keep_last_pose:
            self._last_state = None

        # Cache the (now current) transforms so that they're guaranteed to exist throughout the duration of this
        # timestep, and available for caching during the next timestep's post_physics_step() call
        if og.sim.is_playing():
            self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())
            self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())
            self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())

    def _set_dof_position_targets(self, data, indices, cast=True):
        # No casting results in better efficiency
        if cast:
            data = self._view._frontend.as_contiguous_float32(data)
            indices = self._view._frontend.as_contiguous_uint32(indices)
        data_desc = self._view._frontend.get_tensor_desc(data)
        indices_desc = self._view._frontend.get_tensor_desc(indices)

        if not self._view._backend.set_dof_position_targets(data_desc, indices_desc):
            raise Exception("Failed to set DOF positions in backend")

    def _set_dof_velocity_targets(self, data, indices, cast=True):
        # No casting results in better efficiency
        if cast:
            data = self._view._frontend.as_contiguous_float32(data)
            indices = self._view._frontend.as_contiguous_uint32(indices)
        data_desc = self._view._frontend.get_tensor_desc(data)
        indices_desc = self._view._frontend.get_tensor_desc(indices)

        if not self._view._backend.set_dof_velocity_targets(data_desc, indices_desc):
            raise Exception("Failed to set DOF velocities in backend")

    def _set_dof_actuation_forces(self, data, indices, cast=True):
        # No casting results in better efficiency
        if cast:
            data = self._view._frontend.as_contiguous_float32(data)
            indices = self._view._frontend.as_contiguous_uint32(indices)
        data_desc = self._view._frontend.get_tensor_desc(data)
        indices_desc = self._view._frontend.get_tensor_desc(indices)

        if not self._view._backend.set_dof_actuation_forces(data_desc, indices_desc):
            raise Exception("Failed to set DOF actuation forces in backend")

    def flush_control(self):
        if "dof_position_targets" in self._write_idx_cache:
            pos_indices = cb.int_array(sorted(self._write_idx_cache["dof_position_targets"]))
            pos_targets = self._read_cache["dof_position_targets"]
            self._set_dof_position_targets(cb.to_torch(pos_targets), cb.to_torch(pos_indices), cast=False)

        if "dof_velocity_targets" in self._write_idx_cache:
            vel_indices = cb.int_array(sorted(self._write_idx_cache["dof_velocity_targets"]))
            vel_targets = self._read_cache["dof_velocity_targets"]
            self._set_dof_velocity_targets(cb.to_torch(vel_targets), cb.to_torch(vel_indices), cast=False)

        if "dof_actuation_forces" in self._write_idx_cache:
            eff_indices = cb.int_array(sorted(self._write_idx_cache["dof_actuation_forces"]))
            eff_targets = self._read_cache["dof_actuation_forces"]
            self._set_dof_actuation_forces(cb.to_torch(eff_targets), cb.to_torch(eff_indices), cast=False)

    def initialize_view(self):
        # First, get all of the controllable objects in the scene (avoiding circular import)
        from omnigibson.robots import Robot

        controllable_objects = [obj for scene in og.sim.scenes for obj in scene.objects if isinstance(obj, Robot)]

        # Get their corresponding prim paths
        expected_prim_paths = {obj.articulation_root_path for obj in controllable_objects}

        # Apply the pattern to find the expected prim paths
        expected_prim_paths = {
            prim_path for prim_path in expected_prim_paths if re.fullmatch(self._pattern.replace("*", ".*"), prim_path)
        }

        # Make sure we have at least one controllable object
        if len(expected_prim_paths) == 0:
            return

        # Create the actual articulation view. Note that even though we search for base_link here,
        # the returned things will not necessarily be the base_link prim paths, but the appropriate
        # articulation root path for every object (base_link for non-fixed, parent for fixed objects)
        self._view = og.sim.physics_sim_view.create_articulation_view(self._pattern)
        view_prim_paths = self._view.prim_paths
        assert (
            set(view_prim_paths) == expected_prim_paths
        ), f"ControllableObjectViewAPI expected prim paths {expected_prim_paths} but got {view_prim_paths}"

        # Create the mapping from prim path to index
        self._idx = {prim_path: i for i, prim_path in enumerate(view_prim_paths)}
        self._link_idx = [
            {link_path.split("/")[-1]: j for j, link_path in enumerate(articulation_link_paths)}
            for articulation_link_paths in self._view.link_paths
        ]
        self._base_footprint_link_names = {
            obj.articulation_root_path: (
                obj.base_footprint_link_name if obj.base_footprint_link_name != obj.root_link_name else None
            )
            for obj in controllable_objects
            if obj.articulation_root_path in expected_prim_paths
        }

    def set_joint_position_targets(self, prim_path, positions, indices):
        assert len(indices) == len(positions), "Indices and values must have the same length"
        idx = self._idx[prim_path]

        # Load the current targets.
        if "dof_position_targets" not in self._read_cache:
            self._read_cache["dof_position_targets"] = cb.from_torch(self._view.get_dof_position_targets())

        # Update the target
        self._read_cache["dof_position_targets"][idx, indices] = positions

        # Add this index to the write cache
        self._write_idx_cache["dof_position_targets"].add(idx)

    def set_joint_velocity_targets(self, prim_path, velocities, indices):
        assert len(indices) == len(velocities), "Indices and values must have the same length"
        idx = self._idx[prim_path]

        # Load the current targets.
        if "dof_velocity_targets" not in self._read_cache:
            self._read_cache["dof_velocity_targets"] = cb.from_torch(self._view.get_dof_velocity_targets())

        # Update the target
        self._read_cache["dof_velocity_targets"][idx, indices] = velocities

        # Add this index to the write cache
        self._write_idx_cache["dof_velocity_targets"].add(idx)

    def set_joint_efforts(self, prim_path, efforts, indices):
        assert len(indices) == len(efforts), "Indices and values must have the same length"
        idx = self._idx[prim_path]

        # Load the current targets.
        if "dof_actuation_forces" not in self._read_cache:
            self._read_cache["dof_actuation_forces"] = cb.from_torch(self._view.get_dof_actuation_forces())

        # Update the target
        self._read_cache["dof_actuation_forces"][idx, indices] = efforts

        # Add this index to the write cache
        self._write_idx_cache["dof_actuation_forces"].add(idx)

    def get_member_view_indices(self, prim_paths):
        """Return view row index for each prim_path (in input order)."""
        return [self._idx[p] for p in prim_paths]

    def set_all_joint_position_targets(self, enabled_rows, controls, dof_idx):
        """
        Args:
            enabled_rows: list[int] — view row indices for enabled members (pre-filtered)
            controls: (N_enabled, len(dof_idx)) compute-backend array — pre-stacked by controller
            dof_idx: DOF column indices (cb.arr_type)
        """
        if "dof_position_targets" not in self._read_cache:
            self._read_cache["dof_position_targets"] = cb.from_torch(self._view.get_dof_position_targets())
        targets = self._read_cache["dof_position_targets"]
        row_idx = cb.int_array(enabled_rows).reshape(-1, 1)
        targets[row_idx, dof_idx] = controls
        self._write_idx_cache["dof_position_targets"].update(enabled_rows)

    def set_all_joint_velocity_targets(self, enabled_rows, velocities, dof_idx):
        if "dof_velocity_targets" not in self._read_cache:
            self._read_cache["dof_velocity_targets"] = cb.from_torch(self._view.get_dof_velocity_targets())
        targets = self._read_cache["dof_velocity_targets"]
        row_idx = cb.int_array(enabled_rows).reshape(-1, 1)
        targets[row_idx, dof_idx] = velocities
        self._write_idx_cache["dof_velocity_targets"].update(enabled_rows)

    def set_all_joint_efforts(self, enabled_rows, efforts, dof_idx):
        if "dof_actuation_forces" not in self._read_cache:
            self._read_cache["dof_actuation_forces"] = cb.from_torch(self._view.get_dof_actuation_forces())
        targets = self._read_cache["dof_actuation_forces"]
        row_idx = cb.int_array(enabled_rows).reshape(-1, 1)
        targets[row_idx, dof_idx] = efforts
        self._write_idx_cache["dof_actuation_forces"].update(enabled_rows)

    def get_all_root_transform(self):
        if "root_transforms" not in self._read_cache:
            self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())
        pose = self._read_cache["root_transforms"]
        return pose[:, :3], pose[:, 3:]

    def get_root_transform(self, prim_path):
        idx = self._idx[prim_path]
        pos, quat = self.get_all_root_transform()
        return pos[idx], quat[idx]

    def get_all_position_orientation(self):
        # Here we want to return the position of the base footprint link.
        # If the base footprint link is None, we return the position of the root link.

        # we assume that in a view, all base link_name is the same
        link_name = next(iter(self._base_footprint_link_names.values()))
        if link_name is None:
            return self.get_all_root_transform()
        else:
            return self.get_all_link_transform(link_name)

    def get_position_orientation(self, prim_path):
        # Here we want to return the position of the base footprint link. If the base footprint link is None,
        # we return the position of the root link.
        if self._base_footprint_link_names[prim_path] is not None:
            link_name = self._base_footprint_link_names[prim_path]
            return self.get_link_transform(prim_path, link_name)
        else:
            return self.get_root_transform(prim_path)

    def _get_all_velocities(self, estimate=False):
        link_name = next(iter(self._base_footprint_link_names.values()))
        if link_name is not None:
            return self._get_all_link_velocities(link_name, estimate=estimate)
        else:
            return self._get_all_root_velocities(estimate=estimate)

    def _get_velocities(self, prim_path, estimate=False):
        """World-frame linear + angular velocity for one articulation (6,) from the batched cache."""
        idx = self._idx[prim_path]
        return self._get_all_velocities(estimate=estimate)[idx]

    def _get_all_relative_velocities(self, estimate=False):
        """Returns (N, n_links+1, 6) relative velocities for all robots; final slot [-1] is the base."""
        vel_str = "velocities_estimate" if estimate else "velocities"

        if f"all_relative_{vel_str}" not in self._read_cache:
            # Warm the (N, L, 6) link velocity cache and fetch it
            any_link_name = next(iter(self._link_idx[0]))
            self._get_all_link_velocities(any_link_name, estimate=estimate)
            link_vels = cb.to_torch(self._read_cache[f"link_{vel_str}"])  # (N, L, 6)

            # Get base velocities (N, 6): reuse link cache if a base footprint link is configured
            # (all robots in a view share the same base footprint link name)
            base_footprint_link_name = next(iter(self._base_footprint_link_names.values()))
            if base_footprint_link_name is not None:
                base_link_idx = self._link_idx[0][base_footprint_link_name]
                base_vels = link_vels[:, base_link_idx, :]  # (N, 6) — already in cache, no extra fetch
            else:
                # Warm root velocities cache and get (N, 6)
                self._get_all_root_velocities(estimate=estimate)
                base_vels = cb.to_torch(self._read_cache[f"root_{vel_str}"])  # (N, 6)

            # Build (N, L+1, 6): link vels followed by base vel (base at final index, matching _get_relative_velocities)
            all_vels = th.cat([link_vels, base_vels.unsqueeze(1)], dim=1)  # (N, L+1, 6)

            # Build block-diagonal rotation transform per robot: (N, 6, 6)
            all_quats = cb.to_torch(self.get_all_position_orientation()[1])  # (N, 4)
            ori_t_batch = TT.quat2mat(all_quats).transpose(-2, -1)  # (N, 3, 3)
            tf = th.zeros(all_vels.shape[0], 6, 6, dtype=all_vels.dtype)
            tf[:, :3, :3] = ori_t_batch
            tf[:, 3:, 3:] = ori_t_batch

            # Batched matmul: (N, 1, 6, 6) @ (N, L+1, 6, 1) → (N, L+1, 6)
            rel_vels = (tf.unsqueeze(1) @ all_vels.unsqueeze(-1)).squeeze(-1)
            self._read_cache[f"all_relative_{vel_str}"] = cb.from_torch(rel_vels)

        return self._read_cache[f"all_relative_{vel_str}"]

    def _get_relative_velocities(self, prim_path, estimate=False):
        idx = self._idx[prim_path]
        return self._get_all_relative_velocities(estimate=estimate)[idx]

    def get_linear_velocity(self, prim_path, estimate=False):
        return self._get_velocities(prim_path, estimate=estimate)[:3]

    def get_angular_velocity(self, prim_path, estimate=False):
        return self._get_velocities(prim_path, estimate=estimate)[3:]

    def _get_all_root_velocities(self, estimate=False):
        vel_str = "velocities_estimate" if estimate else "velocities"

        # Use estimated calculation if requested and we have prior history info
        if f"root_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                # Compute root velocities estimate as delta between prior timestep and current timestep
                vels = cb.zeros((self._last_state["root_transforms"].shape[0], 6))

                if "root_transforms" not in self._read_cache:
                    self._read_cache["root_transforms"] = cb.from_torch(self._view.get_root_transforms())

                vels[:, :3] = self._read_cache["root_transforms"][:, :3] - self._last_state["root_transforms"][:, :3]
                vels[:, 3:] = cb.T.quat2axisangle(
                    cb.T.quat_distance(
                        self._read_cache["root_transforms"][:, 3:], self._last_state["root_transforms"][:, 3:]
                    )
                )
                self._read_cache[f"root_{vel_str}"] = vels / og.sim.get_physics_dt()
            else:
                self._read_cache[f"root_{vel_str}"] = cb.from_torch(self._view.get_root_velocities())

        return self._read_cache[f"root_{vel_str}"]

    def get_relative_linear_velocity(self, prim_path, estimate=False):
        # base corresponds to final index
        return self._get_relative_velocities(prim_path, estimate=estimate)[-1, :3]

    def get_relative_angular_velocity(self, prim_path, estimate=False):
        # base corresponds to final index
        return self._get_relative_velocities(prim_path, estimate=estimate)[-1, 3:]

    def get_link_index(self, link_name):
        """Returns the integer body index for the named link in the articulation view's link_paths."""
        return self._link_idx[0][link_name]

    def get_all_link_relative_position_orientation(self, link_name):
        """Returns (N, 3) positions and (N, 4) quaternions for the given link across all robots."""
        cache_key = f"all_link_rel_pose_{link_name}"
        if cache_key not in self._read_cache:
            link_idx = self._link_idx[0][link_name]
            # _get_all_relative_poses returns (N, n_links, 7); slice the desired link: (N, 7)
            poses = self._get_all_relative_poses()[:, link_idx, :]
            self._read_cache[cache_key] = poses
        poses = self._read_cache[cache_key]
        return poses[:, :3], poses[:, 3:]

    def get_all_link_relative_linear_velocity(self, link_name, estimate=False):
        """Returns (N, 3) link linear velocities for all robots."""
        cache_key = f"all_link_rel_lin_vel{'_est' if estimate else ''}_{link_name}"
        if cache_key not in self._read_cache:
            link_idx = self._link_idx[0][link_name]
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, link_idx, :3]
        return self._read_cache[cache_key]

    def get_all_link_relative_angular_velocity(self, link_name, estimate=False):
        """Returns (N, 3) link angular velocities for all robots."""
        cache_key = f"all_link_rel_ang_vel{'_est' if estimate else ''}_{link_name}"
        if cache_key not in self._read_cache:
            link_idx = self._link_idx[0][link_name]
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, link_idx, 3:]
        return self._read_cache[cache_key]

    def get_all_relative_linear_velocity(self, estimate=False):
        """Returns (N, 3) base linear velocities for all robots in this view."""
        cache_key = f"all_relative_lin_vel{'_est' if estimate else ''}"
        if cache_key not in self._read_cache:
            # Base is appended at the final index in _get_all_relative_velocities
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, -1, :3]
        return self._read_cache[cache_key]

    def get_all_relative_angular_velocity(self, estimate=False):
        """Returns (N, 3) base angular velocities for all robots in this view."""
        cache_key = f"all_relative_ang_vel{'_est' if estimate else ''}"
        if cache_key not in self._read_cache:
            # Base is appended at the final index in _get_all_relative_velocities
            self._read_cache[cache_key] = self._get_all_relative_velocities(estimate=estimate)[:, -1, 3:]
        return self._read_cache[cache_key]

    def get_all_joint_positions(self):
        """Returns (N, n_dof) joint positions for all robots in this view."""
        if "dof_positions" not in self._read_cache:
            self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())
        return self._read_cache["dof_positions"]

    def get_joint_positions(self, prim_path):
        return self.get_all_joint_positions()[self._idx[prim_path]]

    def get_all_joint_velocities(self, estimate=False):
        """Returns (N, n_dof) joint velocities for all robots in this view."""
        vel_str = "velocities_estimate" if estimate else "velocities"
        if f"dof_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                if "dof_positions" not in self._read_cache:
                    self._read_cache["dof_positions"] = cb.from_torch(self._view.get_dof_positions())
                self._read_cache[f"dof_{vel_str}"] = (
                    self._read_cache["dof_positions"] - self._last_state["dof_positions"]
                ) / og.sim.get_physics_dt()
            else:
                self._read_cache[f"dof_{vel_str}"] = cb.from_torch(self._view.get_dof_velocities())
        return self._read_cache[f"dof_{vel_str}"]

    def get_joint_velocities(self, prim_path, estimate=False):
        return self.get_all_joint_velocities(estimate=estimate)[self._idx[prim_path]]

    def get_all_joint_efforts(self):
        """Returns (N, n_dof) joint efforts for all robots in this view."""
        if "dof_projected_joint_forces" not in self._read_cache:
            self._read_cache["dof_projected_joint_forces"] = cb.from_torch(self._view.get_dof_projected_joint_forces())
        return self._read_cache["dof_projected_joint_forces"]

    def get_joint_efforts(self, prim_path):
        return self.get_all_joint_efforts()[self._idx[prim_path]]

    def get_all_generalized_mass_matrices(self):
        """Returns (N, n_dof, n_dof) mass matrices for all robots in this view."""
        if "mass_matrices" not in self._read_cache:
            self._read_cache["mass_matrices"] = cb.from_torch(self._view.get_generalized_mass_matrices())
        return self._read_cache["mass_matrices"]

    def get_generalized_mass_matrices(self, prim_path):
        return self.get_all_generalized_mass_matrices()[self._idx[prim_path]]

    def get_all_gravity_compensation_forces(self):
        """Returns (N, n_dof) gravity compensation forces for all robots in this view."""
        if "generalized_gravity_forces" not in self._read_cache:
            self._read_cache["generalized_gravity_forces"] = cb.from_torch(self._view.get_gravity_compensation_forces())
        return self._read_cache["generalized_gravity_forces"]

    def get_gravity_compensation_forces(self, prim_path):
        return self.get_all_gravity_compensation_forces()[self._idx[prim_path]]

    def get_all_coriolis_and_centrifugal_compensation_forces(self):
        """Returns (N, n_dof) Coriolis/centrifugal forces for all robots in this view."""
        if "coriolis_and_centrifugal_forces" not in self._read_cache:
            self._read_cache["coriolis_and_centrifugal_forces"] = cb.from_torch(
                self._view.get_coriolis_and_centrifugal_compensation_forces()
            )
        return self._read_cache["coriolis_and_centrifugal_forces"]

    def get_coriolis_and_centrifugal_compensation_forces(self, prim_path):
        return self.get_all_coriolis_and_centrifugal_compensation_forces()[self._idx[prim_path]]

    def get_link_transform(self, prim_path, link_name):
        idx = self._idx[prim_path]
        pos, quat = self.get_all_link_transform(link_name)
        return pos[idx], quat[idx]

    def get_all_link_transform(self, link_name):
        if "link_transforms" not in self._read_cache:
            self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())

        # We assume that in a view, link_idx for the same link_name is the same across all members
        link_idx = self._link_idx[0][link_name]
        pose = self._read_cache["link_transforms"][:, link_idx]
        return pose[:, :3], pose[:, 3:]

    def _get_relative_poses(self, prim_path):
        idx = self._idx[prim_path]
        return self._get_all_relative_poses()[idx]

    def get_link_relative_position_orientation(self, prim_path, link_name):
        idx = self._idx[prim_path]
        pos, quat = self.get_all_link_relative_position_orientation(link_name)
        return pos[idx], quat[idx]

    def _get_all_link_velocities(self, link_name, estimate=False):
        """Returns (N, 6) velocities (linear + angular) for the given link across all robots."""
        vel_str = "velocities_estimate" if estimate else "velocities"

        # Build and cache the full (N, L, 6) tensor for all robots and all links
        if f"link_{vel_str}" not in self._read_cache:
            if estimate and self._last_state is not None:
                # Compute link velocities estimate as delta between prior timestep and current timestep
                N, L, _ = self._last_state["link_transforms"].shape
                vels = cb.zeros((N, L, 6))

                if "link_transforms" not in self._read_cache:
                    self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())

                vels[:, :, :3] = (
                    self._read_cache["link_transforms"][:, :, :3] - self._last_state["link_transforms"][:, :, :3]
                )
                vels[:, :, 3:] = cb.view(
                    cb.T.quat2axisangle(
                        cb.T.quat_distance(
                            cb.view(self._read_cache["link_transforms"][:, :, 3:], (-1, 4)),
                            cb.view(self._last_state["link_transforms"][:, :, 3:], (-1, 4)),
                        )
                    ),
                    (N, L, 3),
                )
                self._read_cache[f"link_{vel_str}"] = vels / og.sim.get_physics_dt()

            # Otherwise, directly grab velocities
            else:
                self._read_cache[f"link_{vel_str}"] = cb.from_torch(self._view.get_link_velocities())

        link_idx = self._link_idx[0][link_name]
        return self._read_cache[f"link_{vel_str}"][:, link_idx, :]  # (N, 6)

    def _get_link_velocities(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        return self._get_all_link_velocities(link_name, estimate=estimate)[idx]

    def get_link_linear_velocity(self, prim_path, link_name, estimate=False):
        return self._get_link_velocities(prim_path, link_name, estimate=estimate)[:3]

    def get_all_link_linear_velocity(self, link_name, estimate=False):
        return self._get_all_link_velocities(link_name, estimate=estimate)[:, :3]

    def get_link_relative_linear_velocity(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        return self._get_relative_velocities(prim_path, estimate=estimate)[link_idx, :3]

    def get_all_link_angular_velocity(self, link_name, estimate=False):
        return self._get_all_link_velocities(link_name, estimate=estimate)[:, 3:]

    def get_link_relative_angular_velocity(self, prim_path, link_name, estimate=False):
        idx = self._idx[prim_path]
        link_idx = self._link_idx[idx][link_name]
        return self._get_relative_velocities(prim_path, estimate=estimate)[link_idx, 3:]

    def get_all_jacobian(self):
        if "jacobians" not in self._read_cache:
            self._read_cache["jacobians"] = cb.from_torch(self._view.get_jacobians())
        return self._read_cache["jacobians"]

    def get_jacobian(self, prim_path):
        idx = self._idx[prim_path]
        return self.get_all_jacobian()[idx]

    def _get_all_relative_poses(self):
        """Returns (N, n_links, 7) relative poses (pos + quat) for all robots in this view, batched."""
        if "relative_poses" not in self._read_cache:
            # All link world transforms: (N, n_links, 7)
            if "link_transforms" not in self._read_cache:
                self._read_cache["link_transforms"] = cb.from_torch(self._view.get_link_transforms())
            all_link_tfs = cb.to_torch(self._read_cache["link_transforms"])  # (N, n_links, 7)

            # All base poses
            all_pos, all_quat = self.get_all_position_orientation()  # (N, 3), (N, 4)
            all_pos = cb.to_torch(all_pos)
            all_quat = cb.to_torch(all_quat)

            N, n_links = all_link_tfs.shape[:2]

            # Build link homogeneous transform matrices: (N, n_links, 4, 4)
            tfs = th.zeros(N, n_links, 4, 4, dtype=th.float32)
            tfs[:, :, 3, 3] = 1.0
            tfs[:, :, :3, 3] = all_link_tfs[:, :, :3]
            # quat2mat doesn't handle rank-3 input; flatten the N*n_links batch dimension
            tfs[:, :, :3, :3] = TT.quat2mat(all_link_tfs[:, :, 3:].reshape(-1, 4)).reshape(N, n_links, 3, 3)

            # Build batched base pose inverses: (N, 4, 4)
            # For a rigid transform [R, t; 0, 1], the inverse is [R^T, -R^T t; 0, 1]
            base_rot_T = TT.quat2mat(all_quat).transpose(-2, -1)  # (N, 3, 3)
            base_tf_inv = th.zeros(N, 4, 4, dtype=th.float32)
            base_tf_inv[:, 3, 3] = 1.0
            base_tf_inv[:, :3, :3] = base_rot_T
            base_tf_inv[:, :3, 3] = -(base_rot_T @ all_pos.unsqueeze(-1)).squeeze(-1)

            # Batched matmul: (N, 1, 4, 4) @ (N, n_links, 4, 4) → (N, n_links, 4, 4)
            rel_tfs = base_tf_inv.unsqueeze(1) @ tfs

            # Convert back to (N, n_links, 7) pos + quat
            rel_poses = th.zeros(N, n_links, 7, dtype=th.float32)
            rel_poses[:, :, :3] = rel_tfs[:, :, :3, 3]
            rel_poses[:, :, 3:] = TT.mat2quat(rel_tfs[:, :, :3, :3].reshape(-1, 3, 3)).reshape(N, n_links, 4)

            self._read_cache["relative_poses"] = cb.from_torch(rel_poses)
        return self._read_cache["relative_poses"]

    def get_all_relative_jacobians(self):
        """Returns (N, n_links, 6, n_dof_total) relative jacobians for all robots in this view."""
        if "relative_jacobians" not in self._read_cache:
            # All raw jacobians: (N, n_links, 6, n_dof_total)
            all_jacobians = cb.to_torch(self.get_all_jacobian())
            # Base orientation quaternions for all robots: (N, 4)
            all_quats = cb.to_torch(self.get_all_position_orientation()[1])
            N = all_quats.shape[0]

            # Rotation matrices transposed per robot: (N, 3, 3)
            ori_t_batch = TT.quat2mat(all_quats).transpose(-2, -1)

            # Build block-diagonal transform tf = [[ori_t, 0], [0, ori_t]]: (N, 6, 6)
            tf = th.zeros(N, 6, 6, dtype=all_jacobians.dtype)
            tf[:, :3, :3] = ori_t_batch
            tf[:, 3:, 3:] = ori_t_batch

            # Batched matmul: (N, 1, 6, 6) @ (N, n_links, 6, n_dof_total) → (N, n_links, 6, n_dof_total)
            # Run in pytorch since it's order of magnitude faster than numpy!
            self._read_cache["relative_jacobians"] = cb.from_torch(tf.unsqueeze(1) @ all_jacobians)
        return self._read_cache["relative_jacobians"]

    def get_relative_jacobian(self, prim_path):
        idx = self._idx[prim_path]
        return self.get_all_relative_jacobians()[idx]


def get_robot_kinematic_tree_pattern(articulation_root_path: str) -> str:
    """
    Returns a glob pattern that matches all robots of the same type and fixedness as the
    given articulation root path.

    The pattern generalizes over scene index and robot instance name, preserving the
    robot-type component and any path suffix (e.g. base link name for floating-base robots).

    Examples:
        "/World/scene_0/controllable__fetch__robot0"
            -> "/World/scene_*/controllable__fetch__*"
        "/World/scene_0/controllable__fetch__robot0/base_link"
            -> "/World/scene_*/controllable__fetch__*/base_link"
    """
    scene_id, robot_name = articulation_root_path.split("/")[2:4]
    assert scene_id.startswith("scene_"), f"Prim path 2nd component {articulation_root_path} does not start with scene_"
    components = robot_name.split("__")
    assert len(components) == 3, (
        f"Robot prim path's 3rd component {robot_name} does not match "
        "expected format of prefix__robottype__robotname."
    )
    assert (
        components[0] == "controllable"
    ), f"Prim path {articulation_root_path} 3rd component does not start with 'controllable__'"
    return articulation_root_path.replace(f"/{scene_id}/", "/scene_*/").replace(
        f"/{robot_name}", f"/{components[0]}__{components[1]}__*"
    )


class ControllableObjectViewAPI:
    """
    An interface that creates BatchControlViewAPIImpl instances for each robot type in the scene.

    This is done to avoid the overhead of reading from many views for each robot in each physics step,
    providing major speed improvements in vector env use cases.

    This class is a singleton, and should be used to access the BatchControlViewAPIImpl instances.

    The pattern used to group the robots is based on the robot prim paths, which is assumed to be in the format
    /World/scene_*/controllable__robottype__robotname.

    The patterns used by the subviews are generated by replacing the robot name with a wildcard, so that all robots
    of the same type are grouped together. If there are fixed base robots, they will be grouped separately from
    non-fixed base robots even within the same robot type, by virtue of their different articulation root paths.

    **Return types:** All kinematic / dynamic getters delegate to :class:`BatchControlViewAPIImpl` and return
    **compute-backend arrays** (``cb.arr_type`` from :mod:`omnigibson.utils.backend_utils`), after converting
    Isaac articulation-view **torch** tensors with ``cb.from_torch``. Batched joint commands from controllers
    should be **compute-backend arrays** (``cb.arr_type``).
    """

    # Dictionary mapping from pattern to BatchControlViewAPIImpl
    _VIEWS_BY_PATTERN = {}

    @classmethod
    def post_physics_step(cls):
        for view in cls._VIEWS_BY_PATTERN.values():
            view.post_physics_step()

    @classmethod
    def clear(cls):
        for view in cls._VIEWS_BY_PATTERN.values():
            view.clear()

    @classmethod
    def clear_object(cls, prim_path):
        if get_robot_kinematic_tree_pattern(prim_path) in cls._VIEWS_BY_PATTERN:
            cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].clear()

    @classmethod
    def flush_control(cls):
        for view in cls._VIEWS_BY_PATTERN.values():
            view.flush_control()

    @classmethod
    def initialize_view(cls):
        cls._VIEWS_BY_PATTERN = {}

        # First, get all of the controllable objects in the scene (avoiding circular import)
        from omnigibson.robots import Robot

        controllable_objects = [obj for scene in og.sim.scenes for obj in scene.objects if isinstance(obj, Robot)]

        # Get their corresponding prim paths
        expected_prim_paths = {obj.articulation_root_path for obj in controllable_objects}

        # Group the prim paths by robot type
        patterns = {get_robot_kinematic_tree_pattern(prim_path) for prim_path in expected_prim_paths}

        # Create the view for each robot type / fixedness combo
        for pattern in patterns:
            if pattern not in cls._VIEWS_BY_PATTERN:
                cls._VIEWS_BY_PATTERN[pattern] = BatchControlViewAPIImpl(pattern)

        # Initialize the views
        for view in cls._VIEWS_BY_PATTERN.values():
            view.initialize_view()

        # Assert that the views' prim paths are disjoint
        all_prim_paths = []
        for view in cls._VIEWS_BY_PATTERN.values():
            all_prim_paths.extend(view._idx.keys())
        counts = collections.Counter(all_prim_paths)

        missing = set(expected_prim_paths) - set(all_prim_paths)
        assert len(missing) == 0, f"Prim paths {missing} are missing from the views!"

        more_than_once = {prim_path: count for prim_path, count in counts.items() if count > 1}
        assert len(more_than_once) == 0, f"Prim paths {more_than_once} are present in multiple views!"

    @classmethod
    def set_joint_position_targets(cls, prim_path, positions, indices):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].set_joint_position_targets(
            prim_path, positions, indices
        )

    @classmethod
    def set_joint_velocity_targets(cls, prim_path, velocities, indices):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].set_joint_velocity_targets(
            prim_path, velocities, indices
        )

    @classmethod
    def set_joint_efforts(cls, prim_path, efforts, indices):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].set_joint_efforts(
            prim_path, efforts, indices
        )

    @classmethod
    def get_member_view_indices(cls, routing_path, prim_paths):
        """Return view row indices for prim_paths (all in same view as routing_path)."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].get_member_view_indices(prim_paths)

    @classmethod
    def set_all_joint_position_targets(cls, routing_path, enabled_rows, controls, dof_idx):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].set_all_joint_position_targets(
            enabled_rows, controls, dof_idx
        )

    @classmethod
    def set_all_joint_velocity_targets(cls, routing_path, enabled_rows, velocities, dof_idx):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].set_all_joint_velocity_targets(
            enabled_rows, velocities, dof_idx
        )

    @classmethod
    def set_all_joint_efforts(cls, routing_path, enabled_rows, efforts, dof_idx):
        cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(routing_path)].set_all_joint_efforts(
            enabled_rows, efforts, dof_idx
        )

    @classmethod
    def get_position_orientation(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_position_orientation(prim_path)

    @classmethod
    def get_root_position_orientation(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_root_transform(prim_path)

    @classmethod
    def get_linear_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_linear_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_angular_velocity(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_angular_velocity(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_all_joint_positions(cls, prim_path):
        """Returns (N, n_dof) joint positions for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_joint_positions()

    @classmethod
    def get_joint_positions(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_joint_positions(prim_path)

    @classmethod
    def get_all_joint_velocities(cls, prim_path, estimate=False):
        """Returns (N, n_dof) joint velocities for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_joint_velocities(
            estimate=estimate
        )

    @classmethod
    def get_joint_velocities(cls, prim_path, estimate=False):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_joint_velocities(
            prim_path, estimate=estimate
        )

    @classmethod
    def get_all_joint_efforts(cls, prim_path):
        """Returns (N, n_dof) joint efforts for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_joint_efforts()

    @classmethod
    def get_joint_efforts(cls, prim_path):
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_joint_efforts(prim_path)

    @classmethod
    def get_all_generalized_mass_matrices(cls, prim_path):
        """Returns (N, n_dof, n_dof) mass matrices for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_generalized_mass_matrices()

    @classmethod
    def get_all_gravity_compensation_forces(cls, prim_path):
        """Returns (N, n_dof) gravity compensation forces for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_gravity_compensation_forces()

    @classmethod
    def get_all_coriolis_and_centrifugal_compensation_forces(cls, prim_path):
        """Returns (N, n_dof) Coriolis/centrifugal forces for all robots of the same type as @prim_path."""
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_all_coriolis_and_centrifugal_compensation_forces()

    @classmethod
    def get_link_relative_position_orientation(cls, prim_path, link_name):
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_link_relative_position_orientation(prim_path, link_name)

    @classmethod
    def get_link_index(cls, prim_path, link_name):
        """Returns the integer body index for the named link in the articulation view's link_paths."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_link_index(link_name)

    @classmethod
    def get_all_relative_jacobians(cls, prim_path):
        """Returns (N, n_links, 6, n_dof_total) relative jacobians for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_relative_jacobians()

    @classmethod
    def get_all_link_relative_position_orientation(cls, prim_path, link_name):
        """Returns (N, 3) positions and (N, 4) quaternions for the given link across all robots."""
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_all_link_relative_position_orientation(link_name)

    @classmethod
    def get_all_link_relative_linear_velocity(cls, prim_path, link_name, estimate=False):
        """Returns (N, 3) link linear velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_link_relative_linear_velocity(
            link_name, estimate=estimate
        )

    @classmethod
    def get_all_link_relative_angular_velocity(cls, prim_path, link_name, estimate=False):
        """Returns (N, 3) link angular velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[
            get_robot_kinematic_tree_pattern(prim_path)
        ].get_all_link_relative_angular_velocity(link_name, estimate=estimate)

    @classmethod
    def get_all_relative_linear_velocity(cls, prim_path, estimate=False):
        """Returns (N, 3) base linear velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_relative_linear_velocity(
            estimate=estimate
        )

    @classmethod
    def get_all_relative_angular_velocity(cls, prim_path, estimate=False):
        """Returns (N, 3) base angular velocities for all robots of the same type."""
        return cls._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(prim_path)].get_all_relative_angular_velocity(
            estimate=estimate
        )


def clear():
    """
    Clear state tied to singleton classes
    """
    CollisionAPI.clear()
    RigidBodyViewAPI.clear()
    RigidContactAPI.clear()
    ArticulatedObjectViewAPI.clear()
    ControllableObjectViewAPI.clear()


def create_mesh_prim_with_default_xform(primitive_type, prim_path, u_patches=None, v_patches=None, stage=None):
    """
    Creates a mesh prim of the specified @primitive_type at the specified @prim_path

    Args:
        primitive_type (str): Primitive mesh type, should be one of PRIMITIVE_MESH_TYPES to be valid
        prim_path (str): Destination prim path to store the mesh prim
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.
        stage (None or Usd.Stage): If specified, stage on which the primitive mesh should be generated. If None, will
            use og.sim.stage
    """
    with og.sim.editing_usd(stage=stage):
        MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING = {
            "Sphere": lazy.omni.kit.primitive.mesh.evaluators.sphere.SphereEvaluator,
            "Disk": lazy.omni.kit.primitive.mesh.evaluators.disk.DiskEvaluator,
            "Plane": lazy.omni.kit.primitive.mesh.evaluators.plane.PlaneEvaluator,
            "Cylinder": lazy.omni.kit.primitive.mesh.evaluators.cylinder.CylinderEvaluator,
            "Torus": lazy.omni.kit.primitive.mesh.evaluators.torus.TorusEvaluator,
            "Cone": lazy.omni.kit.primitive.mesh.evaluators.cone.ConeEvaluator,
            "Cube": lazy.omni.kit.primitive.mesh.evaluators.cube.CubeEvaluator,
        }

        assert primitive_type in PRIMITIVE_MESH_TYPES, "Invalid primitive mesh type: {primitive_type}"
        evaluator = MESH_PRIM_TYPE_TO_EVALUATOR_MAPPING[primitive_type]
        u_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_U_SCALE)
        v_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_V_SCALE)
        hs_backup = lazy.carb.settings.get_settings().get(evaluator.SETTING_OBJECT_HALF_SCALE)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, 1)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, 1)
        stage = og.sim.stage if stage is None else stage

        # Default half_scale (i.e. half-extent, half_height, radius) is 1.
        # TODO (eric): change it to 0.5 once the mesh generator API accepts floating-number HALF_SCALE
        #  (currently it only accepts integer-number and floors 0.5 into 0).
        lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, 1)
        kwargs = dict(prim_type=primitive_type, prim_path=prim_path, stage=stage)
        if u_patches is not None and v_patches is not None:
            kwargs["u_patches"] = u_patches
            kwargs["v_patches"] = v_patches

        # Import now to avoid too-eager load of Omni classes due to inheritance
        from omnigibson.utils.deprecated_utils import CreateMeshPrimWithDefaultXformCommand

        CreateMeshPrimWithDefaultXformCommand(**kwargs).do()

        lazy.carb.settings.get_settings().set(evaluator.SETTING_U_SCALE, u_backup)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_V_SCALE, v_backup)
        lazy.carb.settings.get_settings().set(evaluator.SETTING_OBJECT_HALF_SCALE, hs_backup)


def mesh_prim_mesh_to_trimesh_mesh(mesh_prim, include_normals=True, include_texcoord=True):
    """
    Generates trimesh mesh from @mesh_prim if mesh_type is "Mesh"

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh
        include_normals (bool): Whether to include the normals in the resulting trimesh or not
        include_texcoord (bool): Whether to include the corresponding 2D-texture coordinates in the resulting
            trimesh or not

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    assert mesh_type == "Mesh", f"Expected mesh prim to have type Mesh, got {mesh_type}"
    face_vertex_counts = vtarray_to_torch(mesh_prim.GetAttribute("faceVertexCounts").Get(), dtype=th.int)
    vertices = vtarray_to_torch(mesh_prim.GetAttribute("points").Get())
    face_indices = vtarray_to_torch(mesh_prim.GetAttribute("faceVertexIndices").Get(), dtype=th.int)

    faces = []
    i = 0
    for count in face_vertex_counts:
        for j in range(count - 2):
            faces.append([face_indices[i], face_indices[i + j + 1], face_indices[i + j + 2]])
        i += count

    kwargs = dict(vertices=vertices, faces=faces)

    if include_normals:
        kwargs["vertex_normals"] = vtarray_to_torch(mesh_prim.GetAttribute("normals").Get())

    if include_texcoord:
        raw_texture = mesh_prim.GetAttribute("primvars:st").Get()
        if raw_texture is not None:
            kwargs["visual"] = trimesh.visual.TextureVisuals(uv=vtarray_to_torch(raw_texture))

    return trimesh.Trimesh(**kwargs)


def mesh_prim_shape_to_trimesh_mesh(mesh_prim):
    """
    Generates trimesh mesh from @mesh_prim if mesh_type is "Sphere", "Cube", "Cone" or "Cylinder"

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
    if mesh_type == "Sphere":
        radius = mesh_prim.GetAttribute("radius").Get()
        trimesh_mesh = trimesh.creation.icosphere(subdivision=3, radius=radius)
    elif mesh_type == "Cube":
        extent = mesh_prim.GetAttribute("size").Get()
        trimesh_mesh = trimesh.creation.box([extent] * 3)
    elif mesh_type == "Cone":
        radius = mesh_prim.GetAttribute("radius").Get()
        height = mesh_prim.GetAttribute("height").Get()
        trimesh_mesh = trimesh.creation.cone(radius=radius, height=height)
        # Trimesh cones are centered at the base. We'll move them down by half the height.
        transform = trimesh.transformations.translation_matrix([0, 0, -height / 2])
        trimesh_mesh.apply_transform(transform)
    elif mesh_type == "Cylinder":
        radius = mesh_prim.GetAttribute("radius").Get()
        height = mesh_prim.GetAttribute("height").Get()
        trimesh_mesh = trimesh.creation.cylinder(radius=radius, height=height)
    else:
        raise ValueError(f"Expected mesh prim to have type Sphere, Cube, Cone or Cylinder, got {mesh_type}")

    return trimesh_mesh


def mesh_prim_to_trimesh_mesh(mesh_prim, include_normals=True, include_texcoord=True, world_frame=False):
    """
    Generates trimesh mesh from @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to convert into trimesh mesh
        include_normals (bool): Whether to include the normals in the resulting trimesh or not
        include_texcoord (bool): Whether to include the corresponding 2D-texture coordinates in the resulting
            trimesh or not
        world_frame (bool): Whether to convert the mesh to the world frame or not

    Returns:
        trimesh.Trimesh: Generated trimesh mesh
    """
    mesh_type = mesh_prim.GetTypeName()
    if mesh_type == "Mesh":
        trimesh_mesh = mesh_prim_mesh_to_trimesh_mesh(mesh_prim, include_normals, include_texcoord)
    else:
        trimesh_mesh = mesh_prim_shape_to_trimesh_mesh(mesh_prim)

    if world_frame:
        trimesh_mesh.apply_transform(get_world_pose_with_scale(mesh_prim.GetPath().pathString))

    return trimesh_mesh


def sample_mesh_keypoints(mesh_prim, n_keypoints, n_keyfaces, seed=None):
    """
    Samples keypoints and keyfaces for mesh @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to be sampled from
        n_keypoints (int): number of (unique) keypoints to randomly sample from @mesh_prim
        n_keyfaces (int): number of (unique) keyfaces to randomly sample from @mesh_prim
        seed (None or int): If set, sets the random seed for deterministic results

    Returns:
        2-tuple:
            - n-array: (n,) 1D int array representing the randomly sampled point idxs from @mesh_prim.
                Note that since this is without replacement, the total length of the array may be less than
                @n_keypoints
            - None or n-array: 1D int array representing the randomly sampled face idxs from @mesh_prim.
                Note that since this is without replacement, the total length of the array may be less than
                @n_keyfaces
    """
    # Set seed if deterministic
    if seed is not None:
        th.manual_seed(seed)

    # Generate trimesh mesh from which to aggregate points
    tm = mesh_prim_mesh_to_trimesh_mesh(mesh_prim=mesh_prim, include_normals=False, include_texcoord=False)
    n_unique_vertices, n_unique_faces = len(tm.vertices), len(tm.faces)
    faces_flat = th.tensor(tm.faces.flatten(), dtype=th.int32)

    # Sample vertices
    unique_vertices = th.unique(faces_flat)
    assert len(unique_vertices) == n_unique_vertices
    keypoint_idx = (
        th.randperm(len(unique_vertices))[:n_keypoints] if n_unique_vertices > n_keypoints else unique_vertices
    )

    # Sample faces
    keyface_idx = th.randperm(n_unique_faces)[:n_keyfaces] if n_unique_faces > n_keyfaces else th.arange(n_unique_faces)

    return keypoint_idx, keyface_idx


def get_mesh_volume_and_com(mesh_prim, world_frame=False):
    """
    Computes the volume and center of mass for @mesh_prim

    Args:
        mesh_prim (Usd.Prim): Mesh prim to compute volume and center of mass for
        world_frame (bool): Whether to return the volume and CoM in the world frame

    Returns:
        Tuple[float, th.tensor]: Tuple containing the (volume, center_of_mass) in the mesh frame or the world frame
    """

    trimesh_mesh = mesh_prim_to_trimesh_mesh(
        mesh_prim, include_normals=False, include_texcoord=False, world_frame=world_frame
    )
    if trimesh_mesh.is_volume:
        volume = trimesh_mesh.volume
        com = th.tensor(trimesh_mesh.center_mass)
    else:
        # If the mesh is not a volume, we compute its convex hull and use that instead
        try:
            trimesh_mesh_convex = trimesh_mesh.convex_hull
            volume = trimesh_mesh_convex.volume
            com = th.tensor(trimesh_mesh_convex.center_mass)
        except:
            # if convex hull computation fails, it usually means the mesh is degenerated: use trivial values.
            volume = 0.0
            com = th.zeros(3)

    return volume, com.to(dtype=th.float32)


def check_extent_radius_ratio(geom_prim, com):
    """
    Checks if the min extent in world frame and the extent radius ratio in local frame of @geom_prim is within the
    acceptable range for PhysX GPU acceleration (not too thin, and not too oblong)

    Ref: https://github.com/NVIDIA-Omniverse/PhysX/blob/561a0df858d7e48879cdf7eeb54cfe208f660f18/physx/source/geomutils/src/convex/GuConvexMeshData.h#L183-L190

    Args:
        geom_prim (GeomPrim): Geom prim to check
        com (th.tensor): Center of mass of the mesh. Obtained from get_mesh_volume_and_com

    Returns:
        bool: True if the min extent (world) and the extent radius ratio (local frame) is acceptable, False otherwise
    """
    mesh_type = geom_prim.prim.GetPrimTypeInfo().GetTypeName()
    # Non-mesh prims are always considered to be within the acceptable range
    if mesh_type != "Mesh":
        return True

    extent = geom_prim.extent
    min_extent = extent.min()
    # If the mesh is too flat in the world frame, omniverse cannot create convex mesh for it
    if min_extent < 1e-5:
        return False

    max_radius = extent.max() / 2.0
    min_radius = th.min(th.norm(geom_prim.points - com, dim=-1), dim=0).values
    ratio = max_radius / min_radius

    # PhysX requires ratio to be < 100.0. We use 95.0 to be safe.
    return ratio < 95.0


def create_primitive_mesh(prim_path, primitive_type, extents=1.0, u_patches=None, v_patches=None, stage=None):
    """
    Helper function that generates a UsdGeom.Mesh prim at specified @prim_path of type @primitive_type.

    NOTE: Generated mesh prim will, by default, have extents equaling [1, 1, 1]

    Args:
        prim_path (str): Where the loaded mesh should exist on the stage
        primitive_type (str): Type of primitive mesh to create. Should be one of:
            {"Cone", "Cube", "Cylinder", "Disk", "Plane", "Sphere", "Torus"}
        extents (float or 3-array): Specifies the extents of the generated mesh. Default is 1.0, i.e.:
            generated mesh will be in be contained in a [1,1,1] sized bounding box
        u_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            u-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
        v_patches (int or None): If specified, should be an integer that represents how many segments to create in the
            v-direction. E.g. 10 means 10 segments (and therefore 11 vertices) will be created.
            Both u_patches and v_patches need to be specified for them to be effective.
        stage (None or Usd.Stage): If specified, stage on which the primitive mesh should be generated. If None, will
            use og.sim.stage

    Returns:
        UsdGeom.Mesh: Generated primitive mesh as a prim on the active stage
    """
    assert_valid_key(key=primitive_type, valid_keys=PRIMITIVE_MESH_TYPES, name="primitive mesh type")
    create_mesh_prim_with_default_xform(
        primitive_type, prim_path, u_patches=u_patches, v_patches=v_patches, stage=stage
    )

    with og.sim.editing_usd(stage=stage):
        mesh = lazy.pxr.UsdGeom.Mesh.Define(og.sim.stage if stage is None else stage, prim_path)

        # Modify the points and normals attributes so that total extents is the desired
        # This means multiplying omni's default by extents * 50.0, as the native mesh generated has extents [-0.01, 0.01]
        # -- i.e.: 2cm-wide mesh
        extents = th.ones(3) * extents if isinstance(extents, float) else th.tensor(extents)
        for attr in (mesh.GetPointsAttr(), mesh.GetNormalsAttr()):
            vals = th.tensor(attr.Get()).double()
            attr.Set(lazy.pxr.Vt.Vec3fArray([lazy.pxr.Gf.Vec3f(*(val * extents * 50.0).tolist()) for val in vals]))
        mesh.GetExtentAttr().Set(
            lazy.pxr.Vt.Vec3fArray(
                [lazy.pxr.Gf.Vec3f(*(-extents / 2.0).tolist()), lazy.pxr.Gf.Vec3f(*(extents / 2.0).tolist())]
            )
        )

    return triangularize_mesh(mesh)


def create_usd_stage(usd_path):
    stage = lazy.pxr.Usd.Stage.CreateNew(usd_path)
    lazy.pxr.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    lazy.pxr.UsdGeom.SetStageUpAxis(stage, "Z")
    return stage


def triangularize_mesh(mesh):
    """
    Triangulates the mesh @mesh, modification in-place
    """
    with og.sim.editing_usd():
        tm = mesh_prim_to_trimesh_mesh(mesh.GetPrim())

        face_vertex_counts = np.array([len(face) for face in tm.faces], dtype=int)
        mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        mesh.GetFaceVertexIndicesAttr().Set(tm.faces.flatten())
        mesh.GetNormalsAttr().Set(lazy.pxr.Vt.Vec3fArray.FromNumpy(tm.vertex_normals[tm.faces.flatten()]))

        # Modify the UV mapping if it exists
        if isinstance(tm.visual, trimesh.visual.TextureVisuals):
            mesh.GetPrim().GetAttribute("primvars:st").Set(
                lazy.pxr.Vt.Vec2fArray.FromNumpy(tm.visual.uv[tm.faces.flatten()])
            )

        return mesh


def add_asset_to_stage(asset_path, prim_path):
    """
    Adds asset file (either USD or OBJ) at @asset_path at the location @prim_path

    Args:
        asset_path (str): Absolute or relative path to the asset file to load
        prim_path (str): Where loaded asset should exist on the stage

    Returns:
        Usd.Prim: Loaded prim as a USD prim
    """
    with og.sim.editing_usd():
        # Make sure this is actually a supported asset type
        asset_type = asset_path.split(".")[-1]
        assert asset_type in {"usd", "usda", "obj", "usdz"}, "Cannot load a non-USD or non-OBJ file as a USD prim!"

        # Make sure the path exists
        assert os.path.exists(
            asset_path
        ), f"Cannot load {asset_type.upper()} file {asset_path} because it does not exist!"

        # Add reference to stage and grab prim
        lazy.isaacsim.core.utils.stage.add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)

        # Make sure prim was loaded correctly
        assert prim, f"Failed to load {asset_type.upper()} object from path: {asset_path}"

        return prim


def get_world_prim():
    """
    Returns:
        Usd.Prim: Active world prim in the current stage
    """
    return lazy.isaacsim.core.utils.prims.get_prim_at_path("/World")


def scene_relative_prim_path_to_absolute(scene, relative_prim_path):
    """
    Converts a scene-relative prim path to an absolute prim path.

    Args:
        scene (Scene or None): Scene object that the prim is in. None if it's global.
        relative_prim_path (str): Relative prim path in the scene

    Returns:
        str: Absolute prim path in the stage
    """
    # Special case for OmniGraph prims
    if relative_prim_path.startswith("/OmniGraph"):
        return relative_prim_path

    # Special case for global floor plane collision prim — already an absolute path
    if (
        og.sim.floor_plane is not None
        and relative_prim_path == og.sim.floor_plane.relative_prim_path + "/collisionPlane"
    ):
        return relative_prim_path

    # Make sure the relative path is actually relative
    assert not relative_prim_path.startswith("/World"), f"Expected relative prim path, got {relative_prim_path}"

    # When the scene is set to None, this prim is not in a scene but is global e.g. like the
    # viewer camera or one of the scene prims.
    if scene is None:
        return "/World" + relative_prim_path

    return scene.prim_path + relative_prim_path


def absolute_prim_path_to_scene_relative(scene, absolute_prim_path):
    """
    Converts an absolute prim path to a scene-relative prim path.

    Args:
        scene (Scene): Scene object that the prim is in. None if it's global.
        absolute_prim_path (str): Absolute prim path in the stage

    Returns:
        str: Relative prim path in the scene
    """
    # Special case for OmniGraph prims
    if absolute_prim_path.startswith("/OmniGraph"):
        return absolute_prim_path

    # Special case for global floor plane collision prim — not scene-scoped, return unchanged
    if (
        og.sim.floor_plane is not None
        and absolute_prim_path == og.sim.floor_plane.relative_prim_path + "/collisionPlane"
    ):
        return absolute_prim_path

    assert absolute_prim_path.startswith("/World"), f"Expected absolute prim path, got {absolute_prim_path}"

    # When the scene is set to None, this prim is not in a scene but is global e.g. like the
    # viewer camera or one of the scene prims.
    if scene is None:
        assert not absolute_prim_path.startswith(
            "/World/scene_"
        ), f"Expected global prim path, got {absolute_prim_path}"
        return absolute_prim_path[len("/World") :]

    return absolute_prim_path[len(scene.prim_path) :]


def delete_or_deactivate_prim(prim_path):
    """
    Attept to delete or deactivate the prim defined at @prim_path.

    Note that the removal of prims usually has an impact on the PhysX state and needs to be followed
    by a call to og.sim.update_handles() to update tensor views etc. - we do not do here to avoid
    performance overhead when lots of prims are removed at once in clear() etc. and instead we
    delegate this to the caller.

    Args:
        prim_path (str): Path defining which prim should be deleted or deactivated

    Returns:
        bool: Whether the operation was successful or not
    """

    # TODO: Replace the weird delete-or-deactivate mechanism here with a concrete deletion
    # using the Sdf layer deletion API.
    with og.sim.editing_usd():
        if not lazy.isaacsim.core.utils.prims.is_prim_path_valid(prim_path):
            return False
        if lazy.isaacsim.core.utils.prims.is_prim_no_delete(prim_path):
            return False
        if lazy.isaacsim.core.utils.prims.get_prim_type_name(prim_path=prim_path) == "PhysicsScene":
            return False
        if prim_path == "/World":
            return False
        if prim_path == "/":
            return False
        # Don't remove any /Render prims as that can cause crashes
        if prim_path.startswith("/Render"):
            return False

        # If the prim is not ancestral, we can delete it.
        if not lazy.isaacsim.core.utils.prims.is_prim_ancestral(prim_path):
            lazy.omni.usd.commands.DeletePrimsCommand([prim_path], destructive=True).do()

        # Otherwise, we can only deactivate it, which essentially serves the same purpose.
        # All objects that are originally in the scene are ancestral because we add the pre-build scene to the stage.
        else:
            # Clear all default attributes before deactivating the prim to ensure clean reactivation.
            # Note: Prim deactivation preserves attribute values, so we must explicitly clear defaults
            # to prevent stale custom values from persisting when the prim is reactivated later.
            prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)
            for attr in prim.GetAttributes():
                assert attr.ClearDefault()
            lazy.omni.usd.commands.DeletePrimsCommand([prim_path], destructive=False).do()

    return True


def activate_prim_and_children(prim_path):
    """
    Recursively activates the prim at @prim_path and all of its children.

    Args:
        prim_path (str): Path to the prim to activate
    """
    current_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path)
    current_prim.SetActive(True)
    # Use GetAllChildren to also find those that are inactive
    for child in current_prim.GetAllChildren():
        activate_prim_and_children(child.GetPath().pathString)


def get_sdf_value_type_name(val):
    """
    Determines the appropriate Sdf value type based on the input value.
    Args:
        val: The input value to determine the type for.
    Returns:
        lazy.pxr.Sdf.ValueTypeName: The corresponding Sdf value type.
    Raises:
        ValueError: If the input value type is not supported.
    """
    SDF_TYPE_MAPPING = {
        lazy.pxr.Gf.Vec3f: lazy.pxr.Sdf.ValueTypeNames.Float3,
        lazy.pxr.Gf.Vec2f: lazy.pxr.Sdf.ValueTypeNames.Float2,
        lazy.pxr.Sdf.AssetPath: lazy.pxr.Sdf.ValueTypeNames.Asset,
        bool: lazy.pxr.Sdf.ValueTypeNames.Bool,
        int: lazy.pxr.Sdf.ValueTypeNames.Int,
        float: lazy.pxr.Sdf.ValueTypeNames.Float,
        str: lazy.pxr.Sdf.ValueTypeNames.String,
    }
    for type_, usd_type in SDF_TYPE_MAPPING.items():
        if isinstance(val, type_):
            return usd_type
    raise ValueError(f"Unsupported input type: {type(val)}")


def replace_collision_blocks(old_usd_path: str, new_usd_path: str, output_usd_path: str):
    """
    Replace all collisions blocks in new_usd_path with those from old_usd_path.
    """

    def extract_collision_blocks(text):
        """
        Extract all top-level 'def [Mesh] "collisions"' blocks using brace matching.
        Returns a list of (start_idx, end_idx, block_text)
        """
        blocks = []
        lines = text.splitlines(keepends=True)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("def") and '"collisions"' in line:
                start = i
                brace_count = 0
                # Find the opening brace
                while "{" not in lines[i]:
                    i += 1
                    if i >= len(lines):
                        break
                if i >= len(lines):
                    break
                brace_count += lines[i].count("{") - lines[i].count("}")
                i += 1
                # Count braces to find block end
                while brace_count > 0 and i < len(lines):
                    brace_count += lines[i].count("{") - lines[i].count("}")
                    i += 1
                end = i
                block_text = "".join(lines[start:end])
                blocks.append((start, end, block_text))
            else:
                i += 1
        return blocks

    # Load USDA files
    with open(old_usd_path, "r") as f:
        source_usda = f.read()
    with open(new_usd_path, "r") as f:
        target_usda = f.read()

    # Extract collision blocks
    source_collision_blocks = extract_collision_blocks(source_usda)
    target_blocks = extract_collision_blocks(target_usda)

    # Replace in target
    if len(target_blocks) != len(source_collision_blocks):
        print(f"Warning: Replacing {min(len(target_blocks), len(source_collision_blocks))} blocks due to mismatch.")

    new_lines = []
    last_idx = 0
    target_lines = target_usda.splitlines(keepends=True)
    for (start, end, _), (_, _, new_block) in zip(target_blocks, source_collision_blocks):
        new_lines.extend(target_lines[last_idx:start])
        new_lines.append(new_block)
        last_idx = end
    new_lines.extend(target_lines[last_idx:])

    new_usda_text = "".join(new_lines)

    # Save result
    with open(output_usd_path, "w") as f:
        f.write(new_usda_text)

    print(f"Finished replacing all {len(source_collision_blocks)} collision blocks.")


@torch_compile
def _compute_relative_poses_torch(
    idx: int,
    n_links: int,
    all_tfs: th.Tensor,
    base_pose: Tuple[th.Tensor, th.Tensor],
):
    tfs = th.zeros((n_links, 4, 4), dtype=th.float32)
    # base vel is the final -1 index
    link_tfs = all_tfs[idx, :]
    tfs[:, 3, 3] = 1.0
    tfs[:, :3, 3] = link_tfs[:, :3]
    tfs[:, :3, :3] = TT.quat2mat(link_tfs[:, 3:])
    base_tf_inv = th.zeros((1, 4, 4), dtype=th.float32)
    base_tf_inv[0, :, :] = TT.pose_inv(TT.pose2mat(base_pose))

    # (1, 4, 4) @ (n_links, 4, 4) -> (n_links, 4, 4)
    rel_tfs = base_tf_inv @ tfs

    # Re-convert to quat form
    rel_poses = th.zeros((n_links, 7), dtype=th.float32)
    rel_poses[:, :3] = rel_tfs[:, :3, 3]
    rel_poses[:, 3:] = TT.mat2quat(rel_tfs[:, :3, :3])

    return rel_poses


@jit(nopython=True, cache=True)
def _compute_relative_poses_numpy(idx, n_links, all_tfs, base_pose):
    tfs = np.zeros((n_links, 4, 4), dtype=np.float32)
    # base vel is the final -1 index
    link_tfs = all_tfs[idx, :]
    tfs[:, 3, 3] = 1.0
    tfs[:, :3, 3] = link_tfs[:, :3]
    tfs[:, :3, :3] = NT._quat2mat(link_tfs[:, 3:])
    # base_tf_inv = np.zeros((1, 4, 4), dtype=np.float32)
    # base_tf_inv[0, :, :] = NT._pose_inv(NT.pose2mat(base_pose))
    base_tf_inv = NT._pose_inv(NT.pose2mat(base_pose))

    # (1, 4, 4) @ (n_links, 4, 4) -> (n_links, 4, 4)
    rel_tfs = np.zeros((n_links, 4, 4), dtype=np.float32)
    for i in prange(n_links):
        rel_tfs[i, :, :] = base_tf_inv @ tfs[i, :, :]
    # rel_tfs = base_tf_inv @ tfs

    # Re-convert to quat form
    rel_poses = np.zeros((n_links, 7), dtype=np.float32)
    rel_poses[:, :3] = rel_tfs[:, :3, 3]
    rel_poses[:, 3:] = NT.mat2quat_batch(rel_tfs[:, :3, :3].copy())

    return rel_poses


# Set these as part of the backend values
add_compute_function(
    name="compute_relative_poses", np_function=_compute_relative_poses_numpy, th_function=_compute_relative_poses_torch
)


def count_joints(prim):
    """
    Search from @prim to count movable joints, fixed joints, and attachment points.

    Args:
        prim (Usd.Prim): Root prim to search from.

    Returns:
        tuple: (n_joints, n_fixed_joints, has_attachment) where
            n_joints (int): number of non-fixed physics joints,
            n_fixed_joints (int): number of fixed physics joints,
            has_attachment (bool): whether any prim name contains "attachment".
    """
    n_joints = 0
    n_fixed_joints = 0
    has_attachment = False
    children = list(prim.GetChildren())
    while children:
        child_prim = children.pop()
        children.extend(child_prim.GetChildren())
        prim_type = child_prim.GetPrimTypeInfo().GetTypeName().lower()
        if "joint" in prim_type:
            if "fixed" in prim_type:
                n_fixed_joints += 1
            else:
                n_joints += 1
        if "attachment" in child_prim.GetName().lower():
            has_attachment = True
    return n_joints, n_fixed_joints, has_attachment


def compute_kinematic_only(fixed_base, scale, n_joints, n_fixed_joints, kinematic_only_config, has_attachment):
    """
    Determine whether an object should be kinematic-only based on its properties.

    Args:
        fixed_base (bool): Whether the object has a fixed base.
        scale (th.Tensor): 3-element scale tensor.
        n_joints (int): Number of non-fixed joints.
        n_fixed_joints (int): Number of fixed joints.
        kinematic_only_config: Value of the kinematic_only load config key (True, False, or None).
        has_attachment (bool): Whether the object has attachment points.

    Returns:
        bool: True if the object should be kinematic only.
    """
    if not fixed_base:
        return False
    if kinematic_only_config is False:
        return False
    return (
        n_joints == 0
        and (th.all(th.isclose(scale, th.ones_like(scale), atol=1e-3)).item() or n_fixed_joints == 0)
        and not has_attachment
    )
