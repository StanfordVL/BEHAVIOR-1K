"""
Standalone implementation of PhysicsBackend using the open-source `newton` physics package (MuJoCo-Warp
solver) directly.
"""

import math
from collections import deque

import torch as th

import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.physics_backends.base import PhysicsBackend
from omnigibson.physics_backends.standalone_sim_context import StandaloneSimulationContext
import omnigibson.utils.transform_utils as T
from omnigibson.utils.constants import PrimType
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import _live_matrix_from_pose

log = create_module_logger(module_name=__name__)


def _wp_to_torch(array):
    import warp as wp

    if array.size == 0:
        return th.from_numpy(array.numpy())
    return wp.to_torch(array)


def _physx_schema_resolver():
    """
    newton's PhysX schema resolver, extended to read joint friction from `physxJoint:jointFriction`, which is
    where JointPrim.friction authors it. Stock SchemaResolverPhysx has no friction entry, so every rebuild
    would otherwise reset joint friction to 0 and discard what the friction setter pushed into the live model.
    """
    import newton
    from newton._src.usd.schema_resolver import SchemaAttribute

    base = newton.usd.SchemaResolverPhysx
    joint_type = newton.usd.PrimType.JOINT

    class _SchemaResolverPhysx(base):
        mapping = {
            **base.mapping,
            joint_type: {**base.mapping[joint_type], "friction": SchemaAttribute("physxJoint:jointFriction", 0.0)},
        }

    return _SchemaResolverPhysx()


_SPHERE_SHAPE_OVERLAP_KERNEL = None


def _sphere_shape_overlap_kernel():
    """
    Warp kernel testing a sphere against each candidate shape's exact geometry: overlap iff the shape's signed
    distance at the sphere center is <= the radius. Follows newton's own particle-vs-shape contact kernel conventions.
    Built lazily so importing this module does not import newton.
    """
    global _SPHERE_SHAPE_OVERLAP_KERNEL
    if _SPHERE_SHAPE_OVERLAP_KERNEL is not None:
        return _SPHERE_SHAPE_OVERLAP_KERNEL

    import warp as wp
    from newton import GeoType
    from newton._src.geometry.kernels import sdf_box, sdf_capsule, sdf_cone, sdf_cylinder, sdf_sphere
    from newton._src.core.types import Axis

    @wp.kernel
    def kernel(
        center: wp.vec3,
        radius: float,
        cand_shape: wp.array(dtype=wp.int32),
        body_q: wp.array(dtype=wp.transform),
        shape_body: wp.array(dtype=wp.int32),
        shape_transform: wp.array(dtype=wp.transform),
        shape_type: wp.array(dtype=wp.int32),
        shape_scale: wp.array(dtype=wp.vec3),
        shape_source_ptr: wp.array(dtype=wp.uint64),
        out: wp.array(dtype=wp.int32),
    ):
        k = wp.tid()
        s = cand_shape[k]
        X_ws = shape_transform[s]
        b = shape_body[s]
        if b >= 0:
            X_ws = wp.transform_multiply(body_q[b], X_ws)
        x = wp.transform_point(wp.transform_inverse(X_ws), center)
        geo_type = shape_type[s]
        scale = shape_scale[s]
        d = float(-1.0e6)  # unsupported types keep the conservative AABB answer (overlap)
        if geo_type == GeoType.SPHERE:
            d = sdf_sphere(x, scale[0])
        elif geo_type == GeoType.BOX:
            d = sdf_box(x, scale[0], scale[1], scale[2])
        elif geo_type == GeoType.CAPSULE:
            d = sdf_capsule(x, scale[0], scale[1], int(Axis.Z))
        elif geo_type == GeoType.CYLINDER:
            d = sdf_cylinder(x, scale[0], scale[1], int(Axis.Z))
        elif geo_type == GeoType.CONE:
            d = sdf_cone(x, scale[0], scale[1], int(Axis.Z))
        elif (geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH) and shape_source_ptr[s] != wp.uint64(0):
            mesh = shape_source_ptr[s]
            # Unbounded search so a point deep inside a solid still finds its closest face (and its sign)
            q = wp.mesh_query_point_sign_normal(mesh, wp.cw_div(x, scale), 1.0e6)
            if q.result:
                closest = wp.cw_mul(wp.mesh_eval_position(mesh, q.face, q.u, q.v), scale)
                d = wp.length(x - closest) * q.sign
        if d <= radius:
            out[k] = 1

    _SPHERE_SHAPE_OVERLAP_KERNEL = kernel
    return kernel


def _assign(wp_array, idx, value):
    """
    Write `value` into `wp_array[idx]`, converting `value` to a plain, flat torch tensor on the same
    device/dtype as `wp_array` first.
    """
    target = _wp_to_torch(wp_array)
    target[idx] = th.as_tensor(value).reshape(-1).to(device=target.device, dtype=target.dtype)


def _xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]


def _quat_mul_batch(q1, q0):
    """Elementwise batched quaternion multiply q1*q0, both (N, 4) xyzw -- q1 applied after q0."""
    x0, y0, z0, w0 = q0.unbind(-1)
    x1, y1, z1, w1 = q1.unbind(-1)
    return th.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=-1,
    )


def _quat_rotate_batch(q, v):
    """Elementwise batched rotate v (N, 3) by q (N, 4) xyzw."""
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * th.linalg.cross(qxyz, v, dim=-1)
    return v + qw * t + th.linalg.cross(qxyz, t, dim=-1)


def _quat_to_abs_rotmat_batch(q):
    """|R| per-element (N, 3, 3), used to compute a rotated AABB's tight world half-extent."""
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = th.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)
    return r.abs()


class _SceneQueryHit:
    """Minimal stand-in for PhysX's scene-query hit object -- only `.rigid_body` is read by every
    current call site (see `object_states/particle_modifier.py`, `contact_particles.py`, etc.); `.collision`
    is set equal to it since this backend doesn't track a separate sub-body collision-mesh path."""

    __slots__ = ("rigid_body", "collision", "position", "normal", "distance")

    def __init__(self, rigid_body, position=None, normal=None, distance=None):
        self.rigid_body = rigid_body
        self.collision = rigid_body
        self.position = position
        self.normal = normal
        self.distance = distance


class _NewtonJointBreakEvent:
    """Synthetic stand-in for PhysX's SimulationEvent."""

    __slots__ = ("joint_path",)

    def __init__(self, joint_path):
        self.joint_path = joint_path


class _AppliedActions:
    """Minimal stand-in for Isaac's `ArticulationActions`, only the fields JointPrim reads."""

    def __init__(self, joint_positions, joint_velocities):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities


def _native_to_handle_dof_cols(handle, dof_start, dof_count):
    """
    Column order mapping a native-DOF-ordered articulation matrix onto a handle's own DOF order.
    """
    handle_cols = [int(g) - dof_start for g in handle._global_qd_idx.tolist()]
    return sorted(set(range(dof_count)) - set(handle_cols)) + handle_cols


def _breadth_first_joint_order(model, global_joint_idx):
    """
    Reorders an articulation's joints into the order PhysX enumerates its DOFs.
    """
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    by_parent = {}
    for j in global_joint_idx:
        by_parent.setdefault(int(joint_parent[j]), []).append(j)
    child_bodies = {int(joint_child[j]) for j in global_joint_idx}
    queue = deque(sorted(set(by_parent) - child_bodies))
    ordered, seen_joints, seen_bodies = [], set(), set(queue)
    while queue:
        body = queue.popleft()
        for j in by_parent.get(body, ()):
            if j in seen_joints:
                continue
            seen_joints.add(j)
            ordered.append(j)
            child = int(joint_child[j])
            if child not in seen_bodies:
                seen_bodies.add(child)
                queue.append(child)
    ordered.extend(j for j in global_joint_idx if j not in seen_joints)
    assert sorted(ordered) == sorted(global_joint_idx), "joint reordering dropped or duplicated a joint"
    return ordered


class NewtonArticulationHandle:
    """
    Per-robot articulation view backed by the shared `newton.Model`/`State`/`Control` owned by
    `NewtonBackend`, implementing the same public surface `PhysXArticulationView` exposes. All
    joint/DOF indexing exposed here is LOCAL to this one robot (0-based), matching the semantics of
    Isaac's per-articulation `ArticulationView`.
    """

    def __init__(self, backend, prim_path):
        self._backend = backend
        self.prim_path = prim_path
        self._root_link_prim_path = prim_path
        self._valid = False
        # Local joint metadata, populated in initialize()
        self._global_joint_idx = []  # global model joint index, per local joint
        self._joint_dof_counts = []  # local dof count, per local joint
        self._joint_dof_offsets = []  # local (cumulative) dof offset, per local joint
        self._joint_names = []
        self._local_dof_to_path = []  # per-local-dof owning joint's prim path (repeated per DOF)
        self._global_qd_idx = None  # torch.long tensor: global qd-space index per local dof
        self._global_q_idx = (
            None  # torch.long tensor: global q-space index per local dof (for revolute/prismatic, 1:1 with qd)
        )
        # q_start of this articulation's own root FREE joint (7-wide: pos(3)+quat(4)), or None for a
        # fixed-base object with no such joint.
        self._root_free_joint_q_start = None

    def initialize(self, physics_sim_view):
        model = self._backend._model
        path_body_map = self._backend._path_body_map
        path_joint_map = self._backend._path_joint_map

        if model is None:
            self._valid = False
            return

        root_body_idx = path_body_map.get(self._root_link_prim_path)
        if root_body_idx is None:
            self._valid = False
            return

        entity_prefix = self._root_link_prim_path.rsplit("/", 1)[0] + "/"
        body_idx_set = {root_body_idx} | {idx for path, idx in path_body_map.items() if path.startswith(entity_prefix)}

        joint_child = model.joint_child.numpy()
        joint_type = model.joint_type.numpy()
        import newton

        JOINT_FREE = int(newton.JointType.FREE)
        root_free_joint = next(
            (j for j in range(model.joint_count) if joint_child[j] == root_body_idx and joint_type[j] == JOINT_FREE),
            None,
        )
        has_free_joint = root_free_joint is not None
        self.prim_path = entity_prefix[:-1] if not has_free_joint else self._root_link_prim_path
        self._root_free_joint_q_start = int(model.joint_q_start.numpy()[root_free_joint]) if has_free_joint else None

        joint_child = model.joint_child.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        n_joints_total = model.joint_count
        path_by_joint_idx = {idx: path for path, idx in path_joint_map.items()}

        def qd_count_at(j):
            return (model.joint_dof_count if j == n_joints_total - 1 else joint_qd_start[j + 1]) - joint_qd_start[j]

        global_joint_idx = sorted(
            i for i in range(n_joints_total) if joint_child[i] in body_idx_set and i in path_by_joint_idx
        )
        # Re-emit joints in PhysX's DOF order, so this handle (and therefore EntityPrim.joints,
        # every *_control_idx, and the serialized state layout) is backend-independent.
        global_joint_idx = _breadth_first_joint_order(model, global_joint_idx)

        self._global_joint_idx = global_joint_idx
        self._joint_dof_counts = []
        self._joint_dof_offsets = []
        self._joint_names = []
        self._local_dof_to_path = []
        global_qd_idx = []
        global_q_idx = []
        offset = 0
        for j in global_joint_idx:
            n_dof = int(qd_count_at(j))
            path = path_by_joint_idx[j]
            name = path.split("/")[-1]
            self._joint_dof_counts.append(n_dof)
            self._joint_dof_offsets.append(offset)
            self._joint_names.append(name)
            for k in range(n_dof):
                global_qd_idx.append(int(joint_qd_start[j]) + k)
                global_q_idx.append(int(joint_q_start[j]) + k)
                self._local_dof_to_path.append(path)
            offset += n_dof

        self._global_qd_idx = th.as_tensor(global_qd_idx, dtype=th.long, device=self._backend.sim.device)
        self._global_q_idx = th.as_tensor(global_q_idx, dtype=th.long, device=self._backend.sim.device)
        self._valid = True

    @property
    def is_valid(self):
        # Looser than is_physics_handle_valid(): whether this handle refers to a prim that structurally
        # exists, not whether it's been bound to the current Newton model yet.
        return True

    def is_physics_handle_valid(self):
        return self._valid

    @property
    def has_dof_metadata(self):
        return self._valid

    @property
    def num_dof(self):
        return int(self._global_qd_idx.numel())

    @property
    def joint_count(self):
        return len(self._global_joint_idx)

    @property
    def joint_dof_counts(self):
        return self._joint_dof_counts

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def joint_dof_offsets(self):
        return self._joint_dof_offsets

    def dof_path(self, dof_index, articulation_index=0):
        return self._local_dof_to_path[dof_index]

    def dof_index_of_path(self, prim_path, articulation_index=0):
        return self._local_dof_to_path.index(prim_path)

    def _idx(self, joint_indices=None):
        return self._global_qd_idx if joint_indices is None else self._global_qd_idx[joint_indices]

    def _idx_q(self, joint_indices=None):
        return self._global_q_idx if joint_indices is None else self._global_q_idx[joint_indices]

    # ---- Joint state get/set ----

    def get_joint_positions(self, joint_indices=None, clone=True):
        return _wp_to_torch(self._backend._state_0.joint_q)[self._idx_q(joint_indices)].unsqueeze(0)

    def get_joint_velocities(self, joint_indices=None, clone=True):
        return _wp_to_torch(self._backend._state_0.joint_qd)[self._idx(joint_indices)].unsqueeze(0)

    def get_measured_joint_efforts(self, joint_indices=None, clone=True):
        idx = self._idx(joint_indices)
        effort = _wp_to_torch(self._backend._control.joint_f)[idx]
        qfrc_actuator = getattr(getattr(self._backend._state_0, "mujoco", None), "qfrc_actuator", None)
        if qfrc_actuator is not None:
            effort = effort + _wp_to_torch(qfrc_actuator)[idx]
        return effort.unsqueeze(0)

    def set_joint_positions(self, positions, joint_indices=None, indices=None):
        idx = self._idx_q(joint_indices)
        # Clamp into the joint's own limit range before writing.
        model = self._backend._model
        qd_idx = self._idx(joint_indices)
        lower = _wp_to_torch(model.joint_limit_lower)[qd_idx]
        upper = _wp_to_torch(model.joint_limit_upper)[qd_idx]
        positions = th.as_tensor(positions).reshape(-1).to(lower.device).clamp(min=lower, max=upper)
        _assign(self._backend._state_0.joint_q, idx, positions)
        _assign(self._backend._state_1.joint_q, idx, positions)
        self._backend._sync_fk()

    def set_joint_velocities(self, velocities, joint_indices=None, indices=None):
        idx = self._idx(joint_indices)
        _assign(self._backend._state_0.joint_qd, idx, velocities)
        _assign(self._backend._state_1.joint_qd, idx, velocities)
        self._backend._sync_fk()

    def set_joint_efforts(self, efforts, joint_indices=None, indices=None):
        idx = self._idx(joint_indices)
        _assign(self._backend._control.joint_f, idx, efforts)

    def get_joint_position_targets(self, joint_indices=None, clone=True):
        return _wp_to_torch(self._backend._control.joint_target_q)[self._idx(joint_indices)].unsqueeze(0)

    def get_joint_velocity_targets(self, joint_indices=None, clone=True):
        return _wp_to_torch(self._backend._control.joint_target_qd)[self._idx(joint_indices)].unsqueeze(0)

    def set_joint_position_targets(self, positions, joint_indices=None, indices=None):
        idx = self._idx(joint_indices)
        _assign(self._backend._control.joint_target_q, idx, positions)

    def set_joint_velocity_targets(self, velocities, joint_indices=None, indices=None):
        idx = self._idx(joint_indices)
        _assign(self._backend._control.joint_target_qd, idx, velocities)

    def get_applied_actions(self):
        pos = _wp_to_torch(self._backend._control.joint_target_q)[self._global_qd_idx].unsqueeze(0)
        vel = _wp_to_torch(self._backend._control.joint_target_qd)[self._global_qd_idx].unsqueeze(0)
        return _AppliedActions(pos, vel)

    def get_dof_is_rotational(self):
        import newton

        joint_type = self._backend._model.joint_type.numpy()
        revolute = int(newton.JointType.REVOLUTE)
        is_rotational = []
        for j, n_dof in zip(self._global_joint_idx, self._joint_dof_counts):
            is_rotational.extend([joint_type[j] == revolute] * n_dof)
        return is_rotational

    # ---- Gains / limits / friction ----

    # MuJoCo's implicit integrator folds kd into the step's effective mass matrix (~kd * physics_dt
    # added to the joint's own inertia).
    def get_gains(self, joint_indices=None):
        idx = self._idx(joint_indices)
        model = self._backend._model
        ke = _wp_to_torch(model.joint_target_ke)[idx].unsqueeze(0)
        kd = _wp_to_torch(model.joint_target_kd)[idx].unsqueeze(0)
        return ke, kd

    def set_gains(self, kps=None, kds=None, joint_indices=None):
        idx = self._idx(joint_indices)
        model = self._backend._model
        if kps is not None:
            _assign(model.joint_target_ke, idx, kps)
        if kds is not None:
            _assign(model.joint_target_kd, idx, kds)
        self._backend._notify_dof_properties_changed()

    def get_max_velocities(self, joint_indices=None):
        idx = self._idx(joint_indices)
        return th.full(
            (1, len(idx) if hasattr(idx, "__len__") else idx.numel()), float("inf"), device=self._backend.sim.device
        )

    def set_max_velocities(self, values, joint_indices=None):
        pass

    def get_max_efforts(self, joint_indices=None):
        idx = self._idx(joint_indices)
        return th.full(
            (1, len(idx) if hasattr(idx, "__len__") else idx.numel()), float("inf"), device=self._backend.sim.device
        )

    def set_max_efforts(self, values, joint_indices=None):
        pass

    def get_friction_coefficients(self, joint_indices=None):
        idx = self._idx(joint_indices)
        model = self._backend._model
        return _wp_to_torch(model.joint_friction)[idx].unsqueeze(0)

    def set_friction_coefficients(self, values, joint_indices=None):
        idx = self._idx(joint_indices)
        _assign(self._backend._model.joint_friction, idx, values)
        self._backend._notify_dof_properties_changed()

    def get_joint_limits(self, joint_indices=None):
        idx = self._idx(joint_indices)
        model = self._backend._model
        lower = _wp_to_torch(model.joint_limit_lower)[idx].unsqueeze(0)
        upper = _wp_to_torch(model.joint_limit_upper)[idx].unsqueeze(0)
        return th.stack([lower, upper], dim=-1)

    def set_joint_limits(self, limits, joint_indices=None):
        idx = self._idx(joint_indices)
        model = self._backend._model
        limits = th.as_tensor(limits)
        _assign(model.joint_limit_lower, idx, limits[..., 0])
        _assign(model.joint_limit_upper, idx, limits[..., 1])
        self._backend._notify_dof_properties_changed()

    # ---- World pose (root link) ----

    def set_world_poses(self, positions, orientations):
        root_body_idx = self._backend._path_body_map[self._root_link_prim_path]
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        positions = th.as_tensor(positions).reshape(-1).to(device=body_q.device, dtype=body_q.dtype)
        orientations = _wxyz_to_xyzw(th.as_tensor(orientations).reshape(-1)).to(
            device=body_q.device, dtype=body_q.dtype
        )
        body_q[root_body_idx, :3] = positions
        body_q[root_body_idx, 3:7] = orientations
        if self._root_free_joint_q_start is not None:
            q_start = self._root_free_joint_q_start
            joint_q = _wp_to_torch(self._backend._state_0.joint_q)
            joint_q[q_start : q_start + 3] = positions
            joint_q[q_start + 3 : q_start + 7] = orientations
        self._backend._sync_fk()

    # ---- Dynamics quantities without a found Newton equivalent ----

    def get_coriolis_and_centrifugal_forces(self, clone=True):
        raise NotImplementedError("Coriolis/centrifugal forces are not implemented for the Newton backend.")

    def get_generalized_gravity_forces(self, clone=True):
        raise NotImplementedError("Generalized gravity forces are not implemented for the Newton backend.")

    def get_mass_matrices(self, clone=True):
        raise NotImplementedError("Mass matrices are not implemented for the Newton backend.")

    def get_jacobians(self, clone=True):
        return self._backend._compute_jacobian(self).unsqueeze(0)


class _AlwaysValidPhysicsView:
    """
    Stand-in for Isaac's private `_physics_view` attribute. Newton has no equivalent
    notion of a tensor view going stale independently of the handle itself, so this is a no-op.
    """

    def check(self):
        return True


class NewtonRigidBodyHandle:
    """Per-link rigid-body view, backed by the shared `newton.Model`/`State`."""

    def __init__(self, backend, prim_path):
        self._backend = backend
        self.prim_path = prim_path
        self._body_idx = None
        self._valid = False
        self._physics_view = _AlwaysValidPhysicsView()
        # q_start of the FREE joint whose child is this body (7-wide: pos(3)+quat(4)), or None if this
        # body isn't a free-joint root.
        self._root_free_joint_q_start = None
        self._root_free_joint_qd_start = None

    def initialize(self, physics_sim_view):
        self._body_idx = self._backend._path_body_map.get(self.prim_path)
        self._valid = self._body_idx is not None
        self._root_free_joint_q_start = None
        self._root_free_joint_qd_start = None
        if self._valid:
            import newton

            model = self._backend._model
            joint_child = model.joint_child.numpy()
            joint_type = model.joint_type.numpy()
            JOINT_FREE = int(newton.JointType.FREE)
            for j in range(model.joint_count):
                if joint_child[j] == self._body_idx and joint_type[j] == JOINT_FREE:
                    self._root_free_joint_q_start = int(model.joint_q_start.numpy()[j])
                    self._root_free_joint_qd_start = int(model.joint_qd_start.numpy()[j])
                    break

    @property
    def is_valid(self):
        return True

    def is_physics_handle_valid(self):
        return self._valid

    def get_world_poses(self, clone=True):
        if not self._valid:
            # Not yet incorporated into the model, so fall back to the plain USD-authored pose.
            from omnigibson.utils.usd_utils import get_world_pose

            pos, quat_wxyz_from_xyzw = get_world_pose(self.prim_path)
            return pos.unsqueeze(0), _xyzw_to_wxyz(quat_wxyz_from_xyzw).unsqueeze(0)
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        pos = body_q[self._body_idx, :3].unsqueeze(0)
        quat_wxyz = _xyzw_to_wxyz(body_q[self._body_idx, 3:7]).unsqueeze(0)
        return (pos.clone(), quat_wxyz.clone()) if clone else (pos, quat_wxyz)

    def set_world_poses(self, positions, orientations):
        if not self._valid:
            return
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        positions = th.as_tensor(positions).reshape(-1).to(device=body_q.device, dtype=body_q.dtype)
        orientations = _wxyz_to_xyzw(th.as_tensor(orientations).reshape(-1)).to(
            device=body_q.device, dtype=body_q.dtype
        )
        body_q[self._body_idx, :3] = positions
        body_q[self._body_idx, 3:7] = orientations
        if self._root_free_joint_q_start is not None:
            q_start = self._root_free_joint_q_start
            joint_q = _wp_to_torch(self._backend._state_0.joint_q)
            joint_q[q_start : q_start + 3] = positions
            joint_q[q_start + 3 : q_start + 7] = orientations
            self._backend._sync_fk()
        self._backend._bvh_dirty = True

    def get_linear_velocities(self, clone=True):
        if not self._valid:
            return th.zeros(1, 3, device=self._backend.sim.device)
        vel = _wp_to_torch(self._backend._state_0.body_qd)[self._body_idx, :3].unsqueeze(0)
        return vel.clone() if clone else vel

    def _set_root_free_joint_qd(self, offset, values):
        if self._root_free_joint_qd_start is None:
            return
        qd_start = self._root_free_joint_qd_start + offset
        joint_qd = _wp_to_torch(self._backend._state_0.joint_qd)
        joint_qd[qd_start : qd_start + 3] = (
            th.as_tensor(values).reshape(-1).to(device=joint_qd.device, dtype=joint_qd.dtype)
        )

    def set_linear_velocities(self, velocities):
        if not self._valid:
            return
        _assign(self._backend._state_0.body_qd, (self._body_idx, slice(0, 3)), velocities)
        self._set_root_free_joint_qd(0, velocities)

    def get_angular_velocities(self, clone=True):
        if not self._valid:
            return th.zeros(1, 3, device=self._backend.sim.device)
        vel = _wp_to_torch(self._backend._state_0.body_qd)[self._body_idx, 3:6].unsqueeze(0)
        return vel.clone() if clone else vel

    def set_angular_velocities(self, velocities):
        if not self._valid:
            return
        _assign(self._backend._state_0.body_qd, (self._body_idx, slice(3, 6)), velocities)
        self._set_root_free_joint_qd(3, velocities)

    def get_coms(self, clone=True):
        com = _wp_to_torch(self._backend._model.body_com)[self._body_idx].reshape(1, 1, 3)
        if clone:
            com = com.clone()
        return com, th.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=com.device)

    def set_coms(self, positions):
        _assign(self._backend._model.body_com, self._body_idx, th.as_tensor(positions).reshape(-1)[:3])
        self._backend._notify_inertial_properties_changed()

    # Mass and density are authored on the body's USD MassAPI, which is read during _rebuild().
    def _mass_api(self, apply=False):
        prim = self._backend.sim.stage.GetPrimAtPath(self.prim_path)
        return lazy.pxr.UsdPhysics.MassAPI.Apply(prim) if apply else lazy.pxr.UsdPhysics.MassAPI(prim)

    def get_masses(self):
        if self._valid:
            return _wp_to_torch(self._backend._model.body_mass)[self._body_idx].unsqueeze(0).clone()
        mass = self._mass_api().GetMassAttr().Get()
        return th.tensor([0.0 if mass is None else float(mass)], device=self._backend.sim.device)

    def set_masses(self, masses):
        mass = float(th.as_tensor(masses).reshape(-1)[0])
        self._mass_api(apply=True).CreateMassAttr().Set(mass)
        if not self._valid:
            return
        if mass <= 0.0:
            # Mass now comes from density x collider volume, which only an import can compute
            self._backend._mass_properties_dirty = True
            return
        self._backend._scale_body_mass(self._body_idx, mass)

    def get_densities(self):
        density = self._mass_api().GetDensityAttr().Get()
        return th.tensor([0.0 if density is None else float(density)], device=self._backend.sim.device)

    def set_densities(self, densities):
        self._mass_api(apply=True).CreateDensityAttr().Set(float(th.as_tensor(densities).reshape(-1)[0]))
        if self._valid:
            self._backend._mass_properties_dirty = True

    def _set_gravcomp(self, value):
        prim = self._backend.sim.stage.GetPrimAtPath(self.prim_path)
        prim.CreateAttribute("mjc:gravcomp", lazy.pxr.Sdf.ValueTypeNames.Float).Set(value)
        if self._valid:
            self._backend._mass_properties_dirty = True

    def enable_gravities(self):
        self._set_gravcomp(0.0)

    def disable_gravities(self):
        self._set_gravcomp(1.0)


class _NewtonBatchArticulationView:
    """
    Aggregates per-robot `NewtonArticulationHandle`s. Exercised for joint-position/velocity/effort control, root/link
    transforms, jacobians, and generalized mass matrices.
    """

    def __init__(self, backend, handles):
        self._backend = backend
        self._handles = handles
        self._link_paths_per_robot = []
        self._body_idx_per_robot = []
        for h in handles:
            entity_prefix = (
                h.prim_path + "/" if h.prim_path not in backend._path_body_map else h.prim_path.rsplit("/", 1)[0] + "/"
            )
            items = sorted(
                (
                    (p, idx)
                    for p, idx in backend._path_body_map.items()
                    if p == h.prim_path or p.startswith(entity_prefix)
                ),
                key=lambda kv: kv[1],
            )
            self._link_paths_per_robot.append([p for p, _ in items])
            self._body_idx_per_robot.append([idx for _, idx in items])

    @property
    def prim_paths(self):
        return [h.prim_path for h in self._handles]

    @property
    def link_paths(self):
        return self._link_paths_per_robot

    def _stack_dof(self, getter):
        per_handle = [getter(h)[0] for h in self._handles]
        max_dof = max((t.numel() for t in per_handle), default=0)
        padded = [t if t.numel() == max_dof else th.cat([t, t.new_zeros(max_dof - t.numel())]) for t in per_handle]
        return th.stack(padded, dim=0)

    def get_dof_positions(self):
        return self._stack_dof(lambda h: h.get_joint_positions())

    def get_dof_velocities(self):
        return self._stack_dof(lambda h: h.get_joint_velocities())

    def get_dof_position_targets(self):
        return self._stack_dof(lambda h: h.get_joint_position_targets())

    def get_dof_velocity_targets(self):
        return self._stack_dof(lambda h: h.get_joint_velocity_targets())

    def get_dof_actuation_forces(self):
        return self._stack_dof(lambda h: h.get_measured_joint_efforts())

    def get_root_transforms(self):
        # Quaternions stay xyzw here, unlike NewtonRigidBodyHandle.get_world_poses().
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        transforms = []
        for idxs in self._body_idx_per_robot:
            root_idx = idxs[0]
            transforms.append(body_q[root_idx, :7].clone())
        return th.stack(transforms, dim=0)

    def get_root_velocities(self):
        body_qd = _wp_to_torch(self._backend._state_0.body_qd)
        return th.stack([body_qd[idxs[0]] for idxs in self._body_idx_per_robot], dim=0)

    def get_link_transforms(self):
        # returns xyzw
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        result = []
        for idxs in self._body_idx_per_robot:
            result.append(th.stack([body_q[idx, :7] for idx in idxs], dim=0))
        return th.stack(result, dim=0)

    def get_link_velocities(self):
        # body_qd is already [linear(3), angular(3)] per body, so no reordering needed here
        body_qd = _wp_to_torch(self._backend._state_0.body_qd)
        return th.stack([th.stack([body_qd[idx] for idx in idxs], dim=0) for idxs in self._body_idx_per_robot], dim=0)

    def get_generalized_mass_matrices(self):
        return th.stack([self._backend._compute_mass_matrix(h) for h in self._handles], dim=0)

    def get_gravity_compensation_forces(self):
        raise NotImplementedError("Gravity compensation forces are not implemented for the Newton backend.")

    def get_coriolis_and_centrifugal_compensation_forces(self):
        raise NotImplementedError(
            "Coriolis/centrifugal compensation forces are not implemented for the Newton backend."
        )

    def get_jacobians(self):
        return th.stack([self._backend._compute_jacobian(h) for h in self._handles], dim=0)

    def get_dof_projected_joint_forces(self):
        raise NotImplementedError("Projected joint forces are not implemented for the Newton backend.")

    def set_dof_position_targets_fast(self, data, indices):
        target = _wp_to_torch(self._backend._control.joint_target_q)
        data = th.as_tensor(data).to(device=target.device, dtype=target.dtype)
        for row in indices.tolist() if hasattr(indices, "tolist") else indices:
            target[self._handles[row]._global_qd_idx] = data[row]

    def set_dof_velocity_targets_fast(self, data, indices):
        target = _wp_to_torch(self._backend._control.joint_target_qd)
        data = th.as_tensor(data).to(device=target.device, dtype=target.dtype)
        for row in indices.tolist() if hasattr(indices, "tolist") else indices:
            target[self._handles[row]._global_qd_idx] = data[row]

    def set_dof_actuation_forces_fast(self, data, indices):
        target = _wp_to_torch(self._backend._control.joint_f)
        data = th.as_tensor(data).to(device=target.device, dtype=target.dtype)
        for row in indices.tolist() if hasattr(indices, "tolist") else indices:
            target[self._handles[row]._global_qd_idx] = data[row]


class _ParticleImportRoot:
    """
    Stands in for a scene object in _rebuild_impl()'s import loop, for a macro particle system. Imported from the
    system prim  while ignoring every other child, e.g. the particle template.
    """

    prim_type = PrimType.RIGID
    fixed_base = False
    is_particle_root = True

    def __init__(self, system_prim):
        self.prim_path = str(system_prim.GetPath())
        self.ignore_paths = [
            str(child.GetPath())
            for child in system_prim.GetChildren()
            if child.GetName() != "particles" and not child.IsA(lazy.pxr.UsdShade.Material)
        ]


def _match_body_paths(backend, pattern):
    """(prim paths, body indices) of model bodies matching @pattern, where `*` matches within one path component
    (PhysX tensor-view semantics) -- so e.g. "/World/scene_*/*/*" matches object links, not macro particles nested
    one level deeper."""
    matched = set(_match_paths(pattern, backend._path_body_map))
    matched = [(path, idx) for path, idx in backend._path_body_map.items() if path in matched]
    return [path for path, _ in matched], [idx for _, idx in matched]


def _match_paths(pattern, paths):
    """The subset of @paths matching @pattern, with `*` matching within a single path component."""
    import re

    regex = re.compile(re.escape(pattern).replace(r"\*", "[^/]*"))
    return [path for path in paths if regex.fullmatch(path)]


class _NewtonRigidBodyBatchView:
    """Stand-in for PhysX's pattern-based `omni.physics.tensors.RigidBodyView`."""

    def __init__(self, backend, body_indices, prim_paths, pattern=None):
        self._backend = backend
        # A pattern-created view re-resolves its bodies whenever the model is rebuilt or a deferred rebuild is
        # pending.
        self._pattern = pattern
        self._generation = backend._model_generation
        self._set_bodies(body_indices, prim_paths)

    def _set_bodies(self, body_indices, prim_paths):
        backend = self._backend
        self._body_indices = th.as_tensor(body_indices, dtype=th.long, device=backend.sim.device)
        self._prim_paths = list(prim_paths)
        # body_idx -> q_start of the FREE joint whose child is that body, for every body in this view
        # that's a free-floating root (e.g. a real MassAPI/RigidBodyAPI particle).
        self._root_free_joint_q_start = self._compute_root_free_joint_starts(qd=False)
        self._root_free_joint_qd_start = self._compute_root_free_joint_starts(qd=True)

    def _refresh_if_stale(self):
        if self._pattern is None:
            return
        backend = self._backend
        # Only a view that would actually see the pending particles forces the deferred rebuild -- scene-wide views
        # (RigidBodyViewAPI, RigidContactAPI) are re-created on every update_handles().
        if backend._macro_particles_dirty and _match_paths(self._pattern, backend._macro_particle_roots()):
            backend.ensure_model_current()
        if self._generation != backend._model_generation:
            self._generation = self._backend._model_generation
            paths, idxs = _match_body_paths(self._backend, self._pattern)
            self._set_bodies(idxs, paths)

    def _compute_root_free_joint_starts(self, qd):
        model = self._backend._model
        if model is None or model.joint_count == 0:
            return {}
        import newton

        joint_child = model.joint_child.numpy()
        joint_type = model.joint_type.numpy()
        joint_start = (model.joint_qd_start if qd else model.joint_q_start).numpy()
        free_joint = int(newton.JointType.FREE)
        body_set = {int(b) for b in self._body_indices.tolist()}
        return {
            int(joint_child[j]): int(joint_start[j])
            for j in range(model.joint_count)
            if joint_type[j] == free_joint and int(joint_child[j]) in body_set
        }

    @property
    def prim_paths(self):
        self._refresh_if_stale()
        return self._prim_paths

    @property
    def count(self):
        self._refresh_if_stale()
        return len(self._prim_paths)

    def get_transforms(self):
        self._refresh_if_stale()
        # [x, y, z, qx, qy, qz, qw] -- PhysX convention (w-last), which also matches Newton's native body_q layout, so
        # no conversion is needed.
        if self.count == 0:
            return th.zeros((0, 7), device=self._backend.sim.device)
        return _wp_to_torch(self._backend._state_0.body_q)[self._body_indices].clone()

    def set_transforms(self, data, indices=None):
        if self.count == 0:
            return
        idx = (
            self._body_indices
            if indices is None
            else self._body_indices[th.as_tensor(indices, device=self._body_indices.device)]
        )
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        data = th.as_tensor(data).to(device=body_q.device, dtype=body_q.dtype)
        body_q[idx] = data
        if self._root_free_joint_q_start:
            joint_q = _wp_to_torch(self._backend._state_0.joint_q)
            for row, body_idx in enumerate(idx.tolist()):
                q_start = self._root_free_joint_q_start.get(body_idx)
                if q_start is not None:
                    joint_q[q_start : q_start + 7] = data[row]
            self._backend._sync_fk()
        self._backend._bvh_dirty = True

    def get_velocities(self):
        if self.count == 0:
            return th.zeros((0, 6), device=self._backend.sim.device)
        return _wp_to_torch(self._backend._state_0.body_qd)[self._body_indices].clone()

    def set_velocities(self, data, indices=None):
        if self.count == 0:
            return
        idx = (
            self._body_indices
            if indices is None
            else self._body_indices[th.as_tensor(indices, device=self._body_indices.device)]
        )
        body_qd = _wp_to_torch(self._backend._state_0.body_qd)
        data = th.as_tensor(data).to(device=body_qd.device, dtype=body_qd.dtype)
        body_qd[idx] = data
        if self._root_free_joint_qd_start:
            joint_qd = _wp_to_torch(self._backend._state_0.joint_qd)
            for row, body_idx in enumerate(idx.tolist()):
                qd_start = self._root_free_joint_qd_start.get(body_idx)
                if qd_start is not None:
                    joint_qd[qd_start : qd_start + 6] = data[row]


class _NewtonRigidContactView:
    """Stand-in for PhysX's `omni.physics.tensors.RigidContactView`, backing RigidContactAPI` and
    `ManipulationRobot`'s assisted-grasping finger-contact-position lookup. World-frame forces/positions come
    from `SolverMuJoCo.update_contacts()` and are converted into PhysX shapes.
    """

    def __init__(self, backend, sensor_paths, sensor_body_idx, filter_paths, filter_body_idx, max_contact_data_count):
        self._backend = backend
        self._sensor_paths = list(sensor_paths)
        self._filter_paths = list(filter_paths)
        # -1 marks both "untracked filter path" (no dedicated Newton body) and, on the contact side,
        # "world/static shape" (Contacts.shape_body == -1, e.g. the ground plane's own collision
        # shape).
        self._body_to_row = {int(b): i for i, b in enumerate(sensor_body_idx) if int(b) >= 0}
        self._body_to_col = {int(b): i for i, b in enumerate(filter_body_idx) if int(b) >= 0}
        self._max_contact_data_count = max_contact_data_count

        n_slots = (backend._model.body_count if backend._model is not None else 0) + 1
        row_lookup = th.full((n_slots,), -1, dtype=th.long, device=backend.sim.device)
        col_lookup = th.full((n_slots,), -1, dtype=th.long, device=backend.sim.device)
        for b, i in self._body_to_row.items():
            row_lookup[b + 1] = i
        for b, i in self._body_to_col.items():
            col_lookup[b + 1] = i
        self._row_lookup = row_lookup
        self._col_lookup = col_lookup

    @property
    def sensor_paths(self):
        return self._sensor_paths

    @property
    def filter_paths(self):
        return self._filter_paths

    @property
    def sensor_count(self):
        return len(self._sensor_paths)

    @property
    def filter_count(self):
        return len(self._filter_paths)

    @property
    def max_contact_data_count(self):
        return self._max_contact_data_count

    def get_contact_force_matrix(self, dt):
        # dt is accepted to match PhysX's signature but unused here.
        matrix = th.zeros((self.sensor_count, self.filter_count, 3), device=self._backend.sim.device)
        if self.sensor_count == 0 or self.filter_count == 0:
            return matrix
        data = self._backend._contact_matrix_data()
        if data is None:
            return matrix
        shape0, shape1, force = data
        if len(shape0) == 0:
            return matrix
        body0, body1 = self._contact_bodies(shape0, shape1)
        # Vectorized equivalent of the per-contact dict-lookup loop.
        r0, c1 = self._row_lookup[body0 + 1], self._col_lookup[body1 + 1]
        r1, c0 = self._row_lookup[body1 + 1], self._col_lookup[body0 + 1]
        fwd = (r0 >= 0) & (c1 >= 0)
        rev = (r1 >= 0) & (c0 >= 0) & ~((r1 == r0) & (c0 == c1))
        if fwd.any():
            matrix.index_put_((r0[fwd], c1[fwd]), force[fwd], accumulate=True)
        if rev.any():
            matrix.index_put_((r1[rev], c0[rev]), -force[rev], accumulate=True)
        return matrix

    def get_net_contact_forces(self, dt):
        net = th.zeros((self.sensor_count, 3), device=self._backend.sim.device)
        if self.sensor_count == 0:
            return net
        data = self._backend._contact_matrix_data()
        if data is None:
            return net
        shape0, shape1, force = data
        if len(shape0) == 0:
            return net
        body0, body1 = self._contact_bodies(shape0, shape1)
        r0, r1 = self._row_lookup[body0 + 1], self._row_lookup[body1 + 1]
        fwd, rev = r0 >= 0, r1 >= 0
        if fwd.any():
            net.index_put_((r0[fwd],), force[fwd], accumulate=True)
        if rev.any():
            net.index_put_((r1[rev],), -force[rev], accumulate=True)
        return net

    def get_contact_data(self, dt):
        device = self._backend.sim.device
        contact_counts = th.zeros((self.sensor_count, self.filter_count), dtype=th.int64, device=device)
        start_indices = th.zeros((self.sensor_count, self.filter_count), dtype=th.int64, device=device)
        empty = (
            th.zeros((0, 1), device=device),
            th.zeros((0, 3), device=device),
            th.zeros((0, 3), device=device),
            th.zeros((0, 1), device=device),
            contact_counts,
            start_indices,
        )
        if self.sensor_count == 0 or self.filter_count == 0:
            return empty
        data = self._backend._contact_matrix_data()
        if data is None:
            return empty
        shape0, shape1, force = data
        if len(shape0) == 0:
            return empty

        body0, body1 = self._contact_bodies(shape0, shape1)
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        point0_local = _wp_to_torch(self._backend._contact_report_contacts.rigid_contact_point0)[: len(shape0)]
        point1_local = _wp_to_torch(self._backend._contact_report_contacts.rigid_contact_point1)[: len(shape0)]
        normal = _wp_to_torch(self._backend._contact_report_contacts.rigid_contact_normal)[: len(shape0)]
        world_point0 = body_q[body0, :3] + _quat_rotate_batch(body_q[body0, 3:7], point0_local)
        world_point1 = body_q[body1, :3] + _quat_rotate_batch(body_q[body1, 3:7], point1_local)
        force_mag = force.norm(dim=-1)

        groups = {}
        for i in range(len(shape0)):
            b0, b1 = int(body0[i]), int(body1[i])
            r0, c1 = self._body_to_row.get(b0), self._body_to_col.get(b1)
            if r0 is not None and c1 is not None:
                groups.setdefault((r0, c1), []).append((world_point0[i], normal[i], force_mag[i]))
            r1, c0 = self._body_to_row.get(b1), self._body_to_col.get(b0)
            if r1 is not None and c0 is not None and (r1, c0) != (r0, c1):
                groups.setdefault((r1, c0), []).append((world_point1[i], -normal[i], force_mag[i]))

        if not groups:
            return empty
        points_list, normals_list, forces_list = [], [], []
        for (r, c), entries in groups.items():
            start_indices[r, c] = len(points_list)
            contact_counts[r, c] = len(entries)
            for pt, n, f in entries:
                points_list.append(pt)
                normals_list.append(n)
                forces_list.append(f)
        points = th.stack(points_list)
        normals = th.stack(normals_list)
        forces = th.stack(forces_list).reshape(-1, 1)
        # separation (penetration/gap distance) isn't populated as it is currently unused.
        separations = th.zeros((len(points_list), 1), device=device)
        return forces, points, normals, separations, contact_counts, start_indices

    def _contact_bodies(self, shape0, shape1):
        shape_body = _wp_to_torch(self._backend._model.shape_body)
        return shape_body[shape0], shape_body[shape1]


class _NewtonPhysicsSimView:
    """
    Stand-in for Isaac's `physics_sim_view` batch-view factory, exposing only `create_articulation_view(pattern)`.
    """

    def __init__(self, backend):
        self._backend = backend

    def create_articulation_view(self, pattern):
        import re

        # vector's ArticulatedObjectViewAPI calls this with a list of exact prim paths rather than a
        # glob pattern string  -- match by direct membership in that case.
        if isinstance(pattern, (list, tuple, set)):
            path_set = set(pattern)

            def is_match(prim_path):
                return prim_path in path_set
        else:
            regex = pattern.replace("*", ".*")

            def is_match(prim_path):
                return re.fullmatch(regex, prim_path) is not None

        # Match against each handle's own (possibly self-corrected, see NewtonArticulationHandle.
        matched = [
            h
            for h in self._backend._articulation_handles.values()
            if h.is_physics_handle_valid() and is_match(h.prim_path)
        ]
        return _NewtonBatchArticulationView(self._backend, matched)

    def create_rigid_contact_view(self, pattern, filter_patterns, max_contact_data_count):
        import newton

        matching = set(_match_paths(pattern, self._backend._path_body_map))
        body_flags = _wp_to_torch(self._backend._model.body_flags) if self._backend._model is not None else None

        sensor_paths, sensor_body_idx = [], []
        for path, idx in self._backend._path_body_map.items():
            if path not in matching:
                continue
            if body_flags is not None and not bool(int(body_flags[idx]) & int(newton.BodyFlags.DYNAMIC)):
                continue
            sensor_paths.append(path)
            sensor_body_idx.append(idx)

        filter_paths = list(filter_patterns)
        filter_body_idx = [self._backend._path_body_map.get(path, -1) for path in filter_paths]

        return _NewtonRigidContactView(
            self._backend, sensor_paths, sensor_body_idx, filter_paths, filter_body_idx, max_contact_data_count
        )

    def create_rigid_body_view(self, pattern):
        paths, idxs = _match_body_paths(self._backend, pattern)
        return _NewtonRigidBodyBatchView(self._backend, body_indices=idxs, prim_paths=paths, pattern=pattern)


class NewtonBackend(PhysicsBackend):
    supports_scene_queries = True
    supports_contact_reporting = True
    supports_cloth = True
    supports_particles = True

    # Newton/MuJoCo imports many BEHAVIOR object joints (cabinets, doors, drawers) as zero-gain effort
    # DOFs instead, so without an explicit passive damper they can drift under contact/gravity even when
    # the scene is supposed to start at rest. Robots are excluded (see _rebuild_impl): their
    # own controllers set real position-servo gains later via update_controller_mode()/set_gains().
    _PASSIVE_OBJECT_JOINT_KD = 5.0
    # Contact friction for a holonomic robot's floor-touching base links (its "wheels"). Those links
    # are non-articulated pads, not driven joints -- a holonomic base is driven through its virtual
    # base_footprint_x/y/rz joints instead, so the wheels can never roll and instead sit on the floor
    # at default_shape_cfg.mu (1.0), pinning the whole robot in place. Real casters roll freely;
    # approximate that by dropping their friction.
    _FLOOR_CONTACT_FRICTION = 0.02
    # Generalized armature / dry-friction floors applied to every articulated robot joint at build time.
    _JOINT_ARMATURE_FLOOR = 1.0e-2
    _JOINT_FRICTION_FLOOR = 2.0e-1
    # Critically damped finger contacts, with MuJoCo's timestep safety clamp enabled. The
    # scene-wide soft contacts allow centimeter-scale penetration during gripper closure.
    _FINGER_CONTACT_SOLREF = (0.01, 1.0)
    _FINGER_CONTACT_SOLMIX = 1000.0
    # Disabled WELD equality slots preallocated on every rebuild for create_attachment_constraint() (by default two
    # arms' worth of assisted grasps for two robots). Can be increased but requires a rebuild of the graph.
    _MIN_ATTACHMENT_SLOTS = 4

    def __init__(self, sim=None):
        super().__init__(sim)
        self._sim_context = None
        self._model = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._contacts = None
        self._solver = None
        self._collision_pipeline = None
        # Newton collision pipeline feeding the plain (rigid-only) SolverMuJoCo path -- see _rebuild()
        self._rigid_collision_pipeline = None
        self._mpm_in_place = False
        self._pre_step_cb = None
        self._post_step_cb = None
        self._joint_break_cb = None
        self._path_body_map = {}
        self._body_path_map = {}
        # Whether the shape BVH (used by raycast_closest/raycast_all) needs a refit before its next use --
        # set on anything that can change body_q.
        self._bvh_dirty = True
        # Dedicated contacts buffer for rigid contact reporting (create_rigid_contact_view).
        self._contact_report_contacts = None
        self._contacts_dirty = True
        self._rebuilding = False
        self._path_joint_map = {}
        self._path_cloth_particle_map = {}  # cloth entity prim_path -> (particle_start, particle_end)
        # Fluid/granular particle systems (MicroPhysicalParticleSystem, e.g. "water") -- unlike cloth,
        # particle count changes at runtime (generate_particles()/remove_particles()), so the
        # authoritative current state lives here (synced from the live model before each mutation),
        # not just derived from a fixed USD mesh.
        self._fluid_systems = {}  # system_name -> {"pos": (N, 3) tensor, "vel": (N, 3) tensor}
        self._path_fluid_particle_map = {}  # system_name -> (particle_start, particle_end) in the CURRENT model
        self._fluid_surface_cache = {}  # see get_fluid_surface_meshes()
        self._articulation_handles = {}
        self._rigid_body_handles = {}
        self._gravity = 9.81
        # Object set the model was last built from (entity prim paths) -- used by
        # refresh_physics_sim_view() to detect when a dynamic add/remove needs a full rebuild.
        self._known_object_paths = frozenset()
        self._joints_dirty = False
        self._joint_change_listener = None
        # CUDA graphs replaying one solver substep.
        self._step_graphs = {}
        self._step_graph_substeps = 0
        self._step_graph_disabled = False
        self._breakable_joints = {}
        self._already_broken_joints = set()
        self._joints_settle_steps_remaining = 0
        self._attachment_constraints = {}
        self._attachment_slots = []
        # Set when a body's density (or a zero mass, meaning "derive from density") is authored on a live
        # model.
        self._mass_properties_dirty = False
        # MacroPhysicalParticleSystem particles are individual rigid-body prims under their system (not scene
        # objects), imported by _rebuild_impl() after every object so object body indices stay stable.
        self._known_macro_particle_paths = frozenset()
        self._macro_particles_dirty = False
        self._ensuring_model_current = False
        self._model_generation = 0
        self._attachment_slot_count = self._MIN_ATTACHMENT_SLOTS
        self._warned_cpu_device = False

    # ---- Lifecycle ----

    @classmethod
    def create_app(cls):
        if gm.RENDER_BACKEND == "kit":
            raise ValueError("Newton physics does not support gm.RENDER_BACKEND = 'kit'; use 'newton' or 'none'.")
        from omnigibson.simulator import _StandaloneApp

        return _StandaloneApp()

    def add_ground_plane(self, prim_path, visible=True, color=None):
        # The physics-side ground plane is added by _rebuild_impl()'s builder.add_ground_plane(); this
        # is just a bare USD placeholder so og.sim.floor_plane stays non-None for code that expects it.
        with self.sim.editing_usd():
            self.sim.stage.DefinePrim(prim_path, "Xform")

    def create_physics_context(self, physics_dt, rendering_dt, device):
        self._sim_context = StandaloneSimulationContext(
            backend=self, physics_dt=physics_dt, rendering_dt=rendering_dt, device=device
        )
        # Watches for Joint-schema prims being added/removed/resynced anywhere on the stage -- e.g.
        # AttachedTo creating or deleting its attachment joint at runtime -- and flags a rebuild, consumed
        # lazily by step_physics_once(). Independent of Simulator's own _usd_guard_listener.
        self._joint_change_listener = lazy.pxr.Tf.Notice.Register(
            lazy.pxr.Usd.Notice.ObjectsChanged, self._on_usd_joint_changed, self._sim_context.stage
        )
        return self._sim_context

    def start_step_callbacks(self, pre_step_fn, post_step_fn, joint_break_fn):
        # No Kit event system to subscribe to standalone; step_physics_once() invokes these directly.
        self._pre_step_cb = pre_step_fn
        self._post_step_cb = post_step_fn
        self._joint_break_cb = joint_break_fn

    def stop_step_callbacks(self):
        self._pre_step_cb = None
        self._post_step_cb = None
        self._joint_break_cb = None

    def apply_engine_settings(
        self,
        gravity,
        enable_ccd,
        use_gpu_dynamics,
        gpu_pairs_capacity,
        gpu_aggr_pairs_capacity,
        gpu_max_particle_contacts,
        gpu_max_rigid_contact_count,
        gpu_max_rigid_patch_count,
    ):
        # Newton has no PhysX-scene-settings concept; gravity is a ModelBuilder-time setting (applied
        # in before_play()'s model build), and CCD/GPU-buffer-capacity concepts don't map onto it.
        self._gravity = gravity

    def get_physics_context(self):
        raise NotImplementedError("No physics-scene-prim concept exists for the Newton backend.")

    @property
    def stage(self):
        return self._sim_context.stage

    @property
    def current_time(self):
        return self._sim_context.current_time

    @property
    def current_time_step_index(self):
        return self._sim_context.current_time_step_index

    @property
    def initial_physics_dt(self):
        return self._sim_context._initial_physics_dt

    @property
    def initial_rendering_dt(self):
        return self._sim_context._initial_rendering_dt

    def is_playing(self):
        return self._sim_context.is_playing()

    def is_stopped(self):
        return self._sim_context.is_stopped()

    def get_physics_dt(self):
        return self._sim_context.get_physics_dt()

    def get_rendering_dt(self):
        return self._sim_context.get_rendering_dt()

    def set_simulation_dt(self, physics_dt=None, rendering_dt=None):
        self._sim_context.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)

    def get_device(self):
        return self._sim_context.device

    def set_device(self, device):
        self._sim_context.device = device

    def play(self):
        self._sim_context.play()

    def pause(self):
        self._sim_context.pause()

    def stop(self):
        self._sim_context.stop()

    def render(self):
        self._sim_context.render()

    def step(self, render):
        # Apply any deferred particle-set change before physics runs.
        self.ensure_model_current()
        self._sim_context.step(render=render)

    # Relative margin added on top of a body's triangle-inequality deficit by _balance_body_inertia().
    # MuJoCo has NO tolerance here (measured: it rejects a deficit of -1e-10 and accepts exactly 0), so
    # this only has to survive the model's own float32 rounding.
    _INERTIA_BALANCE_MARGIN = 1e-6

    @staticmethod
    def _dof_effective_inertia(builder, joint_idx, qd_start, is_revolute):
        """
        Inertia (kg*m^2) or mass (kg) a single-DOF joint moves: its child body's, about the joint axis for a
        revolute joint. Ignores bodies further down the chain, which BEHAVIOR's doors/drawers/lids don't have.
        """
        import numpy as np

        child = builder.joint_child[joint_idx]
        mass = float(builder.body_mass[child])
        if not is_revolute:
            return mass
        x_c = builder.joint_X_c[joint_idx]
        # Joint axis (authored in the joint frame) and anchor, in the child body's frame
        q = np.array(x_c[3:7], dtype=np.float64)  # xyzw
        v = np.array(builder.joint_axis[qd_start], dtype=np.float64)
        t = 2.0 * np.cross(q[:3], v)
        axis = v + q[3] * t + np.cross(q[:3], t)
        axis /= np.linalg.norm(axis)
        r = np.array(builder.body_com[child], dtype=np.float64) - np.array(x_c[0:3], dtype=np.float64)
        r_perp = r - (r @ axis) * axis
        inertia = np.array(builder.body_inertia[child], dtype=np.float64).reshape(3, 3)
        return float(axis @ inertia @ axis + mass * (r_perp @ r_perp))

    def _balance_body_inertia(self):
        """
        Make every body's inertia satisfy MuJoCo's triangle inequality (principal moments A + B >= C),
        the equivalent of MuJoCo's own ``balanceinertia`` compiler flag, which newton never sets.

        Without this, a single bad body aborts SolverMuJoCo construction for the WHOLE scene and the
        run silently continues with no physics at all.

        Corrects by adding a scalar to the diagonal, which shifts all three principal moments equally
        and so preserves the principal axes exactly (the same correction newton's
        verify_and_correct_inertia() applies).
        """
        import numpy as np

        inertia = self._model.body_inertia.numpy()
        if inertia.size == 0:
            return

        moments = np.linalg.eigvalsh(inertia.astype(np.float64))  # (n_bodies, 3), ascending
        largest = moments[:, 2]
        deficit = largest - moments[:, 0] - moments[:, 1]
        # Anything that does not already clear the inequality by at least the margin gets affected.
        # Bodies with no inertia at all (massless/static) are left alone.
        needs_fix = (deficit > -self._INERTIA_BALANCE_MARGIN * largest) & (largest > 0.0)
        if not needs_fix.any():
            return

        shift = (deficit + self._INERTIA_BALANCE_MARGIN * largest)[needs_fix]
        fixed = inertia.copy()
        idx = np.flatnonzero(needs_fix)
        fixed[idx] += np.eye(3, dtype=fixed.dtype) * shift[:, None, None].astype(fixed.dtype)
        self._model.body_inertia.assign(fixed)

        # body_inv_inertia must stay the actual inverse of body_inertia; only the touched bodies move,
        # and each is invertible now that its smallest principal moment has grown.
        inv = self._model.body_inv_inertia.numpy().copy()
        inv[idx] = np.linalg.inv(fixed[idx].astype(np.float64)).astype(inv.dtype)
        self._model.body_inv_inertia.assign(inv)

        log.debug(
            f"Balanced inertia for {len(idx)} body(ies) violating MuJoCo's A + B >= C "
            f"(worst deficit {deficit[needs_fix].max():.3e})"
        )

    # Substeps to run uncaptured before attempting a capture: warp forbids kernel-module loads during
    # a capture, and SolverMuJoCo loads its modules lazily over its first few steps.
    _STEP_GRAPH_WARMUP_SUBSTEPS = 3

    def _step_graph(self, dt):
        """
        CUDA graph replaying one SolverMuJoCo substep for the current state-buffer pair, or None to
        take the uncaptured path (graph disabled, still warming up, CPU device, or capture failed).

        Returns a graph that has NOT been launched so the caller must launch the returned graph to actually
        advance this substep.
        """
        if self._step_graph_disabled or not gm.USE_NEWTON_STEP_GRAPH:
            return None
        if self._solver is None or self._model is None or self._model.device.is_cpu:
            return None

        key = (id(self._state_0), id(self._state_1), dt)
        graph = self._step_graphs.get(key)
        if graph is not None:
            return graph

        if self._step_graph_substeps < self._STEP_GRAPH_WARMUP_SUBSTEPS:
            self._step_graph_substeps += 1
            return None

        import warp as wp

        try:
            with wp.ScopedCapture() as capture:
                self._solver.step(self._state_0, self._state_1, self._control, self._contacts, dt)
            graph = capture.graph
        except Exception as e:
            # Not fatal: the uncaptured path is always available, just slower. Fallback permanently
            # rather than paying a failed capture attempt every substep.
            self._step_graph_disabled = True
            log.warning(f"Newton physics step graph capture failed ({e}); falling back to uncaptured steps.")
            return None

        self._step_graphs[key] = graph
        return graph

    def step_physics_once(self, current_time):
        if self._mass_properties_dirty or self._macro_particles_dirty:
            self._mass_properties_dirty = False
            self._macro_particles_dirty = False
            self._joints_dirty = True
        if self._joints_dirty:
            self._joints_dirty = False
            self._rebuild()

        if self._pre_step_cb is not None:
            self._pre_step_cb()

        if self._solver is not None:
            if self._mpm_in_place:
                applied_body_f = self._applied_body_forces()
                self._state_0.clear_forces()
                self._restore_applied_body_forces(applied_body_f)
                if self._collision_pipeline is not None:
                    self._collision_pipeline.collide(self._state_0, self._contacts)
                self._solver.step(self._state_0, self._state_0, self._control, self._contacts, self.get_physics_dt())
            elif self._collision_pipeline is not None:
                substep_dt = self.get_physics_dt() / self._PARTICLE_SUBSTEPS
                # Hold an applied force through every substep, so it delivers one full physics step's impulse
                applied_body_f = self._applied_body_forces()
                for _ in range(self._PARTICLE_SUBSTEPS):
                    self._state_0.clear_forces()
                    self._restore_applied_body_forces(applied_body_f)
                    self._collision_pipeline.collide(self._state_0, self._contacts)
                    self._solver.step(self._state_0, self._state_1, self._control, self._contacts, substep_dt)
                    self._state_0, self._state_1 = self._state_1, self._state_0
            else:
                # Replay this substep from a CUDA graph when one is available.
                step_dt = self.get_physics_dt()
                if self._rigid_collision_pipeline is not None:
                    self._rigid_collision_pipeline.collide(self._state_0, self._contacts)
                graph = self._step_graph(step_dt)
                if graph is not None:
                    import warp as wp

                    wp.capture_launch(graph)
                else:
                    self._solver.step(self._state_0, self._state_1, self._control, self._contacts, step_dt)
                self._state_0, self._state_1 = self._state_1, self._state_0
            self._state_0.clear_forces()

            self._bvh_dirty = True
            self._contacts_dirty = True
            self._check_joint_breaks()

        if self._post_step_cb is not None:
            self._post_step_cb()

    def _applied_body_forces(self):
        """Copy of the apply_force_at_pos()/apply_torque() wrenches pending on state_0, or None."""
        body_f = self._state_0.body_f
        return None if body_f is None else _wp_to_torch(body_f).clone()

    def _restore_applied_body_forces(self, applied_body_f):
        if applied_body_f is not None:
            _wp_to_torch(self._state_0.body_f).copy_(applied_body_f)

    def _compute_mass_matrix(self, handle):
        """
        Generalized (joint-space) mass matrix for one articulation handle, shape (n_dof, n_dof),
        matching PhysX's ArticulationView.get_generalized_mass_matrices() convention.
        """
        import newton

        model = self._model
        H_all = _wp_to_torch(newton.eval_mass_matrix(model, self._state_0))

        art_idx = int(_wp_to_torch(model.joint_articulation)[handle._global_joint_idx[0]].item())
        joint_qd_start = _wp_to_torch(model.joint_qd_start)
        art_start = int(_wp_to_torch(model.articulation_start)[art_idx].item())
        art_end = int(_wp_to_torch(model.articulation_end)[art_idx].item())
        dof_start = int(joint_qd_start[art_start].item())
        dof_end = int(joint_qd_start[art_end].item())
        dof_count = dof_end - dof_start

        # Both axes are DOF-indexed, so both follow the handle's order.
        cols = _native_to_handle_dof_cols(handle, dof_start, dof_count)
        H = H_all[art_idx, :dof_count, :dof_count]
        return H[cols][:, cols]

    def _compute_jacobian(self, handle):
        """
        World-frame spatial Jacobian for one articulation handle with convention: shape
        (n_links, 6, n_dof), where n_links excludes the root body (row 0 is the first non-root link)
        and the 6 rows are [linear(3), angular(3)] per link.
        """
        import newton

        model = self._model
        J_all = _wp_to_torch(newton.eval_jacobian(model, self._state_0))

        art_idx = int(_wp_to_torch(model.joint_articulation)[handle._global_joint_idx[0]].item())
        joint_qd_start = _wp_to_torch(model.joint_qd_start)
        art_start = int(_wp_to_torch(model.articulation_start)[art_idx].item())
        art_end = int(_wp_to_torch(model.articulation_end)[art_idx].item())
        link_count = art_end - art_start
        dof_start = int(joint_qd_start[art_start].item())
        dof_end = int(joint_qd_start[art_end].item())
        dof_count = dof_end - dof_start

        max_links = model.max_joints_per_articulation
        max_dofs = model.max_dofs_per_articulation
        # Drop row-block 0 (the root body's own jacobian) and trim trailing zero-padding on both axes.
        J = J_all[art_idx].reshape(max_links, 6, max_dofs)[1:link_count, :, :dof_count]
        return J[:, :, _native_to_handle_dof_cols(handle, dof_start, dof_count)]

    @property
    def physics_sim_view(self):
        return _NewtonPhysicsSimView(self)

    def _macro_particle_roots(self):
        """{particle prim path: owning system} for every live MacroPhysicalParticleSystem particle."""
        from omnigibson.systems.macro_particle_system import MacroPhysicalParticleSystem

        paths = {}
        for scene in self.sim.scenes:
            for system in scene.active_systems.values():
                if not isinstance(system, MacroPhysicalParticleSystem) or not system.initialized:
                    continue
                root = self.sim.stage.GetPrimAtPath(f"{system.prim_path}/particles")
                if not root.IsValid():
                    continue
                for child in root.GetChildren():
                    if child.IsActive() and child.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
                        paths[str(child.GetPath())] = system
        return paths

    def ensure_model_current(self):
        """
        Applies a lazily-deferred model change (e.g. new macro particles) now, if one is pending and it's safe to.
        """
        if (
            not self._macro_particles_dirty
            or self._rebuilding
            or self._ensuring_model_current
            or self.sim.currently_stepping
        ):
            return
        self._ensuring_model_current = True
        try:
            self._macro_particles_dirty = False
            self._rebuild()
            self.sim.update_handles()
        finally:
            self._ensuring_model_current = False

    def refresh_physics_sim_view(self):
        if self._model is None or not self.is_playing():
            return
        if frozenset(self._macro_particle_roots()) != self._known_macro_particle_paths:
            self._macro_particles_dirty = True
        current_paths = frozenset(obj.prim_path for scene in self.sim.scenes for obj in scene.objects)
        if current_paths != self._known_object_paths:
            self._rebuild()

    def invalidate_physics_sim_view(self):
        pass

    def flush_changes(self):
        pass

    def sync_to_render_layer(self):
        # No render-facing scene mirror: Newton's own viewers read live body poses from the model state.
        pass

    # ---- Batched DOF target writes ----

    def set_dof_position_targets(self, view, data, indices, cast=False):
        view.set_dof_position_targets_fast(data, indices)

    def set_dof_velocity_targets(self, view, data, indices, cast=False):
        view.set_dof_velocity_targets_fast(data, indices)

    def set_dof_actuation_forces(self, view, data, indices, cast=False):
        view.set_dof_actuation_forces_fast(data, indices)

    # ---- Prim transform reads ----

    def get_world_transform_with_scale(self, prim_path):
        # Compute the raw USD transform first; if this prim is itself a tracked physics body,
        # add the live translation/rotation instead of the stale USD ones.
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        raw_matrix = lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        live_pose = self.get_live_world_pose(prim_path)
        if live_pose is not None:
            return _live_matrix_from_pose(raw_matrix, live_pose)

        ancestor_prim = prim.GetParent()
        while ancestor_prim and ancestor_prim.IsValid():
            ancestor_live_pose = self.get_live_world_pose(str(ancestor_prim.GetPath()))
            if ancestor_live_pose is not None:
                ancestor_raw_matrix = lazy.pxr.UsdGeom.Xformable(ancestor_prim).ComputeLocalToWorldTransform(
                    lazy.pxr.Usd.TimeCode.Default()
                )
                local_to_ancestor = raw_matrix * ancestor_raw_matrix.GetInverse()
                return local_to_ancestor * _live_matrix_from_pose(ancestor_raw_matrix, ancestor_live_pose)
            ancestor_prim = ancestor_prim.GetParent()
        return raw_matrix

    def get_local_transform_with_scale(self, prim_path):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        return lazy.pxr.UsdGeom.Xformable(prim).GetLocalTransformation(lazy.pxr.Usd.TimeCode.Default())

    def get_live_world_pose(self, prim_path):
        if self._rebuilding or self._state_0 is None or self._state_0.body_q is None:
            return None
        body_idx = self._path_body_map.get(prim_path)
        if body_idx is None or body_idx >= self._state_0.body_q.shape[0]:
            return None
        body_q = _wp_to_torch(self._state_0.body_q)
        return body_q[body_idx, :3].clone(), body_q[body_idx, 3:7].clone()

    # ---- Model build (before_play hook) ----

    def before_play(self, sim):
        self._rebuild()

    def _rebuild(self):
        """
        (Re)builds the entire Newton model from the current scene's USD stage. Used both for the
        initial model build (Simulator.play()'s before_play() hook) and for a mid-episode rebuild
        triggered by a dynamic object add/remove. This is an expensive operation.

        A rebuild-from-USD would silently reset every already-simulated object to its initial position,
        so capture every existing handle's state before rebuilding and restore it after.
        """
        import newton
        import warp as wp

        self._joints_dirty = False
        self._step_graphs = {}
        self._step_graph_substeps = 0
        self._rebuilding = True
        try:
            self._rebuild_impl(newton, wp)
        finally:
            self._rebuilding = False

    def _on_usd_joint_changed(self, notice, stage):
        if self._joints_dirty or self._rebuilding:
            return
        for path in notice.GetResyncedPaths():
            prim = stage.GetPrimAtPath(path.GetPrimPath())
            # A removed prim also needs to be handled as dirty.
            if not prim or not prim.IsValid() or prim.IsA(lazy.pxr.UsdPhysics.Joint):
                self._joints_dirty = True
                return

    def _apply_floor_contact_friction(self, builder, robot, robot_path_body_map):
        """Drop contact friction on @robot's floor-touching base links, returning the newton body
        indices affected.

        The link names are read from the robot's own definition (`locomotion.floor_touching_base_link_names`)

        Lowering mu here is necessary but not sufficient: MuJoCo mixes the two geoms' friction and the
        larger wins at equal priority, so a 0.02 pad against the 0.9 ground plane still resolves to
        0.9. _elevate_low_friction_geom_priority() (called once the solver exists) makes these geoms'
        friction authoritative.
        """
        # Holonomic bases only. Their floor-touching links are undriven pads, so dropping friction is
        # the right caster approximation. A DIFFERENTIAL-drive robot propels itself through friction at
        # exactly these links, so doing this to one would destroy its traction instead. No shipped
        # diff-drive asset declares floor_touching_base_link_names today (only r1/r1pro/tiago do, all
        # holonomic), but the field is a generic enough concept that one plausibly could later.
        if not getattr(robot, "is_holonomic_base", False):
            return set()
        link_names = getattr(robot, "floor_touching_base_link_names", None) or []
        if not link_names:
            return set()

        body_idx = set()
        for link_name in link_names:
            idx = robot_path_body_map.get(f"{robot.prim_path}/{link_name}")
            if idx is not None:
                body_idx.add(int(idx))
        if not body_idx:
            log.warning(
                f"Robot {robot.name} declares floor_touching_base_link_names={link_names}, but none "
                f"resolved to an imported Newton body -- leaving their contact friction at the default."
            )
            return set()

        for shape_idx, shape_body in enumerate(builder.shape_body):
            if int(shape_body) in body_idx and shape_idx < len(builder.shape_material_mu):
                builder.shape_material_mu[shape_idx] = self._FLOOR_CONTACT_FRICTION
        return body_idx

    def _elevate_low_friction_geom_priority(self, body_idx):
        """Make the low-friction floor-contact geoms win MuJoCo's contact-parameter mix.

        MuJoCo combines each contact's friction from both geoms, with the larger value winning at
        equal priority -- so _apply_floor_contact_friction()'s 0.02 pads would still produce a 0.9
        contact against the ground plane. Raising just those geoms' `geom_priority` makes their own
        friction authoritative (mujoco_warp honors geom_priority in its contact kernels).
        """
        if not body_idx:
            return
        solver = self._get_mujoco_solver()
        if solver is None or getattr(solver, "mj_model", None) is None:
            return
        mj_model = solver.mj_model
        mjc_body_to_newton = _wp_to_torch(solver.mjc_body_to_newton)[0]
        changed = False
        for geom_idx in range(mj_model.ngeom):
            mjc_body = int(mj_model.geom_bodyid[geom_idx])
            if not (0 <= mjc_body < len(mjc_body_to_newton)):
                continue
            if int(mjc_body_to_newton[mjc_body]) in body_idx:
                mj_model.geom_priority[geom_idx] = 1
                changed = True
        if changed and getattr(solver, "mjw_model", None) is not None:
            solver.mjw_model.geom_priority.assign(mj_model.geom_priority)

    def _configure_finger_contacts(self, body_idx):
        """Keep finger contacts firm without stiffening resting contacts throughout the scene."""
        if not body_idx:
            return
        solver = self._get_mujoco_solver()
        if solver is None:
            return
        mj_model = solver.mj_model
        body_map = solver.mjc_body_to_newton.numpy()[0]
        for geom_idx, mjc_body in enumerate(mj_model.geom_bodyid):
            if int(body_map[mjc_body]) in body_idx:
                mj_model.geom_solref[geom_idx] = self._FINGER_CONTACT_SOLREF
                # Weight finger contact response over the object's soft settings without changing
                # geom priority, which would also override friction mixing.
                mj_model.geom_solmix[geom_idx] = self._FINGER_CONTACT_SOLMIX
        solver.mjw_model.geom_solref.assign(mj_model.geom_solref)
        solver.mjw_model.geom_solmix.assign(mj_model.geom_solmix)

    def _find_cross_object_joints(self, objects):
        """
        Returns {owning_obj: [(joint_prim, body0_path, body1_path), ...]} for every UsdPhysics.Joint
        prim physically nested under one of `objects`' own subtrees whose body0/body1 targets belong
        to two DIFFERENT objects -- e.g. AttachedTo's runtime-created attachment joint, connecting a
        link on the parent object to a link on the (separately, independently add_usd()-imported)
        child object.
        """
        prefixes = [(obj.prim_path + "/", obj) for obj in objects]

        def owning_object(path):
            for prefix, obj in prefixes:
                if path == prefix[:-1] or path.startswith(prefix):
                    return obj
            return None

        cross_joints = {}
        for obj in objects:
            if obj.prim_type == PrimType.CLOTH:
                continue
            root_prim = self.sim.stage.GetPrimAtPath(obj.prim_path)
            if not root_prim.IsValid():
                continue
            for prim in lazy.pxr.Usd.PrimRange(root_prim):
                if not prim.IsA(lazy.pxr.UsdPhysics.Joint):
                    continue
                joint_api = lazy.pxr.UsdPhysics.Joint(prim)
                body0_targets = joint_api.GetBody0Rel().GetTargets()
                body1_targets = joint_api.GetBody1Rel().GetTargets()
                body0_path = str(body0_targets[0]) if body0_targets else None
                body1_path = str(body1_targets[0]) if body1_targets else None
                owner0 = owning_object(body0_path) if body0_path else None
                owner1 = owning_object(body1_path) if body1_path else None
                if owner0 is not None and owner1 is not None and owner0 is not owner1:
                    cross_joints.setdefault(obj, []).append((prim, body0_path, body1_path))
        return cross_joints

    def _read_joint_local_frames(self, prim, wp):
        """Reads physics:localPos0/localRot0 (parent frame) and localPos1/localRot1 (child frame) as
        (parent_xform, child_xform) wp.transform pairs, matching create_joint()'s own attribute
        writes."""

        def read_xform(pos_attr, rot_attr):
            pos = prim.GetAttribute(pos_attr).Get()
            rot = prim.GetAttribute(rot_attr).Get()
            pos = (0.0, 0.0, 0.0) if pos is None else tuple(pos)
            if rot is None:
                quat = (0.0, 0.0, 0.0, 1.0)
            else:
                imag = rot.GetImaginary()
                quat = (imag[0], imag[1], imag[2], rot.GetReal())
            return wp.transform(pos, quat)

        parent_xform = read_xform("physics:localPos0", "physics:localRot0")
        child_xform = read_xform("physics:localPos1", "physics:localRot1")
        return parent_xform, child_xform

    def _read_joint_break_thresholds(self, prim):
        """Returns (break_force, break_torque), each None if unauthored/infinite (never breaks)."""
        break_force = prim.GetAttribute("physics:breakForce").Get()
        break_torque = prim.GetAttribute("physics:breakTorque").Get()
        break_force = None if break_force is None or math.isinf(break_force) else float(break_force)
        break_torque = None if break_torque is None or math.isinf(break_torque) else float(break_torque)
        return break_force, break_torque

    def _rebuild_impl(self, newton, wp):
        from pxr import Gf

        from omnigibson.physics_backends.newton_visuals import add_usd_visual_shapes, raw_world_transform

        for system_name in list(self._path_fluid_particle_map):
            self._sync_fluid_system_from_model(system_name)
        captured_state = self._capture_state()
        self._already_broken_joints = set()

        builder = newton.ModelBuilder(gravity=[0.0, 0.0, -self._gravity])
        # Use Newton defaults ke=2.5e3, kd=1e2, mu=1.0 and a 1 cm contact gap
        builder.default_shape_cfg.ke = 2.5e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.01
        # Keep the ground plane collidable but stop it from rendering to prevent visual fighting with floor meshes.
        ground_shape = builder.add_ground_plane()
        builder.shape_flags[ground_shape] = int(builder.shape_flags[ground_shape]) & ~int(newton.ShapeFlags.VISIBLE)

        # Lets add_usd() import mjc:* body/joint/shape attributes (e.g. mjc:gravcomp)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        if self._fluid_systems:
            # Must be registered before any particles referencing "mpm:*" custom attributes are added.
            from newton.solvers import SolverImplicitMPM

            SolverImplicitMPM.register_custom_attributes(builder)

        objects = [obj for scene in self.sim.scenes for obj in scene.objects]

        from omnigibson.robots import Robot

        objects.sort(key=lambda obj: 0 if isinstance(obj, Robot) else 1)

        # AttachedTo-style joints span two objects and must be added back manually only after every object's
        # bodies exist in the builder.
        cross_object_joints = self._find_cross_object_joints(objects)
        ignore_paths_by_obj = {
            obj: [str(prim.GetPath()) for prim, _, _ in joints] for obj, joints in cross_object_joints.items()
        }

        path_body_map = {}
        path_joint_map = {}
        path_cloth_particle_map = {}
        robot_joint_paths = set()
        low_friction_body_idx = set()
        finger_body_idx = set()
        # Macro-particle roots go last, so object body indices are unaffected by particle changes
        macro_particle_paths = self._macro_particle_roots()
        particle_systems = sorted({system.prim_path for system in macro_particle_paths.values()})
        import_roots = objects + [_ParticleImportRoot(self.sim.stage.GetPrimAtPath(path)) for path in particle_systems]
        for obj in import_roots:
            is_particle_root = getattr(obj, "is_particle_root", False)
            if obj.prim_type == PrimType.CLOTH:
                self._add_cloth_object(builder, obj, path_cloth_particle_map)
                continue

            joint_count_before = len(builder.joint_articulation)
            info = builder.add_usd(
                self.sim.stage,
                root_path=obj.prim_path,
                floating=not obj.fixed_base,
                enable_self_collisions=False,
                collapse_fixed_joints=False,
                # Apply each collider's authored physics:approximation (convexHull on BEHAVIOR assets).
                # Scenes with fluid keep the raw meshes:
                # SolverImplicitMPM cannot build a collider from a convex mesh ("Shape type 10 not supported")
                skip_mesh_approximation=bool(self._fluid_systems),
                load_visual_shapes=False,
                schema_resolvers=[
                    newton.usd.SchemaResolverNewton(),
                    _physx_schema_resolver(),
                    newton.usd.SchemaResolverMjc(),
                ],
                # We never load add_usd()'s own visual geometry (load_visual_shapes=False):
                # omnigibson/utils/newton_viewer_recording.py registers real textured visual meshes
                # separately, directly with the viewer. Collision shapes should stay non-rendered.
                hide_collision_shapes=True,
                ignore_paths=getattr(obj, "ignore_paths", None) or ignore_paths_by_obj.get(obj),
            )

            add_usd_visual_shapes(newton, builder, self.sim.stage, obj.prim_path, info["path_body_map"])

            path_body_map.update(info["path_body_map"])
            path_joint_map.update(info["path_joint_map"])
            if isinstance(obj, Robot):
                robot_joint_paths.update(info["path_joint_map"].keys())
                low_friction_body_idx |= self._apply_floor_contact_friction(builder, obj, info["path_body_map"])
                if obj.is_manipulation:
                    for names in obj.finger_link_names.values():
                        for name in names:
                            idx = info["path_body_map"].get(f"{obj.prim_path}/{name}")
                            if idx is not None:
                                finger_body_idx.add(int(idx))

            # add_usd() only auto-wraps an object's joints into an articulation when the object's own
            # USD authors a UsdPhysics.ArticulationRootAPI marker -- true for robots and most fixed/floating-base
            # objects with real joints, but NOT for objects whose only joints are "meta-link"
            # attachments (e.g. a ceiling's light-fixture mount points, imported as ordinary
            # PhysicsFixedJoints to a massless child body).
            # Give it a fixed or free joint to attach it to the world root.
            object_body_idx = set(info["path_body_map"].values())
            connected_bodies = set(builder.joint_child[joint_count_before:])
            for body_idx in sorted(object_body_idx - connected_bodies):
                if obj.fixed_base:
                    builder.add_joint_fixed(parent=-1, child=body_idx)
                else:
                    builder.add_joint_free(parent=-1, child=body_idx)

            # Wrap any still-unassigned joints from this object into the articulation.
            unassigned = sorted(
                idx
                for idx in range(joint_count_before, len(builder.joint_articulation))
                if builder.joint_articulation[idx] < 0
            )
            if is_particle_root:
                # Each particle is its own free body, not one articulation
                for idx in unassigned:
                    builder.add_articulation([idx], label=f"{obj.prim_path}_{idx}")
                continue
            run_start = None
            for i, idx in enumerate(unassigned):
                if run_start is None:
                    run_start = idx
                if i + 1 == len(unassigned) or unassigned[i + 1] != idx + 1:
                    builder.add_articulation(list(range(run_start, idx + 1)), label=f"{obj.prim_path}_meta_links")
                    run_start = None

        # Now that every object's bodies exist in the builder, add back the cross-object joints.
        self._breakable_joints = {}
        # (parent_body, child_body, is_spherical, parent pos/quat, child pos/quat) per cross-object joint.
        self._cross_object_constraints = []
        for joints in cross_object_joints.values():
            for prim, body0_path, body1_path in joints:
                body0_idx = path_body_map.get(body0_path)
                body1_idx = path_body_map.get(body1_path)
                if body0_idx is None or body1_idx is None:
                    # One endpoint no longer exists (possibly removed in the same rebuild), so there's nothing to connect.
                    continue
                parent_xform, child_xform = self._read_joint_local_frames(prim, wp)
                joint_kwargs = dict(
                    parent=body0_idx,
                    child=body1_idx,
                    parent_xform=parent_xform,
                    child_xform=child_xform,
                    collision_filter_parent=True,
                )
                is_spherical = prim.IsA(lazy.pxr.UsdPhysics.SphericalJoint)
                if is_spherical:
                    joint_idx = builder.add_joint_ball(**joint_kwargs)
                else:
                    joint_idx = builder.add_joint_fixed(**joint_kwargs)
                # USD joint frames live in each body's *scaled* local frame; MuJoCo anchors are unscaled
                self._cross_object_constraints.append(
                    (
                        body0_idx,
                        body1_idx,
                        is_spherical,
                        th.tensor(tuple(parent_xform.p)) * self._body_scale(body0_path),
                        th.tensor(tuple(parent_xform.q)),
                        th.tensor(tuple(child_xform.p)) * self._body_scale(body1_path),
                        th.tensor(tuple(child_xform.q)),
                    )
                )
                joint_path = str(prim.GetPath())
                path_joint_map[joint_path] = joint_idx
                break_force, break_torque = self._read_joint_break_thresholds(prim)
                if break_force is not None or break_torque is not None:
                    self._breakable_joints[joint_path] = (break_force, break_torque, body0_idx, body1_idx)
        if self._breakable_joints:
            # Allow the joint to settle before break-force checking starts since the initial attachment force
            # may be high.
            self._joints_settle_steps_remaining = 1

        # Allocate slots for create_attachment_constraint() so that we don't require a rebuild later.
        self._attachment_slot_count = max(self._attachment_slot_count, len(self._attachment_constraints))
        self._attachment_slots = []
        if builder.body_count > 0:
            for _ in range(self._attachment_slot_count):
                slot = builder.add_custom_values(
                    **{
                        "mujoco:equality_constraint_type": int(newton.EqType.WELD),
                        "mujoco:equality_constraint_body1": -1,
                        "mujoco:equality_constraint_body2": 0,
                        "mujoco:equality_constraint_enabled": False,
                        "mujoco:equality_constraint_label": "og_attachment_slot",
                    }
                )
                self._attachment_slots.append(slot["mujoco:equality_constraint_type"])

        self._path_body_map = path_body_map
        self._body_path_map = {idx: path for path, idx in path_body_map.items()}
        self._path_joint_map = path_joint_map
        self._path_cloth_particle_map = path_cloth_particle_map

        # Fluid/granular particle systems -- added after cloth, each system's CURRENT particle state
        # (self._fluid_systems, kept in sync by generate_particles()/remove_particles()/_capture_state())
        # is the source of truth, not anything derived from a USD mesh.
        path_fluid_particle_map = {}
        for system_name, particles in self._fluid_systems.items():
            particle_start = builder.particle_count
            pos = particles["pos"]
            vel = particles["vel"]
            builder.add_particles(
                pos=[wp.vec3(*p) for p in pos.tolist()],
                vel=[wp.vec3(*v) for v in vel.tolist()],
                mass=[self._FLUID_DEFAULT_PARTICLE_MASS] * len(pos),
                radius=[self._FLUID_DEFAULT_PARTICLE_RADIUS] * len(pos),
                custom_attributes=self._FLUID_MATERIAL_DEFAULTS,
            )
            path_fluid_particle_map[system_name] = (particle_start, builder.particle_count)
        self._path_fluid_particle_map = path_fluid_particle_map

        # add_usd() substitutes a large nonzero default target_ke/target_kd for any joint whose
        # authored UsdPhysics.DriveAPI stiffness/damping is exactly 0 -- found empirically (ke=1e7,
        # kd=1e5 imported for EVERY Fetch joint, even though every single one authors stiffness=0 in
        # its USD; OmniGibson's joints are driven by direct per-step effort application computed in
        # Python, not the physics engine's built-in position servo, so 0 is correct and deliberate).
        # This is not cosmetic: JointPrim._initialize() infers each joint's ControlType from
        # get_gains()'s returned kp (kp == 0 -> EFFORT/VELOCITY, kp != 0 -> POSITION) -- importing a
        # phantom nonzero kp silently flips every joint to POSITION control, driven by a completely
        # untuned spring the robot's own controllers never asked for. For lightweight, low-effort-limit
        # joints (e.g. head_tilt_joint: 0.68 N*m effort limit) this saturates every step from the
        # tiniest position error, causing sustained max-torque chattering that snowballs into a NaN
        # divergence within a few dozen steps -- root-caused via mass-bisection (crushing every
        # upper-body link's mass to near-zero except head_tilt_link still reproduced the divergence
        # alone) plus reading the USD drive directly (UsdPhysics.DriveAPI(...).GetStiffnessAttr()) to
        # confirm every joint authors 0. Overwrite from the authored USD values directly rather than
        # trust add_usd()'s defaulting here.
        joint_qd_start = builder.joint_qd_start
        for joint_path, joint_idx in path_joint_map.items():
            qd_start = joint_qd_start[joint_idx]
            # Zero-DOF joints (e.g. the FIXED joints attaching massless reference frames like
            # laser_link/eyes/eef_link) don't reserve a target_ke/kd slot at all -- their qd_start is
            # just a "one past the end" placeholder shared with whatever joint comes NEXT (or the total
            # dof count, if they're last). Writing at that index therefore lands on the next joint's
            # DOF and silently overwrites its gains, effort limit and target mode with the fixed
            # joint's (absent) drive values. Bounding on `qd_start >= len(...)` only catches the case
            # where such a joint happens to be last; r1pro has six aliased slots, among them
            # left/right_gripper_joint sitting immediately before each gripper's finger_joint1, and
            # torso_joint1 sharing with all six wheel/steer joints. Skip on the joint's real DOF count.
            next_qd_start = (
                joint_qd_start[joint_idx + 1] if joint_idx + 1 < len(joint_qd_start) else builder.joint_dof_count
            )
            if next_qd_start - qd_start <= 0 or qd_start >= len(builder.joint_target_ke):
                continue
            prim = self.sim.stage.GetPrimAtPath(joint_path)
            drive = lazy.pxr.UsdPhysics.DriveAPI.Get(prim, "angular") or lazy.pxr.UsdPhysics.DriveAPI.Get(
                prim, "linear"
            )
            stiffness = drive.GetStiffnessAttr().Get() if drive else 0.0
            damping = drive.GetDampingAttr().Get() if drive else 0.0
            builder.joint_target_ke[qd_start] = stiffness if stiffness is not None else 0.0
            builder.joint_target_kd[qd_start] = damping if damping is not None else 0.0
            # DatasetObject authors its passive joint damping as an "acceleration" drive (per degree for
            # revolute joints), which PhysX scales by the joint's effective inertia. MuJoCo applies kd as a
            # plain torque, so convert: raw, a laptop lid (I ~3e-3 kg*m^2) decayed at 16/s instead of 2.9/s
            if (
                drive
                and damping
                and joint_path not in robot_joint_paths
                and drive.GetTypeAttr().Get() == "acceleration"
            ):
                is_revolute = builder.joint_type[joint_idx] == int(newton.JointType.REVOLUTE)
                builder.joint_target_kd[qd_start] = (
                    damping
                    * (180.0 / math.pi if is_revolute else 1.0)
                    * self._dof_effective_inertia(builder, joint_idx, qd_start, is_revolute)
                )
            # Critically damp object joint limits. SolverMuJoCo maps limit (ke, kd) to a mass-normalized solref
            # with damping ratio kd / (2 sqrt(ke)): the importer's uniform 1e4/10 gives 0.05, and a cabinet door
            # starting 0.75 cm inside a neighbouring sink bounced between the sink and its closed stop, gaining
            # energy until the whole scene went NaN
            if joint_path not in robot_joint_paths and builder.joint_type[joint_idx] in (
                int(newton.JointType.REVOLUTE),
                int(newton.JointType.PRISMATIC),
            ):
                builder.joint_limit_kd[qd_start] = max(
                    float(builder.joint_limit_kd[qd_start]), 2.0 * math.sqrt(float(builder.joint_limit_ke[qd_start]))
                )
            # add_usd() infers joint_target_mode from the SAME authored (ke, kd) via
            # JointTargetMode.from_gains(): a drive present with both gains at 0 -> EFFORT, which the
            # SolverMuJoCo backend takes literally -- "No MuJoCo actuator is created for this DOF" (see
            # newton.JointTargetMode's own docstring). That's fine for a joint that's genuinely only
            # ever driven by set_joint_efforts(), but every one of these joints is actually driven by
            # OmniGibson's JointController, which for POSITION/VELOCITY control_type calls
            # ArticulationView.set_gains() (isaac_kp/isaac_kd, defaulting to a real nonzero value -- see
            # controller_base.py's DEFAULT_ISAAC_KP) + set_joint_position/velocity_targets() at runtime,
            # never joint_f. Under PhysX this always works, since Isaac's per-DOF PD servo exists
            # unconditionally regardless of the USD-authored drive gains. Under MuJoCo, with mode left
            # at EFFORT, there is no actuator for set_gains()'s later writes to ever reach -- confirmed
            # empirically: gains + position targets update correctly in the Newton model, but the joint
            # never moves. Force POSITION_VELOCITY here (independent of the ke=kd=0 above, which stays
            # correct/deliberate -- ineffective and passive until a controller sets real gains) so the
            # actuator infrastructure exists once a controller does; joint_f-based effort control (the
            # other path some controllers use) is applied additively regardless of this mode, so this
            # doesn't affect effort-driven joints either.
            builder.joint_target_mode[qd_start] = int(newton.JointTargetMode.POSITION_VELOCITY)

            if joint_path in robot_joint_paths:
                builder.joint_effort_limit[qd_start] = float("inf")

            # Passive damping for non-robot object joints with no authored drive (cabinets, doors,
            # drawers) -- see _PASSIVE_OBJECT_JOINT_KD's class-level comment. Robots are excluded: their
            # own controllers set real position-servo gains later via update_controller_mode(), and this
            # floor would fight that in the meantime (harmless once overwritten, but pointless).
            # Free/fixed root joints are excluded too -- damping a free-floating root would resist an
            # object's natural fall/motion, which is not what a "settled articulation" damper is for.
            if (
                joint_path not in robot_joint_paths
                and builder.joint_target_ke[qd_start] == 0.0
                and builder.joint_target_kd[qd_start] == 0.0
                and builder.joint_type[joint_idx] not in (int(newton.JointType.FIXED), int(newton.JointType.FREE))
            ):
                builder.joint_target_kd[qd_start] = self._PASSIVE_OBJECT_JOINT_KD

        # Armature / dry-friction floors -- see _JOINT_ARMATURE_FLOOR.
        if self._JOINT_ARMATURE_FLOOR or self._JOINT_FRICTION_FLOOR:
            # Robot joints only, as IsaacLab adds no such floors. On a light object part they swamp its real
            # inertia: a 50 g spoon's free joint fell at 0.8 g and could not tip over a bowl's rim, and a laptop
            # lid spun at 2 rad/s stopped within 0.1 rad (PhysX: 1.5 rad)
            robot_joint_idxs = {path_joint_map[p] for p in robot_joint_paths if p in path_joint_map}
            skipped = (int(newton.JointType.FIXED), int(newton.JointType.FREE))
            n_dofs = len(builder.joint_qd)
            for joint_idx, joint_type in enumerate(builder.joint_type):
                if int(joint_type) in skipped or joint_idx not in robot_joint_idxs:
                    continue
                qd_start = builder.joint_qd_start[joint_idx]
                qd_stop = (
                    builder.joint_qd_start[joint_idx + 1] if joint_idx + 1 < len(builder.joint_qd_start) else n_dofs
                )
                for dof_idx in range(qd_start, qd_stop):
                    if dof_idx < len(builder.joint_armature):
                        builder.joint_armature[dof_idx] = max(
                            float(builder.joint_armature[dof_idx]), self._JOINT_ARMATURE_FLOOR
                        )
                    if dof_idx < len(builder.joint_friction):
                        builder.joint_friction[dof_idx] = max(
                            float(builder.joint_friction[dof_idx]), self._JOINT_FRICTION_FLOOR
                        )

        # OmniGibson's default compute backend (gm.USE_NUMPY_CONTROLLER_BACKEND, True by default) expects
        # plain numpy-convertible (i.e. CPU) tensors from controllers; default to CPU unless the caller
        # explicitly requested a device (e.g. gm.USE_NUMPY_CONTROLLER_BACKEND=False + an explicit cuda
        # device for a torch-backed controller pipeline).
        self._model = builder.finalize(device=self._sim_context.device or "cpu")
        if self._model.device.is_cpu and wp.is_cuda_available() and not self._warned_cpu_device:
            # MuJoCo-Warp on CPU is ~100x slower (seconds per step on a house) and disables the step graph
            self._warned_cpu_device = True
            log.warning(
                "Newton is simulating on CPU although CUDA is available; pass device='cuda' to og.launch() / "
                "the env config to run it on the GPU."
            )

        self._balance_body_inertia()

        # add_usd()'s default up-axis handling applies a spurious rotation to each object's root
        # joint (found empirically: a 90-degree-class rotation even though the source stage and the
        # builder both agree on Z-up) -- neither the default nor apply_up_axis_from_stage=True
        # produced a correct value for it, and this corrupts BOTH the position and orientation
        # components (not just orientation). Confirmed to affect not just FREE joints (a
        # floating-base object's synthetic root joint) but also FIXED joints that are themselves an
        # object's root but have downstream articulation (e.g. a door: a fixed-base object whose
        # base_link is welded to the world via a FIXED joint, but which has a hinge-jointed door
        # panel as a child) -- for a door, add_usd() imported a joint_X_p representing a genuinely
        # different rotation (~120 degrees about a diagonal axis) than the door's actual authored
        # placement (~90 degrees about world Z), visibly rendering the door on its side.
        #
        # Rather than trust the importer's axis-conversion math, overwrite each affected root
        # joint's initial pose directly from raw USD ground truth -- exactly what
        # set_world_poses/set_position_orientation would have placed it at pre-play anyway. Uses
        # raw_world_transform() (manual per-ancestor local-transform composition -- no
        # UsdGeom.XformCache "world" resolution, no Fabric/live-simulation involvement at all)
        # rather than get_world_pose(): confirmed empirically that get_world_pose() and even
        # XformCache.GetLocalToWorldTransform() can, in some calling contexts, read back a live
        # Newton-simulated value instead of the raw authored one for a body affected by this exact
        # bug -- circular, since that live value is what we're trying to correct in the first place.
        # physics_backends/newton_visuals.py's own mesh-to-body baking uses this same
        # raw_world_transform() ground truth, so once the joint anchor here is corrected, the live
        # simulated body pose used at render time matches what visual shapes were baked against.
        joint_q = _wp_to_torch(self._model.joint_q)
        joint_q_start = self._model.joint_q_start.numpy()
        joint_X_p = _wp_to_torch(self._model.joint_X_p)
        joint_type = self._model.joint_type.numpy()
        joint_parent = self._model.joint_parent.numpy()
        joint_child = self._model.joint_child.numpy()
        body_idx_to_path = {idx: path for path, idx in path_body_map.items()}
        JOINT_FREE = int(newton.JointType.FREE)

        def _raw_root_pose(body_path):
            prim = self.sim.stage.GetPrimAtPath(body_path)
            gf_xf = Gf.Transform(raw_world_transform(prim))
            t = gf_xf.GetTranslation()
            q = gf_xf.GetRotation().GetQuat()
            position = th.tensor([t[0], t[1], t[2]], dtype=th.float32)
            orientation = th.tensor([*q.GetImaginary(), q.GetReal()], dtype=th.float32)
            return position, orientation

        for j in range(self._model.joint_count):
            if int(joint_parent[j]) != -1:
                continue
            body_path = body_idx_to_path.get(int(joint_child[j]))
            if joint_type[j] == JOINT_FREE:
                q_start = int(joint_q_start[j])
                if body_path is not None:
                    position, orientation = _raw_root_pose(body_path)
                    joint_q[q_start : q_start + 3] = position.to(device=joint_q.device, dtype=joint_q.dtype)
                    joint_q[q_start + 3 : q_start + 7] = orientation.to(device=joint_q.device, dtype=joint_q.dtype)
                else:
                    joint_q[q_start + 3 : q_start + 7] = th.tensor(
                        [0.0, 0.0, 0.0, 1.0], dtype=joint_q.dtype, device=joint_q.device
                    )
            elif body_path is not None:
                position, orientation = _raw_root_pose(body_path)
                joint_X_p[j, :3] = position.to(device=joint_X_p.device, dtype=joint_X_p.dtype)
                joint_X_p[j, 3:7] = orientation.to(device=joint_X_p.device, dtype=joint_X_p.dtype)

        # Some robot assets (e.g. R1Pro's holonomic base) decompose a multi-DOF joint into a serial
        # chain of single-DOF joints connected by massless intermediate bodies (MassAPI applied with
        # mass=0, used purely as a kinematic pass-through -- confirmed via mass/inertia dump: exactly
        # 0 for these, unlike every real link). Newton's own inertia sanitization already corrects
        # malformed (e.g. non-positive-definite) inertia tensors, but doesn't floor genuinely-zero
        # mass -- harmless for a body reached only via FIXED joints (MuJoCo's converter treats these
        # as non-dynamic, e.g. Fetch's massless eef_link/eyes/laser_link work fine), but MuJoCo's
        # conversion pass requires every body reached via an actively-moving (non-FIXED) joint to have
        # mass/inertia above mjMINVAL, and otherwise silently fails to build a solver at all (caught
        # below as "no joints yet", which is also the legitimate, different case of an empty warmup
        # model -- this loop is the fix for this other, real case). Found via R1Pro/A1/Franka all
        # hitting the same "ControllableObjectViewAPI ... got []" symptom (no solver -> no
        # articulation view at all), traced to this exact mjMINVAL failure in each case.
        body_mass = _wp_to_torch(self._model.body_mass)
        body_inertia = _wp_to_torch(self._model.body_inertia)
        body_inv_mass = _wp_to_torch(self._model.body_inv_mass)
        body_inv_inertia = _wp_to_torch(self._model.body_inv_inertia)
        JOINT_FIXED = int(newton.JointType.FIXED)
        MIN_MASS = 1e-4
        MIN_INERTIA = 1e-8
        for j in range(self._model.joint_count):
            if joint_type[j] == JOINT_FIXED:
                continue
            body_idx = int(joint_child[j])
            if body_mass[body_idx].item() <= 0:
                body_mass[body_idx] = MIN_MASS
                body_inertia[body_idx] = th.eye(3, dtype=body_inertia.dtype, device=body_inertia.device) * MIN_INERTIA
                body_inv_mass[body_idx] = 1.0 / MIN_MASS
                body_inv_inertia[body_idx] = th.eye(3, dtype=body_inv_inertia.dtype, device=body_inv_inertia.device) * (
                    1.0 / MIN_INERTIA
                )

        # Some BEHAVIOR-1K robot assets author a default/reset joint position that falls OUTSIDE that
        # same joint's own authored limit range (found empirically: Fetch's head_tilt_joint rests at
        # -0.9412 rad, but its own limit_lower is -0.76 rad -- a real, if minor, asset inconsistency).
        # PhysX enforces joint limits as a near-exact, unyielding constraint, so this silently just
        # holds the joint pinned at its limit boundary with no ill effect there. MuJoCo/Newton instead
        # models limits as a soft spring (joint_limit_ke/kd) that continuously fights a position stuck
        # in permanent violation -- this alone (independent of any drive/actuator gains, reproduced
        # with every joint's target_ke/kd forced to exactly zero) destabilizes the whole coupled
        # articulation within ~100 steps. Clamp the initial position into the joint's own valid range
        # here, matching what PhysX's hard limit enforcement effectively already does to it at rest.
        joint_limit_lower = _wp_to_torch(self._model.joint_limit_lower)
        joint_limit_upper = _wp_to_torch(self._model.joint_limit_upper)
        joint_qd_start_np = self._model.joint_qd_start.numpy()
        JOINT_BALL = int(newton.JointType.BALL)
        for j in range(self._model.joint_count):
            if joint_type[j] in (JOINT_FREE, JOINT_BALL):
                continue
            q_start = int(joint_q_start[j])
            qd_start = int(joint_qd_start_np[j])
            n_dof = (
                self._model.joint_dof_count if j == self._model.joint_count - 1 else int(joint_qd_start_np[j + 1])
            ) - qd_start
            for k in range(n_dof):
                lower = joint_limit_lower[qd_start + k]
                upper = joint_limit_upper[qd_start + k]
                joint_q[q_start + k] = joint_q[q_start + k].clamp(min=lower.item(), max=upper.item())

        # joint_limit_ke/kd (10000/10) are Newton importer defaults, uniform across every joint --
        # USD's UsdPhysics limit schema has no per-joint stiffness/damping equivalent (PhysX enforces
        # limits as a near-exact constraint, not a spring), so there's no authored ground truth to read
        # here the way there was for target_ke/kd above. For a light, low-torque joint like
        # head_tilt_joint (0.68 N*m effort limit), this default implies a limit-violation force at even
        # a modest ~0.05 rad overshoot thousands of times its own actuation capability -- found
        # empirically to still destabilize the whole coupled articulation within ~100 steps even after
        # the initial-position clamp above (dynamic drift back toward the boundary re-triggers it).
        # Cap limit_ke (scaling kd by the same ratio, to preserve its damping character) so a modest
        # overshoot produces at most ~10x the joint's own effort_limit -- generous enough to still act
        # as a firm stop, but not so stiff it destabilizes MuJoCo's solver for light/low-effort joints.
        # Joints with large effort limits relative to their mass (e.g. the arm) are unaffected, since
        # the cap only binds when the default would imply an ~unbounded-looking force by comparison.
        CHARACTERISTIC_VIOLATION_RAD = 0.05
        LIMIT_FORCE_SAFETY_FACTOR = 10.0
        joint_limit_ke = _wp_to_torch(self._model.joint_limit_ke)
        joint_limit_kd = _wp_to_torch(self._model.joint_limit_kd)
        joint_effort_limit = _wp_to_torch(self._model.joint_effort_limit)
        for j in range(self._model.joint_count):
            if joint_type[j] in (JOINT_FREE, JOINT_BALL):
                continue
            qd_start = int(joint_qd_start_np[j])
            n_dof = (
                self._model.joint_dof_count if j == self._model.joint_count - 1 else int(joint_qd_start_np[j + 1])
            ) - qd_start
            for k in range(n_dof):
                idx = qd_start + k
                effort_limit = joint_effort_limit[idx].item()
                ke = joint_limit_ke[idx].item()
                if not (0 < effort_limit < float("inf")) or ke <= 0:
                    continue
                capped_ke = min(ke, LIMIT_FORCE_SAFETY_FACTOR * effort_limit / CHARACTERISTIC_VIOLATION_RAD)
                if capped_ke < ke:
                    ratio = capped_ke / ke
                    joint_limit_ke[idx] = capped_ke
                    joint_limit_kd[idx] = joint_limit_kd[idx] * ratio

        # model.joint_target_q/qd (QD-space sized -- NOT the same array as joint_q, despite the
        # confusingly similar name) are the *defaults* model.control() clones into a fresh Control.
        # These default to all-zero, which for a POSITION-mode actuator means "servo hard toward
        # q=0" -- found empirically to fling an idle robot's arm from its actual (non-zero) rest pose
        # toward zero at max effort, destabilizing the whole articulation within ~20-30 steps even
        # with no commanded action at all. Seed them from the (already free-joint-corrected) initial
        # joint_q instead, so an idle robot holds its actual starting pose by default. Free/ball joint
        # DOFs are skipped (no scalar q<->qd correspondence, and their target mode is NONE anyway --
        # the mobile base isn't exposed as a controllable joint, see NewtonArticulationHandle).
        joint_qd_start = joint_qd_start_np
        default_target_q = th.zeros_like(_wp_to_torch(self._model.joint_target_q))
        for j in range(self._model.joint_count):
            if joint_type[j] in (JOINT_FREE, JOINT_BALL):
                continue
            q_start = int(joint_q_start[j])
            qd_start = int(joint_qd_start[j])
            n_dof = (
                self._model.joint_dof_count if j == self._model.joint_count - 1 else int(joint_qd_start[j + 1])
            ) - qd_start
            default_target_q[qd_start : qd_start + n_dof] = joint_q[q_start : q_start + n_dof]
        _wp_to_torch(self._model.joint_target_q).copy_(default_target_q)

        # SolverMuJoCo fills this per step only when the state carries it -- see get_measured_joint_efforts()
        if self._model.joint_dof_count > 0:
            self._model.request_state_attributes("mujoco:qfrc_actuator")
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        # model.contacts() lazily builds a default CollisionPipeline sized via
        # _estimate_rigid_contact_max() (from the model's *initial* contact count) unless
        # model.rigid_contact_max is already set -- match SolverMuJoCo's own nconmax (see solver
        # construction below) so MuJoCo doesn't reject this shared Contacts buffer as too small
        # ("MuJoCo naconmax (N) exceeds contacts.rigid_contact_max (M)").
        self._model.rigid_contact_max = self._RIGID_CONTACT_MAX
        self._contacts = self._model.contacts()

        if self._model.joint_count > 0:
            # model.state() does NOT inherit joint_q/joint_qd from the model -- each state starts with
            # its own (zero-initialized) copy. _sync_fk() (called after every set_joint_positions/
            # velocities) reads FROM the state's own joint_q, so it must be seeded from the model's
            # values now, or the free/floating joint's identity fix above would only ever apply to
            # model.joint_q and never actually take effect.
            _wp_to_torch(self._state_0.joint_q).copy_(joint_q)
            _wp_to_torch(self._state_1.joint_q).copy_(joint_q)
            _wp_to_torch(self._state_0.joint_qd).copy_(_wp_to_torch(self._model.joint_qd))
            _wp_to_torch(self._state_1.joint_qd).copy_(_wp_to_torch(self._model.joint_qd))
            newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_0)
            newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_1)

        # Build the shape BVH used by raycast_closest()/raycast_all() -- finalize() already builds one,
        # but scoped to VISIBLE shapes by default, which would silently miss invisible collision-only
        # meshes (the visuals/collisions split used throughout this dataset). Already matches the
        # current (just-computed) body poses, so it isn't dirty yet -- see _bvh_dirty's own docstring.
        if self._model.shape_count > 0:
            self._model.bvh_build_shapes(self._state_0, shape_flags=newton.ShapeFlags.COLLIDE_SHAPES)
        self._bvh_dirty = False

        self._mpm_in_place = False
        has_cloth = bool(path_cloth_particle_map)
        # A system with 0 particles (freshly created, or just cleared) still gets an entry in
        # path_fluid_particle_map -- SolverImplicitMPM requires at least one particle to construct, so
        # only count systems that actually contributed particles to the model.
        has_fluid = any(end > start for start, end in path_fluid_particle_map.values())
        has_rigid = self._model.joint_count > 0
        self._rigid_collision_pipeline = None
        if has_cloth and has_fluid:
            # Not yet supported simultaneously (would need 3-way SolverCoupledProxy MuJoCo+XPBD+MPM
            # entries) -- fall through to the fluid path below, which will silently leave the cloth
            # particles unsimulated (still present in the model, just not driven by any solver). Narrow
            # enough a scenario (a scene needing both a cloth object AND a live fluid system) that this
            # is being left as a known, documented gap rather than blocking either feature individually.
            log.warning(
                "Newton backend does not yet support cloth and fluid particles simultaneously in the "
                "same scene -- only the fluid system(s) will be simulated this rebuild."
            )

        if has_fluid and has_rigid:
            # Fluid + rigid (articulated) content exist -- SolverMuJoCo alone has no particle physics,
            # so couple it with SolverImplicitMPM via the experimental SolverCoupledProxy, matching
            # newton's own example_mujoco_mpm_coupled_solver.py exactly (in-place stepping, MPM handles
            # collider contact internally so its own Proxy explicitly disables the shared collision
            # pipeline for it -- unlike XPBD below, which needs one).
            from newton.solvers.experimental.coupled import SolverCoupledProxy
            from newton.solvers import SolverImplicitMPM

            rigid_body_indices = list(range(self._model.body_count))
            # MPM collides KINEMATIC bodies (e.g. kinematic_only furniture) as immovable, but a lagged proxy then
            # rewinds their velocity by coupling_force / mass every step -- nothing integrates it back, so MPM saw a
            # welded wall moving at ~10 m/s and flung the water. A staggered proxy skips that rewind.
            body_flags = self._model.body_flags.numpy()
            kinematic = int(newton.BodyFlags.KINEMATIC)
            kinematic_body_indices = [i for i in rigid_body_indices if body_flags[i] & kinematic]
            dynamic_body_indices = [i for i in rigid_body_indices if not body_flags[i] & kinematic]
            # One shared factory: SolverCoupledProxy requires every mjc->mpm proxy to use the same one
            no_proxy_collisions = lambda _model: None
            mpm_config = self._make_mpm_config()
            self._solver = SolverCoupledProxy(
                model=self._model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mpm",
                        solver=lambda v: SolverImplicitMPM(model=v, config=mpm_config),
                        particles=list(range(self._model.particle_count)),
                        in_place=True,
                    ),
                    SolverCoupledProxy.Entry(
                        name="mjc",
                        solver=lambda v: newton.solvers.SolverMuJoCo(model=v, **self._mujoco_solver_kwargs()),
                        bodies=rigid_body_indices,
                        joints=list(range(self._model.joint_count)),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mjc",
                            destination="mpm",
                            bodies=bodies,
                            mass_scale=1.0,
                            mode=mode,
                            collision_pipeline=no_proxy_collisions,
                        )
                        for bodies, mode in ((dynamic_body_indices, "lagged"), (kinematic_body_indices, "staggered"))
                        if bodies
                    ],
                    iterations=1,
                ),
            )
            self._collision_pipeline = newton.CollisionPipeline(self._model, soft_contact_max=0)
            self._contacts = self._collision_pipeline.contacts()
            self._mpm_in_place = True
        elif has_fluid:
            # Fluid but no rigid bodies -- a bare SolverImplicitMPM, in-place, no external
            # CollisionPipeline needed at all (validated standalone: MPM handles ground/collider
            # contact internally via its own grid-based representation).
            mpm_config = self._make_mpm_config()
            self._solver = SolverImplicitMPM(model=self._model, config=mpm_config)
            self._collision_pipeline = None
            self._mpm_in_place = True
        elif self._model.particle_count > 0 and has_rigid:
            # Cloth/particle AND rigid (articulated) content exist -- SolverMuJoCo alone has no
            # particle physics and SolverXPBD alone has no articulated-rigid physics, so couple them via
            # the experimental SolverCoupledProxy (see NewtonBackend module docstring / this session's
            # plan doc for the validation done before wiring this in). Deliberately NOT the default
            # path: scenes with no cloth/particles keep using the plain, already-extensively-validated
            # SolverMuJoCo-only path below unchanged.
            from newton.solvers.experimental.coupled import SolverCoupledProxy

            self._model.soft_contact_ke = 1.0e2
            self._model.soft_contact_kd = 1.0
            rigid_body_indices = list(range(self._model.body_count))
            self._solver = SolverCoupledProxy(
                model=self._model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="xpbd",
                        solver=lambda v: newton.solvers.SolverXPBD(model=v, iterations=40),
                        particles=list(range(self._model.particle_count)),
                    ),
                    SolverCoupledProxy.Entry(
                        name="mjc",
                        solver=lambda v: newton.solvers.SolverMuJoCo(model=v, **self._mujoco_solver_kwargs()),
                        bodies=rigid_body_indices,
                        joints=list(range(self._model.joint_count)),
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mjc", destination="xpbd", bodies=rigid_body_indices, mass_scale=1.0
                        )
                    ],
                    iterations=1,
                ),
            )
            self._collision_pipeline = newton.CollisionPipeline(self._model)
            self._contacts = self._collision_pipeline.contacts()
            self._mpm_in_place = False
        elif self._model.particle_count > 0:
            # Particles/cloth but no rigid bodies to couple with (e.g. a lone cloth object) -- a bare
            # SolverXPBD, no SolverCoupledProxy wrapper needed (found empirically: wrapping a
            # single-entry, zero-proxy SolverCoupledProxy around just SolverXPBD produced zero net force
            # on the particles -- gravity/dynamics never got applied -- unlike a plain SolverXPBD).
            self._model.soft_contact_ke = 1.0e2
            self._model.soft_contact_kd = 1.0
            self._solver = newton.solvers.SolverXPBD(model=self._model, iterations=40)
            self._collision_pipeline = newton.CollisionPipeline(self._model)
            self._contacts = self._collision_pipeline.contacts()
            self._mpm_in_place = False
        elif self._model.joint_count > 0:
            self._collision_pipeline = None
            try:
                self._solver = newton.solvers.SolverMuJoCo(self._model, **self._mujoco_solver_kwargs())
                # Contacts from Newton's own collision pipeline, as IsaacLab's manipulation tasks (lift,
                # stack) run it, instead of MuJoCo's one-point-per-convex-pair collision
                self._rigid_collision_pipeline = newton.CollisionPipeline(
                    self._model,
                    broad_phase="explicit",
                    rigid_contact_max=max(self._RIGID_CONTACT_MAX, self._solver.get_max_contact_count()),
                    soft_contact_max=0,
                )
                self._contacts = self._rigid_collision_pipeline.contacts()
            except ValueError as e:
                # Every ValueError lands here, not just the expected "no joints in the model yet" one
                # during early startup -- and the consequence is severe (no solver means the scene
                # runs with NO physics at all), so anything else is surfaced loudly and in full.
                if "joint" in str(e).lower():
                    log.warning(f"No Newton solver created (model has no joints yet): {e}")
                else:
                    log.error(
                        f"Newton solver could NOT be created -- physics will not run until the next "
                        f"successful rebuild: {e}"
                    )
                self._solver = None
        else:
            self._collision_pipeline = None
            self._solver = None

        # Must run after the solver exists (it edits the solver's own compiled mjc model), and applies
        # to every solver branch above -- see the method's own docstring for why the mu drop alone
        # isn't enough.
        self._elevate_low_friction_geom_priority(low_friction_body_idx)
        self._configure_finger_contacts(finger_body_idx)
        # Re-apply any attachment constraints that were live before this rebuild onto the fresh pool.
        self._sync_attachment_slots()
        self._fix_cross_object_constraint_frames()

        # Bind every already-registered articulation/rigid-body handle to this freshly-built model now,
        # rather than waiting for each owning object's own (lazy, one-at-a-time) Simulator.update_handles()
        # call. PhysX's stage-wide physics view creation binds every object's articulation atomically the
        # moment play() starts, regardless of each object's individual Python-side initialization state --
        # our handles only need the model (already built above), not anything from the owning object's own
        # _initialize(), so there's no reason to wait. Found empirically to matter for multi-robot scenes:
        # a robot's own _initialize() internally calls reset(), which queries the pattern-matched batch
        # view spanning ALL robots (e.g. "controllable__fetch__*") via ControllableObjectViewAPI -- if a
        # sibling robot loaded earlier in the same object-initialization queue hasn't had its own handle
        # bound yet, that batch view either errors (indexing a None _global_q_idx) or silently omits it,
        # tripping ControllableObjectViewAPI.initialize_view()'s "every controllable object is represented"
        # assertion. Binding all handles immediately here avoids the ordering dependency entirely.
        for handle in self._articulation_handles.values():
            handle.initialize(None)
        for handle in self._rigid_body_handles.values():
            handle.initialize(None)

        # Existing handles (constructed at object-load time, before the model existed) get re-bound
        # to the freshly-built model by Simulator.update_handles() calling .initialize() on each again
        # right after play(); nothing further to do here.

        # A dedicated Contacts buffer for rigid contact reporting, separate from self._contacts (which the
        # solver consumes). solver.update_contacts() fills it from MuJoCo's own post-solve contact list, so it
        # never needs a collision pipeline of its own.
        self._contact_report_contacts = None
        if has_rigid:
            self._contact_report_contacts = newton.Contacts(
                self._RIGID_CONTACT_MAX,
                0,
                device=self._model.device,
                requested_attributes={"force"},
            )
        self._contacts_dirty = True

        self._known_object_paths = frozenset(obj.prim_path for obj in objects)
        self._known_macro_particle_paths = frozenset(macro_particle_paths)
        self._model_generation += 1
        self._restore_state(captured_state)

    # Rough starting-point stiffness values for cloth -- NOT a principled port of PhysX's spring-based
    # spring{Bend,Shear,Stretch}Stiffness/springDamping values (no 1:1 formula exists between PhysX's
    # spring model and Newton's constraint model); expect this to need further empirical retuning
    # against real cloth assets. tri_ke/tri_ka/tri_kd are dead parameters for SolverXPBD specifically
    # (verified against newton's own xpbd/kernels.py -- never referenced there; they matter for
    # SolverVBD/SolverStyle3D's true FEM triangle-stress model instead) -- SolverXPBD's actual
    # elasticity comes entirely from spring_ke/spring_kd (a PBD distance-constraint solver over
    # add_springs=True's edges) plus edge_ke/edge_kd for bending. spring_ke/kd here are ~100x lower
    # than a first attempt (1e2/1.0) that exploded catastrophically on a real, finely-remeshed
    # (~15k-particle) asset -- short high-resolution springs at that original stiffness, combined with
    # the correspondingly tiny per-particle mass add_cloth_mesh() derives from density*area/3, is a
    # classic explicit spring-mass instability (effective stiffness-to-mass ratio implies a much
    # smaller stable timestep than at the coarse resolution this was first tuned against).
    _CLOTH_DEFAULT_TRI_KE = 1.0e2
    _CLOTH_DEFAULT_TRI_KA = 1.0e2
    _CLOTH_DEFAULT_TRI_KD = 1.0e-1
    _CLOTH_DEFAULT_EDGE_KE = 1.0
    _CLOTH_DEFAULT_EDGE_KD = 1.0e-2
    _CLOTH_DEFAULT_SPRING_KE = 1.0
    _CLOTH_DEFAULT_SPRING_KD = 1.0e-2
    _CLOTH_DEFAULT_PARTICLE_RADIUS = 0.01
    _CLOTH_DEFAULT_DENSITY = 200.0
    # Number of XPBD sub-steps per OmniGibson physics step (see step_physics_once()) -- matches
    # newton's own example_mujoco_xpbd_coupled_solver.py's substep count.
    _PARTICLE_SUBSTEPS = 32

    # Fluid (SolverImplicitMPM) defaults: water as in newton's example_mpm_water_dam_break.py -- frictionless, with
    # newton's own (incompressible) yield defaults. Don't zero mpm:yield_pressure: that removes all resistance to
    # compression and water lands as a motionless sheet. MPM gives each particle a (2r)^3 volume, so the mass below
    # is water's density. MicroPhysicalParticleSystem doesn't pass per-particle mass/radius through
    # generate_particles() (PhysX derives them from its own particle_contact_offset), so these are fixed constants.
    _FLUID_DEFAULT_PARTICLE_RADIUS = 0.01
    _FLUID_DEFAULT_PARTICLE_MASS = 1000.0 * (2 * _FLUID_DEFAULT_PARTICLE_RADIUS) ** 3
    _FLUID_MATERIAL_DEFAULTS = {
        "mpm:friction": 0.0,
        "mpm:tensile_yield_ratio": 1.0,
        "mpm:viscosity": 0.0,
    }
    _FLUID_VOXEL_SIZE = 0.05

    # Sized generously for a populated household scene, not tuned per-scene. Must match (or exceed)
    # every SolverMuJoCo construction's own nconmax below -- MuJoCo validates that any Contacts buffer
    # handed to it (including the default one from the generic self._model.contacts() call in
    # _rebuild_impl(), and the explicit external CollisionPipeline used by the fluid/cloth-coupled
    # paths) is at least as large as its own nconmax, and raises otherwise ("MuJoCo naconmax (N)
    # exceeds contacts.rigid_contact_max (M)").
    _RIGID_CONTACT_MAX = 250_000

    def _mujoco_solver_kwargs(self):
        """
        SolverMuJoCo settings shared by every solver branch in _rebuild_impl(), so rigid bodies behave the same
        whether or not the scene also holds cloth or fluid. IsaacLab's MJWarpSolverCfg defaults
        (isaaclab_newton/physics/mjwarp_manager_cfg.py), with the implicitfast integrator its tasks all select.
        Contacts come from Newton's CollisionPipeline (use_mujoco_contacts=False), so MuJoCo's own collision
        options (ccd_iterations, enable_multiccd) don't apply.

        nconmax is otherwise auto-estimated from the *initial* contact count -- too small for a populated
        household scene. An undersized buffer doesn't fail gracefully: mujoco_warp's EPA/CCD scratch arrays get
        indexed past their capacity, corrupting heap memory and segfaulting later at an unrelated location
        (confirmed with wp.config.mode = "debug", which turns it into an array bounds assertion).
        """
        return dict(
            solver="newton",
            integrator="implicitfast",
            iterations=100,
            ls_iterations=50,
            cone="pyramidal",
            impratio=1.0,
            tolerance=1e-6,
            use_mujoco_contacts=False,
            njmax=32_768,
            nconmax=self._RIGID_CONTACT_MAX,
        )

    def _make_mpm_config(self):
        """SolverImplicitMPM.Config, validated standalone against a water-like blob (settles/spreads
        flat under gravity, no explosion) -- matches newton's own coupled-MPM example's tuning."""
        from newton.solvers import SolverImplicitMPM

        config = SolverImplicitMPM.Config()
        config.voxel_size = self._FLUID_VOXEL_SIZE
        config.grid_type = "fixed"
        config.grid_padding = 50
        config.max_active_cell_count = 1 << 15
        config.strain_basis = "P0"
        config.max_iterations = 50
        config.critical_fraction = 0.0
        return config

    def _add_cloth_object(self, builder, obj, path_cloth_particle_map):
        """
        Imports one cloth object's mesh into `builder` via `add_cloth_mesh()`, and records its particle
        index range (keyed by the mesh prim's own path -- matching `ClothPrim.prim_path`, since
        `ClothPrim` is a `GeomPrim` wrapping the promoted mesh itself, not the parent entity) so
        `get/set_cloth_particle_positions/velocities` can slice `state.particle_q/qd` for it later.
        """
        import warp as wp

        mesh_prim = self.sim.stage.GetPrimAtPath(obj.root_link.prim_path)
        points = mesh_prim.GetAttribute("points").Get()
        face_vertex_counts = mesh_prim.GetAttribute("faceVertexCounts").Get()
        face_vertex_indices = mesh_prim.GetAttribute("faceVertexIndices").Get()

        # Fan-triangulate arbitrary polygon faces into a flat triangle index list.
        indices = []
        offset = 0
        for count in face_vertex_counts:
            face = face_vertex_indices[offset : offset + count]
            for i in range(1, count - 1):
                indices.extend([face[0], face[i], face[i + 1]])
            offset += count

        matrix = lazy.pxr.UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        transform = lazy.pxr.Gf.Transform(matrix)
        pos = transform.GetTranslation()
        rot = transform.GetRotation().GetQuat()
        # add_cloth_mesh() only accepts a single uniform scale -- bake any (possibly non-uniform)
        # object scale directly into the vertices instead, and pass scale=1.0 below.
        scale = transform.GetScale()
        vertices = [(p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]) for p in points]

        particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=wp.vec3(pos[0], pos[1], pos[2]),
            rot=wp.quat(rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2], rot.GetReal()),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=self._CLOTH_DEFAULT_DENSITY,
            tri_ke=self._CLOTH_DEFAULT_TRI_KE,
            tri_ka=self._CLOTH_DEFAULT_TRI_KA,
            tri_kd=self._CLOTH_DEFAULT_TRI_KD,
            edge_ke=self._CLOTH_DEFAULT_EDGE_KE,
            edge_kd=self._CLOTH_DEFAULT_EDGE_KD,
            add_springs=True,
            spring_ke=self._CLOTH_DEFAULT_SPRING_KE,
            spring_kd=self._CLOTH_DEFAULT_SPRING_KD,
            particle_radius=self._CLOTH_DEFAULT_PARTICLE_RADIUS,
        )
        particle_end = builder.particle_count
        path_cloth_particle_map[obj.root_link.prim_path] = (particle_start, particle_end)

    def _capture_state(self):
        """
        Snapshots every currently-bound handle's joint/world state, keyed by prim path, so a mid-episode
        _rebuild() (triggered by a dynamic object add/remove) can restore it afterward instead of
        silently resetting every object back to its authored USD defaults. Returns None on the very
        first build (self._model is None yet -- nothing to preserve).
        """
        if self._model is None:
            return None
        captured = {"articulations": {}, "rigid_bodies": {}, "cloth": {}, "macro_particles": {}}
        # Macro particles have no per-prim handles; snapshot their bodies directly by path
        if self._known_macro_particle_paths and self._state_0 is not None and self._state_0.body_q is not None:
            body_q = _wp_to_torch(self._state_0.body_q)
            body_qd = _wp_to_torch(self._state_0.body_qd)
            for path in self._known_macro_particle_paths:
                idx = self._path_body_map.get(path)
                if idx is not None:
                    captured["macro_particles"][path] = (body_q[idx].clone(), body_qd[idx].clone())
        for path in self._path_cloth_particle_map:
            captured["cloth"][path] = {
                "particle_q": self.get_cloth_particle_positions(path).clone(),
                "particle_qd": self.get_cloth_particle_velocities(path).clone(),
            }
        for path, handle in self._articulation_handles.items():
            if not handle.is_physics_handle_valid():
                continue
            root_body_idx = self._path_body_map.get(handle._root_link_prim_path)
            world_pos, world_quat = (None, None)
            if root_body_idx is not None:
                body_q = _wp_to_torch(self._state_0.body_q)
                world_pos = body_q[root_body_idx, :3].clone()
                world_quat = _xyzw_to_wxyz(body_q[root_body_idx, 3:7]).clone()
            # Drive gains must be preserved too. add_usd() re-imports joint_target_ke/kd from the
            # authored USD DriveAPI on every rebuild, and _rebuild_impl() deliberately forces those
            # back to the authored values (which are 0 for every OmniGibson joint -- see the long
            # comment there). The *real* gains are the ones a robot's controllers push at runtime via
            # update_controller_mode() -> set_gains(), and nothing re-pushes them after a rebuild:
            # Simulator.play() only re-runs update_controller_mode() when `was_stopped` AND the robot
            # is already `initialized`. A rebuild therefore silently zeroed every controller gain,
            # leaving each joint in POSITION control with kp=kd=0 -- i.e. no servo at all, so a
            # gravity-loaded joint just sagged to its limit stop. Measured on a fresh env before this
            # fix: r1pro's torso_joint1 reported kp=0.0/kd=0.0 despite its controller having set
            # 20000/2000, and the trunk drifted from its 1.025 rad command to its 1.833 rad stop.
            # (Rebuilds happen more often than play(): one is triggered from step_physics_once()
            # itself whenever _joints_dirty is set, e.g. an AttachedTo attach/detach.)
            gains_ke, gains_kd = (None, None)
            if handle.num_dof > 0:
                gains_ke, gains_kd = handle.get_gains()
                gains_ke, gains_kd = gains_ke.clone(), gains_kd.clone()
            captured["articulations"][path] = {
                "joint_q": handle.get_joint_positions().clone() if handle.num_dof > 0 else None,
                "joint_qd": handle.get_joint_velocities().clone() if handle.num_dof > 0 else None,
                "gains_ke": gains_ke,
                "gains_kd": gains_kd,
                "world_pos": world_pos,
                "world_quat": world_quat,
            }
        for path, handle in self._rigid_body_handles.items():
            if not handle.is_physics_handle_valid():
                continue
            pos, quat = handle.get_world_poses()
            captured["rigid_bodies"][path] = {
                "world_pos": pos.clone(),
                "world_quat": quat.clone(),
                "lin_vel": handle.get_linear_velocities().clone(),
                "ang_vel": handle.get_angular_velocities().clone(),
            }
        return captured

    def _restore_state(self, captured):
        """Writes a _capture_state() snapshot into the just-(re)built model, for every path that still exists."""
        if captured is None:
            return
        for path, data in captured["cloth"].items():
            if path not in self._path_cloth_particle_map:
                continue
            self.set_cloth_particle_positions(path, data["particle_q"])
            self.set_cloth_particle_velocities(path, data["particle_qd"])
        for path, data in captured["articulations"].items():
            handle = self._articulation_handles.get(path)
            if handle is None or not handle.is_physics_handle_valid():
                continue
            if data["joint_q"] is not None:
                handle.set_joint_positions(data["joint_q"])
                handle.set_joint_velocities(data["joint_qd"])
            # Re-push the pre-rebuild drive gains -- see _capture_state()'s own comment. Only restored
            # when the DOF count still matches: a rebuild that actually changed this articulation's
            # topology invalidates the old per-DOF vector, and in that case the authored defaults plus
            # the next update_controller_mode() are the honest fallback.
            if data.get("gains_ke") is not None and handle.num_dof == data["gains_ke"].shape[-1]:
                handle.set_gains(kps=data["gains_ke"], kds=data["gains_kd"])
            if data["world_pos"] is not None and self._path_body_map.get(handle._root_link_prim_path) is not None:
                handle.set_world_poses(data["world_pos"], data["world_quat"])
        particle_paths = [p for p in captured.get("macro_particles", {}) if p in self._path_body_map]
        if particle_paths:
            view = _NewtonRigidBodyBatchView(self, [self._path_body_map[p] for p in particle_paths], particle_paths)
            view.set_transforms(th.stack([captured["macro_particles"][p][0] for p in particle_paths]))
            view.set_velocities(th.stack([captured["macro_particles"][p][1] for p in particle_paths]))
        for path, data in captured["rigid_bodies"].items():
            handle = self._rigid_body_handles.get(path)
            if handle is None or not handle.is_physics_handle_valid():
                continue
            handle.set_world_poses(data["world_pos"], data["world_quat"])
            handle.set_linear_velocities(data["lin_vel"])
            handle.set_angular_velocities(data["ang_vel"])

    # ---- Cloth particle state I/O ----
    # particle_q/particle_qd on state_0 are already world-frame (add_cloth_mesh's pos/rot/scale args
    # position the mesh in world space at build time), unlike PhysX's local-frame `points` USD attr --
    # no local<->world conversion needed here, matching the WORLD-frame contract in PhysicsBackend.

    def _fallback_cloth_particle_positions(self, prim_path, idxs=None):
        # Not yet bound into a built Newton model (e.g. called from ClothPrim._post_load(), before the
        # first play()/_rebuild() ever runs) -- fall back to the plain USD-authored local-frame `points`
        # attribute + the prim's own world transform, mirroring how NewtonRigidBodyHandle.get_world_poses()
        # falls back to plain USD while unbound/stopped.
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        points = prim.GetAttribute("points").Get()
        p_local = vtarray_to_torch(points)
        p_local = p_local[idxs] if idxs is not None else p_local
        matrix = lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        linear = th.tensor([[matrix[i][j] for j in range(3)] for i in range(3)], dtype=th.float32)
        translation = th.tensor([matrix[3][j] for j in range(3)], dtype=th.float32)
        return p_local @ linear + translation

    def get_cloth_particle_positions(self, prim_path, idxs=None):
        if prim_path not in self._path_cloth_particle_map:
            return self._fallback_cloth_particle_positions(prim_path, idxs=idxs)
        start, end = self._path_cloth_particle_map[prim_path]
        particle_q = _wp_to_torch(self._state_0.particle_q)[start:end, :3]
        return particle_q[th.as_tensor(idxs, device=particle_q.device)] if idxs is not None else particle_q

    def set_cloth_particle_positions(self, prim_path, positions, idxs=None):
        if prim_path not in self._path_cloth_particle_map:
            # Not yet bound -- write straight to the USD points attribute (local frame), matching the
            # fallback read path above and PhysX's own always-USD-backed representation.
            prim = self.sim.stage.GetPrimAtPath(prim_path)
            matrix = lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
            linear = th.tensor([[matrix[i][j] for j in range(3)] for i in range(3)], dtype=th.float32)
            translation = th.tensor([matrix[3][j] for j in range(3)], dtype=th.float32)
            p_local = (
                th.as_tensor(positions, dtype=th.float32, device=translation.device) - translation
            ) @ th.linalg.inv(linear)
            if idxs is not None:
                p_local_full = vtarray_to_torch(prim.GetAttribute("points").Get())
                p_local_full[idxs] = p_local
                p_local = p_local_full
            prim.GetAttribute("points").Set(lazy.pxr.Vt.Vec3fArray(p_local.tolist()))
            return
        start, end = self._path_cloth_particle_map[prim_path]
        particle_q = _wp_to_torch(self._state_0.particle_q)
        idx = (
            th.arange(start, end, device=particle_q.device)[th.as_tensor(idxs, device=particle_q.device)]
            if idxs is not None
            else th.arange(start, end, device=particle_q.device)
        )
        particle_q[idx, :3] = th.as_tensor(positions, dtype=particle_q.dtype, device=particle_q.device).reshape(-1, 3)

    def get_cloth_particle_velocities(self, prim_path):
        if prim_path not in self._path_cloth_particle_map:
            return th.zeros(len(self.sim.stage.GetPrimAtPath(prim_path).GetAttribute("points").Get()), 3)
        start, end = self._path_cloth_particle_map[prim_path]
        return _wp_to_torch(self._state_0.particle_qd)[start:end, :3]

    def set_cloth_particle_velocities(self, prim_path, velocities):
        if prim_path not in self._path_cloth_particle_map:
            return
        start, end = self._path_cloth_particle_map[prim_path]
        particle_qd = _wp_to_torch(self._state_0.particle_qd)
        particle_qd[start:end, :3] = th.as_tensor(
            velocities, dtype=particle_qd.dtype, device=particle_qd.device
        ).reshape(-1, 3)

    def get_cloth_stiffness(self, prim_path):
        # No live per-object override tracked yet -- every cloth uses the same build-time defaults
        # (see _add_cloth_object). Report those; set_cloth_stiffness below can still adjust the live
        # model arrays even though this getter doesn't yet distinguish per-object customization.
        return {
            "bend": self._CLOTH_DEFAULT_EDGE_KE,
            "damping": self._CLOTH_DEFAULT_TRI_KD,
            "shear": self._CLOTH_DEFAULT_TRI_KA,
            "stretch": self._CLOTH_DEFAULT_TRI_KE,
        }

    def set_cloth_stiffness(self, prim_path, bend=None, damping=None, shear=None, stretch=None):
        # Cloth stiffness lives on per-TRIANGLE/per-EDGE model arrays (tri_ke/ka/kd, edge_ke/kd), not a
        # single per-object value -- finding the exact triangle/edge index range for one cloth object
        # (analogous to _path_cloth_particle_map, but for tri_count/edge_count) isn't tracked yet.
        # Deferred: not needed for the initial working end-to-end path (build-time defaults apply).
        pass

    # ---- Fluid/granular particle-system state I/O ----
    # A system mapped in self._path_fluid_particle_map lives in the model's state; otherwise (unbuilt, or just
    # resized) self._fluid_systems holds it, and every rebuild re-imports from there.
    # generate_particles()/remove_particles() are structural changes (particle count itself changes) and
    # always trigger an explicit _rebuild() -- unlike everything else in this file, there's no
    # dirty-check/deferred-rebuild path for these, since they're already-explicit, infrequent operations
    # (matching how before_play() itself unconditionally rebuilds).

    def _sync_fluid_system_from_model(self, system_name):
        """Write the live model's current state for `system_name` back into self._fluid_systems, if built."""
        if system_name not in self._path_fluid_particle_map:
            return
        start, end = self._path_fluid_particle_map[system_name]
        if end == start:
            # A 0-particle model leaves particle_q/particle_qd as None (Warp never allocates a
            # zero-size array), which happens whenever every fluid system in the scene is empty.
            self._fluid_systems[system_name] = {
                "pos": th.zeros((0, 3), dtype=th.float32, device=self.sim.device),
                "vel": th.zeros((0, 3), dtype=th.float32, device=self.sim.device),
            }
            return
        self._fluid_systems[system_name] = {
            "pos": _wp_to_torch(self._state_0.particle_q)[start:end, :3].clone(),
            "vel": _wp_to_torch(self._state_0.particle_qd)[start:end, :3].clone(),
        }

    def create_particle_system(self, system_name):
        # Idempotent -- safe to call defensively from generate_particles() without first checking
        # whether the system already exists (and without wiping out any particles it already has).
        if system_name not in self._fluid_systems:
            self._fluid_systems[system_name] = {
                "pos": th.zeros((0, 3), device=self.sim.device),
                "vel": th.zeros((0, 3), device=self.sim.device),
            }

    def generate_particles(self, system_name, positions, velocities=None):
        self._sync_fluid_system_from_model(system_name)
        # Callers may pass plain lists (device info already stripped, e.g. via .tolist()) or tensors on
        # any device -- always land on self.sim.device here rather than trusting the caller.
        positions = th.as_tensor(positions, dtype=th.float32, device=self.sim.device).reshape(-1, 3)
        velocities = (
            th.zeros_like(positions)
            if velocities is None
            else th.as_tensor(velocities, dtype=th.float32, device=self.sim.device).reshape(-1, 3)
        )
        system = self._fluid_systems[system_name]
        system["pos"] = th.cat([system["pos"], positions])
        system["vel"] = th.cat([system["vel"], velocities])
        # The live model no longer matches this system; unmap it so _rebuild_impl()'s sync can't clobber the change
        self._path_fluid_particle_map.pop(system_name, None)
        self._rebuild()

    def remove_particles(self, system_name, idxs):
        self._sync_fluid_system_from_model(system_name)
        system = self._fluid_systems[system_name]
        keep = th.ones(len(system["pos"]), dtype=th.bool, device=system["pos"].device)
        keep[th.as_tensor(idxs, dtype=th.long, device=system["pos"].device)] = False
        system["pos"] = system["pos"][keep]
        system["vel"] = system["vel"][keep]
        self._path_fluid_particle_map.pop(system_name, None)
        self._rebuild()

    def get_particle_positions(self, system_name):
        if system_name in self._path_fluid_particle_map:
            start, end = self._path_fluid_particle_map[system_name]
            # A 0-particle model leaves particle_q as None (Warp never allocates a zero-size array),
            # which happens whenever every fluid system in the scene is currently empty.
            if end == start:
                return th.zeros((0, 3), dtype=th.float32, device=self.sim.device)
            return _wp_to_torch(self._state_0.particle_q)[start:end, :3]
        return self._fluid_systems[system_name]["pos"]

    def set_particle_positions(self, system_name, positions, idxs=None):
        positions = th.as_tensor(positions, dtype=th.float32, device=self.sim.device).reshape(-1, 3)
        if system_name in self._path_fluid_particle_map:
            start, end = self._path_fluid_particle_map[system_name]
            if end == start:
                return
            particle_q = _wp_to_torch(self._state_0.particle_q)
            idx = (
                th.arange(start, end, device=particle_q.device)[idxs]
                if idxs is not None
                else th.arange(start, end, device=particle_q.device)
            )
            particle_q[idx, :3] = positions
        else:
            system = self._fluid_systems[system_name]
            if idxs is not None:
                system["pos"][idxs] = positions.to(system["pos"].device)
            else:
                system["pos"] = positions

    def get_particle_velocities(self, system_name):
        if system_name in self._path_fluid_particle_map:
            start, end = self._path_fluid_particle_map[system_name]
            if end == start:
                return th.zeros((0, 3), dtype=th.float32, device=self.sim.device)
            return _wp_to_torch(self._state_0.particle_qd)[start:end, :3]
        return self._fluid_systems[system_name]["vel"]

    def set_particle_velocities(self, system_name, velocities, idxs=None):
        velocities = th.as_tensor(velocities, dtype=th.float32, device=self.sim.device).reshape(-1, 3)
        if system_name in self._path_fluid_particle_map:
            start, end = self._path_fluid_particle_map[system_name]
            if end == start:
                return
            particle_qd = _wp_to_torch(self._state_0.particle_qd)
            idx = (
                th.arange(start, end, device=particle_qd.device)[idxs]
                if idxs is not None
                else th.arange(start, end, device=particle_qd.device)
            )
            particle_qd[idx, :3] = velocities
        else:
            system = self._fluid_systems[system_name]
            if idxs is not None:
                system["vel"][idxs] = velocities.to(system["vel"].device)
            else:
                system["vel"] = velocities

    # Surface reconstruction settings from newton's example_mpm_water_dam_break.py, with the voxel and kernel sized
    # to _FLUID_VOXEL_SIZE and the 2 cm particle spacing
    _FLUID_SURFACE_VOXEL_SIZE = 0.3 * _FLUID_VOXEL_SIZE
    _FLUID_SURFACE_KWARGS = dict(
        kernel_radius=max(6.0 * _FLUID_DEFAULT_PARTICLE_RADIUS, 1.5 * _FLUID_SURFACE_VOXEL_SIZE),
        threshold=0.25,
        smooth_lambda=0.0,
        anisotropic=True,
        kernel_scale=0.6,
        anisotropy_ratio=16.0,
        anisotropy_scale=2.0,
        anisotropy_min_neighbors=4,
        anisotropy_binning=True,
        anisotropy_strength=0.95,
        field_smooth_iterations=1,
        field_smooth_radius=2,
        field_mode="sdf",
        redistance_iterations=1,
        mesh_smooth_iterations=1,
    )

    def get_fluid_surface_meshes(self):
        """
        Surface meshes reconstructed from the live MPM particles, for rendering fluid as a liquid instead of beads.

        Returns:
            dict: system_name -> (vertices, indices, normals) warp arrays, or None if the system has no surface
                (e.g. a few isolated droplets). Empty if no fluid is currently simulated.
        """
        import warp as wp

        if not self._mpm_in_place:
            return {}
        mpm = self._solver.solver("mpm") if hasattr(self._solver, "solver") else self._solver
        if self._fluid_surface_cache.get("solver") is not mpm:
            # Each rebuild makes a new solver (and particle layout), so the extraction contexts are per solver
            self._fluid_surface_cache = {"solver": mpm, "surfaces": {}, "flags": {}}
        cache = self._fluid_surface_cache
        systems = {name: span for name, span in self._path_fluid_particle_map.items() if span[1] > span[0]}
        meshes = {}
        for system_name, (start, end) in systems.items():
            if system_name not in cache["surfaces"]:
                cache["surfaces"][system_name] = mpm.create_particle_surface(
                    voxel_size=self._FLUID_SURFACE_VOXEL_SIZE, **self._FLUID_SURFACE_KWARGS
                )
                flags = None
                if len(systems) > 1:
                    # Surface only this system's particles, so separate fluids don't merge into one mesh
                    flags = mpm._mpm_model.particle_flags.numpy().copy()
                    flags[:start] = 0
                    flags[end:] = 0
                    flags = wp.array(flags, dtype=wp.int32, device=self._model.device)
                cache["flags"][system_name] = flags
            mesh = mpm.extract_particle_surface(
                self._state_0,
                cache["surfaces"][system_name],
                extrapolate_into_colliders=True,
                collider_extrapolation_depth=2.0 * self._FLUID_SURFACE_VOXEL_SIZE,
                particle_flags=cache["flags"][system_name],
            )
            vertices, indices, normals = mesh.to_arrays()
            meshes[system_name] = None if vertices is None else (vertices, indices, normals)
        return meshes

    def _sync_fk(self):
        import newton

        if self._model is not None and self._model.joint_count > 0:
            newton.eval_fk(self._model, self._state_0.joint_q, self._state_0.joint_qd, self._state_0)
        self._bvh_dirty = True

    def _scale_body_mass(self, body_idx, mass):
        """Sets a live body's mass, scaling its inertia by the same factor (same geometry, uniform density)."""
        model = self._model
        old_mass = float(_wp_to_torch(model.body_mass)[body_idx])
        ratio = mass / old_mass if old_mass > 0.0 else 1.0
        _wp_to_torch(model.body_mass)[body_idx] = mass
        _wp_to_torch(model.body_inv_mass)[body_idx] = 1.0 / mass
        _wp_to_torch(model.body_inertia)[body_idx] *= ratio
        _wp_to_torch(model.body_inv_inertia)[body_idx] /= ratio
        self._notify_inertial_properties_changed()

    def _notify_inertial_properties_changed(self):
        if self._solver is not None:
            import newton

            self._solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            self._step_graphs = {}

    def _notify_dof_properties_changed(self):
        if self._solver is not None:
            import newton

            self._solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            # That call does host-side solver bookkeeping which a captured replay of step() would
            # skip, so re-capture rather than replaying a graph recorded before the change.
            self._step_graphs = {}

    # ---- Per-prim articulation / rigid-body view I/O ----

    def create_articulation_view(self, prim_path):
        handle = NewtonArticulationHandle(self, prim_path)
        self._articulation_handles[prim_path] = handle
        return handle

    def create_rigid_body_view(self, prim_path):
        handle = NewtonRigidBodyHandle(self, prim_path)
        self._rigid_body_handles[prim_path] = handle
        return handle

    # ---- Sleep / wake (no Newton equivalent found -- bodies never sleep in this backend) ----

    def is_asleep(self, prim_path):
        return False

    def wake(self, prim_path):
        pass

    def sleep(self, prim_path):
        pass

    # ---- External forces ----
    # state.body_f is a real per-body external wrench (world-frame linear force applied at the body's
    # COM, plus torque) that SolverMuJoCo's own apply_mjc_body_f_kernel reads every step when non-None.
    # Written onto self._state_0 (the buffer about to be consumed as this step's input); the plain-rigid
    # branch of step_physics_once() clears it right after each step so an applied force takes effect for
    # exactly one step, matching PhysX's apply_force_at_pos/apply_torque one-shot semantics.

    def _accumulate_body_wrench(self, prim_path, force, torque):
        if self._state_0 is None or self._state_0.body_f is None:
            return
        body_idx = self._path_body_map.get(prim_path)
        if body_idx is None:
            return
        body_f = _wp_to_torch(self._state_0.body_f)
        if force is not None:
            body_f[body_idx, :3] += th.as_tensor(force, dtype=body_f.dtype, device=body_f.device)
        if torque is not None:
            body_f[body_idx, 3:6] += th.as_tensor(torque, dtype=body_f.dtype, device=body_f.device)

    def apply_force_at_pos(self, prim_path, force, pos):
        body_idx = self._path_body_map.get(prim_path)
        if body_idx is None or self._state_0 is None or self._state_0.body_f is None:
            return
        body_q = _wp_to_torch(self._state_0.body_q)
        com_local = _wp_to_torch(self._model.body_com)[body_idx]
        com_world = body_q[body_idx, :3] + _quat_rotate_batch(
            body_q[body_idx, 3:7].unsqueeze(0), com_local.unsqueeze(0)
        ).squeeze(0)
        force_t = th.as_tensor(force, dtype=com_world.dtype, device=com_world.device)
        pos_t = th.as_tensor(pos, dtype=com_world.dtype, device=com_world.device)
        torque_t = th.linalg.cross(pos_t - com_world, force_t)
        self._accumulate_body_wrench(prim_path, force_t, torque_t)

    def apply_torque(self, prim_path, torque):
        self._accumulate_body_wrench(prim_path, None, torque)

    # ---- Scene queries ----
    # No PhysX scene-query interface exists standalone. raycast_closest/raycast_all are implemented via
    # newton.intersect_ray against the model's own shape BVH (real, public API -- see _rebuild()'s
    # bvh_build_shapes() call). overlap_sphere/overlap_box/overlap_sphere_any have no equivalent public
    # Newton API, so they're implemented as an AABB-based approximate overlap test against every tracked
    # rigid body's own collision-shape geometry (world AABB computed from body_q + each shape's local
    # transform/AABB, see _shape_world_aabbs()) -- every real call site in this codebase (contact_particles,
    # cloth_prim, system_base, toggle, heat_source_or_sink, particle_modifier) only ever checks whether a
    # small query volume overlaps a rigid body, reported via a `hit.rigid_body` path, never against
    # particles (those are already handled separately via direct position tensors) or with a need for
    # exact mesh-level precision -- PhysX's own call sites already do a coarse relaxed-AABB pre-filter
    # before their own precise narrowphase check, so AABB precision here is not a functional downgrade
    # for these use cases. overlap_mesh/overlap_shape remain unimplemented (see below) -- only reachable
    # from IK-based motion planning, itself unsupported on this backend.

    def _resolve_body_path(self, body_idx):
        """Prim path for a body index, with a fallback for -1 (world/static shapes -- in practice
        always the ground plane, the only shape with no owning body in a normal scene) to a synthetic
        ground-plane collision path, matching the convention already used elsewhere in this codebase
        (RigidContactAPIImpl.get_body_filters()'s `og.sim.floor_plane.prim_path + "/collisionPlane"`).
        Used by raycast_closest/raycast_all, whose hits can land on any shape including the ground
        plane -- unlike overlap_sphere/overlap_box/overlap_sphere_any, which already exclude -1 bodies
        upstream via _shape_world_aabbs()'s own valid_mask (a query volume overlapping a rigid body is
        never meaningfully "the ground plane" for those use cases the same way a raycast hit is).
        Returns None only if body_idx is neither a tracked body nor -1 with a resolvable floor plane
        (e.g. no floor exists in this scene at all) -- found necessary because several existing call
        sites (e.g. object_states/adjacency.py, heat_source_or_sink.py, contact_particles.py) assume
        `hit.rigid_body` is always a real path and call `.split("/")` on it directly, matching what
        real PhysX guarantees (every collidable shape there has a real backing prim, including its own
        ground plane) but this backend didn't for the -1 case until this fallback was added.
        """
        path = self._body_path_map.get(body_idx)
        if path is not None:
            return path
        if body_idx == -1 and self.sim is not None and getattr(self.sim, "floor_plane", None) is not None:
            return self.sim.floor_plane.prim_path + "/collisionPlane"
        return None

    def _shape_world_aabbs(self):
        """(valid_mask, lower, upper, shape_body) world-space AABB per shape, or None if there are no
        shapes with an owning body. Shapes with shape_body == -1 (e.g. the ground plane, which has no
        tracked prim path) are excluded via valid_mask."""
        model = self._model
        if model is None or model.shape_count == 0:
            return None
        shape_body = _wp_to_torch(model.shape_body)
        valid = shape_body >= 0
        if not bool(valid.any()):
            return None
        body_q = _wp_to_torch(self._state_0.body_q)
        body_idx = shape_body.clamp(min=0).long()
        body_pos = body_q[body_idx, :3]
        body_quat = body_q[body_idx, 3:7]
        shape_tf = _wp_to_torch(model.shape_transform)
        shape_local_pos = shape_tf[:, :3]
        shape_local_quat = shape_tf[:, 3:7]

        world_quat = _quat_mul_batch(body_quat, shape_local_quat)
        world_pos = body_pos + _quat_rotate_batch(body_quat, shape_local_pos)

        aabb_lower_local = _wp_to_torch(model.shape_collision_aabb_lower)
        aabb_upper_local = _wp_to_torch(model.shape_collision_aabb_upper)
        c_local = (aabb_lower_local + aabb_upper_local) / 2.0
        he_local = (aabb_upper_local - aabb_lower_local) / 2.0

        # Tight world AABB of a rotated local AABB: rotate the center, and take |R| @ half_extent for the
        # new half-extent (standard closed-form transformed-AABB formula, exact for any rotation).
        r_abs = _quat_to_abs_rotmat_batch(world_quat)
        c_world = world_pos + _quat_rotate_batch(world_quat, c_local)
        he_world = th.einsum("nij,nj->ni", r_abs, he_local)

        return valid, c_world - he_world, c_world + he_world, shape_body

    def _overlap_shapes(self, query_lower, query_upper, sphere_center=None, sphere_radius=None):
        """Body indices whose world AABB overlaps the query AABB [query_lower, query_upper]. If a sphere
        is also given, further refines with a closest-point-on-AABB distance check (tighter than the
        AABB-only test, matching a real sphere query more closely)."""
        result = self._shape_world_aabbs()
        if result is None:
            return []
        valid, lower, upper, shape_body = result
        q_lower = th.as_tensor(query_lower, dtype=lower.dtype, device=lower.device)
        q_upper = th.as_tensor(query_upper, dtype=lower.dtype, device=lower.device)
        mask = valid & (lower <= q_upper).all(dim=-1) & (upper >= q_lower).all(dim=-1)
        idxs = mask.nonzero(as_tuple=True)[0]
        if sphere_center is not None and len(idxs) > 0:
            center = th.as_tensor(sphere_center, dtype=lower.dtype, device=lower.device)
            clamped = th.maximum(lower[idxs], th.minimum(upper[idxs], center))
            dist = (clamped - center).norm(dim=-1)
            idxs = idxs[dist <= sphere_radius]
            if len(idxs) > 0:
                # AABBs alone wildly over-report for concave objects (every point inside a pot "overlaps"
                # it), so confirm the survivors against each shape's actual geometry
                idxs = idxs[self._sphere_overlaps_shapes(sphere_center, sphere_radius, idxs)]
        return shape_body[idxs].tolist()

    def _sphere_overlaps_shapes(self, center, radius, shape_idxs):
        """Bool mask over @shape_idxs: whether a sphere at @center with @radius overlaps each shape's geometry."""
        import warp as wp

        model = self._model
        dev = model.device
        cand = wp.from_torch(shape_idxs.to(device=_wp_to_torch(model.shape_body).device, dtype=th.int32))
        out = wp.zeros(len(shape_idxs), dtype=wp.int32, device=dev)
        wp.launch(
            _sphere_shape_overlap_kernel(),
            dim=len(shape_idxs),
            inputs=[
                wp.vec3(*[float(c) for c in center]),
                float(radius),
                cand,
                self._state_0.body_q,
                model.shape_body,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                out,
            ],
            device=dev,
        )
        return _wp_to_torch(out).bool().to(shape_idxs.device)

    def overlap_sphere(self, radius, pos, reportFn, anyHit=False):
        query_lower = [p - radius for p in pos]
        query_upper = [p + radius for p in pos]
        for body_idx in self._overlap_shapes(query_lower, query_upper, sphere_center=pos, sphere_radius=radius):
            rigid_body = self._body_path_map.get(int(body_idx))
            if rigid_body is None:
                continue
            if not reportFn(_SceneQueryHit(rigid_body)):
                return

    def overlap_box(self, halfExtent, pos, rot, reportFn, anyHit=False):
        # Every real call site passes an identity rotation; a non-identity one is only used to compute
        # the query box's own AABB here (not rotated further against shape geometry).
        if tuple(rot) not in ((0, 0, 0, 1), (0.0, 0.0, 0.0, 1.0)):
            log.warning("NewtonBackend.overlap_box: non-identity `rot` is approximated via its own AABB only.")
        query_lower = [p - h for p, h in zip(pos, halfExtent)]
        query_upper = [p + h for p, h in zip(pos, halfExtent)]
        for body_idx in self._overlap_shapes(query_lower, query_upper):
            rigid_body = self._body_path_map.get(int(body_idx))
            if rigid_body is None:
                continue
            if not reportFn(_SceneQueryHit(rigid_body)):
                return

    def overlap_mesh(self, *args, **kwargs):
        raise NotImplementedError(
            "overlap_mesh is not implemented for the Newton backend -- only reachable from IK-based "
            "motion planning (utils/motion_planning_utils.py), which is itself unsupported on this "
            "backend (see NewtonArticulationHandle's Jacobian methods)."
        )

    def overlap_shape(self, *args, **kwargs):
        raise NotImplementedError(
            "overlap_shape is not implemented for the Newton backend -- only reachable from IK-based "
            "motion planning (utils/motion_planning_utils.py), which is itself unsupported on this "
            "backend (see NewtonArticulationHandle's Jacobian methods)."
        )

    def overlap_sphere_any(self, radius, pos):
        query_lower = [p - radius for p in pos]
        query_upper = [p + radius for p in pos]
        return len(self._overlap_shapes(query_lower, query_upper, sphere_center=pos, sphere_radius=radius)) > 0

    def _ensure_bvh_fresh(self):
        if self._bvh_dirty:
            self._model.bvh_refit_shapes(self._state_0)
            self._bvh_dirty = False

    def _get_mujoco_solver(self):
        """The live SolverMuJoCo instance backing this model's rigid dynamics, or None if there's no
        rigid content at all (cloth-only/fluid-only scenes). Unwraps SolverCoupledProxy (used whenever
        cloth or fluid coexists with rigid content) via its own public solver("mjc") accessor -- see
        _rebuild()'s solver-construction branching for where the "mjc" entry name comes from."""
        if self._solver is None:
            return None
        import newton

        if isinstance(self._solver, newton.solvers.SolverMuJoCo):
            return self._solver
        from newton.solvers.experimental.coupled import SolverCoupledProxy

        if isinstance(self._solver, SolverCoupledProxy):
            try:
                return self._solver.solver("mjc")
            except KeyError:
                return None
        return None

    def _ensure_contacts_fresh(self):
        """Lazily refresh self._contact_report_contacts from the live solver state, at most once per
        physics step (mirrors _ensure_bvh_fresh()'s dirty-flag pattern) -- matches how
        RigidContactAPI.add_contacts_from_physics_step() already calls this data's PhysX equivalent
        exactly once per physics substep, so this isn't meaningfully more work than PhysX already does."""
        if not self._contacts_dirty:
            return
        solver = self._get_mujoco_solver()
        if solver is not None and self._contact_report_contacts is not None:
            solver.update_contacts(self._contact_report_contacts, self._state_0)
        self._contacts_dirty = False

    def _contact_matrix_data(self):
        """(shape0, shape1, force) long/long/float32 tensors, one row per currently-valid rigid contact
        (truncated from the fixed-size _RIGID_CONTACT_MAX allocation), or None if there's no rigid
        contact-reporting pipeline for this model (cloth-only/fluid-only scenes -- see
        _get_mujoco_solver()). force is Newton's own shape0-relative world-frame linear contact force
        (Contacts.force's first 3 components; the last 3 are a torque about shape0's body COM, always
        zero for the condim=3 contacts this dataset's collision shapes produce -- not used here)."""
        self._ensure_contacts_fresh()
        if self._contact_report_contacts is None:
            return None
        n = int(_wp_to_torch(self._contact_report_contacts.rigid_contact_count).reshape(-1)[0].item())
        shape0 = _wp_to_torch(self._contact_report_contacts.rigid_contact_shape0)[:n].long()
        shape1 = _wp_to_torch(self._contact_report_contacts.rigid_contact_shape1)[:n].long()
        force = _wp_to_torch(self._contact_report_contacts.force)[:n, :3]
        return shape0, shape1, force

    def create_attachment_constraint(
        self,
        prim_path,
        joint_type,
        body0,
        body1,
        joint_frame_in_parent_frame_pos,
        joint_frame_in_parent_frame_quat,
        joint_frame_in_child_frame_pos,
        joint_frame_in_child_frame_quat,
    ):
        """
        Assisted grasping attaches and detaches objects many times per episode. The base implementation's
        USD joint would trigger a full model rebuild each time (see _on_usd_joint_changed()), so instead
        this retargets one of the model's preallocated, disabled WELD equality slots onto (body0, body1)
        and enables it in place, without a rebuild. A SphericalJoint is the same WELD with torquescale=0, which zeroes both the
        rotational Jacobian rows and residual, leaving a pure point (CONNECT-like) constraint.

        Returns @prim_path as the handle; no USD prim is created.
        """
        assert joint_type in ("FixedJoint", "SphericalJoint"), f"Unsupported attachment joint type: {joint_type}"
        # The joint frames follow USD's convention of being expressed in each body's *scaled* local frame,
        # whereas MuJoCo's weld anchors live in the unscaled body frame.
        parent_quat = th.as_tensor(joint_frame_in_parent_frame_quat, dtype=th.float32).cpu()
        child_quat = th.as_tensor(joint_frame_in_child_frame_quat, dtype=th.float32).cpu()
        self._attachment_constraints[prim_path] = {
            "body0": body0,
            "body1": body1,
            "anchor0": th.as_tensor(joint_frame_in_parent_frame_pos, dtype=th.float32).cpu() * self._body_scale(body0),
            "anchor1": th.as_tensor(joint_frame_in_child_frame_pos, dtype=th.float32).cpu() * self._body_scale(body1),
            # Orientation of body1 relative to body0 implied by both sides' joint frames coinciding.
            "rel_quat": T.quat_multiply(parent_quat, T.quat_inverse(child_quat)),
            "torquescale": 1.0 if joint_type == "FixedJoint" else 0.0,
        }
        if len(self._attachment_constraints) > len(self._attachment_slots):
            # Pool exhausted -- grow it with a rebuild, which step_physics_once() runs before the next step.
            self._joints_dirty = True
        else:
            self._sync_attachment_slots()
        return prim_path

    def remove_attachment_constraint(self, handle):
        if self._attachment_constraints.pop(handle, None) is not None:
            self._sync_attachment_slots()

    def _body_scale(self, prim_path):
        from pxr import Gf

        from omnigibson.physics_backends.newton_visuals import raw_world_transform

        scale = Gf.Transform(raw_world_transform(self.sim.stage.GetPrimAtPath(prim_path))).GetScale()
        return th.tensor(tuple(scale), dtype=th.float32)

    def _sync_attachment_slots(self):
        """Writes every live attachment constraint onto the current model's slot pool (in insertion order)
        and disables the rest. Rewrites the whole pool each call -- it's a handful of rows."""
        solver = self._get_mujoco_solver()
        if solver is None or not self._attachment_slots or solver.mjw_model is None:
            return
        import newton

        if solver.model is not self._model:
            # SolverCoupledProxy hands its MuJoCo entry a view model whose equality rows may be re-indexed.
            if self._attachment_constraints:
                log.warning("Attachment constraints are not supported alongside cloth/fluid; ignoring them.")
            return

        eq_map = _wp_to_torch(solver.mjc_eq_to_newton_eq)[0].tolist()
        newton_to_mjc_eq = {int(n): i for i, n in enumerate(eq_map) if n >= 0}
        body_map = _wp_to_torch(solver.mjc_body_to_newton)[0].tolist()
        newton_to_mjc_body = {int(n): i for i, n in enumerate(body_map)}
        newton_to_mjc_body[-1] = 0

        attrs = self._model.mujoco
        eq_obj1id = solver.mjw_model.eq_obj1id.numpy()
        eq_obj2id = solver.mjw_model.eq_obj2id.numpy()
        anchor = attrs.equality_constraint_anchor.numpy()
        relpose = attrs.equality_constraint_relpose.numpy()
        torquescale = attrs.equality_constraint_torquescale.numpy()
        enabled = attrs.equality_constraint_enabled.numpy()

        specs = list(self._attachment_constraints.items())
        for i, slot in enumerate(self._attachment_slots):
            enabled[slot] = False
            mjc_eq = newton_to_mjc_eq.get(slot)
            if mjc_eq is None or i >= len(specs):
                continue
            handle, spec = specs[i]
            mjc_body0 = newton_to_mjc_body.get(self._path_body_map.get(spec["body0"], -2))
            mjc_body1 = newton_to_mjc_body.get(self._path_body_map.get(spec["body1"], -2))
            if mjc_body0 is None or mjc_body1 is None:
                log.warning(f"Attachment constraint {handle} references a body missing from the model; skipping.")
                continue
            # MuJoCo weld: obj1's anchor is relpose's translation, obj2's anchor is eq anchor, and
            # relpose's rotation is obj2's orientation in obj1's frame.
            eq_obj1id[mjc_eq] = mjc_body0
            eq_obj2id[mjc_eq] = mjc_body1
            anchor[slot] = spec["anchor1"].numpy()
            relpose[slot] = th.cat([spec["anchor0"], spec["rel_quat"]]).numpy()
            torquescale[slot] = spec["torquescale"]
            enabled[slot] = True

        solver.mjw_model.eq_obj1id.assign(eq_obj1id)
        solver.mjw_model.eq_obj2id.assign(eq_obj2id)
        attrs.equality_constraint_anchor.assign(anchor)
        attrs.equality_constraint_relpose.assign(relpose)
        attrs.equality_constraint_torquescale.assign(torquescale)
        attrs.equality_constraint_enabled.assign(enabled)
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
        # Matches _notify_dof_properties_changed(): don't replay a graph captured before a notify.
        self._step_graphs = {}

    def _fix_cross_object_constraint_frames(self):
        """
        SolverMuJoCo turns each cross-object (loop-closure) joint into a WELD / CONNECT whose relative pose is
        left for spec.compile() to auto-compute from the bodies' poses *at compile time* -- the build-time poses
        seeded from raw USD, not where the bodies actually are when the joint was created (e.g. AttachedTo's
        attach pose). The constraint then yanks the bodies toward that stale relative pose and can launch them.
        Overwrite each row with the joint's own authored frames instead, using the same data layout as the
        attachment slots (see _sync_attachment_slots()).
        """
        solver = self._get_mujoco_solver()
        if solver is None or solver.mjw_model is None or not self._cross_object_constraints:
            return
        eq_body_pairs = self._resolve_eq_body_pairs(solver)
        eq_obj1id = _wp_to_torch(solver.mjw_model.eq_obj1id)
        body_map = _wp_to_torch(solver.mjc_body_to_newton)[0]
        eq_data = _wp_to_torch(solver.mjw_model.eq_data)
        for body0, body1, is_spherical, p_pos, p_quat, c_pos, c_quat in self._cross_object_constraints:
            eq = eq_body_pairs.get((body0, body1))
            if eq is None:
                continue
            # The row may list the bodies in either order; obj1 is "first" in MuJoCo's semantics below.
            if int(body_map[int(eq_obj1id[eq])]) != body0:
                p_pos, p_quat, c_pos, c_quat = c_pos, c_quat, p_pos, p_quat
            row = eq_data[:, eq]
            device = row.device
            if is_spherical:
                # CONNECT: data[0:3] = anchor on obj1, data[3:6] = anchor on obj2
                row[:, 0:3] = p_pos.to(device)
                row[:, 3:6] = c_pos.to(device)
            else:
                # WELD: data[0:3] = anchor on obj2, data[3:6] = anchor on obj1, data[6:10] = obj2's
                # orientation in obj1's frame (wxyz), implied by both sides' joint frames coinciding
                rel = T.quat_multiply(p_quat, T.quat_inverse(c_quat))
                row[:, 0:3] = c_pos.to(device)
                row[:, 3:6] = p_pos.to(device)
                row[:, 6:10] = rel[[3, 0, 1, 2]].to(device)
        self._step_graphs = {}

    def _resolve_eq_body_pairs(self, solver):
        """{(newton_body0, newton_body1): mjc_eq_idx} for every equality constraint (WELD or CONNECT)
        in the live mjc model, resolved via eq_obj1id/eq_obj2id (mjc body indices) composed with
        solver.mjc_body_to_newton (a real, persistent mjc-body-index -> newton-body-index array).
        Both orderings are included since AttachedTo's own body0/body1 order isn't guaranteed to
        match eq_obj1id/eq_obj2id's order. Recomputed each call (cheap -- typically 0-1 entries; no
        dirty-flag caching needed unlike _ensure_contacts_fresh(), since equality-constraint identity
        only changes on a full _rebuild(), which callers already re-derive this from fresh."""
        model = solver.mjw_model
        eq_obj1id = _wp_to_torch(model.eq_obj1id)
        eq_obj2id = _wp_to_torch(model.eq_obj2id)
        body_map = _wp_to_torch(solver.mjc_body_to_newton)[0]
        pairs = {}
        for i in range(eq_obj1id.shape[0]):
            nb0 = int(body_map[int(eq_obj1id[i])])
            nb1 = int(body_map[int(eq_obj2id[i])])
            pairs[(nb0, nb1)] = i
            pairs[(nb1, nb0)] = i
        return pairs

    def _check_joint_breaks(self):
        """Compares each breakable joint's live equality-constraint reaction force/torque (read
        straight from the mjc solver's constraint-space output, mjw_data.efc.{type,id,force} -- no
        native Newton "joint break" concept exists, so this is the engine-agnostic-interface-facing
        equivalent of what PhysX does internally) against its authored break_force/break_torque, and
        fires self._joint_break_cb for any that just exceeded theirs. A WELD constraint (the common
        FIXED-joint case) spans 6 consecutive constraint rows in a fixed [fx,fy,fz,tx,ty,tz] order --
        confirmed empirically against a real attach/break scenario -- so the first 3 rows are the
        linear reaction force [N] (compared to break_force) and the last 3 are the angular reaction
        torque [N*m] (compared to break_torque). A CONNECT constraint (SPHERICAL joints) spans only 3
        rows (translation only, matching a ball joint's free rotation) -- compared to break_force
        only. Called once per step from step_physics_once(); cheap no-op when there are no breakable
        joints (the common case)."""
        if not self._breakable_joints:
            return
        if self._joints_settle_steps_remaining > 0:
            self._joints_settle_steps_remaining -= 1
            return
        solver = self._get_mujoco_solver()
        if solver is None or solver.mjw_data is None:
            return
        eq_body_pairs = self._resolve_eq_body_pairs(solver)
        d = solver.mjw_data
        efc_type = _wp_to_torch(d.efc.type)[0]
        efc_id = _wp_to_torch(d.efc.id)[0]
        efc_force = _wp_to_torch(d.efc.force)[0]
        nefc = int(_wp_to_torch(d.nefc)[0].item())
        equality_type = 0  # mujoco_warp._src.types.ConstraintType.EQUALITY

        for joint_path, (break_force, break_torque, body0_idx, body1_idx) in list(self._breakable_joints.items()):
            if joint_path in self._already_broken_joints:
                continue
            eq_idx = eq_body_pairs.get((body0_idx, body1_idx))
            if eq_idx is None:
                continue
            row_mask = (efc_type[:nefc] == equality_type) & (efc_id[:nefc] == eq_idx)
            rows = efc_force[:nefc][row_mask]
            if rows.numel() == 0:
                continue
            if rows.numel() >= 6:
                broke = (break_force is not None and th.linalg.norm(rows[:3]).item() > break_force) or (
                    break_torque is not None and th.linalg.norm(rows[3:6]).item() > break_torque
                )
            else:
                broke = break_force is not None and th.linalg.norm(rows).item() > break_force
            if broke:
                self._already_broken_joints.add(joint_path)
                if self._joint_break_cb is not None:
                    self._joint_break_cb(_NewtonJointBreakEvent(joint_path))

    def raycast_closest(self, origin, dir, distance):
        # Both entry points funnel into one private implementation rather than into each other: the
        # base raycast_closest_batch() is itself a loop over raycast_closest(), so having this method
        # call the public batch method would mutually recurse for any subclass that drops the override.
        return self._raycast_closest_rays([origin], [dir], [distance])[0]

    def raycast_closest_batch(self, origins, dirs, distances):
        """
        Closest hit per ray, from a SINGLE intersect_ray launch for the whole batch.

        Worth overriding the base per-ray loop because newton.intersect_ray() declares its kernel
        inside itself with ``module="unique"``, so every call reconstructs that kernel and re-hashes
        its whole reference closure -- ~2.7 ms, essentially independent of how many rays the call
        carries (measured: 1 ray and 32 rays both cost 2.7 ms). Casting a batch one ray at a time
        therefore pays that cost per ray: assisted grasping's 32 rays per physics step measured
        ~103 ms unbatched versus ~2.7 ms batched. The output readbacks (and shape_body) are device
        syncs too, so those are also done once per batch rather than once per ray.
        """
        return self._raycast_closest_rays(origins, dirs, distances)

    def _raycast_closest_rays(self, origins, dirs, distances):
        """Shared implementation of raycast_closest()/raycast_closest_batch() -- see the latter."""
        import newton
        import warp as wp

        n_rays = len(origins)
        if self._model is None or self._model.shape_count == 0 or n_rays == 0:
            return [{"hit": False} for _ in range(n_rays)]
        self._ensure_bvh_fresh()

        dev = self._model.device
        out_dist = wp.zeros(n_rays, dtype=wp.float32, device=dev)
        out_shape_id = wp.full(n_rays, -1, dtype=wp.int32, device=dev)
        out_normal = wp.zeros(n_rays, dtype=wp.vec3, device=dev)
        newton.intersect_ray(
            self._model,
            ray_origins=wp.array(origins, dtype=wp.vec3, device=dev),
            ray_directions=wp.array(dirs, dtype=wp.vec3, device=dev),
            ray_worlds=wp.full(n_rays, -1, dtype=wp.int32, device=dev),
            out_dist=out_dist,
            out_shape_id=out_shape_id,
            out_normal=out_normal,
        )

        shape_ids = out_shape_id.numpy()
        hit_dists = out_dist.numpy()
        hit_normals = out_normal.numpy()
        shape_body = self._model.shape_body.numpy()

        results = []
        for i in range(n_rays):
            shape_id = int(shape_ids[i])
            hit_dist = float(hit_dists[i])
            # intersect_ray has no max-distance cutoff -- treat a hit beyond the requested distance as a miss.
            if shape_id < 0 or hit_dist > float(distances[i]):
                results.append({"hit": False})
                continue
            rigid_body = self._resolve_body_path(int(shape_body[shape_id]))
            origin_t = th.as_tensor(origins[i], dtype=th.float32)
            dir_t = th.as_tensor(dirs[i], dtype=th.float32)
            results.append(
                {
                    "hit": True,
                    "position": (origin_t + dir_t * hit_dist).tolist(),
                    "normal": hit_normals[i].tolist(),
                    "distance": hit_dist,
                    "collision": rigid_body,
                    "rigidBody": rigid_body,
                }
            )
        return results

    def raycast_all(self, origin, dir, distance, reportFn):
        import newton
        import warp as wp

        if self._model is None or self._model.shape_count == 0:
            return
        self._ensure_bvh_fresh()

        dev = self._model.device
        origin_t = th.as_tensor(origin, dtype=th.float32)
        dir_t = th.as_tensor(dir, dtype=th.float32)
        traveled = 0.0
        epsilon = 1e-4
        # newton.intersect_ray only returns the closest hit per ray -- emulate "all hits" by repeatedly
        # re-casting from just past the previous hit point ("peeling"), bounded for safety.
        for _ in range(64):
            remaining = distance - traveled
            if remaining <= 0:
                break
            cur_origin = (origin_t + dir_t * traveled).tolist()
            out_dist = wp.zeros(1, dtype=wp.float32, device=dev)
            out_shape_id = wp.full(1, -1, dtype=wp.int32, device=dev)
            out_normal = wp.zeros(1, dtype=wp.vec3, device=dev)
            newton.intersect_ray(
                self._model,
                ray_origins=wp.array([cur_origin], dtype=wp.vec3, device=dev),
                ray_directions=wp.array([dir], dtype=wp.vec3, device=dev),
                ray_worlds=wp.array([-1], dtype=wp.int32, device=dev),
                out_dist=out_dist,
                out_shape_id=out_shape_id,
                out_normal=out_normal,
            )
            shape_id = int(out_shape_id.numpy()[0])
            if shape_id < 0:
                break
            seg_dist = float(out_dist.numpy()[0])
            if seg_dist > remaining:
                break
            hit_dist = traveled + seg_dist
            body_idx = int(self._model.shape_body.numpy()[shape_id])
            rigid_body = self._resolve_body_path(body_idx)
            position = (origin_t + dir_t * hit_dist).tolist()
            hit = _SceneQueryHit(
                rigid_body, position=position, normal=out_normal.numpy()[0].tolist(), distance=hit_dist
            )
            if not reportFn(hit):
                return
            traveled = hit_dist + epsilon

    # ---- Joint-break events ----
    # No native engine event stream exists standalone (unlike PhysX's simulation-event subscription)
    # -- _check_joint_breaks() detects breaks itself each step and synthesizes a _NewtonJointBreakEvent,
    # which these two just recognize/unpack.

    def is_joint_break_event(self, event):
        return isinstance(event, _NewtonJointBreakEvent)

    def decode_joint_break_event(self, event):
        return event.joint_path
