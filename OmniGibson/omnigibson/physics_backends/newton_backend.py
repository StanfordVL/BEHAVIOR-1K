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


def raw_world_transform(prim):
    from pxr import Gf, UsdGeom

    stage = prim.GetStage()
    pseudo_root_path = stage.GetPseudoRoot().GetPath()
    world = Gf.Matrix4d(1.0)
    p = prim
    while p and p.GetPath() != pseudo_root_path:
        xformable = UsdGeom.Xformable(p)
        if xformable:
            world = world * xformable.GetLocalTransformation()
        p = p.GetParent()
    return world


def _wp_to_torch(array):
    import warp as wp

    if array.size == 0:
        return th.from_numpy(array.numpy())
    return wp.to_torch(array)


def _physx_schema_resolver():
    """Stock SchemaResolverPhysx has no joint friction entry, so rebuilds would reset joint friction to 0."""
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
        d = float(-1.0e6)
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
            q = wp.mesh_query_point_sign_normal(mesh, wp.cw_div(x, scale), 1.0e6)
            if q.result:
                closest = wp.cw_mul(wp.mesh_eval_position(mesh, q.face, q.u, q.v), scale)
                d = wp.length(x - closest) * q.sign
        if d <= radius:
            out[k] = 1

    _SPHERE_SHAPE_OVERLAP_KERNEL = kernel
    return kernel


def _assign(wp_array, idx, value):
    target = _wp_to_torch(wp_array)
    target[idx] = th.as_tensor(value).reshape(-1).to(device=target.device, dtype=target.dtype)


def _xyzw_to_wxyz(quat):
    return quat[..., [3, 0, 1, 2]]


def _wxyz_to_xyzw(quat):
    return quat[..., [1, 2, 3, 0]]


def _quat_mul_batch(q1, q0):
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
    qxyz = q[..., :3]
    qw = q[..., 3:4]
    t = 2.0 * th.linalg.cross(qxyz, v, dim=-1)
    return v + qw * t + th.linalg.cross(qxyz, t, dim=-1)


def _quat_to_abs_rotmat_batch(q):
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
    __slots__ = ("rigid_body", "collision", "position", "normal", "distance")

    def __init__(self, rigid_body, position=None, normal=None, distance=None):
        self.rigid_body = rigid_body
        self.collision = rigid_body
        self.position = position
        self.normal = normal
        self.distance = distance


class _NewtonJointBreakEvent:
    __slots__ = ("joint_path",)

    def __init__(self, joint_path):
        self.joint_path = joint_path


class _AppliedActions:
    def __init__(self, joint_positions, joint_velocities):
        self.joint_positions = joint_positions
        self.joint_velocities = joint_velocities


def _native_to_handle_dof_cols(handle, dof_start, dof_count):
    handle_cols = [int(g) - dof_start for g in handle._global_qd_idx.tolist()]
    return sorted(set(range(dof_count)) - set(handle_cols)) + handle_cols


def _breadth_first_joint_order(model, global_joint_idx):
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
    def __init__(self, backend, prim_path):
        self._backend = backend
        self.prim_path = prim_path
        self._root_link_prim_path = prim_path
        self._valid = False
        self._global_joint_idx = []
        self._joint_dof_counts = []
        self._joint_dof_offsets = []
        self._joint_names = []
        self._local_dof_to_path = []
        self._global_qd_idx = None
        self._global_q_idx = None
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
        # Re-emit joints in PhysX's DOF order so indices and serialized state are backend-independent.
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

    def get_coriolis_and_centrifugal_forces(self, clone=True):
        raise NotImplementedError("Coriolis/centrifugal forces are not implemented for the Newton backend.")

    def get_generalized_gravity_forces(self, clone=True):
        raise NotImplementedError("Generalized gravity forces are not implemented for the Newton backend.")

    def get_mass_matrices(self, clone=True):
        raise NotImplementedError("Mass matrices are not implemented for the Newton backend.")

    def get_jacobians(self, clone=True):
        return self._backend._compute_jacobian(self).unsqueeze(0)


class _AlwaysValidPhysicsView:
    def check(self):
        return True


class NewtonRigidBodyHandle:
    def __init__(self, backend, prim_path):
        self._backend = backend
        self.prim_path = prim_path
        self._body_idx = None
        self._valid = False
        self._physics_view = _AlwaysValidPhysicsView()
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
            from omnigibson.utils.usd_utils import get_world_pose

            pos, quat_wxyz_from_xyzw = get_world_pose(self.prim_path)
            return pos.unsqueeze(0), _xyzw_to_wxyz(quat_wxyz_from_xyzw).unsqueeze(0)
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        pos = body_q[self._body_idx, :3].unsqueeze(0)
        quat_wxyz = _xyzw_to_wxyz(body_q[self._body_idx, 3:7]).unsqueeze(0)
        # Always copy: an int index gives a live view of body_q, and callers rely on fresh tensors like PhysX's.
        return pos.clone(), quat_wxyz.clone()

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
        return _wp_to_torch(self._backend._state_0.body_qd)[self._body_idx, :3].unsqueeze(0).clone()

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
        return _wp_to_torch(self._backend._state_0.body_qd)[self._body_idx, 3:6].unsqueeze(0).clone()

    def set_angular_velocities(self, velocities):
        if not self._valid:
            return
        _assign(self._backend._state_0.body_qd, (self._body_idx, slice(3, 6)), velocities)
        self._set_root_free_joint_qd(3, velocities)

    def get_coms(self, clone=True):
        com = _wp_to_torch(self._backend._model.body_com)[self._body_idx].reshape(1, 1, 3).clone()
        return com, th.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=com.device)

    def set_coms(self, positions):
        _assign(self._backend._model.body_com, self._body_idx, th.as_tensor(positions).reshape(-1)[:3])
        self._backend._notify_inertial_properties_changed()

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
        body_q = _wp_to_torch(self._backend._state_0.body_q)
        result = []
        for idxs in self._body_idx_per_robot:
            result.append(th.stack([body_q[idx, :7] for idx in idxs], dim=0))
        return th.stack(result, dim=0)

    def get_link_velocities(self):
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


def _match_body_paths(backend, pattern):
    matched = set(_match_paths(pattern, backend._path_body_map))
    matched = [(path, idx) for path, idx in backend._path_body_map.items() if path in matched]
    return [path for path, _ in matched], [idx for _, idx in matched]


def _match_paths(pattern, paths):
    import re

    regex = re.compile(re.escape(pattern).replace(r"\*", "[^/]*"))
    return [path for path in paths if regex.fullmatch(path)]


class _NewtonRigidBodyBatchView:
    def __init__(self, backend, body_indices, prim_paths, pattern=None):
        self._backend = backend
        self._pattern = pattern
        self._generation = backend._model_generation
        self._set_bodies(body_indices, prim_paths)

    def _set_bodies(self, body_indices, prim_paths):
        backend = self._backend
        self._body_indices = th.as_tensor(body_indices, dtype=th.long, device=backend.sim.device)
        self._prim_paths = list(prim_paths)
        self._root_free_joint_q_start = self._compute_root_free_joint_starts(qd=False)
        self._root_free_joint_qd_start = self._compute_root_free_joint_starts(qd=True)

    def _refresh_if_stale(self):
        if self._pattern is None:
            return
        backend = self._backend
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
    def __init__(self, backend, sensor_paths, sensor_body_idx, filter_paths, filter_body_idx, max_contact_data_count):
        self._backend = backend
        self._sensor_paths = list(sensor_paths)
        self._filter_paths = list(filter_paths)
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
        separations = th.zeros((len(points_list), 1), device=device)
        return forces, points, normals, separations, contact_counts, start_indices

    def _contact_bodies(self, shape0, shape1):
        shape_body = _wp_to_torch(self._backend._model.shape_body)
        return shape_body[shape0], shape_body[shape1]


class _NewtonPhysicsSimView:
    def __init__(self, backend):
        self._backend = backend

    def create_articulation_view(self, pattern):
        import re

        if isinstance(pattern, (list, tuple, set)):
            path_set = set(pattern)

            def is_match(prim_path):
                return prim_path in path_set
        else:
            regex = pattern.replace("*", ".*")

            def is_match(prim_path):
                return re.fullmatch(regex, prim_path) is not None

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
    supports_contact_reporting = True

    # Imported object joints are zero-gain DOFs that drift under contact/gravity without passive damping.
    _PASSIVE_OBJECT_JOINT_KD = 5.0
    _FLOOR_CONTACT_FRICTION = 0.02
    _JOINT_ARMATURE_FLOOR = 1.0e-2
    # Arm joints need more: 1e-2 let held objects slip in the hand and go NaN.
    _ARM_JOINT_ARMATURE_FLOOR = 1.0e-1
    _JOINT_FRICTION_FLOOR = 2.0e-1

    def __init__(self, sim=None):
        super().__init__(sim)
        self._sim_context = None
        self._model = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._contacts = None
        self._solver = None
        self._rigid_collision_pipeline = None
        self._pre_step_cb = None
        self._post_step_cb = None
        self._joint_break_cb = None
        self._path_body_map = {}
        self._body_path_map = {}
        self._bvh_dirty = True
        self._contact_report_contacts = None
        self._contacts_dirty = True
        self._rebuilding = False
        self._path_joint_map = {}
        self._articulation_handles = {}
        self._rigid_body_handles = {}
        self._gravity = 9.81
        self._known_object_paths = frozenset()
        self._joints_dirty = False
        self._joint_change_listener = None
        self._step_graphs = {}
        self._step_graph_substeps = 0
        self._step_graph_disabled = False
        self._breakable_joints = {}
        self._already_broken_joints = set()
        self._joints_settle_steps_remaining = 0
        self._mass_properties_dirty = False
        self._model_generation = 0
        self._warned_cpu_device = False

    @classmethod
    def create_app(cls):
        if gm.RENDER_BACKEND == "kit":
            raise ValueError("Newton physics does not support gm.RENDER_BACKEND = 'kit'; use 'newton' or 'none'.")
        from omnigibson.simulator import _StandaloneApp

        return _StandaloneApp()

    def add_ground_plane(self, prim_path, visible=True, color=None):
        with self.sim.editing_usd():
            self.sim.stage.DefinePrim(prim_path, "Xform")

    def create_physics_context(self, physics_dt, rendering_dt, device):
        self._sim_context = StandaloneSimulationContext(
            backend=self, physics_dt=physics_dt, rendering_dt=rendering_dt, device=device
        )
        self._joint_change_listener = lazy.pxr.Tf.Notice.Register(
            lazy.pxr.Usd.Notice.ObjectsChanged, self._on_usd_joint_changed, self._sim_context.stage
        )
        return self._sim_context

    def start_step_callbacks(self, pre_step_fn, post_step_fn, joint_break_fn):
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
        self._sim_context.step(render=render)

    # MuJoCo rejects even a -1e-10 deficit, so this margin only has to cover float32 rounding.
    _INERTIA_BALANCE_MARGIN = 1e-6

    @staticmethod
    def _dof_effective_inertia(builder, joint_idx, qd_start, is_revolute):
        import numpy as np

        child = builder.joint_child[joint_idx]
        mass = float(builder.body_mass[child])
        if not is_revolute:
            return mass
        x_c = builder.joint_X_c[joint_idx]
        q = np.array(x_c[3:7], dtype=np.float64)
        v = np.array(builder.joint_axis[qd_start], dtype=np.float64)
        t = 2.0 * np.cross(q[:3], v)
        axis = v + q[3] * t + np.cross(q[:3], t)
        axis /= np.linalg.norm(axis)
        r = np.array(builder.body_com[child], dtype=np.float64) - np.array(x_c[0:3], dtype=np.float64)
        r_perp = r - (r @ axis) * axis
        inertia = np.array(builder.body_inertia[child], dtype=np.float64).reshape(3, 3)
        return float(axis @ inertia @ axis + mass * (r_perp @ r_perp))

    def _balance_body_inertia(self):
        """Fix MuJoCo's A + B >= C inertia check; one failing body leaves the whole scene without a solver."""
        import numpy as np

        inertia = self._model.body_inertia.numpy()
        if inertia.size == 0:
            return

        moments = np.linalg.eigvalsh(inertia.astype(np.float64))
        largest = moments[:, 2]
        deficit = largest - moments[:, 0] - moments[:, 1]
        needs_fix = (deficit > -self._INERTIA_BALANCE_MARGIN * largest) & (largest > 0.0)
        if not needs_fix.any():
            return

        shift = (deficit + self._INERTIA_BALANCE_MARGIN * largest)[needs_fix]
        fixed = inertia.copy()
        idx = np.flatnonzero(needs_fix)
        fixed[idx] += np.eye(3, dtype=fixed.dtype) * shift[:, None, None].astype(fixed.dtype)
        self._model.body_inertia.assign(fixed)

        inv = self._model.body_inv_inertia.numpy().copy()
        inv[idx] = np.linalg.inv(fixed[idx].astype(np.float64)).astype(inv.dtype)
        self._model.body_inv_inertia.assign(inv)

        log.debug(
            f"Balanced inertia for {len(idx)} body(ies) violating MuJoCo's A + B >= C "
            f"(worst deficit {deficit[needs_fix].max():.3e})"
        )

    # Warp forbids kernel-module loads during capture, and SolverMuJoCo loads its modules lazily.
    _STEP_GRAPH_WARMUP_SUBSTEPS = 3

    def _step_graph(self, dt):
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
            self._step_graph_disabled = True
            log.warning(f"Newton physics step graph capture failed ({e}); falling back to uncaptured steps.")
            return None

        self._step_graphs[key] = graph
        return graph

    def step_physics_once(self, current_time):
        if self._mass_properties_dirty:
            self._mass_properties_dirty = False
            self._joints_dirty = True
        if self._joints_dirty:
            self._joints_dirty = False
            self._rebuild()

        if self._pre_step_cb is not None:
            self._pre_step_cb()

        if self._solver is not None:
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

    def _compute_mass_matrix(self, handle):
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

        cols = _native_to_handle_dof_cols(handle, dof_start, dof_count)
        H = H_all[art_idx, :dof_count, :dof_count]
        return H[cols][:, cols]

    def _compute_jacobian(self, handle):
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
        J = J_all[art_idx].reshape(max_links, 6, max_dofs)[1:link_count, :, :dof_count]
        return J[:, :, _native_to_handle_dof_cols(handle, dof_start, dof_count)]

    @property
    def physics_sim_view(self):
        return _NewtonPhysicsSimView(self)

    def refresh_physics_sim_view(self):
        if self._model is None or not self.is_playing():
            return
        current_paths = frozenset(obj.prim_path for scene in self.sim.scenes for obj in scene.objects)
        if current_paths != self._known_object_paths:
            self._rebuild()

    def invalidate_physics_sim_view(self):
        pass

    def flush_changes(self):
        pass

    def sync_to_render_layer(self):
        pass

    def set_dof_position_targets(self, view, data, indices, cast=False):
        view.set_dof_position_targets_fast(data, indices)

    def set_dof_velocity_targets(self, view, data, indices, cast=False):
        view.set_dof_velocity_targets_fast(data, indices)

    def set_dof_actuation_forces(self, view, data, indices, cast=False):
        view.set_dof_actuation_forces_fast(data, indices)

    def get_world_transform_with_scale(self, prim_path):
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

    def before_play(self, sim):
        self._rebuild()

    def _rebuild(self):
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
            if not prim or not prim.IsValid() or prim.IsA(lazy.pxr.UsdPhysics.Joint):
                self._joints_dirty = True
                return

    def _apply_floor_contact_friction(self, builder, robot, robot_path_body_map):
        # Holonomic only: a differential-drive robot needs friction at these links for traction.
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
        """MuJoCo uses the larger geom friction at equal priority, so the low-friction pads need higher priority."""
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

    def _find_cross_object_joints(self, objects):
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
        break_force = prim.GetAttribute("physics:breakForce").Get()
        break_torque = prim.GetAttribute("physics:breakTorque").Get()
        break_force = None if break_force is None or math.isinf(break_force) else float(break_force)
        break_torque = None if break_torque is None or math.isinf(break_torque) else float(break_torque)
        return break_force, break_torque

    def _rebuild_impl(self, newton, wp):
        from pxr import Gf

        captured_state = self._capture_state()
        self._already_broken_joints = set()

        builder = newton.ModelBuilder(gravity=[0.0, 0.0, -self._gravity])
        builder.default_shape_cfg.ke = 2.5e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.default_shape_cfg.mu = 1.0
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.01
        ground_shape = builder.add_ground_plane()
        builder.shape_flags[ground_shape] = int(builder.shape_flags[ground_shape]) & ~int(newton.ShapeFlags.VISIBLE)

        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        objects = [obj for scene in self.sim.scenes for obj in scene.objects]

        from omnigibson.robots import Robot

        objects.sort(key=lambda obj: 0 if isinstance(obj, Robot) else 1)

        cross_object_joints = self._find_cross_object_joints(objects)
        ignore_paths_by_obj = {
            obj: [str(prim.GetPath()) for prim, _, _ in joints] for obj, joints in cross_object_joints.items()
        }

        path_body_map = {}
        path_joint_map = {}
        robot_joint_paths = set()
        arm_joint_paths = set()
        low_friction_body_idx = set()
        for obj in objects:
            if obj.prim_type == PrimType.CLOTH:
                continue

            joint_count_before = len(builder.joint_articulation)
            info = builder.add_usd(
                self.sim.stage,
                root_path=obj.prim_path,
                floating=not obj.fixed_base,
                enable_self_collisions=False,
                collapse_fixed_joints=False,
                skip_mesh_approximation=False,
                load_visual_shapes=False,
                schema_resolvers=[
                    newton.usd.SchemaResolverNewton(),
                    _physx_schema_resolver(),
                    newton.usd.SchemaResolverMjc(),
                ],
                hide_collision_shapes=True,
                ignore_paths=ignore_paths_by_obj.get(obj),
            )

            path_body_map.update(info["path_body_map"])
            path_joint_map.update(info["path_joint_map"])
            if isinstance(obj, Robot):
                robot_joint_paths.update(info["path_joint_map"].keys())
                low_friction_body_idx |= self._apply_floor_contact_friction(builder, obj, info["path_body_map"])
                if obj.is_manipulation:
                    arm_names = {name for names in obj.arm_joint_names.values() for name in names}
                    arm_joint_paths.update(p for p in info["path_joint_map"] if p.rsplit("/", 1)[-1] in arm_names)

            # add_usd() only wraps joints into an articulation when ArticulationRootAPI is authored, so
            # give remaining bodies a fixed or free joint to the world root.
            object_body_idx = set(info["path_body_map"].values())
            connected_bodies = set(builder.joint_child[joint_count_before:])
            for body_idx in sorted(object_body_idx - connected_bodies):
                if obj.fixed_base:
                    builder.add_joint_fixed(parent=-1, child=body_idx)
                else:
                    builder.add_joint_free(parent=-1, child=body_idx)

            unassigned = sorted(
                idx
                for idx in range(joint_count_before, len(builder.joint_articulation))
                if builder.joint_articulation[idx] < 0
            )
            run_start = None
            for i, idx in enumerate(unassigned):
                if run_start is None:
                    run_start = idx
                if i + 1 == len(unassigned) or unassigned[i + 1] != idx + 1:
                    builder.add_articulation(list(range(run_start, idx + 1)), label=f"{obj.prim_path}_meta_links")
                    run_start = None

        self._breakable_joints = {}
        self._cross_object_constraints = []
        for joints in cross_object_joints.values():
            for prim, body0_path, body1_path in joints:
                body0_idx = path_body_map.get(body0_path)
                body1_idx = path_body_map.get(body1_path)
                if body0_idx is None or body1_idx is None:
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
            self._joints_settle_steps_remaining = 1

        self._path_body_map = path_body_map
        self._body_path_map = {idx: path for path, idx in path_body_map.items()}
        self._path_joint_map = path_joint_map

        joint_qd_start = builder.joint_qd_start
        for joint_path, joint_idx in path_joint_map.items():
            qd_start = joint_qd_start[joint_idx]
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
            # Passive damping is authored as a PhysX acceleration drive (per degree); MuJoCo kd is a torque.
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
            if joint_path not in robot_joint_paths and builder.joint_type[joint_idx] in (
                int(newton.JointType.REVOLUTE),
                int(newton.JointType.PRISMATIC),
            ):
                builder.joint_limit_kd[qd_start] = max(
                    float(builder.joint_limit_kd[qd_start]), 2.0 * math.sqrt(float(builder.joint_limit_ke[qd_start]))
                )
            builder.joint_target_mode[qd_start] = int(newton.JointTargetMode.POSITION_VELOCITY)

            if joint_path in robot_joint_paths:
                builder.joint_effort_limit[qd_start] = float("inf")

            if (
                joint_path not in robot_joint_paths
                and builder.joint_target_ke[qd_start] == 0.0
                and builder.joint_target_kd[qd_start] == 0.0
                and builder.joint_type[joint_idx] not in (int(newton.JointType.FIXED), int(newton.JointType.FREE))
            ):
                builder.joint_target_kd[qd_start] = self._PASSIVE_OBJECT_JOINT_KD

        if self._JOINT_ARMATURE_FLOOR or self._JOINT_FRICTION_FLOOR:
            robot_joint_idxs = {path_joint_map[p] for p in robot_joint_paths if p in path_joint_map}
            arm_joint_idxs = {path_joint_map[p] for p in arm_joint_paths if p in path_joint_map}
            skipped = (int(newton.JointType.FIXED), int(newton.JointType.FREE))
            n_dofs = len(builder.joint_qd)
            for joint_idx, joint_type in enumerate(builder.joint_type):
                if int(joint_type) in skipped or joint_idx not in robot_joint_idxs:
                    continue
                qd_start = builder.joint_qd_start[joint_idx]
                qd_stop = (
                    builder.joint_qd_start[joint_idx + 1] if joint_idx + 1 < len(builder.joint_qd_start) else n_dofs
                )
                armature = self._ARM_JOINT_ARMATURE_FLOOR if joint_idx in arm_joint_idxs else self._JOINT_ARMATURE_FLOOR
                for dof_idx in range(qd_start, qd_stop):
                    if dof_idx < len(builder.joint_armature):
                        builder.joint_armature[dof_idx] = max(float(builder.joint_armature[dof_idx]), armature)
                    if dof_idx < len(builder.joint_friction):
                        builder.joint_friction[dof_idx] = max(
                            float(builder.joint_friction[dof_idx]), self._JOINT_FRICTION_FLOOR
                        )

        self._model = builder.finalize(device=self._sim_context.device or "cpu")
        if self._model.device.is_cpu and wp.is_cuda_available() and not self._warned_cpu_device:
            self._warned_cpu_device = True
            log.warning(
                "Newton is simulating on CPU although CUDA is available; pass device='cuda' to og.launch() / "
                "the env config to run it on the GPU."
            )

        self._balance_body_inertia()

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

        if self._model.joint_dof_count > 0:
            self._model.request_state_attributes("mujoco:qfrc_actuator")
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._model.rigid_contact_max = self._RIGID_CONTACT_MAX
        self._contacts = self._model.contacts()

        if self._model.joint_count > 0:
            _wp_to_torch(self._state_0.joint_q).copy_(joint_q)
            _wp_to_torch(self._state_1.joint_q).copy_(joint_q)
            _wp_to_torch(self._state_0.joint_qd).copy_(_wp_to_torch(self._model.joint_qd))
            _wp_to_torch(self._state_1.joint_qd).copy_(_wp_to_torch(self._model.joint_qd))
            newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_0)
            newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_1)

        if self._model.shape_count > 0:
            self._model.bvh_build_shapes(self._state_0, shape_flags=newton.ShapeFlags.COLLIDE_SHAPES)
        self._bvh_dirty = False

        has_rigid = self._model.joint_count > 0
        self._rigid_collision_pipeline = None
        if has_rigid:
            try:
                self._solver = newton.solvers.SolverMuJoCo(self._model, **self._mujoco_solver_kwargs())
                self._rigid_collision_pipeline = newton.CollisionPipeline(
                    self._model,
                    broad_phase="explicit",
                    rigid_contact_max=max(self._RIGID_CONTACT_MAX, self._solver.get_max_contact_count()),
                    soft_contact_max=0,
                )
                self._contacts = self._rigid_collision_pipeline.contacts()
            except ValueError as e:
                if "joint" in str(e).lower():
                    log.warning(f"No Newton solver created (model has no joints yet): {e}")
                else:
                    log.error(
                        f"Newton solver could NOT be created -- physics will not run until the next "
                        f"successful rebuild: {e}"
                    )
                self._solver = None
        else:
            self._solver = None

        self._elevate_low_friction_geom_priority(low_friction_body_idx)
        self._fix_cross_object_constraint_frames()

        for handle in self._articulation_handles.values():
            handle.initialize(None)
        for handle in self._rigid_body_handles.values():
            handle.initialize(None)

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
        self._model_generation += 1
        self._restore_state(captured_state)

    # Must match or exceed every SolverMuJoCo construction's nconmax below.
    _RIGID_CONTACT_MAX = 250_000

    def _mujoco_solver_kwargs(self):
        """An undersized nconmax corrupts heap memory and segfaults later at an unrelated location."""
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

    def _capture_state(self):
        if self._model is None:
            return None
        captured = {"articulations": {}, "rigid_bodies": {}}
        for path, handle in self._articulation_handles.items():
            if not handle.is_physics_handle_valid():
                continue
            root_body_idx = self._path_body_map.get(handle._root_link_prim_path)
            world_pos, world_quat = (None, None)
            if root_body_idx is not None:
                body_q = _wp_to_torch(self._state_0.body_q)
                world_pos = body_q[root_body_idx, :3].clone()
                world_quat = _xyzw_to_wxyz(body_q[root_body_idx, 3:7]).clone()
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
        if captured is None:
            return
        for path, data in captured["articulations"].items():
            handle = self._articulation_handles.get(path)
            if handle is None or not handle.is_physics_handle_valid():
                continue
            if data["joint_q"] is not None:
                handle.set_joint_positions(data["joint_q"])
                handle.set_joint_velocities(data["joint_qd"])
            if data.get("gains_ke") is not None and handle.num_dof == data["gains_ke"].shape[-1]:
                handle.set_gains(kps=data["gains_ke"], kds=data["gains_kd"])
            if data["world_pos"] is not None and self._path_body_map.get(handle._root_link_prim_path) is not None:
                handle.set_world_poses(data["world_pos"], data["world_quat"])
        for path, data in captured["rigid_bodies"].items():
            handle = self._rigid_body_handles.get(path)
            if handle is None or not handle.is_physics_handle_valid():
                continue
            handle.set_world_poses(data["world_pos"], data["world_quat"])
            handle.set_linear_velocities(data["lin_vel"])
            handle.set_angular_velocities(data["ang_vel"])

    def _cloth_world_transform(self, prim):
        matrix = lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default())
        linear = th.tensor([[matrix[i][j] for j in range(3)] for i in range(3)], dtype=th.float32)
        translation = th.tensor([matrix[3][j] for j in range(3)], dtype=th.float32)
        return linear, translation

    def get_cloth_particle_positions(self, prim_path, idxs=None):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        p_local = vtarray_to_torch(prim.GetAttribute("points").Get())
        p_local = p_local[idxs] if idxs is not None else p_local
        linear, translation = self._cloth_world_transform(prim)
        return p_local @ linear + translation

    def set_cloth_particle_positions(self, prim_path, positions, idxs=None):
        prim = self.sim.stage.GetPrimAtPath(prim_path)
        linear, translation = self._cloth_world_transform(prim)
        p_local = (th.as_tensor(positions, dtype=th.float32, device=translation.device) - translation) @ th.linalg.inv(
            linear
        )
        if idxs is not None:
            p_local_full = vtarray_to_torch(prim.GetAttribute("points").Get())
            p_local_full[idxs] = p_local
            p_local = p_local_full
        prim.GetAttribute("points").Set(lazy.pxr.Vt.Vec3fArray(p_local.tolist()))

    def get_cloth_particle_velocities(self, prim_path):
        return th.zeros(len(self.sim.stage.GetPrimAtPath(prim_path).GetAttribute("points").Get()), 3)

    def set_cloth_particle_velocities(self, prim_path, velocities):
        pass

    def get_cloth_stiffness(self, prim_path):
        return {"bend": 0.0, "damping": 0.0, "shear": 0.0, "stretch": 0.0}

    def set_cloth_stiffness(self, prim_path, bend=None, damping=None, shear=None, stretch=None):
        pass

    def _sync_fk(self):
        import newton

        if self._model is not None and self._model.joint_count > 0:
            newton.eval_fk(self._model, self._state_0.joint_q, self._state_0.joint_qd, self._state_0)
        self._bvh_dirty = True

    def _scale_body_mass(self, body_idx, mass):
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
            self._step_graphs = {}

    def create_articulation_view(self, prim_path):
        handle = NewtonArticulationHandle(self, prim_path)
        self._articulation_handles[prim_path] = handle
        return handle

    def create_rigid_body_view(self, prim_path):
        handle = NewtonRigidBodyHandle(self, prim_path)
        self._rigid_body_handles[prim_path] = handle
        return handle

    def is_asleep(self, prim_path):
        return False

    def wake(self, prim_path):
        pass

    def sleep(self, prim_path):
        pass

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

    def _resolve_body_path(self, body_idx):
        path = self._body_path_map.get(body_idx)
        if path is not None:
            return path
        if body_idx == -1 and self.sim is not None and getattr(self.sim, "floor_plane", None) is not None:
            return self.sim.floor_plane.prim_path + "/collisionPlane"
        return None

    def _shape_world_aabbs(self):
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

        r_abs = _quat_to_abs_rotmat_batch(world_quat)
        c_world = world_pos + _quat_rotate_batch(world_quat, c_local)
        he_world = th.einsum("nij,nj->ni", r_abs, he_local)

        return valid, c_world - he_world, c_world + he_world, shape_body

    def _overlap_shapes(self, query_lower, query_upper, sphere_center=None, sphere_radius=None):
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
                idxs = idxs[self._sphere_overlaps_shapes(sphere_center, sphere_radius, idxs)]
        return shape_body[idxs].tolist()

    def _sphere_overlaps_shapes(self, center, radius, shape_idxs):
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
        return self._solver

    def _ensure_contacts_fresh(self):
        if not self._contacts_dirty:
            return
        solver = self._get_mujoco_solver()
        if solver is not None and self._contact_report_contacts is not None:
            solver.update_contacts(self._contact_report_contacts, self._state_0)
        self._contacts_dirty = False

    def _contact_matrix_data(self):
        self._ensure_contacts_fresh()
        if self._contact_report_contacts is None:
            return None
        n = int(_wp_to_torch(self._contact_report_contacts.rigid_contact_count).reshape(-1)[0].item())
        shape0 = _wp_to_torch(self._contact_report_contacts.rigid_contact_shape0)[:n].long()
        shape1 = _wp_to_torch(self._contact_report_contacts.rigid_contact_shape1)[:n].long()
        force = _wp_to_torch(self._contact_report_contacts.force)[:n, :3]
        return shape0, shape1, force

    def _body_scale(self, prim_path):
        from pxr import Gf

        scale = Gf.Transform(raw_world_transform(self.sim.stage.GetPrimAtPath(prim_path))).GetScale()
        return th.tensor(tuple(scale), dtype=th.float32)

    def _fix_cross_object_constraint_frames(self):
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
            if int(body_map[int(eq_obj1id[eq])]) != body0:
                p_pos, p_quat, c_pos, c_quat = c_pos, c_quat, p_pos, p_quat
            row = eq_data[:, eq]
            device = row.device
            if is_spherical:
                row[:, 0:3] = p_pos.to(device)
                row[:, 3:6] = c_pos.to(device)
            else:
                rel = T.quat_multiply(p_quat, T.quat_inverse(c_quat))
                row[:, 0:3] = c_pos.to(device)
                row[:, 3:6] = p_pos.to(device)
                row[:, 6:10] = rel[[3, 0, 1, 2]].to(device)
        self._step_graphs = {}

    def _resolve_eq_body_pairs(self, solver):
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
        return self._raycast_closest_rays([origin], [dir], [distance])[0]

    def raycast_closest_batch(self, origins, dirs, distances):
        return self._raycast_closest_rays(origins, dirs, distances)

    def _raycast_closest_rays(self, origins, dirs, distances):
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

    def is_joint_break_event(self, event):
        return isinstance(event, _NewtonJointBreakEvent)

    def decode_joint_break_event(self, event):
        return event.joint_path
