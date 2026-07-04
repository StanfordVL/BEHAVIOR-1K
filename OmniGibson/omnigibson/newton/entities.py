"""Newton-backed runtime entity handles."""

from dataclasses import dataclass, field
from functools import cached_property
import math
from pathlib import Path
from typing import Any

import newton
import torch as th
import warp as wp

from omnigibson.runtime.entity import SimBody, SimEntity, SimJoint, SimShape


@dataclass
class NewtonBody(SimBody):
    """Runtime handle for one Newton body."""

    simulator: Any = field(compare=False, repr=False)

    def get_pose(self):
        return _array_item(self.simulator.state_0.body_q, self.index)

    def get_velocity(self):
        return _array_item(self.simulator.state_0.body_qd, self.index)

    def get_position_orientation(self, frame="world", clone=True):
        if frame != "world":
            raise NotImplementedError("Newton bodies currently support world-frame pose queries only.")
        pose = _array_to_tensor(self.simulator.state_0.body_q)
        if pose is None or self.index >= len(pose):
            pos = th.zeros(3)
            quat = th.tensor([0.0, 0.0, 0.0, 1.0])
        else:
            pos = pose[self.index, :3]
            quat = pose[self.index, 3:7]
        return (pos.detach().cpu().clone(), quat.detach().cpu().clone()) if clone else (pos, quat)

    def get_linear_velocity(self, clone=True):
        vel = _array_to_tensor(self.simulator.state_0.body_qd)
        result = th.zeros(3) if vel is None or self.index >= len(vel) else vel[self.index, :3]
        return result.detach().cpu().clone() if clone else result

    def get_angular_velocity(self, clone=True):
        vel = _array_to_tensor(self.simulator.state_0.body_qd)
        result = th.zeros(3) if vel is None or self.index >= len(vel) else vel[self.index, 3:6]
        return result.detach().cpu().clone() if clone else result


@dataclass
class NewtonJoint(SimJoint):
    """Runtime handle for one Newton joint."""

    simulator: Any = field(compare=False, repr=False)

    def get_position(self):
        model = self.simulator.model
        state = self.simulator.state_0
        q = getattr(state, "joint_q", None) or model.joint_q
        q_start = _array_item(model.joint_q_start, self.index)
        q_stop = _next_start_or_length(model.joint_q_start, self.index, q)
        return _array_slice(q, q_start, q_stop)

    def get_velocity(self):
        model = self.simulator.model
        state = self.simulator.state_0
        qd = getattr(state, "joint_qd", None) or model.joint_qd
        qd_start = _array_item(model.joint_qd_start, self.index)
        qd_stop = _next_start_or_length(model.joint_qd_start, self.index, qd)
        return _array_slice(qd, qd_start, qd_stop)

    @property
    def joint_type(self):
        from omnigibson.utils.constants import JointType

        name = self.name.split("/")[-1].lower()
        if "finger" in name or "lift" in name or "slide" in name:
            return JointType.JOINT_PRISMATIC
        return JointType.JOINT_REVOLUTE

    @property
    def lower_limit(self):
        qd_index = _array_item(self.simulator.model.joint_qd_start, self.index, 0)
        limits = _array_to_tensor(getattr(self.simulator.model, "joint_limit_lower", None))
        if limits is None or qd_index >= len(limits):
            return -float("inf")
        return _tensor_or_scalar_to_float(limits[qd_index])

    @property
    def upper_limit(self):
        qd_index = _array_item(self.simulator.model.joint_qd_start, self.index, 0)
        limits = _array_to_tensor(getattr(self.simulator.model, "joint_limit_upper", None))
        if limits is None or qd_index >= len(limits):
            return float("inf")
        return _tensor_or_scalar_to_float(limits[qd_index])


@dataclass
class NewtonShape(SimShape):
    """Runtime handle for one Newton shape."""


@dataclass
class NewtonEntity(SimEntity):
    """Runtime handle for an imported Newton asset instance."""

    simulator: Any = field(compare=False, repr=False)
    _scale: th.Tensor = field(default_factory=lambda: th.ones(3), init=False, repr=False)

    @cached_property
    def bodies(self) -> dict[str, NewtonBody]:
        model = self.simulator.model
        result = {}
        for index in self.body_indices:
            name = _unique_name(_safe_label(model.body_label, index, f"body_{index}"), result)
            result[name] = NewtonBody(name=name, index=index, simulator=self.simulator)
        return result

    @cached_property
    def links(self):
        return {_simple_label(name): body for name, body in self.bodies.items()}

    @cached_property
    def joints(self) -> dict[str, NewtonJoint]:
        model = self.simulator.model
        result = {}
        for index in self.joint_indices:
            name = _unique_name(_safe_label(model.joint_label, index, f"joint_{index}"), result)
            result[name] = NewtonJoint(name=name, index=index, simulator=self.simulator)
        return result

    @cached_property
    def shapes(self) -> dict[str, NewtonShape]:
        model = self.simulator.model
        shape_body = _array_to_list(model.shape_body)
        result = {}
        for index in self.shape_indices:
            name = _unique_name(_safe_label(model.shape_label, index, f"shape_{index}"), result)
            body_index = shape_body[index] if index < len(shape_body) and shape_body[index] >= 0 else None
            result[name] = NewtonShape(name=name, index=index, body_index=body_index)
        return result

    @property
    def prim_path(self):
        return f"/{self.name}"

    @property
    def root_link(self):
        return next(iter(self.links.values()), None)

    @property
    def n_dof(self):
        return len(self._joint_dofs)

    @property
    def n_joints(self):
        return self.n_dof

    @property
    def joint_names(self):
        return [item["name"] for item in self._joint_dofs]

    @property
    def control_limits(self):
        lower, upper = self._joint_limits()
        return {
            "position": (lower, upper),
            "velocity": (-th.ones(self.n_dof) * 10.0, th.ones(self.n_dof) * 10.0),
        }

    @property
    def scale(self):
        return self._scale.clone()

    @scale.setter
    def scale(self, value):
        value = _to_scale_tensor(value)
        if th.any(value <= 0):
            raise ValueError(f"Scale must be positive, got {value}.")
        ratio = value / self._scale
        self._scale_shape_array("shape_scale", ratio)
        self._scale_shape_array("shape_collision_aabb_lower", ratio)
        self._scale_shape_array("shape_collision_aabb_upper", ratio)
        self._scale = value

    def set_loaded_scale(self, value):
        """Record scale that was already baked into the imported Newton model."""
        value = _to_scale_tensor(value)
        if th.any(value <= 0):
            raise ValueError(f"Scale must be positive, got {value}.")
        self._scale = value

    @property
    def aabb_extent(self):
        lower, upper = self._world_aabb()
        return upper - lower

    @property
    def aabb_center(self):
        lower, upper = self._world_aabb()
        return (lower + upper) * 0.5

    def get_position_orientation(self, frame="world"):
        if frame != "world":
            raise NotImplementedError("NewtonEntity currently supports world-frame pose queries only.")
        pose = self._root_pose_tensor()
        if pose is None:
            return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])
        return pose[:3].detach().cpu().clone(), pose[3:7].detach().cpu().clone()

    def set_position_orientation(self, position=None, orientation=None, frame="world"):
        if frame != "world":
            raise NotImplementedError("NewtonEntity currently supports world-frame pose updates only.")
        if not self.body_indices:
            return

        body_q = _array_to_tensor(self.simulator.state_0.body_q)
        if body_q is None:
            return

        root_idx = self.body_indices[0]
        old_root_pos = body_q[root_idx, :3].clone()
        new_root_pos = _to_tensor3(position, device=old_root_pos.device) if position is not None else old_root_pos
        delta = new_root_pos - old_root_pos

        for body_idx in self.body_indices:
            body_q[body_idx, :3] += delta
        if orientation is not None:
            body_q[root_idx, 3:7] = _to_tensor4(orientation, device=body_q.device)

        self._set_root_joint_pose(new_root_pos, orientation)

        # Keep the paired output state coherent for callers that inspect it before the next step.
        other_body_q = _array_to_tensor(self.simulator.state_1.body_q)
        if other_body_q is not None and len(other_body_q) == len(body_q):
            for body_idx in self.body_indices:
                other_body_q[body_idx, :3] += delta
            if orientation is not None:
                other_body_q[root_idx, 3:7] = _to_tensor4(orientation, device=other_body_q.device)

    def get_joint_positions(self, normalized=False):
        q = _array_to_tensor(getattr(self.simulator.state_0, "joint_q", None))
        if q is None:
            q = _array_to_tensor(getattr(self.simulator.model, "joint_q", None))
        if q is None or self.n_dof == 0:
            return th.zeros(0)
        values = th.stack([q[item["q_index"]] for item in self._joint_dofs]).detach().cpu()
        if normalized:
            lower, upper = self._joint_limits(device=values.device)
            span = th.clamp(upper - lower, min=1.0e-6)
            values = 2.0 * (values - lower) / span - 1.0
        return values

    def get_joint_velocities(self):
        qd = _array_to_tensor(getattr(self.simulator.state_0, "joint_qd", None))
        if qd is None or self.n_dof == 0:
            return th.zeros(0)
        return th.stack([qd[item["target_index"]] for item in self._joint_dofs]).detach().cpu()

    def set_joint_positions(self, positions, indices=None, normalized=False, drive=False):
        if self.n_dof == 0:
            return

        positions = _to_1d_tensor(positions)
        indices = _normalize_dof_indices(indices, self.n_dof)
        if positions.numel() != len(indices):
            raise ValueError(f"Expected {len(indices)} joint positions, got {positions.numel()}.")

        lower, upper = self._joint_limits()
        if normalized:
            positions = 0.5 * (positions + 1.0) * (upper[list(indices)] - lower[list(indices)]) + lower[list(indices)]
        positions = th.clamp(positions, lower[list(indices)], upper[list(indices)])

        for owner in (self.simulator.model, self.simulator.state_0, self.simulator.state_1):
            q = _array_to_tensor(getattr(owner, "joint_q", None))
            if q is None:
                continue
            values = positions.to(device=q.device)
            for local_idx, value in zip(indices, values):
                q[self._joint_dofs[local_idx]["q_index"]] = value

            qd = _array_to_tensor(getattr(owner, "joint_qd", None))
            if qd is not None:
                for local_idx in indices:
                    qd[self._joint_dofs[local_idx]["target_index"]] = 0.0

        if drive:
            target = _array_to_tensor(getattr(self.simulator.control, "joint_target_pos", None))
            if target is not None:
                values = positions.to(device=target.device)
                for local_idx, value in zip(indices, values):
                    target[self._joint_dofs[local_idx]["target_index"]] = value

        newton.eval_fk(
            self.simulator.model,
            self.simulator.state_0.joint_q,
            self.simulator.state_0.joint_qd,
            self.simulator.state_0,
        )
        newton.eval_fk(
            self.simulator.model,
            self.simulator.state_1.joint_q,
            self.simulator.state_1.joint_qd,
            self.simulator.state_1,
        )

    def set_joint_velocities(self, velocities, indices=None, drive=False):
        if self.n_dof == 0:
            return

        velocities = _to_1d_tensor(velocities)
        indices = _normalize_dof_indices(indices, self.n_dof)
        if velocities.numel() != len(indices):
            raise ValueError(f"Expected {len(indices)} joint velocities, got {velocities.numel()}.")

        for owner in (self.simulator.model, self.simulator.state_0, self.simulator.state_1):
            qd = _array_to_tensor(getattr(owner, "joint_qd", None))
            if qd is None:
                continue
            values = velocities.to(device=qd.device)
            for local_idx, value in zip(indices, values):
                qd[self._joint_dofs[local_idx]["target_index"]] = value

        if drive:
            target = _array_to_tensor(getattr(self.simulator.control, "joint_target_vel", None))
            if target is not None:
                values = velocities.to(device=target.device)
                for local_idx, value in zip(indices, values):
                    target[self._joint_dofs[local_idx]["target_index"]] = value

    def keep_still(self):
        body_qd = _array_to_tensor(self.simulator.state_0.body_qd)
        if body_qd is not None:
            for body_idx in self.body_indices:
                body_qd[body_idx] = 0.0
        other_body_qd = _array_to_tensor(self.simulator.state_1.body_qd)
        if body_qd is not None and other_body_qd is not None and len(other_body_qd) == len(body_qd):
            for body_idx in self.body_indices:
                other_body_qd[body_idx] = 0.0

    def _root_pose_tensor(self):
        if not self.body_indices:
            return None
        body_q = _array_to_tensor(self.simulator.state_0.body_q)
        if body_q is None:
            return None
        return body_q[self.body_indices[0]]

    def _scale_shape_array(self, attr_name, ratio):
        array = getattr(self.simulator.model, attr_name, None)
        values = _array_to_tensor(array)
        if values is None:
            return
        ratio = ratio.to(device=values.device)
        for shape_idx in self.shape_indices:
            values[shape_idx] *= ratio

    def _set_root_joint_pose(self, position, orientation):
        root_joint_idx = self._root_joint_index()
        if root_joint_idx is None:
            return

        model = self.simulator.model
        q_starts = _array_to_list(model.joint_q_start)
        qd_starts = _array_to_list(model.joint_qd_start)
        if root_joint_idx + 1 >= len(q_starts):
            return

        q_start = q_starts[root_joint_idx]
        q_stop = q_starts[root_joint_idx + 1]
        if q_stop - q_start != 7:
            return

        joint_owners = [model, self.simulator.state_0, self.simulator.state_1]
        for owner in joint_owners:
            q = _array_to_tensor(getattr(owner, "joint_q", None))
            if q is None:
                continue
            q[q_start : q_start + 3] = position.to(device=q.device)
            if orientation is not None:
                q[q_start + 3 : q_start + 7] = _to_tensor4(orientation, device=q.device)

            qd = _array_to_tensor(getattr(owner, "joint_qd", None))
            if qd is not None and root_joint_idx + 1 < len(qd_starts):
                qd_start = qd_starts[root_joint_idx]
                qd_stop = qd_starts[root_joint_idx + 1]
                qd[qd_start:qd_stop] = 0.0

        newton.eval_fk(model, self.simulator.state_0.joint_q, self.simulator.state_0.joint_qd, self.simulator.state_0)
        newton.eval_fk(model, self.simulator.state_1.joint_q, self.simulator.state_1.joint_qd, self.simulator.state_1)

    def _root_joint_index(self):
        if not self.body_indices:
            return None
        root_idx = self.body_indices[0]
        joint_child = _array_to_list(getattr(self.simulator.model, "joint_child", []))
        for joint_idx in self.joint_indices:
            if joint_idx < len(joint_child) and joint_child[joint_idx] == root_idx:
                return joint_idx
        return None

    def _world_aabb(self):
        lower = _array_to_tensor(getattr(self.simulator.model, "shape_collision_aabb_lower", None))
        upper = _array_to_tensor(getattr(self.simulator.model, "shape_collision_aabb_upper", None))
        shape_transform = _array_to_tensor(getattr(self.simulator.model, "shape_transform", None))
        shape_body = _array_to_list(getattr(self.simulator.model, "shape_body", []))
        body_q = _array_to_tensor(self.simulator.state_0.body_q)
        if lower is None or upper is None or body_q is None or not self.shape_indices:
            pos, _ = self.get_position_orientation()
            return pos - 0.5, pos + 0.5

        lower = lower.to(device=body_q.device)
        upper = upper.to(device=body_q.device)
        if shape_transform is not None:
            shape_transform = shape_transform.to(device=body_q.device)
        points = []
        for shape_idx in self.shape_indices:
            if shape_idx >= len(lower) or shape_idx >= len(upper):
                continue
            corners = _aabb_corners(lower[shape_idx], upper[shape_idx])
            if shape_transform is not None and shape_idx < len(shape_transform):
                shape_tf = shape_transform[shape_idx]
                corners = _rotate_vectors_xyzw(corners, shape_tf[3:7]) + shape_tf[:3]
            body_idx = shape_body[shape_idx] if shape_idx < len(shape_body) else -1
            if body_idx >= 0 and body_idx < len(body_q):
                body_pose = body_q[body_idx]
                corners = _rotate_vectors_xyzw(corners, body_pose[3:7]) + body_pose[:3]
            points.append(corners)

        if not points:
            pos, _ = self.get_position_orientation()
            return pos - 0.5, pos + 0.5
        stacked = th.cat(points, dim=0)
        return stacked.min(dim=0).values.detach().cpu(), stacked.max(dim=0).values.detach().cpu()

    @cached_property
    def _joint_dofs(self):
        model = self.simulator.model
        q_starts = _array_to_list(getattr(model, "joint_q_start", []))
        qd_starts = _array_to_list(getattr(model, "joint_qd_start", []))
        labels = _array_to_list(getattr(model, "joint_label", []))
        lower = _array_to_list(getattr(model, "joint_limit_lower", []))

        dofs = []
        for joint_idx in self.joint_indices:
            if joint_idx + 1 >= len(q_starts) or joint_idx + 1 >= len(qd_starts):
                continue
            if q_starts[joint_idx + 1] - q_starts[joint_idx] != 1:
                continue
            if qd_starts[joint_idx + 1] - qd_starts[joint_idx] != 1:
                continue

            target_index = qd_starts[joint_idx]
            if target_index >= len(lower):
                continue
            label = labels[joint_idx] if joint_idx < len(labels) else f"joint_{joint_idx}"
            dofs.append(
                {
                    "joint_index": joint_idx,
                    "q_index": q_starts[joint_idx],
                    "target_index": target_index,
                    "name": str(label).split("/")[-1],
                }
            )
        return tuple(dofs)

    def _joint_limits(self, indices=None, device=None):
        lower = _array_to_tensor(getattr(self.simulator.model, "joint_limit_lower", None))
        upper = _array_to_tensor(getattr(self.simulator.model, "joint_limit_upper", None))
        if lower is None or upper is None:
            lower = th.full((self.n_dof,), -float("inf"), dtype=th.float32)
            upper = th.full((self.n_dof,), float("inf"), dtype=th.float32)
        else:
            local_indices = tuple(range(self.n_dof)) if indices is None else tuple(indices)
            target_indices = [self._joint_dofs[idx]["target_index"] for idx in local_indices]
            lower = lower[target_indices]
            upper = upper[target_indices]
        if device is not None:
            lower = lower.to(device=device)
            upper = upper.to(device=device)
        return lower.detach().cpu() if device is None else lower, upper.detach().cpu() if device is None else upper


@dataclass
class NewtonRobotEntity(NewtonEntity):
    """Newton-backed robot adapter with minimal joint-position control.

    This intentionally bypasses legacy OmniGibson controller classes, which are
    tied to Isaac articulation views. The runtime command surface is compatible
    with the current robot control example, but commands are applied directly to
    Newton/MuJoCo joint target buffers.
    """

    default_joint_pos: tuple[float, ...] | None = None
    action_normalize: bool = True
    wheel_radius: float | None = None
    wheel_axle_length: float | None = None
    eef_link_names: tuple[str, ...] = ()
    _controller_config: dict = field(default_factory=dict, init=False, repr=False)
    _controllers: dict = field(default_factory=dict, init=False, repr=False)
    _control_joint_cache: tuple[dict, ...] | None = field(default=None, init=False, repr=False)
    _joint_group_cache: dict[str, tuple[int, ...]] | None = field(default=None, init=False, repr=False)
    _action_space: Any = field(default=None, init=False, repr=False)
    _last_joint_target: th.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.reload_controllers()

    @property
    def n_dof(self):
        return len(self._control_joints)

    @property
    def action_dim(self):
        return sum(controller.command_dim for controller in self._controllers.values())

    @property
    def action_space(self):
        if self._action_space is None:
            self._action_space = _NewtonBoxActionSpace(self.action_dim)
        return self._action_space

    @property
    def controllers(self):
        return self._controllers

    @property
    def controller_order(self):
        return list(self._controllers)

    @property
    def _default_controller_config(self):
        config = {}
        groups = self._joint_groups
        if groups.get("base"):
            config["base"] = {
                "DifferentialDriveController": {"name": "DifferentialDriveController"},
                "JointController": {"name": "JointController"},
                "NullJointController": {"name": "NullJointController"},
            }
        if groups.get("trunk"):
            config["trunk"] = {
                "JointController": {"name": "JointController"},
                "NullJointController": {"name": "NullJointController"},
            }
        if groups.get("arm_0"):
            config["arm_0"] = {
                "JointController": {"name": "JointController"},
                "InverseKinematicsController": {"name": "InverseKinematicsController"},
                "OperationalSpaceController": {"name": "OperationalSpaceController"},
                "NullJointController": {"name": "NullJointController"},
            }
        if groups.get("gripper_0"):
            config["gripper_0"] = {
                "MultiFingerGripperController": {"name": "MultiFingerGripperController"},
                "JointController": {"name": "JointController"},
                "NullJointController": {"name": "NullJointController"},
            }
        if groups.get("camera"):
            config["camera"] = {
                "JointController": {"name": "JointController"},
                "NullJointController": {"name": "NullJointController"},
            }
        return config

    @cached_property
    def links(self):
        return {_simple_label(name): body for name, body in self.bodies.items()}

    @cached_property
    def joints(self) -> dict[str, NewtonJoint]:
        result = {}
        for item in self._control_joints:
            name = _unique_name(item["name"], result)
            result[name] = NewtonJoint(name=name, index=item["joint_index"], simulator=self.simulator)
        return result

    @property
    def root_link(self):
        return next(iter(self.links.values()), None)

    @property
    def joint_names(self):
        return [item["name"] for item in self._control_joints]

    @property
    def control_limits(self):
        lower, upper = self._joint_limits()
        return {
            "position": (lower, upper),
            "velocity": (-th.ones(self.n_dof) * 10.0, th.ones(self.n_dof) * 10.0),
        }

    def reload_controllers(self, controller_config=None):
        controller_config = controller_config or self._default_controller_selection()
        self._controller_config = controller_config

        self._controllers = {}
        command_start = 0
        for group, cfg in controller_config.items():
            dof_idx = self._joint_groups.get(group, ())
            controller_type = cfg.get("name", "JointController")
            command_dim = self._controller_command_dim(controller_type, dof_idx)
            controller_defaults = self._controller_defaults(group, controller_type)
            self._controllers[group] = _NewtonControllerGroup(
                controller_type=controller_type,
                dof_idx=tuple(dof_idx),
                command_dim=command_dim,
                command_start=command_start,
                motor_type=cfg.get("motor_type", controller_defaults["motor_type"]),
                use_delta_commands=bool(cfg.get("use_delta_commands", controller_defaults["use_delta_commands"])),
                mode=cfg.get("mode", controller_defaults["mode"]),
                inverted=bool(cfg.get("inverted", controller_defaults["inverted"])),
                position_kp=float(cfg.get("position_kp", controller_defaults["position_kp"])),
                position_kd=float(cfg.get("position_kd", controller_defaults["position_kd"])),
                velocity_kd=float(cfg.get("velocity_kd", controller_defaults["velocity_kd"])),
            )
            command_start += command_dim

        self._action_space = _NewtonBoxActionSpace(self.action_dim)
        self._ensure_joint_drives()
        self._sync_targets_to_current_position()

    def apply_action(self, action):
        if action is None:
            return
        if isinstance(action, dict):
            action = action.get(self.name, next(iter(action.values()), None))
            if action is None:
                return

        action = _to_1d_tensor(action)
        if action.numel() != self.action_dim:
            raise ValueError(f"Action for {self.name} must have dimension {self.action_dim}, got {action.numel()}.")

        for controller in self._controllers.values():
            if controller.command_dim == 0:
                continue
            command = action[controller.command_start : controller.command_start + controller.command_dim]
            self._apply_controller_command(controller, command)

    def _action_to_joint_target(self, action, indices=None, *, use_delta_commands=False):
        # The legacy JointController has an explicit no-op path. Preserve that
        # behavior here so zero actions hold the previous target instead of
        # mapping to the middle of the normalized joint range.
        if th.allclose(action, th.zeros_like(action)):
            if self._last_joint_target is not None:
                if indices is None:
                    return self._last_joint_target.clone()
                return self._last_joint_target[list(indices)].clone()
            current = self.get_joint_positions()
            return current if indices is None else current[list(indices)]

        lower, upper = self._joint_limits(indices=indices)
        if use_delta_commands:
            current = self.get_joint_positions()
            current = current if indices is None else current[list(indices)]
            target = current + th.clamp(action, -1.0, 1.0)
        elif self.action_normalize:
            target = 0.5 * (action + 1.0) * (upper - lower) + lower
        else:
            target = action
        return th.clamp(target, lower, upper)

    def _set_joint_position_targets(self, positions, indices=None):
        positions = _to_1d_tensor(positions)
        indices = tuple(range(self.n_dof)) if indices is None else tuple(indices)
        if positions.numel() != len(indices):
            raise ValueError(f"Expected {len(indices)} target positions, got {positions.numel()}.")

        lower, upper = self._joint_limits(indices=indices)
        positions = th.clamp(positions, lower, upper)
        target = _array_to_tensor(getattr(self.simulator.control, "joint_target_pos", None))
        if target is not None:
            values = positions.to(device=target.device)
            for local_idx, value in zip(indices, values):
                target[self._control_joints[local_idx]["target_index"]] = value
        if self._last_joint_target is None or self._last_joint_target.numel() != self.n_dof:
            self._last_joint_target = self.get_joint_positions()
        self._last_joint_target[list(indices)] = positions.detach().cpu()

    def reset(self):
        if self.default_joint_pos is not None and len(self.default_joint_pos) == self.n_dof:
            self.set_joint_positions(th.tensor(self.default_joint_pos, dtype=th.float32), drive=True)
        else:
            self._sync_targets_to_current_position()
        self.keep_still()

    def get_joint_positions(self, normalized=False):
        q = _array_to_tensor(getattr(self.simulator.state_0, "joint_q", None))
        if q is None:
            q = _array_to_tensor(getattr(self.simulator.model, "joint_q", None))
        if q is None or self.n_dof == 0:
            return th.zeros(0)
        values = th.stack([q[item["q_index"]] for item in self._control_joints]).detach().cpu()
        if normalized:
            lower, upper = self._joint_limits(device=values.device)
            span = th.clamp(upper - lower, min=1.0e-6)
            values = 2.0 * (values - lower) / span - 1.0
        return values

    def get_joint_velocities(self):
        qd = _array_to_tensor(getattr(self.simulator.state_0, "joint_qd", None))
        if qd is None or self.n_dof == 0:
            return th.zeros(0)
        return th.stack([qd[item["target_index"]] for item in self._control_joints]).detach().cpu()

    def set_joint_velocities(self, velocities, indices=None, drive=False):
        if self.n_dof == 0:
            return

        velocities = _to_1d_tensor(velocities)
        if indices is None:
            indices = tuple(range(self.n_dof))
        elif isinstance(indices, th.Tensor):
            indices = tuple(int(i) for i in indices.detach().cpu().tolist())
        else:
            indices = tuple(int(i) for i in indices)

        if velocities.numel() != len(indices):
            raise ValueError(f"Expected {len(indices)} joint velocities, got {velocities.numel()}.")

        owners = [self.simulator.model, self.simulator.state_0, self.simulator.state_1]
        for owner in owners:
            qd = _array_to_tensor(getattr(owner, "joint_qd", None))
            if qd is None:
                continue
            values = velocities.to(device=qd.device)
            for local_idx, value in zip(indices, values):
                qd[self._control_joints[local_idx]["target_index"]] = value

        if drive:
            self._set_joint_velocity_targets(velocities, indices)

    def set_joint_positions(self, positions, indices=None, normalized=False, drive=False):
        if self.n_dof == 0:
            return

        positions = _to_1d_tensor(positions)
        if indices is None:
            indices = tuple(range(self.n_dof))
        elif isinstance(indices, th.Tensor):
            indices = tuple(int(i) for i in indices.detach().cpu().tolist())
        else:
            indices = tuple(int(i) for i in indices)

        if positions.numel() != len(indices):
            raise ValueError(f"Expected {len(indices)} joint positions, got {positions.numel()}.")

        lower, upper = self._joint_limits()
        if normalized:
            positions = 0.5 * (positions + 1.0) * (upper[list(indices)] - lower[list(indices)]) + lower[list(indices)]
        positions = th.clamp(positions, lower[list(indices)], upper[list(indices)])

        owners = [self.simulator.model, self.simulator.state_0, self.simulator.state_1]
        for owner in owners:
            q = _array_to_tensor(getattr(owner, "joint_q", None))
            if q is None:
                continue
            values = positions.to(device=q.device)
            for local_idx, value in zip(indices, values):
                q[self._control_joints[local_idx]["q_index"]] = value

            qd = _array_to_tensor(getattr(owner, "joint_qd", None))
            if qd is not None:
                for local_idx in indices:
                    qd[self._control_joints[local_idx]["target_index"]] = 0.0

        if drive:
            target = _array_to_tensor(getattr(self.simulator.control, "joint_target_pos", None))
            if target is not None:
                values = positions.to(device=target.device)
                for local_idx, value in zip(indices, values):
                    target[self._control_joints[local_idx]["target_index"]] = value
            if len(indices) == self.n_dof:
                self._last_joint_target = positions.detach().cpu().clone()

        newton.eval_fk(
            self.simulator.model,
            self.simulator.state_0.joint_q,
            self.simulator.state_0.joint_qd,
            self.simulator.state_0,
        )
        newton.eval_fk(
            self.simulator.model,
            self.simulator.state_1.joint_q,
            self.simulator.state_1.joint_qd,
            self.simulator.state_1,
        )

    @property
    def _control_joints(self):
        if self._control_joint_cache is None:
            self._control_joint_cache = self._find_control_joints()
        return self._control_joint_cache

    @property
    def _joint_groups(self):
        if self._joint_group_cache is None:
            self._joint_group_cache = self._infer_joint_groups()
        return self._joint_group_cache

    def _find_control_joints(self):
        model = self.simulator.model
        q_starts = _array_to_list(getattr(model, "joint_q_start", []))
        qd_starts = _array_to_list(getattr(model, "joint_qd_start", []))
        labels = _array_to_list(getattr(model, "joint_label", []))
        lower = _array_to_list(getattr(model, "joint_limit_lower", []))

        control_joints = []
        for joint_idx in self.joint_indices:
            if joint_idx + 1 >= len(q_starts) or joint_idx + 1 >= len(qd_starts):
                continue
            if q_starts[joint_idx + 1] - q_starts[joint_idx] != 1:
                continue
            if qd_starts[joint_idx + 1] - qd_starts[joint_idx] != 1:
                continue

            target_index = qd_starts[joint_idx]
            if target_index >= len(lower):
                continue
            label = labels[joint_idx] if joint_idx < len(labels) else f"joint_{joint_idx}"
            control_joints.append(
                {
                    "joint_index": joint_idx,
                    "q_index": q_starts[joint_idx],
                    "target_index": target_index,
                    "name": str(label).split("/")[-1],
                }
            )
        return tuple(control_joints)

    def _infer_joint_groups(self):
        groups = {"base": [], "trunk": [], "arm_0": [], "gripper_0": [], "camera": []}
        for idx, item in enumerate(self._control_joints):
            name = item["name"].lower()
            if "wheel" in name or name.startswith("base_"):
                group = "base"
            elif "head" in name or "camera" in name:
                group = "camera"
            elif "gripper" in name or "finger" in name:
                group = "gripper_0"
            elif "torso" in name or "trunk" in name:
                group = "trunk"
            elif any(token in name for token in ("shoulder", "upperarm", "elbow", "forearm", "wrist", "arm")):
                group = "arm_0"
            else:
                group = "arm_0"
            groups[group].append(idx)
        return {group: tuple(indices) for group, indices in groups.items()}

    def _default_controller_selection(self):
        defaults = {}
        for group, options in self._default_controller_config.items():
            defaults[group] = {"name": next(iter(options))}
        return defaults

    def _controller_command_dim(self, controller_type, dof_idx):
        if controller_type == "NullJointController" or not dof_idx:
            return 0
        if controller_type == "DifferentialDriveController":
            return 2
        if controller_type in {"InverseKinematicsController", "OperationalSpaceController"}:
            return 6
        if controller_type == "MultiFingerGripperController":
            return 1 if len(dof_idx) > 1 else len(dof_idx)
        if controller_type == "HolonomicBaseJointController":
            return min(3, len(dof_idx))
        return len(dof_idx)

    def _controller_defaults(self, group, controller_type):
        if controller_type == "DifferentialDriveController":
            return _controller_defaults(
                motor_type="velocity",
                use_delta_commands=False,
                position_kp=0.0,
                position_kd=0.0,
                velocity_kd=60.0,
            )
        if controller_type in {"InverseKinematicsController", "OperationalSpaceController"}:
            return _controller_defaults(
                motor_type="position",
                use_delta_commands=True,
                position_kp=1200.0,
                position_kd=120.0,
            )
        if controller_type == "MultiFingerGripperController":
            return _controller_defaults(
                motor_type="position",
                use_delta_commands=False,
                mode="binary",
                position_kp=800.0,
                position_kd=80.0,
            )
        if controller_type == "HolonomicBaseJointController":
            return _controller_defaults(
                motor_type="velocity",
                use_delta_commands=False,
                position_kp=0.0,
                position_kd=0.0,
                velocity_kd=60.0,
            )
        if controller_type == "JointController":
            if group == "base":
                return _controller_defaults(
                    motor_type="velocity",
                    use_delta_commands=False,
                    position_kp=0.0,
                    position_kd=0.0,
                    velocity_kd=80.0,
                )
            if group == "gripper_0":
                # Newton/MuJoCo accepts pure velocity targets for imported USD
                # finger joints, but the Panda prismatic finger drives barely
                # move in that mode. Use small position deltas to preserve the
                # interactive JointController behavior from OG teleop.
                return _controller_defaults(
                    motor_type="position",
                    use_delta_commands=True,
                    position_kp=800.0,
                    position_kd=80.0,
                )
            return _controller_defaults(
                motor_type="position",
                use_delta_commands=True,
                position_kp=1200.0,
                position_kd=120.0,
            )
        return _controller_defaults()

    def _apply_controller_command(self, controller, command):
        if controller.controller_type == "DifferentialDriveController":
            self._apply_differential_drive(command, controller.dof_idx)
        elif controller.controller_type in {"InverseKinematicsController", "OperationalSpaceController"}:
            self._apply_cartesian_delta(command, controller)
        elif controller.controller_type == "MultiFingerGripperController" and controller.command_dim == 1:
            self._apply_gripper_scalar(command, controller)
        elif controller.controller_type != "NullJointController":
            if controller.motor_type == "velocity":
                self._set_joint_velocity_targets(
                    self._action_to_joint_velocity(command, indices=controller.dof_idx),
                    controller.dof_idx,
                )
            else:
                target = self._action_to_joint_target(
                    command,
                    indices=controller.dof_idx,
                    use_delta_commands=controller.use_delta_commands,
                )
                self._set_joint_position_targets(target, indices=controller.dof_idx)

    def _apply_differential_drive(self, command, indices):
        if len(indices) < 2:
            return
        cmd = _to_1d_tensor(command)
        if self.action_normalize:
            cmd = th.clamp(cmd, -1.0, 1.0)
        radius = self.wheel_radius or 0.05
        axle_length = self.wheel_axle_length or 0.3
        max_wheel_speed = self._max_joint_velocity(indices[:2])
        max_linear = max_wheel_speed * radius
        max_angular = max_linear * 2.0 / axle_length
        linear = cmd[0] * max_linear
        angular = (cmd[1] if cmd.numel() > 1 else th.tensor(0.0)) * max_angular
        half_axle = axle_length / 2.0
        velocities = th.tensor(
            [
                (linear - angular * half_axle) / radius,
                (linear + angular * half_axle) / radius,
            ],
            dtype=th.float32,
        )
        self._set_joint_velocity_targets(velocities, indices[:2])

    def _apply_cartesian_delta(self, command, controller):
        indices = controller.dof_idx
        if not indices:
            return
        cmd = _to_1d_tensor(command)
        if th.allclose(cmd, th.zeros_like(cmd)):
            return
        eef_body = self._eef_body()
        if eef_body is None:
            self._apply_arm_delta(command, indices)
            return

        pos_delta = th.clamp(cmd[:3], -1.0, 1.0) * 0.2
        if th.linalg.norm(pos_delta) <= 1.0e-8:
            return

        current = self.get_joint_positions()
        current_subset = current[list(indices)]
        base_pos = self._body_position(eef_body.index).detach().cpu()
        jacobian_cols = []
        eps = 1.0e-3
        for local_idx in indices:
            perturbed = current_subset.clone()
            column_idx = indices.index(local_idx)
            perturbed[column_idx] += eps
            self.set_joint_positions(perturbed, indices=indices, drive=False)
            jacobian_cols.append((self._body_position(eef_body.index).detach().cpu() - base_pos) / eps)

        self.set_joint_positions(current_subset, indices=indices, drive=False)
        jacobian = th.stack(jacobian_cols, dim=1)
        try:
            dq = th.linalg.pinv(jacobian) @ pos_delta
        except RuntimeError:
            self._apply_arm_delta(command, indices)
            return
        dq = th.clamp(dq, -0.12, 0.12)
        self._set_joint_position_targets(current_subset + dq, indices=indices)

    def _apply_arm_delta(self, command, indices):
        if not indices:
            return
        cmd = _to_1d_tensor(command)
        if th.allclose(cmd, th.zeros_like(cmd)):
            return
        count = min(cmd.numel(), len(indices))
        current = self.get_joint_positions()[list(indices[:count])]
        delta = th.clamp(cmd[:count], -1.0, 1.0) * 0.05
        target = current + delta
        self._set_joint_position_targets(target, indices=indices[:count])

    def _apply_gripper_scalar(self, command, controller):
        indices = controller.dof_idx
        if not indices:
            return
        command_value = float(_to_1d_tensor(command)[0])
        should_open = command_value >= 0.0
        if controller.inverted:
            should_open = not should_open
        lower, upper = self._joint_limits(indices=indices)
        target = upper if should_open else lower
        self._set_joint_position_targets(target, indices=indices)

    def _action_to_joint_velocity(self, action, indices=None):
        action = _to_1d_tensor(action)
        if th.allclose(action, th.zeros_like(action)):
            return th.zeros_like(action)
        lower, upper = self._joint_velocity_limits(indices=indices)
        if self.action_normalize:
            velocity = 0.5 * (th.clamp(action, -1.0, 1.0) + 1.0) * (upper - lower) + lower
        else:
            velocity = action
        return th.clamp(velocity, lower, upper)

    def _set_joint_velocity_targets(self, velocities, indices):
        target = _array_to_tensor(getattr(self.simulator.control, "joint_target_vel", None))
        if target is None:
            return
        values = _to_1d_tensor(velocities).to(device=target.device)
        for local_idx, value in zip(indices, values):
            target[self._control_joints[local_idx]["target_index"]] = value

    def _joint_limits(self, indices=None, device=None):
        lower = _array_to_tensor(getattr(self.simulator.model, "joint_limit_lower", None))
        upper = _array_to_tensor(getattr(self.simulator.model, "joint_limit_upper", None))
        if lower is None or upper is None:
            lower = th.full((self.n_dof,), -float("inf"), dtype=th.float32)
            upper = th.full((self.n_dof,), float("inf"), dtype=th.float32)
        else:
            local_indices = tuple(range(self.n_dof)) if indices is None else tuple(indices)
            target_indices = [self._control_joints[idx]["target_index"] for idx in local_indices]
            lower = lower[target_indices]
            upper = upper[target_indices]
        if device is not None:
            lower = lower.to(device=device)
            upper = upper.to(device=device)
        return lower.detach().cpu() if device is None else lower, upper.detach().cpu() if device is None else upper

    def _joint_velocity_limits(self, indices=None, device=None):
        local_indices = tuple(range(self.n_dof)) if indices is None else tuple(indices)
        limit = th.tensor([self._max_joint_velocity((idx,)) for idx in local_indices], dtype=th.float32)
        if device is not None:
            limit = limit.to(device=device)
        return -limit, limit

    def _max_joint_velocity(self, indices):
        upper = _array_to_tensor(getattr(self.simulator.model, "joint_limit_upper", None))
        lower = _array_to_tensor(getattr(self.simulator.model, "joint_limit_lower", None))
        if upper is None or lower is None:
            return 10.0
        values = []
        for local_idx in indices:
            target_index = self._control_joints[local_idx]["target_index"]
            if target_index < len(upper) and target_index < len(lower):
                bound = max(abs(float(lower[target_index])), abs(float(upper[target_index])))
                if math.isfinite(bound) and bound > 0:
                    values.append(min(bound, 30.0))
        return min(values) if values else 10.0

    def _eef_body(self):
        for name in self.eef_link_names:
            body = self.links.get(name)
            if body is not None:
                return body
        for fallback in ("eef_link", "gripper_link", "panda_hand", "wrist_roll_link"):
            body = self.links.get(fallback)
            if body is not None:
                return body
        arm_links = [
            body
            for name, body in self.links.items()
            if any(token in name.lower() for token in ("eef", "gripper", "wrist", "hand"))
        ]
        return arm_links[-1] if arm_links else None

    def _body_position(self, body_index):
        pose = _array_to_tensor(getattr(self.simulator.state_0, "body_q", None))
        if pose is None or body_index >= len(pose):
            return th.zeros(3)
        return pose[body_index, :3]

    def _ensure_joint_drives(self):
        ke = _array_to_tensor(getattr(self.simulator.model, "joint_target_ke", None))
        kd = _array_to_tensor(getattr(self.simulator.model, "joint_target_kd", None))
        mode = _array_to_tensor(getattr(self.simulator.model, "joint_target_mode", None))
        if ke is None:
            return

        controller_by_dof = {}
        for controller in self._controllers.values():
            for local_idx in controller.dof_idx:
                controller_by_dof[local_idx] = controller

        for local_idx, item in enumerate(self._control_joints):
            target_index = item["target_index"]
            if target_index >= len(ke):
                continue
            controller = controller_by_dof.get(local_idx)
            if controller is None:
                continue
            is_gripper = local_idx in self._joint_groups.get("gripper_0", ())
            is_velocity = controller.motor_type == "velocity"
            if mode is not None and target_index < len(mode):
                target_mode = newton.JointTargetMode.VELOCITY if is_velocity else newton.JointTargetMode.POSITION
                mode[target_index] = int(target_mode)
            if is_velocity:
                ke[target_index] = 0.0
                if kd is not None and target_index < len(kd):
                    kd[target_index] = controller.velocity_kd
            else:
                ke[target_index] = 800.0 if is_gripper else controller.position_kp
                if kd is not None and target_index < len(kd):
                    kd[target_index] = 80.0 if is_gripper else controller.position_kd

    def _sync_targets_to_current_position(self):
        target = _array_to_tensor(getattr(self.simulator.control, "joint_target_pos", None))
        q = _array_to_tensor(getattr(self.simulator.state_0, "joint_q", None))
        if target is None or q is None:
            return
        values = []
        for item in self._control_joints:
            value = q[item["q_index"]].to(device=target.device)
            target[item["target_index"]] = value
            values.append(value.detach().cpu())
        self._last_joint_target = th.stack(values) if values else th.zeros(0)


@dataclass(frozen=True)
class _NewtonControllerGroup:
    controller_type: str
    dof_idx: tuple[int, ...]
    command_dim: int
    command_start: int
    motor_type: str = "position"
    use_delta_commands: bool = False
    mode: str = "independent"
    inverted: bool = False
    position_kp: float = 1200.0
    position_kd: float = 120.0
    velocity_kd: float = 60.0


def _controller_defaults(
    *,
    motor_type="position",
    use_delta_commands=False,
    mode="independent",
    inverted=False,
    position_kp=1200.0,
    position_kd=120.0,
    velocity_kd=60.0,
):
    return {
        "motor_type": motor_type,
        "use_delta_commands": use_delta_commands,
        "mode": mode,
        "inverted": inverted,
        "position_kp": position_kp,
        "position_kd": position_kd,
        "velocity_kd": velocity_kd,
    }


class _NewtonBoxActionSpace:
    def __init__(self, action_dim):
        self.shape = (action_dim,)
        self.low = -th.ones(action_dim, dtype=th.float32).numpy()
        self.high = th.ones(action_dim, dtype=th.float32).numpy()

    def sample(self):
        return (th.rand(self.shape[0]) * 2.0 - 1.0).numpy()


def make_newton_entity(
    *,
    simulator,
    name: str,
    category: str,
    kind: str,
    source_path,
    before_counts: tuple[int, int, int],
    after_counts: tuple[int, int, int],
) -> NewtonEntity:
    """Create an entity from body/joint/shape count snapshots."""
    before_bodies, before_joints, before_shapes = before_counts
    after_bodies, after_joints, after_shapes = after_counts
    model = simulator.model
    before_bodies, after_bodies = _clamped_range_bounds(before_bodies, after_bodies, model.body_count)
    before_joints, after_joints = _clamped_range_bounds(before_joints, after_joints, model.joint_count)
    before_shapes, after_shapes = _clamped_range_bounds(before_shapes, after_shapes, model.shape_count)
    return NewtonEntity(
        name=name,
        category=category,
        kind=kind,
        source_path=Path(source_path).expanduser().resolve(),
        body_indices=tuple(range(before_bodies, after_bodies)),
        joint_indices=tuple(range(before_joints, after_joints)),
        shape_indices=tuple(range(before_shapes, after_shapes)),
        simulator=simulator,
    )


def make_newton_entity_from_labels(
    *,
    simulator,
    name: str,
    category: str,
    kind: str,
    source_path,
    label_prefix: str,
    default_joint_pos=None,
    controller_metadata=None,
    action_normalize=True,
) -> NewtonEntity:
    """Create an entity by matching finalized Newton labels and ownership arrays."""
    model = simulator.model
    body_indices = tuple(
        index for index, label in enumerate(model.body_label) if _label_matches_prefix(label, label_prefix)
    )
    body_index_set = set(body_indices)

    shape_body = _array_to_list(model.shape_body)
    shape_indices = []
    for index, label in enumerate(model.shape_label):
        owned_by_body = index < len(shape_body) and shape_body[index] in body_index_set
        if owned_by_body or _label_matches_prefix(label, label_prefix):
            shape_indices.append(index)

    joint_child = _array_to_list(model.joint_child)
    joint_indices = []
    for index, label in enumerate(model.joint_label):
        owned_by_body = index < len(joint_child) and joint_child[index] in body_index_set
        if owned_by_body or _label_matches_prefix(label, label_prefix):
            joint_indices.append(index)

    entity_cls = NewtonRobotEntity if kind == "robot" else NewtonEntity
    kwargs = {}
    if entity_cls is NewtonRobotEntity:
        controller_metadata = controller_metadata or {}
        kwargs = {
            "default_joint_pos": tuple(default_joint_pos) if default_joint_pos is not None else None,
            "action_normalize": action_normalize,
            "wheel_radius": controller_metadata.get("wheel_radius"),
            "wheel_axle_length": controller_metadata.get("wheel_axle_length"),
            "eef_link_names": tuple(controller_metadata.get("eef_link_names") or ()),
        }

    return entity_cls(
        name=name,
        category=category,
        kind=kind,
        source_path=Path(source_path).expanduser().resolve(),
        body_indices=body_indices,
        joint_indices=tuple(joint_indices),
        shape_indices=tuple(shape_indices),
        simulator=simulator,
        **kwargs,
    )


def _label_matches_prefix(label, prefix):
    label = str(label)
    return label == prefix or label.startswith(f"{prefix}/")


def _clamped_range_bounds(start, stop, count):
    start = min(start, count)
    stop = min(stop, count)
    if stop < start:
        stop = start
    return start, stop


def _array_item(array, index, default=None):
    values = _array_to_list(array)
    if index >= len(values):
        return default
    return values[index]


def _array_slice(array, start, stop):
    values = _array_to_list(array)
    return values[start:stop]


def _next_start_or_length(starts_array, index, values_array):
    starts = _array_to_list(starts_array)
    if index + 1 < len(starts):
        return starts[index + 1]
    return len(_array_to_list(values_array))


def _array_to_list(array):
    if isinstance(array, list):
        return array
    if isinstance(array, tuple):
        return list(array)
    if hasattr(array, "numpy"):
        try:
            return array.numpy().tolist()
        except RuntimeError:
            pass
    try:
        return wp.to_torch(array).detach().cpu().tolist()
    except Exception:
        return []


def _array_to_tensor(array):
    if array is None:
        return None
    if isinstance(array, th.Tensor):
        return array
    try:
        return wp.to_torch(array)
    except Exception:
        return None


def _to_tensor3(value, device=None):
    if isinstance(value, th.Tensor):
        result = value.detach().to(dtype=th.float32).reshape(3)
    else:
        result = th.tensor(value, dtype=th.float32).reshape(3)
    return result.to(device=device) if device is not None else result


def _to_tensor4(value, device=None):
    if isinstance(value, th.Tensor):
        result = value.detach().to(dtype=th.float32).reshape(4)
    else:
        result = th.tensor(value, dtype=th.float32).reshape(4)
    return result.to(device=device) if device is not None else result


def _to_1d_tensor(value):
    if isinstance(value, th.Tensor):
        return value.detach().to(dtype=th.float32).flatten()
    return th.tensor(value, dtype=th.float32).flatten()


def _normalize_dof_indices(indices, n_dof):
    if indices is None:
        return tuple(range(n_dof))
    if isinstance(indices, th.Tensor):
        return tuple(int(i) for i in indices.detach().cpu().tolist())
    return tuple(int(i) for i in indices)


def _tensor_or_scalar_to_float(value):
    if isinstance(value, th.Tensor):
        return value.detach().cpu().item()
    return value


def _to_scale_tensor(value):
    if isinstance(value, th.Tensor):
        value = value.detach().cpu().to(dtype=th.float32)
    elif isinstance(value, (int, float)):
        value = th.tensor([float(value)] * 3, dtype=th.float32)
    else:
        value = th.tensor(value, dtype=th.float32)
    if value.numel() == 1:
        value = value.repeat(3)
    return value.reshape(3)


def _aabb_corners(lower, upper):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    return th.stack(
        (
            th.stack((x0, y0, z0)),
            th.stack((x0, y0, z1)),
            th.stack((x0, y1, z0)),
            th.stack((x0, y1, z1)),
            th.stack((x1, y0, z0)),
            th.stack((x1, y0, z1)),
            th.stack((x1, y1, z0)),
            th.stack((x1, y1, z1)),
        )
    )


def _rotate_vectors_xyzw(vectors, quat):
    norm = th.linalg.norm(quat)
    if norm <= 0:
        return vectors
    quat = quat / norm
    q_xyz = quat[:3]
    q_w = quat[3]
    uv = th.cross(q_xyz.expand_as(vectors), vectors, dim=-1)
    uuv = th.cross(q_xyz.expand_as(vectors), uv, dim=-1)
    return vectors + 2.0 * (q_w * uv + uuv)


def _safe_label(labels, index, fallback):
    if index < len(labels) and labels[index]:
        return str(labels[index])
    return fallback


def _simple_label(name):
    return str(name).rstrip("/").split("/")[-1]


def _unique_name(name, existing):
    if name not in existing:
        return name
    suffix = 1
    while f"{name}_{suffix}" in existing:
        suffix += 1
    return f"{name}_{suffix}"
