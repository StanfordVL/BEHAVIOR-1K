import math

import numpy as np
import torch as th
from numba import jit

import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers.controller_base import BaseController, ControlType
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.python_utils import assert_valid_key, torch_compile
from omnigibson.utils.ui_utils import create_module_logger
from copy import deepcopy

# Create module logger
log = create_module_logger(module_name=__name__)

# Different modes
OSC_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,  # 6DOF (x,y,z,ax,ay,az) control of pose, whether both position and orientation is given in absolute coordinates
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}
OSC_MODES = set(OSC_MODE_COMMAND_DIMS.keys())


class OperationalSpaceController(BaseController):
    """
    Controller class to convert (delta or absolute) EEF commands into joint efforts using Operational Space Control

    This controller expects 6DOF delta commands (dx, dy, dz, dax, day, daz), where the delta orientation
    commands are in axis-angle form, and outputs low-level torque commands.

    Gains may also be considered part of the action space as well. In this case, the action space would be:
        (
            dx, dy, dz, dax, day, daz                       <-- 6DOF delta eef commands
            [, kpx, kpy, kpz, kpax, kpay, kpaz]             <-- kp gains
            [, drx dry, drz, drax, dray, draz]              <-- damping ratio gains
            [, kpnx, kpny, kpnz, kpnax, kpnay, kpnaz]       <-- kp null gains
        )

    Note that in this case, we ASSUME that the inputted gains are normalized to be in the range [-1, 1], and will
    be mapped appropriately to their respective ranges, as defined by XX_limits

    Alternatively, parameters (in this case, kp or damping_ratio) can either be set during initialization or provided
    from an external source; if the latter, the control_dict should include the respective parameter(s) as
    a part of its keys

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run OSC to back out joint efforts for a desired task frame command
        3. Clips the resulting command by the motor (effort) limits
    """

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        control_dim = len(config["dof_idx"])
        config["use_gravity_compensation"] = config.get("use_gravity_compensation", False)
        config["use_cc_compensation"] = config.get("use_cc_compensation", True)
        if config["use_gravity_compensation"]:
            log.warning(
                "OperationalSpaceController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        kp = config.get("kp", 150.0)
        damping_ratio = config.get("damping_ratio", 1.0)
        kp_null = config.get("kp_null", 10.0)
        config["kp"] = cls.nums2array(nums=kp, dim=6) if kp is not None else None
        config["damping_ratio"] = damping_ratio
        config["kp_null"] = cls.nums2array(nums=kp_null, dim=control_dim) if kp_null is not None else None
        config["kd_null"] = 2 * cb.sqrt(config["kp_null"]) if kp_null is not None else None
        config["kp_limits"] = cb.array(config.get("kp_limits", (10.0, 300.0)))
        config["damping_ratio_limits"] = cb.array(config.get("damping_ratio_limits", (0.0, 2.0)))
        config["kp_null_limits"] = cb.array(config.get("kp_null_limits", (0.0, 50.0)))

        config["variable_kp"] = config["kp"] is None
        config["variable_damping_ratio"] = config["damping_ratio"] is None
        config["variable_kp_null"] = config["kp_null"] is None
        assert True not in {
            config["variable_kp"],
            config["variable_damping_ratio"],
            config["variable_kp_null"],
        }, "Variable gains with OSC is not supported yet!"

        mode = config.get("mode", "pose_delta_ori")
        assert_valid_key(key=mode, valid_keys=OSC_MODES, name="OSC mode")
        config["mode"] = mode

        command_input_limits = config.get("command_input_limits", "default")
        command_output_limits = config.get("command_output_limits", ((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5)))

        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        if mode == "pose_absolute_ori":
            if command_input_limits is not None:
                if type(command_input_limits) is str and command_input_limits == "default":
                    command_input_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_input_limits[0][3:] = -math.pi
                    command_input_limits[1][3:] = math.pi
            if command_output_limits is not None:
                if type(command_output_limits) is str and command_output_limits == "default":
                    command_output_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_output_limits[0][3:] = -math.pi
                    command_output_limits[1][3:] = math.pi

        is_input_limits_numeric = not (command_input_limits is None or isinstance(command_input_limits, str))
        is_output_limits_numeric = not (command_output_limits is None or isinstance(command_output_limits, str))
        if is_input_limits_numeric:
            command_input_limits = [cls.nums2array(lim, dim=6) for lim in command_input_limits]
        if is_output_limits_numeric:
            command_output_limits = [cls.nums2array(lim, dim=6) for lim in command_output_limits]

        config["command_dim"] = OSC_MODE_COMMAND_DIMS[mode]
        for variable_gain, gain_limits, dim in zip(
            (config["variable_kp"], config["variable_damping_ratio"], config["variable_kp_null"]),
            (config["kp_limits"], config["damping_ratio_limits"], config["kp_null_limits"]),
            (6, 6, control_dim),
        ):
            if variable_gain:
                if is_input_limits_numeric:
                    command_input_limits = [
                        cb.cat([lim, cls.nums2array(nums=val, dim=dim)])
                        for lim, val in zip(command_input_limits, (-1, 1))
                    ]
                if is_output_limits_numeric:
                    command_output_limits = [
                        cb.cat([lim, cls.nums2array(nums=val, dim=dim)])
                        for lim, val in zip(command_output_limits, gain_limits)
                    ]
                config["command_dim"] += dim

        config["decouple_pos_ori"] = config.get("decouple_pos_ori", False)
        config["workspace_pose_limiter"] = config.get("workspace_pose_limiter", None)
        reset_joint_pos = config.get("reset_joint_pos")
        config["reset_joint_pos"] = reset_joint_pos[config["dof_idx"]]

        config["command_input_limits"] = command_input_limits
        config["command_output_limits"] = command_output_limits

        return super()._process_config(controller_id, config)

    @classmethod
    def _init_state(cls, controller_id: str):
        cls._state[controller_id]["fixed_quat_target"] = None
    
    @classmethod
    def _clear_variable_gains(cls, controller_id: str):
        """
        Helper function to clear any gains that are variable and considered part of actions
        """
        if cls._configs[controller_id]["variable_kp"]:
            cls._configs[controller_id]["kp"] = None
        if cls._configs[controller_id]["variable_damping_ratio"]:
            cls._configs[controller_id]["damping_ratio"] = None
        if cls._configs[controller_id]["variable_kp_null"]:
            cls._configs[controller_id]["kp_null"] = None
            cls._configs[controller_id]["kd_null"] = None

    @classmethod
    def reset(cls, controller_id: str):
        super().reset(controller_id=controller_id)
        cls._state[controller_id]["fixed_quat_target"] = None
        cls._clear_variable_gains(controller_id=controller_id)

    @classmethod
    def _load_state(cls, controller_id: str, state):
        super()._load_state(controller_id=controller_id, state=state)
        if cls._goals[controller_id] is not None and cls._configs[controller_id]["mode"] == "position_fixed_ori":
            cls._state[controller_id]["fixed_quat_target"] = cls._goals[controller_id]["target_quat"]

    @classmethod
    def _update_variable_gains(cls, controller_id: str, gains):
        """
        Helper function to update any gains that are variable and considered part of actions

        Args:
            gains (n-array): array where n dim is parsed based on which gains are being learned
        """
        idx = 0
        if cls._configs[controller_id]["variable_kp"]:
            cls._configs[controller_id]["kp"] = gains[:, idx : idx + 6]
            idx += 6
        if cls._configs[controller_id]["variable_damping_ratio"]:
            cls._configs[controller_id]["damping_ratio"] = gains[:, idx : idx + 6]
            idx += 6
        if cls._configs[controller_id]["variable_kp_null"]:
            cls._configs[controller_id]["kp_null"] = gains[:, idx : idx + cls.control_dim(controller_id)]
            cls._configs[controller_id]["kd_null"] = 2 * cb.sqrt(cls._configs[controller_id]["kp_null"])  # critically damped
            idx += cls.control_dim(controller_id)

    @classmethod
    def _update_goal(cls, controller_id: str, command, control_dict):
        config = cls._configs[controller_id]
        pos_relative = cb.copy(control_dict[f"{config['task_name']}_pos_relative"])
        quat_relative = cb.copy(control_dict[f"{config['task_name']}_quat_relative"])

        if config["mode"] == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        if config["mode"] == "position_fixed_ori":
            if cls._state[controller_id]["fixed_quat_target"] is None:
                cls._state[controller_id]["fixed_quat_target"] = (
                    quat_relative if (cls._goals[controller_id] is None) else cls._goals[controller_id]["target_quat"]
                )
            target_quat = cls._state[controller_id]["fixed_quat_target"]
        elif config["mode"] == "position_compliant_ori":
            target_quat = quat_relative
        elif config["mode"] == "pose_absolute_ori" or config["mode"] == "absolute_pose":
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        if config["workspace_pose_limiter"] is not None:
            target_pos, target_quat = config["workspace_pose_limiter"](target_pos, target_quat, control_dict)

        gains = None  # TODO! command[OSC_MODE_COMMAND_DIMS[self.mode]:]
        if gains is not None:
            cls._update_variable_gains(controller_id=controller_id, gains=gains)

        # Set goals and return
        return dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat)),
        )

    @classmethod
    def compute_control(cls, controller_id: str, control_dict):
        goal_dict = cls._goals[controller_id]
        config = cls._configs[controller_id]
        kp = config["kp"]
        damping_ratio = config["damping_ratio"]
        kd = 2 * cb.sqrt(kp) * damping_ratio

        dof_idxs_mat = tuple(cb.meshgrid(cls.dof_idx(controller_id), cls.dof_idx(controller_id)))
        q = control_dict["joint_position"][cls.dof_idx(controller_id)]
        qd = control_dict["joint_velocity"][cls.dof_idx(controller_id)]
        mm = control_dict["mass_matrix"][dof_idxs_mat]
        j_eef = control_dict[f"{config['task_name']}_jacobian_relative"][:, cls.dof_idx(controller_id)]
        ee_pos = control_dict[f"{config['task_name']}_pos_relative"]
        ee_quat = control_dict[f"{config['task_name']}_quat_relative"]
        ee_vel = cb.cat(
            [
                control_dict[f"{config['task_name']}_lin_vel_relative"],
                control_dict[f"{config['task_name']}_ang_vel_relative"],
            ]
        )
        base_lin_vel = control_dict["root_rel_lin_vel"]
        base_ang_vel = control_dict["root_rel_ang_vel"]

        u = cb.get_custom_method("compute_osc_torques")(
            q=q,
            qd=qd,
            mm=mm,
            j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(cb.T.quat2mat(ee_quat)),
            ee_lin_vel=cb.as_float32(ee_vel[:3]),
            ee_ang_vel_err=cb.as_float32(
                cb.T.quat2axisangle(
                    cb.T.quat_multiply(cb.T.axisangle2quat(-ee_vel[3:]), cb.T.axisangle2quat(base_ang_vel))
                )
            ),
            goal_pos=goal_dict["target_pos"],
            goal_ori_mat=goal_dict["target_ori_mat"],
            kp=kp,
            kd=kd,
            kp_null=config["kp_null"],
            kd_null=config["kd_null"],
            rest_qpos=config["reset_joint_pos"],
            control_dim=cls.control_dim(controller_id),
            decouple_pos_ori=config["decouple_pos_ori"],
            base_lin_vel=cb.as_float32(base_lin_vel),
            base_ang_vel=cb.as_float32(base_ang_vel),
        ).flatten()

        if config["use_gravity_compensation"]:
            u += control_dict["gravity_force"][cls.dof_idx(controller_id)]
        if config["use_cc_compensation"]:
            u += control_dict["cc_force"][cls.dof_idx(controller_id)]
        return u

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        target_pos = cb.copy(control_dict[f"{config['task_name']}_pos_relative"])
        target_quat = cb.copy(control_dict[f"{config['task_name']}_quat_relative"])
        return dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat))
        )

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        pos_relative = control_dict[f"{config['task_name']}_pos_relative"]
        quat_relative = control_dict[f"{config['task_name']}_quat_relative"]
        command = cb.zeros(6)
        if config["mode"] == "absolute_pose":
            command[:3] = pos_relative
        else:
            # We can leave it as zero for delta mode.
            pass
        if config["mode"] in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = cb.T.quat2axisangle(quat_relative)
        else:
            # For these modes, we don't need to add orientation to the command
            pass

        return command

    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        return dict(target_pos=(3,), target_ori_mat=(3, 3))

    @classmethod
    def control_type(cls, controller_id: str):
        return ControlType.EFFORT

    @classmethod
    def command_dim(cls, controller_id: str):
        return cls._configs[controller_id]["command_dim"]


@th.jit.script
def _compute_osc_torques_torch(
    q: th.Tensor,
    qd: th.Tensor,
    mm: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    ee_lin_vel: th.Tensor,
    ee_ang_vel_err: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    kp: th.Tensor,
    kd: th.Tensor,
    kp_null: th.Tensor,
    kd_null: th.Tensor,
    rest_qpos: th.Tensor,
    control_dim: int,
    decouple_pos_ori: bool,
    base_lin_vel: th.Tensor,
    base_ang_vel: th.Tensor,
):
    # Compute the inverse
    mm_inv = th.linalg.inv(mm)

    # Calculate error
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat((pos_err, ori_err))

    # Vel target is the base velocity as experienced by the end effector
    # For angular velocity, this is just the base angular velocity
    # For linear velocity, this is the base linear velocity PLUS the net linear velocity experienced
    #   due to the base linear velocity
    # For angular velocity, we need to make sure we compute the difference between the base and eef velocity
    # properly, not simply "subtraction" as in the linear case
    lin_vel_err = base_lin_vel + th.linalg.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = th.cat((lin_vel_err, ee_ang_vel_err))

    # Determine desired wrench
    err = th.unsqueeze(kp * err + kd * vel_err, dim=-1)
    m_eef_inv = j_eef @ mm_inv @ j_eef.T
    m_eef = th.linalg.inv(m_eef_inv)

    if decouple_pos_ori:
        # # More efficient, but numba doesn't support 3D tensor operations yet
        # j_eef_batch = j_eef.reshape(2, 3, -1)
        # m_eef_pose_inv = j_eef_batch @ th.unsqueeze(mm_inv, dim=0) @ th.transpose(j_eef_batch, 0, 2, 1)
        # m_eef_pose = th.linalg.inv_ex(m_eef_pose_inv).inverse  # Shape (2, 3, 3)
        # wrench = (m_eef_pose @ err.reshape(2, 3, 1)).flatten()
        m_eef_pos_inv = j_eef[:3, :] @ mm_inv @ j_eef[:3, :].T
        m_eef_ori_inv = j_eef[3:, :] @ mm_inv @ j_eef[3:, :].T
        m_eef_pos = th.linalg.inv(m_eef_pos_inv)
        m_eef_ori = th.linalg.inv(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:3, :]
        wrench_ori = m_eef_ori @ err[3:, :]
        wrench = th.cat((wrench_pos, wrench_ori))
    else:
        wrench = m_eef @ err

    # Compute OSC torques
    u = j_eef.T @ wrench

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * wrap_angle(rest_qpos - q)
        u_null = mm @ th.unsqueeze(u_null, dim=-1)
        u += (th.eye(control_dim, dtype=th.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u


# Use numba since faster
@jit(nopython=True)
def _compute_osc_torques_numpy(
    q,
    qd,
    mm,
    j_eef,
    ee_pos,
    ee_mat,
    ee_lin_vel,
    ee_ang_vel_err,
    goal_pos,
    goal_ori_mat,
    kp,
    kd,
    kp_null,
    kd_null,
    rest_qpos,
    control_dim,
    decouple_pos_ori,
    base_lin_vel,
    base_ang_vel,
):
    # Compute the inverse
    mm_inv = np.linalg.inv(mm)

    # Calculate error
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate((pos_err, ori_err))

    # Vel target is the base velocity as experienced by the end effector
    # For angular velocity, this is just the base angular velocity
    # For linear velocity, this is the base linear velocity PLUS the net linear velocity experienced
    #   due to the base linear velocity
    # For angular velocity, we need to make sure we compute the difference between the base and eef velocity
    # properly, not simply "subtraction" as in the linear case
    lin_vel_err = base_lin_vel + np.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = np.concatenate((lin_vel_err, ee_ang_vel_err))

    # Determine desired wrench
    err = np.expand_dims(kp * err + kd * vel_err, axis=-1)
    m_eef_inv = j_eef @ mm_inv @ j_eef.T
    m_eef = np.linalg.inv(m_eef_inv)

    if decouple_pos_ori:
        # # More efficient, but numba doesn't support 3D tensor operations yet
        # j_eef_batch = j_eef.reshape(2, 3, -1)
        # m_eef_pose_inv = np.matmul(np.matmul(j_eef_batch, np.expand_dims(mm_inv, axis=0)), np.transpose(j_eef_batch, (0, 2, 1)))
        # m_eef_pose = np.linalg.inv(m_eef_pose_inv)  # Shape (2, 3, 3)
        # wrench = np.matmul(m_eef_pose, err.reshape(2, 3, 1)).flatten()
        m_eef_pos_inv = j_eef[:3, :] @ mm_inv @ j_eef[:3, :].T
        m_eef_ori_inv = j_eef[3:, :] @ mm_inv @ j_eef[3:, :].T
        m_eef_pos = np.linalg.inv(m_eef_pos_inv)
        m_eef_ori = np.linalg.inv(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:3, :]
        wrench_ori = m_eef_ori @ err[3:, :]
        wrench = np.concatenate((wrench_pos, wrench_ori))
    else:
        wrench = m_eef @ err

    # Compute OSC torques
    u = j_eef.T @ wrench

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * ((rest_qpos - q + np.pi) % (2 * np.pi) - np.pi)
        u_null = mm @ np.expand_dims(u_null, axis=-1).astype(np.float32)
        u += (np.eye(control_dim, dtype=np.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u


# Set these as part of the backend values
add_compute_function(
    name="compute_osc_torques", np_function=_compute_osc_torques_numpy, th_function=_compute_osc_torques_torch
)
