from copy import deepcopy
import math
from collections.abc import Iterable

import numpy as np
import torch as th
from numba import jit

import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers.controller_base import ControlType
from omnigibson.controllers.joint_controller import JointController
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Different modes
IK_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,  # 6DOF (x,y,z,ax,ay,az) control of pose, whether both position and orientation is given in absolute coordinates
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}
IK_MODES = set(IK_MODE_COMMAND_DIMS.keys())


class InverseKinematicsController(JointController):
    """
    Controller class to convert (delta) EEF commands into joint velocities using Inverse Kinematics (IK).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run Inverse Kinematics to back out joint velocities for a desired task frame command
        3. Clips the resulting command by the motor (velocity) limits
    """

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        config["motor_type"] = "position"
        config["use_delta_commands"] = False
        config["mode"] = config.get("mode", "pose_delta_ori")
        assert config["mode"] in IK_MODES, f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {config['mode']}"
        config["use_impedances"] = config.get("use_impedances", False)
        config["smoothing_filter_size"] = config.get("smoothing_filter_size", None)
        config["workspace_pose_limiter"] = config.get("workspace_pose_limiter", None)
        config["condition_on_current_position"] = config.get("condition_on_current_position", True)
        config["reset_joint_pos"] = config["reset_joint_pos"][config["dof_idx"]]

        command_input_limits = config.get("command_input_limits", "default")
        command_output_limits = config.get("command_output_limits", (
            (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
            (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
        ))

        # If mode is absolute pose, make sure command input limits / output limits are None
        if config["mode"] == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        # If the mode is set as absolute orientation and using default config,
        # change input and output limits accordingly.
        # By default, the input limits are set as 1, so we modify this to have a correct range.
        # The output orientation limits are also set to be values assuming delta commands, so those are updated too
        if config["mode"] == "pose_absolute_ori":
            if command_input_limits is not None:
                if type(command_input_limits) is str and command_input_limits == "default":
                    command_input_limits = [
                        cb.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi]),
                        cb.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi]),
                    ]
                else:
                    command_input_limits[0][3:] = cb.array([-math.pi] * len(command_input_limits[0][3:]))
                    command_input_limits[1][3:] = cb.array([math.pi] * len(command_input_limits[1][3:]))
            if command_output_limits is not None:
                if not isinstance(command_output_limits, str) and isinstance(command_output_limits, Iterable):
                    command_output_limits = [
                        cb.array(command_output_limits[0]),
                        cb.array(command_output_limits[1]),
                    ]
                if type(command_output_limits) is str and command_output_limits == "default":
                    command_output_limits = [
                        cb.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi]),
                        cb.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi]),
                    ]
                else:
                    command_output_limits[0][3:] = cb.array([-math.pi] * len(command_output_limits[0][3:]))
                    command_output_limits[1][3:] = cb.array([math.pi] * len(command_output_limits[1][3:]))
        config["command_input_limits"] = command_input_limits
        config["command_output_limits"] = command_output_limits

        return super()._process_config(controller_id, config)

    @classmethod
    def _init_state(cls, controller_id: str):
        config = cls._configs[controller_id]
        cls._state[controller_id]["fixed_quat_target"] = None
        cls._state[controller_id]["control_filter"] = None if config.get("smoothing_filter_size", None) in {None, 0} else MovingAverageFilter(
                obs_dim=len(cls.dof_idx(controller_id)), filter_width=config["smoothing_filter_size"]
            )

    @classmethod
    def reset(cls, controller_id: str):
        super().reset(controller_id=controller_id)
        if cls._state[controller_id]["control_filter"] is not None:
            cls._state[controller_id]["control_filter"].reset()
        cls._state[controller_id]["fixed_quat_target"] = None
    
    @classmethod
    def state_size(cls, controller_id: str):
        # Add state size from the control filter
        control_filter = cls._state[controller_id]["control_filter"]
        return super().state_size(controller_id) + (0 if control_filter is None else control_filter.state_size)


    @classmethod
    def _dump_state(cls, controller_id: str):
        state = super()._dump_state(controller_id=controller_id)
        if cls._state[controller_id]["control_filter"] is not None:
            state["control_filter"] = cls._state[controller_id]["control_filter"].dump_state(serialized=False)
        return state

    @classmethod
    def _load_state(cls, controller_id: str, state):
        super()._load_state(controller_id=controller_id, state=state)
        # If self._goal is populated, then set fixed_quat_target as well if the mode uses it
        if cls._goals[controller_id] is not None:
            if cls._configs[controller_id]["mode"] == "position_fixed_ori":
                cls._state[controller_id]["fixed_quat_target"] = cls._goals[controller_id]["target_quat"]
            # Load relevant info for this controller
            if cls._state[controller_id]["control_filter"] is not None:
                cls._state[controller_id]["control_filter"].load_state(state["control_filter"], serialized=False)

    @classmethod
    def serialize(cls, controller_id: str, state):
        state_flat = super().serialize(controller_id=controller_id, state=state)
        return th.cat(
            [
                state_flat,
                (
                    th.tensor([])
                    if cls._state[controller_id]["control_filter"] is None
                    else cls._state[controller_id]["control_filter"].serialize(state=state["control_filter"])
                ),
            ]
        )

    @classmethod
    def deserialize(cls, controller_id: str, state):
        state_dict, idx = super().deserialize(controller_id=controller_id, state=state)
        if cls._state[controller_id]["control_filter"] is not None:
            state_dict["control_filter"], deserialized_items = cls._state[controller_id]["control_filter"].deserialize(
                state=state[idx:]
            )
            idx += deserialized_items
        return state_dict, idx

    @classmethod
    def _update_goal(cls, controller_id: str, command, control_dict):
        config = cls._configs[controller_id]
        pos_relative = control_dict[f"{config['task_name']}_pos_relative"]
        quat_relative = control_dict[f"{config['task_name']}_quat_relative"]

        # Convert position command to absolute values if needed
        if config["mode"] == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        # Compute orientation
        if config["mode"] == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if cls._state[controller_id]["fixed_quat_target"] is None:
                cls._state[controller_id]["fixed_quat_target"] = (
                    quat_relative if (cls._goals[controller_id] is None) else cls._goals[controller_id]["target_quat"]
                )
            target_quat = cls._state[controller_id]["fixed_quat_target"]
        elif config["mode"] == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif config["mode"] == "pose_absolute_ori" or config["mode"] == "absolute_pose":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if config.get("workspace_pose_limiter", None) is not None:
            target_pos, target_quat = config["workspace_pose_limiter"](target_pos, target_quat, control_dict)

        goal_dict = dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat))
        )
        return goal_dict

    @classmethod
    def compute_control(cls, controller_id: str, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal.
        This processes the command based on mode, possibly clips the command based on self.workspace_pose_limiter,

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    target_pos: robot-frame (x,y,z) desired end effector position
                    target_ori_mat: robot-frame desired end effector quaternion orientation matrix
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    base_pos: (x,y,z) cartesian position of the robot's base relative to the static global frame
                    base_quat: (x,y,z,w) quaternion orientation of the robot's base relative to the static global frame
                    <@self.task_name>_pos_relative: (x,y,z) relative cartesian position of the desired task frame to
                        control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_quat_relative: (x,y,z,w) relative quaternion orientation of the desired task
                        frame to control, computed in its local frame (e.g.: robot base frame)

        Returns:
            Array[float]: outputted (non-clipped!) velocity control signal to deploy
        """
        # Calculate and return IK-backed out joint angles
        config = cls._configs[controller_id]
        goal_dict = cls._goals[controller_id]
        q = control_dict["joint_position"][cls.dof_idx(controller_id)]
        j_eef = control_dict[f"{config['task_name']}_jacobian_relative"][:, cls.dof_idx(controller_id)]
        ee_pos = control_dict[f"{config['task_name']}_pos_relative"]
        ee_quat = control_dict[f"{config['task_name']}_quat_relative"]


        # Calculate desired joint positions
        target_joint_pos = cb.get_custom_method("compute_ik_qpos")(
            q=q,
            j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(cb.T.quat2mat(ee_quat)),
            goal_pos=goal_dict["target_pos"],
            goal_ori_mat=goal_dict["target_ori_mat"],
            q_lower_limit=config["control_limits"][ControlType.get_type("position")][0][cls.dof_idx(controller_id)],
            q_upper_limit=config["control_limits"][ControlType.get_type("position")][1][cls.dof_idx(controller_id)],
        )

        # Optionally pass through smoothing filter for better stability
        if cls._state[controller_id]["control_filter"] is not None:
            target_joint_pos = cls._state[controller_id]["control_filter"].estimate(target_joint_pos)

        # Run super to reach desired position / velocity setpoint
        return super().compute_control(controller_id=controller_id, control_dict=control_dict, goal_dict=dict(target=target_joint_pos))

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        return dict(
            target_pos=cb.as_float32(control_dict[f"{config['task_name']}_pos_relative"]),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(control_dict[f"{config['task_name']}_quat_relative"])),
        )

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        pos_relative = control_dict[f"{config['task_name']}_pos_relative"]
        quat_relative = control_dict[f"{config['task_name']}_quat_relative"]

        command = cb.zeros(6)
        mode = config[controller_id]["mode"]
        # Handle position
        if mode == "absolute_pose":
            command[:3] = pos_relative
        else:
            # We can leave it as zero for delta mode.
            pass

        # Handle orientation
        if mode in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = cb.T.quat2axisangle(quat_relative)
        else:
            # For these modes, we don't need to add orientation to the command
            pass

        return command

    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        return dict(target_pos=(3,), target_ori_mat=(3, 3), target_quat=(4,))

    @classmethod
    def step_batch(cls, controller_ids):
        """
        Batched step for InverseKinematicsController instances.
        Batches the expensive IK solve (Jacobian pseudoinverse), then applies
        per-instance smoothing filter and JointController tail sequentially.
        """
        # Fill no-op goals
        for cid in controller_ids:
            if cls._goals[cid] is None:
                cls._goals[cid] = cls.compute_no_op_goal(cid, cls._control_dicts[cid])

        N = len(controller_ids)
        dims = [cls.control_dim(cid) for cid in controller_ids]
        max_dim = max(dims)

        # Gather and zero-pad IK data into batched tensors
        q = cb.zeros((N, max_dim))
        j_eef = cb.zeros((N, 6, max_dim))
        ee_pos = cb.zeros((N, 3))
        ee_mat = cb.zeros((N, 3, 3))
        goal_pos = cb.zeros((N, 3))
        goal_ori_mat = cb.zeros((N, 3, 3))
        q_lower = cb.zeros((N, max_dim))
        q_upper = cb.zeros((N, max_dim))

        for i, cid in enumerate(controller_ids):
            config = cls._configs[cid]
            d = dims[i]
            dof_idx = cls.dof_idx(cid)
            cd = cls._control_dicts[cid]

            q[i, :d] = cd["joint_position"][dof_idx]
            j_eef[i, :, :d] = cd[f"{config['task_name']}_jacobian_relative"][:, dof_idx]
            ee_pos[i] = cd[f"{config['task_name']}_pos_relative"]
            ee_quat = cd[f"{config['task_name']}_quat_relative"]
            ee_mat[i] = cb.as_float32(cb.T.quat2mat(ee_quat))
            goal_pos[i] = cls._goals[cid]["target_pos"]
            goal_ori_mat[i] = cls._goals[cid]["target_ori_mat"]
            q_lower[i, :d] = config["control_limits"][ControlType.get_type("position")][0][dof_idx]
            q_upper[i, :d] = config["control_limits"][ControlType.get_type("position")][1][dof_idx]

        # Batched IK solve: [N, max_dim] joint position targets
        target_batch = cb.get_custom_method("compute_ik_qpos_batch")(
            q=q,
            j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(ee_mat),
            goal_pos=cb.as_float32(goal_pos),
            goal_ori_mat=cb.as_float32(goal_ori_mat),
            q_lower_limit=q_lower,
            q_upper_limit=q_upper,
        )

        # Per-instance post-processing: smoothing filter + JointController compute tail
        results = []
        for i, cid in enumerate(controller_ids):
            d = dims[i]
            target_joint_pos = target_batch[i, :d]

            # Smoothing filter (per-instance, stateful)
            if cls._state[cid]["control_filter"] is not None:
                target_joint_pos = cls._state[cid]["control_filter"].estimate(target_joint_pos)

            # JointController compute_control tail (inlined to avoid cross-class dict issues)
            config = cls._configs[cid]
            if config["use_impedances"]:
                cd = cls._control_dicts[cid]
                dof_idx = cls.dof_idx(cid)
                base_value = cd[f"joint_{config['motor_type']}"][dof_idx]
                if config["motor_type"] == "position":
                    u = (target_joint_pos - base_value) * config["pos_kp"] + (-cd["joint_velocity"][dof_idx]) * config["pos_kd"]
                elif config["motor_type"] == "velocity":
                    u = (target_joint_pos - base_value) * config["vel_kp"]
                else:
                    u = target_joint_pos
                u = cb.get_custom_method("compute_joint_torques")(u, cd["mass_matrix"], dof_idx)
                if config["use_gravity_compensation"]:
                    u += cd["gravity_force"][dof_idx]
                if config["use_cc_compensation"]:
                    u += cd["cc_force"][dof_idx]
            else:
                u = target_joint_pos

            u = cls.clip_control(cid, u)
            cls._controls[cid] = u
            results.append(u)

        return results

    @classmethod
    def command_dim(cls, controller_id: str):
        return IK_MODE_COMMAND_DIMS[cls._configs[controller_id]["mode"]]


@th.jit.script
def _compute_ik_qpos_torch(
    q: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    q_lower_limit: th.Tensor,
    q_upper_limit: th.Tensor,
):
    # Compute the pose error. Note that this is computed NOT in the EEF frame but still
    # in the base frame.
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat([pos_err, ori_err])

    # Use the jacobian to compute a local approximation
    j_eef_pinv = th.linalg.pinv(j_eef)
    delta_j = j_eef_pinv @ err
    target_joint_pos = q + delta_j

    # Clip values to be within the joint limits
    return target_joint_pos.clip(
        min=q_lower_limit,
        max=q_upper_limit,
    )


# Use numba since faster
@jit(nopython=True)
def _compute_ik_qpos_numpy(
    q,
    j_eef,
    ee_pos,
    ee_mat,
    goal_pos,
    goal_ori_mat,
    q_lower_limit,
    q_upper_limit,
):
    # Compute the pose error. Note that this is computed NOT in the EEF frame but still
    # in the base frame.
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate((pos_err, ori_err))

    # Use the jacobian to compute a local approximation
    j_eef_pinv = np.linalg.pinv(j_eef)
    delta_j = j_eef_pinv @ err
    target_joint_pos = q + delta_j

    # Clip values to be within the joint limits
    return target_joint_pos.clip(q_lower_limit, q_upper_limit)


# Set these as part of the backend values
add_compute_function(name="compute_ik_qpos", np_function=_compute_ik_qpos_numpy, th_function=_compute_ik_qpos_torch)


def _compute_ik_qpos_batch_torch(
    q: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    q_lower_limit: th.Tensor,
    q_upper_limit: th.Tensor,
):
    """Batched IK solve for N controllers. All inputs have a leading batch dim N."""
    pos_err = goal_pos - ee_pos  # [N, 3]
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)  # [N, 3]
    err = th.cat([pos_err, ori_err], dim=-1)  # [N, 6]

    j_eef_pinv = th.linalg.pinv(j_eef)  # [N, D, 6]
    delta_j = (j_eef_pinv @ err.unsqueeze(-1)).squeeze(-1)  # [N, D]
    target_joint_pos = q + delta_j

    return target_joint_pos.clip(min=q_lower_limit, max=q_upper_limit)


def _compute_ik_qpos_batch_numpy(
    q,
    j_eef,
    ee_pos,
    ee_mat,
    goal_pos,
    goal_ori_mat,
    q_lower_limit,
    q_upper_limit,
):
    """Batched IK solve for N controllers. All inputs have a leading batch dim N."""
    pos_err = goal_pos - ee_pos  # [N, 3]
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)  # [N, 3]
    err = np.concatenate([pos_err, ori_err], axis=-1)  # [N, 6]

    j_eef_pinv = np.linalg.pinv(j_eef)  # [N, D, 6]
    delta_j = (j_eef_pinv @ err[..., None])[..., 0]  # [N, D]
    target_joint_pos = q + delta_j

    return target_joint_pos.clip(q_lower_limit, q_upper_limit)


add_compute_function(
    name="compute_ik_qpos_batch", np_function=_compute_ik_qpos_batch_numpy, th_function=_compute_ik_qpos_batch_torch
)
