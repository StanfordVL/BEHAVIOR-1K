import math
from collections.abc import Iterable

import numpy as np
import torch as th
from numba import jit

import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers import ControlType, ManipulationController
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


class InverseKinematicsController(JointController, ManipulationController):
    """
    Controller class to convert (delta) EEF commands into joint velocities using Inverse Kinematics (IK).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run Inverse Kinematics to back out joint velocities for a desired task frame command
        3. Clips the resulting command by the motor (velocity) limits
    """

    def __init__(
        self,
        task_name,
        control_freq,
        reset_joint_pos,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=(
            (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
            (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
        ),
        isaac_kp=None,
        isaac_kd=None,
        pos_kp=None,
        pos_damping_ratio=None,
        vel_kp=None,
        use_impedances=False,
        mode="pose_delta_ori",
        smoothing_filter_size=None,
        workspace_pose_limiter=None,
        condition_on_current_position=True,
    ):
        """
        Args:
            task_name (str): name assigned to this task frame for computing IK control. During control calculations,
                the inputted control_dict should include entries named <@task_name>_pos_relative and
                <@task_name>_quat_relative. See self._command_to_control() for what these values should entail.
            control_freq (int): controller loop frequency
            reset_joint_pos (Array[float]): reset joint positions, used as part of nullspace controller in IK.
                Note that this should correspond to ALL the joints; the exact indices will be extracted via @dof_idx
            control_limits (Dict[str, Tuple[Array[float], Array[float]]]): The min/max limits to the outputted
                    control signal. Should specify per-dof type limits, i.e.:

                    "position": [[min], [max]]
                    "velocity": [[min], [max]]
                    "effort": [[min], [max]]
                    "has_limit": [...bool...]

                Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
            dof_idx (Array[int]): specific dof indices controlled by this robot. Used for inferring
                controller-relevant values during control computations
            command_input_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max acceptable inputted command. Values outside this range will be clipped.
                If None, no clipping will be used. If "default", range will be set to (-1, 1)
            command_output_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max scaled command. If both this value and @command_input_limits is not None,
                then all inputted command values will be scaled from the input range to the output range.
                If either is None, no scaling will be used. If "default", then this range will automatically be set
                to the @control_limits entry corresponding to self.control_type
            isaac_kp (None or float or Array[float]): If specified, stiffness gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers.
                Should only be nonzero if self.control_type is position
            isaac_kd (None or float or Array[float]): If specified, damping gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers
                Should only be nonzero if self.control_type is position or velocity
            pos_kp (None or float): If @motor_type is "position" and @use_impedances=True, this is the
                proportional gain applied to the joint controller. If None, a default value will be used.
            pos_damping_ratio (None or float): If @motor_type is "position" and @use_impedances=True, this is the
                damping ratio applied to the joint controller. If None, a default value will be used.
            vel_kp (None or float): If @motor_type is "velocity" and @use_impedances=True, this is the
                proportional gain applied to the joint controller. If None, a default value will be used.
            use_impedances (bool): If True, will use impedances via the mass matrix to modify the desired efforts
                applied
            mode (str): mode to use when computing IK. In all cases, position commands are 3DOF delta (dx,dy,dz)
                cartesian values, relative to the robot base frame. Valid options are:
                    - "absolute_pose": 6DOF (dx,dy,dz,ax,ay,az) control over pose,
                        where both the position and the orientation is given in absolute axis-angle coordinates
                    - "pose_absolute_ori": 6DOF (dx,dy,dz,ax,ay,az) control over pose,
                        where the orientation is given in absolute axis-angle coordinates
                    - "pose_delta_ori": 6DOF (dx,dy,dz,dax,day,daz) control over pose
                    - "position_fixed_ori": 3DOF (dx,dy,dz) control over position,
                        with orientation commands being kept as fixed initial absolute orientation
                    - "position_compliant_ori": 3DOF (dx,dy,dz) control over position,
                        with orientation commands automatically being sent as 0s (so can drift over time)
            smoothing_filter_size (None or int): if specified, sets the size of a moving average filter to apply
                on all outputted IK joint positions.
            workspace_pose_limiter (None or function): if specified, callback method that should clip absolute
                target (x,y,z) cartesian position and absolute quaternion orientation (x,y,z,w) to a specific workspace
                range (i.e.: this can be unique to each robot, and implemented by each embodiment).
                Function signature should be:

                    def limiter(target_pos: Array[float], target_quat: Array[float], control_dict: Dict[str, Any]) --> Tuple[Array[float], Array[float]]

                where target_pos is (x,y,z) cartesian position values, target_quat is (x,y,z,w) quarternion orientation
                values, and the returned tuple is the processed (pos, quat) command.
            condition_on_current_position (bool): if True, will use the current joint position as the initial guess for the IK algorithm.
                Otherwise, will use the reset_joint_pos as the initial guess.
        """
        # Store arguments
        control_dim = len(dof_idx)
        self.control_filter = (
            None
            if smoothing_filter_size in {None, 0}
            else MovingAverageFilter(obs_dim=control_dim, filter_width=smoothing_filter_size)
        )
        assert mode in IK_MODES, f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {mode}"

        # If mode is absolute pose, make sure command input limits / output limits are None
        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        self.mode = mode
        self.workspace_pose_limiter = workspace_pose_limiter
        self.task_name = task_name
        self.reset_joint_pos = reset_joint_pos[dof_idx]
        self.condition_on_current_position = condition_on_current_position

        # Other variables that will be filled in at runtime
        self._fixed_quat_target = None

        # If the mode is set as absolute orientation and using default config,
        # change input and output limits accordingly.
        # By default, the input limits are set as 1, so we modify this to have a correct range.
        # The output orientation limits are also set to be values assuming delta commands, so those are updated too
        if self.mode == "pose_absolute_ori":
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
        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            vel_kp=vel_kp,
            motor_type="position",
            use_delta_commands=False,
            use_impedances=use_impedances,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
        )

    def reset(self):
        # Call super first
        super().reset()

        # Reset the filter and clear internal control state
        if self.control_filter is not None:
            self.control_filter.reset()
        self._fixed_quat_target = None

    @property
    def state_size(self):
        # Add state size from the control filter
        return super().state_size + (0 if self.control_filter is None else self.control_filter.state_size)

    def _dump_state(self):
        # Run super first
        state = super()._dump_state()

        # Add internal quaternion target and filter state
        if self.control_filter is not None:
            state["control_filter"] = self.control_filter.dump_state(serialized=False)

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # If self._goal is populated, then set fixed_quat_target as well if the mode uses it
        if self._goal is not None:
            if self.mode == "position_fixed_ori":
                self._fixed_quat_target = self._goal["target_quat"]

            # Load relevant info for this controller
            if self.control_filter is not None:
                self.control_filter.load_state(state["control_filter"], serialized=False)

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize state for this controller
        return th.cat(
            [
                state_flat,
                (
                    th.tensor([])
                    if self.control_filter is None
                    else self.control_filter.serialize(state=state["control_filter"])
                ),
            ]
        )

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize state for this controller
        if self.control_filter is not None:
            state_dict["control_filter"], deserialized_items = self.control_filter.deserialize(state=state[idx:])
            idx += deserialized_items

        return state_dict, idx

    def _update_goal(self, command, control_dict):
        # Grab important info from control dict
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]

        # Convert position command to absolute values if needed
        if self.mode == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        # Compute orientation
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self._fixed_quat_target is None:
                self._fixed_quat_target = quat_relative if (self._goal is None) else self._goal["target_quat"]
            target_quat = self._fixed_quat_target
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif self.mode == "pose_absolute_ori" or self.mode == "absolute_pose":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat, control_dict)

        goal_dict = dict(
            target_pos=cb.as_float32(target_pos),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(target_quat)),
        )

        return goal_dict

    def compute_control(self, goal_dict, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal.
        This processes the command based on self.mode, possibly clips the command based on self.workspace_pose_limiter,

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
        q = control_dict["joint_position"][self.dof_idx]
        j_eef = control_dict[f"{self.task_name}_jacobian_relative"][:, self.dof_idx]
        ee_pos = control_dict[f"{self.task_name}_pos_relative"]
        ee_quat = control_dict[f"{self.task_name}_quat_relative"]

        # Calculate desired joint positions
        target_joint_pos = cb.get_custom_method("compute_ik_qpos")(
            q=q,
            j_eef=j_eef,
            ee_pos=cb.as_float32(ee_pos),
            ee_mat=cb.as_float32(cb.T.quat2mat(ee_quat)),
            goal_pos=goal_dict["target_pos"],
            goal_ori_mat=goal_dict["target_ori_mat"],
            q_lower_limit=self._control_limits[ControlType.get_type("position")][0][self.dof_idx],
            q_upper_limit=self._control_limits[ControlType.get_type("position")][1][self.dof_idx],
        )

        # Optionally pass through smoothing filter for better stability
        if self.control_filter is not None:
            target_joint_pos = self.control_filter.estimate(target_joint_pos)

        # Run super to reach desired position / velocity setpoint
        return super().compute_control(goal_dict=dict(target=target_joint_pos), control_dict=control_dict)

    def compute_no_op_goal(self, control_dict):
        # No-op is maintaining current pose
        # Convert quat into eef ori mat
        return dict(
            target_pos=cb.as_float32(control_dict[f"{self.task_name}_pos_relative"]),
            target_ori_mat=cb.as_float32(cb.T.quat2mat(control_dict[f"{self.task_name}_quat_relative"])),
        )

    def _compute_no_op_command(self, control_dict):
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]

        command = cb.zeros(6)

        # Handle position
        if self.mode == "absolute_pose":
            command[:3] = pos_relative
        else:
            # We can leave it as zero for delta mode.
            pass

        # Handle orientation
        if self.mode in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = cb.T.quat2axisangle(quat_relative)
        else:
            # For these modes, we don't need to add orientation to the command
            pass

        return command

    def _get_goal_shapes(self):
        return dict(
            target_pos=(3,),
            target_ori_mat=(3, 3),
        )

    @property
    def command_dim(self):
        return IK_MODE_COMMAND_DIMS[self.mode]


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
