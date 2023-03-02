import numpy as np

import omnigibson.utils.transform_utils as T
from omnigibson.controllers import ControlType, ManipulationController
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.control_utils import IKSolver
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Different modes
IK_MODE_COMMAND_DIMS = {
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}
IK_MODES = set(IK_MODE_COMMAND_DIMS.keys())


class InverseKinematicsController(ManipulationController):
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
        robot_description_path,
        robot_urdf_path,
        eef_name,
        control_freq,
        default_joint_pos,          # TODO: Currently doesn't do anything in Lula
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5)),
        motor_type="velocity",
        kv=2.0,
        mode="pose_delta_ori",
        smoothing_filter_size=None,
        workspace_pose_limiter=None,
    ):
        """
        Args:
            task_name (str): name assigned to this task frame for computing IK control. During control calculations,
                the inputted control_dict should include entries named <@task_name>_pos_relative and
                <@task_name>_quat_relative. See self._command_to_control() for what these values should entail.
            robot_description_path (str): path to robot descriptor yaml file
            robot_urdf_path (str): path to robot urdf file
            eef_name (str): end effector frame name
            control_freq (int): controller loop frequency
            default_joint_pos (Array[float]): default joint positions, used as part of nullspace controller in IK.
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
            motor_type (str): type of motor being controlled, one of {position, velocity}
            kv (float): Gain applied to error between IK-commanded joint positions and current joint positions if
                using @motor_type = velocity
            mode (str): mode to use when computing IK. In all cases, position commands are 3DOF delta (dx,dy,dz)
                cartesian values, relative to the robot base frame. Valid options are:
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

                    def limiter(command_pos: Array[float], command_quat: Array[float], control_dict: Dict[str, Any]) --> Tuple[Array[float], Array[float]]

                where pos_command is (x,y,z) cartesian position values, command_quat is (x,y,z,w) quarternion orientation
                values, and the returned tuple is the processed (pos, quat) command.
        """
        # Store arguments
        control_dim = len(dof_idx)
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self._motor_type = motor_type.lower()
        self.control_filter = (
            None
            if smoothing_filter_size in {None, 0}
            else MovingAverageFilter(obs_dim=control_dim, filter_width=smoothing_filter_size)
        )
        assert mode in IK_MODES, "Invalid ik mode specified! Valid options are: {IK_MODES}, got: {mode}"
        self.mode = mode
        self.kv = kv
        self.workspace_pose_limiter = workspace_pose_limiter
        self.task_name = task_name
        self.default_joint_pos = default_joint_pos[dof_idx]

        # Create the lula IKSolver
        self.solver = IKSolver(
            robot_description_path=robot_description_path,
            robot_urdf_path=robot_urdf_path,
            eef_name=eef_name,
            default_joint_pos=default_joint_pos,
        )

        # Other variables that will be filled in at runtime
        self._quat_target = None

        # If the mode is set as absolute orientation and using default config,
        # change input and output limits accordingly.
        # By default, the input limits are set as 1, so we modify this to have a correct range.
        # The output orientation limits are also set to be values assuming delta commands, so those are updated too
        if self.mode == "pose_absolute_ori":
            if command_input_limits is not None:
                if command_input_limits == "default":
                    command_input_limits = [
                        [-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi],
                        [1.0, 1.0, 1.0, np.pi, np.pi, np.pi],
                    ]
                else:
                    command_input_limits[0][3:] = -np.pi
                    command_input_limits[1][3:] = np.pi
            if command_output_limits is not None:
                if command_output_limits == "default":
                    command_output_limits = [
                        [-1.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi],
                        [1.0, 1.0, 1.0, np.pi, np.pi, np.pi],
                    ]
                else:
                    command_output_limits[0][3:] = -np.pi
                    command_output_limits[1][3:] = np.pi

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self):
        # Reset the filter and clear internal control state
        if self.control_filter is not None:
            self.control_filter.reset()
        self._quat_target = None

    @property
    def state_size(self):
        # Add 4 for internal quat target and the state size from the control filter
        return super().state_size + 4 + self.control_filter.state_size

    def _dump_state(self):
        # Run super first
        state = super()._dump_state()

        # Add internal quaternion target and filter state
        state["quat_target"] = self._quat_target
        state["control_filter"] = self.control_filter.dump_state(serialized=False)

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load relevant info for this controller
        self._quat_target = state["quat_target"]
        self.control_filter.load_state(state["control_filter"], serialized=False)

    def _serialize(self, state):
        # Run super first
        state_flat = super()._serialize(state=state)

        # Serialize state for this controller
        return np.concatenate([
            state_flat,
            np.zeros(4) if state["quat_target"] is None else state["quat_target"],      # Encode None as zeros for consistent serialization size
            self.control_filter.serialize(state=state["control_filter"]),
        ]).astype(float)

    def _deserialize(self, state):
        # Run super first
        state_dict, idx = super()._deserialize(state=state)

        # Deserialize state for this controller
        state_dict["quat_target"] = None if np.all(state[idx: idx + 4] == 0.0) else state[idx: idx + 4]
        state_dict["control_filter"] = self.control_filter.deserialize(state=state[idx + 4: idx + 4 + self.control_filter.state_size])

        return state_dict, idx + 4 + self.control_filter.state_size

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal.
        This processes the command based on self.mode, possibly clips the command based on self.workspace_pose_limiter,

        Args:
            command (Array[float]): desired (already preprocessed) command to convert into control signals
                Is one of:
                    (dx,dy,dz) - desired delta cartesian position
                    (dx,dy,dz,dax,day,daz) - desired delta cartesian position and delta axis-angle orientation
                    (dx,dy,dz,ax,ay,az) - desired delta cartesian position and global axis-angle orientation
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
        # Grab important info from control dict
        pos_relative = np.array(control_dict["{}_pos_relative".format(self.task_name)])
        quat_relative = np.array(control_dict["{}_quat_relative".format(self.task_name)])

        # The first three values of the command are always the (delta) position, convert to absolute values
        dpos = command[:3]
        target_pos = pos_relative + dpos

        # Compute orientation
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self._quat_target is None:
                self._quat_target = quat_relative
            target_quat = self._quat_target
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif self.mode == "pose_absolute_ori":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = T.axisangle2quat(command[3:])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = T.quat2mat(T.axisangle2quat(command[3:]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat, control_dict)

        # Calculate and return IK-backed out joint angles
        current_joint_pos = control_dict["joint_position"][self.dof_idx]
        target_joint_pos = self.solver.solve(
            target_pos=target_pos,
            target_quat=target_quat,
            initial_joint_pos=current_joint_pos,
        )

        if target_joint_pos is None:
            # Print warning that we couldn't find a valid solution, and return the current joint configuration
            # instead so that we execute a no-op control
            log.warning(f"Could not find valid IK configuration! Returning no-op control instead.")
            target_joint_pos = current_joint_pos

        # Optionally pass through smoothing filter for better stability
        if self.control_filter is not None:
            target_joint_pos = self.control_filter.estimate(target_joint_pos)

        # Grab the resulting error and scale it by the velocity gain, or else simply use the target_joint_pos
        u = -self.kv * (current_joint_pos - target_joint_pos) if \
            self.control_type == ControlType.VELOCITY else target_joint_pos

        # Return these commanded velocities (this only includes the relevant dof idx)
        return u

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return IK_MODE_COMMAND_DIMS[self.mode]
