import numpy as np

from omnigibson.controllers import IsGraspingState, ControlType, LocomotionController, ManipulationController, \
    GripperController
from omnigibson.utils.python_utils import assert_valid_key
import omnigibson.utils.transform_utils as T


class JointController(LocomotionController, ManipulationController, GripperController):
    """
    Controller class for joint control. Because omniverse can handle direct position / velocity / effort
    control signals, this is merely a pass-through operation from command to control (with clipping / scaling built in).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. If using delta commands, then adds the command to the current joint state
        2b. Clips the resulting command by the motor limits
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        use_delta_commands=False,
        compute_delta_in_quat_space=None,
    ):
        """
        Args:
            control_freq (int): controller loop frequency
            motor_type (str): type of motor being controlled, one of {position, velocity, effort}
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
            use_delta_commands (bool): whether inputted commands should be interpreted as delta or absolute values
            compute_delta_in_quat_space (None or List[(rx_idx, ry_idx, rz_idx), ...]): if specified, groups of
                joints that need to be processed in quaternion space to avoid gimbal lock issues normally faced by
                3 DOF rotation joints. Each group needs to consist of three idxes corresponding to the indices in
                the input space. This is only used in the delta_commands mode.
        """
        # Store arguments
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self._motor_type = motor_type.lower()
        self._use_delta_commands = use_delta_commands
        self._compute_delta_in_quat_space = [] if compute_delta_in_quat_space is None else compute_delta_in_quat_space

        # When in delta mode, it doesn't make sense to infer output range using the joint limits (since that's an
        # absolute range and our values are relative). So reject the default mode option in that case.
        assert not (
            self._use_delta_commands and command_output_limits == "default"
        ), "Cannot use 'default' command output limits in delta commands mode of JointController. Try None instead."

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self):
        # Nothing to reset.
        pass

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal

        Args:
            command (Array[float]): desired (already preprocessed) command to convert into control signals
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities
                    joint_effort: Array of current joint effort

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        # If we're using delta commands, add this value
        if self._use_delta_commands:
            # Compute the base value for the command.
            base_value = control_dict["joint_{}".format(self._motor_type)][self.dof_idx]

            # Apply the command to the base value.
            u = base_value + command

            # Correct any gimbal lock issues using the compute_delta_in_quat_space group.
            for rx_ind, ry_ind, rz_ind in self._compute_delta_in_quat_space:
                # Grab the starting rotations of these joints.
                start_rots = base_value[[rx_ind, ry_ind, rz_ind]]

                # Grab the delta rotations.
                delta_rots = command[[rx_ind, ry_ind, rz_ind]]

                # Compute the final rotations in the quaternion space.
                _, end_quat = T.pose_transform(np.zeros(3), T.euler2quat(delta_rots),
                                               np.zeros(3), T.euler2quat(start_rots))
                end_rots = T.quat2euler(end_quat)

                # Update the command
                u[[rx_ind, ry_ind, rz_ind]] = end_rots

        # Otherwise, control is simply the command itself
        else:
            u = command

        # Return control
        return u

    def is_grasping(self):
        # No good heuristic to determine grasping, so return UNKNOWN
        return IsGraspingState.UNKNOWN

    @property
    def use_delta_commands(self):
        """
        Returns:
            bool: Whether this controller is using delta commands or not
        """
        return self._use_delta_commands

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return len(self.dof_idx)
