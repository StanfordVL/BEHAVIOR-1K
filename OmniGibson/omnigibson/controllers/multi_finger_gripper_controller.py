import torch as th

from omnigibson.controllers import ControlType, GripperController, IsGraspingState
from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.python_utils import assert_valid_key

VALID_MODES = {
    "binary",
    "smooth",
    "independent",
}


# Create settings for this module
m = create_module_macros(module_path=__file__)

# is_grasping heuristics parameters
m.POS_TOLERANCE = 0.002  # arbitrary heuristic
m.VEL_TOLERANCE = 0.02  # arbitrary heuristic


class MultiFingerGripperController(GripperController):
    """
    Controller class for multi finger gripper control. This either interprets an input as a binary
    command (open / close), continuous command (open / close with scaled velocities), or per-joint continuous command

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. Convert command into gripper joint control signals
        2b. Clips the resulting control by the motor limits
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        isaac_kp=None,
        isaac_kd=None,
        inverted=False,
        mode="binary",
        open_qpos=None,
        closed_qpos=None,
        limit_tolerance=0.001,
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
            isaac_kp (None or float or Array[float]): If specified, stiffness gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers.
                Should only be nonzero if self.control_type is position
            isaac_kd (None or float or Array[float]): If specified, damping gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers
                Should only be nonzero if self.control_type is position or velocity
            inverted (bool): whether or not the command direction (grasp is negative) and the control direction are
                inverted, e.g. to grasp you need to move the joint in the positive direction.
            mode (str): mode for this controller. Valid options are:

                "binary": 1D command, if preprocessed value > 0 is interpreted as an max open
                    (send max pos / vel / tor signal), otherwise send max close control signals
                "smooth": 1D command, sends symmetric signal to all finger joints equal to the preprocessed commands
                "independent": n-dimensional command, sends independent signals to each finger joint equal to the preprocessed command

            open_qpos (None or Array[float]): If specified, the joint positions representing a fully-opened gripper.
                This is to allow representing the open state as a partially opened gripper, rather than the full
                opened gripper. If None, will simply use the native joint limits of the gripper joints. Only relevant
                if using @mode=binary and @motor_type=position
            closed_qpos (None or Array[float]): If specified, the joint positions representing a fully-closed gripper.
                This is to allow representing the closed state as a partially closed gripper, rather than the full
                closed gripper. If None, will simply use the native joint limits of the gripper joints. Only relevant
                if using @mode=binary and @motor_type=position
            limit_tolerance (float): sets the tolerance from the joint limit ends, below which controls will be zeroed
                out if the control is using velocity or torque control
        """
        # Store arguments
        assert_valid_key(key=motor_type.lower(), valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        self._motor_type = motor_type.lower()
        assert_valid_key(key=mode, valid_keys=VALID_MODES, name="mode for multi finger gripper")
        self._inverted = inverted
        self._mode = mode
        self._limit_tolerance = limit_tolerance
        self._open_qpos = open_qpos if open_qpos is None else cb.array(open_qpos)
        self._closed_qpos = closed_qpos if closed_qpos is None else cb.array(closed_qpos)

        # Create other args to be filled in at runtime
        self._is_grasping = IsGraspingState.FALSE

        # Create ring buffer for velocity history to avoid high frequency nosie during grasp state inference
        self._vel_filter = MovingAverageFilter(obs_dim=len(dof_idx), filter_width=5)

        # If we're using binary signal, these values will be overridden manually, so set to default for now
        if mode == "binary":
            command_output_limits = "default"

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
        )

    def _generate_default_command_output_limits(self):
        # By default (independent mode), this is simply the super call
        command_output_limits = super()._generate_default_command_output_limits()

        # If we're in binary mode, output limits should just be (-1.0, 1.0)
        if self._mode == "binary":
            command_output_limits = (-1.0, 1.0)
        # If we're in smoothing mode, output limits should be the average of the independent limits
        elif self._mode == "smooth":
            command_output_limits = (
                cb.mean(command_output_limits[0]),
                cb.mean(command_output_limits[1]),
            )
        elif self._mode == "independent":
            pass
        else:
            raise ValueError(f"Invalid mode {self._mode}")

        return command_output_limits

    def reset(self):
        # Call super first
        super().reset()

        # Reset the filter
        self._vel_filter.reset()

        # reset grasping state
        self._is_grasping = IsGraspingState.FALSE

    @property
    def state_size(self):
        # Add state size from the control filter
        return super().state_size + self._vel_filter.state_size

    def _preprocess_command(self, command):
        # We extend this method to make sure command is always n-dimensional
        if self._mode != "independent":
            command = (
                cb.array([command] * self.command_dim)
                if type(command) in {int, float}
                else cb.array([command[0]] * self.command_dim)
            )

        # Flip the command if the direction is inverted.
        if self._inverted:
            command = self._command_input_limits[1] - (command - self._command_input_limits[0])

        # Return from super method
        return super()._preprocess_command(command=command)

    def _update_goal(self, command, control_dict):
        # Directly store command as the goal
        return dict(target=command)

    def compute_control(self, goal_dict, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) gripper
        joint control signal

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    target: desired gripper target
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        target = goal_dict["target"]
        joint_pos = control_dict["joint_position"][self.dof_idx]
        # Choose what to do based on control mode
        if self._mode == "binary":
            # Use max control signal
            should_open = target[0] >= 0.0 if not self._inverted else target[0] > 0.0
            if should_open:
                u = (
                    self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx]
                    if self._open_qpos is None
                    else self._open_qpos
                )
            else:
                u = (
                    self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx]
                    if self._closed_qpos is None
                    else self._closed_qpos
                )
        else:
            # Use continuous signal. Make sure to go from command to control dim.
            u = cb.full((self.control_dim,), target[0]) if len(target) == 1 else target

        # If we're near the joint limits and we're using velocity / torque control, we zero out the action
        if self._motor_type in {"velocity", "torque"}:
            violate_upper_limit = (
                joint_pos > self._control_limits[ControlType.POSITION][1][self.dof_idx] - self._limit_tolerance
            )
            violate_lower_limit = (
                joint_pos < self._control_limits[ControlType.POSITION][0][self.dof_idx] + self._limit_tolerance
            )
            violation = cb.logical_or(violate_upper_limit * (u > 0), violate_lower_limit * (u < 0))
            u *= ~violation

        # Update whether we're grasping or not
        self._update_grasping_state(control_dict=control_dict)

        # Return control
        return u

    def _update_grasping_state(self, control_dict):
        """
        Updates internal inferred grasping state of the gripper being controlled by this gripper controller

        Args:
            control_dict (dict): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:

                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities
        """
        # Update velocity history
        finger_vel = self._vel_filter.estimate(control_dict["joint_velocity"][self.dof_idx])

        # Calculate grasping state based on mode of this controller
        # Independent mode of MultiFingerGripperController does not have any good heuristics to determine is_grasping
        if self._mode == "independent":
            is_grasping = IsGraspingState.UNKNOWN

        # No control has been issued before -- we assume not grasping
        elif self._control is None:
            is_grasping = IsGraspingState.FALSE

        #  Different values in the command for non-independent mode - cannot use heuristics
        elif not cb.all(self._control == self._control[0]):
            is_grasping = IsGraspingState.UNKNOWN

        # Joint position tolerance for is_grasping heuristics checking is smaller than or equal to the gripper
        # controller's tolerance of zero-ing out velocities, which makes the heuristics invalid.
        elif not m.POS_TOLERANCE > self._limit_tolerance:
            is_grasping = IsGraspingState.UNKNOWN

        else:
            finger_pos = control_dict["joint_position"][self.dof_idx]

            # For joint position control, if the desired positions are the same as the current positions, is_grasping unknown
            if self._motor_type == "position" and cb.abs(finger_pos - self._control).mean() < m.POS_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN

            # For joint velocity / torque control, if the desired velocities / torques are zeros, is_grasping unknown
            elif self._motor_type in {"velocity", "torque"} and cb.abs(self._control).mean() < m.VEL_TOLERANCE:
                is_grasping = IsGraspingState.UNKNOWN

            # Otherwise, the last control signal intends to "move" the gripper
            else:
                min_pos = self._control_limits[ControlType.POSITION][0][self.dof_idx]
                max_pos = self._control_limits[ControlType.POSITION][1][self.dof_idx]

                # Make sure we don't have any invalid values (i.e.: fingers should be within the limits)
                finger_pos = finger_pos.clip(min_pos, max_pos)

                # Check distance from both ends of the joint limits
                dist_from_lower_limit = finger_pos - min_pos
                dist_from_upper_limit = max_pos - finger_pos

                # If either of the joint positions are not near the joint limits with some tolerance (m.POS_TOLERANCE)
                valid_grasp_pos = (
                    dist_from_lower_limit.mean() > m.POS_TOLERANCE or dist_from_upper_limit.mean() > m.POS_TOLERANCE
                )

                # And the joint velocities are close to zero with some tolerance (m.VEL_TOLERANCE)
                valid_grasp_vel = cb.all(cb.abs(finger_vel) < m.VEL_TOLERANCE)

                # Then the gripper is grasping something, which stops the gripper from reaching its desired state
                is_grasping = IsGraspingState.TRUE if valid_grasp_pos and valid_grasp_vel else IsGraspingState.FALSE

        # Store calculated state
        self._is_grasping = is_grasping

    def compute_no_op_goal(self, control_dict):
        # Take care of the special case of binary control
        if self._mode == "binary":
            goal_sign = -1 if self.is_grasping() == IsGraspingState.TRUE else 1
            if self._inverted:
                goal_sign = -1 * goal_sign
            target = cb.array([goal_sign])

        else:
            if self._motor_type == "position":
                target = control_dict["joint_position"][self.dof_idx]
            elif self._motor_type == "velocity":
                target = cb.zeros(self.command_dim)
            else:
                raise ValueError("Cannot compute noop action for effort motor type.")

            # Convert to binary / smooth mode if necessary
            if self._mode == "smooth":
                target = cb.mean(target, dim=-1, keepdim=True)

        return dict(target=target)

    def _compute_no_op_command(self, control_dict):
        # Take care of the special case of binary control
        if self._mode == "binary":
            command_val = -1 if self.is_grasping() == IsGraspingState.TRUE else 1
            if self._inverted:
                command_val = -1 * command_val
            return cb.array([command_val])

        if self._motor_type == "position":
            command = control_dict["joint_position"][self.dof_idx]
        elif self._motor_type == "velocity":
            command = cb.zeros(self.command_dim)
        else:
            raise ValueError("Cannot compute noop action for effort motor type.")

        # Convert to binary / smooth mode if necessary
        if self._mode == "smooth":
            command = cb.mean(command, dim=-1, keepdim=True)

        return command

    def _get_goal_shapes(self):
        return dict(target=(self.command_dim,))

    def is_grasping(self):
        # Return cached value
        return self._is_grasping

    def _dump_state(self):
        # Run super first
        state = super()._dump_state()

        # Add filter state
        state["vel_filter"] = self._vel_filter.dump_state(serialized=False)

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Also load velocity filter state if we've set a goal
        if self._goal is not None:
            self._vel_filter.load_state(state["vel_filter"], serialized=False)

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize state for this controller
        return th.cat([state_flat, self._vel_filter.serialize(state=state["vel_filter"])])

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize state for the velocity filter
        state_dict["vel_filter"], deserialized_items = self._vel_filter.deserialize(state=state[idx:])
        idx += deserialized_items

        return state_dict, idx

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return len(self.dof_idx) if self._mode == "independent" else 1
