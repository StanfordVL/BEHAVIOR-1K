import torch as th

from omnigibson.controllers import ControlType, GripperController, IsGraspingState
from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

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

        # Per-member grasping state and velocity filters (indexed by controller_idx)
        self._is_grasping = []  # list of IsGraspingState per member
        self._vel_filter = None  # single batched MovingAverageFilter for all members
        # Last control per member (for grasping heuristic)
        self._controls = []

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

    def add_member(self, articulation_root_path, control_enabled=True):
        idx = super().add_member(articulation_root_path, control_enabled=control_enabled)
        if idx < len(self._is_grasping):
            # Reusing a tombstoned slot — reset per-member grasping state
            self._is_grasping[idx] = IsGraspingState.FALSE
        else:
            # New slot — append state
            self._is_grasping.append(IsGraspingState.FALSE)
        if self._vel_filter is None:
            # First-ever member: create the batched filter (idx is always 0 here)
            self._vel_filter = MovingAverageFilter(obs_dim=len(self.dof_idx), filter_width=5, n_members=1)
        else:
            # Pass idx so the filter reuses the slot in-place or appends as appropriate
            self._vel_filter.add_member(idx)
        # Note: _controls is managed by BaseController.add_member; do not append here
        return idx

    def unregister_member(self, controller_idx):
        """Mark member at controller_idx as a tombstone in both controller and velocity filter.

        Args:
            controller_idx (int): index of the member to unregister
        """
        super().unregister_member(controller_idx)
        if self._vel_filter is not None:
            self._vel_filter.unregister_member(controller_idx)

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

    def reset(self, controller_idx):
        # Call super first
        super().reset(controller_idx)

        # Reset the filter and grasping state
        self._vel_filter.reset(controller_idx)
        self._is_grasping[controller_idx] = IsGraspingState.FALSE

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

    def _update_goal(self, controller_idx, command):
        # Directly store command as the goal (compute-backend array)
        return dict(target=command)

    def compute_control(self, goals):
        """
        Converts the (already preprocessed) batched goals into deployable (non-clipped!) gripper
        joint control signals for all N group members.

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    target: (N, command_dim) desired gripper target

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """
        target_batch = goals["target"]  # (N, command_dim)

        rows = self.view_row_indices
        all_joint_pos = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :][
            :, self.dof_idx
        ]  # (N, ctrl_dim)

        unregistered_mask = self._unregistered_controllers == 1  # (N,)

        # Choose what to do based on control mode
        if self._mode == "binary":
            should_open = target_batch[:, 0] >= 0.0 if not self._inverted else target_batch[:, 0] > 0.0  # (N,)
            open_limit = (
                self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx]
                if self._open_qpos is None
                else self._open_qpos
            )  # (ctrl_dim,)
            closed_limit = (
                self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx]
                if self._closed_qpos is None
                else self._closed_qpos
            )  # (ctrl_dim,)
            u = cb.where(should_open[:, None], open_limit, closed_limit)  # (N, ctrl_dim)
        else:
            # Broadcast single-column target across control_dim if needed
            if target_batch.shape[1] == 1:
                u = target_batch * cb.ones(self.control_dim)
            else:
                u = target_batch  # (N, ctrl_dim)

        # If we're near the joint limits and we're using velocity / effort control, we zero out the action
        if self._motor_type in {"velocity", "effort"}:
            pos_hi = self._control_limits[ControlType.POSITION][1][self.dof_idx]  # (ctrl_dim,)
            pos_lo = self._control_limits[ControlType.POSITION][0][self.dof_idx]  # (ctrl_dim,)
            violate_upper_limit = all_joint_pos > pos_hi - self._limit_tolerance  # (N, ctrl_dim)
            violate_lower_limit = all_joint_pos < pos_lo + self._limit_tolerance  # (N, ctrl_dim)
            violation = (violate_upper_limit & (u > 0)) | (violate_lower_limit & (u < 0))
            u = u * ~violation

        # Update grasping state for all members
        self._update_grasping_state(all_joint_pos, u)

        # Zero out unregistered members
        u[unregistered_mask] = 0.0

        return u  # array with shape (N, control_dim)

    def _update_grasping_state(self, joint_pos, control):
        """
        Updates internal inferred grasping state for the controller at @controller_idx.

        Args:
            joint_pos (Array): joint positions for this group's members' controlled DOFs, shape (N, ctrl_dim)
            control (Array): the control signal being applied, shape (N, ctrl_dim)
        """
        rows = self.view_row_indices
        all_joint_vel = ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=True)[rows, :][
            :, self.dof_idx
        ]  # (N, ctrl_dim)

        # Update velocity history for all members
        finger_vels = self._vel_filter.estimate_batch(all_joint_vel)  # (N, ctrl_dim)

        # Calculate grasping state based on mode of this controller
        if self._mode == "independent":
            is_grasping_result = [IsGraspingState.UNKNOWN] * self.n_members

        else:
            # Different values in the command for non-independent mode - cannot use heuristics
            non_uniform_mask = ~cb.all(control == control[:, :1], 1)  # (N,)

            # Joint position tolerance for is_grasping heuristics checking is smaller than or equal to the gripper
            # controller's tolerance of zero-ing out velocities, which makes the heuristics invalid.
            if not m.POS_TOLERANCE > self._limit_tolerance:
                is_grasping_result = [IsGraspingState.UNKNOWN] * self.n_members

            else:
                # For joint position control, if the desired positions are the same as the current positions, is_grasping unknown
                if self._motor_type == "position":
                    no_move_mask = cb.mean(cb.abs(control - joint_pos), dim=1) < m.POS_TOLERANCE  # (N,)
                # For joint velocity / effort control, if the desired velocities / efforts are zeros, is_grasping unknown
                elif self._motor_type in {"velocity", "effort"}:
                    no_move_mask = cb.mean(cb.abs(control), dim=1) < m.VEL_TOLERANCE  # (N,)
                else:
                    no_move_mask = cb.bool_zeros(self.n_members)  # all-False

                # Otherwise, the last control signal intends to "move" the gripper
                min_pos = self._control_limits[ControlType.POSITION][0][self.dof_idx]  # (ctrl_dim,)
                max_pos = self._control_limits[ControlType.POSITION][1][self.dof_idx]  # (ctrl_dim,)
                # Make sure we don't have any invalid values (i.e.: fingers should be within the limits)
                finger_pos = joint_pos.clip(min_pos, max_pos)  # (N, ctrl_dim)
                # Check distance from both ends of the joint limits
                dist_from_lower_limit = finger_pos - min_pos  # (N, ctrl_dim)
                dist_from_upper_limit = max_pos - finger_pos  # (N, ctrl_dim)

                # If either of the joint positions are not near the joint limits with some tolerance (m.POS_TOLERANCE)
                valid_grasp_pos = (cb.mean(dist_from_lower_limit, dim=1) > m.POS_TOLERANCE) | (
                    cb.mean(dist_from_upper_limit, dim=1) > m.POS_TOLERANCE
                )  # (N,)

                # And the joint velocities are close to zero with some tolerance (m.VEL_TOLERANCE)
                valid_grasp_vel = cb.all(cb.abs(finger_vels) < m.VEL_TOLERANCE, 1)  # (N,)

                # Then the gripper is grasping something, which stops the gripper from reaching its desired state
                is_grasping_true = valid_grasp_pos & valid_grasp_vel  # (N,)

                # Build per-member result: UNKNOWN overrides where non_uniform or no_move
                is_grasping_result = [
                    IsGraspingState.UNKNOWN
                    if (non_uniform_mask[i] or no_move_mask[i])
                    else (IsGraspingState.TRUE if is_grasping_true[i] else IsGraspingState.FALSE)
                    for i in range(self.n_members)
                ]

        # Store calculated state
        self._is_grasping = is_grasping_result

    def compute_no_op_goal(self, controller_idx):
        """
        Returns:
            dict: ``target`` as a compute-backend array (shape matches command space).
        """
        prim_path = self._articulation_root_paths[controller_idx]

        # Take care of the special case of binary control
        if self._mode == "binary":
            goal_sign = -1 if self._is_grasping[controller_idx] == IsGraspingState.TRUE else 1
            if self._inverted:
                goal_sign = -1 * goal_sign
            target = cb.array([goal_sign])

        else:
            if self._motor_type == "position":
                target = ControllableObjectViewAPI.get_joint_positions(prim_path)[self.dof_idx]
            elif self._motor_type == "velocity":
                target = cb.zeros(self.command_dim)
            else:
                raise ValueError("Cannot compute noop action for effort motor type.")

            # Convert to binary / smooth mode if necessary
            if self._mode == "smooth":
                target = cb.mean(target, dim=-1, keepdim=True)

        return dict(target=target)

    def _compute_no_op_command(self, controller_idx):
        prim_path = self._articulation_root_paths[controller_idx]

        # Take care of the special case of binary control
        if self._mode == "binary":
            command_val = -1 if self._is_grasping[controller_idx] == IsGraspingState.TRUE else 1
            if self._inverted:
                command_val = -1 * command_val
            return cb.array([command_val])

        if self._motor_type == "position":
            command = ControllableObjectViewAPI.get_joint_positions(prim_path)[self.dof_idx]
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

    def is_grasping(self, controller_idx=0):
        # Return cached value for this member
        return self._is_grasping[controller_idx]

    def _dump_state(self, controller_idx):
        # Run super first
        state = super()._dump_state(controller_idx=controller_idx)
        state["vel_filter"] = None if self._vel_filter is None else self._vel_filter.dump_state(controller_idx)
        return state

    def _load_state(self, controller_idx, state):
        # Run super first
        super()._load_state(controller_idx=controller_idx, state=state)

        # Also load velocity filter state for this single member.
        if self._vel_filter is not None and state.get("vel_filter") is not None:
            self._vel_filter.load_state(controller_idx, state["vel_filter"])
        elif self._vel_filter is not None and not state["goal_set"]:
            self._vel_filter.reset(controller_idx)

    def serialize(self, state, controller_idx):
        # Run super first
        state_flat = super().serialize(state=state, controller_idx=controller_idx)
        filter_flat = (
            self._vel_filter.serialize(state["vel_filter"], controller_idx)
            if self._vel_filter is not None and state.get("vel_filter") is not None
            else th.tensor([])
        )
        return th.cat([state_flat, filter_flat])

    def deserialize(self, state, controller_idx):
        state_dict, idx = super().deserialize(state=state, controller_idx=controller_idx)
        state_dict["vel_filter"] = None
        if self._vel_filter is not None:
            state_dict["vel_filter"], samples_len = self._vel_filter.deserialize(state[idx:], controller_idx)
            idx += samples_len
        return state_dict, idx

    @property
    def state_size(self):
        if self._vel_filter is None:
            return super().state_size
        return super().state_size + self._vel_filter.state_size

    @property
    def control_type(self):
        return ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return len(self.dof_idx) if self._mode == "independent" else 1
