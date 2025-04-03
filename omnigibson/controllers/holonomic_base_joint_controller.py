from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.controllers.joint_controller import JointController
from omnigibson.utils.geometry_utils import wrap_angle


class HolonomicBaseJointController(JointController):
    """
    Controller class for holonomic base joint control. This is a very specific type of controller used to model control of a 3DOF
    holonomic robot base -- i.e.: free motion along (x, y, rz).

    NOTE: Inputted commands are ALWAYS assumed to be in the form of absolute values (defined in the robot's local frame), not deltas!
    NOTE: Also assumes commands are ALWAYS inputted in the form of [x, y, rz]!

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
        command_input_limits=None,
        command_output_limits=None,
        isaac_kp=None,
        isaac_kd=None,
        pos_kp=None,
        pos_damping_ratio=None,
        vel_kp=None,
        use_impedances=False,
        use_gravity_compensation=False,
        use_cc_compensation=True,
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
            pos_kp (None or float): If @motor_type is "position" and @use_impedances=True, this is the
                proportional gain applied to the joint controller. If None, a default value will be used.
            pos_damping_ratio (None or float): If @motor_type is "position" and @use_impedances=True, this is the
                damping ratio applied to the joint controller. If None, a default value will be used.
            vel_kp (None or float): If @motor_type is "velocity" and @use_impedances=True, this is the
                proportional gain applied to the joint controller. If None, a default value will be used.
            use_impedances (bool): If True, will use impedances via the mass matrix to modify the desired efforts
                applied
            use_gravity_compensation (bool): If True, will add gravity compensation to the computed efforts. This is
                an experimental feature that only works on fixed base robots. We do not recommend enabling this.
            use_cc_compensation (bool): If True, will add Coriolis / centrifugal compensation to the computed efforts.
        """
        # Make sure we're controlling exactly 3 DOFs
        assert len(dof_idx) == 3, f"Expected 3 DOFs for holonomic base control, got {len(dof_idx)}"

        # Run super init
        super().__init__(
            control_freq=control_freq,
            motor_type=motor_type,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            vel_kp=vel_kp,
            use_impedances=use_impedances,
            use_gravity_compensation=use_gravity_compensation,
            use_cc_compensation=use_cc_compensation,
            use_delta_commands=False,
            compute_delta_in_quat_space=None,
        )

    def _update_goal(self, command, control_dict):
        """
        Updates the goal command by transforming it from the robot's local frame to its canonical frame.
        """
        base_pose = cb.T.pose2mat((control_dict["root_pos"], control_dict["root_quat"]))
        canonical_pose = cb.T.pose2mat((control_dict["canonical_pos"], control_dict["canonical_quat"]))
        canonical_to_base_pose = cb.T.pose_inv(canonical_pose) @ base_pose

        if self.motor_type == "position":
            # Handle position control mode
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]  # Set x,y translation

            # Transform command to canonical frame
            command_in_canonical_frame = canonical_to_base_pose @ command_in_base_frame

            # Extract x,y translation from transformed command
            position = command_in_canonical_frame[:2, 3]

            # Since our virtual joints are in the order of ["x", "y", "z", "rx", "ry", "rz"],
            # the "base_footprint_rz_link" is aligned with @self.base_footprint_link in the z-axis
            # We just need to directly apply the command as delta to the current joint position of "base_footprint_rz_joint"
            # Note that the current joint position is guaranteed to be in the range of [-pi, pi] because
            # @HolonomicBaseRobot.apply_action explicitly wraps the joint position to [-pi, pi] if it's out of range
            rz_joint_pos = control_dict["joint_position"][self.dof_idx][2:3]

            # Wrap the delta joint position to [-pi, pi]. In other words, we don't expect the commanded delta joint position
            # to have a magnitude greater than pi, because the robot can't reasonably rotate more than pi in a single step
            delta_joint_pos = wrap_angle(command[2])

            # Calculate the new joint position. This is guaranteed to be in the range of [-pi * 2, pi * 2], because both quantities
            # are in the range of [-pi, pi]. This is important because revolute joints in Isaac Sim have a joint position (target) range of [-pi * 2, pi * 2]
            new_joint_pos = rz_joint_pos + delta_joint_pos

            command = cb.cat([position, new_joint_pos])
        else:
            # Handle velocity/effort control modes
            # Note: Only rotate the commands, don't translate
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]  # Set x,y linear velocity

            canonical_to_base_pose_rotation = cb.as_float32(cb.eye(4))
            canonical_to_base_pose_rotation[:3, :3] = canonical_to_base_pose[:3, :3]

            # Transform command to canonical frame
            command_in_canonical_frame = canonical_to_base_pose_rotation @ command_in_base_frame

            # Extract x,y linear velocity from transformed command
            linear_velocity = command_in_canonical_frame[:2, 3]

            angular_velocity = command[2:3]

            command = cb.cat([linear_velocity, angular_velocity])

        return super()._update_goal(command=command, control_dict=control_dict)

    # For "position" control mode, this controller behaves similar to use_delta_commands=True,
    # where the command [dx, dy, drz] means the robot should move by [dx, dy] and rotate by drz (in the base link frame)
    # For "velocity" and "effort" control modes, this controller behaves similar to use_delta_commands=False,
    # where the command [vx, vy, vrz] means the robot should move with linear velocity [vx, vy] and angular velocity vrz (in the base link frame)
    # In all cases, no-op commands should be [0, 0, 0].
    def _compute_no_op_command(self, control_dict):
        return cb.zeros(self.command_dim)
