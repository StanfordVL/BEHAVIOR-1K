import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.controllers import JointController
from omnigibson.macros import create_module_macros


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
        pos_kp=None,
        pos_damping_ratio=None,
        pos_ki=None,
        max_integral_error=None,
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
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            pos_ki=pos_ki,
            max_integral_error=max_integral_error,
            vel_kp=vel_kp,
            use_impedances=use_impedances,
            use_gravity_compensation=use_gravity_compensation,
            use_cc_compensation=use_cc_compensation,
        )

    def _update_goal(self, command, control_dict):
        """
        Updates the goal command by transforming it from the robot's local frame to its canonical frame.
        """
        base_pose = T.pose2mat((control_dict["root_pos"], control_dict["root_quat"]))
        canonical_pose = T.pose2mat((control_dict["canonical_pos"], control_dict["canonical_quat"]))
        canonical_to_base_pose = T.pose_inv(canonical_pose) @ base_pose

        if self.motor_type == "position":
            # Handle position control mode
            command_in_base_frame = th.eye(4)
            command_in_base_frame[:2, 3] = command[:2]  # Set x,y translation
            command_in_base_frame[:3, :3] = T.euler2mat(th.tensor([0.0, 0.0, command[2]]))  # Set rotation

            # Transform command to canonical frame
            command_in_canonical_frame = canonical_to_base_pose @ command_in_base_frame

            # Extract position and yaw from transformed command
            position = command_in_canonical_frame[:2, 3]
            yaw = T.mat2euler(command_in_canonical_frame[:3, :3])[2:3]
            command = th.cat([position, yaw])
        else:
            # Handle velocity/effort control modes
            # Note: Only rotate the commands, don't translate
            rotation_matrix = canonical_to_base_pose[:3, :3]

            # Prepare poses for transformation
            rotation_poses = th.zeros((2, 3, 3))
            rotation_poses[:, :3, :3] = rotation_matrix

            local_vectors = th.zeros((2, 3, 1))
            local_vectors[0, :, 0] = th.tensor([command[0], command[1], 0.0])  # Linear
            local_vectors[1, :, 0] = th.tensor([0.0, 0.0, command[2]])  # Angular

            # Transform to global frame
            global_vectors = rotation_poses @ local_vectors
            linear_global = global_vectors[0]
            angular_global = global_vectors[1]

            command = th.tensor([linear_global[0], linear_global[1], angular_global[2]])

        return super()._update_goal(command=command, control_dict=control_dict)
