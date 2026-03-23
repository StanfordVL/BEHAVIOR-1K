from omnigibson.controllers import ControlType, LocomotionController
from omnigibson.utils.backend_utils import _compute_backend as cb


class DifferentialDriveController(LocomotionController):
    """
    Differential drive (DD) controller for controlling two independently controlled wheeled joints.

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Convert desired (lin_vel, ang_vel) command into (left, right) wheel joint velocity control signals
        3. Clips the resulting command by the joint velocity limits
    """

    def __init__(
        self,
        wheel_radius,
        wheel_axle_length,
        control_freq,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        isaac_kp=None,
        isaac_kd=None,
    ):
        """
        Args:
            wheel_radius (float): radius of the wheels (both assumed to be same radius)
            wheel_axle_length (float): perpendicular distance between the two wheels
            control_freq (int): controller loop frequency
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
                to the maximum linear and angular velocities calculated from @wheel_radius, @wheel_axle_length, and
                @control_limits velocity limits entry
            isaac_kp (None or float or Array[float]): If specified, stiffness gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers.
                Should only be nonzero if self.control_type is position
            isaac_kd (None or float or Array[float]): If specified, damping gains to apply to the underlying
                isaac DOFs. Can either be a single number or a per-DOF set of numbers
                Should only be nonzero if self.control_type is position or velocity
        """
        # Store internal variables
        self._wheel_radius = wheel_radius
        self._wheel_axle_halflength = wheel_axle_length / 2.0

        # If we're using default command output limits, map this to maximum linear / angular velocities
        if type(command_output_limits) is str and command_output_limits == "default":
            min_vels = control_limits["velocity"][0][dof_idx]
            assert (
                min_vels[0] == min_vels[1]
            ), "Differential drive requires both wheel joints to have same min velocities!"
            max_vels = control_limits["velocity"][1][dof_idx]
            assert (
                max_vels[0] == max_vels[1]
            ), "Differential drive requires both wheel joints to have same max velocities!"
            assert abs(min_vels[0]) == abs(
                max_vels[0]
            ), "Differential drive requires both wheel joints to have same min and max absolute velocities!"
            max_lin_vel = max_vels[0] * wheel_radius
            max_ang_vel = max_lin_vel * 2.0 / wheel_axle_length
            command_output_limits = ((-max_lin_vel, -max_ang_vel), (max_lin_vel, max_ang_vel))

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

    def _update_goal(self, command, control_dict):
        # Directly store command as the velocity goal
        return dict(vel=command)

    def compute_control(self, goal_dict, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal.
        This processes converts the desired (lin_vel, ang_vel) command into (left, right) wheel joint velocity control
        signals.

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    vel: desired (lin_vel, ang_vel) of the controlled body
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:

        Returns:
            Array[float]: outputted (non-clipped!) velocity control signal to deploy
                to the [left, right] wheel joints
        """
        lin_vel, ang_vel = goal_dict["vel"]

        # Convert to wheel velocities
        left_wheel_joint_vel = (lin_vel - ang_vel * self._wheel_axle_halflength) / self._wheel_radius
        right_wheel_joint_vel = (lin_vel + ang_vel * self._wheel_axle_halflength) / self._wheel_radius

        # Return desired velocities
        return cb.array([left_wheel_joint_vel, right_wheel_joint_vel])

    def compute_no_op_goal(self, control_dict):
        # This is zero-vector, since we want zero linear / angular velocity
        return dict(vel=cb.zeros(2))

    def _compute_no_op_command(self, control_dict):
        return cb.zeros(2)

    def _get_goal_shapes(self):
        # Add (2, )-array representing linear, angular velocity
        return dict(vel=(2,))

    @property
    def control_type(self):
        return ControlType.VELOCITY

    @property
    def command_dim(self):
        # [lin_vel, ang_vel]
        return 2
