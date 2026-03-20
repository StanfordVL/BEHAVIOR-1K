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
            dof_idx (Array[int]): specific dof indices controlled by this controller. Used for inferring
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

        # Precompute (2, 2) transform: vel_batch (N, 2) @ _wheel_vel_transform -> (N, 2) [left, right] wheel vels
        # left  = lin_vel / r - ang_vel * half / r
        # right = lin_vel / r + ang_vel * half / r
        inv_r = 1.0 / wheel_radius
        half_inv_r = self._wheel_axle_halflength / wheel_radius
        self._wheel_vel_transform = cb.array([[inv_r, inv_r], [-half_inv_r, half_inv_r]])  # (2, 2)

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

    def _update_goal(self, controller_idx, command):
        # Directly store command as the velocity goal (compute-backend array)
        return dict(vel=command)

    def compute_control(self, goals):
        """
        Converts the (already preprocessed) batched goals into deployable (non-clipped!) joint control signals
        for all N group members.

        Args:
            goals (Dict[str, Tensor]): batched goals with shape (N, *shape) per key.
                Must include:
                    vel: (N, 2) desired (lin_vel, ang_vel) of the controlled bodies

        Returns:
            Tensor: (N, 2) outputted (non-clipped!) velocity control signal to deploy to the [left, right] wheel joints
        """
        # (N, 2) @ (2, 2) -> (N, 2) [left, right] wheel joint velocities
        return goals["vel"] @ self._wheel_vel_transform

    def compute_no_op_goal(self, controller_idx):
        # Zero (lin, ang) velocity as ``cb`` array
        return dict(vel=cb.zeros(2))

    def _compute_no_op_command(self, controller_idx):
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
