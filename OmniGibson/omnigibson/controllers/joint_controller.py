import math

import numpy as np
import torch as th
from numba import jit

from omnigibson.controllers import (
    ControlType,
    GripperController,
    IsGraspingState,
    LocomotionController,
    ManipulationController,
)
from omnigibson.macros import create_module_macros
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.utils.python_utils import assert_valid_key, torch_compile
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)
m.DEFAULT_JOINT_POS_KP = 50.0
m.DEFAULT_JOINT_POS_DAMPING_RATIO = 1.0  # critically damped
m.DEFAULT_JOINT_VEL_KP = 2.0


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
        isaac_kp=None,
        isaac_kd=None,
        pos_kp=None,
        pos_damping_ratio=None,
        vel_kp=None,
        use_impedances=False,
        use_gravity_compensation=False,
        use_cc_compensation=True,
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

        # Store control gains
        if self._motor_type == "position":
            pos_kp = m.DEFAULT_JOINT_POS_KP if pos_kp is None else pos_kp
            pos_damping_ratio = m.DEFAULT_JOINT_POS_DAMPING_RATIO if pos_damping_ratio is None else pos_damping_ratio
        elif self._motor_type == "velocity":
            vel_kp = m.DEFAULT_JOINT_VEL_KP if vel_kp is None else vel_kp
            assert (
                pos_damping_ratio is None
            ), "Cannot set pos_damping_ratio for JointController with motor_type=velocity!"
        else:  # effort
            assert pos_kp is None, "Cannot set pos_kp for JointController with motor_type=effort!"
            assert pos_damping_ratio is None, "Cannot set pos_damping_ratio for JointController with motor_type=effort!"
            assert vel_kp is None, "Cannot set vel_kp for JointController with motor_type=effort!"
        self.pos_kp = pos_kp
        self.pos_kd = None if pos_kp is None or pos_damping_ratio is None else 2 * math.sqrt(pos_kp) * pos_damping_ratio
        self.vel_kp = vel_kp
        self._use_impedances = use_impedances
        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        # Warn the user about gravity compensation being experimental.
        if self._use_gravity_compensation:
            log.warning(
                "JointController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        # When in delta mode, it doesn't make sense to infer output range using the joint limits (since that's an
        # absolute range and our values are relative). So reject the default mode option in that case.
        assert not (
            self._use_delta_commands and type(command_output_limits) is str and command_output_limits == "default"
        ), "Cannot use 'default' command output limits in delta commands mode of JointController. Try None instead."

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
        # Use motor type instead of default control type, since, e.g, use_impedances is commanding joint positions
        # but controls low-level efforts
        return (
            self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx],
            self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx],
        )

    def _update_goal(self, command, control_dict):
        # If we're using delta commands, add this value
        if self._use_delta_commands:
            # Compute the base value for the command
            base_value = control_dict[f"joint_{self._motor_type}"][self.dof_idx]

            # Apply the command to the base value.
            target = base_value + command

            # Correct any gimbal lock issues using the compute_delta_in_quat_space group.
            for rx_ind, ry_ind, rz_ind in self._compute_delta_in_quat_space:
                # Grab the starting rotations of these joints.
                start_rots = base_value[[rx_ind, ry_ind, rz_ind]]

                # Grab the delta rotations.
                delta_rots = command[[rx_ind, ry_ind, rz_ind]]

                # Compute the final rotations in the quaternion space.
                _, end_quat = cb.T.pose_transform(
                    cb.zeros(3), cb.T.euler2quat(delta_rots), cb.zeros(3), cb.T.euler2quat(start_rots)
                )
                end_rots = cb.T.quat2euler(end_quat)

                # Update the command
                target[[rx_ind, ry_ind, rz_ind]] = end_rots

        # Otherwise, goal is simply the command itself
        else:
            target = command

        # Clip the command based on the limits
        target = target.clip(
            self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx],
            self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx],
        )

        return dict(target=target)

    def compute_control(self, goal_dict, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    target: desired N-dof absolute joint values used as setpoint
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities
                    joint_effort: Array of current joint effort

        Returns:
            Array[float]: outputted (non-clipped!) control signal to deploy
        """
        base_value = control_dict[f"joint_{self._motor_type}"][self.dof_idx]
        target = goal_dict["target"]

        # Convert control into efforts
        if self._use_impedances:
            if self._motor_type == "position":
                # Run impedance controller -- effort = pos_err * kp + vel_err * kd
                position_error = target - base_value
                vel_pos_error = -control_dict["joint_velocity"][self.dof_idx]
                u = position_error * self.pos_kp + vel_pos_error * self.pos_kd
            elif self._motor_type == "velocity":
                # Compute command torques via PI velocity controller plus gravity compensation torques
                velocity_error = target - base_value
                u = velocity_error * self.vel_kp
            else:  # effort
                u = target

            u = cb.get_custom_method("compute_joint_torques")(u, control_dict["mass_matrix"], self.dof_idx)

            # Add gravity compensation
            if self._use_gravity_compensation:
                u += control_dict["gravity_force"][self.dof_idx]

            # Add Coriolis / centrifugal compensation
            if self._use_cc_compensation:
                u += control_dict["cc_force"][self.dof_idx]

        else:
            # Desired is the exact goal
            u = target

        # Return control
        return u

    def compute_no_op_goal(self, control_dict):
        # Compute based on mode
        if self._motor_type == "position":
            # Maintain current qpos
            target = control_dict[f"joint_{self._motor_type}"][self.dof_idx]
        else:
            # For velocity / effort, directly set to 0
            target = cb.zeros(self.control_dim)

        return dict(target=target)

    def _compute_no_op_command(self, control_dict):
        if self.motor_type == "position":
            if self._use_delta_commands:
                return cb.zeros(self.command_dim)
            else:
                return control_dict["joint_position"][self.dof_idx]
        elif self.motor_type == "velocity":
            if self._use_delta_commands:
                return -control_dict["joint_velocity"][self.dof_idx]
            else:
                return cb.zeros(self.command_dim)

        raise ValueError("Cannot compute noop action for effort motor type.")

    def _get_goal_shapes(self):
        return dict(target=(self.control_dim,))

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
    def motor_type(self):
        """
        Returns:
            str: The type of motor being simulated by this controller. One of {"position", "velocity", "effort"}
        """
        return self._motor_type

    @property
    def control_type(self):
        return ControlType.EFFORT if self._use_impedances else ControlType.get_type(type_str=self._motor_type)

    @property
    def command_dim(self):
        return len(self.dof_idx)


@torch_compile
def _compute_joint_torques_torch(
    u: th.Tensor,
    mm: th.Tensor,
    dof_idx: th.Tensor,
):
    dof_idxs_mat = th.meshgrid(dof_idx, dof_idx, indexing="xy")
    return mm[dof_idxs_mat] @ u


# Use numba since faster
@jit(nopython=True)
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.

    Implementation from:
    https://github.com/numba/numba/issues/5894#issuecomment-974701551

    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start : start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@jit(nopython=True)
def _compute_joint_torques_numpy(
    u,
    mm,
    dof_idx,
):
    return numba_ix(mm, dof_idx, dof_idx) @ u


# Set these as part of the backend values
add_compute_function(
    name="compute_joint_torques", np_function=_compute_joint_torques_numpy, th_function=_compute_joint_torques_torch
)
