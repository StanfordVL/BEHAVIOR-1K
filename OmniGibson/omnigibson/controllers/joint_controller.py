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
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

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
        smoothing_filter_size=None,
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
            dof_idx (Array[int]): specific dof indices controlled by this controller. Used for inferring
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
            smoothing_filter_size (None or int): if specified, sets the size of a moving average filter to apply
                on all outputted joint positions.
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

        # Possibly create control filter
        command_dim = len(dof_idx)
        self.control_filter = (
            None
            if smoothing_filter_size in {None, 0}
            else MovingAverageFilter(obs_dim=command_dim, filter_width=smoothing_filter_size)
        )

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
        self._smoothing_filter_size = smoothing_filter_size
        self._filter_obs_dim = len(dof_idx)
        self._control_filter = None  # single batched filter for all members

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

    def add_member(self, articulation_root_path, control_enabled=True):
        idx = super().add_member(articulation_root_path, control_enabled=control_enabled)
        if self._smoothing_filter_size not in {None, 0}:
            if self._control_filter is None:
                # First-ever member: create the batched filter (idx is always 0 here)
                self._control_filter = MovingAverageFilter(
                    obs_dim=self._filter_obs_dim,
                    filter_width=self._smoothing_filter_size,
                    n_members=1,
                )
            else:
                # Pass idx so the filter reuses the slot in-place or appends as appropriate
                self._control_filter.add_member(idx)
        return idx

    def unregister_member(self, controller_idx):
        """Mark member at controller_idx as a tombstone in both controller and smoothing filter.

        Args:
            controller_idx (int): index of the member to unregister
        """
        super().unregister_member(controller_idx)
        if self._control_filter is not None:
            self._control_filter.unregister_member(controller_idx)

    def reset(self, controller_idx):
        super().reset(controller_idx)
        if self._control_filter is not None:
            self._control_filter.reset(controller_idx)

    @property
    def state_size(self):
        if self._control_filter is None:
            return super().state_size
        return super().state_size + self._control_filter.state_size

    def _dump_state(self, controller_idx):
        state = super()._dump_state(controller_idx=controller_idx)
        state["control_filter"] = (
            None if self._control_filter is None else self._control_filter.dump_state(controller_idx)
        )
        return state

    def _load_state(self, controller_idx, state):
        super()._load_state(controller_idx=controller_idx, state=state)
        if self._control_filter is not None and state.get("control_filter") is not None:
            self._control_filter.load_state(controller_idx, state["control_filter"])

    def serialize(self, state, controller_idx):
        state_flat = super().serialize(state=state, controller_idx=controller_idx)
        filter_part = (
            th.tensor([])
            if self._control_filter is None or state.get("control_filter") is None
            else self._control_filter.serialize(state["control_filter"], controller_idx)
        )
        return th.cat([state_flat, filter_part])

    def deserialize(self, state, controller_idx):
        state_dict, idx = super().deserialize(state=state, controller_idx=controller_idx)
        state_dict["control_filter"] = None
        if self._control_filter is not None:
            state_dict["control_filter"], samples_len = self._control_filter.deserialize(state[idx:], controller_idx)
            idx += samples_len
        return state_dict, idx

    def _generate_default_command_output_limits(self):
        # Use motor type instead of default control type, since, e.g, use_impedances is commanding joint positions
        # but controls low-level efforts
        return (
            self._control_limits[ControlType.get_type(self._motor_type)][0][self.dof_idx],
            self._control_limits[ControlType.get_type(self._motor_type)][1][self.dof_idx],
        )

    def _update_goal(self, controller_idx, command):
        """
        Returns:
            dict: ``target`` joint setpoint as a compute-backend array
        """
        # If we're using delta commands, add this value
        if self._use_delta_commands:
            prim_path = self._articulation_root_paths[controller_idx]
            # Compute the base value for the command
            if self._motor_type == "position":
                base_value = ControllableObjectViewAPI.get_joint_positions(prim_path)[self.dof_idx]
            elif self._motor_type == "velocity":
                base_value = ControllableObjectViewAPI.get_joint_velocities(prim_path, estimate=True)[self.dof_idx]
            else:
                base_value = ControllableObjectViewAPI.get_joint_efforts(prim_path)[self.dof_idx]

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

    def compute_control(self, goals):
        """
        Converts the (already preprocessed) batched goals into deployable (non-clipped!) joint control signals
        for all N group members.

        Args:
            goals (Dict[str, Array]): batched goals with shape (N, *shape) per key.
                Must include:
                    target: (N, control_dim) desired joint values used as setpoint

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """

        target = goals["target"]  # (N, control_dim)

        # Optionally pass through smoothing filter for better stability
        if self._control_filter is not None:
            target = self._control_filter.estimate_batch(target)

        # Convert control into efforts
        if self._use_impedances:
            rows = self.view_row_indices
            # Joint indices are defined over actuated joints; generalized dynamics tensors may include extra base DoFs.
            all_joint_positions = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)[rows, :]

            if self._motor_type == "position":
                base_value = all_joint_positions[:, self.dof_idx]
                vel_base = ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=True)[
                    rows, :
                ][:, self.dof_idx]
                position_error = target - base_value
                vel_pos_error = -vel_base
                u = position_error * self.pos_kp + vel_pos_error * self.pos_kd
            elif self._motor_type == "velocity":
                base_value = ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=True)[
                    rows, :
                ][:, self.dof_idx]
                velocity_error = target - base_value
                u = velocity_error * self.vel_kp
            else:  # effort
                u = target

            # Apply impedances via mass matrix (batched over all N members)
            all_mm = ControllableObjectViewAPI.get_all_generalized_mass_matrices(self.routing_path)[
                rows, :, :
            ]  # (N, n_dof_total, n_dof_total)

            # Compute offset between generalized DoFs and actuated joint DoFs (handles floating-base robots).
            base_dof_offset = all_mm.shape[-1] - all_joint_positions.shape[-1]
            if base_dof_offset < 0:
                base_dof_offset = 0
            effective_dof_idx = [idx + base_dof_offset for idx in self.dof_idx]
            dof_idx_arr = cb.int_array(effective_dof_idx)

            u = cb.get_custom_method("compute_joint_torques_batch")(u, all_mm, dof_idx_arr)  # (N, control_dim)

            if self._use_gravity_compensation:
                u = (
                    u
                    + ControllableObjectViewAPI.get_all_gravity_compensation_forces(self.routing_path)[rows, :][
                        :, effective_dof_idx
                    ]
                )

            if self._use_cc_compensation:
                u = (
                    u
                    + ControllableObjectViewAPI.get_all_coriolis_and_centrifugal_compensation_forces(self.routing_path)[
                        rows, :
                    ][:, effective_dof_idx]
                )

        else:
            u = target

        return u

    def compute_no_op_goal(self, controller_idx):
        """
        Returns:
            dict: ``target`` as a compute-backend array (hold position or zeros by motor type)
        """
        prim_path = self._articulation_root_paths[controller_idx]

        if self._motor_type == "position":
            target = ControllableObjectViewAPI.get_joint_positions(prim_path)[self.dof_idx]
        else:
            target = cb.zeros(self.control_dim)

        return dict(target=target)

    def _compute_no_op_command(self, controller_idx):
        prim_path = self._articulation_root_paths[controller_idx]

        if self.motor_type == "position":
            if self._use_delta_commands:
                return cb.zeros(self.command_dim)
            else:
                return ControllableObjectViewAPI.get_joint_positions(prim_path)[self.dof_idx]
        elif self.motor_type == "velocity":
            if self._use_delta_commands:
                return -ControllableObjectViewAPI.get_joint_velocities(prim_path, estimate=True)[self.dof_idx]
            else:
                return cb.zeros(self.command_dim)

        raise ValueError("Cannot compute noop action for effort motor type.")

    def _get_goal_shapes(self):
        return dict(target=(self.control_dim,))

    def is_grasping(self, controller_idx):
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


@torch_compile
def _compute_joint_torques_batch_torch(
    u: th.Tensor,
    mm: th.Tensor,
    dof_idx: th.Tensor,
):
    dof_idxs_mat = th.meshgrid(dof_idx, dof_idx, indexing="xy")
    mm_sub = mm[:, dof_idxs_mat[0], dof_idxs_mat[1]]  # (N, ctrl_dim, ctrl_dim)
    return (mm_sub @ u.unsqueeze(-1)).squeeze(-1)  # (N, ctrl_dim)


@jit(nopython=True)
def _compute_joint_torques_batch_numpy(
    u,
    mm,
    dof_idx,
):
    N = u.shape[0]
    result = np.zeros_like(u)
    for i in range(N):
        result[i] = numba_ix(mm[i], dof_idx, dof_idx) @ u[i]
    return result


add_compute_function(
    name="compute_joint_torques_batch",
    np_function=_compute_joint_torques_batch_numpy,
    th_function=_compute_joint_torques_batch_torch,
)
