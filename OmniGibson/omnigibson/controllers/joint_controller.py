from copy import deepcopy
import math

import numpy as np
import torch as th
from numba import jit

from omnigibson.controllers.controller_base import BaseController, ControlType, IsGraspingState
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


class JointController(BaseController):
    """
    Controller class for joint control. Because omniverse can handle direct position / velocity / effort
    control signals, this is merely a pass-through operation from command to control (with clipping / scaling built in).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2a. If using delta commands, then adds the command to the current joint state
        2b. Clips the resulting command by the motor limits
    """

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        motor_type = config["motor_type"].lower()
        assert_valid_key(key=motor_type, valid_keys=ControlType.VALID_TYPES_STR, name="motor_type")
        config["motor_type"] = motor_type
        config["use_delta_commands"] = config.get("use_delta_commands", False)
        config["compute_delta_in_quat_space"] = (
            [] if config.get("compute_delta_in_quat_space", None) is None else config["compute_delta_in_quat_space"]
        )

        # Control gains
        pos_kp = config.get("pos_kp", None)
        pos_damping_ratio = config.get("pos_damping_ratio", None)
        vel_kp = config.get("vel_kp", None)
        if motor_type == "position":
            pos_kp = m.DEFAULT_JOINT_POS_KP if pos_kp is None else pos_kp
            pos_damping_ratio = m.DEFAULT_JOINT_POS_DAMPING_RATIO if pos_damping_ratio is None else pos_damping_ratio
        elif motor_type == "velocity":
            vel_kp = m.DEFAULT_JOINT_VEL_KP if vel_kp is None else vel_kp
            assert (
                pos_damping_ratio is None
            ), "Cannot set pos_damping_ratio for JointController with motor_type=velocity!"
        else:  # effort
            assert pos_kp is None, "Cannot set pos_kp for JointController with motor_type=effort!"
            assert pos_damping_ratio is None, "Cannot set pos_damping_ratio for JointController with motor_type=effort!"
            assert vel_kp is None, "Cannot set vel_kp for JointController with motor_type=effort!"

        config["pos_kp"] = pos_kp
        config["pos_kd"] = None if pos_kp is None or pos_damping_ratio is None else 2 * math.sqrt(pos_kp) * pos_damping_ratio
        config["vel_kp"] = vel_kp
        config["use_impedances"] = config.get("use_impedances", False)
        config["use_gravity_compensation"] = config.get("use_gravity_compensation", False)
        config["use_cc_compensation"] = config.get("use_cc_compensation", True)

        if config["use_gravity_compensation"]:
            log.warning(
                "JointController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        # Delta mode cannot use default command_output_limits
        if config["use_delta_commands"] and config.get("command_output_limits", "default") == "default":
            raise AssertionError(
                "Cannot use 'default' command output limits in delta commands mode of JointController. Try None instead."
            )

        return super()._process_config(controller_id, config)

    @classmethod
    def _generate_default_command_output_limits(cls, controller_id: str):
        config = cls._configs[controller_id]
        motor_type = config["motor_type"]
        return (
            config["control_limits"][ControlType.get_type(motor_type)][0][cls.dof_idx(controller_id)],
            config["control_limits"][ControlType.get_type(motor_type)][1][cls.dof_idx(controller_id)],
        )

    @classmethod
    def _update_goal(cls, controller_id: str, command, control_dict):
        config = cls._configs[controller_id]
        # If we're using delta commands, add this value
        if config["use_delta_commands"]:
            # Compute the base value for the command
            base_value = control_dict[f"joint_{config['motor_type']}"][cls.dof_idx(controller_id)]

            # Apply the command to the base value.
            target = base_value + command

            # Correct any gimbal lock issues using the compute_delta_in_quat_space group.
            for rx_ind, ry_ind, rz_ind in config["compute_delta_in_quat_space"]:
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
            config["control_limits"][ControlType.get_type(config["motor_type"])][0][cls.dof_idx(controller_id)],
            config["control_limits"][ControlType.get_type(config["motor_type"])][1][cls.dof_idx(controller_id)],
        )
        return dict(target=target)

    @classmethod
    def compute_control(cls, controller_id: str, control_dict, goal_dict=None):
        config = cls._configs[controller_id]
        base_value = control_dict[f"joint_{config['motor_type']}"][cls.dof_idx(controller_id)]
        target = cls._goals[controller_id]["target"] if goal_dict is None else goal_dict["target"]

        if config["use_impedances"]:
            if config["motor_type"] == "position":
                position_error = target - base_value
                vel_pos_error = -control_dict["joint_velocity"][cls.dof_idx(controller_id)]
                u = position_error * config["pos_kp"] + vel_pos_error * config["pos_kd"]
            elif config["motor_type"] == "velocity":
                velocity_error = target - base_value
                u = velocity_error * config["vel_kp"]
            else:
                u = target

            u = cb.get_custom_method("compute_joint_torques")(u, control_dict["mass_matrix"], cls.dof_idx(controller_id))

            if config["use_gravity_compensation"]:
                u += control_dict["gravity_force"][cls.dof_idx(controller_id)]
            if config["use_cc_compensation"]:
                u += control_dict["cc_force"][cls.dof_idx(controller_id)]
        else:
            u = target

        return u

    @classmethod
    def step_batch(cls, controller_ids):
        """
        Batched step for JointController instances.
        Non-impedance: u = target (trivial pass-through, processed sequentially).
        Impedance: batched PD control + mass matrix multiply with zero-padded tensors.
        """
        # Fill no-op goals
        for cid in controller_ids:
            if cls._goals[cid] is None:
                cls._goals[cid] = cls.compute_no_op_goal(cid, cls._control_dicts[cid])

        # Split by impedance usage
        impedance_indices = []
        non_impedance_indices = []
        for idx, cid in enumerate(controller_ids):
            if cls._configs[cid]["use_impedances"]:
                impedance_indices.append(idx)
            else:
                non_impedance_indices.append(idx)

        results = [None] * len(controller_ids)

        # Non-impedance: u = target, clip, done
        for idx in non_impedance_indices:
            cid = controller_ids[idx]
            u = cb.copy(cls._goals[cid]["target"])
            u = cls.clip_control(cid, u)
            cls._controls[cid] = u
            results[idx] = u

        # Impedance: batched PD + mass matrix multiply
        if impedance_indices:
            imp_cids = [controller_ids[idx] for idx in impedance_indices]
            N = len(imp_cids)
            dims = [cls.control_dim(cid) for cid in imp_cids]
            max_dim = max(dims)

            # Build zero-padded [N, max_dim] tensors
            targets = cb.zeros((N, max_dim))
            base_values = cb.zeros((N, max_dim))
            velocities = cb.zeros((N, max_dim))
            gain = cb.zeros((N, max_dim))
            damping = cb.zeros((N, max_dim))
            gravity = cb.zeros((N, max_dim))
            cc = cb.zeros((N, max_dim))
            mass_matrices = cb.zeros((N, max_dim, max_dim))
            is_effort = [False] * N

            for i, cid in enumerate(imp_cids):
                config = cls._configs[cid]
                d = dims[i]
                dof_idx = cls.dof_idx(cid)
                cd = cls._control_dicts[cid]

                targets[i, :d] = cls._goals[cid]["target"]
                base_values[i, :d] = cd[f"joint_{config['motor_type']}"][dof_idx]
                velocities[i, :d] = cd["joint_velocity"][dof_idx]

                if config["motor_type"] == "position":
                    gain[i, :d] = config["pos_kp"]
                    damping[i, :d] = config["pos_kd"]
                elif config["motor_type"] == "velocity":
                    gain[i, :d] = config["vel_kp"]
                else:
                    is_effort[i] = True

                # Extract sub mass matrix: full_mm[dof_idx][:, dof_idx] -> (d, d)
                mass_matrices[i, :d, :d] = cd["mass_matrix"][dof_idx][:, dof_idx]

                if config["use_gravity_compensation"]:
                    gravity[i, :d] = cd["gravity_force"][dof_idx]
                if config["use_cc_compensation"]:
                    cc[i, :d] = cd["cc_force"][dof_idx]

            # Vectorized PD: u = (target - base) * gain + (-vel) * damping
            u = (targets - base_values) * gain + (-velocities) * damping
            # For effort controllers, override with just target
            for i in range(N):
                if is_effort[i]:
                    u[i] = targets[i]

            # Batched mass matrix multiply: [N, D, D] @ [N, D, 1] -> [N, D]
            u = (mass_matrices @ u[..., None])[..., 0]

            # Add compensation forces (zeros for controllers that don't use them)
            u = u + gravity + cc

            # Unpad, clip, store
            for i, (cid, idx) in enumerate(zip(imp_cids, impedance_indices)):
                d = dims[i]
                control = u[i, :d]
                control = cls.clip_control(cid, control)
                cls._controls[cid] = control
                results[idx] = control

        return results

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        if config["motor_type"] == "position":
            target = control_dict[f"joint_{config['motor_type']}"][cls.dof_idx(controller_id)]
        else:
            target = cb.zeros(cls.control_dim(controller_id))
        return dict(target=target)

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        config = cls._configs[controller_id]
        if config["motor_type"] == "position":
            if config["use_delta_commands"]:
                return cb.zeros(cls.command_dim(controller_id))
            return control_dict["joint_position"][cls.dof_idx(controller_id)]
        if config["motor_type"] == "velocity":
            if config["use_delta_commands"]:
                return -control_dict["joint_velocity"][cls.dof_idx(controller_id)]
            return cb.zeros(cls.command_dim(controller_id))
        raise ValueError("Cannot compute noop action for effort motor type.")

    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        return dict(target=(cls.control_dim(controller_id),))

    @classmethod
    def is_grasping(cls, controller_id: str):
        return IsGraspingState.UNKNOWN

    @classmethod
    def use_delta_commands(cls, controller_id: str):
        config = cls._configs[controller_id]
        return config["use_delta_commands"]

    @classmethod
    def motor_type(cls, controller_id: str):
        config = cls._configs[controller_id]
        return config["motor_type"]

    @classmethod
    def control_type(cls, controller_id: str):
        config = cls._configs[controller_id]
        return ControlType.EFFORT if config["use_impedances"] else ControlType.get_type(type_str=config["motor_type"])

    @classmethod
    def command_dim(cls, controller_id: str):
        config = cls._configs[controller_id]
        return len(cls.dof_idx(controller_id))


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
