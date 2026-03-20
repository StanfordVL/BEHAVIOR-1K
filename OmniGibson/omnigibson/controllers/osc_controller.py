import math

import numpy as np
import torch as th
from numba import jit

import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers import ControlType, ManipulationController
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Create module logger
log = create_module_logger(module_name=__name__)


@th.jit.script
def _quat_multiply_batch(q1: th.Tensor, q0: th.Tensor) -> th.Tensor:
    """Batched quaternion multiplication q1 * q0. Inputs are (N, 4) in (x,y,z,w) convention."""
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    return th.stack(
        [
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ],
        dim=-1,
    )


# Different modes
OSC_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,  # 6DOF (x,y,z,ax,ay,az) control of pose, whether both position and orientation is given in absolute coordinates
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}
OSC_MODES = set(OSC_MODE_COMMAND_DIMS.keys())


class OperationalSpaceController(ManipulationController):
    """
    Controller class to convert (delta or absolute) EEF commands into joint efforts using Operational Space Control

    This controller expects 6DOF delta commands (dx, dy, dz, dax, day, daz), where the delta orientation
    commands are in axis-angle form, and outputs low-level torque commands.

    Gains may also be considered part of the action space as well. In this case, the action space would be:
        (
            dx, dy, dz, dax, day, daz                       <-- 6DOF delta eef commands
            [, kpx, kpy, kpz, kpax, kpay, kpaz]             <-- kp gains
            [, drx dry, drz, drax, dray, draz]              <-- damping ratio gains
            [, kpnx, kpny, kpnz, kpnax, kpnay, kpnaz]       <-- kp null gains
        )

    Note that in this case, we ASSUME that the inputted gains are normalized to be in the range [-1, 1], and will
    be mapped appropriately to their respective ranges, as defined by XX_limits

    Alternatively, parameters (in this case, kp or damping_ratio) can either be set during initialization or provided
    from an external source; if the latter, the control_dict should include the respective parameter(s) as
    a part of its keys

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run OSC to back out joint efforts for a desired task frame command
        3. Clips the resulting command by the motor (effort) limits
    """

    def __init__(
        self,
        control_freq,
        reset_joint_pos,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5)),
        isaac_kp=None,
        isaac_kd=None,
        kp=150.0,
        kp_limits=(10.0, 300.0),
        damping_ratio=1.0,
        damping_ratio_limits=(0.0, 2.0),
        kp_null=10.0,
        kp_null_limits=(0.0, 50.0),
        mode="pose_delta_ori",
        decouple_pos_ori=False,
        workspace_pose_limiter=None,
        use_gravity_compensation=False,
        use_cc_compensation=True,
        link_name=None,
    ):
        """
        Args:
            control_freq (int): controller loop frequency
            reset_joint_pos (Array[float]): reset joint positions, used as part of nullspace controller in IK.
                Note that this should correspond to ALL the joints; the exact indices will be extracted via @dof_idx
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
            kp (None, int, float, or array): Gain values to apply to 6DOF error.
                If None, will be variable (part of action space)
            kp_limits (2-array): (min, max) values of kp
            damping_ratio (None, int, float, or array): Damping ratio to apply to 6DOF error controller gain
                If None, will be variable (part of action space)
            damping_ratio_limits (2-array): (min, max) values of damping ratio
            kp_null (None, int, float, or array): Gain applied when calculating null torques
                If None, will be variable (part of action space)
            kp_null_limits (2-array): (min, max) values of kp_null
            mode (str): mode to use when computing IK. In all cases, position commands are 3DOF delta (dx,dy,dz)
                cartesian values, relative to the robot base frame. Valid options are:
                    - "pose_absolute_ori": 6DOF (dx,dy,dz,ax,ay,az) control over pose,
                        where the orientation is given in absolute axis-angle coordinates
                    - "pose_delta_ori": 6DOF (dx,dy,dz,dax,day,daz) control over pose
                    - "position_fixed_ori": 3DOF (dx,dy,dz) control over position,
                        with orientation commands being kept as fixed initial absolute orientation
                    - "position_compliant_ori": 3DOF (dx,dy,dz) control over position,
                        with orientation commands automatically being sent as 0s (so can drift over time)
            decouple_pos_ori (bool): Whether to decouple position and orientation control or not
            workspace_pose_limiter (None or function): if specified, callback method that should clip absolute
                target (x,y,z) cartesian position and absolute quaternion orientation (x,y,z,w) to a specific workspace
                range (i.e.: this can be unique to each robot, and implemented by each embodiment).
                Function signature should be:

                    def limiter(target_pos: Array[float], target_quat: Array[float]) --> Tuple[Array[float], Array[float]]

                where target_pos is (x,y,z) cartesian position values, target_quat is (x,y,z,w) quarternion orientation
                values, and the returned tuple is the processed (pos, quat) command.
            use_gravity_compensation (bool): If True, will add gravity compensation to the computed efforts. This is
                an experimental feature that only works on fixed base robots. We do not recommend enabling this.
            use_cc_compensation (bool): If True, will add Coriolis / centrifugal compensation to the computed efforts.
            link_name (str or None): name of eef or trunk link
        """
        # Store arguments
        control_dim = len(dof_idx)
        self._use_gravity_compensation = use_gravity_compensation
        self._use_cc_compensation = use_cc_compensation

        # Warn the user about gravity compensation and Coriolis / centrifugal compensation being experimental.
        if self._use_gravity_compensation:
            log.warning(
                "OperationalSpaceController is using gravity compensation. This is an experimental feature that only works on "
                "fixed base robots. We do not recommend enabling this."
            )

        # Store gains for direct use in the solver
        self.kp = self.nums2array(nums=kp, dim=6) if kp is not None else None
        self.damping_ratio = damping_ratio
        self.kp_null = self.nums2array(nums=kp_null, dim=control_dim) if kp_null is not None else None
        self.kd_null = 2 * cb.sqrt(self.kp_null) if kp_null is not None else None  # critically damped
        self.kp_limits = cb.array(list(kp_limits))
        self.damping_ratio_limits = cb.array(list(damping_ratio_limits))
        self.kp_null_limits = cb.array(list(kp_null_limits))

        # Store settings for whether we're learning gains or not
        self.variable_kp = self.kp is None
        self.variable_damping_ratio = self.damping_ratio is None
        self.variable_kp_null = self.kp_null is None

        # TODO: Add support for variable gains -- for now, just raise an error
        assert True not in {
            self.variable_kp,
            self.variable_damping_ratio,
            self.variable_kp_null,
        }, "Variable gains with OSC is not supported yet!"

        # If the mode is set as absolute orientation and using default config,
        # change input and output limits accordingly.
        # By default, the input limits are set as 1, so we modify this to have a correct range.
        # The output orientation limits are also set to be values assuming delta commands, so those are updated too
        assert_valid_key(key=mode, valid_keys=OSC_MODES, name="OSC mode")

        # If mode is absolute pose, make sure command input limits / output limits are None
        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        self.mode = mode
        if self.mode == "pose_absolute_ori":
            if command_input_limits is not None:
                if type(command_input_limits) is str and command_input_limits == "default":
                    command_input_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_input_limits[0][3:] = -math.pi
                    command_input_limits[1][3:] = math.pi
            if command_output_limits is not None:
                if type(command_output_limits) is str and command_output_limits == "default":
                    command_output_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_output_limits[0][3:] = -math.pi
                    command_output_limits[1][3:] = math.pi

        is_input_limits_numeric = not (command_input_limits is None or isinstance(command_input_limits, str))
        is_output_limits_numeric = not (command_output_limits is None or isinstance(command_output_limits, str))
        command_input_limits = (
            [self.nums2array(lim, dim=6) for lim in command_input_limits]
            if is_input_limits_numeric
            else command_input_limits
        )
        command_output_limits = (
            [self.nums2array(lim, dim=6) for lim in command_output_limits]
            if is_output_limits_numeric
            else command_output_limits
        )

        # Modify input / output scaling based on whether we expect gains to be part of the action space
        self._command_dim = OSC_MODE_COMMAND_DIMS[self.mode]
        for variable_gain, gain_limits, dim in zip(
            (self.variable_kp, self.variable_damping_ratio, self.variable_kp_null),
            (self.kp_limits, self.damping_ratio_limits, self.kp_null_limits),
            (6, 6, control_dim),
        ):
            if variable_gain:
                # Add this to input / output limits
                if is_input_limits_numeric:
                    command_input_limits = [
                        cb.cat([lim, self.nums2array(nums=val, dim=dim)])
                        for lim, val in zip(command_input_limits, (-1, 1))
                    ]
                if is_output_limits_numeric:
                    command_output_limits = [
                        cb.cat([lim, self.nums2array(nums=val, dim=dim)])
                        for lim, val in zip(command_output_limits, gain_limits)
                    ]
                # Update command dim
                self._command_dim += dim

        # Other values
        self.decouple_pos_ori = decouple_pos_ori
        self.workspace_pose_limiter = workspace_pose_limiter
        self.reset_joint_pos = cb.array(reset_joint_pos[dof_idx])

        # member state that will be filled in at runtime
        self._link_name = link_name  # eef/trunk link name (same for all members in the group)
        self._fixed_quat_targets = []  # per-member fixed quat target for position_fixed_ori mode

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
        """
        Register a member and store its EEF link name.

        Reuses a tombstoned slot when available (tombstone reuse is handled by the base class).

        Args:
            articulation_root_path (str): articulation root prim path of the new group member

        Returns:
            int: controller_idx
        """
        idx = super().add_member(articulation_root_path, control_enabled=control_enabled)
        if idx < len(self._fixed_quat_targets):
            # Reusing a tombstoned slot — reset the fixed orientation target
            self._fixed_quat_targets[idx] = None
        else:
            self._fixed_quat_targets.append(None)
        return idx

    def reset(self, controller_idx):
        # Call super first
        super().reset(controller_idx)

        # Clear internal variables
        self._fixed_quat_targets[controller_idx] = None
        self._clear_variable_gains()

    def _load_state(self, controller_idx, state):
        # Run super first
        super()._load_state(controller_idx=controller_idx, state=state)

        # Restore per-member fixed orientation targets from loaded goals.
        if self.mode == "position_fixed_ori":
            if cb.item_bool(self._goal_set[controller_idx]):
                self._fixed_quat_targets[controller_idx] = cb.T.mat2quat(self._goals["target_ori_mat"][controller_idx])

    def _clear_variable_gains(self):
        """
        Helper function to clear any gains that are variable and considered part of actions
        """
        if self.variable_kp:
            self.kp = None
        if self.variable_damping_ratio:
            self.damping_ratio = None
        if self.variable_kp_null:
            self.kp_null = None
            self.kd_null = None

    def _update_variable_gains(self, gains):
        """
        Helper function to update any gains that are variable and considered part of actions

        Args:
            gains (n-array): array where n dim is parsed based on which gains are being learned
        """
        idx = 0
        if self.variable_kp:
            self.kp = gains[:, idx : idx + 6]
            idx += 6
        if self.variable_damping_ratio:
            self.damping_ratio = gains[:, idx : idx + 6]
            idx += 6
        if self.variable_kp_null:
            self.kp_null = gains[:, idx : idx + self.control_dim]
            self.kd_null = 2 * cb.sqrt(self.kp_null)  # critically damped
            idx += self.control_dim

    def _update_goal(self, controller_idx, command):
        """
        Updates the internal goal (ee pos and ee ori mat) based on the inputted delta command

        Args:
            command (n-array): Preprocessed command
            controller_idx (int): idx of the controller that need to update goal

        Returns:
            dict: ``target_pos`` and ``target_ori_mat`` as compute-backend (``cb``) arrays
        """
        prim_path = self._articulation_root_paths[controller_idx]
        link_name = self._link_name

        # Get current EEF pose
        pos_relative, quat_relative = ControllableObjectViewAPI.get_link_relative_position_orientation(
            prim_path, link_name
        )

        # Convert position command to absolute values if needed
        if self.mode == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        # Compute orientation
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self._fixed_quat_targets[controller_idx] is None:
                self._fixed_quat_targets[controller_idx] = (
                    cb.copy(quat_relative)
                    if not cb.item_bool(self._goal_set[controller_idx])
                    else cb.T.mat2quat(self._goals["target_ori_mat"][controller_idx])
                )
            target_quat = self._fixed_quat_targets[controller_idx]
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif self.mode == "pose_absolute_ori" or self.mode == "absolute_pose":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = cb.T.axisangle2quat(command[3:6])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6]))
            target_quat = cb.T.mat2quat(dori @ cb.T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat)

        gains = None  # TODO! command[OSC_MODE_COMMAND_DIMS[self.mode]:]
        if gains is not None:
            self._update_variable_gains(gains=gains)

        # Set goals and return
        return dict(
            target_pos=target_pos,
            target_ori_mat=cb.T.quat2mat(target_quat),
        )

    def compute_control(self, goals):
        """
        Computes low-level torque controls for all N group members using internal EEF goal pos/ori.

        Args:
            goals (Dict[str, Array]): batched goals with shape (N, *shape) per key.
                Must include:
                    target_pos: (N, 3) desired EEF positions
                    target_ori_mat: (N, 3, 3) desired EEF orientation matrices

        Returns:
            Array: (N, control_dim) low-level effort control actions
        """
        link_name = self._link_name
        rows = self.view_row_indices

        kp = self.kp  # (6,)
        kd = 2 * cb.sqrt(kp) * self.damping_ratio  # (6,)

        # Batched joint state reads — convert from Isaac (torch) to compute backend type
        all_q = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)  # (N_view, n_joint_dof)
        q_all = all_q[rows, :][:, self.dof_idx]  # (N, ctrl_dim)
        qd_all = ControllableObjectViewAPI.get_all_joint_velocities(self.routing_path, estimate=True)[rows, :][
            :, self.dof_idx
        ]  # (N, ctrl_dim)

        # Batched mass matrix: slice to (N, ctrl_dim, ctrl_dim)
        all_mm_full = ControllableObjectViewAPI.get_all_generalized_mass_matrices(
            self.routing_path
        )  # (N_view, n_dof_total, n_dof_total)
        mm_col_offset = all_mm_full.shape[-1] - all_q.shape[-1]
        mm_dof_idx = self.dof_idx + mm_col_offset
        mm_dof_idx_arr = cb.int_array(mm_dof_idx)
        dof_idxs_mat = cb.meshgrid(mm_dof_idx_arr, mm_dof_idx_arr)
        mm_all = all_mm_full[rows, :, :][:, dof_idxs_mat[0], dof_idxs_mat[1]]  # (N, ctrl_dim, ctrl_dim)

        # Batched jacobians
        jac_all = ControllableObjectViewAPI.get_all_relative_jacobians(
            self.routing_path
        )  # (N_view, n_links, 6, n_dof_total)
        eef_body_idx = ControllableObjectViewAPI.get_link_index(self.routing_path, link_name)
        jac_row = eef_body_idx - 1  # Jacobian excludes root body (index 0)
        jac_col_offset = jac_all.shape[-1] - all_q.shape[-1]
        jac_dof_idx = self.dof_idx + jac_col_offset
        j_eef_all = jac_all[rows][:, jac_row, :, :][:, :, jac_dof_idx]  # (N, 6, ctrl_dim)

        # Batched EEF pose and velocities
        ee_pos_all, ee_quat_all = ControllableObjectViewAPI.get_all_link_relative_position_orientation(
            self.routing_path, link_name
        )  # (N_view, 3), (N_view, 4)
        ee_pos_all = ee_pos_all[rows]
        ee_quat_all = ee_quat_all[rows]
        ee_mat_all = cb.T.quat2mat(ee_quat_all)  # (N, 3, 3)
        ee_lin_vel_all = ControllableObjectViewAPI.get_all_link_relative_linear_velocity(
            self.routing_path, link_name, estimate=True
        )[rows]  # (N, 3)
        ee_ang_vel_all = ControllableObjectViewAPI.get_all_link_relative_angular_velocity(
            self.routing_path, link_name, estimate=True
        )[rows]  # (N, 3)
        base_lin_vel_all = ControllableObjectViewAPI.get_all_relative_linear_velocity(self.routing_path, estimate=True)[
            rows
        ]  # (N, 3)
        base_ang_vel_all = ControllableObjectViewAPI.get_all_relative_angular_velocity(
            self.routing_path, estimate=True
        )[rows]  # (N, 3)

        # Batched angular velocity error
        ee_ang_vel_err_all = cb.T.quat2axisangle(
            cb.T.quat_multiply(
                cb.T.axisangle2quat(-ee_ang_vel_all),
                cb.T.axisangle2quat(base_ang_vel_all),
            )
        )  # (N, 3)

        # Add leading dim (1, *) — broadcasts over N in the batch solver
        kp_batch = cb.view(kp, (1, -1))
        kd_batch = cb.view(kd, (1, -1))
        kp_null_batch = cb.view(self.kp_null, (1, -1))
        kd_null_batch = cb.view(self.kd_null, (1, -1))
        rest_qpos_batch = cb.view(self.reset_joint_pos, (1, -1))

        u = cb.get_custom_method("compute_osc_torques_batch")(
            q=q_all,
            qd=qd_all,
            mm=mm_all,
            j_eef=j_eef_all,
            ee_pos=ee_pos_all,
            ee_mat=ee_mat_all,
            ee_lin_vel=ee_lin_vel_all,
            ee_ang_vel_err=ee_ang_vel_err_all,
            goal_pos=goals["target_pos"],
            goal_ori_mat=goals["target_ori_mat"],
            kp=kp_batch,
            kd=kd_batch,
            kp_null=kp_null_batch,
            kd_null=kd_null_batch,
            rest_qpos=rest_qpos_batch,
            control_dim=self.control_dim,
            decouple_pos_ori=self.decouple_pos_ori,
            base_lin_vel=base_lin_vel_all,
            base_ang_vel=base_ang_vel_all,
        )  # (N, ctrl_dim)

        if self._use_gravity_compensation:
            all_gravity = ControllableObjectViewAPI.get_all_gravity_compensation_forces(self.routing_path)
            gravity_col_offset = all_gravity.shape[-1] - all_q.shape[-1]
            gravity_dof_idx = self.dof_idx + gravity_col_offset
            u = u + all_gravity[rows][:, gravity_dof_idx]

        if self._use_cc_compensation:
            all_cc = ControllableObjectViewAPI.get_all_coriolis_and_centrifugal_compensation_forces(self.routing_path)
            cc_col_offset = all_cc.shape[-1] - all_q.shape[-1]
            cc_dof_idx = self.dof_idx + cc_col_offset
            u = u + all_cc[rows][:, cc_dof_idx]

        return u

    def compute_no_op_goal(self, controller_idx):
        """
        Returns:
            dict: Current EEF pose as ``cb`` arrays (``target_pos``, ``target_ori_mat``).
        """
        # No-op is maintaining current pose
        prim_path = self._articulation_root_paths[controller_idx]
        link_name = self._link_name

        target_pos, target_quat = ControllableObjectViewAPI.get_link_relative_position_orientation(prim_path, link_name)

        # Convert quat into eef ori mat
        return dict(
            target_pos=cb.copy(target_pos),
            target_ori_mat=cb.T.quat2mat(target_quat),
        )

    def _compute_no_op_command(self, controller_idx):
        prim_path = self._articulation_root_paths[controller_idx]
        link_name = self._link_name

        pos_relative, quat_relative = ControllableObjectViewAPI.get_link_relative_position_orientation(
            prim_path, link_name
        )

        command = cb.zeros(6)

        # Handle position
        if self.mode == "absolute_pose":
            command[:3] = pos_relative
        else:
            # We can leave it as zero for delta mode.
            pass

        # Handle orientation
        if self.mode in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = cb.T.quat2axisangle(quat_relative)
        else:
            # For these modes, we don't need to add orientation to the command
            pass

        return command

    def _get_goal_shapes(self):
        return dict(
            target_pos=(3,),
            target_ori_mat=(3, 3),
        )

    @property
    def control_type(self):
        return ControlType.EFFORT

    @property
    def command_dim(self):
        return self._command_dim


@th.jit.script
def _compute_osc_torques_torch(
    q: th.Tensor,
    qd: th.Tensor,
    mm: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    ee_lin_vel: th.Tensor,
    ee_ang_vel_err: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    kp: th.Tensor,
    kd: th.Tensor,
    kp_null: th.Tensor,
    kd_null: th.Tensor,
    rest_qpos: th.Tensor,
    control_dim: int,
    decouple_pos_ori: bool,
    base_lin_vel: th.Tensor,
    base_ang_vel: th.Tensor,
):
    # Compute the inverse
    mm_inv = th.linalg.inv(mm)

    # Calculate error
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat((pos_err, ori_err))

    # Vel target is the base velocity as experienced by the end effector
    # For angular velocity, this is just the base angular velocity
    # For linear velocity, this is the base linear velocity PLUS the net linear velocity experienced
    #   due to the base linear velocity
    # For angular velocity, we need to make sure we compute the difference between the base and eef velocity
    # properly, not simply "subtraction" as in the linear case
    lin_vel_err = base_lin_vel + th.linalg.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = th.cat((lin_vel_err, ee_ang_vel_err))

    # Determine desired wrench
    err = th.unsqueeze(kp * err + kd * vel_err, dim=-1)
    m_eef_inv = j_eef @ mm_inv @ j_eef.T
    m_eef = th.linalg.inv(m_eef_inv)

    if decouple_pos_ori:
        # # More efficient, but numba doesn't support 3D tensor operations yet
        # j_eef_batch = j_eef.reshape(2, 3, -1)
        # m_eef_pose_inv = j_eef_batch @ th.unsqueeze(mm_inv, dim=0) @ th.transpose(j_eef_batch, 0, 2, 1)
        # m_eef_pose = th.linalg.inv_ex(m_eef_pose_inv).inverse  # Shape (2, 3, 3)
        # wrench = (m_eef_pose @ err.reshape(2, 3, 1)).flatten()
        m_eef_pos_inv = j_eef[:3, :] @ mm_inv @ j_eef[:3, :].T
        m_eef_ori_inv = j_eef[3:, :] @ mm_inv @ j_eef[3:, :].T
        m_eef_pos = th.linalg.inv(m_eef_pos_inv)
        m_eef_ori = th.linalg.inv(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:3, :]
        wrench_ori = m_eef_ori @ err[3:, :]
        wrench = th.cat((wrench_pos, wrench_ori))
    else:
        wrench = m_eef @ err

    # Compute OSC torques
    u = j_eef.T @ wrench

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * wrap_angle(rest_qpos - q)
        u_null = mm @ th.unsqueeze(u_null, dim=-1)
        u += (th.eye(control_dim, dtype=th.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u


# Use numba since faster
@jit(nopython=True)
def _compute_osc_torques_numpy(
    q,
    qd,
    mm,
    j_eef,
    ee_pos,
    ee_mat,
    ee_lin_vel,
    ee_ang_vel_err,
    goal_pos,
    goal_ori_mat,
    kp,
    kd,
    kp_null,
    kd_null,
    rest_qpos,
    control_dim,
    decouple_pos_ori,
    base_lin_vel,
    base_ang_vel,
):
    # Compute the inverse
    mm_inv = np.linalg.inv(mm)

    # Calculate error
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate((pos_err, ori_err))

    # Vel target is the base velocity as experienced by the end effector
    # For angular velocity, this is just the base angular velocity
    # For linear velocity, this is the base linear velocity PLUS the net linear velocity experienced
    #   due to the base linear velocity
    # For angular velocity, we need to make sure we compute the difference between the base and eef velocity
    # properly, not simply "subtraction" as in the linear case
    lin_vel_err = base_lin_vel + np.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = np.concatenate((lin_vel_err, ee_ang_vel_err))

    # Determine desired wrench
    err = np.expand_dims(kp * err + kd * vel_err, axis=-1)
    m_eef_inv = j_eef @ mm_inv @ j_eef.T
    m_eef = np.linalg.inv(m_eef_inv)

    if decouple_pos_ori:
        # # More efficient, but numba doesn't support 3D tensor operations yet
        # j_eef_batch = j_eef.reshape(2, 3, -1)
        # m_eef_pose_inv = np.matmul(np.matmul(j_eef_batch, np.expand_dims(mm_inv, axis=0)), np.transpose(j_eef_batch, (0, 2, 1)))
        # m_eef_pose = np.linalg.inv(m_eef_pose_inv)  # Shape (2, 3, 3)
        # wrench = np.matmul(m_eef_pose, err.reshape(2, 3, 1)).flatten()
        m_eef_pos_inv = j_eef[:3, :] @ mm_inv @ j_eef[:3, :].T
        m_eef_ori_inv = j_eef[3:, :] @ mm_inv @ j_eef[3:, :].T
        m_eef_pos = np.linalg.inv(m_eef_pos_inv)
        m_eef_ori = np.linalg.inv(m_eef_ori_inv)
        wrench_pos = m_eef_pos @ err[:3, :]
        wrench_ori = m_eef_ori @ err[3:, :]
        wrench = np.concatenate((wrench_pos, wrench_ori))
    else:
        wrench = m_eef @ err

    # Compute OSC torques
    u = j_eef.T @ wrench

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    if rest_qpos is not None:
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = kd_null * -qd + kp_null * ((rest_qpos - q + np.pi) % (2 * np.pi) - np.pi)
        u_null = mm @ np.expand_dims(u_null, axis=-1).astype(np.float32)
        u += (np.eye(control_dim, dtype=np.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u


# Set these as part of the backend values
add_compute_function(
    name="compute_osc_torques", np_function=_compute_osc_torques_numpy, th_function=_compute_osc_torques_torch
)


def _compute_osc_torques_batch_torch(
    q: th.Tensor,
    qd: th.Tensor,
    mm: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    ee_lin_vel: th.Tensor,
    ee_ang_vel_err: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    kp: th.Tensor,
    kd: th.Tensor,
    kp_null: th.Tensor,
    kd_null: th.Tensor,
    rest_qpos: th.Tensor,
    control_dim: int,
    decouple_pos_ori: bool,
    base_lin_vel: th.Tensor,
    base_ang_vel: th.Tensor,
):
    mm_inv = th.linalg.inv(mm)
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    lin_vel_err = base_lin_vel + th.linalg.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = th.cat([lin_vel_err, ee_ang_vel_err], dim=-1)
    task_err = th.cat([pos_err, ori_err], dim=-1)
    err = (kp * task_err + kd * vel_err).unsqueeze(-1)
    j_eef_T = j_eef.transpose(-2, -1)

    if decouple_pos_ori:
        j_pos = j_eef[:, :3, :]
        j_ori = j_eef[:, 3:, :]
        m_eef_pos = th.linalg.inv(j_pos @ mm_inv @ j_pos.transpose(-2, -1))
        m_eef_ori = th.linalg.inv(j_ori @ mm_inv @ j_ori.transpose(-2, -1))
        wrench = th.cat([m_eef_pos @ err[:, :3, :], m_eef_ori @ err[:, 3:, :]], dim=1)
        m_eef = th.linalg.inv(j_eef @ mm_inv @ j_eef_T)
    else:
        m_eef = th.linalg.inv(j_eef @ mm_inv @ j_eef_T)
        wrench = m_eef @ err

    u = j_eef_T @ wrench

    j_eef_inv = m_eef @ j_eef @ mm_inv
    angle_diff = (rest_qpos - q + math.pi) % (2 * math.pi) - math.pi
    u_null = (kd_null * (-qd) + kp_null * angle_diff).unsqueeze(-1)
    u_null = mm @ u_null
    eye = th.eye(control_dim, dtype=th.float32).unsqueeze(0)
    nullspace_proj = eye - j_eef_T @ j_eef_inv
    u = u + nullspace_proj @ u_null

    return u.squeeze(-1)


def _compute_osc_torques_batch_numpy(
    q,
    qd,
    mm,
    j_eef,
    ee_pos,
    ee_mat,
    ee_lin_vel,
    ee_ang_vel_err,
    goal_pos,
    goal_ori_mat,
    kp,
    kd,
    kp_null,
    kd_null,
    rest_qpos,
    control_dim,
    decouple_pos_ori,
    base_lin_vel,
    base_ang_vel,
):
    mm_inv = np.linalg.inv(mm)
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    lin_vel_err = base_lin_vel + np.cross(base_ang_vel, ee_pos) - ee_lin_vel
    vel_err = np.concatenate([lin_vel_err, ee_ang_vel_err], axis=-1)
    task_err = np.concatenate([pos_err, ori_err], axis=-1)
    err = np.expand_dims(kp * task_err + kd * vel_err, axis=-1)
    j_eef_T = np.swapaxes(j_eef, -2, -1)

    if decouple_pos_ori:
        j_pos = j_eef[:, :3, :]
        j_ori = j_eef[:, 3:, :]
        m_eef_pos = np.linalg.inv(j_pos @ mm_inv @ np.swapaxes(j_pos, -2, -1))
        m_eef_ori = np.linalg.inv(j_ori @ mm_inv @ np.swapaxes(j_ori, -2, -1))
        wrench = np.concatenate([m_eef_pos @ err[:, :3, :], m_eef_ori @ err[:, 3:, :]], axis=1)
        m_eef = np.linalg.inv(j_eef @ mm_inv @ j_eef_T)
    else:
        m_eef = np.linalg.inv(j_eef @ mm_inv @ j_eef_T)
        wrench = m_eef @ err

    u = j_eef_T @ wrench

    j_eef_inv = m_eef @ j_eef @ mm_inv
    angle_diff = (rest_qpos - q + math.pi) % (2 * math.pi) - math.pi
    u_null = np.expand_dims(kd_null * (-qd) + kp_null * angle_diff, axis=-1)
    u_null = mm @ u_null
    eye = np.eye(control_dim, dtype=np.float32)[None]
    nullspace_proj = eye - j_eef_T @ j_eef_inv
    u = u + nullspace_proj @ u_null

    return u[..., 0]


add_compute_function(
    name="compute_osc_torques_batch",
    np_function=_compute_osc_torques_batch_numpy,
    th_function=_compute_osc_torques_batch_torch,
)
