import math
from collections.abc import Iterable

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as TT
import omnigibson.utils.transform_utils_np as NT
from omnigibson.controllers import ManipulationController
from omnigibson.controllers.joint_controller import JointController
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.backend_utils import add_compute_function
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Create module logger
log = create_module_logger(module_name=__name__)

# Different modes
IK_MODE_COMMAND_DIMS = {
    "absolute_pose": 6,  # 6DOF (x,y,z,ax,ay,az) control of pose, whether both position and orientation is given in absolute coordinates
    "pose_absolute_ori": 6,  # 6DOF (dx,dy,dz,ax,ay,az) control over pose, where the orientation is given in absolute axis-angle coordinates
    "pose_delta_ori": 6,  # 6DOF (dx,dy,dz,dax,day,daz) control over pose
    "position_fixed_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands being kept as fixed initial absolute orientation
    "position_compliant_ori": 3,  # 3DOF (dx,dy,dz) control over position, with orientation commands automatically being sent as 0s (so can drift over time)
}
IK_MODES = set(IK_MODE_COMMAND_DIMS.keys())


class InverseKinematicsController(JointController, ManipulationController):
    """
    Controller class to convert (delta) EEF commands into joint velocities using Inverse Kinematics (IK).

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Run Inverse Kinematics to back out joint velocities for a desired task frame command
        3. Clips the resulting command by the motor (velocity) limits
    """

    def __init__(
        self,
        control_freq,
        reset_joint_pos,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=(
            (-0.2, -0.2, -0.2, -0.5, -0.5, -0.5),
            (0.2, 0.2, 0.2, 0.5, 0.5, 0.5),
        ),
        isaac_kp=None,
        isaac_kd=None,
        pos_kp=None,
        pos_damping_ratio=None,
        vel_kp=None,
        use_impedances=False,
        mode="pose_delta_ori",
        smoothing_filter_size=None,
        workspace_pose_limiter=None,
        condition_on_current_position=True,
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
            dof_idx (Array[int]): specific dof indices controlled by this controller group. Used for inferring
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
            mode (str): mode to use when computing IK. In all cases, position commands are 3DOF delta (dx,dy,dz)
                cartesian values, relative to the robot base frame. Valid options are:
                    - "absolute_pose": 6DOF (dx,dy,dz,ax,ay,az) control over pose,
                        where both the position and the orientation is given in absolute axis-angle coordinates
                    - "pose_absolute_ori": 6DOF (dx,dy,dz,ax,ay,az) control over pose,
                        where the orientation is given in absolute axis-angle coordinates
                    - "pose_delta_ori": 6DOF (dx,dy,dz,dax,day,daz) control over pose
                    - "position_fixed_ori": 3DOF (dx,dy,dz) control over position,
                        with orientation commands being kept as fixed initial absolute orientation
                    - "position_compliant_ori": 3DOF (dx,dy,dz) control over position,
                        with orientation commands automatically being sent as 0s (so can drift over time)
            smoothing_filter_size (None or int): if specified, sets the size of a moving average filter to apply
                on all outputted IK joint positions.
            workspace_pose_limiter (None or function): if specified, callback method that should clip absolute
                target (x,y,z) cartesian position and absolute quaternion orientation (x,y,z,w) to a specific workspace
                range (i.e.: this can be unique to each robot, and implemented by each embodiment).
                Function signature should be:

                    def limiter(target_pos: Array[float], target_quat: Array[float]) --> Tuple[Array[float], Array[float]]

                where target_pos is (x,y,z) cartesian position values, target_quat is (x,y,z,w) quarternion orientation
                values, and the returned tuple is the processed (pos, quat) command.
            condition_on_current_position (bool): if True, will use the current joint position as the initial guess for the IK algorithm.
                Otherwise, will use the reset_joint_pos as the initial guess.
            link_name (str or None): name of the EEF or trunk link.
        """
        # Store arguments
        assert mode in IK_MODES, f"Invalid ik mode specified! Valid options are: {IK_MODES}, got: {mode}"

        # If mode is absolute pose, make sure command input limits / output limits are None
        if mode == "absolute_pose":
            assert command_input_limits is None, "command_input_limits should be None if using absolute_pose mode!"
            assert command_output_limits is None, "command_output_limits should be None if using absolute_pose mode!"

        self.mode = mode
        self.workspace_pose_limiter = workspace_pose_limiter
        self.reset_joint_pos = reset_joint_pos[dof_idx]
        self.condition_on_current_position = condition_on_current_position

        self._link_name = link_name  # eef/trunk link name (same for all members in the group)
        self._fixed_quat_targets = []  # per-member fixed quat target for position_fixed_ori mode

        # If the mode is set as absolute orientation and using default config,
        # change input and output limits accordingly.
        # By default, the input limits are set as 1, so we modify this to have a correct range.
        # The output orientation limits are also set to be values assuming delta commands, so those are updated too
        if self.mode == "pose_absolute_ori":
            if command_input_limits is not None:
                if type(command_input_limits) is str and command_input_limits == "default":
                    command_input_limits = [
                        cb.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi]),
                        cb.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi]),
                    ]
                else:
                    command_input_limits[0][3:] = cb.full((len(command_input_limits[0][3:]),), -math.pi)
                    command_input_limits[1][3:] = cb.full((len(command_input_limits[1][3:]),), math.pi)
            if command_output_limits is not None:
                if not isinstance(command_output_limits, str) and isinstance(command_output_limits, Iterable):
                    command_output_limits = [
                        cb.array(list(command_output_limits[0])),
                        cb.array(list(command_output_limits[1])),
                    ]
                if type(command_output_limits) is str and command_output_limits == "default":
                    command_output_limits = [
                        cb.array([-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi]),
                        cb.array([1.0, 1.0, 1.0, math.pi, math.pi, math.pi]),
                    ]
                else:
                    command_output_limits[0][3:] = cb.full((len(command_output_limits[0][3:]),), -math.pi)
                    command_output_limits[1][3:] = cb.full((len(command_output_limits[1][3:]),), math.pi)
        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            pos_kp=pos_kp,
            pos_damping_ratio=pos_damping_ratio,
            vel_kp=vel_kp,
            motor_type="position",
            smoothing_filter_size=smoothing_filter_size,
            use_delta_commands=False,
            use_impedances=use_impedances,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
            isaac_kp=isaac_kp,
            isaac_kd=isaac_kd,
        )
        # Reuse the limits already cached by the base class; adding a leading dim lets clip() broadcast over N
        self._q_lower = cb.view(self._clip_lo, (1, -1))
        self._q_upper = cb.view(self._clip_hi, (1, -1))

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
        self._fixed_quat_targets[controller_idx] = None

    def _load_state(self, controller_idx, state):
        # Run super first
        super()._load_state(controller_idx=controller_idx, state=state)

        # Restore per-member fixed orientation targets from loaded goals.
        if self.mode == "position_fixed_ori":
            if cb.item_bool(self._goal_set[controller_idx]):
                self._fixed_quat_targets[controller_idx] = cb.T.mat2quat(self._goals["target_ori_mat"][controller_idx])
            else:
                self._fixed_quat_targets[controller_idx] = None

    def _update_goal(self, controller_idx, command):
        """
        Returns:
            dict: ``target_pos`` and ``target_ori_mat`` as compute-backend (``cb``) arrays
        """
        prim_path = self._articulation_root_paths[controller_idx]
        link_name = self._link_name

        # Get current EEF pose relative to robot base
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
                self._fixed_quat_targets[controller_idx] = cb.copy(quat_relative)
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
        return dict(
            target_pos=target_pos,
            target_ori_mat=cb.T.quat2mat(target_quat),
        )

    def compute_control(self, goals):
        """
        Converts the (already preprocessed) batched goals into deployable (non-clipped!) joint control signals
        for all N group members.

        Args:
            goals (Dict[str, Array]): batched goals with shape (N, *shape) per key.
                Must include:
                    target_pos: (N, 3) desired EEF positions
                    target_ori_mat: (N, 3, 3) desired EEF orientation matrices

        Returns:
            Array: (N, control_dim) outputted (non-clipped!) control signal to deploy
        """

        link_name = self._link_name
        rows = self.view_row_indices

        # Batched state reads — convert from Isaac (torch) to compute backend type
        all_q = ControllableObjectViewAPI.get_all_joint_positions(self.routing_path)  # (N_view, n_joint_dof)
        q_all = all_q[rows, :][:, self.dof_idx]  # (N, ctrl_dim)
        jac_all = ControllableObjectViewAPI.get_all_relative_jacobians(
            self.routing_path
        )  # (N_view, n_links, 6, n_dof_total)
        eef_body_idx = ControllableObjectViewAPI.get_link_index(self.routing_path, link_name)
        jac_row = eef_body_idx - 1  # Jacobian excludes root body (index 0)
        # Floating-base robots expose Jacobian columns as [virtual_base(6), joints].
        # dof_idx indexes the joint block, so we need an offset for the Jacobian columns.
        # Compute offset from full tensor shapes before row-slicing.
        jac_col_offset = jac_all.shape[-1] - all_q.shape[-1]
        jac_dof_idx = self.dof_idx + jac_col_offset
        j_eef_all = jac_all[rows][:, jac_row, :, :][:, :, jac_dof_idx]  # (N, 6, ctrl_dim)
        ee_pos_all, ee_quat_all = ControllableObjectViewAPI.get_all_link_relative_position_orientation(
            self.routing_path, link_name
        )  # (N_view, 3), (N_view, 4)
        ee_pos_all = ee_pos_all[rows]
        ee_quat_all = ee_quat_all[rows]
        ee_mat_all = cb.T.quat2mat(ee_quat_all)  # (N, 3, 3)

        target_joint_pos_batch = cb.get_custom_method("compute_ik_qpos_batch")(
            q=q_all,
            j_eef=j_eef_all,
            ee_pos=ee_pos_all,
            ee_mat=ee_mat_all,
            goal_pos=goals["target_pos"],
            goal_ori_mat=goals["target_ori_mat"],
            q_lower_limit=self._q_lower,  # (1, ctrl_dim) broadcasts over N via clip()
            q_upper_limit=self._q_upper,
        )  # (N, ctrl_dim)

        # Delegate to JointController.compute_control for impedance handling
        return super().compute_control(dict(target=target_joint_pos_batch))

    def compute_no_op_goal(self, controller_idx):
        """
        Returns:
            dict: Current relative EEF pose as ``cb`` arrays (``target_pos``, ``target_ori_mat``).
        """
        prim_path = self._articulation_root_paths[controller_idx]
        link_name = self._link_name

        pos_relative, quat_relative = ControllableObjectViewAPI.get_link_relative_position_orientation(
            prim_path, link_name
        )
        return dict(
            target_pos=cb.copy(pos_relative),
            target_ori_mat=cb.T.quat2mat(quat_relative),
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
    def command_dim(self):
        return IK_MODE_COMMAND_DIMS[self.mode]


def _jparse_compute_torch(
    jacobian: th.Tensor,
    gamma: float = 0.1,
    singular_direction_gain: float = 1.0,
) -> th.Tensor:
    """Batched J-PARSE pseudo-inverse (torch implementation).

    Vectorized over the batch dimension for use with the IK controller.
    J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities
    Effectively. Clamps small singular values and adds smooth feedback in
    singular directions for singularity-robust IK control.
    Reference: https://github.com/armlabstanford/jparse
    """
    J = jacobian.double()
    N, m, _ = J.shape

    U, S, Vh = th.linalg.svd(J, full_matrices=False)

    sigma_max = S[:, 0]  # (N,)
    threshold = gamma * sigma_max  # (N,)

    nonsing_mask = S > threshold[:, None]  # (N, k)
    sing_mask = ~nonsing_mask

    # ---- J_safety: clamp singular values below threshold ----
    S_safety = th.clamp(S, min=threshold[:, None])  # (N, k)
    J_safety = U @ (S_safety.unsqueeze(-1) * Vh)  # (N, m, n)

    # ---- J_proj: retain only non-singular directions ----
    n_nonsing = nonsing_mask.sum(dim=1)  # (N,)

    if (n_nonsing > 0).all():
        S_proj = S * nonsing_mask  # (N, k)
        J_proj = U @ (S_proj.unsqueeze(-1) * Vh)
    else:
        J_proj = th.zeros_like(J)
        has_nonsing = n_nonsing > 0
        S_proj = S * nonsing_mask
        J_proj[has_nonsing] = (U[has_nonsing] * S_proj[has_nonsing, None, :]) @ Vh[has_nonsing]

    # ---- Phi_singular: smooth feedback in singular directions ----
    n_sing = sing_mask.sum(dim=1)  # (N,)
    Phi_singular = th.zeros(N, m, m, dtype=J.dtype, device=J.device)

    has_sing = n_sing > 0
    if has_sing.any():
        gains = th.full((m,), singular_direction_gain, dtype=J.dtype, device=J.device)
        Kp = th.diag(gains)  # (m, m)

        S_ratio = S[has_sing] / sigma_max[has_sing, None]  # (N_sing, k)
        phi_vals = S_ratio / gamma  # (N_sing, k)
        phi_vals = phi_vals * sing_mask[has_sing]  # zero out non-singular

        U_sing = U[has_sing]  # (N_sing, m, k)
        Phi_mat = th.diag_embed(phi_vals)  # (N_sing, k, k)
        Phi_singular[has_sing] = U_sing @ Phi_mat @ U_sing.transpose(-2, -1) @ Kp

    # ---- Combine ----
    J_safety_pinv = th.linalg.pinv(J_safety)
    J_proj_pinv = th.linalg.pinv(J_proj)

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv
    J_parse = J_parse + J_safety_pinv @ Phi_singular

    return J_parse.float()


def _jparse_compute_numpy(
    jacobian,
    gamma=0.1,
    singular_direction_gain=1.0,
):
    """Batched J-PARSE pseudo-inverse (numpy implementation).

    Vectorized over the batch dimension for use with the IK controller.
    J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities
    Effectively. Clamps small singular values and adds smooth feedback in
    singular directions for singularity-robust IK control.
    Reference: https://github.com/armlabstanford/jparse
    """
    J = jacobian.astype(np.float64)
    N, m, _ = J.shape

    U, S, Vh = np.linalg.svd(J, full_matrices=False)

    sigma_max = S[:, 0]  # (N,)
    threshold = gamma * sigma_max  # (N,)

    nonsing_mask = S > threshold[:, None]  # (N, k)
    sing_mask = ~nonsing_mask

    # ---- J_safety: clamp singular values below threshold ----
    S_safety = np.clip(S, threshold[:, None], None)  # (N, k)
    J_safety = U @ (S_safety[:, :, None] * Vh)  # (N, m, n)

    # ---- J_proj: retain only non-singular directions ----
    S_proj = S * nonsing_mask  # (N, k)
    J_proj = U @ (S_proj[:, :, None] * Vh)

    # ---- Phi_singular: smooth feedback in singular directions ----
    n_sing = sing_mask.sum(axis=1)  # (N,)
    Phi_singular = np.zeros((N, m, m), dtype=np.float64)

    has_sing = n_sing > 0
    if has_sing.any():
        gains = np.full(m, singular_direction_gain, dtype=np.float64)
        Kp = np.diag(gains)  # (m, m)

        S_ratio = S[has_sing] / sigma_max[has_sing, None]  # (N_sing, k)
        phi_vals = (S_ratio / gamma) * sing_mask[has_sing]  # (N_sing, k)

        U_sing = U[has_sing]  # (N_sing, m, k)
        Phi_mat = np.apply_along_axis(np.diag, -1, phi_vals)  # (N_sing, k, k)
        Phi_singular[has_sing] = U_sing @ Phi_mat @ U_sing.transpose(0, 2, 1) @ Kp

    # ---- Combine ----
    J_safety_pinv = np.linalg.pinv(J_safety)
    J_proj_pinv = np.linalg.pinv(J_proj)

    J_parse = J_safety_pinv @ J_proj @ J_proj_pinv
    J_parse = J_parse + J_safety_pinv @ Phi_singular

    return J_parse.astype(np.float32)


def _compute_ik_qpos_batch_torch(
    q: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    goal_pos: th.Tensor,
    goal_ori_mat: th.Tensor,
    q_lower_limit: th.Tensor,
    q_upper_limit: th.Tensor,
):
    pos_err = goal_pos - ee_pos
    ori_err = TT.orientation_error(goal_ori_mat, ee_mat)
    err = th.cat([pos_err, ori_err], dim=-1)
    j_eef_pinv = _jparse_compute_torch(j_eef)
    delta_j = (j_eef_pinv @ err.unsqueeze(-1)).squeeze(-1)
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(min=q_lower_limit, max=q_upper_limit)


def _compute_ik_qpos_batch_numpy(
    q,
    j_eef,
    ee_pos,
    ee_mat,
    goal_pos,
    goal_ori_mat,
    q_lower_limit,
    q_upper_limit,
):
    pos_err = goal_pos - ee_pos
    ori_err = NT.orientation_error(goal_ori_mat, ee_mat).astype(np.float32)
    err = np.concatenate([pos_err, ori_err], axis=-1)
    j_eef_pinv = _jparse_compute_numpy(j_eef)
    delta_j = (j_eef_pinv @ err[..., None])[..., 0]
    target_joint_pos = q + delta_j
    return target_joint_pos.clip(q_lower_limit, q_upper_limit)


add_compute_function(
    name="compute_ik_qpos_batch",
    np_function=_compute_ik_qpos_batch_numpy,
    th_function=_compute_ik_qpos_batch_torch,
)
