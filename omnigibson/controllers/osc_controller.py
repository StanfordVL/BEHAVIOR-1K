import math

import torch as th

import omnigibson.utils.transform_utils as T
from omnigibson.controllers import ControlType, ManipulationController
from omnigibson.utils.control_utils import orientation_error
from omnigibson.utils.processing_utils import MovingAverageFilter
from omnigibson.utils.python_utils import assert_valid_key, nums2array
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

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
        task_name,
        control_freq,
        reset_joint_pos,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits=((-0.2, -0.2, -0.2, -0.5, -0.5, -0.5), (0.2, 0.2, 0.2, 0.5, 0.5, 0.5)),
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
    ):
        """
        Args:
            task_name (str): name assigned to this task frame for computing OSC control. During control calculations,
                the inputted control_dict should include entries named <@task_name>_pos_relative and
                <@task_name>_quat_relative. See self._command_to_control() for what these values should entail.
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

                    def limiter(target_pos: Array[float], target_quat: Array[float], control_dict: Dict[str, Any]) --> Tuple[Array[float], Array[float]]

                where target_pos is (x,y,z) cartesian position values, target_quat is (x,y,z,w) quarternion orientation
                values, and the returned tuple is the processed (pos, quat) command.
            use_gravity_compensation (bool): If True, will add gravity compensation to the computed efforts. This is
                an experimental feature that only works on fixed base robots. We do not recommend enabling this.
            use_cc_compensation (bool): If True, will add Coriolis / centrifugal compensation to the computed efforts.
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

        # Store gains
        self.kp = nums2array(nums=kp, dim=6, dtype=th.float32) if kp is not None else None
        self.damping_ratio = damping_ratio
        self.kp_null = nums2array(nums=kp_null, dim=control_dim, dtype=th.float32) if kp_null is not None else None
        self.kd_null = 2 * th.sqrt(self.kp_null) if kp_null is not None else None  # critically damped
        self.kp_limits = th.tensor(kp_limits, dtype=th.float32)
        self.damping_ratio_limits = th.tensor(damping_ratio_limits, dtype=th.float32)
        self.kp_null_limits = th.tensor(kp_null_limits, dtype=th.float32)

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
                if type(command_input_limits) == str and command_input_limits == "default":
                    command_input_limits = [
                        [-1.0, -1.0, -1.0, -math.pi, -math.pi, -math.pi],
                        [1.0, 1.0, 1.0, math.pi, math.pi, math.pi],
                    ]
                else:
                    command_input_limits[0][3:] = -math.pi
                    command_input_limits[1][3:] = math.pi
            if command_output_limits is not None:
                if type(command_output_limits) == str and command_output_limits == "default":
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
            [nums2array(lim, dim=6, dtype=th.float32) for lim in command_input_limits]
            if is_input_limits_numeric
            else command_input_limits
        )
        command_output_limits = (
            [nums2array(lim, dim=6, dtype=th.float32) for lim in command_output_limits]
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
                        th.cat([lim, nums2array(nums=val, dim=dim, dtype=th.float32)])
                        for lim, val in zip(command_input_limits, (-1, 1))
                    ]
                if is_output_limits_numeric:
                    command_output_limits = [
                        th.cat([lim, nums2array(nums=val, dim=dim, dtype=th.float32)])
                        for lim, val in zip(command_output_limits, gain_limits)
                    ]
                # Update command dim
                self._command_dim += dim

        # Other values
        self.decouple_pos_ori = decouple_pos_ori
        self.workspace_pose_limiter = workspace_pose_limiter
        self.task_name = task_name
        self.reset_joint_pos = reset_joint_pos[dof_idx]

        # Other variables that will be filled in at runtime
        self._fixed_quat_target = None

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self):
        # Call super first
        super().reset()

        # Clear internal variables
        self._fixed_quat_target = None
        self._clear_variable_gains()

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # If self._goal is populated, then set fixed_quat_target as well if the mode uses it
        if self.mode == "position_fixed_ori" and self._goal is not None:
            self._fixed_quat_target = self._goal["target_quat"]

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
            self.kd_null = 2 * th.sqrt(self.kp_null)  # critically damped
            idx += self.control_dim

    def _update_goal(self, command, control_dict):
        """
        Updates the internal goal (ee pos and ee ori mat) based on the inputted delta command

        Args:
            command (n-array): Preprocessed command
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    <@self.task_name>_pos_relative: (x,y,z) relative cartesian position of the desired task frame to
                        control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_quat_relative: (x,y,z,w) relative quaternion orientation of the desired task
                        frame to control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_lin_vel_relative: (x,y,z) relative linear velocity of the desired task frame to
                        control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_ang_vel_relative: (ax, ay, az) relative angular velocity of the desired task
                        frame to control, computed in its local frame (e.g.: robot base frame)
        """
        # Grab important info from control dict
        pos_relative = control_dict[f"{self.task_name}_pos_relative"].clone()
        quat_relative = control_dict[f"{self.task_name}_quat_relative"].clone()

        # Convert position command to absolute values if needed
        if self.mode == "absolute_pose":
            target_pos = command[:3]
        else:
            dpos = command[:3]
            target_pos = pos_relative + dpos

        # Compute orientation
        if self.mode == "position_fixed_ori":
            # We need to grab the current robot orientation as the commanded orientation if there is none saved
            if self._fixed_quat_target is None:
                self._fixed_quat_target = quat_relative if (self._goal is None) else self._goal["target_quat"]
            target_quat = self._fixed_quat_target
        elif self.mode == "position_compliant_ori":
            # Target quat is simply the current robot orientation
            target_quat = quat_relative
        elif self.mode == "pose_absolute_ori" or self.mode == "absolute_pose":
            # Received "delta" ori is in fact the desired absolute orientation
            target_quat = T.axisangle2quat(command[3:6])
        else:  # pose_delta_ori control
            # Grab dori and compute target ori
            dori = T.quat2mat(T.axisangle2quat(command[3:6]))
            target_quat = T.mat2quat(dori @ T.quat2mat(quat_relative))

        # Possibly limit to workspace if specified
        if self.workspace_pose_limiter is not None:
            target_pos, target_quat = self.workspace_pose_limiter(target_pos, target_quat, control_dict)

        gains = None  # TODO! command[OSC_MODE_COMMAND_DIMS[self.mode]:]
        if gains is not None:
            self._update_variable_gains(gains=gains)

        # Set goals and return
        return dict(
            target_pos=target_pos,
            target_ori_mat=T.quat2mat(target_quat),
        )

    def compute_control(self, goal_dict, control_dict):
        """
        Computes low-level torque controls using internal eef goal pos / ori.

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    target_pos: robot-frame (x,y,z) desired end effector position
                    target_quat: robot-frame (x,y,z,w) desired end effector quaternion orientation
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:
                    joint_position: Array of current joint positions
                    joint_velocity: Array of current joint velocities
                    mass_matrix: (N_dof, N_dof) Current mass matrix
                    <@self.task_name>_jacobian_relative: (6, N_dof) Current jacobian matrix for desired task frame
                    <@self.task_name>_pos_relative: (x,y,z) relative cartesian position of the desired task frame to
                        control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_quat_relative: (x,y,z,w) relative quaternion orientation of the desired task
                        frame to control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_lin_vel_relative: (x,y,z) relative linear velocity of the desired task frame to
                        control, computed in its local frame (e.g.: robot base frame)
                    <@self.task_name>_ang_vel_relative: (ax, ay, az) relative angular velocity of the desired task
                        frame to control, computed in its local frame (e.g.: robot base frame)

            control_dict (dict): Dictionary of state tensors including relevant info for controller computation

        Returns:
            n-array: low-level effort control actions, NOT post-processed
        """
        # TODO: Update to possibly grab parameters from dict
        # For now, always use internal values
        kp = self.kp
        damping_ratio = self.damping_ratio
        kd = 2 * th.sqrt(kp) * damping_ratio

        # Extract relevant values from the control dict
        dof_idxs_mat = tuple(th.meshgrid(self.dof_idx, self.dof_idx))
        q = control_dict["joint_position"][self.dof_idx]
        qd = control_dict["joint_velocity"][self.dof_idx]
        mm = control_dict["mass_matrix"][dof_idxs_mat]
        j_eef = control_dict[f"{self.task_name}_jacobian_relative"][:, self.dof_idx]
        ee_pos = control_dict[f"{self.task_name}_pos_relative"]
        ee_quat = control_dict[f"{self.task_name}_quat_relative"]
        ee_vel = th.cat(
            [control_dict[f"{self.task_name}_lin_vel_relative"], control_dict[f"{self.task_name}_ang_vel_relative"]]
        )
        base_lin_vel = control_dict["root_rel_lin_vel"]
        base_ang_vel = control_dict["root_rel_ang_vel"]

        # Calculate torques
        u = _compute_osc_torques(
            q=q,
            qd=qd,
            mm=mm,
            j_eef=j_eef,
            ee_pos=ee_pos,
            ee_mat=T.quat2mat(ee_quat),
            ee_vel=ee_vel,
            goal_pos=goal_dict["target_pos"],
            goal_ori_mat=goal_dict["target_ori_mat"],
            kp=kp,
            kd=kd,
            kp_null=self.kp_null,
            kd_null=self.kd_null,
            rest_qpos=self.reset_joint_pos,
            control_dim=self.control_dim,
            decouple_pos_ori=self.decouple_pos_ori,
            base_lin_vel=base_lin_vel,
            base_ang_vel=base_ang_vel,
        ).flatten()

        # Add gravity compensation
        if self._use_gravity_compensation:
            u += control_dict["gravity_force"][self.dof_idx]

        # Add Coriolis / centrifugal compensation
        if self._use_cc_compensation:
            u += control_dict["cc_force"][self.dof_idx]

        # Return the control torques
        return u

    def compute_no_op_goal(self, control_dict):
        # No-op is maintaining current pose
        target_pos = control_dict[f"{self.task_name}_pos_relative"].clone()
        target_quat = control_dict[f"{self.task_name}_quat_relative"].clone()

        # Convert quat into eef ori mat
        return dict(
            target_pos=target_pos,
            target_ori_mat=T.quat2mat(target_quat),
        )

    def _compute_no_op_action(self, control_dict):
        pos_relative = control_dict[f"{self.task_name}_pos_relative"]
        quat_relative = control_dict[f"{self.task_name}_quat_relative"]

        command = th.zeros(6, dtype=th.float32, device=pos_relative.device)

        # Handle position
        if self.mode == "absolute_pose":
            command[:3] = pos_relative
        else:
            # We can leave it as zero for delta mode.
            pass

        # Handle orientation
        if self.mode in ("pose_absolute_ori", "absolute_pose"):
            command[3:] = T.quat2axisangle(quat_relative)
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
def _compute_osc_torques(
    q: th.Tensor,
    qd: th.Tensor,
    mm: th.Tensor,
    j_eef: th.Tensor,
    ee_pos: th.Tensor,
    ee_mat: th.Tensor,
    ee_vel: th.Tensor,
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
    ori_err = orientation_error(goal_ori_mat, ee_mat)
    err = th.cat((pos_err, ori_err))

    # Vel target is the base velocity as experienced by the end effector
    # For angular velocity, this is just the base angular velocity
    # For linear velocity, this is the base linear velocity PLUS the net linear velocity experienced
    #   due to the base linear velocity
    # For angular velocity, we need to make sure we compute the difference between the base and eef velocity
    # properly, not simply "subtraction" as in the linear case
    lin_vel_err = base_lin_vel + th.linalg.cross(base_ang_vel, ee_pos) - ee_vel[:3]
    ang_vel_err = T.quat2axisangle(T.quat_multiply(T.axisangle2quat(-ee_vel[3:]), T.axisangle2quat(base_ang_vel)))
    vel_err = th.cat((lin_vel_err, ang_vel_err))

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
        u_null = kd_null * -qd + kp_null * ((rest_qpos - q + math.pi) % (2 * math.pi) - math.pi)
        u_null = mm @ th.unsqueeze(u_null, dim=-1)
        u += (th.eye(control_dim, dtype=th.float32) - j_eef.T @ j_eef_inv) @ u_null

    return u
