import os

import numpy as np

from omnigibson.controllers import ControlType
from omnigibson.macros import gm
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

log = create_module_logger(module_name=__name__)


class R1(ManipulationRobot, LocomotionRobot):
    """
    R1 Robot
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        uuid=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=False,
        load_config=None,
        fixed_base=False,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to ControllableObject hierarchy
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # Unique to BaseRobot
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        disable_grasp_handling=False,
        # Unique to r1
        rigid_trunk=False,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            control_freq (float): control frequency (in Hz) at which to control the object. If set to be None,
                we will automatically set the control frequency to be at the render frequency by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self._default_joint_pos will be used instead.
                Note that _default_joint_pos are hardcoded & precomputed, and thus should not be modified by the user.
                Set this value instead if you want to initialize the robot with a different rese joint position.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is ["rgb", "proprio"].
                Valid options are "all", or a list containing any subset of omnigibson.sensors.ALL_SENSOR_MODALITIES.
                Note: If @sensor_config explicitly specifies `modalities` for a given sensor class, it will
                    override any values specified from @obs_modalities!
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            sensor_config (None or dict): nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. This will override any default values specified by this class.
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            disable_grasp_handling (bool): If True, will disable all grasp handling for this object. This means that
                sticky and assisted grasp modes will not work unless the connection/release methodsare manually called.
            rigid_trunk (bool) if True, will prevent the trunk from moving during execution.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store args
        self.rigid_trunk = rigid_trunk

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            disable_grasp_handling=disable_grasp_handling,
            **kwargs,
        )

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("R1 does not support discrete actions!")

    def _initialize(self):
        # Run super method first
        super()._initialize()

    def _postprocess_control(self, control, control_type):
        # Run super method first
        u_vec, u_type_vec = super()._postprocess_control(control=control, control_type=control_type)

        # Override trunk value if we're keeping the trunk rigid
        if self.rigid_trunk:
            u_vec[self.trunk_control_idx] = self._default_joint_pos[self.trunk_control_idx]
            u_type_vec[self.trunk_control_idx] = ControlType.POSITION

        # Return control
        return u_vec, u_type_vec

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        joint_positions = ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path)
        joint_velocities = ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path)
        dic["trunk_qpos"] = joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        controllers = ["base"]
        for arm in self.arm_names:
            controllers += [f"arm_{arm}", f"gripper_{arm}"]
        return controllers

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        # We use IK and multi finger gripper controllers as default
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Need to override joint idx being controlled to include trunk in default arm controller configs
        for arm_cfg in cfg[f"arm_{self.default_arm}"].values():
            arm_control_idx = np.concatenate([self.trunk_control_idx, self.arm_control_idx[self.default_arm]])
            arm_cfg["dof_idx"] = arm_control_idx

            # Need to modify the default joint positions also if this is a null joint controller
            if arm_cfg["name"] == "NullJointController":
                arm_cfg["default_command"] = self.reset_joint_pos[arm_control_idx]

            # If using rigid trunk, we also clamp its limits
            if self.rigid_trunk:
                arm_cfg["control_limits"]["position"][0][self.trunk_control_idx] = self.untucked_default_joint_pos[
                    self.trunk_control_idx
                ]
                arm_cfg["control_limits"]["position"][1][self.trunk_control_idx] = self.untucked_default_joint_pos[
                    self.trunk_control_idx
                ]

        return cfg

    @property
    def _default_joint_pos(self):
        return np.zeros(len(self.joints))

    @property
    def wheel_radius(self):
        # TODO
        return 0.0613

    @property
    def wheel_axle_length(self):
        # TODO
        return 0.372

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def assisted_grasp_start_points(self):
        # TODO
        return {
            self.default_arm: [
                GraspingPoint(link_name="r_gripper_finger_link", position=[0.025, -0.012, 0.0]),
                GraspingPoint(link_name="r_gripper_finger_link", position=[-0.025, -0.012, 0.0]),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        # TODO
        return {
            self.default_arm: [
                GraspingPoint(link_name="l_gripper_finger_link", position=[0.025, 0.012, 0.0]),
                GraspingPoint(link_name="l_gripper_finger_link", position=[-0.025, 0.012, 0.0]),
            ]
        }

    @property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joints.
        """
        return np.array([list(self.joints.keys()).index(name) for name in self.trunk_joint_names])

    @property
    def base_joint_names(self):
        # TODO
        return []

    @property
    def trunk_joint_names(self):
        return [f"torso_joint{i}" for i in range(1, 5)]

    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @property
    def arm_link_names(self):
        return {arm: [f"{arm}_arm_link{i}" for i in range(1, 3)] for arm in self.arm_names}

    @property
    def arm_joint_names(self):
        return {arm: [f"{arm}_arm_joint{i}" for i in range(1, 7)] for arm in self.arm_names}

    @property
    def eef_link_names(self):
        return {arm: f"{arm}_arm_link6" for arm in self.arm_names}

    @property
    def finger_link_names(self):
        return {arm: [f"{arm}_gripper_link{i}" for i in range(1, 3)] for arm in self.arm_names}

    @property
    def finger_joint_names(self):
        return {arm: [f"{arm}_gripper_axis{i}" for i in range(1, 3)] for arm in self.arm_names}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/r1/r1.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {arm: os.path.join(gm.ASSET_PATH, f"models/r1/r1_{arm}_descriptor.yaml") for arm in self.arm_names}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/r1/r1.urdf")

    @property
    def arm_workspace_range(self):
        return {arm: [np.deg2rad(-45), np.deg2rad(45)] for arm in self.arm_names}

    @property
    def eef_usd_path(self):
        return {arm: os.path.join(gm.ASSET_PATH, "models/r1/r1_eef.usd") for arm in self.arm_names}
