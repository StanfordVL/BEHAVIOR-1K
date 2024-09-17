import os

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class R1(ManipulationRobot, HolonomicBaseRobot):
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
        default_reset_mode="untuck",
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
            default_reset_mode (str): Default reset mode for the robot. Should be one of: {"tuck", "untuck"}
                If reset_joint_pos is not None, this will be ignored (since _default_joint_pos won't be used during initialization).
            rigid_trunk (bool) if True, will prevent the trunk from moving during execution.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store args
        self.rigid_trunk = rigid_trunk
        assert_valid_key(key=default_reset_mode, valid_keys=RESET_JOINT_OPTIONS, name="default_reset_mode")
        self.default_reset_mode = default_reset_mode

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=True,
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

    # Name of the actual root link that we are interested in. Note that this is different from self.root_link_name,
    # which is "base_footprint_x", corresponding to the first of the 6 1DoF joints to control the base.
    @property
    def base_footprint_link_name(self):
        return "base_link"

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("R1 does not support discrete actions!")

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
        # We use joint controllers for base as default
        controllers["base"] = "JointController"
        # We use IK and multi finger gripper controllers as default
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        for arm in self.arm_names:
            for arm_cfg in cfg["arm_{}".format(arm)].values():

                if arm == "left":
                    # Need to override joint idx being controlled to include trunk in default arm controller configs
                    arm_control_idx = th.cat([self.trunk_control_idx, self.arm_control_idx[arm]])
                    arm_cfg["dof_idx"] = arm_control_idx

                    # Need to modify the default joint positions also if this is a null joint controller
                    if arm_cfg["name"] == "NullJointController":
                        arm_cfg["default_command"] = self.reset_joint_pos[arm_control_idx]

                # If using rigid trunk, we also clamp its limits
                # TODO: How to handle for right arm which has a fixed trunk internally even though the trunk is moving
                # via the left arm??
                if self.rigid_trunk:
                    arm_cfg["control_limits"]["position"][0][self.trunk_control_idx] = th.zeros(4)
                    arm_cfg["control_limits"]["position"][1][self.trunk_control_idx] = th.zeros(4)

        return cfg

    def tuck(self):
        """
        Immediately set this robot's configuration to be in tucked mode
        """
        self.set_joint_positions(self.tucked_default_joint_pos)

    def untuck(self):
        """
        Immediately set this robot's configuration to be in untucked mode
        """
        self.set_joint_positions(self.untucked_default_joint_pos)

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        for arm in self.arm_names:
            pos[self.arm_control_idx[arm]] = th.tensor([0.0, 1.906, -0.991, 1.571, 0.915, -1.571])
        return pos

    @property
    def _default_joint_pos(self):
        return self.tucked_default_joint_pos if self.default_reset_mode == "tuck" else self.untucked_default_joint_pos

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def assisted_grasp_start_points(self):
        return {
            arm: [
                GraspingPoint(link_name=f"{arm}_gripper_link1", position=th.tensor([-0.032, 0.0, -0.009])),
                GraspingPoint(link_name=f"{arm}_gripper_link1", position=th.tensor([0.025, 0.0, -0.009])),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            arm: [
                GraspingPoint(link_name=f"{arm}_gripper_link1", position=th.tensor([-0.032, 0.0, -0.009])),
                GraspingPoint(link_name=f"{arm}_gripper_link1", position=th.tensor([0.025, 0.0, -0.009])),
            ]
            for arm in self.arm_names
        }

    @property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joints.
        """
        return th.tensor([list(self.joints.keys()).index(name) for name in self.trunk_joint_names])

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
        return {arm: f"{arm}_hand" for arm in self.arm_names}

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
        return {arm: [th.deg2rad(-45), th.deg2rad(45)] for arm in self.arm_names}

    @property
    def eef_usd_path(self):
        return {arm: os.path.join(gm.ASSET_PATH, "models/r1/r1_eef.usd") for arm in self.arm_names}

    @property
    def disabled_collision_pairs(self):
        # badly modeled gripper collision meshes
        return [
            ["left_gripper_link1", "left_gripper_link2"],
            ["right_gripper_link1", "right_gripper_link2"],
        ]
