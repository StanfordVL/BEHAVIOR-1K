import os

import numpy as np


import igibson
from igibson.controllers import ControlType
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.robots.locomotion_robot import LocomotionRobot
from igibson.utils.python_utils import assert_valid_key
from igibson.utils.usd_utils import JointType
import igibson.macros as m

DEFAULT_ARM_POSES = {
    "vertical",
    "diagonal15",
    "diagonal30",
    "diagonal45",
    "horizontal",
}

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class Tiago(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    Tiago Robot
    Reference: https://pal-robotics.com/robots/tiago/
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        prim_path,
        name=None,
        category="agent",
        class_id=None,
        uuid=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
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
        obs_modalities="all",
        proprio_obs="default",

        # Unique to ManipulationRobot
        grasping_mode="physical",

        # Unique to Fetch
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_arm_pose="vertical",

        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param uuid: Unique unsigned-integer identifier to assign to this object (max 8-numbers).
            If None is specified, then it will be auto-generated
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            robot abilities and parameters.
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific
            controller configurations for this object. This will override any default values specified by this class.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
            corresponds to all modalities being used.
            Otherwise, valid options should be part of igibson.sensors.ALL_SENSOR_MODALITIES.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or Array[float], if specified, should be the joint positions that the robot should
            be set to during a reset. If None (default), self.default_joint_pos will be used instead.
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        :param rigid_trunk: bool, if True, will prevent the trunk from moving during execution.
        :param default_trunk_offset: float, sets the default height of the robot's trunk
        :param default_arm_pose: Default pose for the robot arm. Should be one of {"vertical", "diagonal15",
            "diagonal30", "diagonal45", "horizontal"}
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store args
        self.rigid_trunk = rigid_trunk
        self.default_trunk_offset = default_trunk_offset
        assert_valid_key(key=default_arm_pose, valid_keys=DEFAULT_ARM_POSES, name="default_arm_pose")
        self.default_arm_pose = default_arm_pose

        # Parse reset joint pos if specifying special string
        if isinstance(reset_joint_pos, str):
            assert (
                reset_joint_pos in RESET_JOINT_OPTIONS
            ), "reset_joint_pos should be one of {} if using a string!".format(RESET_JOINT_OPTIONS)
            reset_joint_pos = (
                self.tucked_default_joint_pos if reset_joint_pos == "tuck" else self.untucked_default_joint_pos
            )

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            rendering_params=rendering_params,
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
            grasping_mode=grasping_mode,
            **kwargs,
        )

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Tiago"

    @property
    def n_arms(self):
        return 2

    @property
    def arm_names(self):
        return ["left", "right"]

    @property
    def tucked_default_joint_pos(self):
        if m.IS_PUBLIC_ISAACSIM:
            vals = np.array(
                [
                    # 0.0,        # wheels
                    # 0.0,
                    0.0,        # trunk
                    -1.10,
                    1.47,
                    2.71,
                    1.71,
                    -1.57,
                    1.39,
                    0.0,
                    0.045,  # gripper
                    0.045,
                    -1.10,
                    1.47,
                    2.71,
                    1.71,
                    -1.57,
                    1.39,
                    0.0,
                    0.045,
                    0.045,
                    0.0,        # head
                    0.0,        # head
                ]
            )
        else:
            vals = np.array(
                [
                    # 0.0,        # wheels
                    # 0.0,
                    0.0,        # trunk
                    -1.10,
                    -1.10,
                    0.0,        # head
                    1.47,
                    1.47,
                    0.0,        # head
                    2.71,
                    2.71,
                    1.71,
                    1.71,
                    -1.57,
                    -1.57,
                    1.39,
                    1.39,
                    0.0,
                    0.0,
                    0.045,  # gripper
                    0.045,
                    0.045,
                    0.045,
                ]
            )

        return vals

    @property
    def untucked_default_joint_pos(self):
        pos = np.zeros(self.n_joints)
        pos[self.base_control_idx] = 0.0
        pos[self.trunk_control_idx] = 0.02 + self.default_trunk_offset
        pos[self.camera_control_idx] = np.array([0.0, 0.45])
        pos[self.gripper_control_idx[self.default_arm]] = np.array([0.05, 0.05])  # open gripper

        # Choose arm based on setting
        if self.default_arm_pose == "vertical":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
            )
        elif self.default_arm_pose == "diagonal15":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
            )
        elif self.default_arm_pose == "diagonal30":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
            )
        elif self.default_arm_pose == "diagonal45":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
            )
        elif self.default_arm_pose == "horizontal":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
            )
        else:
            raise ValueError("Unknown default arm pose: {}".format(self.default_arm_pose))
        return pos

    def _create_discrete_action_space(self):
        # Tiago does not support discrete actions
        raise ValueError("Fetch does not support discrete actions!")

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

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Set the joint friction for EEF to be higher
        for arm in self.arm_names:
            for joint in self.finger_joints[arm]:
                if joint.joint_type != JointType.JOINT_FIXED:
                    joint.friction = 500

    def _actions_to_control(self, action):
        # Run super method first
        u_vec, u_type_vec = super()._actions_to_control(action=action)

        # Override trunk value if we're keeping the trunk rigid
        if self.rigid_trunk:
            u_vec[self.trunk_control_idx] = self.untucked_default_joint_pos[self.trunk_control_idx]
            u_type_vec[self.trunk_control_idx] = ControlType.POSITION

        # Return control
        return u_vec, u_type_vec

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        joints_state = self.get_joints_state(normalized=False)
        dic["trunk_qpos"] = joints_state.positions[self.trunk_control_idx]
        dic["trunk_qvel"] = joints_state.velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        controllers = ["base", "camera"]
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]

        return controllers

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        # TODO: Get omnidirectional base working
        controllers["base"] = "NullJointController"
        controllers["camera"] = "JointController"
        # TODO: Revert to IK once implemented
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "JointController" #"InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        for arm in self.arm_names:
            if arm == "left":
                # Use default IK controller -- also need to override joint idx being controlled to include trunk in default
                # IK arm controller
                cfg["arm_{}".format(arm)]["InverseKinematicsController"]["dof_idx"] = np.concatenate(
                    [self.trunk_control_idx, self.arm_control_idx[arm]]
                )

                cfg["arm_{}".format(arm)]["JointController"]["dof_idx"] = np.concatenate(
                    [self.trunk_control_idx, self.arm_control_idx[arm]]
                )

            # If using rigid trunk, we also clamp its limits
            if self.rigid_trunk:
                cfg["arm_{}".format(arm)]["InverseKinematicsController"]["control_limits"]["position"][0][
                    self.trunk_control_idx
                ] = self.untucked_default_joint_pos[self.trunk_control_idx]
                cfg["arm_{}".format(arm)]["InverseKinematicsController"]["control_limits"]["position"][1][
                    self.trunk_control_idx
                ] = self.untucked_default_joint_pos[self.trunk_control_idx]
                cfg["arm_{}".format(arm)]["JointController"]["control_limits"]["position"][0][
                    self.trunk_control_idx
                ] = self.untucked_default_joint_pos[self.trunk_control_idx]
                cfg["arm_{}".format(arm)]["JointController"]["control_limits"]["position"][1][
                    self.trunk_control_idx
                ] = self.untucked_default_joint_pos[self.trunk_control_idx]

            # Must also create a default controller for each gripper


        return cfg

    @property
    def default_joint_pos(self):
        return self.tucked_default_joint_pos

    @property
    def wheel_radius(self):
        # TODO
        return 0.0613

    @property
    def wheel_axle_length(self):
        # TODO
        return 0.372

    @property
    def gripper_link_to_grasp_point(self):
        return {self.default_arm: np.array([0.1, 0, 0])}

    @property
    def assisted_grasp_start_points(self):
        return {
            arm: [
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[0.04, -0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[0.04, -0.012, -0.014]),
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[-0.04, -0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[-0.04, -0.012, -0.014]),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            arm: [
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[0.04, 0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[0.04, 0.012, -0.014]),
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[-0.04, 0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[-0.04, 0.012, -0.014]),
            ]
            for arm in self.arm_names
        }

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([], dtype=int)

    @property
    def trunk_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([0])

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([19, 20]) if m.IS_PUBLIC_ISAACSIM else np.array([3, 6])

    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        if m.IS_PUBLIC_ISAACSIM:
            vals = {"left": np.array([1, 2, 3, 4, 5, 6, 7]), "right": np.array([10, 11, 12, 13, 14, 15, 16])}
        else:
            vals = {"left": np.array([1, 4, 7, 9, 11, 13, 15]), "right": np.array([2, 5, 8, 10, 12, 14, 16])}
        return vals

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        if m.IS_PUBLIC_ISAACSIM:
            vals = {"left": np.array([8, 9]), "right": np.array([17, 18])}
        else:
            vals = {"left": np.array([17, 18]), "right": np.array([19, 20])}
        return vals

    @property
    def finger_lengths(self):
        return {arm: 0.12 for arm in self.arm_names}

    @property
    def disabled_collision_pairs(self):
        return []

    @property
    def arm_link_names(self):
        return {arm: [f"arm_{arm}_{i}_link" for i in range(1, 8)] for arm in self.arm_names}

    @property
    def eef_link_names(self):
        return {arm: "gripper_{}_grasping_frame".format(arm) for arm in self.arm_names}

    @property
    def finger_link_names(self):
        return {arm: ["gripper_{}_right_finger_link".format(arm), "gripper_{}_left_finger_link".format(arm)] for arm in
                self.arm_names}

    @property
    def finger_joint_names(self):
        return {arm: ["gripper_{}_right_finger_joint".format(arm), "gripper_{}_left_finger_joint".format(arm)] for arm
                in self.arm_names}

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford.usd")

    def dump_config(self):
        cfg = super().dump_config()

        cfg["rigid_trunk"] = self.rigid_trunk
        cfg["default_trunk_offset"] = self.default_trunk_offset
        cfg["default_arm_pose"] = self.default_arm_pose

        return cfg
