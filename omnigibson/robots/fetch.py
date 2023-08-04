import os
import numpy as np

from omnigibson.macros import gm
from omnigibson.controllers import ControlType
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import JointType

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


class Fetch(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        prim_path=None,
        class_id=None,
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
        obs_modalities="all",
        proprio_obs="default",

        # Unique to ManipulationRobot
        grasping_mode="physical",

        # Unique to Fetch
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_arm_pose="diagonal30",

        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
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
                simulator.import_object will automatically set the control frequency to be 1 / render_timestep by default.
            controller_config (None or dict): nested dictionary mapping controller name(s) to specific controller
                configurations for this object. This will override any default values specified by this class.
            action_type (str): one of {discrete, continuous} - what type of action space to use
            action_normalize (bool): whether to normalize inputted actions. This will override any default values
                specified by this class.
            reset_joint_pos (None or n-array): if specified, should be the joint positions that the object should
                be set to during a reset. If None (default), self.default_joint_pos will be used instead.
            obs_modalities (str or list of str): Observation modalities to use for this robot. Default is "all", which
                corresponds to all modalities being used.
                Otherwise, valid options should be part of omnigibson.sensors.ALL_SENSOR_MODALITIES.
            proprio_obs (str or list of str): proprioception observation key(s) to use for generating proprioceptive
                observations. If str, should be exactly "default" -- this results in the default proprioception
                observations being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict
                for valid key choices
            grasping_mode (str): One of {"physical", "assisted", "sticky"}.
                If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
                If "assisted", will magnetize any object touching and within the gripper's fingers.
                If "sticky", will magnetize any object touching the gripper's fingers.
            rigid_trunk (bool) if True, will prevent the trunk from moving during execution.
            default_trunk_offset (float): sets the default height of the robot's trunk
            default_arm_pose (str): Default pose for the robot arm. Should be one of:
                {"vertical", "diagonal15", "diagonal30", "diagonal45", "horizontal"}
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
            class_id=class_id,
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
            grasping_mode=grasping_mode,
            **kwargs,
        )

    @property
    def model_name(self):
        return "Fetch"

    @property
    def tucked_default_joint_pos(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                1.1707963267948966,
                0.0,  # head
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
                0.0,  # arm
                0.05,
                0.05,  # gripper
            ]
        )

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
                [-0.94121, -0.64134, 1.55186, 1.65672, -0.93218, 1.53416, 2.14474]
            )
        elif self.default_arm_pose == "diagonal15":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-0.95587, -0.34778, 1.46388, 1.47821, -0.93813, 1.4587, 1.9939]
            )
        elif self.default_arm_pose == "diagonal30":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996]
            )
        elif self.default_arm_pose == "diagonal45":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-1.11479, -0.0685, 1.5696, 1.37304, -0.74273, 1.3983, 1.79618]
            )
        elif self.default_arm_pose == "horizontal":
            pos[self.arm_control_idx[self.default_arm]] = np.array(
                [-1.43016, 0.20965, 1.86816, 1.77576, -0.27289, 1.31715, 2.01226]
            )
        else:
            raise ValueError("Unknown default arm pose: {}".format(self.default_arm_pose))
        return pos

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
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
        joint_positions = self.get_joint_positions(normalized=False)
        joint_velocities = self.get_joint_velocities(normalized=False)
        dic["trunk_qpos"] = joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "camera", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Need to override joint idx being controlled to include trunk in default arm controller configs
        for arm_cfg in cfg[f"arm_{self.default_arm}"].values():
            arm_cfg["dof_idx"] = np.concatenate([self.trunk_control_idx, self.arm_control_idx[self.default_arm]])

            # If using rigid trunk, we also clamp its limits
            if self.rigid_trunk:
                arm_cfg["control_limits"]["position"][0][self.trunk_control_idx] = \
                    self.untucked_default_joint_pos[self.trunk_control_idx]
                arm_cfg["control_limits"]["position"][1][self.trunk_control_idx] = \
                    self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg

    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def finger_lengths(self):
        return {self.default_arm: 0.1}

    @property
    def assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="r_gripper_finger_link", position=[0.04, -0.012, 0.014]),
                GraspingPoint(link_name="r_gripper_finger_link", position=[0.04, -0.012, -0.014]),
                GraspingPoint(link_name="r_gripper_finger_link", position=[-0.04, -0.012, 0.014]),
                GraspingPoint(link_name="r_gripper_finger_link", position=[-0.04, -0.012, -0.014]),
            ]
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="l_gripper_finger_link", position=[0.04, 0.012, 0.014]),
                GraspingPoint(link_name="l_gripper_finger_link", position=[0.04, 0.012, -0.014]),
                GraspingPoint(link_name="l_gripper_finger_link", position=[-0.04, 0.012, 0.014]),
                GraspingPoint(link_name="l_gripper_finger_link", position=[-0.04, 0.012, -0.014]),
            ]
        }

    @property
    def base_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([2])

    @property
    def camera_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([3, 5])

    @property
    def arm_control_idx(self):
        return {self.default_arm: np.array([4, 6, 7, 8, 9, 10, 11])}

    @property
    def gripper_control_idx(self):
        return {self.default_arm: np.array([12, 13])}

    @property
    def disabled_collision_pairs(self):
        return [
            ["torso_lift_link", "shoulder_lift_link"],
            ["torso_lift_link", "torso_fixed_link"],
        ]

    @property
    def arm_link_names(self):
        return {self.default_arm: [
            "shoulder_pan_link",
            "shoulder_lift_link",
            "upperarm_roll_link",
            "elbow_flex_link",
            "forearm_roll_link",
            "wrist_flex_link",
            "wrist_roll_link",
        ]}

    @property
    def arm_joint_names(self):
        return {self.default_arm: [
            "torso_lift_joint",
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "upperarm_roll_joint",
            "elbow_flex_joint",
            "forearm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]}

    @property
    def eef_link_names(self):
        return {self.default_arm: "gripper_link"}

    @property
    def finger_link_names(self):
        return {self.default_arm: ["r_gripper_finger_link", "l_gripper_finger_link"]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["r_gripper_finger_joint", "l_gripper_finger_joint"]}

    @property
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/fetch/fetch.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {self.default_arm: os.path.join(gm.ASSET_PATH, "models/fetch/fetch_descriptor.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/fetch/fetch.urdf")
