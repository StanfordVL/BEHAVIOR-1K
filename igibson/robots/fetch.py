import os

import numpy as np


import igibson
from igibson.controllers import ControlType
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key

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
        name=None,
        control_freq=None,
        action_type="continuous",
        action_normalize=True,
        proprio_obs="default",
        reset_joint_pos=None,
        controller_config=None,
        base_name=None,
        scale=1.0,
        self_collision=True,
        class_id=SemanticClass.ROBOTS,
        rendering_params=None,
        grasping_mode="physical",
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_arm_pose="vertical",
    ):
        """
        :param name: None or str, name of the robot object
        :param control_freq: float, control frequency (in Hz) at which to control the robot. If set to be None,
            simulator.import_robot will automatically set the control frequency to be 1 / render_timestep by default.
        :param action_type: str, one of {discrete, continuous} - what type of action space to use
        :param action_normalize: bool, whether to normalize inputted actions. This will override any default values
         specified by this class.
        :param proprio_obs: str or tuple of str, proprioception observation key(s) to use for generating proprioceptive
            observations. If str, should be exactly "default" -- this results in the default proprioception observations
            being used, as defined by self.default_proprio_obs. See self._get_proprioception_dict for valid key choices
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param controller_config: None or Dict[str, ...], nested dictionary mapping controller name(s) to specific controller
            configurations for this robot. This will override any default values specified by this class.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the first link name used in @model_file
        :param scale: int, scaling factor for model (default is 1)
        :param self_collision: bool, whether to enable self collision
        :param class_id: SemanticClass, semantic class this robot belongs to. Default is SemanticClass.ROBOTS.
        :param rendering_params: None or Dict[str, Any], If not None, should be keyword-mapped rendering options to set.
            See DEFAULT_RENDERING_PARAMS for the values passed by default.
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        :param rigid_trunk: bool, if True, will prevent the trunk from moving during execution.
        :param default_trunk_offset: float, sets the default height of the robot's trunk
        :param default_arm_pose: Default pose for the robot arm. Should be one of {"vertical", "diagonal15",
            "diagonal30", "diagonal45", "horizontal"}
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
            name=name,
            control_freq=control_freq,
            action_type=action_type,
            action_normalize=action_normalize,
            proprio_obs=proprio_obs,
            reset_joint_pos=reset_joint_pos,
            controller_config=controller_config,
            base_name=base_name,
            scale=scale,
            self_collision=self_collision,
            class_id=class_id,
            rendering_params=rendering_params,
            grasping_mode=grasping_mode,
        )

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Fetch"

    @property
    def tucked_default_joint_pos(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                0.0,  # head
                1.1707963267948966,
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

    def _load(self, simulator):
        # Run super method
        ids = super()._load(simulator)

        assert len(self.get_body_ids()) == 1, "Fetch robot is expected to have only one body ID."

        # Extend super method by increasing laterial friction for EEF
        for link in self.finger_joint_ids[self.default_arm]:
            p.changeDynamics(self.base_link.body_id, link, lateralFriction=500)

        return ids

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
        dic["trunk_qpos"] = self.joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = self.joint_velocities[self.trunk_control_idx]

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

        # We use parallel jaw gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "ParallelJawGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Use default IK controller -- also need to override joint idx being controlled to include trunk in default
        # IK arm controller
        cfg["arm_{}".format(self.default_arm)]["InverseKinematicsController"]["joint_idx"] = np.concatenate(
            [self.trunk_control_idx, self.arm_control_idx[self.default_arm]]
        )

        # If using rigid trunk, we also clamp its limits
        if self.rigid_trunk:
            cfg["arm_{}".format(self.default_arm)]["InverseKinematicsController"]["control_limits"]["position"][0][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]
            cfg["arm_{}".format(self.default_arm)]["InverseKinematicsController"]["control_limits"]["position"][1][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]

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
    def gripper_link_to_grasp_point(self):
        return {self.default_arm: np.array([0.1, 0, 0])}

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
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([0, 1])

    @property
    def trunk_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([2])

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([3, 4])

    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {self.default_arm: np.array([5, 6, 7, 8, 9, 10, 11])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        return {self.default_arm: np.array([12, 13])}

    @property
    def disabled_collision_pairs(self):
        return [
            ["torso_lift_link", "shoulder_lift_link"],
            ["torso_lift_link", "torso_fixed_link"],
            ["caster_wheel_link", "estop_link"],
            ["caster_wheel_link", "laser_link"],
            ["caster_wheel_link", "torso_fixed_link"],
            ["caster_wheel_link", "l_wheel_link"],
            ["caster_wheel_link", "r_wheel_link"],
        ]

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
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/fetch/fetch_gripper.urdf")

    def dump_config(self):
        """Dump robot config"""
        dump = super(Fetch, self).dump_config()
        dump.update(
            {
                "rigid_trunk": self.rigid_trunk,
                "default_trunk_offset": self.default_trunk_offset,
                "default_arm_pose": self.default_arm_pose,
            }
        )
        return dump
