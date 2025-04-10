import math
from functools import cached_property

import torch as th

from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.robots.untucked_arm_pose_robot import UntuckedArmPoseRobot
from omnigibson.utils.transform_utils import euler2quat
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class Fetch(TwoWheelRobot, ArticulatedTrunkRobot, UntuckedArmPoseRobot, ActiveCameraRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
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
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        disable_grasp_handling=False,
        finger_static_friction=None,
        finger_dynamic_friction=None,
        # Unique to MobileManipulationRobot
        default_reset_mode="untuck",
        # Unique to UntuckedArmPoseRobot
        default_arm_pose="vertical",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            fixed_base (bool): whether to fix the base of this object or not
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
            include_sensor_names (None or list of str): If specified, substring(s) to check for in all raw sensor prim
                paths found on the robot. A sensor must include one of the specified substrings in order to be included
                in this robot's set of sensors
            exclude_sensor_names (None or list of str): If specified, substring(s) to check against in all raw sensor
                prim paths found on the robot. A sensor must not include any of the specified substrings in order to
                be included in this robot's set of sensors
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
            finger_static_friction (None or float): If specified, specific static friction to use for robot's fingers
            finger_dynamic_friction (None or float): If specified, specific dynamic friction to use for robot's fingers.
                Note: If specified, this will override any ways that are found within @link_physics_materials for any
                robot finger gripper links
            default_reset_mode (str): Default reset mode for the robot. Should be one of: {"tuck", "untuck"}
                If reset_joint_pos is not None, this will be ignored (since _default_joint_pos won't be used during initialization).
            default_arm_pose (str): Default pose for the robot arm. Should be one of:
                {"vertical", "diagonal15", "diagonal30", "diagonal45", "horizontal"}
                If either reset_joint_pos is not None or default_reset_mode is "tuck", this will be ignored.
                Otherwise the reset_joint_pos will be initialized to the precomputed joint positions that represents default_arm_pose.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            control_freq=control_freq,
            controller_config=controller_config,
            action_type=action_type,
            action_normalize=action_normalize,
            reset_joint_pos=reset_joint_pos,
            obs_modalities=obs_modalities,
            include_sensor_names=include_sensor_names,
            exclude_sensor_names=exclude_sensor_names,
            proprio_obs=proprio_obs,
            sensor_config=sensor_config,
            grasping_mode=grasping_mode,
            disable_grasp_handling=disable_grasp_handling,
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            default_reset_mode=default_reset_mode,
            default_arm_pose=default_arm_pose,
            **kwargs,
        )

    @property
    def tucked_default_joint_pos(self):
        return th.tensor(
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
        pos = super().untucked_default_joint_pos
        pos[self.base_control_idx] = 0.0
        pos[self.trunk_control_idx] = 0.385
        pos[self.camera_control_idx] = th.tensor([0.0, 0.45])
        pos[self.gripper_control_idx[self.default_arm]] = th.tensor([0.05, 0.05])  # open gripper
        return pos

    @property
    def default_arm_poses(self):
        return {
            "vertical": th.tensor([-0.94121, -0.64134, 1.55186, 1.65672, -0.93218, 1.53416, 2.14474]),
            "diagonal15": th.tensor([-0.95587, -0.34778, 1.46388, 1.47821, -0.93813, 1.4587, 1.9939]),
            "diagonal30": th.tensor([-1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996]),
            "diagonal45": th.tensor([-1.11479, -0.0685, 1.5696, 1.37304, -0.74273, 1.3983, 1.79618]),
            "horizontal": th.tensor([-1.43016, 0.20965, 1.86816, 1.77576, -0.27289, 1.31715, 2.01226]),
        }

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("Fetch does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "trunk", "camera", f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["trunk"] = "JointController"
        controllers["camera"] = "JointController"
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"

        return controllers

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @cached_property
    def base_joint_names(self):
        return ["l_wheel_joint", "r_wheel_joint"]

    @cached_property
    def camera_joint_names(self):
        return ["head_pan_joint", "head_tilt_joint"]

    @cached_property
    def trunk_joint_names(self):
        return ["torso_lift_joint"]

    @cached_property
    def manipulation_link_names(self):
        return [
            "torso_lift_link",
            "head_pan_link",
            "head_tilt_link",
            "shoulder_pan_link",
            "shoulder_lift_link",
            "upperarm_roll_link",
            "elbow_flex_link",
            "forearm_roll_link",
            "wrist_flex_link",
            "wrist_roll_link",
            "eef_link",
            "l_gripper_finger_link",
            "r_gripper_finger_link",
        ]

    @cached_property
    def arm_link_names(self):
        return {
            self.default_arm: [
                "shoulder_pan_link",
                "shoulder_lift_link",
                "upperarm_roll_link",
                "elbow_flex_link",
                "forearm_roll_link",
                "wrist_flex_link",
                "wrist_roll_link",
            ]
        }

    @cached_property
    def arm_joint_names(self):
        return {
            self.default_arm: [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ]
        }

    @cached_property
    def eef_link_names(self):
        return {self.default_arm: "eef_link"}

    @cached_property
    def finger_link_names(self):
        return {self.default_arm: ["r_gripper_finger_link", "l_gripper_finger_link"]}

    @cached_property
    def finger_joint_names(self):
        return {self.default_arm: ["r_gripper_finger_joint", "l_gripper_finger_joint"]}

    @property
    def arm_workspace_range(self):
        return {self.default_arm: th.deg2rad(th.tensor([-45, 45], dtype=th.float32))}

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: euler2quat(th.tensor([0, math.pi / 2, math.pi], dtype=th.float32))}
