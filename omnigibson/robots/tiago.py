import math
import os
from typing import Literal

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.manipulation_robot import GraspingPoint
from omnigibson.robots.untucked_arm_pose_robot import UntuckedArmPoseRobot
from omnigibson.utils.python_utils import assert_valid_key, classproperty
from omnigibson.utils.usd_utils import ControllableObjectViewAPI


class Tiago(HolonomicBaseRobot, ArticulatedTrunkRobot, UntuckedArmPoseRobot, ActiveCameraRobot):
    """
    Tiago Robot
    Reference: https://pal-robotics.com/robots/tiago/
    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
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
        # Unique to ArticulatedTrunkRobot
        rigid_trunk=False,
        default_trunk_offset=0.2,
        # Unique to MobileManipulationRobot
        default_reset_mode="untuck",
        # Unique to UntuckedArmPoseRobot
        default_arm_pose="diagonal15",
        # Unique to Tiago
        variant="default",
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
            rigid_trunk (bool): If True, will prevent the trunk from moving during execution.
            default_trunk_offset (float): The default height of the robot's trunk
            default_reset_mode (str): Default reset mode for the robot. Should be one of: {"tuck", "untuck"}
                If reset_joint_pos is not None, this will be ignored (since _default_joint_pos won't be used during initialization).
            default_arm_pose (str): Default pose for the robot arm. Should be one of:
                {"vertical", "diagonal15", "diagonal30", "diagonal45", "horizontal"}
                If either reset_joint_pos is not None or default_reset_mode is "tuck", this will be ignored.
                Otherwise the reset_joint_pos will be initialized to the precomputed joint positions that represents default_arm_pose.
            variant (str): Which variant of the robot should be loaded. One of "default", "wrist_cam"
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store args
        assert variant in ("default", "wrist_cam"), f"Invalid Tiago variant specified {variant}!"
        self._variant = variant

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
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
            rigid_trunk=rigid_trunk,
            default_trunk_offset=default_trunk_offset,
            default_reset_mode=default_reset_mode,
            default_arm_pose=default_arm_pose,
            **kwargs,
        )

    @property
    def arm_joint_names(self):
        names = dict()
        for arm in self.arm_names:
            names[arm] = [f"arm_{arm}_{i}_joint" for i in range(1, 8)]
        return names

    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        pos[self.trunk_control_idx] = 0
        pos[self.camera_control_idx] = th.tensor([0.0, 0.0])
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.045, 0.045])  # open gripper
            pos[self.arm_control_idx[arm]] = th.tensor([-1.10, 1.47, 2.71, 1.71, -1.57, 1.39, 0])
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = super().untucked_default_joint_pos
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        pos[self.trunk_control_idx] = 0.02 + self.default_trunk_offset
        pos[self.camera_control_idx] = th.tensor([0.0, -0.45])
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.045, 0.045])  # open gripper
        return pos

    @property
    def default_arm_poses(self):
        return {
            "vertical": th.tensor([0.85846, -0.14852, 1.81008, 1.63368, 0.13764, -1.32488, -0.68415]),
            "diagonal15": th.tensor([0.90522, -0.42811, 2.23505, 1.64627, 0.76867, -0.79464, -1.08908]),
            "diagonal30": th.tensor([0.71883, -0.02787, 1.86002, 1.52897, 0.52204, -0.99741, -1.11046]),
            "diagonal45": th.tensor([0.66058, -0.14251, 1.77547, 1.43345, 0.65988, -1.02741, -1.32857]),
            "horizontal": th.tensor([0.61511, 0.49229, 1.46306, 1.24919, 1.08282, -1.28865, 1.50910]),
        }

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("Tiago does not support discrete actions!")

    def _post_load(self):
        super()._post_load()
        # The eef gripper links should be visual-only. They only contain a "ghost" box volume for detecting objects
        # inside the gripper, in order to activate attachments (AG for Cloths).
        for arm in self.arm_names:
            self.eef_links[arm].visual_only = True
            self.eef_links[arm].visible = False

    # Name of the actual root link that we are interested in. Note that this is different from self.root_link_name,
    # which is "base_footprint_x", corresponding to the first of the 6 1DoF joints to control the base.
    @property
    def base_footprint_link_name(self):
        return "base_footprint"

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
        # We use joint controllers for base and camera as default
        controllers["base"] = "JointController"
        controllers["camera"] = "JointController"
        # We use multi finger gripper, and IK controllers for eefs as default
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def assisted_grasp_start_points(self):
        return {
            arm: [
                GraspingPoint(
                    link_name="gripper_{}_right_finger_link".format(arm), position=th.tensor([0.002, 0.0, -0.2])
                ),
                GraspingPoint(
                    link_name="gripper_{}_right_finger_link".format(arm), position=th.tensor([0.002, 0.0, -0.13])
                ),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            arm: [
                GraspingPoint(
                    link_name="gripper_{}_left_finger_link".format(arm), position=th.tensor([-0.002, 0.0, -0.2])
                ),
                GraspingPoint(
                    link_name="gripper_{}_left_finger_link".format(arm), position=th.tensor([-0.002, 0.0, -0.13])
                ),
            ]
            for arm in self.arm_names
        }

    @property
    def arm_control_idx(self):
        # Add combined entry
        idxs = super().arm_control_idx
        # Concatenate all values and sort them
        idxs["combined"] = th.sort(th.cat([val for val in idxs.values()]))[0]
        return idxs

    @property
    def finger_lengths(self):
        return {arm: 0.12 for arm in self.arm_names}

    @property
    def disabled_collision_link_names(self):
        # These should NEVER have collisions in the first place (i.e.: these are poorly modeled geoms from the source
        # asset) -- they are strictly engulfed within ANOTHER collision mesh from a DIFFERENT link
        return [
            name
            for arm in self.arm_names
            for name in [f"arm_{arm}_tool_link", f"wrist_{arm}_ft_link", f"wrist_{arm}_ft_tool_link"]
        ]

    @property
    def disabled_collision_pairs(self):
        return [
            ["arm_left_1_link", "arm_left_2_link"],
            ["arm_left_2_link", "arm_left_3_link"],
            ["arm_left_3_link", "arm_left_4_link"],
            ["arm_left_4_link", "arm_left_5_link"],
            ["arm_left_5_link", "arm_left_6_link"],
            ["arm_left_6_link", "arm_left_7_link"],
            ["arm_right_1_link", "arm_right_2_link"],
            ["arm_right_2_link", "arm_right_3_link"],
            ["arm_right_3_link", "arm_right_4_link"],
            ["arm_right_4_link", "arm_right_5_link"],
            ["arm_right_5_link", "arm_right_6_link"],
            ["arm_right_6_link", "arm_right_7_link"],
            ["gripper_right_right_finger_link", "gripper_right_left_finger_link"],
            ["gripper_right_link", "wrist_right_ft_link"],
            ["arm_right_6_link", "gripper_right_link"],
            ["arm_right_6_link", "wrist_right_ft_tool_link"],
            ["arm_right_6_link", "wrist_right_ft_link"],
            ["arm_right_6_link", "arm_right_tool_link"],
            ["arm_right_5_link", "wrist_right_ft_link"],
            ["arm_right_5_link", "arm_right_tool_link"],
            ["gripper_left_right_finger_link", "gripper_left_left_finger_link"],
            ["gripper_left_link", "wrist_left_ft_link"],
            ["arm_left_6_link", "gripper_left_link"],
            ["arm_left_6_link", "wrist_left_ft_tool_link"],
            ["arm_left_6_link", "wrist_left_ft_link"],
            ["arm_left_6_link", "arm_left_tool_link"],
            ["arm_left_5_link", "wrist_left_ft_link"],
            ["arm_left_5_link", "arm_left_tool_link"],
            ["torso_lift_link", "torso_fixed_column_link"],
            ["torso_fixed_link", "torso_fixed_column_link"],
            ["base_antenna_left_link", "torso_fixed_link"],
            ["base_antenna_right_link", "torso_fixed_link"],
            ["base_link", "wheel_rear_left_link"],
            ["base_link", "wheel_rear_right_link"],
            ["base_link", "wheel_front_left_link"],
            ["base_link", "wheel_front_right_link"],
            ["base_link", "base_dock_link"],
            ["base_link", "base_antenna_right_link"],
            ["base_link", "base_antenna_left_link"],
            ["base_link", "torso_fixed_column_link"],
            ["base_link", "suspension_front_left_link"],
            ["base_link", "suspension_front_right_link"],
            ["base_link", "torso_fixed_link"],
            ["suspension_front_left_link", "wheel_front_left_link"],
            ["torso_lift_link", "arm_right_1_link"],
            ["torso_lift_link", "arm_right_2_link"],
            ["torso_lift_link", "arm_left_1_link"],
            ["torso_lift_link", "arm_left_2_link"],
            ["arm_left_tool_link", "wrist_left_ft_link"],
            ["wrist_left_ft_link", "wrist_left_ft_tool_link"],
            ["wrist_left_ft_tool_link", "gripper_left_link"],
            ["gripper_left_grasping_frame", "gripper_left_left_finger_link"],
            ["gripper_left_grasping_frame", "gripper_left_right_finger_link"],
            ["wrist_right_ft_link", "arm_right_tool_link"],
            ["wrist_right_ft_tool_link", "wrist_right_ft_link"],
            ["gripper_right_link", "wrist_right_ft_tool_link"],
            ["head_1_link", "head_2_link"],
            ["torso_fixed_column_link", "arm_right_1_link"],
            ["torso_fixed_column_link", "arm_left_1_link"],
            ["arm_left_1_link", "arm_left_3_link"],
            ["arm_right_1_link", "arm_right_3_link"],
            ["base_link", "arm_right_4_link"],
            ["base_link", "arm_right_5_link"],
            ["base_link", "arm_left_4_link"],
            ["base_link", "arm_left_5_link"],
            ["wrist_left_ft_tool_link", "arm_left_5_link"],
            ["wrist_right_ft_tool_link", "arm_right_5_link"],
            ["arm_left_tool_link", "wrist_left_ft_tool_link"],
            ["arm_right_tool_link", "wrist_right_ft_tool_link"],
        ]

    @property
    def camera_joint_names(self):
        return ["head_1_joint", "head_2_joint"]

    @property
    def trunk_joint_names(self):
        return ["torso_lift_joint"]

    @property
    def manipulation_link_names(self):
        return [
            "torso_fixed_link",
            "torso_lift_link",
            "arm_left_1_link",
            "arm_left_2_link",
            "arm_left_3_link",
            "arm_left_4_link",
            "arm_left_5_link",
            "arm_left_6_link",
            "arm_left_7_link",
            "arm_left_tool_link",
            "wrist_left_ft_link",
            "wrist_left_ft_tool_link",
            "gripper_left_link",
            # "gripper_left_grasping_frame",
            "gripper_left_left_finger_link",
            "gripper_left_right_finger_link",
            "gripper_left_tool_link",
            "arm_right_1_link",
            "arm_right_2_link",
            "arm_right_3_link",
            "arm_right_4_link",
            "arm_right_5_link",
            "arm_right_6_link",
            "arm_right_7_link",
            "arm_right_tool_link",
            "wrist_right_ft_link",
            "wrist_right_ft_tool_link",
            "gripper_right_link",
            # "gripper_right_grasping_frame",
            "gripper_right_left_finger_link",
            "gripper_right_right_finger_link",
            "gripper_right_tool_link",
            "head_1_link",
            "head_2_link",
            "xtion_link",
        ]

    @property
    def arm_link_names(self):
        return {arm: [f"arm_{arm}_{i}_link" for i in range(1, 8)] for arm in self.arm_names}

    @property
    def eef_link_names(self):
        return {arm: "gripper_{}_grasping_frame".format(arm) for arm in self.arm_names}

    @property
    def finger_link_names(self):
        return {arm: [f"gripper_{arm}_right_finger_link", f"gripper_{arm}_left_finger_link"] for arm in self.arm_names}

    @property
    def finger_joint_names(self):
        return {
            arm: [f"gripper_{arm}_right_finger_joint", f"gripper_{arm}_left_finger_joint"] for arm in self.arm_names
        }

    @property
    def usd_path(self):
        if self._variant == "wrist_cam":
            return os.path.join(
                gm.ASSET_PATH,
                "models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33_with_wrist_cam.usd",
            )

        # Default variant
        return os.path.join(
            gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33.usd"
        )

    @property
    def simplified_mesh_usd_path(self):
        # TODO: How can we make this more general - maybe some automatic way to generate these?
        return os.path.join(
            gm.ASSET_PATH,
            "models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33_simplified_collision_mesh.usd",
        )

    @property
    def robot_arm_descriptor_yamls(self):
        # TODO: Remove the need to do this by making the arm descriptor yaml files generated automatically
        return {
            "left": os.path.join(
                gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford_left_arm_descriptor.yaml"
            ),
            "left_fixed": os.path.join(
                gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford_left_arm_fixed_trunk_descriptor.yaml"
            ),
            "right": os.path.join(
                gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford_right_arm_fixed_trunk_descriptor.yaml"
            ),
            "combined": os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford.yaml"),
        }

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford.urdf")

    @property
    def arm_workspace_range(self):
        return {
            "left": th.deg2rad(th.tensor([15, 75], dtype=th.float32)),
            "right": th.deg2rad(th.tensor([-75, -15], dtype=th.float32)),
        }

    @property
    def eef_usd_path(self):
        return {
            arm: os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford/tiago_eef.usd")
            for arm in self.arm_names
        }
