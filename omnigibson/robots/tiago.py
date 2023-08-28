import os

import numpy as np
from pxr import Gf

from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.utils.python_utils import assert_valid_key
from omnigibson.utils.usd_utils import JointType

from omni.isaac.core.utils.prims import get_prim_at_path

# Create settings for this module
m = create_module_macros(module_path=__file__)


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

m.MAX_LINEAR_VELOCITY = 1.5  # linear velocity in meters/second
m.MAX_ANGULAR_VELOCITY = np.pi  # angular velocity in radians/second


class Tiago(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    Tiago Robot
    Reference: https://pal-robotics.com/robots/tiago/

    NOTE: If using IK Control for both the right and left arms, note that the left arm dictates control of the trunk,
    and the right arm passively must follow. That is, sending desired delta position commands to the right end effector
    will be computed independently from any trunk motion occurring during that timestep.
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

        # Unique to Tiago
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_arm_pose="vertical",

        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            category (str): Category for the object. Defaults to "object".
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
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
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

        # Other args that will be created at runtime
        self._world_base_fixed_joint_prim = None

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
            grasping_mode=grasping_mode,
            **kwargs,
        )

    @property
    def arm_joint_names(self):
        names = dict()
        for arm in self.arm_names:
            names[arm] = ["torso_lift_joint"] + [
                f"arm_{arm}_{i}_joint" for i in range(1, 8)
            ]
        return names

    @property
    def model_name(self):
        return "Tiago"

    @property
    def n_arms(self):
        return 2

    @property
    def arm_names(self):
        return ["left", "right"]

    @property
    def tucked_default_joint_pos(self):
        pos = np.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        pos[self.trunk_control_idx] = 0
        pos[self.camera_control_idx] = np.array([0.0, 0.0])
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = np.array([0.045, 0.045])  # open gripper
            pos[self.arm_control_idx[arm]] = np.array(
                [-1.10, 1.47, 2.71, 1.71, -1.57, 1.39, 0]
            )
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = np.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        pos[self.trunk_control_idx] = 0.02 + self.default_trunk_offset
        pos[self.camera_control_idx] = np.array([0.0, 0.45])
        pos[self.gripper_control_idx[self.default_arm]] = np.array([0.045, 0.045])  # open gripper

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

    @property
    def discrete_action_list(self):
        # Not supported for this robot
        raise NotImplementedError()

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

    def reset(self):
        """
        Reset should not change the robot base pose.
        We need to cache and restore the base joints to the world.
        """
        base_joint_positions = self.get_joint_positions()[self.base_idx]
        super().reset()
        self.set_joint_positions(base_joint_positions, indices=self.base_idx)

    def _post_load(self):
        super()._post_load()
        # The eef gripper links should be visual-only. They only contain a "ghost" box volume for detecting objects
        # inside the gripper, in order to activate attachments (AG for Cloths).
        for arm in self.arm_names:
            self.eef_links[arm].visual_only = True
            self.eef_links[arm].visible = False

        self._world_base_fixed_joint_prim = get_prim_at_path(f"{self._prim_path}/rootJoint")
        position, orientation = self.get_position_orientation()
        # Set the world-to-base fixed joint to be at the robot's current pose
        self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
        self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*orientation[[3, 0, 1, 2]]))

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Set the joint friction for EEF to be higher
        for arm in self.arm_names:
            for joint in self.finger_joints[arm]:
                if joint.joint_type != JointType.JOINT_FIXED:
                    joint.friction = 500

    # Name of the actual root link that we are interested in. Note that this is different from self.root_link_name,
    # which is "base_footprint_x", corresponding to the first of the 6 1DoF joints to control the base.
    @property
    def base_footprint_link_name(self):
        return "base_footprint"

    @property
    def base_footprint_link(self):
        """
        Returns:
            RigidPrim: base footprint link of this object prim
        """
        return self._links[self.base_footprint_link_name]

    def _actions_to_control(self, action):
        # Run super method first
        u_vec, u_type_vec = super()._actions_to_control(action=action)

        # Change the control from base_footprint_link ("base_footprint") frame to root_link ("base_footprint_x") frame
        base_orn = self.base_footprint_link.get_orientation()
        root_link_orn = self.root_link.get_orientation()

        cur_orn = T.mat2quat(T.quat2mat(root_link_orn).T  @ T.quat2mat(base_orn))

        # Rotate the linear and angular velocity to the desired frame
        lin_vel_global, _ = T.pose_transform([0, 0, 0], cur_orn, u_vec[self.base_idx[:3]], [0, 0, 0, 1])
        ang_vel_global, _ = T.pose_transform([0, 0, 0], cur_orn, u_vec[self.base_idx[3:]], [0, 0, 0, 1])

        u_vec[self.base_control_idx] = np.array([lin_vel_global[0], lin_vel_global[1], ang_vel_global[2]])
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
    def control_limits(self):
        # Overwrite the control limits with the maximum linear and angular velocities for the purpose of clip_control
        # Note that when clip_control happens, the control is still in the base_footprint_link ("base_footprint") frame
        # Omniverse still thinks these joints have no limits because when the control is transformed to the root_link
        # ("base_footprint_x") frame, it can go above this limit.
        limits = super().control_limits
        limits["velocity"][0][self.base_idx[:3]] = -m.MAX_LINEAR_VELOCITY
        limits["velocity"][1][self.base_idx[:3]] = m.MAX_LINEAR_VELOCITY
        limits["velocity"][0][self.base_idx[3:]] = -m.MAX_ANGULAR_VELOCITY
        limits["velocity"][1][self.base_idx[3:]] = m.MAX_ANGULAR_VELOCITY
        return limits

    def get_control_dict(self):
        # Modify the right hand's pos_relative in the z-direction based on the trunk's value
        # We do this so we decouple the trunk's dynamic value from influencing the IK controller solution for the right
        # hand, which does not control the trunk
        dic = super().get_control_dict()
        dic["eef_right_pos_relative"][2] = dic["eef_right_pos_relative"][2] - self.get_joint_positions()[self.trunk_control_idx]

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
        controllers["base"] = "JointController"
        controllers["camera"] = "JointController"
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_base_controller_configs(self):
        dic = {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": False,
            "motor_type": "velocity",
            "compute_delta_in_quat_space": [(3, 4, 5)],
            "dof_idx": self.base_control_idx,
        }
        return dic

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Get default base controller for omnidirectional Tiago
        cfg["base"] = {"JointController": self._default_base_controller_configs}

        for arm in self.arm_names:
            for arm_cfg in cfg["arm_{}".format(arm)].values():

                if arm == "left":
                    # Need to override joint idx being controlled to include trunk in default arm controller configs
                    arm_cfg["dof_idx"] = np.concatenate([self.trunk_control_idx, self.arm_control_idx[arm]])

                # If using rigid trunk, we also clamp its limits
                # TODO: How to handle for right arm which has a fixed trunk internally even though the trunk is moving
                # via the left arm??
                if self.rigid_trunk:
                    arm_cfg["control_limits"]["position"][0][self.trunk_control_idx] = \
                        self.untucked_default_joint_pos[self.trunk_control_idx]
                    arm_cfg["control_limits"]["position"][1][self.trunk_control_idx] = \
                        self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg

    @property
    def default_joint_pos(self):
        return self.tucked_default_joint_pos

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
        Returns:
            n-array: Indices in low-level control vector corresponding to the three controllable 1DoF base joints
        """
        joints = list(self.joints.keys())
        return np.array(
            [
                joints.index(f"base_footprint_{component}_joint")
                for component in ["x", "y", "rz"]
            ]
        )

    @property
    def base_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to the six 1DoF base joints
        """
        joints = list(self.joints.keys())
        return np.array(
            [
                joints.index(f"base_footprint_{component}_joint")
                for component in ["x", "y", "z", "rx", "ry", "rz"]
            ]
        )

    @property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([6])

    @property
    def camera_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([9, 12])

    @property
    def arm_control_idx(self):
        return {"left": np.array([7, 10, 13, 15, 17, 19, 21]), "right": np.array([8, 11, 14, 16, 18, 20, 22])}

    @property
    def gripper_control_idx(self):
        return {"left": np.array([23, 24]), "right": np.array([25, 26])}

    @property
    def finger_lengths(self):
        return {arm: 0.12 for arm in self.arm_names}

    @property
    def disabled_collision_link_names(self):
        # These should NEVER have collisions in the first place (i.e.: these are poorly modeled geoms from the source
        # asset) -- they are strictly engulfed within ANOTHER collision mesh from a DIFFERENT link
        return [name for arm in self.arm_names for name in [f"arm_{arm}_tool_link", f"wrist_{arm}_ft_link", f"wrist_{arm}_ft_tool_link"]]

    @property
    def disabled_collision_pairs(self):
        return [
            ["torso_fixed_column_link", "torso_fixed_link"],
            ["torso_fixed_column_link", "torso_lift_link"],
            ["arm_left_6_link", "gripper_left_link"],
            ["arm_right_6_link", "gripper_right_link"],
        ]

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
    def usd_path(self):
        return os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford/tiago_dual_omnidirectional_stanford_33.usd")

    @property
    def robot_arm_descriptor_yamls(self):
        return {"left": os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford_left_arm_descriptor.yaml"),
                "right": os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford_right_arm_fixed_trunk_descriptor.yaml")}

    @property
    def urdf_path(self):
        return os.path.join(gm.ASSET_PATH, "models/tiago/tiago_dual_omnidirectional_stanford.urdf")

    def get_position_orientation(self):
        # If the simulator is playing, return the pose of the base_footprint link frame
        if self._dc is not None and self._dc.is_simulating():
            return self.base_footprint_link.get_position_orientation()

        # Else, return the pose of the robot frame
        else:
            return super().get_position_orientation()

    def set_position_orientation(self, position=None, orientation=None):
        current_position, current_orientation = self.get_position_orientation()
        if position is None:
            position = current_position
        if orientation is None:
            orientation = current_orientation

        # If the simulator is playing, set the 6 base joints to achieve the desired pose of base_footprint link frame
        if self._dc is not None and self._dc.is_simulating():
            # Find the relative transformation from base_footprint_link ("base_footprint") frame to root_link
            # ("base_footprint_x") frame. Assign it to the 6 1DoF joints that control the base.
            # Note that the 6 1DoF joints are originated from the root_link ("base_footprint_x") frame.
            joint_pos, joint_orn = self.root_link.get_position_orientation()
            inv_joint_pos, inv_joint_orn = T.mat2pose(T.pose_inv(T.pose2mat((joint_pos, joint_orn))))

            relative_pos, relative_orn = T.pose_transform(inv_joint_pos, inv_joint_orn, position, orientation)
            relative_rpy = T.quat2euler(relative_orn)
            self.joints["base_footprint_x_joint"].set_pos(relative_pos[0], drive=False)
            self.joints["base_footprint_y_joint"].set_pos(relative_pos[1], drive=False)
            self.joints["base_footprint_z_joint"].set_pos(relative_pos[2], drive=False)
            self.joints["base_footprint_rx_joint"].set_pos(relative_rpy[0], drive=False)
            self.joints["base_footprint_ry_joint"].set_pos(relative_rpy[1], drive=False)
            self.joints["base_footprint_rz_joint"].set_pos(relative_rpy[2], drive=False)

        # Else, set the pose of the robot frame, and then move the joint frame of the world_base_joint to match it
        else:
            # Call the super() method to move the robot frame first
            super().set_position_orientation(position, orientation)
            # Move the joint frame for the world_base_joint
            if self._world_base_fixed_joint_prim is not None:
                self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(Gf.Quatf(*orientation[[3, 0, 1, 2]]))

    def set_linear_velocity(self, velocity: np.ndarray):
        # Transform the desired linear velocity from the world frame to the root_link ("base_footprint_x") frame
        # Note that this will also set the target to be the desired linear velocity (i.e. the robot will try to maintain
        # such velocity), which is different from the default behavior of set_linear_velocity for all other objects.
        orn = self.root_link.get_orientation()
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_x_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_y_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_z_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_linear_velocity(self) -> np.ndarray:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_linear_velocity()

    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        # See comments of self.set_linear_velocity
        orn = self.root_link.get_orientation()
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_rx_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_ry_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_rz_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_angular_velocity(self) -> np.ndarray:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_angular_velocity()
