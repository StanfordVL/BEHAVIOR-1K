import os

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import ControlType
from omnigibson.macros import create_module_macros, gm
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

m = create_module_macros(module_path=__file__)
m.MAX_LINEAR_VELOCITY = 1.5  # linear velocity in meters/second
m.MAX_ANGULAR_VELOCITY = th.pi  # angular velocity in radians/second


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
        # Other args that will be created at runtime
        self._world_base_fixed_joint_prim = None

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

        self._world_base_fixed_joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(
            f"{self.prim_path}/rootJoint"
        )
        position, orientation = self.get_position_orientation()
        # Set the world-to-base fixed joint to be at the robot's current pose
        self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
        self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
            lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
        )

    # Name of the actual root link that we are interested in. Note that this is different from self.root_link_name,
    # which is "base_footprint_x", corresponding to the first of the 6 1DoF joints to control the base.
    @property
    def base_footprint_link_name(self):
        return "base_link"

    def _postprocess_control(self, control, control_type):
        # Run super method first
        u_vec, u_type_vec = super()._postprocess_control(control=control, control_type=control_type)

        # Override trunk value if we're keeping the trunk rigid
        if self.rigid_trunk:
            u_vec[self.trunk_control_idx] = self._default_joint_pos[self.trunk_control_idx]
            u_type_vec[self.trunk_control_idx] = ControlType.POSITION

        # Change the control from base_footprint_link ("base_footprint") frame to root_link ("base_footprint_x") frame
        base_orn = self.base_footprint_link.get_orientation()
        root_link_orn = self.root_link.get_orientation()

        cur_orn = T.mat2quat(T.quat2mat(root_link_orn).T @ T.quat2mat(base_orn))

        # Rotate the linear and angular velocity to the desired frame
        lin_vel_global, _ = T.pose_transform(th.zeros(3), cur_orn, u_vec[self.base_idx[:3]], th.tensor([0, 0, 0, 1]))
        ang_vel_global, _ = T.pose_transform(th.zeros(3), cur_orn, u_vec[self.base_idx[3:]], th.tensor([0, 0, 0, 1]))

        u_vec[self.base_control_idx] = th.tensor([lin_vel_global[0], lin_vel_global[1], ang_vel_global[2]])
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
        fcns = super().get_control_dict()
        native_fcn = fcns.get_fcn("eef_right_pos_relative")
        fcns["eef_right_pos_relative"] = lambda: (
            native_fcn() + th.tensor([0, 0, -self.get_joint_positions()[self.trunk_control_idx[0]]])
        )

        return fcns

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
        controllers["base"] = "JointController"
        # We use IK and multi finger gripper controllers as default
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
            "use_impedances": False,
            "motor_type": "velocity",
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

    @property
    def _default_joint_pos(self):
        return th.zeros(len(self.joints))

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
    def base_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to the six 1DoF base joints
        """
        joints = list(self.joints.keys())
        return th.tensor(
            [joints.index(f"base_footprint_{component}_joint") for component in ["x", "y", "z", "rx", "ry", "rz"]]
        )

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
    def base_joint_names(self):
        return [f"base_footprint_{component}_joint" for component in ("x", "y", "rz")]

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

    def get_position_orientation(self):
        # TODO: Investigate the need for this custom behavior.
        return self.base_footprint_link.get_position_orientation()

    def set_position_orientation(self, position=None, orientation=None):
        current_position, current_orientation = self.get_position_orientation()
        if position is None:
            position = current_position
        if orientation is None:
            orientation = current_orientation
        position, orientation = th.tensor(position), th.tensor(orientation)
        assert th.isclose(
            th.linalg.norm(orientation), 1, atol=1e-3
        ), f"{self.name} desired orientation {orientation} is not a unit quaternion."

        # TODO: Reconsider the need for this. Why can't these behaviors be unified? Does the joint really need to move?
        # If the simulator is playing, set the 6 base joints to achieve the desired pose of base_footprint link frame
        if og.sim.is_playing() and self.initialized:
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
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
                    lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
                )

    def set_linear_velocity(self, velocity: th.Tensor):
        # Transform the desired linear velocity from the world frame to the root_link ("base_footprint_x") frame
        # Note that this will also set the target to be the desired linear velocity (i.e. the robot will try to maintain
        # such velocity), which is different from the default behavior of set_linear_velocity for all other objects.
        orn = self.root_link.get_orientation()
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_x_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_y_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_z_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_linear_velocity(self) -> th.Tensor:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_linear_velocity()

    def set_angular_velocity(self, velocity: th.Tensor) -> None:
        # See comments of self.set_linear_velocity
        orn = self.root_link.get_orientation()
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_rx_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_ry_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_rz_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_angular_velocity(self) -> th.Tensor:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_angular_velocity()

    @property
    def disabled_collision_pairs(self):
        # badly modeled gripper collision meshes
        return [
            ["left_gripper_link1", "left_gripper_link2"],
            ["right_gripper_link1", "right_gripper_link2"],
        ]

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        action = ManipulationRobot.teleop_data_to_action(self, teleop_action)
        action[self.base_action_idx] = teleop_action.base * 0.1
        return action
