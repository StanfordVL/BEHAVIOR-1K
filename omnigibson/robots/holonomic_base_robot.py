from abc import abstractmethod
from functools import cached_property
from typing import Literal

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.utils.python_utils import classproperty

m = create_module_macros(module_path=__file__)
m.MAX_LINEAR_VELOCITY = 1.5  # linear velocity in meters/second
m.MAX_ANGULAR_VELOCITY = th.pi  # angular velocity in radians/second


class HolonomicBaseRobot(LocomotionRobot):
    """
    LocomotionRobot that is is equipped with holonomic base capabilities.
    Usually achived by having 6 1-DOF joints (3 for position and 3 for orientation) connected to the world.

    NOTE: controller_config should, at the minimum, contain:
        base: controller specifications for the controller to control this robot's base (locomotion).
            Should include:

            - name: Controller to create
            - <other kwargs> relevant to the controller being created. Note that all values will have default
                values specified, but setting these individual kwargs will override them

    """

    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
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
        obs_modalities=("rgb", "proprio"),
        proprio_obs="default",
        sensor_config=None,
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
            fixed_base (bool): whether to fix the base of this object or not
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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        self._world_base_fixed_joint_prim = None

        # Call super() method
        super().__init__(
            name=name,
            relative_prim_path=relative_prim_path,
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
            **kwargs,
        )

    def _post_load(self):
        super()._post_load()

        self._world_base_fixed_joint_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(
            f"{self.prim_path}/rootJoint"
        )
        position, orientation = self.get_position_orientation()
        # Set the world-to-base fixed joint to be at the robot's current pose
        self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
        self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
            lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
        )

    def _initialize(self):
        super()._initialize()

        for i, component in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            joint_name = f"base_footprint_{component}_joint"
            assert joint_name in self.joints, f"Missing base joint: {joint_name}"
            if i < 3:
                self.joints[joint_name].max_velocity = m.MAX_LINEAR_VELOCITY
            else:
                self.joints[joint_name].max_velocity = m.MAX_ANGULAR_VELOCITY

        # Force the recomputation of this cached property
        del self.control_limits

        # Reload the controllers to update command_output_limits, which read the updated control limits
        self.reload_controllers(self._controller_config)

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
    def base_joint_names(self):
        return [f"base_footprint_{component}_joint" for component in ("x", "y", "rz")]

    def reset(self):
        """
        Reset should not change the robot base pose.
        We need to cache and restore the base joints to the world.
        """
        base_joint_positions = self.get_joint_positions()[self.base_idx]
        super().reset()
        self.set_joint_positions(base_joint_positions, indices=self.base_idx)

    def get_position_orientation(self, frame: Literal["world", "scene"] = "world", clone=True):
        """
        Gets tiago's pose with respect to the specified frame.

        Args:
            frame (Literal): frame to get the pose with respect to. Default to world.
                scene frame gets position relative to the scene.
            clone (bool): Whether to clone the internal buffer or not when grabbing data

        Returns:
            2-tuple:
                - th.Tensor: (x,y,z) position in the specified frame
                - th.Tensor: (x,y,z,w) quaternion orientation in the specified frame
        """
        return self.base_footprint_link.get_position_orientation(frame=frame, clone=clone)

    def set_position_orientation(self, position=None, orientation=None, frame: Literal["world", "scene"] = "world"):
        """
        Sets tiago's pose with respect to the specified frame

        Args:
            position (None or 3-array): if specified, (x,y,z) position in the world frame
                Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the world frame.
                Default is None, which means left unchanged.
            frame (Literal): frame to set the pose with respect to, defaults to "world".
                scene frame sets position relative to the scene.
        """
        assert frame in ["world", "scene"], f"Invalid frame '{frame}'. Must be 'world' or 'scene'."

        # If no position or no orientation are given, get the current position and orientation of the object
        if position is None or orientation is None:
            current_position, current_orientation = self.get_position_orientation(frame=frame)
        position = current_position if position is None else position
        orientation = current_orientation if orientation is None else orientation

        # Convert to th.Tensor if necessary
        position = th.as_tensor(position, dtype=th.float32)
        orientation = th.as_tensor(orientation, dtype=th.float32)

        # Convert to from scene-relative to world if necessary
        if frame == "scene":
            assert self.scene is not None, "cannot set position and orientation relative to scene without a scene"
            position, orientation = self.scene.convert_scene_relative_pose_to_world(position, orientation)

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
        orn = self.root_link.get_position_orientation()[1]
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_x_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_y_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_z_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_linear_velocity(self) -> th.Tensor:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_linear_velocity()

    def set_angular_velocity(self, velocity: th.Tensor) -> None:
        # See comments of self.set_linear_velocity
        orn = self.root_link.get_position_orientation()[1]
        velocity_in_root_link = T.quat2mat(orn).T @ velocity
        self.joints["base_footprint_rx_joint"].set_vel(velocity_in_root_link[0], drive=False)
        self.joints["base_footprint_ry_joint"].set_vel(velocity_in_root_link[1], drive=False)
        self.joints["base_footprint_rz_joint"].set_vel(velocity_in_root_link[2], drive=False)

    def get_angular_velocity(self) -> th.Tensor:
        # Note that the link we are interested in is self.base_footprint_link, not self.root_link
        return self.base_footprint_link.get_angular_velocity()

    def _postprocess_control(self, control, control_type):
        # Run super method first
        u_vec, u_type_vec = super()._postprocess_control(control=control, control_type=control_type)

        # Change the control from base_footprint_link ("base_footprint") frame to root_link ("base_footprint_x") frame
        base_orn = self.base_footprint_link.get_position_orientation()[1]
        root_link_orn = self.root_link.get_position_orientation()[1]

        cur_orn = T.mat2quat(T.quat2mat(root_link_orn).T @ T.quat2mat(base_orn))

        # Rotate the linear and angular velocity to the desired frame
        lin_vel_global, _ = T.pose_transform(th.zeros(3), cur_orn, u_vec[self.base_idx[:3]], th.tensor([0, 0, 0, 1]))
        ang_vel_global, _ = T.pose_transform(th.zeros(3), cur_orn, u_vec[self.base_idx[3:]], th.tensor([0, 0, 0, 1]))

        u_vec[self.base_control_idx] = th.tensor([lin_vel_global[0], lin_vel_global[1], ang_vel_global[2]])

        return u_vec, u_type_vec

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        action = ManipulationRobot.teleop_data_to_action(self, teleop_action)
        action[self.base_action_idx] = th.tensor(teleop_action.base).float() * 0.1
        return action

    @property
    def base_footprint_link_name(self):
        raise NotImplementedError("base_footprint_link_name is not implemented for HolonomicBaseRobot")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("HolonomicBaseRobot")
        return classes
