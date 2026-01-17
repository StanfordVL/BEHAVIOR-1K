import math
import os
from functools import cached_property

import torch as th

from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.transform_utils import euler2quat


class Yam(ManipulationRobot):
    """
    The Yam 6DOF robot arm with a parallel gripper
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
        fixed_base=True,
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
        finger_static_friction=None,
        finger_dynamic_friction=None,
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
                Set this value instead if you want to initialize the robot with a different reset joint position.
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
            finger_static_friction (None or float): If specified, specific static friction to use for robot's fingers
            finger_dynamic_friction (None or float): If specified, specific dynamic friction to use for robot's fingers.
                Note: If specified, this will override any ways that are found within @link_physics_materials for any
                robot finger gripper links
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
            grasping_direction="lower",
            finger_static_friction=finger_static_friction,
            finger_dynamic_friction=finger_dynamic_friction,
            **kwargs,
        )

        # Add callback
        import omnigibson as og
        og.sim.add_callback_on_play(name=f"{self.name}_set_gains", callback=self._set_gains)

    def _set_limits(self):
        effort_limits = th.tensor([28.0, 28.0, 28.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        for i, joint in enumerate(self.joints.values()):
            joint.max_effort = effort_limits[i]

    def _set_gains(self):
        kp_limits = th.tensor([40.0, 40.0, 40.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        kv_limits = th.tensor([2.5, 2.5, 2.5, 1.0, 1.0, 1.0, 10.0, 10.0])
        for i, joint in enumerate(self.joints.values()):
            joint.stiffness = kp_limits[i]
            joint.damping = kv_limits[i]

    def _initialize(self):
        out = super()._initialize()
        self._set_limits()
        self.links["link_6"].visible = True
        return out

    @property
    def discrete_action_list(self):
        raise NotImplementedError()

    def _create_discrete_action_space(self):
        raise ValueError("Yam does not support discrete actions!")

    @property
    def _raw_controller_order(self):
        return [f"arm_{self.default_arm}", f"gripper_{self.default_arm}"]

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        controllers[f"arm_{self.default_arm}"] = "InverseKinematicsController"
        controllers[f"gripper_{self.default_arm}"] = "MultiFingerGripperController"
        return controllers

    @property
    def _default_joint_pos(self):
        # Default joint positions: 6 arm joints + 2 finger joints
        return th.tensor([0.0, 1.047, 1.047, 0.0, 0.0, 0.0, 0.0, 0.0])

    @cached_property
    def arm_link_names(self):
        return {
            self.default_arm: [
                "arm",
                "link_1",
                "link_2",
                "link_3",
                "link_4",
                "link_5",
                "link_6",
            ]
        }

    @cached_property
    def arm_joint_names(self):
        return {
            self.default_arm: [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
            ]
        }

    @cached_property
    def eef_link_names(self):
        return {self.default_arm: "link_6"}

    @cached_property
    def finger_link_names(self):
        return {self.default_arm: ["left_link_finger", "right_link_finger"]}

    @cached_property
    def finger_joint_names(self):
        return {self.default_arm: ["left_finger", "right_finger"]}

    @property
    def usd_path(self):
        return os.path.join(
            get_dataset_path("omnigibson-robot-assets"),
            "models/yam/usd/yam.usda",
        )

    @property
    def teleop_rotation_offset(self):
        return {self.default_arm: euler2quat([-math.pi, 0, 0])}

    @property
    def _assisted_grasp_start_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="left_link_finger", position=th.tensor([0.0, 0.0, 0.0])),
            ]
        }

    @property
    def _assisted_grasp_end_points(self):
        return {
            self.default_arm: [
                GraspingPoint(link_name="right_link_finger", position=th.tensor([0.0, 0.0, 0.0])),
            ]
        }

