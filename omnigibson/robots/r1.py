from functools import cached_property

import torch as th

from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.mobile_manipulation_robot import MobileManipulationRobot
from omnigibson.utils.python_utils import classproperty


class R1(HolonomicBaseRobot, ArticulatedTrunkRobot, MobileManipulationRobot):
    """
    R1 Robot
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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            scale=scale,
            visible=visible,
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
            **kwargs,
        )

    def _post_load(self):
        super()._post_load()

        # R1 and R1Pro's URDFs still use the mesh type for the collision meshes of the wheels (see the source URDFs)
        # as opposed to sphere primitives. As a result, even though import robot script changes to sphere approximation,
        # GeomPrim will change it back to convex hull approximation during post load. We need to manually set it back to sphere.
        # TODO: replace the mesh collision mesh with sphere primitives in the import robot script if use_sphere_wheels=True.
        for wheel_name in self.floor_touching_base_link_names:
            wheel_link = self.links[wheel_name]
            assert set(wheel_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
            wheel_link.collision_meshes["collisions"].set_collision_approximation("boundingSphere")

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

    @property
    def _raw_controller_order(self):
        controllers = ["base", "trunk"]
        for arm in self.arm_names:
            controllers += [f"arm_{arm}", f"gripper_{arm}"]
        return controllers

    @property
    def _default_controllers(self):
        controllers = super()._default_controllers
        # We use joint controllers for base as default
        controllers["base"] = "HolonomicBaseJointController"
        controllers["trunk"] = "JointController"
        # We use IK and multi finger gripper controllers as default
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"
        return controllers

    @property
    def tucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.05, 0.05])  # open gripper
        return pos

    @property
    def untucked_default_joint_pos(self):
        pos = th.zeros(self.n_dof)
        # Keep the current joint positions for the base joints
        pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = th.tensor([0.05, 0.05])  # open gripper
            pos[self.arm_control_idx[arm]] = th.tensor([0.0, 1.906, -0.991, 1.571, 0.915, -1.571])
        return pos

    @cached_property
    def floor_touching_base_link_names(self):
        return ["wheel_link1", "wheel_link2", "wheel_link3"]

    @cached_property
    def trunk_link_names(self):
        return ["torso_link1", "torso_link2", "torso_link3", "torso_link4"]

    @cached_property
    def trunk_joint_names(self):
        return [f"torso_joint{i}" for i in range(1, 5)]

    @classproperty
    def n_arms(cls):
        return 2

    @classproperty
    def arm_names(cls):
        return ["left", "right"]

    @cached_property
    def arm_link_names(self):
        return {arm: [f"{arm}_arm_link{i}" for i in range(1, 7)] for arm in self.arm_names}

    @cached_property
    def arm_joint_names(self):
        return {arm: [f"{arm}_arm_joint{i}" for i in range(1, 7)] for arm in self.arm_names}

    @cached_property
    def eef_link_names(self):
        return {arm: f"{arm}_eef_link" for arm in self.arm_names}

    @cached_property
    def finger_link_names(self):
        return {arm: [f"{arm}_gripper_link{i}" for i in range(1, 3)] for arm in self.arm_names}

    @cached_property
    def finger_joint_names(self):
        return {arm: [f"{arm}_gripper_axis{i}" for i in range(1, 3)] for arm in self.arm_names}

    @property
    def arm_workspace_range(self):
        return {arm: th.deg2rad(th.tensor([-45, 45], dtype=th.float32)) for arm in self.arm_names}

    @property
    def disabled_collision_pairs(self):
        # badly modeled gripper collision meshes
        return [
            ["left_gripper_link1", "left_gripper_link2"],
            ["right_gripper_link1", "right_gripper_link2"],
            ["base_link", "wheel_link1"],
            ["base_link", "wheel_link2"],
            ["base_link", "wheel_link3"],
            ["torso_link2", "torso_link4"],
        ]
