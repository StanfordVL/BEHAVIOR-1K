import os
from copy import deepcopy
import math
import torch as th
from functools import cached_property
from typing import Literal
from abc import abstractmethod

import gymnasium as gym
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.objects.usd_object import USDObject
from omnigibson.sensors import (
    ALL_SENSOR_MODALITIES,
    SENSOR_PRIMS_TO_SENSOR_CLS,
    ScanSensor,
    VisionSensor,
    create_sensor,
)
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.constants import PrimType
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty, merge_nested_dicts, CachedFunctions, assert_valid_key
from omnigibson.utils.usd_utils import ControllableObjectViewAPI, absolute_prim_path_to_scene_relative
from omnigibson.utils.vision_utils import segmentation_to_rgb, change_pcd_frame
from omnigibson.controllers import create_controller
from omnigibson.controllers.controller_base import ControlType
from omnigibson.controllers.joint_controller import JointController
from omnigibson.objects.object_base import BaseObject
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)

# Global dicts that will contain mappings
REGISTERED_ROBOTS = dict()

# Add proprio sensor modality to ALL_SENSOR_MODALITIES
ALL_SENSOR_MODALITIES.add("proprio")

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Name of the category to assign to all robots
m.ROBOT_CATEGORY = "agent"


class Robot(USDObject, BaseObject, GymObservable):
    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        relative_prim_path=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=True,
        link_physics_materials=None,
        load_config=None,
        # Unique to USDObject hierarchy
        abilities=None,
        # Unique to Robot
        # for control
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        # for robot base
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
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
                Set this value instead if you want to initialize the object with a different rese joint position.
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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store control-related inputs
        self._control_freq = control_freq
        self._controller_config = controller_config
        if reset_joint_pos is None:
            self._reset_joint_pos = None
        elif isinstance(reset_joint_pos, th.Tensor):
            self._reset_joint_pos = reset_joint_pos
        else:
            self._reset_joint_pos = th.tensor(reset_joint_pos, dtype=th.float)

        # Make sure action type is valid, and also save
        assert_valid_key(key=action_type, valid_keys={"discrete", "continuous"}, name="action type")
        self._action_type = action_type
        self._action_normalize = action_normalize

        # Store internal placeholders that will be filled in later (from ControllableObject)
        self._dof_to_joints = None  # dict that will map DOF indices to JointPrims
        self._last_action = None
        self._controllers = None
        self.dof_names_ordered = None
        self._control_enabled = True

        # Store robot-specific inputs
        self._obs_modalities = (
            obs_modalities
            if obs_modalities == "all"
            else {obs_modalities}
            if isinstance(obs_modalities, str)
            else set(obs_modalities)
        )  # this will get updated later when we fill in our sensors
        self._proprio_obs = self.default_proprio_obs if proprio_obs == "default" else list(proprio_obs)
        self._sensor_config = sensor_config

        # Process abilities
        robot_abilities = {"robot": {}}
        abilities = robot_abilities if abilities is None else robot_abilities.update(abilities)

        # Initialize internal attributes that will be loaded later
        self._include_sensor_names = None if include_sensor_names is None else set(include_sensor_names)
        self._exclude_sensor_names = None if exclude_sensor_names is None else set(exclude_sensor_names)
        self._sensors = None  # e.g.: scan sensor, vision sensor

        # All BaseRobots should have xform properties pre-loaded
        load_config = {} if load_config is None else load_config
        load_config["xform_props_pre_loaded"] = True

        class_name = self.__class__.__name__.lower()
        if relative_prim_path:
            # If prim path is specified, assert that the last element starts with the right prefix to ensure that
            # the object will be included in the ControllableObjectViewAPI.
            assert relative_prim_path.split("/")[-1].startswith(f"controllable__{class_name}__"), (
                "If relative_prim_path is specified, the last element of the path must look like "
                f"'controllable__{class_name}__robotname' where robotname can be an arbitrary "
                "string containing no double underscores."
            )
            assert relative_prim_path.split("/")[-1].count("__") == 2, (
                "If relative_prim_path is specified, the last element of the path must look like "
                f"'controllable__{class_name}__robotname' where robotname can be an arbitrary "
                "string containing no double underscores."
            )
        else:
            # If prim path is not specified, set it to the default path, but prepend controllable.
            relative_prim_path = f"/controllable__{class_name}__{name}"

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            usd_path=self.usd_path,
            name=name,
            category=m.ROBOT_CATEGORY,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=PrimType.RIGID,
            include_default_states=True,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

        assert not isinstance(self._load_config["scale"], th.Tensor) or th.all(
            self._load_config["scale"] == self._load_config["scale"][0]
        ), f"Robot scale must be uniform! Got: {self._load_config['scale']}"

    def load(self, scene):
        # Run super first
        prim = super().load(scene)

        # Set the control frequency if one was not provided.
        expected_control_freq = 1.0 / og.sim.get_sim_step_dt()
        if self._control_freq is None:
            log.info(
                "Control frequency is None - being set to default of render_frequency: %.4f", expected_control_freq
            )
            self._control_freq = expected_control_freq
        else:
            assert math.isclose(
                expected_control_freq, self._control_freq
            ), "Stored control frequency does not match environment's render timestep."

        return prim
    
    def _post_load(self):
        # Run super post load first
        super()._post_load()

        # For controllable objects, we disable gravity of all links that are not fixed to the base link.
        # This is because we cannot accurately apply gravity compensation in the absence of a working
        # generalized gravity force computation. This may have some side effects on the measured
        # torque on each of these links, but it provides a greatly improved joint control behavior.
        # Note that we do NOT disable gravity for links that are fixed to the base link, as these links
        # are typically where most of the downward force on the robot is applied. Disabling gravity
        # for these links would result in the robot floating in the air easily. Also note that here
        # we use the base link footprint which takes into account the presence of virtual joints.
        fixed_link_names = self.get_fixed_link_names_in_subtree(self.base_footprint_link_name)

        # Find the links that are NOT fixed.
        other_link_names = set(self.links.keys()) - fixed_link_names

        # Disable gravity for those links.
        for link_name in other_link_names:
            self.links[link_name].disable_gravity()

        # Load the sensors
        self._load_sensors()

        # TODO: move into config
        # special case handling
        if self.model_name.lower() in ["r1", "r1pro"]:
            # R1 and R1Pro's URDFs still use the mesh type for the collision meshes of the wheels
            # We need to manually set it back to sphere approximation
            for wheel_name in self.floor_touching_base_link_names:
                wheel_link = self.links[wheel_name]
                assert set(wheel_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
                wheel_link.collision_meshes["collisions"].set_collision_approximation("boundingSphere")
        elif self.model_name.lower() == "tiago":
            # The eef gripper links should be visual-only. They only contain a "ghost" box volume
            # for detecting objects inside the gripper, in order to activate attachments (AG for Cloths).
            for arm in self.arm_names:
                self.eef_links[arm].visual_only = True
                self.eef_links[arm].visible = False
    def _load_controllers(self):
        """
        Loads controller(s) to map inputted actions into executable (pos, vel, and / or effort) signals on this object.
        Stores created controllers as dictionary mapping controller names to specific controller
        instances used by this object.
        """
        # Generate the controller config
        self._controller_config = self._generate_controller_config(custom_config=self._controller_config)

        # We copy the controller config here because we add/remove some keys in-place that shouldn't persist
        _controller_config = deepcopy(self._controller_config)

        # Store dof idx mapping to dof name
        self.dof_names_ordered = list(self._joints.keys())

        # Initialize controllers to create
        self._controllers = dict()
        # Keep track of any controllers that are subsumed by other controllers
        # We will not instantiate subsumed controllers
        controller_subsumes = dict()  # Maps independent controller name to list of subsumed controllers
        subsume_names = set()
        for name in self._raw_controller_order:
            # Make sure we have the valid controller name specified
            assert_valid_key(key=name, valid_keys=_controller_config, name="controller name")
            cfg = _controller_config[name]
            subsume_controllers = cfg.pop("subsume_controllers", [])
            # If this controller subsumes other controllers, it cannot be subsumed by another controller
            # (i.e.: we don't allow nested / cyclical subsuming)
            if len(subsume_controllers) > 0:
                assert (
                    name not in subsume_names
                ), f"Controller {name} subsumes other controllers, and therefore cannot be subsumed by another controller!"
                controller_subsumes[name] = subsume_controllers
                for subsume_name in subsume_controllers:
                    # Make sure it doesn't already exist -- a controller should only be subsumed by up to one other
                    assert (
                        subsume_name not in subsume_names
                    ), f"Controller {subsume_name} cannot be subsumed by more than one other controller!"
                    assert (
                        subsume_name not in controller_subsumes
                    ), f"Controller {name} subsumes other controllers, and therefore cannot be subsumed by another controller!"
                    subsume_names.add(subsume_name)

        # Loop over all controllers, in the order corresponding to @action dim
        for name in self._raw_controller_order:
            # If this controller is subsumed by another controller, simply skip it
            if name in subsume_names:
                continue
            cfg = _controller_config[name]
            # If we subsume other controllers, prepend the subsumed' dof idxs to this controller's idxs
            if name in controller_subsumes:
                for subsumed_name in controller_subsumes[name]:
                    subsumed_cfg = _controller_config[subsumed_name]
                    cfg["dof_idx"] = th.concatenate([subsumed_cfg["dof_idx"], cfg["dof_idx"]])
            # If we're using normalized action space, override the inputs for all controllers
            if self._action_normalize:
                cfg["command_input_limits"] = "default"  # default is normalized (-1, 1)

            # Create the controller
            controller = create_controller(**cb.from_torch_recursive(cfg))
            # Verify the controller's DOFs can all be driven
            for idx in controller.dof_idx:
                assert self._joints[
                    self.dof_names_ordered[idx]
                ].driven, "Controllers should only control driveable joints!"
            self._controllers[name] = controller
        self.update_controller_mode()
    
    def update_controller_mode(self):
        """
        Helper function to force the joints to use the internal specified control mode and gains
        """
        # Update the control modes of each joint based on the outputted control from the controllers
        unused_dofs = {i for i in range(self.n_dof)}
        for controller in self._controllers.values():
            for i, dof in enumerate(controller.dof_idx):
                # Make sure the DOF has not already been set yet, and remove it afterwards
                assert dof.item() in unused_dofs
                unused_dofs.remove(dof.item())
                control_type = controller.control_type
                dof_joint = self._joints[self.dof_names_ordered[dof]]
                dof_joint.set_control_type(
                    control_type=control_type,
                    kp=None if controller.isaac_kp is None or dof_joint.is_mimic_joint else controller.isaac_kp[i],
                    kd=None if controller.isaac_kd is None or dof_joint.is_mimic_joint else controller.isaac_kd[i],
                )

        # For all remaining DOFs not controlled, we assume these are free DOFs (e.g.: virtual joints representing free
        # motion wrt a specific axis), so explicitly set kp / kd to 0 to avoid silent bugs when
        # joint positions / velocities are set
        for unused_dof in unused_dofs:
            unused_joint = self._joints[self.dof_names_ordered[unused_dof]]
            assert not unused_joint.driven, (
                f"All unused joints not mapped to any controller should not have DriveAPI attached to it! "
                f"However, joint {unused_joint.name} is driven!"
            )
            unused_joint.set_control_type(
                control_type=ControlType.NONE,
                kp=None,
                kd=None,
            )
    
    def _generate_controller_config(self, custom_config=None):
        """
        Generates a fully-populated controller config, overriding any default values with the corresponding values
        specified in @custom_config

        Args:
            custom_config (None or Dict[str, ...]): nested dictionary mapping controller name(s) to specific custom
                controller configurations for this object. This will override any default values specified by this class

        Returns:
            dict: Fully-populated nested dictionary mapping controller name(s) to specific controller configurations for
                this object
        """
        controller_config = {} if custom_config is None else deepcopy(custom_config)

        # Update the configs
        for group in self._raw_controller_order:
            group_controller_name = (
                controller_config[group]["name"]
                if group in controller_config and "name" in controller_config[group]
                else self._default_controllers[group]
            )
            controller_config[group] = merge_nested_dicts(
                base_dict=self._default_controller_config[group][group_controller_name],
                extra_dict=controller_config.get(group, {}),
            )

        return controller_config
    
    def reload_controllers(self, controller_config=None):
        """
        Reloads controllers based on the specified new @controller_config

        Args:
            controller_config (None or Dict[str, ...]): nested dictionary mapping controller name(s) to specific
                controller configurations for this object. This will override any default values specified by this class.
        """
        self._controller_config = {} if controller_config is None else controller_config

        # (Re-)load controllers
        self._load_controllers()

        # (Re-)create the action space
        self._action_space = (
            self._create_discrete_action_space()
            if self._action_type == "discrete"
            else self._create_continuous_action_space()
        )
    
    def reset(self):
        # Call super first
        super().reset()

        # Override the reset joint state based on reset values
        self.set_joint_positions(positions=self._reset_joint_pos, drive=False)

    def _create_discrete_action_space(self):
        """
        Create a discrete action space for this object.
        By default, subclass does not support this type of action space.
        If otherwise, should be implemented by the subclass.

        Returns:
            gym.space: Object-specific discrete action space
        """
        raise ValueError("Does not support discrete actions!")

    def _create_continuous_action_space(self):
        """
        Create a continuous action space for this object. By default, this loops over all controllers and
        appends their respective input command limits to set the action space.
        Any custom behavior should be implemented by the subclass (e.g.: if a subclass does not
        support this type of action space, it should raise an error).

        Returns:
            gym.space.Box: Object-specific continuous action space
        """
        # Action space is ordered according to the order in _default_controller_config control
        low, high = [], []
        for controller in self._controllers.values():
            limits = controller.command_input_limits
            low.append(th.tensor([-float("inf")] * controller.command_dim) if limits is None else limits[0])
            high.append(th.tensor([float("inf")] * controller.command_dim) if limits is None else limits[1])

        return gym.spaces.Box(
            shape=(self.action_dim,),
            low=cb.to_numpy(cb.cat(low)),
            high=cb.to_numpy(cb.cat(high)),
            dtype=NumpyTypes.FLOAT32,
        )

    def apply_action(self, action):
        """
        Converts inputted actions into low-level control signals

        NOTE: This does NOT deploy control on the object. Use self.step() instead.

        Args:
            action (n-array): n-DOF length array of actions to apply to this object's internal controllers
        """
        # Store last action as the current action being applied
        self._last_action = action

        # If we're using discrete action space, we grab the specific action and use that to convert to control
        if self._action_type == "discrete":
            action = th.tensor(self.discrete_action_list[action], dtype=th.float32)

        # Sanity check that action is 1D array
        assert len(action.shape) == 1, f"Action must be 1D array, got {len(action.shape)}D array!"

        # Sanity check that action is 1D array
        assert len(action.shape) == 1, f"Action must be 1D array, got {len(action.shape)}D array!"

        # Check if the input action's length matches the action dimension
        assert len(action) == self.action_dim, "Action must be dimension {}, got dim {} instead.".format(
            self.action_dim, len(action)
        )

        # Convert action from torch if necessary
        action = cb.from_torch(action)

        # First, loop over all controllers, and update the desired command
        idx = 0

        for name, controller in self._controllers.items():
            # Set command, then take a controller step
            controller.update_goal(
                command=action[idx : idx + controller.command_dim], control_dict=self.get_control_dict()
            )
            # Update idx
            idx += controller.command_dim

    @property
    def is_driven(self) -> bool:
        """
        Returns:
            bool: Whether this object is actively controlled/driven or not
        """
        return True

    @property
    def control_enabled(self):
        return self._control_enabled

    @control_enabled.setter
    def control_enabled(self, value):
        self._control_enabled = value

    def step(self):
        """
        Takes a controller step across all controllers and deploys the computed control signals onto the object.
        """
        # Skip if we don't have control enabled
        if not self.control_enabled:
            return

        # Skip this step if our articulation view is not valid
        if self._articulation_view_direct is None or not self._articulation_view_direct.initialized:
            return

        # First, loop over all controllers, and calculate the computed control
        control = dict()
        idx = 0

        # Compose control_dict
        control_dict = self.get_control_dict()

        for name, controller in self._controllers.items():
            control[name] = {
                "value": controller.step(control_dict=control_dict),
                "type": controller.control_type,
            }
            # Update idx
            idx += controller.command_dim

        # Compose controls
        u_vec = cb.zeros(self.n_dof)
        # By default, the control type is Effort and the control value is 0 (th.zeros) - i.e. no control applied
        u_type_vec = cb.array([ControlType.EFFORT] * self.n_dof)
        for group, ctrl in control.items():
            idx = self._controllers[group].dof_idx
            u_vec[idx] = ctrl["value"]
            u_type_vec[idx] = ctrl["type"]

        u_vec, u_type_vec = self._postprocess_control(control=u_vec, control_type=u_type_vec)

        # Deploy control signals
        self.deploy_control(control=u_vec, control_type=u_type_vec)

    def _postprocess_control(self, control, control_type):
        """
        Runs any postprocessing on @control with corresponding @control_type on this entity. Default is no-op.
        Deploys control signals @control with corresponding @control_type on this entity.

        Args:
            control (k- or n-array): control signals to deploy. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @control must
                be the same length as @indices!
            control_type (k- or n-array): control types for each DOF. Each entry should be one of ControlType.
                 This should be n-DOF length if all joints are being set, or k-length (k < n) if specific
                 indices are being set. In this case, the length of @control must be the same length as @indices!

        Returns:
            2-tuple:
                - n-array: raw control signals to send to the object's joints
                - list: control types for each joint
        """
        return control, control_type

    def deploy_control(self, control, control_type):
        """
        Deploys control signals @control with corresponding @control_type on this entity.

        Note: This is DIFFERENT than self.set_joint_positions/velocities/efforts, because in this case we are only
            setting target values (i.e.: we subject this entity to physical dynamics in order to reach the desired
            @control setpoints), compared to set_joint_XXXX which manually sets the actual state of the joints.

            This function is intended to be used with motorized entities, e.g.: robot agents or machines (e.g.: a
            conveyor belt) to simulation physical control of these entities.

            In contrast, use set_joint_XXXX for simulation-specific logic, such as simulator resetting or "magic"
            action implementations.

        Args:
            control (n-array): control signals to deploy. This should be n-DOF length for all joints being set.
            control_type (n-array): control types for each DOF. Each entry should be one of ControlType.
                 This should be n-DOF length for all joints being set.
        """
        # Run sanity check
        assert len(control) == len(control_type) == self.n_dof, (
            f"Control signals, control types, and number of DOF should all be the same!"
            f"Got {len(control)}, {len(control_type)}, and {self.n_dof} respectively."
        )

        # set the targets for joints
        pos_idxs = cb.where(control_type == ControlType.POSITION)[0]
        if len(pos_idxs) > 0:
            ControllableObjectViewAPI.set_joint_position_targets(
                self.articulation_root_path,
                positions=control[pos_idxs],
                indices=pos_idxs,
            )
            # If we're setting joint position targets, we should also set velocity targets to 0
            ControllableObjectViewAPI.set_joint_velocity_targets(
                self.articulation_root_path,
                velocities=cb.zeros(len(pos_idxs)),
                indices=pos_idxs,
            )
        vel_idxs = cb.where(control_type == ControlType.VELOCITY)[0]
        if len(vel_idxs) > 0:
            ControllableObjectViewAPI.set_joint_velocity_targets(
                self.articulation_root_path,
                velocities=control[vel_idxs],
                indices=vel_idxs,
            )
        eff_idxs = cb.where(control_type == ControlType.EFFORT)[0]
        if len(eff_idxs) > 0:
            ControllableObjectViewAPI.set_joint_efforts(
                self.articulation_root_path,
                efforts=control[eff_idxs],
                indices=eff_idxs,
            )

    def get_control_dict(self):
        """
        Grabs all relevant information that should be passed to each controller during each controller step. This
        automatically caches information

        Returns:
            CachedFunctions: Keyword-mapped control values for this object, mapping names to n-arrays.
                By default, returns the following (can be queried via [] or get()):

                - joint_position: (n_dof,) joint positions
                - joint_velocity: (n_dof,) joint velocities
                - joint_effort: (n_dof,) joint efforts
                - root_pos: (3,) (x,y,z) global cartesian position of the object's root link
                - root_quat: (4,) (x,y,z,w) global cartesian orientation of ths object's root link
                - mass_matrix: (n_dof, n_dof) mass matrix
                - gravity_force: (n_dof,) per-joint generalized gravity forces
                - cc_force: (n_dof,) per-joint centripetal and centrifugal forces
        """
        # Note that everything here uses the ControllableObjectViewAPI because these are faster implementations of
        # the functions that this class also implements. The API centralizes access for all of the robots in the scene
        # removing the need for multiple reads and writes.
        # TODO(cgokmen): CachedFunctions can now be entirely removed since the ControllableObjectViewAPI already implements caching.
        fcns = CachedFunctions()
        fcns["_root_pos_quat"] = lambda: ControllableObjectViewAPI.get_position_orientation(self.articulation_root_path)
        fcns["root_pos"] = lambda: fcns["_root_pos_quat"][0]
        fcns["root_quat"] = lambda: fcns["_root_pos_quat"][1]

        # NOTE: We explicitly compute hand-calculated (i.e.: non-Isaac native) values for velocity because
        # Isaac has some numerical inconsistencies for low velocity values, which cause downstream issues for
        # controllers when computing accurate control. This is why we explicitly set the `estimate=True` flag here,
        # which is not used anywhere else in the codebase
        fcns["root_lin_vel"] = lambda: ControllableObjectViewAPI.get_linear_velocity(
            self.articulation_root_path, estimate=True
        )
        fcns["root_ang_vel"] = lambda: ControllableObjectViewAPI.get_angular_velocity(
            self.articulation_root_path, estimate=True
        )
        fcns["root_rel_lin_vel"] = lambda: ControllableObjectViewAPI.get_relative_linear_velocity(
            self.articulation_root_path,
            estimate=True,
        )
        fcns["root_rel_ang_vel"] = lambda: ControllableObjectViewAPI.get_relative_angular_velocity(
            self.articulation_root_path,
            estimate=True,
        )
        fcns["joint_position"] = lambda: ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path)
        fcns["joint_velocity"] = lambda: ControllableObjectViewAPI.get_joint_velocities(
            self.articulation_root_path, estimate=True
        )
        fcns["joint_effort"] = lambda: ControllableObjectViewAPI.get_joint_efforts(self.articulation_root_path)
        # Similar to the jacobians, there may be an additional 6 entries at the beginning of the mass matrix, if this robot does
        # not have a fixed base (i.e.: the 6DOF --> "floating" joint)
        fcns["mass_matrix"] = lambda: (
            ControllableObjectViewAPI.get_generalized_mass_matrices(self.articulation_root_path)
            if self.fixed_base
            else ControllableObjectViewAPI.get_generalized_mass_matrices(self.articulation_root_path)[6:, 6:]
        )
        fcns["gravity_force"] = lambda: ControllableObjectViewAPI.get_gravity_compensation_forces(
            self.articulation_root_path
        )
        fcns["cc_force"] = lambda: ControllableObjectViewAPI.get_coriolis_and_centrifugal_compensation_forces(
            self.articulation_root_path
        )

        return fcns

    def _add_task_frame_control_dict(self, fcns, task_name, link_name):
        """
        Internally helper function to generate per-link control dictionary entries. Useful for generating relevant
        control values needed for IK / OSC for a given @task_name. Should be called within @get_control_dict()

        Args:
            fcns (CachedFunctions): Keyword-mapped control values for this object, mapping names to n-arrays.
            task_name (str): name to assign for this task_frame. It will be prepended to all fcns generated
            link_name (str): the corresponding link name from this controllable object that @task_name is referencing
        """
        fcns[f"_{task_name}_pos_quat_relative"] = (
            lambda: ControllableObjectViewAPI.get_link_relative_position_orientation(
                self.articulation_root_path, link_name
            )
        )
        fcns[f"{task_name}_pos_relative"] = lambda: fcns[f"_{task_name}_pos_quat_relative"][0]
        fcns[f"{task_name}_quat_relative"] = lambda: fcns[f"_{task_name}_pos_quat_relative"][1]

        # NOTE: We explicitly compute hand-calculated (i.e.: non-Isaac native) values for velocity because
        # Isaac has some numerical inconsistencies for low velocity values, which cause downstream issues for
        # controllers when computing accurate control. This is why we explicitly set the `estimate=True` flag here,
        # which is not used anywhere else in the codebase
        fcns[f"{task_name}_lin_vel_relative"] = lambda: ControllableObjectViewAPI.get_link_relative_linear_velocity(
            self.articulation_root_path,
            link_name,
            estimate=True,
        )
        fcns[f"{task_name}_ang_vel_relative"] = lambda: ControllableObjectViewAPI.get_link_relative_angular_velocity(
            self.articulation_root_path,
            link_name,
            estimate=True,
        )
        # -n_joints because there may be an additional 6 entries at the beginning of the array, if this robot does
        # not have a fixed base (i.e.: the 6DOF --> "floating" joint)
        # see self.get_relative_jacobian() for more info
        # We also count backwards for the link frame because if the robot is fixed base, the jacobian returned has one
        # less index than the number of links. This is presumably because the 1st link of a fixed base robot will
        # always have a zero jacobian since it can't move. Counting backwards resolves this issue.
        start_idx = 0 if self.fixed_base else 6
        link_idx = self._articulation_view.get_body_index(link_name)
        fcns[f"{task_name}_jacobian_relative"] = lambda: ControllableObjectViewAPI.get_relative_jacobian(
            self.articulation_root_path
        )[-(self.n_links - link_idx), :, start_idx : start_idx + self.n_joints]

    def q_to_action(self, q):
        """
        Converts a target joint configuration to an action that can be applied to this object.
        All controllers should be JointController with use_delta_commands=False
        """
        action = []
        for name, controller in self.controllers.items():
            assert (
                isinstance(controller, JointController) and not controller.use_delta_commands
            ), f"Controller [{name}] should be a JointController with use_delta_commands=False!"
            command = q[controller.dof_idx]
            action.append(controller._reverse_preprocess_command(command))
        action = th.cat(action, dim=0)
        assert (
            action.shape[0] == self.action_dim
        ), f"Action should have dimension {self.action_dim}, got {action.shape[0]}"
        return action

    def dump_action(self):
        """
        Dump the last action applied to this object. For use in demo collection.
        """
        return self._last_action

    def set_position_orientation(self, position=None, orientation=None, frame: Literal["world", "scene"] = "world"):
        # Run super first
        super().set_position_orientation(position, orientation, frame)

        # Clear the controllable view's backend since state has changed
        ControllableObjectViewAPI.clear_object(prim_path=self.articulation_root_path)

    def set_joint_positions(self, positions, indices=None, normalized=False, drive=False):
        # Call super first
        super().set_joint_positions(positions=positions, indices=indices, normalized=normalized, drive=drive)

        # If we're not driving the joints, reset the controllers so that the goals are updated wrt to the new state
        # Also clear the controllable view's backend since state has changed
        if not drive:
            ControllableObjectViewAPI.clear_object(prim_path=self.articulation_root_path)
            for controller in self._controllers.values():
                controller.reset()

    def _dump_state(self):
        # Grab super state
        state = super()._dump_state()

        # Add in controller states
        controller_states = dict()
        for controller_name, controller in self._controllers.items():
            controller_states[controller_name] = controller.dump_state()

        state["controllers"] = controller_states

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load controller states
        controller_states = state["controllers"]
        for controller_name, controller in self._controllers.items():
            controller.load_state(state=controller_states[controller_name])

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize the controller states sequentially
        controller_states_flat = th.cat(
            [c.serialize(state=state["controllers"][c_name]) for c_name, c in self._controllers.items()]
        )

        # Concatenate and return
        return th.cat([state_flat, controller_states_flat])

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize the controller states sequentially
        controller_states = dict()
        for c_name, c in self._controllers.items():
            controller_states[c_name], deserialized_items = c.deserialize(state=state[idx:])
            idx += deserialized_items
        state_dict["controllers"] = controller_states

        return state_dict, idx


    def _initialize(self):
        # Run super
        super()._initialize()

        # Assert that the prim path matches ControllableObjectViewAPI's expected format
        scene_id, robot_name = self.articulation_root_path.split("/")[2:4]
        assert scene_id.startswith(
            "scene_"
        ), "Second component of articulation root path (scene ID) must start with 'scene_'"
        robot_name_components = robot_name.split("__")
        assert (
            len(robot_name_components) == 3
        ), "Third component of articulation root path (robot name) must have 3 components separated by '__'"
        assert (
            robot_name_components[0] == "controllable"
        ), "Third component of articulation root path (robot name) must start with 'controllable'"
        assert (
            robot_name_components[1] == self.__class__.__name__.lower()
        ), "Third component of articulation root path (robot name) must contain the class name as the second part"

        # Fill in the DOF to joint mapping
        self._dof_to_joints = dict()
        idx = 0
        for joint in self._joints.values():
            for _ in range(joint.n_dof):
                self._dof_to_joints[idx] = joint
                idx += 1

        # Update the reset joint pos
        if self._reset_joint_pos is None:
            self._reset_joint_pos = self._default_joint_pos

        # Load controllers
        self._load_controllers()

         # Setup action space
        self._action_space = (
            self._create_discrete_action_space()
            if self._action_type == "discrete"
            else self._create_continuous_action_space()
        )

        # Reset the object and keep all joints still after loading
        self.reset()
        self.keep_still()


        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Load the observation space for this robot
        self.load_observation_space()

        # Validate this robot configuration
        self._validate_configuration()

        self._reset_joint_pos_aabb_extent = self.aabb_extent

    def _load_sensors(self):
        """
        Loads sensor(s) to retrieve observations from this object.
        Stores created sensors as dictionary mapping sensor names to specific sensor
        instances used by this object.
        """
        # Populate sensor config
        self._sensor_config = self._generate_sensor_config(custom_config=self._sensor_config)

        # Search for any sensors this robot might have attached to any of its links
        self._sensors = dict()
        obs_modalities = set()
        for link_name, link in self._links.items():
            # Search through all children prims and see if we find any sensor
            sensor_counts = {p: 0 for p in SENSOR_PRIMS_TO_SENSOR_CLS.keys()}
            for prim in link.prim.GetChildren():
                prim_type = prim.GetPrimTypeInfo().GetTypeName()
                if prim_type in SENSOR_PRIMS_TO_SENSOR_CLS:
                    # Possibly filter out the sensor based on name
                    prim_path = str(prim.GetPrimPath())
                    not_blacklisted = self._exclude_sensor_names is None or not any(
                        name in prim_path for name in self._exclude_sensor_names
                    )
                    whitelisted = self._include_sensor_names is None or any(
                        name in prim_path for name in self._include_sensor_names
                    )
                    # Also make sure that the include / exclude sensor names are mutually exclusive
                    if self._exclude_sensor_names is not None and self._include_sensor_names is not None:
                        assert (
                            len(set(self._exclude_sensor_names).intersection(set(self._include_sensor_names))) == 0
                        ), (
                            f"include_sensor_names and exclude_sensor_names must be mutually exclusive! "
                            f"Got: {self._include_sensor_names} and {self._exclude_sensor_names}"
                        )
                    if not (not_blacklisted and whitelisted):
                        continue

                    # Infer what obs modalities to use for this sensor
                    sensor_cls = SENSOR_PRIMS_TO_SENSOR_CLS[prim_type]
                    sensor_kwargs = self._sensor_config[sensor_cls.__name__]
                    if "modalities" not in sensor_kwargs:
                        sensor_kwargs["modalities"] = (
                            sensor_cls.all_modalities
                            if self._obs_modalities == "all"
                            else sensor_cls.all_modalities.intersection(self._obs_modalities)
                        )
                    # If the modalities list is empty, don't import the sensor.
                    if not sensor_kwargs["modalities"]:
                        continue

                    obs_modalities = obs_modalities.union(sensor_kwargs["modalities"])
                    # Create the sensor and store it internally
                    sensor = create_sensor(
                        sensor_type=prim_type,
                        relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, prim_path),
                        name=f"{self.name}:{link_name}:{prim_type}:{sensor_counts[prim_type]}",
                        **sensor_kwargs,
                    )
                    sensor.load(self.scene)
                    self._sensors[sensor.name] = sensor
                    sensor_counts[prim_type] += 1

        # Since proprioception isn't an actual sensor, we need to possibly manually add it here as well
        if self._obs_modalities == "all" or "proprio" in self._obs_modalities:
            obs_modalities.add("proprio")

        # Update our overall obs modalities
        self._obs_modalities = obs_modalities

    def _generate_sensor_config(self, custom_config=None):
        """
        Generates a fully-populated sensor config, overriding any default values with the corresponding values
        specified in @custom_config

        Args:
            custom_config (None or Dict[str, ...]): nested dictionary mapping sensor class name(s) to specific custom
                sensor configurations for this object. This will override any default values specified by this class

        Returns:
            dict: Fully-populated nested dictionary mapping sensor class name(s) to specific sensor configurations
                for this object
        """
        sensor_config = {} if custom_config is None else deepcopy(custom_config)

        # Merge the sensor dictionaries
        sensor_config = merge_nested_dicts(
            base_dict=self._default_sensor_config,
            extra_dict=sensor_config,
        )

        return sensor_config

    def _validate_configuration(self):
        """
        Run any needed sanity checks to make sure this robot was created correctly.
        """
        pass

    def get_obs(self):
        """
        Grabs all observations from the robot. This is keyword-mapped based on each observation modality
            (e.g.: proprio, rgb, etc.)

        Returns:
            2-tuple:
                dict: Keyword-mapped dictionary mapping observation modality names to
                    observations (usually np arrays)
                dict: Keyword-mapped dictionary mapping observation modality names to
                    additional info
        """
        # Our sensors already know what observation modalities it has, so we simply iterate over all of them
        # and grab their observations, processing them into a flat dict
        obs_dict = dict()
        info_dict = dict()
        for sensor_name, sensor in self._sensors.items():
            obs_dict[sensor_name], info_dict[sensor_name] = sensor.get_obs()
            for key in obs_dict[sensor_name]:
                if "pointcloud" in key:
                    # convert point cloud from world frame to robot base frame
                    obs_dict[sensor_name][key] = change_pcd_frame(
                        pcd=obs_dict[sensor_name][key],
                        rel_pose=th.cat(self.get_position_orientation()),
                    )

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_dict["proprio"], info_dict["proprio"] = self.get_proprioception()

        return obs_dict, info_dict

    def get_proprioception(self):
        """
        Returns:
            n-array: numpy array of all robot-specific proprioceptive observations.
            dict: empty dictionary, a placeholder for additional info
        """
        proprio_dict = self._get_proprioception_dict()
        return th.cat([proprio_dict[obs] for obs in self._proprio_obs]), {}

    def _get_proprioception_dict(self):
        """
        Returns:
            dict: keyword-mapped proprioception observations available for this robot.
                Can be extended by subclasses
        """
        joint_positions = cb.to_torch(
            cb.copy(ControllableObjectViewAPI.get_joint_positions(self.articulation_root_path))
        )
        joint_velocities = cb.to_torch(
            cb.copy(ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path))
        )
        joint_efforts = cb.to_torch(cb.copy(ControllableObjectViewAPI.get_joint_efforts(self.articulation_root_path)))
        pos, quat = ControllableObjectViewAPI.get_position_orientation(self.articulation_root_path)
        pos, quat = cb.to_torch(cb.copy(pos)), cb.to_torch(cb.copy(quat))
        ori = T.quat2euler(quat)

        ori_2d = T.z_angle_from_quat(quat).unsqueeze(0)  # Convert to 1D tensor

        # Pack everything together
        return dict(
            joint_qpos=joint_positions,
            joint_qpos_sin=th.sin(joint_positions),
            joint_qpos_cos=th.cos(joint_positions),
            joint_qvel=joint_velocities,
            joint_qeffort=joint_efforts,
            robot_pos=pos,
            robot_ori_cos=th.cos(ori),
            robot_ori_sin=th.sin(ori),
            robot_2d_ori=ori_2d,
            robot_2d_ori_cos=th.cos(ori_2d),
            robot_2d_ori_sin=th.sin(ori_2d),
            robot_lin_vel=cb.to_torch(
                cb.copy(ControllableObjectViewAPI.get_linear_velocity(self.articulation_root_path))
            ),
            robot_ang_vel=cb.to_torch(
                cb.copy(ControllableObjectViewAPI.get_angular_velocity(self.articulation_root_path))
            ),
        )

    def _load_observation_space(self):
        # We compile observation spaces from our sensors
        obs_space = dict()

        for sensor_name, sensor in self._sensors.items():
            # Load the sensor observation space
            obs_space[sensor_name] = sensor.load_observation_space()

        # Have to handle proprio separately since it's not an actual sensor
        if "proprio" in self._obs_modalities:
            obs_space["proprio"] = self._build_obs_box_space(
                shape=(self.proprioception_dim,), low=-float("inf"), high=float("inf"), dtype=NumpyTypes.FLOAT32
            )

        return obs_space

    def add_obs_modality(self, modality):
        """
        Adds observation modality @modality to this robot. Note: Should be one of omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to add to this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we add it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.add_modality(modality=modality)

    def remove_obs_modality(self, modality):
        """
        Remove observation modality @modality from this robot. Note: Should be one of
        omnigibson.sensors.ALL_SENSOR_MODALITIES

        Args:
            modality (str): Observation modality to remove from this robot
        """
        # Iterate over all sensors we own, and if the requested modality is a part of its possible valid modalities,
        # then we remove it
        for sensor in self._sensors.values():
            if modality in sensor.all_modalities:
                sensor.remove_modality(modality=modality)

    def visualize_sensors(self):
        """
        Renders this robot's key sensors, visualizing them via matplotlib plots
        """
        frames = dict()
        remaining_obs_modalities = deepcopy(self.obs_modalities)
        for sensor in self.sensors.values():
            obs, _ = sensor.get_obs()
            sensor_frames = []
            if isinstance(sensor, VisionSensor):
                # We check for rgb, depth, normal, seg_instance
                for modality in ["rgb", "depth", "normal", "seg_instance"]:
                    if modality in sensor.modalities:
                        ob = obs[modality]
                        if modality == "rgb":
                            # Ignore alpha channel, map to floats
                            ob = ob[:, :, :3] / 255.0
                        elif modality == "seg_instance":
                            # Map IDs to rgb
                            ob = segmentation_to_rgb(ob, N=256) / 255.0
                        elif modality == "normal":
                            # Re-map to 0 - 1 range
                            ob = (ob + 1.0) / 2.0
                        else:
                            # Depth, nothing to do here
                            pass
                        # Add this observation to our frames and remove the modality
                        sensor_frames.append((modality, ob))
                        remaining_obs_modalities -= {modality}
                    else:
                        # Warn user that we didn't find this modality
                        print(f"Modality {modality} is not active in sensor {sensor.name}, skipping...")
            elif isinstance(sensor, ScanSensor):
                # We check for occupancy_grid
                occupancy_grid = obs.get("occupancy_grid", None)
                if occupancy_grid is not None:
                    sensor_frames.append(("occupancy_grid", occupancy_grid))
                    remaining_obs_modalities -= {"occupancy_grid"}

            # Map the sensor name to the frames for that sensor
            frames[sensor.name] = sensor_frames

        # Warn user that any remaining modalities are not able to be visualized
        if len(remaining_obs_modalities) > 0:
            print(f"Modalities: {remaining_obs_modalities} cannot be visualized, skipping...")

        # Write all the frames to a plot
        import matplotlib.pyplot as plt

        for sensor_name, sensor_frames in frames.items():
            n_sensor_frames = len(sensor_frames)
            if n_sensor_frames > 0:
                fig, axes = plt.subplots(nrows=1, ncols=n_sensor_frames)
                if n_sensor_frames == 1:
                    axes = [axes]
                # Dump frames and set each subtitle
                for i, (modality, frame) in enumerate(sensor_frames):
                    axes[i].imshow(frame)
                    axes[i].set_title(modality)
                    axes[i].set_axis_off()
                # Set title
                fig.suptitle(sensor_name)
                plt.show(block=False)

        # One final plot show so all the figures get rendered
        plt.show()

    def remove(self):
        """
        Do NOT call this function directly to remove a prim - call og.sim.remove_prim(prim) for proper cleanup
        """
        # Remove all sensors
        for sensor in self._sensors.values():
            sensor.remove()

        # Run super
        super().remove()

    @property
    def reset_joint_pos_aabb_extent(self):
        """
        This is the aabb extent of the robot in the robot frame after resetting the joints.
        Returns:
            3-array: Axis-aligned bounding box extent of the robot base
        """
        return self._reset_joint_pos_aabb_extent

    def teleop_data_to_action(self, teleop_action) -> th.Tensor:
        """
        Generate action data from teleoperation action data
        Args:
            teleop_action (TeleopAction): teleoperation action data
        Returns:
            th.tensor: array of action data filled with update value
        """
        return th.zeros(self.action_dim)

    @property
    def sensors(self):
        """
        Returns:
            dict: Keyword-mapped dictionary mapping sensor names to BaseSensor instances owned by this robot
        """
        return self._sensors

    @property
    def obs_modalities(self):
        """
        Returns:
            set of str: Observation modalities used for this robot (e.g.: proprio, rgb, etc.)
        """
        assert self._loaded, "Cannot check observation modalities until we load this robot!"
        return self._obs_modalities

    @property
    def proprioception_dim(self):
        """
        Returns:
            int: Size of self.get_proprioception() vector
        """
        return len(self.get_proprioception()[0])

    @property
    def _default_sensor_config(self):
        """
        Returns:
            dict: default nested dictionary mapping sensor class name(s) to specific sensor
                configurations for this object. See kwargs from omnigibson/sensors/__init__/create_sensor for more
                details

                Expected structure is as follows:
                    SensorClassName1:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    SensorClassName2:
                        modalities: ...
                        enabled: ...
                        noise_type: ...
                        noise_kwargs:
                            ...
                        sensor_kwargs:
                            ...
                    ...
        """
        return {
            "VisionSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    "image_height": 128,
                    "image_width": 128,
                },
            },
            "ScanSensor": {
                "enabled": True,
                "noise_type": None,
                "noise_kwargs": None,
                "sensor_kwargs": {
                    # Basic LIDAR kwargs
                    "min_range": 0.05,
                    "max_range": 10.0,
                    "horizontal_fov": 360.0,
                    "vertical_fov": 1.0,
                    "yaw_offset": 0.0,
                    "horizontal_resolution": 1.0,
                    "vertical_resolution": 1.0,
                    "rotation_rate": 0.0,
                    "draw_points": False,
                    "draw_lines": False,
                    # Occupancy Grid kwargs
                    "occupancy_grid_resolution": 128,
                    "occupancy_grid_range": 5.0,
                    "occupancy_grid_inner_radius": 0.5,
                    "occupancy_grid_local_link": None,
                },
            },
        }

    @property
    def default_proprio_obs(self):
        """
        Returns:
            list of str: Default proprioception observations to use
        """
        return []

    @property
    def model_name(self):
        """
        Returns:
            str: name of this robot model. usually corresponds to the class name of a given robot model
        """
        return self.__class__.__name__

    @property
    def usd_path(self):
        # By default, sets the standardized path
        model = self.model_name.lower()
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{model}/usd/{model}.usda")

    @property
    def urdf_path(self):
        """
        Returns:
            str: file path to the robot urdf file.
        """
        # By default, sets the standardized path
        model = self.model_name.lower()
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{model}/urdf/{model}.urdf")

    @classproperty
    def _do_not_register_classes(cls):
        # Don't register this class since it's an abstract template
        classes = super()._do_not_register_classes
        classes.add("BaseRobot")
        return classes

    @classproperty
    def _cls_registry(cls):
        # Global robot registry -- override super registry
        global REGISTERED_ROBOTS
        return REGISTERED_ROBOTS
    
    @property
    def base_footprint_link_name(self):
        """
        Get the base footprint link name for the controllable object.

        The base footprint link is the link that should be considered the base link for the object
        even in the presence of virtual joints that may be present in the object's articulation. For
        robots without virtual joints, this is the same as the root link. For robots with virtual joints,
        this is the link that is the child of the last virtual joint in the robot's articulation.

        Returns:
            str: Name of the base footprint link for this object
        """
        return self.root_link_name

    @property
    def base_footprint_link(self):
        """
        Get the base footprint link for the controllable object.

        The base footprint link is the link that should be considered the base link for the object
        even in the presence of virtual joints that may be present in the object's articulation. For
        robots without virtual joints, this is the same as the root link. For robots with virtual joints,
        this is the link that is the child of the last virtual joint in the robot's articulation.

        Returns:
            RigidDynamicPrim: Base footprint link for this object
        """
        return self.links[self.base_footprint_link_name]

    @property
    def action_dim(self):
        """
        Returns:
            int: Dimension of action space for this object. By default,
                is the sum over all controller action dimensions
        """
        return sum([controller.command_dim for controller in self._controllers.values()])

    @property
    def action_space(self):
        """
        Action space for this object.

        Returns:
            gym.space: Action space, either discrete (Discrete) or continuous (Box)
        """
        return deepcopy(self._action_space)

    @property
    def discrete_action_list(self):
        """
        Discrete choices for actions for this object. Only needs to be implemented if the object supports discrete
        actions.

        Returns:
            dict: Mapping from single action identifier (e.g.: a string, or a number) to array of continuous
                actions to deploy via this object's controllers.
        """
        raise NotImplementedError()

    @property
    def controllers(self):
        """
        Returns:
            dict: Controllers owned by this object, mapping controller name to controller object
        """
        return self._controllers

    @property
    def controller_order(self):
        """
        Returns:
            list: Ordering of the actions, corresponding to the controllers. e.g., ["base", "arm", "gripper"],
                to denote that the action vector should be interpreted as first the base action, then arm command, then
                gripper command. Note that this may be a subset of all possible controllers due to some controllers
                subsuming others (e.g.: arm controller subsuming the trunk controller if using IK)
        """
        assert self._controllers is not None, "Can only view controller_order after controllers are loaded!"
        return list(self._controllers.keys())

    @property
    @abstractmethod
    def _raw_controller_order(self):
        """
        Returns:
            list: Raw ordering of the actions, corresponding to the controllers. e.g., ["base", "arm", "gripper"],
                to denote that the action vector should be interpreted as first the base action, then arm command, then
                gripper command. Note that external users should query @controller_order, which is the post-processed
                ordering of actions, which may be a subset of the controllers due to some controllers subsuming others
                (e.g.: arm controller subsuming the trunk controller if using IK)
        """
        raise NotImplementedError

    @property
    def controller_action_idx(self):
        """
        Returns:
            dict: Mapping from controller names (e.g.: head, base, arm, etc.) to corresponding
                indices (list) in the action vector
        """
        dic = {}
        idx = 0
        for controller in self.controller_order:
            cmd_dim = self._controllers[controller].command_dim
            dic[controller] = th.arange(idx, idx + cmd_dim)
            idx += cmd_dim

        return dic

    @property
    def controller_joint_idx(self):
        """
        Returns:
            dict: Mapping from controller names (e.g.: head, base, arm, etc.) to corresponding
                indices (list) of the joint state vector controlled by each controller
        """
        dic = {}
        for controller in self.controller_order:
            dic[controller] = self._controllers[controller].dof_idx

        return dic

    # TODO: These are cached, but they are not updated when the joint limit is changed
    @cached_property
    def control_limits(self):
        """
        Returns:
            dict: Keyword-mapped limits for this object. Dict contains:
                position: (min, max) joint limits, where min and max are N-DOF arrays
                velocity: (min, max) joint velocity limits, where min and max are N-DOF arrays
                effort: (min, max) joint effort limits, where min and max are N-DOF arrays
                has_limit: (n_dof,) array where each element is True if that corresponding joint has a position limit
                    (otherwise, joint is assumed to be limitless)
        """
        return {
            "position": (self.joint_lower_limits, self.joint_upper_limits),
            "velocity": (-self.max_joint_velocities, self.max_joint_velocities),
            "effort": (-self.max_joint_efforts, self.max_joint_efforts),
            "has_limit": self.joint_has_limits,
        }

    @property
    def reset_joint_pos(self):
        """
        Returns:
            n-array: reset joint positions for this robot
        """
        return self._reset_joint_pos

    @reset_joint_pos.setter
    def reset_joint_pos(self, value):
        """
        Args:
            value: the new reset joint positions for this robot
        """
        self._reset_joint_pos = value

    @property
    # @abstractmethod
    def _default_joint_pos(self):
        """
        Returns:
            n-array: Default joint positions for this robot
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _default_controller_config(self):
        """
        Returns:
            dict: default nested dictionary mapping controller name(s) to specific controller
                configurations for this object. Note that the order specifies the sequence of actions to be received
                from the environment.

                Expected structure is as follows:
                    group1:
                        controller_name1:
                            controller_name1_params
                            ...
                        controller_name2:
                            ...
                    group2:
                        ...

                The @group keys specify the control type for various aspects of the object,
                e.g.: "head", "arm", "base", etc. @controller_name keys specify the supported controllers for
                that group. A default specification MUST be specified for each controller_name.
                e.g.: IKController, DifferentialDriveController, JointController, etc.
        """
        return {}

    @property
    @abstractmethod
    def _default_controllers(self):
        """
        Returns:
            dict: Maps object group (e.g. base, arm, etc.) to default controller class name to use
            (e.g. IKController, JointController, etc.)
        """
        return {}
