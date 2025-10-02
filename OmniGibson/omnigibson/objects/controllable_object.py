import math
from abc import abstractmethod
from copy import deepcopy
from functools import cached_property
from typing import Literal

import gymnasium as gym
import torch as th

import omnigibson as og
from omnigibson.controllers import create_controller
from omnigibson.controllers.controller_base import ControlType
from omnigibson.controllers.joint_controller import JointController
from omnigibson.objects.object_base import BaseObject
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.constants import PrimType
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import CachedFunctions, assert_valid_key, merge_nested_dicts
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

# Create module logger
log = create_module_logger(module_name=__name__)


class ControllableObject(BaseObject):
    """
    Simple class that extends object functionality for controlling joints -- this assumes that at least some joints
    are motorized (i.e.: non-zero low-level simulator joint motor gains) and intended to be controlled,
    e.g.: a conveyor belt or a robot agent
    """

    def __init__(
        self,
        name,
        relative_prim_path=None,
        category="object",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        link_physics_materials=None,
        load_config=None,
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (None or str): The path relative to its scene prim for this object. If not specified, it defaults to /<name>.
            category (str): Category for the object. Defaults to "object".
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
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
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store inputs
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

        # Store internal placeholders that will be filled in later
        self._dof_to_joints = None  # dict that will map DOF indices to JointPrims
        self._last_action = None
        self._controllers = None
        self.dof_names_ordered = None
        self._control_enabled = True

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
            name=name,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            **kwargs,
        )

    def _initialize(self):
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

        # Run super first
        super()._initialize()
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
        # Call super first
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

    @abstractmethod
    def _create_discrete_action_space(self):
        """
        Create a discrete action space for this object. Should be implemented by the subclass (if a subclass does not
        support this type of action space, it should raise an error).

        Returns:
            gym.space: Object-specific discrete action space
        """
        raise NotImplementedError

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
    @abstractmethod
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
