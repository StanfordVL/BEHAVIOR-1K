import os
from copy import deepcopy
import math
import torch as th
from functools import cached_property
from omegaconf import OmegaConf

from omnigibson.robots.definition_schema import (
    RobotDefinition,
    EndEffectorDefinition,
)
from typing import Literal
from collections import namedtuple
import networkx as nx
import gymnasium as gym
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.sensors import (
    ALL_SENSOR_MODALITIES,
    SENSOR_PRIMS_TO_SENSOR_CLS,
    ScanSensor,
    VisionSensor,
    create_sensor,
)
import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.gym_utils import GymObservable
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import merge_nested_dicts, CachedFunctions, assert_valid_key
from omnigibson.utils.vision_utils import segmentation_to_rgb, change_pcd_frame
from omnigibson.utils.geometry_utils import wrap_angle
from omnigibson.controllers import (
    create_controller,
    ControlType,
    GripperController,
    InverseKinematicsController,
    IsGraspingState,
    ManipulationController,
    MultiFingerGripperController,
    OperationalSpaceController,
    LocomotionController,
    JointController,
    HolonomicBaseJointController,
    DifferentialDriveController,
)
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.object_states import ContactBodies
from omnigibson.prims.geom_prim import VisualGeomPrim
from omnigibson.utils.constants import JointType, PrimType, ROBOT_CATEGORY
from omnigibson.utils.sampling_utils import raytest_batch
from omnigibson.utils.usd_utils import (
    ControllableObjectViewAPI,
    GripperRigidContactAPI,
    create_joint,
    create_primitive_mesh,
    absolute_prim_path_to_scene_relative,
    delete_or_deactivate_prim,
)

# Create module logger
log = create_module_logger(module_name=__name__)

# Add proprio sensor modality to ALL_SENSOR_MODALITIES
ALL_SENSOR_MODALITIES.add("proprio")

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.ASSIST_FRACTION = 1.0
m.ASSIST_GRASP_MASS_THRESHOLD = 10.0
m.ARTICULATED_ASSIST_FRACTION = 0.7
m.MIN_ASSIST_FORCE = 0
m.MAX_ASSIST_FORCE = 100
m.MIN_AG_DEFAULT_GRASP_POINT_PROP = 0.2
m.MAX_AG_DEFAULT_GRASP_POINT_PROP = 0.95
m.AG_DEFAULT_GRASP_POINT_Z_PROP = 0.4
m.CONSTRAINT_VIOLATION_THRESHOLD = 0.1
m.GRASP_WINDOW = 3.0
m.RELEASE_WINDOW = 1 / 30.0
m.MAX_LINEAR_VELOCITY = 1.5
m.MAX_ANGULAR_VELOCITY = th.pi
m.MAX_EFFORT = 1000.0
m.BASE_JOINT_CONTROLLER_POSITION_KP = 100.0


AG_MODES = {
    "physical",
    "assisted",
    "sticky",
}
GraspingPoint = namedtuple("GraspingPoint", ["link_name", "position"])  # link_name (str), position (x,y,z tuple)


class Robot(USDObject, GymObservable):
    def __init__(
        self,
        # Shared kwargs in hierarchy
        name,
        model,
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
        control_freq=None,
        controller_config=None,
        action_type="continuous",
        action_normalize=True,
        reset_joint_pos=None,
        obs_modalities=("rgb", "proprio"),
        include_sensor_names=None,
        exclude_sensor_names=None,
        proprio_obs="default",
        sensor_config=None,
        # Unique to ManipulationRobot
        grasping_mode="physical",
        grasping_direction="lower",
        disable_grasp_handling=False,
        finger_static_friction=None,
        finger_dynamic_friction=None,
        # Unique to MobileManipulationRobot
        default_reset_mode="untuck",
        # Unique to robots with multiple arm poses
        default_arm_pose=None,
        # Unique to Tiago
        variant="default",
        # Unique to A1, Franka
        end_effector="gripper",
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            model (str): Model of Robot.
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
                Set this value instead if you want to initialize the object with a different reset joint position.
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
                If "assisted", will magnetize any object touching and within the gripper's fingers. In this mode,
                    at least two "fingers" need to touch the object.
                If "sticky", will magnetize any object touching the gripper's fingers. In this mode, only one finger
                    needs to touch the object.
            grasping_direction (str): One of {"lower", "upper"}. If "lower", lower limit represents a closed grasp,
                otherwise upper limit represents a closed grasp.
            disable_grasp_handling (bool): If True, the robot will not automatically handle assisted or sticky grasps.
                Instead, you will need to call the grasp handling methods yourself.
            finger_static_friction (None or float): If specified, specific static friction to use for robot's fingers
            finger_dynamic_friction (None or float): If specified, specific dynamic friction to use for robot's fingers.
                Note: If specified, this will override any ways that are found within @link_physics_materials for any
                robot finger gripper links
            default_reset_mode (str): Default reset mode for the robot. Should be one of: {"tuck", "untuck"}
                If reset_joint_pos is not None, this will be ignored (since _default_joint_pos won't be used during initialization).
            end_effector (str): The end effector type to use.
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject).
        """
        self.model = model
        # Read and validate robot definition YAML file using OmegaConf
        definition_dir = os.path.dirname(__file__)
        definition_path = os.path.join(definition_dir, "definitions", self.model + ".yaml")
        yaml_definition = OmegaConf.load(definition_path)
        schema = OmegaConf.structured(RobotDefinition)
        merged_definition = OmegaConf.merge(schema, yaml_definition)
        self._definition: RobotDefinition = OmegaConf.to_object(merged_definition)

        if self.has_end_effector_variants:
            self.end_effector = end_effector
            grasping_direction = "lower" if end_effector == "gripper" else "upper"
            self._init_ag_points()

        if self.is_manipulation:
            # Store relevant internal vars
            assert_valid_key(key=grasping_mode, valid_keys=AG_MODES, name="grasping_mode")
            assert_valid_key(key=grasping_direction, valid_keys=["lower", "upper"], name="grasping direction")
            self._grasping_mode = grasping_mode
            self._grasping_direction = grasping_direction
            self._disable_grasp_handling = disable_grasp_handling

            # Other variables filled in at runtime
            self._eef_to_fingertip_lengths = None  # dict mapping arm name to finger name to offset

            # Initialize other variables used for assistive grasping
            self._ag_obj_in_hand = {arm: None for arm in self.arm_names}
            self._ag_obj_constraints = {arm: None for arm in self.arm_names}
            self._ag_obj_constraint_params = {arm: {} for arm in self.arm_names}
            self._ag_freeze_gripper = {arm: None for arm in self.arm_names}
            self._ag_release_counter = {arm: None for arm in self.arm_names}
            self._ag_grasp_counter = {arm: None for arm in self.arm_names}

        if self.is_holonomic_base:
            fixed_base = True
            self._world_base_fixed_joint_prim = None

            # Sanity check that the base controller is a HolonomicBaseJointController
            if controller_config is not None and "base" in controller_config:
                assert (
                    controller_config["base"]["name"] == "HolonomicBaseJointController"
                ), "Base controller must be a HolonomicBaseJointController!"

        if self.is_mobile_manipulation:
            assert_valid_key(key=default_reset_mode, valid_keys=RESET_JOINT_OPTIONS, name="default_reset_mode")
            self.default_reset_mode = default_reset_mode

        if self.has_multiple_arm_poses:
            if default_arm_pose is not None:
                assert_valid_key(key=default_arm_pose, valid_keys=self.default_arm_poses, name="default_arm_pose")
            else:
                default_arm_pose = self._definition.mobile_manipulation.default_arm_pose_key
            self.default_arm_pose = default_arm_pose

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

        # Store internal placeholders that will be filled in later
        self._dof_to_joints = None  # dict that will map DOF indices to JointPrims
        self._last_action = None
        self._controllers = None
        self.dof_names_ordered = None
        self._control_enabled = True

        class_name = self.kinematic_tree_identifier.lower()
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
            category=ROBOT_CATEGORY,
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

        if self.is_manipulation:
            # Update finger link material dictionary based on desired values
            if finger_static_friction is not None or finger_dynamic_friction is not None:
                for arm, finger_link_names in self.finger_link_names.items():
                    for finger_link_name in finger_link_names:
                        if finger_link_name not in self._link_physics_materials:
                            self._link_physics_materials[finger_link_name] = dict()
                        if finger_static_friction is not None:
                            self._link_physics_materials[finger_link_name]["static_friction"] = finger_static_friction
                        if finger_dynamic_friction is not None:
                            self._link_physics_materials[finger_link_name]["dynamic_friction"] = finger_dynamic_friction

    @property
    def is_two_wheel(self) -> bool:
        """Returns True if this robot has a two-wheel differential drive base."""
        return self._definition.two_wheel is not None

    @property
    def is_holonomic_base(self) -> bool:
        """Returns True if this robot has a holonomic base."""
        return self._definition.holonomic_base is not None

    @property
    def is_articulated_trunk(self) -> bool:
        """Returns True if this robot has an articulated trunk."""
        return self._definition.articulated_trunk is not None

    @property
    def is_active_camera(self) -> bool:
        """Returns True if this robot has an active (controllable) camera."""
        return self._definition.active_camera is not None

    @property
    def has_multiple_arm_poses(self) -> bool:
        """Returns True if this robot has multiple arm pose configurations."""
        return (
            self._definition.mobile_manipulation is not None
            and self._definition.mobile_manipulation.default_arm_poses is not None
        )

    @property
    def is_mobile_manipulation(self) -> bool:
        """Returns True if this robot is a mobile manipulation robot."""
        return self._definition.mobile_manipulation is not None

    @property
    def is_manipulation(self) -> bool:
        """Returns True if this robot has manipulation capabilities."""
        return self._definition.manipulation is not None or self.is_mobile_manipulation or self.is_articulated_trunk

    @property
    def is_locomotion(self) -> bool:
        """Returns True if this robot has locomotion capabilities."""
        return self.is_holonomic_base or self.is_two_wheel or self._definition.locomotion is not None

    @property
    def linear_velocity_gain_for_primitives(self) -> float:
        """Returns the linear velocity proportional gain for action primitives."""
        assert (
            self._definition.linear_velocity_gain_for_primitives is not None
        ), f"linear_velocity_gain_for_primitives not defined for robot {self.model}"
        return self._definition.linear_velocity_gain_for_primitives

    @property
    def angular_velocity_gain_for_primitives(self) -> float:
        """Returns the angular velocity proportional gain for action primitives."""
        assert (
            self._definition.angular_velocity_gain_for_primitives is not None
        ), f"angular_velocity_gain_for_primitives not defined for robot {self.model}"
        return self._definition.angular_velocity_gain_for_primitives

    @property
    def has_end_effector_variants(self) -> bool:
        """Returns True if this robot supports multiple end effector types."""
        return (
            self._definition.manipulation is not None
            and self._definition.manipulation.supported_end_effector is not None
        )

    def _convert_to_grasping_points(self, li):
        result = []
        for item in li:
            link_name, position = item
            result.append(GraspingPoint(link_name=link_name, position=th.tensor(position)))
        return result

    def _init_ag_points(self):
        """Initialize assisted grasp points from end effector definition."""
        eef_def = self._get_end_effector_definition()
        if eef_def is None or eef_def.ag_start_points is None:
            return

        self._ag_start_points = self._convert_to_grasping_points(eef_def.ag_start_points)
        self._ag_end_points = self._convert_to_grasping_points(eef_def.ag_end_points)

    def _get_end_effector_definition(self) -> "EndEffectorDefinition | None":
        """Get the current end effector configuration if this robot has end effector variants."""
        if not self.has_end_effector_variants:
            return None
        return self._definition.manipulation.end_effectors.get(self.end_effector)

    @property
    def kinematic_tree_identifier(self):
        """
        A string that uniquely identifies the kinematic tree of this particular robot+endeffector combination.
        This is used to generate the prim path of the robot, which is then used for glob matching by the batched
        controller APIs. This allows robots to be grouped into views based on their number, and type, of joints.

        If the robot has a manipulation config with a supported end effector that defines
        a model, that end-effector model is returned; otherwise the robot's base model.
        """
        if self._definition.manipulation and self._definition.manipulation.supported_end_effector:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.model:
                return eef_def.model
        return self.model

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

        if self.is_holonomic_base:
            self._world_base_fixed_joint_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(
                f"{self.prim_path}/rootJoint"
            )
            position, orientation = self.get_position_orientation()
            # Set the world-to-base fixed joint to be at the robot's current pose
            self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
            self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
                lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
            )

        force_sphere = (
            self.is_holonomic_base
            and self._definition.holonomic_base
            and self._definition.holonomic_base.force_sphere_wheel_approximation
        )
        if force_sphere:
            # R1 and R1Pro's URDFs still use the mesh type for the collision meshes of the wheels
            # We need to manually set it back to sphere approximation
            for wheel_name in self.floor_touching_base_link_names:
                wheel_link = self.links[wheel_name]
                assert set(wheel_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
                wheel_link.collision_meshes["collisions"].set_collision_approximation("boundingSphere")
        if self._definition.visual_only_eef_links:
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
        if self.is_holonomic_base:
            base_joint_positions = self.get_joint_positions()[self.base_idx]
        # Call super first
        super().reset()

        # Override the reset joint state based on reset values
        self.set_joint_positions(positions=self._reset_joint_pos, drive=False)

        if self.is_holonomic_base:
            self.set_joint_positions(base_joint_positions, indices=self.base_idx)

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
        if self.is_holonomic_base:
            rz_joint_dof_indices = self.joints["base_footprint_rz_joint"].dof_indices
            j_pos = self.get_joint_positions()[rz_joint_dof_indices]
            # In preparation for the base controller's @update_goal, we need to wrap the current joint pos
            # to be in range [-pi, pi], so that once the command (a delta joint pos in range [-pi, pi])
            # is applied, the final target joint pos is in range [-pi * 2, pi * 2], which is required by Isaac.
            if j_pos < -math.pi or j_pos > math.pi:
                j_pos = wrap_angle(j_pos)
                self.set_joint_positions(j_pos, indices=rz_joint_dof_indices, drive=False)

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
        if self.is_manipulation:
            # We intercept the gripper control and replace it with the current joint position if we're freezing our gripper
            for arm in self.arm_names:
                if self._ag_freeze_gripper[arm]:
                    control[self.gripper_control_idx[arm]] = (
                        self._ag_obj_constraint_params[arm]["gripper_pos"]
                        if self.controllers[f"gripper_{arm}"].control_type == ControlType.POSITION
                        else 0.0
                    )
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

        if self.is_manipulation:
            # Then run assisted grasping
            if self.grasping_mode != "physical" and not self._disable_grasp_handling:
                self._handle_assisted_grasping()

    def _update_constraint_cloth(self, arm="default"):
        """
        Update the AG constraint for cloth: for the fixed joint between the attachment point and the world, we set
        the local pos to match the current eef link position plus the attachment_point_pos_local offset. As a result,
        the joint will drive the attachment point to the updated position, which will then drive the cloth.
        See _establish_grasp_cloth for more details.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        assert self.is_manipulation
        attachment_point_pos_local = self._ag_obj_constraint_params[arm]["attachment_point_pos_local"]
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        attachment_point_pos, _ = T.pose_transform(
            eef_link_pos, eef_link_orn, attachment_point_pos_local, th.tensor([0, 0, 0, 1], dtype=th.float32)
        )
        joint_prim = self._ag_obj_constraints[arm]
        joint_prim.GetAttribute("physics:localPos1").Set(lazy.pxr.Gf.Vec3f(*attachment_point_pos))

    def _handle_assisted_grasping(self):
        """
        Handles assisted grasping by creating or removing constraints.
        """
        assert self.is_manipulation
        # Loop over all arms
        for arm in self.arm_names:
            # We apply a threshold based on the control rather than the command here so that the behavior
            # stays the same across different controllers and control modes (absolute / delta). This way,
            # a zero action will actually keep the AG setting where it already is.
            controller = self._controllers[f"gripper_{arm}"]
            controlled_joints = controller.dof_idx
            control = cb.to_torch(controller.control)
            if control is None:
                applying_grasp = False
            elif self._grasping_direction == "lower":
                applying_grasp = (
                    th.any(control < self.joint_upper_limits[controlled_joints])
                    if controller.control_type == ControlType.POSITION
                    else th.any(control < 0)
                )
            else:
                applying_grasp = (
                    th.any(control > self.joint_lower_limits[controlled_joints])
                    if controller.control_type == ControlType.POSITION
                    else th.any(control > 0)
                )
            # Execute gradual release of object
            if self._ag_obj_in_hand[arm]:
                if self._ag_release_counter[arm] is not None:
                    self._handle_release_window(arm=arm)
                else:
                    if gm.AG_CLOTH:
                        self._update_constraint_cloth(arm=arm)

                    if not applying_grasp:
                        self._release_grasp(arm=arm)
            elif applying_grasp:
                current_ag_data = self._calculate_in_hand_object(arm=arm)
                if self._ag_grasp_counter[arm] is not None:
                    # We're in a grasp window already
                    if current_ag_data is None:
                        # Lost contact with object, reset window
                        self._ag_grasp_counter[arm] = None
                    else:
                        self._ag_grasp_counter[arm] += 1

                        # Check if window is complete
                        time_in_grasp = self._ag_grasp_counter[arm] * og.sim.get_sim_step_dt()
                        if time_in_grasp >= m.GRASP_WINDOW:
                            # Establish grasp with the LATEST ag_data
                            self._establish_grasp(arm=arm, ag_data=current_ag_data)
                            # Reset the grasp window tracking
                            self._ag_grasp_counter[arm] = None
                elif current_ag_data is not None:
                    # Start tracking a new potential grasp
                    self._ag_grasp_counter[arm] = 0
            else:
                # Not trying to grasp, reset any pending grasp window
                self._ag_grasp_counter[arm] = None

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

        if self.is_holonomic_base:
            # Add canonical position and orientation
            fcns["_canonical_pos_quat"] = lambda: ControllableObjectViewAPI.get_root_position_orientation(
                self.articulation_root_path
            )
            fcns["canonical_pos"] = lambda: fcns["_canonical_pos_quat"][0]
            fcns["canonical_quat"] = lambda: fcns["_canonical_pos_quat"][1]

        if self.is_articulated_trunk:
            self._add_task_frame_control_dict(
                fcns=fcns, task_name="trunk", link_name=self.joints[self.trunk_joint_names[-1]].body1.split("/")[-1]
            )

        if self.is_manipulation:
            for arm in self.arm_names:
                eef_link_name = self.eef_link_names[arm] if self.eef_link_names else None
                if eef_link_name is None:
                    raise ValueError(f"eef_link_names is None for arm {arm}. Check robot definition YAML.")
                # Verify the link actually exists
                if eef_link_name not in self._links:
                    raise ValueError(
                        f"EEF link '{eef_link_name}' for arm '{arm}' not found in robot links. Available links: {list(self._links.keys())}"
                    )

                self._add_task_frame_control_dict(fcns=fcns, task_name=f"eef_{arm}", link_name=eef_link_name)

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

    def dump_action(self):
        """
        Dump the last action applied to this object. For use in demo collection.
        """
        return self._last_action

    def _base_set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        # Run super first
        super().set_position_orientation(position, orientation, frame)

        # Clear the controllable view's backend since state has changed
        ControllableObjectViewAPI.clear_object(prim_path=self.articulation_root_path)

    def set_position_orientation(
        self, position=None, orientation=None, frame: Literal["world", "parent", "scene"] = "world"
    ):
        """
        Sets robot's pose with respect to the specified frame
        ...
        """
        if self.is_holonomic_base:
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

            # If the simulator is playing, set the 6 base joints to achieve the desired pose of base_footprint link frame
            if og.sim.is_playing() and self.initialized:
                # Find the relative transformation from base_footprint_link ("base_footprint") frame to root_link
                # ("base_footprint_x") frame. Assign it to the 6 1DoF joints that control the base.
                # Note that the 6 1DoF joints are originated from the root_link ("base_footprint_x") frame.
                joint_pos, joint_orn = self.root_link.get_position_orientation()
                inv_joint_pos, inv_joint_orn = T.invert_pose_transform(joint_pos, joint_orn)
                relative_pos, relative_orn = T.pose_transform(inv_joint_pos, inv_joint_orn, position, orientation)
                intrinsic_eulers = T.mat2euler_intrinsic(T.quat2mat(relative_orn))
                joint_positions = th.concatenate((relative_pos, intrinsic_eulers))
                self.set_joint_positions(positions=joint_positions, indices=self.base_idx, drive=False)

            # Else, set the pose of the robot frame, and then move the joint frame of the world_base_joint to match it
            else:
                # Call the super() method to move the robot frame first
                self._base_set_position_orientation(position, orientation, frame)
                # Move the joint frame for the world_base_joint
                if self._world_base_fixed_joint_prim is not None:
                    self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(tuple(position))
                    self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(
                        lazy.pxr.Gf.Quatf(*orientation[[3, 0, 1, 2]].tolist())
                    )
            return

        if self.is_manipulation:
            # Store the original EEF poses.
            original_poses = {}
            for arm in self.arm_names:
                original_poses[arm] = (self.get_eef_position(arm), self.get_eef_orientation(arm))

            self._base_set_position_orientation(position, orientation, frame)

            # Now for each hand, if it was holding an AG object, teleport it.
            for arm in self.arm_names:
                if self._ag_obj_in_hand[arm] is not None:
                    original_eef_pose = T.pose2mat(original_poses[arm])
                    inv_original_eef_pose = T.pose_inv(pose_mat=original_eef_pose)
                    original_obj_pose = T.pose2mat(self._ag_obj_in_hand[arm].get_position_orientation())
                    new_eef_pose = T.pose2mat((self.get_eef_position(arm), self.get_eef_orientation(arm)))
                    # New object pose is transform:
                    # original --> "De"transform the original EEF pose --> "Re"transform the new EEF pose
                    new_obj_pose = new_eef_pose @ inv_original_eef_pose @ original_obj_pose
                    self._ag_obj_in_hand[arm].set_position_orientation(*T.mat2pose(hmat=new_obj_pose))
            return
        self._base_set_position_orientation(position, orientation, frame)

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

        # If we're using actual physical grasping, no extra state needed to save
        if self.is_manipulation and self.grasping_mode != "physical":
            # Include AG_state
            ag_params = self._ag_obj_constraint_params.copy()
            for arm in ag_params.keys():
                if len(ag_params[arm]) > 0:
                    assert (
                        self.scene is not None
                    ), "Cannot get position and orientation relative to scene without a scene"
                    ag_params[arm]["contact_pos"], _ = self.scene.convert_world_pose_to_scene_relative(
                        ag_params[arm]["contact_pos"],
                        th.tensor([0, 0, 0, 1], dtype=th.float32),
                    )
            state["ag_obj_constraint_params"] = ag_params

        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)

        # Load controller states
        controller_states = state["controllers"]
        for controller_name, controller in self._controllers.items():
            controller.load_state(state=controller_states[controller_name])

        if self.is_manipulation:
            # No additional loading needed if we're using physical grasping
            if self.grasping_mode == "physical":
                return

            # Include AG_state
            # TODO: currently does not take care of cloth objects
            # TODO: add unit tests
            for arm in self.arm_names:
                current_ag_constraint = self._ag_obj_constraint_params[arm]
                has_current_constraint = len(current_ag_constraint) > 0

                loaded_ag_constraint = {}
                has_loaded_constraint = False
                if "ag_obj_constraint_params" in state:
                    loaded_ag_constraint = state["ag_obj_constraint_params"][arm]
                    has_loaded_constraint = len(loaded_ag_constraint) > 0

                # Release existing grasp if needed
                should_release = False
                if has_current_constraint:
                    if not has_loaded_constraint:
                        should_release = True
                    else:
                        # Check if constraints are different
                        are_equal = current_ag_constraint.keys() == loaded_ag_constraint.keys() and all(
                            th.equal(v1, v2) if isinstance(v1, th.Tensor) and isinstance(v2, th.Tensor) else v1 == v2
                            for v1, v2 in zip(current_ag_constraint.values(), loaded_ag_constraint.values())
                        )
                        should_release = not are_equal

                if should_release:
                    self.release_grasp_immediately(arm=arm)

                # Establish new grasp if needed
                if has_loaded_constraint and (not has_current_constraint or should_release):
                    obj = self.scene.object_registry("prim_path", loaded_ag_constraint["ag_obj_prim_path"])
                    link = obj.links[loaded_ag_constraint["ag_link_prim_path"].split("/")[-1]]
                    contact_pos_global = loaded_ag_constraint["contact_pos"]
                    assert (
                        self.scene is not None
                    ), "Cannot set position and orientation relative to scene without a scene"
                    contact_pos_global, _ = self.scene.convert_scene_relative_pose_to_world(
                        contact_pos_global,
                        th.tensor([0, 0, 0, 1], dtype=th.float32),
                    )
                    self._establish_grasp(arm=arm, ag_data=(obj, link), contact_pos=contact_pos_global)

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        # Serialize the controller states sequentially
        controller_states_flat = th.cat(
            [c.serialize(state=state["controllers"][c_name]) for c_name, c in self._controllers.items()]
        )

        # Concatenate and return
        state_flat = th.cat([state_flat, controller_states_flat])
        if self.is_manipulation:
            # No additional serialization needed if we're using physical grasping
            if self.grasping_mode == "physical":
                return state_flat
        return state_flat

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        # Deserialize the controller states sequentially
        controller_states = dict()
        for c_name, c in self._controllers.items():
            controller_states[c_name], deserialized_items = c.deserialize(state=state[idx:])
            idx += deserialized_items
        state_dict["controllers"] = controller_states

        if self.is_manipulation:
            # No additional deserialization needed if we're using physical grasping
            if self.grasping_mode == "physical":
                return state_dict, idx
        return state_dict, idx

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
            robot_name_components[1] == self.kinematic_tree_identifier.lower()
        ), "Third component of articulation root path (robot name) must contain the class name as the second part"
        # Run super
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

        # Initialize all sensors
        for sensor in self._sensors.values():
            sensor.initialize()

        # Load the observation space for this robot
        self.load_observation_space()

        # Validate this robot configuration
        self._validate_configuration()

        self._reset_joint_pos_aabb_extent = self.aabb_extent

        if self.is_manipulation:
            # make eef link not visible
            for arm in self.arm_names:
                self._links[self.eef_link_names[arm]].visible = False

            # Infer relevant link properties, e.g.: fingertip location, AG grasping points
            # We use a try / except to maintain backwards-compatibility with robots that do not follow our
            # OG-specified convention
            try:
                self._infer_finger_properties()
            except AssertionError as e:
                log.warning(f"Could not infer relevant finger link properties because:\n\n{e}")

        if self.is_holonomic_base:
            for i, component in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
                joint_name = f"base_footprint_{component}_joint"
                assert joint_name in self.joints, f"Missing base joint: {joint_name}"

                # Set the linear and angular velocity limits for the base joints (the default value is too large)
                if i < 3:
                    self.joints[joint_name].max_velocity = m.MAX_LINEAR_VELOCITY
                else:
                    self.joints[joint_name].max_velocity = m.MAX_ANGULAR_VELOCITY

                # Set the effort limits for the base joints (the default value is too small)
                self.joints[joint_name].max_effort = m.MAX_EFFORT

            # Force the recomputation of this cached property
            del self.control_limits

            # Overwrite with the new control limits
            self._controller_config["base"]["control_limits"]["velocity"] = self.control_limits["velocity"]
            self._controller_config["base"]["control_limits"]["effort"] = self.control_limits["effort"]

            # Reload the controllers to update their command_output_limits and control_limits
            self.reload_controllers(self._controller_config)

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
        if self.is_manipulation:
            # Iterate over all arms
            for arm in self.arm_names:
                # If we have an arm controller, make sure it is a manipulation controller
                if f"arm_{arm}" in self._controllers:
                    assert isinstance(
                        self._controllers["arm_{}".format(arm)], ManipulationController
                    ), "Arm {} controller must be a ManipulationController!".format(arm)

                # If we have a gripper controller, make sure it is a manipulation controller
                if f"gripper_{arm}" in self._controllers:
                    assert isinstance(
                        self._controllers["gripper_{}".format(arm)], GripperController
                    ), "Gripper {} controller must be a GripperController!".format(arm)

        if self.is_locomotion:
            # If we have a base controller, make sure it is a locomotion controller
            if "base" in self._controllers:
                assert isinstance(
                    self._controllers["base"], LocomotionController
                ), "Base controller must be a LocomotionController!"
        if self.is_two_wheel:
            assert (
                len(self.base_control_idx) == 2
            ), "Differential drive can only be used with robot with two base joints!"

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
        dic = th.cat([proprio_dict[obs] for obs in self._proprio_obs]), {}

        return dic

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
        dic = dict(
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

        if self.is_manipulation:
            # Loop over all arms to grab proprio info
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]
            for arm in self.arm_names:
                # Add arm info
                dic["arm_{}_qpos".format(arm)] = joint_positions[self.arm_control_idx[arm]]
                dic["arm_{}_qpos_sin".format(arm)] = th.sin(joint_positions[self.arm_control_idx[arm]])
                dic["arm_{}_qpos_cos".format(arm)] = th.cos(joint_positions[self.arm_control_idx[arm]])
                dic["arm_{}_qvel".format(arm)] = joint_velocities[self.arm_control_idx[arm]]

                # Add eef and grasping info
                eef_pos, eef_quat = ControllableObjectViewAPI.get_link_relative_position_orientation(
                    self.articulation_root_path, self.eef_link_names[arm]
                )
                dic["eef_{}_pos".format(arm)], dic["eef_{}_quat".format(arm)] = (
                    cb.to_torch(eef_pos),
                    cb.to_torch(eef_quat),
                )
                dic["grasp_{}".format(arm)] = th.tensor([self.is_grasping(arm)])
                dic["gripper_{}_qpos".format(arm)] = joint_positions[self.gripper_control_idx[arm]]
                dic["gripper_{}_qvel".format(arm)] = joint_velocities[self.gripper_control_idx[arm]]
        if self.is_articulated_trunk:
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]
            dic["trunk_qpos"] = joint_positions[self.trunk_control_idx]
            dic["trunk_qvel"] = joint_velocities[self.trunk_control_idx]
        if self.is_locomotion:
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]

            # Add base info
            dic["base_qpos"] = joint_positions[self.base_control_idx]
            dic["base_qpos_sin"] = th.sin(joint_positions[self.base_control_idx])
            dic["base_qpos_cos"] = th.cos(joint_positions[self.base_control_idx])
            dic["base_qvel"] = joint_velocities[self.base_control_idx]
        if self.is_two_wheel:
            # Grab wheel joint velocity info
            l_vel, r_vel = ControllableObjectViewAPI.get_joint_velocities(self.articulation_root_path)[
                self.base_control_idx
            ]

            # Compute linear and angular velocities
            lin_vel = (l_vel + r_vel) / 2.0 * self.wheel_radius
            ang_vel = (r_vel - l_vel) / self.wheel_axle_length

            # Add info
            dic["dd_base_lin_vel"] = th.tensor([lin_vel])
            dic["dd_base_ang_vel"] = th.tensor([ang_vel])
        if self.is_active_camera:
            joint_positions = dic["joint_qpos"]
            joint_velocities = dic["joint_qvel"]
            dic["camera_qpos"] = joint_positions[self.camera_control_idx]
            dic["camera_qpos_sin"] = th.sin(joint_positions[self.camera_control_idx])
            dic["camera_qpos_cos"] = th.cos(joint_positions[self.camera_control_idx])
            dic["camera_qvel"] = joint_velocities[self.camera_control_idx]

        return dic

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

    def _infer_finger_properties(self):
        """
        Infers relevant finger properties based on the given finger meshes of the robot

        NOTE: This assumes the given EEF convention for parallel jaw grippers -- i.e.:
        z points in the direction of the fingers, y points in the direction of grasp articulation, and x
        is then inferred automatically
        """
        assert self.is_manipulation
        # Calculate and cache fingertip to eef frame offsets, as well as AG grasping points
        self._eef_to_fingertip_lengths = dict()
        self._default_ag_start_points = dict()
        self._default_ag_end_points = dict()
        for arm, finger_links in self.finger_links.items():
            self._eef_to_fingertip_lengths[arm] = dict()
            eef_link = self.eef_links[arm]
            world_to_eef_tf = T.pose2mat(eef_link.get_position_orientation())
            eef_to_world_tf = T.pose_inv(world_to_eef_tf)

            # Infer parent link for this finger
            finger_parent_link, finger_parent_max_z = None, None
            is_parallel_jaw = len(finger_links) == 2
            assert (
                is_parallel_jaw
            ), "Inferring finger link information can only be done for parallel jaw gripper robots!"
            finger_pts_in_eef_frame = []
            for i, finger_link in enumerate(finger_links):
                # Find parent, and make sure one exists
                parent_prim_path, parent_link = None, None
                for joint in self.joints.values():
                    if finger_link.prim_path == joint.body1:
                        parent_prim_path = joint.body0
                        break
                assert (
                    parent_prim_path is not None
                ), f"Expected articulated parent joint for finger link {finger_link.name} but found none!"
                for link in self.links.values():
                    if parent_prim_path == link.prim_path:
                        parent_link = link
                        break
                assert parent_link is not None, f"Expected parent link located at {parent_prim_path} but found none!"
                # Make sure all fingers share the same parent
                if finger_parent_link is None:
                    finger_parent_link = parent_link
                    finger_parent_pts = finger_parent_link.collision_boundary_points_world
                    assert (
                        finger_parent_pts is not None
                    ), f"Expected finger parent points to be defined for parent link {finger_parent_link.name}, but got None!"
                    # Convert from world frame -> eef frame
                    finger_parent_pts = th.concatenate([finger_parent_pts, th.ones(len(finger_parent_pts), 1)], dim=-1)
                    finger_parent_pts = (finger_parent_pts @ eef_to_world_tf.T)[:, :3]
                    finger_parent_max_z = finger_parent_pts[:, 2].max().item()
                else:
                    assert (
                        finger_parent_link == parent_link
                    ), f"Expected all fingers to have same parent link, but found multiple parents at {finger_parent_link.prim_path} and {parent_link.prim_path}"

                # Calculate this finger's collision boundary points in the world frame
                finger_pts = finger_link.collision_boundary_points_world
                assert (
                    finger_pts is not None
                ), f"Expected finger points to be defined for link {finger_link.name}, but got None!"
                # Convert from world frame -> eef frame
                finger_pts = th.concatenate([finger_pts, th.ones(len(finger_pts), 1)], dim=-1)
                finger_pts = (finger_pts @ eef_to_world_tf.T)[:, :3]
                finger_pts_in_eef_frame.append(finger_pts)

            # Determine how each finger is located relative to the other in the EEF frame along the y-axis
            # This is used to infer which side of each finger's set of points correspond to the "inner" surface
            finger_pts_mean = [finger_pts[:, 1].mean().item() for finger_pts in finger_pts_in_eef_frame]
            first_finger_is_lower_y_finger = finger_pts_mean[0] < finger_pts_mean[1]
            is_lower_y_fingers = [first_finger_is_lower_y_finger, not first_finger_is_lower_y_finger]

            for i, (finger_link, finger_pts, is_lower_y_finger) in enumerate(
                zip(finger_links, finger_pts_in_eef_frame, is_lower_y_fingers)
            ):
                # Since we know the EEF frame always points with z outwards towards the fingers, the outer-most point /
                # fingertip is the maximum z value
                finger_max_z = finger_pts[:, 2].max().item()
                assert (
                    finger_max_z > 0
                ), f"Expected positive fingertip to eef frame offset for link {finger_link.name}, but got: {finger_max_z}. Does the EEF frame z-axis point in the direction of the fingers?"
                self._eef_to_fingertip_lengths[arm][finger_link.name] = finger_max_z

                # Now, only keep points that are above the parent max z by 20% for inferring y values
                finger_range = finger_max_z - finger_parent_max_z
                valid_idxs = th.where(
                    finger_pts[:, 2] > (finger_parent_max_z + finger_range * m.MIN_AG_DEFAULT_GRASP_POINT_PROP)
                )[0]
                finger_pts = finger_pts[valid_idxs]
                # Infer which side of the gripper corresponds to the inner side (i.e.: the side that touches between the
                # two fingers
                # We use the heuristic that given a set of points defining a gripper finger, we assume that it's one
                # of (y_min, y_max) over all points, with the selection being chosen by inferring which of the limits
                # corresponds to the inner side of the finger.
                # This is the upper side of the y values if this finger is the lower finger, else the lower side
                # of the y values
                y_min, y_max = finger_pts[:, 1].min(), finger_pts[:, 1].max()
                y_offset = y_max if is_lower_y_finger else y_min
                y_sign = 1.0 if is_lower_y_finger else -1.0

                # Compute the default grasping points for this finger
                # For now, we only have a strong heuristic defined for parallel jaw grippers, which assumes that
                # there are exactly 2 fingers
                # In this case, this is defined as the x2 (x,y,z) tuples where:
                # z - the +/-40% from the EEF frame, bounded by the 20% and 100% length between the range from
                #       [finger_parent_max_z, finger_max_z]
                #       This is synonymous to inferring the length of the finger (lower bounded by the gripper base,
                #       assumed to be the parent link), and then taking the values +/-%, bounded by the MIN% and MAX%
                #       along its length
                # y - the value closest to the edge of the finger surface in the direction of +/- EEF y-axis.
                #       This assumes that each individual finger lies completely on one side of the EEF y-axis
                # x - 0. This assumes that the EEF axis evenly splits each finger symmetrically on each side
                # (x,y,z,1) -- homogenous form for efficient transforming into finger link frame
                z_lower = max(
                    finger_parent_max_z + finger_range * m.MIN_AG_DEFAULT_GRASP_POINT_PROP,
                    -finger_range * m.AG_DEFAULT_GRASP_POINT_Z_PROP,
                )
                z_upper = min(
                    finger_parent_max_z + finger_range * m.MAX_AG_DEFAULT_GRASP_POINT_PROP,
                    finger_range * m.AG_DEFAULT_GRASP_POINT_Z_PROP,
                )
                # We want to ensure the z value is symmetric about the EEF z frame, so make sure z_lower is negative
                # and z_upper is positive, and use +/- the absolute minimum value between the two
                assert (
                    z_lower < 0 and z_upper > 0
                ), f"Expected computed z_lower / z_upper bounds for finger grasping points to be negative / positive, but instead got: {z_lower}, {z_upper}"
                z_offset = min(abs(z_lower), abs(z_upper))

                grasp_pts = th.tensor(
                    [
                        [
                            0,
                            y_offset + 0.002 * y_sign,
                            -z_offset,
                            1,
                        ],
                        [
                            0,
                            y_offset + 0.002 * y_sign,
                            z_offset,
                            1,
                        ],
                    ]
                )
                # Convert the grasping points from the EEF frame -> finger frame
                finger_to_world_tf = T.pose_inv(T.pose2mat(finger_link.get_position_orientation()))
                finger_to_eef_tf = finger_to_world_tf @ world_to_eef_tf
                grasp_pts = (grasp_pts @ finger_to_eef_tf.T)[:, :3]
                grasping_points = [
                    GraspingPoint(link_name=finger_link.body_name, position=grasp_pt) for grasp_pt in grasp_pts
                ]
                if i == 0:
                    # Start point
                    self._default_ag_start_points[arm] = grasping_points
                else:
                    # End point
                    self._default_ag_end_points[arm] = grasping_points

        # For each grasping point, if we're in DEBUG mode, visualize with spheres
        if gm.DEBUG:
            for ag_points in (self.assisted_grasp_start_points, self.assisted_grasp_end_points):
                for arm_ag_points in ag_points.values():
                    # Skip if None exist
                    if arm_ag_points is None:
                        continue
                    # For each ag point, generate a small sphere at that point
                    for i, arm_ag_point in enumerate(arm_ag_points):
                        link = self.links[arm_ag_point.link_name]
                        local_pos = arm_ag_point.position
                        vis_mesh_prim_path = f"{link.prim_path}/ag_point_{i}"
                        create_primitive_mesh(
                            prim_path=vis_mesh_prim_path,
                            extents=0.005,
                            primitive_type="Sphere",
                        )
                        vis_geom = VisualGeomPrim(
                            relative_prim_path=absolute_prim_path_to_scene_relative(
                                scene=self.scene,
                                absolute_prim_path=vis_mesh_prim_path,
                            ),
                            name=f"ag_point_{i}",
                        )
                        vis_geom.load(self.scene)
                        vis_geom.set_position_orientation(
                            position=local_pos,
                            frame="parent",
                        )

    def is_grasping(self, arm="default", candidate_obj=None):
        """
        Returns True if the robot is grasping the target option @candidate_obj or any object if @candidate_obj is None.

        Args:
            arm (str): specific arm to check for grasping. Default is "default" which corresponds to the first entry
                in self.arm_names
            candidate_obj (StatefulObject or None): object to check if this robot is currently grasping. If None, then
                will be a general (object-agnostic) check for grasping.
                Note: if self.grasping_mode is "physical", then @candidate_obj will be ignored completely

        Returns:
            IsGraspingState: For the specific manipulator appendage, returns IsGraspingState.TRUE if it is grasping
                (potentially @candidate_obj if specified), IsGraspingState.FALSE if it is not grasping,
                and IsGraspingState.UNKNOWN if unknown.
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        if self.grasping_mode != "physical":
            is_grasping_obj = (
                self._ag_obj_in_hand[arm] is not None
                if candidate_obj is None
                else self._ag_obj_in_hand[arm] == candidate_obj
            )
            is_grasping = (
                IsGraspingState.TRUE
                if is_grasping_obj and self._ag_release_counter[arm] is None
                else IsGraspingState.FALSE
            )
        else:
            # Infer from the gripper controller the state
            is_grasping = self._controllers["gripper_{}".format(arm)].is_grasping()
            # If candidate obj is not None, we also check to see if our fingers are in contact with the object
            if is_grasping == IsGraspingState.TRUE and candidate_obj is not None:
                finger_links = {link for link in self.finger_links[arm]}
                if len(candidate_obj.states[ContactBodies].get_value().intersection(finger_links)) == 0:
                    is_grasping = IsGraspingState.FALSE
        return is_grasping

    def _find_gripper_contacts(self, arm="default", return_contact_positions=False):
        """
        Specific for Manipulation Robot
        For arm @arm, calculate any body IDs and corresponding link IDs that are not part of the robot
        itself that are in contact with any of this arm's gripper's fingers
        Args:
            arm (str): specific arm whose gripper will be checked for contact. Default is "default" which
                corresponds to the first entry in self.arm_names
            return_contact_positions (bool): if True, will additionally return the contact (x,y,z) position
        Returns:
            2-tuple:
                - set: set of unique contact prim_paths that are not the robot self-collisions.
                    If @return_contact_positions is True, then returns (prim_path, pos), where pos is the contact
                    (x,y,z) position
                    Note: if no objects that are not the robot itself are intersecting, the set will be empty.
                - dict: dictionary mapping unique contact objects defined by the contact prim_path to
                    set of unique robot link prim_paths that it is in contact with
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm

        # Get robot finger links
        finger_paths = set([link.prim_path for link in self.finger_links[arm]])

        # Get robot links
        link_paths = set(self.link_prim_paths)

        if not return_contact_positions:
            raw_contact_data = {
                (row, col)
                for row, col in GripperRigidContactAPI.get_contact_pairs(self.scene.idx, column_prim_paths=finger_paths)
                if row not in link_paths
            }
        else:
            raw_contact_data = {
                (row, col, point)
                for row, col, force, point, normal, sep in GripperRigidContactAPI.get_contact_data(
                    self.scene.idx, column_prim_paths=finger_paths
                )
                if row not in link_paths
            }

        # Translate to robot contact data
        robot_contact_links = dict()
        contact_data = set()
        for con_data in raw_contact_data:
            if not return_contact_positions:
                other_contact, link_contact = con_data
                contact_data.add(other_contact)
            else:
                other_contact, link_contact, point = con_data
                contact_data.add((other_contact, point))
            if other_contact not in robot_contact_links:
                robot_contact_links[other_contact] = set()
            robot_contact_links[other_contact].add(link_contact)

        return contact_data, robot_contact_links

    def _release_grasp(self, arm="default"):
        """
        Magic action to release this robot's grasp on an object

        Args:
            arm (str): specific arm whose grasp will be released.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm

        # Remove joint and filtered collision restraints
        delete_or_deactivate_prim(self._ag_obj_constraint_params[arm]["ag_joint_prim_path"])
        self._ag_obj_constraints[arm] = None
        self._ag_obj_constraint_params[arm] = {}
        self._ag_freeze_gripper[arm] = False
        self._ag_release_counter[arm] = 0

    def release_grasp_immediately(self, arm="default"):
        """
        Magic action to release this robot's grasp for one arm.
        As opposed to @_release_grasp, this method would bypass the release window mechanism and immediately release.
        """
        assert self.is_manipulation
        if self._ag_obj_constraints[arm] is not None:
            self._release_grasp(arm=arm)
            self._ag_release_counter[arm] = int(math.ceil(m.RELEASE_WINDOW / og.sim.get_sim_step_dt()))
            self._handle_release_window(arm=arm)
            assert not self._ag_obj_in_hand[arm], "Object still in ag list after release!"
            # TODO: Verify not needed!
            # for finger_link in self.finger_links[arm]:
            #     finger_link.remove_filtered_collision_pair(prim=self._ag_obj_in_hand[arm])

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
        if self.is_two_wheel:
            action = th.zeros(self.action_dim)
            assert isinstance(
                self._controllers["base"], DifferentialDriveController
            ), "Only DifferentialDriveController is supported!"
            action[self.base_action_idx] = th.tensor([teleop_action.base[0], teleop_action.base[2]]).float() * 0.3
            return action

        if self.is_holonomic_base:
            action = th.zeros(self.action_dim)
            hands = ["left", "right"] if self.n_arms == 2 else ["right"]
            for i, hand in enumerate(hands):
                arm_name = self.arm_names[i]
                arm_action = th.tensor(teleop_action[hand]).float()
                # arm action
                assert isinstance(self._controllers[f"arm_{arm_name}"], InverseKinematicsController) or isinstance(
                    self._controllers[f"arm_{arm_name}"], OperationalSpaceController
                ), f"Only IK and OSC controllers are supported for arm {arm_name}!"
                target_pos, target_orn = arm_action[:3], T.quat2axisangle(T.euler2quat(arm_action[3:6]))
                action[self.arm_action_idx[arm_name]] = th.cat((target_pos, target_orn))
                # gripper action
                assert isinstance(
                    self._controllers[f"gripper_{arm_name}"], MultiFingerGripperController
                ), f"Only MultiFingerGripperController is supported for gripper {arm_name}!"
                action[self.gripper_action_idx[arm_name]] = arm_action[6]
            action[self.base_action_idx] = th.tensor(teleop_action.base).float()
            return action

        if self.is_manipulation:
            action = th.zeros(self.action_dim)
            hands = ["left", "right"] if self.n_arms == 2 else ["right"]
            for i, hand in enumerate(hands):
                arm_name = self.arm_names[i]
                arm_action = th.tensor(teleop_action[hand]).float()
                # arm action
                assert isinstance(self._controllers[f"arm_{arm_name}"], InverseKinematicsController) or isinstance(
                    self._controllers[f"arm_{arm_name}"], OperationalSpaceController
                ), f"Only IK and OSC controllers are supported for arm {arm_name}!"
                target_pos, target_orn = arm_action[:3], T.quat2axisangle(T.euler2quat(arm_action[3:6]))
                action[self.arm_action_idx[arm_name]] = th.cat((target_pos, target_orn))
                # gripper action
                assert isinstance(
                    self._controllers[f"gripper_{arm_name}"], MultiFingerGripperController
                ), f"Only MultiFingerGripperController is supported for gripper {arm_name}!"
                action[self.gripper_action_idx[arm_name]] = arm_action[6]
                return action

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
        obs_keys = []
        if self.is_manipulation:
            for arm in self.arm_names:
                obs_keys += [
                    "arm_{}_qpos_sin".format(arm),
                    "arm_{}_qpos_cos".format(arm),
                    "eef_{}_pos".format(arm),
                    "eef_{}_quat".format(arm),
                    "gripper_{}_qpos".format(arm),
                    "grasp_{}".format(arm),
                ]
            return obs_keys
        if self.is_locomotion:
            obs_keys += ["base_qpos_sin", "base_qpos_cos", "robot_lin_vel", "robot_ang_vel"]
        if self.is_articulated_trunk:
            obs_keys += ["trunk_qpos", "trunk_qvel"]
        if self.is_two_wheel:
            obs_keys += ["dd_base_lin_vel", "dd_base_ang_vel"]
        if self.is_active_camera:
            obs_keys += ["camera_qpos_sin", "camera_qpos_cos"]
        return obs_keys

    @property
    def usd_path(self):
        # Check top-level usd_path
        if self._definition.usd_path:
            return os.path.join(get_dataset_path("omnigibson-robot-assets"), self._definition.usd_path)
        # Check end-effector specific usd_path
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.usd_path:
                return os.path.join(get_dataset_path("omnigibson-robot-assets"), eef_def.usd_path)

        # By default, sets the standardized path
        model = self.model.lower()
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{model}/usd/{model}.usda")

    @property
    def urdf_path(self):
        """
        Returns:
            str: file path to the robot urdf file.
        """
        # Check top-level urdf_path
        if self._definition.urdf_path:
            return os.path.join(get_dataset_path("omnigibson-robot-assets"), self._definition.urdf_path)
        # Check end-effector specific urdf_path
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def:
                assert not eef_def.not_support_urdf, "Robot doesn't support URDF."
                if eef_def.urdf_path:
                    return os.path.join(get_dataset_path("omnigibson-robot-assets"), eef_def.urdf_path)

        # By default, sets the standardized path
        model = self.model.lower()
        return os.path.join(get_dataset_path("omnigibson-robot-assets"), f"models/{model}/urdf/{model}.urdf")

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
        if self._definition.base_footprint_link_name:
            return self._definition.base_footprint_link_name
        if self.is_holonomic_base:
            raise NotImplementedError("base_footprint_link_name is not implemented for HolonomicBaseRobot")
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
    def _raw_controller_order(self):
        """
        Returns:
            list: Raw ordering of the actions, corresponding to the controllers. e.g., ["base", "arm", "gripper"],
                to denote that the action vector should be interpreted as first the base action, then arm command, then
                gripper command. Note that external users should query @controller_order, which is the post-processed
                ordering of actions, which may be a subset of the controllers due to some controllers subsuming others
                (e.g.: arm controller subsuming the trunk controller if using IK)
        """
        # Check top-level raw_controller_order
        if self._definition.raw_controller_order:
            return self._definition.raw_controller_order

        # Fall back to defaults based on capabilities
        controllers = []
        if self.is_manipulation:
            for arm in self.arm_names:
                controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]
            return controllers
        if self.is_active_camera:
            return ["camera"]
        if self.is_locomotion:
            return ["base"]

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
    def _default_joint_pos(self):
        """
        Returns:
            n-array: Default joint positions for this robot
        """
        # Check top-level default_joint_pos
        if self._definition.default_joint_pos:
            return self._convert_yaml_list_to_tensor(self._definition.default_joint_pos)
        # Check end-effector specific default_joint_pos
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.default_joint_pos:
                return self._convert_yaml_list_to_tensor(eef_def.default_joint_pos)

        if self.is_mobile_manipulation:
            return (
                self.tucked_default_joint_pos if self.default_reset_mode == "tuck" else self.untucked_default_joint_pos
            )

    @property
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
        cfg = {}
        if self.is_manipulation:
            arm_ik_configs = self._default_arm_ik_controller_configs
            arm_osc_configs = self._default_arm_osc_controller_configs
            arm_joint_configs = self._default_arm_joint_controller_configs
            arm_null_joint_configs = self._default_arm_null_joint_controller_configs
            gripper_pj_configs = self._default_gripper_multi_finger_controller_configs
            gripper_joint_configs = self._default_gripper_joint_controller_configs
            gripper_null_configs = self._default_gripper_null_controller_configs

            # Add arm and gripper defaults, per arm
            for arm in self.arm_names:
                cfg["arm_{}".format(arm)] = {
                    arm_ik_configs[arm]["name"]: arm_ik_configs[arm],
                    arm_osc_configs[arm]["name"]: arm_osc_configs[arm],
                    arm_joint_configs[arm]["name"]: arm_joint_configs[arm],
                    arm_null_joint_configs[arm]["name"]: arm_null_joint_configs[arm],
                }
                cfg["gripper_{}".format(arm)] = {
                    gripper_pj_configs[arm]["name"]: gripper_pj_configs[arm],
                    gripper_joint_configs[arm]["name"]: gripper_joint_configs[arm],
                    gripper_null_configs[arm]["name"]: gripper_null_configs[arm],
                }
        if self.is_locomotion:
            # Add supported base controllers
            cfg["base"] = {
                self._default_base_joint_controller_config["name"]: self._default_base_joint_controller_config,
                self._default_base_null_joint_controller_config[
                    "name"
                ]: self._default_base_null_joint_controller_config,
            }
        if self.is_holonomic_base:
            # Add supported base controllers
            cfg["base"] = {
                self._default_holonomic_base_joint_controller_config[
                    "name"
                ]: self._default_holonomic_base_joint_controller_config,
                self._default_base_null_joint_controller_config[
                    "name"
                ]: self._default_base_null_joint_controller_config,
            }
        if self.is_articulated_trunk:
            cfg["trunk"] = {
                self._default_trunk_joint_controller_config["name"]: self._default_trunk_joint_controller_config,
                self._default_trunk_null_joint_controller_config[
                    "name"
                ]: self._default_trunk_null_joint_controller_config,
                self._default_trunk_ik_controller_config["name"]: self._default_trunk_ik_controller_config,
                self._default_trunk_osc_controller_config["name"]: self._default_trunk_osc_controller_config,
            }
        if self.is_two_wheel:
            cfg["base"][self._default_base_differential_drive_controller_config["name"]] = (
                self._default_base_differential_drive_controller_config
            )
        if self.is_active_camera:
            cfg["camera"] = {
                self._default_camera_joint_controller_config["name"]: self._default_camera_joint_controller_config,
                self._default_camera_null_joint_controller_config[
                    "name"
                ]: self._default_camera_null_joint_controller_config,
            }

        return cfg

    def _get_assisted_grasp_joint_type(self, ag_obj, ag_link):
        """
        Check whether an object @obj can be grasped. If so, return the joint type to use for assisted grasping.
        Otherwise, return None.

        Args:
            ag_obj (BaseObject): Object targeted for an assisted grasp
            ag_link (RigidDynamicPrim): Link of the object to be grasped

        Returns:
            (None or str): If obj can be grasped, returns the joint type to use for assisted grasping.
        """
        assert self.is_manipulation
        # Deny objects that are too heavy and are not a non-base link of a fixed-base object)
        mass = ag_link.mass
        if mass > m.ASSIST_GRASP_MASS_THRESHOLD and not (ag_obj.fixed_base and ag_link != ag_obj.root_link):
            return None

        # Otherwise, compute the joint type. We use a fixed joint unless the link is a non-fixed link.
        # A link is non-fixed if it has any non-fixed parent joints.
        joint_type = "FixedJoint"
        for edge in nx.edge_dfs(ag_obj.articulation_tree, ag_link.body_name, orientation="reverse"):
            joint = ag_obj.articulation_tree.edges[edge[:2]]
            if joint["joint_type"] != JointType.JOINT_FIXED:
                joint_type = "SphericalJoint"
                break

        return joint_type

    def _establish_grasp_rigid(self, arm="default", ag_data=None, contact_pos=None):
        """
        Establishes an ag-assisted grasp, if enabled.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
            ag_data (None or 2-tuple): if specified, assisted-grasp object, link tuple (i.e. :(BaseObject, RigidDynamicPrim)).
                Otherwise, does a no-op
            contact_pos (None or th.tensor): if specified, contact position to use for grasp.
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return
        ag_obj, ag_link = ag_data
        # Get the appropriate joint type
        joint_type = self._get_assisted_grasp_joint_type(ag_obj, ag_link)
        if joint_type is None:
            return

        if contact_pos is None:
            force_data, _ = self._find_gripper_contacts(arm=arm, return_contact_positions=True)
            for c_link_prim_path, c_contact_pos in force_data:
                if c_link_prim_path == ag_link.prim_path:
                    contact_pos = c_contact_pos
                    break

        assert contact_pos is not None, (
            "contact_pos in self._find_gripper_contacts(return_contact_positions=True) is not found in "
            "self._find_gripper_contacts(return_contact_positions=False). This is likely because "
            "GripperRigidContactAPI.get_contact_pairs and get_contact_data return inconsistent results."
        )

        # Joint frame set at the contact point
        # Need to find distance between robot and contact point in robot link's local frame and
        # ag link and contact point in ag link's local frame
        joint_frame_pos = contact_pos
        joint_frame_orn = th.tensor([0, 0, 0, 1.0])
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        parent_frame_pos, parent_frame_orn = T.relative_pose_transform(
            joint_frame_pos, joint_frame_orn, eef_link_pos, eef_link_orn
        )
        obj_link_pos, obj_link_orn = ag_link.get_position_orientation()
        child_frame_pos, child_frame_orn = T.relative_pose_transform(
            joint_frame_pos, joint_frame_orn, obj_link_pos, obj_link_orn
        )

        # Create the joint
        joint_prim_path = f"{self.eef_links[arm].prim_path}/ag_constraint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=self.eef_links[arm].prim_path,
            body1=ag_link.prim_path,
            enabled=True,
            exclude_from_articulation=True,
            joint_frame_in_parent_frame_pos=parent_frame_pos / self.scale,
            joint_frame_in_parent_frame_quat=parent_frame_orn,
            joint_frame_in_child_frame_pos=child_frame_pos / ag_obj.scale,
            joint_frame_in_child_frame_quat=child_frame_orn,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        assist_force = m.MIN_ASSIST_FORCE + (m.MAX_ASSIST_FORCE - m.MIN_ASSIST_FORCE) * m.ASSIST_FRACTION
        max_force = assist_force if joint_type == "FixedJoint" else assist_force * m.ARTICULATED_ASSIST_FRACTION
        # joint_prim.GetAttribute("physics:breakForce").Set(max_force)

        self._ag_obj_constraint_params[arm] = {
            "ag_obj_prim_path": ag_obj.prim_path,
            "ag_link_prim_path": ag_link.prim_path,
            "ag_joint_prim_path": joint_prim_path,
            "joint_type": joint_type,
            "gripper_pos": self.get_joint_positions()[self.gripper_control_idx[arm]],
            "max_force": max_force,
            "contact_pos": contact_pos,
        }
        self._ag_obj_in_hand[arm] = ag_obj
        self._ag_freeze_gripper[arm] = True

    @property
    def _default_controllers(self):
        """
        Returns:
            dict: Maps object group (e.g. base, arm, etc.) to default controller class name to use
            (e.g. IKController, JointController, etc.)
        """
        controllers = {}

        if self.is_manipulation:
            for arm in self.arm_names:
                controllers["arm_{}".format(arm)] = "JointController"
                controllers["gripper_{}".format(arm)] = "JointController"
        if self.is_locomotion:
            controllers["base"] = "JointController"
        if self.is_holonomic_base:
            controllers["base"] = "HolonomicBaseJointController"
        if self.is_articulated_trunk:
            controllers["trunk"] = "JointController"
        if self.is_two_wheel:
            controllers["base"] = "DifferentialDriveController"
        if self.is_active_camera:
            controllers["camera"] = "JointController"

        # Override with config-specified defaults
        if self._definition.default_controllers:
            for key, value in self._definition.default_controllers.items():
                controllers[key] = value

        return controllers

    @property
    def grasping_mode(self):
        """
        Grasping mode of this robot. Is one of AG_MODES

        Returns:
            str: Grasping mode for this robot
        """
        assert self.is_manipulation
        return self._grasping_mode

    @property
    def n_arms(self):
        """
        Specific for Manipulation Robot.
        Returns:
            int: Number of arms this robot has. Returns 1 by default
        """
        if self._definition.manipulation:
            return self._definition.manipulation.n_arms
        return 1

    @property
    def arm_names(self):
        """
        Specific for Manipulation Robot.
        Returns:
            list of str: List of arm names for this robot. Should correspond to the keys used to index into
                arm- and gripper-related dictionaries, e.g.: eef_link_names, finger_link_names, etc.
                Default is string enumeration based on @self.n_arms.
        """
        if self._definition.manipulation and self._definition.manipulation.arm_names:
            return self._definition.manipulation.arm_names
        return [str(i) for i in range(self.n_arms)]

    @property
    def default_arm(self):
        """
        Returns:
            str: Default arm name for this robot, corresponds to the first entry in @arm_names by default
        """
        assert self.is_manipulation
        return self.arm_names[0]

    @property
    def arm_action_idx(self):
        assert self.is_manipulation
        arm_action_idx = {}
        for arm_name in self.arm_names:
            controller_idx = self.controller_order.index(f"arm_{arm_name}")
            action_start_idx = sum(
                [self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)]
            )
            arm_action_idx[arm_name] = th.arange(
                action_start_idx, action_start_idx + self.controllers[f"arm_{arm_name}"].command_dim
            )
        return arm_action_idx

    @property
    def gripper_action_idx(self):
        assert self.is_manipulation
        gripper_action_idx = {}
        for arm_name in self.arm_names:
            controller_idx = self.controller_order.index(f"gripper_{arm_name}")
            action_start_idx = sum(
                [self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)]
            )
            gripper_action_idx[arm_name] = th.arange(
                action_start_idx, action_start_idx + self.controllers[f"gripper_{arm_name}"].command_dim
            )
        return gripper_action_idx

    @cached_property
    def arm_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding arm link names,
                should correspond to specific link names in this robot's underlying model file

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        assert self.is_manipulation
        if self._definition.manipulation and self._definition.manipulation.arm_link_names:
            return self._definition.manipulation.arm_link_names
        return {}

    @cached_property
    def arm_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding arm joint names,
                should correspond to specific joint names in this robot's underlying model file

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        assert self.is_manipulation
        if self._definition.manipulation and self._definition.manipulation.arm_joint_names:
            return self._definition.manipulation.arm_joint_names
        return {}

    @cached_property
    def eef_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to corresponding name of the EEF link,
                should correspond to specific link name in this robot's underlying model file
        """
        assert self.is_manipulation
        # Check manipulation definition
        if self._definition.manipulation and self._definition.manipulation.eef_link_names:
            return self._definition.manipulation.eef_link_names
        # Check end-effector specific
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.eef_link_names:
                return eef_def.eef_link_names
        raise ValueError(f"eef_link_names not found in model definition for model={self.model}")

    @cached_property
    def gripper_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of link names corresponding to
                this robot's gripper. Should be mutual exclusive from self.arm_link_names and self.finger_link_names!

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        assert self.is_manipulation
        if self._definition.manipulation and self._definition.manipulation.gripper_link_names:
            return self._definition.manipulation.gripper_link_names
        raise NotImplementedError

    @cached_property
    def finger_link_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of link names corresponding to
                this robot's fingers

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding idxs.
        """
        assert self.is_manipulation
        # Check manipulation definition
        if self._definition.manipulation and self._definition.manipulation.finger_link_names:
            return self._definition.manipulation.finger_link_names
        # Check end-effector specific
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.finger_link_names:
                return eef_def.finger_link_names
        return {}

    @cached_property
    def finger_joint_names(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to array of joint names corresponding to
                this robot's fingers.

                Note: the ordering within the dictionary is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        assert self.is_manipulation
        # Check manipulation definition
        if self._definition.manipulation and self._definition.manipulation.finger_joint_names:
            return self._definition.manipulation.finger_joint_names
        # Check end-effector specific
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.finger_joint_names:
                return eef_def.finger_joint_names
        return {}

    @cached_property
    def arm_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to indices in low-level control
                vector corresponding to arm joints.
        """
        assert self.is_manipulation
        idxs = {
            arm: th.tensor([list(self.joints.keys()).index(name) for name in self.arm_joint_names[arm]])
            for arm in self.arm_names
        }
        if self._definition.manipulation and self._definition.manipulation.add_combined_arm_control_idx:
            idxs["combined"] = th.sort(th.cat([val for val in idxs.values()]))[0]
        return idxs

    @cached_property
    def gripper_control_idx(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to indices in low-level control
                vector corresponding to gripper joints.
        """
        assert self.is_manipulation
        return {
            arm: th.tensor([list(self.joints.keys()).index(name) for name in self.finger_joint_names[arm]])
            for arm in self.arm_names
        }

    @cached_property
    def arm_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's links
        """
        assert self.is_manipulation
        return {arm: [self._links[link] for link in self.arm_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def eef_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot link corresponding to that arm's
                eef link. NOTE: These links always have a canonical local orientation frame -- assuming a parallel jaw
                eef morphology, it is assumed that the eef z-axis points out from the tips of the fingers, the y-axis
                points from the left finger to the right finger, and the x-axis inferred programmatically
        """
        assert self.is_manipulation
        return {arm: self._links[self.eef_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def gripper_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's gripper links
        """
        assert self.is_manipulation
        return {arm: [self._links[link] for link in self.gripper_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def finger_links(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot links corresponding to
                that arm's finger links
        """
        assert self.is_manipulation
        return {arm: [self._links[link] for link in self.finger_link_names[arm]] for arm in self.arm_names}

    @cached_property
    def finger_joints(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to robot joints corresponding to
                that arm's finger joints
        """
        assert self.is_manipulation
        return {arm: [self._joints[joint] for joint in self.finger_joint_names[arm]] for arm in self.arm_names}

    @property
    def _assisted_grasp_start_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping start points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_end_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        if self._definition.manipulation and self._definition.manipulation.assisted_grasp_start_points:
            return self._convert_to_grasping_points(self._definition.manipulation.assisted_grasp_start_points)
        if self.is_manipulation and hasattr(self, "_ag_start_points"):
            return {self.default_arm: self._ag_start_points}
        return None

    @property
    def assisted_grasp_start_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping start points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_end_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        assert self.is_manipulation
        return (
            self._assisted_grasp_start_points
            if self._assisted_grasp_start_points is not None
            else self._default_ag_start_points
        )

    @property
    def _assisted_grasp_end_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping end points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_start_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        if self._definition.manipulation and self._definition.manipulation.assisted_grasp_end_points:
            return self._convert_to_grasping_points(self._definition.manipulation.assisted_grasp_end_points)
        if self.is_manipulation and hasattr(self, "_ag_end_points"):
            return {self.default_arm: self._ag_end_points}
        return None

    @property
    def assisted_grasp_end_points(self):
        """
        Returns:
            dict: Dictionary mapping individual arm appendage names to array of GraspingPoint tuples,
                composed of (link_name, position) values specifying valid grasping end points located at
                cartesian (x,y,z) coordinates specified in link_name's local coordinate frame.
                These values will be used in conjunction with
                @self.assisted_grasp_start_points to trigger assisted grasps, where objects that intersect
                with any ray starting at any point in @self.assisted_grasp_start_points and terminating at any point in
                @self.assisted_grasp_end_points will trigger an assisted grasp (calculated individually for each gripper
                appendage). By default, each entry returns None, and must be implemented by any robot subclass that
                wishes to use assisted grasping.
        """
        assert self.is_manipulation
        return (
            self._assisted_grasp_end_points
            if self._assisted_grasp_end_points is not None
            else self._default_ag_end_points
        )

    @property
    def eef_to_fingertip_lengths(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to per-finger corresponding z-distance between EEF and each
                respective fingertip
        """
        assert self.is_manipulation
        return self._eef_to_fingertip_lengths

    @property
    def arm_workspace_range(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to a tuple indicating the start and end of the
                angular range of the arm workspace around the Z axis of the robot, where 0 is facing
                forward.
        """
        assert self.is_manipulation
        dic = dict()
        workspace_range = {}
        if self._definition.manipulation and self._definition.manipulation.arm_workspace_range:
            workspace_range = self._definition.manipulation.arm_workspace_range
        for k, v in workspace_range.items():
            dic[k] = th.deg2rad(th.tensor(v, dtype=th.float32))
        return dic

    def get_eef_pose(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab eef pose. Default is "default" which corresponds to the first entry
                in self.arm_names

        Returns:
            2-tuple: End-effector pose, in (pos, quat) format, corresponding to arm @arm
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        return self._links[self.eef_link_names[arm]].get_position_orientation()

    def get_eef_position(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab eef position. Default is "default" which corresponds to the first entry
                in self.arm_names

        Returns:
            3-array: (x,y,z) global end-effector Cartesian position for this robot's end-effector corresponding
                to arm @arm
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        return self.get_eef_pose(arm=arm)[0]

    def get_eef_orientation(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab eef orientation. Default is "default" which corresponds to the first entry
                in self.arm_names

        Returns:
            3-array: (x,y,z,w) global quaternion orientation for this robot's end-effector corresponding
                to arm @arm
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        return self.get_eef_pose(arm=arm)[1]

    def get_relative_eef_pose(self, arm="default", mat=False):
        """
        Args:
            arm (str): specific arm to grab eef pose. Default is "default" which corresponds to the first entry
                in self.arm_names
            mat (bool): whether to return pose in matrix form (mat=True) or (pos, quat) tuple (mat=False)

        Returns:
            2-tuple or (4, 4)-array: End-effector pose, either in 4x4 homogeneous
                matrix form (if @mat=True) or (pos, quat) tuple (if @mat=False), corresponding to arm @arm
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        eef_link_pose = self.eef_links[arm].get_position_orientation()
        base_link_pose = self.get_position_orientation()
        pose = T.relative_pose_transform(*eef_link_pose, *base_link_pose)
        return T.pose2mat(pose) if mat else pose

    def get_relative_eef_position(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef pos.
                Default is "default" which corresponds to the first entry in self.arm_names


        Returns:
            3-array: (x,y,z) Cartesian position of end-effector relative to robot base frame
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        return self.get_relative_eef_pose(arm=arm)[0]

    def get_relative_eef_orientation(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef orientation.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            4-array: (x,y,z,w) quaternion orientation of end-effector relative to robot base frame
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        return self.get_relative_eef_pose(arm=arm)[1]

    def get_relative_eef_lin_vel(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef linear velocity.
                Default is "default" which corresponds to the first entry in self.arm_names


        Returns:
            3-array: (x,y,z) Linear velocity of end-effector relative to robot base frame
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        base_link_quat = self.get_position_orientation()[1]
        return T.quat2mat(base_link_quat).T @ self.eef_links[arm].get_linear_velocity()

    def get_relative_eef_ang_vel(self, arm="default"):
        """
        Args:
            arm (str): specific arm to grab relative eef angular velocity.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            3-array: (ax,ay,az) angular velocity of end-effector relative to robot base frame
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        base_link_quat = self.get_position_orientation()[1]
        return T.quat2mat(base_link_quat).T @ self.eef_links[arm].get_angular_velocity()

    def _calculate_in_hand_object_rigid(self, arm="default"):
        """
        Calculates which object to assisted-grasp for arm @arm. Returns an (object_id, link_id) tuple or None
        if no valid AG-enabled object can be found.

        Args:
            arm (str): specific arm to calculate in-hand object for.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            None or 2-tuple: If a valid assisted-grasp object is found, returns the corresponding
                (object, object_link) (i.e.: (BaseObject, RigidDynamicPrim)) pair to the contacted in-hand object.
                Otherwise, returns None
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm

        # If we're not using physical grasping, we check for gripper contact
        if self.grasping_mode != "physical":
            candidates_set, robot_contact_links = self._find_gripper_contacts(arm=arm)
            # If we're using assisted grasping, we further filter candidates via ray-casting
            if self.grasping_mode == "assisted":
                candidates_set_raycast = self._find_gripper_raycast_collisions(arm=arm)
                candidates_set = candidates_set.intersection(candidates_set_raycast)
        else:
            raise ValueError("Invalid grasping mode for calculating in hand object: {}".format(self.grasping_mode))

        # Immediately return if there are no valid candidates
        if len(candidates_set) == 0:
            return None

        # Find the closest object to the gripper center
        gripper_center_pos = self.eef_links[arm].get_position_orientation()[0]

        candidate_data = []
        for prim_path in candidates_set:
            # Calculate position of the object link. Only allow this for objects currently.
            obj_prim_path, link_name = prim_path.rsplit("/", 1)
            candidate_obj = self.scene.object_registry("prim_path", obj_prim_path, None)
            if candidate_obj is None or link_name not in candidate_obj.links:
                continue
            candidate_link = candidate_obj.links[link_name]
            dist = th.norm(candidate_link.get_position_orientation()[0] - gripper_center_pos)
            candidate_data.append((prim_path, dist))

        if not candidate_data:
            return None

        candidate_data = sorted(candidate_data, key=lambda x: x[-1])
        ag_prim_path, _ = candidate_data[0]

        # Make sure the ag_prim_path is not a self collision
        assert ag_prim_path not in self.link_prim_paths, "assisted grasp object cannot be the robot itself!"

        # Make sure at least two fingers are in contact with this object
        robot_contacts = robot_contact_links[ag_prim_path]
        touching_at_least_two_fingers = (
            True
            if self.grasping_mode == "sticky"
            else len({link.prim_path for link in self.finger_links[arm]}.intersection(robot_contacts)) >= 2
        )

        # TODO: Better heuristic, hacky, we assume the parent object prim path is the prim_path minus the last "/" item
        ag_obj_prim_path = "/".join(ag_prim_path.split("/")[:-1])
        ag_obj_link_name = ag_prim_path.split("/")[-1]
        ag_obj = self.scene.object_registry("prim_path", ag_obj_prim_path)

        # Return None if object cannot be assisted grasped or not touching at least two fingers
        if ag_obj is None or not touching_at_least_two_fingers:
            return None

        # Get object and its contacted link
        return ag_obj, ag_obj.links[ag_obj_link_name]

    def _find_gripper_raycast_collisions(self, arm="default"):
        """
        For arm @arm, calculate any prims that are not part of the robot
        itself that intersect with rays cast between any of the gripper's start and end points

        Args:
            arm (str): specific arm whose gripper will be checked for raycast collisions. Default is "default"
            which corresponds to the first entry in self.arm_names

        Returns:
            set[str]: set of prim path of detected raycast intersections that
            are not the robot itself. Note: if no objects that are not the robot itself are intersecting,
            the set will be empty.
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        # First, make sure start and end grasp points exist (i.e.: aren't None)
        assert (
            self.assisted_grasp_start_points[arm] is not None
        ), "In order to use assisted grasping, assisted_grasp_start_points must not be None!"
        assert (
            self.assisted_grasp_end_points[arm] is not None
        ), "In order to use assisted grasping, assisted_grasp_end_points must not be None!"

        # Iterate over all start and end grasp points and calculate their x,y,z positions in the world frame
        # (per arm appendage)
        # Since we'll be calculating the cartesian cross product between start and end points, we stack the start points
        # by the number of end points and repeat the individual elements of the end points by the number of start points
        n_start_points = len(self.assisted_grasp_start_points[arm])
        n_end_points = len(self.assisted_grasp_end_points[arm])
        start_and_end_points = th.zeros(n_start_points + n_end_points, 3)
        link_positions = th.zeros(n_start_points + n_end_points, 3)
        link_quats = th.zeros(n_start_points + n_end_points, 4)
        idx = 0
        for grasp_start_point in self.assisted_grasp_start_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_start_point.link_name].get_position_orientation()
            link_positions[idx] = link_pos
            link_quats[idx] = link_orn
            start_and_end_points[idx] = grasp_start_point.position
            idx += 1

        for grasp_end_point in self.assisted_grasp_end_points[arm]:
            # Get world coordinates of link base frame
            link_pos, link_orn = self.links[grasp_end_point.link_name].get_position_orientation()
            link_positions[idx] = link_pos
            link_quats[idx] = link_orn
            start_and_end_points[idx] = grasp_end_point.position
            idx += 1

        # Transform start / end points into world frame (batched call for efficiency sake)
        start_and_end_points = link_positions + (T.quat2mat(link_quats) @ start_and_end_points.unsqueeze(-1)).squeeze(
            -1
        )
        # Stack the start points and repeat the end points, and add these values to the raycast dicts
        raycast_startpoints = th.tile(start_and_end_points[:n_start_points], (n_end_points, 1))
        raycast_endpoints = th.repeat_interleave(start_and_end_points[n_start_points:], n_start_points, dim=0) + 1e-8
        ray_data = set()
        # Calculate raycasts from each start point to end point -- this is n_startpoints * n_endpoints total rays
        for result in raytest_batch(raycast_startpoints, raycast_endpoints, only_closest=True):
            if result["hit"]:
                # filter out self body parts (we currently assume that the robot cannot grasp itself)
                if self.prim_path not in result["rigidBody"]:
                    ray_data.add(result["rigidBody"])
        return ray_data

    def _handle_release_window(self, arm="default"):
        """
        Handles releasing an object from arm @arm

        Args:
            arm (str): specific arm to handle release window.
                Default is "default" which corresponds to the first entry in self.arm_names
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm
        self._ag_release_counter[arm] += 1
        time_since_release = self._ag_release_counter[arm] * og.sim.get_sim_step_dt()
        if time_since_release >= m.RELEASE_WINDOW:
            self._ag_obj_in_hand[arm] = None
            self._ag_release_counter[arm] = None

    @property
    def curobo_path(self):
        """
        Returns:
            str or Dict[CuRoboEmbodimentSelection, str]: file path to the robot curobo file or a mapping from
                CuRoboEmbodimentSelection to the file path
        """
        # Check top-level curobo_path
        if self._definition.curobo_path:
            return os.path.join(get_dataset_path("omnigibson-robot-assets"), self._definition.curobo_path)
        # Check end-effector specific curobo_path
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.curobo_path:
                return os.path.join(get_dataset_path("omnigibson-robot-assets"), eef_def.curobo_path)
            else:
                assert False, f"Robot not supported for curobo."

        # Import here to avoid circular imports
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        # By default, sets the standardized path
        model = self.model.lower()
        return {
            emb_sel: os.path.join(
                get_dataset_path("omnigibson-robot-assets"),
                f"models/{model}/curobo/{model}_description_curobo_{emb_sel.value}.yaml",
            )
            for emb_sel in CuRoboEmbodimentSelection
        }

    @property
    def curobo_attached_object_link_names(self):
        """
        Returns:
            Dict[str, str]: mapping from robot eef link names to the link names of the attached objects
        """
        if (
            self._definition.manipulation
            and self._definition.manipulation.eef_support_curobo_attached_object_link_names
        ):
            assert (
                self.end_effector in self._definition.manipulation.eef_support_curobo_attached_object_link_names
            ), f"Robot not supported for curobo."

        assert self.is_manipulation
        # By default, sets the standardized path
        return {eef_link_name: f"attached_object_{eef_link_name}" for eef_link_name in self.eef_link_names.values()}

    @property
    def _default_arm_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config to control that
                robot's arm. Uses velocity control by default.
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": None,
                "motor_type": "position",
                "use_delta_commands": True,
                "use_impedances": False,
            }
        return dic

    @property
    def _default_arm_ik_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config for an
                Inverse kinematics controller to control this robot's arm
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "InverseKinematicsController",
                "task_name": f"eef_{arm}",
                "control_freq": self._control_freq,
                "reset_joint_pos": self.reset_joint_pos,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": (
                    th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                    th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
                ),
                "mode": "pose_delta_ori",
                "smoothing_filter_size": 2,
                "workspace_pose_limiter": None,
            }
        return dic

    @property
    def _default_arm_osc_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config for an
                operational space controller to control this robot's arm
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "OperationalSpaceController",
                "task_name": f"eef_{arm}",
                "control_freq": self._control_freq,
                "reset_joint_pos": self.reset_joint_pos,
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "command_output_limits": (
                    th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                    th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
                ),
                "mode": "pose_delta_ori",
                "workspace_pose_limiter": None,
            }
        return dic

    @property
    def _default_arm_null_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default arm null controller config
                to control this robot's arm i.e. dummy controller
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "NullJointController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.arm_control_idx[arm],
                "default_goal": self.reset_joint_pos[self.arm_control_idx[arm]],
                "use_impedances": False,
            }
        return dic

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default controller config to control
                this robot's multi finger gripper. Assumes robot gripper idx has exactly two elements
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "MultiFingerGripperController",
                "control_freq": self._control_freq,
                "motor_type": "position",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "mode": "binary",
                "limit_tolerance": 0.001,
                "inverted": self._grasping_direction == "upper",
            }
        return dic

    @property
    def _default_gripper_joint_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default gripper joint controller config
                to control this robot's gripper
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "command_output_limits": "default",
                "use_delta_commands": False,
                "use_impedances": False,
            }
        return dic

    @property
    def _default_gripper_null_controller_configs(self):
        """
        Returns:
            dict: Dictionary mapping arm appendage name to default gripper null controller config
                to control this robot's (non-prehensile) gripper i.e. dummy controller
        """
        assert self.is_manipulation
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "NullJointController",
                "control_freq": self._control_freq,
                "motor_type": "velocity",
                "control_limits": self.control_limits,
                "dof_idx": self.gripper_control_idx[arm],
                "default_goal": th.zeros(len(self.gripper_control_idx[arm])),
                "use_impedances": False,
            }
        return dic

    def _calculate_in_hand_object(self, arm="default"):
        assert self.is_manipulation
        if gm.AG_CLOTH:
            return self._calculate_in_hand_object_cloth(arm)
        else:
            return self._calculate_in_hand_object_rigid(arm)

    def _establish_grasp(self, arm="default", ag_data=None, contact_pos=None):
        assert self.is_manipulation
        if gm.AG_CLOTH:
            return self._establish_grasp_cloth(arm, ag_data)
        else:
            return self._establish_grasp_rigid(arm, ag_data, contact_pos)

    def _calculate_in_hand_object_cloth(self, arm="default"):
        """
        Same as _calculate_in_hand_object_rigid, except for cloth. Only one should be used at any given time.

        Calculates which object to assisted-grasp for arm @arm. Returns an (BaseObject, RigidDynamicPrim, th.Tensor) tuple or
        None if no valid AG-enabled object can be found.

        1) Check if the gripper is closed enough
        2) Go through each of the cloth object, and check if its attachment point link position is within the "ghost"
        box volume of the gripper link.

        Only returns the first valid object and ignore the rest.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names

        Returns:
            None or 3-tuple: If a valid assisted-grasp object is found,
                returns the corresponding (object, object_link, attachment_point_position), i.e.
                ((BaseObject, RigidDynamicPrim, th.Tensor)) to the contacted in-hand object. Otherwise, returns None
        """
        assert self.is_manipulation
        # TODO (eric): Assume joint_pos = 0 means fully closed
        GRIPPER_FINGER_CLOSE_THRESHOLD = 0.03
        gripper_finger_pos = self.get_joint_positions()[self.gripper_control_idx[arm]]
        gripper_finger_close = th.sum(gripper_finger_pos) < GRIPPER_FINGER_CLOSE_THRESHOLD
        if not gripper_finger_close:
            return None

        cloth_objs = self.scene.object_registry("prim_type", PrimType.CLOTH)
        if cloth_objs is None:
            return None

        # TODO (eric): Only AG one cloth at any given moment.
        # Returns the first cloth that overlaps with the "ghost" box volume
        for cloth_obj in cloth_objs:
            attachment_point_pos = cloth_obj.links["attachment_point"].get_position_orientation()[0]
            particles_in_volume = self.eef_links[arm].check_points_in_volume([attachment_point_pos])
            if particles_in_volume.sum() > 0:
                return cloth_obj, cloth_obj.links["attachment_point"], attachment_point_pos

        return None

    def _establish_grasp_cloth(self, arm="default", ag_data=None):
        """
        Same as _establish_grasp_cloth, except for cloth. Only one should be used at any given time.
        Establishes an ag-assisted grasp, if enabled.

        Create a fixed joint between the attachment point link of the cloth object and the world.
        In theory, we could have created a fixed joint to the eef link, but omni doesn't support this as the robot has
        an articulation root API attached to it, which is incompatible with the attachment API.

        We also store attachment_point_pos_local as the attachment point position in the eef link frame when the fixed
        joint is created. As the eef link frame changes its pose, we will use attachment_point_pos_local to figure out
        the new attachment_point_pos in the world frame and set the fixed joint to there. See _update_constraint_cloth
        for more details.

        Args:
            arm (str): specific arm to establish grasp.
                Default is "default" which corresponds to the first entry in self.arm_names
            ag_data (None or 3-tuple): If specified, should be the corresponding
                (object, object_link, attachment_point_position), i.e. ((BaseObject, RigidPrim, th.Tensor)) to the]
                contacted in-hand object
        """
        assert self.is_manipulation
        arm = self.default_arm if arm == "default" else arm

        # Return immediately if ag_data is None
        if ag_data is None:
            return

        ag_obj, ag_link, attachment_point_pos = ag_data

        # Find the attachment point position in the eef frame
        eef_link_pos, eef_link_orn = self.eef_links[arm].get_position_orientation()
        attachment_point_pos_local, _ = T.relative_pose_transform(
            attachment_point_pos, th.tensor([0, 0, 0, 1], dtype=th.float32), eef_link_pos, eef_link_orn
        )

        # Create the joint
        joint_prim_path = f"{ag_link.prim_path}/ag_constraint"
        joint_type = "FixedJoint"
        joint_prim = create_joint(
            prim_path=joint_prim_path,
            joint_type=joint_type,
            body0=ag_link.prim_path,
            body1=None,
            enabled=False,
            exclude_from_articulation=True,
            joint_frame_in_child_frame_pos=attachment_point_pos,
        )

        # Save a reference to this joint prim
        self._ag_obj_constraints[arm] = joint_prim

        # Modify max force based on user-determined assist parameters
        # TODO
        max_force = m.MIN_ASSIST_FORCE + (m.MAX_ASSIST_FORCE - m.MIN_ASSIST_FORCE) * m.ASSIST_FRACTION
        # joint_prim.GetAttribute("physics:breakForce").Set(max_force)

        self._ag_obj_constraint_params[arm] = {
            "ag_obj_prim_path": ag_obj.prim_path,
            "ag_link_prim_path": ag_link.prim_path,
            "ag_joint_prim_path": joint_prim_path,
            "joint_type": joint_type,
            "gripper_pos": self.get_joint_positions()[self.gripper_control_idx[arm]],
            "max_force": max_force,
            "attachment_point_pos_local": attachment_point_pos_local,
            "contact_pos": attachment_point_pos,
        }
        self._ag_obj_in_hand[arm] = ag_obj
        self._ag_freeze_gripper[arm] = True

    def _convert_to_math_pi(self, ele):
        """
        Convert string expressions involving pi (e.g., "pi/8", "0.5*pi", "-pi/2")
        to their numeric values.
        """
        if not isinstance(ele, str) or "pi" not in ele:
            return float(ele)

        expression = ele.replace("-pi", f"(-{math.pi})")
        expression = expression.replace("pi", str(math.pi))
        safe_dict = {
            "__builtins__": {},
            "math": math,
            "abs": abs,
            "round": round,
        }
        result = eval(expression, safe_dict)
        return float(result)

    def _convert_yaml_list_to_tensor(self, li):
        ret = []
        for element in li:
            ret.append(self._convert_to_math_pi(element))
        return th.tensor(ret)

    def _get_teleop_rotation_offset(self, prop):
        dic = dict()
        for key, value in prop.items():
            tensor_val = self._convert_yaml_list_to_tensor(value)
            # Check if it's quaternion (4 values) or euler angles (3 values)
            if len(value) == 4:
                # Already a quaternion, return as-is
                dic[key] = tensor_val
            elif len(value) == 3:
                # Euler angles, convert to quaternion
                dic[key] = T.euler2quat(tensor_val)
            else:
                raise ValueError(
                    f"teleop_rotation_offset must have 3 (euler) or 4 (quaternion) values, got {len(tensor_val)}"
                )
        return dic

    @property
    def teleop_rotation_offset(self):
        """
        Rotational offset that will be applied for teleoperation
        such that [0, 0, 0, 1] as action will keep the robot eef pointing at +x axis
        """
        assert self.is_manipulation
        # Check end-effector specific
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.teleop_rotation_offset:
                return self._get_teleop_rotation_offset(eef_def.teleop_rotation_offset)
        # Check manipulation definition
        if self._definition.manipulation and self._definition.manipulation.teleop_rotation_offset:
            return self._get_teleop_rotation_offset(self._definition.manipulation.teleop_rotation_offset)
        return {arm: th.tensor([0, 0, 0, 1]) for arm in self.arm_names}

    @property
    def _default_base_joint_controller_config(self):
        """
        Returns:
            dict: Default base joint controller config to control this robot's base. Uses velocity
                control by default.
        """
        assert self.is_locomotion
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "dof_idx": self.base_control_idx,
            "command_output_limits": "default",
            "use_delta_commands": False,
        }

    @property
    def _default_base_null_joint_controller_config(self):
        """
        Returns:
            dict: Default null joint controller config to control this robot's base i.e. dummy controller
        """
        assert self.is_locomotion
        return {
            "name": "NullJointController",
            "control_freq": self._control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "dof_idx": self.base_control_idx,
            "default_goal": th.zeros(len(self.base_control_idx)),
            "use_impedances": False,
        }

    def move_by(self, delta):
        """
        Move robot base without physics simulation

        Args:
            delta (float):float], (x,y,z) cartesian delta base position
        """
        assert self.is_locomotion
        new_pos = th.tensor(delta) + self.get_position_orientation()[0]
        self.set_position_orientation(position=new_pos)

    def move_forward(self, delta=0.05):
        """
        Move robot base forward without physics simulation

        Args:
            delta (float): delta base position forward
        """
        assert self.is_locomotion
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([delta, 0, 0])))

    def move_backward(self, delta=0.05):
        """
        Move robot base backward without physics simulation

        Args:
            delta (float): delta base position backward
        """
        assert self.is_locomotion
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([-delta, 0, 0])))

    def move_left(self, delta=0.05):
        """
        Move robot base left without physics simulation

        Args:
            delta (float): delta base position left
        """
        assert self.is_locomotion
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([0, -delta, 0])))

    def move_right(self, delta=0.05):
        """
        Move robot base right without physics simulation

        Args:
            delta (float): delta base position right
        """
        assert self.is_locomotion
        self.move_by(T.quat2mat(self.get_position_orientation()[1]).dot(th.tensor([0, delta, 0])))

    def turn_left(self, delta=0.03):
        """
        Rotate robot base left without physics simulation

        Args:
            delta (float): delta angle to rotate the base left
        """
        assert self.is_locomotion
        quat = self.get_position_orientation()[1]
        quat = T.quat_multiply((T.euler2quat(-delta, 0, 0)), quat)
        self.set_position_orientation(orientation=quat)

    def turn_right(self, delta=0.03):
        """
        Rotate robot base right without physics simulation

        Args:
            delta (float): angle to rotate the base right
        """
        assert self.is_locomotion
        quat = self.get_position_orientation()[1]
        quat = T.quat_multiply((T.euler2quat(delta, 0, 0)), quat)
        self.set_position_orientation(orientation=quat)

    @cached_property
    def non_floor_touching_base_links(self):
        assert self.is_locomotion
        return [self.links[name] for name in self.non_floor_touching_base_link_names]

    @cached_property
    def non_floor_touching_base_link_names(self):
        assert self.is_locomotion
        return [self.base_footprint_link_name]

    @cached_property
    def floor_touching_base_links(self):
        assert self.is_locomotion
        return [self.links[name] for name in self.floor_touching_base_link_names]

    @cached_property
    def floor_touching_base_link_names(self):
        assert self.is_locomotion
        return self._definition.locomotion.floor_touching_base_link_names or []

    @property
    def base_action_idx(self):
        assert self.is_locomotion
        controller_idx = self.controller_order.index("base")
        action_start_idx = sum([self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)])
        return th.arange(action_start_idx, action_start_idx + self.controllers["base"].command_dim)

    @property
    def base_joint_names(self):
        """
        Returns:
            list: Array of joint names corresponding to this robot's base joints (e.g.: wheels).

                Note: the ordering within the list is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        assert self.is_locomotion
        return self._definition.locomotion.base_joint_names

    @cached_property
    def base_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to base joints.
        """
        assert self.is_locomotion
        return th.tensor([list(self.joints.keys()).index(name) for name in self.base_joint_names])

    @property
    def _default_holonomic_base_joint_controller_config(self):
        """
        Returns:
            dict: Default base joint controller config to control this robot's base. Uses velocity
                control by default.
        """
        assert self.is_holonomic_base
        return {
            "name": "HolonomicBaseJointController",
            "control_freq": self._control_freq,
            "motor_type": "velocity",
            "control_limits": self.control_limits,
            "dof_idx": self.base_control_idx,
            "command_output_limits": "default",
        }

    @cached_property
    def base_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to the six 1DoF base joints
        """
        assert self.is_holonomic_base
        joints = list(self.joints.keys())
        return th.tensor(
            [joints.index(f"base_footprint_{component}_joint") for component in ["x", "y", "z", "rx", "ry", "rz"]]
        )

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
        if self.is_holonomic_base:
            return self.base_footprint_link.get_position_orientation(frame=frame, clone=clone)
        return super().get_position_orientation()

    def set_linear_velocity(self, velocity: th.Tensor):
        if self.is_holonomic_base:
            # Transform the desired linear velocity from the world frame to the root_link ("base_footprint_x") frame
            # Note that this will also set the target to be the desired linear velocity (i.e. the robot will try to maintain
            # such velocity), which is different from the default behavior of set_linear_velocity for all other objects.
            orn = self.root_link.get_position_orientation()[1]
            velocity_in_root_link = T.quat2mat(orn).T @ velocity
            self.set_joint_velocities(velocity_in_root_link, indices=self.base_idx[:3], drive=False)
        else:
            super().set_linear_velocity(velocity)

    def get_linear_velocity(self) -> th.Tensor:
        if self.is_holonomic_base:
            # Note that the link we are interested in is self.base_footprint_link, not self.root_link
            return self.base_footprint_link.get_linear_velocity()
        else:
            super().get_linear_velocity()

    def set_angular_velocity(self, velocity: th.Tensor) -> None:
        if self.is_holonomic_base:
            # 1e-3 is emperically tuned to be a good value for the time step
            delta_t = 1e-3 / (velocity.norm() + 1e-6)
            delta_mat = T.delta_rotation_matrix(velocity, delta_t)
            base_link_orn = self.get_position_orientation()[1]
            rot_mat = T.quat2mat(base_link_orn)
            desired_mat = delta_mat @ rot_mat
            root_link_orn = self.root_link.get_position_orientation()[1]
            desired_mat_in_root_link = T.quat2mat(root_link_orn).T @ desired_mat
            desired_intrinsic_eulers = T.mat2euler_intrinsic(desired_mat_in_root_link)

            cur_joint_pos = self.get_joint_positions()[self.base_idx[3:]]
            delta_intrinsic_eulers = desired_intrinsic_eulers - cur_joint_pos
            velocity_intrinsic = delta_intrinsic_eulers / delta_t

            self.set_joint_velocities(velocity_intrinsic, indices=self.base_idx[3:], drive=False)
        else:
            super().set_angular_velocity(velocity)

    def get_angular_velocity(self) -> th.Tensor:
        if self.is_holonomic_base:
            # Note that the link we are interested in is self.base_footprint_link, not self.root_link
            return self.base_footprint_link.get_angular_velocity()
        else:
            super().get_angular_velocity()

    def q_to_action(self, q):
        """
        Converts a target joint configuration to an action that can be applied to this object.
        All controllers should be JointController with use_delta_commands=False
        """
        if not self.is_holonomic_base:
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

        action = []
        for name, controller in self.controllers.items():
            assert (
                isinstance(controller, JointController) and not controller.use_delta_commands
            ), f"Controller [{name}] should be a JointController/HolonomicBaseJointController with use_delta_commands=False!"
            command = q[controller.dof_idx]
            if isinstance(controller, HolonomicBaseJointController):
                # For a holonomic base joint controller, the command should be in the robot local frame
                # For orientation, we need to convert the command to a delta angle
                cur_rz_joint_pos = self.get_joint_positions()[self.base_idx][5]
                delta_q = wrap_angle(command[2] - cur_rz_joint_pos)

                # For translation, we need to convert the command to the robot local frame
                body_pose = self.get_position_orientation()
                canonical_pos = th.tensor([command[0], command[1], body_pose[0][2]], dtype=th.float32)
                local_pos = T.relative_pose_transform(canonical_pos, th.tensor([0.0, 0.0, 0.0, 1.0]), *body_pose)[0]
                command = th.tensor([local_pos[0], local_pos[1], delta_q])
            action.append(controller._reverse_preprocess_command(command))
        action = th.cat(action, dim=0)
        assert (
            action.shape[0] == self.action_dim
        ), f"Action should have dimension {self.action_dim}, got {action.shape[0]}"
        return action

    @property
    def tucked_default_joint_pos(self):
        assert self.is_mobile_manipulation
        return self._convert_yaml_list_to_tensor(self._definition.mobile_manipulation.tucked_default_joint_pos)

    @property
    def untucked_default_joint_pos(self):
        assert self.is_mobile_manipulation
        return self._convert_yaml_list_to_tensor(self._definition.mobile_manipulation.untucked_default_joint_pos)

    def tuck(self):
        """
        Immediately set this robot's configuration to be in tucked mode
        """
        assert self.is_mobile_manipulation
        pos = self.tucked_default_joint_pos
        if self.is_holonomic_base:
            pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        self.set_joint_positions(pos)

    def untuck(self):
        """
        Immediately set this robot's configuration to be in untucked mode
        """
        assert self.is_mobile_manipulation
        pos = self.untucked_default_joint_pos
        if self.is_holonomic_base:
            pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
        self.set_joint_positions(pos)

    @cached_property
    def trunk_links(self):
        assert self.is_articulated_trunk
        return [self.links[name] for name in self.trunk_link_names]

    @cached_property
    def trunk_link_names(self):
        assert self.is_articulated_trunk
        if self._definition.is_articulated_trunk:
            return self._definition.articulated_trunk.trunk_link_names
        return []

    @cached_property
    def trunk_joint_names(self):
        assert self.is_articulated_trunk
        if self._definition.articulated_trunk:
            return self._definition.articulated_trunk.trunk_joint_names
        return []

    @cached_property
    def trunk_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to trunk joints.
        """
        assert self.is_articulated_trunk
        return th.tensor([list(self.joints.keys()).index(name) for name in self.trunk_joint_names])

    @property
    def trunk_action_idx(self):
        assert self.is_articulated_trunk
        controller_idx = self.controller_order.index("trunk")
        action_start_idx = sum([self.controllers[self.controller_order[i]].command_dim for i in range(controller_idx)])
        return th.arange(action_start_idx, action_start_idx + self.controllers["trunk"].command_dim)

    @property
    def _default_trunk_ik_controller_config(self):
        """
        Returns:
            dict: Default controller config for an Inverse kinematics controller to control this robot's trunk
        """
        assert self.is_articulated_trunk
        return {
            "name": "InverseKinematicsController",
            "task_name": "trunk",
            "control_freq": self._control_freq,
            "reset_joint_pos": self.reset_joint_pos,
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "command_output_limits": (
                th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
            ),
            "mode": "pose_delta_ori",
            "smoothing_filter_size": 2,
            "workspace_pose_limiter": None,
        }

    @property
    def _default_trunk_osc_controller_config(self):
        """
        Returns:
            dict: Default controller config for an Operational Space controller to control this robot's trunk
        """
        assert self.is_articulated_trunk
        return {
            "name": "OperationalSpaceController",
            "task_name": "trunk",
            "control_freq": self._control_freq,
            "reset_joint_pos": self.reset_joint_pos,
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "command_output_limits": (
                th.tensor([-0.2, -0.2, -0.2, -0.5, -0.5, -0.5]),
                th.tensor([0.2, 0.2, 0.2, 0.5, 0.5, 0.5]),
            ),
            "mode": "pose_delta_ori",
            "workspace_pose_limiter": None,
        }

    @property
    def _default_trunk_joint_controller_config(self):
        """
        Returns:
            dict: Default base joint controller config to control this robot's base. Uses position
                control by default.
        """
        assert self.is_articulated_trunk
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "command_output_limits": None,
            "use_delta_commands": True,
        }

    @property
    def _default_trunk_null_joint_controller_config(self):
        """
        Returns:
            dict: Default null joint controller config to control this robot's base i.e. dummy controller
        """
        assert self.is_articulated_trunk
        return {
            "name": "NullJointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.trunk_control_idx,
            "default_goal": self.reset_joint_pos[self.trunk_control_idx],
            "use_impedances": False,
        }

    @property
    def default_arm_poses(self):
        assert self.has_multiple_arm_poses
        dic = dict()
        if self._definition.mobile_manipulation and self._definition.mobile_manipulation.default_arm_poses:
            for key, value in self._definition.mobile_manipulation.default_arm_poses.items():
                dic[key] = self._convert_yaml_list_to_tensor(value)
        return dic

    def _create_discrete_action_space(self):
        if not self.is_two_wheel:
            raise ValueError(f"{self.model} does not support discrete actions!")

        # Set action list based on controller (joint or DD) used

        # We set straight velocity to be 50% of max velocity for the wheels
        max_wheel_joint_vels = self.control_limits["velocity"][1][self.base_control_idx]
        assert len(max_wheel_joint_vels) == 2, "TwoWheelRobot must only have two base (wheel) joints!"
        assert max_wheel_joint_vels[0] == max_wheel_joint_vels[1], "Both wheels must have the same max speed!"
        wheel_straight_vel = 0.5 * max_wheel_joint_vels[0]
        wheel_rotate_vel = 0.5
        if self._controller_config["base"]["name"] == "JointController":
            action_list = [
                [wheel_straight_vel, wheel_straight_vel],
                [-wheel_straight_vel, -wheel_straight_vel],
                [wheel_rotate_vel, -wheel_rotate_vel],
                [-wheel_rotate_vel, wheel_rotate_vel],
                [0, 0],
            ]
        else:
            # DifferentialDriveController
            lin_vel = wheel_straight_vel * self.wheel_radius
            ang_vel = wheel_rotate_vel * self.wheel_radius * 2.0 / self.wheel_axle_length
            action_list = [
                [lin_vel, 0],
                [-lin_vel, 0],
                [0, ang_vel],
                [0, -ang_vel],
                [0, 0],
            ]

        self.action_list = action_list

        # Return this action space
        return gym.spaces.Discrete(n=len(self.action_list))

    @property
    def _default_base_differential_drive_controller_config(self):
        """
        Returns:
            dict: Default differential drive controller config to
                control this robot's base.
        """
        assert self.is_two_wheel
        return {
            "name": "DifferentialDriveController",
            "control_freq": self._control_freq,
            "wheel_radius": self.wheel_radius,
            "wheel_axle_length": self.wheel_axle_length,
            "control_limits": self.control_limits,
            "dof_idx": self.base_control_idx,
        }

    @property
    def wheel_radius(self):
        """
        Returns:
            float: radius of each wheel at the base, in metric units
        """
        assert self.is_two_wheel
        return self._definition.two_wheel.wheel_radius

    @property
    def wheel_axle_length(self):
        """
        Returns:
            float: perpendicular distance between the robot's two wheels, in metric units
        """
        assert self.is_two_wheel
        return self._definition.two_wheel.wheel_axle_length

    @property
    def _default_camera_joint_controller_config(self):
        """
        Returns:
            dict: Default camera joint controller config to control this robot's camera
        """
        assert self.is_active_camera
        return {
            "name": "JointController",
            "control_freq": self._control_freq,
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
            "command_output_limits": None,
            "motor_type": "position",
            "use_delta_commands": True,
            "use_impedances": False,
        }

    @property
    def _default_camera_null_joint_controller_config(self):
        """
        Returns:
            dict: Default null joint controller config to control this robot's camera i.e. dummy controller
        """
        assert self.is_active_camera
        return {
            "name": "NullJointController",
            "control_freq": self._control_freq,
            "motor_type": "position",
            "control_limits": self.control_limits,
            "dof_idx": self.camera_control_idx,
            "default_goal": self.reset_joint_pos[self.camera_control_idx],
            "use_impedances": False,
        }

    @cached_property
    def camera_joint_names(self):
        """
        Returns:
            list: Array of joint names corresponding to this robot's camera joints.

                Note: the ordering within the list is assumed to be intentional, and is
                directly used to define the set of corresponding control idxs.
        """
        assert self.is_active_camera
        if self._definition.active_camera:
            return self._definition.active_camera.camera_joint_names
        return []

    @cached_property
    def camera_control_idx(self):
        """
        Returns:
            n-array: Indices in low-level control vector corresponding to camera joints.
        """
        assert self.is_active_camera
        return th.tensor([list(self.joints.keys()).index(name) for name in self.camera_joint_names])

    @property
    def disabled_collision_link_names(self):
        return self._definition.disabled_collision_link_names or []

    @property
    def disabled_collision_pairs(self):
        if self.has_end_effector_variants:
            eef_def = self._get_end_effector_definition()
            if eef_def and eef_def.disabled_collision_pairs:
                return eef_def.disabled_collision_pairs
        return self._definition.disabled_collision_pairs or []

    @cached_property
    def manipulation_link_names(self):
        return self._definition.manipulation.manipulation_link_names or []
