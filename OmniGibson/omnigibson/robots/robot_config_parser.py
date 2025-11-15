from pathlib import Path
import importlib
import math
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple

import torch as th
import yaml
from omnigibson.utils.transform_utils import euler2quat
import os
from omnigibson.utils.asset_utils import get_dataset_path

from omnigibson.robots.robot_base import BaseRobot, REGISTERED_ROBOTS
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.mobile_manipulation_robot import MobileManipulationRobot
from omnigibson.robots.untucked_arm_pose_robot import UntuckedArmPoseRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot

CAPABILITY_BASES = {
    "two_wheel": TwoWheelRobot,
    "manipulation": ManipulationRobot,
    "holonomic_base": HolonomicBaseRobot,
    "articulated_trunk": ArticulatedTrunkRobot,
    "active_camera": ActiveCameraRobot,
    "mobile_manipulation": MobileManipulationRobot,
    "untucked_arm_pose": UntuckedArmPoseRobot,
    "locomotion": LocomotionRobot,
}

EXPECTED_PROPS = {
    "two_wheel_locomotion": {"wheel_radius", "wheel_axle_length", "base_joint_names"},
    "untucked_arm_pose": {"arm_link_names","arm_joint_names","eef_link_names","finger_link_names","finger_joint_names"},
    "mobile_manipulation": {"arm_link_names","arm_joint_names","eef_link_names","finger_link_names","finger_joint_names"},
    "manipulation": {"arm_link_names","arm_joint_names","eef_link_names","finger_link_names","finger_joint_names"},
    "locomotion":{"base_joint_names"},
    "holonomic_base": set(),  # Empty set, not empty dict
    "articulated_trunk": {"arm_link_names","arm_joint_names","eef_link_names","finger_link_names","finger_joint_names"},
    "active_camera":{"camera_joint_names"},
}
# Allowed top-level YAML keys and whether they are required
ALLOWED_TOP_KEYS: Dict[str, bool] = {
    "name": True,                # name of the robot class
    "description": False,        # docstring of the robot class
    "module_path": True,         # fully qualified module to expose the class (e.g., omnigibson.robots.robot_configs.fetch)
    "extends": False,            # optional import path "omnigibson.robots.r1:R1" or "omnigibson.robots.r1.R1"
    "capabilities": False,       # list[str] matching CAPABILITY_BASES keys
    "property": False,         # dict[str, Any] exposed via @property returning the literal
    "classproperty": False,   # dict[str, Any] exposed via @cached_property returning the literal
    "init": False,              # dict[str, Any] for __init__ method parameters and default values
}

def _set_post_load(cfg, robot_cls):
    """
    Creates _post_load method based on robot name.
    Only work for tiago, r1 and r1pro for now.
    
    """
    robot_name = cfg.get("name", "").lower()
    
    if robot_name.lower() in ["r1", "r1pro"]:
        def _post_load(self):
            super(robot_cls, self)._post_load()
            # R1 and R1Pro's URDFs still use the mesh type for the collision meshes of the wheels
            # We need to manually set it back to sphere approximation
            for wheel_name in self.floor_touching_base_link_names:
                wheel_link = self.links[wheel_name]
                assert set(wheel_link.collision_meshes) == {"collisions"}, "Wheel link should only have 1 collision!"
                wheel_link.collision_meshes["collisions"].set_collision_approximation("boundingSphere")
        setattr(robot_cls, "_post_load", _post_load)
    
    elif robot_name.lower() == "tiago":
        def _post_load(self):
            super(robot_cls, self)._post_load()
            # The eef gripper links should be visual-only. They only contain a "ghost" box volume 
            # for detecting objects inside the gripper, in order to activate attachments (AG for Cloths).
            for arm in self.arm_names:
                self.eef_links[arm].visual_only = True
                self.eef_links[arm].visible = False
        setattr(robot_cls, "_post_load", _post_load)

def _set_arm_workspace_range(cfg, robot_cls):
    properties = cfg.get("property", {})
    if "arm_workspace_range" not in properties:
        return
    arm_workspace_range = properties["arm_workspace_range"].copy()
    for k in arm_workspace_range:
        arm_workspace_range[k] = th.deg2rad(th.tensor(arm_workspace_range[k], dtype=th.float32))
    
    # Create a property that returns the processed arm_workspace_range
    # This is necessary because arm_workspace_range must be a @property
    # (required by ManipulationRobot's @property definition)
    @property
    def arm_workspace_range_prop(self):
        return arm_workspace_range
    setattr(robot_cls, "arm_workspace_range", arm_workspace_range_prop)

def _set_default_arm_poses(cfg, robot_cls):
    properties = cfg.get("property", {})
    if "default_arm_poses" not in properties:
        return
    default_arm_poses = properties["default_arm_poses"].copy()
    for k in default_arm_poses:
        default_arm_poses[k] = th.tensor(default_arm_poses[k])
    @property
    def default_arm_poses_prop(self):
        return default_arm_poses
    setattr(robot_cls, "default_arm_poses", default_arm_poses_prop)

def _set_end_effector_properties(cfg, robot_cls):
    """
    Handle end-effector-specific properties for A1 and FrankaPanda robots.
    These properties are defined under end-effector keys (e.g., "gripper", "inspire") 
    in the YAML and need to be dynamically selected based on self.end_effector.
    """
    robot_name = cfg.get("name", "").lower()
    if robot_name not in ["a1", "frankapanda"]:
        return
    
    properties = cfg.get("property", {})
    
    # Find all end-effector configs (keys that are not standard property names)
    # Common end-effector names: gripper, inspire, allegro, leap_right, leap_left
    end_effector_names = ["gripper", "inspire", "allegro", "leap_right", "leap_left"]
    end_effector_configs = {
        name: properties[name] 
        for name in end_effector_names 
        if name in properties and isinstance(properties[name], dict)
    }
    
    if not end_effector_configs:
        return
    
    # Get property names from the first end-effector config (assuming all have same keys)
    first_ee_config = next(iter(end_effector_configs.values()))
    prop_names = set(first_ee_config.keys())
    
    # instance attributes, not @property
    instance_attr_props = {
        "_eef_link_names", "_finger_link_names", "_finger_joint_names",
        "_default_robot_model_joint_pos", "_teleop_rotation_offset",
        "_ag_start_points", "_ag_end_points", "_model_name", "_gripper_control_idx"
    }
    
    tensor_props = {"_default_robot_model_joint_pos","_teleop_rotation_offset"}
    grasping_point_props = {"_ag_start_points", "_ag_end_points"}
    
    # Helper function to convert list of [link_name, [x, y, z]] to GraspingPoint objects
    def _convert_to_grasping_points(data):
        result = []
        for item in data:
            link_name, position = item
            result.append(GraspingPoint(link_name=link_name, position=th.tensor(position)))
        return result


    # Store end-effector configs in class for access in __init__ and properties
    robot_cls._end_effector_configs = end_effector_configs
    
    # Modify __init__ to set instance attributes based on end_effector
    original_init = robot_cls.__init__
    
    def __init_with_end_effector(self, *args, **kwargs):
        # Call original __init__ first (which sets self.end_effector)
        original_init(self, *args, **kwargs)
        
        # Get config for current end_effector
        ee_config = end_effector_configs.get(self.end_effector)
        if ee_config is None:
            return
        
        # Set instance attributes based on end_effector config
        for prop_name in prop_names:
            if prop_name not in instance_attr_props:
                continue
            
            value = ee_config.get(prop_name)
            if value is None:
                continue
            
            # Apply special conversions
            if prop_name in tensor_props:
                # Convert list to tensor
                if isinstance(value, list):
                    li = []
                    for ele in value:
                        li.append(float(_convert_to_math_pi(ele)))
                    value = th.tensor(li)
            elif prop_name in grasping_point_props:
                # Convert to GraspingPoint objects
                value = _convert_to_grasping_points(value)
            setattr(self, prop_name, value)
    
    setattr(robot_cls, "__init__", __init_with_end_effector)
    
    # @property that return values based on end_effector
    
    
    for prop_name in prop_names:
        if prop_name in instance_attr_props | grasping_point_props | tensor_props :
            # Skip private properties that are already set as instance attributes
            continue
        
        # Create property that dynamically returns value based on end_effector
        def _make_dynamic_property_factory(prop_name):
            if prop_name in ["usd_path", "urdf_path", "curobo_path"]:
                @property
                def prop_func(self):
                    ee_config = end_effector_configs.get(self.end_effector)
                    path = ee_config[prop_name]
                    path = os.path.join(get_dataset_path("omnigibson-robot-assets"), path)
                    return path
            
            else:
                # Generic property that returns value from end_effector config
                @property
                def prop_func(self):
                    ee_config = end_effector_configs.get(self.end_effector)
                    value=ee_config.get(prop_name)
                    if prop_name in ["_default_joint_pos","teleop_rotation_offset"]:
                        if isinstance(value, list):
                            li = []
                            for ele in value:
                                li.append(float(_convert_to_math_pi(ele)))
                            value = th.tensor(li)
                    return value
            
            return prop_func
        
        setattr(robot_cls, prop_name, _make_dynamic_property_factory(prop_name))
    
    # Add properties for _assisted_grasp_start_points and _assisted_grasp_end_points
    # These are computed from instance attributes set in __init__
    def _assisted_grasp_start_points_getter(self):
        return {self.default_arm: getattr(self, "_ag_start_points", [])}
    
    def _assisted_grasp_end_points_getter(self):
        return {self.default_arm: getattr(self, "_ag_end_points", [])}

    setattr(robot_cls, "_assisted_grasp_start_points", property(_assisted_grasp_start_points_getter))
    setattr(robot_cls, "_assisted_grasp_end_points", property(_assisted_grasp_end_points_getter))
    
def _set_general_properties(cfg, robot_cls):
    """
    Set all properties from the 'property' section of YAML config as @property.
    This handles properties that haven't been specially processed by other functions.
    """
    properties = cfg.get("property", {})
    
    # List of properties that are handled by special functions
    special_properties = {
        "_default_controllers",
        "_default_joint_pos",
        "tucked_default_joint_pos",
        "untucked_default_joint_pos",
        "teleop_rotation_offset",
        "arm_workspace_range",
        "default_arm_poses"
    }
    
    # For A1 and FrankaPanda, skip end_effector-specific configs (sub-dictionaries)
    # These are handled in __init__ based on end_effector parameter
    is_a1_or_franka = cfg.get("name", "") in ["A1", "FrankaPanda"]

    for prop_name, prop_value in properties.items():
        # Skip properties that are handled specially
        if prop_name in special_properties:
            continue
        
        # Skip end_effector-specific configs for A1 and FrankaPanda
        if is_a1_or_franka and (prop_name in ["gripper","inspire", "allegro"] or "leap" in prop_name):
            continue
        
        # Create a property that returns the literal value
        # We need to capture the value in a closure with a default argument
        # to avoid late binding issues in loops
        def _make_property(name, value):
            if prop_name in ["usd_path", "urdf_path", "curobo_path"]:
                @property
                def prop_func(self):
                    path = os.path.join(get_dataset_path("omnigibson-robot-assets"), value)
                    return path
                return prop_func
            
            else:
                @property
                def prop_func(self):
                    return value
                return prop_func
        
        setattr(robot_cls, prop_name, _make_property(prop_name, prop_value))

def _set_classproperties(cfg, robot_cls):
    """
    Set all properties from the 'classproperty' section of YAML config as @classproperty.
    """
    from omnigibson.utils.python_utils import classproperty
    
    classproperties = cfg.get("classproperty", {})
    
    for prop_name, prop_value in classproperties.items():
        # Create a classproperty that returns the literal value
        # We need to capture the value in a closure with a default argument
        # to avoid late binding issues in loops
        def _make_classproperty(name, value):
            @classproperty
            def prop_func(cls):
                return value
            return prop_func
        
        setattr(robot_cls, prop_name, _make_classproperty(prop_name, prop_value))

def _set_default_controllers(config, robot_cls):
    """
    Resets/overrides the _default_controllers property on robot_cls by calling super() and applying updates from config.
    This will override any existing _default_controllers property that might already exist on the class.
    """
    properties = config.get("property", {})
    updates = properties.get("_default_controllers")
    if updates is None:
        return
    
    updates_dict = updates.copy()
    
    def _make_property(update_dict):
        @property
        def prop_func(self):
            controllers = getattr(super(robot_cls, self), "_default_controllers")
            controllers = dict(controllers)
            for key, value in update_dict.items():
                controllers[key] = value
            return controllers
        return prop_func
    setattr(robot_cls, "_default_controllers", _make_property(updates_dict))

def _convert_to_math_pi(ele):
    """
    Convert string expressions involving pi (e.g., "pi/8", "0.5*pi", "-pi/2") 
    to their numeric values.
    """
    if not isinstance(ele, str) or "pi" not in ele:
        return ele
    
    expression = ele.replace("-pi", f"(-{math.pi})")
    expression = expression.replace("pi", str(math.pi))
    safe_dict = {
        "__builtins__": {},
        "math": math,
        "abs": abs,
        "round": round,
    }
    
    try:
        result = eval(expression, safe_dict)
        return result
    except (SyntaxError, NameError, ZeroDivisionError) as e:
        return ele

def _set_teleop_rotation_offset(cfg, robot_cls):
    def _convert_to_quat(li):
        ret = []
        for element in li:
            ret.append(float(_convert_to_math_pi(element)))
        return euler2quat(th.tensor(ret))
    properties = cfg.get("property", {})
    if "teleop_rotation_offset" not in properties:
        return
    value = properties["teleop_rotation_offset"]
    if isinstance(value, list):
        value = _convert_to_quat(value)
    elif isinstance(value, dict):
        value = {key: _convert_to_quat(val) for key, val in value.items()}
    
    # Create a property that returns the processed teleop_rotation_offset
    # This is necessary because teleop_rotation_offset must be a @property
    # (required by ManipulationRobot's @property definition)
    @property
    def teleop_rotation_offset_prop(self):
        return value
    setattr(robot_cls, "teleop_rotation_offset", teleop_rotation_offset_prop)

def _set_default_joint_pos(cfg, robot_cls):
    properties = cfg.get("property", {})
    if "_default_joint_pos" in properties:
        val = properties["_default_joint_pos"]
        if isinstance(val, list):
            li = []
            for ele in val:
                li.append(float(_convert_to_math_pi(ele)))
            tensor_val = th.tensor(li)
            
            @property
            def _default_joint_pos_prop(self):
                return tensor_val
            setattr(robot_cls, "_default_joint_pos", _default_joint_pos_prop)
        elif val == 0:
            @property
            def _default_joint_pos_prop(self):
                return th.zeros(self.n_joints)
            setattr(robot_cls, "_default_joint_pos", _default_joint_pos_prop)

def _set_tucked_untucked_default_joint_pos(cfg, robot_cls, k):
    """
    Creates (un)tucked_default_joint_pos property
    """
    properties = cfg.get("property", {}) 
    if k not in properties:
        return
    updates = properties[k]
    if not isinstance(updates, dict):
        return
    
    def _make_property(update_dict=updates):
        @property
        def prop_func(self):
            if "init" in update_dict.keys():
                pos = th.zeros(self.n_dof)
            else:
                # Call the parent class's property to get the actual tensor value
                # (not the property descriptor)
                pos = getattr(super(robot_cls, self), k).clone()
            
            for key, value in update_dict.items():
                if key == "init":
                    continue
                elif key == "base_idx" and value == "current":
                    pos[self.base_idx] = self.get_joint_positions()[self.base_idx]
                elif key == "gripper_control_idx":
                    for arm in update_dict[key].keys():
                        pos[self.gripper_control_idx[arm]] = th.tensor(update_dict[key][arm])
                elif key == "arm_control_idx" and isinstance(value, dict):
                    for arm in update_dict[key].keys():
                        pos[self.arm_control_idx[arm]] = th.tensor(update_dict[key][arm])
                elif key == "trunk_control_idx":
                        pos[self.trunk_control_idx] = value
                elif key == "camera_control_idx":
                    pos[self.camera_control_idx] = th.tensor(value)
                else:
                    # Direct index update
                    # If key is an integer, use it directly; otherwise get the instance attribute
                    if isinstance(key, int):
                        pos[key] = value
                    else:
                        pos[getattr(self, key)] = value
            return pos
        return prop_func
    
    setattr(robot_cls, k, _make_property())

def _create_init_method(cfg, robot_cls, bases):
    """
    Create __init__ method from YAML 'init' section.
    The init section in yaml contains the robot class' parameter names and default values for __init__().
    These can override parent class defaults or add new parameters.

    Also add another attr _init_param_names_from_yaml to store all support params in the final __init__().
    """
    init_params_from_yaml = cfg.get("init", {})

    # Step 1: get all init() params for this robot_cls
    
    # Get the params of all base class' __init__()
    import inspect
    base_params = set()
    for base in bases:
        if not (hasattr(base, "__init__") and base.__init__ is not object.__init__):
            continue
        if hasattr(base, "_init_param_names_from_yaml"):
            base_params |= set(base._init_param_names_from_yaml)
        else:
            sig = inspect.signature(base.__init__)
            base_param |= set(sig.parameters.keys())
    base_params.discard('self')  # Remove 'self'

    robot_cls._init_param_names_from_yaml = base_params | set(init_params_from_yaml.keys())
    
    # Step 2: create init() for this robot_cls
    
    if not init_params_from_yaml:
        # No custom __init__ specified, use base class __init__
        return

    # Identify custom parameters (not in base __init__)
    custom_params = {k: v for k, v in init_params_from_yaml.items() if k not in base_params}
    
    # Create final __init__() for robot classs
    def __init__(self, *args, **kwargs):
        # Apply defaults from YAML init section (overrides parent defaults)
        for param_name, default_value in init_params_from_yaml.items():
            if param_name not in kwargs:
                kwargs[param_name] = default_value
        
        # Store custom parameters as instance attributes before calling super
        # If robot class need to add special cases in init(), add them here
        for param_name, default_value in custom_params.items():
            value = kwargs.get(param_name, default_value)
            
            if param_name == "variant" and cfg["name"] == "Tiago":
                valid_variants = ("default", "wrist_cam")
                if value not in valid_variants:
                    raise ValueError(f"Invalid Tiago variant specified {value}! Must be one of {valid_variants}")
                self._variant = value
            elif param_name == "end_effector" and cfg["name"].lower()=='a1':
                if value not in ["gripper","inspire"]:
                    raise ValueError(f"Invalid A1 end effector.")
                self.end_effector = value
            elif param_name == "end_effector" and cfg["name"].lower()=='frankapanda':
                if value not in ["gripper","inspire","leap_right","leap_left","allegro"]:
                    raise ValueError(f"Invalid A1 end effector.")
                self.end_effector = value
          
        # Set grasping_direction based on end_effector for A1 and FrankaPanda
        if cfg["name"].lower() in ['a1', 'frankapanda']:
            end_effector_value = kwargs.get("end_effector", init_params_from_yaml.get("end_effector", "gripper"))
            kwargs["grasping_direction"] = "lower" if end_effector_value == "gripper" else "upper"
          
        # Call super().__init__ with all kwargs (including overridden defaults)
        super(robot_cls, self).__init__(*args, **kwargs)
    setattr(robot_cls, "__init__", __init__)

def _validate_config(cfg: Dict[str, Any]) -> None:
    unknown = set(cfg.keys()) - set(ALLOWED_TOP_KEYS.keys())
    if unknown:
        raise ValueError(f"Unknown top-level keys in robot YAML: {sorted(list(unknown))}")
    missing = [k for k, req in ALLOWED_TOP_KEYS.items() if req and k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in robot YAML: {missing}")
    # check to see if all required props are provided
    # if "properties" in cfg and not isinstance(cfg["properties"], dict):
    #     raise TypeError("properties must be a mapping of name->value")

    # if cfg['name'].lower() in ['a1','frankapanda']:
    #     return
    
    # properties = cfg.get("property", {})
    # cached_property = cfg.get("cached_property", {})
    set_of_props = set(cfg.get("property", {}).keys()) | set(cfg.get("classproperty", {}).keys())
    if cfg['name'].lower() in ['a1','frankapanda']:
        set_of_props = set_of_props | set(cfg['property']['gripper'].keys())
        # return
    
    for cap in cfg.get("capabilities", []):
        if cap not in CAPABILITY_BASES:
            
            raise ValueError(f"{cfg['name']}: Unknown capability '{cap}'. Allowed: {sorted(list(CAPABILITY_BASES.keys()))}")
        # breakpoint()
        if cap in EXPECTED_PROPS:
            required_props = EXPECTED_PROPS[cap]
            # print(cfg['name'],type(required_props),type(set_of_props))
            missing_props = required_props - set_of_props
            if missing_props:
                raise ValueError(
                    f"{cfg['name']}: Capability '{cap}' requires the following properties in YAML: {sorted(list(required_props))}. "
                    f"Missing: {sorted(list(missing_props))}"
                )

def _import_class(import_path: str):
    """
    Used when a robot config uses the extends key to inherit 
    from another robot class specified as a string path.
    """
    # Parse the import path
    if ":" in import_path:
        module_path, cls_name = import_path.split(":", 1)
    elif "." in import_path:
        parts = import_path.split(".")
        module_path, cls_name = ".".join(parts[:-1]), parts[-1]
    else:
        raise ValueError(f"Invalid extends path: {import_path}")
    
    # First check if the class is already registered (from a YAML file loaded earlier)
    # This handles the case where one YAML-defined class extends another
    if cls_name in REGISTERED_ROBOTS:
        return REGISTERED_ROBOTS[cls_name]
    
    # If it's a robot_configs path, try to load the YAML file directly
    if "robot_configs" in module_path:
        # Extract the YAML filename from the module path
        # e.g., "omnigibson.robots.robot_configs.franka" -> "franka.yaml"
        yaml_name = module_path.split("robot_configs.")[-1]
        # Use the same directory structure as autodiscover
        robot_config_dir = Path(__file__).parent / "robot_configs"
        yaml_path = robot_config_dir / f"{yaml_name}.yaml"
        if yaml_path.exists():
            # Load the YAML file to create the class
            cls = create_robot_class_from_yaml(yaml_path)
            if cls.__name__ == cls_name:
                return cls
    
    # Otherwise, try to import it from the module (for non-YAML classes)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not import class {cls_name} from {import_path}. "
                        f"Make sure the class is already loaded or the module path is correct: {e}")

def _inject_module(module_path: str, class_name: str, cls: type) -> None:
    """
    Expose the generated class at module_path,
    so imports like `from X import Class` work.
    """
    mod = ModuleType(module_path)
    setattr(mod, class_name, cls)
    mod.__all__ = [class_name]
    sys.modules[module_path] = mod


def create_robot_class_from_yaml(config_path: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    _validate_config(cfg)

    bases: List[type] = []
    if "extends" in cfg:
        bases.append(_import_class(cfg["extends"]))
    for cap in cfg.get("capabilities", []):
        bases.append(CAPABILITY_BASES[cap])
    if not bases:
        bases = [BaseRobot]

    class_attrs: Dict[str, Any] = {"__doc__": cfg.get("description", ""), "_yaml_config": cfg}

    for key, value in (cfg.get("class_attributes", {}) or {}).items():
        class_attrs[key] = value

    robot_cls = type(cfg["name"], tuple(bases), class_attrs)

    # Create __init__ method from YAML 'init' section if specified
    _create_init_method(cfg, robot_cls, bases)
    
    
    # Set up special properties that need dynamic creation
    _set_default_controllers(cfg, robot_cls)
    _set_tucked_untucked_default_joint_pos(cfg, robot_cls, "tucked_default_joint_pos")
    _set_tucked_untucked_default_joint_pos(cfg, robot_cls, "untucked_default_joint_pos")
    _set_default_joint_pos(cfg, robot_cls)
    _set_teleop_rotation_offset(cfg, robot_cls)
    _set_arm_workspace_range(cfg, robot_cls)
    _set_default_arm_poses(cfg, robot_cls)
    _set_end_effector_properties(cfg, robot_cls)  # Must be before _set_general_properties
    _set_post_load(cfg, robot_cls)
    
    # Set all other properties from 'property' section as @property
    _set_general_properties(cfg, robot_cls)
    
    # Set all properties from 'classproperty' section as @classproperty
    _set_classproperties(cfg, robot_cls)


    # Register in registry
    REGISTERED_ROBOTS[robot_cls.__name__] = robot_cls

    # Inject module for import-compat
    _inject_module(cfg["module_path"], robot_cls.__name__, robot_cls)

    return robot_cls


def autodiscover_and_register(yaml_dir: Path) -> List[Tuple[str, type]]:
    """
    Auto-discover and register all YAML robot configs.
    Files are loaded in alphabetical order, so classes that extend others
    should come after their base classes (e.g., franka_mounted.yaml after franka.yaml).
    """
    out = []
    if not yaml_dir.exists():
        return out
    for p in sorted(yaml_dir.glob("*.yaml")):
        cls = create_robot_class_from_yaml(p)
        out.append((cls.__name__, cls))
    return out


# Auto-discover and create classes for all robot configs in robot_configs/*.yaml
ROBOT_CONFIG_DIR = Path(__file__).parent / "robot_configs"
__all__: List[str] = []
_registered = autodiscover_and_register(ROBOT_CONFIG_DIR)
for _name, _cls in _registered:
    globals()[_name] = _cls
    __all__.append(_name) 

 