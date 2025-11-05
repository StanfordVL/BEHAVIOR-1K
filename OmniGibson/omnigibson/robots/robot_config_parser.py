from pathlib import Path
import importlib
import math
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple

import torch as th
import yaml
from omnigibson.utils.transform_utils import euler2quat


from omnigibson.robots.robot_base import BaseRobot, REGISTERED_ROBOTS
from omnigibson.robots.two_wheel_robot import TwoWheelRobot
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.robots.holonomic_base_robot import HolonomicBaseRobot
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.active_camera_robot import ActiveCameraRobot
from omnigibson.robots.mobile_manipulation_robot import MobileManipulationRobot
from omnigibson.robots.untucked_arm_pose_robot import UntuckedArmPoseRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot

CAPABILITY_BASES = {
    "two_wheel_locomotion": TwoWheelRobot,
    "manipulation": ManipulationRobot,
    "holonomic_base": HolonomicBaseRobot,
    "articulated_trunk": ArticulatedTrunkRobot,
    "active_camera": ActiveCameraRobot,
    "mobile_manipulation": MobileManipulationRobot,
    "untucked_arm_pose": UntuckedArmPoseRobot,
    "locomotion": LocomotionRobot,
}

EXPECTED_PROPS = {
    "two_wheel_locomotion": {"wheel_radius", "wheel_axle_length"},
    "untucked_arm_pose": {"default_arm_poses"},
    "mobile_manipulation": {"tucked_default_joint_pos","untucked_default_joint_pos"},
    "manipulation": {"arm_link_names","arm_joint_names","eef_link_names","gripper_link_names","finger_link_names","finger_joint_names","arm_workspace_range"},
    "locomotion":{"floor_touching_base_link_names", "base_joint_names"},
    "holonomic_base":{"base_footprint_link_name"},
    "articulated_trunk": {"trunk_link_names","trunk_joint_names"},
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
    "cached_property": False,   # dict[str, Any] exposed via @cached_property returning the literal
}

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
    if "teleop_rotation_offset" not in cfg:
        return
    value = cfg["teleop_rotation_offset"]
    if isinstance(value, list):
        value = _convert_to_quat(value)
    elif isinstance(value, dict):
        value =  {key: _convert_to_quat(val) for key, val in value.items()}
    setattr(robot_cls, "teleop_rotation_offset", value)

def _set_discrete_action_prop(robot_cls, cfg):
    """
    - _create_discrete_action_space: If value is like ValueError("message"), creates a method that raises that exception
    - discrete_action_list: If value is NotImplementedError, creates a property that raises NotImplementedError
    """
    if "_create_discrete_action_space" in cfg:
        value = cfg["_create_discrete_action_space"]
        if isinstance(value, list) and value[0].endswith("Error"):
            exception_type_name = value[0]
            exception_message = value[1]
            exception_class = getattr(__builtins__, exception_type_name, None)
            if exception_class is None or not issubclass(exception_class, BaseException):
                raise ValueError(f"Unknown error type: {exception_type_name}")
            def _create_method():
                def method():
                    raise exception_class(exception_message)
                return method()
            setattr(robot_cls, "_create_discrete_action_space", _create_method())
    
    properties = cfg.get("property", {})
    if "discrete_action_list" in properties:
        if properties["discrete_action_list"] == "NotImplementedError":
            @property
            def discrete_action_list_prop(self):
                raise NotImplementedError()
            setattr(robot_cls, "discrete_action_list", discrete_action_list_prop)

def _set_default_joint_pos(robot_cls, cfg):
    properties = cfg.get("property", {})
    if "_default_joint_pos" in properties:
        val = properties["_default_joint_pos"]
        if isinstance(val, list):
            li = []
            for ele in val:
                li.append(float(_convert_to_math_pi(ele)))
            val = th.tensor(li)
            setattr(robot_cls, "_default_joint_pos", val)
        elif val == 0:
            @property
            def _default_joint_pos_prop(self):
                return th.zeros(self.n_joints)
            setattr(robot_cls, "_default_joint_pos", _default_joint_pos_prop)

def _validate_config(cfg: Dict[str, Any]) -> None:
    unknown = set(cfg.keys()) - set(ALLOWED_TOP_KEYS.keys())
    if unknown:
        raise ValueError(f"Unknown top-level keys in robot YAML: {sorted(list(unknown))}")
    missing = [k for k, req in ALLOWED_TOP_KEYS.items() if req and k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in robot YAML: {missing}")
    # check to see if all required props are provided
    if "properties" in cfg and not isinstance(cfg["properties"], dict):
        raise TypeError("properties must be a mapping of name->value")
    properties = cfg.get("properties", {})
    for cap in cfg.get("capabilities", []):
        if cap not in CAPABILITY_BASES:
            raise ValueError(f"Unknown capability '{cap}'. Allowed: {sorted(list(CAPABILITY_BASES.keys()))}")
        if cap in EXPECTED_PROPS:
            required_props = EXPECTED_PROPS[cap]
            missing_props = required_props - set(properties.keys())
            if missing_props:
                raise ValueError(
                    f"Capability '{cap}' requires the following properties in YAML: {sorted(list(required_props))}. "
                    f"Missing: {sorted(list(missing_props))}"
                )

def _import_class(import_path: str):
    """
    Used when a robot config uses the extends key to inherit 
    from another robot class specified as a string path.
    """
    if ":" in import_path:
        module_path, cls_name = import_path.split(":", 1)
    elif "." in import_path:
        parts = import_path.split(".")
        module_path, cls_name = ".".join(parts[:-1]), parts[-1]
    else:
        raise ValueError(f"Invalid extends path: {import_path}")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

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

    # if cfg["support_discrete_action"] is false, then set a property 

    # Register in registry
    REGISTERED_ROBOTS[robot_cls.__name__] = robot_cls

    # Inject module for import-compat
    _inject_module(cfg["module_path"], robot_cls.__name__, robot_cls)

    return robot_cls


def autodiscover_and_register(yaml_dir: Path) -> List[Tuple[str, type]]:
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

 