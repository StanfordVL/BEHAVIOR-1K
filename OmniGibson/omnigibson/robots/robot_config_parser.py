from pathlib import Path
import importlib
import sys
from types import ModuleType
from typing import Any, Dict, List, Tuple

import torch as th
import yaml

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


# Allowed top-level YAML keys and whether they are required
ALLOWED_TOP_KEYS: Dict[str, bool] = {
    "name": True,                # name of the robot class
    "description": False,        # docstring of the robot class
    "module_path": True,         # fully qualified module to expose the class (e.g., omnigibson.robots.robot_configs.fetch)
    "extends": False,            # optional import path "omnigibson.robots.r1:R1" or "omnigibson.robots.r1.R1"
    "capabilities": False,       # list[str] matching CAPABILITY_BASES keys
    "class_attributes": False,   # dict[str, Any] set as class attributes
    "properties": False,         # dict[str, Any] exposed via @property returning the literal
    "tensor_properties": False,  # list[str] keys in properties to convert to th.tensor
}


def _validate_config(cfg: Dict[str, Any]) -> None:
    unknown = set(cfg.keys()) - set(ALLOWED_TOP_KEYS.keys())
    if unknown:
        raise ValueError(f"Unknown top-level keys in robot YAML: {sorted(list(unknown))}")
    missing = [k for k, req in ALLOWED_TOP_KEYS.items() if req and k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in robot YAML: {missing}")
    if "capabilities" in cfg and not isinstance(cfg["capabilities"], list):
        raise TypeError("capabilities must be a list of strings")
    if "class_attributes" in cfg and not isinstance(cfg["class_attributes"], dict):
        raise TypeError("class_attributes must be a mapping of name->value")
    if "properties" in cfg and not isinstance(cfg["properties"], dict):
        raise TypeError("properties must be a mapping of name->value")
    if "tensor_properties" in cfg and not isinstance(cfg["tensor_properties"], list):
        raise TypeError("tensor_properties must be a list of property names")
    for cap in cfg.get("capabilities", []):
        if cap not in CAPABILITY_BASES:
            raise ValueError(f"Unknown capability '{cap}'. Allowed: {sorted(list(CAPABILITY_BASES.keys()))}")
    for sec in ("class_attributes", "properties"):
        for key in cfg.get(sec, {}).keys():
            if not str(key).isidentifier():
                raise ValueError(f"Invalid {sec} name '{key}' (must be a valid identifier)")


def _import_class(import_path: str):
    if ":" in import_path:
        module_path, cls_name = import_path.split(":", 1)
    elif "." in import_path:
        parts = import_path.split(".")
        module_path, cls_name = ".".join(parts[:-1]), parts[-1]
    else:
        raise ValueError(f"Invalid extends path: {import_path}")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

def _tensorize(val: Any) -> th.Tensor:
    return val if isinstance(val, th.Tensor) else th.tensor(val)


def _inject_module(module_path: str, class_name: str, cls: type) -> None:
    """Expose the generated class at module_path so imports like `from X import Class` work."""
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

    # tensor_props = set(cfg.get("tensor_properties", []))
    # properties = cfg.get("properties", {})
    class_attrs: Dict[str, Any] = {"__doc__": cfg.get("description", ""), "_yaml_config": cfg}

    for key, value in (cfg.get("class_attributes", {}) or {}).items():
        class_attrs[key] = value

    # for key, value in (properties or {}).items():
    #     literal = _coerce_property_value(value)
    #     if key in tensor_props:
    #         def _make_tensor_prop(v):
    #             return property(lambda self, _v=v: _maybe_tensorize(_v))
    #         class_attrs[key] = _make_tensor_prop(literal)
    #     else:
    #         def _make_prop(v):
    #             return property(lambda self, _v=v: _v)
    #         class_attrs[key] = _make_prop(literal)

    robot_cls = type(cfg["name"], tuple(bases), class_attrs)

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

 