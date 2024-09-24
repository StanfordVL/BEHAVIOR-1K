"""
Constant Definitions
"""

import hashlib
import os
from enum import Enum, IntEnum
from functools import cache

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_all_object_categories, get_all_system_categories

MAX_INSTANCE_COUNT = th.iinfo(th.int32).max
MAX_CLASS_COUNT = th.iinfo(th.int32).max
MAX_VIEWER_SIZE = 2048


class ViewerMode(IntEnum):
    NAVIGATION = 0
    MANIPULATION = 1
    PLANNING = 2


class LightingMode(str, Enum):
    # See https://stackoverflow.com/a/58608362 for info on string enums
    STAGE = "stage"
    CAMERA = "camera"
    RIG_DEFAULT = "Default"
    RIG_GREY = "Grey Studio"
    RIG_COLORED = "Colored Lights"


class SimulatorMode(IntEnum):
    GUI = 1
    HEADLESS = 2
    VR = 3


# Specific methods for applying / removing particles
class ParticleModifyMethod(str, Enum):
    ADJACENCY = "adjacency"
    PROJECTION = "projection"


# Specific condition types for applying / removing particles
class ParticleModifyCondition(str, Enum):
    FUNCTION = "function"
    SATURATED = "saturated"
    TOGGLEDON = "toggled_on"
    GRAVITY = "gravity"


# Structure categories that need to always be loaded for stability purposes
STRUCTURE_CATEGORIES = frozenset({"floors", "walls", "ceilings", "lawn", "driveway", "fence", "roof", "background"})

# Ground categories / prim names used for filtering collisions, e.g.: during motion planning
GROUND_CATEGORIES = frozenset({"floors", "lawn", "driveway", "carpet"})

# Joint friction magic values to assign to objects based on their category
DEFAULT_JOINT_FRICTION = 10.0
SPECIAL_JOINT_FRICTIONS = {
    "oven": 30.0,
    "dishwasher": 30.0,
    "toilet": 3.0,
}


class PrimType(IntEnum):
    RIGID = 0
    CLOTH = 1


class EmitterType(IntEnum):
    FIRE = 0
    STEAM = 1


# Valid primitive mesh types
PRIMITIVE_MESH_TYPES = {
    "Cone",
    "Cube",
    "Cylinder",
    "Disk",
    "Plane",
    "Sphere",
    "Torus",
}

# Valid geom types
GEOM_TYPES = {"Sphere", "Cube", "Cone", "Cylinder", "Mesh"}

# Valid joint axis
JointAxis = ["X", "Y", "Z"]


# TODO: Clean up this class to be better enum with sanity checks
# Joint types
class JointType:
    JOINT = "Joint"
    JOINT_FIXED = "FixedJoint"
    JOINT_PRISMATIC = "PrismaticJoint"
    JOINT_REVOLUTE = "RevoluteJoint"
    JOINT_SPHERICAL = "SphericalJoint"

    _STR_TO_TYPE = {
        "Joint": JOINT,
        "FixedJoint": JOINT_FIXED,
        "PrismaticJoint": JOINT_PRISMATIC,
        "RevoluteJoint": JOINT_REVOLUTE,
        "SphericalJoint": JOINT_SPHERICAL,
    }

    _TYPE_TO_STR = {
        JOINT: "Joint",
        JOINT_FIXED: "FixedJoint",
        JOINT_PRISMATIC: "PrismaticJoint",
        JOINT_REVOLUTE: "RevoluteJoint",
        JOINT_SPHERICAL: "SphericalJoint",
    }

    @classmethod
    def get_type(cls, str_type):
        assert str_type in cls._STR_TO_TYPE, f"Invalid string joint type name received: {str_type}"
        return cls._STR_TO_TYPE[str_type]

    @classmethod
    def get_str(cls, joint_type):
        assert joint_type in cls._TYPE_TO_STR, f"Invalid joint type name received: {joint_type}"
        return cls._TYPE_TO_STR[joint_type]

    @classmethod
    def is_valid(cls, joint_type):
        return joint_type in cls._TYPE_TO_STR if isinstance(joint_type, cls) else joint_type in cls._STR_TO_TYPE


# Object category specs
AVERAGE_OBJ_DENSITY = 67.0


class OccupancyGridState:
    OBSTACLES = 0.0
    UNKNOWN = 0.5
    FREESPACE = 1.0


MAX_TASK_RELEVANT_OBJS = 50
TASK_RELEVANT_OBJS_OBS_DIM = 9
AGENT_POSE_DIM = 6

# TODO: What the hell is this magic list?? It's not used anywhere
UNDER_OBJECTS = [
    "breakfast_table",
    "coffee_table",
    "console_table",
    "desk",
    "gaming_table",
    "pedestal_table",
    "pool_table",
    "stand",
    "armchair",
    "chaise_longue",
    "folding_chair",
    "highchair",
    "rocking_chair",
    "straight_chair",
    "swivel_chair",
    "bench",
]


@cache
def semantic_class_name_to_id():
    """
    Get mapping from semantic class name to class id

    Returns:
        dict: class name to class id
    """
    categories = get_all_object_categories()
    systems = get_all_system_categories(include_cloth=True)

    all_semantics = sorted(set(categories + systems + ["background", "unlabelled", "object", "light", "agent"]))

    # Assign a unique class id to each class name with hashing, the upper limit here is the max of int32
    max_int32 = th.iinfo(th.int32).max + 1
    class_name_to_class_id = {s: int(hashlib.md5(s.encode()).hexdigest(), 16) % max_int32 for s in all_semantics}

    return class_name_to_class_id


@cache
def semantic_class_id_to_name():
    """
    Get mapping from semantic class id to class name

    Returns:
        dict: class id to class name
    """
    return {v: k for k, v in semantic_class_name_to_id().items()}
