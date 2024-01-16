"""
Constant Definitions
"""
import os
from enum import Enum, IntEnum

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_og_avg_category_specs

MAX_INSTANCE_COUNT = 1024
MAX_CLASS_COUNT = 4096
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


class SemanticClass(IntEnum):
    BACKGROUND = 0
    ROBOTS = 1
    USER_ADDED_OBJS = 2
    SCENE_OBJS = 3
    # The following class ids count backwards from MAX_CLASS_COUNT (instead of counting forward from 4) because we want
    # to maintain backward compatibility
    GRASS = 506
    DIRT = 507
    STAIN = 508
    WATER = 509
    HEAT_SOURCE_MARKER = 510
    TOGGLE_MARKER = 511


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


# Valid omni characters for specifying strings, e.g. prim paths
VALID_OMNI_CHARS = frozenset({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '/'})

# Structure categories that need to always be loaded for stability purposes
STRUCTURE_CATEGORIES = frozenset({"floors", "walls", "ceilings", "lawn", "driveway", "fence"})

# Note that we are starting this from bit 6 since bullet seems to be giving special meaning to groups 0-5.
# Collision groups for objects. For special logic, different categories can be assigned different collision groups.
ALL_COLLISION_GROUPS_MASK = -1
DEFAULT_COLLISION_GROUP = "default"
SPECIAL_COLLISION_GROUPS = {
    "floors": "floors",
    "carpet": "carpet",
}


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
GEOM_TYPES = {"Sphere", "Cube", "Capsule", "Cone", "Cylinder", "Mesh"}

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
AVERAGE_CATEGORY_SPECS = get_og_avg_category_specs()


def get_collision_group_mask(groups_to_exclude=[]):
    """Get a collision group mask that has collisions enabled for every group except those in groups_to_exclude."""
    collision_mask = ALL_COLLISION_GROUPS_MASK
    for group in groups_to_exclude:
        collision_mask &= ~(1 << group)
    return collision_mask


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

hdr_texture = os.path.join(gm.DATASET_PATH, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(gm.DATASET_PATH, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    gm.DATASET_PATH, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(gm.DATASET_PATH, "scenes", "background", "urban_street_01.jpg")


def get_class_name_to_class_id():
    """
    Get mapping from semantic class name to class id

    Returns:
        dict: starting class id for scene objects
    """
    existing_classes = {item.value for item in SemanticClass}
    category_txt = os.path.join(gm.DATASET_PATH, "metadata/categories.txt")
    class_name_to_class_id = {"agent": SemanticClass.ROBOTS}  # Agents should have the robot semantic class.
    starting_class_id = 0
    if os.path.isfile(category_txt):
        with open(category_txt) as f:
            for line in f.readlines():
                while starting_class_id in existing_classes:
                    starting_class_id += 1
                assert starting_class_id < MAX_CLASS_COUNT, "Class ID overflow: MAX_CLASS_COUNT is {}.".format(
                    MAX_CLASS_COUNT
                )
                class_name_to_class_id[line.strip()] = starting_class_id
                starting_class_id += 1

    return class_name_to_class_id


CLASS_NAME_TO_CLASS_ID = get_class_name_to_class_id()
