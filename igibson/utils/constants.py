"""
Constant Definitions
"""

import os
from enum import IntEnum, Enum

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.assets_utils import get_ig_avg_category_specs

AVAILABLE_MODALITIES = ("rgb", "normal", "3d", "seg", "optical_flow", "scene_flow", "ins_seg")
MAX_INSTANCE_COUNT = 1024
MAX_CLASS_COUNT = 512


class ViewerMode(IntEnum):
    NAVIGATION = 0
    MANIPULATION = 1
    PLANNING = 2


class SimulatorMode(IntEnum):
    GUI_INTERACTIVE = 1
    GUI_NON_INTERACTIVE = 2
    HEADLESS = 3
    HEADLESS_TENSOR = 4
    VR = 5


class SemanticClass(IntEnum):
    BACKGROUND = 0
    ROBOTS = 1
    USER_ADDED_OBJS = 2
    SCENE_OBJS = 3
    # The following class ids count backwards from MAX_CLASS_COUNT (instead of counting forward from 4) because we want
    # to maintain backward compatibility
    DIRT = 507
    STAIN = 508
    WATER = 509
    HEAT_SOURCE_MARKER = 510
    TOGGLE_MARKER = 511


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
AVERAGE_CATEGORY_SPECS = get_ig_avg_category_specs()


def get_collision_group_mask(groups_to_exclude=[]):
    """Get a collision group mask that has collisions enabled for every group except those in groups_to_exclude."""
    collision_mask = ALL_COLLISION_GROUPS_MASK
    for group in groups_to_exclude:
        collision_mask &= ~(1 << group)
    return collision_mask


class ShadowPass(IntEnum):
    NO_SHADOW = 0
    HAS_SHADOW_RENDER_SHADOW = 1
    HAS_SHADOW_RENDER_SCENE = 2


class CoordinateSystem(IntEnum):
    OPENCV = 0
    OPENGL = 1
    PYBULLET = 2
    SUNRGBD = 3


class OccupancyGridState(object):
    OBSTACLES = 0.0
    UNKNOWN = 0.5
    FREESPACE = 1.0


# PyBullet-related
class PyBulletSleepState(IntEnum):
    AWAKE = 1
    ISLAND_AWAKE = 3


PYBULLET_BASE_LINK_INDEX = -1


# BEHAVIOR-related
FLOOR_SYNSET = "floor.n.01"
NON_SAMPLEABLE_OBJECTS = []
non_sampleable_category_txt = os.path.join(igibson.ig_dataset_path, "metadata/non_sampleable_categories.txt")
if os.path.isfile(non_sampleable_category_txt):
    with open(non_sampleable_category_txt) as f:
        NON_SAMPLEABLE_OBJECTS = [FLOOR_SYNSET] + [line.strip() for line in f.readlines()]

MAX_TASK_RELEVANT_OBJS = 50
TASK_RELEVANT_OBJS_OBS_DIM = 9
AGENT_POSE_DIM = 6

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

hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

NamedRenderingPresets = {
    "NO_PBR": MeshRendererSettings(enable_pbr=False, enable_shadow=False),
    "PBR_NOSHADOW": MeshRendererSettings(enable_pbr=True, enable_shadow=True),
    "PBR_SHADOW_MSAA": MeshRendererSettings(enable_pbr=True, enable_shadow=True, msaa=True),
    "NO_PBR_OPT": MeshRendererSettings(enable_pbr=False, enable_shadow=False, optimized=True),
    "PBR_NOSHADOW_OPT": MeshRendererSettings(enable_pbr=True, enable_shadow=True, optimized=True),
    "PBR_SHADOW_MSAA_OPT": MeshRendererSettings(enable_pbr=True, enable_shadow=True, msaa=True, optimized=True),
    "HQ_WITH_BG_OPT": MeshRendererSettings(
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        msaa=True,
        light_dimming_factor=1.0,
        optimized=True,
    ),
    "VISUAL_RL": MeshRendererSettings(enable_pbr=True, enable_shadow=False, msaa=False, optimized=True),
    "PERCEPTION": MeshRendererSettings(
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        msaa=True,
        light_dimming_factor=1.0,
        optimized=True,
    ),
}

# Encodings
RAW_ENCODING = 0
COPY_RECTANGLE_ENCODING = 1
RRE_ENCODING = 2
CORRE_ENCODING = 4
HEXTILE_ENCODING = 5
ZLIB_ENCODING = 6
TIGHT_ENCODING = 7
ZLIBHEX_ENCODING = 8
ZRLE_ENCODING = 16
# 0xffffff00 to 0xffffffff tight options
PSEUDO_CURSOR_ENCODING = -239

# Keycodes
KEY_BackSpace = 0xFF08
KEY_Tab = 0xFF09
KEY_Return = 0xFF0D
KEY_Escape = 0xFF1B
KEY_Insert = 0xFF63
KEY_Delete = 0xFFFF
KEY_Home = 0xFF50
KEY_End = 0xFF57
KEY_PageUp = 0xFF55
KEY_PageDown = 0xFF56
KEY_Left = 0xFF51
KEY_Up = 0xFF52
KEY_Right = 0xFF53
KEY_Down = 0xFF54
KEY_F1 = 0xFFBE
KEY_F2 = 0xFFBF
KEY_F3 = 0xFFC0
KEY_F4 = 0xFFC1
KEY_F5 = 0xFFC2
KEY_F6 = 0xFFC3
KEY_F7 = 0xFFC4
KEY_F8 = 0xFFC5
KEY_F9 = 0xFFC6
KEY_F10 = 0xFFC7
KEY_F11 = 0xFFC8
KEY_F12 = 0xFFC9
KEY_F13 = 0xFFCA
KEY_F14 = 0xFFCB
KEY_F15 = 0xFFCC
KEY_F16 = 0xFFCD
KEY_F17 = 0xFFCE
KEY_F18 = 0xFFCF
KEY_F19 = 0xFFD0
KEY_F20 = 0xFFD1
KEY_ShiftLeft = 0xFFE1
KEY_ShiftRight = 0xFFE2
KEY_ControlLeft = 0xFFE3
KEY_ControlRight = 0xFFE4
KEY_MetaLeft = 0xFFE7
KEY_MetaRight = 0xFFE8
KEY_AltLeft = 0xFFE9
KEY_AltRight = 0xFFEA

KEY_Scroll_Lock = 0xFF14
KEY_Sys_Req = 0xFF15
KEY_Num_Lock = 0xFF7F
KEY_Caps_Lock = 0xFFE5
KEY_Pause = 0xFF13
KEY_Super_L = 0xFFEB
KEY_Super_R = 0xFFEC
KEY_Hyper_L = 0xFFED
KEY_Hyper_R = 0xFFEE

KEY_KP_0 = 0xFFB0
KEY_KP_1 = 0xFFB1
KEY_KP_2 = 0xFFB2
KEY_KP_3 = 0xFFB3
KEY_KP_4 = 0xFFB4
KEY_KP_5 = 0xFFB5
KEY_KP_6 = 0xFFB6
KEY_KP_7 = 0xFFB7
KEY_KP_8 = 0xFFB8
KEY_KP_9 = 0xFFB9
KEY_KP_Enter = 0xFF8D

KEY_ForwardSlash = 0x002F
KEY_BackSlash = 0x005C
KEY_SpaceBar = 0x0020

# TODO: build this programmatically?
KEYMAP = {
    "bsp": KEY_BackSpace,
    "tab": KEY_Tab,
    "return": KEY_Return,
    "enter": KEY_Return,
    "esc": KEY_Escape,
    "ins": KEY_Insert,
    "delete": KEY_Delete,
    "del": KEY_Delete,
    "home": KEY_Home,
    "end": KEY_End,
    "pgup": KEY_PageUp,
    "pgdn": KEY_PageDown,
    "ArrowLeft": KEY_Left,
    "left": KEY_Left,
    "ArrowUp": KEY_Up,
    "up": KEY_Up,
    "ArrowRight": KEY_Right,
    "right": KEY_Right,
    "ArrowDown": KEY_Down,
    "down": KEY_Down,
    "slash": KEY_BackSlash,
    "bslash": KEY_BackSlash,
    "fslash": KEY_ForwardSlash,
    "spacebar": KEY_SpaceBar,
    "space": KEY_SpaceBar,
    "sb": KEY_SpaceBar,
    "f1": KEY_F1,
    "f2": KEY_F2,
    "f3": KEY_F3,
    "f4": KEY_F4,
    "f5": KEY_F5,
    "f6": KEY_F6,
    "f7": KEY_F7,
    "f8": KEY_F8,
    "f9": KEY_F9,
    "f10": KEY_F10,
    "f11": KEY_F11,
    "f12": KEY_F12,
    "f13": KEY_F13,
    "f14": KEY_F14,
    "f15": KEY_F15,
    "f16": KEY_F16,
    "f17": KEY_F17,
    "f18": KEY_F18,
    "f19": KEY_F19,
    "f20": KEY_F20,
    "lshift": KEY_ShiftLeft,
    "shift": KEY_ShiftLeft,
    "rshift": KEY_ShiftRight,
    "lctrl": KEY_ControlLeft,
    "ctrl": KEY_ControlLeft,
    "rctrl": KEY_ControlRight,
    "lmeta": KEY_MetaLeft,
    "meta": KEY_MetaLeft,
    "rmeta": KEY_MetaRight,
    "lalt": KEY_AltLeft,
    "alt": KEY_AltLeft,
    "ralt": KEY_AltRight,
    "scrlk": KEY_Scroll_Lock,
    "sysrq": KEY_Sys_Req,
    "numlk": KEY_Num_Lock,
    "caplk": KEY_Caps_Lock,
    "pause": KEY_Pause,
    "lsuper": KEY_Super_L,
    "super": KEY_Super_L,
    "rsuper": KEY_Super_R,
    "lhyper": KEY_Hyper_L,
    "hyper": KEY_Hyper_L,
    "rhyper": KEY_Hyper_R,
    "kp0": KEY_KP_0,
    "kp1": KEY_KP_1,
    "kp2": KEY_KP_2,
    "kp3": KEY_KP_3,
    "kp4": KEY_KP_4,
    "kp5": KEY_KP_5,
    "kp6": KEY_KP_6,
    "kp7": KEY_KP_7,
    "kp8": KEY_KP_8,
    "kp9": KEY_KP_9,
    "kpenter": KEY_KP_Enter,
}
