import torch as th
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
from omnigibson.utils import transform_utils as T
from omnigibson.transition_rules import (
    ToggleableMachineRule,
    MixingToolRule,
    CookingSystemRule,
)

# Define room dependencies by scene
# Format: {scene_name: {room_A: [room_B, room_C], ...}}
ROOM_DEPENDENCIES = {
    "house_single_floor": {
        "dining_room": ["kitchen", "living_room"],
        "kitchen": ["living_room", "dining_room"],
        "living_room": ["kitchen", "dining_room"],
    },
    "house_double_floor_lower": {
        "kitchen": ["living_room", "corridor"],
        "living_room": ["kitchen", "corridor"],
        "garage": ["corridor"],
    },
    "house_double_floor_upper": {},
    "restaurant_diner": {
        "dining_room": ["bar", "corridor"],
        "bar": ["dining_room", "corridor", "kitchen"]
    },
    "Rs_int": {
        "living_room": ["kitchen"],
        "kitchen": ["living_room"],
    },
    "hotel_suite_large": {
        "bedroom": ["bathroom"],
        "bathroom": ["bedroom"]
    },
    "office_cubicles_right": {},
}

TASK_SPECIFIC_EXTRA_ROOMS = {
    "bringing_in_kindling": {
        "house_double_floor_lower": ["corridor"],
    },
    "bringing_newspaper_in": {
        "house_double_floor_lower": ["corridor"],
    },
    "bringing_paper_to_recycling": {
        "house_double_floor_lower": ["corridor"],
    },
    "chopping_wood": {
        "house_double_floor_lower": ["garage"],
    },
    "dispose_of_batteries": {
        "office_cubicles_right": ["corridor"],
    },
}


# Viewing mode configuration
class ViewingMode(str, Enum):
    SINGLE_VIEW = "single_view"
    VR = "vr"
    MULTI_VIEW_1 = "multi_view_1"


# Feature flags
USE_FLUID = False
USE_CLOTH = False
USE_ARTICULATED = False
FULL_SCENE = False
VIEWING_MODE = ViewingMode.MULTI_VIEW_1
SIMPLIFIED_TRUNK_CONTROL = True
APPLY_EXTRA_GRIP = False
# Grasping mode the teleoperated robot loads with, one of "assisted" (default), "sticky" or
# "physical". og_launch's --grasping-mode overrides this per run.
#
# "physical" is the escape hatch for frame rate: it is the only mode Robot.post_step() skips its
# assisted-grasping handling in, and under the Newton backend that handling dominates the teleop
# tick -- measured on r1pro at 103 ms of a 186 ms tick (5.4 fps), because it casts 32 single rays per
# env.step() and each one pays newton.intersect_ray's per-call kernel rebuild. Dropping it takes the
# tick to ~84 ms (~12 fps). The cost is real grasping behavior: nothing holds an object but friction,
# and is_grasping() falls back to inferring from the gripper controller.
GRASPING_MODE = "assisted"


# ─── Robot-specific teleop configuration ────────────────────────────────────


@dataclass
class RobotTeleopConfig:
    """All robot-specific parameters needed for teleoperation.

    To add a new robot, create a new instance in ROBOT_CONFIGS.
    Values that can be derived from the OmniGibson robot object
    (link names, joint indices, etc.) are NOT included here.
    """

    robot_type: str

    # Controller configuration passed to OmniGibson
    controller_config: Dict

    # Default reset joint positions
    reset_joint_pos: th.Tensor

    # Torso keyframe positions for trunk translate mapping
    # trunk_translate=0.0 -> torso_upright, 1.0 -> torso_downward, 2.0 -> torso_ground
    torso_upright: th.Tensor
    torso_downward: th.Tensor
    torso_ground: th.Tensor
    trunk_translate_range: Tuple[float, float] = (0.0, 2.0)

    # Link names (only if asset defaults need overriding)
    wrist_camera_link: Dict[str, str] = field(default_factory=dict)
    head_camera_link: Optional[str] = None

    # Camera local offsets (only if asset defaults need correction)
    wrist_camera_pos: Optional[th.Tensor] = None
    wrist_camera_ori: Optional[th.Tensor] = None
    head_camera_pos: Optional[th.Tensor] = None
    head_camera_ori: Optional[th.Tensor] = None

    # Arm geometry: direction multiplier for shoulder tilt
    shoulder_directions: Dict[str, float] = field(
        default_factory=lambda: {"left": -1.0, "right": 1.0}
    )

    # Ghost robot joint structure
    arm_dof: int = 6
    finger_joints_per_arm: int = 2
    torso_joint_count: int = 4


# Pre-defined robot configurations
ROBOT_CONFIGS: Dict[str, RobotTeleopConfig] = {
    "r1": RobotTeleopConfig(
        robot_type="r1",
        controller_config={
            "arm_left": {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
            },
            "arm_right": {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
            },
            "gripper_left": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                "command_input_limits": "default",
                "command_output_limits": "default",
            },
            "gripper_right": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                "command_input_limits": "default",
                "command_output_limits": "default",
            },
            "base": {
                "name": "HolonomicBaseJointController",
                "motor_type": "velocity",
                "vel_kp": 150,
                "command_input_limits": [-th.ones(3), th.ones(3)],
                "command_output_limits": [
                    -th.tensor([0.75, 0.75, 1.0]),
                    th.tensor([0.75, 0.75, 1.0]),
                ],
                "use_impedances": False,
            },
            "trunk": {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
            },
            "camera": {
                "name": "NullJointController",
            },
        },
        reset_joint_pos=(
            th.tensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,  # 6 virtual base joints
                    0,
                    0,
                    0,
                    0,  # 4 trunk joints
                    33,
                    -33,  # L, R arm joints
                    162,
                    162,
                    -108,
                    -108,
                    34,
                    -34,
                    73,
                    -73,
                    -65,
                    65,
                    0,
                    0,  # 2 L gripper
                    0,
                    0,  # 2 R gripper
                ]
            )
            * th.pi
            / 180
        ),
        torso_upright=th.tensor([0.45, -0.4, 0.0, 0.0], dtype=th.float32),
        torso_downward=th.tensor([1.6, -2.5, -0.94, 0.0], dtype=th.float32),
        torso_ground=th.tensor([1.735, -2.57, -2.1, 0.0], dtype=th.float32),
        wrist_camera_link={"left": "left_eef_link", "right": "right_eef_link"},
        head_camera_link="eyes",
        wrist_camera_pos=th.tensor([0.1, 0.0, -0.1], dtype=th.float32),
        wrist_camera_ori=th.tensor(
            [
                0.6830127018922194,
                0.6830127018922193,
                0.18301270189221927,
                0.18301270189221946,
            ],
            dtype=th.float32,
        ),
        arm_dof=6,
        finger_joints_per_arm=2,
        torso_joint_count=4,
    ),
    "r1pro": RobotTeleopConfig(
        robot_type="r1pro",
        controller_config={
            "arm_left": {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
            },
            "arm_right": {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
            },
            "gripper_left": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                "command_input_limits": "default",
                "command_output_limits": "default",
            },
            "gripper_right": {
                "name": "MultiFingerGripperController",
                "mode": "smooth",
                "command_input_limits": "default",
                "command_output_limits": "default",
            },
            "base": {
                "name": "HolonomicBaseJointController",
                "motor_type": "velocity",
                "vel_kp": 150,
                "command_input_limits": [-th.ones(3), th.ones(3)],
                "command_output_limits": [
                    -th.tensor([0.75, 0.75, 1.0]),
                    th.tensor([0.75, 0.75, 1.0]),
                ],
                "use_impedances": False,
            },
            "trunk": {
                "name": "JointController",
                "motor_type": "position",
                "pos_kp": 150,
                "command_input_limits": None,
                "command_output_limits": None,
                "use_impedances": False,
                "use_delta_commands": False,
            },
            "camera": {
                "name": "NullJointController",
            },
        },
        reset_joint_pos=th.zeros(28) * th.pi / 180,
        torso_upright=th.tensor([0.45, -0.4, 0.0, 0.0], dtype=th.float32),
        torso_downward=th.tensor([1.6, -2.5, -0.94, 0.0], dtype=th.float32),
        torso_ground=th.tensor([1.735, -2.57, -2.1, 0.0], dtype=th.float32),
        wrist_camera_link={
            "left": "left_realsense_link",
            "right": "right_realsense_link",
        },
        head_camera_link="zed_link",
        head_camera_pos=th.tensor([0.06, 0.0, 0.01], dtype=th.float32),
        head_camera_ori=th.tensor([-1.0, 0.0, 0.0, 0.0], dtype=th.float32),
        arm_dof=7,
        finger_joints_per_arm=2,
        torso_joint_count=4,
    ),
}

# Default parameters
DEFAULT_TRUNK_TRANSLATE = 0.5
DEFAULT_RESET_DELTA_SPEED = 10.0  # deg / sec
N_COOLDOWN_SECS = 1.5
FLASHLIGHT_INTENSITY = 2000.0

# Visualization settings
RESOLUTION = [1080, 1080]  # [H, W]
# Newton GL viewer only (see OGRobotServer._setup_newton_viewer_panels): resolution of the docked
# secondary camera panels that stand in for Kit's docked viewports, and how many viewer ticks pass
# between panel refreshes. PANEL_RESOLUTION mirrors the 256x256 texture resolution setup_cameras()
# gives those Kit viewports. Each refresh renders ONE panel (round-robin), so with the default 6
# panels (4 secondary cameras + the 2 EXTRA_VIEWPOINT_CAMERA_CONFIGS gripper views) and a 30 fps
# viewer the stride below means every panel updates at 30 / (6 * 1) = 5 fps for ~3 ms of extra work
# per tick; raise it to spend less, at the cost of staler panels.
PANEL_RESOLUTION = (256, 256)  # (W, H)
PANEL_EVERY_N_FRAMES = 1
# Name of the single viewer window all panels are logged into. Newton's image logger shows exactly
# ONE logged name at a time (picked from its sidebar dropdown, auto-selecting whichever name was
# logged first), but a batched log under one name renders as a tile grid -- so all the panels go out
# together under this name to be visible at once. Renaming it renames that window.
PANEL_WINDOW_NAME = "robot cameras"
USE_VISUAL_SPHERES = False
USE_VERTICAL_VISUALIZERS = False
GHOST_APPEAR_THRESHOLD = 0.1
GHOST_APPEAR_TIME = 10
USE_REACHABILITY_VISUALIZERS = True
AUTO_CHECKPOINTING = False
STEPS_TO_AUTO_CHECKPOINT = 6000  # ~5 min at 20fps

# Visualization cylinder configs
VIS_GEOM_COLORS = {
    False: [
        th.tensor([1.0, 0, 0]),
        th.tensor([0, 1.0, 0]),
        th.tensor([0, 0, 1.0]),
    ],
    True: [
        th.tensor([1.0, 0.5, 0.5]),
        th.tensor([0.5, 1.0, 0.5]),
        th.tensor([0.5, 0.5, 1.0]),
    ],
}
BEACON_LENGTH = 5.0

# Global whitelist of visual-only objects
VISUAL_ONLY_CATEGORIES = {
    # "bush",
    # "tree",
    # "pot_plant",
}

# Global whitelist of task-relevant objects
EXTRA_TASK_RELEVANT_CATEGORIES = {
    "floors",
    "driveway",
    "lawn",
}

# OmniGibson simulator settings
OMNIGIBSON_MACROS = {
    "USE_NUMPY_CONTROLLER_BACKEND": True,
    "USE_GPU_DYNAMICS": (USE_FLUID or USE_CLOTH),
    "ENABLE_OBJECT_STATES": True,
    "ENABLE_TRANSITION_RULES": True,
    "ENABLE_CCD": True,
    "ENABLE_HQ_RENDERING": USE_FLUID,
    "GUI_VIEWPORT_ONLY": True,
}
REACHABILITY_VISUALIZER_CONFIG = {
    "beam_width": 0.005,
    "square_distance": 0.6,
    "square_width": 0.4,
    "square_height": 0.3,
    "beam_color": [0.7, 0.7, 0.7],
}

# Visualization cylinder configurations
VIS_CYLINDER_CONFIG = {
    "width": 0.01,
    "lengths": [0.25, 0.25, 0.5],  # x,y,z
    "proportion_offsets": [0.0, 0.0, 0.5],  # x,y,z
    "quat_offsets": [
        T.euler2quat(th.tensor([0.0, th.pi / 2, 0.0])),
        T.euler2quat(th.tensor([-th.pi / 2, 0.0, 0.0])),
        T.euler2quat(th.tensor([0.0, 0.0, 0.0])),
    ],
}

# External camera parameters
EXTERNAL_CAMERA_CONFIGS = {
    "external_sensor0": {
        "position": [-0.4, 0, 2.0],
        "orientation": [0.2706, -0.2706, -0.6533, 0.6533],
    },
    "external_sensor1": {
        "position": [-0.2, 0.6, 2.0],
        "orientation": [-0.1930, 0.4163, 0.8062, -0.3734],
    },
    "external_sensor2": {
        "position": [-0.2, -0.6, 2.0],
        "orientation": [0.4164, -0.1929, -0.3737, 0.8060],
    },
}

# Extra viewpoint cameras docked as panels inside the Newton GL viewer (see
# OGRobotServer._setup_newton_viewer_panels). Unlike the shoulder/wrist panels -- which reuse camera
# prims the robot asset and the env's external sensors already provide -- these are plain USD camera
# prims that setup_cameras_standalone() creates on the fly, so any robot link can be given a
# viewpoint without adding a VisionSensor for it (which would also add it to the robot's observation
# space, and therefore to recorded data, as a side effect). Newton-viewer only: the Kit path has its
# own docked viewports and ignores these.
#
# Keys are the panel labels (tiles of the PANEL_WINDOW_NAME window). Every entry names what the
# camera is mounted on and where it sits:
#   - "arm": mount on that arm's gripper body (the link its fingers hang off -- see
#     _gripper_mount_link), with the pose below authored in the GRASP frame: +z points out
#     through the fingertips, +/-y is the finger-opening axis, so +/-x is the side a camera can watch
#     the jaws from without a finger in the way. Robot-agnostic, so no per-robot link names needed.
#     Mutually exclusive with "link", which names a link on the robot directly and takes the pose in
#     that link's own frame.
#   - "position" / "look_at": the camera sits at "position" and aims at "look_at", with "up"
#     (default +z) breaking the roll tie. Note that the Newton viewer re-derives its camera from
#     position + view direction only, keeping image-up aligned with WORLD up, so the roll "up"
#     implies is authored on the prim but not honored there (it only matters to consumers that use
#     the prim's full orientation) -- a gripper view therefore rotates with the wrist.
#   - "focal_length" / "horizontal_aperture": optional USD camera intrinsics; the defaults below give
#     a ~83 deg horizontal fov, wide enough to keep both fingers in frame from this close.
# The gripper defaults below sit on the +x face of the gripper body, just behind the jaws, and look
# out along the approach direction past the fingertips -- so the jaws frame the near field and
# whatever is about to be grasped sits ahead of them. Verified by capture against r1pro.
# Each entry costs one more off-screen render in the viewer's round-robin, so adding cameras here
# lowers every panel's refresh rate proportionally (see PANEL_EVERY_N_FRAMES).
EXTRA_VIEWPOINT_CAMERA_CONFIGS = {
    "left_gripper": {
        "arm": "left",
        "position": [0.08, 0.0, -0.12],
        "look_at": [0.0, 0.0, 0.2],
    },
    "right_gripper": {
        "arm": "right",
        "position": [0.08, 0.0, -0.12],
        "look_at": [0.0, 0.0, 0.2],
    },
}
VIEWPOINT_CAMERA_UP = [0.0, 0.0, 1.0]
VIEWPOINT_CAMERA_FOCAL_LENGTH = 17.0
VIEWPOINT_CAMERA_HORIZONTAL_APERTURE = 30.0

# UI visual settings
UI_SETTINGS = {
    "goal_satisfied_color": 0xFF00FF00,  # Green (ABGR)
    "goal_unsatisfied_color": 0xFF0000FF,  # Red (ABGR)
    "font_size": 25,
    "top_margin": 50,
    "left_margin": 50,
}

# Status display settings
STATUS_DISPLAY_SETTINGS = {
    "event_duration": 3.0,  # seconds
    "persistent_duration": 0.1,  # For persistent events - very short
    "persistent_states": ["in_cooldown", "waiting_to_resume"],
    "event_colors": {
        "checkpoint": 0xFF00FF00,  # Green
        "rollback": 0xFFFF00FF,  # Magenta
        "cooldown": 0xFF00FFFF,  # Yellow
        "waiting": 0xFFFF0000,  # White
        "reset": 0xFF00AAFF,  # Orange
    },
    "font_size": 20,
    "bottom_margin": 50,
    "right_margin": 50,
    "line_spacing": 5,
}

INCLUDE_TRUNK_CONTACT_OBS = True
INCLUDE_BASE_CONTACT_OBS = True
INCLUDE_ARM_CONTACT_OBS = False

INCLUDE_JACOBIAN_OBS = False
GHOST_UPDATE_FREQ = 3

BLINK_WHEN_IN_CONTACT = True

DISABLED_TRANSITION_RULES = [ToggleableMachineRule, MixingToolRule, CookingSystemRule]
