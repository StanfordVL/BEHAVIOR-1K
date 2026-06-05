"""
VR teleoperation demo for Franka + SharpaWave dexterous hand.

Usage:
  # VR teleop with wrist camera views
  python vr_sharpa_teleop.py --hand right --view all

  # Desktop-only (no VR headset), single wrist view
  python vr_sharpa_teleop.py --no-vr --hand right --view wrist_palm

  # Headless render to video
  OMNIGIBSON_HEADLESS=true python vr_sharpa_teleop.py --no-vr --view all --save-video out.mp4 --steps 300

Options:
  --hand right|left          Which hand (default: right)
  --view front|top|side|ego|wrist_back|wrist_palm|all
                             Camera view (default: front)
  --no-vr                    Desktop-only mode, no Quest needed
  --save-video PATH          Record viewer camera to mp4
  --steps N                  Sim steps (default: 10000)
  --mode controller|hand     VR tracking mode (default: controller)
  --scene MODEL              Scene model (default: Rs_int)
  --no-vr-display            Disable VR headset mirror (saves GPU)
"""

import argparse
import math
import os
import time

import av
import numpy as np
import torch as th
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.teleop_utils import OVXRSystem
from omnigibson.utils.ui_utils import KeyboardEventHandler
from omnigibson.controllers.controller_view import ControllerView

# Import this BEFORE setting gm flags below — build_arat_scene sets its own gm
# values at module import, and we want our settings (FLATCACHE on) to win.
import build_arat_scene  # noqa: F401

gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True
# Required: switches OmniGibson to the omnigibson_5_1_0_vr.kit experience file,
# which loads omni.kit.xr.* extensions. Without this, OVXRSystem.__init__ fails
# with `module lazy_omni.kit has no attribute xr`.
gm.ENABLE_VR = True

LOG_FILE = "/home/eeg/Desktop/BEHAVIOR-1K/vr_teleop_debug.log"


# Equivalent to MuJoCo's <key name="home"/>; without this the low-effort
# fingers drift past their URDF limits at startup under gravity.
_RESET_POSE = th.tensor(
    [-6e-4, -1.30, 6e-4, -2.87, 1e-3, 1.999, 0.749]   # 7 Franka arm joints
    + [0.0] * 22                                       # 22 SharpaWave finger joints
)


# Per-joint effort multipliers — pinky starved (cmd=0 + effort 1×) to break
# the chain-runaway feedback loop; moderate boosts for the 4 gripping fingers.
# See STATE_VR_TELEOP.md for the tuning history that led to these numbers.
_PINKY_CMC_BOOST      =   1.0
_PINKY_MCP_FE_BOOST   =   1.0
_PINKY_MCP_AA_BOOST   =   1.0
_PINKY_PIP_BOOST      =   1.0
_INDEX_MCP_FE_BOOST   =  25.0
_MIDDLE_MCP_FE_BOOST  =  25.0
_RING_MCP_FE_BOOST    =  30.0


def _build_per_joint_effort(default_mult: float, dof_names_ordered) -> dict:
    """Map every joint name in `dof_names_ordered` → effort multiplier."""
    overrides = {
        "right_pinky_CMC":     _PINKY_CMC_BOOST,
        "right_pinky_MCP_FE":  _PINKY_MCP_FE_BOOST,
        "right_pinky_MCP_AA":  _PINKY_MCP_AA_BOOST,
        "right_pinky_PIP":     _PINKY_PIP_BOOST,
        "right_index_MCP_FE":  _INDEX_MCP_FE_BOOST,
        "right_middle_MCP_FE": _MIDDLE_MCP_FE_BOOST,
        "right_ring_MCP_FE":   _RING_MCP_FE_BOOST,
    }
    return {name: overrides.get(name, default_mult) for name in dof_names_ordered}


def _robot_cfg(hand_side: str, position=None, orientation=None):
    robot_model = "franka_mounted_sharpa_right" if hand_side == "right" else "franka_mounted_sharpa_left"
    cfg = {
        "model": robot_model,
        "grasping_direction": "upper",
        "obs_modalities": ["rgb"],
        "action_normalize": False,
        "fixed_base": True,
        "reset_joint_pos": _RESET_POSE,
        "controller_config": {
            "arm_0": {
                "name": "InverseKinematicsController",
                "mode": "absolute_pose",
                "command_input_limits": None,
                "command_output_limits": None,
            },
            "gripper_0": {
                "name": "MultiFingerGripperController",
                "mode": "independent",
                "motor_type": "position",
                "inverted": False,
                "command_input_limits": None,
                "command_output_limits": None,
            },
        },
    }
    if position is not None:
        cfg["position"] = position
    if orientation is not None:
        cfg["orientation"] = orientation
    return cfg


def build_config(hand_side: str, scene_model: str, scene_file=None, arat=False):
    if arat:
        from build_arat_scene import OBJECTS, build_object_cfgs
        obj_cfgs = build_object_cfgs(OBJECTS)
        obj_cfgs.extend([
            {"type": "LightObject", "light_type": "Sphere", "name": "light0",
             "radius": 0.01, "intensity": 1e5, "position": [-2, -2, 2]},
            {"type": "LightObject", "light_type": "Sphere", "name": "light1",
             "radius": 0.01, "intensity": 1e5, "position": [2, 2, 2]},
        ])
        return dict(
            scene={"type": "Scene"},
            objects=obj_cfgs,
            robots=[_robot_cfg(hand_side, position=[-0.4, 0.25, -0.08], orientation=[0, 0, 0, 1])],
        )
    if scene_file is not None:
        return dict(
            scene={"type": "Scene", "scene_file": scene_file},
            robots=[_robot_cfg(hand_side, position=[-0.4, 0.25, -0.08], orientation=[0, 0, 0, 1])],
        )
    return dict(
        scene={"type": "InteractiveTraversableScene", "scene_model": scene_model},
        robots=[_robot_cfg(hand_side)],
    )


# ---------------------------------------------------------------------------
# Wrist-mounted tracking cameras (move with the hand each frame).
# All offsets & orientations are in the hand_C_MC local frame.
# Hand frame: +Z = finger direction, +X = palm/dorsal axis.
# ---------------------------------------------------------------------------
WRIST_VIEWS = {
    "wrist_back": {
        "link_suffix": "hand_C_MC",
        "offset": (-0.09, 0.0, -0.05),
        "orn": (0.7010, -0.7010, -0.0923, 0.0923),
        "desc": "Mirror of palm: -X (dorsal) side, looking +Z along fingers.",
    },
    "wrist_palm": {
        "link_suffix": "hand_C_MC",
        "offset": (0.09, 0, -0.05),
        "orn": (0.7010, 0.7010, 0.0923, 0.0923),
        "desc": "At +X, looking +Z along fingers. Screen up = +X.",
    },
}


def compute_tracking_pose(robot, link_name, local_offset, local_orn):
    """Compute world-frame position + orientation for a camera tracking a robot link."""
    link = robot.links[link_name]
    link_pos, link_quat = link.get_position_orientation()
    world_offset = T.quat_apply(link_quat, local_offset)
    return link_pos + world_offset, T.quat_multiply(link_quat, local_orn)


def update_tracking_camera(robot, link_name, local_offset, local_orn):
    """Update viewer_camera to follow a robot link with a local offset each frame."""
    cam_pos, cam_orn = compute_tracking_pose(robot, link_name, local_offset, local_orn)
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_orn)


def create_camera_prim(cam_path):
    """Create a USD Camera prim with translate+orient xform ops (matching VisionSensor)."""
    stage = og.sim.stage
    cam = lazy.pxr.UsdGeom.Camera.Define(stage, cam_path)
    cam.GetClippingRangeAttr().Set(lazy.pxr.Gf.Vec2f(0.001, 10000000.0))
    cam.GetFocalLengthAttr().Set(17.0)
    prim = stage.GetPrimAtPath(cam_path)
    xformable = lazy.pxr.UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    orient_op = xformable.AddOrientOp()
    return (translate_op, orient_op)


def set_camera_xform(ops, position, orientation):
    """Set camera world transform using separate translate+orient ops.
    orientation is (x, y, z, w) quaternion."""
    translate_op, orient_op = ops
    p = [float(x) for x in position]
    q = [float(x) for x in orientation]  # x, y, z, w
    translate_op.Set(lazy.pxr.Gf.Vec3d(*p))
    orient_op.Set(lazy.pxr.Gf.Quatf(q[3], q[0], q[1], q[2]))


def create_vision_sensor(prim_path, name, image_height=480, image_width=640, viewport_name=None):
    """Create a VisionSensor that can render RGB offscreen."""
    sensor = VisionSensor(
        relative_prim_path=prim_path,
        name=name,
        modalities="rgb",
        image_height=image_height,
        image_width=image_width,
        viewport_name=viewport_name,
    )
    sensor.load(None)
    sensor.clipping_range = [0.001, 10000000.0]
    sensor.focal_length = 17.0
    sensor.initialize()
    return sensor


def get_sensor_rgb(sensor):
    """Get RGB numpy array (H, W, 3) uint8 from a VisionSensor."""
    obs = sensor.get_obs()[0]
    rgb = obs["rgb"][:, :, :3]
    if isinstance(rgb, th.Tensor):
        rgb = rgb.cpu().numpy()
    return np.ascontiguousarray(rgb).astype(np.uint8)


def tile_views(main_rgb, wrist_rgbs, target_h=480):
    """Tile main (ego) view with wrist views into a single composite frame.

    Layout: [  ego (large)  | wrist_palm ]
            [               | wrist_back ]
    """
    h_main, w_main = main_rgb.shape[:2]
    scale = target_h / h_main
    new_w_main = int(w_main * scale)
    main_resized = np.array(
        av.VideoFrame.from_ndarray(main_rgb, format="rgb24")
        .reformat(width=new_w_main, height=target_h)
        .to_ndarray(format="rgb24")
    )

    n_wrist = len(wrist_rgbs)
    wrist_h = target_h // max(n_wrist, 1)
    wrist_resized = []
    for wrgb in wrist_rgbs:
        wr_h, wr_w = wrgb.shape[:2]
        wr_scale = wrist_h / wr_h
        new_wr_w = int(wr_w * wr_scale)
        wr = np.array(
            av.VideoFrame.from_ndarray(wrgb, format="rgb24")
            .reformat(width=new_wr_w, height=wrist_h)
            .to_ndarray(format="rgb24")
        )
        wrist_resized.append(wr)

    if wrist_resized:
        wrist_w = max(wr.shape[1] for wr in wrist_resized)
        wrist_stack_parts = []
        for wr in wrist_resized:
            if wr.shape[1] < wrist_w:
                pad = np.zeros((wr.shape[0], wrist_w - wr.shape[1], 3), dtype=np.uint8)
                wr = np.concatenate([wr, pad], axis=1)
            wrist_stack_parts.append(wr)
        wrist_col = np.concatenate(wrist_stack_parts, axis=0)
        if wrist_col.shape[0] < target_h:
            pad = np.zeros((target_h - wrist_col.shape[0], wrist_col.shape[1], 3), dtype=np.uint8)
            wrist_col = np.concatenate([wrist_col, pad], axis=0)
        elif wrist_col.shape[0] > target_h:
            wrist_col = wrist_col[:target_h]
        composite = np.concatenate([main_resized, wrist_col], axis=1)
    else:
        composite = main_resized

    return composite


def setup_all_views(robot, hand_side, use_main_viewport=False, headless=False):
    """
    Set up ego view as an independent VisionSensor + wrist tracking cameras.
    Returns (ego_sensor, wrist_trackers).
    The ego sensor is static (positioned once); wrist trackers need per-frame updates.
    """
    ego_key = f"ego_{hand_side}"
    v = AXIS_ALIGNED_VIEWS[ego_key]

    base_pos, base_orn = robot.get_position_orientation()
    local_offset = th.tensor(v["offset"], dtype=th.float32)
    local_cam_orn = th.tensor(v["orn"], dtype=th.float32)
    shoulder_height = th.tensor([0.0, 0.0, 1.2], dtype=th.float32)
    world_offset = T.quat_apply(base_orn, local_offset + shoulder_height)
    ego_pos = base_pos + world_offset
    ego_orn = T.quat_multiply(base_orn, local_cam_orn)

    ego_sensor = create_vision_sensor("/ego_cam", "ego_cam", image_height=480, image_width=640)
    ego_sensor.set_position_orientation(position=ego_pos, orientation=ego_orn)

    # Hide the auto-created viewport (VisionSensor always makes one internally)
    if ego_sensor._viewport is not None:
        ego_sensor._viewport.visible = False

    _VP_LAYOUT_X = 0
    _VP_LAYOUT_Y = 30
    _VP_EGO_W, _VP_EGO_H = 640, 480
    _VP_WRIST_W, _VP_WRIST_H = 400, 300

    # NOTE (Isaac Sim 5.1 / Kit 107.3): omni.kit.viewport.utility.create_viewport_window
    # changed signature — position_x/position_y/camera_path must now be passed as
    # kwargs AT CREATION; assigning them as attributes on the returned window is
    # silently a no-op in 5.1 (the new ViewportWindow exposes setPosition() instead).
    # Older 4.x scripts that did `vp_win.position_x = X` end up with all viewports
    # stacked at (0,0), which looks like "tiled views aren't appearing".
    if not headless:
        try:
            vp_win = lazy.omni.kit.viewport.utility.create_viewport_window(
                "Ego View",
                width=_VP_EGO_W, height=_VP_EGO_H,
                position_x=_VP_LAYOUT_X, position_y=_VP_LAYOUT_Y,
                camera_path=ego_sensor.prim_path,
            )
            print(f"[Multi-view] Viewport 'Ego View' → {ego_sensor.prim_path}  "
                  f"@ ({_VP_LAYOUT_X},{_VP_LAYOUT_Y}) {_VP_EGO_W}x{_VP_EGO_H}")
        except Exception as e:
            print(f"[Multi-view] Could not create ego viewport: {e}")
    print(f"[Multi-view] Ego sensor at {[round(x,2) for x in ego_pos.tolist()]}  ({v['desc']})")

    _wrist_x = _VP_LAYOUT_X + _VP_EGO_W + 10
    trackers = []
    for wi, (view_name, wv) in enumerate(WRIST_VIEWS.items()):
        prim_path = f"/{view_name}_cam"
        link_name = f"{hand_side}_{wv['link_suffix']}"
        offset = th.tensor(wv["offset"], dtype=th.float32)
        orn = th.tensor(wv["orn"], dtype=th.float32)

        sensor = create_vision_sensor(prim_path, view_name)
        pos, quat = compute_tracking_pose(robot, link_name, offset, orn)
        sensor.set_position_orientation(position=pos, orientation=quat)

        if sensor._viewport is not None:
            sensor._viewport.visible = False

        if not headless:
            try:
                label = view_name.replace("_", " ").title()
                _wy = _VP_LAYOUT_Y + wi * (_VP_WRIST_H + 10)
                vp_win = lazy.omni.kit.viewport.utility.create_viewport_window(
                    label,
                    width=_VP_WRIST_W, height=_VP_WRIST_H,
                    position_x=_wrist_x, position_y=_wy,
                    camera_path=sensor.prim_path,
                )
                print(f"[Multi-view] Viewport '{label}' → {sensor.prim_path}  "
                      f"@ ({_wrist_x},{_wy}) {_VP_WRIST_W}x{_VP_WRIST_H}  (tracking {link_name})")
            except Exception as e:
                print(f"[Multi-view] Could not create viewport for {view_name}: {e}")
        else:
            print(f"[Multi-view] Sensor '{view_name}' → {sensor.prim_path}  (tracking {link_name})")

        trackers.append((sensor, link_name, offset, orn))

    return ego_sensor, trackers


def update_all_trackers(robot, trackers):
    """Update every wrist-tracking camera each frame."""
    for sensor, link_name, offset, orn in trackers:
        pos, quat = compute_tracking_pose(robot, link_name, offset, orn)
        sensor.set_position_orientation(position=pos, orientation=quat)


AXIS_ALIGNED_VIEWS = {
    "front": {
        "offset": (0.0, -2.5, 0.0),
        "orn": (0.7071, 0.0, 0.0, 0.7071),
        "desc": "Front: robot +X = screen right, +Z = screen up",
    },
    "top": {
        "offset": (0.0, 0.0, 2.5),
        "orn": (0.0, 0.0, 0.0, 1.0),
        "desc": "Top-down: robot +X = screen right, +Y = screen up",
    },
    "side": {
        "offset": (2.5, 0.0, 0.0),
        "orn": (0.5, 0.5, 0.5, 0.5),
        "desc": "Right side: robot +Y = screen right, +Z = screen up",
    },
    "ego_right": {
        "offset": (-0.8, -0.4, 0.7),
        "orn": (0.326, -0.326, -0.627, 0.627),
        "desc": "Ego (right arm): behind-left, tilted down ~20° into workspace",
    },
    "ego_left": {
        "offset": (-0.8, 0.4, 0.6),
        "orn": (0.430, -0.430, -0.561, 0.561),
        "desc": "Ego (left arm): behind-right, looking forward-left into workspace",
    },
}


def setup_third_person_viewport(robot, view="front", use_main_viewport=False):
    """
    Set up a camera view. If use_main_viewport, use the primary viewport directly.
    Otherwise create a second viewport (VR takes the main one).

    Offsets are in the robot's local frame so the camera stays correct
    regardless of how the robot is placed/rotated in the scene.
    """
    base_pos, base_orn = robot.get_position_orientation()

    v = AXIS_ALIGNED_VIEWS[view]
    local_offset = th.tensor(v["offset"], dtype=th.float32)
    local_cam_orn = th.tensor(v["orn"], dtype=th.float32)

    shoulder_height = th.tensor([0.0, 0.0, 1.2], dtype=th.float32)
    world_offset = T.quat_apply(base_orn, local_offset + shoulder_height)
    cam_pos = base_pos + world_offset
    cam_orn = T.quat_multiply(base_orn, local_cam_orn)

    print(f"[Camera] base_pos={[round(x,3) for x in base_pos.tolist()]}  "
          f"base_orn={[round(x,3) for x in base_orn.tolist()]}")

    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_orn)
    print(f"[Camera] {v['desc']}")
    print(f"[Camera] pos={[round(x, 2) for x in cam_pos.tolist()]}")

    if use_main_viewport:
        try:
            viewport = lazy.omni.kit.viewport.utility.get_active_viewport()
            viewport.camera_path = "/World/viewer_camera"
            print(f"[Camera] Main viewport → /World/viewer_camera")
        except Exception:
            pass
    else:
        try:
            # See note in setup_all_views: 5.1 needs camera_path as a kwarg, not attr.
            vp_window = lazy.omni.kit.viewport.utility.create_viewport_window(
                "Third Person", width=800, height=600,
                camera_path="/World/viewer_camera",
            )
            print(f"[Camera] Second viewport 'Third Person' → /World/viewer_camera")
            print(f"[Camera] Main viewport stays on xrCamera for VR.")
        except Exception as e:
            print(f"[Camera] Could not create second viewport ({e}), falling back to main viewport")
            try:
                viewport = lazy.omni.kit.viewport.utility.get_active_viewport()
                viewport.camera_path = "/World/viewer_camera"
            except Exception:
                pass

    try:
        og.sim.enable_viewer_camera_teleoperation()
        print("[Camera] WASD+mouse enabled for viewer_camera adjustment")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--scene", default="Rs_int",
                        help="InteractiveTraversableScene model name (ignored when --scene-file is set)")
    parser.add_argument("--scene-file", type=str, default=None,
                        help="Path to a saved scene JSON (uses bare Scene instead of InteractiveTraversableScene)")
    parser.add_argument("--arat", action="store_true",
                        help="Use ARAT scene (table + objects from build_arat_scene.py, robot at [-0.4, 0, -0.08])")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--mode", choices=["hand", "controller"], default="controller")
    parser.add_argument("--view",
                        choices=["front", "top", "side", "ego", "ego_right", "ego_left",
                                 "wrist_back", "wrist_palm", "all"],
                        default="front",
                        help="Camera view: front/top/side, ego, wrist_back/wrist_palm, or 'all' for multi-view")
    parser.add_argument("--no-vr-display", action="store_true",
                        help="Disable VR headset display (desktop-only mode, saves GPU)")
    parser.add_argument("--no-vr", action="store_true",
                        help="View-only mode: no Quest needed, just visualize the robot + camera")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save viewer camera to mp4 file (e.g. --save-video output.mp4)")
    # === TEST: HAND OPEN/CLOSE -- arg begin ===
    parser.add_argument("--test-hand", action="store_true",
                        help="(TEST) In --no-vr mode, alternate hand close/open every 50 frames via env.step")
    # === TEST: HAND OPEN/CLOSE -- arg end ===
    # ----------------------------------------------------------------------
    # Hand-tracking offline test flags (added 2026-04 for Phase 1-3 dev).
    # See OmniGibson/omnigibson/utils/hand_retargeting.py for assumptions.
    #
    # --mock-hand          : in --no-vr mode, build a procedural 5x3 OpenXR
    #                        bend tensor each frame (open->close cycle), wrap
    #                        it in a TeleopAction, and route through
    #                        robot.teleop_data_to_action() so the full
    #                        retargeting + smoothing path is exercised.
    # --record-hand PATH   : when running with a real Quest in --mode hand,
    #                        pickle the frame-by-frame hand_data + palm pose
    #                        to PATH for later offline replay.
    # --replay-hand PATH   : in --no-vr mode, load saved frames from PATH and
    #                        feed them into the same retargeting path.
    # ----------------------------------------------------------------------
    parser.add_argument("--mock-hand", action="store_true",
                        help="(no-vr) Build synthetic OpenXR 5x3 bend tensor each frame "
                             "(open<->close cycle) and route it through teleop_data_to_action.")
    parser.add_argument("--record-hand", type=str, default=None,
                        help="(VR mode) Pickle live OpenXR hand_data + palm pose to this path.")
    parser.add_argument("--replay-hand", type=str, default=None,
                        help="(no-vr) Load hand_data frames from this pickle path and replay them.")
    args = parser.parse_args()

    if args.view == "ego":
        args.view = f"ego_{args.hand}"
        print(f"[View] ego → {args.view} (based on --hand {args.hand})")

    logf = open(LOG_FILE, "w")

    def log(msg):
        logf.write(msg + "\n")
        logf.flush()

    cfg = build_config(args.hand, args.scene, scene_file=args.scene_file, arat=args.arat)
    env = og.Environment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    # right_hand_C_MC is the physical palm — restore visibility hidden by robot._initialize()
    for arm in robot.arm_names:
        robot.links[robot.eef_link_names[arm]].visible = True

    # Enable hand tracking component AFTER OG init (so carb is available) but
    # BEFORE OVXRSystem.start() which creates the OpenXR instance.
    if args.mode == "hand":
        lazy.carb.settings.get_settings().set(
            'xr.openxr.components."omni.kit.xr.openxr.ext.hand_tracking".enabled', True)
        lazy.carb.settings.get_settings().set(
            'xr.openxr.components."isaacsim.xr.openxr.hand_tracking".enabled', True)
        print("[Hand] Set hand tracking component carb settings")

    base_pos, base_orn = robot.get_position_orientation()
    eef_pos, eef_orn = robot.eef_links[robot.arm_names[0]].get_position_orientation()
    eef_rel, _ = T.relative_pose_transform(eef_pos, eef_orn, base_pos, base_orn)
    log(f"=== INIT ===")
    log(f"robot_class={robot.__class__.__name__}")
    log(f"base_pos={base_pos.tolist()}")
    log(f"base_orn={base_orn.tolist()}")
    log(f"eef_world={eef_pos.tolist()}")
    log(f"eef_in_base={eef_rel.tolist()}")
    log(f"arm_joints={robot.get_joint_positions()[:7].tolist()}")
    log(f"all_link_names={list(robot.links.keys())[:15]}")

    link0_pos = robot.links.get("panda_link0")
    if link0_pos is not None:
        l0_pos, l0_orn = link0_pos.get_position_orientation()
        l0_rel = l0_pos - base_pos
        log(f"panda_link0_world={l0_pos.tolist()}")
        log(f"panda_link0_in_base={l0_rel.tolist()}")

    is_wrist_view = args.view in WRIST_VIEWS
    is_all_view = args.view == "all"
    wrist_track = None
    all_trackers = None

    # Disable OmniGibson's USD-edit guard for the rest of this teleop session.
    # The guard crashes on ANY USD edit outside og.sim.editing_usd(); VR teleop
    # legitimately makes many render-layer edits (viewport creation, `.visible`
    # toggles, replicator SDGPipeline writes during raw app.update()), several
    # inside Omniverse code we can't wrap. Disable rather than wrap per-site.
    if hasattr(og.sim, "_disable_usd_guard"):
        og.sim._disable_usd_guard()
        print("[USD] Edit guard disabled for VR teleop (render-layer edits are expected).")

    arm_name = robot.arm_names[0]
    orig_rot_offset = robot.teleop_rotation_offset[arm_name].tolist()
    log(f"teleop_rotation_offset={orig_rot_offset}")
    print(f"[Teleop] Rotation offset: {orig_rot_offset}")

    ego_sensor = None
    if is_all_view:
        headless = gm.HEADLESS
        ego_sensor, all_trackers = setup_all_views(robot, args.hand, use_main_viewport=args.no_vr, headless=headless)
        log(f"multi_view=all  ego={ego_sensor.prim_path}  wrist_cams={list(WRIST_VIEWS.keys())}")
    elif is_wrist_view:
        wv = WRIST_VIEWS[args.view]
        wrist_link_name = f"{args.hand}_{wv['link_suffix']}"
        wrist_offset = th.tensor(wv["offset"], dtype=th.float32)
        wrist_orn = th.tensor(wv["orn"], dtype=th.float32)
        wrist_track = (wrist_link_name, wrist_offset, wrist_orn)

        if args.no_vr:
            try:
                viewport = lazy.omni.kit.viewport.utility.get_active_viewport()
                viewport.camera_path = "/World/viewer_camera"
            except Exception:
                pass
        else:
            try:
                # See note in setup_all_views: 5.1 needs camera_path as a kwarg.
                vp_window = lazy.omni.kit.viewport.utility.create_viewport_window(
                    "Wrist Camera", width=800, height=600,
                    camera_path="/World/viewer_camera",
                )
            except Exception:
                try:
                    viewport = lazy.omni.kit.viewport.utility.get_active_viewport()
                    viewport.camera_path = "/World/viewer_camera"
                except Exception:
                    pass

        update_tracking_camera(robot, *wrist_track)
        print(f"[Camera] {wv['desc']}")
        print(f"[Camera] Tracking link: {wrist_link_name}")
        print(f"[Camera] offset={wv['offset']}  orn={wv['orn']}")
        link_p, link_q = robot.links[wrist_link_name].get_position_orientation()
        print(f"[Camera] link_pos={[round(x,3) for x in link_p.tolist()]}  "
              f"link_orn={[round(x,3) for x in link_q.tolist()]}")
        cam_p, cam_q = og.sim.viewer_camera.get_position_orientation()
        print(f"[Camera] cam_pos={[round(x,3) for x in cam_p.tolist()]}  "
              f"cam_orn={[round(x,3) for x in cam_q.tolist()]}")
        log(f"wrist_view={args.view} link={wrist_link_name} offset={wv['offset']} orn={wv['orn']}")
    else:
        setup_third_person_viewport(robot, view=args.view, use_main_viewport=args.no_vr)

    video_writer = None
    video_first_frame = True
    if args.save_video:
        vpath = os.path.abspath(args.save_video)
        os.makedirs(os.path.dirname(vpath) or ".", exist_ok=True)
        video_container = av.open(vpath, mode="w")
        video_stream = video_container.add_stream("libx264", rate=30)
        video_stream.pix_fmt = "yuv420p"
        video_writer = (video_container, video_stream)
        print(f"[Video] Recording to {vpath}")

    def capture_frame():
        nonlocal video_first_frame
        if video_writer is None:
            return
        try:
            main_cam = ego_sensor if ego_sensor is not None else og.sim.viewer_camera
            main_rgb = get_sensor_rgb(main_cam)
            if is_all_view and all_trackers:
                wrist_rgbs = [get_sensor_rgb(s) for s, _, _, _ in all_trackers]
                composite = tile_views(main_rgb, wrist_rgbs) if wrist_rgbs else main_rgb
            else:
                composite = main_rgb
            if video_first_frame:
                video_writer[1].width = composite.shape[1]
                video_writer[1].height = composite.shape[0]
                video_first_frame = False
            frame = av.VideoFrame.from_ndarray(composite, format="rgb24")
            for packet in video_writer[1].encode(frame):
                video_writer[0].mux(packet)
        except Exception:
            pass

    if args.no_vr:
        print(f"\n{'='*60}")
        print(f"  View-only mode (no VR): {args.hand} hand, view={args.view}")
        if not args.save_video:
            print(f"  Use WASD + right-click drag to move camera")
        else:
            print(f"  Recording video to: {args.save_video}")
        print(f"  Press Ctrl+C to exit")
        print(f"{'='*60}\n")

        # === TEST: HAND OPEN/CLOSE -- setup begin ===
        _test_hand = False
        _CYCLE_FRAMES = 360
        _RAMP_FRAMES = 180
        if args.test_hand:
            from omnigibson.robots.franka_mounted_sharpa import SHARPA_JOINT_LIMITS
            import sys

            def _log(msg):
                print(msg, file=sys.stderr, flush=True)

            _arm_name = robot.arm_names[0]

            # Hold arm steady via IK
            _eef_pos, _eef_quat = robot.eef_links[_arm_name].get_position_orientation()
            _base_pos, _base_quat = robot.get_position_orientation()
            _eef_rel_pos, _eef_rel_quat = T.relative_pose_transform(
                _eef_pos, _eef_quat, _base_pos, _base_quat
            )
            _arm_hold_cmd = th.cat([_eef_rel_pos, T.quat2axisangle(_eef_rel_quat)])

            # Open = straight (zeros). Close = 85% of upper URDF limit, with margin.
            # AA joints stay 0 in both poses; DIPs follow open. Pinky skipped (chain
            # runaway). See vr_sharpa_teleop.py / STATE_VR_TELEOP.md for rationale.
            _hand_open = th.zeros(22)
            _hand_close = SHARPA_JOINT_LIMITS[:, 1].clone() * 0.85
            for _aa_idx in [1, 3, 6, 10, 14, 19]:
                _hand_close[_aa_idx] = 0.0
                _hand_open[_aa_idx] = 0.0
            for _dip_idx in [4, 8, 12, 16, 21]:
                _hand_close[_dip_idx] = _hand_open[_dip_idx]
            for _pinky_idx in [17, 18, 19, 20, 21]:
                _hand_close[_pinky_idx] = 0.0
                _hand_open[_pinky_idx] = 0.0

            # Per-joint effort (default 20×, pinky 1×, see _build_per_joint_effort).
            _PER_JOINT_EFFORT = _build_per_joint_effort(20.0, robot.dof_names_ordered)
            _gripper_ctrl = robot.controllers.get("gripper_arm_0") or robot.controllers.get("gripper_0")
            if _gripper_ctrl is not None:
                _finger_dof_idx = ControllerView.get_dof_idx(_gripper_ctrl[0])  # robot.controllers[name] = (group_key, controller_idx)
                _effort_limits = robot.control_limits["effort"]
                for dof in _finger_dof_idx:
                    jname = robot.dof_names_ordered[dof]
                    mult = _PER_JOINT_EFFORT[jname]
                    _effort_limits[0][dof] *= mult
                    _effort_limits[1][dof] *= mult
                    robot.joints[jname].max_effort = robot.joints[jname].max_effort * mult
                _log(f"  [TEST] Per-joint effort applied (default 20×, pinky 1×)")

            _test_action = th.zeros(robot.action_dim)
            _log(f"  [TEST] env.step with raw radians, {_CYCLE_FRAMES}f/phase, {_RAMP_FRAMES}f ramp")
            _log(f"  [TEST] close[:5]={[round(x,3) for x in _hand_close[:5].tolist()]}")
            _test_hand = True
        # === TEST: HAND OPEN/CLOSE -- setup end ===

        # ----------------------------------------------------------------------
        # MOCK / REPLAY HAND TRACKING setup (Phase 1-3 offline test path).
        # Assumptions: see OmniGibson/omnigibson/utils/hand_retargeting.py
        # docstring (OpenXR joint order, Sharpa joint order, sign convention).
        # ----------------------------------------------------------------------
        _mock_or_replay = args.mock_hand or args.replay_hand
        _replay_frames = None
        _hand_pose_action = None
        _hand_finger_dof_idx = None
        if _mock_or_replay:
            import sys, pickle as _pickle
            from omnigibson.utils.teleop_utils import TeleopAction as _TA

            def _hlog(msg):
                print(msg, file=sys.stderr, flush=True)
                logf.write(msg + "\n"); logf.flush()

            if args.replay_hand:
                with open(args.replay_hand, "rb") as f:
                    _replay_frames = _pickle.load(f)
                _hlog(f"  [HAND] Loaded {len(_replay_frames)} replay frames from {args.replay_hand}")

            _arm_name = robot.arm_names[0]

            # Cache initial EEF pose in robot frame so the arm "holds still"
            # while we play hand tracking through the gripper.
            _eef_pos, _eef_quat = robot.eef_links[_arm_name].get_position_orientation()
            _base_pos, _base_quat = robot.get_position_orientation()
            _eef_rel_pos, _eef_rel_quat = T.relative_pose_transform(
                _eef_pos, _eef_quat, _base_pos, _base_quat
            )
            _eef_rel_euler = T.quat2euler(_eef_rel_quat)
            # The 7-tuple consumed by robot.teleop_data_to_action: [pos(3),
            # euler(3), trigger(1)].  Trigger=0 because the gripper command
            # comes from hand_data instead.
            _hand_pose_action = th.cat([
                _eef_rel_pos.float(), _eef_rel_euler.float(), th.zeros(1)
            ])

            # On this branch `robot._controllers[name]` stores
            # `(group_key_str, controller_idx)` tuples — the actual controller
            # instance lives at ControllerView._controller_groups[group_key].
            # We resolve through ControllerView to get .dof_idx for both the
            # effort-boost and per-finger logging.
            from omnigibson.controllers import ControllerView as _CV
            _gripper_key_arm = (
                robot._controllers.get("gripper_arm_0")
                or robot._controllers.get("gripper_0")
            )
            if isinstance(_gripper_key_arm, (tuple, list)) and len(_gripper_key_arm) > 0:
                _gripper_group_key = _gripper_key_arm[0]
                _hand_gripper_ctrl = _CV._controller_groups.get(_gripper_group_key)
            else:
                _hand_gripper_ctrl = None
            if _hand_gripper_ctrl is not None and hasattr(_hand_gripper_ctrl, "dof_idx"):
                _hand_finger_dof_idx = ControllerView.get_dof_idx(_hand_gripper_ctrl[0])  # robot.controllers[name] = (group_key, controller_idx)
                _MULT = 20.0
                _effort_limits_h = robot.control_limits["effort"]
                _effort_limits_h[0][_hand_finger_dof_idx] *= _MULT
                _effort_limits_h[1][_hand_finger_dof_idx] *= _MULT
                for _dof in _hand_finger_dof_idx:
                    _jname = robot.dof_names_ordered[_dof]
                    robot.joints[_jname].max_effort = robot.joints[_jname].max_effort * _MULT
                _hlog(f"  [HAND] Effort boosted {_MULT}x for {len(_hand_finger_dof_idx)} finger DOFs")
            else:
                _hlog(f"  [HAND] Could not resolve gripper controller; skipping effort boost")

            # gripper_action_idx slice — should be 22 for Sharpa.
            _hand_gripper_idx = robot.gripper_action_idx[_arm_name]
            _hlog(f"  [HAND] gripper_action_idx={list(_hand_gripper_idx)} (len={len(_hand_gripper_idx)})")
            _hlog(f"  [HAND] action_dim={robot.action_dim}, arm_idx={list(robot.arm_action_idx[_arm_name])}")
            _hlog(f"  [HAND] mode={'mock' if args.mock_hand else 'replay'}, "
                  f"hand={args.hand}, hold pose={[round(x,3) for x in _hand_pose_action[:6].tolist()]}")

        # Procedural mock generator: a smooth open->close->open cycle.
        # Returns a (5, 3) tensor of bend angles in radians, [0, max_bend],
        # matching what _update_hand_tracking_data would produce live.
        _MOCK_CYCLE_FRAMES = 240
        _MOCK_MAX_BEND = 1.3   # ~75 deg, safely within Sharpa flex limits
        def _mock_5x3(s: int) -> th.Tensor:
            t = (s % _MOCK_CYCLE_FRAMES) / _MOCK_CYCLE_FRAMES   # 0..1
            # triangle wave 0->1->0 over the cycle
            tri = 1.0 - abs(2.0 * t - 1.0)
            # smoothstep for nicer visual
            sm = tri * tri * (3.0 - 2.0 * tri)
            return th.full((5, 3), sm * _MOCK_MAX_BEND, dtype=th.float32)

        for step in range(args.steps):

            # === TEST: HAND OPEN/CLOSE -- step begin ===
            if _mock_or_replay:
                # Pull the 5x3 bend tensor from the mock generator or the
                # replay file.  This is the same shape that
                # _update_hand_tracking_data writes to teleop_action.hand_data
                # in the live VR path, so the rest of the pipeline is shared.
                if args.replay_hand:
                    _frame = _replay_frames[step % len(_replay_frames)]
                    _bend_5x3 = _frame["bend_5x3"]
                    if not th.is_tensor(_bend_5x3):
                        _bend_5x3 = th.tensor(_bend_5x3, dtype=th.float32)
                else:
                    _bend_5x3 = _mock_5x3(step)

                # Build a TeleopAction the same way OVXRSystem would.
                _ta = _TA()
                _ta[args.hand] = _hand_pose_action          # arm IK pose, trigger=0
                _ta.is_valid = {args.hand: True, "head": True}
                _ta.hand_data = {args.hand: _bend_5x3}

                # robot.teleop_data_to_action() detects hand_data and routes
                # through the retargeting + smoothing path (see robot.py
                # is_manipulation branch).
                _action = robot.teleop_data_to_action(_ta)
                env.step(_action)

                if step % 30 == 0 and _hand_finger_dof_idx is not None:
                    _actual = robot.get_joint_positions()[_hand_finger_dof_idx]
                    _cmd = _action[_hand_gripper_idx]
                    # Index_MCP_FE is at Sharpa idx 5; index_PIP at 7.
                    _hlog(f"  [HAND step {step:5d}]"
                          f"  bend[idx,MCP]={_bend_5x3[1,0]:+.2f}"
                          f"  bend[mid,PIP]={_bend_5x3[2,1]:+.2f}"
                          f"  cmd[idx_MCP_FE]={_cmd[5]:+.2f}"
                          f"  cmd[mid_PIP]={_cmd[11]:+.2f}"
                          f"  actual[idx_MCP_FE]={_actual[5]:+.2f}"
                          f"  actual[mid_PIP]={_actual[11]:+.2f}")
            elif _test_hand:
                _test_action[:6] = _arm_hold_cmd

                _phase = (step // _CYCLE_FRAMES) % 2
                _frame_in_phase = step % _CYCLE_FRAMES
                _t = min(_frame_in_phase / max(_RAMP_FRAMES, 1), 1.0)

                if _phase == 0:
                    _hand_target = _hand_open + _t * (_hand_close - _hand_open)
                else:
                    _hand_target = _hand_close + _t * (_hand_open - _hand_close)

                _test_action[6:28] = _hand_target
                env.step(_test_action)

                if step % _CYCLE_FRAMES == 0 or step % 50 == 0:
                    _actual = robot.get_joint_positions()[_finger_dof_idx]
                    _log(f"  [TEST step {step:5d}] {'CLOSE' if _phase==0 else 'OPEN'} t={_t:.2f}  "
                         f"cmd[:4]={[round(x,2) for x in _hand_target[:4].tolist()]}  "
                         f"actual[:4]={[round(x,3) for x in _actual[:4].tolist()]}")
            else:
                og.sim.step()
            # === TEST: HAND OPEN/CLOSE -- step end ===

            if all_trackers:
                update_all_trackers(robot, all_trackers)
            elif wrist_track is not None:
                update_tracking_camera(robot, *wrist_track)
            capture_frame()
            if step % 300 == 0:
                eef_p, _ = robot.eef_links[robot.arm_names[0]].get_position_orientation()
                bp, _ = robot.get_position_orientation()
                eef_r = eef_p - bp
                print(f"[step {step:5d}]  eef={[round(x,3) for x in eef_r.tolist()]}")
        if video_writer:
            for packet in video_writer[1].encode():
                video_writer[0].mux(packet)
            video_writer[0].close()
            print(f"[Video] Saved to {os.path.abspath(args.save_video)}")
        print("Done.")
        og.shutdown()
        return

    # Hand tracking requires the OpenXR backend so it can query XR_EXT_hand_tracking.
    # SteamVR registers as the system OpenXR runtime, so "OpenXR" still connects to it.
    # Isaac Sim 5.1 only ships the OpenXR backend (no omni.kit.xr.system.steamvr).
    # OpenXR routes to SteamVR (or whatever runtime ~/.config/openxr/1/active_runtime.json points at).
    xr_system = "OpenXR"

    if args.mode == "hand":
        log(f"[Hand] XR backend={xr_system}, hand tracking components requested pre-environment")

    vrsys = OVXRSystem(
        robot=robot,
        show_control_marker=not args.no_vr_display,
        system=xr_system,
        eef_tracking_mode=args.mode,
        align_anchor_to="base",
        disable_display_output=args.no_vr_display,
    )

    vrsys.set_anchor_with_prim(og.sim.viewer_camera)
    cam_pos, _ = og.sim.viewer_camera.get_position_orientation()
    log(f"VR anchor → viewer_camera at {cam_pos.tolist()} (ego view)")
    print(f"[VR] Anchor set to viewer_camera (pos={[round(x,2) for x in cam_pos.tolist()]})")

    vrsys.start()

    # During VR startup, XR internally calls clear_controller_model(), which
    # removes the physics view from the robot's articulation views. After that,
    # robot.get_joint_positions() returns None. Rebuild the physics handles
    # before the teleop loop reads joint state.
    og.sim.update_handles()
    log("[VR] Rebuilt physics handles after vrsys.start() (XR invalidated articulation view)")

    log(f"\n=== VR STARTED ===")
    log(f"robot_arms={vrsys.robot_arms}")
    log(f"align_anchor_to={vrsys.align_anchor_to}")
    log(f"eef_tracking_mode={vrsys.eef_tracking_mode}")

    # Diagnostic: check controller input availability
    log(f"\n=== CONTROLLER INPUT DIAGNOSTICS ===")
    log(f"controllers_detected={list(vrsys.controllers.keys())}")
    for cname, cdev in vrsys.controllers.items():
        try:
            inp_names = list(cdev.get_input_names())
            out_names = list(cdev.get_output_names())
            log(f"  {cname}: inputs={inp_names[:5]}{'...' if len(inp_names)>5 else ''} ({len(inp_names)} total)")
            log(f"  {cname}: outputs={out_names[:5]}{'...' if len(out_names)>5 else ''} ({len(out_names)} total)")
        except Exception as e:
            log(f"  {cname}: ERROR getting input/output names: {e}")
    print(f"[VR] Controller diagnostics logged to debug file")

    grasp_state = {"closed": False, "t": 0.0}
    # Slower ramp (~1.5s @120Hz) gives the compliant PD time to settle.
    _GRASP_RAMP_SECONDS = 1.5
    _GRASP_RAMP_FRAMES = max(1, int(_GRASP_RAMP_SECONDS * 120))
    def _smoothstep(t):
        return t * t * (3.0 - 2.0 * t)

    from omnigibson.robots.franka_mounted_sharpa import SHARPA_JOINT_LIMITS
    # Open = straight (zeros). Close = 85% of upper URDF limit. AA zeroed in
    # both. DIPs follow open. Pinky skipped (chain runaway).
    _hand_open = th.zeros(22)
    _hand_close = SHARPA_JOINT_LIMITS[:, 1].clone() * 0.85
    _aa_indices = [1, 3, 6, 10, 14, 19]
    _dip_indices = [4, 8, 12, 16, 21]
    for idx in _aa_indices:
        _hand_close[idx] = 0.0
        _hand_open[idx] = 0.0
    for idx in _dip_indices:
        _hand_close[idx] = _hand_open[idx]
    for _pinky_idx in [17, 18, 19, 20, 21]:
        _hand_close[_pinky_idx] = 0.0
        _hand_open[_pinky_idx] = 0.0

    _arm_name = robot.arm_names[0]
    _gripper_idx = robot.gripper_action_idx[_arm_name]

    _gripper_ctrl = robot.controllers.get(f"gripper_{_arm_name}") or robot.controllers.get("gripper_0")
    if _gripper_ctrl is not None:
        _PER_JOINT_EFFORT = _build_per_joint_effort(20.0, robot.dof_names_ordered)
        _finger_dof_idx = ControllerView.get_dof_idx(_gripper_ctrl[0])  # robot.controllers[name] = (group_key, controller_idx)
        _effort_limits = robot.control_limits["effort"]
        for dof in _finger_dof_idx:
            jname = robot.dof_names_ordered[dof]
            mult = _PER_JOINT_EFFORT[jname]
            _effort_limits[0][dof] *= mult
            _effort_limits[1][dof] *= mult
            robot.joints[jname].max_effort = robot.joints[jname].max_effort * mult
        print(f"[Grasp] Per-joint effort applied (default 20×, pinky 1×)")
        log(f"[Grasp] Per-joint effort applied (default 20×, pinky 1×)")

    log(f"[Grasp] open[:5]={[round(x,3) for x in _hand_open[:5].tolist()]}")
    log(f"[Grasp] close[:5]={[round(x,3) for x in _hand_close[:5].tolist()]}")

    # ALVR/Quest A button bounces — single physical press fires multiple
    # click edges within ~50ms. Debounce so one press = one toggle.
    # (Only relevant in --mode controller; hand-tracking mode never invokes this.)
    _last_toggle_t = [0.0]
    _DEBOUNCE_S = 0.25

    def on_grasp_toggle():
        now = time.monotonic()
        if now - _last_toggle_t[0] < _DEBOUNCE_S:
            log(f"[BUTTON A] debounced (Δt={now - _last_toggle_t[0]:.3f}s)")
            return
        _last_toggle_t[0] = now
        grasp_state["closed"] = not grasp_state["closed"]
        grasp_state["t"] = 0.0
        state_str = "CLOSED" if grasp_state["closed"] else "OPEN"
        print(f"  >>> [Button A] Grasp → {state_str} (ramp {_GRASP_RAMP_FRAMES} frames)")
        log(f"[BUTTON A] grasp_toggle → {state_str}")

    ctrl_name = args.hand
    KeyboardEventHandler.KEYBOARD_CALLBACKS[f"{ctrl_name}_a"] = on_grasp_toggle
    log(f"Registered button callback: {ctrl_name}_a → grasp toggle")

    # --- Button B: cycle wrist rotation offset ---
    # Analytically computed offset: inv(Q_c_handshake) * Q_d_handshake
    #   Q_c avg from steps 150-360: [0.6920, -0.6588, -0.2133, 0.2037]
    #   Q_d (fingers→+X, palm→+Y, thumb→+Z): [0.5, 0.5, 0.5, 0.5]
    _ANALYTIC_OFFSET = th.tensor([-0.0214, 0.8839, -0.4669, 0.0118])
    _ANALYTIC_OFFSET = _ANALYTIC_OFFSET / _ANALYTIC_OFFSET.norm()

    _rot_presets = [
        ("analytic",         _ANALYTIC_OFFSET.clone()),
        ("Ry-90",            T.euler2quat(th.tensor([0.0, -math.pi / 2, 0.0]))),
        ("identity",         th.tensor([0.0, 0.0, 0.0, 1.0])),
        ("Rx+90",            T.euler2quat(th.tensor([math.pi / 2, 0.0, 0.0]))),
        ("Rx-90",            T.euler2quat(th.tensor([-math.pi / 2, 0.0, 0.0]))),
        ("Ry+90",            T.euler2quat(th.tensor([0.0, math.pi / 2, 0.0]))),
        ("Rz+90",            T.euler2quat(th.tensor([0.0, 0.0, math.pi / 2]))),
        ("Rz-90",            T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2]))),
        ("orig(Rx180)",      robot.teleop_rotation_offset[arm_name].clone()),
        ("Rx180_Ry90",       T.euler2quat(th.tensor([math.pi, math.pi / 2, 0.0]))),
        ("Rx180_Ry-90",      T.euler2quat(th.tensor([math.pi, -math.pi / 2, 0.0]))),
        ("Rx180_Rz90",       T.euler2quat(th.tensor([math.pi, 0.0, math.pi / 2]))),
        ("Rx180_Rz-90",      T.euler2quat(th.tensor([math.pi, 0.0, -math.pi / 2]))),
    ]
    _rot_idx = {"v": 0}

    robot._teleop_rotation_offset = _rot_presets[0][1]
    # Desired EEF orientation at startup: fingers +X, palm +Y, thumb +Z (handshake)
    _DESIRED_START_QUAT = th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32)
    robot._teleop_desired_start_quat = _DESIRED_START_QUAT
    _init_euler = [round(math.degrees(x), 1) for x in T.quat2euler(_rot_presets[0][1]).tolist()]
    log(f"[ROT_OFFSET] initial preset: {_rot_presets[0][0]} quat={[round(x,4) for x in _rot_presets[0][1].tolist()]} euler={_init_euler}")
    log(f"[ROT_OFFSET] desired_start_quat={_DESIRED_START_QUAT.tolist()}")
    print(f"[Teleop] Wrist rotation preset: {_rot_presets[0][0]} euler={_init_euler}°")

    def on_rot_cycle():
        _rot_idx["v"] = (_rot_idx["v"] + 1) % len(_rot_presets)
        name, quat = _rot_presets[_rot_idx["v"]]
        robot._teleop_rotation_offset = quat
        euler_deg = [round(math.degrees(x), 1) for x in T.quat2euler(quat).tolist()]
        print(f"  >>> [Button B] Wrist preset #{_rot_idx['v']}: {name}  euler={euler_deg}°")
        log(f"[BUTTON B] rot_preset idx={_rot_idx['v']} name={name} euler_deg={euler_deg}")

    KeyboardEventHandler.KEYBOARD_CALLBACKS[f"{ctrl_name}_b"] = on_rot_cycle
    log(f"Registered button callback: {ctrl_name}_b → cycle wrist rotation ({len(_rot_presets)} presets)")

    print(f"\n{'='*60}")
    print(f"  VR Teleop: {args.hand} hand, mode={args.mode}, view={args.view}")
    print(f"  Debug log: {LOG_FILE}")
    print(f"  Button A ({args.hand} controller) = toggle grasp")
    print(f"  Button B ({args.hand} controller) = cycle wrist rotation ({len(_rot_presets)} presets)")
    print(f"{'='*60}")
    print(f"  Hold your {args.hand} hand/controller steady. Calibration on first frame.\n")

    button_names_logged = False

    for step in range(args.steps):
        vrsys.update()
        action = vrsys.get_robot_teleop_action()

        if not button_names_logged and "button_data" in vrsys.raw_data:
            bd = vrsys.raw_data["button_data"]
            log(f"\n=== BUTTON DATA DUMP (step {step}) ===")
            if not bd:
                log(f"  WARNING: button_data is EMPTY (no controller inputs detected)")
                log(f"  controllers keys: {list(vrsys.controllers.keys())}")
                for cn, cd in vrsys.controllers.items():
                    try:
                        log(f"  {cn} input_names: {list(cd.get_input_names())}")
                    except Exception as e:
                        log(f"  {cn} get_input_names() failed: {e}")
            for ctrl, inputs in bd.items():
                log(f"  controller={ctrl}")
                for inp_name, gestures in inputs.items():
                    log(f"    input={inp_name}  gestures={gestures}")
            print(f"[Buttons] Available inputs logged to debug file (empty={not bd})")
            button_names_logged = True

        if step % 30 == 0:
            ta = vrsys.teleop_action
            valid = {k: bool(v) for k, v in ta.is_valid.items()} if hasattr(ta, "is_valid") else {}
            clamped = getattr(robot, "_teleop_prev_target", None)
            calibrated = getattr(robot, "_teleop_calibrated", False)

            eef_p, _ = robot.eef_links[robot.arm_names[0]].get_position_orientation()
            bp, _ = robot.get_position_orientation()
            eef_r = eef_p - bp
            jpos = robot.get_joint_positions()

            vr_raw = ta[args.hand][:6] if hasattr(ta, args.hand) else th.zeros(6)

            # Log HMD info if available
            hmd_info = ""
            try:
                hmd_phys = vrsys.xr2og(vrsys.hmd.get_pose())
                hmd_virt = vrsys.raw_data["transforms"]["head"]
                hmd_info = (f"  hmd_phys_pos={[round(x,3) for x in hmd_phys[0].tolist()]}"
                           f"  hmd_virt_pos={[round(x,3) for x in hmd_virt[0].tolist()]}")
            except Exception:
                pass

            # Log ALL button values every 30 frames to detect if presses register
            btn_info = ""
            if "button_data" in vrsys.raw_data:
                for ctrl_name, inputs in vrsys.raw_data["button_data"].items():
                    nonzero = {inp: {g: v for g, v in gestures.items() if v != 0.0}
                               for inp, gestures in inputs.items()}
                    nonzero = {k: v for k, v in nonzero.items() if v}
                    if nonzero:
                        btn_info += f"  buttons[{ctrl_name}]={nonzero}"
                        print(f"  [step {step}] BUTTON ACTIVE: {ctrl_name} → {nonzero}")

            log(f"\n[step {step:5d}] valid={valid} cal={calibrated}")
            log(f"  vr_raw_pos={[round(x,3) for x in vr_raw[:3].tolist()]}")
            log(f"  vr_raw_orn={[round(x,3) for x in vr_raw[3:6].tolist()]}")
            corr_orn = getattr(vrsys, "_corrected_orn", {}).get(args.hand, None)
            if corr_orn is not None:
                corr_list = [round(x, 4) for x in corr_orn.tolist()]
                log(f"  corrected_orn(xyzw)={corr_list}")
                final_orn = T.quat_multiply(corr_orn, robot.teleop_rotation_offset[robot.arm_names[0]])
                final_list = [round(x, 4) for x in final_orn.tolist()]
                log(f"  final_orn(xyzw)    ={final_list}")
                if step % 150 == 0:
                    print(f"  >>> corrected_orn(xyzw)={corr_list}  final_orn(xyzw)={final_list}")
            if clamped is not None:
                log(f"  clamped   ={[round(x,3) for x in clamped.tolist()]}")
            log(f"  eef_base  ={[round(x,3) for x in eef_r.tolist()]}")
            log(f"  arm_j     ={[round(x,3) for x in jpos[:7].tolist()]}")
            log(hmd_info)
            if btn_info:
                log(btn_info)
            elif step % 150 == 0:
                log(f"  buttons: all zero")

        if step % 60 == 0:
            clamped = getattr(robot, "_teleop_prev_target", None)
            eef_p, _ = robot.eef_links[robot.arm_names[0]].get_position_orientation()
            bp, _ = robot.get_position_orientation()
            eef_r = eef_p - bp
            ct = [round(x, 3) for x in clamped.tolist()] if clamped is not None else "N/A"
            er = [round(x, 3) for x in eef_r.tolist()]
            print(f"[step {step:5d}]  clamped={ct}  eef={er}")

        grasp_state["t"] = min(grasp_state["t"] + 1.0 / _GRASP_RAMP_FRAMES, 1.0)
        s = _smoothstep(grasp_state["t"])
        if grasp_state["closed"]:
            action[_gripper_idx] = _hand_open + s * (_hand_close - _hand_open)
        else:
            action[_gripper_idx] = _hand_close + s * (_hand_open - _hand_close)
        if step % 30 == 0 and grasp_state["t"] < 1.0:
            tgt = action[_gripper_idx]
            log(f"[Grasp ramp] t={grasp_state['t']:.2f} s={s:.2f} tgt[:5]={[round(x,3) for x in tgt[:5].tolist()]}")

        if step % 120 == 0:
            jpos = robot.get_joint_positions()
            finger_pos = jpos[7:29]
            fp = [round(x, 3) for x in finger_pos.tolist()]
            tgt = action[_gripper_idx]
            ft = [round(x, 3) for x in tgt.tolist()]
            state_str = "CLOSED" if grasp_state["closed"] else "OPEN"
            log(f"[Finger pos] state={state_str} t={grasp_state['t']:.2f}")
            log(f"  actual={fp}")
            log(f"  target={ft}")
            labels = [
                "th_CMC_FE","th_CMC_AA","th_MCP_FE","th_MCP_AA","th_IP",
                "ix_MCP_FE","ix_MCP_AA","ix_PIP","ix_DIP",
                "md_MCP_FE","md_MCP_AA","md_PIP","md_DIP",
                "rg_MCP_FE","rg_MCP_AA","rg_PIP","rg_DIP",
                "pk_CMC","pk_MCP_FE","pk_MCP_AA","pk_PIP","pk_DIP",
            ]
            for i, lbl in enumerate(labels):
                err = abs(fp[i] - ft[i])
                if err > 0.1:
                    log(f"  ** {lbl}: actual={fp[i]:.3f} target={ft[i]:.3f} err={err:.3f}")

        env.step(action)

        if all_trackers:
            update_all_trackers(robot, all_trackers)
        elif wrist_track is not None:
            update_tracking_camera(robot, *wrist_track)

    print("Done.")
    vrsys.stop()
    logf.close()
    og.shutdown()


if __name__ == "__main__":
    main()
