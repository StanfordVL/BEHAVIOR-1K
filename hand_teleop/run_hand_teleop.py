"""Sharpa hand-tracking teleop in OmniGibson + ARAT scene, via WebXR.

This is the analogue of `vr_sharpa_hand_teleop.py` but using the WebXR-based
finger retargeting pipeline (no OVXRSystem, no SteamVR/ALVR, no
`omni.kit.xr.*`). The Quest's native browser opens this server's webpage,
streams hand-tracking joints over a WebSocket, and a geometric retargeter
produces clean Sharpa joint angles every tick.

Run:
    python hand_teleop/run_hand_teleop.py --hand right --arat

On the Quest, open Chrome / Meta Browser at:
    http://<host-ip>:8012
Tap "Enter VR", make sure both hands are tracked.

Press SPACE in the OmniGibson viewer (or set --auto-anchor) to capture the
rest pose; from then on retargeting is anchored to your current pose.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
import yaml

# Make `hand_teleop` importable as a package when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent))   # so `hand_teleop` is a top-level package

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.controllers.controller_view import ControllerView

# Make sure build_arat_scene's gm overrides land FIRST, then ours win.
import build_arat_scene  # noqa: F401

gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_FLATCACHE = True
# We don't need ENABLE_VR — no OVXRSystem path. WebXR runs purely in the
# Quest's browser and ships JSON over WebSocket; OmniGibson never opens
# an XR session itself.

from build_arat_scene import OBJECTS, build_object_cfgs

from hand_teleop.og_bridge import (
    BridgeConfig, OGBridge, SHARPA_GRIPPER_INDEX, build_joint_limits_from_robot,
)
from hand_teleop.retargeter import RetargeterConfig
from hand_teleop.safety import SafetyConfig
from hand_teleop.vr_input import VRInputServer
from hand_teleop.views import setup_all_views, update_all_trackers

LOG_FILE = "/home/eeg/Desktop/BEHAVIOR-1K/hand_teleop_debug.log"
_LOG = logging.getLogger("hand_teleop")


# --- Reused from vr_sharpa_teleop.py: Franka home + 22 zero finger joints. ---
_RESET_POSE = th.tensor(
    [-6e-4, -1.30, 6e-4, -2.87, 1e-3, 1.999, 0.749]
    + [0.0] * 22
)


# --- Per-joint effort multipliers (validated baseline from STATE_VR_TELEOP.md). ---
_PINKY_CMC_BOOST      =   1.0
_PINKY_MCP_FE_BOOST   =   1.0
_PINKY_MCP_AA_BOOST   =   1.0
_PINKY_PIP_BOOST      =   1.0
_INDEX_MCP_FE_BOOST   =  25.0
_MIDDLE_MCP_FE_BOOST  =  25.0
_RING_MCP_FE_BOOST    =  30.0


def _build_per_joint_effort(default_mult: float, dof_names_ordered):
    overrides = {
        "right_pinky_CMC":     _PINKY_CMC_BOOST,
        "right_pinky_MCP_FE":  _PINKY_MCP_FE_BOOST,
        "right_pinky_MCP_AA":  _PINKY_MCP_AA_BOOST,
        "right_pinky_PIP":     _PINKY_PIP_BOOST,
        "right_index_MCP_FE":  _INDEX_MCP_FE_BOOST,
        "right_middle_MCP_FE": _MIDDLE_MCP_FE_BOOST,
        "right_ring_MCP_FE":   _RING_MCP_FE_BOOST,
        "left_pinky_CMC":     _PINKY_CMC_BOOST,
        "left_pinky_MCP_FE":  _PINKY_MCP_FE_BOOST,
        "left_pinky_MCP_AA":  _PINKY_MCP_AA_BOOST,
        "left_pinky_PIP":     _PINKY_PIP_BOOST,
        "left_index_MCP_FE":  _INDEX_MCP_FE_BOOST,
        "left_middle_MCP_FE": _MIDDLE_MCP_FE_BOOST,
        "left_ring_MCP_FE":   _RING_MCP_FE_BOOST,
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


def _build_env_config(hand_side: str, scene_model: str, scene_file=None, arat=False):
    if arat:
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


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _build_retargeter_cfg(yaml_cfg: dict) -> RetargeterConfig:
    r = (yaml_cfg.get("retargeter") or {})
    return RetargeterConfig(
        finger_flex_gain=float(r.get("finger_flex_gain", 1.0)),
        thumb_flex_gain=float(r.get("thumb_flex_gain", 1.5)),
        thumb_opp_gain=float(r.get("thumb_opp_gain", 2.0)),
        thumb_aa_gain=float(r.get("thumb_aa_gain", 1.5)),
        pinky_cmc_gain=float(r.get("pinky_cmc_gain", 0.5)),
        abduction_gain=float(r.get("abduction_gain", 1.0)),
        finger_abduction_sign_left=float(r.get("finger_abduction_sign_left", 1.0)),
        finger_abduction_sign_right=float(r.get("finger_abduction_sign_right", -1.0)),
        thumb_cmc_fe_sign_left=float(r.get("thumb_cmc_fe_sign_left", -1.0)),
        thumb_cmc_fe_sign_right=float(r.get("thumb_cmc_fe_sign_right", 1.0)),
        thumb_cmc_aa_sign_left=float(r.get("thumb_cmc_aa_sign_left", 1.0)),
        thumb_cmc_aa_sign_right=float(r.get("thumb_cmc_aa_sign_right", -1.0)),
    )


def _build_safety_cfg(yaml_cfg: dict) -> SafetyConfig:
    s = (yaml_cfg.get("safety") or {})
    return SafetyConfig(
        enabled=bool(s.get("enabled", True)),
        smoothing_alpha=float(s.get("smoothing_alpha", 0.85)),
        max_arm_velocity=float(s.get("max_arm_velocity", 2.0)),
        max_hand_velocity=float(s.get("max_hand_velocity", 6.0)),
        max_arm_delta_per_tick=float(s.get("max_arm_delta_per_tick", 0.10)),
        max_hand_delta_per_tick=float(s.get("max_hand_delta_per_tick", 0.20)),
        log_clips=bool(s.get("log_clips", True)),
    )


def _build_bridge_cfg(yaml_cfg: dict, side: str) -> BridgeConfig:
    b = (yaml_cfg.get("bridge") or {})
    ws = (b.get("workspace_box") or {})
    return BridgeConfig(
        side=side,
        workspace_min=tuple(ws.get("min", (-0.6, -0.3, 0.05))),
        workspace_max=tuple(ws.get("max", (0.6, 0.9, 1.20))),
        position_sensitivity=float(b.get("position_sensitivity", 1.0)),
        vr_to_world_quat_xyzw=tuple(
            float(x) for x in (b.get("vr_to_world_quat_xyzw")
                              or (0.7071067811865476, 0.0, 0.0, 0.7071067811865476))
        ),
        orientation_mode=str(b.get("orientation_mode", "absolute")),
        orn_align_quat_xyzw=tuple(
            float(x) for x in (b.get("orn_align_quat_xyzw")
                              or (1.0, 0.0, 0.0, 0.0))
        ),
        skip_pinky=bool(b.get("skip_pinky", True)),
        close_target_clamp=float(b.get("close_target_clamp", 0.85)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--scene-file", type=str, default=None)
    parser.add_argument("--arat", action="store_true",
                        help="Use the ARAT scene (table + Franka + light), same as vr_sharpa_*teleop --arat")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Maximum simulation steps; default ~ infinite")
    parser.add_argument("--ws-port", type=int, default=8012,
                        help="WebSocket / HTTP port for the WebXR client")
    parser.add_argument("--auto-anchor", action="store_true",
                        help="Capture anchor pose automatically on the first valid frame")
    parser.add_argument("--view", choices=["all", "ego", "none"], default="ego",
                        help="`all` = ego + wrist_back + wrist_palm (3 render products, heaviest); "
                             "`ego` = just the over-the-shoulder ego cam (1 extra render, recommended); "
                             "`none` = no extra cameras (lightest, only the default OmniGibson viewer)")
    parser.add_argument("--config", type=Path, default=_THIS_DIR / "config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    debug_log = open(LOG_FILE, "w")
    def log(msg):
        debug_log.write(msg + "\n")
        debug_log.flush()

    yaml_cfg = _load_yaml(args.config)

    # Build env.
    cfg = _build_env_config(args.hand, args.scene, scene_file=args.scene_file, arat=args.arat)
    env = og.Environment(configs=cfg)
    env.reset()
    robot = env.robots[0]
    for arm_n in robot.arm_names:
        robot.links[robot.eef_link_names[arm_n]].visible = True

    # Camera: third-person side view.
    base_pos, base_orn = robot.get_position_orientation()
    shoulder_off = th.tensor([2.5, 0.0, 1.2], dtype=th.float32)
    cam_pos = base_pos + T.quat_apply(base_orn, shoulder_off)
    cam_orn = T.quat_multiply(base_orn, th.tensor([0.5, 0.5, 0.5, 0.5], dtype=th.float32))
    og.sim.viewer_camera.set_position_orientation(position=cam_pos, orientation=cam_orn)
    try:
        og.sim.enable_viewer_camera_teleoperation()
    except Exception:
        pass

    # Multi-view setup. `ego` is the recommended default — adds a single
    # over-the-shoulder VisionSensor (1 extra render product, ~25% extra
    # GPU). `all` adds two more wrist-tracking cams (~3× extra GPU and
    # noticeable lag). `none` skips entirely.
    all_trackers = None
    if args.view in ("all", "ego"):
        headless = bool(gm.HEADLESS)
        include_wrist = (args.view == "all")
        ego_sensor, all_trackers = setup_all_views(
            robot, args.hand, headless=headless, include_wrist=include_wrist,
        )
        log(f"multi_view={args.view}  ego={ego_sensor.prim_path}  "
            f"wrist_cams={['wrist_back', 'wrist_palm'] if include_wrist else []}")

    # Per-joint effort tuning (matches our validated baseline).
    arm_name = robot.arm_names[0]
    gripper_handle = robot.controllers.get(f"gripper_{arm_name}") or robot.controllers.get("gripper_0")
    finger_dof_idx = ControllerView.get_dof_idx(gripper_handle[0]).tolist()
    per_joint_effort = _build_per_joint_effort(20.0, robot.dof_names_ordered)
    effort_lims = robot.control_limits["effort"]
    for dof in finger_dof_idx:
        jname = robot.dof_names_ordered[dof]
        mult = per_joint_effort[jname]
        effort_lims[0][dof] *= mult
        effort_lims[1][dof] *= mult
        robot.joints[jname].max_effort = robot.joints[jname].max_effort * mult
    log(f"[Effort] applied per-joint multipliers (default 20×, pinky 1×)")

    # Joint limits dict for the retargeter.
    joint_limits = build_joint_limits_from_robot(robot, finger_dof_idx, side_prefix=args.hand)

    # WebXR server. WebXR requires HTTPS on non-localhost origins (Meta
    # Browser/Wolvic enforcement); auto-pick up TLS cert + key if generated
    # in this dir as `cert.pem` / `key.pem`.
    web_dir = _THIS_DIR / "web"
    cert_file = _THIS_DIR / "cert.pem"
    key_file = _THIS_DIR / "key.pem"
    use_tls = cert_file.exists() and key_file.exists()
    vr_input = VRInputServer(
        host="0.0.0.0", port=args.ws_port, web_dir=web_dir,
        cert_file=cert_file if use_tls else None,
        key_file=key_file if use_tls else None,
    )
    vr_input.start()
    scheme = "https" if use_tls else "http"
    print(f"[hand_teleop] WebXR server: {scheme}://<host>:{args.ws_port}")
    if use_tls:
        print(f"[hand_teleop] TLS enabled (cert.pem + key.pem). On the Quest, "
              f"the browser will warn about a self-signed cert — tap 'Advanced' "
              f"→ 'Proceed anyway'.")

    # Bridge.
    bridge = OGBridge(
        env=env,
        robot=robot,
        gripper_finger_dof_idx=finger_dof_idx,
        bridge_cfg=_build_bridge_cfg(yaml_cfg, args.hand),
        retarg_cfg=_build_retargeter_cfg(yaml_cfg),
        safety_cfg=_build_safety_cfg(yaml_cfg),
        joint_limits=joint_limits,
        side_prefix=args.hand,
    )

    # Anchor on space; or auto-anchor on first valid VR frame if requested.
    from omnigibson.utils.ui_utils import KeyboardEventHandler
    def _on_anchor():
        bridge.request_anchor()
        print("[hand_teleop] anchor requested — next valid frame becomes rest pose")
    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.SPACE, callback_fn=_on_anchor,
    )
    if args.auto_anchor:
        bridge.request_anchor()
        print("[hand_teleop] auto-anchor armed; first valid VR frame will calibrate.")

    log(f"=== INIT ===")
    log(f"hand={args.hand}  arat={args.arat}  ws_port={args.ws_port}")
    log(f"finger_dof_idx={finger_dof_idx}")
    log(f"joint_limits={joint_limits}")

    print(f"\n{'='*60}")
    print(f"  hand_teleop running. Quest browser: http://<host>:{args.ws_port}")
    print(f"  Press SPACE in viewer to set anchor (or pass --auto-anchor)")
    print(f"  Ctrl+C to exit.")
    print(f"{'='*60}\n")

    try:
        for step in range(args.steps):
            payload = vr_input.get_latest()
            action = bridge.step(payload)
            env.step(action)
            # Wrist-mounted cameras follow the hand each frame.
            if all_trackers:
                update_all_trackers(robot, all_trackers)

            if step % 60 == 0:
                eef_p, _ = robot.eef_links[arm_name].get_position_orientation()
                bp, _ = robot.get_position_orientation()
                eef_r = (eef_p - bp).tolist()
                jpos = robot.get_joint_positions().tolist()
                fp = jpos[7:29]
                pinky_q = fp[17:22]
                log(f"[step {step:6d}] eef_rel={[round(x, 3) for x in eef_r]}  "
                    f"pinky={[round(x, 3) for x in pinky_q]}  "
                    f"calib={bridge._calibrated}  "
                    f"vr_active={payload.get('left') is not None or payload.get('right') is not None}")
    except KeyboardInterrupt:
        print("\n[hand_teleop] Ctrl+C — shutting down")
    finally:
        try:
            vr_input.stop()
        except Exception:
            pass
        debug_log.close()
        try:
            og.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
