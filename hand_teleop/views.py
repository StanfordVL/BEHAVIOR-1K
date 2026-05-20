"""Multi-view (ego + wrist back + wrist palm) setup for hand_teleop.

Mirrors the `--view all` setup in vr_sharpa_teleop.py: an ego camera looking
over the robot's shoulder + two wrist-mounted tracking cameras (palm side,
dorsal side) that follow the SharpaWave hand each frame.

Each camera is a `VisionSensor`; on a non-headless run, each gets its own
floating viewport window. The ego camera is positioned once and held; wrist
cameras must be updated every sim step via `update_all_trackers`.
"""

from __future__ import annotations

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.sensors.vision_sensor import VisionSensor


# All offsets / orientations are in the hand_C_MC link's local frame.
# Hand frame: +Z = finger direction, +X = palm/dorsal axis.
WRIST_VIEWS = {
    "wrist_back": {
        "link_suffix": "hand_C_MC",
        "offset": (-0.09, 0.0, -0.05),
        "orn": (0.7010, -0.7010, -0.0923, 0.0923),
        "desc": "Mirror of palm: -X (dorsal) side, looking +Z along fingers.",
    },
    "wrist_palm": {
        "link_suffix": "hand_C_MC",
        "offset": (0.09, 0.0, -0.05),
        "orn": (0.7010, 0.7010, 0.0923, 0.0923),
        "desc": "At +X, looking +Z along fingers. Screen up = +X.",
    },
}


# Static / world-frame views used for the ego camera (and useful third-person
# alternatives if a script wants to reuse this module).
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


def compute_tracking_pose(robot, link_name, local_offset, local_orn):
    """Compute world-frame position + orientation for a camera tracking a robot link."""
    link = robot.links[link_name]
    link_pos, link_quat = link.get_position_orientation()
    world_offset = T.quat_apply(link_quat, local_offset)
    return link_pos + world_offset, T.quat_multiply(link_quat, local_orn)


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


def setup_all_views(robot, hand_side, headless=False, include_wrist=True):
    """
    Set up ego + (optionally) wrist_back + wrist_palm cameras.

    Args:
        include_wrist: if False, skip the two wrist tracking cameras and
            return an empty trackers list. The single ego camera is much
            cheaper than the full 3-camera setup (~3× less GPU per tick).

    Returns:
      ego_sensor (VisionSensor): static, positioned once.
      trackers (list[tuple]): [(sensor, link_name, local_offset, local_orn), ...]
        — call `update_all_trackers(robot, trackers)` each sim step.
        Empty when `include_wrist=False`.
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
    if ego_sensor._viewport is not None:
        ego_sensor._viewport.visible = False

    _VP_LAYOUT_X = 0
    _VP_LAYOUT_Y = 30
    _VP_EGO_W, _VP_EGO_H = 640, 480
    _VP_WRIST_W, _VP_WRIST_H = 400, 300

    # Isaac 5.1 / Kit 107.3: position_x/position_y/camera_path must be passed
    # AS KWARGS to create_viewport_window — assigning them as attributes on
    # the returned window is silently a no-op in 5.1.
    if not headless:
        try:
            lazy.omni.kit.viewport.utility.create_viewport_window(
                "Ego View",
                width=_VP_EGO_W, height=_VP_EGO_H,
                position_x=_VP_LAYOUT_X, position_y=_VP_LAYOUT_Y,
                camera_path=ego_sensor.prim_path,
            )
            print(f"[Multi-view] Viewport 'Ego View' → {ego_sensor.prim_path}  "
                  f"@ ({_VP_LAYOUT_X},{_VP_LAYOUT_Y}) {_VP_EGO_W}x{_VP_EGO_H}")
        except Exception as e:
            print(f"[Multi-view] Could not create ego viewport: {e}")
    print(f"[Multi-view] Ego sensor at {[round(x, 2) for x in ego_pos.tolist()]}  ({v['desc']})")

    trackers = []
    if not include_wrist:
        return ego_sensor, trackers

    _wrist_x = _VP_LAYOUT_X + _VP_EGO_W + 10
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
                lazy.omni.kit.viewport.utility.create_viewport_window(
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
    """Update every wrist-tracking camera each frame. ego camera is static."""
    for sensor, link_name, offset, orn in trackers:
        pos, quat = compute_tracking_pose(robot, link_name, offset, orn)
        sensor.set_position_orientation(position=pos, orientation=quat)
