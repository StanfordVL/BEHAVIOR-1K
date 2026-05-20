# hand_teleop — WebXR hand-tracking teleop for SharpaWave in OmniGibson

Self-contained alternative to `vr_sharpa_hand_teleop.py` that bypasses
the entire `OVXRSystem` / OpenXR / SteamVR / ALVR chain. The Quest's
native browser opens this server's webpage, ships hand-tracking joints
over a WebSocket, and a geometric retargeter produces SharpaWave joint
angles every tick.

Why a separate path: Isaac Sim 5.1 removed `XRCoreEventType.hand_joints`,
breaking `OVXRSystem`'s hand-tracking subscription. Until OmniGibson
upstream ports to the new polling API, WebXR is the most reliable way
to get clean hand-joint data into the simulator.

## Files

```
hand_teleop/
├── README.md              — this file
├── config.yaml            — knobs (rotation preset, gains, safety limits)
├── run_hand_teleop.py     — entry point (analogous to vr_sharpa_hand_teleop.py)
├── og_bridge.py           — glues WebXR keypoints → OmniGibson env
├── retargeter.py          — geometric WebXR → SharpaWave joint angles
├── safety.py              — EMA + velocity + Δq-per-tick filter
├── vr_input.py            — WebSocket server (aiohttp)
└── web/index.html         — WebXR client served on the Quest browser
```

## Run

1. Activate the conda env that has OmniGibson + Isaac Sim:
   ```bash
   /home/eeg/anaconda3/bin/conda activate behavior
   ```

2. Make sure `aiohttp` is installed (the only extra dep beyond OmniGibson):
   ```bash
   pip install aiohttp pyyaml
   ```

3. Start the teleop:
   ```bash
   python hand_teleop/run_hand_teleop.py --hand right --arat --auto-anchor
   ```

   The OmniGibson window opens with the ARAT scene + Franka + SharpaWave;
   a WebSocket+HTTP server starts on port 8012.

4. On the Quest, open the device's native browser (Meta Browser or
   Wolvic) and navigate to:
   ```
   http://<host-machine-ip>:8012
   ```
   Tap **Enter VR** and grant hand-tracking permission.

5. Hold your physical hand in your "rest" pose, then either:
   - Press SPACE in the OmniGibson viewer to capture the anchor, or
   - Pass `--auto-anchor` to the script so the first valid frame is used.

   Subsequent finger curls are computed *relative to* this rest pose.

6. Move your physical hand → the EEF tracks the wrist; curl your fingers
   → the SharpaWave hand mirrors. The pinky stays at home (PhysX chain
   runaway); thumb + 4 fingers do the gripping.

## Calibration model — analogous to teleop_utils.py

The bridge implements the same calibration the restored
`OVXRSystem.update()` does, adapted for the WebXR data path:

| Concept | OVXRSystem (controller mode) | WebXR bridge |
|---|---|---|
| Fixed XR-to-world rotation | `_xr_to_world_rot` (captured once from HMD) | `vr_to_world_quat_xyzw` from `config.yaml` |
| Per-frame robot-frame transform | `xr_to_robot_rot = inv(base) · _xr_to_world_rot` | same: `inv(base) · vr_to_world` |
| Head-rotation invariance | subtract `phys_ctrl - phys_head` | WebXR `local-floor` reference space is anchored to physical room (head-rotation-decoupled by construction) |
| First-valid-frame EEF anchor | `_teleop_pos_offset[hand] = eef_rel_pos − corrected_pos` | `_teleop_pos_offset = eef_rel_pos − wrist_xyz_robot` |
| Per-frame EEF target | `corrected_pos + _teleop_pos_offset` | `wrist_xyz_robot + _teleop_pos_offset` |
| Wrist orientation | `quat2axisangle(corrected_orn · teleop_rotation_offset)` | same — applied on `wrist_q_robot` |
| Workspace clamp | not in OVXRSystem | `np.clip(target_pos, ws.min, ws.max)` (added here) |

Result: at startup the EEF is at its current home pose. Hand motion is
anchored relative to that. Head turns don't drag the arm.

## What didn't change

- ARAT scene loaded via `build_arat_scene.OBJECTS` + `build_object_cfgs(...)`.
- Robot config: same Franka + SharpaWave with `reset_joint_pos`, IK arm
  controller in `absolute_pose` mode, `MultiFingerGripperController` in
  `independent` position mode.
- Per-joint effort tuning: validated baseline from `STATE_VR_TELEOP.md`
  (default 20×, pinky 1×, index/middle/ring MCP_FE 25–30×).
- Pinky-skip in command (chain runaway prevention).
- 0.85 close target clamp (PhysX limit margin).

## Troubleshooting

- **No video on the Quest**: make sure the Quest is on the same Wi-Fi as
  the host PC. Try `ping <host-ip>` from a phone on the same network.
- **WebXR unsupported**: Meta Browser supports WebXR + hand-tracking by
  default on Quest 2/3. Wolvic (sideloadable) also works.
- **EEF jumps on first frame**: normal — the bridge anchors on the first
  valid frame. If you see persistent drift, press SPACE to re-anchor with
  your hand in the desired rest pose.
- **Forward/back or left/right swapped**: change `vr_to_world_quat_xyzw`
  in `config.yaml` to one of the four presets shown in the comment.
- **Fingers spread when you close**: flip `finger_abduction_sign_right`
  to its opposite (1.0 ↔ -1.0) in `config.yaml`.

## What still needs work

- **Wrist orientation calibration**: currently uses a fixed
  `vr_to_world_quat` from config. A future version could auto-calibrate
  the rotation by snapshotting the wrist quaternion at anchor time.
- **Pinky**: skipped from the grip (chain runaway in PhysX). If
  upstream effort/limit tuning improves, set `skip_pinky: false`.
- **Safety filter on IK action**: filters in axis-angle space, which is
  fine for small per-tick deltas but not strictly correct for large ones.
  For ARAT-scale motion (slow, deliberate), this is good enough.
