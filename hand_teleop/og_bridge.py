"""WebXR → OmniGibson bridge.

Connects the WebXR hand-tracking pipeline (vr_input + retargeter) to
OmniGibson's Franka + SharpaWave robot. Each control tick:

  1. Pull the latest WebXR pose payload from `VRInputServer.get_latest()`.
  2. Decode the active side's keypoints.
  3. Compute arm-wrist pose in robot base frame (with workspace clamp +
     vr_to_robot rotation), feed it to OmniGibson's
     `InverseKinematicsController` (absolute_pose) for the arm.
  4. Run `Retargeter` on the same keypoints to get 22 SharpaWave joint
     angles (suffix → angle), pack into the gripper action vector.
  5. Run the result through a `SafetyFilter` (EMA + velocity / Δq caps).
  6. Return the (action_dim,) action tensor to feed `env.step(action)`.

This module is intentionally engine-light: it imports torch + numpy + the
two ported helpers (`retargeter.py`, `safety.py`). No `OVXRSystem`, no
`omni.kit.xr.*` — every Isaac 5.1 XR breakage is bypassed by going
through WebXR.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch as th

import omnigibson.utils.transform_utils as T

from .retargeter import Retargeter, RetargeterConfig
from .safety import SafetyConfig, SafetyFilter
from .vr_input import VRInputServer, hand_payload_to_keypoints


_LOG = logging.getLogger("hand_teleop.bridge")


# Sharpa joint suffix (as emitted by the retargeter) → index in the
# 22-element gripper command vector. The order matches what
# `ControllerView.get_dof_idx(gripper_handle[0])` returns for the
# `franka_mounted_sharpa_right` robot in OmniGibson; see vr_sharpa_teleop's
# debug log under `[Finger pos] labels = [...]`.
SHARPA_GRIPPER_INDEX = {
    "thumb_CMC_FE":  0,
    "thumb_CMC_AA":  1,
    "thumb_MCP_FE":  2,
    "thumb_MCP_AA":  3,
    "thumb_IP":      4,
    "index_MCP_FE":  5,
    "index_MCP_AA":  6,
    "index_PIP":     7,
    "index_DIP":     8,
    "middle_MCP_FE": 9,
    "middle_MCP_AA": 10,
    "middle_PIP":    11,
    "middle_DIP":    12,
    "ring_MCP_FE":   13,
    "ring_MCP_AA":   14,
    "ring_PIP":      15,
    "ring_DIP":      16,
    "pinky_CMC":     17,
    "pinky_MCP_FE":  18,
    "pinky_MCP_AA":  19,
    "pinky_PIP":     20,
    "pinky_DIP":     21,
}


@dataclass
class BridgeConfig:
    """Tunable knobs for the WebXR → OmniGibson bridge."""
    # Which physical hand the user is teleoperating with.
    side: str = "right"                  # "left" | "right"

    # Arm IK target shaping.
    position_sensitivity: float = 1.0
    # --- Frame mapping (mirrors vr_sharpa_teleop's _YAW_FIX) ----------------
    # WebXR local-floor is Y-up, -Z forward (the direction you faced entering
    # VR), +X right. We rotate that into OmniGibson world (Z-up, +X arm-forward,
    # +Y left) so your physical motion maps the same way as the controller path:
    #   WebXR forward (-Z) -> robot +X (forward),  WebXR up (+Y) -> +Z (up),
    #   WebXR right (+X)   -> robot -Y (right).
    # This (0.5,-0.5,-0.5,0.5) = Rz(-90deg) o Rx(90deg); the old (0.7071,0,0,0.7071)
    # had no yaw fix, which is why forward/back and left/right felt swapped.
    vr_to_world_quat_xyzw: tuple[float, float, float, float] = (
        0.5, -0.5, -0.5, 0.5,
    )

    # --- Orientation: first-frame calibration (mirrors vr_sharpa) -----------
    # At the first valid frame we anchor the EEF orientation to
    # `desired_start_quat_xyzw` (the SharpaWave "handshake" rest pose, same value
    # as vr_sharpa_teleop's _DESIRED_START_QUAT), then track the wrist's deltas
    # 1:1. No jump at t=0; "hand rotates left -> robot rotates left".
    desired_start_quat_xyzw: tuple[float, float, float, float] = (
        0.5, 0.5, 0.5, 0.5,
    )
    # Body-frame tweak applied to the mapped wrist before calibration (the
    # analog of the YAML teleop_rotation_offset / vr_sharpa rotation presets).
    # Identity by default — the calibration handles the rest-pose alignment, so
    # this only nudges the tracking frame. Try 90/180deg presets if a hand
    # rotation drives the wrist the wrong way.
    orn_align_quat_xyzw: tuple[float, float, float, float] = (
        0.0, 0.0, 0.0, 1.0,   # identity
    )

    # --- Reachable workspace (shoulder-sphere; mirrors vr_sharpa) -----------
    # Base frame, shoulder at Z=1.194m. Clamp targets to a reach sphere + floor/
    # ceiling + per-step rate limit so IK never chases unreachable poses.
    shoulder_pos: tuple[float, float, float] = (0.0, 0.0, 1.194)
    shoulder_reach: float = 0.80
    z_min: float = 0.82
    z_max: float = 1.75
    max_pos_step: float = 0.02

    # Pinky skip — same as vr_sharpa_teleop: the chain runaway in PhysX
    # makes pinky unstable, so leave it at zero regardless of retargeter
    # output. Set False to send retargeted pinky angles.
    skip_pinky: bool = True

    # Soft limit on the close target. SharpaWave joints overshoot if you
    # command them at the URDF upper limit; clamp targets to this fraction
    # before they go to PhysX. 0.85 matches our validated baseline.
    close_target_clamp: float = 0.85


def _quat_apply_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (xyzw, normalized)."""
    qx, qy, qz, qw = q_xyzw
    # v' = q * v * q^-1, expanded to vector form.
    t = 2.0 * np.array([
        qy * v[2] - qz * v[1],
        qz * v[0] - qx * v[2],
        qx * v[1] - qy * v[0],
    ])
    return v + qw * t + np.cross([qx, qy, qz], t)


def _quat_mul_wxyz(a, b):
    """Hamilton product of two wxyz quaternions (numpy)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def _xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_inv_wxyz(q):
    """Inverse (conjugate, assuming unit) of a wxyz quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _clamp_to_workspace_sphere(pos, cfg, prev_pos=None):
    """Clamp a base-frame target to the Franka reach (mirrors vr_sharpa's
    _clamp_to_workspace): floor/ceiling, a shoulder-centered reach sphere, and a
    per-step rate limit so the Jacobian IK stays stable. numpy in/out."""
    shoulder = np.asarray(cfg.shoulder_pos, dtype=float)
    clamped = np.asarray(pos, dtype=float).copy()
    clamped[2] = float(np.clip(clamped[2], cfg.z_min, cfg.z_max))
    to_shoulder = clamped - shoulder
    dist = float(np.linalg.norm(to_shoulder))
    if dist > cfg.shoulder_reach:
        clamped = shoulder + to_shoulder * (cfg.shoulder_reach / dist)
    if prev_pos is not None:
        delta = clamped - prev_pos
        dn = float(np.linalg.norm(delta))
        if dn > cfg.max_pos_step:
            clamped = prev_pos + delta * (cfg.max_pos_step / dn)
    return clamped


class OGBridge:
    """Owns the per-tick WebXR-to-OmniGibson translation."""

    def __init__(
        self,
        env,
        robot,
        gripper_finger_dof_idx: list[int],
        bridge_cfg: BridgeConfig,
        retarg_cfg: RetargeterConfig,
        safety_cfg: SafetyConfig,
        joint_limits: dict[str, tuple[float, float]],
        side_prefix: str = "right",
        scaled_close_targets: Optional[th.Tensor] = None,
    ):
        self.env = env
        self.robot = robot
        self.cfg = bridge_cfg
        self.retargeter = Retargeter(cfg=retarg_cfg, joint_limits=joint_limits)
        self.side_prefix = side_prefix  # robot URDF side prefix (right/left)
        self.gripper_finger_dof_idx = list(gripper_finger_dof_idx)
        self.scaled_close_targets = scaled_close_targets  # optional cap hint

        # Safety filter dimensions: 7 arm joints + 22 gripper joints (Franka + Sharpa).
        # We feed it (arm_qpos, hand_qpos) snapshots — even though our action is
        # IK pose + gripper position, we filter on the *commanded joint targets*.
        # For the arm we filter on the IK target's pose components separately so
        # the filter behaves coherently in one space.
        self.safety = SafetyFilter(
            n_arm=6, n_hand=22, dt=safety_cfg_dt(safety_cfg),
            cfg=safety_cfg,
        )

        # Calibration state.
        self._anchor_armed = True   # set by run loop on space-press / startup
        self._calibrated = False
        self._teleop_pos_offset: Optional[np.ndarray] = None  # eef base-frame ref - first vr_pos
        # Orientation calibration (mirrors vr_sharpa compute_calibrated_arm_action):
        # at the first valid frame, _teleop_orn_offset = desired_start ⊗ inv(mapped_wrist),
        # then target_orn = _teleop_orn_offset ⊗ mapped_wrist. → at t=0 the EEF is at
        # desired_start_quat (handshake); afterward it tracks the wrist's delta.
        self._teleop_orn_offset_wxyz: Optional[np.ndarray] = None
        self._prev_target_pos: Optional[np.ndarray] = None   # for the rate limiter
        self._home_eef_pose_robot: Optional[tuple[np.ndarray, np.ndarray]] = None

        # Arm name and gripper action index.
        self.arm_name = robot.arm_names[0]
        self.gripper_action_idx = robot.gripper_action_idx[self.arm_name]
        self.action_dim = int(robot.action_dim)

    # ------------------------------------------------------------------
    def request_anchor(self) -> None:
        """Mark the next valid frame as the anchor (rest pose)."""
        self.retargeter.set_anchor()
        self._anchor_armed = True
        self._calibrated = False
        self._teleop_pos_offset = None
        self._teleop_orn_offset_wxyz = None
        self._prev_target_pos = None

    # ------------------------------------------------------------------
    def step(self, vr_payload: dict, fallback_close: float = 0.0) -> th.Tensor:
        """Build a single (action_dim,) command tensor from a VR payload.

        If the VR payload is missing / not yet anchored, returns a zero
        action that holds the arm in place via IK and keeps the gripper open.
        `fallback_close` (0..1) lets the caller drive a manual close ramp
        when there is no VR data (e.g. while waiting on the WebXR client).
        """
        # Default: zero gripper command, arm IK target = current EEF pose
        # (i.e. holds in place).
        action = th.zeros(self.action_dim, dtype=th.float32)
        eef_pos_world, eef_quat_world = self.robot.eef_links[self.arm_name].get_position_orientation()
        base_pos, base_quat = self.robot.get_position_orientation()
        eef_rel_pos, eef_rel_quat = T.relative_pose_transform(
            eef_pos_world, eef_quat_world, base_pos, base_quat,
        )
        # IK action = (rel_pos[3], axisangle[3]) for absolute_pose mode.
        action[:3] = eef_rel_pos
        action[3:6] = T.quat2axisangle(eef_rel_quat)

        # Pull the active-side payload.
        side_payload = (vr_payload or {}).get(self.cfg.side)
        wrist_pose, kps = hand_payload_to_keypoints(side_payload)

        if kps is not None and "wrist" in kps:
            # ---- Arm: wrist pose → robot base frame ----
            wrist_xyz_vr = np.asarray(kps["wrist"]["position"], dtype=float)
            wrist_quat_wxyz_vr = np.asarray(kps["wrist"]["quaternion_wxyz"], dtype=float)
            # vr_to_world_quat_xyzw rotates a vector from WebXR space (Y-up,
            # -Z forward) into OmniGibson world (Z-up). We use it for both
            # the position vector and (composed) the wrist quaternion.
            q_xyzw = np.asarray(self.cfg.vr_to_world_quat_xyzw, dtype=float)
            wrist_xyz_world = _quat_apply_xyzw(q_xyzw, wrist_xyz_vr)
            wrist_xyz_world *= self.cfg.position_sensitivity

            wrist_xyz_robot = wrist_xyz_world - base_pos.numpy()

            # Compose orientations: wrist_q_world = vr_to_world * wrist_q_vr,
            # then express in robot base by left-multiplying base_q^-1.
            q_vr_to_world_wxyz = _xyzw_to_wxyz(q_xyzw)
            wrist_q_world_wxyz = _quat_mul_wxyz(q_vr_to_world_wxyz, wrist_quat_wxyz_vr)
            base_q_wxyz = _xyzw_to_wxyz(base_quat.numpy())
            base_q_inv_wxyz = np.array([base_q_wxyz[0], -base_q_wxyz[1], -base_q_wxyz[2], -base_q_wxyz[3]])
            wrist_q_robot_wxyz = _quat_mul_wxyz(base_q_inv_wxyz, wrist_q_world_wxyz)
            # OmniGibson uses xyzw everywhere except some specific APIs; the
            # `quat_multiply` / `quat2axisangle` helpers expect xyzw.
            wrist_q_robot_xyzw = _wxyz_to_xyzw(wrist_q_robot_wxyz)

            # First-valid-frame anchoring: snapshot the EEF rest pose so the
            # commanded target == actual EEF at t=0 (no jump). Mirrors the
            # algorithm now in teleop_utils.OVXRSystem.update().
            if self._teleop_pos_offset is None:
                self._teleop_pos_offset = eef_rel_pos.numpy() - wrist_xyz_robot
                self._home_eef_pose_robot = (
                    eef_rel_pos.numpy().copy(),
                    eef_rel_quat.numpy().copy(),
                )

            target_pos = wrist_xyz_robot + self._teleop_pos_offset
            # Workspace clamp: shoulder-reach sphere + floor/ceiling + per-step
            # rate limit, identical to vr_sharpa_teleop's _clamp_to_workspace.
            target_pos = _clamp_to_workspace_sphere(
                target_pos, self.cfg, prev_pos=self._prev_target_pos
            )
            self._prev_target_pos = target_pos.copy()

            # Orientation: first-frame calibration anchored to desired_start_quat,
            # then track the wrist's delta — same algorithm as vr_sharpa's
            # compute_calibrated_arm_action. orn_align is a body-frame tweak
            # baked into the mapped wrist (the teleop_rotation_offset analog).
            orn_align_wxyz = _xyzw_to_wxyz(np.asarray(self.cfg.orn_align_quat_xyzw, dtype=float))
            mapped_wrist_wxyz = _quat_mul_wxyz(wrist_q_robot_wxyz, orn_align_wxyz)
            if self._teleop_orn_offset_wxyz is None:
                desired_wxyz = _xyzw_to_wxyz(np.asarray(self.cfg.desired_start_quat_xyzw, dtype=float))
                self._teleop_orn_offset_wxyz = _quat_mul_wxyz(
                    desired_wxyz, _quat_inv_wxyz(mapped_wrist_wxyz)
                )
            final_q_wxyz = _quat_mul_wxyz(self._teleop_orn_offset_wxyz, mapped_wrist_wxyz)
            final_q_xyzw = _wxyz_to_xyzw(final_q_wxyz)

            action[:3] = th.tensor(target_pos, dtype=th.float32)
            action[3:6] = T.quat2axisangle(th.tensor(final_q_xyzw, dtype=th.float32))

            # ---- Hand: retargeter → 22 finger angles ----
            angles = self.retargeter.compute(kps, side=self.cfg.side)
            if angles:
                self._calibrated = True
                hand_cmd = th.zeros(22, dtype=th.float32)
                for suffix, angle in angles.items():
                    idx = SHARPA_GRIPPER_INDEX.get(suffix)
                    if idx is None:
                        continue
                    if self.cfg.skip_pinky and idx >= 17:
                        continue
                    val = float(angle) * self.cfg.close_target_clamp
                    hand_cmd[idx] = val
                action[6:6 + 22] = hand_cmd

        # ---- Safety filter: smooth + clamp ----
        # We filter on (xyz target | gripper command), not the arm joint
        # qpos (which is what mink-stack filtered on). This is fine: it
        # bounds per-tick change in the commanded action space, which is
        # ultimately what the IK + gripper see.
        arm_action_np = action[:6].cpu().numpy()
        hand_action_np = action[6:6 + 22].cpu().numpy()
        full_np = np.concatenate([arm_action_np, hand_action_np])
        full_np = self.safety.filter(full_np, log=_LOG)
        action[:6] = th.tensor(full_np[:6], dtype=th.float32)
        action[6:6 + 22] = th.tensor(full_np[6:], dtype=th.float32)
        return action


def safety_cfg_dt(_safety_cfg: SafetyConfig) -> float:
    """Resolve the dt for SafetyFilter — keep consistent with control_hz."""
    # SafetyFilter uses dt for the velocity cap. Default control_hz=60 -> dt≈0.0167.
    # The runner is the source of truth; we mirror that here in case someone
    # constructs OGBridge directly.
    return 1.0 / 60.0


def build_joint_limits_from_robot(robot, gripper_finger_dof_idx: list[int],
                                  side_prefix: str = "right") -> dict[str, tuple[float, float]]:
    """Read SharpaWave joint limits from the OmniGibson robot.

    Returns a dict keyed by retargeter suffix (e.g. "index_MCP_FE") so the
    retargeter can clip its outputs to the URDF range — which exactly matches
    what `BaseController.clip_control()` does downstream anyway.
    """
    limits: dict[str, tuple[float, float]] = {}
    full_to_suffix = {f"{side_prefix}_{s}": s for s in SHARPA_GRIPPER_INDEX.keys()}
    for dof in gripper_finger_dof_idx:
        jname = robot.dof_names_ordered[dof]
        suffix = full_to_suffix.get(jname)
        if suffix is None:
            continue
        j = robot.joints[jname]
        limits[suffix] = (float(j.lower_limit), float(j.upper_limit))
    return limits
