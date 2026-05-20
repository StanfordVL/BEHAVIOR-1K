"""Geometric WebXR-to-Sharpa hand retargeter.

This module implements direct kinematic retargeting from WebXR Hand Input
keypoints to Sharpa joint angles, replacing the older approach where finger
mocap bodies were dragged through MuJoCo connect constraints. The geometric
approach is dramatically more natural because:

  * Flexion angles (MCP, PIP, DIP, IP) are taken directly from the angles
    between consecutive WebXR bone segments — physically meaningful and 1:1
    with the user's actual finger curl.
  * Abduction is computed in a stable per-frame palm frame built from
    metacarpal keypoints, so it's robust to hand orientation.
  * Thumb opposition (CMC) is driven by the thumb metacarpal's deviation in
    the palm plane, with a gain to compensate for kinematic mismatch between
    a human thumb and the Sharpa thumb.

The approach mirrors `adoshi25/robot_teleop`'s Tesollo retargeter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------- WebXR keypoint chains ----------------
# Order: wrist-side -> tip. The flexion angles are between consecutive bones
# (a "bone" is the segment between two adjacent keypoints).
_FINGER_CHAINS: dict[str, tuple[str, ...]] = {
    "index": (
        "index-finger-metacarpal",
        "index-finger-phalanx-proximal",
        "index-finger-phalanx-intermediate",
        "index-finger-phalanx-distal",
        "index-finger-tip",
    ),
    "middle": (
        "middle-finger-metacarpal",
        "middle-finger-phalanx-proximal",
        "middle-finger-phalanx-intermediate",
        "middle-finger-phalanx-distal",
        "middle-finger-tip",
    ),
    "ring": (
        "ring-finger-metacarpal",
        "ring-finger-phalanx-proximal",
        "ring-finger-phalanx-intermediate",
        "ring-finger-phalanx-distal",
        "ring-finger-tip",
    ),
    "pinky": (
        "pinky-finger-metacarpal",
        "pinky-finger-phalanx-proximal",
        "pinky-finger-phalanx-intermediate",
        "pinky-finger-phalanx-distal",
        "pinky-finger-tip",
    ),
    "thumb": (
        "thumb-metacarpal",
        "thumb-phalanx-proximal",
        "thumb-phalanx-distal",
        "thumb-tip",
    ),
}


@dataclass
class RetargeterConfig:
    """Tunable knobs for retargeting. All angles in radians."""

    # 1:1 multiplier on each non-thumb finger's flexion (MCP/PIP/DIP).
    # >1 = amplify curl. The Sharpa joint limits clip the result anyway.
    finger_flex_gain: float = 1.0
    # Sharpa thumb has very limited human-mismatched range; amplify so a
    # natural human thumb-tuck reaches a usable Sharpa fist.
    thumb_flex_gain: float = 1.5
    # Out-of-palm-plane folding (CMC_FE).  Larger gain = thumb folds toward
    # palm-front more aggressively for the same WebXR motion.
    thumb_opp_gain: float = 2.0
    # In-palm-plane lateral motion (CMC_AA).  Larger gain = thumb tucks
    # across the palm toward the fingers more aggressively.
    thumb_aa_gain: float = 1.5
    # Pinky CMC tracks a fraction of pinky's medial pull (palm cupping).
    pinky_cmc_gain: float = 0.5
    # Multiplier on per-finger MCP_AA abduction (lateral spread). Set to a
    # negative value to invert direction if the hand spreads when you close
    # your fingers (or vice versa) — Sharpa's MCP_AA axes can flip depending
    # on the side, see RetargeterConfig.finger_abduction_sign.
    abduction_gain: float = 1.0
    # Per-side sign for the four-finger MCP_AA. Sharpa's left-hand MCP_AA
    # joints are oriented opposite to the natural palm-frame lateral-angle
    # delta, so spreading the user's hand was producing a "fingers together"
    # pose. +1 for left maps spread->spread; flip to -1 if the right-hand
    # configuration ever inverts again.
    finger_abduction_sign_left: float = 1.0
    finger_abduction_sign_right: float = -1.0
    # Per-side signs for the thumb CMC joints.  Defaults are derived from
    # an empirical perturbation test of the Sharpa MJCF (forward kinematics
    # of each joint -> thumb-tip displacement in palm-local coords):
    #   * +CMC_FE moves the tip toward palm-front (negative palm-Z), so we
    #     want negative `elev` delta to produce positive CMC_FE -> sign = -1.
    #   * +CMC_AA moves the tip toward the fingers (-X, +Y), so we want
    #     positive `ip_palm` delta to produce positive CMC_AA -> sign = +1.
    # Right hand mirrors.  Flip if your specific Sharpa side feels inverted.
    thumb_cmc_fe_sign_left: float = -1.0
    thumb_cmc_fe_sign_right: float = 1.0
    thumb_cmc_aa_sign_left: float = 1.0
    thumb_cmc_aa_sign_right: float = -1.0


@dataclass
class Retargeter:
    """Maps WebXR hand keypoints to Sharpa joint angles via geometric retargeting.

    Stateful: capture-on-anchor flow.
        retargeter.set_anchor()              -> next valid frame becomes the
                                                rest pose for abduction.
        retargeter.compute(keypoints, side)  -> returns {joint_name: angle}.

    Anchored quantities (only abductions, thumb CMC, pinky CMC) are recorded
    on the first call after `set_anchor()`. Flexion is absolute: 0° at
    straight, increases with curl, clipped to joint limits.
    """

    cfg: RetargeterConfig = field(default_factory=RetargeterConfig)
    # Joint limits keyed by short joint suffix (e.g. "index_MCP_FE").
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Internal anchor state.
    _anchor_pending: bool = True
    _anchor_set: bool = False
    _anchor_abd: dict[str, float] = field(default_factory=dict)
    _anchor_thumb_ip_palm: Optional[float] = None
    _anchor_thumb_elev: Optional[float] = None
    _anchor_pinky_lat: Optional[float] = None
    # Quest hand tracking reports ~10°+ residual thumb flex even when the
    # user's thumb is fully straight; anchor the thumb flex angles too so
    # "rest pose at anchor" -> 0° at the joint.
    _anchor_thumb_mcp_flex: Optional[float] = None
    _anchor_thumb_ip_flex: Optional[float] = None

    # ------------------------------------------------------------------
    def set_anchor(self) -> None:
        """Mark anchor pending — next valid call to `compute` snapshots it."""
        self._anchor_pending = True
        self._anchor_set = False
        self._anchor_abd.clear()
        self._anchor_thumb_ip_palm = None
        self._anchor_thumb_elev = None
        self._anchor_pinky_lat = None
        self._anchor_thumb_mcp_flex = None
        self._anchor_thumb_ip_flex = None

    # ------------------------------------------------------------------
    def compute(self, keypoints: dict | None, side: str) -> dict[str, float]:
        """Return a dict mapping joint suffix (e.g. "index_PIP") to angle.

        Joints not in the result should be left at their home value by the
        caller (we only emit joints we actively retarget).
        """
        if not keypoints:
            return {}

        # Build palm frame; bail if we don't have enough metacarpals visible.
        frame = self._build_palm_frame(keypoints)
        if frame is None:
            return {}
        fwd, lat, nrm = frame

        # Sign convention is split because the Sharpa joint axes don't agree
        # for fingers vs thumb. `side_sign` (Tesollo-borrowed) drives the
        # thumb CMC; `abd_sign` is what the per-finger MCP_AA actually wants.
        side_sign = -1.0 if side == "left" else 1.0
        abd_sign = (self.cfg.finger_abduction_sign_left if side == "left"
                    else self.cfg.finger_abduction_sign_right)
        out: dict[str, float] = {}

        # ---- Per-finger flexion (4-finger fingers) ----
        for finger in ("index", "middle", "ring", "pinky"):
            flex = self._flex_angles(keypoints, _FINGER_CHAINS[finger])
            if flex is None:
                continue
            mcp_fe, pip, dip = flex
            self._set(out, f"{finger}_MCP_FE",
                      mcp_fe * self.cfg.finger_flex_gain)
            self._set(out, f"{finger}_PIP",
                      pip * self.cfg.finger_flex_gain)
            self._set(out, f"{finger}_DIP",
                      dip * self.cfg.finger_flex_gain)

        # ---- Thumb flexion ----
        thumb_flex = self._flex_angles(keypoints, _FINGER_CHAINS["thumb"])
        if thumb_flex is not None:
            mcp_fe_raw, ip_raw = thumb_flex
            if self._anchor_pending:
                # Capture the user's rest-pose thumb flex; subtracting this
                # later cancels the Quest hand-tracker's residual thumb bend.
                self._anchor_thumb_mcp_flex = mcp_fe_raw
                self._anchor_thumb_ip_flex = ip_raw
            mcp_anchor = self._anchor_thumb_mcp_flex or 0.0
            ip_anchor = self._anchor_thumb_ip_flex or 0.0
            mcp_fe = (mcp_fe_raw - mcp_anchor) * self.cfg.thumb_flex_gain
            ip = (ip_raw - ip_anchor) * self.cfg.thumb_flex_gain
            self._set(out, "thumb_MCP_FE", mcp_fe)
            self._set(out, "thumb_IP", ip)

        # ---- Per-finger abduction (anchored) ----
        for finger in ("index", "middle", "ring", "pinky"):
            ang = self._finger_lateral_angle(keypoints, finger, fwd, lat, nrm)
            if ang is None:
                continue
            if self._anchor_pending:
                self._anchor_abd[finger] = ang
                continue
            rest = self._anchor_abd.get(finger, ang)
            abd = abd_sign * (ang - rest) * self.cfg.abduction_gain
            self._set(out, f"{finger}_MCP_AA", abd)

        # ---- Thumb CMC / MCP_AA from palm-relative metacarpal direction ----
        thumb_extras = self._compute_thumb_palm_angles(
            keypoints, fwd, lat, nrm, side,
        )
        if thumb_extras is not None:
            cmc_fe, cmc_aa = thumb_extras
            if cmc_fe is not None:
                self._set(out, "thumb_CMC_FE", cmc_fe)
            if cmc_aa is not None:
                self._set(out, "thumb_CMC_AA", cmc_aa)
            # MCP_AA is small / mostly zero. Leave at home (omit from out).

        # ---- Pinky CMC (palm cupping) ----
        pinky_cmc = self._compute_pinky_cmc(keypoints, fwd, lat, nrm)
        if pinky_cmc is not None:
            self._set(out, "pinky_CMC", pinky_cmc)

        # First call after set_anchor finalizes the anchor.
        if self._anchor_pending and (self._anchor_abd or
                                     self._anchor_thumb_ip_palm is not None):
            self._anchor_pending = False
            self._anchor_set = True

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set(self, out: dict[str, float], suffix: str, val: float) -> None:
        lo, hi = self.joint_limits.get(suffix, (-np.inf, np.inf))
        out[suffix] = float(np.clip(val, lo, hi))

    @staticmethod
    def _kp_pos(kps: dict, name: str) -> Optional[np.ndarray]:
        kp = kps.get(name)
        if kp is None:
            return None
        p = kp.get("position")
        if p is None:
            return None
        return np.asarray(p, dtype=float)

    def _build_palm_frame(self, kps: dict):
        """Right-handed (forward, lateral, normal) frame anchored on the palm.

        forward = wrist -> middle metacarpal (along the palm into the fingers)
        lateral = pinky metacarpal - index metacarpal (across the palm)
                  (re-orthogonalized vs forward)
        normal  = forward x lateral  (out of the palm; sign depends on hand)
        """
        wrist = self._kp_pos(kps, "wrist")
        mid = self._kp_pos(kps, "middle-finger-metacarpal")
        idx = self._kp_pos(kps, "index-finger-metacarpal")
        pky = self._kp_pos(kps, "pinky-finger-metacarpal")
        if any(p is None for p in (wrist, mid, idx, pky)):
            return None

        fwd = mid - wrist
        n = np.linalg.norm(fwd)
        if n < 1e-6:
            return None
        fwd /= n

        lat = pky - idx
        lat = lat - np.dot(lat, fwd) * fwd
        n = np.linalg.norm(lat)
        if n < 1e-6:
            return None
        lat /= n

        nrm = np.cross(fwd, lat)
        return fwd, lat, nrm

    def _flex_angles(self, kps: dict, chain: tuple[str, ...]):
        """Return the flexion angles between consecutive bones along a chain.

        For a chain of N keypoints there are (N-1) bones and (N-2) flex
        angles. Each angle is in [0, π] — 0 = straight, π = fully folded.
        Returns None if any keypoint is missing.
        """
        pts = [self._kp_pos(kps, n) for n in chain]
        if any(p is None for p in pts):
            return None
        bones = [pts[i + 1] - pts[i] for i in range(len(pts) - 1)]
        out = []
        for i in range(len(bones) - 1):
            a = bones[i]
            b = bones[i + 1]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                out.append(0.0)
                continue
            cos_t = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
            out.append(float(np.arccos(cos_t)))
        return out

    def _finger_lateral_angle(
        self, kps: dict, finger: str,
        fwd: np.ndarray, lat: np.ndarray, nrm: np.ndarray,
    ) -> Optional[float]:
        """In-palm-plane lateral angle of the proximal phalanx, in radians."""
        prox = self._kp_pos(kps, f"{finger}-finger-phalanx-proximal")
        inter = self._kp_pos(kps, f"{finger}-finger-phalanx-intermediate")
        if prox is None or inter is None:
            return None
        bone = inter - prox
        bone = bone - np.dot(bone, nrm) * nrm
        n = np.linalg.norm(bone)
        if n < 1e-6:
            return None
        bone /= n
        return float(np.arctan2(np.dot(bone, lat), np.dot(bone, fwd)))

    def _compute_thumb_palm_angles(
        self, kps: dict,
        fwd: np.ndarray, lat: np.ndarray, nrm: np.ndarray,
        side: str,
    ):
        """Drive Sharpa thumb CMC_FE / CMC_AA from the palm-relative thumb
        metacarpal direction.

        Empirical Sharpa joint axes (perturb each joint, watch the thumb tip
        in palm-local coords):

          * CMC_FE: tip moves +Z (palm-back) when negative, -Z (palm-front)
            when positive.  This is "thumb folds DOWN toward palm-front" —
            i.e., OUT-OF-PALM-PLANE motion.  Range -10..+110°.
          * CMC_AA: tip moves toward the fingers (-X, +Y) when positive,
            away from fingers when negative.  This is IN-PALM-PLANE motion.
            Range -20..+20°.

        So we drive CMC_FE from the metacarpal's out-of-plane (elev) delta,
        and CMC_AA from its in-plane (lateral) delta.  Earlier versions had
        these swapped, which is why no amount of sign fiddling produced
        natural-feeling thumb motion.
        """
        meta = self._kp_pos(kps, "thumb-metacarpal")
        prox = self._kp_pos(kps, "thumb-phalanx-proximal")
        if meta is None or prox is None:
            return None
        bone = prox - meta
        n = np.linalg.norm(bone)
        if n < 1e-6:
            return None
        bone_n = bone / n

        # Out-of-plane angle: thumb's signed elevation above the palm plane.
        # When the user folds the thumb toward palm-front, nrm goes negative
        # so elev goes negative — and we want CMC_FE to swing positive
        # (Sharpa's "thumb tucked toward palm-front" direction), hence the
        # default sign of -1 below.
        elev = float(np.arcsin(np.clip(np.dot(bone_n, nrm), -1.0, 1.0)))

        # In-plane angle: where the metacarpal points within the palm plane.
        # When the user tucks the thumb across the palm toward the fingers,
        # the metacarpal rotates toward `lat` so this angle increases — and
        # we want CMC_AA positive (Sharpa's "thumb toward fingers" direction).
        bone_proj = bone - np.dot(bone, nrm) * nrm
        pn = np.linalg.norm(bone_proj)
        if pn < 1e-6:
            return None
        bone_proj_n = bone_proj / pn
        ip_palm = float(np.arctan2(
            np.dot(bone_proj_n, lat), np.dot(bone_proj_n, fwd),
        ))

        if self._anchor_pending:
            self._anchor_thumb_ip_palm = ip_palm
            self._anchor_thumb_elev = elev
            return None, None

        rest_ip = self._anchor_thumb_ip_palm
        rest_elev = self._anchor_thumb_elev
        if rest_ip is None or rest_elev is None:
            return None, None

        cmc_fe_sign = (self.cfg.thumb_cmc_fe_sign_left if side == "left"
                       else self.cfg.thumb_cmc_fe_sign_right)
        cmc_aa_sign = (self.cfg.thumb_cmc_aa_sign_left if side == "left"
                       else self.cfg.thumb_cmc_aa_sign_right)
        # CMC_FE: out-of-plane elevation delta, amplified to bridge the
        # human / Sharpa thumb-folding range mismatch.
        cmc_fe = cmc_fe_sign * (elev - rest_elev) * self.cfg.thumb_opp_gain
        # CMC_AA: in-plane (lateral-vs-forward) delta of the metacarpal.
        cmc_aa = cmc_aa_sign * (ip_palm - rest_ip) * self.cfg.thumb_aa_gain
        return cmc_fe, cmc_aa

    def _compute_pinky_cmc(
        self, kps: dict,
        fwd: np.ndarray, lat: np.ndarray, nrm: np.ndarray,
    ) -> Optional[float]:
        """Pinky CMC tracks how far the pinky metacarpal pulls toward the
        thumb side relative to anchor — captures palm cupping. Limit is
        small (0..15°) so we just need the right sign + magnitude."""
        meta = self._kp_pos(kps, "pinky-finger-metacarpal")
        wrist = self._kp_pos(kps, "wrist")
        if meta is None or wrist is None:
            return None
        v = meta - wrist
        v = v - np.dot(v, nrm) * nrm
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        v /= n
        # Lateral component normalised by the in-plane direction.
        ang = float(np.arctan2(np.dot(v, lat), np.dot(v, fwd)))
        if self._anchor_pending:
            self._anchor_pinky_lat = ang
            return None
        if self._anchor_pinky_lat is None:
            return None
        # Pinky cupping = pinky metacarpal swings toward thumb side. The
        # joint limit is positive-only, so take the "tuck-in" direction.
        cmc = max(0.0, self._anchor_pinky_lat - ang) * self.cfg.pinky_cmc_gain
        return cmc
