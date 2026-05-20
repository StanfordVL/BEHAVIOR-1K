"""Safety filter for the published 29-DoF teleop qpos.

Sits between the arm IK + hand retargeter outputs and the TCP/state publish.
Three layers, applied in order:

  1. EMA smoothing (low-pass) — knocks out single-frame jitter.
  2. Max joint velocity (rad/s) — clamps |Δqpos / dt|.
  3. Max per-tick Δqpos (rad)  — hard absolute cap on each tick's delta.

All three are optional and tunable per joint group (arm vs hand). The filter
holds state (`prev_qpos`) so it must be `reset(qpos)`-ed on anchor / scene
reset to avoid slewing the new home pose through the rate limiter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SafetyConfig:
    enabled: bool = True
    # EMA smoothing coefficient applied to the *target* qpos before
    # rate limits.  1.0 = no smoothing (pass-through), 0.0 = freeze.
    # Typical 0.7 - 1.0 for crisp tracking; lower values lag more.
    smoothing_alpha: float = 1.0
    # Max joint velocity (rad/s). Applied as |Δq| <= max_vel * dt.
    max_arm_velocity: float = 2.0      # iiwa joint vel limit ≈ 1.4 - 2.1 rad/s
    max_hand_velocity: float = 6.0     # finger joints can move faster
    # Hard absolute cap on Δq per tick (rad). Useful as an emergency brake
    # decoupled from the configured control rate. Set <=0 or None to disable.
    max_arm_delta_per_tick: float = 0.10    # ~5.7° per tick
    max_hand_delta_per_tick: float = 0.20   # ~11.5° per tick
    # If True, log a warning the first time each layer clips the output.
    log_clips: bool = True


class SafetyFilter:
    """Rate-limit + smooth a stream of qpos vectors.

    Args:
        n_arm: number of arm joints in the qpos vector (assumed contiguous at the front).
        n_hand: number of hand joints (after the arm joints).
        dt: control timestep in seconds (= 1 / control_hz).
        cfg: SafetyConfig.
    """

    def __init__(self, n_arm: int, n_hand: int, dt: float, cfg: SafetyConfig):
        self.cfg = cfg
        self.n_arm = int(n_arm)
        self.n_hand = int(n_hand)
        self.n = self.n_arm + self.n_hand
        self.dt = float(dt)

        self._prev: Optional[np.ndarray] = None

        # Pre-build per-joint velocity / delta caps as (n,) arrays so the
        # hot path is one np.clip per layer.
        self._max_vel = np.concatenate([
            np.full(self.n_arm, cfg.max_arm_velocity, dtype=float),
            np.full(self.n_hand, cfg.max_hand_velocity, dtype=float),
        ])
        if cfg.max_arm_delta_per_tick is None or cfg.max_arm_delta_per_tick <= 0:
            arm_delta = np.full(self.n_arm, np.inf, dtype=float)
        else:
            arm_delta = np.full(self.n_arm, cfg.max_arm_delta_per_tick, dtype=float)
        if cfg.max_hand_delta_per_tick is None or cfg.max_hand_delta_per_tick <= 0:
            hand_delta = np.full(self.n_hand, np.inf, dtype=float)
        else:
            hand_delta = np.full(self.n_hand, cfg.max_hand_delta_per_tick, dtype=float)
        self._max_delta = np.concatenate([arm_delta, hand_delta])

        # One-shot warn flags so we don't spam the log every clipped tick.
        self._warned_smooth = False
        self._warned_vel = False
        self._warned_delta = False

    def reset(self, qpos: np.ndarray) -> None:
        """Snap the filter state to `qpos`. Call on anchor / scene reset."""
        self._prev = np.asarray(qpos, dtype=float).copy()

    def filter(self, qpos: np.ndarray, log=None) -> np.ndarray:
        """Return a safety-filtered copy of `qpos`.

        Pass-through if the filter is disabled or this is the first call
        (state not yet primed) — in the first-call case we just record the
        qpos so that subsequent calls have a baseline for the rate limit.
        """
        q = np.asarray(qpos, dtype=float).copy()
        if not self.cfg.enabled or self._prev is None:
            self._prev = q.copy()
            return q

        # ---- 1. EMA smoothing ----
        a = float(self.cfg.smoothing_alpha)
        if 0.0 <= a < 1.0:
            target = a * q + (1.0 - a) * self._prev
        else:
            target = q

        # ---- 2. Max velocity ----
        delta = target - self._prev
        vel_cap = self._max_vel * self.dt
        clipped_v = np.clip(delta, -vel_cap, vel_cap)
        if log is not None and self.cfg.log_clips and not self._warned_vel \
                and np.any(np.abs(delta) > vel_cap + 1e-9):
            log.warning("SafetyFilter: velocity cap engaged "
                        "(|Δq|/dt exceeded max_*_velocity).")
            self._warned_vel = True
        delta = clipped_v

        # ---- 3. Hard per-tick Δq cap ----
        if np.isfinite(self._max_delta).any():
            clipped_d = np.clip(delta, -self._max_delta, self._max_delta)
            if log is not None and self.cfg.log_clips and not self._warned_delta \
                    and np.any(np.abs(delta) > self._max_delta + 1e-9):
                log.warning("SafetyFilter: per-tick Δq cap engaged "
                            "(exceeded max_*_delta_per_tick).")
                self._warned_delta = True
            delta = clipped_d

        out = self._prev + delta
        self._prev = out.copy()
        return out
