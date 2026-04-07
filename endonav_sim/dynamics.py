"""Realistic robotic ureteroscope dynamics layer.

Wraps the clean kinematic ``KidneySimulator.command`` with all the
imperfections a real tendon-driven scope exhibits:

  * encoder quantization, dead-zones and rate limits
  * tendon-sheath backlash hysteresis on roll and deflection
  * multiplicative Gaussian noise on commanded magnitudes
  * shaft buckling on collision (productive in wide cavities, blocking in
    narrow ones)
  * retraction slip
  * dead-reckoning pose-estimate drift

Default values come from the literature cited in the plan document
(Pietrow 2004 buckling pressures, Wang 2015 / Zhang 2017 tendon-sheath
nonlinearities, LithoVue / BD Aptra ureteroscope specs).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScopeLimits:
    """Hardware envelope and noise model for one scope instance."""

    # Geometric limits — anchored to real flexible ureteroscope specs.
    shaft_outer_diameter_mm: float = 3.0  # ~9 Fr
    working_length_mm: float = 700.0  # 65-75 cm clinical
    max_deflection_deg: float = 270.0  # bidirectional 270° up + 270° down
    # Per-step rate limits (motor-driven robotic stage).
    max_advance_per_step_mm: float = 2.0
    max_roll_per_step_deg: float = 15.0
    max_deflection_per_step_deg: float = 20.0
    # Encoder resolution / dead zones.
    advance_resolution_mm: float = 0.05
    roll_resolution_deg: float = 0.5
    deflection_resolution_deg: float = 1.0
    deflection_deadzone_deg: float = 2.0
    # Multiplicative Gaussian noise (sd as fraction of commanded magnitude).
    advance_noise_frac: float = 0.06
    roll_noise_frac: float = 0.10
    deflection_noise_frac: float = 0.05
    # Backlash hysteresis: cable play eaten when the command direction reverses.
    deflection_hysteresis_deg: float = 4.0
    roll_hysteresis_deg: float = 2.0
    # Buckling: when an advance is rejected by the wall, the shaft bows
    # sideways. Easy to trigger (real scopes buckle at 6-12g axial load).
    buckling_wiggle_deg: float = 8.0
    # Retraction slip — tip catches on mucosa as you pull back.
    retraction_slip_frac: float = 0.15
    # Dead-reckoning pose-estimate drift, applied per step.
    pose_drift_pos_mm: float = 0.05
    pose_drift_angle_deg: float = 0.3
    # Wall-clearance proprioception noise (additive, mm).
    wall_clearance_quantum_mm: float = 0.5
    wall_clearance_noise_mm: float = 0.3


@dataclass
class CommandFeedback:
    """Telemetry returned to the controller after each command attempt.

    This is everything a real robotic ureteroscope can plausibly report:
    encoder readback, contact-force estimate, buckling flag, noisy wall-
    clearance estimate, and a dead-reckoned pose. NO ground-truth anatomy
    leaks through this channel.
    """

    actual_advance_mm: float
    actual_roll_deg: float
    actual_deflection_deg: float
    contact_force_norm: float  # 0..1
    buckled: bool
    wall_clearance_mm: float  # quantized + noisy
    tip_pose_estimate: np.ndarray  # (4,4)
    collided: bool  # tip motion fully blocked despite buckling


class ScopeDynamics:
    """Stateful imperfection layer wrapping a kinematic command callback.

    Owns its own RNG seeded off ``seed`` so all noise is reproducible.
    Tracks per-instance hysteresis state and a separate dead-reckoned pose
    estimate that drifts away from the true pose over time.
    """

    def __init__(self, limits: ScopeLimits, seed: int = 0) -> None:
        self.limits = limits
        self.rng = np.random.default_rng(seed)
        # Sign of the previous deflection / roll *increment* (not absolute
        # value). When the sign flips we eat a hysteresis chunk.
        self._last_deflection_dir: int = 0  # -1, 0, +1
        self._last_roll_dir: int = 0
        # Pose estimate (commanded-integration). Initialized lazily on the
        # first call to :meth:`apply` from the simulator's true pose.
        self.pose_estimate: np.ndarray | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _quantize(value: float, step: float) -> float:
        if step <= 0.0:
            return float(value)
        return float(np.round(value / step) * step)

    def reset(self, true_pose: np.ndarray) -> None:
        self._last_deflection_dir = 0
        self._last_roll_dir = 0
        self.pose_estimate = true_pose.copy()

    # ------------------------------------------------------------------
    def shape_command(
        self,
        advance_mm: float,
        roll_deg: float,
        deflection_deg: float,
        prev_deflection_deg: float,
    ) -> tuple[float, float, float]:
        """Apply quantization, dead-zone, hysteresis and noise.

        Returns the (possibly distorted) values that should actually be
        forwarded to the kinematic backend.
        """
        L = self.limits

        # 1. Quantize / clip / rate-limit
        adv = float(np.clip(advance_mm, -L.max_advance_per_step_mm, L.max_advance_per_step_mm))
        rl = float(np.clip(roll_deg, -L.max_roll_per_step_deg, L.max_roll_per_step_deg))
        defl_target = float(np.clip(deflection_deg, -L.max_deflection_deg, L.max_deflection_deg))

        adv = self._quantize(adv, L.advance_resolution_mm)
        rl = self._quantize(rl, L.roll_resolution_deg)
        defl_target = self._quantize(defl_target, L.deflection_resolution_deg)

        # 2. Deflection dead-zone — small *changes* are absorbed by cable slack.
        defl_delta = defl_target - prev_deflection_deg
        if abs(defl_delta) < L.deflection_deadzone_deg:
            defl_target = prev_deflection_deg
            defl_delta = 0.0

        # 3. Hysteresis on direction reversal
        cur_dir = int(np.sign(defl_delta))
        if cur_dir != 0 and self._last_deflection_dir != 0 and cur_dir != self._last_deflection_dir:
            # Eat backlash slack opposite to the new motion direction
            slack = L.deflection_hysteresis_deg
            defl_target = prev_deflection_deg + (defl_delta - cur_dir * min(slack, abs(defl_delta)))
            # Re-quantize after slack subtraction
            defl_target = self._quantize(defl_target, L.deflection_resolution_deg)
            defl_delta = defl_target - prev_deflection_deg
            cur_dir = int(np.sign(defl_delta))
        if cur_dir != 0:
            self._last_deflection_dir = cur_dir

        # Same idea for roll
        roll_dir = int(np.sign(rl))
        if roll_dir != 0 and self._last_roll_dir != 0 and roll_dir != self._last_roll_dir:
            slack = L.roll_hysteresis_deg
            rl = rl - roll_dir * min(slack, abs(rl))
        if roll_dir != 0:
            self._last_roll_dir = roll_dir

        # 4. Multiplicative Gaussian noise
        if adv != 0.0:
            adv *= 1.0 + float(self.rng.normal(0.0, L.advance_noise_frac))
        if rl != 0.0:
            rl *= 1.0 + float(self.rng.normal(0.0, L.roll_noise_frac))
        if defl_delta != 0.0:
            # Add noise to the *delta*, then re-anchor
            defl_target = prev_deflection_deg + defl_delta * (
                1.0 + float(self.rng.normal(0.0, L.deflection_noise_frac))
            )

        # 5. Retraction slip
        if adv < 0.0:
            adv *= 1.0 - L.retraction_slip_frac

        return adv, rl, defl_target

    # ------------------------------------------------------------------
    def buckle_perturbation(self) -> float:
        """Random extra deflection (deg) when the shaft bows under load."""
        return float(self.rng.normal(0.0, self.limits.buckling_wiggle_deg))

    # ------------------------------------------------------------------
    def integrate_pose_estimate(
        self,
        commanded_advance_mm: float,
        commanded_roll_deg: float,
        commanded_deflection_deg: float,
        true_pose: np.ndarray,
    ) -> np.ndarray:
        """Dead-reckon the pose estimate from commanded deltas plus drift.

        We do *not* peek at the true pose for content — we only use it to
        seed the estimate on the first call so the two start aligned.
        """
        L = self.limits
        if self.pose_estimate is None:
            self.pose_estimate = true_pose.copy()
        est = self.pose_estimate.copy()
        # Translate along the local forward axis (-Z column of the pose).
        forward = -est[:3, 2]
        est[:3, 3] = est[:3, 3] + forward * commanded_advance_mm
        # Add Gaussian drift to position
        est[:3, 3] = est[:3, 3] + self.rng.normal(0.0, L.pose_drift_pos_mm, size=3)
        # Approximate orientation drift as a small rotation around a random axis
        if L.pose_drift_angle_deg > 0.0:
            axis = self.rng.normal(0.0, 1.0, size=3)
            axis /= max(float(np.linalg.norm(axis)), 1e-9)
            theta = float(np.deg2rad(self.rng.normal(0.0, L.pose_drift_angle_deg)))
            c, s = float(np.cos(theta)), float(np.sin(theta))
            K = np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
            est[:3, :3] = R @ est[:3, :3]
        self.pose_estimate = est
        return est.copy()

    # ------------------------------------------------------------------
    def quantize_clearance(self, true_clearance_mm: float) -> float:
        L = self.limits
        noisy = true_clearance_mm + float(self.rng.normal(0.0, L.wall_clearance_noise_mm))
        return float(self._quantize(max(noisy, 0.0), L.wall_clearance_quantum_mm))


__all__ = ["ScopeLimits", "CommandFeedback", "ScopeDynamics"]
