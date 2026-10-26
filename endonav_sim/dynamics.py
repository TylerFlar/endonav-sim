"""Realistic robotic ureteroscope dynamics layer.

Wraps the clean kinematic ``KidneySimulator.command`` with all the
imperfections a real tendon-driven, multi-section continuum scope exhibits:

  * encoder quantization, dead-zones and rate limits
  * tendon-sheath backlash hysteresis on roll and per-section curvature
  * multiplicative Gaussian noise on commanded magnitudes
  * tendon cross-coupling between proximal / distal sections
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
    # Per-step rate limits (motor-driven robotic stage).
    max_advance_per_step_mm: float = 2.0
    max_roll_per_step_deg: float = 15.0
    # Encoder resolution.
    advance_resolution_mm: float = 0.05
    roll_resolution_deg: float = 0.5
    # Multiplicative Gaussian noise (sd as fraction of commanded magnitude).
    advance_noise_frac: float = 0.06
    roll_noise_frac: float = 0.10
    # Backlash hysteresis on roll direction reversal.
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

    # ---------------- Per-section continuum limits ----------------------
    # Section lengths (mm). Mirror endonav_sim.continuum defaults but
    # duplicated here so the dynamics layer is self-contained.
    section1_length_mm: float = 8.0
    section2_length_mm: float = 5.0
    # Curvature targets are commanded in 1/mm. With shorter sections the
    # max curvature can be higher: 0.15/mm on an 8mm section gives ~69°
    # of bend, and 0.25/mm on a 5mm section gives ~72°, both within the
    # range of clinical multi-section flexible scopes.
    max_kappa1_per_mm: float = 0.15
    max_kappa2_per_mm: float = 0.25
    # Per-step rate limits on the curvature target (1/mm per command).
    max_kappa1_rate_per_step: float = 0.030
    max_kappa2_rate_per_step: float = 0.050
    # Per-step rate limit on the bending plane angle (deg per command).
    max_phi_rate_per_step_deg: float = 25.0
    # Encoder resolution / dead zone for curvature.
    kappa_resolution_per_mm: float = 0.001
    kappa_deadzone_per_mm: float = 0.0015
    # Multiplicative Gaussian noise on curvature deltas.
    kappa_noise_frac: float = 0.06
    # Backlash hysteresis on curvature direction reversals (1/mm).
    kappa_hysteresis_per_mm: float = 0.003
    # Tendon coupling: bending S1 perturbs S2 (and vice versa) by a
    # fraction of the parent's curvature. Real cable-driven multi-section
    # scopes typically show ~5–15 % cross-coupling.
    tendon_coupling_s1_to_s2: float = 0.10
    tendon_coupling_s2_to_s1: float = 0.05


@dataclass
class CommandFeedback:
    """Per-section telemetry returned after dynamics shaping.

    Exposes the curvature / phi targets that actually made it through the
    dynamics layer (after rate-limiting, quantization, hysteresis, noise
    and tendon coupling) plus the shaped advance and roll.
    """

    actual_advance_mm: float
    actual_roll_deg: float
    actual_kappa1_per_mm: float
    actual_phi1_rad: float
    actual_kappa2_per_mm: float
    actual_phi2_rad: float


class ScopeDynamics:
    """Stateful imperfection layer wrapping a kinematic command callback.

    Owns its own RNG seeded off ``seed`` so all noise is reproducible.
    Tracks per-instance hysteresis state and a separate dead-reckoned pose
    estimate that drifts away from the true pose over time.
    """

    def __init__(self, limits: ScopeLimits, seed: int = 0) -> None:
        self.limits = limits
        self.rng = np.random.default_rng(seed)
        self._last_roll_dir: int = 0
        self._last_kappa1_dir: int = 0
        self._last_kappa2_dir: int = 0
        self.pose_estimate: np.ndarray | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _quantize(value: float, step: float) -> float:
        if step <= 0.0:
            return float(value)
        return float(np.round(value / step) * step)

    def reset(self, true_pose: np.ndarray) -> None:
        self._last_roll_dir = 0
        self._last_kappa1_dir = 0
        self._last_kappa2_dir = 0
        self.pose_estimate = true_pose.copy()

    # ------------------------------------------------------------------
    def shape_command(
        self,
        advance_mm: float,
        roll_deg: float,
        kappa1_target: float,
        phi1_target: float,
        kappa2_target: float,
        phi2_target: float,
        prev_kappa1: float,
        prev_phi1: float,
        prev_kappa2: float,
        prev_phi2: float,
    ) -> CommandFeedback:
        """Apply rate-limit / quantize / dead-zone / hysteresis / noise /
        tendon-coupling to a continuum scope command.

        ``advance_mm`` and ``roll_deg`` go through translation/roll shaping
        (clip, quantize, hysteresis, noise, retraction slip). Per-section
        ``(kappa, phi)`` targets are slewed toward the request with rate
        limits, dead-zoned, hysteresis-corrected on direction reversals,
        then perturbed by Gaussian noise. Finally a fixed cross-coupling
        matrix bleeds each section's curvature into the other to mimic
        shared-cable behavior.
        """
        L = self.limits

        # ---- advance + roll ------------------------------------------------
        adv = float(np.clip(advance_mm, -L.max_advance_per_step_mm, L.max_advance_per_step_mm))
        rl = float(np.clip(roll_deg, -L.max_roll_per_step_deg, L.max_roll_per_step_deg))
        adv = self._quantize(adv, L.advance_resolution_mm)
        rl = self._quantize(rl, L.roll_resolution_deg)

        roll_dir = int(np.sign(rl))
        if roll_dir != 0 and self._last_roll_dir != 0 and roll_dir != self._last_roll_dir:
            slack = L.roll_hysteresis_deg
            rl = rl - roll_dir * min(slack, abs(rl))
        if roll_dir != 0:
            self._last_roll_dir = roll_dir

        if adv != 0.0:
            adv *= 1.0 + float(self.rng.normal(0.0, L.advance_noise_frac))
        if rl != 0.0:
            rl *= 1.0 + float(self.rng.normal(0.0, L.roll_noise_frac))
        if adv < 0.0:
            adv *= 1.0 - L.retraction_slip_frac

        # ---- per-section curvature shaping ---------------------------------
        def _shape_section(
            target: float,
            prev: float,
            max_kappa: float,
            max_rate: float,
            last_dir_attr: str,
        ) -> float:
            target = float(np.clip(target, -max_kappa, max_kappa))
            target = self._quantize(target, L.kappa_resolution_per_mm)
            delta = target - prev
            if abs(delta) > max_rate:
                delta = float(np.sign(delta)) * max_rate
                target = prev + delta
            if abs(delta) < L.kappa_deadzone_per_mm:
                return prev
            cur_dir = int(np.sign(delta))
            last_dir = int(getattr(self, last_dir_attr))
            if cur_dir != 0 and last_dir != 0 and cur_dir != last_dir:
                slack = L.kappa_hysteresis_per_mm
                eaten = min(slack, abs(delta))
                delta = delta - cur_dir * eaten
                target = prev + delta
            if cur_dir != 0:
                setattr(self, last_dir_attr, cur_dir)
            if delta != 0.0:
                target = prev + delta * (1.0 + float(self.rng.normal(0.0, L.kappa_noise_frac)))
            return float(np.clip(target, -max_kappa, max_kappa))

        new_k1 = _shape_section(
            kappa1_target, prev_kappa1, L.max_kappa1_per_mm, L.max_kappa1_rate_per_step,
            "_last_kappa1_dir",
        )
        new_k2 = _shape_section(
            kappa2_target, prev_kappa2, L.max_kappa2_per_mm, L.max_kappa2_rate_per_step,
            "_last_kappa2_dir",
        )

        # Tendon coupling: each section pulls a fraction of its delta into
        # the other section.
        d1 = new_k1 - prev_kappa1
        d2 = new_k2 - prev_kappa2
        new_k1 = float(np.clip(new_k1 + L.tendon_coupling_s2_to_s1 * d2,
                                -L.max_kappa1_per_mm, L.max_kappa1_per_mm))
        new_k2 = float(np.clip(new_k2 + L.tendon_coupling_s1_to_s2 * d1,
                                -L.max_kappa2_per_mm, L.max_kappa2_per_mm))

        # ---- phi: rate-limited slew ----------------------------------------
        def _shape_phi(target: float, prev: float) -> float:
            d = float(target) - float(prev)
            d = (d + np.pi) % (2.0 * np.pi) - np.pi
            max_d = float(np.deg2rad(L.max_phi_rate_per_step_deg))
            if abs(d) > max_d:
                d = float(np.sign(d)) * max_d
            return float(prev + d)

        new_p1 = _shape_phi(phi1_target, prev_phi1)
        new_p2 = _shape_phi(phi2_target, prev_phi2)

        return CommandFeedback(
            actual_advance_mm=float(adv),
            actual_roll_deg=float(rl),
            actual_kappa1_per_mm=float(new_k1),
            actual_phi1_rad=float(new_p1),
            actual_kappa2_per_mm=float(new_k2),
            actual_phi2_rad=float(new_p2),
        )

    # ------------------------------------------------------------------
    def buckle_perturbation(self) -> float:
        """Random extra deflection (deg) when the shaft bows under load."""
        return float(self.rng.normal(0.0, self.limits.buckling_wiggle_deg))

    # ------------------------------------------------------------------
    def integrate_pose_estimate(
        self,
        commanded_advance_mm: float,
        commanded_roll_deg: float,
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
        forward = -est[:3, 2]
        est[:3, 3] = est[:3, 3] + forward * commanded_advance_mm
        est[:3, 3] = est[:3, 3] + self.rng.normal(0.0, L.pose_drift_pos_mm, size=3)
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
