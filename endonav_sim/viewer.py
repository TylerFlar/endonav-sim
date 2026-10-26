"""Realtime pygame viewer + input handler for :class:`EndonavEnv`.

Drives an EndonavEnv at a fixed-timestep rate (default 30 Hz) and renders
each frame in a desktop window. Inputs (keyboard, gamepad, on-screen
analog sticks) are mapped onto the env's continuum action space so the
same env can be driven by a human OR by an agent callable.

This file is intentionally self-contained: import only when you actually
want a window. The env module does NOT import pygame.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pygame
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "endonav_sim.viewer requires `pygame`. Install with `pip install pygame`."
    ) from e

from .env import TOOL_BASKET, TOOL_LASER, TOOL_NONE, TOOL_RESET_POSE, EndonavEnv
from .visuals import compose_visuals

AgentFn = Callable[[dict], dict]


@dataclass
class _ContinuumTargets:
    """Operator's current per-section bend targets, integrated from input."""

    kappa1: float = 0.0
    phi1: float = 0.0
    kappa2: float = 0.0
    phi2: float = 0.0


@dataclass
class _Stats:
    fps: float = 0.0
    last_buckled: bool = False
    last_collided: bool = False
    last_clearance: float = 0.0
    last_contact: float = 0.0
    last_node: str = ""


class RealtimeViewer:
    """Pygame event loop that drives an :class:`EndonavEnv`.

    Usage:
        env = EndonavEnv(seed=42)
        viewer = RealtimeViewer(env, target_hz=30)
        viewer.run()

    Optional ``agent`` is any callable ``obs -> action_dict``. When set,
    pressing ``Tab`` toggles between human input and agent control.
    """

    def __init__(
        self,
        env: EndonavEnv,
        target_hz: int = 30,
        window_title: str = "endonav-sim — realtime",
        agent: AgentFn | None = None,
    ) -> None:
        self.env = env
        self.target_hz = int(target_hz)
        self.dt = 1.0 / self.target_hz
        self.agent = agent
        self.agent_mode = False

        pygame.init()
        pygame.display.set_caption(window_title)
        self.surface = pygame.display.set_mode(
            (env._width, env._height + 120)  # extra strip for HUD/sticks
        )
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_big = pygame.font.SysFont("consolas", 18, bold=True)
        self.clock = pygame.time.Clock()

        # Optional gamepad
        pygame.joystick.init()
        self.joy: Optional[pygame.joystick.Joystick] = None
        if pygame.joystick.get_count() > 0:
            self.joy = pygame.joystick.Joystick(0)
            self.joy.init()

        self.targets = _ContinuumTargets()
        self.stats = _Stats()
        self._jitter_rng = np.random.default_rng(0)

        # Last observation cache for the agent path
        self._last_obs: dict | None = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        obs, _info = self.env.reset()
        self._last_obs = obs
        running = True
        accumulator = 0.0
        last_time = time.monotonic()
        try:
            while running:
                now = time.monotonic()
                accumulator += now - last_time
                last_time = now

                # Pump events even if we are not stepping yet
                running = self._pump_events()
                if not running:
                    break

                # Fixed-step physics
                while accumulator >= self.dt:
                    action = self._build_action()
                    obs, _r, _term, _trunc, _info = self.env.step(action)
                    self._last_obs = obs
                    self.stats.last_buckled = bool(obs["buckled"])
                    self.stats.last_collided = bool(obs["collided"])
                    self.stats.last_clearance = float(obs["wall_clearance_mm"])
                    self.stats.last_contact = float(obs["contact_force"])
                    self.stats.last_node = str(_info.get("current_tree_node", ""))
                    accumulator -= self.dt

                # Render the most recent frame
                self._draw(obs)
                self.clock.tick(self.target_hz)
                self.stats.fps = self.clock.get_fps()
        finally:
            pygame.quit()

    # ------------------------------------------------------------------
    def _pump_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.env.reset()
                    self.targets = _ContinuumTargets()
                if event.key == pygame.K_TAB and self.agent is not None:
                    self.agent_mode = not self.agent_mode
        return True

    # ------------------------------------------------------------------
    def _build_action(self) -> dict:
        if self.agent_mode and self.agent is not None and self._last_obs is not None:
            try:
                a = self.agent(self._last_obs)
            except Exception as e:  # noqa: BLE001
                print(f"agent error: {e}")
                a = {}
            return self._fill_default_action(a)

        keys = pygame.key.get_pressed()
        L = self.env._scope_limits

        advance = 0.0
        roll = 0.0
        tool = TOOL_NONE

        # Keyboard advance / roll — full step per frame for responsiveness.
        if keys[pygame.K_w]:
            advance += L.max_advance_per_step_mm
        if keys[pygame.K_s]:
            advance -= L.max_advance_per_step_mm
        if keys[pygame.K_q]:
            roll -= L.max_roll_per_step_deg
        if keys[pygame.K_e]:
            roll += L.max_roll_per_step_deg

        # Per-section bending: IJKL / arrows act as a virtual joystick.
        # Hold a direction key = bend that way; release = spring back to straight.
        # This gives immediate, responsive "stick-like" control.
        bend_strength = 0.6  # fraction of max curvature when a key is held
        decay = 0.80  # per-frame decay toward straight when no key is held

        # S2 (distal tip): IJKL
        dx2 = (1.0 if keys[pygame.K_l] else 0.0) - (1.0 if keys[pygame.K_j] else 0.0)
        dy2 = (1.0 if keys[pygame.K_i] else 0.0) - (1.0 if keys[pygame.K_k] else 0.0)
        if abs(dx2) + abs(dy2) > 0.01:
            mag = min(1.0, float(np.hypot(dx2, dy2)))
            self.targets.kappa2 = L.max_kappa2_per_mm * bend_strength * mag
            self.targets.phi2 = float(np.arctan2(dy2, dx2))
        else:
            self.targets.kappa2 *= decay

        # S1 (proximal): arrow keys
        dx1 = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (1.0 if keys[pygame.K_LEFT] else 0.0)
        dy1 = (1.0 if keys[pygame.K_UP] else 0.0) - (1.0 if keys[pygame.K_DOWN] else 0.0)
        if abs(dx1) + abs(dy1) > 0.01:
            mag = min(1.0, float(np.hypot(dx1, dy1)))
            self.targets.kappa1 = L.max_kappa1_per_mm * bend_strength * mag
            self.targets.phi1 = float(np.arctan2(dy1, dx1))
        else:
            self.targets.kappa1 *= decay

        if keys[pygame.K_SPACE]:
            tool = TOOL_BASKET
        if keys[pygame.K_f]:
            tool = TOOL_LASER

        # Gamepad: left stick = (kappa1, phi1), right stick = (kappa2, phi2)
        if self.joy is not None:
            try:
                lx = self._dz(self.joy.get_axis(0))
                ly = self._dz(-self.joy.get_axis(1))
                rx = self._dz(self.joy.get_axis(2))
                ry = self._dz(-self.joy.get_axis(3))
                if abs(lx) + abs(ly) > 1e-3:
                    self.targets.kappa1 = float(min(L.max_kappa1_per_mm,
                                                    np.hypot(lx, ly) * L.max_kappa1_per_mm))
                    self.targets.phi1 = float(np.arctan2(ly, lx))
                if abs(rx) + abs(ry) > 1e-3:
                    self.targets.kappa2 = float(min(L.max_kappa2_per_mm,
                                                    np.hypot(rx, ry) * L.max_kappa2_per_mm))
                    self.targets.phi2 = float(np.arctan2(ry, rx))
                # Triggers (axes 4/5 on most pads): advance ±
                try:
                    lt = (self.joy.get_axis(4) + 1.0) * 0.5
                    rt = (self.joy.get_axis(5) + 1.0) * 0.5
                    advance += (rt - lt) * adv_step
                except Exception:  # noqa: BLE001
                    pass
                # Buttons
                if self.joy.get_numbuttons() > 0:
                    if self.joy.get_button(0):  # A
                        tool = TOOL_BASKET
                    if self.joy.get_button(1):  # B
                        tool = TOOL_LASER
                    if self.joy.get_button(7):  # Start
                        tool = TOOL_RESET_POSE
                    if self.joy.get_button(4):  # LB
                        roll -= roll_step
                    if self.joy.get_button(5):  # RB
                        roll += roll_step
            except Exception:  # noqa: BLE001
                pass

        return {
            "advance_mm": float(advance),
            "roll_deg": float(roll),
            "kappa1": float(self.targets.kappa1),
            "phi1": float(self.targets.phi1),
            "kappa2": float(self.targets.kappa2),
            "phi2": float(self.targets.phi2),
            "tool": int(tool),
        }

    @staticmethod
    def _dz(x: float, dz: float = 0.12) -> float:
        return 0.0 if abs(x) < dz else float(x)

    @staticmethod
    def _fill_default_action(a: dict) -> dict:
        return {
            "advance_mm": float(a.get("advance_mm", 0.0)),
            "roll_deg": float(a.get("roll_deg", 0.0)),
            "kappa1": float(a.get("kappa1", 0.0)),
            "phi1": float(a.get("phi1", 0.0)),
            "kappa2": float(a.get("kappa2", 0.0)),
            "phi2": float(a.get("phi2", 0.0)),
            "tool": int(a.get("tool", TOOL_NONE)),
        }

    # ------------------------------------------------------------------
    def _draw(self, obs: dict) -> None:
        rgb = obs["rgb"]
        # Compose tool visual effects on top of the clean frame.
        tvs = self.env.sim.tool_visual_state
        cinematic = compose_visuals(
            rgb, tool_state=tvs, jitter_rng=None, apply_motion=False
        )
        # pygame surfarray expects (W,H,3); transpose.
        surf_rgb = np.transpose(cinematic, (1, 0, 2)).copy()
        cam_surface = pygame.surfarray.make_surface(surf_rgb)
        self.surface.blit(cam_surface, (0, 0))
        # HUD strip
        self._draw_hud()
        pygame.display.flip()

    def _draw_hud(self) -> None:
        h_top = self.env._height
        w = self.env._width
        pygame.draw.rect(self.surface, (16, 16, 22), (0, h_top, w, 120))

        s = self.stats
        mode = "AGENT" if self.agent_mode else "HUMAN"
        status_flags = ""
        if s.last_buckled:
            status_flags += " BUCKLED"
        if s.last_collided:
            status_flags += " COLLIDED"
        lines = [
            f"{mode}  fps={s.fps:5.1f}  node={s.last_node}"
            f"  clearance={s.last_clearance:5.2f}mm  contact={s.last_contact:.2f}"
            f"{status_flags}",
            f"k1={self.targets.kappa1:.4f}/mm phi1={np.rad2deg(self.targets.phi1):+6.1f}deg"
            f"   k2={self.targets.kappa2:.4f}/mm phi2={np.rad2deg(self.targets.phi2):+6.1f}deg",
            "W/S=advance  Q/E=roll  Arrows=bend S1  IJKL=bend S2  Space=basket  F=laser  R=reset",
        ]
        for i, line in enumerate(lines):
            color = (130, 130, 140) if i == 2 else (220, 220, 230)
            txt = self.font.render(line, True, color)
            self.surface.blit(txt, (10, h_top + 8 + i * 18))

        # On-screen analog sticks for the two sections.
        L = self.env._scope_limits
        self._draw_stick(
            cx=w - 220, cy=h_top + 60,
            radius=45,
            kappa=self.targets.kappa1, phi=self.targets.phi1,
            kappa_max=L.max_kappa1_per_mm,
            label="S1",
        )
        self._draw_stick(
            cx=w - 110, cy=h_top + 60,
            radius=45,
            kappa=self.targets.kappa2, phi=self.targets.phi2,
            kappa_max=L.max_kappa2_per_mm,
            label="S2",
        )

    def _draw_stick(
        self, cx: int, cy: int, radius: int,
        kappa: float, phi: float, kappa_max: float, label: str,
    ) -> None:
        pygame.draw.circle(self.surface, (60, 60, 70), (cx, cy), radius, 2)
        mag = float(min(1.0, abs(kappa) / max(kappa_max, 1e-9)))
        x = int(cx + np.cos(phi) * mag * (radius - 4))
        y = int(cy - np.sin(phi) * mag * (radius - 4))
        pygame.draw.circle(self.surface, (180, 220, 255), (x, y), 6)
        txt = self.font.render(label, True, (200, 200, 210))
        self.surface.blit(txt, (cx - 8, cy + radius + 2))


__all__ = ["RealtimeViewer", "AgentFn"]
