"""Gymnasium environment wrapper around :class:`KidneySimulator`.

Exposes a 2-section continuum scope action space + structured observation
dict so any RL library (Stable-Baselines3, CleanRL, RLlib) or scripted /
LLM agent can drive the simulator through a single uniform interface.

The action space is a Dict — verbose but self-documenting:

  advance_mm  Box(-2, +2)        translate along the shaft skeleton
  roll_deg    Box(-15, +15)      INCREMENTAL axial rotation
  kappa1      Box(-K1, +K1)      proximal-section curvature target (1/mm)
  phi1        Box(-pi, +pi)      proximal-section bending plane (rad)
  kappa2      Box(-K2, +K2)      distal-section curvature target (1/mm)
  phi2        Box(-pi, +pi)      distal-section bending plane (rad)
  tool        Discrete(4)        0=none, 1=basket, 2=laser, 3=reset_pose

Ground-truth keys (true pose, current_tree_node, visible stones) live in
``info`` rather than ``obs`` so policies cannot accidentally cheat.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "endonav_sim.env requires `gymnasium`. Install with `pip install gymnasium`."
    ) from e

from .anatomy import AnatomyParams
from .dynamics import ScopeLimits
from .simulator import KidneySimulator, ToolMode
from .stones import StoneParams

TOOL_NONE = 0
TOOL_BASKET = 1
TOOL_LASER = 2
TOOL_RESET_POSE = 3


class EndonavEnv(gym.Env):
    """Gymnasium env for the kidney ureteroscopy simulator.

    Parameters
    ----------
    seed:
        Seed used for anatomy, stones, and dynamics RNG. Calling
        :meth:`reset` with a different ``seed`` rebuilds the kidney mesh
        from scratch (slow). Calling it with the same seed only resets the
        scope pose (cheap).
    width, height:
        Render resolution.
    realistic:
        Toggle the dynamics imperfection layer. When ``False`` commands
        execute perfectly, useful for sanity tests.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        seed: int = 0,
        width: int = 1024,
        height: int = 768,
        realistic: bool = True,
        anatomy_params: AnatomyParams | None = None,
        stone_params: StoneParams | None = None,
        scope_limits: ScopeLimits | None = None,
        max_episode_steps: int = 2000,
    ) -> None:
        super().__init__()
        self._build_seed = int(seed)
        self._width = int(width)
        self._height = int(height)
        self._realistic = bool(realistic)
        self._anatomy_params = anatomy_params
        self._stone_params = stone_params if stone_params is not None else StoneParams(seed=seed)
        self._scope_limits = scope_limits or ScopeLimits()
        self._max_episode_steps = int(max_episode_steps)
        self._step_count = 0
        self._last_obs_rgb: np.ndarray | None = None

        self.sim = self._build_sim(self._build_seed)
        L = self._scope_limits

        self.action_space = spaces.Dict(
            {
                "advance_mm": spaces.Box(
                    low=-L.max_advance_per_step_mm,
                    high=L.max_advance_per_step_mm,
                    shape=(),
                    dtype=np.float32,
                ),
                "roll_deg": spaces.Box(
                    low=-L.max_roll_per_step_deg,
                    high=L.max_roll_per_step_deg,
                    shape=(),
                    dtype=np.float32,
                ),
                "kappa1": spaces.Box(
                    low=-L.max_kappa1_per_mm, high=L.max_kappa1_per_mm,
                    shape=(), dtype=np.float32,
                ),
                "phi1": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(), dtype=np.float32,
                ),
                "kappa2": spaces.Box(
                    low=-L.max_kappa2_per_mm, high=L.max_kappa2_per_mm,
                    shape=(), dtype=np.float32,
                ),
                "phi2": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(), dtype=np.float32,
                ),
                "tool": spaces.Discrete(4),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=255, shape=(height, width, 3), dtype=np.uint8,
                ),
                "wall_clearance_mm": spaces.Box(
                    low=0.0, high=100.0, shape=(), dtype=np.float32,
                ),
                "contact_force": spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32,
                ),
                "buckled": spaces.Discrete(2),
                "collided": spaces.Discrete(2),
                "tip_pose_estimate": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float32,
                ),
                "kappa1": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "phi1": spaces.Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                "kappa2": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32),
                "phi2": spaces.Box(low=-np.pi, high=np.pi, shape=(), dtype=np.float32),
                "actual_advance_mm": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(), dtype=np.float32,
                ),
                "actual_roll_deg": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(), dtype=np.float32,
                ),
            }
        )

    # ------------------------------------------------------------------
    def _build_sim(self, seed: int) -> KidneySimulator:
        anatomy = self._anatomy_params or AnatomyParams(seed=seed)
        return KidneySimulator(
            anatomy_params=anatomy,
            stone_params=self._stone_params,
            scope_limits=self._scope_limits,
            seed=seed,
            width=self._width,
            height=self._height,
            realistic=self._realistic,
        )

    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        if seed is not None and seed != self._build_seed:
            self._build_seed = int(seed)
            self.sim = self._build_sim(self._build_seed)
        else:
            self.sim.reset()
        self._step_count = 0
        obs, info = self._observe(
            buckled=False,
            collided=False,
            contact_force=0.0,
            wall_clearance_mm=0.0,
            actual_advance_mm=0.0,
            actual_roll_deg=0.0,
            tip_pose_estimate=self.sim.pose,
        )
        return obs, info

    # ------------------------------------------------------------------
    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict, float, bool, bool, dict]:
        tool = int(action.get("tool", TOOL_NONE))
        if tool == TOOL_RESET_POSE:
            self.sim.reset()
            result = None
            capture = None
        else:
            result = self.sim.command(
                advance_mm=float(action.get("advance_mm", 0.0)),
                roll_deg=float(action.get("roll_deg", 0.0)),
                kappa1_target=float(action.get("kappa1", 0.0)),
                phi1_target=float(action.get("phi1", 0.0)),
                kappa2_target=float(action.get("kappa2", 0.0)),
                phi2_target=float(action.get("phi2", 0.0)),
            )
            capture = None
            if tool == TOOL_BASKET:
                capture = self.sim.attempt_capture(ToolMode.BASKET)
            elif tool == TOOL_LASER:
                capture = self.sim.attempt_capture(ToolMode.LASER)

        # Reward: small step cost + capture bonus + collision penalty.
        reward = -0.01
        if result is not None and result.collided:
            reward -= 1.0
        if capture is not None and capture.success:
            reward += 5.0

        obs, info = self._observe(
            buckled=bool(result.buckled) if result else False,
            collided=bool(result.collided) if result else False,
            contact_force=float(result.contact_force_norm) if result else 0.0,
            wall_clearance_mm=float(result.wall_clearance_mm) if result else 0.0,
            actual_advance_mm=float(result.actual_advance_mm) if result else 0.0,
            actual_roll_deg=float(result.actual_roll_deg) if result else 0.0,
            tip_pose_estimate=result.tip_pose_estimate if result else self.sim.pose,
        )
        if capture is not None:
            info["capture"] = {
                "success": capture.success,
                "mode": capture.mode.value,
                "stone_id": capture.stone_id,
                "stone_size_mm": capture.stone_size_mm,
                "failure_reason": capture.failure_reason,
            }
        if result is not None:
            info["wall_clearance_mm"] = float(result.wall_clearance_mm)

        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        terminated = False
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _observe(
        self,
        buckled: bool,
        collided: bool,
        contact_force: float,
        wall_clearance_mm: float,
        actual_advance_mm: float,
        actual_roll_deg: float,
        tip_pose_estimate: np.ndarray,
    ) -> tuple[dict, dict]:
        rendered = self.sim.render(with_depth=False, with_stones_visible=True)
        rgb = rendered["rgb"]
        self._last_obs_rgb = rgb
        obs = {
            "rgb": rgb,
            "wall_clearance_mm": np.float32(wall_clearance_mm),
            "contact_force": np.float32(contact_force),
            "buckled": int(buckled),
            "collided": int(collided),
            "tip_pose_estimate": np.asarray(tip_pose_estimate, dtype=np.float32),
            "kappa1": np.float32(self.sim.kappa1),
            "phi1": np.float32(self.sim.phi1),
            "kappa2": np.float32(self.sim.kappa2),
            "phi2": np.float32(self.sim.phi2),
            "actual_advance_mm": np.float32(actual_advance_mm),
            "actual_roll_deg": np.float32(actual_roll_deg),
        }
        info = {
            "true_pose": rendered["pose"],
            "true_nearest_wall_mm": rendered["true_nearest_wall_mm"],
            "current_tree_node": rendered["current_tree_node"],
            "current_tree_progress": rendered["current_tree_progress"],
            "stones_visible": rendered["stones_visible"],
            "tool_visual_state": self.sim.tool_visual_state,
        }
        return obs, info

    # ------------------------------------------------------------------
    def render(self) -> np.ndarray | None:
        if self._last_obs_rgb is None:
            rendered = self.sim.render(with_depth=False)
            self._last_obs_rgb = rendered["rgb"]
        return self._last_obs_rgb

    def close(self) -> None:
        # Renderer holds GL resources; release if available.
        rel = getattr(self.sim.renderer, "release", None)
        if callable(rel):
            try:
                rel()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["EndonavEnv", "TOOL_NONE", "TOOL_BASKET", "TOOL_LASER", "TOOL_RESET_POSE"]
