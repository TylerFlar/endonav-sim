"""Physical visual effects produced by the kidney simulator.

These overlays represent things that happen IN THE WORLD: the laser firing,
smoke from a fragmentation event, the basket extending into view, and the
small cardiac/respiratory micro-motion of the camera relative to the kidney.

They live in the sim package — not in the agent — because they describe the
sim's own physical state, not the controller's metadata. An external display
layer (e.g. ``endonav-agent``'s realtime renderer) reads
``KidneySimulator.tool_visual_state`` and calls ``compose_visuals(...)`` to
get a cinematic frame.

NOTE: ``KidneySimulator.render()`` itself does NOT apply these effects — it
returns a clean rasterized frame so existing perception/test code is
unaffected. Effects are opt-in via this module.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

# ----------------------------------------------------------------------
# Image geometry (must match CoaxialRenderer defaults)
# ----------------------------------------------------------------------
IMG_W: int = 1024
IMG_H: int = 768
CX: int = IMG_W // 2  # 512
CY: int = IMG_H // 2  # 384


# ----------------------------------------------------------------------
# Tool visual state
# ----------------------------------------------------------------------
@dataclass
class ToolVisualState:
    """Snapshot of the sim's tool-effect timeline.

    Stored on KidneySimulator and updated whenever ``attempt_capture`` is
    invoked. The agent's display layer reads this via the public
    ``tool_visual_state`` property and passes it to ``compose_visuals``.

    Times are ``time.monotonic()`` epochs. ``None`` means the corresponding
    tool has never fired.
    """

    laser_kind: str | None = None  # "hit" | "dust" | "miss"
    laser_ts: float | None = None
    last_laser_stone_id: str | None = None
    last_laser_composition: str | None = None

    basket_kind: str | None = None  # "success" | "use"
    basket_ts: float | None = None
    last_basket_stone_id: str | None = None

    def note_laser(
        self,
        kind: str,
        stone_id: str | None = None,
        composition: str | None = None,
    ) -> None:
        self.laser_kind = kind
        self.laser_ts = time.monotonic()
        self.last_laser_stone_id = stone_id
        self.last_laser_composition = composition

    def note_basket(
        self,
        kind: str,
        stone_id: str | None = None,
    ) -> None:
        self.basket_kind = kind
        self.basket_ts = time.monotonic()
        self.last_basket_stone_id = stone_id

    def snapshot(self) -> "ToolVisualState":
        """Return a copy safe to pass across thread/process boundaries."""
        return ToolVisualState(
            laser_kind=self.laser_kind,
            laser_ts=self.laser_ts,
            last_laser_stone_id=self.last_laser_stone_id,
            last_laser_composition=self.last_laser_composition,
            basket_kind=self.basket_kind,
            basket_ts=self.basket_ts,
            last_basket_stone_id=self.last_basket_stone_id,
        )


# ----------------------------------------------------------------------
# Precomputed assets
# ----------------------------------------------------------------------
def _build_smoke_kernel(size: int = 256) -> np.ndarray:
    """Soft radial gradient (1.0 center → 0 edge) with mild noise.

    Multiplied by an age-dependent alpha at draw time. Stored as float32 in
    [0, 1] so the per-frame compositor can scale + alpha-blend cheaply.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rmax = size / 2.0
    falloff = np.clip(1.0 - r / rmax, 0.0, 1.0)
    falloff = np.sqrt(falloff)
    rng = np.random.default_rng(0)
    noise = rng.uniform(0.85, 1.0, size=(size, size)).astype(np.float32)
    falloff = falloff * noise
    falloff = cv2.GaussianBlur(falloff, (15, 15), 0)
    falloff /= max(float(falloff.max()), 1e-6)
    return falloff


_SMOKE_KERNEL: np.ndarray = _build_smoke_kernel(256)
_SMOKE_BASE_COLOR = np.array([235, 230, 220], dtype=np.float32)  # warm off-white


# ----------------------------------------------------------------------
# Camera micro-motion (cardiac / respiratory jitter)
# ----------------------------------------------------------------------
def apply_jitter(
    rgb: np.ndarray,
    rng: np.random.Generator,
    intensity_px: float = 1.5,
) -> np.ndarray:
    """Small 2D translation simulating patient motion against the static scope.

    Returns a NEW array. Cheap (one warpAffine call). The patient's
    cardiac/respiratory cycle moves the kidney relative to the camera by a
    fraction of a millimeter per frame; in image space that's a few pixels
    of translation. Doesn't change scale or rotation.
    """
    dx = float(rng.normal(0.0, intensity_px))
    dy = float(rng.normal(0.0, intensity_px))
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        rgb,
        M,
        (IMG_W, IMG_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ----------------------------------------------------------------------
# Tool overlays
# ----------------------------------------------------------------------
def draw_laser_beam(
    rgb: np.ndarray,
    age_s: float,
    *,
    max_age: float = 0.20,
    anchor_xy: tuple[int, int] | None = None,
) -> None:
    """Bright cyan flash + fiber-tip glint at the image center."""
    if age_s > max_age:
        return
    cx, cy = anchor_xy if anchor_xy is not None else (CX, CY)
    fade = 1.0 - (age_s / max_age)
    fade = max(0.0, min(1.0, fade)) ** 0.6  # ease-out

    halo = np.zeros_like(rgb)
    cv2.circle(halo, (cx, cy), 60, (60, 220, 220), -1, lineType=cv2.LINE_AA)
    cv2.circle(halo, (cx, cy), 30, (180, 255, 255), -1, lineType=cv2.LINE_AA)
    halo = cv2.GaussianBlur(halo, (51, 51), 0)
    cv2.addWeighted(rgb, 1.0, halo, 0.85 * fade, 0, dst=rgb)

    core_radius = max(3, int(8 * fade))
    cv2.circle(rgb, (cx, cy), core_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(rgb, (cx, cy), core_radius + 2, (200, 255, 255), 2, lineType=cv2.LINE_AA)


def draw_smoke(
    rgb: np.ndarray,
    age_s: float,
    *,
    max_age: float = 1.5,
    anchor_xy: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> None:
    """Expanding off-white puff anchored at the image center."""
    if age_s < 0 or age_s > max_age:
        return
    cx, cy = anchor_xy if anchor_xy is not None else (CX, CY)
    t = age_s / max_age  # 0..1
    base_size = int(180 * scale)
    sz = int(base_size * (0.6 + 0.8 * t))
    sz = max(8, sz)
    alpha = 0.55 * (1.0 - t) ** 1.4
    if alpha <= 0.005:
        return
    kernel = cv2.resize(_SMOKE_KERNEL, (sz, sz), interpolation=cv2.INTER_LINEAR)
    x0 = cx - sz // 2
    y0 = cy - sz // 2
    x1 = x0 + sz
    y1 = y0 + sz
    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = min(IMG_W, x1)
    fy1 = min(IMG_H, y1)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    kx0 = fx0 - x0
    ky0 = fy0 - y0
    kx1 = kx0 + (fx1 - fx0)
    ky1 = ky0 + (fy1 - fy0)
    sub = rgb[fy0:fy1, fx0:fx1].astype(np.float32)
    a = (kernel[ky0:ky1, kx0:kx1] * alpha)[..., None]
    smoke = _SMOKE_BASE_COLOR.reshape(1, 1, 3)
    blended = sub * (1.0 - a) + smoke * a
    rgb[fy0:fy1, fx0:fx1] = np.clip(blended, 0, 255).astype(np.uint8)


def draw_basket_cage(
    rgb: np.ndarray,
    age_s: float,
    *,
    max_age: float = 0.80,
    anchor_xy: tuple[int, int] | None = None,
    success: bool = False,
) -> None:
    """Wireframe basket cage drawn slightly forward-of-center.

    The cage extends instantly when deployed (no fade-in — a real basket
    pops out of the working channel under cable tension), holds at full
    alpha for most of its lifetime, then fades out over the last third.
    On a successful capture, the cage closes (radius shrinks during the
    last half of its lifetime) and turns gold.
    """
    if age_s < 0 or age_s > max_age:
        return
    cx, cy = anchor_xy if anchor_xy is not None else (CX, CY)
    t = age_s / max_age
    # Hold full alpha for the first 60% of life, then fade out.
    if t < 0.6:
        alpha = 1.0
    else:
        alpha = max(0.0, 1.0 - (t - 0.6) / 0.4)
    base_radius = 70
    if success and t > 0.5:
        close_t = (t - 0.5) / 0.5
        radius = int(base_radius * (1.0 - 0.5 * close_t))
    else:
        radius = base_radius
    color_fg = (255, 215, 90) if success else (220, 220, 230)

    overlay = np.zeros_like(rgb)
    n_wires = 6
    for i in range(n_wires):
        ang = math.pi * i / n_wires
        axes = (radius, max(8, int(radius * 0.35)))
        cv2.ellipse(
            overlay,
            (cx, cy),
            axes,
            math.degrees(ang),
            0,
            360,
            color_fg,
            1,
            lineType=cv2.LINE_AA,
        )
    cv2.circle(overlay, (cx, cy), radius, color_fg, 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(rgb, 1.0, overlay, alpha * 0.85, 0, dst=rgb)


# ----------------------------------------------------------------------
# Top-level compositor
# ----------------------------------------------------------------------
_LASER_MAX_AGE = {
    "miss": 0.20,
    "dust": 0.80,
    "hit": 1.5,  # smoke duration > flash duration
}
_BASKET_MAX_AGE = {
    "use": 0.80,
    "success": 0.80,
}


def compose_visuals(
    canonical_rgb: np.ndarray,
    tool_state: ToolVisualState | None = None,
    *,
    jitter_rng: np.random.Generator | None = None,
    apply_motion: bool = True,
    now: float | None = None,
) -> np.ndarray:
    """Take a clean sim-rendered frame and produce the cinematic version.

    Steps (each optional via flags / state):
      1. Copy the input so the caller's array is unaffected.
      2. Apply cardiac/respiratory jitter (when ``apply_motion`` and a RNG
         is provided).
      3. Draw the active laser overlay (if recent enough).
      4. Draw the active basket overlay (if recent enough).

    The agent's HUD overlay is NOT drawn here — that's the controller's
    metadata, layered on top of the cinematic frame by the display layer.
    """
    if canonical_rgb is None:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    if apply_motion and jitter_rng is not None:
        frame = apply_jitter(canonical_rgb, jitter_rng)
    else:
        frame = canonical_rgb.copy()

    if tool_state is not None:
        cur = now if now is not None else time.monotonic()

        # Laser overlay (with optional smoke for hit/dust)
        if tool_state.laser_ts is not None and tool_state.laser_kind is not None:
            age = cur - tool_state.laser_ts
            kind = tool_state.laser_kind
            max_age = _LASER_MAX_AGE.get(kind, 0.20)
            if 0.0 <= age <= max_age:
                # Always draw the flash within its 0.20s window.
                draw_laser_beam(frame, age, max_age=0.20)
                # Smoke is conditional on a hit/dust event.
                if kind == "hit":
                    draw_smoke(frame, age, max_age=1.5, scale=1.1)
                elif kind == "dust":
                    draw_smoke(frame, age, max_age=0.80, scale=0.7)

        # Basket overlay
        if tool_state.basket_ts is not None and tool_state.basket_kind is not None:
            age = cur - tool_state.basket_ts
            kind = tool_state.basket_kind
            max_age = _BASKET_MAX_AGE.get(kind, 0.80)
            if 0.0 <= age <= max_age:
                draw_basket_cage(
                    frame, age, max_age=max_age, success=(kind == "success")
                )

    return frame


__all__ = [
    "ToolVisualState",
    "apply_jitter",
    "draw_laser_beam",
    "draw_smoke",
    "draw_basket_cage",
    "compose_visuals",
    "IMG_W",
    "IMG_H",
    "CX",
    "CY",
]
