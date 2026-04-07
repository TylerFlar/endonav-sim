"""Proximity / safety-reflex perception.

Combines a 3x3 brightness grid (closest wall is brightest under coaxial light)
with a Farneback optical-flow expansion estimate to flag impending collisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ProximityResult:
    danger_score: float
    danger_direction_polar: tuple[float, float]
    flow_magnitude: float
    flow_expansion: float
    is_dangerous: bool


class ProximityDetector:
    def __init__(self, grid: tuple[int, int] = (3, 3), danger_threshold: float = 0.7) -> None:
        self.grid = grid
        self.danger_threshold = float(danger_threshold)
        self._prev_gray: np.ndarray | None = None

    def process(self, frame: np.ndarray, cumulative_roll: float = 0.0) -> ProximityResult:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        H, W = gray.shape[:2]
        gh, gw = self.grid

        # ----- 3x3 (or NxM) brightness grid -----
        cell_h = H // gh
        cell_w = W // gw
        cell_means = np.zeros((gh, gw), dtype=np.float32)
        cell_centers = np.zeros((gh, gw, 2), dtype=np.float32)
        for i in range(gh):
            for j in range(gw):
                y0 = i * cell_h
                y1 = (i + 1) * cell_h if i < gh - 1 else H
                x0 = j * cell_w
                x1 = (j + 1) * cell_w if j < gw - 1 else W
                cell_means[i, j] = float(gray[y0:y1, x0:x1].mean())
                cell_centers[i, j] = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]

        max_idx = np.unravel_index(int(np.argmax(cell_means)), cell_means.shape)
        max_val = float(cell_means[max_idx])
        danger_score = max_val / 255.0

        cx_img = W / 2.0
        cy_img = H / 2.0
        radius_norm = min(H, W) / 2.0
        bcx, bcy = cell_centers[max_idx]
        dx = float(bcx - cx_img)
        dy = float(bcy - cy_img)
        angle_image = float(np.arctan2(-dy, dx))
        magnitude_norm = float(np.hypot(dx, dy) / radius_norm)
        danger_direction_polar = (angle_image + float(cumulative_roll), magnitude_norm)

        # ----- Farneback optical flow -----
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            flow_magnitude = 0.0
            flow_expansion = 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            fx = flow[..., 0]
            fy = flow[..., 1]
            flow_magnitude = float(np.sqrt(fx * fx + fy * fy).mean())

            ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
            rx = xs - cx_img
            ry = ys - cy_img
            r = np.sqrt(rx * rx + ry * ry) + 1e-6
            rx /= r
            ry /= r
            flow_expansion = float((fx * rx + fy * ry).mean())

        self._prev_gray = gray

        is_dangerous = danger_score >= self.danger_threshold

        return ProximityResult(
            danger_score=danger_score,
            danger_direction_polar=danger_direction_polar,
            flow_magnitude=flow_magnitude,
            flow_expansion=flow_expansion,
            is_dangerous=is_dangerous,
        )
