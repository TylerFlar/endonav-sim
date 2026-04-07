"""Stub junction detector at multiple anatomical bifurcations.

The Sampaio Type A1 anatomy has *two* junction levels:
  1. Pelvis -> upper + lower major calyces  (expect >= 2 blobs)
  2. Each major -> 3 minor infundibula      (expect >= 3 blobs)

We threshold low-brightness regions inside the endoscope aperture, run
connected components, and check the count at each viewpoint."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy import ndimage

from endonav_sim.sim.simulator import KidneySimulator

OUT = Path("junction_detection.png")
DARK_THRESHOLD = 20
MIN_BLOB_AREA = 60

TESTS = [
    ("pelvis bifurcation", "pelvis", 0.92, 2),
    ("upper major junction", "major_upper", 0.92, 3),
    ("lower major junction", "major_lower", 0.92, 3),
]


def detect(rgb: np.ndarray) -> tuple[np.ndarray, list[int]]:
    gray = rgb.mean(axis=2)
    H, W = gray.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    radius = min(H, W) / 2.0 * 0.93
    aperture = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
    mask = (gray < DARK_THRESHOLD) & aperture
    labels, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= MIN_BLOB_AREA]
    return labels, keep


def main() -> None:
    sim = KidneySimulator()
    panels = []
    all_pass = True
    for name, node, t, expected in TESTS:
        sim.follow_skeleton(node, t)
        rgb = sim.render()["rgb"]
        labels, keep = detect(rgb)
        ok = len(keep) >= expected
        all_pass &= ok
        print(
            f"{name:24s} node={node:12s} blobs={len(keep)} expected>={expected}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
        overlay = rgb.copy()
        for lab in keep:
            edge = ndimage.binary_dilation(labels == lab) & ~(labels == lab)
            overlay[edge] = (0, 255, 0)
        panels.append(np.concatenate([rgb, overlay], axis=1))

    grid = np.concatenate(panels, axis=0)
    iio.imwrite(OUT, grid)
    print(f"wrote {OUT}")
    print("OVERALL:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
