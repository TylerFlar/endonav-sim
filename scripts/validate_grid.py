"""Render the 9 anatomically interesting viewpoints into a 3x3 PNG grid."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from endonav_sim.sim.simulator import KidneySimulator

OUT = Path("grid.png")

VIEWS = [
    ("ureter distal", "ureter_distal", 0.50),
    ("ureter iliac", "ureter_iliac", 0.50),
    ("UPJ", "ureter_upj", 0.80),
    ("pelvis (2-major)", "pelvis", 0.95),
    ("upper major", "major_upper", 0.85),
    ("lower major", "major_lower", 0.85),
    ("calyx upper #1", "calyx_u1", 0.55),
    ("calyx lower #2", "calyx_l2", 0.55),
    ("calyx lower #3", "calyx_l3", 0.55),
]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    out[:18, : 8 * len(text) + 4] = (0, 0, 0)
    # Lazy: skip actual text rasterization, rely on filename grid + console.
    return out


def main() -> None:
    sim = KidneySimulator()
    tiles = []
    for name, node, t in VIEWS:
        sim.follow_skeleton(node, t)
        out = sim.render()
        print(f"{name:22s} node={out['current_tree_node']:12s} near={out['nearest_wall_mm']:.2f}mm")
        tiles.append(out["rgb"])
    h, w = tiles[0].shape[:2]
    grid = np.zeros((3 * h, 3 * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 3)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    iio.imwrite(OUT, grid)
    print(f"wrote {OUT} ({3 * w}x{3 * h})")
    print("Tile order (row-major):")
    for i, (name, _, _) in enumerate(VIEWS):
        r, c = divmod(i, 3)
        print(f"  ({r},{c}) {name}")


if __name__ == "__main__":
    main()
