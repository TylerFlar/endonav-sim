"""Render the 9 anatomically interesting viewpoints into a 3x3 PNG grid."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from endonav_sim import AnatomyParams, KidneySimulator

OUT = Path("artifacts/grid.png")


def main() -> None:
    sim = KidneySimulator(anatomy_params=AnatomyParams(seed=0), realistic=False)
    meta = sim.anatomy_meta
    upper_calyces = [c for c in meta.calyx_node_ids if c.startswith("calyx_upper")]
    lower_calyces = [c for c in meta.calyx_node_ids if c.startswith("calyx_lower")]

    views = [
        ("ureter distal", "ureter_distal", 0.50),
        ("ureter iliac", "ureter_iliac", 0.50),
        ("UPJ", "ureter_upj", 0.80),
        ("pelvis", "pelvis", 0.95),
        ("upper major", "major_upper", 0.85),
        ("lower major", "major_lower", 0.85),
        ("upper calyx", upper_calyces[0], 0.55),
        ("lower calyx #1", lower_calyces[0], 0.55),
        ("lower calyx #2", lower_calyces[min(1, len(lower_calyces) - 1)], 0.55),
    ]

    tiles = []
    for name, node, t in views:
        sim.follow_skeleton(node, t)
        out = sim.render()
        print(
            f"{name:18s} node={out['current_tree_node']:18s} near={out['nearest_wall_mm']:.2f} mm"
        )
        tiles.append(out["rgb"])

    h, w = tiles[0].shape[:2]
    grid = np.zeros((3 * h, 3 * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 3)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile

    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, grid)
    print(f"\nwrote {OUT} ({3 * w}x{3 * h})")


if __name__ == "__main__":
    main()
