"""Generate 4 procedural kidneys with stones and render preview frames.

Drives the simulator at fixed seeds 0..3, prints anatomy + stone metadata,
collects a single rendered frame from each, then arranges them into a 2x2
contact sheet at ``artifacts/procedural_kidneys.png``.
"""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from endonav_sim import (
    AnatomyParams,
    KidneySimulator,
    StoneParams,
)

OUT = Path("artifacts/procedural_kidneys.png")
SEEDS = [0, 1, 2, 3]


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    """Burn a 1-line caption into the top-left of an RGB frame."""
    import cv2

    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tiles: list[np.ndarray] = []

    for seed in SEEDS:
        t0 = time.time()
        sim = KidneySimulator(
            anatomy_params=AnatomyParams(seed=seed),
            stone_params=StoneParams(seed=seed),
            seed=seed,
            realistic=True,
        )
        meta = sim.anatomy_meta
        n_upper = meta.realized_branch_counts["major_upper"]
        n_lower = meta.realized_branch_counts["major_lower"]
        ipa = meta.infundibulopelvic_angle_deg
        n_stones = len(sim.stones)
        sizes = [round(2 * s.radius, 1) for s in sim.stones]
        comps = sorted({s.composition for s in sim.stones})
        print(
            f"\n[seed {seed}]  build={time.time() - t0:.1f}s  "
            f"upper={n_upper}  lower={n_lower}  IPA={ipa:.0f} deg  "
            f"stones={n_stones}  sizes_mm={sizes}  compositions={comps}"
        )

        # Teleport the camera into the *infundibulum* feeding a stone-bearing
        # calyx, looking forward into the calyx. follow_skeleton lands us on
        # the centerline (with zero deflection / roll) so the view is well
        # inside the lumen, never clipped against a calyx back wall.
        stones_in_calyces = [s for s in sim.stones if s.node_id.startswith("calyx_")]
        if stones_in_calyces:
            target = stones_in_calyces[0]
            host_calyx = target.node_id
            host_infundibulum = sim.tree[host_calyx]["parent"]
            sim.follow_skeleton(host_infundibulum, 0.75)
        elif sim.stones:
            sim.follow_skeleton(sim.stones[0].node_id, 0.20)
        else:
            cal = meta.calyx_node_ids[0]
            sim.follow_skeleton(sim.tree[cal]["parent"], 0.75)
        rgb = sim.render()["rgb"]
        caption = (
            f"seed={seed}  upper={n_upper}/lower={n_lower}  IPA={ipa:.0f} deg  stones={n_stones}"
        )
        tiles.append(_label(rgb, caption))

    # 2x2 grid
    h, w = tiles[0].shape[:2]
    grid = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, 2)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    iio.imwrite(OUT, grid)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
