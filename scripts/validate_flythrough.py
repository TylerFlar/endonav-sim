"""Render a fly-through video that walks the procedural skeleton in DFS order."""

from __future__ import annotations

import time
from pathlib import Path

import imageio.v2 as imageio

from endonav_sim import AnatomyParams, KidneySimulator
from endonav_sim.skeleton import dfs_order

OUT = Path("artifacts/flythrough.mp4")
FPS = 24
STEPS_PER_NODE = 30


def main() -> None:
    sim = KidneySimulator(anatomy_params=AnatomyParams(seed=0), realistic=False)
    order = dfs_order(sim.tree)
    print("DFS order:", order)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=7)
    t0 = time.time()
    n_frames = 0
    for node in order:
        for k in range(STEPS_PER_NODE):
            t = k / max(1, STEPS_PER_NODE - 1)
            sim.follow_skeleton(node, t)
            frame = sim.render()["rgb"]
            writer.append_data(frame)
            n_frames += 1
    writer.close()
    dt = time.time() - t0
    print(f"wrote {OUT} ({n_frames} frames, {n_frames / dt:.1f} fps render rate)")


if __name__ == "__main__":
    main()
