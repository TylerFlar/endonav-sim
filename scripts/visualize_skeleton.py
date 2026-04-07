"""Render a 3D matplotlib visualization of the skeleton + mesh wireframe."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endonav_sim.mesh_gen import build_mesh
from endonav_sim.skeleton import build_skeleton
from endonav_sim.tree import dfs_order

OUT = Path("skeleton_overlay.png")

NODE_COLORS = {
    # ureter sub-segments — greys
    "ureter_distal": "#bbbbbb",
    "ureter_iliac": "#999999",
    "ureter_proximal": "#777777",
    "ureter_upj": "#555555",
    "pelvis": "#ffd866",
    # major calyces — saturated
    "major_upper": "#66c8ff",
    "major_lower": "#ff78c8",
    # upper-pole minor inf + calyces — blues/cyans
    "minf_u1": "#3ca0dc",
    "calyx_u1": "#1e6cb0",
    "minf_u2": "#5acdcd",
    "calyx_u2": "#208080",
    "minf_u3": "#7e8cdc",
    "calyx_u3": "#3848b0",
    # lower-pole minor inf + calyces — pinks/reds
    "minf_l1": "#dc6c8c",
    "calyx_l1": "#b02858",
    "minf_l2": "#dca06c",
    "calyx_l2": "#b06028",
    "minf_l3": "#dc3ca0",
    "calyx_l3": "#a01878",
}


def main() -> None:
    skel = build_skeleton()
    mesh = build_mesh(skel).mesh

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample mesh vertices as a translucent point cloud (proxy wireframe).
    v = mesh.vertices
    idx = np.random.default_rng(0).choice(len(v), size=min(8000, len(v)), replace=False)
    ax.scatter(v[idx, 0], v[idx, 1], v[idx, 2], s=1, c="#cc9090", alpha=0.08)

    for name in dfs_order():
        pts = np.stack([s.pos for s in skel[name]])
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2], "-", color=NODE_COLORS[name], linewidth=3, label=name
        )
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10, color=NODE_COLORS[name])

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Kidney collecting system: skeleton over wall point cloud")
    ax.legend(loc="upper left", fontsize=8)
    # Equal aspect.
    bounds = mesh.bounds
    ranges = bounds[1] - bounds[0]
    mid = bounds.mean(axis=0)
    half = ranges.max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
