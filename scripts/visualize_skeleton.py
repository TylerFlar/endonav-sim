"""Render a 3D matplotlib visualization of the procedural skeleton.

Usage:
    python scripts/visualize_skeleton.py [--seed N] [--variant A1|A2|B1|B2] [--no-stones]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endonav_sim import AnatomyParams, StoneParams, generate_anatomy, generate_stones
from endonav_sim.mesh_gen import build_mesh
from endonav_sim.skeleton import build_skeleton, dfs_order

STONE_COMP_COLORS = {
    "calcium_oxalate": "#dcd6c0",
    "uric_acid": "#c89858",
    "struvite": "#d0c8a8",
    "cystine": "#e8dca0",
}


def _color_for(name: str) -> str:
    if name.startswith("ureter"):
        return "#888888"
    if name == "pelvis":
        return "#ffd866"
    if name.startswith("major_upper"):
        return "#66c8ff"
    if name.startswith("major_lower"):
        return "#ff78c8"
    if name.startswith("major_middle"):
        return "#7ac060"
    if name.startswith("minf_upper") or name.startswith("calyx_upper"):
        return "#3ca0dc"
    if name.startswith("minf_lower") or name.startswith("calyx_lower"):
        return "#dc6c8c"
    if name.startswith("minf_middle") or name.startswith("calyx_middle"):
        return "#5aa040"
    return "#aaaaaa"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0, help="Procedural anatomy seed")
    p.add_argument(
        "--variant",
        choices=["A1", "A2", "B1", "B2"],
        default="A1",
        help="Sampaio pelvicalyceal variant",
    )
    p.add_argument("--no-stones", action="store_true", help="Skip stone overlay")
    p.add_argument("--out", type=Path, default=Path("artifacts/skeleton_overlay.png"))
    args = p.parse_args()

    tree, meta = generate_anatomy(AnatomyParams(seed=args.seed, variant=args.variant))
    skel = build_skeleton(tree)
    stones = [] if args.no_stones else generate_stones(skel, meta, StoneParams(seed=args.seed))

    title = (
        f"procedural anatomy {args.variant} seed={args.seed}  "
        f"({meta.n_dead_ends} calyces, IPA={meta.infundibulopelvic_angle_deg:.0f} deg)"
    )

    mesh = build_mesh(skel, tree=tree, stones=stones).mesh

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    v = mesh.vertices
    idx = np.random.default_rng(0).choice(len(v), size=min(8000, len(v)), replace=False)
    ax.scatter(v[idx, 0], v[idx, 1], v[idx, 2], s=1, c="#cc9090", alpha=0.08)

    for name in dfs_order(tree):
        pts = np.stack([s.pos for s in skel[name]])
        c = _color_for(name)
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color=c, linewidth=2.5)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, color=c)

    if stones:
        for s in stones:
            ax.scatter(
                [s.center[0]],
                [s.center[1]],
                [s.center[2]],
                s=80 * (s.radius**1.5),
                color=STONE_COMP_COLORS.get(s.composition, "#dddddd"),
                edgecolors="black",
                linewidths=0.5,
                depthshade=False,
            )
        ax.text2D(
            0.02,
            0.02,
            f"stones: {len(stones)}",
            transform=ax.transAxes,
            fontsize=10,
            color="black",
        )

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(title)
    bounds = mesh.bounds
    ranges = bounds[1] - bounds[0]
    mid = bounds.mean(axis=0)
    half = ranges.max() / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
