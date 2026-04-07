"""Brightness vs distance: place the camera facing a wall at several
distances and verify the mean response follows ~1/r^2 (modulo Reinhard
tonemap saturation at very small r)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from endonav_sim.simulator import KidneySimulator, _pose_from_forward

OUT = Path("brightness_falloff.png")
DISTANCES_MM = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]


def main() -> None:
    sim = KidneySimulator()

    # Use a minor calyx terminal centerline as the reference. The papilla
    # is offset ~85% of the calyx radius perpendicular to the axis, so the
    # central optical-axis ray hits the un-papilla'd back wall directly.
    target_node = "calyx_l2"
    sim.follow_skeleton(target_node, 1.0)
    end_sample = sim.skel[target_node][-1]
    wall_pos = end_sample.pos
    forward = end_sample.tangent
    # Camera sits along -forward from the wall by `r` and looks at the wall.
    means = []
    centers = []
    for r in DISTANCES_MM:
        cam_pos = wall_pos - forward * r
        sim.pose = _pose_from_forward(cam_pos, forward)
        rgb = sim.render()["rgb"]
        # Sample a small central patch (avoid edges seeing other walls).
        H, W = rgb.shape[:2]
        cy, cx = H // 2, W // 2
        patch = rgb[cy - 8 : cy + 8, cx - 8 : cx + 8]
        m = patch.mean()
        means.append(m)
        centers.append(r)
        print(f"r={r:5.1f} mm  mean={m:.1f}")

    means = np.asarray(means)
    centers = np.asarray(centers)

    # Reference 1/r^2 curve, normalised to the brightest non-saturated point.
    # Saturation makes very-near samples flat; pick a mid sample as the anchor.
    anchor_idx = len(centers) // 2
    ref = means[anchor_idx] * (centers[anchor_idx] ** 2) / (centers**2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(centers, means, "o-", label="rendered mean brightness")
    ax.plot(centers, ref, "--", label=f"1/r^2 (anchored at r={centers[anchor_idx]:.0f}mm)")
    ax.set_xlabel("distance from wall (mm)")
    ax.set_ylabel("mean pixel brightness (0-255)")
    ax.set_title("Coaxial-light brightness falloff")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
