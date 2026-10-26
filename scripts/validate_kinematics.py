"""Validate the ureteroscope kinematic model: roll + 2-section PCC bending.

Renders a 3x4 grid (curvature rows x roll columns) at a stable viewpoint
in the renal pelvis, plus runs assertions on the kinematic invariants.
Uses ``realistic=False`` so the underlying clean kinematic path is tested
without dynamics noise.
"""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from endonav_sim import AnatomyParams, KidneySimulator

OUT = Path("artifacts/validate_kinematics.png")

# Curvature targets for the distal tip section (1/mm).
# kappa * L2(5mm) = bend angle in radians.
# 0.1047 → ~30°,  0.2094 → ~60°
KAPPAS = [0.0, 0.1047, 0.2094]
ROLLS_DEG = [0.0, 90.0, 180.0, 270.0]
# Default section 2 length (mm) for angle conversion.
L2 = 5.0


def view_dir(sim: KidneySimulator) -> np.ndarray:
    return -sim.pose[:3, 2]


def main() -> None:
    sim = KidneySimulator(anatomy_params=AnatomyParams(seed=0), realistic=False)

    # Stable viewpoint with forward clearance: mid-pelvis.
    sim.follow_skeleton("pelvis", 0.4)
    base_tangent = view_dir(sim).copy()

    # 1. kappa=0 -> view_dir == tangent regardless of roll.
    for r in ROLLS_DEG:
        sim.follow_skeleton("pelvis", 0.4)
        sim.command(roll_deg=r, kappa2_target=0.0)
        d = view_dir(sim)
        assert np.allclose(d, base_tangent, atol=1e-6), (
            f"roll {r} with zero curvature changed view_dir: {d} vs {base_tangent}"
        )

    # 2 & 3. Same curvature at roll=0 vs roll=90 should produce the same
    # angular offset from the base tangent (just in different bending planes).
    k30 = np.deg2rad(30.0) / L2  # curvature for ~30° bend
    sim.follow_skeleton("pelvis", 0.4)
    sim.command(roll_deg=0.0, kappa2_target=k30)
    v0 = view_dir(sim).copy()
    sim.follow_skeleton("pelvis", 0.4)
    sim.command(roll_deg=90.0, kappa2_target=k30)
    v90 = view_dir(sim).copy()
    a0 = np.arccos(np.clip(np.dot(v0, base_tangent), -1, 1))
    a90 = np.arccos(np.clip(np.dot(v90, base_tangent), -1, 1))
    assert abs(a0 - np.deg2rad(30)) < 0.02, f"bend angle 0: {np.rad2deg(a0):.2f}"
    assert abs(a90 - np.deg2rad(30)) < 0.02, f"bend angle 90: {np.rad2deg(a90):.2f}"

    # 4. Sweeping roll 0..360 with constant curvature traces a cone.
    angles_to_tan = []
    for r in np.linspace(0, 360, 13):
        sim.follow_skeleton("pelvis", 0.4)
        sim.command(roll_deg=float(r), kappa2_target=k30)
        d = view_dir(sim)
        angles_to_tan.append(np.arccos(np.clip(np.dot(d, base_tangent), -1, 1)))
    angles_to_tan = np.array(angles_to_tan)
    assert np.allclose(angles_to_tan, np.deg2rad(30), atol=0.02), (
        f"cone broken: {np.rad2deg(angles_to_tan)}"
    )

    # 5. Advance follows skeleton path (bending doesn't move base position).
    sim.follow_skeleton("pelvis", 0.2)
    p_before, _ = sim._sample_path(sim.path_node, sim.path_progress)
    sim.command(roll_deg=45.0, kappa2_target=0.02)
    p_after_aim, _ = sim._sample_path(sim.path_node, sim.path_progress)
    assert np.allclose(p_before, p_after_aim), "aiming moved the base position"
    sim.command(advance_mm=2.0)
    p_after_adv, _ = sim._sample_path(sim.path_node, sim.path_progress)
    assert not np.allclose(p_before, p_after_adv), "advance did not move base position"

    print("All kinematic assertions passed.")

    tiles: list[np.ndarray] = []
    for kappa in KAPPAS:
        for roll in ROLLS_DEG:
            sim.follow_skeleton("pelvis", 0.4)
            sim.command(roll_deg=roll, kappa2_target=kappa)
            tiles.append(sim.render()["rgb"])
            bend_deg = np.rad2deg(kappa * L2)
            print(f"bend={bend_deg:5.1f}deg roll={roll:6.1f}  ok")

    rows, cols = len(KAPPAS), len(ROLLS_DEG)
    h, w = tiles[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile

    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, grid)
    print(f"wrote {OUT} ({cols * w}x{rows * h})")


if __name__ == "__main__":
    main()
