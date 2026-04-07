"""Validate the ureteroscope kinematic model: roll + deflection composition.

Renders a 3x4 grid (deflection rows x roll columns) at a stable viewpoint
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

DEFLECTIONS_DEG = [0.0, 30.0, 60.0]
ROLLS_DEG = [0.0, 90.0, 180.0, 270.0]


def view_dir(sim: KidneySimulator) -> np.ndarray:
    return -sim.pose[:3, 2]


def main() -> None:
    sim = KidneySimulator(anatomy_params=AnatomyParams(seed=0), realistic=False)

    # Stable viewpoint with forward clearance: mid-pelvis.
    sim.follow_skeleton("pelvis", 0.4)
    base_tangent = view_dir(sim).copy()

    # 1. deflection=0 -> view_dir == tangent regardless of roll.
    for r in ROLLS_DEG:
        sim.follow_skeleton("pelvis", 0.4)
        sim.command(roll_deg=r, deflection_deg=0.0)
        d = view_dir(sim)
        assert np.allclose(d, base_tangent, atol=1e-9), (
            f"roll {r} with zero deflection changed view_dir: {d} vs {base_tangent}"
        )

    # 2 & 3. deflection=30, roll=0 vs roll=90 should differ by 90deg around tangent.
    sim.follow_skeleton("pelvis", 0.4)
    sim.command(roll_deg=0.0, deflection_deg=30.0)
    v0 = view_dir(sim).copy()
    sim.follow_skeleton("pelvis", 0.4)
    sim.command(roll_deg=90.0, deflection_deg=30.0)
    v90 = view_dir(sim).copy()
    a0 = np.arccos(np.clip(np.dot(v0, base_tangent), -1, 1))
    a90 = np.arccos(np.clip(np.dot(v90, base_tangent), -1, 1))
    assert abs(a0 - np.deg2rad(30)) < 1e-6, f"deflection angle 0: {np.rad2deg(a0)}"
    assert abs(a90 - np.deg2rad(30)) < 1e-6, f"deflection angle 90: {np.rad2deg(a90)}"

    # 4. Sweeping roll 0..360 with constant deflection traces a cone.
    angles_to_tan = []
    for r in np.linspace(0, 360, 13):
        sim.follow_skeleton("pelvis", 0.4)
        sim.command(roll_deg=float(r), deflection_deg=30.0)
        d = view_dir(sim)
        angles_to_tan.append(np.arccos(np.clip(np.dot(d, base_tangent), -1, 1)))
    angles_to_tan = np.array(angles_to_tan)
    assert np.allclose(angles_to_tan, np.deg2rad(30), atol=1e-6), (
        f"cone broken: {np.rad2deg(angles_to_tan)}"
    )

    # 5. Advance follows skeleton path (roll/deflection don't move base).
    sim.follow_skeleton("pelvis", 0.2)
    p_before, _ = sim._sample_path(sim.path_node, sim.path_progress)
    sim.command(roll_deg=45.0, deflection_deg=20.0)
    p_after_aim, _ = sim._sample_path(sim.path_node, sim.path_progress)
    assert np.allclose(p_before, p_after_aim), "aiming moved the base position"
    sim.command(advance_mm=2.0)
    p_after_adv, _ = sim._sample_path(sim.path_node, sim.path_progress)
    assert not np.allclose(p_before, p_after_adv), "advance did not move base position"

    print("All kinematic assertions passed.")

    tiles: list[np.ndarray] = []
    for defl in DEFLECTIONS_DEG:
        for roll in ROLLS_DEG:
            sim.follow_skeleton("pelvis", 0.4)
            sim.command(roll_deg=roll, deflection_deg=defl)
            tiles.append(sim.render()["rgb"])
            print(f"defl={defl:5.1f} roll={roll:6.1f}  ok")

    rows, cols = len(DEFLECTIONS_DEG), len(ROLLS_DEG)
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
