"""End-to-end validation of the perception stack against the simulator.

Runs six tests:
  1. Junction detection at pelvis (expect >=2 blobs)
  2. Junction detection at major_upper (expect >=3 blobs)
  3. Dead-end detection deep in a minor calyx
  4. Place-recognition confusion matrix across all named nodes
  5. Blob polar coords vs ground-truth branch directions at pelvis
  6. Roll invariance of place-recognition descriptors

All artifacts are written to the repo root.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

from endonav_sim.perception import (
    JunctionDetector,
    JunctionResult,
    PlaceRecognition,
    ProximityDetector,
)
from endonav_sim.sim.simulator import KidneySimulator

OUT_DIR = Path(".")

# ---------- helpers ---------------------------------------------------------


def _burn_in(detector: JunctionDetector, frame: np.ndarray, n: int = 25) -> JunctionResult:
    res = None
    for _ in range(n):
        res = detector.process(frame, cumulative_roll=0.0)
    assert res is not None
    return res


def _draw_result(rgb: np.ndarray, res: JunctionResult, title: str) -> np.ndarray:
    overlay = rgb.copy()
    mask_rgb = np.zeros_like(rgb)
    mask_rgb[..., 1] = res.dark_mask  # green tint for dark mask
    overlay = cv2.addWeighted(overlay, 1.0, mask_rgb, 0.35, 0.0)
    H, W = rgb.shape[:2]
    cx, cy = W // 2, H // 2
    for b in res.blobs:
        cv2.drawContours(overlay, [b.contour], -1, (0, 255, 0), 2)
        cv2.circle(overlay, b.centroid_xy, 6, (255, 0, 0), -1)
        cv2.line(overlay, (cx, cy), b.centroid_xy, (255, 255, 0), 2)
    cv2.putText(
        overlay,
        f"{title}: {res.classification} (n={len(res.blobs)})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


# ---------- tests -----------------------------------------------------------


def test_junctions(sim: KidneySimulator) -> tuple[bool, list[np.ndarray]]:
    cases = [
        ("pelvis bifurcation", "pelvis", 0.92, 2, "pelvis"),
        ("upper major trifurcation", "major_upper", 0.92, 3, "major"),
    ]
    panels: list[np.ndarray] = []
    all_pass = True
    for name, node, t, expected, fname in cases:
        sim.follow_skeleton(node, t)
        rgb = sim.render()["rgb"]
        det = JunctionDetector(min_blob_fraction=0.005)
        res = _burn_in(det, rgb)
        ok = res.classification == "junction" and len(res.blobs) >= expected
        all_pass &= ok
        print(
            f"[Test 1/2] {name:30s} blobs={len(res.blobs)} expected>={expected} "
            f"class={res.classification} {'PASS' if ok else 'FAIL'}"
        )
        panel = _draw_result(rgb, res, name)
        panels.append(panel)
        iio.imwrite(OUT_DIR / f"validate_perception_{fname}.png", panel)

    # Test 3: dead end
    sim.follow_skeleton("calyx_u1", 0.95)
    rgb = sim.render()["rgb"]
    det = JunctionDetector(min_blob_fraction=0.005)
    res = _burn_in(det, rgb, n=30)
    ok = res.classification == "dead_end"
    all_pass &= ok
    print(
        f"[Test 3]   minor calyx dead end       blobs={len(res.blobs)} "
        f"class={res.classification} {'PASS' if ok else 'FAIL'}"
    )
    panel = _draw_result(rgb, res, "dead end")
    panels.append(panel)
    iio.imwrite(OUT_DIR / "validate_perception_deadend.png", panel)

    return all_pass, panels


# ---------- place recognition -----------------------------------------------

PR_NODES = [
    "pelvis",
    "major_upper",
    "major_lower",
    "minf_u1",
    "minf_u2",
    "minf_u3",
    "minf_l1",
    "minf_l2",
    "minf_l3",
    "calyx_u1",
    "calyx_u2",
    "calyx_u3",
    "calyx_l1",
    "calyx_l2",
    "calyx_l3",
]


def test_place_recognition(sim: KidneySimulator) -> bool:
    pr = PlaceRecognition()
    print(f"[Test 4]   place-recognition backend = {pr.backend}")

    frames: dict[str, np.ndarray] = {}
    for node in PR_NODES:
        sim.follow_skeleton(node, 0.5)
        frames[node] = sim.render()["rgb"].copy()
        pr.add_node(node, frames[node], cumulative_roll=0.0)
    pr.finalize()

    n = len(PR_NODES)
    sim_mat = np.zeros((n, n), dtype=np.float32)
    for i, node in enumerate(PR_NODES):
        res = pr.match(frames[node], cumulative_roll=0.0)
        for j, other in enumerate(PR_NODES):
            sim_mat[i, j] = res.all_similarities.get(other, 0.0)

    diag = np.diag(sim_mat)
    off = sim_mat.copy()
    np.fill_diagonal(off, np.nan)
    diag_mean = float(np.nanmean(diag))
    off_mean = float(np.nanmean(off))
    separation = diag_mean - off_mean
    ok = separation > 0.05
    print(
        f"           diag_mean={diag_mean:.3f}  off_mean={off_mean:.3f}  "
        f"separation={separation:.3f}  {'PASS' if ok else 'FAIL'}"
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(PR_NODES, rotation=90)
    ax.set_yticklabels(PR_NODES)
    ax.set_title(
        f"Place recognition cosine similarity ({pr.backend})\n"
        f"diag={diag_mean:.3f}  off={off_mean:.3f}  sep={separation:.3f}"
    )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validate_perception_pr_confusion.png", dpi=120)
    plt.close(fig)
    return ok


# ---------- polar alignment -------------------------------------------------


def _project_world_to_image(
    point_world: np.ndarray, pose: np.ndarray, fov_y_deg: float, W: int, H: int
) -> tuple[float, float] | None:
    """Project world point to image pixel coords. Returns None if behind camera."""
    cam_to_world = pose
    world_to_cam = np.linalg.inv(cam_to_world)
    p_h = np.array([point_world[0], point_world[1], point_world[2], 1.0])
    p_cam = world_to_cam @ p_h
    x_c, y_c, z_c = p_cam[0], p_cam[1], p_cam[2]
    # Camera looks down -Z; visible points have z_c < 0.
    if z_c >= -1e-3:
        return None
    f = (H / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    u = (W / 2.0) + f * (x_c / -z_c)
    v = (H / 2.0) - f * (y_c / -z_c)
    return float(u), float(v)


def test_blob_polar(sim: KidneySimulator) -> bool:
    sim.follow_skeleton("pelvis", 0.92)
    out = sim.render()
    rgb = out["rgb"]
    pose = out["pose"]
    H, W = rgb.shape[:2]
    cx, cy = W / 2.0, H / 2.0

    det = JunctionDetector(min_blob_fraction=0.005)
    res = _burn_in(det, rgb)
    if len(res.blobs) < 2:
        print(f"[Test 5]   FAIL — only {len(res.blobs)} blob(s) detected at pelvis")
        return False

    # Ground truth: first waypoint of major_upper / major_lower in world coords.
    gt: list[tuple[str, tuple[float, float], float]] = []
    for child in ("major_upper", "major_lower"):
        wp = sim.skel[child][2].pos  # a couple of mm into the child
        proj = _project_world_to_image(wp, pose, sim.renderer.fov_y_deg, W, H)
        if proj is None:
            print(f"[Test 5]   FAIL — branch {child} not in front of camera")
            return False
        u, v = proj
        angle = float(np.arctan2(-(v - cy), u - cx))
        gt.append((child, (u, v), angle))

    # Match each blob to its closest ground-truth branch by angle.
    blobs_top2 = res.blobs[:2]
    used = set()
    errors_deg: list[float] = []
    pairs: list[tuple[tuple[int, int], tuple[float, float], float]] = []
    for b in blobs_top2:
        best_idx = -1
        best_err = 1e9
        for k, (_, _, gt_angle) in enumerate(gt):
            if k in used:
                continue
            d = abs(((b.polar_image[0] - gt_angle) + np.pi) % (2 * np.pi) - np.pi)
            if d < best_err:
                best_err = d
                best_idx = k
        used.add(best_idx)
        err_deg = float(np.rad2deg(best_err))
        errors_deg.append(err_deg)
        pairs.append((b.centroid_xy, gt[best_idx][1], err_deg))

    ok = all(e < 20.0 for e in errors_deg)
    print(
        f"[Test 5]   polar errors (deg) = {[round(e,1) for e in errors_deg]}  "
        f"{'PASS' if ok else 'FAIL'}"
    )

    overlay = rgb.copy()
    for (bxy, gxy, err) in pairs:
        cv2.arrowedLine(overlay, (int(cx), int(cy)), bxy, (255, 0, 0), 3, tipLength=0.05)
        cv2.arrowedLine(
            overlay, (int(cx), int(cy)), (int(gxy[0]), int(gxy[1])), (0, 255, 0), 3, tipLength=0.05
        )
        cv2.putText(
            overlay,
            f"{err:.1f} deg",
            (bxy[0] + 8, bxy[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        "red=detected, green=ground-truth branch",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    iio.imwrite(OUT_DIR / "validate_perception_polar.png", overlay)
    return ok


# ---------- roll invariance -------------------------------------------------


def test_roll_invariance(sim: KidneySimulator) -> bool:
    sim.follow_skeleton("major_upper", 0.5)
    pr = PlaceRecognition()
    rolls_deg = [0.0, 90.0, 180.0, 270.0]
    descriptors: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    prev = 0.0
    for r in rolls_deg:
        delta = r - prev
        if delta != 0.0:
            sim.command(roll_deg=delta)
        prev = r
        frame = sim.render()["rgb"].copy()
        frames.append(frame)
        d = pr.extract_descriptor(frame, cumulative_roll=sim.cumulative_roll)
        descriptors.append(d)

    desc = np.stack(descriptors, axis=0)
    sim_mat = desc @ desc.T
    np.fill_diagonal(sim_mat, 1.0)
    min_off = float(sim_mat[~np.eye(len(rolls_deg), dtype=bool)].min())
    ok = min_off > 0.95
    print(
        f"[Test 6]   roll-invariance min similarity = {min_off:.4f} {'PASS' if ok else 'FAIL'}"
    )

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, f, r in zip(axes[:4], frames, rolls_deg, strict=False):
        ax.imshow(f)
        ax.set_title(f"roll={r:.0f} deg")
        ax.axis("off")
    im = axes[4].imshow(sim_mat, vmin=0.8, vmax=1.0, cmap="viridis")
    axes[4].set_title(f"cosine sim (min={min_off:.3f})")
    axes[4].set_xticks(range(4))
    axes[4].set_yticks(range(4))
    axes[4].set_xticklabels([f"{r:.0f}" for r in rolls_deg])
    axes[4].set_yticklabels([f"{r:.0f}" for r in rolls_deg])
    fig.colorbar(im, ax=axes[4])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validate_perception_roll_invariance.png", dpi=120)
    plt.close(fig)
    return ok


# ---------- proximity sanity check (not in the 6 spec tests, but cheap) -----


def test_proximity_smoke(sim: KidneySimulator) -> None:
    sim.follow_skeleton("ureter_distal", 0.5)
    prox = ProximityDetector()
    f1 = sim.render()["rgb"]
    r1 = prox.process(f1, cumulative_roll=0.0)
    sim.command(advance_mm=2.0)
    f2 = sim.render()["rgb"]
    r2 = prox.process(f2, cumulative_roll=0.0)
    print(
        f"[smoke]    proximity danger={r2.danger_score:.2f} "
        f"flow_mag={r2.flow_magnitude:.2f} expansion={r2.flow_expansion:.3f}"
    )


# ---------- main ------------------------------------------------------------


def main() -> None:
    sim = KidneySimulator()
    results: dict[str, bool] = {}

    ok_junc, _ = test_junctions(sim)
    results["junctions+deadend"] = ok_junc

    results["place_recognition"] = test_place_recognition(sim)
    results["blob_polar"] = test_blob_polar(sim)
    results["roll_invariance"] = test_roll_invariance(sim)
    test_proximity_smoke(sim)

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:22s} {'PASS' if v else 'FAIL'}")
    print("OVERALL:", "PASS" if all(results.values()) else "FAIL")


if __name__ == "__main__":
    main()
