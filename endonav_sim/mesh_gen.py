"""Generate a watertight inner-wall mesh from the skeleton via an implicit
swept-sphere field + marching cubes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from skimage import measure

from .skeleton import Sample, Skeleton, build_skeleton
from .tree import TREE

GRID_STEP_MM = 0.28  # finer grid → less visible faceting under coaxial light
PADDING_MM = 3.0
SMOOTH_K = 2.0  # 1/mm — higher = sharper junctions
PAPILLA_RADIUS_FRAC = 0.50  # of the calyx end radius
PAPILLA_OFFSET_FRAC = 0.85  # lateral offset of the bulge from the centerline
PAPILLA_PROGRESS = 0.80  # how far along the calyx to plant the papilla


@dataclass
class Papilla:
    center: np.ndarray  # (3,) world-space center of the carved sphere
    radius: float  # mm
    calyx_node: str


@dataclass
class MeshBuildResult:
    mesh: trimesh.Trimesh
    bbox_min: np.ndarray
    grid_shape: tuple[int, int, int]
    papillae: list[Papilla]


def _bbox(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    pts = np.stack([s.pos for s in samples])
    rads = np.array([s.radius for s in samples])
    lo = pts.min(axis=0) - rads.max() - PADDING_MM
    hi = pts.max(axis=0) + rads.max() + PADDING_MM
    return lo, hi


def _build_grid(
    lo: np.ndarray, hi: np.ndarray, step: float
) -> tuple[np.ndarray, tuple[int, int, int]]:
    n = np.ceil((hi - lo) / step).astype(int) + 1
    xs = lo[0] + np.arange(n[0]) * step
    ys = lo[1] + np.arange(n[1]) * step
    zs = lo[2] + np.arange(n[2]) * step
    # Use 'ij' indexing so axis 0=x, 1=y, 2=z (matches marching_cubes spacing).
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1)  # (Nx, Ny, Nz, 3)
    return grid, tuple(int(v) for v in n)


def _signed_distance_field(
    grid: np.ndarray, samples: list[Sample], smooth_k: float = SMOOTH_K
) -> np.ndarray:
    """Smooth-min over per-sample distance-minus-radius fields.

    f < 0 inside the lumen, f > 0 outside, f = 0 on the wall. Uses
    log-sum-exp with the running min as a stability shift."""
    flat = grid.reshape(-1, 3)  # (P,3)
    P = flat.shape[0]

    # Pass 1: hard min (also gives us the LSE shift).
    hard_min = np.full(P, np.inf, dtype=np.float64)
    for s in samples:
        d = np.linalg.norm(flat - s.pos[None, :], axis=1) - s.radius
        np.minimum(hard_min, d, out=hard_min)

    # Pass 2: log-sum-exp accumulation with shift for numerical stability.
    acc = np.zeros(P, dtype=np.float64)
    for s in samples:
        d = np.linalg.norm(flat - s.pos[None, :], axis=1) - s.radius
        acc += np.exp(-smooth_k * (d - hard_min))
    smin = hard_min - np.log(acc) / smooth_k

    return smin.reshape(grid.shape[:3])


def _build_papillae(skel: Skeleton, tree: dict = TREE) -> list[Papilla]:
    """Place one papilla blob inside each leaf (calyx) segment.

    The bulge is centered at PAPILLA_PROGRESS along the calyx axis and
    laterally offset perpendicular to that axis so it sits against the back
    wall on one side, leaving a curved free passage on the other side."""
    papillae: list[Papilla] = []
    for name, node in tree.items():
        if node["children"]:
            continue  # only leaves get papillae
        samples = skel[name]
        # Pick a sample near PAPILLA_PROGRESS along the segment.
        idx = int(round(PAPILLA_PROGRESS * (len(samples) - 1)))
        s = samples[idx]
        tangent = s.tangent / np.linalg.norm(s.tangent)
        # Pick a perpendicular direction (any vector not parallel to tangent).
        helper = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        perp = np.cross(helper, tangent)
        perp /= np.linalg.norm(perp)
        radius = PAPILLA_RADIUS_FRAC * s.radius
        center = s.pos + perp * (PAPILLA_OFFSET_FRAC * s.radius)
        papillae.append(Papilla(center=center, radius=float(radius), calyx_node=name))
    return papillae


def _papilla_field(
    grid: np.ndarray, papillae: list[Papilla], smooth_k: float = SMOOTH_K
) -> np.ndarray:
    """Smooth-min over papilla sphere SDFs. Same convention as the lumen
    field: f<0 inside the papilla volume."""
    flat = grid.reshape(-1, 3)
    P = flat.shape[0]
    if not papillae:
        return np.full(grid.shape[:3], np.inf, dtype=np.float64)
    hard_min = np.full(P, np.inf, dtype=np.float64)
    for p in papillae:
        d = np.linalg.norm(flat - p.center[None, :], axis=1) - p.radius
        np.minimum(hard_min, d, out=hard_min)
    acc = np.zeros(P, dtype=np.float64)
    for p in papillae:
        d = np.linalg.norm(flat - p.center[None, :], axis=1) - p.radius
        acc += np.exp(-smooth_k * (d - hard_min))
    smin = hard_min - np.log(acc) / smooth_k
    return smin.reshape(grid.shape[:3])


def _smooth_max(a: np.ndarray, b: np.ndarray, k: float = SMOOTH_K) -> np.ndarray:
    """Stable smooth-max via log-sum-exp with the running max as the shift."""
    m = np.maximum(a, b)
    return m + np.log(np.exp(k * (a - m)) + np.exp(k * (b - m))) / k


def build_mesh(
    skel: Skeleton | None = None,
    grid_step: float = GRID_STEP_MM,
    smooth_k: float = SMOOTH_K,
) -> MeshBuildResult:
    if skel is None:
        skel = build_skeleton()

    samples: list[Sample] = []
    for v in skel.values():
        samples.extend(v)

    lo, hi = _bbox(samples)
    grid, shape = _build_grid(lo, hi, grid_step)

    sdf_lumen = _signed_distance_field(grid, samples, smooth_k=smooth_k)

    # Carve papillae out of the lumen by combining: keep points that are
    # inside the lumen AND outside the papilla volume. With f<0 = inside,
    # the desired final SDF is smooth_max(f_lumen, -f_papilla).
    papillae = _build_papillae(skel)
    sdf_papilla = _papilla_field(grid, papillae, smooth_k=smooth_k)
    sdf = _smooth_max(sdf_lumen, -sdf_papilla, k=smooth_k)

    # Marching cubes at the zero level set. spacing matches grid_step on each axis.
    verts, faces, normals, _ = measure.marching_cubes(
        sdf, level=0.0, spacing=(grid_step, grid_step, grid_step)
    )
    # verts come out in (x,y,z) physical coordinates relative to grid origin.
    verts = verts + lo[None, :]

    # `normals` are analytic gradients of the SDF: ∇f points from inside
    # (f<0) toward outside (f>0), i.e. outward from the lumen. We want
    # inward (toward the camera that lives inside). Negate.
    # Also flip face winding so screen-space orientation matches the inward
    # normal direction (consistent for any GL culling logic downstream).
    # skimage's marching_cubes returns gradient-based normals oriented
    # toward LOWER volume values (i.e. toward the interior of the iso-
    # surface). For our SDF (f<0 inside the lumen), that's already inward.
    inward_normals = normals
    faces = faces[:, ::-1]

    # process=False so trimesh does NOT run fix_normals / merge_vertices /
    # other heuristics that would re-orient faces and clobber the inward
    # winding we just set. We provide vertex_normals explicitly so trimesh
    # never recomputes them from face winding (which is the source of the
    # black-pinch artifacts when the two get out of sync).
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=inward_normals,
        process=False,
    )

    return MeshBuildResult(mesh=mesh, bbox_min=lo, grid_shape=shape, papillae=papillae)


if __name__ == "__main__":
    res = build_mesh()
    m = res.mesh
    print(f"vertices={len(m.vertices)} faces={len(m.faces)} watertight={m.is_watertight}")
    print(f"bbox: {m.bounds}")
