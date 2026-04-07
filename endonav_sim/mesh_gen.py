"""Generate a watertight inner-wall mesh from the skeleton via an implicit
swept-sphere field + marching cubes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from skimage import measure

from .skeleton import Sample, Skeleton, build_skeleton
from .stones import Stone

GRID_STEP_MM = 0.22  # fine enough that Gouraud shading reads as smooth at close range
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
    stones: list[Stone]
    sdf_grid: np.ndarray  # Final combined SDF (lumen ∩ ¬papillae ∩ ¬stones)
    sdf_origin: np.ndarray  # World-space coords of voxel (0,0,0)
    sdf_step: float  # Voxel spacing in mm


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
    log-sum-exp with the running min as a stability shift.

    Optimized: build a KD-tree on the centerline samples and, for each
    voxel, only LSE over the K nearest samples. The far-away samples
    contribute negligibly to the soft-min (their `exp(-k * d)` term is
    nearly zero) so this is essentially exact for K ≈ 24.
    """
    flat = grid.reshape(-1, 3).astype(np.float64)
    P = flat.shape[0]
    pos = np.stack([s.pos for s in samples]).astype(np.float64)  # (S, 3)
    rad = np.array([s.radius for s in samples], dtype=np.float64)  # (S,)
    S = len(samples)

    K = min(24, S)
    kd = cKDTree(pos)

    CHUNK = 16384
    out = np.empty(P, dtype=np.float64)
    for i in range(0, P, CHUNK):
        chunk = flat[i : i + CHUNK]  # (C, 3)
        dists, idxs = kd.query(chunk, k=K, workers=-1)  # (C, K)
        if K == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        d = dists - rad[idxs]  # (C, K)
        hard_min = d.min(axis=1)  # (C,)
        acc = np.exp(-smooth_k * (d - hard_min[:, None])).sum(axis=1)
        out[i : i + CHUNK] = hard_min - np.log(acc) / smooth_k

    return out.reshape(grid.shape[:3])


def _build_papillae(skel: Skeleton, tree: dict) -> list[Papilla]:
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


def _sphere_blob_field(
    grid: np.ndarray,
    centers: np.ndarray,  # (N, 3)
    radii: np.ndarray,  # (N,)
    smooth_k: float = SMOOTH_K,
) -> np.ndarray:
    """KD-tree-pruned smooth-min over sphere SDFs (papillae or stones).

    Each entity is ``f(p) = ||p - center|| - radius``. Returns the smooth-min
    of all entities at every grid point. f < 0 inside, f > 0 outside.
    """
    if len(centers) == 0:
        return np.full(grid.shape[:3], np.inf, dtype=np.float64)
    flat = grid.reshape(-1, 3).astype(np.float64)
    P = flat.shape[0]
    K = min(8, len(centers))
    kd = cKDTree(centers)
    out = np.empty(P, dtype=np.float64)
    CHUNK = 16384
    for i in range(0, P, CHUNK):
        chunk = flat[i : i + CHUNK]
        dists, idxs = kd.query(chunk, k=K, workers=-1)
        if K == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        d = dists - radii[idxs]
        hard_min = d.min(axis=1)
        acc = np.exp(-smooth_k * (d - hard_min[:, None])).sum(axis=1)
        out[i : i + CHUNK] = hard_min - np.log(acc) / smooth_k
    return out.reshape(grid.shape[:3])


def _papilla_field(
    grid: np.ndarray, papillae: list[Papilla], smooth_k: float = SMOOTH_K
) -> np.ndarray:
    if not papillae:
        return np.full(grid.shape[:3], np.inf, dtype=np.float64)
    centers = np.stack([p.center for p in papillae]).astype(np.float64)
    radii = np.array([p.radius for p in papillae], dtype=np.float64)
    return _sphere_blob_field(grid, centers, radii, smooth_k=smooth_k)


def _smooth_max(a: np.ndarray, b: np.ndarray, k: float = SMOOTH_K) -> np.ndarray:
    """Stable smooth-max via log-sum-exp with the running max as the shift."""
    m = np.maximum(a, b)
    return m + np.log(np.exp(k * (a - m)) + np.exp(k * (b - m))) / k


def _stone_field(grid: np.ndarray, stones: list[Stone], smooth_k: float = SMOOTH_K) -> np.ndarray:
    """Smooth-min over stone sphere SDFs (skips removed stones)."""
    active = [s for s in stones if not s.removed]
    if not active:
        return np.full(grid.shape[:3], np.inf, dtype=np.float64)
    centers = np.stack([s.center for s in active]).astype(np.float64)
    radii = np.array([s.radius for s in active], dtype=np.float64)
    return _sphere_blob_field(grid, centers, radii, smooth_k=smooth_k)


def build_mesh(
    skel: Skeleton,
    tree: dict,
    grid_step: float = GRID_STEP_MM,
    smooth_k: float = SMOOTH_K,
    stones: list[Stone] | None = None,
) -> MeshBuildResult:
    if stones is None:
        stones = []

    samples: list[Sample] = []
    for v in skel.values():
        samples.extend(v)

    lo, hi = _bbox(samples)
    grid, shape = _build_grid(lo, hi, grid_step)

    sdf_lumen = _signed_distance_field(grid, samples, smooth_k=smooth_k)

    # Carve papillae out of the lumen by combining: keep points that are
    # inside the lumen AND outside the papilla volume. With f<0 = inside,
    # the desired final SDF is smooth_max(f_lumen, -f_papilla).
    papillae = _build_papillae(skel, tree)
    sdf_papilla = _papilla_field(grid, papillae, smooth_k=smooth_k)
    sdf = _smooth_max(sdf_lumen, -sdf_papilla, k=smooth_k)

    # Carve stones the same way: keep points inside the lumen AND outside
    # any active (un-removed) stone volume.
    if stones:
        sdf_stones = _stone_field(grid, stones, smooth_k=smooth_k)
        sdf = _smooth_max(sdf, -sdf_stones, k=smooth_k)

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

    return MeshBuildResult(
        mesh=mesh,
        bbox_min=lo,
        grid_shape=shape,
        papillae=papillae,
        stones=list(stones),
        sdf_grid=sdf,
        sdf_origin=lo,
        sdf_step=float(grid_step),
    )


if __name__ == "__main__":
    from .anatomy import AnatomyParams, generate_anatomy

    tree, _ = generate_anatomy(AnatomyParams(seed=0))
    skel = build_skeleton(tree)
    res = build_mesh(skel, tree=tree)
    m = res.mesh
    print(f"vertices={len(m.vertices)} faces={len(m.faces)} watertight={m.is_watertight}")
    print(f"bbox: {m.bounds}")
