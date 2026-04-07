"""Trilinearly-interpolated voxel signed distance field.

Used as the per-frame collision query backend for :class:`KidneySimulator`.
The SDF grid is computed once at mesh-build time inside
:func:`endonav_sim.mesh_gen.build_mesh` and reused without
re-querying ``trimesh.proximity`` (which is two orders of magnitude
slower than the trilinear lookup below).

Sign convention: ``f < 0`` inside the lumen, ``f > 0`` outside. The
:meth:`VoxelSDF.clearance` helper flips the sign so that "positive
clearance" means "safely inside the lumen" — matching the legacy
:class:`endonav_sim.collision.ClearanceField` semantics.
"""

from __future__ import annotations

import numpy as np


class VoxelSDF:
    """Regular-grid signed distance field with vectorized trilinear sampling.

    Parameters
    ----------
    grid : np.ndarray
        Float array of shape ``(Nx, Ny, Nz)``. ``f < 0`` is inside the lumen.
    origin : np.ndarray
        World-space coordinates of voxel ``(0, 0, 0)``. Shape ``(3,)``.
    step : float
        Voxel spacing in mm (isotropic).
    """

    __slots__ = ("grid", "origin", "step", "_shape", "_max_idx")

    def __init__(self, grid: np.ndarray, origin: np.ndarray, step: float) -> None:
        self.grid = np.ascontiguousarray(grid, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.step = float(step)
        self._shape = self.grid.shape
        self._max_idx = np.array(
            [s - 2 for s in self._shape], dtype=np.int64
        )  # last sampleable cell

    # ------------------------------------------------------------------
    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinearly interpolate the SDF at one or more world-space points.

        ``points`` may be shape ``(3,)`` for a single sample or ``(N, 3)``
        for a batch. Returns a scalar or ``(N,)`` array of float32. Points
        outside the grid bbox are clamped to the nearest cell — that's
        always far outside the lumen so the sign and rough magnitude are
        still correct.
        """
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]

        # Local grid coordinates (continuous voxel indices)
        local = (pts - self.origin) / self.step  # (N, 3)
        i0 = np.floor(local).astype(np.int64)  # (N, 3)
        # Clamp so i0 .. i0+1 stays in-bounds
        i0 = np.minimum(np.maximum(i0, 0), self._max_idx)
        f = (local - i0).astype(np.float32)  # (N, 3) in [0,1]

        ix, iy, iz = i0[:, 0], i0[:, 1], i0[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        g = self.grid

        c000 = g[ix, iy, iz]
        c100 = g[ix + 1, iy, iz]
        c010 = g[ix, iy + 1, iz]
        c110 = g[ix + 1, iy + 1, iz]
        c001 = g[ix, iy, iz + 1]
        c101 = g[ix + 1, iy, iz + 1]
        c011 = g[ix, iy + 1, iz + 1]
        c111 = g[ix + 1, iy + 1, iz + 1]

        c00 = c000 * (1.0 - fx) + c100 * fx
        c10 = c010 * (1.0 - fx) + c110 * fx
        c01 = c001 * (1.0 - fx) + c101 * fx
        c11 = c011 * (1.0 - fx) + c111 * fx
        c0 = c00 * (1.0 - fy) + c10 * fy
        c1 = c01 * (1.0 - fy) + c11 * fy
        out = c0 * (1.0 - fz) + c1 * fz

        return float(out[0]) if single else out

    # ------------------------------------------------------------------
    def clearance(self, point: np.ndarray) -> float:
        """Distance from the wall: positive when safely inside the lumen,
        negative when the query point is outside (in the wall or beyond)."""
        return -float(self.sample(point))

    def is_clear(self, point: np.ndarray, clearance_mm: float = 0.5) -> bool:
        return self.clearance(point) >= clearance_mm


__all__ = ["VoxelSDF"]
