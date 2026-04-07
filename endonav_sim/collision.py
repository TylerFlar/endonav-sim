"""Collision / clearance queries against the precomputed voxel SDF.

Replaces the legacy ``trimesh.proximity``-based implementation, which
queried both a BVH closest-point AND ``mesh.contains`` per call (tens of
ms per single point). The voxel SDF is built once at mesh-build time
inside :func:`endonav_sim.mesh_gen.build_mesh` and queried via
trilinear interpolation in :class:`endonav_sim.sdf.VoxelSDF` —
sub-microsecond per point.
"""

from __future__ import annotations

import numpy as np

from .sdf import VoxelSDF


class ClearanceField:
    """Wraps :class:`VoxelSDF` for camera collision tests.

    Sign convention: ``+`` inside the lumen (safe), ``-`` outside.
    """

    def __init__(self, sdf: VoxelSDF) -> None:
        self.sdf = sdf

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(points)
        return -np.asarray(self.sdf.sample(pts))

    def is_clear(self, point: np.ndarray, clearance_mm: float = 0.5) -> bool:
        return self.sdf.clearance(point) >= clearance_mm

    def nearest_wall_distance(self, point: np.ndarray) -> float:
        return self.sdf.clearance(point)


__all__ = ["ClearanceField"]
