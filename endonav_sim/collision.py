"""SDF-based clearance checks against the wall mesh."""

from __future__ import annotations

import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery


class ClearanceField:
    """Wraps trimesh.proximity for camera collision tests.

    Sign convention: trimesh.signed_distance is positive inside the
    watertight volume. Our mesh encloses the lumen as its interior, so
    positive == inside the lumen (the safe region) directly."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        self.mesh = mesh
        self.pq = ProximityQuery(mesh)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Return clearance from the wall: + inside lumen, - outside (in wall).

        trimesh.signed_distance uses nearest-face-normal sign tests that are
        unreliable on concave smoothed surfaces (e.g. inside the calyces).
        We compute unsigned distance via closest_point and determine sign
        independently with mesh.contains (winding-number / ray casting)."""
        pts = np.atleast_2d(points)
        _, unsigned, _ = self.pq.on_surface(pts)
        inside = self.mesh.contains(pts)
        sign = np.where(inside, 1.0, -1.0)
        return sign * unsigned

    def is_clear(self, point: np.ndarray, clearance_mm: float = 0.5) -> bool:
        return float(self.signed_distance(point)[0]) >= clearance_mm

    def nearest_wall_distance(self, point: np.ndarray) -> float:
        return float(self.signed_distance(point)[0])
