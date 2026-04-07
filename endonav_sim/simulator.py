"""Public KidneySimulator API: builds the world, owns the camera pose,
exposes render/command/follow_skeleton/reset."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .collision import ClearanceField
from .mesh_gen import build_mesh
from .renderer import CoaxialRenderer
from .skeleton import Skeleton, build_skeleton, flatten_skeleton
from .texture import color_mesh, displace_mesh
from .tree import TREE, root_node


def _pose_from_forward(pos: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Build a camera-to-world pose so the camera's local -Z aligns with `forward`."""
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward  # camera looks down its local -Z
    pose[:3, 3] = pos
    return pose


def _rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    K = np.array(
        [
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0],
        ]
    )
    return np.eye(3) + s * K + (1 - c) * (K @ K)


class KidneySimulator:
    def __init__(
        self,
        tree_definition: dict | None = None,
        width: int = 1024,
        height: int = 768,
        fov_y_deg: float = 95.0,
    ) -> None:
        self.tree = tree_definition or TREE
        self.skel: Skeleton = build_skeleton(self.tree)
        res = build_mesh(self.skel)
        mesh = displace_mesh(res.mesh)
        mesh = color_mesh(mesh, papillae=res.papillae)
        self.mesh = mesh
        self.papillae = res.papillae

        self.clearance = ClearanceField(mesh)
        self.renderer = CoaxialRenderer(mesh, width, height, fov_y_deg)

        # KD-tree over flattened skeleton samples for "where am I?" lookup.
        self._skel_pts, self._skel_radii, self._skel_names, self._skel_progress = flatten_skeleton(
            self.skel
        )
        self._kd = cKDTree(self._skel_pts)

        self.pose = np.eye(4, dtype=np.float64)
        self.reset()

    # ----- pose helpers -----------------------------------------------------

    def _forward(self) -> np.ndarray:
        return -self.pose[:3, 2]

    def _right(self) -> np.ndarray:
        return self.pose[:3, 0]

    def _up(self) -> np.ndarray:
        return self.pose[:3, 1]

    # ----- public API -------------------------------------------------------

    def reset(self) -> None:
        root = root_node(self.tree)
        first = self.skel[root][0]
        self.pose = _pose_from_forward(first.pos.copy(), first.tangent.copy())
        # Nudge slightly forward so we are unambiguously inside the lumen.
        self.pose[:3, 3] += first.tangent * 0.5

    def follow_skeleton(self, node_name: str, progress: float) -> None:
        samples = self.skel[node_name]
        progress = float(np.clip(progress, 0.0, 1.0))
        # Linear interp between adjacent samples.
        idx_f = progress * (len(samples) - 1)
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, len(samples) - 1)
        t = idx_f - i0
        pos = (1 - t) * samples[i0].pos + t * samples[i1].pos
        tangent = (1 - t) * samples[i0].tangent + t * samples[i1].tangent
        tangent = tangent / np.linalg.norm(tangent)
        self.pose = _pose_from_forward(pos, tangent)

    def command(self, advance_mm: float, yaw_deg: float, pitch_deg: float) -> bool:
        """Yaw (around local up) + pitch (around local right), then advance
        along the (new) forward direction. Reverts and returns False if the
        proposed position would intrude on the wall."""
        new_pose = self.pose.copy()
        R = new_pose[:3, :3]

        if yaw_deg != 0.0:
            Ry = _rot_axis_angle(R[:, 1], np.deg2rad(yaw_deg))
            R = Ry @ R
        if pitch_deg != 0.0:
            Rp = _rot_axis_angle(R[:, 0], np.deg2rad(pitch_deg))
            R = Rp @ R
        new_pose[:3, :3] = R

        forward = -R[:, 2]
        new_pos = new_pose[:3, 3] + forward * advance_mm
        new_pose[:3, 3] = new_pos

        if not self.clearance.is_clear(new_pos, clearance_mm=0.5):
            return False

        self.pose = new_pose
        return True

    def get_skeleton(self) -> dict[str, list[tuple[float, float, float]]]:
        return {
            name: [tuple(s.pos.tolist()) for s in samples] for name, samples in self.skel.items()
        }

    def render(self) -> dict:
        rgb, depth = self.renderer.render(self.pose.astype(np.float32))
        cam_pos = self.pose[:3, 3]
        nearest_mm = self.clearance.nearest_wall_distance(cam_pos)
        _, idx = self._kd.query(cam_pos)
        return {
            "rgb": rgb,
            "depth": depth,
            "pose": self.pose.copy(),
            "nearest_wall_mm": float(nearest_mm),
            "current_tree_node": self._skel_names[idx],
            "current_tree_progress": float(self._skel_progress[idx]),
        }
