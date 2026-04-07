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
        self.path_node = root_node(self.tree)
        # Start a touch inside the root so we are unambiguously in the lumen.
        node_len = float(self.tree[self.path_node]["length"])
        self.path_progress = float(np.clip(0.5 / max(node_len, 1e-6), 0.0, 1.0))
        self.cumulative_roll = 0.0
        self.current_deflection = 0.0
        self._update_pose()

    def _sample_path(self, node_name: str, progress: float) -> tuple[np.ndarray, np.ndarray]:
        samples = self.skel[node_name]
        progress = float(np.clip(progress, 0.0, 1.0))
        idx_f = progress * (len(samples) - 1)
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, len(samples) - 1)
        t = idx_f - i0
        pos = (1 - t) * samples[i0].pos + t * samples[i1].pos
        tangent = (1 - t) * samples[i0].tangent + t * samples[i1].tangent
        tangent = tangent / np.linalg.norm(tangent)
        return pos, tangent

    def _compute_pose(
        self,
        node_name: str,
        progress: float,
        cumulative_roll: float,
        current_deflection: float,
    ) -> np.ndarray:
        pos, tangent = self._sample_path(node_name, progress)

        # Stable basis perpendicular to the tangent (the unrolled bending plane).
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tangent, world_up)) > 0.95:
            world_up = np.array([0.0, 1.0, 0.0])
        local_up0 = world_up - np.dot(world_up, tangent) * tangent
        local_up0 /= np.linalg.norm(local_up0)
        local_right0 = np.cross(tangent, local_up0)

        # Roll: rotate the bending-plane basis around the shaft axis.
        Rroll = _rot_axis_angle(tangent, cumulative_roll)
        bend_axis = Rroll @ local_right0  # axis around which the tip deflects
        cam_up = Rroll @ local_up0

        # Deflect: bend the tip in the rolled bending plane.
        Rdef = _rot_axis_angle(bend_axis, current_deflection)
        view_dir = Rdef @ tangent
        cam_up_final = Rdef @ cam_up

        right = np.cross(view_dir, cam_up_final)
        right /= np.linalg.norm(right)
        cam_up_final = np.cross(right, view_dir)
        cam_up_final /= np.linalg.norm(cam_up_final)

        pose = np.eye(4, dtype=np.float64)
        pose[:3, 0] = right
        pose[:3, 1] = cam_up_final
        pose[:3, 2] = -view_dir
        pose[:3, 3] = pos
        return pose

    def _update_pose(self) -> None:
        self.pose = self._compute_pose(
            self.path_node, self.path_progress, self.cumulative_roll, self.current_deflection
        )

    def _advance_along_path(
        self, node_name: str, progress: float, advance_mm: float
    ) -> tuple[str, float]:
        """Walk a signed arc length along the skeleton, descending into the
        first child on overshoot and into the parent on undershoot."""
        node = node_name
        # Convert to absolute arc length within node, walk, then re-normalize.
        remaining = advance_mm
        while True:
            node_len = float(self.tree[node]["length"])
            arc = progress * node_len + remaining
            if arc < 0.0:
                parent = self.tree[node].get("parent")
                if parent is None:
                    return node, 0.0
                parent_len = float(self.tree[parent]["length"])
                # Carry the negative remainder into the parent at its end.
                remaining = arc  # still negative
                node = parent
                progress = 1.0
                # Re-express: arc-from-parent-end = arc; new_arc = parent_len + arc
                new_arc = parent_len + remaining
                if new_arc >= 0.0:
                    return node, float(np.clip(new_arc / parent_len, 0.0, 1.0))
                # Need to keep ascending; reset remaining for next iter.
                remaining = new_arc - 0.0  # but using parent_len as the base
                # Simplify: set progress to 0 and remaining = new_arc, loop.
                progress = 0.0
                remaining = new_arc
                continue
            if arc > node_len:
                children = self.tree[node].get("children", [])
                if not children:
                    return node, 1.0
                # Descend into first child carrying the overflow.
                remaining = arc - node_len
                node = children[0]
                progress = 0.0
                continue
            return node, float(arc / node_len)

    def follow_skeleton(self, node_name: str, progress: float) -> None:
        self.path_node = node_name
        self.path_progress = float(np.clip(progress, 0.0, 1.0))
        self.cumulative_roll = 0.0
        self.current_deflection = 0.0
        self._update_pose()

    def command(
        self,
        advance_mm: float = 0.0,
        roll_deg: float = 0.0,
        deflection_deg: float = 0.0,
    ) -> bool:
        """Ureteroscope command.

        advance_mm: translate along shaft (passively follows centerline).
        roll_deg: INCREMENTAL axial shaft rotation (adds to cumulative roll).
        deflection_deg: ABSOLUTE tip bending in the (rolled) bending plane.

        Returns False (and reverts state) if the new tip position collides.
        """
        prev_state = (
            self.path_node,
            self.path_progress,
            self.cumulative_roll,
            self.current_deflection,
        )

        new_roll = self.cumulative_roll + np.deg2rad(roll_deg)
        new_deflection = np.deg2rad(deflection_deg)
        if advance_mm != 0.0:
            new_node, new_progress = self._advance_along_path(
                self.path_node, self.path_progress, advance_mm
            )
        else:
            new_node, new_progress = self.path_node, self.path_progress

        new_pose = self._compute_pose(new_node, new_progress, new_roll, new_deflection)
        new_pos = new_pose[:3, 3]
        if not self.clearance.is_clear(new_pos, clearance_mm=0.5):
            # Revert: state unchanged.
            (
                self.path_node,
                self.path_progress,
                self.cumulative_roll,
                self.current_deflection,
            ) = prev_state
            return False

        self.path_node = new_node
        self.path_progress = new_progress
        self.cumulative_roll = float(new_roll)
        self.current_deflection = float(new_deflection)
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
