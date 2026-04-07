"""Public KidneySimulator API: builds the world, owns the camera pose,
exposes render/command/follow_skeleton/reset/attempt_capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree

from .anatomy import AnatomyParams, generate_anatomy
from .collision import ClearanceField
from .dynamics import CommandFeedback, ScopeDynamics, ScopeLimits
from .mesh_gen import build_mesh
from .renderer import CoaxialRenderer
from .sdf import VoxelSDF
from .skeleton import Skeleton, build_skeleton, flatten_skeleton, root_node
from .stones import Stone, StoneParams, generate_stones
from .texture import color_mesh, displace_mesh


class ToolMode(Enum):
    BASKET = "basket"  # nitinol basket: grabs whole stones up to ~3.5 mm
    LASER = "laser"  # holmium / thulium fiber: fragments stones up to ~15 mm


@dataclass
class CaptureResult:
    """Outcome of an :meth:`KidneySimulator.attempt_capture` call."""

    success: bool
    mode: ToolMode
    stone_id: str | None = None
    stone_size_mm: float = 0.0
    stone_composition: str | None = None
    fragments_produced: list = field(default_factory=list)
    failure_reason: str | None = None


# Tool/channel capacities — driven by real 3.6 Fr working channel + 3 Fr tools.
BASKET_MAX_DIAMETER_MM = 3.5
LASER_MAX_DIAMETER_MM = 15.0
TIP_REACH_MM = 4.0
CAPTURE_CONE_DEG = 30.0


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
        anatomy_params: AnatomyParams | None = None,
        stone_params: StoneParams | None = None,
        scope_limits: ScopeLimits | None = None,
        seed: int = 0,
        width: int = 1024,
        height: int = 768,
        fov_y_deg: float = 95.0,
        realistic: bool = True,
    ) -> None:
        # Generate a fresh anatomy. ``anatomy_params`` defaults to
        # ``AnatomyParams(seed=seed)`` so a bare ``KidneySimulator()`` still
        # produces a valid kidney.
        if anatomy_params is None:
            anatomy_params = AnatomyParams(seed=seed)
        self.tree, self.anatomy_meta = generate_anatomy(anatomy_params)

        self.skel: Skeleton = build_skeleton(self.tree)

        # Stones: optional, sampled fresh from a seed if requested.
        self.stones: list[Stone] = []
        if stone_params is not None:
            self.stones = generate_stones(self.skel, self.anatomy_meta, stone_params)

        res = build_mesh(self.skel, tree=self.tree, stones=self.stones)
        mesh = displace_mesh(res.mesh)
        mesh = color_mesh(mesh, papillae=res.papillae, stones=self.stones)
        self.mesh = mesh
        self.papillae = res.papillae

        # Voxel SDF cached at build time → sub-microsecond per-point clearance.
        # Bump the effective clearance margin by half a voxel so trilinear-
        # interp error never lets the camera tunnel through the wall.
        self._sdf_grid_step = float(res.sdf_step)
        self._effective_clearance_mm = 0.5 + 0.5 * self._sdf_grid_step
        self.clearance = ClearanceField(VoxelSDF(res.sdf_grid, res.sdf_origin, res.sdf_step))
        self.renderer = CoaxialRenderer(mesh, width, height, fov_y_deg)

        # Realistic dynamics layer (encoder noise, hysteresis, buckling, ...).
        self.realistic = bool(realistic)
        self.dynamics = ScopeDynamics(scope_limits or ScopeLimits(), seed=seed)

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
        self.dynamics.reset(self.pose)

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
    ) -> CommandFeedback:
        """Issue a robot command and return realistic telemetry.

        advance_mm: translate along shaft (passively follows centerline).
        roll_deg: INCREMENTAL axial shaft rotation (adds to cumulative roll).
        deflection_deg: ABSOLUTE tip bending in the (rolled) bending plane.

        Returns a :class:`CommandFeedback` with encoder readback, contact
        force, buckling flag, noisy wall-clearance, and a dead-reckoned pose
        estimate. When ``realistic=False`` (constructor flag) the dynamics
        layer is bypassed entirely and commands execute perfectly.
        """
        prev_deflection_deg = float(np.rad2deg(self.current_deflection))

        if self.realistic:
            adv, rl, defl_target_deg = self.dynamics.shape_command(
                advance_mm, roll_deg, deflection_deg, prev_deflection_deg
            )
        else:
            adv, rl, defl_target_deg = float(advance_mm), float(roll_deg), float(deflection_deg)

        prev_state = (
            self.path_node,
            self.path_progress,
            self.cumulative_roll,
            self.current_deflection,
        )

        new_roll = self.cumulative_roll + np.deg2rad(rl)
        new_deflection = np.deg2rad(defl_target_deg)
        if adv != 0.0:
            new_node, new_progress = self._advance_along_path(
                self.path_node, self.path_progress, adv
            )
        else:
            new_node, new_progress = self.path_node, self.path_progress

        new_pose = self._compute_pose(new_node, new_progress, new_roll, new_deflection)
        new_pos = new_pose[:3, 3]

        buckled = False
        collided = False
        actual_advance = adv
        actual_roll = rl
        actual_deflection = defl_target_deg
        contact_force = 0.0

        if not self.clearance.is_clear(new_pos, clearance_mm=self._effective_clearance_mm):
            # Collision: try the buckling fallback only if running realistic.
            if self.realistic:
                # The shaft bows: tip stays at the previous along-path position
                # but picks up an extra random deflection wiggle.
                wiggle_deg = self.dynamics.buckle_perturbation()
                wiggled_deflection = np.deg2rad(prev_deflection_deg + wiggle_deg)
                wiggled_pose = self._compute_pose(
                    self.path_node, self.path_progress, self.cumulative_roll, wiggled_deflection
                )
                if self.clearance.is_clear(
                    wiggled_pose[:3, 3], clearance_mm=self._effective_clearance_mm
                ):
                    # Productive buckle: tip nudged sideways but no progress
                    self.current_deflection = float(wiggled_deflection)
                    self.pose = wiggled_pose
                    actual_advance = 0.0
                    actual_deflection = float(np.rad2deg(wiggled_deflection))
                    buckled = True
                    contact_force = 0.6
                else:
                    # Wall blocked even the wiggled pose — revert everything
                    (
                        self.path_node,
                        self.path_progress,
                        self.cumulative_roll,
                        self.current_deflection,
                    ) = prev_state
                    actual_advance = 0.0
                    actual_roll = 0.0
                    actual_deflection = prev_deflection_deg
                    buckled = True
                    collided = True
                    contact_force = 1.0
            else:
                # Clean kinematic mode: hard reject, identical to old behaviour.
                (
                    self.path_node,
                    self.path_progress,
                    self.cumulative_roll,
                    self.current_deflection,
                ) = prev_state
                actual_advance = 0.0
                actual_roll = 0.0
                actual_deflection = prev_deflection_deg
                collided = True
                contact_force = 1.0
        else:
            self.path_node = new_node
            self.path_progress = new_progress
            self.cumulative_roll = float(new_roll)
            self.current_deflection = float(new_deflection)
            self.pose = new_pose
            # Contact-force estimate from how close to the wall the tip is now
            true_clearance = float(self.clearance.nearest_wall_distance(new_pos))
            contact_force = float(np.clip(1.0 - (true_clearance / 4.0), 0.0, 1.0))

        # Wall clearance feedback (noisy in realistic mode)
        true_clearance = float(self.clearance.nearest_wall_distance(self.pose[:3, 3]))
        if self.realistic:
            wall_clearance_mm = self.dynamics.quantize_clearance(true_clearance)
            tip_pose_estimate = self.dynamics.integrate_pose_estimate(
                actual_advance, actual_roll, actual_deflection, self.pose
            )
        else:
            wall_clearance_mm = true_clearance
            tip_pose_estimate = self.pose.copy()

        return CommandFeedback(
            actual_advance_mm=float(actual_advance),
            actual_roll_deg=float(actual_roll),
            actual_deflection_deg=float(actual_deflection),
            contact_force_norm=float(contact_force),
            buckled=bool(buckled),
            wall_clearance_mm=float(wall_clearance_mm),
            tip_pose_estimate=tip_pose_estimate,
            collided=bool(collided),
        )

    # ----- stones / tool channel ------------------------------------------

    def attempt_capture(self, mode: ToolMode = ToolMode.BASKET) -> CaptureResult:
        """Try to grab or fragment the nearest reachable stone.

        A stone is reachable when it is within ``TIP_REACH_MM`` of the camera
        and inside a ``CAPTURE_CONE_DEG`` half-angle cone in front of the
        camera. BASKET captures whole stones up to 3.5 mm diameter; LASER
        fragments stones up to 15 mm diameter into 2-6 smaller pieces and
        rebuilds the rendered mesh.
        """
        cam_pos = self.pose[:3, 3]
        cam_fwd = -self.pose[:3, 2]
        cam_fwd = cam_fwd / max(float(np.linalg.norm(cam_fwd)), 1e-9)

        best: Stone | None = None
        best_d = np.inf
        for s in self.stones:
            if s.removed:
                continue
            delta = s.center - cam_pos
            dist = float(np.linalg.norm(delta))
            if dist > TIP_REACH_MM + s.radius:
                continue
            direction = delta / max(dist, 1e-9)
            cosang = float(np.dot(direction, cam_fwd))
            if cosang < float(np.cos(np.deg2rad(CAPTURE_CONE_DEG))):
                continue
            if dist < best_d:
                best = s
                best_d = dist

        if best is None:
            return CaptureResult(success=False, mode=mode, failure_reason="no_reachable_stone")

        diameter = 2.0 * best.radius
        if mode == ToolMode.BASKET:
            if diameter > BASKET_MAX_DIAMETER_MM:
                return CaptureResult(
                    success=False,
                    mode=mode,
                    stone_id=best.id,
                    stone_size_mm=diameter,
                    stone_composition=best.composition,
                    failure_reason="stone_too_large_for_basket",
                )
            best.removed = True
            self._rebuild_mesh()
            return CaptureResult(
                success=True,
                mode=mode,
                stone_id=best.id,
                stone_size_mm=diameter,
                stone_composition=best.composition,
            )

        # LASER fragmentation
        if diameter > LASER_MAX_DIAMETER_MM:
            return CaptureResult(
                success=False,
                mode=mode,
                stone_id=best.id,
                stone_size_mm=diameter,
                stone_composition=best.composition,
                failure_reason="stone_too_large_for_laser",
            )
        rng = self.dynamics.rng if self.realistic else np.random.default_rng(0)
        n_frags = int(rng.integers(2, 7))
        # Fragment volumes sum to ~80% of the original; the remaining 20% is
        # "dust" (irrigated out).
        original_vol = (4.0 / 3.0) * np.pi * best.radius**3
        target_total = 0.8 * original_vol
        weights = rng.dirichlet(np.ones(n_frags))
        frags: list[Stone] = []
        for i, w in enumerate(weights):
            r = float((3.0 * w * target_total / (4.0 * np.pi)) ** (1.0 / 3.0))
            offset = rng.normal(0.0, 0.4 * best.radius, size=3)
            frags.append(
                Stone(
                    id=f"{best.id}_frag{i}",
                    center=best.center + offset,
                    radius=max(r, 0.5),
                    node_id=best.node_id,
                    composition=best.composition,
                    is_fragment=True,
                )
            )
        best.removed = True
        self.stones.extend(frags)
        self._rebuild_mesh()
        return CaptureResult(
            success=True,
            mode=mode,
            stone_id=best.id,
            stone_size_mm=diameter,
            stone_composition=best.composition,
            fragments_produced=frags,
        )

    def _rebuild_mesh(self) -> None:
        """Rebuild the wall mesh after a capture or fragmentation event."""
        res = build_mesh(self.skel, tree=self.tree, stones=self.stones)
        mesh = displace_mesh(res.mesh)
        mesh = color_mesh(mesh, papillae=res.papillae, stones=self.stones)
        self.mesh = mesh
        self.papillae = res.papillae
        self.clearance = ClearanceField(VoxelSDF(res.sdf_grid, res.sdf_origin, res.sdf_step))
        # The CoaxialRenderer holds persistent GL buffers tied to the old
        # mesh; rebuild it to swap geometry safely.
        w, h = self.renderer.width, self.renderer.height
        fov = self.renderer.fov_y_deg
        self.renderer.release()
        self.renderer = CoaxialRenderer(mesh, w, h, fov)

    def get_skeleton(self) -> dict[str, list[tuple[float, float, float]]]:
        return {
            name: [tuple(s.pos.tolist()) for s in samples] for name, samples in self.skel.items()
        }

    def render(
        self,
        with_depth: bool = False,
        with_stones_visible: bool = False,
    ) -> dict:
        """Render a frame and return a dict of perception data.

        Both ``depth`` and the ground-truth ``stones_visible`` list are
        opt-in because they are the dominant per-frame costs and most
        callers don't need them every frame.
        """
        rgb, depth = self.renderer.render(self.pose.astype(np.float32), with_depth=with_depth)
        cam_pos = self.pose[:3, 3]
        nearest_mm = float(self.clearance.nearest_wall_distance(cam_pos))
        if self.realistic:
            reported_nearest = self.dynamics.quantize_clearance(nearest_mm)
        else:
            reported_nearest = nearest_mm
        _, idx = self._kd.query(cam_pos)

        stones_visible: list[dict] | None = None
        if with_stones_visible:
            cam_fwd = -self.pose[:3, 2]
            cam_fwd = cam_fwd / max(float(np.linalg.norm(cam_fwd)), 1e-9)
            cos_half = float(np.cos(np.deg2rad(self.renderer.fov_y_deg * 0.5)))
            max_range_mm = 60.0
            stones_visible = []
            for s in self.stones:
                if s.removed:
                    continue
                delta = s.center - cam_pos
                d = float(np.linalg.norm(delta))
                if d < 1e-6 or d > max_range_mm:
                    continue
                direction = delta / d
                if float(np.dot(direction, cam_fwd)) < cos_half:
                    continue
                stones_visible.append(
                    {
                        "id": s.id,
                        "distance_mm": d,
                        "radius_mm": float(s.radius),
                        "composition": s.composition,
                    }
                )

        return {
            "rgb": rgb,
            "depth": depth,
            "pose": self.pose.copy(),
            "nearest_wall_mm": float(reported_nearest),
            "true_nearest_wall_mm": nearest_mm,
            "current_tree_node": self._skel_names[idx],  # GROUND TRUTH — eval only
            "current_tree_progress": float(self._skel_progress[idx]),
            "stones_visible": stones_visible,  # GROUND TRUTH — eval only
        }
