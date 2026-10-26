"""KidneySimulator using SOFA BeamAdapter InterventionalRadiologyController.

Advance/retract is handled by SOFA's ``InterventionalRadiologyController``
which manages beam node deployment along the wire rest shape. Tip bending
comes from the pre-curved ``RodSpireSection`` rest shape combined with
axial rotation. Wall contact is handled by SOFA's collision pipeline
with friction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.spatial import cKDTree

from .anatomy import AnatomyParams, generate_anatomy
from .dynamics import CommandFeedback, ScopeDynamics, ScopeLimits
from .mesh_gen import build_mesh
from .renderer import CoaxialRenderer
from .skeleton import Skeleton, build_skeleton, flatten_skeleton, root_node
from .sofa_setup import init_sofa
from .stones import Stone, StoneParams, generate_stones
from .texture import color_mesh, displace_mesh
from .visuals import ToolVisualState

init_sofa()

import Sofa  # noqa: E402
import Sofa.Simulation  # noqa: E402

from .sofa_scene import (  # noqa: E402
    SHAFT_LENGTH_MM,
    TIP_LENGTH_MM,
    create_sofa_scene,
    extract_deformed_vertices,
    extract_stone_positions,
    extract_tip_pose,
    get_tip_deployed_length,
    set_tip_deployed_length,
    set_tip_rotation,
    step_sofa,
)


class ToolMode(Enum):
    BASKET = "basket"
    LASER = "laser"


@dataclass
class CommandResult:
    actual_advance_mm: float
    actual_roll_deg: float
    actual_kappa1_per_mm: float
    actual_phi1_rad: float
    actual_kappa2_per_mm: float
    actual_phi2_rad: float
    contact_force_norm: float
    buckled: bool
    collided: bool
    wall_clearance_mm: float
    tip_pose_estimate: np.ndarray


@dataclass
class CaptureResult:
    success: bool
    mode: ToolMode
    stone_id: str | None = None
    stone_size_mm: float = 0.0
    stone_composition: str | None = None
    fragments_produced: list = field(default_factory=list)
    failure_reason: str | None = None


BASKET_MAX_DIAMETER_MM = 3.5
LASER_MAX_DIAMETER_MM = 15.0
TIP_REACH_MM = 4.0
CAPTURE_CONE_DEG = 30.0
DUST_RADIUS_MM = 0.6
MIN_FRAGMENT_RADIUS_MM = 0.3


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
        if anatomy_params is None:
            anatomy_params = AnatomyParams(seed=seed)
        self.tree, self.anatomy_meta = generate_anatomy(anatomy_params)
        self.skel: Skeleton = build_skeleton(self.tree)

        self.stones: list[Stone] = []
        if stone_params is not None:
            self.stones = generate_stones(self.skel, self.anatomy_meta, stone_params)

        res = build_mesh(self.skel, tree=self.tree, stones=self.stones)
        mesh = displace_mesh(res.mesh)
        mesh = color_mesh(mesh, papillae=res.papillae, stones=self.stones)
        self.mesh = mesh
        self.papillae = res.papillae
        self.renderer = CoaxialRenderer(mesh, width, height, fov_y_deg)

        self.realistic = bool(realistic)
        self.dynamics = ScopeDynamics(scope_limits or ScopeLimits(), seed=seed)
        self._tool_visual = ToolVisualState()

        self._skel_pts, self._skel_radii, self._skel_names, self._skel_progress = (
            flatten_skeleton(self.skel)
        )
        self._kd = cKDTree(self._skel_pts)

        # SOFA scene
        self._sofa_root = Sofa.Core.Node("root")
        self._sofa = create_sofa_scene(
            self._sofa_root, mesh, self.skel, self.tree, self.stones,
        )
        Sofa.Simulation.init(self._sofa_root)

        # Displacement transfer for deformed mesh rendering
        self._physics_rest_verts = self._sofa["physics_verts"].copy()
        physics_kd = cKDTree(self._physics_rest_verts)
        render_verts = np.asarray(self.mesh.vertices, dtype=np.float64)
        _, self._phys_to_render_map = physics_kd.query(render_verts)
        self._render_rest_verts = render_verts.copy()
        self._render_rest_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float32).copy()

        # State tracking
        self._deployed_length = 0.0  # current tip deployment (mm)
        self._rotation_deg = 0.0  # current axial rotation
        self.kappa1 = self.phi1 = self.kappa2 = self.phi2 = 0.0
        self.cumulative_roll = 0.0

        self.pose = np.eye(4, dtype=np.float64)
        self.path_node = root_node(self.tree)
        self.path_progress = 0.0
        self._sync_pose()

    # ----- internal ----------------------------------------------------------

    @property
    def tool_visual_state(self) -> ToolVisualState:
        return self._tool_visual.snapshot()

    def _sync_pose(self) -> None:
        """Read the beam tip pose from SOFA and update path tracking."""
        self.pose = extract_tip_pose(
            self._sofa["beam_dofs"], self._sofa["deploy_controller"]
        )
        cam_pos = self.pose[:3, 3]
        if not np.any(np.isnan(cam_pos)):
            _, idx = self._kd.query(cam_pos)
            self.path_node = self._skel_names[idx]
            self.path_progress = float(self._skel_progress[idx])

    def _sample_path(self, node_name: str, progress: float) -> tuple[np.ndarray, np.ndarray]:
        samples = self.skel[node_name]
        progress = float(np.clip(progress, 0.0, 1.0))
        idx_f = progress * (len(samples) - 1)
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, len(samples) - 1)
        t = idx_f - i0
        pos = (1 - t) * samples[i0].pos + t * samples[i1].pos
        tangent = (1 - t) * samples[i0].tangent + t * samples[i1].tangent
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
        return pos, tangent

    # ----- public API -------------------------------------------------------

    def reset(self) -> None:
        Sofa.Simulation.reset(self._sofa_root)
        self._deployed_length = 0.0
        self._rotation_deg = 0.0
        self.kappa1 = self.phi1 = self.kappa2 = self.phi2 = 0.0
        self.cumulative_roll = 0.0
        self.path_node = root_node(self.tree)
        self.path_progress = 0.0
        self._sync_pose()
        self.dynamics.reset(self.pose)

    def follow_skeleton(self, node_name: str, progress: float) -> None:
        """Deploy the scope to reach a specific skeleton location."""
        # Estimate the curvilinear distance to this point
        target_pos, _ = self._sample_path(node_name, progress)
        # Find nearest skeleton point and estimate cumulative arc length
        _, idx = self._kd.query(target_pos)
        # Rough deployment length = index / total_points * total_length
        total_pts = len(self._skel_pts)
        deploy = float(idx) / max(total_pts - 1, 1) * SHAFT_LENGTH_MM
        deploy = float(np.clip(deploy, 0, SHAFT_LENGTH_MM + TIP_LENGTH_MM))

        self.reset()
        self._deployed_length = deploy
        set_tip_deployed_length(self._sofa["deploy_controller"], deploy)
        # Let SOFA settle with the new deployment
        for _ in range(10):
            step_sofa(self._sofa_root)
        self._sync_pose()
        self.path_node = node_name
        self.path_progress = float(np.clip(progress, 0.0, 1.0))

    def command(
        self,
        advance_mm: float = 0.0,
        roll_deg: float = 0.0,
        kappa1_target: float = 0.0,
        phi1_target: float = 0.0,
        kappa2_target: float = 0.0,
        phi2_target: float | None = None,
    ) -> CommandResult:
        if phi2_target is None:
            phi2_target = phi1_target

        # Dynamics shaping
        if self.realistic:
            shaped = self.dynamics.shape_command(
                advance_mm, roll_deg,
                kappa1_target, phi1_target, kappa2_target, phi2_target,
                float(self.kappa1), float(self.phi1),
                float(self.kappa2), float(self.phi2),
            )
        else:
            shaped = CommandFeedback(
                actual_advance_mm=float(advance_mm),
                actual_roll_deg=float(roll_deg),
                actual_kappa1_per_mm=float(kappa1_target),
                actual_phi1_rad=float(phi1_target),
                actual_kappa2_per_mm=float(kappa2_target),
                actual_phi2_rad=float(phi2_target),
            )

        prev_pos = self.pose[:3, 3].copy()

        # Advance: update deployed length via InterventionalRadiologyController
        adv = shaped.actual_advance_mm
        max_deploy = SHAFT_LENGTH_MM + TIP_LENGTH_MM
        self._deployed_length = float(np.clip(
            self._deployed_length + adv, 0, max_deploy
        ))
        set_tip_deployed_length(self._sofa["deploy_controller"], self._deployed_length)

        # Roll: rotate the scope axially (steers the pre-curved tip)
        self._rotation_deg += shaped.actual_roll_deg
        set_tip_rotation(self._sofa["deploy_controller"], self._rotation_deg)

        # Step SOFA
        step_sofa(self._sofa_root)

        # Extract new pose
        self._sync_pose()

        # Sync stone positions
        if self._sofa["stone_dofs"]:
            stone_positions = extract_stone_positions(self._sofa["stone_dofs"])
            for i, stone_idx in enumerate(self._sofa["active_stone_indices"]):
                if i < len(stone_positions):
                    self.stones[stone_idx].center = stone_positions[i]

        # Update state
        self.kappa1 = float(shaped.actual_kappa1_per_mm)
        self.phi1 = float(shaped.actual_phi1_rad)
        self.kappa2 = float(shaped.actual_kappa2_per_mm)
        self.phi2 = float(shaped.actual_phi2_rad)
        self.cumulative_roll = float(np.deg2rad(self._rotation_deg))

        # Contact force heuristic
        displacement = float(np.linalg.norm(self.pose[:3, 3] - prev_pos))
        contact_force = 0.0
        if abs(adv) > 0.01:
            contact_force = float(np.clip(1.0 - displacement / abs(adv), 0.0, 1.0))

        dist_to_skel, _ = self._kd.query(self.pose[:3, 3])
        wall_clearance = max(0.0, float(dist_to_skel))

        if self.realistic:
            tip_pose_estimate = self.dynamics.integrate_pose_estimate(
                shaped.actual_advance_mm, shaped.actual_roll_deg, self.pose
            )
            wall_clearance = self.dynamics.quantize_clearance(wall_clearance)
        else:
            tip_pose_estimate = self.pose.copy()

        return CommandResult(
            actual_advance_mm=float(shaped.actual_advance_mm),
            actual_roll_deg=float(shaped.actual_roll_deg),
            actual_kappa1_per_mm=float(shaped.actual_kappa1_per_mm),
            actual_phi1_rad=float(shaped.actual_phi1_rad),
            actual_kappa2_per_mm=float(shaped.actual_kappa2_per_mm),
            actual_phi2_rad=float(shaped.actual_phi2_rad),
            contact_force_norm=contact_force,
            buckled=False,
            collided=False,
            wall_clearance_mm=float(wall_clearance),
            tip_pose_estimate=tip_pose_estimate,
        )

    # ----- rendering ---------------------------------------------------------

    def render(self, with_depth=False, with_stones_visible=False) -> dict:
        deformed_physics = extract_deformed_vertices(self._sofa["kidney_dofs"])
        displacements = deformed_physics - self._physics_rest_verts
        render_displacements = displacements[self._phys_to_render_map]
        deformed_render = (self._render_rest_verts + render_displacements).astype(np.float32)
        self.renderer.update_vertices(deformed_render, self._render_rest_normals)

        rgb, depth = self.renderer.render(self.pose.astype(np.float32), with_depth=with_depth)
        cam_pos = self.pose[:3, 3]
        dist_to_skel, idx = self._kd.query(cam_pos)
        nearest_mm = max(0.0, float(dist_to_skel))
        reported_nearest = (
            self.dynamics.quantize_clearance(nearest_mm) if self.realistic else nearest_mm
        )

        stones_visible = None
        if with_stones_visible:
            cam_fwd = -self.pose[:3, 2]
            cam_fwd /= max(float(np.linalg.norm(cam_fwd)), 1e-9)
            cos_half = float(np.cos(np.deg2rad(self.renderer.fov_y_deg * 0.5)))
            stones_visible = []
            for s in self.stones:
                if s.removed:
                    continue
                delta = s.center - cam_pos
                d = float(np.linalg.norm(delta))
                if d < 1e-6 or d > 60.0:
                    continue
                if float(np.dot(delta / d, cam_fwd)) < cos_half:
                    continue
                stones_visible.append({
                    "id": s.id, "distance_mm": d,
                    "radius_mm": float(s.radius), "composition": s.composition,
                })

        return {
            "rgb": rgb, "depth": depth, "pose": self.pose.copy(),
            "nearest_wall_mm": float(reported_nearest),
            "true_nearest_wall_mm": nearest_mm,
            "current_tree_node": self._skel_names[idx],
            "current_tree_progress": float(self._skel_progress[idx]),
            "stones_visible": stones_visible,
        }

    # ----- stones / tool channel ------------------------------------------

    def attempt_capture(self, mode: ToolMode = ToolMode.BASKET) -> CaptureResult:
        cam_pos = self.pose[:3, 3]
        cam_fwd = -self.pose[:3, 2]
        cam_fwd /= max(float(np.linalg.norm(cam_fwd)), 1e-9)

        best: Stone | None = None
        best_d = np.inf
        for s in self.stones:
            if s.removed:
                continue
            delta = s.center - cam_pos
            dist = float(np.linalg.norm(delta))
            if dist > TIP_REACH_MM + s.radius:
                continue
            if float(np.dot(delta / max(dist, 1e-9), cam_fwd)) < float(
                np.cos(np.deg2rad(CAPTURE_CONE_DEG))
            ):
                continue
            if dist < best_d:
                best = s
                best_d = dist

        if best is None:
            if mode == ToolMode.LASER:
                self._tool_visual.note_laser("miss")
            else:
                self._tool_visual.note_basket("use")
            return CaptureResult(success=False, mode=mode, failure_reason="no_reachable_stone")

        diameter = 2.0 * best.radius
        if mode == ToolMode.BASKET:
            if diameter > BASKET_MAX_DIAMETER_MM:
                self._tool_visual.note_basket("use", stone_id=best.id)
                return CaptureResult(
                    success=False, mode=mode, stone_id=best.id,
                    stone_size_mm=diameter, stone_composition=best.composition,
                    failure_reason="stone_too_large_for_basket",
                )
            self._tool_visual.note_basket("success", stone_id=best.id)
            best.removed = True
            self._rebuild_mesh()
            return CaptureResult(
                success=True, mode=mode, stone_id=best.id,
                stone_size_mm=diameter, stone_composition=best.composition,
            )

        if diameter > LASER_MAX_DIAMETER_MM:
            self._tool_visual.note_laser("miss", stone_id=best.id, composition=best.composition)
            return CaptureResult(
                success=False, mode=mode, stone_id=best.id,
                stone_size_mm=diameter, stone_composition=best.composition,
                failure_reason="stone_too_large_for_laser",
            )

        if best.radius < DUST_RADIUS_MM:
            self._tool_visual.note_laser("dust", stone_id=best.id, composition=best.composition)
            best.removed = True
            self._rebuild_mesh()
            return CaptureResult(
                success=True, mode=mode, stone_id=best.id,
                stone_size_mm=diameter, stone_composition=best.composition,
                fragments_produced=[],
            )

        rng = self.dynamics.rng if self.realistic else np.random.default_rng(0)
        n_frags = int(rng.integers(2, 7))
        original_vol = (4.0 / 3.0) * np.pi * best.radius**3
        weights = rng.dirichlet(np.ones(n_frags))
        frags: list[Stone] = []
        for i, w in enumerate(weights):
            r = float((3.0 * w * 0.8 * original_vol / (4.0 * np.pi)) ** (1.0 / 3.0))
            if r < MIN_FRAGMENT_RADIUS_MM:
                continue
            placed = self._place_fragment(rng, best, r)
            if placed is None:
                continue
            frags.append(Stone(
                id=f"{best.id}_frag{i}", center=placed, radius=r,
                node_id=best.node_id, composition=best.composition, is_fragment=True,
            ))
        self._tool_visual.note_laser("hit", stone_id=best.id, composition=best.composition)
        best.removed = True
        self.stones.extend(frags)
        self._rebuild_mesh()
        return CaptureResult(
            success=True, mode=mode, stone_id=best.id,
            stone_size_mm=diameter, stone_composition=best.composition,
            fragments_produced=frags,
        )

    def _place_fragment(self, rng, parent, radius):
        samples = self.skel.get(parent.node_id) if isinstance(self.skel, dict) else None
        if not samples:
            return None
        positions = np.stack([s.pos for s in samples])
        dists = np.linalg.norm(positions - parent.center[None, :], axis=1)
        idx = int(np.argmin(dists))
        for ci in sorted(range(max(0, idx - 3), min(len(samples), idx + 4)),
                         key=lambda i: dists[i]):
            s = samples[ci]
            rmax = float(s.radius) - radius - 0.3
            if rmax <= 0.0:
                continue
            tangent = s.tangent / max(float(np.linalg.norm(s.tangent)), 1e-9)
            helper = np.array([0, 0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([1, 0, 0.0])
            perp1 = np.cross(helper, tangent)
            perp1 /= max(float(np.linalg.norm(perp1)), 1e-9)
            perp2 = np.cross(tangent, perp1)
            theta = float(rng.uniform(0, 2 * np.pi))
            rho = float(rng.uniform(0, rmax))
            return s.pos + rho * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
        return None

    def _rebuild_mesh(self) -> None:
        res = build_mesh(self.skel, tree=self.tree, stones=self.stones)
        mesh = displace_mesh(res.mesh)
        mesh = color_mesh(mesh, papillae=res.papillae, stones=self.stones)
        self.mesh = mesh
        self.papillae = res.papillae
        w, h = self.renderer.width, self.renderer.height
        fov = self.renderer.fov_y_deg
        self.renderer.release()
        self.renderer = CoaxialRenderer(mesh, w, h, fov)
        self._render_rest_verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
        self._render_rest_normals = np.asarray(mesh.vertex_normals, dtype=np.float32).copy()
        physics_kd = cKDTree(self._physics_rest_verts)
        _, self._phys_to_render_map = physics_kd.query(self._render_rest_verts)

    def get_skeleton(self):
        return {
            name: [tuple(s.pos.tolist()) for s in samples]
            for name, samples in self.skel.items()
        }
