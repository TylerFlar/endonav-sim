"""SOFA scene using the canonical BeamAdapter InterventionalRadiologyController.

Based on the official ``SingleBeamDeploymentCollision.py`` example from the
BeamAdapter plugin. The scope is modeled as:

  - **RodStraightSection** (shaft, ~90mm) + **RodSpireSection** (tip, ~10mm
    with a J-curve rest shape for pre-curved articulation)
  - **WireRestShape** composing the sections
  - **InterventionalRadiologyController** handling advance/retract/rotate
  - **RestShapeSpringsForceField** anchoring undeployed portions (anti-drift)
  - **BTDLinearSolver** (O(n) for beam chains, numerically stable)
  - **MultiAdaptiveBeamMapping** for collision geometry

The kidney is a surface-FEM deformable mesh with friction contact.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .sofa_setup import init_sofa

init_sofa()

import Sofa  # noqa: E402
import Sofa.Simulation  # noqa: E402

# ---- Scope geometry (modeled after a 9Fr flexible ureteroscope) ----------
SHAFT_LENGTH_MM = 90.0
TIP_LENGTH_MM = 10.0
SCOPE_RADIUS_MM = 1.5  # 3.0 mm OD / 2
SCOPE_YOUNG_MODULUS = 20000  # Effective stiffness (matches BeamAdapter examples)
SCOPE_TIP_YOUNG_MODULUS = 10000  # Tip is softer (more flexible)
SCOPE_DENSITY = 0.00000155  # Matches BeamAdapter canonical density
SCOPE_POISSON = 0.3
SPIRE_DIAMETER = 15.0  # Pre-curved tip: 15mm diameter J-curve
SPIRE_HEIGHT = 0.0  # Planar curve (not helical)
N_BEAMS_SHAFT = 30
N_BEAMS_TIP = 5
N_EDGES_COLLIS_SHAFT = 30
N_EDGES_COLLIS_TIP = 5
N_BEAM_NODES = 41  # RegularGridTopology nx (node pool for deployment)

# ---- Kidney ---------------------------------------------------------------
KIDNEY_YOUNG_MODULUS = 30_000  # Pa
KIDNEY_POISSON_RATIO = 0.45
KIDNEY_DENSITY = 1040.0

# ---- Simulation -----------------------------------------------------------
DT = 0.01
COLLISION_ALARM_DIST = 2.0
COLLISION_CONTACT_DIST = 0.5
FRICTION_MU = 0.3

REQUIRED_PLUGINS = [
    "BeamAdapter",
    "SoftRobots",
    "Sofa.Component.AnimationLoop",
    "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection",
    "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact",
    "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.IO.Mesh",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.SolidMechanics.Spring",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Visual",
]


def _decimate_for_physics(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    import fast_simplification as fs
    reduction = 1.0 - (target_faces / len(mesh.faces))
    verts, faces = fs.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_reduction=float(np.clip(reduction, 0.01, 0.99)),
    )
    decimated = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if not decimated.is_watertight:
        decimated.fill_holes()
        if not decimated.is_watertight:
            decimated = decimated.convex_hull
    return decimated


def create_sofa_scene(
    root: Sofa.Core.Node,
    mesh: trimesh.Trimesh,
    skeleton: dict,
    tree: dict,
    stones: list,
    starting_pos: list[float] | None = None,
    **kwargs,
) -> dict:
    """Build the SOFA scene using the BeamAdapter interventional radiology pattern.

    ``starting_pos`` is [x, y, z, qx, qy, qz, qw] for the insertion point.
    Defaults to the first skeleton sample's position oriented along +Z.
    """
    # ---- Plugins + solver pipeline -----------------------------------------
    root.addObject("RequiredPlugin", pluginName=REQUIRED_PLUGINS)
    root.addObject("FreeMotionAnimationLoop")
    root.addObject("DefaultVisualManagerLoop")
    root.addObject("LCPConstraintSolver", mu=str(FRICTION_MU),
                    tolerance=1e-10, maxIt=1000, build_lcp=False)
    root.addObject("CollisionPipeline", depth=6)
    root.addObject("BruteForceBroadPhase")
    root.addObject("BVHNarrowPhase")
    root.addObject("LocalMinDistance",
                    alarmDistance=COLLISION_ALARM_DIST,
                    contactDistance=COLLISION_CONTACT_DIST,
                    angleCone=0.2)
    root.addObject("CollisionResponse", response="FrictionContactConstraint")
    root.gravity = [0.0, 0.0, 0.0]
    root.dt = DT

    # ---- Compute insertion point from skeleton -----------------------------
    if starting_pos is None:
        from .skeleton import root_node
        root_name = root_node(tree)
        samples = skeleton[root_name]
        pos = samples[0].pos
        tangent = samples[0].tangent
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
        # Quaternion that rotates +X to align with tangent
        # (BeamAdapter convention: instrument extends along +X)
        x_axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(tangent, x_axis, atol=1e-6):
            quat = [0.0, 0.0, 0.0, 1.0]
        elif np.allclose(tangent, -x_axis, atol=1e-6):
            quat = [0.0, 0.0, 1.0, 0.0]
        else:
            axis = np.cross(x_axis, tangent)
            axis /= max(float(np.linalg.norm(axis)), 1e-9)
            angle = float(np.arccos(np.clip(np.dot(x_axis, tangent), -1, 1)))
            half = angle * 0.5
            quat = list(axis * np.sin(half)) + [float(np.cos(half))]
        starting_pos = list(pos) + quat

    # ---- Wire rest shape (shaft + tip) -------------------------------------
    edge_topo = root.addChild("EdgeTopology")
    edge_topo.addObject("RodStraightSection", name="StraightSection",
                         length=SHAFT_LENGTH_MM, radius=SCOPE_RADIUS_MM,
                         nbBeams=N_BEAMS_SHAFT, nbEdgesCollis=N_EDGES_COLLIS_SHAFT,
                         nbEdgesVisu=100,
                         youngModulus=SCOPE_YOUNG_MODULUS,
                         massDensity=SCOPE_DENSITY,
                         poissonRatio=SCOPE_POISSON)
    edge_topo.addObject("RodSpireSection", name="SpireSection",
                         length=TIP_LENGTH_MM, radius=SCOPE_RADIUS_MM,
                         nbBeams=N_BEAMS_TIP, nbEdgesCollis=N_EDGES_COLLIS_TIP,
                         nbEdgesVisu=50,
                         spireDiameter=SPIRE_DIAMETER,
                         spireHeight=SPIRE_HEIGHT,
                         youngModulus=SCOPE_TIP_YOUNG_MODULUS,
                         massDensity=SCOPE_DENSITY,
                         poissonRatio=SCOPE_POISSON)
    edge_topo.addObject("WireRestShape", name="BeamRestShape", template="Rigid3d",
                         wireMaterials="@StraightSection @SpireSection")
    edge_topo.addObject("EdgeSetTopologyContainer", name="meshLines")
    edge_topo.addObject("EdgeSetTopologyModifier")
    edge_topo.addObject("EdgeSetGeometryAlgorithms", template="Rigid3d")
    edge_topo.addObject("MechanicalObject", name="dofTopo", template="Rigid3d")

    # ---- Beam mechanics (the core physics) ---------------------------------
    beam = root.addChild("BeamModel")
    beam.addObject("EulerImplicitSolver", rayleighStiffness=0.2, rayleighMass=0.1)
    beam.addObject("BTDLinearSolver", verification=False, subpartSolve=False)
    beam.addObject("RegularGridTopology", name="MeshLines",
                    nx=N_BEAM_NODES, ny=1, nz=1,
                    xmax=0, xmin=0, ymin=0, ymax=0, zmax=0, zmin=0,
                    p0=[0, 0, 0])
    beam.addObject("MechanicalObject", name="DOFs", template="Rigid3d",
                    showIndices=False)
    beam.addObject("WireBeamInterpolation", name="BeamInterpolation",
                    WireRestShape="@../EdgeTopology/BeamRestShape")
    beam.addObject("AdaptiveBeamForceFieldAndMass", name="BeamForceField",
                    massDensity=SCOPE_DENSITY,
                    interpolation="@BeamInterpolation")
    beam.addObject("InterventionalRadiologyController", name="DeployController",
                    template="Rigid3d",
                    instruments="BeamInterpolation",
                    topology="@MeshLines",
                    startingPos=starting_pos,
                    xtip=[0],
                    rotationInstrument=[0],
                    step=2.0,  # mm per keyboard step
                    speed=0.0,  # continuous speed (we set this programmatically)
                    listening=True,
                    controlledInstrument=0)
    beam.addObject("LinearSolverConstraintCorrection", wire_optimization=True)
    beam.addObject("FixedProjectiveConstraint", indices=0)
    beam.addObject("RestShapeSpringsForceField",
                    points="@DeployController.indexFirstNode",
                    angularStiffness=1e8, stiffness=1e8)

    # Collision model
    beam_col = beam.addChild("CollisionModel")
    beam_col.activated = True
    beam_col.addObject("EdgeSetTopologyContainer", name="collisEdgeSet")
    beam_col.addObject("EdgeSetTopologyModifier")
    beam_col.addObject("MechanicalObject", name="CollisionDOFs")
    beam_col.addObject("MultiAdaptiveBeamMapping",
                         controller="../DeployController",
                         useCurvAbs=True)
    beam_col.addObject("LineCollisionModel", contactDistance=0.0)
    beam_col.addObject("PointCollisionModel", contactDistance=0.0)

    # ---- Kidney (deformable collision mesh) --------------------------------
    physics_mesh = _decimate_for_physics(mesh, 16000)
    physics_verts = np.asarray(physics_mesh.vertices, dtype=np.float64)
    physics_faces = np.asarray(physics_mesh.faces, dtype=np.int32)

    kidney = root.addChild("Kidney")
    kidney.addObject("MeshTopology",
                      position=physics_verts.tolist(),
                      triangles=physics_faces.tolist())
    kidney.addObject("MechanicalObject", name="dofs", template="Vec3d")
    kidney.addObject("TriangleCollisionModel", moving=False, simulated=False)
    kidney.addObject("LineCollisionModel", moving=False, simulated=False)

    # ---- Stone rigid bodies ------------------------------------------------
    stone_dofs_list = []
    active_stones = [s for s in stones if not s.removed]
    for i, stone in enumerate(active_stones):
        stone_node = root.addChild(f"stone_{i}")
        stone_node.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=0.1)
        stone_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
        pos = list(stone.center) + [0.0, 0.0, 0.0, 1.0]
        stone_node.addObject("MechanicalObject", name="dofs", template="Rigid3d",
                              position=[pos])
        stone_node.addObject("UniformMass", totalMass=0.001)
        stone_node.addObject("UncoupledConstraintCorrection")
        stone_node.addObject("SphereCollisionModel",
                              radius=float(stone.radius),
                              selfCollision=False)
        stone_dofs_list.append(stone_node.getObject("dofs"))

    return {
        "beam_dofs": beam.getObject("DOFs"),
        "deploy_controller": beam.getObject("DeployController"),
        "kidney_dofs": kidney.getObject("dofs"),
        "physics_verts": physics_verts,
        "physics_faces": physics_faces,
        "physics_mesh": physics_mesh,
        "stone_dofs": stone_dofs_list,
        "active_stone_indices": [i for i, s in enumerate(stones) if not s.removed],
        "starting_pos": starting_pos,
    }


# ---- Per-step helpers ----------------------------------------------------

def step_sofa(root: Sofa.Core.Node, dt: float | None = None) -> None:
    Sofa.Simulation.animate(root, dt or root.dt.value)


def get_tip_deployed_length(deploy_controller) -> float:
    """Current deployed length (curvilinear abscissa of the tip)."""
    return float(deploy_controller.xtip.value[0])


def set_tip_deployed_length(deploy_controller, length_mm: float) -> None:
    """Set the deployed length (advance the scope to this position)."""
    deploy_controller.xtip.value = [float(length_mm)]


def set_tip_rotation(deploy_controller, rotation_deg: float) -> None:
    """Set the axial rotation of the scope."""
    deploy_controller.rotationInstrument.value = [float(rotation_deg)]


def extract_tip_pose(beam_dofs, deploy_controller) -> np.ndarray:
    """Camera-to-world pose from the most-distal deployed beam node.

    BeamAdapter convention: beam extends along +X, with +Y right and +Z up.
    Camera convention (OpenGL): camera looks along local -Z, +Y up, +X right.
    We remap: camera_forward(-Z) = beam_forward(+X).
    """
    positions = beam_dofs.position.value
    tip = positions[-1]
    pos = np.array(tip[:3], dtype=np.float64)
    qx, qy, qz, qw = tip[3], tip[4], tip[5], tip[6]
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)

    # BeamAdapter +X = forward (deployment direction).
    # Build a camera frame with -Z = beam forward, using a stable world-up
    # to prevent the camera from rolling as the beam curves.
    forward = R[:, 0]  # beam +X = deployment direction
    forward = forward / max(float(np.linalg.norm(forward)), 1e-9)

    # Y is the stable "up" for the anatomy (the ureter runs along Z,
    # so Y is perpendicular to the main traversal direction).
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.95:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward  # camera looks along -Z
    pose[:3, 3] = pos
    return pose


def extract_deformed_vertices(kidney_dofs) -> np.ndarray:
    return np.array(kidney_dofs.position.value, dtype=np.float64)


def extract_stone_positions(stone_dofs_list: list) -> list[np.ndarray]:
    return [np.array(d.position.value[0][:3], dtype=np.float64) for d in stone_dofs_list]


__all__ = [
    "create_sofa_scene",
    "step_sofa",
    "get_tip_deployed_length",
    "set_tip_deployed_length",
    "set_tip_rotation",
    "extract_tip_pose",
    "extract_deformed_vertices",
    "extract_stone_positions",
    "SHAFT_LENGTH_MM",
    "TIP_LENGTH_MM",
    "DT",
]
