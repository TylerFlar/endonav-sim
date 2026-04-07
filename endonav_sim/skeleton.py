"""Build per-segment centerline waypoints (the skeleton) from a tree dict.

Tree dicts come from :func:`endonav_sim.anatomy.generate_anatomy`. Each
node has ``parent``, ``length``, ``radius_start``, ``radius_end``,
``children``, and (for non-root nodes) ``branch_angle`` / ``branch_azimuth``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_STEP_MM = 1.0


def root_node(tree: dict[str, dict]) -> str:
    """Return the unique node in ``tree`` with no parent."""
    roots = [n for n, d in tree.items() if d.get("parent") is None]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {roots}")
    return roots[0]


def dfs_order(tree: dict[str, dict]) -> list[str]:
    """Depth-first traversal order starting at the root."""
    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for c in tree[name]["children"]:
            visit(c)

    visit(root_node(tree))
    return order


@dataclass
class Sample:
    """One centerline sample: world position, local radius, unit tangent."""

    pos: np.ndarray  # (3,)
    radius: float
    tangent: np.ndarray  # (3,)


# A node's skeleton is the ordered list of samples from its start (at the
# parent's end point) to its end. Adjacent nodes share an endpoint sample
# only conceptually — each list is self-contained for that node.
Skeleton = dict[str, list[Sample]]


def _orthonormal_basis(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors x, y orthogonal to z (and to each other)."""
    z = z / np.linalg.norm(z)
    # Pick a helper not parallel to z.
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y


def _rotate_axis(parent_tangent: np.ndarray, angle_deg: float, azimuth_deg: float) -> np.ndarray:
    """Tilt parent_tangent by `angle` around an axis in the perpendicular plane
    chosen by `azimuth`. Returns a new unit vector."""
    if angle_deg == 0.0:
        return parent_tangent / np.linalg.norm(parent_tangent)
    x, y = _orthonormal_basis(parent_tangent)
    az = np.deg2rad(azimuth_deg)
    # Axis to rotate around lies in the (x, y) plane perpendicular to parent.
    rot_axis = np.cos(az) * x + np.sin(az) * y
    # Rodrigues' rotation formula.
    theta = np.deg2rad(angle_deg)
    k = rot_axis
    v = parent_tangent / np.linalg.norm(parent_tangent)
    rotated = (
        v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
    )
    return rotated / np.linalg.norm(rotated)


def _sample_segment(
    start: np.ndarray,
    tangent: np.ndarray,
    length: float,
    r0: float,
    r1: float,
    step: float = SAMPLE_STEP_MM,
) -> list[Sample]:
    n = max(2, int(np.ceil(length / step)) + 1)
    ts = np.linspace(0.0, 1.0, n)
    out: list[Sample] = []
    for t in ts:
        pos = start + tangent * (t * length)
        radius = (1.0 - t) * r0 + t * r1
        out.append(Sample(pos=pos, radius=float(radius), tangent=tangent.copy()))
    return out


def build_skeleton(tree: dict[str, dict]) -> Skeleton:
    """Walk the tree and produce per-node centerline samples in world space.

    Children may branch from any point along the parent's centerline by
    setting ``start_progress`` (fraction of parent length, 0..1; default 1.0
    = parent's endpoint). This is how anterior/posterior infundibular rows
    are staggered along a major calyx.
    """
    skel: Skeleton = {}
    root = root_node(tree)

    def _interp_parent(
        parent_samples: list[Sample], progress: float
    ) -> tuple[np.ndarray, np.ndarray]:
        progress = float(np.clip(progress, 0.0, 1.0))
        idx_f = progress * (len(parent_samples) - 1)
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, len(parent_samples) - 1)
        t = idx_f - i0
        pos = (1 - t) * parent_samples[i0].pos + t * parent_samples[i1].pos
        tan = parent_samples[i1].tangent
        return pos, tan

    def visit(name: str, start_pos: np.ndarray, parent_tangent: np.ndarray | None) -> None:
        node = tree[name]
        if parent_tangent is None:
            tangent = np.array([0.0, 0.0, 1.0])
        else:
            tangent = _rotate_axis(
                parent_tangent,
                float(node.get("branch_angle", 0.0)),
                float(node.get("branch_azimuth", 0.0)),
            )
        samples = _sample_segment(
            start_pos,
            tangent,
            float(node["length"]),
            float(node["radius_start"]),
            float(node["radius_end"]),
        )
        skel[name] = samples
        for child in node["children"]:
            child_node = tree[child]
            sp = float(child_node.get("start_progress", 1.0))
            child_start, child_parent_tan = _interp_parent(samples, sp)
            visit(child, child_start, child_parent_tan)

    visit(root, np.zeros(3), None)
    return skel


def flatten_skeleton(skel: Skeleton) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Flatten all samples for KD-tree lookup.

    Returns (positions[N,3], radii[N], node_names[N], progress[N]) where
    progress is in [0,1] along that node's segment."""
    positions: list[np.ndarray] = []
    radii: list[float] = []
    names: list[str] = []
    progress: list[float] = []
    for name, samples in skel.items():
        n = len(samples)
        for i, s in enumerate(samples):
            positions.append(s.pos)
            radii.append(s.radius)
            names.append(name)
            progress.append(i / max(1, n - 1))
    return (
        np.asarray(positions, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
        names,
        np.asarray(progress, dtype=np.float64),
    )
