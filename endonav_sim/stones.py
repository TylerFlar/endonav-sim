"""Procedural kidney stone generation.

Stones are placed inside the lumen of a procedurally-generated kidney with
clinically realistic size, composition and location distributions. The
result is a list of :class:`Stone` objects that the simulator carves into
the rendered mesh and tracks for capture/removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .anatomy import AnatomyMeta
from .skeleton import Skeleton

Composition = Literal["calcium_oxalate", "uric_acid", "struvite", "cystine"]


@dataclass
class Stone:
    id: str
    center: np.ndarray  # (3,) world position (inside lumen)
    radius: float  # mm
    node_id: str  # which node it sits in
    composition: Composition
    removed: bool = False  # set True after a successful capture
    is_fragment: bool = False  # produced by a laser fragmentation


@dataclass
class StoneParams:
    seed: int = 0
    count: tuple[int, int] = (1, 8)
    # Lognormal radius sampler. mu/sigma chosen so exp(mu) ≈ 4 mm typical
    # diameter (radius ≈ 2 mm) with a long tail toward larger stones.
    size_mm: tuple[float, float] = (2.0, 20.0)  # diameter clip
    size_lognormal_mu_sigma: tuple[float, float] = (1.4, 0.55)
    composition_weights: dict[str, float] = field(
        default_factory=lambda: {
            "calcium_oxalate": 0.80,
            "struvite": 0.10,
            "uric_acid": 0.09,
            "cystine": 0.01,
        }
    )
    location_weights: dict[str, float] = field(
        default_factory=lambda: {
            "lower_pole_calyx": 0.50,
            "upper_pole_calyx": 0.20,
            "infundibulum": 0.15,
            "pelvis": 0.10,
            "ureter_upj": 0.05,
        }
    )
    staghorn_probability: float = 0.05


def _sample_int(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    w = np.asarray([weights[k] for k in keys], dtype=np.float64)
    w = w / w.sum()
    return str(rng.choice(keys, p=w))


def _candidate_nodes_for_location(location: str, meta: AnatomyMeta) -> list[str]:
    """Return the list of node ids that are valid hosts for a stone of the
    requested anatomical location."""
    if location == "lower_pole_calyx":
        # Calyces under major_lower
        return [c for c in meta.calyx_node_ids if c.startswith("calyx_lower")]
    if location == "upper_pole_calyx":
        return [c for c in meta.calyx_node_ids if c.startswith("calyx_upper")]
    if location == "infundibulum":
        return list(meta.infundibulum_node_ids)
    if location == "pelvis":
        return [meta.pelvis_id]
    if location == "ureter_upj":
        return [n for n in meta.ureter_node_ids if n.endswith("upj")]
    return []


def _sample_stone_radius(rng: np.random.Generator, params: StoneParams) -> float:
    mu, sigma = params.size_lognormal_mu_sigma
    diameter = float(rng.lognormal(mean=mu, sigma=sigma))
    diameter = float(np.clip(diameter, params.size_mm[0], params.size_mm[1]))
    return diameter * 0.5


def _place_inside_node(
    rng: np.random.Generator,
    skel: Skeleton,
    node_id: str,
    stone_radius: float,
) -> tuple[np.ndarray, float] | None:
    """Pick a position along the node centerline, offset perpendicularly into
    the lumen, that leaves at least ``stone_radius + clearance`` between the
    stone surface and the wall.

    Returns ``(center, local_radius)`` or ``None`` if the node is too narrow
    for this stone.
    """
    samples = skel[node_id]
    if len(samples) < 2:
        return None
    # Pick a random sample roughly in the middle 80% of the segment
    idx = int(
        rng.integers(
            int(0.1 * len(samples)), max(int(0.9 * len(samples)), int(0.1 * len(samples)) + 1)
        )
    )
    s = samples[idx]
    local_r = float(s.radius)
    clearance = 0.4
    if local_r - stone_radius - clearance < 0.0:
        return None
    # Random perpendicular offset up to (local_r - stone_radius - clearance)
    tangent = s.tangent / float(np.linalg.norm(s.tangent))
    helper = np.array([0.0, 0.0, 1.0]) if abs(tangent[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    perp1 = np.cross(helper, tangent)
    perp1 /= float(np.linalg.norm(perp1))
    perp2 = np.cross(tangent, perp1)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    rmax = local_r - stone_radius - clearance
    rho = float(rng.uniform(0.0, max(rmax, 0.0)))
    offset = rho * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    center = s.pos + offset
    return center, local_r


def generate_stones(
    skel: Skeleton,
    meta: AnatomyMeta,
    params: StoneParams,
) -> list[Stone]:
    """Sample a list of stones for this kidney instance."""
    rng = np.random.default_rng(params.seed)
    n = _sample_int(rng, params.count)
    stones: list[Stone] = []
    next_id = 0

    # Optional staghorn: occupies a major calyx with a single very large stone
    if rng.random() < params.staghorn_probability and meta.calyx_node_ids:
        host = str(rng.choice(meta.calyx_node_ids))
        s = skel[host][len(skel[host]) // 2]
        stag_radius = float(s.radius * 0.85)
        stones.append(
            Stone(
                id=f"stone_{next_id}",
                center=s.pos.copy(),
                radius=stag_radius,
                node_id=host,
                composition="struvite",
            )
        )
        next_id += 1
        n = max(0, n - 1)

    attempts = 0
    while len(stones) < n + (1 if next_id > 0 else 0) and attempts < n * 30 + 30:
        attempts += 1
        location = _weighted_choice(rng, params.location_weights)
        candidates = _candidate_nodes_for_location(location, meta)
        if not candidates:
            continue
        node_id = str(rng.choice(candidates))
        stone_radius = _sample_stone_radius(rng, params)
        placement = _place_inside_node(rng, skel, node_id, stone_radius)
        if placement is None:
            # Try a smaller stone in this same node
            stone_radius = max(stone_radius * 0.5, params.size_mm[0] * 0.5)
            placement = _place_inside_node(rng, skel, node_id, stone_radius)
            if placement is None:
                continue
        center, _local_r = placement
        composition: Composition = _weighted_choice(rng, params.composition_weights)  # type: ignore[assignment]
        stones.append(
            Stone(
                id=f"stone_{next_id}",
                center=center,
                radius=stone_radius,
                node_id=node_id,
                composition=composition,
            )
        )
        next_id += 1
    return stones


__all__ = ["Stone", "StoneParams", "generate_stones", "Composition"]
