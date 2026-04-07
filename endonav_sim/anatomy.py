"""Procedural seeded kidney anatomy generator.

Samples a fresh anatomy from anatomical distributions and returns a tree
dict that :func:`endonav_sim.skeleton.build_skeleton` and
:func:`endonav_sim.mesh_gen.build_mesh` consume directly. Numerical ranges
follow Sampaio / Soni / Elbahnasy / StatPearls (see README references).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Variant = Literal["A1", "A2", "B1", "B2"]


@dataclass
class AnatomyParams:
    """Parameters for procedural kidney generation.

    All `(mean, sd)` pairs are sampled with a Gaussian and then clipped to
    physiological lower bounds. ``seed`` controls reproducibility.
    """

    seed: int = 0
    variant: Variant = "A1"

    # Ureter — TWO narrowings only (UVJ + UPJ); the iliac region is just a
    # bend, NOT a constriction (per Yamashita 2021 CT studies).
    ureter_total_length: tuple[float, float] = (80.0, 8.0)  # compressed from 220-300mm
    ureter_uvj_radius: tuple[float, float] = (1.55, 0.15)  # narrowest, intramural
    ureter_upj_radius: tuple[float, float] = (2.05, 0.20)
    ureter_mid_radius: tuple[float, float] = (2.45, 0.25)
    ureter_iliac_bend_deg: tuple[float, float] = (18.0, 4.0)  # bend only, no narrowing
    ureter_proximal_counterbend_deg: tuple[float, float] = (22.0, 5.0)
    ureter_upj_bend_deg: tuple[float, float] = (12.0, 3.0)

    # Pelvis
    pelvis_length: tuple[float, float] = (14.0, 2.0)
    pelvis_radius_end: tuple[float, float] = (8.0, 1.5)
    pelvis_branch_angle_deg: tuple[float, float] = (8.0, 2.0)

    # Sampaio counts: 70% of kidneys have 7-9 total minor calyces, drained by
    # 2-4 lower pole infundibula in 56.8% of kidneys. We cap the upper end at
    # 6 so each major calyx hosts at most 3 branches per anterior/posterior
    # row, which is the largest count that fits in a half-cone without sibling
    # tip overlap given the infundibulum length and calyx radius.
    minor_calyces_total: tuple[int, int] = (5, 7)
    lower_pole_infundibula: tuple[int, int] = (2, 3)

    # Branch angles for the major calyces and the infundibula coming off them.
    major_branch_angle: tuple[float, float] = (42.0, 6.0)
    minor_branch_angle: tuple[float, float] = (35.0, 6.0)

    # Infundibulum geometry (Elbahnasy method); compressed from in-vivo
    # means of 26mm length / 7.8mm width.
    infundibulum_length: tuple[float, float] = (12.0, 3.0)
    infundibulum_width: tuple[float, float] = (3.6, 0.7)

    # Lower pole infundibulopelvic angle (IPA): mean ~59°, can be < 35°.
    # Used to bias the major_lower branch angle — lower IPA = harder to reach
    # the lower pole.
    infundibulopelvic_angle: tuple[float, float] = (59.0, 18.0)

    # Calyx (leaf chamber containing the papilla). End radius kept modest so
    # adjacent calyx tips don't blob into one another.
    calyx_length: tuple[float, float] = (9.0, 1.5)
    calyx_radius_end: tuple[float, float] = (3.5, 0.4)

    # Branch placement noise. Kept small so anterior and posterior calyx
    # rows don't drift into each other's hemispheres.
    azimuth_jitter_deg: float = 8.0

    # Hard physiological floors
    min_uvj_radius: float = 1.30  # below this a 9 Fr scope can't enter
    min_calyx_radius: float = 2.5
    min_infundibulum_width: float = 2.0


@dataclass
class AnatomyMeta:
    """Realized anatomy metadata. Used by metrics, never visible through the
    realistic command interface."""

    seed: int
    variant: Variant
    ureter_node_ids: list[str]
    pelvis_id: str
    major_node_ids: list[str]
    infundibulum_node_ids: list[str]
    calyx_node_ids: list[str]
    realized_branch_counts: dict[str, int]  # major_id -> number of infundibula
    infundibulopelvic_angle_deg: float  # the realized lower-pole IPA
    minor_calyces_total: int
    n_dead_ends: int


def _sample_normal(rng: np.random.Generator, params: tuple[float, float], floor: float) -> float:
    mean, sd = params
    val = float(rng.normal(mean, sd))
    return max(val, floor)


def _sample_int(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    lo, hi = lo_hi
    return int(rng.integers(lo, hi + 1))


def generate_anatomy(params: AnatomyParams) -> tuple[dict, AnatomyMeta]:
    """Generate a fresh kidney anatomy.

    Returns ``(tree, meta)``. ``tree`` has one entry per anatomical segment;
    each value is a dict with ``parent``, ``length``, ``radius_start``,
    ``radius_end``, ``children``, and (for non-root nodes) ``branch_angle``
    and ``branch_azimuth``.
    """
    if params.variant != "A1":
        raise NotImplementedError(
            f"variant {params.variant!r} not implemented yet — only Sampaio A1 is supported"
        )

    rng = np.random.default_rng(params.seed)
    tree: dict[str, dict] = {}

    # ------------------------------------------------------------------ureter
    total_len = _sample_normal(rng, params.ureter_total_length, floor=40.0)
    # Split the ureter into 4 sub-segments (distal/iliac/proximal/upj) so the
    # mesh can carry an S-curve and the two narrowings without creating
    # spurious intermediate constrictions.
    seg_fracs = np.array([0.275, 0.250, 0.275, 0.200])
    seg_lengths = (total_len * seg_fracs).tolist()

    r_uvj = max(
        _sample_normal(rng, params.ureter_uvj_radius, params.min_uvj_radius), params.min_uvj_radius
    )
    r_upj = _sample_normal(rng, params.ureter_upj_radius, floor=1.6)
    r_mid = _sample_normal(rng, params.ureter_mid_radius, floor=1.8)
    # Distal end -> iliac mid -> proximal mid -> upj. Two narrowings only.
    r_distal_end = (r_uvj + r_mid) * 0.5  # taper from UVJ up to mid

    iliac_bend = _sample_normal(rng, params.ureter_iliac_bend_deg, floor=4.0)
    counterbend = _sample_normal(rng, params.ureter_proximal_counterbend_deg, floor=4.0)
    upj_bend = _sample_normal(rng, params.ureter_upj_bend_deg, floor=2.0)

    tree["ureter_distal"] = {
        "parent": None,
        "length": float(seg_lengths[0]),
        "radius_start": float(r_uvj),
        "radius_end": float(r_distal_end),
        "children": ["ureter_iliac"],
    }
    tree["ureter_iliac"] = {
        "parent": "ureter_distal",
        "branch_angle": float(iliac_bend),
        "branch_azimuth": 0.0,
        "length": float(seg_lengths[1]),
        "radius_start": float(r_distal_end),
        "radius_end": float(r_mid),
        "children": ["ureter_proximal"],
    }
    tree["ureter_proximal"] = {
        "parent": "ureter_iliac",
        "branch_angle": float(counterbend),
        "branch_azimuth": 180.0,  # counter-bend completes the S
        "length": float(seg_lengths[2]),
        "radius_start": float(r_mid),
        "radius_end": float(r_upj * 1.15),
        "children": ["ureter_upj"],
    }
    tree["ureter_upj"] = {
        "parent": "ureter_proximal",
        "branch_angle": float(upj_bend),
        "branch_azimuth": float(rng.uniform(0.0, 60.0)),
        "length": float(seg_lengths[3]),
        "radius_start": float(r_upj),
        "radius_end": float(r_upj * 1.7),
        "children": ["pelvis"],
    }

    # -----------------------------------------------------------------pelvis
    pelvis_len = _sample_normal(rng, params.pelvis_length, floor=8.0)
    pelvis_r_end = _sample_normal(rng, params.pelvis_radius_end, floor=5.0)
    pelvis_branch = _sample_normal(rng, params.pelvis_branch_angle_deg, floor=2.0)
    tree["pelvis"] = {
        "parent": "ureter_upj",
        "branch_angle": float(pelvis_branch),
        "branch_azimuth": 0.0,
        "length": float(pelvis_len),
        "radius_start": float(r_upj * 1.7),
        "radius_end": float(pelvis_r_end),
        "children": ["major_upper", "major_lower"],
    }

    # -------------------------------------------------------- major calyces
    # Total minor calyces, then split between upper and lower poles. The
    # lower pole is drawn from `lower_pole_infundibula`; the upper gets the
    # remainder. Both clipped to be at least 1.
    n_total = _sample_int(rng, params.minor_calyces_total)
    n_lower = _sample_int(rng, params.lower_pole_infundibula)
    n_lower = int(np.clip(n_lower, 1, n_total - 1))
    n_upper = n_total - n_lower

    # Lower IPA biases the lower-pole branch angle. Lower IPA -> sharper bend.
    ipa = _sample_normal(rng, params.infundibulopelvic_angle, floor=15.0)
    ipa = float(np.clip(ipa, 15.0, 110.0))
    # Map IPA to branch_angle: IPA=90° -> 40°, IPA=30° -> 70°, IPA=110° -> 32°.
    # branch_angle = max_branch - 0.5 * (IPA - 30)  ish, simple linear.
    lower_branch_angle = float(np.clip(80.0 - 0.5 * (ipa - 30.0), 30.0, 80.0))
    upper_branch_angle = _sample_normal(rng, params.major_branch_angle, floor=20.0)

    major_radius_start = pelvis_r_end * 0.65
    major_radius_end = pelvis_r_end * 0.55
    major_len = _sample_normal(rng, (12.5, 2.0), floor=8.0)

    # Stagger the two majors along the pelvis: upper branches earlier
    # (anterior side), lower from the distal end. This is closer to real
    # anatomy and prevents the two majors from converging right at the
    # shared pelvis end.
    tree["major_upper"] = {
        "parent": "pelvis",
        "branch_angle": float(upper_branch_angle),
        "branch_azimuth": 0.0,
        "start_progress": 0.55,
        "length": float(major_len),
        "radius_start": float(major_radius_start),
        "radius_end": float(major_radius_end),
        "children": [],
    }
    tree["major_lower"] = {
        "parent": "pelvis",
        "branch_angle": float(lower_branch_angle),
        "branch_azimuth": 180.0,
        "start_progress": 1.0,
        "length": float(_sample_normal(rng, (13.0, 2.0), floor=8.0)),
        "radius_start": float(major_radius_start),
        "radius_end": float(major_radius_end),
        "children": [],
    }

    # ------------------------------------------------ infundibula + calyces
    def _make_infundibula_for_major(major_id: str, n: int) -> tuple[list[str], list[str]]:
        """Create n infundibulum + calyx pairs under ``major_id``.

        Branches are split into the two anatomical rows that real renal
        calyces form: anterior calyces (along the front of the kidney) and
        posterior calyces (along the back). Modeled as two half-cones on
        opposite sides of the major-axis tangent: anterior covers azimuth
        [-90°, +90°] at progress ~0.55 of the major, posterior covers
        [+90°, +270°] at progress ~0.95.
        """
        infundibulum_ids: list[str] = []
        calyx_ids: list[str] = []
        n_a = (n + 1) // 2
        n_b = n - n_a

        # Each row is a half-circle (180°) of azimuths, sampled inside
        # [-90°, +90°] for anterior and [+90°, +270°] for posterior.
        def _half_azimuths(count: int, center: float) -> np.ndarray:
            if count == 0:
                return np.empty(0)
            if count == 1:
                return np.array([center])
            half_span = 80.0  # degrees on each side of `center`
            return center + np.linspace(-half_span, half_span, count)

        rows = [
            # (count, start_progress, azimuths, branch_angle_bias, length_scale)
            # Anterior row branches off well before the major's tip and bends
            # sharply sideways so its calyces stay clear of the posterior row.
            (n_a, 0.40, _half_azimuths(n_a, 0.0), 18.0, 0.85),
            # Posterior row branches at the major's endpoint with a moderate
            # bend, in the opposite hemisphere.
            (n_b, 1.00, _half_azimuths(n_b, 180.0), 0.0, 1.0),
        ]
        i = 0
        for row_count, start_progress, base_azimuths, angle_bias, length_scale in rows:
            if row_count == 0:
                continue
            for k in range(row_count):
                inf_id = f"minf_{major_id.split('_')[1]}_{i}"
                cal_id = f"calyx_{major_id.split('_')[1]}_{i}"
                infundibulum_ids.append(inf_id)
                calyx_ids.append(cal_id)

                inf_len = _sample_normal(rng, params.infundibulum_length, floor=5.0) * length_scale
                inf_w = max(
                    _sample_normal(
                        rng, params.infundibulum_width, floor=params.min_infundibulum_width
                    ),
                    params.min_infundibulum_width,
                )
                br_angle = _sample_normal(rng, params.minor_branch_angle, floor=22.0) + angle_bias
                br_angle = float(min(br_angle, 65.0))
                azimuth = float(
                    base_azimuths[k]
                    + rng.uniform(-params.azimuth_jitter_deg, params.azimuth_jitter_deg)
                )

                tree[inf_id] = {
                    "parent": major_id,
                    "branch_angle": float(br_angle),
                    "branch_azimuth": azimuth,
                    "start_progress": float(start_progress),
                    "length": float(inf_len),
                    "radius_start": float(inf_w),
                    "radius_end": float(inf_w * 0.85),
                    "children": [cal_id],
                }
                tree[major_id]["children"].append(inf_id)

                cal_len = _sample_normal(rng, params.calyx_length, floor=5.0)
                cal_r_end = max(
                    _sample_normal(rng, params.calyx_radius_end, floor=params.min_calyx_radius),
                    params.min_calyx_radius,
                )
                tree[cal_id] = {
                    "parent": inf_id,
                    "branch_angle": 0.0,
                    "branch_azimuth": 0.0,
                    "length": float(cal_len),
                    "radius_start": float(inf_w * 0.85),
                    "radius_end": float(cal_r_end),
                    "children": [],
                }
                i += 1
        return infundibulum_ids, calyx_ids

    upper_inf_ids, upper_cal_ids = _make_infundibula_for_major("major_upper", n_upper)
    lower_inf_ids, lower_cal_ids = _make_infundibula_for_major("major_lower", n_lower)

    meta = AnatomyMeta(
        seed=params.seed,
        variant=params.variant,
        ureter_node_ids=["ureter_distal", "ureter_iliac", "ureter_proximal", "ureter_upj"],
        pelvis_id="pelvis",
        major_node_ids=["major_upper", "major_lower"],
        infundibulum_node_ids=upper_inf_ids + lower_inf_ids,
        calyx_node_ids=upper_cal_ids + lower_cal_ids,
        realized_branch_counts={"major_upper": n_upper, "major_lower": n_lower},
        infundibulopelvic_angle_deg=ipa,
        minor_calyces_total=n_total,
        n_dead_ends=n_total,
    )
    return tree, meta


__all__ = ["AnatomyParams", "AnatomyMeta", "generate_anatomy"]
