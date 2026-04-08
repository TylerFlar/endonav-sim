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

    # Pelvis. Real renal pelvis is a short flattened funnel, ~10 mm tall;
    # the major calyces hinge off it almost perpendicular to the ureter.
    pelvis_length: tuple[float, float] = (10.0, 1.5)
    pelvis_radius_end: tuple[float, float] = (8.0, 1.5)
    pelvis_branch_angle_deg: tuple[float, float] = (8.0, 2.0)

    # Sampaio counts: 70% of kidneys have 7-9 total minor calyces, drained by
    # 2-4 lower pole infundibula in 56.8% of kidneys. We cap the upper end at
    # 6 so each major calyx hosts at most 3 branches per anterior/posterior
    # row, which is the largest count that fits in a half-cone without sibling
    # tip overlap given the infundibulum length and calyx radius.
    minor_calyces_total: tuple[int, int] = (5, 7)
    lower_pole_infundibula: tuple[int, int] = (2, 3)

    # Branch angles for the major calyces. Upper pole bends modestly off the
    # pelvis tangent so the major continues toward the upper pole with a
    # bit of lateral splay; lower pole is derived from IPA (180° - IPA) so
    # the major-axis relationship is anatomically faithful.
    upper_pole_branch_angle: tuple[float, float] = (30.0, 5.0)
    minor_branch_angle: tuple[float, float] = (42.0, 6.0)

    # Infundibulum geometry (Elbahnasy method). In-vivo means: 26 mm length,
    # 7.8 mm width. We compress moderately for the simulator while keeping
    # the resulting pelvicalyceal envelope close to real CT measurements.
    infundibulum_length: tuple[float, float] = (12.0, 2.0)
    infundibulum_width: tuple[float, float] = (3.4, 0.5)

    # Lower pole infundibulopelvic angle (IPA): mean ~59°, can be < 35°.
    # IPA is the *anatomical* angle between the lower-pole infundibular axis
    # and the upper-ureter (pelvis) axis. The lower major's branch_angle in
    # the skeleton is therefore (180° − IPA): IPA=180° → 0° bend (lower
    # parallels upper, anatomically impossible upper limit), IPA=60° → 120°
    # (lower clearly points down-and-out, the typical case), IPA=20° →
    # 160° (lower nearly parallels the ureter, the steep "hard to access"
    # case where instruments can't reach the lower pole).
    infundibulopelvic_angle: tuple[float, float] = (62.0, 18.0)

    # Calyx (leaf chamber containing the papilla). Real minor calyces are
    # short cups (4-7 mm long) wrapping around a renal pyramid's papilla.
    calyx_length: tuple[float, float] = (6.0, 1.0)
    calyx_radius_end: tuple[float, float] = (3.7, 0.4)

    # Branch placement noise. Tight enough that the anterior and posterior
    # rows stay in their respective sagittal planes (real calyces are not
    # 3D-fanned, they form two coplanar rows around the renal pyramid axis).
    azimuth_jitter_deg: float = 6.0

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
    if params.variant not in ("A1", "A2", "B1", "B2"):
        raise ValueError(f"unknown Sampaio variant {params.variant!r}")

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

    # IPA = anatomical angle between lower-pole infundibular axis and the
    # upper-ureter axis. Map directly to branch_angle: a 60° IPA gives a
    # 120° bend off the pelvis tangent, which means the lower major points
    # down-and-out — the normal case.
    ipa = _sample_normal(rng, params.infundibulopelvic_angle, floor=15.0)
    ipa = float(np.clip(ipa, 15.0, 130.0))
    lower_branch_angle = 180.0 - ipa
    upper_branch_angle = _sample_normal(rng, params.upper_pole_branch_angle, floor=35.0)
    upper_branch_angle = float(np.clip(upper_branch_angle, 35.0, 95.0))

    major_radius_start = pelvis_r_end * 0.65
    major_radius_end = pelvis_r_end * 0.55
    major_len = _sample_normal(rng, (16.0, 2.0), floor=10.0)

    # Both majors bend in the SAME plane (azimuth 0 around the pelvis
    # tangent), but at angles on opposite sides of 90°: the upper major
    # bends ~35° from the pelvis tangent (mostly continuing the +pole
    # direction with some lateral splay), the lower major bends 180°-IPA
    # so its tangent flips below horizontal — pointing toward the lower
    # pole. This is the layout that makes upper-pole calyces actually
    # appear in the upper half of the kidney and lower-pole calyces in
    # the lower half.
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
        "branch_azimuth": 0.0,  # same bending plane as upper
        "start_progress": 1.0,
        "length": float(_sample_normal(rng, (12.0, 1.5), floor=8.0)),
        "radius_start": float(major_radius_start),
        "radius_end": float(major_radius_end),
        "children": [],
    }

    # ------------------------------------------------ infundibula + calyces
    def _make_infundibula_for_major(
        major_id: str, label: str, n: int
    ) -> tuple[list[str], list[str]]:
        """Create n infundibulum + calyx pairs under ``major_id``.

        Real calyces are arrayed in TWO planar rows on opposite sides of the
        major-calyx axis: an anterior row (front of the kidney) and a
        posterior row (back). Each row contains 1-3 minor calyces; the
        branches in a row differ by their *start_progress* along the major,
        not by azimuth — they all branch in the same plane. This produces
        the flat, two-row layout you see on a real CT urogram instead of a
        3D candelabra.
        """
        infundibulum_ids: list[str] = []
        calyx_ids: list[str] = []
        n_a = (n + 1) // 2  # anterior row gets the larger half on odd N
        n_b = n - n_a

        def _row_progresses(count: int) -> np.ndarray:
            if count == 0:
                return np.empty(0)
            if count == 1:
                return np.array([0.7])
            return np.linspace(0.30, 0.95, count)

        rows = [
            # (count, start_progress array, azimuth_center)
            # Azimuth 90° vs 270° = anterior vs posterior in the major's
            # local frame, perpendicular to the major's own bending plane.
            # This creates the real anterior/posterior thickness of the
            # pelvicalyceal system in the y-axis.
            (n_a, _row_progresses(n_a), 90.0),
            (n_b, _row_progresses(n_b), 270.0),
        ]
        i = 0
        for row_count, progresses, az_center in rows:
            if row_count == 0:
                continue
            # Within a row, fan the sibling branch angles from sharp (closer
            # to perpendicular) at the proximal end to shallow (closer to
            # parallel) at the distal end. Combined with the staggered
            # start_progress this gives the row a true fan layout instead of
            # parallel translations.
            if row_count == 1:
                row_angles = np.array([_sample_normal(rng, params.minor_branch_angle, 22.0)])
            else:
                row_angles = np.linspace(55.0, 25.0, row_count)
                row_angles = row_angles + rng.normal(0.0, 3.0, size=row_count)
            for k in range(row_count):
                inf_id = f"minf_{label}_{i}"
                cal_id = f"calyx_{label}_{i}"
                infundibulum_ids.append(inf_id)
                calyx_ids.append(cal_id)

                inf_len = _sample_normal(rng, params.infundibulum_length, floor=5.0)
                inf_w = max(
                    _sample_normal(
                        rng, params.infundibulum_width, floor=params.min_infundibulum_width
                    ),
                    params.min_infundibulum_width,
                )
                br_angle = float(np.clip(row_angles[k], 18.0, 65.0))
                azimuth = float(
                    az_center + rng.uniform(-params.azimuth_jitter_deg, params.azimuth_jitter_deg)
                )

                tree[inf_id] = {
                    "parent": major_id,
                    "branch_angle": float(br_angle),
                    "branch_azimuth": azimuth,
                    "start_progress": float(progresses[k]),
                    "length": float(inf_len),
                    "radius_start": float(inf_w),
                    "radius_end": float(inf_w * 0.85),
                    "children": [cal_id],
                }
                tree[major_id]["children"].append(inf_id)

                cal_len = _sample_normal(rng, params.calyx_length, floor=4.0)
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

    upper_inf_ids, upper_cal_ids = _make_infundibula_for_major("major_upper", "upper", n_upper)
    lower_inf_ids, lower_cal_ids = _make_infundibula_for_major("major_lower", "lower", n_lower)

    major_node_ids = ["major_upper", "major_lower"]
    infundibulum_node_ids = upper_inf_ids + lower_inf_ids
    calyx_node_ids = upper_cal_ids + lower_cal_ids
    realized_branch_counts = {"major_upper": n_upper, "major_lower": n_lower}

    # ------------------------------------------------------ Sampaio variants
    # A1 (the default) is now built. The other three layer additions or
    # reparenting on top of it.
    if params.variant == "A2":
        # A2: identical layout to A1 except one minor calyx near the kidney
        # equator "crosses over" — its infundibulum drains into the opposite
        # major. Pick the most distal anterior infundibulum of the smaller
        # major and reparent it onto the other major.
        donor_major = "major_upper" if n_upper >= n_lower else "major_lower"
        recipient_major = "major_lower" if donor_major == "major_upper" else "major_upper"
        # Find the donor's most-distal anterior infundibulum (largest start_progress)
        donor_infs = [
            nid
            for nid in tree[donor_major]["children"]
            if abs(tree[nid].get("branch_azimuth", 0.0) - 90.0) < 30.0  # anterior row
        ]
        if donor_infs:
            crossing = max(donor_infs, key=lambda nid: tree[nid].get("start_progress", 1.0))
            tree[donor_major]["children"].remove(crossing)
            tree[recipient_major]["children"].append(crossing)
            tree[crossing]["parent"] = recipient_major
            # Anchor it near the recipient's middle so it physically branches
            # from the contralateral major rather than reaching across space.
            tree[crossing]["start_progress"] = 0.55
            realized_branch_counts[donor_major] -= 1
            realized_branch_counts[recipient_major] += 1

    elif params.variant == "B1":
        # B1: a third major calyx drains the middle zone of the kidney
        # directly off the renal pelvis, between the upper and lower poles.
        # It branches laterally (perpendicular to the upper/lower bending
        # plane) and hosts 1-3 of its own infundibula.
        n_middle = max(1, _sample_int(rng, (1, 3)))
        middle_branch_angle = 80.0  # nearly perpendicular to pelvis tangent
        tree["major_middle"] = {
            "parent": "pelvis",
            "branch_angle": middle_branch_angle,
            "branch_azimuth": 90.0,  # lateral, perpendicular to upper/lower plane
            "start_progress": 0.80,
            "length": float(_sample_normal(rng, (12.0, 1.5), floor=8.0)),
            "radius_start": float(major_radius_start),
            "radius_end": float(major_radius_end),
            "children": [],
        }
        tree["pelvis"]["children"].append("major_middle")
        middle_inf_ids, middle_cal_ids = _make_infundibula_for_major(
            "major_middle", "middle", n_middle
        )
        major_node_ids.append("major_middle")
        infundibulum_node_ids += middle_inf_ids
        calyx_node_ids += middle_cal_ids
        realized_branch_counts["major_middle"] = n_middle
        n_total += n_middle

    elif params.variant == "B2":
        # B2: 1-3 minor calyces drain the middle zone DIRECTLY into the
        # renal pelvis, with no intervening major chamber. Each one is just
        # an infundibulum + calyx pair attached to the pelvis itself,
        # branching laterally between the upper and lower poles.
        n_middle = max(1, _sample_int(rng, (1, 3)))
        # Spread them along the pelvis at progressive start positions, all
        # branching laterally (azimuth 90, perpendicular to upper/lower plane).
        progresses = [0.80] if n_middle == 1 else list(np.linspace(0.55, 0.95, n_middle))
        b2_inf_ids: list[str] = []
        b2_cal_ids: list[str] = []
        for k in range(n_middle):
            inf_id = f"minf_middle_{k}"
            cal_id = f"calyx_middle_{k}"
            inf_len = _sample_normal(rng, params.infundibulum_length, floor=5.0)
            inf_w = max(
                _sample_normal(rng, params.infundibulum_width, floor=params.min_infundibulum_width),
                params.min_infundibulum_width,
            )
            tree[inf_id] = {
                "parent": "pelvis",
                "branch_angle": 75.0,
                "branch_azimuth": 90.0 + float(rng.uniform(-8.0, 8.0)),
                "start_progress": float(progresses[k]),
                "length": float(inf_len),
                "radius_start": float(inf_w),
                "radius_end": float(inf_w * 0.85),
                "children": [cal_id],
            }
            tree["pelvis"]["children"].append(inf_id)
            cal_len = _sample_normal(rng, params.calyx_length, floor=4.0)
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
            b2_inf_ids.append(inf_id)
            b2_cal_ids.append(cal_id)
        infundibulum_node_ids += b2_inf_ids
        calyx_node_ids += b2_cal_ids
        n_total += n_middle

    meta = AnatomyMeta(
        seed=params.seed,
        variant=params.variant,
        ureter_node_ids=["ureter_distal", "ureter_iliac", "ureter_proximal", "ureter_upj"],
        pelvis_id="pelvis",
        major_node_ids=major_node_ids,
        infundibulum_node_ids=infundibulum_node_ids,
        calyx_node_ids=calyx_node_ids,
        realized_branch_counts=realized_branch_counts,
        infundibulopelvic_angle_deg=ipa,
        minor_calyces_total=n_total,
        n_dead_ends=n_total,
    )
    return tree, meta


__all__ = ["AnatomyParams", "AnatomyMeta", "generate_anatomy"]
