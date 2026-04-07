"""Anatomical tree definition for the kidney collecting system.

Models a Sampaio Type A1 pelvicalyceal system (the most common variant,
~60-70% of human kidneys), with two major calyceal groups (upper + lower
pole) draining into a broad renal pelvis. The middle zone is drained by
calyces of either pole. The ureter is broken into 4 sub-segments to give
it the characteristic S-curve and the three physiologic narrowings (UVJ,
iliac crossing, UPJ) that an endoscope would feel during navigation.

References:
- Sampaio classification of pelvicalyceal patterns
- Real anatomy: ureter ~26 cm with 3 narrowings; 7-13 minor calyces per kidney;
  major calyces formed by ~2-4 minor calyces converging.
The ureter is intentionally compressed (~80 mm vs 260 mm in vivo) so the
sim is tractable to traverse but still has its qualitative shape.
"""

from __future__ import annotations

# All units: millimeters / degrees.
# Root segment (no parent) defines the world origin and lies along +Z.
# Child segments specify branch_angle (deg from parent end-tangent) and
# branch_azimuth (deg around the parent end-tangent).
TREE: dict[str, dict] = {
    # ---- ureter: four sub-segments giving an S-curve + three narrowings ----
    # 1. distal ureter (just above the bladder / UVJ — narrowing #1)
    "ureter_distal": {
        "parent": None,
        "length": 22.0,
        "radius_start": 1.6,  # UVJ narrowing
        "radius_end": 2.2,
        "children": ["ureter_iliac"],
    },
    # 2. iliac crossing (slight bend over the iliac vessels — narrowing #2)
    "ureter_iliac": {
        "parent": "ureter_distal",
        "branch_angle": 18.0,
        "branch_azimuth": 0.0,
        "length": 20.0,
        "radius_start": 1.7,  # iliac narrowing
        "radius_end": 2.3,
        "children": ["ureter_proximal"],
    },
    # 3. proximal ureter (counter-bend back — completes the S)
    "ureter_proximal": {
        "parent": "ureter_iliac",
        "branch_angle": 22.0,
        "branch_azimuth": 180.0,
        "length": 22.0,
        "radius_start": 2.3,
        "radius_end": 2.6,
        "children": ["ureter_upj"],
    },
    # 4. UPJ (third narrowing) joining the renal pelvis
    "ureter_upj": {
        "parent": "ureter_proximal",
        "branch_angle": 12.0,
        "branch_azimuth": 30.0,
        "length": 14.0,
        "radius_start": 2.0,  # UPJ narrowing
        "radius_end": 3.5,
        "children": ["pelvis"],
    },
    # ---- renal pelvis: broad funnel ---------------------------------------
    "pelvis": {
        "parent": "ureter_upj",
        "branch_angle": 8.0,
        "branch_azimuth": 0.0,
        "length": 14.0,
        "radius_start": 5.5,
        "radius_end": 8.0,
        "children": ["major_upper", "major_lower"],
    },
    # ---- two major calyces (Sampaio Type A1) ------------------------------
    "major_upper": {
        "parent": "pelvis",
        "branch_angle": 38.0,
        "branch_azimuth": 0.0,  # upper pole = +X side, "above"
        "length": 12.0,
        "radius_start": 5.0,
        "radius_end": 4.2,
        "children": ["minf_u1", "minf_u2", "minf_u3"],
    },
    "major_lower": {
        "parent": "pelvis",
        "branch_angle": 42.0,
        "branch_azimuth": 180.0,  # lower pole = opposite side
        "length": 13.0,
        "radius_start": 5.0,
        "radius_end": 4.2,
        "children": ["minf_l1", "minf_l2", "minf_l3"],
    },
    # ---- minor infundibula off the upper major ----------------------------
    "minf_u1": {
        "parent": "major_upper",
        "branch_angle": 35.0,
        "branch_azimuth": 0.0,
        "length": 11.0,
        "radius_start": 3.6,
        "radius_end": 3.0,
        "children": ["calyx_u1"],
    },
    "minf_u2": {
        "parent": "major_upper",
        "branch_angle": 8.0,
        "branch_azimuth": 120.0,
        "length": 12.0,
        "radius_start": 3.6,
        "radius_end": 3.0,
        "children": ["calyx_u2"],
    },
    "minf_u3": {
        "parent": "major_upper",
        "branch_angle": 38.0,
        "branch_azimuth": 240.0,
        "length": 10.0,
        "radius_start": 3.6,
        "radius_end": 3.0,
        "children": ["calyx_u3"],
    },
    # ---- minor infundibula off the lower major ----------------------------
    "minf_l1": {
        "parent": "major_lower",
        "branch_angle": 38.0,
        "branch_azimuth": 0.0,
        "length": 12.0,
        "radius_start": 3.6,
        "radius_end": 2.9,
        "children": ["calyx_l1"],
    },
    "minf_l2": {
        "parent": "major_lower",
        "branch_angle": 10.0,
        "branch_azimuth": 130.0,
        "length": 13.0,
        "radius_start": 3.6,
        "radius_end": 2.9,
        "children": ["calyx_l2"],
    },
    "minf_l3": {
        "parent": "major_lower",
        "branch_angle": 36.0,
        "branch_azimuth": 230.0,
        "length": 11.0,
        "radius_start": 3.6,
        "radius_end": 2.9,
        "children": ["calyx_l3"],
    },
    # ---- minor calyces (each gets a papilla) ------------------------------
    "calyx_u1": {
        "parent": "minf_u1",
        "branch_angle": 0.0,
        "branch_azimuth": 0.0,
        "length": 9.0,
        "radius_start": 3.0,
        "radius_end": 4.8,
        "children": [],
    },
    "calyx_u2": {
        "parent": "minf_u2",
        "branch_angle": 0.0,
        "branch_azimuth": 0.0,
        "length": 9.0,
        "radius_start": 3.0,
        "radius_end": 4.8,
        "children": [],
    },
    "calyx_u3": {
        "parent": "minf_u3",
        "branch_angle": 0.0,
        "branch_azimuth": 0.0,
        "length": 9.0,
        "radius_start": 3.0,
        "radius_end": 4.8,
        "children": [],
    },
    "calyx_l1": {
        "parent": "minf_l1",
        "branch_angle": 0.0,
        "branch_azimuth": 0.0,
        "length": 10.0,
        "radius_start": 2.9,
        "radius_end": 4.8,
        "children": [],
    },
    "calyx_l2": {
        "parent": "minf_l2",
        "branch_angle": 0.0,
        "branch_azimuth": 0.0,
        "length": 9.0,
        "radius_start": 2.9,
        "radius_end": 4.8,
        "children": [],
    },
    "calyx_l3": {
        "parent": "minf_l3",
        "branch_angle": 0.0,
        "branch_azimuth": 0.0,
        "length": 10.0,
        "radius_start": 2.9,
        "radius_end": 4.8,
        "children": [],
    },
}


def root_node(tree: dict[str, dict] = TREE) -> str:
    """Return the unique node with no parent."""
    roots = [n for n, d in tree.items() if d.get("parent") is None]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {roots}")
    return roots[0]


def dfs_order(tree: dict[str, dict] = TREE) -> list[str]:
    """Depth-first traversal order starting at the root."""
    order: list[str] = []

    def visit(name: str) -> None:
        order.append(name)
        for c in tree[name]["children"]:
            visit(c)

    visit(root_node(tree))
    return order
