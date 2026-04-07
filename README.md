# endonav-sim

A procedural endoscopic simulator of a kidney collecting system, built as a testbed for autonomous ureteroscope navigation algorithms (junction detection, place recognition, DFS exploration). No CT scan required — the entire pelvicalyceal anatomy is generated procedurally and rendered with a coaxial lighting model derived from published endoscopic-rendering literature.

![hero — looking from the upper major calyx into its three minor infundibula](docs/images/hero_major_upper.png)

## What it does

- **Procedural anatomy.** A 19-node Sampaio Type A1 pelvicalyceal tree (S-curved ureter with three physiologic narrowings → broad renal pelvis → 2 major calyces → 6 minor calyces, each terminating in a papilla) is generated from a tree dictionary, swept with sphere SDFs, blended via smooth-min, carved with papilla domes via smooth-max, and meshed with marching cubes into a single watertight inner-wall mesh.
- **Mucosal detail.** Vertex displacement (3D value noise) creates the bumpy mucosal folds; per-vertex color noise creates vascular streaks; cribriform dark dots on the papillae and sparse Randall's plaque white speckles add the diagnostically distinctive features urologists look for.
- **EndoPBR-style lighting.** Coaxial point light at the camera, EndoPBR spotlight emission `cos^n(θ)/r²`, GGX/Cook-Torrance specular collapsed for the L=V case, wrap-around diffuse with warm SSS bleed for skin-like terminator softening, ACES filmic tonemap. References: [EndoPBR](https://arxiv.org/abs/2502.20669) (arXiv 2502.20669, 2025), [NVIDIA GPU Gems Ch 16](https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-16-real-time-approximations-subsurface-scattering), Dey et al. MICCAI 2005.
- **Phantom-camera matched optics.** 1024×768 output at 4:3, with 870×760 active region letterboxed by black bars. 2× supersample AA, mild radial chromatic aberration, blocky h264-style sensor noise.
- **Ureteroscope kinematics.** `KidneySimulator` models a flexible ureteroscope (ROEN Surgical Zamenix R style) with 3 real DOFs: `advance` along the shaft, `roll` (incremental axial shaft rotation), and `deflection` (absolute single-plane tip bending). The shaft passively conforms to the lumen centerline; aiming is polar — roll picks the bending direction, deflection picks the bending amount. Rolling the shaft also rotates the rendered camera image, just like a real scope. Exposes `reset / render / command(advance_mm, roll_deg, deflection_deg) / follow_skeleton / get_skeleton`. `render()` returns RGB + metric depth + clearance + current tree node + progress. SDF-based collision detection prevents the camera from intruding on the wall.

## Example renders

| Inside the upper major calyx | Inside a minor calyx with papilla |
|---|---|
| ![upper major](docs/images/hero_major_upper.png) | ![calyx with papilla](docs/images/calyx_papilla.png) |

| Down the iliac ureter | Anatomy validation grid (3×3 viewpoints) |
|---|---|
| ![ureter](docs/images/ureter.png) | ![validation grid](docs/images/grid.png) |

The grid shows nine canonical viewpoints from the validation suite — distal/iliac/UPJ ureter, the pelvis bifurcation, both major calyces, and three minor calyces. Each tile is a real `KidneySimulator.render()` output at the phantom-camera resolution.

### Skeleton
![skeleton overlay](docs/images/skeleton_overlay.png)

The 3D skeleton (one color per tree node) overlaid on a translucent point cloud of the wall mesh. Each calyx leaf has a small papilla blob carved into the back wall.

### Brightness validation
![brightness 1/r² falloff](docs/images/brightness_falloff.png)

The central pixel patch over a flat-ish wall, sampled at distances 1–25 mm. The rendered curve (solid) tracks the analytic 1/r² reference (dashed) — confirming the inverse-square coaxial light is faithful and that brightness-based proximity perception modules will transfer sim→real.

### Junction detection
![junction detection](docs/images/junction_detection.png)

A simple dark-blob counter (placeholder for the real junction detector) at the three anatomical bifurcation levels:
- Pelvis: ≥2 dark openings (the two major calyces) ✓
- Upper major: ≥3 dark openings (the three minor infundibula) ✓
- Lower major: ≥3 dark openings ✓

## Install

Requires Python 3.10–3.12. Uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone <this-repo> endonav-sim
cd endonav-sim
uv sync               # production deps
uv sync --extra dev   # + ruff, pytest
```

## Quickstart

```python
from endonav_sim.simulator import KidneySimulator

sim = KidneySimulator()                # builds anatomy, mesh, renderer
sim.reset()                             # camera at ureter entry
out = sim.render()                      # dict: rgb, depth, pose, nearest_wall_mm, current_tree_node, current_tree_progress
sim.command(advance_mm=2.0)                            # push the scope 2 mm deeper
sim.command(roll_deg=90.0, deflection_deg=30.0)        # roll 90°, then deflect tip 30°
# returns False (and reverts state) if the new tip pose collides with the wall
sim.follow_skeleton("calyx_u1", 0.5)    # teleport to a skeleton waypoint
```

## Validation scripts

```bash
uv run python -m scripts.validate_grid               # 3x3 anatomical viewpoint grid
uv run python -m scripts.validate_kinematics         # 3x4 roll/deflection grid + invariants
uv run python -m scripts.validate_junction_detector  # dark-blob detection at all bifurcations
uv run python -m scripts.validate_brightness_falloff # 1/r² falloff plot
uv run python -m scripts.validate_flythrough         # MP4 fly-through of the entire DFS
uv run python -m scripts.visualize_skeleton          # 3D skeleton + mesh point cloud
```

Outputs are written to the repo root (and gitignored).

## Layout

```
endonav_sim/
  tree.py            Sampaio Type A1 anatomy as a dict of segments
  skeleton.py        Walks the tree, builds per-node 1mm-spaced waypoints with tangents
  mesh_gen.py        Implicit swept-sphere field, smooth-min/max, marching cubes
  texture.py         Value-noise displacement, vertex coloring, cribriform, plaque
  collision.py       SDF clearance check (mesh.contains + closest_point)
  renderer.py        moderngl renderer, SSAA, two-pass coaxial + endoscope post
  simulator.py       Public KidneySimulator API
  shader/
    coaxial.{vert,frag}     EndoPBR-style coaxial BRDF
    postprocess.{vert,frag} Letterbox + chroma + h264-style noise resolve
scripts/
  validate_grid.py
  validate_junction_detector.py
  validate_brightness_falloff.py
  validate_flythrough.py
  visualize_skeleton.py
docs/images/         README assets
```

## Dev

```bash
uv run ruff check endonav_sim scripts
uv run ruff format endonav_sim scripts
```

## License

MIT — see [LICENSE](LICENSE).

## References

- **EndoPBR**: Material and Lighting Estimation for Photorealistic Surgical Simulations via Physically-based Rendering. arXiv:2502.20669 (2025). https://arxiv.org/abs/2502.20669
- **Sampaio classification** of pelvicalyceal patterns: Sampaio FJB, *Anatomical background for nephron-sparing surgery in renal cell carcinoma.* J Urol 1992; see [PMC10953598](https://pmc.ncbi.nlm.nih.gov/articles/PMC10953598/) for an endourology-focused narrative review.
- **NVIDIA GPU Gems Ch 16**, Real-Time Approximations to Subsurface Scattering. https://developer.nvidia.com/gpugems/gpugems/part-iii-materials/chapter-16-real-time-approximations-subsurface-scattering
- **Dey et al.**, *Photo-Realistic Tissue Reflectance Modelling for Minimally Invasive Surgical Simulation*, MICCAI 2005.
