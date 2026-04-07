"""Mucosal displacement and per-vertex base coloring.

vnoise is broken on Python 3.12, so we use a small numpy-only value-noise
sampler: a random scalar grid + trilinear interpolation. This is sufficient
for low-frequency organic perturbations of vertex positions and colors."""

from __future__ import annotations

import numpy as np
import trimesh

DISPLACE_AMP_MM = 0.28
# Stay well under the marching-cubes Nyquist limit (~1/(2*grid_step) ≈ 2.27/mm
# at the 0.22 mm grid). 0.65/mm gives ~7 verts/cycle so the noise reads as
# smooth folds rather than per-vertex spikes.
DISPLACE_FREQ = 0.65
COLOR_FREQ = 0.10
BASE_COLOR_RGB = np.array([0.78, 0.47, 0.51])  # pink-red

# Stone composition palette (per-vertex base color before noise modulation).
# Calcium oxalate is the most common (~80%), off-white. Uric acid is
# yellow-orange. Struvite (the staghorn culprit) is chalky tan. Cystine is
# pale waxy yellow.
STONE_COLORS = {
    "calcium_oxalate": np.array([0.86, 0.82, 0.74]),
    "uric_acid": np.array([0.78, 0.60, 0.32]),
    "struvite": np.array([0.82, 0.78, 0.66]),
    "cystine": np.array([0.92, 0.88, 0.62]),
}


class ValueNoise3D:
    """Periodic value noise via a random scalar lattice + trilinear interp."""

    def __init__(self, period: int = 64, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.period = period
        # Pad by 1 so we never need to wrap during interpolation reads.
        self.lattice = rng.random((period + 1, period + 1, period + 1)).astype(np.float64)

    def sample(self, points: np.ndarray, freq: float) -> np.ndarray:
        """Sample noise at world-space points (N,3) at the given frequency.
        Returns values in approximately [-1, 1]."""
        coords = points * freq
        # Wrap into lattice space.
        coords = np.mod(coords, self.period)
        i = np.floor(coords).astype(int)
        f = coords - i
        # Smoothstep for nicer continuity than raw trilinear.
        f = f * f * (3.0 - 2.0 * f)

        ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        L = self.lattice
        c000 = L[ix, iy, iz]
        c100 = L[ix + 1, iy, iz]
        c010 = L[ix, iy + 1, iz]
        c110 = L[ix + 1, iy + 1, iz]
        c001 = L[ix, iy, iz + 1]
        c101 = L[ix + 1, iy, iz + 1]
        c011 = L[ix, iy + 1, iz + 1]
        c111 = L[ix + 1, iy + 1, iz + 1]
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        c = c0 * (1 - fz) + c1 * fz
        return c * 2.0 - 1.0  # remap [0,1] -> [-1,1]


def displace_mesh(
    mesh: trimesh.Trimesh,
    amp: float = DISPLACE_AMP_MM,
    freq: float = DISPLACE_FREQ,
    seed: int = 1,
) -> trimesh.Trimesh:
    """Displace each vertex along its (inward) normal by Perlin-ish noise.

    Mesh winding has already been flipped to face inward, so vertex normals
    point into the lumen — displacing along +normal pushes the wall toward
    the camera (creates protruding folds), -normal pushes it away (recesses).
    We use a centered noise so we get both."""
    noise = ValueNoise3D(seed=seed)
    # Snapshot the (reliable) inward normals before displacement.
    old_n = np.asarray(mesh.vertex_normals).copy()
    delta = noise.sample(mesh.vertices, freq) * amp
    new_verts = mesh.vertices + old_n * delta[:, None]
    faces = np.asarray(mesh.faces)

    # Recompute vertex normals from the DISPLACED geometry so the lighting
    # actually reflects the mucosal bumps. We compute area-weighted face
    # normals, accumulate to vertices, then flip per-vertex against the
    # snapshotted inward direction so orientation stays consistent (this
    # avoids relying on face winding, which is the original pinch bug).
    v = new_verts
    tri = v[faces]  # (F,3,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # area-weighted
    vn = np.zeros_like(v)
    np.add.at(vn, faces[:, 0], fn)
    np.add.at(vn, faces[:, 1], fn)
    np.add.at(vn, faces[:, 2], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = vn / np.where(norms > 1e-12, norms, 1.0)
    # Per-vertex sign-align to the original inward direction.
    flip = np.einsum("ij,ij->i", vn, old_n) < 0.0
    vn[flip] *= -1.0

    # Multiple passes of Laplacian smoothing on the vertex normals: average
    # each vertex's normal with its 1-ring neighbours via the face list.
    # 5 passes on the denser 0.22 mm-grid mesh eliminates the high-frequency
    # wobble that the 1/r^2 lighting amplifies into visible polygon facets at
    # close range, without flattening the displacement-driven mucosal folds.
    for _ in range(5):
        smoothed = np.zeros_like(vn)
        counts = np.zeros(len(v), dtype=np.int64)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ia = faces[:, a]
            ib = faces[:, b]
            np.add.at(smoothed, ia, vn[ib])
            np.add.at(smoothed, ib, vn[ia])
            np.add.at(counts, ia, 1)
            np.add.at(counts, ib, 1)
        smoothed = smoothed + vn * counts[:, None]  # include self
        n2 = np.linalg.norm(smoothed, axis=1, keepdims=True)
        vn = smoothed / np.where(n2 > 1e-12, n2, 1.0)

    mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=faces,
        vertex_normals=vn,
        process=False,
    )
    return mesh


def color_mesh(
    mesh: trimesh.Trimesh,
    base: np.ndarray = BASE_COLOR_RGB,
    freq: float = COLOR_FREQ,
    seed: int = 2,
    papillae: list | None = None,
    stones: list | None = None,
) -> trimesh.Trimesh:
    """Assign per-vertex colors:
    - low-frequency vascular red modulation
    - cribriform dark spots on vertices that lie on a papilla surface
    - sparse Randall's plaque white speckles globally"""
    noise_r = ValueNoise3D(seed=seed)
    noise_g = ValueNoise3D(seed=seed + 7)
    cribr = ValueNoise3D(seed=seed + 13)
    plaque = ValueNoise3D(seed=seed + 23)

    v = np.asarray(mesh.vertices)
    nr = noise_r.sample(v, freq)
    ng = noise_g.sample(v, freq)
    colors = np.tile(base, (len(v), 1))
    colors[:, 0] += 0.14 * nr  # vascular red streaks (a bit stronger now)
    colors[:, 1] += 0.05 * ng
    colors[:, 2] += 0.05 * ng

    # Cribriform: dark dots on the papilla surface. Identify papilla vertices
    # by proximity to any papilla center, then threshold a high-freq noise
    # field so a fraction of those vertices darkens.
    if papillae:
        centers = np.stack([p.center for p in papillae])
        radii = np.array([p.radius for p in papillae])
        # (V, P) distances → mask vertices within 1.3 * radius of any papilla.
        d = np.linalg.norm(v[:, None, :] - centers[None, :, :], axis=2)
        on_papilla = (d < (1.3 * radii)[None, :]).any(axis=1)
        cn = cribr.sample(v, 4.0)
        dot_mask = on_papilla & (cn > 0.55)
        colors[dot_mask] *= 0.45  # darken to ~45% (the duct-opening dots)

    # Randall's plaque: rare bright white-ish speckles, anywhere on the wall.
    pn = plaque.sample(v, 6.0)
    plaque_mask = pn > 0.78
    colors[plaque_mask] = np.minimum(colors[plaque_mask] + 0.45, 1.0)

    # Kidney stones: paint vertices that ended up on (or near) a stone surface
    # with a composition-specific palette. The stones were carved into the
    # mesh by mesh_gen via smooth-max with -stone_field, so the wall now
    # bulges around each stone — we identify those vertices by proximity
    # within stone.radius + 0.4 mm.
    if stones:
        from scipy.spatial import cKDTree

        active = [s for s in stones if not getattr(s, "removed", False)]
        if active:
            stone_centers = np.stack([s.center for s in active])
            stone_radii = np.array([s.radius for s in active])
            stone_palette = np.stack(
                [STONE_COLORS.get(s.composition, np.array([0.85, 0.85, 0.85])) for s in active]
            )
            kd = cKDTree(stone_centers)
            d_nearest, idx_nearest = kd.query(v, k=1, workers=-1)
            within = d_nearest < (stone_radii[idx_nearest] + 0.45)
            if within.any():
                idx = np.where(within)[0]
                stone_color_pick = stone_palette[idx_nearest[idx]]
                noise_s = ValueNoise3D(seed=seed + 31).sample(v[idx], 2.5)
                stone_color_pick = stone_color_pick + 0.05 * noise_s[:, None]
                colors[idx] = np.clip(stone_color_pick, 0.0, 1.0)

    colors = np.clip(colors, 0.0, 1.0)
    rgba = np.concatenate([colors, np.ones((len(v), 1))], axis=1)
    mesh.visual.vertex_colors = (rgba * 255).astype(np.uint8)
    return mesh


if __name__ == "__main__":
    from .mesh_gen import build_mesh

    res = build_mesh()
    m = displace_mesh(res.mesh)
    m = color_mesh(m, papillae=res.papillae)
    print(
        f"verts={len(m.vertices)} watertight={m.is_watertight} colors={m.visual.vertex_colors.shape}"
    )
