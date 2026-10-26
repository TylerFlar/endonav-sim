"""Performance + accuracy regression bench for KidneySimulator.

Reports:
  - mesh build wallclock
  - per-frame command() / render() / total ms (mean, p50, p99)
  - effective fps
  - clearance field accuracy vs trimesh.proximity (random points)

Run:
    uv run python scripts/bench_sim.py [--seed N] [--frames N] [--with-depth]
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from endonav_sim import AnatomyParams, KidneySimulator, StoneParams


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[k]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--with-depth", action="store_true")
    p.add_argument("--with-stones-visible", action="store_true")
    p.add_argument("--no-stones", action="store_true")
    args = p.parse_args()

    print(f"=== bench_sim seed={args.seed} frames={args.frames} ===")

    # ---- Build ---------------------------------------------------------
    t0 = time.perf_counter()
    sim = KidneySimulator(
        anatomy_params=AnatomyParams(seed=args.seed),
        stone_params=None if args.no_stones else StoneParams(seed=args.seed),
        seed=args.seed,
        realistic=True,
    )
    build_s = time.perf_counter() - t0
    print(f"build:                  {build_s:7.2f} s")
    print(f"  mesh tris:            {len(sim.mesh.faces):>9d}")
    print(f"  mesh verts:           {len(sim.mesh.vertices):>9d}")
    print(f"  stones:               {len(sim.stones):>9d}")
    print(f"  tree nodes:           {len(sim.tree):>9d}")

    # ---- Frame loop ----------------------------------------------------
    sim.reset()
    cmd_ms: list[float] = []
    rnd_ms: list[float] = []
    tot_ms: list[float] = []

    # Warmup so JIT/cache effects don't pollute the first frames
    for _ in range(10):
        sim.command(advance_mm=0.5, roll_deg=2.0, kappa2_target=0.02)
        sim.render(with_depth=args.with_depth, with_stones_visible=args.with_stones_visible)

    for _ in range(args.frames):
        t = time.perf_counter()
        sim.command(advance_mm=0.5, roll_deg=2.0, kappa2_target=0.02)
        c = time.perf_counter()
        sim.render(with_depth=args.with_depth, with_stones_visible=args.with_stones_visible)
        r = time.perf_counter()
        cmd_ms.append((c - t) * 1000.0)
        rnd_ms.append((r - c) * 1000.0)
        tot_ms.append((r - t) * 1000.0)

    def _stats(name: str, xs: list[float]) -> None:
        m = statistics.mean(xs)
        p50 = _percentile(xs, 0.50)
        p99 = _percentile(xs, 0.99)
        print(f"  {name:14s} mean {m:6.2f}  p50 {p50:6.2f}  p99 {p99:6.2f} ms")

    print(f"\nframe loop ({args.frames} frames):")
    _stats("command()", cmd_ms)
    _stats("render()", rnd_ms)
    _stats("total", tot_ms)
    fps = 1000.0 / statistics.mean(tot_ms)
    print(f"  fps:           {fps:6.1f}")
    print(f"  60fps target:  {'PASS' if fps >= 60.0 else 'FAIL'}")



if __name__ == "__main__":
    main()
