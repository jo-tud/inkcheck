"""Time one representative inkcheck run on CPU and size a rented GPU from it.

My machine's GPUs were reserved for other work, so this is the shape of
run I actually do day to day: stream a small patch from S3, render a stack of
layers along a surface normal, run one ink model checkpoint over it. Real S3 reads and a real
`scrollprize/ink_9um` checkpoint; the surface itself is a synthetic flat patch (there is no
real tifxyz mesh checked into this repo) so the script is self-contained and reproducible from a
clean checkout -- it exercises the same render + inference code path real experiments use, just
without a scientifically meaningful mesh. Not a validation result; a timing fixture.

The render runs twice: once against an empty cache (network + compute, and the bytes it pulled)
and once more against the now-warm cache (compute only). The difference is the time spent waiting
on the network, which is what a better-connected machine would save.

Usage: python scripts/bench_cpu_run.py [--pixels 256] [--threads N] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from inkcheck.catalog import reference_volume
from inkcheck.gpu import cap_cpu_threads
from inkcheck.ink import load_model, predict
from inkcheck.render import render
from inkcheck.surface import Surface

CHECKPOINT = "data/models/ink_9um/hybrid_3d2d-seed42/step-075000.pth"
# Chunk-aligned origin inside PHerc0343, level 0 (same region used by scripts/bench_volume.py).
Z0, Y0, X0 = 74 * 128, 23 * 128, 23 * 128


def flat_surface(pixels: int, z0: int, y0: int, x0: int) -> tuple[Surface, np.ndarray]:
    """A synthetic flat patch: one grid node per render pixel, normal along +z."""
    rows, cols = np.meshgrid(np.arange(pixels, dtype=np.float64), np.arange(pixels, dtype=np.float64), indexing="ij")
    zyx = np.stack([np.full_like(rows, float(z0)), rows + y0, cols + x0], axis=-1)
    surf = Surface(zyx=zyx, step=1.0, meta={}, name="synthetic-flat")
    n = np.zeros_like(zyx)
    n[..., 0] = 1.0  # unit normal along +z
    return surf, n


def peak_rss_gb() -> float:
    # ru_maxrss is KB on Linux, bytes on macOS; this project only runs on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pixels", type=int, default=256, help="render region side length in pixels")
    ap.add_argument("--threads", type=int, default=None, help="override INKCHECK_CPU_THREADS")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="chunk cache to use (default: a fresh empty temp dir, so the read is cold)")
    args = ap.parse_args()
    tmp = None
    if args.cache_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="inkcheck-bench-cpu-")
        args.cache_dir = Path(tmp.name)

    cap_cpu_threads(args.threads)
    t_start = time.perf_counter()

    vi = reference_volume()
    from inkcheck.volume import Volume, cache_usage
    vol = Volume(vi.url, level=0, cache_dir=args.cache_dir)
    bytes_before, _ = cache_usage(args.cache_dir)

    surf, n = flat_surface(args.pixels, Z0, Y0, X0)
    offsets = np.arange(-16, 17)  # 33 layers

    t0 = time.perf_counter()
    layers, valid = render(surf, n, vol, offsets, workers=4)
    t_render = time.perf_counter() - t0
    fetched = cache_usage(args.cache_dir)[0] - bytes_before
    print(f"render: {t_render:.1f}s, {layers.shape}, valid frac {valid.mean():.2f}, "
          f"fetched {fetched / 2**20:.1f} MiB ({fetched / 2**20 / t_render:.1f} MiB/s), "
          f"peak RSS so far {peak_rss_gb():.2f} GB")

    t0 = time.perf_counter()
    render(surf, n, vol, offsets, workers=4)
    t_render_warm = time.perf_counter() - t0
    print(f"render again, warm cache: {t_render_warm:.1f}s "
          f"-> network wait in the cold render ~{t_render - t_render_warm:.1f}s")

    t0 = time.perf_counter()
    m = load_model(CHECKPOINT, "ink9-seed42", device="cpu")
    t_load = time.perf_counter() - t0
    print(f"load_model: {t_load:.1f}s, patch={m.patch} depth={m.depth}")

    t0 = time.perf_counter()
    pr = predict(m, layers, valid, direction="reverse")
    t_predict = time.perf_counter() - t0
    print(f"predict: {t_predict:.1f}s, output {pr.shape}, mean prob {np.nanmean(pr):.3f}")

    t_total = time.perf_counter() - t_start - t_render_warm
    peak_gb = peak_rss_gb()
    print(f"\ntotal wall-clock: {t_total:.1f}s  (render {t_render:.1f}s, load {t_load:.1f}s, "
          f"predict {t_predict:.1f}s; excludes the warm re-render)")
    print(f"bytes fetched: {fetched / 2**20:.1f} MiB; render compute (warm) {t_render_warm:.1f}s")
    print(f"peak RSS: {peak_gb:.2f} GB")
    print(f"pixels: {args.pixels}x{args.pixels}, layers: {len(offsets)}, "
          f"threads: {args.threads or 'default'}")
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
