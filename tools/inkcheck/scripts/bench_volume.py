"""Measure inkcheck.volume read throughput and cache behaviour on a real S3 volume.

Three reads of the same representative patch (one z-chunk x 4x4 xy-chunks = 16 chunks, 32 MiB
uncompressed, PHerc0343 level 0):

  1. cold, chunk-aligned   -- fresh empty cache, box == whole chunks
  2. cold, misaligned      -- fresh empty cache, same extent shifted by half a chunk in y and x,
                               so it touches a 2x5x5 chunk neighbourhood instead of 1x4x4
  3. warm, chunk-aligned   -- same box as (1), cache now populated from (1)

(2) isolates the cost of chunk misalignment on an otherwise identical read; (3) isolates cache
hit speed. Bytes transferred is measured from actual cache file sizes written (the chunks here
are stored uncompressed, so on-disk bytes == network bytes), not inferred from array shape.

Usage: python scripts/bench_volume.py [--cache-gb N]
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from inkcheck.catalog import reference_volume
from inkcheck.volume import ZarrLevel, cache_usage

CHUNK = 128
# Chunk-boundary-aligned origin inside the PHerc0343 masked volume (arbitrary, mid-scroll).
# Alignment matters for test (1): an unaligned box of the same extent already touches extra
# chunks on its own, which would make the misalignment comparison in test (2) meaningless.
Z0, Y0, X0 = 74 * CHUNK, 23 * CHUNK, 23 * CHUNK


def timed_read(level: ZarrLevel, z0, y0, x0, z1, y1, x1):
    t0 = time.perf_counter()
    arr = level.read(z0, z1, y0, y1, x0, x1)
    dt = time.perf_counter() - t0
    return arr, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-gb", type=float, default=None, help="override INKCHECK_CACHE_MAX_GB for this run")
    args = ap.parse_args()

    vol = reference_volume()
    print(f"volume: {vol.url} ({vol.voxel_um} um/voxel)")

    with tempfile.TemporaryDirectory(prefix="inkcheck-bench-") as cold_dir:
        cold_dir = Path(cold_dir)
        level = ZarrLevel(vol.url, level=0, cache_dir=cold_dir)
        assert level.meta.chunks == (CHUNK, CHUNK, CHUNK), level.meta.chunks
        assert level.meta.compressor is None, "benchmark assumes uncompressed chunks"

        # 1. cold, chunk-aligned: exactly 1x4x4 = 16 chunks
        z1, y1, x1 = Z0 + CHUNK, Y0 + 4 * CHUNK, X0 + 4 * CHUNK
        _, dt_aligned = timed_read(level, Z0, Y0, X0, z1, y1, x1)
        bytes_aligned, files_aligned = cache_usage(cold_dir)
        print(f"\n1. cold, chunk-aligned:   {dt_aligned:6.2f} s, {bytes_aligned/2**20:7.1f} MiB, "
              f"{files_aligned:3d} chunk files -> {bytes_aligned/2**20/dt_aligned:6.1f} MB/s")

        # 2. cold, misaligned by half a chunk: same extent, shifted -> touches a 2x5x5 neighbourhood
        shutil.rmtree(cold_dir)
        cold_dir.mkdir()
        level2 = ZarrLevel(vol.url, level=0, cache_dir=cold_dir)
        shift = CHUNK // 2
        _, dt_misaligned = timed_read(
            level2, Z0 + shift, Y0 + shift, X0 + shift, z1 + shift, y1 + shift, x1 + shift
        )
        bytes_misaligned, files_misaligned = cache_usage(cold_dir)
        print(f"2. cold, misaligned:      {dt_misaligned:6.2f} s, {bytes_misaligned/2**20:7.1f} MiB, "
              f"{files_misaligned:3d} chunk files -> {bytes_misaligned/2**20/dt_misaligned:6.1f} MB/s "
              f"({files_misaligned/files_aligned:.2f}x the chunks for the same extent)")

        # 3. warm, chunk-aligned: re-read (1) from the now-populated cache
        _, dt_warm = timed_read(level, Z0, Y0, X0, z1, y1, x1)
        print(f"3. warm, chunk-aligned:   {dt_warm:6.2f} s, {bytes_aligned/2**20:7.1f} MiB "
              f"(same bytes, 0 network) -> {bytes_aligned/2**20/dt_warm:6.1f} MB/s, "
              f"{dt_aligned/dt_warm:.0f}x faster than cold")


if __name__ == "__main__":
    main()
