"""Time one surface render end to end and count the bytes it pulls; optionally check its output.

Compares the two chunk readers in `inkcheck.render`: `stencil` (default, fetches only the
chunks the trilinear samples touch) and `bbox` (every chunk of each tile's bounding box). For a
cold number, point `--cache` at an empty directory; it is deleted afterwards unless `--keep-cache`.

    python scripts/bench_render.py --tifxyz PATH --umbilicus PATH --region row0 row1 col0 col1
        [--reader stencil|bbox] [--cache DIR] [--reference layers.npy] [--workers 6]

Prints JSON: wall-clock, chunks/bytes downloaded and from cache, MB/s, the sha256 of the float32
render (equal hashes = bit-identical outputs), and whether the uint8 render (rint, clip 0..255,
as the experiments save it) equals `--reference`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from inkcheck.catalog import REFERENCE_SAMPLE, REFERENCE_VOLUME_ID, find_volume, load_catalog
from inkcheck.render import render
from inkcheck.surface import axis_from_umbilicus, load_tifxyz, normals, orient_outward
from inkcheck.volume import Volume, cache_usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifxyz", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True, help="umbilicus JSON with control_points")
    ap.add_argument("--region", type=int, nargs=4, default=None, metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--layers", type=int, default=66)
    ap.add_argument("--reader", choices=["stencil", "bbox"], default="stencil")
    ap.add_argument("--cache", type=Path, default=None, help="chunk cache dir (default: the shared one)")
    ap.add_argument("--keep-cache", action="store_true", help="don't delete --cache afterwards")
    ap.add_argument("--reference", type=Path, default=None, help="uint8 layers.npy to compare against")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sample", default="PHerc0343")
    ap.add_argument("--volume-id", default=None,
                    help=f"volume id prefix (default for {REFERENCE_SAMPLE}: the pinned volume {REFERENCE_VOLUME_ID})")
    args = ap.parse_args()

    surf = load_tifxyz(args.tifxyz)
    umb = json.load(open(args.umbilicus))["control_points"]
    axis = axis_from_umbilicus(np.array([(p["z"], p["y"], p["x"]) for p in umb]))
    n, _ = orient_outward(surf, normals(surf), axis)
    offsets = np.arange(args.layers) - (args.layers - 1) / 2
    region = tuple(args.region) if args.region else None

    vid = args.volume_id or (REFERENCE_VOLUME_ID if args.sample == REFERENCE_SAMPLE else None)
    vi = find_volume(load_catalog(), args.sample, vid)
    cache_was_empty = args.cache is not None and cache_usage(args.cache)[1] == 0
    vol = Volume(vi.url, 0, cache_dir=args.cache)
    t = time.time()
    layers, valid = render(surf, n, vol, offsets, region=region, workers=args.workers,
                           sparse=args.reader == "stencil")
    dt = time.time() - t
    st = vol.array.stats()
    out = {
        "reader": args.reader,
        "region_px": list(region) if region else None,
        "shape": list(layers.shape),
        "valid_fraction": round(float(valid.mean()), 4),
        "workers": args.workers,
        "cold_cache": cache_was_empty,
        "wall_s": round(dt, 1),
        **st,
        "MiB_downloaded": round(st["bytes_downloaded"] / 2**20, 1),
        "MB_per_s_downloaded": round(st["bytes_downloaded"] / 1e6 / dt, 2),
        "sha256_float32": hashlib.sha256(np.ascontiguousarray(layers).tobytes()).hexdigest(),
    }
    if args.reference:
        ref = np.load(args.reference, mmap_mode="r")
        u8 = np.clip(np.rint(layers), 0, 255).astype(np.uint8)
        out["reference"] = str(args.reference)
        out["equal_to_reference_uint8"] = bool(ref.shape == u8.shape and np.array_equal(ref, u8))
    if args.cache is not None and not args.keep_cache:
        shutil.rmtree(args.cache, ignore_errors=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
