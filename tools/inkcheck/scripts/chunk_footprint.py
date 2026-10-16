"""How many S3 chunks, and how many bytes, a surface render will pull -- without pulling them.

Replays `inkcheck.render.render`'s tiling: each tile fetches only the chunks its trilinear sample
stencil touches (`inkcheck.volume.linear_stencil_chunks`). For comparison it also lists
what the old reader fetched, every chunk of each tile's bounding box. It checks which chunks are
already in the shared chunk cache and sizes the rest from the cached chunks' mean compressed size.
No chunk data is downloaded (one small `.zarray` request).

Use it to price a job before it runs: bytes / link throughput = network wall-clock.

    python scripts/chunk_footprint.py --tifxyz PATH --umbilicus PATH [--region row0 row1 col0 col1]
        [--layers 66] [--tile 384] [--sample PHerc0343]

`chunks` / `bytes_*` describe the current reader; `bbox_reader` the original bounding-box one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inkcheck.catalog import REFERENCE_SAMPLE, REFERENCE_VOLUME_ID, find_volume, load_catalog
from inkcheck.render import pixel_geometry
from inkcheck.surface import axis_from_umbilicus, load_tifxyz, normals, orient_outward
from inkcheck.volume import ZarrLevel, default_cache_dir, linear_stencil_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifxyz", type=Path, required=True)
    ap.add_argument("--umbilicus", type=Path, required=True, help="umbilicus JSON with control_points")
    ap.add_argument("--region", type=int, nargs=4, default=None, metavar=("ROW0", "ROW1", "COL0", "COL1"))
    ap.add_argument("--layers", type=int, default=66)
    ap.add_argument("--tile", type=int, default=384, help="render tile side (render() default 384)")
    ap.add_argument("--sample", default="PHerc0343")
    ap.add_argument("--volume-id", default=None,
                    help=f"volume id prefix (default for {REFERENCE_SAMPLE}: the pinned volume {REFERENCE_VOLUME_ID})")
    args = ap.parse_args()

    surf = load_tifxyz(args.tifxyz)
    umb = json.load(open(args.umbilicus))["control_points"]
    axis = axis_from_umbilicus(np.array([(p["z"], p["y"], p["x"]) for p in umb]))
    n, _ = orient_outward(surf, normals(surf), axis)
    H, W = surf.shape
    r0, r1, c0, c1 = args.region or (0, int((H - 1) * surf.step) + 1, 0, int((W - 1) * surf.step) + 1)
    offsets = np.arange(args.layers) - (args.layers - 1) / 2

    vid = args.volume_id or (REFERENCE_VOLUME_ID if args.sample == REFERENCE_SAMPLE else None)
    vi = find_volume(load_catalog(), args.sample, vid)
    arr = ZarrLevel(vi.url, 0)
    cs = np.array(arr.meta.chunks)
    shp = np.array(arr.meta.shape)

    box_ids: set[tuple[int, int, int]] = set()
    stencil_ids: set[tuple[int, int, int]] = set()
    box_reads = stencil_reads = 0  # chunk reads summed over tiles, before the cache dedups
    pixels = 0
    for a in range(r0, r1, args.tile):
        for b in range(c0, c1, args.tile):
            p, nn = pixel_geometry(surf, n, a, min(a + args.tile, r1), b, min(b + args.tile, c1))
            ok = np.all(np.isfinite(p), -1) & np.all(np.isfinite(nn), -1)
            if not ok.any():
                continue
            pixels += int(ok.sum())
            pts = (p[ok][None] + offsets[:, None, None] * nn[ok][None]).reshape(-1, 3)
            lo = np.maximum(np.floor(pts.min(0)).astype(int) - 2, 0)
            hi = np.minimum(np.ceil(pts.max(0)).astype(int) + 3, shp)
            rng = [range(lo[d] // cs[d], (hi[d] - 1) // cs[d] + 1) for d in range(3)]
            ids = {(z, y, x) for z in rng[0] for y in rng[1] for x in rng[2]}
            box_reads += len(ids)
            box_ids |= ids
            sids = {c for c in linear_stencil_chunks(pts, cs) if c in ids}  # read() clips to the box
            stencil_reads += len(sids)
            stencil_ids |= sids

    def census(chunk_ids):
        cached, cached_bytes, missing = 0, 0, 0
        for cz, cy, cx in chunk_ids:
            f = arr.cache / f"{cz}.{cy}.{cx}"
            if f.exists():
                cached += 1
                cached_bytes += f.stat().st_size
            elif (arr.cache / f"{cz}.{cy}.{cx}.missing").exists():
                missing += 1
        return cached, cached_bytes, missing

    mib = 2**20
    b_cached, b_cached_bytes, b_missing = census(box_ids)
    cached, cached_bytes, missing = census(stencil_ids)
    mean_chunk = b_cached_bytes / b_cached if b_cached else float(np.prod(cs) * arr.meta.dtype.itemsize)

    def est(n_ids, n_cached, n_cached_bytes, n_missing):
        unknown = n_ids - n_cached - n_missing
        return n_cached_bytes + unknown * mean_chunk, unknown * mean_chunk

    est_total, est_fetch = est(len(stencil_ids), cached, cached_bytes, missing)
    b_total, b_fetch = est(len(box_ids), b_cached, b_cached_bytes, b_missing)
    out = {
        "volume": vi.url,
        "chunks_shape": cs.tolist(),
        "compressor": (arr.meta.compressor or {}).get("id"),
        "region_px": [r0, r1, c0, c1],
        "valid_pixels": pixels,
        "layers": args.layers,
        "tile": args.tile,
        "chunks": len(stencil_ids),
        "chunk_reads_summed_over_tiles": stencil_reads,
        "chunks_in_cache": cached,
        "chunks_known_missing": missing,
        "cache_hit_rate": round(cached / len(stencil_ids), 4) if stencil_ids else None,
        "mean_chunk_MiB": round(mean_chunk / mib, 3),
        "bytes_total_est_MiB": round(est_total / mib, 1),
        "bytes_still_to_fetch_est_MiB": round(est_fetch / mib, 1),
        "MiB_per_Mpx": round(est_total / mib / (pixels / 1e6), 1) if pixels else None,
        "bbox_reader": {
            "chunks": len(box_ids),
            "chunk_reads_summed_over_tiles": box_reads,
            "bytes_total_est_MiB": round(b_total / mib, 1),
            "bytes_still_to_fetch_est_MiB": round(b_fetch / mib, 1),
        },
        "bytes_saved_vs_bbox": round(1 - est_total / b_total, 4) if b_total else None,
        "cache_dir": str(default_cache_dir()),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
