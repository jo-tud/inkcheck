"""Grow small single-sheet patches from a binary surface prediction (m7), snap them to the CT
sheet, and certify them with the ``inkcheck.geometry`` sheet-peak measure.

The grower is deliberately simple and conservative: a height field over a local plane, grown
breadth-first from a seed. Each new node goes on the prediction run nearest the height its placed
neighbours predict. A node is **refused** when no run is close enough, or when a second run
competes nearby — so the patch stops at a sheet contact instead of jumping to the next wrap.

Coordinates: everything public is level-0 voxels (z, y, x) of the CT volume. The prediction is
read at its own level (``pred_scale`` level-0 voxels per prediction voxel; 4 for the m7 runs on
2.4 um volumes, which were computed on level 2) Prediction voxel k is centred at level-0
``pred_scale * k + (pred_scale - 1) / 2``, as a pyramid level (``volume.to_level``).
"""

from __future__ import annotations

import heapq
import json
import uuid as _uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

from .geometry import GeometryParams, analyze_surface
from .surface import Surface, normals, shifted
from .volume import Volume, ZarrLevel, from_level, to_level


@dataclass
class GrowParams:
    step: int = 5  # node spacing in prediction voxels (5 x 4 = 20 level-0 voxels, VC3D's step)
    half_nodes: int = 80  # the patch grid is (2 * half_nodes + 1)^2 nodes at most (0.59 cm²)
    search: float = 3.0  # accept a run whose centre is within +-search of the predicted height
    ambiguity: float = 6.0  # refuse if a second run centre is within +-ambiguity of the prediction
    max_slope: float = 1.0  # |dh| per node step, in node steps (45 degrees)
    normal_sigma: float = 3.0  # structure-tensor smoothing for the seed normal (prediction voxels)
    snap_level: int = 1  # CT level for snapping
    snap_radius: int = 4  # max snap distance (snap-level voxels)
    snap_prominence: float = 4.0


def _runs(on: np.ndarray, offs: np.ndarray) -> np.ndarray:
    d = np.diff(np.concatenate([[0], on.astype(np.int8), [0]]))
    st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
    return (offs[st] + offs[en]) / 2 if len(st) else np.empty(0)


def seed_normal(box: np.ndarray, p: np.ndarray, sigma: float) -> np.ndarray:
    """Unit normal of a binary sheet at p (box index coordinates) from its structure tensor."""
    r = int(4 * sigma) + 2
    c = np.round(p).astype(int)
    sub = box[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1, c[2] - r:c[2] + r + 1].astype(np.float32)
    g = [ndimage.gaussian_filter(sub, 1.0, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    w = np.exp(-((np.indices(sub.shape) - r) ** 2).sum(0) / (2 * sigma ** 2))
    T = np.array([[np.sum(w * g[i] * g[j]) for j in range(3)] for i in range(3)])
    _, V = np.linalg.eigh(T)
    return V[:, 2]


def grow(pred: ZarrLevel, seed_l0, params: GrowParams | None = None, pred_scale: int = 4):
    """Grow one patch. Returns (zyx_l0 grid with NaN, stats dict)."""
    gp = params or GrowParams()
    K = gp.half_nodes
    reach = K * gp.step
    H = reach + 40  # a tilted plane's corners may leave the box; those nodes count as no-hit
    sp = to_level(np.asarray(seed_l0, float), pred_scale)
    c = np.round(sp).astype(int)
    box = pred.read(c[0] - H, c[0] + H, c[1] - H, c[1] + H, c[2] - H, c[2] + H) > 0
    org = c - H
    p0 = sp - org
    n = seed_normal(box, p0, gp.normal_sigma)
    # in-plane axes: u as close to +z as possible (rows run along the scroll axis), v = n x u
    ez = np.array([1.0, 0, 0])
    u = ez - n * (n @ ez)
    if np.linalg.norm(u) < 0.3:
        u = np.array([0, 1.0, 0]) - n * n[1]
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    offs = np.arange(-gp.ambiguity - 1, gp.ambiguity + 1.01, 0.5)

    def runs_at(i, j, h):
        base = p0 + (i * gp.step) * u + (j * gp.step) * v
        pts = base[None, :] + (h + offs)[:, None] * n[None, :]
        if np.any(pts < 0) or np.any(pts >= np.array(box.shape) - 1):
            return None
        on = ndimage.map_coordinates(box, pts.T, order=0, mode="constant", cval=0) > 0
        return h + _runs(on, offs)

    hgt = np.full((2 * K + 1, 2 * K + 1), np.nan)
    state = np.zeros(hgt.shape, np.int8)  # 0 untried, 1 placed, 2 refused-ambiguous, 3 no-hit
    r0 = runs_at(0, 0, 0.0)
    if r0 is None or len(r0) == 0:
        return None, {"reason": "seed not on prediction"}
    j0 = int(np.argmin(np.abs(r0)))
    if np.sum(np.abs(r0 - r0[j0]) <= gp.ambiguity) > 1:
        return None, {"reason": "seed ambiguous"}
    hgt[K, K] = r0[j0]
    state[K, K] = 1
    heap = []

    def push_nbrs(a, b):
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            aa, bb = a + da, b + db
            if 0 <= aa <= 2 * K and 0 <= bb <= 2 * K and state[aa, bb] == 0:
                heapq.heappush(heap, ((aa - K) ** 2 + (bb - K) ** 2, aa, bb))

    push_nbrs(K, K)
    max_dh = gp.max_slope * gp.step
    while heap:
        _, a, b = heapq.heappop(heap)
        if state[a, b] != 0:
            continue
        preds = []
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a1, b1, a2, b2 = a + da, b + db, a + 2 * da, b + 2 * db
            if 0 <= a1 <= 2 * K and 0 <= b1 <= 2 * K and state[a1, b1] == 1:
                h1 = hgt[a1, b1]
                if 0 <= a2 <= 2 * K and 0 <= b2 <= 2 * K and state[a2, b2] == 1:
                    preds.append(2 * h1 - hgt[a2, b2])
                else:
                    preds.append(h1)
        if not preds:
            continue
        hp = float(np.mean(preds))
        rr = runs_at(a - K, b - K, hp)
        if rr is None or len(rr) == 0:
            state[a, b] = 3
            continue
        d = np.abs(rr - hp)
        k = int(np.argmin(d))
        if d[k] > gp.search:
            state[a, b] = 3
            continue
        if np.sum(d <= gp.ambiguity) > 1:
            state[a, b] = 2
            continue
        nb = [hgt[a + da, b + db] for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
              if 0 <= a + da <= 2 * K and 0 <= b + db <= 2 * K and state[a + da, b + db] == 1]
        if any(abs(rr[k] - h) > max_dh for h in nb):
            state[a, b] = 3
            continue
        hgt[a, b] = rr[k]
        state[a, b] = 1
        push_nbrs(a, b)

    ii, jj = np.meshgrid(np.arange(-K, K + 1), np.arange(-K, K + 1), indexing="ij")
    P = (p0[None, None, :] + (ii * gp.step)[..., None] * u + (jj * gp.step)[..., None] * v
         + hgt[..., None] * n)
    zyx = from_level(P + org, pred_scale)
    zyx[state != 1] = np.nan
    rows = np.nonzero((state == 1).any(1))[0]
    cols = np.nonzero((state == 1).any(0))[0]
    zyx = zyx[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    st = state[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    stats = {"placed": int((st == 1).sum()), "refused_ambiguous": int((state == 2).sum()),
             "refused_nohit": int((state == 3).sum()), "seed_normal_zyx": n.round(4).tolist()}
    return zyx, stats


def snap(surf: Surface, ct: Volume, gp: GrowParams, tile: int = 12, workers: int = 4) -> tuple[Surface, np.ndarray]:
    """Move each node along its normal to the nearest CT peak within +-snap_radius (CT-level voxels).

    Returns the snapped surface and the applied offsets (level-0 voxels; NaN = no peak, node kept).
    """
    n = normals(surf)
    s = ct.scale
    R = gp.snap_radius + 8
    offs = np.arange(-R, R + 1, dtype=np.float64)
    H, W = surf.shape
    out = np.full((H, W), np.nan)

    def one(rc):
        r, c = rc
        P = surf.zyx[r:r + tile, c:c + tile]
        N = n[r:r + tile, c:c + tile]
        ok = np.all(np.isfinite(P), -1) & np.all(np.isfinite(N), -1)
        blk = np.full(ok.shape, np.nan)
        if not ok.any():
            return rc, blk
        pts = P[ok][:, None, :] + (offs * s)[None, :, None] * N[ok][:, None, :]
        prof = ndimage.uniform_filter1d(ct.sample(pts), 3, axis=1)
        d = np.full(len(prof), np.nan)
        for k, pr in enumerate(prof):
            pk, _ = signal.find_peaks(pr, prominence=gp.snap_prominence)
            if len(pk):
                o = offs[pk]
                j = int(np.argmin(np.abs(o)))
                if abs(o[j]) <= gp.snap_radius:
                    d[k] = o[j] * s
        blk[ok] = d
        return rc, blk

    with ThreadPoolExecutor(workers) as ex:
        for (r, c), blk in ex.map(one, [(r, c) for r in range(0, H, tile) for c in range(0, W, tile)]):
            out[r:r + tile, c:c + tile] = blk
    # robust: median-filter the offset field so one noisy profile does not kink the mesh
    filled = np.where(np.isfinite(out), out, 0.0)
    cnt = ndimage.uniform_filter(np.isfinite(out).astype(float), 3)
    med = ndimage.median_filter(filled, 3)
    use = np.isfinite(out) & (cnt > 0.5)
    off = np.where(use, med, 0.0)
    off[~surf.valid] = np.nan
    snapped = Surface(zyx=surf.zyx + off[..., None] * n, step=surf.step, meta=dict(surf.meta), name=surf.name)
    return snapped, np.where(use, med, np.nan)


def _scaled(surf: Surface, f: float) -> Surface:
    return Surface(zyx=to_level(surf.zyx, f), step=surf.step / f, meta=surf.meta, name=surf.name)


def single_sheet_fields(surf: Surface, ct_l1: Volume, workers: int = 4) -> dict:
    """``inkcheck.geometry`` fields on CT level 1 (geometry defaults), plus the single-sheet flag.

    single = a prominent peak within +-rival_window of the node and no rival peak there.
    """
    gp = GeometryParams(workers=workers)
    s = _scaled(surf, ct_l1.scale)
    f = analyze_surface(s, normals(s), ct_l1, None, gp)
    ok = np.isfinite(f["st_angle"])
    single = ok & (np.abs(f["d0"]) <= gp.rival_window) & (f["rivals"] == 0)
    f["single"] = np.where(ok, single.astype(float), np.nan)
    return f


def m7_fields(surf: Surface, ct_lp: Volume, pred_lp: Volume, workers: int = 4) -> dict:
    """``inkcheck.geometry`` m7 support at the prediction's own level."""
    gp = GeometryParams(workers=workers)
    s = _scaled(surf, ct_lp.scale)
    return analyze_surface(s, normals(s), ct_lp, pred_lp, gp)


def tears(surf: Surface) -> float:
    """Fraction of neighbouring valid node pairs further apart than 2 x step."""
    p = surf.zyx
    d = [np.linalg.norm(p[1:] - p[:-1], axis=-1), np.linalg.norm(p[:, 1:] - p[:, :-1], axis=-1)]
    d = np.concatenate([x[np.isfinite(x)] for x in d])
    return float((d > 2 * surf.step).mean()) if len(d) else float("nan")


def write_tifxyz(path: str | Path, surf: Surface, voxel_um: float, extra: dict | None = None) -> dict:
    """Write a VC3D tifxyz directory (x/y/z.tif float32, invalid = -1, meta.json)."""
    import tifffile

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    p = surf.zyx.astype(np.float32)
    bad = ~np.all(np.isfinite(p), -1)
    for k, name in ((2, "x"), (1, "y"), (0, "z")):
        a = p[..., k].copy()
        a[bad] = -1.0
        tifffile.imwrite(path / f"{name}.tif", a)
    v = p[~bad]
    area = surf.area_cm2(voxel_um)
    meta = {
        "bbox": [v[:, ::-1].min(0).tolist(), v[:, ::-1].max(0).tolist()],  # (x, y, z) like VC3D
        "format": "tifxyz", "type": "seg", "uuid": path.name,
        "scale": [1.0 / surf.step, 1.0 / surf.step],
        "area_cm2": area, "area_vx2": area / (voxel_um * 1e-4) ** 2,
        "source": "inkcheck.patchgrow",
    }
    meta.update(extra or {})
    (path / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def new_name(prefix: str = "pg") -> str:
    return f"{prefix}_{_uuid.uuid4().hex[:12]}"


def main(argv=None) -> None:
    """One patch from one seed:

    python -m inkcheck.patchgrow --volume s3://.../X.zarr --pred s3://.../X-m7.zarr \\
        --seed Z,Y,X --out DIR [--voxel-um 2.403] [--pred-scale 4]
    """
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True, help="CT OME-zarr (level 1 is read for snap + certificate)")
    ap.add_argument("--pred", required=True, help="binary surface prediction zarr (its level 0 is read)")
    ap.add_argument("--seed", required=True, help="Z,Y,X in level-0 voxels of the CT volume")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voxel-um", type=float, default=2.403)
    ap.add_argument("--pred-scale", type=int, default=4, help="level-0 voxels per prediction voxel")
    a = ap.parse_args(argv)
    gp = GrowParams()
    pred = ZarrLevel(a.pred, 0)
    ct1 = Volume(a.volume, 1)
    seed = np.array([float(v) for v in a.seed.split(",")])
    zyx, st = grow(pred, seed, gp, a.pred_scale)
    if zyx is None:
        raise SystemExit(f"no patch: {st}")
    surf = Surface(zyx=zyx, step=float(gp.step * a.pred_scale))
    sn, off = snap(surf, ct1, gp)
    f = single_sheet_fields(sn, ct1)
    q = {"single_sheet_fraction": float(np.nanmean(f["single"])), "tears": tears(sn), **st,
         "snapped_fraction": float(np.mean(np.isfinite(off[sn.valid])))}
    meta = write_tifxyz(Path(a.out), sn, a.voxel_um, extra={"quality": q, "grow_params": asdict(gp),
                                                            "seed_l0": seed.tolist()})
    print(json.dumps({"out": a.out, "area_cm2": round(meta["area_cm2"], 4), **q}, indent=1))


if __name__ == "__main__":
    main()


__all__ = ["GrowParams", "grow", "snap", "single_sheet_fields", "m7_fields", "tears", "write_tifxyz",
           "shifted", "asdict"]
