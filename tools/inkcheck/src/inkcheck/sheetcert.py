"""Single-sheet certificate v2: is each node of a surface on one papyrus sheet?

**Status (2026-10-13): this does not work as a certificate yet.** On PHerc0139, PHerc0846A and
PHerc1203 it fails its own half-spacing null (H/F 0.63-0.85, where it needs 0.5 or less): these
scrolls are dense with gap-bounded bands at 4.8 um. It stays here as a measuring tool (the nulls
and the displacement sweep). The test runs behind this are not part of this repository.

The v1 rule (one dominant CT peak near the node, ``patchgrow.single_sheet_fields``) does not
discriminate at 2.4 µm: fibre texture gives several peaks per profile on and off the sheet
(in my 2.4 µm patch study, a surface displaced 12 voxels off the sheet scored 0.94 of the
on-sheet median). v2 looks for the *gaps* instead. Per node:

1. CT profile along the normal, averaged over a square tangential stencil, over a long range
   (default ±56 voxels of the CT level read).
2. Inside a window (default ±16 voxels) around the node: take the brightest point near the
   centre, the half level between it and the darkest point of the long profile, and the band
   above that level. The node passes when the band closes inside the window on both sides, both
   gaps are deep, the band is a single sheet thick, the node is in it, and there is contrast.

Every measurement comes with two nulls computed by the same code on the same nodes (so a
certificate can check itself):

* tangent null: the sampling frame turned 90° (profile along each tangent, stencil over the
  normal and the other tangent);
* half-spacing null: the window centred half a local sheet spacing off the surface, in the gap
  between two sheets. The spacing is the first autocorrelation peak of the long profile, median
  per tile of nodes.

Reported beside them: the full-spacing shift (the neighbouring sheet; the certificate is
wrap-blind there by design) and the fixed ±12-voxel shift of that patch study.

Coordinates: surfaces are level-0 voxels (z, y, x); ``vol`` is any CT level; every length in
``BandParams`` is in voxels of that level except the widths, which are in µm.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .geometry import _tangents
from .surface import Surface, normals
from .volume import Volume


@dataclass
class SampleParams:
    """What is read per node. Fixed: both stencils (3 x 3 and 5 x 5) are read at once."""

    long_radius: int = 56  # long normal profile, -R..R voxels
    window: int = 16  # measurement window half-width (voxels); also the tangent profile range
    stencil_max: int = 2  # 5 x 5 stencil read; s = 1 uses the inner 3 x 3
    stencil_spacing: float = 4.0  # voxels between stencil points
    smooth: int = 3  # running mean along the profile


@dataclass
class BandParams:
    stencil: int = 1  # stencil half-size: 1 -> 3 x 3, 2 -> 5 x 5
    gap_depth: float = 0.65  # (top - gap) / (top - air) on both sides
    w_max_um: float = 90.0
    w_min_um: float = 10.0
    anchor: int = 6  # band peak searched within +-anchor of the window centre
    level_frac: float = 0.5  # band = profile >= air + level_frac * (top - air)
    node_tol: float = 2.0  # window centre may sit this far outside the band edges
    min_contrast: float = 8.0  # top - air (uint8 units)
    lag_min: int = 6  # sheet spacing: first autocorrelation peak in lag_min..lag_max
    lag_max: int = 40
    null_shift: int = 12  # the patch study's fixed displacement, reported for continuity


# ----------------------------------------------------------------------------------------------
# sampling


def _frames(surf: Surface):
    n = normals(surf)
    t, b = _tangents(surf, n)
    return n, t, b


def _sample_tile(vol: Volume, P, N, T, B, sp: SampleParams) -> dict:
    """Profiles for one tile of nodes (all inputs (k, 3) in the coordinates of ``vol``)."""
    S = sp.stencil_max
    st = np.arange(-S, S + 1) * sp.stencil_spacing
    su, sv = [a.ravel() for a in np.meshgrid(st, st, indexing="ij")]
    ol = np.arange(-sp.long_radius, sp.long_radius + 1, dtype=np.float64)
    ow = np.arange(-sp.window, sp.window + 1, dtype=np.float64)

    def pts(d, e1, e2, offs):  # (k, stencil, offsets, 3)
        return (P[:, None, None, :] + su[None, :, None, None] * e1[:, None, None, :]
                + sv[None, :, None, None] * e2[:, None, None, :] + offs[None, None, :, None] * d[:, None, None, :])

    groups = {"normal": pts(N, T, B, ol), "tan_t": pts(T, N, B, ow), "tan_b": pts(B, N, T, ow)}
    allp = np.concatenate([g.reshape(-1, 3) for g in groups.values()])
    lo = np.floor(allp.min(0)).astype(int) - 2
    hi = np.ceil(allp.max(0)).astype(int) + 3
    box = vol.read_box(lo, hi).astype(np.float32)
    inner = (np.abs(su) <= sp.stencil_spacing + 1e-6) & (np.abs(sv) <= sp.stencil_spacing + 1e-6)
    out = {}
    for name, g in groups.items():
        v = ndimage.map_coordinates(box, (g.reshape(-1, 3) - lo).T, order=1, mode="nearest").reshape(g.shape[:3])
        for s, m in ((1, inner), (2, np.ones_like(inner))):
            prof = v[:, m].mean(1)
            out[f"{name}_s{s}"] = ndimage.uniform_filter1d(prof, sp.smooth, axis=1, mode="nearest")
            out[f"{name}_masked_s{s}"] = (v[:, m] < 0.5).any(axis=(1, 2))
    return out


def sample_profiles(surf: Surface, vol: Volume, tiles: list[tuple[int, int]] | None = None,
                    tile_nodes: int = 12, sp: SampleParams | None = None, workers: int = 4,
                    inner: int = 0) -> dict:
    """Profiles for every valid node in ``tiles`` (default: all tiles with a valid node).

    ``inner``: drop this many border rows/cols of each tile from the output (they are still used
    for normals; for tiles cut with a 1-node border, as the patch study's transfer check did).
    Returns arrays over the sampled nodes plus their (row, col) and tile index.
    """
    sp = sp or SampleParams()
    s = Surface(zyx=surf.zyx / vol.scale, step=surf.step / vol.scale, name=surf.name)
    n, t, b = _frames(s)
    H, W = s.shape
    if tiles is None:
        tiles = [(r, c) for r in range(0, H, tile_nodes) for c in range(0, W, tile_nodes)
                 if s.valid[r:r + tile_nodes, c:c + tile_nodes].any()]

    def run(k):
        r, c = tiles[k]
        sl = (slice(r + inner, r + tile_nodes - inner), slice(c + inner, c + tile_nodes - inner))
        ok = (s.valid[sl] & np.all(np.isfinite(n[sl]), -1) & np.all(np.isfinite(t[sl]), -1))
        if not ok.any():
            return k, None
        rr, cc = np.nonzero(ok)
        res = _sample_tile(vol, s.zyx[sl][ok], n[sl][ok], t[sl][ok], b[sl][ok], sp)
        res["rc"] = np.stack([rr + r + inner, cc + c + inner], 1)
        res["tile"] = np.full(len(rr), k)
        return k, res

    parts = []
    with ThreadPoolExecutor(workers) as ex:
        for _, res in ex.map(run, range(len(tiles))):
            if res is not None:
                parts.append(res)
    if not parts:
        return {}
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


# ----------------------------------------------------------------------------------------------
# the band test


def band_test(w: np.ndarray, air: np.ndarray, bp: BandParams, voxel_um: float) -> dict:
    """w: (N, 2W+1) window profiles centred on the point under test; air: (N,) dark reference."""
    N, K = w.shape
    m = K // 2
    a = bp.anchor
    out = {"pass": np.zeros(N, bool), "width_um": np.full(N, np.nan), "closed": np.zeros(N, bool),
           "depth_min": np.full(N, np.nan), "contrast": np.full(N, np.nan)}
    for i in range(N):
        p = w[i]
        if not np.isfinite(air[i]) or not np.all(np.isfinite(p)):
            continue
        c = m - a + int(np.argmax(p[m - a:m + a + 1]))
        top = p[c]
        rng = top - air[i]
        out["contrast"][i] = rng
        if rng < bp.min_contrast:
            continue
        L = air[i] + bp.level_frac * rng
        lft = c
        while lft > 0 and p[lft - 1] >= L:
            lft -= 1
        rgt = c
        while rgt < K - 1 and p[rgt + 1] >= L:
            rgt += 1
        if lft == 0 or rgt == K - 1:
            continue
        out["closed"][i] = True
        xl = (lft - 1) + (L - p[lft - 1]) / (p[lft] - p[lft - 1])
        xr = rgt + (p[rgt] - L) / (p[rgt] - p[rgt + 1])
        dl = (top - p[:lft].min()) / rng
        dr = (top - p[rgt + 1:].min()) / rng
        width = (xr - xl) * voxel_um
        out["width_um"][i] = width
        out["depth_min"][i] = min(dl, dr)
        out["pass"][i] = (dl >= bp.gap_depth and dr >= bp.gap_depth and bp.w_min_um <= width <= bp.w_max_um
                          and xl - bp.node_tol <= m <= xr + bp.node_tol)
    return out


def sheet_spacing(long: np.ndarray, bp: BandParams) -> np.ndarray:
    """Per node: first local maximum of the autocorrelation at lag lag_min..lag_max (voxels)."""
    x = long - long.mean(1, keepdims=True)
    den = (x * x).sum(1)
    K = bp.lag_max + 1
    ac = np.full((len(x), K + 1), np.nan)
    for k in range(1, K + 1):
        ac[:, k] = (x[:, :-k] * x[:, k:]).sum(1) / np.maximum(den, 1e-9)
    out = np.full(len(x), np.nan)
    for k in range(bp.lag_min, bp.lag_max + 1):
        hit = np.isnan(out) & (ac[:, k] > ac[:, k - 1]) & (ac[:, k] >= ac[:, k + 1])
        out[hit] = k
    return out


def _window(long: np.ndarray, shift: np.ndarray, half: int) -> np.ndarray:
    R = long.shape[1] // 2
    fin = np.isfinite(shift)
    idx = R + np.round(np.where(fin, shift, 0)).astype(int)[:, None] + np.arange(-half, half + 1)[None, :]
    ok = fin & (idx.min(1) >= 0) & (idx.max(1) < long.shape[1])
    w = np.full((len(long), 2 * half + 1), np.nan)
    w[ok] = np.take_along_axis(long[ok], idx[ok], axis=1)
    return w


def evaluate(prof: dict, bp: BandParams, voxel_um: float, half: int = 16) -> dict:
    """Per-node pass flags for the surface and every null, from ``sample_profiles`` output."""
    s = bp.stencil
    long = prof[f"normal_s{s}"]
    masked = prof[f"normal_masked_s{s}"]
    air = long.min(1)
    N = len(long)
    sp_node = sheet_spacing(long, bp)
    sp_tile = np.full(N, np.nan)
    for k in np.unique(prof["tile"]):
        sel = prof["tile"] == k
        if np.isfinite(sp_node[sel]).any():
            sp_tile[sel] = np.nanmedian(sp_node[sel])

    def run(shift, extra_mask=None):
        w = _window(long, shift, half)
        r = band_test(w, air, bp, voxel_um)
        ok = r["pass"] & ~masked
        if extra_mask is not None:
            ok &= ~extra_mask
        defined = np.all(np.isfinite(w), 1)
        return ok, defined, r

    zero = np.zeros(N)
    real, _, rr = run(zero)
    out = {"real": real, "width_um": rr["width_um"], "depth_min": rr["depth_min"], "contrast": rr["contrast"],
           "spacing_vox": sp_tile, "spacing_node_vox": sp_node}
    for name in ("tan_t", "tan_b"):
        r = band_test(prof[f"{name}_s{s}"], air, bp, voxel_um)
        out[name] = r["pass"] & ~prof[f"{name}_masked_s{s}"] & ~masked
    for tag, sh in (("half", sp_tile / 2), ("full", sp_tile), ("fixed", np.full(N, bp.null_shift))):
        for sign, sg in (("p", 1), ("m", -1)):
            ok, defined, _ = run(np.where(np.isfinite(sh), sg * sh, np.nan))
            out[f"{tag}_{sign}"] = np.where(defined & np.isfinite(sh), ok, np.nan)
    return out


def summarise(ev: dict) -> dict:
    """Fractions over nodes: F (real), T (tangent null), H (half-spacing null) and the rest."""
    def frac(x):
        x = np.asarray(x, float)
        return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")

    F = frac(ev["real"])
    T = 0.5 * (frac(ev["tan_t"]) + frac(ev["tan_b"]))
    H = 0.5 * (frac(ev["half_p"]) + frac(ev["half_m"]))
    res = {"n_nodes": int(len(ev["real"])), "F": F, "T": T, "H": H,
           "T_t": frac(ev["tan_t"]), "T_b": frac(ev["tan_b"]),
           "H_coverage": frac(np.isfinite(np.asarray(ev["half_p"], float))),
           "full_spacing": 0.5 * (frac(ev["full_p"]) + frac(ev["full_m"])),
           "full_coverage": frac(np.isfinite(np.asarray(ev["full_p"], float))),
           "fixed_12": 0.5 * (frac(ev["fixed_p"]) + frac(ev["fixed_m"])),
           "spacing_um_median_vox": float(np.nanmedian(ev["spacing_vox"])) if np.isfinite(ev["spacing_vox"]).any()
           else float("nan"),
           "width_um_median_pass": float(np.nanmedian(ev["width_um"][ev["real"]])) if ev["real"].any()
           else float("nan")}
    res["T_over_F"] = res["T"] / F if F > 0 else float("inf")
    res["H_over_F"] = res["H"] / F if F > 0 else float("inf")
    return res


def certificate(summary: dict, bar: float, null_ratio: float = 0.5) -> bool:
    """Certified single-sheet: F >= bar and both own nulls <= null_ratio x F."""
    F = summary["F"]
    return bool(F >= bar and summary["T"] <= null_ratio * F and summary["H"] <= null_ratio * F)


def pass_map(surf: Surface, prof: dict, ev: dict) -> np.ndarray:
    """(H, W) float map: 1 pass, 0 fail, NaN not measured."""
    m = np.full(surf.shape, np.nan)
    m[prof["rc"][:, 0], prof["rc"][:, 1]] = ev["real"].astype(float)
    return m


def main(argv=None) -> None:
    """Certify one tifxyz surface:

    python -m inkcheck.sheetcert --volume s3://.../X.zarr --surface DIR --params calibration.json
        [--level 1] [--voxel-um 2.403] [--write]

    Prints F, T, H and whether the patch is certified; --write adds quality.sheetcert_v2 to DIR/meta.json and
    writes DIR/band.npy (per-node pass map).
    """
    import argparse
    from pathlib import Path

    from .surface import load_tifxyz

    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--surface", required=True)
    ap.add_argument("--params", required=True, help="calibration.json (band params and patch bar)")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--voxel-um", type=float, default=2.403, help="level-0 voxel size")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cal = json.loads(Path(a.params).read_text())
    bp = BandParams(**cal["band_params"])
    vol = Volume(a.volume, a.level)
    surf = load_tifxyz(a.surface)
    prof = sample_profiles(surf, vol, workers=a.workers)
    ev = evaluate(prof, bp, a.voxel_um * vol.scale)
    sm = summarise(ev)
    sm["certified"] = certificate(sm, cal["patch_bar"])
    sm["patch_bar"] = cal["patch_bar"]
    sm["band_params"] = asdict(bp)
    print(json.dumps(sm, indent=1))
    if a.write:
        d = Path(a.surface)
        meta = json.loads((d / "meta.json").read_text())
        meta.setdefault("quality", {})["sheetcert_v2"] = sm
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        np.save(d / "band.npy", pass_map(surf, prof, ev))


if __name__ == "__main__":
    main()


__all__ = ["SampleParams", "BandParams", "sample_profiles", "band_test", "sheet_spacing", "evaluate",
           "summarise", "certificate", "pass_map"]
