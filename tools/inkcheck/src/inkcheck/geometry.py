"""Does the surface follow the papyrus sheet?

Per grid node, three independent measurements:

* CT profile along the mesh normal (averaged over a small tangential stencil): position,
  prominence and width of the bright papyrus band nearest the mesh.
* m7 surface-prediction profile along the same normal: support on the mesh minus support a few
  voxels away (the rule from villa #1906, as used by the Bullo27 survey).
* Local sheet orientation from the CT structure tensor: angle between the mesh normal and the
  dominant intensity-gradient direction. A mesh that cuts through layers ("swirls") has a large
  angle even where it happens to sit on bright material.

Work is done tile by tile (a tile is a square block of grid nodes) so that each tile needs one
compact read of the volume.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from .surface import Surface
from .volume import Volume


@dataclass
class GeometryParams:
    radius: int = 16  # sample offsets -radius..radius along the normal (voxels)
    stencil: int = 1  # tangential stencil half-size in nodes-of-4-voxels: 1 -> 3x3
    stencil_spacing: float = 4.0  # voxels between stencil points
    tile_nodes: int = 12  # tile edge in grid nodes
    st_downsample: int = 2  # structure tensor computed on a 2x block-mean of the box
    st_sigma_d: float = 1.0  # derivative sigma (downsampled voxels)
    st_sigma_i: float = 3.0  # integration sigma (downsampled voxels)
    peak_prominence: float = 4.0  # minimum CT peak prominence (uint8 units) to count as a sheet
    m7_near: int = 2  # |offset| <= near counts as "on the mesh"
    m7_far: tuple[int, int] = (5, 8)  # background band for m7 support
    rival_window: int = 8  # "rivals" counts other peaks with |offset| <= this ...
    rival_ratio: float = 0.5  # ... and prominence >= this fraction of the nearest peak's
    workers: int = 4


NODE_FIELDS = (
    "d0",  # offset of the CT sheet peak nearest the mesh (voxels, + along normal); NaN if none
    "prominence",  # prominence of that peak (uint8 units)
    "width",  # FWHM of that peak (voxels)
    "n_peaks",  # prominent peaks within +-radius
    "rivals",  # other peaks within +-rival_window of the mesh at >= rival_ratio of d0's prominence
    "center_contrast",  # (p(0) - min) / (max - min) of the profile
    "m7_support",  # m7 hit rate |o|<=near minus hit rate in the far band
    "m7_on",  # m7 hit rate |o|<=near
    "m7_d0",  # offset of the centre of the m7 run nearest the mesh (voxels); NaN if no run
    "st_angle",  # degrees between mesh normal and CT sheet normal
    "st_coherence",  # (l1 - l2) / (l1 + l2) of the structure tensor
    "radial_align",  # |n . r_hat|: 1 = normal points away from the scroll axis (only with an axis)
)


def _tangents(surf: Surface, n: np.ndarray):
    p = surf.zyx
    t = np.full_like(p, np.nan)
    t[:, 1:-1] = p[:, 2:] - p[:, :-2]
    t[:, 0] = p[:, 1] - p[:, 0]
    t[:, -1] = p[:, -1] - p[:, -2]
    t -= np.einsum("hwc,hwc->hw", t, n)[..., None] * n
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    b = np.cross(n, t)
    return t, b


def _profile_features(prof: np.ndarray, offsets: np.ndarray, params: GeometryParams):
    """prof: (N, K) CT profiles. Returns dict of (N,) arrays."""
    N = prof.shape[0]
    out = {k: np.full(N, np.nan) for k in ("d0", "prominence", "width", "n_peaks", "rivals", "center_contrast")}
    sm = ndimage.uniform_filter1d(prof, 3, axis=1)
    mid = len(offsets) // 2
    for i in range(N):
        p = sm[i]
        if not np.all(np.isfinite(p)):
            continue
        lo, hi = p.min(), p.max()
        out["center_contrast"][i] = (p[mid] - lo) / (hi - lo) if hi > lo else np.nan
        peaks, props = signal.find_peaks(p, prominence=params.peak_prominence)
        out["n_peaks"][i] = len(peaks)
        if len(peaks) == 0:
            continue
        j = int(np.argmin(np.abs(offsets[peaks])))
        pk = peaks[j]
        # sub-voxel refinement with a parabola through the peak and its neighbours
        d = float(offsets[pk])
        if 0 < pk < len(p) - 1:
            a, b, c = p[pk - 1], p[pk], p[pk + 1]
            den = a - 2 * b + c
            if den < 0:
                d += 0.5 * (a - c) / den
        out["d0"][i] = d
        out["prominence"][i] = props["prominences"][j]
        near = np.abs(offsets[peaks]) <= params.rival_window
        strong = props["prominences"] >= params.rival_ratio * props["prominences"][j]
        out["rivals"][i] = int(np.sum(near & strong)) - int(near[j] and strong[j])
        w = signal.peak_widths(p, [pk], rel_height=0.5)[0][0]
        out["width"][i] = w * (offsets[1] - offsets[0])
    return out


def _nearest_run_centre(on: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """on: (N, K) bool. Centre offset of the run of True nearest offset 0, per row."""
    out = np.full(on.shape[0], np.nan)
    for i, row in enumerate(on):
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0] - 1
        centres = (offsets[starts] + offsets[ends]) / 2
        out[i] = centres[np.argmin(np.abs(centres))]
    return out


def _structure_tensor_at(box: np.ndarray, origin: np.ndarray, pts_zyx: np.ndarray, params: GeometryParams):
    """Dominant gradient direction (sheet normal) and coherence at points, from a CT box."""
    f = params.st_downsample
    if f > 1:
        zs, ys, xs = (s // f * f for s in box.shape)
        b = box[:zs, :ys, :xs].reshape(zs // f, f, ys // f, f, xs // f, f).mean(axis=(1, 3, 5))
    else:
        b = box
    b = b.astype(np.float32)
    g = [ndimage.gaussian_filter(b, params.st_sigma_d, order=o) for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    rel = ((pts_zyx - origin) / f - (0.5 * (f - 1) / f)).T  # block-mean centre correction
    comps = {}
    for i in range(3):
        for j in range(i, 3):
            t = ndimage.gaussian_filter(g[i] * g[j], params.st_sigma_i)
            comps[(i, j)] = ndimage.map_coordinates(t, rel, order=1, mode="nearest")
    n = pts_zyx.shape[0]
    T = np.empty((n, 3, 3), dtype=np.float64)
    for (i, j), v in comps.items():
        T[:, i, j] = v
        T[:, j, i] = v
    w, V = np.linalg.eigh(T)  # ascending
    e1 = V[:, :, 2]
    l1, l2 = w[:, 2], w[:, 1]
    coh = (l1 - l2) / np.maximum(l1 + l2, 1e-12)
    return e1, coh


def analyze_tile(vol: Volume, pred: Volume | None, p: np.ndarray, n: np.ndarray, t: np.ndarray, b: np.ndarray,
                 params: GeometryParams) -> dict:
    """Measure one tile. p, n, t, b: (h, w, 3) node positions, normals, tangents (NaN = invalid)."""
    h, w = p.shape[:2]
    ok = np.all(np.isfinite(p), -1) & np.all(np.isfinite(n), -1) & np.all(np.isfinite(t), -1)
    res = {k: np.full((h, w), np.nan) for k in NODE_FIELDS}
    if not ok.any():
        return res
    P, N, T, B = p[ok], n[ok], t[ok], b[ok]
    R = params.radius
    offsets = np.arange(-R, R + 1, dtype=np.float64)
    s = params.stencil
    st = np.arange(-s, s + 1) * params.stencil_spacing
    su, sv = np.meshgrid(st, st, indexing="ij")
    su, sv = su.ravel(), sv.ravel()
    # points: (nodes, stencil, offsets, 3)
    pts = (P[:, None, None, :]
           + su[None, :, None, None] * T[:, None, None, :]
           + sv[None, :, None, None] * B[:, None, None, :]
           + offsets[None, None, :, None] * N[:, None, None, :])
    flat = pts.reshape(-1, 3)
    margin = 4 + int(3 * params.st_sigma_i * params.st_downsample)
    lo = np.floor(flat.min(0)).astype(int) - margin
    hi = np.ceil(flat.max(0)).astype(int) + margin + 1
    box = vol.read_box(lo, hi).astype(np.float32)
    ct = ndimage.map_coordinates(box, (flat - lo).T, order=1, mode="nearest").reshape(pts.shape[:3])
    prof = ct.mean(axis=1)
    feats = _profile_features(prof, offsets, params)
    for k, v in feats.items():
        res[k][ok] = v

    if pred is not None:
        pb = pred.read_box(lo, hi)
        mv = ndimage.map_coordinates((pb > 0).astype(np.float32), (flat - lo).T, order=0, mode="nearest")
        mprof = mv.reshape(pts.shape[:3]).mean(axis=1)
        near = np.abs(offsets) <= params.m7_near
        far = (np.abs(offsets) >= params.m7_far[0]) & (np.abs(offsets) <= params.m7_far[1])
        on = mprof[:, near].mean(1)
        res["m7_on"][ok] = on
        res["m7_support"][ok] = on - mprof[:, far].mean(1)
        res["m7_d0"][ok] = _nearest_run_centre(mprof > 0.5, offsets)

    e1, coh = _structure_tensor_at(box, lo, P, params)
    cosang = np.abs(np.einsum("nc,nc->n", e1, N))
    res["st_angle"][ok] = np.degrees(np.arccos(np.clip(cosang, 0, 1)))
    res["st_coherence"][ok] = coh
    return res


def radial_alignment(surf: Surface, n: np.ndarray, axis_fn) -> np.ndarray:
    """|n . r_hat| per node, r_hat the horizontal direction away from the scroll axis."""
    p = surf.zyx
    yx = axis_fn(p[..., 0])
    r = np.zeros_like(p)
    r[..., 1:] = p[..., 1:] - yx
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return np.abs(np.einsum("hwc,hwc->hw", n, r))


def analyze_surface(surf: Surface, n: np.ndarray, vol: Volume, pred: Volume | None = None,
                    params: GeometryParams | None = None, progress=None, axis_fn=None,
                    tiles: list[tuple[int, int]] | None = None) -> dict:
    """Per-node geometry fields: dict of (H, W) arrays (NaN outside the analysed tiles).

    ``tiles`` restricts the analysis to these tile origins (row, col in nodes), e.g. a random
    sample for cheap patch-level estimates; default all tiles with valid nodes.
    """
    params = params or GeometryParams()
    t, b = _tangents(surf, n)
    H, W = surf.shape
    out = {k: np.full((H, W), np.nan) for k in NODE_FIELDS}
    ts = params.tile_nodes
    if tiles is None:
        tiles = list_tiles(surf, ts)

    def run(rc):
        r, c = rc
        sl = (slice(r, r + ts), slice(c, c + ts))
        return rc, analyze_tile(vol, pred, surf.zyx[sl], n[sl], t[sl], b[sl], params)

    done = 0
    with ThreadPoolExecutor(params.workers) as ex:
        for (r, c), res in ex.map(run, tiles):
            for k in NODE_FIELDS:
                out[k][r:r + ts, c:c + ts] = res[k]
            done += 1
            if progress:
                progress(done, len(tiles))
    if axis_fn is not None:
        ra = radial_alignment(surf, n, axis_fn)
        out["radial_align"] = np.where(np.isfinite(out["st_angle"]), ra, np.nan)
    return out


def list_tiles(surf: Surface, tile_nodes: int, min_valid: float = 0.0) -> list[tuple[int, int]]:
    """Tile origins (row, col) whose fraction of valid nodes exceeds ``min_valid``."""
    H, W = surf.shape
    v = surf.valid
    return [(r, c) for r in range(0, H, tile_nodes) for c in range(0, W, tile_nodes)
            if v[r:r + tile_nodes, c:c + tile_nodes].mean() > min_valid and v[r:r + tile_nodes, c:c + tile_nodes].any()]
