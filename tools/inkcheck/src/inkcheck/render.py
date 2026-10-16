"""Render a surface into a stack of layers along its normal ("surface volume").

Pixel (row, col) of the render corresponds to grid coordinate (row / step, col / step), i.e. one
pixel per level-0 voxel along the surface, as ``vc_render_tifxyz --scale 1``. Layer k samples the
CT at ``offsets[k]`` voxels along the oriented normal (trilinear). Positions and normals between
grid nodes are interpolated bilinearly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

from .surface import Surface
from .volume import Volume, linear_stencil_chunks


def _interp_grid(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of an (H, W, C) grid at fractional (rows, cols); NaN propagates."""
    H, W, C = field.shape
    r0 = np.clip(np.floor(rows).astype(int), 0, H - 2)
    c0 = np.clip(np.floor(cols).astype(int), 0, W - 2)
    fr = (rows - r0)[..., None]
    fc = (cols - c0)[..., None]
    f00 = field[r0, c0]
    f01 = field[r0, c0 + 1]
    f10 = field[r0 + 1, c0]
    f11 = field[r0 + 1, c0 + 1]
    return (1 - fr) * ((1 - fc) * f00 + fc * f01) + fr * ((1 - fc) * f10 + fc * f11)


def pixel_geometry(surf: Surface, n: np.ndarray, r0: int, r1: int, c0: int, c1: int):
    """Positions (h, w, 3) and unit normals (h, w, 3) for render pixels [r0:r1, c0:c1]."""
    rr, cc = np.meshgrid(np.arange(r0, r1) / surf.step, np.arange(c0, c1) / surf.step, indexing="ij")
    p = _interp_grid(surf.zyx, rr, cc)
    nn = _interp_grid(n, rr, cc)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
    return p, nn


def render_region(surf: Surface, n: np.ndarray, vol: Volume, r0: int, r1: int, c0: int, c1: int,
                  offsets: np.ndarray, sparse: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Render pixels [r0:r1, c0:c1] at the given normal offsets.

    ``sparse=False`` fetches every chunk of the tile's bounding box (the original reader, same
    output, about twice the bytes on a curved sheet); kept for benchmarking.

    Returns (layers (D, h, w) float32, valid (h, w) bool).
    """
    p, nn = pixel_geometry(surf, n, r0, r1, c0, c1)
    valid = np.all(np.isfinite(p), -1) & np.all(np.isfinite(nn), -1)
    D = len(offsets)
    out = np.zeros((D, r1 - r0, c1 - c0), dtype=np.float32)
    if not valid.any():
        return out, valid
    P = p[valid]
    N = nn[valid]
    pts = P[None, :, :] + np.asarray(offsets, float)[:, None, None] * N[None, :, :]
    flat = pts.reshape(-1, 3) / vol.scale
    lo = np.floor(flat.min(0)).astype(int) - 2
    hi = np.ceil(flat.max(0)).astype(int) + 3
    # fetch only the chunks the trilinear stencil touches; the rest of the box stays fill value and
    # is never read by map_coordinates (order=1: no spline prefilter, 2-voxel support per axis)
    only = linear_stencil_chunks(flat, vol.array.meta.chunks) if sparse else None
    box = vol.read_box(lo, hi, only=only).astype(np.float32)
    vals = ndimage.map_coordinates(box, (flat - lo).T, order=1, mode="nearest").reshape(D, -1)
    out[:, valid] = vals
    return out, valid


def render(surf: Surface, n: np.ndarray, vol: Volume, offsets: np.ndarray, region=None,
           tile: int = 384, workers: int = 4, progress=None, sparse: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Render a region (r0, r1, c0, c1) in pixels, default the whole surface, tile by tile."""
    H, W = surf.shape
    full = (0, int((H - 1) * surf.step) + 1, 0, int((W - 1) * surf.step) + 1)
    r0, r1, c0, c1 = region or full
    D = len(offsets)
    layers = np.zeros((D, r1 - r0, c1 - c0), dtype=np.float32)
    valid = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    jobs = [(a, min(a + tile, r1), b, min(b + tile, c1)) for a in range(r0, r1, tile) for b in range(c0, c1, tile)]

    def run(j):
        return j, render_region(surf, n, vol, *j, offsets, sparse=sparse)

    with ThreadPoolExecutor(workers) as ex:
        for i, ((a, a1, b, b1), (lay, ok)) in enumerate(ex.map(run, jobs)):
            layers[:, a - r0:a1 - r0, b - c0:b1 - c0] = lay
            valid[a - r0:a1 - r0, b - c0:b1 - c0] = ok
            if progress:
                progress(i + 1, len(jobs))
    return layers, valid
