"""tifxyz surfaces: loading, normals, orientation, and perturbation.

A tifxyz surface is a grid of 3D points (x.tif, y.tif, z.tif, float32, level-0 voxel units) plus
meta.json. ``scale`` in meta.json is grid density (1/step): 0.05 means a node every 20 voxels.
Invalid nodes have z <= 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


@dataclass
class Surface:
    zyx: np.ndarray  # (H, W, 3) float64 level-0 voxel coordinates, NaN where invalid
    step: float  # voxels between neighbouring grid nodes
    meta: dict = field(default_factory=dict)
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return self.zyx.shape[:2]

    @property
    def valid(self) -> np.ndarray:
        return np.all(np.isfinite(self.zyx), axis=-1)

    def area_cm2(self, voxel_um: float) -> float:
        """Area from the grid: sum of |du x dv| per valid cell."""
        p = self.zyx
        du = p[:-1, 1:] - p[:-1, :-1]
        dv = p[1:, :-1] - p[:-1, :-1]
        a = np.linalg.norm(np.cross(du, dv), axis=-1)
        return float(np.nansum(a) * (voxel_um * 1e-4) ** 2)


def load_tifxyz(path: str | Path) -> Surface:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    x = tifffile.imread(path / "x.tif").astype(np.float64)
    y = tifffile.imread(path / "y.tif").astype(np.float64)
    z = tifffile.imread(path / "z.tif").astype(np.float64)
    zyx = np.stack([z, y, x], axis=-1)
    bad = (z <= 0) | ~np.isfinite(z) | ~np.isfinite(y) | ~np.isfinite(x)
    zyx[bad] = np.nan
    scale = meta.get("scale")
    if scale is None:
        step = 20.0
    else:
        s = float(scale[0]) if isinstance(scale, (list, tuple)) else float(scale)
        step = 1.0 / s if s < 1 else s  # tolerate files that store the step instead of the scale
    return Surface(zyx=zyx, step=step, meta=meta, name=path.name)


def normals(surf: Surface) -> np.ndarray:
    """Unit normals (H, W, 3) in zyx from central differences on the grid; NaN at invalid nodes.

    Sign follows the grid orientation (rows x cols); use ``orient_outward`` for a physical sign.
    """
    p = surf.zyx
    dr = np.full_like(p, np.nan)
    dc = np.full_like(p, np.nan)
    dr[1:-1] = (p[2:] - p[:-2]) / 2
    dc[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / 2
    # one-sided differences at the borders and next to invalid nodes
    fwd_r = p[1:] - p[:-1]
    fwd_c = p[:, 1:] - p[:, :-1]
    fill_r = np.isnan(dr[..., 0])
    dr[:-1][fill_r[:-1]] = fwd_r[fill_r[:-1]]
    fill_r = np.isnan(dr[..., 0])
    dr[1:][fill_r[1:]] = fwd_r[fill_r[1:]]
    fill_c = np.isnan(dc[..., 0])
    dc[:, :-1][fill_c[:, :-1]] = fwd_c[fill_c[:, :-1]]
    fill_c = np.isnan(dc[..., 0])
    dc[:, 1:][fill_c[:, 1:]] = fwd_c[fill_c[:, 1:]]
    n = np.cross(dr, dc)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def axis_from_umbilicus(points_zyx: np.ndarray):
    """Return f(z) -> (y, x) of the scroll axis by linear interpolation of umbilicus points."""
    pts = np.asarray(points_zyx, dtype=float)
    pts = pts[np.argsort(pts[:, 0])]

    def f(z):
        z = np.asarray(z, dtype=float)
        return np.stack([np.interp(z, pts[:, 0], pts[:, 1]), np.interp(z, pts[:, 0], pts[:, 2])], -1)

    return f


def orient_outward(surf: Surface, n: np.ndarray, axis_fn) -> tuple[np.ndarray, float]:
    """Flip normals so that most point away from the scroll axis.

    Returns the oriented normals and the fraction of nodes that agreed with the chosen sign
    (close to 1 for a sheet-following surface; near 0.5 for one crossing the axis).
    """
    p = surf.zyx
    ok = surf.valid & np.all(np.isfinite(n), -1)
    yx_axis = axis_fn(p[..., 0])
    radial = np.zeros_like(p)
    radial[..., 1:] = p[..., 1:] - yx_axis
    dot = np.einsum("hwc,hwc->hw", n, radial)
    frac_pos = float((dot[ok] > 0).mean()) if ok.any() else 0.5
    if frac_pos < 0.5:
        n = -n
        frac_pos = 1 - frac_pos
    return n, frac_pos


def estimate_axis(volume_lowres, level: int, z_step_l0: int = 256):
    """Crude scroll axis: per z slab, centroid of nonzero voxels at a coarse pyramid level.

    ``volume_lowres`` is an inkcheck.volume.Volume at ``level``. Good enough to orient normals,
    not to fit a spiral.
    """
    s = 2 ** level
    zl, yl, xl = volume_lowres.shape
    pts = []
    for z0 in range(0, zl * s, z_step_l0):
        zc = min(z0 // s, zl - 1)
        sl = volume_lowres.read_box((zc, 0, 0), (zc + 1, yl, xl))[0]
        m = ndimage.binary_opening(sl > 0, iterations=1)
        if m.sum() < 50:
            continue
        yy, xx = np.nonzero(m)
        pts.append((zc * s, yy.mean() * s, xx.mean() * s))
    if not pts:
        raise ValueError("volume looks empty at this level")
    return axis_from_umbilicus(np.array(pts))


def shifted(surf: Surface, n: np.ndarray, offset: float) -> Surface:
    """Copy of the surface moved by ``offset`` voxels along its normals (for controlled tests)."""
    return Surface(zyx=surf.zyx + offset * n, step=surf.step, meta=dict(surf.meta), name=f"{surf.name}+{offset:g}")
