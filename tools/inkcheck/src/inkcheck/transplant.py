"""Detectability via ink transplant.

Idea: take real ink from a scroll where letters are known (e.g. PHerc0343 at 8.64 µm), isolate
its signature in the surface volume, add it at controlled strength into a target surface volume,
and measure whether the ink model finds it there. This separates "the pipeline cannot use ink of
known strength on this background" from "there is no ink here".

Ink signature. For every pixel inside a known ink mask, the signature is the difference between
the observed layers and what the same pixel would look like without ink, estimated by
normalised-convolution inpainting from nearby non-ink pixels of the same layer. Outside the mask
the signature is zero, so only the letter strokes are moved.

Caveat (to state with every result): the transplant is additive in the rendered surface volume.
It tests the model and the target's background/noise, not the physics of ink in that scroll.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def inpaint_background(layers: np.ndarray, ink: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """Estimate ink-free layers inside ``ink`` from surrounding pixels (normalised convolution).

    layers: (D, H, W) float; ink: (H, W) bool. Returns (D, H, W) float32.
    """
    keep = (~ink).astype(np.float32)
    w = ndimage.gaussian_filter(keep, sigma)
    out = np.empty_like(layers, dtype=np.float32)
    for k in range(layers.shape[0]):
        num = ndimage.gaussian_filter(layers[k].astype(np.float32) * keep, sigma)
        out[k] = np.where(w > 1e-3, num / np.maximum(w, 1e-3), layers[k])
    return out


@dataclass
class InkSignature:
    residual: np.ndarray  # (D, h, w) float32, zero outside the stroke mask
    mask: np.ndarray  # (h, w) bool stroke mask
    center: int  # layer index of the sheet / best ink depth within ``residual``
    voxel_um: float
    source: str = ""


def extract_signatures(layers: np.ndarray, ink: np.ndarray, boxes: list[tuple[int, int, int, int]],
                       center: int, half_depth: int, voxel_um: float, sigma: float = 12.0,
                       source: str = "") -> list[InkSignature]:
    """Cut one signature per box (r0, r1, c0, c1) from a source surface volume.

    ``ink`` is the stroke mask over the whole render; ``center`` the layer index where ink sits
    (e.g. the best depth of the positive control); layers center±half_depth are kept.
    """
    lo, hi = center - half_depth, center + half_depth + 1
    sub = layers[lo:hi].astype(np.float32)
    bg = inpaint_background(sub, ink, sigma)
    res = (sub - bg) * ink[None]
    sigs = []
    for r0, r1, c0, c1 in boxes:
        m = ink[r0:r1, c0:c1]
        if m.sum() < 50:
            continue
        sigs.append(InkSignature(res[:, r0:r1, c0:c1].copy(), m.copy(), half_depth, voxel_um, source))
    return sigs


def resample(sig: InkSignature, voxel_um: float) -> InkSignature:
    """Rescale a signature to another voxel size (in-plane and depth)."""
    if abs(sig.voxel_um - voxel_um) < 1e-6:
        return sig
    f = sig.voxel_um / voxel_um
    res = ndimage.zoom(sig.residual, (f, f, f), order=1)
    m = ndimage.zoom(sig.mask.astype(np.float32), (f, f), order=1) > 0.5
    h, w = min(res.shape[1], m.shape[0]), min(res.shape[2], m.shape[1])
    return InkSignature(res[:, :h, :w], m[:h, :w], int(round(sig.center * f)), voxel_um, sig.source)


def place(target: np.ndarray, valid: np.ndarray, sigs: list[InkSignature], alpha: float, center: int | np.ndarray,
          rng: np.random.Generator, n: int, scale: float = 1.0, avoid: np.ndarray | None = None,
          max_tries: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Add ``n`` randomly chosen signatures at random valid positions.

    ``center`` is the target layer index where the sheet surface is (scalar, or (H, W) map from
    geometry). ``scale`` converts source intensity units into target units (ratio of robust
    spreads). Returns (modified copy of target, placed-ink mask).
    """
    out = target.astype(np.float32).copy()
    D, H, W = target.shape
    placed = np.zeros((H, W), bool)
    blocked = np.zeros((H, W), bool) if avoid is None else avoid.copy()
    for _ in range(n):
        s = sigs[rng.integers(len(sigs))]
        sd, sh, sw = s.residual.shape
        if sh >= H or sw >= W:
            continue
        for _try in range(max_tries):
            r = int(rng.integers(0, H - sh))
            c = int(rng.integers(0, W - sw))
            win_valid = valid[r:r + sh, c:c + sw]
            if win_valid[s.mask].all() and not blocked[r:r + sh, c:c + sw][s.mask].any():
                break
        else:
            continue
        ctr = int(center if np.isscalar(center) else np.nanmedian(center[r:r + sh, c:c + sw]))
        z0 = ctr - s.center
        a, b = max(z0, 0), min(z0 + sd, D)
        if b <= a:
            continue
        out[a:b, r:r + sh, c:c + sw] += alpha * scale * s.residual[a - z0:b - z0]
        placed[r:r + sh, c:c + sw] |= s.mask
        blocked[max(r - 20, 0):r + sh + 20, max(c - 20, 0):c + sw + 20] = True
    return out, placed


def robust_spread(layers: np.ndarray, valid: np.ndarray, center: int, half: int = 4) -> float:
    """MAD-based spread of the layers around the sheet, used to match intensity units."""
    v = layers[center - half:center + half + 1][:, valid]
    med = np.median(v)
    return float(1.4826 * np.median(np.abs(v - med)))


def separation(pred: np.ndarray, pos: np.ndarray, neg: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    """P(pred at a random positive pixel > pred at a random negative pixel)."""
    a, b = pred[pos], pred[neg]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    a = rng.choice(a, min(n, len(a)))
    b = np.sort(rng.choice(b, min(n, len(b))))
    return float(np.searchsorted(b, a, side="left").mean() / len(b))
