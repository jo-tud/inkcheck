"""Run villa flat ink models (e.g. scrollprize/ink_9um) on in-memory surface volumes.

Mirrors ``vesuvius.ink_detection.inference.infer``: per-patch robust normalisation, centred
layer window, sliding window with Hann blending. Working in memory lets the diagnostics change the
input (depth window, transplanted ink) without writing zarr files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LoadedModel:
    name: str
    model: torch.nn.Module
    patch: int
    depth: int
    preprocessing: str
    amp_dtype: torch.dtype | None
    device: torch.device


def load_model(checkpoint: str, name: str | None = None, device: str | None = None,
                vram_gb: float | None = None, cpu_threads: int | None = None,
                amp: str = "auto") -> LoadedModel:
    """Load a checkpoint onto an explicit device with an explicit VRAM and CPU-thread cap.

    ``device``/``vram_gb``/``cpu_threads`` default to ``INKCHECK_DEVICE``/``INKCHECK_VRAM_GB``/
    ``INKCHECK_CPU_THREADS`` (see ``inkcheck.gpu.resolve_device``), so two jobs sharing
    one machine do not collide by both defaulting to every core or the same card.
    """
    from vesuvius.ink_detection.inference.infer import configure_model

    from .gpu import resolve_device

    cm = configure_model(argparse.Namespace(checkpoint=checkpoint, amp_dtype=amp))
    choice = resolve_device(device, vram_gb, label=name or checkpoint, cpu_threads=cpu_threads)
    model = cm.model.to(choice.device).eval()
    return LoadedModel(name or checkpoint, model, cm.patch_size, cm.input_depth, cm.preprocessing,
                        cm.amp_dtype, choice.device)


def _hann2d(size: int) -> np.ndarray:
    w = np.hanning(size)
    w2 = np.outer(w, w)
    w2 /= w2.max()
    return np.maximum(w2, 1e-3).astype(np.float32)


def _normalize(patch_zyx: np.ndarray, preprocessing: str) -> np.ndarray:
    from vesuvius.ink_detection.inference.infer import normalize_flat_patch

    if preprocessing == "divide_255":
        return np.ascontiguousarray(np.clip(patch_zyx, 0, 255), dtype=np.float32) / 255.0
    return normalize_flat_patch(np.asarray(patch_zyx, dtype=np.float32), preprocessing)


def predict(m: LoadedModel, layers: np.ndarray, valid: np.ndarray | None = None, center: int | None = None,
            direction: str = "forward", stride: int | None = None, batch: int = 16) -> np.ndarray:
    """Ink probability (H, W) for a surface volume ``layers`` (D, H, W).

    ``center`` is the layer index the model window is centred on (default: middle layer).
    ``direction='reverse'`` feeds the window in reverse layer order. Pixels outside ``valid``
    are NaN.
    """
    D, H, W = layers.shape
    k = m.depth
    c = D // 2 if center is None else int(center)
    lo = c - k // 2
    hi = lo + k
    if lo < 0 or hi > D:
        raise ValueError(f"window [{lo}, {hi}) outside {D} layers")
    win = layers[lo:hi]
    if direction == "reverse":
        win = win[::-1]
    P = m.patch
    stride = stride or P // 2
    ys = list(range(0, max(H - P, 0) + 1, stride))
    xs = list(range(0, max(W - P, 0) + 1, stride))
    if ys[-1] + P < H:
        ys.append(H - P)
    if xs[-1] + P < W:
        xs.append(W - P)
    pad_h, pad_w = max(P - H, 0), max(P - W, 0)
    if pad_h or pad_w:
        win = np.pad(win, ((0, 0), (0, pad_h), (0, pad_w)))
    acc = np.zeros(win.shape[1:], np.float32)
    wsum = np.zeros(win.shape[1:], np.float32)
    wmap = _hann2d(P)
    vmask = None if valid is None else np.pad(valid, ((0, pad_h), (0, pad_w)))
    coords = [(y, x) for y in ys for x in xs]
    if vmask is not None:
        coords = [(y, x) for y, x in coords if vmask[y:y + P, x:x + P].any()]
    autocast = torch.autocast("cuda", dtype=m.amp_dtype) if (m.device.type == "cuda" and m.amp_dtype) else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), autocast:
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            arr = np.stack([_normalize(win[:, y:y + P, x:x + P], m.preprocessing) for y, x in chunk])
            t = torch.from_numpy(arr).unsqueeze(1).to(m.device)
            logits = m.model(t)
            prob = torch.sigmoid(logits.float())
            if prob.shape[-2:] != (P, P):
                prob = torch.nn.functional.interpolate(prob, size=(P, P), mode="bilinear", align_corners=False)
            prob = prob[:, 0].cpu().numpy()
            for (y, x), pr in zip(chunk, prob):
                acc[y:y + P, x:x + P] += pr * wmap
                wsum[y:y + P, x:x + P] += wmap
    out = np.where(wsum > 0, acc / np.maximum(wsum, 1e-6), np.nan)[:H, :W]
    if valid is not None:
        out[~valid] = np.nan
    return out
