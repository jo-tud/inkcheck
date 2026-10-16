"""PHerc. 343 positive control, v3: between-line negatives, the reference's four-model ensemble, parity check
(pass/fail rules in README.md).

Stages, each cached so a crashed run resumes:
  1. infer   dnative reverse d=-15..+15 and hecate reverse d=+1 on the v2 render (CPU) -> probs_dn.npy, hecate_d+1.npy
  2. synth   synthetic check of the scoring code (planted ink must pass, noise must fail) -> synth.json
  3. score   held-out selection, surrogate null, parity with the published map, figures -> results.json

    python run.py [--stage infer|synth|score|all]

Reuses the v2 render and ink9 maps (../2026-10-12-pherc343-positive-control-v2); v2's run.py rebuilds them if absent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2DIR = HERE.parent / "2026-10-12-pherc343-positive-control-v2"
REF = ROOT / "external/pherc343-first-letters"
CKPT = ROOT / "data/models/pherc343_ref"
NAME = "concat_w047-w048_R5B2_z9500-11000"

DEPTHS = list(range(-15, 16))
HECATE_D = 1
CLEAR, BAND_H = 16, 78                          # between-line bands: clearance and depth (raw px), README
N_NULL = 1000
PUB_LO, PUB_HI = -0.711, 3.864                  # published ensemble stretch: z 0.89 at u 0.35, 2.72 at u 0.75
PARITY_R = 0.80


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = load_module("pherc343_v2", V2DIR / "run.py")


def threads():
    return int(os.environ.get("INKCHECK_CPU_THREADS", "12"))


# ---------------------------------------------------------------- stage 1: infer

def ensure_v2():
    if not (V2DIR / "layers.npy").exists():
        v2.stage_render()
    done = V2DIR / "probs_done.json"
    if not done.exists() or len(json.load(open(done))) < len(v2.SEEDS) * len(v2.DIRS) * len(v2.DEPTHS):
        v2.stage_infer()


def hecate_input(layers, d):
    """hecate_prep.py for a 66-layer render: 16 planes at 9.6 um centred on layer 32.5 + d, in-plane 8.64 -> 9.6 um."""
    import cv2
    step = 9.6 / 8.64
    pos = 32.5 + d + (np.arange(16) - 7.5) * step
    lo = np.floor(pos).astype(int)
    frac = (pos - lo).astype(np.float32)
    hi = np.minimum(lo + 1, layers.shape[0] - 1)
    H, W = layers.shape[1:]
    s = 8.64 / 9.6
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    out = np.empty((16, nh, nw), np.uint8)
    for i in range(16):
        z = np.asarray(layers[lo[i]], np.float32) * (1 - frac[i]) + np.asarray(layers[hi[i]], np.float32) * frac[i]
        out[i] = np.clip(cv2.resize(z, (nw, nh), interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)
    return out


def stage_infer():
    import tempfile
    import torch
    from inkcheck.ink import load_model, predict
    (HERE / "tmp").mkdir(exist_ok=True)
    tempfile.tempdir = str(HERE / "tmp")             # hecate's disk accumulators; outlives a temporary scratch dir
    ensure_v2()
    layers = np.load(V2DIR / "layers.npy", mmap_mode="r")
    valid = np.load(V2DIR / "valid.npy")
    donef = HERE / "infer_done.json"
    done = json.load(open(donef)) if donef.exists() else {}

    pf = HERE / "probs_dn.npy"
    probs = (np.lib.format.open_memmap(pf, mode="r+") if pf.exists()
             else np.lib.format.open_memmap(pf, mode="w+", dtype=np.uint8, shape=(len(DEPTHS),) + valid.shape))
    m = load_model(str(CKPT / "dense_native-016000.rebuilt.pth"), "dnative")
    for d in v2.depth_order():
        key = f"dnative/reverse/{d}"
        if key in done:
            continue
        t = time.time()
        win = np.asarray(layers[25 + d:42 + d]).astype(np.float32)   # 17 layers centred on 33 + d
        pr = predict(m, win, valid, center=8, direction="reverse")
        probs[DEPTHS.index(d)] = np.clip(np.rint(np.nan_to_num(pr) * 255), 0, 255).astype(np.uint8)
        probs.flush()
        done[key] = round(time.time() - t, 1)
        json.dump(done, open(donef, "w"), indent=0)
        print(f"infer dnative reverse d={d:+d} {done[key]:.0f}s ({len(done)})", flush=True)
    del m

    key = f"hecate/reverse/{HECATE_D}"
    if key not in done:
        hec = load_module("hecate", CKPT / "hecate.py")
        torch.set_num_threads(threads())
        hm = hec.load_model(str(CKPT / "hecate_9.6um.pth"), "cpu")
        t = time.time()
        inp = hecate_input(layers, HECATE_D)
        out = np.zeros(inp.shape[1:], np.uint8)
        hec.predict(hm, inp, out, reverse=True, batch_size=4)
        np.save(HERE / f"hecate_d{HECATE_D:+d}.npy", out)
        done[key] = round(time.time() - t, 1)
        json.dump(done, open(donef, "w"), indent=0)
        print(f"infer hecate reverse d={HECATE_D:+d} {done[key]:.0f}s", flush=True)



# ---------------------------------------------------------------- geometry: letters and between-line bands

P = v2.POOL


def raw_to_pool_in(r):
    """raw rect (x0, y0, x1, y1) -> pooled (r0, r1, c0, c1), rounded inward (cells fully inside)."""
    x0, y0, x1, y1 = r
    return (int(np.ceil((y0 - v2.R0) / P)), int(np.floor((y1 - v2.R0) / P)),
            int(np.ceil((x0 - v2.C0) / P)), int(np.floor((x1 - v2.C0) / P)))


def bands_of(box):
    x0, y0, x1, y1 = box
    return [(x0, y1 + CLEAR, x1, y1 + CLEAR + BAND_H), (x0, y0 - CLEAR - BAND_H, x1, y0 - CLEAR)]


def exclusion_mask(shape, boxes):
    """pooled cells within CLEAR px of any raw box (rounded outward)."""
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        r0, r1, c0, c1 = v2.to_pool((x0 - CLEAR, y0 - CLEAR, x1 + CLEAR, y1 + CLEAR))
        m[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)] = True
    return m


class Fold:
    """One fold on the pooled grid: letter boxes (positives) and between-line bands (negatives)."""

    def __init__(self, name, pv, letters_raw, all_boxes_raw):
        self.name, self.pv = name, pv
        a, b = v2.FOLDS[name]
        self.ca, self.cb = int(np.ceil((a - v2.C0) / P)), int(np.floor((b - v2.C0) / P))
        self.set_letters(letters_raw, all_boxes_raw)

    def inside(self, r):
        x0, _, x1, _ = r
        a, b = v2.FOLDS[self.name]
        return x0 >= a and x1 <= b

    def set_letters(self, letters_raw, all_boxes_raw):
        pv = self.pv
        excl = exclusion_mask(pv.shape, all_boxes_raw)
        self.letter_cells, self.band_cells = [], []
        pos = np.zeros_like(pv)
        for bx in letters_raw:
            r0, r1, c0, c1 = v2.to_pool(bx)
            if pv[r0:r1, c0:c1].mean() >= 0.5:
                cells = np.zeros_like(pv); cells[r0:r1, c0:c1] = True
                self.letter_cells.append(cells & pv)
                pos |= cells & pv
            for band in bands_of(bx):
                if not self.inside(band):
                    continue
                r0, r1, c0, c1 = raw_to_pool_in(band)
                r0, r1 = max(r0, 0), min(r1, pv.shape[0])
                if r1 <= r0 or c1 <= c0:
                    continue
                cells = np.zeros_like(pv); cells[r0:r1, c0:c1] = True
                use = cells & pv & ~excl
                if use.sum() >= 0.5 * cells.sum():
                    self.band_cells.append(use)
        self.pos_px = pos
        self.neg_px = np.zeros_like(pv)
        for c in self.band_cells:
            self.neg_px |= c
        self.idx_letters = [np.flatnonzero(c) for c in self.letter_cells]
        self.idx_bands = [np.flatnonzero(c) for c in self.band_cells]
        self.idx_pos, self.idx_neg = np.flatnonzero(self.pos_px), np.flatnonzero(self.neg_px)

    def box_scores(self, pmap):
        f = pmap.ravel()
        return ([float(np.nanpercentile(f[i], 95)) for i in self.idx_letters],
                [float(np.nanpercentile(f[i], 95)) for i in self.idx_bands])

    def score(self, pmap):
        pos, neg = self.box_scores(pmap)
        f = pmap.ravel()
        return {"box_auc": v2.auc(pos, neg), "pixel_auc": v2.auc(f[self.idx_pos], f[self.idx_neg]),
                "n_letters": len(pos), "n_bands": len(neg), "n_pos_px": int(len(self.idx_pos)),
                "n_neg_px": int(len(self.idx_neg))}


def make_folds(pv):
    boxes = v2.load_boxes()
    all_raw = [b["box"] for b in boxes]
    folds = {}
    for f in v2.FOLDS:
        ids = [b["id"] for b in boxes if v2.fold_of(b["box"]) == f]
        folds[f] = Fold(f, pv, [b["box"] for b in boxes if b["id"] in ids], all_raw)
        folds[f].ids = ids
    return folds, boxes


# ---------------------------------------------------------------- nulls

def surrogate(crop, valid, rng):
    """Same power spectrum (phase randomised) and same value distribution (rank remap); invalid stays NaN."""
    vals = crop[valid]
    x = np.where(valid, crop, vals.mean())
    amp = np.abs(np.fft.rfft2(x))
    ph = np.angle(np.fft.rfft2(rng.standard_normal(x.shape)))
    s = np.fft.irfft2(amp * np.exp(1j * ph), s=x.shape)
    out = np.full(crop.shape, np.nan, np.float32)
    order = np.argsort(s[valid], kind="stable")
    remap = np.empty_like(vals)
    remap[order] = np.sort(vals)
    out[valid] = remap
    return out


def surrogate_null(fold, pmap, rng, n=N_NULL):
    c0, c1 = fold.ca, fold.cb
    crop = pmap[:, c0:c1]
    valid = np.isfinite(crop)
    full = np.full(pmap.shape, np.nan, np.float32)
    box, px = [], []
    for _ in range(n):
        full[:, c0:c1] = surrogate(crop, valid, rng)
        s = fold.score(full)
        box.append(s["box_auc"]); px.append(s["pixel_auc"])
    return np.array(box), np.array(px)


def random_box_null(fold, pmap, letters_raw, all_raw, rng, n=N_NULL):
    """v2-style: move each letter box (size kept) to a random spot in the fold (>= 50 % valid, no overlap with a real
    letter box); bands are re-derived around the moved boxes. Reported only, not part of the pass/fail rule."""
    a, b = v2.FOLDS[fold.name]
    pv = fold.pv
    occ = np.zeros_like(pv)
    for bx in all_raw:
        r0, r1, c0, c1 = v2.to_pool(bx); occ[r0:r1, c0:c1] = True
    out_b, out_p = [], []
    for _ in range(n):
        pseudo = []
        for x0, y0, x1, y1 in letters_raw:
            w, h = x1 - x0, y1 - y0
            for _t in range(500):
                nx, ny = rng.uniform(a, b - w), rng.uniform(v2.R0, v2.R1 - h)
                r0, r1, c0, c1 = v2.to_pool((nx, ny, nx + w, ny + h))
                if occ[r0:r1, c0:c1].any() or pv[r0:r1, c0:c1].mean() < 0.5:
                    continue
                pseudo.append((nx, ny, nx + w, ny + h)); break
        g = Fold.__new__(Fold); g.__dict__.update(fold.__dict__)
        g.set_letters(pseudo, pseudo + list(all_raw))
        s = g.score(pmap)
        out_b.append(s["box_auc"]); out_p.append(s["pixel_auc"])
    return np.array(out_b), np.array(out_p)


def label_permutation(fold, pmap, rng, n=N_NULL):
    pos, neg = fold.box_scores(pmap)
    allsc = np.array(pos + neg)
    return np.array([v2.auc(*np.split(rng.permutation(allsc), [len(pos)])) for _ in range(n)])


def meets_bar(box, px):
    return bool(box["observed"] >= 0.80 and px["observed"] >= 0.60
                and box["observed"] > box["null_p99"] and px["observed"] > px["null_p99"])


def evaluate(fold, pmap, rng, extra=True):
    s = fold.score(pmap)
    nb, npx = surrogate_null(fold, pmap, rng)
    out = {"n_letters": s["n_letters"], "n_bands": s["n_bands"], "n_pos_px": s["n_pos_px"], "n_neg_px": s["n_neg_px"],
           "box_auc": v2.null_summary(s["box_auc"], nb), "pixel_auc": v2.null_summary(s["pixel_auc"], npx)}
    out["meets_bar"] = meets_bar(out["box_auc"], out["pixel_auc"])
    if extra:
        boxes = v2.load_boxes()
        letters_raw = [b["box"] for b in boxes if b["id"] in fold.ids]
        rb, rp = random_box_null(fold, pmap, letters_raw, [b["box"] for b in boxes], rng, n=200)
        out["box_auc_random_box_null"] = v2.null_summary(s["box_auc"], rb)
        out["pixel_auc_random_box_null"] = v2.null_summary(s["pixel_auc"], rp)
        out["box_auc_label_permutation"] = v2.null_summary(s["box_auc"], label_permutation(fold, pmap, rng))
    return out


# ---------------------------------------------------------------- stage 2: synthetic check

def smooth_noise(shape, rng, sigma=6):
    from scipy.ndimage import gaussian_filter
    x = gaussian_filter(rng.standard_normal(shape), sigma)
    return (x - x.mean()) / x.std()


def stage_synth():
    from scipy.ndimage import gaussian_filter
    valid = np.load(V2DIR / "valid.npy")
    _, pv = v2.pool(valid.astype(np.float32), valid)
    folds, boxes = make_folds(pv)
    rng = np.random.default_rng(1)
    noise = smooth_noise(pv.shape, rng)
    planted = noise.copy()
    ink = np.zeros(pv.shape)
    for b in boxes:                                   # strokes: a ring per letter box, about a quarter of its area
        r0, r1, c0, c1 = v2.to_pool(b["box"])
        yy, xx = np.mgrid[r0:r1, c0:c1]
        cy, cx, ry, rx = (r0 + r1) / 2, (c0 + c1) / 2, (r1 - r0) / 2.6, (c1 - c0) / 2.6
        rr = np.hypot((yy - cy) / ry, (xx - cx) / rx)
        ink[r0:r1, c0:c1] = np.maximum(ink[r0:r1, c0:c1], (np.abs(rr - 0.8) < 0.15) * 1.0)
    planted += 3 * gaussian_filter(ink, 1.5) / max(gaussian_filter(ink, 1.5).max(), 1e-6)
    # added after the ring result (README, "Synthetic check, result"): the v2-style plant, ink over the whole box
    fill = np.zeros(pv.shape)
    for b in boxes:
        r0, r1, c0, c1 = v2.to_pool(b["box"])
        fill[r0:r1, c0:c1] = 1.0
    planted_fill = noise + 3 * gaussian_filter(fill, 1.5)
    out = {}
    for label, m in (("planted", planted), ("planted_fill", planted_fill), ("noise", noise)):
        m = np.where(pv, m, np.nan).astype(np.float32)
        res = {f: evaluate(fo, m, np.random.default_rng(2), extra=False) for f, fo in folds.items()}
        out[label] = {"folds": res, "passes_bar": all(r["meets_bar"] for r in res.values())}
        print(label, json.dumps(out[label]), flush=True)
    out["ok"] = out["planted"]["passes_bar"] and not out["noise"]["passes_bar"]          # the ring plant, as planned before the run
    out["ok_fill"] = out["planted_fill"]["passes_bar"] and not out["noise"]["passes_bar"]
    json.dump(out, open(HERE / "synth.json", "w"), indent=1)
    print("synthetic check", "OK" if out["ok"] else "FAILED", flush=True)
    return out



# ---------------------------------------------------------------- stage 3: score

MODELS = ("ink9-42", "ink9-43", "ink9-mean", "dnative", "ens3")
COLORS = {"ink9-42": "#2a78d6", "ink9-43": "#eb6834", "ink9-mean": "#1baf7a", "dnative": "#eda100", "ens3": "#e87ba4"}
PUB_PNG = REF / f"outputs/{NAME}.png"


def villa_t(v):
    """ensemble_maps.py: villa uint8 -> clip((v - 64) / 128, 0, 1)."""
    return np.clip((np.asarray(v, np.float32) - 64.0) / 128.0, 0, 1)


def hist_stats(maps_u8, valid):
    """exact mean and SD of villa_t over the valid pixels of all given uint8 maps."""
    h = np.zeros(256, np.int64)
    for m in maps_u8:
        h += np.bincount(np.asarray(m)[valid], minlength=256)
    t = villa_t(np.arange(256))
    n = h.sum()
    mu = float((h * t).sum() / n)
    return mu, float(np.sqrt(max((h * t * t).sum() / n - mu * mu, 0.0))) + 1e-6


def published_alpha(ct34, valid):
    """overlay strength alpha = (R - B) / 135 of the published image, area located by matching the CT (README, parity check).
    Returns alpha in the raw frame (NaN on the drawn row lines and outside valid), offset, CT match r."""
    import cv2
    img = cv2.imread(str(PUB_PNG), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    alpha = (R - B) / 135.0
    ct = np.rot90(ct34.astype(np.float32), 2)
    vr = np.rot90(valid, 2)
    H, W = ct.shape
    f = 4
    ds = lambda x: cv2.resize(x, (x.shape[1] // f, x.shape[0] // f), interpolation=cv2.INTER_AREA)
    cts, vs, gs, zs = ds(ct), ds(vr.astype(np.float32)) > 0.99, ds(R), ds((np.abs(alpha) < 0.01).astype(np.float32)) > 0.99
    best = (-1.0, 0, 0)
    for oy in range((img.shape[0] - H) // f + 1):
        for ox in range((img.shape[1] - W) // f + 1):
            m = vs & zs[oy:oy + cts.shape[0], ox:ox + cts.shape[1]]
            if m.sum() < 1000:
                continue
            r = np.corrcoef(gs[oy:oy + cts.shape[0], ox:ox + cts.shape[1]][m], cts[m])[0, 1]
            best = max(best, (float(r), oy * f, ox * f))
    _, oy0, ox0 = best
    best = (-1.0, 0, 0)
    for oy in range(max(oy0 - 6, 0), oy0 + 7):
        for ox in range(max(ox0 - 6, 0), ox0 + 7):
            m = vr & (np.abs(alpha[oy:oy + H, ox:ox + W]) < 0.01)
            r = np.corrcoef(R[oy:oy + H, ox:ox + W][m][::7], ct[m][::7])[0, 1]
            best = max(best, (float(r), oy, ox))
    r, oy, ox = best
    a = alpha[oy:oy + H, ox:ox + W]
    gres = np.abs(G - (R - 20 * alpha))[oy:oy + H, ox:ox + W]
    black = (img[oy:oy + H, ox:ox + W].sum(-1) == 0)       # the reference draws its own invalid pixels black
    a = np.where(vr & (a >= 0) & ~black, a, np.nan)
    return np.rot90(a, 2).astype(np.float32), {"offset_rows_cols_in_png": [oy, ox], "ct_layer34_match_r": round(r, 4),
                                              "green_channel_consistent_fraction": round(float((gres[a > 0.01] <= 1.01).mean()), 4),
                                              "masked_black_or_line_fraction_of_valid": round(float((~np.isfinite(a))[vr].mean()), 4)}


def overlay_alpha(z):
    u = np.clip((z - PUB_LO) / (PUB_HI - PUB_LO), 0, 1)
    return 0.9 * np.clip((u - 0.35) / 0.40, 0, 1)


def stage_score():
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    valid = np.load(V2DIR / "valid.npy")
    layers = np.load(V2DIR / "layers.npy", mmap_mode="r")
    probs = np.load(V2DIR / "probs.npy", mmap_mode="r")
    dn = np.load(HERE / "probs_dn.npy", mmap_mode="r")
    hec_raw = np.load(HERE / f"hecate_d{HECATE_D:+d}.npy")
    done = json.load(open(HERE / "infer_done.json"))
    assert len(done) == len(DEPTHS) + 1, f"infer incomplete: {len(done)}"
    synth = json.load(open(HERE / "synth.json"))
    rev = v2.DIRS.index("reverse")
    H, W = valid.shape

    # per-model z-score statistics (reverse, valid pixels, all depths; hecate: d = +1 only)
    stats = {f"ink9-{s}": hist_stats([probs[si, rev, k] for k in range(len(DEPTHS))], valid) for si, s in enumerate(v2.SEEDS)}
    stats["dnative"] = hist_stats([dn[k] for k in range(len(DEPTHS))], valid)
    hec = cv2.resize(hec_raw.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_LINEAR)
    hv = hec[valid]
    stats["hecate"] = (float(hv.mean()), float(hv.std()) + 1e-6)
    print("stats", stats, flush=True)

    # pooled maps per configuration
    t0 = time.time()
    maps, configs = [], []
    pv = None
    for k, d in enumerate(DEPTHS):
        raw, zs = {}, {}
        for si, s in enumerate(v2.SEEDS):
            m = f"ink9-{s}"
            raw[m], pv = v2.pool(probs[si, rev, k] / 255.0, valid)
            tz, _ = v2.pool(villa_t(probs[si, rev, k]), valid)
            zs[m] = (tz - stats[m][0]) / stats[m][1]
        raw["ink9-mean"] = (raw["ink9-42"] + raw["ink9-43"]) / 2
        raw["dnative"], _ = v2.pool(dn[k] / 255.0, valid)
        tz, _ = v2.pool(villa_t(dn[k]), valid)
        zs["dnative"] = (tz - stats["dnative"][0]) / stats["dnative"][1]
        raw["ens3"] = (zs["dnative"] + zs["ink9-42"] + zs["ink9-43"]) / 3
        for m in MODELS:
            maps.append(raw[m].astype(np.float32)); configs.append({"model": m, "dir": "reverse", "d": d})
        if d == HECATE_D:
            hz, _ = v2.pool((hec - stats["hecate"][0]) / stats["hecate"][1], valid)
            ens4_pooled = ((zs["dnative"] + hz + zs["ink9-42"] + zs["ink9-43"]) / 4).astype(np.float32)
            hec_pooled, _ = v2.pool(hec, valid)
    print(f"pooled {len(maps)} maps {time.time() - t0:.0f}s", flush=True)

    folds, boxes = make_folds(pv)
    sweep = []
    for cfg, pm in zip(configs, maps):
        row = dict(cfg)
        for f, fo in folds.items():
            sc = fo.score(pm)
            row[f"{f}_box_auc"], row[f"{f}_pixel_auc"] = round(sc["box_auc"], 4), round(sc["pixel_auc"], 4)
        sweep.append(row)

    def best_on(f, rows):
        return max(rows, key=lambda r: (r[f"{f}_box_auc"], r[f"{f}_pixel_auc"], -abs(r["d"])))

    rng = np.random.default_rng(0)
    heldout = {}
    for sel, ev in (("A", "B"), ("B", "A")):
        b = best_on(sel, sweep)
        i = sweep.index(b)
        res = evaluate(folds[ev], maps[i], rng)
        heldout[f"select_{sel}_eval_{ev}"] = {"config": {k: b[k] for k in ("model", "dir", "d")},
                                              "selection_box_auc": b[f"{sel}_box_auc"], **res}
        print(sel, "->", ev, json.dumps(heldout[f"select_{sel}_eval_{ev}"]), flush=True)

    fixed = {}
    for label, pm in (("ens4_reverse_d+1", ens4_pooled), ("hecate_reverse_d+1", hec_pooled.astype(np.float32))):
        fixed[label] = {f: evaluate(fo, pm, rng, extra=(label.startswith("ens4"))) for f, fo in folds.items()}
        print(label, json.dumps(fixed[label]), flush=True)

    # published map: parity check, and the same held-out test applied to it
    pub, pub_info = published_alpha(np.asarray(layers[34]), valid)
    zfull = (villa_t(dn[DEPTHS.index(HECATE_D)]) - stats["dnative"][0]) / stats["dnative"][1]
    zfull += (hec - stats["hecate"][0]) / stats["hecate"][1]
    for si, s in enumerate(v2.SEEDS):
        zfull += (villa_t(probs[si, rev, DEPTHS.index(HECATE_D)]) - stats[f"ink9-{s}"][0]) / stats[f"ink9-{s}"][1]
    ours = overlay_alpha(zfull / 4)
    m = valid & np.isfinite(pub)
    r_pix = float(np.corrcoef(ours[m], pub[m])[0, 1])
    sub = np.flatnonzero(m.ravel())[::50]
    rho = float(spearmanr(ours.ravel()[sub], pub.ravel()[sub]).statistic)
    pub_pooled, _ = v2.pool(np.nan_to_num(pub), m)
    ours_pooled, _ = v2.pool(ours, valid)
    mp = np.isfinite(pub_pooled) & np.isfinite(ours_pooled)
    r_pool = float(np.corrcoef(ours_pooled[mp], pub_pooled[mp])[0, 1])
    parity = {"pearson_r_pixels": round(r_pix, 4), "spearman_rho_pixels_every_50th": round(rho, 4),
              "pearson_r_pooled_4x4": round(r_pool, 4), "n_pixels": int(m.sum()),
              "published_nonzero_fraction": round(float((pub[m] > 0).mean()), 4),
              "ours_nonzero_fraction": round(float((ours[m] > 0).mean()), 4), **pub_info}
    pub_pooled = np.where(np.isfinite(pub_pooled), pub_pooled, np.nan).astype(np.float32)
    pub_eval = {f: evaluate(fo, pub_pooled, rng) for f, fo in folds.items()}
    print("parity", parity, "published map", json.dumps(pub_eval), flush=True)

    # ---- checks
    meets = all(h["meets_bar"] for h in heldout.values())
    checks = {
        "letters_visible": "needs a human look at fig_reading.png (see README)",
        "heldout_auc": {"pass": bool(meets), "per_direction_of_split": {k: h["meets_bar"] for k, h in heldout.items()},
                           "ens4_d+1_meets_bar": {f: v["meets_bar"] for f, v in fixed["ens4_reverse_d+1"].items()}},
        "normal_direction": "settled in v2 (pass)",
        "baseline_reported": {"pass": True},
        "depth_sweep": {"villa_models_d": [DEPTHS[0], DEPTHS[-1]], "hecate_d": [HECATE_D],
                           "note": "full sweep for the villa models, one depth for hecate (decided before the run)"},
        "reproducible": "checked separately (see README)",
        "parity": {"pass": bool(r_pix >= PARITY_R), "pearson_r": round(r_pix, 4), "bar": PARITY_R},
        "published_map_meets_bar": {f: v["meets_bar"] for f, v in pub_eval.items()},
        "synthetic_check": {"ring_plant_ok": synth["ok"], "whole_box_plant_ok": synth["ok_fill"]},
    }
    timings = {k: v for k, v in done.items()}
    results = {
        "scan": {"scroll": "PHerc0343", "volume": "20250521140437-8.640um-1.2m-116keV-masked", "voxel_um": 8.64,
                 "energy_keV": 116, "level": 0},
        "surface": {"mesh": NAME, "normals": "outward (away from Nieuwlaar umbilicus)", "direction": "reverse"},
        "region_raw_px": {"rows": [v2.R0, v2.R1], "cols": [v2.C0, v2.C1]},
        "layers": {"n": v2.N_LAYERS, "offset_of_layer_k": "k - 32.5", "d": "k - 33", "depth_range": [DEPTHS[0], DEPTHS[-1]],
                   "hecate_d": HECATE_D},
        "split": {"folds_raw_cols": v2.FOLDS, "guard_band_raw_cols": [v2.FOLDS["A"][1], v2.FOLDS["B"][0]],
                  "fold_letter_ids": {f: fo.ids for f, fo in folds.items()},
                  "n_bands": {f: len(fo.band_cells) for f, fo in folds.items()},
                  "band_clearance_px": CLEAR, "band_depth_px": BAND_H, "pool": P},
        "zscore_stats": {k: {"mean": round(v[0], 5), "sd": round(v[1], 5)} for k, v in stats.items()},
        "heldout": heldout,
        "fixed_reference_config": fixed,
        "parity_with_published_map": parity,
        "published_map_through_the_test": pub_eval,
        "synthetic": {k: synth[k] for k in ("ok", "ok_fill")},
        "cpu": {"threads": threads(), "map_seconds": timings},
        "checks": checks,
        "sweep": sweep,
    }
    json.dump(results, open(HERE / "results.json", "w"), indent=1)

    # ---- figures
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey="row")
    for j, f in enumerate(v2.FOLDS):
        for metric, i in (("box_auc", 0), ("pixel_auc", 1)):
            ax = axs[i, j]
            for mname in MODELS:
                rows = [r for r in sweep if r["model"] == mname]
                ax.plot([r["d"] for r in rows], [r[f"{f}_{metric}"] for r in rows], "-", color=COLORS[mname], lw=2, label=mname)
            ax.axhline(0.5, color="#888", lw=1)
            ax.axhline(0.80 if metric == "box_auc" else 0.60, color="#888", lw=1, ls=":")
            ax.set_title(f"fold {f}: {metric.replace('_', ' ')} (letters vs between-line bands)", fontsize=10)
            ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("d (voxels, + = away from scroll axis), reverse")
    axs[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("PHerc0343 8.64 µm: depth sweep per fold (selection data; dotted = pass line; held-out numbers in results.json)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "fig_depth_sweep.png", dpi=110); plt.close(fig)

    panels = []
    for key, h in heldout.items():
        i = next(j for j, c in enumerate(configs) if c == h["config"])
        c = h["config"]
        ev = key.split("_")[-1]
        panels.append((maps[i], f"{c['model']} reverse d={c['d']:+d}, chosen on fold {key.split('_')[1]}; held-out fold {ev}: "
                                f"box AUC {h['box_auc']['observed']:.2f} (null p99 {h['box_auc']['null_p99']:.2f}), "
                                f"pixel AUC {h['pixel_auc']['observed']:.2f} (null p99 {h['pixel_auc']['null_p99']:.2f})"))
    panels.append((ens4_pooled, "4-model ensemble (dnative, hecate, ink9-42, ink9-43), reverse d=+1, reference setting (not held out)"))
    panels.append((pub_pooled, f"published map (overlay strength from the reference PNG); parity with the ensemble r = {r_pix:.2f}"))
    fig, axs = plt.subplots(len(panels), 1, figsize=(14, 5.6 * len(panels)))
    for ax, (pm, title) in zip(axs, panels):
        im = np.rot90(pm, 2)                      # reading orientation = raw rotated 180 deg
        Hp, Wp = im.shape
        ax.imshow(im, cmap="gray", vmin=np.nanpercentile(pm, 1), vmax=np.nanpercentile(pm, 99.5))
        for b in boxes:
            r0, r1, c0, c1 = v2.to_pool(b["box"])
            ax.add_patch(plt.Rectangle((Wp - c1, Hp - r1), c1 - c0, r1 - r0, fill=False, lw=1,
                                       ec="#eda100" if v2.fold_of(b["box"]) else "#888"))
            ax.text(Wp - c1 + 2, Hp - r1 - 3, str(b["id"]), color="#eda100", fontsize=7)
        for fo in folds.values():
            for cells in fo.band_cells:
                rr, cc = np.nonzero(cells)
                ax.add_patch(plt.Rectangle((Wp - cc.max() - 1, Hp - rr.max() - 1), cc.max() - cc.min() + 1,
                                           rr.max() - rr.min() + 1, fill=False, lw=0.6, ec="#3987e5"))
        for x in v2.FOLDS["A"][1], v2.FOLDS["B"][0]:
            ax.axvline(Wp - (x - v2.C0) / P, color="#e87ba4", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(HERE / "fig_reading.png", dpi=100); plt.close(fig)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["infer", "synth", "score", "all"])
    a = ap.parse_args()
    os.chdir(HERE)
    if a.stage in ("infer", "all"):
        stage_infer()
    if a.stage in ("synth", "all"):
        stage_synth()
    if a.stage in ("score", "all"):
        stage_score()
