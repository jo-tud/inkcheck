"""PHerc. 343 positive control, v2: held-out scoring of the PHerc. 343 letters (pass/fail rules in README.md).

Stages, each cached so a crashed run resumes:
  1. render  66 layers along the outward normal over Nieuwlaar's submission area -> layers.npy
  2. infer   ink_9um seed42/43 x forward/reverse x d=-15..+15 (CPU)            -> probs.npy
  3. score   two-fold held-out selection, shifted-label null, sign test, figures -> results.json

    python run.py [--stage render|infer|score|all]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REF = ROOT / "external/pherc343-first-letters"
MODELS = ROOT / "data/models/ink_9um"
NAME = "concat_w047-w048_R5B2_z9500-11000"

R0, R1, C0, C1 = 105, 2560, 380, 6021          # raw render px, Nieuwlaar's submission area
N_LAYERS = 66
OFFSETS = np.arange(N_LAYERS) - 32.5            # layer k at (k - 32.5) vox along the outward normal
DEPTHS = list(range(-15, 16))                   # d = k - 33, window centred on layer 33 + d
SEEDS = (42, 43)
DIRS = ("forward", "reverse")
POOL = 4                                        # scores on 4x4 mean-pooled maps
FOLDS = {"A": (380, 2486), "B": (2610, 6021)}   # raw columns; 2486-2610 is the guard band
MARGIN = 100                                    # blank pixels/boxes keep this far from every letter box
BOX_W, BOX_H = 292, 334                         # median letter box size (raw px)
N_NULL = 1000
V1_REGION = (550, 1650, 600, 5600)              # v1 region (rows, cols), for the continuity metric


def depth_order():
    out = [0]
    for a in range(1, 16):
        out += [-a, a]
    return out


# ---------------------------------------------------------------- inputs

def load_surface():
    from inkcheck.surface import load_tifxyz, normals, orient_outward, axis_from_umbilicus
    surf = load_tifxyz(REF / f"outputs/{NAME}.tifxyz")
    umb = json.load(open(REF / "inputs/umbilicus_PHerc0343.json"))["control_points"]
    axis = axis_from_umbilicus(np.array([(p["z"], p["y"], p["x"]) for p in umb]))
    n, agree = orient_outward(surf, normals(surf), axis)
    return surf, n, agree, axis


def load_boxes():
    letters = json.load(open(REF / f"inputs/{NAME}_letter_boxes.json"))["letters"]
    return [dict(id=l["id"], row=l["row"], box=[float(v) for v in l["box"]]) for l in letters]


def fold_of(box):
    x0, _, x1, _ = box
    for f, (a, b) in FOLDS.items():
        if x0 >= a and x1 <= b:
            return f
    return None  # crosses the guard band -> dropped


# ---------------------------------------------------------------- stage 1: render

def stage_render():
    from inkcheck.catalog import reference_volume
    from inkcheck.volume import Volume
    from inkcheck.render import render
    if (HERE / "layers.npy").exists():
        print("render: cached layers.npy")
        return
    surf, n, agree, _ = load_surface()
    vi = reference_volume()
    print("volume", vi.url, "outward agreement", round(agree, 4), flush=True)
    t = time.time()
    layers, valid = render(surf, n, Volume(vi.url, 0), OFFSETS, region=(R0, R1, C0, C1), workers=6,
                           progress=lambda i, k: print(f"render {i}/{k} {time.time()-t:.0f}s", flush=True) if i % 10 == 0 else None)
    dt = time.time() - t
    np.save(HERE / "layers.npy", np.clip(np.rint(layers), 0, 255).astype(np.uint8))
    np.save(HERE / "valid.npy", valid)
    json.dump({"render_s": dt, "volume_url": vi.url, "outward_agreement": agree, "shape": list(layers.shape),
               "valid_fraction": float(valid.mean())}, open(HERE / "render.json", "w"), indent=1)
    print(f"render {layers.shape} {dt:.0f}s valid {valid.mean():.3f}", flush=True)


# ---------------------------------------------------------------- stage 2: infer

def stage_infer():
    from inkcheck.ink import load_model, predict
    layers = np.load(HERE / "layers.npy", mmap_mode="r")
    valid = np.load(HERE / "valid.npy")
    shape = (len(SEEDS), len(DIRS), len(DEPTHS)) + valid.shape
    pf, donef = HERE / "probs.npy", HERE / "probs_done.json"
    probs = (np.lib.format.open_memmap(pf, mode="r+") if pf.exists()
             else np.lib.format.open_memmap(pf, mode="w+", dtype=np.uint8, shape=shape))
    done = json.load(open(donef)) if donef.exists() else {}
    models = {s: load_model(str(MODELS / f"hybrid_3d2d-seed{s}/step-075000.pth"), f"ink9-{s}") for s in SEEDS}
    for d in depth_order():
        c = 33 + d
        lo = c - 8
        win = np.asarray(layers[lo:lo + 17]).astype(np.float32)  # layers 25+d .. 41+d, centre 33+d
        for si, s in enumerate(SEEDS):
            for di, direction in enumerate(DIRS):
                key = f"{s}/{direction}/{d}"
                if key in done:
                    continue
                t = time.time()
                pr = predict(models[s], win, valid, center=8, direction=direction)
                probs[si, di, DEPTHS.index(d)] = np.clip(np.rint(np.nan_to_num(pr) * 255), 0, 255).astype(np.uint8)
                probs.flush()
                done[key] = round(time.time() - t, 1)
                json.dump(done, open(donef, "w"), indent=0)
                print(f"infer seed{s} {direction} d={d:+d} {done[key]:.0f}s ({len(done)}/{np.prod(shape[:3])})", flush=True)


# ---------------------------------------------------------------- stage 3: score

def pool(a, valid):
    """4x4 mean over valid pixels; returns (pooled, pooled_valid)."""
    H, W = (s // POOL * POOL for s in valid.shape)
    v = valid[:H, :W].reshape(H // POOL, POOL, W // POOL, POOL)
    vs = v.sum((1, 3))
    a = a[..., :H, :W].astype(np.float32)
    s = (a * valid[:H, :W]).reshape(a.shape[:-2] + (H // POOL, POOL, W // POOL, POOL)).sum((-3, -1))
    pv = vs >= 0.75 * POOL * POOL
    return np.where(pv, s / np.maximum(vs, 1), np.nan), pv


def to_pool(box):
    """raw-frame box [x0, y0, x1, y1] -> pooled (r0, r1, c0, c1), clipped to the region."""
    x0, y0, x1, y1 = box
    r0 = int(np.floor((max(y0, R0) - R0) / POOL)); r1 = int(np.ceil((min(y1, R1) - R0) / POOL))
    c0 = int(np.floor((max(x0, C0) - C0) / POOL)); c1 = int(np.ceil((min(x1, C1) - C0) / POOL))
    return r0, r1, c0, c1


def auc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Fold:
    """Pooled-grid geometry of one fold: letter boxes, blank pixels and blank boxes."""

    def __init__(self, name, pvalid, letter_boxes, all_boxes):
        self.name = name
        Hp, Wp = pvalid.shape
        a, b = FOLDS[name]
        self.c0, self.c1 = int(np.ceil((a - C0) / POOL)), int(np.floor((b - C0) / POOL))
        self.pvalid = pvalid
        self.infold = np.zeros_like(pvalid)
        self.infold[:, self.c0:self.c1] = True
        self.set_letters(letter_boxes, all_boxes)

    def set_letters(self, letter_boxes, exclusion_boxes):
        """letter_boxes, exclusion_boxes: pooled (r0, r1, c0, c1)."""
        Hp, Wp = self.pvalid.shape
        m = MARGIN // POOL
        self.letters = [b for b in letter_boxes if self.pvalid[b[0]:b[1], b[2]:b[3]].mean() >= 0.5]
        inbox = np.zeros_like(self.pvalid)
        for r0, r1, c0, c1 in self.letters:
            inbox[r0:r1, c0:c1] = True
        near = np.zeros_like(self.pvalid)
        for r0, r1, c0, c1 in exclusion_boxes:
            near[max(r0 - m, 0):r1 + m, max(c0 - m, 0):c1 + m] = True
        self.pos_px = inbox & self.pvalid & self.infold
        self.blank_px = ~near & self.pvalid & self.infold
        bh, bw = int(round(BOX_H / POOL)), int(round(BOX_W / POOL))
        self.blanks = []
        for r in range(0, Hp - bh + 1, bh):
            for c in range(self.c0, self.c1 - bw + 1, bw):
                if near[r:r + bh, c:c + bw].any():
                    continue
                if self.pvalid[r:r + bh, c:c + bw].mean() >= 0.8:
                    self.blanks.append((r, r + bh, c, c + bw))

    def box_scores(self, pmap, boxes):
        out = []
        for r0, r1, c0, c1 in boxes:
            v = pmap[r0:r1, c0:c1]
            v = v[np.isfinite(v)]
            out.append(float(np.percentile(v, 95)) if v.size else np.nan)
        return np.array(out)

    def score(self, pmap):
        pos, neg = self.box_scores(pmap, self.letters), self.box_scores(pmap, self.blanks)
        return {"box_auc": auc(pos, neg), "pixel_auc": auc(pmap[self.pos_px], pmap[self.blank_px]),
                "n_letters": len(pos), "n_blank_boxes": len(neg)}


def random_box_null(fold: Fold, pmap, letters, true_boxes, rng, n=N_NULL):
    """Move each of the fold's letter boxes (sizes kept) to a random spot in the fold that is
    >= 50 % valid and does not overlap any real letter box; rescore exactly as for the real labels
    (blank set excludes the margin around both the pseudo boxes and the real letter boxes).

    Changed 2026-10-12, before any real map was scored: the planned rigid shift landed on the
    real letters so often that its 99th percentile reached box AUC 1.0 on a synthetic map with
    perfect planted ink, so the AUC bar could never be met (README, "Change to the null").
    """
    Hp = fold.pvalid.shape[0]
    occupied = np.zeros_like(fold.pvalid)
    for r0, r1, c0, c1 in true_boxes:
        occupied[r0:r1, c0:c1] = True
    box_aucs, px_aucs = [], []
    for _ in range(n):
        pseudo = []
        for r0, r1, c0, c1 in letters:
            h, w = r1 - r0, c1 - c0
            for _try in range(500):
                nr, nc = rng.integers(0, Hp - h), rng.integers(fold.c0, fold.c1 - w)
                if occupied[nr:nr + h, nc:nc + w].any() or fold.pvalid[nr:nr + h, nc:nc + w].mean() < 0.5:
                    continue
                pseudo.append((nr, nr + h, nc, nc + w))
                break
        g = Fold.__new__(Fold)
        g.__dict__.update(fold.__dict__)
        g.set_letters(pseudo, pseudo + list(true_boxes))
        s = g.score(pmap)
        box_aucs.append(s["box_auc"]); px_aucs.append(s["pixel_auc"])
    return np.array(box_aucs), np.array(px_aucs)


def null_summary(obs, null):
    null = null[np.isfinite(null)]
    return {"observed": round(obs, 4), "null_p50": round(float(np.percentile(null, 50)), 4),
            "null_p99": round(float(np.percentile(null, 99)), 4),
            "p_value": round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4), "n_null": int(len(null))}


def geometry_sign_test(surf, axis):
    """Which side of the sheet does the unmirrored raw render face? (README, "Normal direction")."""
    p = surf.zyx[..., ::-1]                              # x, y, z
    dr = np.full_like(p, np.nan); dc = np.full_like(p, np.nan)
    dr[1:-1] = (p[2:] - p[:-2]) / 2
    dc[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / 2
    viewer = np.cross(dr, dc)                            # e_r x e_c: side a viewer of the raw image stands on
    yx = axis(surf.zyx[..., 0])                          # (y, x) of the axis at each node's z
    radial = np.stack([p[..., 0] - yx[..., 1], p[..., 1] - yx[..., 0], np.zeros(p.shape[:2])], -1)
    dot = np.einsum("hwc,hwc->hw", viewer, radial)
    ok = np.isfinite(dot)
    side = json.load(open(REF / f"outputs/{NAME}.json"))["orientation"]
    vc3d_normal_outward = float((np.einsum("hwc,hwc->hw", np.cross(dc, dr), radial)[ok] > 0).mean())
    frac_axis = float((dot[ok] < 0).mean())
    return {"nodes": int(ok.sum()),
            "fraction_viewer_side_toward_axis": round(frac_axis, 4),
            "vc3d_normal_dPdcol_x_dPdrow_outward_fraction": round(vc3d_normal_outward, 4),
            "reference_reading_transform": {"flip_lr": side["flip_lr"], "flip_ud": side["flip_ud"],
                                            "is_rotation_not_mirror": bool(side["flip_lr"] == side["flip_ud"])},
            "written_face": "toward the scroll axis" if frac_axis > 0.5 else "away from the scroll axis",
            "predicted_direction_with_outward_normals": "reverse" if frac_axis > 0.5 else "forward",
            "assumption": "volume x, y, z axes form a right-handed frame (reconstruction not mirrored)"}


def stage_score():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf, n, agree, axis = load_surface()
    layers = np.load(HERE / "layers.npy", mmap_mode="r")
    valid = np.load(HERE / "valid.npy")
    probs = np.load(HERE / "probs.npy", mmap_mode="r")
    done = json.load(open(HERE / "probs_done.json"))
    assert len(done) == len(SEEDS) * len(DIRS) * len(DEPTHS), f"infer incomplete: {len(done)} maps"
    boxes = load_boxes()
    pooled_boxes = {b["id"]: to_pool(b["box"]) for b in boxes}
    all_pb = list(pooled_boxes.values())

    # pooled maps for every configuration: model in {s42, s43, mean}
    t = time.time()
    maps, configs = [], []
    for di, direction in enumerate(DIRS):
        for k, d in enumerate(DEPTHS):
            per_seed = []
            for si, s in enumerate(SEEDS):
                pm, pv = pool(probs[si, di, k] / 255.0, valid)
                per_seed.append(pm)
                maps.append(pm); configs.append({"model": f"ink9-{s}", "dir": direction, "d": d})
            maps.append((per_seed[0] + per_seed[1]) / 2); configs.append({"model": "ink9-mean", "dir": direction, "d": d})
    print(f"pooled {len(maps)} maps {time.time()-t:.0f}s", flush=True)

    folds = {}
    for f in FOLDS:
        ids = [b["id"] for b in boxes if fold_of(b["box"]) == f]
        folds[f] = Fold(f, pv, [pooled_boxes[i] for i in ids], all_pb)
        folds[f].ids = ids
    dropped = [b["id"] for b in boxes if fold_of(b["box"]) is None]

    sweep = []
    for cfg, pm in zip(configs, maps):
        row = dict(cfg)
        for f, fo in folds.items():
            s = fo.score(pm)
            row[f"{f}_box_auc"] = round(s["box_auc"], 4)
            row[f"{f}_pixel_auc"] = round(s["pixel_auc"], 4)
        sweep.append(row)

    def best_on(f, rows):
        return max(rows, key=lambda r: (r[f"{f}_box_auc"], r[f"{f}_pixel_auc"], -abs(r["d"])))

    rng = np.random.default_rng(0)
    heldout = {}
    for sel, ev in (("A", "B"), ("B", "A")):
        b = best_on(sel, sweep)
        i = sweep.index(b)
        fo = folds[ev]
        s = fo.score(maps[i])
        nb, npx = random_box_null(fo, maps[i], fo.letters, all_pb, rng)
        pos, neg = fo.box_scores(maps[i], fo.letters), fo.box_scores(maps[i], fo.blanks)
        allsc = np.concatenate([pos, neg])
        perm = np.array([auc(*np.split(rng.permutation(allsc), [len(pos)])) for _ in range(N_NULL)])
        # best of each direction chosen on the selection fold, scored on the held-out fold (direction check)
        per_dir = {}
        for direction in DIRS:
            bd = best_on(sel, [r for r in sweep if r["dir"] == direction])
            per_dir[direction] = {"config": {k: bd[k] for k in ("model", "dir", "d")},
                                  "heldout_box_auc": bd[f"{ev}_box_auc"], "heldout_pixel_auc": bd[f"{ev}_pixel_auc"]}
        heldout[f"select_{sel}_eval_{ev}"] = {
            "config": {k: b[k] for k in ("model", "dir", "d")},
            "selection_box_auc": b[f"{sel}_box_auc"],
            "n_letters": s["n_letters"], "n_blank_boxes": s["n_blank_boxes"],
            "box_auc": null_summary(s["box_auc"], nb),
            "pixel_auc": null_summary(s["pixel_auc"], npx),
            "box_auc_label_permutation": null_summary(s["box_auc"], perm),
            "best_per_direction": per_dir,
        }
        print(sel, "->", ev, heldout[f"select_{sel}_eval_{ev}"], flush=True)

    # v1 continuity metric: row-1 boxes vs all other valid pixels, v1 region, held-out configs
    vr0, vr1, vc0, vc1 = V1_REGION
    sl = (slice((vr0 - R0) // POOL, (vr1 - R0) // POOL), slice((vc0 - C0) // POOL, (vc1 - C0) // POOL))
    inb = np.zeros_like(pv)
    for b in boxes:
        if b["row"] == "1":
            r0, r1, c0, c1 = pooled_boxes[b["id"]]
            inb[r0:r1, c0:c1] = True
    v1 = {}
    for key, h in heldout.items():
        i = next(j for j, c in enumerate(configs) if c == h["config"])
        pm = maps[i][sl]; m = pv[sl]; ib = inb[sl]
        v1[key] = round(auc(pm[ib & m], pm[~ib & m]), 4)

    # model-free profiles along the normal
    lay = np.asarray(layers)
    vm = valid
    prof = [float(lay[k][vm].mean()) for k in range(N_LAYERS)]
    inbox_full = np.zeros_like(valid); near_full = np.zeros_like(valid)
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        inbox_full[max(int(y0) - R0, 0):int(y1) - R0, max(int(x0) - C0, 0):int(x1) - C0] = True
        near_full[max(int(y0) - R0 - MARGIN, 0):int(y1) - R0 + MARGIN, max(int(x0) - C0 - MARGIN, 0):int(x1) - C0 + MARGIN] = True
    pin, pbl = inbox_full & vm, ~near_full & vm
    contrast = [float((lay[k][pin].mean() - lay[k][pbl].mean()) / lay[k][pbl].std()) for k in range(N_LAYERS)]
    del lay

    geo = geometry_sign_test(surf, axis)
    pred = geo["predicted_direction_with_outward_normals"]

    # ---- checks
    bar_detail = {}
    meets = True
    for key, h in heldout.items():
        ok = (h["box_auc"]["observed"] >= 0.80 and h["pixel_auc"]["observed"] >= 0.60
              and h["box_auc"]["observed"] > h["box_auc"]["null_p99"] and h["pixel_auc"]["observed"] > h["pixel_auc"]["null_p99"])
        bar_detail[key] = ok
        meets &= ok
    direction_ok = geo["fraction_viewer_side_toward_axis"] > 0.9 or geo["fraction_viewer_side_toward_axis"] < 0.1
    direction_ok &= all(h["config"]["dir"] == pred for h in heldout.values())
    checks = {
        "letters_visible": "needs a human look at fig_heldout_reading.png (see README)",
        "heldout_auc": {"pass": bool(meets), "per_direction_of_split": bar_detail},
        "normal_direction": {"pass": bool(direction_ok), "predicted": pred,
                                 "selected": {k: h["config"]["dir"] for k, h in heldout.items()}},
        "baseline_reported": {"pass": True},
        "depth_sweep": {"pass": len(done) == 124, "depths": [DEPTHS[0], DEPTHS[-1]], "maps": len(done)},
        "reproducible": "checked separately from a clean checkout (see README)",
    }
    timings = np.array(list(done.values()))
    render_info = json.load(open(HERE / "render.json")) if (HERE / "render.json").exists() else {}
    results = {
        "scan": {"scroll": "PHerc0343", "volume": "20250521140437-8.640um-1.2m-116keV-masked", "voxel_um": 8.64,
                 "energy_keV": 116, "level": 0},
        "surface": {"mesh": NAME, "normals": "outward (away from Nieuwlaar umbilicus)", "outward_agreement": round(agree, 4)},
        "region_raw_px": {"rows": [R0, R1], "cols": [C0, C1]},
        "layers": {"n": N_LAYERS, "offset_of_layer_k": "k - 32.5", "d": "k - 33", "depth_range": [DEPTHS[0], DEPTHS[-1]]},
        "split": {"folds_raw_cols": FOLDS, "guard_band_raw_cols": [FOLDS["A"][1], FOLDS["B"][0]],
                  "fold_letter_ids": {f: fo.ids for f, fo in folds.items()}, "dropped_letter_ids": dropped,
                  "blank_margin_px": MARGIN, "pool": POOL, "blank_box_px": [BOX_H, BOX_W]},
        "heldout": heldout,
        "v1_metric_on_v1_region": v1,
        "sign_test": geo,
        "profile": {"d": [k - 33 for k in range(N_LAYERS)], "mean_ct": [round(v, 3) for v in prof],
                    "letter_vs_blank_contrast_sd": [round(v, 4) for v in contrast]},
        "cpu": {"threads": int(os.environ.get("INKCHECK_CPU_THREADS", 0)) or None,
                "render_s": render_info.get("render_s"),
                "map_s_median": float(np.median(timings)), "maps_total_s": float(timings.sum())},
        "checks": checks,
        "sweep": sweep,
    }
    json.dump(results, open(HERE / "results.json", "w"), indent=1)

    # ---- figures
    colors = {"ink9-42": "#2a78d6", "ink9-43": "#eb6834", "ink9-mean": "#1baf7a"}
    styles = {"reverse": "-", "forward": "--"}
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey="row")
    for j, f in enumerate(FOLDS):
        for metric, i in (("box_auc", 0), ("pixel_auc", 1)):
            ax = axs[i, j]
            for m in colors:
                for dr in DIRS:
                    rows = [r for r in sweep if r["model"] == m and r["dir"] == dr]
                    ax.plot([r["d"] for r in rows], [r[f"{f}_{metric}"] for r in rows], styles[dr], color=colors[m],
                            lw=2, label=f"{m} {dr}")
            ax.axhline(0.5, color="#888", lw=1)
            ax.set_title(f"fold {f}: {metric.replace('_', ' ')}", fontsize=11)
            ax.grid(alpha=0.25)
            if i == 1:
                ax.set_xlabel("d (voxels, + = away from scroll axis)")
    axs[0, 0].legend(fontsize=8, ncol=2, frameon=False)
    fig.suptitle("PHerc0343 8.64 µm, ink_9um: depth sweep per fold (selection data; held-out numbers in results.json)", fontsize=11)
    fig.tight_layout(); fig.savefig(HERE / "fig_depth_sweep.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(len(heldout), 1, figsize=(14, 6.5 * len(heldout)))
    for ax, (key, h) in zip(np.atleast_1d(axs), heldout.items()):
        i = next(j for j, c in enumerate(configs) if c == h["config"])
        im = np.rot90(maps[i], 2)                      # reading orientation = raw rotated 180 deg (reference sidecar)
        Hp, Wp = im.shape
        ax.imshow(im, cmap="gray", vmin=np.nanpercentile(maps[i], 1), vmax=np.nanpercentile(maps[i], 99.5))
        for b in boxes:
            r0, r1, c0, c1 = pooled_boxes[b["id"]]
            f = fold_of(b["box"])
            ax.add_patch(plt.Rectangle((Wp - c1, Hp - r1), c1 - c0, r1 - r0, fill=False, lw=1,
                                       ec="#eda100" if f else "#888"))
            ax.text(Wp - c1 + 2, Hp - r1 - 3, str(b["id"]), color="#eda100", fontsize=7)
        for x in FOLDS["A"][1], FOLDS["B"][0]:
            ax.axvline(Wp - (x - C0) / POOL, color="#e87ba4", lw=1)
        ev = key.split("_")[-1]
        c = h["config"]
        ax.set_title(f"{c['model']} {c['dir']} d={c['d']:+d}, chosen on fold {key.split('_')[1]}; held-out fold {ev}: "
                     f"box AUC {h['box_auc']['observed']:.2f} (null p99 {h['box_auc']['null_p99']:.2f}), "
                     f"pixel AUC {h['pixel_auc']['observed']:.2f} (null p99 {h['pixel_auc']['null_p99']:.2f})", fontsize=10)
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(HERE / "fig_heldout_reading.png", dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    dd = [k - 33 for k in range(N_LAYERS)]
    axs[0].plot(dd, prof, "-", color="#2a78d6", lw=2); axs[0].set_title("mean CT intensity (valid pixels)", fontsize=11)
    axs[1].plot(dd, contrast, "-", color="#2a78d6", lw=2); axs[1].axhline(0, color="#888", lw=1)
    axs[1].set_title("raw CT: letter boxes − blank, in blank SD", fontsize=11)
    for ax in axs:
        ax.set_xlabel("d (voxels, + = away from scroll axis)"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(HERE / "fig_profile.png", dpi=110); plt.close(fig)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["render", "infer", "score", "all"])
    a = ap.parse_args()
    os.chdir(HERE)
    if a.stage in ("render", "all"):
        stage_render()
    if a.stage in ("infer", "all"):
        stage_infer()
    if a.stage in ("score", "all"):
        stage_score()
