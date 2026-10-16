"""Diagnostic, written after the v3 scores: why the published map meets the bar on fold B while the ens4 map of
run.py at the same setting does not.

The published map is the same four-model ensemble after the published overlay transform (clip at u = 0.35, 85 % of
pixels zero). Here the ens4 map of run.py goes through that same transform (`overlay_alpha`, as in the parity check)
and is scored with the unchanged v3 test. If it passes like the published map, the gap comes from the display
transform (a thresholded map has a narrower surrogate null), not from signal lost in this pipeline.

    python diag_overlay.py      -> diag_overlay.json
"""
from __future__ import annotations

import json

import cv2
import numpy as np

import run as v3

v2 = v3.v2


def main():
    valid = np.load(v3.V2DIR / "valid.npy")
    probs = np.load(v3.V2DIR / "probs.npy", mmap_mode="r")
    dn = np.load(v3.HERE / "probs_dn.npy", mmap_mode="r")
    hec_raw = np.load(v3.HERE / f"hecate_d{v3.HECATE_D:+d}.npy")
    rev = v2.DIRS.index("reverse")
    k = v3.DEPTHS.index(v3.HECATE_D)
    H, W = valid.shape

    stats = {f"ink9-{s}": v3.hist_stats([probs[si, rev, j] for j in range(len(v3.DEPTHS))], valid)
             for si, s in enumerate(v2.SEEDS)}
    stats["dnative"] = v3.hist_stats([dn[j] for j in range(len(v3.DEPTHS))], valid)
    hec = cv2.resize(hec_raw.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_LINEAR)
    stats["hecate"] = (float(hec[valid].mean()), float(hec[valid].std()) + 1e-6)

    z = (v3.villa_t(dn[k]) - stats["dnative"][0]) / stats["dnative"][1]
    z += (hec - stats["hecate"][0]) / stats["hecate"][1]
    for si, s in enumerate(v2.SEEDS):
        z += (v3.villa_t(probs[si, rev, k]) - stats[f"ink9-{s}"][0]) / stats[f"ink9-{s}"][1]
    z /= 4

    out = {"note": "diagnostic written after the main scores; same folds, bands and surrogate null as results.json"}
    rng = np.random.default_rng(0)
    for label, full in (("ens4_d+1_raw_z", z), ("ens4_d+1_overlay_alpha", v3.overlay_alpha(z))):
        pm, pv = v2.pool(full.astype(np.float32), valid)
        folds, _ = v3.make_folds(pv)
        pm = pm.astype(np.float32)
        out[label] = {f: v3.evaluate(fo, pm, rng, extra=False) for f, fo in folds.items()}
        print(label, json.dumps(out[label]), flush=True)
    json.dump(out, open(v3.HERE / "diag_overlay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
