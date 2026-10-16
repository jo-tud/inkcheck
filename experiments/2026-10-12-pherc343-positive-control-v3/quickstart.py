"""PHerc. 343 positive-control quickstart: the two held-out evaluations, from a fresh clone, on CPU
(about 16 min measured, see tools/inkcheck/README.md).

The full run (`run.py`, about 7 h on CPU from a clean clone, 3 h of it v3-specific) sweeps 4 models x 31 depths, picks one configuration per fold and scores it
on the other fold. This script skips the sweep: it takes the two configurations that sweep chose (recorded in
results.json), renders the area, computes only the three maps they need and scores them with the same code, the same
labels and the same random seed as run.py. It then compares every number with results.json.

    python quickstart.py [--out DIR]

  1. render  66 layers along the outward normal over the reference area (streamed from the open-data bucket)
             -> OUT/layers.npy, OUT/valid.npy      (reused from ../...-v2/ if a full run already made them)
  2. infer   dnative reverse d=+3, ink9-42 and ink9-43 reverse d=-1       -> OUT/<model>_d<d>.npy (uint8, as run.py)
  3. score   held-out fold B (dnative d=+3) and fold A (ink9-mean d=-1)   -> OUT/quickstart_results.json

It changes nothing in run.py and writes nothing outside OUT.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pherc343_v3", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run = load_run()
v2 = run.v2

# the two configurations run.py's sweep selected (results.json "heldout"); fixed here, not re-selected
HELDOUT = (("select_A_eval_B", "B", "dnative", 3), ("select_B_eval_A", "A", "ink9-mean", -1))


def peak_rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def stage_render(out):
    """Same call as v2.stage_render (region, offsets, volume, level 0, 6 workers); uint8 conversion done in place."""
    for d in (v2.HERE, out):
        if (d / "layers.npy").exists() and (d / "valid.npy").exists():
            print(f"render: cached {d / 'layers.npy'}", flush=True)
            return d
    from inkcheck.catalog import reference_volume
    from inkcheck.render import render
    from inkcheck.volume import Volume
    surf, n, agree, _ = v2.load_surface()
    vi = reference_volume()
    print("volume", vi.url, "outward agreement", round(agree, 4), flush=True)
    t = time.time()
    layers, valid = render(surf, n, Volume(vi.url, 0), v2.OFFSETS, region=(v2.R0, v2.R1, v2.C0, v2.C1), workers=6,
                           progress=lambda i, k: print(f"render {i}/{k} {time.time() - t:.0f}s", flush=True)
                           if i % 10 == 0 else None)
    dt = time.time() - t
    np.rint(layers, out=layers)
    np.clip(layers, 0, 255, out=layers)
    np.save(out / "layers.npy", layers.astype(np.uint8))
    np.save(out / "valid.npy", valid)
    del layers
    json.dump({"render_s": round(dt, 1), "volume_url": vi.url, "outward_agreement": agree,
               "valid_fraction": float(valid.mean())}, open(out / "render.json", "w"), indent=1)
    print(f"render {dt:.0f}s valid {valid.mean():.3f} peak RSS {peak_rss_gib():.1f} GiB", flush=True)
    return out


def stage_infer(src, out):
    """run.py / v2 run.py inference for the three maps the held-out configurations need."""
    from inkcheck.ink import load_model, predict
    layers = np.load(src / "layers.npy", mmap_mode="r")
    valid = np.load(src / "valid.npy")
    jobs = [("dnative", 3, str(run.CKPT / "dense_native-016000.rebuilt.pth"))]
    jobs += [(f"ink9-{s}", -1, str(v2.MODELS / f"hybrid_3d2d-seed{s}/step-075000.pth")) for s in v2.SEEDS]
    times = {}
    for name, d, ckpt in jobs:
        f = out / f"{name}_d{d:+d}.npy"
        if f.exists():
            print(f"infer: cached {f.name}", flush=True)
            continue
        t = time.time()
        m = load_model(ckpt, name)
        win = np.asarray(layers[25 + d:42 + d]).astype(np.float32)   # 17 layers centred on 33 + d
        pr = predict(m, win, valid, center=8, direction="reverse")
        np.save(f, np.clip(np.rint(np.nan_to_num(pr) * 255), 0, 255).astype(np.uint8))
        times[f"{name}/reverse/{d}"] = round(time.time() - t, 1)
        del m
        print(f"infer {name} reverse d={d:+d} {times[f'{name}/reverse/{d}']:.0f}s peak RSS {peak_rss_gib():.1f} GiB",
              flush=True)
    return times


def stage_score(src, out):
    """stage_score's pooling and evaluate() calls for the held-out maps, in the same order with the same seed."""
    valid = np.load(src / "valid.npy")
    dn, pv = v2.pool(np.load(out / "dnative_d+3.npy") / 255.0, valid)
    i42, _ = v2.pool(np.load(out / "ink9-42_d-1.npy") / 255.0, valid)
    i43, _ = v2.pool(np.load(out / "ink9-43_d-1.npy") / 255.0, valid)
    maps = {"dnative": dn.astype(np.float32), "ink9-mean": ((i42 + i43) / 2).astype(np.float32)}
    folds, _ = run.make_folds(pv)
    rng = np.random.default_rng(0)                    # run.py: same seed, B evaluated first, then A
    res = {}
    for key, fold, model, d in HELDOUT:
        t = time.time()
        res[key] = {"config": {"model": model, "dir": "reverse", "d": d}, **run.evaluate(folds[fold], maps[model], rng)}
        print(f"score {key} {time.time() - t:.0f}s", flush=True)
    return res


def compare(res):
    ref = json.load(open(HERE / "results.json"))["heldout"]
    rows, same = [], True
    for key, fold, model, d in HELDOUT:
        for metric in ("box_auc", "pixel_auc"):
            a, b = res[key][metric], ref[key][metric]
            ok = all(a[k] == b[k] for k in ("observed", "null_p50", "null_p99", "p_value"))
            same &= ok
            rows.append((f"{model} d={d:+d}", fold, metric, a, b, ok))
        ok = res[key]["meets_bar"] == ref[key]["meets_bar"]
        same &= ok
    print()
    print("PHerc. 343 positive control, held-out evaluations (PHerc0343 20250521140437, 8.64 um, 116 keV, level 0, reverse normals)")
    print(f"{'map':<18}{'held-out fold':<15}{'metric':<11}{'observed':>9}{'null p50':>10}{'null p99':>10}{'p':>8}"
          f"   results.json")
    for name, fold, metric, a, b, ok in rows:
        print(f"{name:<18}{fold:<15}{metric:<11}{a['observed']:>9.4f}{a['null_p50']:>10.4f}{a['null_p99']:>10.4f}"
              f"{a['p_value']:>8.4f}   {'same' if ok else 'DIFFERENT: ' + json.dumps(b)}")
    for key, fold, model, d in HELDOUT:
        print(f"fold {fold} meets the bar (box >= 0.80, pixel >= 0.60, both > null p99): {res[key]['meets_bar']}")
    print("box level: held-out box AUC above 0.80 and above the null p99 on both folds:",
          all(res[k]["box_auc"]["observed"] >= 0.80 and res[k]["box_auc"]["observed"] > res[k]["box_auc"]["null_p99"]
              for k, *_ in HELDOUT))
    print("ALL NUMBERS MATCH results.json" if same else "MISMATCH with results.json (see rows above)")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "quickstart"))
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    src = stage_render(out)
    t1 = time.time()
    times = stage_infer(src, out)
    t2 = time.time()
    res = stage_score(src, out)
    t3 = time.time()
    same = compare(res)
    summary = {"heldout": res, "matches_results_json": same, "layers_from": str(src), "map_seconds": times,
               "wall_s": {"render": round(t1 - t0, 1), "infer": round(t2 - t1, 1), "score": round(t3 - t2, 1),
                          "total": round(t3 - t0, 1)},
               "threads": run.threads(), "peak_rss_gib_this_process": round(peak_rss_gib(), 2)}
    json.dump(summary, open(out / "quickstart_results.json", "w"), indent=1)
    print(f"wall {summary['wall_s']} s, peak RSS {summary['peak_rss_gib_this_process']} GiB -> {out / 'quickstart_results.json'}")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
