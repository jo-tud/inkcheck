# inkcheck v0.1.2

**What v0.1 measures.** Whether my own code, run from the public scan, re-makes the published
PHerc. 343 First Letters ink map (https://github.com/Nieuwlaar/pherc343-first-letters) and the
scores that map gets against the reference author's letter boxes, on a part of the area I did not use
to choose model and depth, against a surrogate null. It does not measure ink: the letter boxes were
drawn on the published map of the same models I run. No GPU needed: a fresh clone gets to the
held-out numbers in 14–17 min on 12 threads once the packages are downloaded (details under
Quickstart).

Where this is going: a diagnostic for why ink detection comes up empty on a surface patch. A blank
ink map has several possible causes, and today they are hard to tell apart:

- the surface is off the sheet or crosses layers;
- the ink sits at a different depth than the model window;
- the scan looks unlike the scrolls the model was trained on (domain gap);
- the ink is too faint for the model at this resolution, or there is no text there.

**v0.1 does not do that yet.** It ships the library parts (streaming zarr reader, renderer, villa
ink-model wrapper, sheet-profile and patch-growing code, an ink-transplant module), a small
`inkcheck` command, and one reproduced result.

## The PHerc. 343 result

`experiments/2026-10-12-pherc343-positive-control-v3/`: inkcheck renders the reference area from the
public scan, runs four published ink models on CPU (ink9-42, ink9-43 and dense_native at 9 µm,
hecate at 9.6 µm) and scores the maps against the reference repository's 18 letter boxes (17
scored; one crosses the fold guard). To keep the choice of model and depth honest, I split the area
into two folds, chose the model and depth on one fold and scored the other.

My four-model ensemble re-makes the published ink map at Pearson r = 0.84 (same checkpoints, same
render geometry, same transform). That checks the reproduction; it is not an ink measurement. I have
no reference value for the r an exact re-run should reach. One of the four models (hecate) has no
published training list, so I cannot tell whether it saw PHerc. 343; the training lists of ink_9um
and dense_native do not name PHerc. 343. The two held-out rows below do not use hecate; the r = 0.84
ensemble does.

**The letter boxes are not ground truth.** The reference author drew them by hand on the published
ink map, and that map is the mean of the same four models I run. So the box AUC below measures
agreement between my maps and a person's box placement on that ensemble map, not independent
evidence that the boxes hold ink. The fold split controls the choice of model and depth; it does not
remove this.

![held-out maps and the published map](../../experiments/2026-10-12-pherc343-positive-control-v3/fig_reading.png)

In reading orientation fold A is right of the pink guard lines and fold B is left of them. Each of
the two held-out panels is held out on one side only: the top panel (chosen on fold A) on the left,
the second panel (chosen on fold B) on the right. The other side of each panel is the data the
configuration was chosen on.

| Held-out map (chosen on the other fold) | Fold | Letter boxes / between-line bands | Box AUC (null p50 / p99, p) | Pixel AUC (null p50 / p99, p) |
|---|---|---|---|---|
| dnative, reverse, d = +3 | B | 8 / 16 | 0.98 (0.64 / 0.91, 0.002) | 0.58 (0.50 / 0.66, 0.16) |
| ink9-mean, reverse, d = −1 | A | 9 / 18 | 0.96 (0.61 / 0.90, 0.003) | 0.73 (0.50 / 0.67, 0.001) |

I fixed the thresholds before running the evaluation (box AUC ≥ 0.80 and pixel AUC ≥ 0.60 on both
folds, both above the null's 99th percentile). Box AUC passes on both folds. Pixel AUC fails on fold
B (0.58, also under the null's 99th percentile of 0.66). So the check I registered fails as a whole
(it needed both levels on both folds), and I rate the box-level agreement as low confidence. My own
unblinded look at the maps: letters are readable in some boxes only (not a scored test).

The selection was a tie: on the selection fold, 9 configurations reach box AUC 1.0 on fold A and 4
on fold B. The tie is broken by the selection fold's pixel AUC, then by the smaller |d|; this uses no
held-out data, but the reported numbers depend on it. Across the 9 tied choices on fold A, the
held-out fold B scores range from 0.93 to 0.98 (box) and 0.58 to 0.65 (pixel); across the 4 tied
choices on fold B, fold A scores 0.92–0.99 (box) and 0.71–0.77 (pixel). So another tie-break could
have put fold B pixel AUC above the 0.60 bar. All fold B values stay under 0.66, the null's 99th
percentile for the chosen map; the null was not computed for the other tied maps.

Volume PHerc0343 `20250521140437-8.640um-1.2m-116keV-masked`, 8.64 µm, 116 keV, level 0. Normals point
outward (away from the scroll axis), and the model window is fed in reverse layer order. Depth sweep
d = −15..+15 voxels. Folds split by raw column: A 380–2486 (9 letters, 18 between-line bands), guard
2486–2610, B 2610–6021 (8 letters, 16 bands). Null: 1000 surrogate maps with the same power spectrum
and value distribution. The experiment README has the full method and tables;
`experiments/2026-10-12-pherc343-positive-control-v2/` has the depth sweep and the normal-direction
test it builds on.

## Quickstart: fresh clone to the held-out numbers

Linux, x86-64, [uv](https://docs.astral.sh/uv/), git, curl, 16 GB RAM, about 15 GB free disk
(after a run the clone holds about 9.3 GB and the chunk cache 2.0 GiB, plus the uv package cache),
network access to GitHub, Hugging Face and the public scroll bucket. No GPU.

Input: none of your own. v0.1 runs one fixed PHerc. 343 area, with the reference repository's mesh,
letter boxes and published map, and the public scan streamed over HTTPS. It has no tested command
for your own segment yet.

```bash
git clone https://github.com/jo-tud/inkcheck && cd inkcheck
bash tools/inkcheck/quickstart.sh 2>&1 | tee quickstart.log
```

This one command:

1. runs `setup.sh`: villa at the pinned commit, its locked venv, inkcheck, the ink_9um checkpoints;
2. fetches the reference repository (mesh, letter labels, published map) at a pinned commit and
   checks its sha256 sums;
3. rebuilds the dense_native checkpoint from its public release (sha256-checked);
4. runs `experiments/2026-10-12-pherc343-positive-control-v3/quickstart.py`. This renders the area
   from the scan (streamed, about 2 GiB of chunks) and computes the three maps the two held-out
   configurations need. It scores them with the experiment's own code and random seed and prints the
   table above. Each number is compared with `results.json`, and the command exits 1 if any differs.

A good run ends with `ALL NUMBERS MATCH results.json` and exit code 0.

Measured on 2026-10-20 for v0.1 (the code is unchanged since): an anonymous clone in an empty
container (12 CPUs, 16 GB memory limit, no swap, no GPU, empty package and chunk caches) took
**16 min 29 s wall-clock, all downloads included (10 min 55 s of it the render, inference and
scoring), 8.1 GB in the largest process**. The container's memory peak, file cache included,
reached the 16 GB limit without running out of memory, so give it 16 GB. The same test on the
pre-release tree four days earlier took 23 min 13 s; the difference is mostly download time, which
depends on your connection (villa is about 775 MiB, the venv packages several GB, mostly torch). A
run with warm package and chunk caches took 14 min 18 s.

The quickstart does **not** re-run the configuration selection. It takes the two configurations the
full sweep chose (recorded in `results.json`) and reproduces their held-out scores.

### Full run

The full run sweeps the models over 31 depths, selects one configuration per fold, and runs hecate,
the parity check and the figures. From a clean clone on CPU (12 threads), the measured stage times add
up to about **7 h**:

| Stage | Maps | Time |
|---|---|---|
| render (v2 `run.py`) | 66 layers | 17 min |
| ink9-42/43 sweep (v2 `run.py`, forward and reverse) | 124 × about 110 s | 3.9 h |
| dnative sweep (reverse) | 31 × about 110 s | 57 min |
| hecate, d = +1 | 1 | 1.8 h |
| scoring and figures (plus the synthetic check, not timed) | | 13 min |

```bash
bash tools/inkcheck/quickstart.sh       # setup, reference repository, dense_native (and the quickstart)
PY=$PWD/external/villa/vesuvius/.venv/bin/python
PY=$PY bash external/pherc343-first-letters/ink/scripts/fetch_checkpoints.sh \
    external/pherc343-first-letters/ink/models.tsv data/models/pherc343_ref      # adds hecate
export CUDA_VISIBLE_DEVICES=-1 INKCHECK_DEVICE=cpu INKCHECK_CPU_THREADS=12
cd experiments/2026-10-12-pherc343-positive-control-v3 && $PY run.py
```

`python run.py --stage score` rescores from cached maps in 13 min and rewrites `results.json` and the
figures. `python diag_overlay.py` (7 min) runs the display-transform diagnostic described in the
experiment README.

## Setup

From the repo root: `bash setup.sh 2>&1 | tee setup.log`. One command, idempotent, pinned to a
specific `villa` commit. Installs into `external/villa/vesuvius/.venv`. CPU only: it hides the GPUs
(`CUDA_VISIBLE_DEVICES=-1`) unless `INKCHECK_ALLOW_LOCAL_GPU=1`.
`python tools/inkcheck/scripts/check_env.py` checks the pins (villa commit, checkpoint hashes, device
rule) at any time. `VILLA_MIRROR=<path to a villa checkout>` copies the pinned commit from a local
checkout instead of GitHub; the result is the same commit. The checkpoint download uses plain HTTPS,
one file at a time (`HF_HUB_DISABLE_XET=1`, `--max-workers 1`): at most 3 open connections, against
16 with Hugging Face's default Xet client.

### Pinned dependencies

- The venv comes from villa's `uv.lock` at the pinned commit (`setup.sh`, `VILLA_COMMIT`), installed
  with `uv sync --python 3.14.4 --extra models --extra tests` (torch 2.12.1, numpy 2.4.6, scipy 1.18.0
  in the tested venv). `check_env.py` (also run by `setup.sh`) prints torch as `2.12.1+cu130`. That
  is the same PyPI wheel, `torch==2.12.1` in `uv.lock` and `requirements.lock`; `+cu130` is the CUDA
  13.0 build tag in the wheel's own version string and shows on CPU-only runs too. The interpreter
  is pinned to the patch release (`INKCHECK_PYTHON`, default 3.14.4); villa's own `.python-version`
  only says `3.14`, which let a fresh run pick 3.14.8.
- `tools/inkcheck/requirements.lock` lists every package and version in the venv the fresh-clone
  quickstart ran in (`uv pip freeze`). Use it to check or rebuild an environment:
  `uv pip install -r tools/inkcheck/requirements.lock` (local-path lines aside).
- Checkpoints are pinned by Hugging Face revision (`setup.sh`) and by sha256 (`check_env.py`,
  `external/pherc343-first-letters/ink/models.tsv`). The reference repository is pinned by commit
  (`quickstart.sh`).
- `pyproject.toml` gives the lower bounds inkcheck itself needs.

## Remote volumes (`inkcheck.volume`)

Reads chunks of the public open-data zarr arrays directly over HTTPS into a disk cache, chunk by
chunk, so a box read only fetches the chunks it actually touches.

- `INKCHECK_CACHE` (default `.cache/inkcheck` under the account's real home directory, not `$HOME`,
  so runs with a throwaway `$HOME` still share one cache and one budget).
- `INKCHECK_CACHE_MAX_GB` (default 200): hard ceiling. Least-recently-used chunks are evicted
  automatically once exceeded.
- `INKCHECK_MAX_CONNECTIONS` (default 8): most chunk downloads one process runs at once, summed over
  every volume, tile and reader thread (a render of 6 tiles with 16 reader threads each would
  otherwise open up to 96). Requests wait for a free connection; cache hits are not limited.
  `inkcheck.volume.connection_stats()` reports the cap and the peak reached.
- `python tools/inkcheck/scripts/bench_volume.py`: measures cold/warm/misaligned throughput against
  a real volume. Run this after touching `volume.py` and compare before/after numbers.
- `python tools/inkcheck/scripts/s3_probe.py`: stdlib-only S3 latency and throughput probe on a
  fixed chunk slab (~96 MiB). Run it first on any new machine so its numbers compare with mine.
- `python tools/inkcheck/scripts/chunk_footprint.py --tifxyz ... --umbilicus ...`: how many chunks
  and bytes a render of a real mesh will pull, and how many are already cached, without downloading
  them.
- `inkcheck.render` fetches only the chunks its trilinear samples touch (`linear_stencil_chunks`),
  not every chunk of each tile's bounding box, with the same output bit for bit (a test checks
  this). `chunk_footprint.py` above prints the chunk and byte counts of both readers for a mesh.
  `render(..., sparse=False)` restores the old reader.
- `python tools/inkcheck/scripts/bench_render.py --tifxyz ... --umbilicus ... --region ...
  [--reader stencil|bbox] [--cache EMPTY_DIR] [--reference layers.npy]`: times a real render, counts
  the bytes it downloads, and checks the output against a saved render.

Never download a whole volume.

## Compute (`inkcheck.gpu`)

CPU is the default and normal path. `inkcheck.ink.load_model(checkpoint, device=..., vram_gb=...,
cpu_threads=...)` resolves compute explicitly instead of grabbing whatever is available:

- `INKCHECK_CPU_THREADS` (default: half the machine's cores): caps torch and, via `threadpoolctl`,
  the BLAS/OpenMP pools under numpy and scipy.
- `INKCHECK_DEVICE` (default `cpu`): asking for a CUDA device raises unless
  `INKCHECK_ALLOW_LOCAL_GPU=1`.
- `INKCHECK_VRAM_GB`: caps the process to that much VRAM and refuses to start if it is not free.
- `python -m inkcheck.gpu`: CPU thread default and, if GPU use is enabled, free/total VRAM per GPU.
- `python tools/inkcheck/scripts/bench_cpu_run.py`: times one render + inference run on CPU and
  reports wall-clock and peak RSS.

## Command line

`setup.sh` installs the `inkcheck` command into the venv (`python -m inkcheck` works too):

- `inkcheck info [--json]`: version, module path, device rule, CPU threads, chunk cache and the
  pinned PHerc. 343 volume. No network.
- `inkcheck volume [SAMPLE] [--volume-id ID]`: resolve a volume from the open-data catalog. Without
  arguments, the pinned PHerc. 343 volume; the command fails if the catalog maps that id to another
  URL or voxel size.

The PHerc. 343 reproduction is `bash tools/inkcheck/quickstart.sh` (above), not a subcommand: it
needs the repository layout (root `setup.sh`, the v2 and v3 experiment folders).

## Tests

`$PWD/external/villa/vesuvius/.venv/bin/python -m pytest tools/inkcheck/tests`, from the root of
the clone after the quickstart (or, in any Python ≥ 3.11 venv: `python -m pip install -e
"tools/inkcheck[tests]" && python -m pytest tools/inkcheck/tests`): 27 tests, no network, no torch, about 3 s (chunk de-duplication, sparse reads, pyramid level centres, patch growing, sheet
certification, and a smoke test of the `inkcheck` command). They cover
the reader, two geometry helpers and the command line. The command-line test runs the installed
`inkcheck` executable only when this checkout is the installed one; otherwise it skips and says so.
They do **not** cover the ink models or the PHerc. 343 scoring; the quickstart's comparison with
`results.json` is the end-to-end check for those.

## What does not work yet

- **No independent ink evidence.** Every score here is against letter boxes drawn on the published
  ensemble map of the models I score (see above). Nothing in v0.1 shows ink in those boxes
  independently of these four models.
- **Box level only.** The pixel-level check failed on one of the two folds (fold B pixel AUC 0.58,
  under 0.60 and under the null's 99th percentile of 0.66). So the result shows agreement with the
  letter-box placement at box level, not ink.
- **The ink-transplant calibration failed.** I planted PHerc. 343 letter signal (cut where these
  models see ink), and in a second run synthetic ink strokes, at other places in the same area (60
  locations per fold) and scored them with the same two held-out configurations. Planted ink was not
  found reliably at box level, and letter-shaped shams (letter shapes with no net added density) were
  flagged about as often as planted ink: on fold B 0.20 for shams against 0.22 for planted letter
  signal; on fold A neither was flagged. So these detectors do not yet tell ink from letter-shaped
  texture. Those runs are not part of this repository.
- **Letters are only partly legible.** My own unblinded look at the maps: letters are readable in
  some boxes only (not a scored test).
- **The null is sensitive to the display transform.** The surrogate null on pixel AUC changes under
  monotone transforms with ties. Putting my ensemble through the published display transform (which
  sets 85 % of pixels to zero) turns the fold B pixel fail into a pass, mostly by narrowing the null
  (p99 0.69 → 0.56) rather than by raising the score (0.643 → 0.628). A future version has to fix the
  map transform before scoring. The null here is a second set of 1000 draws; for the same raw map
  the scored run gives p99 0.70. Null percentiles move by about 0.01 between draws.
- **Only PHerc. 343 so far.** This is a positive control on one area of one scan, with labels and a
  mesh made by someone else. It shows that my code reproduces the published maps there. It says
  nothing yet about ink in any other scroll, segment or scan.
- **The PHerc. 343 signal does not survive another scan's grey values.** In a separate test I
  remapped the PHerc. 343 area to the grey-value distribution of each of nine other 8.64 µm, 116 keV
  scans and ran ink_9um seeds 42 and 43 on it: they no longer reproduced their own PHerc. 343 output
  on any of the nine (those letter rows come from ink_9um's own map, so this measures the model's
  robustness, not ink). A content-matched remap (papyrus and non-papyrus fitted separately) passed on
  0 of 2 scans: on PHerc. 175B the 343 rows came back above the bar but a forward-direction peak
  scored higher; on PHerc. 1218 they stayed below the null. So an empty map on another scan can come
  from the scan's grey values alone. That test is not part of this repository.
- **A check on a second scroll did not pass.** On three held-out PHerc. 139 segments (w024, w047,
  w053; scan at 9.362 µm, 113 keV), dense_native agreed with a separate 59 keV, 1.129 µm ink map at
  AUC 0.915, 0.899 and 0.763. I had asked for ≥ 0.80 on all three, so the check failed on one of the
  three segments. The registration check between the two scans failed on w047 and could not be
  measured on w024 and w053. The 59 keV map's training data is not published, so even a pass would
  have shown agreement between models, not ink. That check is not part of this repository.
- **The configuration selection is tuned on this area.** The held-out protocol picks the best of
  5 configurations (3 models, the mean of the two ink_9um seeds, and a 3-model ensemble) × 31 depths
  on one fold and scores the other (155 rows in `results.json`). hecate ran at one depth only (d = +1, the
  reference author's setting).
- **The sheet certificate (`inkcheck.sheetcert`) does not work as a certificate yet**: it fails its
  own half-spacing null on PHerc0139, PHerc0846A and PHerc1203 (H/F 0.65-0.85, where it needs 0.5
  or less; see the module docstring). The PHerc0139 number was re-made after the fix that reads
  pyramid levels above 0 at voxel centres; the other two were measured before that fix
  (their surfaces are not on disk).
- dense_native is rebuilt from its public release. The exact training checkpoint is not public.

## Data notice

The code in this repository is MIT-licensed (`LICENSE`). **The scroll data is not.** Vesuvius
Challenge CT data is licensed CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/; terms at
https://scrollprize.org/data). The renders, ink maps and figures this code produces from it are
derived from that data. They fall under the same terms: attribution, non-commercial use only. This
repository does not contain or redistribute scan data. The code streams it from the public bucket at
run time.

Scan used by v0.1: PHerc. 343 (ESRF, 8.64 µm, 116 keV), acquisition scan
`20250509220833-8.640um-1.2m-116keV`, reconstructed volume `20250521140437`
(`s3://vesuvius-challenge-open-data/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr`).
The volume is pinned by id and URL (`inkcheck.catalog.REFERENCE_VOLUME_ID`, `reference_volume()`),
so a later re-publication of PHerc0343 cannot silently change the scan under the mesh coordinates.
As scrollprize.org/data asks, cite the dataset:

> Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul
> Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey,
> Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge – CT Scans of Herculaneum
> Papyri.* Vesuvius Challenge.

## Credits

inkcheck builds on other people's open work.

- **villa** (https://github.com/ScrollPrize/villa, MIT, © 2024 Vesuvius Challenge), pinned at commit
  `47111d1a31381a5ff2c05046ed3e311b0dbeec64` in `setup.sh`. inkcheck loads its ink models through
  villa (`vesuvius.ink_detection.inference.infer`: `configure_model`, `normalize_flat_patch`).
  `inkcheck.ink.predict` re-implements its sliding-window inference in memory (patch normalisation,
  centred layer window, Hann blending). villa's ink detection was originally developed by Youssef
  Nader and Luke Farritor and extended by the Vesuvius Challenge team.
- **ink_9um checkpoints**, `hybrid_3d2d` seeds 42 and 43, step 75 000
  (https://huggingface.co/scrollprize/ink_9um, MIT, Vesuvius Challenge; revision
  `7109667e2607db1b90c37c8b09cb876ea7fe7bb1`).
- **dense_native**, step 16 000 (https://huggingface.co/Nieuwlaar/ink9um-dense-native, MIT, Erwin
  Nieuwlaar). Its model card says it is fine-tuned from KLAVIS's (DomRusso2) public `ink9um-dense`
  checkpoints and the ink_9um checkpoints and labels. I rebuild it from the public safetensors with the reference
  repository's `rebuild_dense_native_pth.py`.
- **Hecate 9.6 µm** (https://huggingface.co/scrollprize/hecate, MIT, Vesuvius Challenge), with its own
  `hecate.py`. It is used in the full run only, not in the quickstart.
- **PHerc. 343 First Letters reference** by Erwin Nieuwlaar
  (https://github.com/Nieuwlaar/pherc343-first-letters, MIT, commit `e8370f1`). It provides the
  segment mesh (`concat_w047-w048_R5B2_z9500-11000.tifxyz`), the scroll axis (umbilicus), the 18
  letter boxes (17 scored; one crosses the fold guard; drawn on the published ensemble map, not
  independent labels), and the
  published ink map I compare against. The v3 experiment also follows its methods closely: the hecate
  input preparation follows `hecate_prep.py`, the per-model z-score ensemble follows
  `ensemble_maps.py`, and the overlay transform used for the parity check follows
  `make_submission_image.py`. Each is re-implemented in
  `experiments/2026-10-12-pherc343-positive-control-v3/run.py` and described in that experiment's
  README.
- **Scan data:** Vesuvius Challenge and the ESRF (see [Data notice](#data-notice)).

## Licence

MIT, © 2026 jo-tud. See `LICENSE`. The licence covers the code only, not the scroll data or anything
derived from it.
