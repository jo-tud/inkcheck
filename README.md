# inkcheck

The aim of inkcheck is to find out why ink detection shows nothing on a Vesuvius Challenge surface
patch: the surface is off the sheet, the ink is at another depth, the scan is unlike the model's
training data, or there is simply no detectable ink. CPU only.

v0.1 is the first step: it reproduces the published PHerc. 343 First Letters result
([Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters)) from the
public scan with my own code, and scores it on held-out data. A detector that cannot find known ink
tells you nothing about unknown ink, so this comes first.

By Johannes Pfeffer ([@jo-tud](https://github.com/jo-tud)).

![held-out maps and the published map](experiments/2026-10-12-pherc343-positive-control-v3/fig_reading.png)

## Run it

Linux, [uv](https://docs.astral.sh/uv/), about 15 GB free disk, 16 GB RAM, no GPU.

```bash
bash tools/inkcheck/quickstart.sh 2>&1 | tee quickstart.log
```

This sets up a pinned villa environment, fetches the reference repository and the public
checkpoints, streams about 2 GiB of the scan, renders the area, runs the ink models on CPU and
checks every number against the stored results. 14–16 min on 12 threads once the packages are
downloaded.

## What came out

- I split the area into two parts, chose model and depth on one and scored the other, both ways
  round. Held-out box AUC: 0.96 and 0.98 (99th percentile of the null: 0.90 and 0.91). This shows
  agreement with the letter-box placement at box level, not ink.
- Held-out pixel AUC: 0.73 and 0.58. I fixed a bar of 0.60 before running the evaluation, so the
  pixel-level check fails on one of the two parts; there 0.58 is also inside the null (99th
  percentile 0.66).
- My four-model ensemble matches the published ink map at Pearson r = 0.84.
- Looking at the maps, letters are readable in some boxes only.

## What does not work yet

- The letter boxes were drawn on the published map of the same models I run, so this is a check of
  the reproduction, not independent proof of ink.
- Only PHerc. 343 so far. A check against a second ink map on three PHerc. 139 segments did not pass
  (one of three segments under the bar).
- An ink-transplant calibration on the same two configurations failed: letter-shaped shams were
  flagged about as often as planted ink.
- The PHerc. 343 signal is lost when the area is remapped to the grey values of other 8.64 µm scans
  (0 of 9), so an empty map elsewhere can come from the scan alone.
- The diagnostic itself (off-sheet, depth, domain gap, transplanted ink) is library code without a
  validated result yet.

Details, the full run and all limits: [`tools/inkcheck/README.md`](tools/inkcheck/README.md).

| Path | Content |
|---|---|
| `tools/inkcheck/` | the tool: code, tests, quickstart |
| `experiments/2026-10-12-pherc343-positive-control-v2/` | depth sweep and normal direction |
| `experiments/2026-10-12-pherc343-positive-control-v3/` | the held-out result above |

## Licence and data

Code: MIT, see `LICENSE`.

The scan data this code reads is published by the Vesuvius Challenge and EduceLab under
CC BY-NC 4.0 and is not redistributed here. Renders and ink maps made from it fall under the same
terms. The PHerc. 343 scan used here is cited as:

> Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian Rafael Dal Prà, Johannes Rudolph, Paul
> Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest McDonald, David Josey,
> Youssef Nader, C. Seth Parker, W. Brent Seales. *Vesuvius Challenge – CT Scans of Herculaneum
> Papyri.* Vesuvius Challenge.

Scans made before 2025 are cited as:

> S. Parsons, C. S. Parker, C. Chapman, M. Hayashida, W. B. Seales, *EduceLab-Scrolls: Verifiable
> Recovery of Text from Herculaneum Papyri using X-ray CT*, arXiv:2304.02084, 2023.

See <https://scrollprize.org/data> for the data agreement.
