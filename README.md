# inkcheck v0.1.2

**What v0.1 measures.** Whether my own code, run from the public scan, re-makes the published
PHerc. 343 First Letters ink map
([Nieuwlaar/pherc343-first-letters](https://github.com/Nieuwlaar/pherc343-first-letters)) and the
scores that map gets against the reference author's letter boxes, on a part of the area I did not use
to choose model and depth. It does not measure ink: the letter boxes were drawn on the published map
of the same models I run. CPU only.

Where this is going: a tool that finds out why ink detection shows nothing on a Vesuvius Challenge
surface patch: the surface is off the sheet, the ink is at another depth, the scan is unlike the
model's training data, or there is simply no detectable ink. v0.1 does not do that yet. A detector
that cannot find the published letters tells you little about unknown ink, so this comes first.

By Johannes Pfeffer ([@jo-tud](https://github.com/jo-tud)).

![held-out maps and the published map](experiments/2026-10-12-pherc343-positive-control-v3/fig_reading.png)

Top two panels: the held-out maps (top: chosen on the right part, held out on the left of the pink
lines; second: chosen on the left part, held out on the right). Third: my four-model ensemble at the
reference setting. Bottom: the published map.

## Run it

Linux, x86-64, [uv](https://docs.astral.sh/uv/), git, curl, 16 GB RAM, about 15 GB free disk, no GPU.

```bash
git clone https://github.com/jo-tud/inkcheck && cd inkcheck
bash tools/inkcheck/quickstart.sh 2>&1 | tee quickstart.log
```

This sets up a pinned villa environment, fetches the reference repository and the public
checkpoints, streams about 2 GiB of the scan, renders the area, runs the ink models on CPU and
checks every number against the stored results. The first run downloads several GB of packages and
models. On 12 threads it takes 14–17 min once the packages are downloaded. From an empty 16 GB
container with all downloads, two runs took 16 min 29 s and 23 min 13 s (the difference is mostly
download time); the largest process used 8.1 GB.

**Input.** None of your own: v0.1 runs this one PHerc. 343 area, with the reference repository's
mesh, letter boxes and published map, and the public scan streamed over HTTPS. It has no tested
command for your own segment yet.

**A good run** ends with `ALL NUMBERS MATCH results.json` and exit code 0; if any number differs,
the command exits 1. The bars the scores were judged against (fixed before the evaluation) are box
AUC ≥ 0.80 and pixel AUC ≥ 0.60 on both parts, each above the null's 99th percentile.

## What came out

- My four-model ensemble re-makes the published ink map at Pearson r = 0.84 (same checkpoints, same
  render geometry). That checks my reproduction, not ink. One of the four models (hecate) has no
  published training list, so I cannot tell whether it saw PHerc. 343; the held-out AUCs below do
  not use it.
- I split the area into two parts, chose model and depth on one and scored the other, both ways
  round. Held-out box AUC: 0.96 (9 letter boxes against 18 between-line bands) and 0.98 (8 against
  16); the 99th percentile of the null is 0.90 and 0.91. This measures agreement with where a person
  drew the letter boxes on the published map, not ink.
- Held-out pixel AUC: 0.73 and 0.58. I fixed a bar of 0.60 before running the evaluation, so the
  pixel-level check fails on one of the two parts; there 0.58 is also inside the null (99th
  percentile 0.66). So the check I registered fails as a whole (it needed both levels on both parts),
  and I rate the box-level agreement as low confidence.
- My own unblinded look at the maps: letters are readable in some boxes only (not a scored test).

## What does not work yet

- The letter boxes were drawn on the published map of the same models I run, so this is a check of
  the reproduction, not independent proof of ink.
- Only PHerc. 343 so far. A check against a second ink map on three PHerc. 139 segments did not pass
  (one of three segments under the bar).
- An ink-transplant calibration on the same two configurations failed. When I planted PHerc. 343
  letter signal and letter-shaped shams in the same scan (60 locations per part), one configuration
  flagged shams about as often as planted ink (0.20 and 0.22) and the other flagged neither. A second
  run with synthetic ink strokes failed the same way. These runs are not part of this repository.
- ink_9um no longer reproduces its own PHerc. 343 output when the area is remapped to the grey
  values of other 8.64 µm scans (0 of 9), so an empty map elsewhere can come from the scan alone.
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
