#!/usr/bin/env bash
# From a fresh clone to the held-out numbers of the PHerc. 343 positive control, CPU only.
#
#   bash tools/inkcheck/quickstart.sh 2>&1 | tee quickstart.log
#
# 1. setup.sh: pinned villa checkout, its locked venv, inkcheck, the ink_9um checkpoints
# 2. the reference repository (mesh, labels, published map), pinned commit
# 3. the dense_native checkpoint, rebuilt from its public release and sha256-checked
# 4. experiments/2026-10-12-pherc343-positive-control-v3/quickstart.py: render, 3 maps, held-out scoring,
#    compared number by number with results.json (exit 1 on any difference)
#
# Optional: VILLA_MIRROR=<villa checkout>, REF_MIRROR=<reference checkout> to copy the pinned commits from
# a local checkout instead of GitHub; INKCHECK_CACHE=<dir> for the chunk cache; INKCHECK_CPU_THREADS (default 12);
# INKCHECK_MAX_CONNECTIONS (default 8) for parallel chunk downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REF_DIR="$ROOT/external/pherc343-first-letters"
REF_REMOTE="https://github.com/Nieuwlaar/pherc343-first-letters.git"
REF_COMMIT="e8370f1f88a211166056df62ea6fcfffe4a1063e"   # 2026-10-07, "PHerc. 343 First Letters: code and outputs"
REF_MIRROR="${REF_MIRROR:-}"
PY="$ROOT/external/villa/vesuvius/.venv/bin/python"

echo "=== 1/4 setup.sh $(date -Is)"
bash "$ROOT/setup.sh"

echo "=== 2/4 reference repository $(date -Is)"
if [ ! -d "$REF_DIR/.git" ]; then
  mkdir -p "$REF_DIR"
  git init -q "$REF_DIR"
  git -C "$REF_DIR" remote add origin "$REF_REMOTE"
  git -C "$REF_DIR" fetch -q --depth 1 "${REF_MIRROR:-origin}" "$REF_COMMIT"
  git -C "$REF_DIR" checkout -q FETCH_HEAD
fi
got="$(git -C "$REF_DIR" rev-parse HEAD)"
[ "$got" = "$REF_COMMIT" ] || { echo "external/pherc343-first-letters is at $got, pinned $REF_COMMIT" >&2; exit 1; }
(cd "$REF_DIR/outputs" && sha256sum -c --quiet SHA256SUMS)
echo "reference $got, outputs/SHA256SUMS ok"

echo "=== 3/4 dense_native checkpoint $(date -Is)"
TSV="$(mktemp)"
trap 'rm -f "$TSV"' EXIT
grep -E '^(key|dnative)' "$REF_DIR/ink/models.tsv" > "$TSV"   # dnative only; hecate is not needed here
PY="$PY" bash "$REF_DIR/ink/scripts/fetch_checkpoints.sh" "$TSV" "$ROOT/data/models/pherc343_ref"

echo "=== 4/4 held-out scoring $(date -Is)"
export CUDA_VISIBLE_DEVICES=-1 INKCHECK_DEVICE=cpu INKCHECK_CPU_THREADS="${INKCHECK_CPU_THREADS:-12}"
"$PY" "$ROOT/experiments/2026-10-12-pherc343-positive-control-v3/quickstart.py"
echo "=== done $(date -Is)"
