#!/usr/bin/env bash
# One-command setup for tools/inkcheck: pinned villa checkout, deps, env verification,
# ink_9um checkpoints. Idempotent -- safe to re-run after a partial failure.
#
# Usage:
#   bash setup.sh 2>&1 | tee setup.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VILLA_DIR="$REPO_ROOT/external/villa"
VILLA_REMOTE="https://github.com/ScrollPrize/villa.git"
# Pinned 2026-10-12. Bump deliberately (and re-run this script end to end) when villa moves;
# do not let a local checkout silently drift from what a fresh clone gets.
VILLA_COMMIT="47111d1a31381a5ff2c05046ed3e311b0dbeec64"
# Optional: an existing villa checkout on this machine to fetch from instead of GitHub (villa is
# ~775 MiB and a home line can be slow). Same pinned commit either way.
VILLA_MIRROR="${VILLA_MIRROR:-}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
# Patch release the venv was tested with. villa's .python-version says only "3.14", so without
# this uv takes the newest 3.14.x it can find (a fresh run on 2026-10-14 got 3.14.8).
INKCHECK_PYTHON="${INKCHECK_PYTHON:-3.14.4}"

# Some job runners hand each run a throwaway $HOME. Keep uv's download cache under the account's
# real home so a re-run does not fetch torch and friends (several GB) again.
REAL_HOME="$(getent passwd "$(id -u)" | cut -d: -f6)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REAL_HOME/.cache/uv}"

# CPU by default: hide the GPUs so not even the verification step below opens a CUDA context
# on them. Set INKCHECK_ALLOW_LOCAL_GPU=1 to let inkcheck use a local GPU.
if [ "${INKCHECK_ALLOW_LOCAL_GPU:-0}" != 1 ]; then
  export CUDA_VISIBLE_DEVICES=-1
fi
export INKCHECK_DEVICE="${INKCHECK_DEVICE:-cpu}"

# torch's cuda:N ordering is by compute capability, not PCI slot, so without this cuda:0 can be
# a *different physical card* than nvidia-smi's GPU 0 -- confirmed on my machine (cuda:0 ==
# the RTX PRO 6000 by default, but nvidia-smi GPU 0 == the RTX 5090). Pin PCI order so cuda:N
# always matches nvidia-smi's GPU N.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_RETRIES=10

command -v "$UV" >/dev/null 2>&1 || {
  echo "uv not found at $UV -- install from https://docs.astral.sh/uv/ (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
  exit 1
}

if [ ! -d "$VILLA_DIR/.git" ]; then
  echo "cloning villa @ $VILLA_COMMIT"
  mkdir -p "$VILLA_DIR"
  git init -q "$VILLA_DIR"
  git -C "$VILLA_DIR" remote add origin "$VILLA_REMOTE"
  if [ -n "$VILLA_MIRROR" ]; then
    echo "fetching villa from local mirror $VILLA_MIRROR"
    git -C "$VILLA_DIR" fetch -q --depth 1 "$VILLA_MIRROR" "$VILLA_COMMIT"
    git -C "$VILLA_DIR" checkout -q FETCH_HEAD
  elif ! git -C "$VILLA_DIR" fetch --depth 1 origin "$VILLA_COMMIT" 2>/dev/null; then
    echo "shallow fetch-by-commit not supported by the remote; falling back to a full clone"
    rm -rf "$VILLA_DIR"
    git clone -q "$VILLA_REMOTE" "$VILLA_DIR"
    git -C "$VILLA_DIR" checkout -q "$VILLA_COMMIT"
  else
    git -C "$VILLA_DIR" checkout -q FETCH_HEAD
  fi
else
  current="$(git -C "$VILLA_DIR" rev-parse HEAD)"
  if [ "$current" != "$VILLA_COMMIT" ]; then
    echo "WARNING: external/villa is at $current, pinned commit is $VILLA_COMMIT." >&2
    echo "Not touching it automatically -- it may hold uncommitted work. 'git -C external/villa checkout $VILLA_COMMIT' yourself if you want the pinned version." >&2
  fi
fi

cd "$VILLA_DIR/vesuvius"
ok=0
for i in 1 2 3 4 5 6 7 8; do
  if "$UV" sync --python "$INKCHECK_PYTHON" --extra models --extra tests --no-install-package volume-cartographer; then
    ok=1
    break
  fi
  echo "uv sync failed, retry $i/8"
  sleep 15
done
[ "$ok" = 1 ] || { echo "uv sync did not succeed after 8 retries" >&2; exit 1; }

"$UV" pip install --python .venv/bin/python -e "$REPO_ROOT/tools/inkcheck"

echo "--- verification ---"
echo "villa $(git -C "$VILLA_DIR" rev-parse HEAD) (pinned $VILLA_COMMIT)"
.venv/bin/python -c "
import sys, numpy, scipy, requests, torch, inkcheck
from inkcheck.gpu import resolve_device
from inkcheck.volume import cache_max_bytes, default_cache_dir
print('python', sys.version.split()[0], '| numpy', numpy.__version__, '| scipy', scipy.__version__,
      '| requests', requests.__version__, '| torch', torch.__version__)
print('inkcheck', inkcheck.__version__, inkcheck.__file__)
print('device', resolve_device(), '| torch threads', torch.get_num_threads(),
      '| cuda visible to torch:', torch.cuda.device_count())
print('chunk cache', default_cache_dir(), '| cap', cache_max_bytes() / 2**30, 'GiB')
"
echo "env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} INKCHECK_DEVICE=$INKCHECK_DEVICE"
echo "     set both in your shell too: export CUDA_VISIBLE_DEVICES=-1 INKCHECK_DEVICE=cpu"

cd "$REPO_ROOT"
# Fixed Hugging Face revision (2026-08-14); check_env.py checks the hashes.
# At most 8 connections per job. The default Xet client opened 16 (measured on a fresh clone);
# plain HTTPS with one worker peaks at 3 (2 x 132 MiB in about 8 s, hashes unchanged).
HF_HUB_DISABLE_XET=1 "$VILLA_DIR/vesuvius/.venv/bin/hf" download scrollprize/ink_9um \
  hybrid_3d2d-seed42/step-075000.pth hybrid_3d2d-seed43/step-075000.pth \
  --revision 7109667e2607db1b90c37c8b09cb876ea7fe7bb1 --local-dir data/models/ink_9um \
  --max-workers 1

echo "--- checkpoints ---"
ls -la data/models/ink_9um/hybrid_3d2d-seed42/step-075000.pth data/models/ink_9um/hybrid_3d2d-seed43/step-075000.pth

# Warn rather than stop, like the villa drift check above; the Docker image treats this as fatal.
"$VILLA_DIR/vesuvius/.venv/bin/python" tools/inkcheck/scripts/check_env.py \
  || echo "WARNING: check_env.py found a mismatch (see above); results may not reproduce." >&2

echo SETUP_DONE
