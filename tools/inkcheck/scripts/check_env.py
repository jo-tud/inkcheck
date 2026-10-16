"""Print the pinned versions and settings an inkcheck run depends on, and fail if they are off.

Default command of the Docker image; also runs on bare metal after setup.sh. Checks the villa
commit, the ink_9um checkpoint hashes, and that the device resolves the way the GPU rule says.
Exit code 1 on any mismatch.

Usage (from the repo root): python tools/inkcheck/scripts/check_env.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

VILLA_COMMIT = "47111d1a31381a5ff2c05046ed3e311b0dbeec64"  # keep in step with setup.sh
CHECKPOINTS = {
    "data/models/ink_9um/hybrid_3d2d-seed42/step-075000.pth":
        "e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab",
    "data/models/ink_9um/hybrid_3d2d-seed43/step-075000.pth":
        "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f",
}


def villa_commit() -> str:
    """The image records it in VILLA_COMMIT; on bare metal ask the external/villa checkout."""
    if os.environ.get("VILLA_COMMIT"):
        return os.environ["VILLA_COMMIT"]
    try:
        return subprocess.run(["git", "-C", "external/villa", "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    import numpy, requests, scipy, torch
    import inkcheck
    from inkcheck.gpu import resolve_device
    from inkcheck.volume import cache_max_bytes, default_cache_dir

    failures = []
    print("python", sys.version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__,
          "| requests", requests.__version__, "| torch", torch.__version__)
    print("inkcheck", inkcheck.__version__, inkcheck.__file__)

    commit = villa_commit()
    print("villa", commit, "(pinned)" if commit == VILLA_COMMIT else f"!= pinned {VILLA_COMMIT}")
    if commit != VILLA_COMMIT:
        failures.append("villa commit")

    for rel, want in CHECKPOINTS.items():
        p = Path(rel)
        got = sha256(p) if p.exists() else "missing"
        print("checkpoint", rel, "ok" if got == want else f"MISMATCH ({got})")
        if got != want:
            failures.append(rel)

    allowed = os.environ.get("INKCHECK_ALLOW_LOCAL_GPU") == "1"
    print("device", resolve_device().device, "| torch threads", torch.get_num_threads(),
          "| cuda visible to torch:", torch.cuda.device_count(),
          "| INKCHECK_ALLOW_LOCAL_GPU", "1" if allowed else "unset",
          "| CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    if not allowed and torch.cuda.device_count() > 0:
        failures.append("a GPU is visible without INKCHECK_ALLOW_LOCAL_GPU=1")

    print("chunk cache", default_cache_dir(), "| cap", cache_max_bytes() / 2**30, "GiB")

    if failures:
        print("ENV_CHECK_FAILED:", "; ".join(failures))
        return 1
    print("ENV_CHECK_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
