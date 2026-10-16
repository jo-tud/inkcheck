"""The ``inkcheck`` command: environment report and volume lookup.

    inkcheck info [--json]                     version, device rule, CPU threads, chunk cache (no network)
    inkcheck volume [SAMPLE] [--volume-id ID]  resolve a volume from the open-data catalog (default: the pinned PHerc. 343 volume)

The PHerc. 343 reproduction itself is `bash tools/inkcheck/quickstart.sh` (see README); it needs the repository layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

from . import __version__


def info() -> dict:
    from .catalog import REFERENCE_VOLUME_ID, REFERENCE_VOLUME_URL
    from .gpu import DEFAULT_DEVICE
    from .volume import cache_max_bytes, default_cache_dir

    threads = os.environ.get("INKCHECK_CPU_THREADS")
    return {
        "version": __version__,
        "module": os.path.dirname(os.path.abspath(__file__)),
        "python": sys.version.split()[0],
        "device": os.environ.get("INKCHECK_DEVICE", DEFAULT_DEVICE),
        "local_gpu_allowed": os.environ.get("INKCHECK_ALLOW_LOCAL_GPU") == "1",
        "cpu_threads": int(threads) if threads else max(1, (os.cpu_count() or 2) // 2),
        "cache_dir": str(default_cache_dir()),
        "cache_max_gb": cache_max_bytes() / 2**30,
        "reference_volume_id": REFERENCE_VOLUME_ID,
        "reference_volume_url": REFERENCE_VOLUME_URL,
    }


def volume(sample: str | None, volume_id: str | None) -> dict:
    from .catalog import find_volume, load_catalog, reference_volume

    vi = reference_volume() if sample is None and volume_id is None else find_volume(load_catalog(), sample or "PHerc0343", volume_id)
    return asdict(vi)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="inkcheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"inkcheck {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="version, device rule, CPU threads and chunk cache; no network")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("volume", help="resolve a volume from the open-data catalog (fetched once, then cached)")
    p.add_argument("sample", nargs="?", help="e.g. PHerc0343; omit for the pinned PHerc. 343 volume")
    p.add_argument("--volume-id", help="volume id prefix; required when the sample has several volumes")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        out = info()
        if args.json:
            print(json.dumps(out, indent=1))
        else:
            for k, v in out.items():
                print(f"{k:18} {v}")
        return 0
    print(json.dumps(volume(args.sample, args.volume_id), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
