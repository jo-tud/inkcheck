"""Explicit compute choice: CPU thread cap by default, GPU device and VRAM cap when allowed.

CPU is the default and the normal path; a local GPU is used only with
``INKCHECK_ALLOW_LOCAL_GPU=1``. Machines are often shared, so ``resolve_device`` always caps CPU
threads too -- a job that quietly takes every core is as much a collision as two jobs on one
GPU. Defaults come from the environment (``INKCHECK_DEVICE``, ``INKCHECK_VRAM_GB``,
``INKCHECK_CPU_THREADS``) so each user can fix a budget once in their shell profile instead
of threading flags through every script. Everything here prints what it picked -- a silent
default is how two jobs end up fighting over the same resource.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

#: Default device: CPU. Asking for CUDA raises unless ``INKCHECK_ALLOW_LOCAL_GPU=1``.
DEFAULT_DEVICE = "cpu"

GiB = 2**30


def cap_cpu_threads(n: int | None = None) -> int:
    """Cap this process to ``n`` CPU threads (default ``INKCHECK_CPU_THREADS``, else half the
    machine's cores) so two jobs sharing one machine don't oversubscribe it between them.

    Applies to torch (``set_num_threads``) and, via ``threadpoolctl``, to the BLAS/OpenMP pools
    numpy and scipy use underneath -- those read ``OMP_NUM_THREADS`` only once, at first use, so
    setting the env var after numpy is already imported (as it is by the time any inkcheck code
    runs) would silently do nothing. ``threadpoolctl`` changes the live limit instead.
    """
    if n is None:
        env = os.environ.get("INKCHECK_CPU_THREADS")
        n = int(env) if env else max(1, (os.cpu_count() or 2) // 2)
    import threadpoolctl
    import torch

    torch.set_num_threads(n)
    threadpoolctl.threadpool_limits(n)
    print(f"[inkcheck cpu] capped to {n} threads (of {os.cpu_count()} cores)", file=sys.stderr)
    return n


@dataclass
class GpuChoice:
    device: object  # torch.device, kept untyped so importing this module doesn't require torch
    index: int | None
    vram_gb: float | None
    free_gb: float | None
    total_gb: float | None


def resolve_device(device: str | None = None, vram_gb: float | None = None, label: str = "inkcheck",
                    cache_dir: Path | None = None, cpu_threads: int | None = None) -> GpuChoice:
    """Pick a device and cap this process's compute use, whichever device it ends up on.

    ``device`` defaults to ``INKCHECK_DEVICE`` (env) then ``DEFAULT_DEVICE`` (cpu). ``vram_gb``
    defaults to ``INKCHECK_VRAM_GB`` and only applies on CUDA, which also requires
    ``INKCHECK_ALLOW_LOCAL_GPU=1``. ``cpu_threads`` (default
    ``INKCHECK_CPU_THREADS``, see ``cap_cpu_threads``) is applied unconditionally -- CPU is the
    normal path here, and the process still runs plenty of numpy/scipy/torch CPU code even when
    a GPU is in play. Raises if the requested VRAM exceeds what is free on the card right now,
    rather than letting the process start and OOM (or starve) later. Also drops an advisory GPU
    reservation file under the cache dir so a second process on the same card can see who else is
    using it (``gpu_status``) -- advisory, not a lock: the whole point of the VRAM cap is that two
    jobs *can* share a card.
    """
    import torch

    cap_cpu_threads(cpu_threads)

    device = device or os.environ.get("INKCHECK_DEVICE", DEFAULT_DEVICE)
    if vram_gb is None:
        env = os.environ.get("INKCHECK_VRAM_GB")
        vram_gb = float(env) if env else None

    if device != "cpu" and os.environ.get("INKCHECK_ALLOW_LOCAL_GPU") != "1":
        raise RuntimeError(
            f"local GPU use is off by default (requested {device!r}). Run on CPU, or set "
            f"INKCHECK_ALLOW_LOCAL_GPU=1 to use a local GPU."
        )

    if device != "cpu" and not torch.cuda.is_available():
        print(f"[inkcheck gpu] CUDA not available, falling back to cpu (requested {device!r})", file=sys.stderr)
        device = "cpu"

    dev = torch.device(device)
    if dev.type != "cuda":
        print("[inkcheck gpu] using cpu", file=sys.stderr)
        return GpuChoice(dev, None, None, None, None)

    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    free, total = torch.cuda.mem_get_info(idx)
    free_gb, total_gb = free / GiB, total / GiB

    if vram_gb is not None:
        if vram_gb > free_gb:
            raise RuntimeError(
                f"requested {vram_gb:.1f} GB on cuda:{idx} ({name}) but only {free_gb:.1f} GB is "
                f"free right now. Pick another device (INKCHECK_DEVICE) or a lower cap "
                f"(INKCHECK_VRAM_GB); run `python -m inkcheck.gpu` to see current usage."
            )
        torch.cuda.set_per_process_memory_fraction(min(vram_gb * GiB / total, 1.0), idx)

    print(
        f"[inkcheck gpu] using cuda:{idx} ({name}): {free_gb:.1f}/{total_gb:.1f} GB free"
        + (f", capped to {vram_gb:.1f} GB for this process" if vram_gb is not None else ", no cap set"),
        file=sys.stderr,
    )
    _write_reservation(cache_dir, idx, label, vram_gb)
    return GpuChoice(dev, idx, vram_gb, free_gb, total_gb)


def _write_reservation(cache_dir: Path | None, idx: int, label: str, vram_gb: float | None) -> None:
    from .volume import default_cache_dir

    root = (cache_dir or default_cache_dir()) / "gpu-status"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"cuda{idx}.pid{os.getpid()}.json"
    import json

    path.write_text(json.dumps({
        "pid": os.getpid(), "label": label, "device": idx,
        "vram_gb": vram_gb, "started": time.time(),
    }))


def gpu_status(cache_dir: Path | None = None) -> list[dict]:
    """Live reservations written by ``resolve_device`` across all processes, pruned of dead pids."""
    from .volume import default_cache_dir

    root = (cache_dir or default_cache_dir()) / "gpu-status"
    out = []
    if not root.exists():
        return out
    import json

    for p in sorted(root.glob("cuda*.pid*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        pid = rec.get("pid")
        try:
            os.kill(pid, 0)
        except (OSError, TypeError):
            p.unlink(missing_ok=True)
            continue
        out.append(rec)
    return out


def _main() -> None:
    import torch

    default_threads = int(os.environ.get("INKCHECK_CPU_THREADS") or max(1, (os.cpu_count() or 2) // 2))
    print(f"cpu: {default_threads} threads (of {os.cpu_count()}) by default; "
          f"INKCHECK_CPU_THREADS to override. (status only -- does not apply the cap)")
    recs = gpu_status()
    for idx in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(idx)
        name = torch.cuda.get_device_name(idx)
        print(f"cuda:{idx} {name}: {free / GiB:.1f}/{total / GiB:.1f} GB free")
        for r in recs:
            if r["device"] == idx:
                cap = f"{r['vram_gb']:.1f} GB" if r.get("vram_gb") else "no cap"
                print(f"  pid {r['pid']} ({r['label']}): {cap}, started {time.ctime(r['started'])}")


if __name__ == "__main__":
    _main()
