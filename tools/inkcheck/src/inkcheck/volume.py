"""Read crops and sample points from remote OME-Zarr scroll volumes.

The open-data volumes are zarr v2 arrays of 128^3 uint8 chunks, mostly uncompressed. Reading
chunks directly over HTTP with a disk cache is simpler and faster than going through a zarr
client, and keeps the tool independent of the zarr version.

The disk cache is shared by every volume and every process (several jobs, one disk) and is
capped at ``INKCHECK_CACHE_MAX_GB`` (default 200 GiB, ``DEFAULT_CACHE_MAX_BYTES``). A cache that can grow without bound
is a disk-full incident waiting to happen, so every new chunk written to disk triggers a throttled
check that evicts the least-recently-used chunks once the cap is exceeded (``evict_lru``).

Downloads are capped per process at ``INKCHECK_MAX_CONNECTIONS`` (default 8,
``DEFAULT_MAX_CONNECTIONS``), however many tiles and reader threads run: every request borrows one
of that many HTTP sessions (one connection each) and waits while all are busy. Cache hits do not
count. A shared home line saturates long before 6 tiles x 16 reader threads = 96 parallel downloads.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import pwd
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage

OPEN_DATA_HTTP = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

DEFAULT_CACHE_MAX_BYTES = 200 * 2**30  # 200 GiB
DEFAULT_MAX_CONNECTIONS = 8  # concurrent downloads per process
_EVICT_TARGET_FRACTION = 0.9  # evict down to 90% of the cap, not right up to the edge (hysteresis)
_EVICT_MIN_INTERVAL_S = 30.0  # throttle: at most one usage scan per process per this many seconds


def to_http(url: str) -> str:
    """Map s3://vesuvius-challenge-open-data/... to its public HTTPS endpoint."""
    if url.startswith("s3://vesuvius-challenge-open-data/"):
        return OPEN_DATA_HTTP + "/" + url[len("s3://vesuvius-challenge-open-data/"):]
    return url


def default_cache_dir() -> Path:
    """``INKCHECK_CACHE``, else ``.cache/inkcheck`` under the account's real home directory.

    Deliberately not ``$HOME``: some job runners give each run a throwaway ``$HOME``, which would
    give every run a private cache (re-downloading everything over a slow line) and multiply the
    byte budget by the number of caches.
    """
    env = os.environ.get("INKCHECK_CACHE")
    if env:
        return Path(env)
    return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".cache" / "inkcheck"


def cache_max_bytes() -> int:
    env = os.environ.get("INKCHECK_CACHE_MAX_GB")
    return int(float(env) * 2**30) if env else DEFAULT_CACHE_MAX_BYTES


def max_connections() -> int:
    env = os.environ.get("INKCHECK_MAX_CONNECTIONS")
    n = int(env) if env else DEFAULT_MAX_CONNECTIONS
    if n < 1:
        raise ValueError(f"INKCHECK_MAX_CONNECTIONS must be >= 1, got {env!r}")
    return n


def cache_usage(root: Path) -> tuple[int, int]:
    """Total bytes and file count of cached chunks under ``root`` (skips the lock file)."""
    if not root.exists():
        return 0, 0
    total = n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.name != ".evict.lock":
            st = _safe_stat(p)
            if st is not None:
                total += st.st_size
                n += 1
    return total, n


def _safe_stat(p: Path):
    try:
        return p.stat()
    except FileNotFoundError:
        return None


def evict_lru(root: Path, max_bytes: int, target_fraction: float = _EVICT_TARGET_FRACTION) -> int:
    """Delete the least-recently-used cached files under ``root`` until usage <= target.

    Best-effort and safe to call from multiple processes: skips in-flight ``.tmp*`` writes, and
    gives up immediately (evicting nothing) if another process already holds the eviction lock.
    Returns the number of bytes evicted.
    """
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".evict.lock", "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0  # another process is already evicting; it'll catch up, nothing to do here

        candidates = [
            p for p in root.rglob("*")
            if p.is_file() and p.name != ".evict.lock" and ".tmp" not in p.name
        ]
        entries = [(p, s) for p, s in ((p, _safe_stat(p)) for p in candidates) if s is not None]
        total = sum(s.st_size for _, s in entries)
        if total <= max_bytes:
            return 0
        target = int(max_bytes * target_fraction)
        entries.sort(key=lambda ps: ps[1].st_mtime)  # oldest-touched first
        evicted = 0
        for p, st in entries:
            if total - evicted <= target:
                break
            try:
                p.unlink()
                evicted += st.st_size
            except FileNotFoundError:
                continue
        print(
            f"[inkcheck cache] {total / 2**30:.1f} GiB > {max_bytes / 2**30:.0f} GiB cap; "
            f"evicted {evicted / 2**30:.2f} GiB, now {(total - evicted) / 2**30:.1f} GiB",
            file=sys.stderr,
        )
        return evicted


_last_evict_check = 0.0
_evict_check_lock = threading.Lock()


def _maybe_evict(root: Path) -> None:
    """Throttled cap check, called after every chunk written to disk (see ``ZarrLevel._chunk``)."""
    global _last_evict_check
    now = time.monotonic()
    with _evict_check_lock:
        if now - _last_evict_check < _EVICT_MIN_INTERVAL_S:
            return
        _last_evict_check = now
    total, _ = cache_usage(root)
    if total > cache_max_bytes():
        evict_lru(root, cache_max_bytes())


class _SessionPool:
    """``n`` HTTP sessions shared by every thread of the process, one connection each.

    A request borrows a session for its whole transfer and blocks while all ``n`` are busy, so at
    most ``n`` downloads run at once and at most ``n`` sockets stay open. ``peak`` is the most
    sessions ever in use at the same time.
    """

    def __init__(self, n: int):
        self.n = n
        self._free: queue.LifoQueue = queue.LifoQueue()
        for _ in range(n):
            self._free.put(None)  # session made on first use
        self._lock = threading.Lock()
        self._in_use = 0
        self.peak = 0

    @contextmanager
    def session(self):
        s = self._free.get()
        if s is None:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=3)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
        with self._lock:
            self._in_use += 1
            self.peak = max(self.peak, self._in_use)
        try:
            yield s
        finally:
            with self._lock:
                self._in_use -= 1
            self._free.put(s)


_pool: _SessionPool | None = None
_pool_lock = threading.Lock()


def _session_pool() -> _SessionPool:
    """The process-wide pool, sized by ``max_connections()`` on first use."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _SessionPool(max_connections())
        return _pool


def connection_stats() -> dict:
    """The per-process download cap and the most downloads that ran at once so far."""
    pool = _session_pool()
    return {"max_connections": pool.n, "peak_connections": pool.peak}


def _get(url: str, timeout: float) -> requests.Response:
    """GET with the body read in full, holding one of the process's ``max_connections()`` slots."""
    with _session_pool().session() as s:
        return s.get(url, timeout=timeout)


def linear_stencil_chunks(pts: np.ndarray, chunks) -> set[tuple[int, int, int]]:
    """Chunk ids holding every voxel a trilinear (``order=1``) sample at ``pts`` reads.

    A trilinear sample at p reads voxels floor(p) and floor(p) + 1 on each axis, so it touches the
    chunk of floor(p) and, where floor(p) + 1 crosses a chunk boundary, the next one. ``pts`` is
    (N, 3) in the array's own voxel coordinates and must be finite. Pass the result as ``only`` to
    ``ZarrLevel.read`` to fetch just these chunks of a bounding box: on a curved sheet the box is
    mostly empty space (1,985 box chunks vs ~1,000 stencil chunks on the PHerc. 343 patch).
    """
    if len(pts) == 0:
        return set()
    pts = np.asarray(pts, dtype=np.float64)
    if not np.abs(pts).max() < 2**31 - 1:
        raise ValueError("sample coordinates must be finite and fit in int32")
    # one contiguous int32 row per axis: ~30x faster than unique() over (N, 3) int64 rows,
    # which made this the slowest step of a warm render
    f = np.floor(pts.T).astype(np.int32, order="C")
    a, nxt = [], []
    for d in range(3):
        c = int(chunks[d])
        a.append(f[d] // c)
        nxt.append(f[d] % c == c - 1)  # floor(p) + 1 lies in the next chunk along this axis
    lo = [int(v.min()) for v in a]
    rel = [v - l for v, l in zip(a, lo)]
    mark = np.zeros([int(v.max()) + 2 for v in rel], dtype=bool)
    mark[rel[0], rel[1], rel[2]] = True
    for o in itertools.product((0, 1), repeat=3):
        if not any(o):
            continue
        sel = np.logical_and.reduce([nxt[d] for d in range(3) if o[d]])
        if sel.any():
            mark[rel[0][sel] + o[0], rel[1][sel] + o[1], rel[2][sel] + o[2]] = True
    return {(int(z) + lo[0], int(y) + lo[1], int(x) + lo[2]) for z, y, x in np.argwhere(mark)}


@dataclass
class ArrayMeta:
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    dtype: np.dtype
    fill_value: int
    separator: str
    compressor: dict | None


class ZarrLevel:
    """One pyramid level of a remote zarr v2 array, read chunk by chunk with a disk cache."""

    def __init__(self, url: str, level: int = 0, cache_dir: Path | None = None, workers: int = 16):
        self.base = to_http(url).rstrip("/")
        self.level = level
        self.url = f"{self.base}/{level}"
        self.meta = self._read_meta()
        key = hashlib.sha1(self.url.encode()).hexdigest()[:16]
        self.cache = (cache_dir or default_cache_dir()) / key
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / "SOURCE").write_text(self.url + "\n")
        self.workers = workers
        # bytes pulled over the network vs served from disk, for egress reporting (``stats()``)
        self._stats = {"chunks_downloaded": 0, "bytes_downloaded": 0, "chunks_from_cache": 0, "chunks_missing": 0}
        self._stats_lock = threading.Lock()
        # Render tiles overlap, so parallel tiles ask for the same edge chunk at the same time.
        # Without this, each would download it (measured: 2,412 downloads for 1,985 chunks).
        # Striped so the lock table stays bounded on a full-volume run.
        self._chunk_locks = [threading.Lock() for _ in range(1024)]
        self._codec = None
        if self.meta.compressor is not None:
            import numcodecs

            self._codec = numcodecs.get_codec(self.meta.compressor)

    def _read_meta(self) -> ArrayMeta:
        r = _get(f"{self.url}/.zarray", timeout=30)
        r.raise_for_status()
        m = r.json()
        if m.get("zarr_format") != 2:
            raise ValueError(f"only zarr v2 is supported, got {m.get('zarr_format')}")
        if m.get("filters"):
            raise ValueError("zarr filters are not supported")
        return ArrayMeta(
            shape=tuple(m["shape"]),
            chunks=tuple(m["chunks"]),
            dtype=np.dtype(m["dtype"]),
            fill_value=m.get("fill_value") or 0,
            separator=m.get("dimension_separator", "."),
            compressor=m.get("compressor"),
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.meta.shape

    def stats(self) -> dict:
        """Counters since this object was created: chunks/bytes downloaded, cache hits, missing."""
        with self._stats_lock:
            return dict(self._stats)

    def _count(self, **inc) -> None:
        with self._stats_lock:
            for k, v in inc.items():
                self._stats[k] += v

    @staticmethod
    def _download(url: str, attempts: int = 6) -> bytes | None:
        """GET one chunk; None if it does not exist. Retries with backoff on dropped connections
        and 5xx: the adapter's own retries do not cover a reset while the body is streaming, and
        one reset used to kill a multi-hour render."""
        for i in range(attempts):
            try:
                r = _get(url, timeout=60)
                if r.status_code in (403, 404):
                    return None
                if r.status_code < 500:
                    r.raise_for_status()
                    return r.content
                err: Exception = requests.HTTPError(f"{r.status_code} for {url}")
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                err = e
            if i == attempts - 1:
                raise err
            time.sleep(min(2 ** i, 30))
        raise AssertionError("unreachable")

    def _chunk(self, cz: int, cy: int, cx: int) -> np.ndarray | None:
        """Return one chunk, or None if it does not exist (all fill value)."""
        with self._chunk_locks[hash((cz, cy, cx)) % len(self._chunk_locks)]:
            raw = self._chunk_bytes(cz, cy, cx)
        if raw is None:
            return None
        if self._codec is not None:
            raw = self._codec.decode(raw)
        return np.frombuffer(raw, dtype=self.meta.dtype).reshape(self.meta.chunks)

    def _chunk_bytes(self, cz: int, cy: int, cx: int) -> bytes | None:
        """Compressed bytes of one chunk from the disk cache or the network; None if missing."""
        sep = self.meta.separator
        path = self.cache / f"{cz}.{cy}.{cx}"
        missing = self.cache / f"{cz}.{cy}.{cx}.missing"
        if missing.exists():
            self._count(chunks_missing=1)
            return None
        if path.exists():
            raw = path.read_bytes()
            path.touch()  # bump mtime: marks this chunk as recently used for LRU eviction
            self._count(chunks_from_cache=1)
        else:
            raw = self._download(f"{self.url}/{cz}{sep}{cy}{sep}{cx}")
            if raw is None:
                missing.touch()
                self._count(chunks_missing=1)
                return None
            self._count(chunks_downloaded=1, bytes_downloaded=len(raw))
            tmp = path.with_suffix(".tmp%d-%d" % (os.getpid(), threading.get_ident()))
            tmp.write_bytes(raw)
            tmp.replace(path)
            _maybe_evict(self.cache.parent)
        return raw

    def read(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
             only: set[tuple[int, int, int]] | None = None) -> np.ndarray:
        """Read the box [z0:z1, y0:y1, x0:x1]; parts outside the array are fill value.

        ``only``: if given, fetch just these chunk ids (e.g. from ``linear_stencil_chunks``) and
        leave the rest of the box at fill value.
        """
        out = np.full((z1 - z0, y1 - y0, x1 - x0), self.meta.fill_value, dtype=self.meta.dtype)
        lo = (z0, y0, x0)
        hi = (z1, y1, x1)
        cs = self.meta.chunks
        shp = self.meta.shape
        ranges = []
        for d in range(3):
            a = max(lo[d], 0)
            b = min(hi[d], shp[d])
            if a >= b:
                return out
            ranges.append(range(a // cs[d], (b - 1) // cs[d] + 1))
        ids = [(cz, cy, cx) for cz in ranges[0] for cy in ranges[1] for cx in ranges[2]]
        if only is not None:
            ids = [cid for cid in ids if cid in only]

        def fetch(cid):
            return cid, self._chunk(*cid)

        with ThreadPoolExecutor(self.workers) as ex:
            for cid, arr in ex.map(fetch, ids):
                if arr is None:
                    continue
                src, dst = [], []
                for d in range(3):
                    c0 = cid[d] * cs[d]
                    a = max(lo[d], c0)
                    b = min(hi[d], c0 + cs[d], shp[d])
                    src.append(slice(a - c0, b - c0))
                    dst.append(slice(a - lo[d], b - lo[d]))
                out[tuple(dst)] = arr[tuple(src)]
        return out


class Volume:
    """A scroll volume (or prediction) at one resolution level, with point sampling."""

    def __init__(self, url: str, level: int = 0, cache_dir: Path | None = None):
        self.array = ZarrLevel(url, level, cache_dir)
        self.level = level
        self.scale = 2 ** level  # level-0 voxels per voxel of this level

    @property
    def shape(self):
        return self.array.shape

    def read_box(self, lo_zyx, hi_zyx, only=None) -> np.ndarray:
        z0, y0, x0 = (int(v) for v in lo_zyx)
        z1, y1, x1 = (int(v) for v in hi_zyx)
        return self.array.read(z0, z1, y0, y1, x0, x1, only=only)

    def sample(self, pts_zyx: np.ndarray, order: int = 1, margin: int = 2) -> np.ndarray:
        """Sample at level-0 (z, y, x) coordinates; returns float32 of shape pts.shape[:-1].

        Reads the bounding box of all points at once, so callers should pass spatially compact
        point sets (one tile at a time).
        """
        pts = np.asarray(pts_zyx, dtype=np.float64) / self.scale
        flat = pts.reshape(-1, 3)
        ok = np.all(np.isfinite(flat), axis=1)
        out = np.full(flat.shape[0], np.nan, dtype=np.float32)
        if not ok.any():
            return out.reshape(pts.shape[:-1])
        lo = np.floor(flat[ok].min(0)).astype(int) - margin
        hi = np.ceil(flat[ok].max(0)).astype(int) + margin + 1
        box = self.read_box(lo, hi).astype(np.float32)
        rel = (flat[ok] - lo).T
        out[ok] = ndimage.map_coordinates(box, rel, order=order, mode="nearest")
        return out.reshape(pts.shape[:-1])


def read_json(url: str) -> dict:
    r = _get(to_http(url), timeout=30)
    r.raise_for_status()
    return r.json()


def cache_json(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    d = read_json(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d))
    return d
