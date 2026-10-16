"""Parallel reads of the same chunk must download it once (render tiles overlap at their edges)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from inkcheck.volume import ArrayMeta, ZarrLevel


def _level(tmp_path):
    level = ZarrLevel.__new__(ZarrLevel)  # skip __init__: no metadata fetch over the network
    level.url = "http://example.invalid/0"
    level.meta = ArrayMeta((8, 8, 8), (8, 8, 8), np.dtype(np.uint8), 0, ".", None)
    level.cache = tmp_path
    level.workers = 4
    level._stats = {"chunks_downloaded": 0, "bytes_downloaded": 0, "chunks_from_cache": 0, "chunks_missing": 0}
    level._stats_lock = threading.Lock()
    level._chunk_locks = [threading.Lock() for _ in range(1024)]
    level._codec = None
    return level


def test_concurrent_requests_download_a_chunk_once(tmp_path, monkeypatch):
    calls = []

    def slow_download(url, attempts=6):
        calls.append(url)
        time.sleep(0.2)  # long enough that all requests overlap
        return bytes(range(256)) * 2

    monkeypatch.setattr(ZarrLevel, "_download", staticmethod(slow_download))
    level = _level(tmp_path)
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda _: level._chunk(0, 0, 0), range(8)))
    assert len(calls) == 1
    assert all(np.array_equal(o, out[0]) for o in out)
    st = level.stats()
    assert st["chunks_downloaded"] == 1 and st["chunks_from_cache"] == 7
