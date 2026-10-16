"""All reads of a process share INKCHECK_MAX_CONNECTIONS downloads, however many tiles and threads."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests

from inkcheck import volume
from inkcheck.volume import ArrayMeta, ZarrLevel


def _level(tmp_path, name, workers=16):
    level = ZarrLevel.__new__(ZarrLevel)  # skip __init__: no metadata fetch over the network
    level.url = f"http://example.invalid/{name}/0"
    level.meta = ArrayMeta((32, 32, 32), (8, 8, 8), np.dtype(np.uint8), 0, ".", None)
    level.cache = tmp_path / name
    level.cache.mkdir()
    level.workers = workers
    level._stats = {"chunks_downloaded": 0, "bytes_downloaded": 0, "chunks_from_cache": 0, "chunks_missing": 0}
    level._stats_lock = threading.Lock()
    level._chunk_locks = [threading.Lock() for _ in range(1024)]
    level._codec = None
    return level


class _FakeHTTP:
    """Stands in for requests.Session.get: counts overlapping GETs and sessions used."""

    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.sessions = set()

    def get(self, session, url, timeout=None):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.sessions.add(id(session))
        time.sleep(0.01)
        with self.lock:
            self.active -= 1
        cz, cy, cx = (int(v) for v in url.rsplit("/", 1)[1].split("."))
        r = requests.Response()
        r.status_code = 200
        r._content = bytes([(cz * 16 + cy * 4 + cx) % 256]) * 512
        return r


@pytest.fixture
def fake_http(monkeypatch):
    fake = _FakeHTTP()
    monkeypatch.setattr(requests.Session, "get", lambda self, url, timeout=None: fake.get(self, url, timeout))
    monkeypatch.setattr(volume, "_pool", None)  # a fresh pool sized from the env of this test
    return fake


def test_default_is_eight(monkeypatch):
    monkeypatch.delenv("INKCHECK_MAX_CONNECTIONS", raising=False)
    assert volume.max_connections() == volume.DEFAULT_MAX_CONNECTIONS == 8


def test_env_override_and_bad_values(monkeypatch):
    monkeypatch.setenv("INKCHECK_MAX_CONNECTIONS", "3")
    assert volume.max_connections() == 3
    monkeypatch.setenv("INKCHECK_MAX_CONNECTIONS", "0")
    with pytest.raises(ValueError):
        volume.max_connections()


@pytest.mark.parametrize("cap", [1, 3, 8])
def test_parallel_tiles_never_exceed_the_cap(tmp_path, monkeypatch, fake_http, cap):
    # a render's shape: 6 tiles at once over two volumes, each read on 16 threads -> up to 96 GETs
    monkeypatch.setenv("INKCHECK_MAX_CONNECTIONS", str(cap))
    levels = [_level(tmp_path, f"v{i}") for i in range(2)]

    def tile(i):
        return levels[i % 2].read(0, 32, 0, 32, 8 * (i // 2), 8 * (i // 2) + 8)

    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(tile, range(6)))

    assert fake_http.peak == cap  # saturated, never above
    assert len(fake_http.sessions) <= cap
    assert volume.connection_stats() == {"max_connections": cap, "peak_connections": cap}
    # same bytes as an uncapped read: chunk (cz, cy, cx) is filled with its own id
    for i, o in enumerate(out):
        cx = i // 2
        for cz in range(4):
            for cy in range(4):
                assert (o[8 * cz:8 * cz + 8, 8 * cy:8 * cy + 8] == (cz * 16 + cy * 4 + cx) % 256).all()
    assert sum(lv.stats()["chunks_downloaded"] for lv in levels) == 6 * 16


def test_failed_request_returns_its_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("INKCHECK_MAX_CONNECTIONS", "1")
    monkeypatch.setattr(volume, "_pool", None)

    def boom(self, url, timeout=None):
        raise requests.ConnectionError("reset")

    monkeypatch.setattr(requests.Session, "get", boom)
    for _ in range(3):  # would block forever on the second call if the slot leaked
        with pytest.raises(requests.ConnectionError):
            volume._get("http://example.invalid/x", timeout=1)
