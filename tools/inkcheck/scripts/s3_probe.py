"""Measure how fast this machine reads scroll chunks from the public S3 bucket. Stdlib only.

Run it first on any new machine (Kaggle cell, Modal function, rented pod) before paying for a
long job, so its numbers compare directly with this repo's baseline: same volume, same chunks.

    python s3_probe.py [--streams 1 16 64] [--chunks 16]

Reads `--chunks` uncompressed 2 MiB chunks (a 128 x 512 x 512 voxel slab of PHerc0343, level 0)
once per stream count, each pass on a different z-slab so nothing is served warm from a nearby
cache. Prints per-request latency (connect, TLS, first byte) and throughput as JSON.
Bytes pulled: chunks x 2 MiB x passes, ~96 MiB with the defaults. The bucket is in the AWS Open
Data program, so the egress is paid by AWS.
"""

from __future__ import annotations

import argparse
import http.client
import json
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor

HOST = "vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
VOL = "/PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/0"
Z0, Y0, X0 = 80, 30, 30  # chunk indices of the slab corner; chunks exist here (checked 2026-10-12)


def timed_get(path: str, timeout: float = 30.0) -> dict:
    """One GET on a fresh connection, with the phases timed separately."""
    t0 = time.perf_counter()
    raw = socket.create_connection((HOST, 443), timeout=timeout)
    t_conn = time.perf_counter()
    sock = ssl.create_default_context().wrap_socket(raw, server_hostname=HOST)
    t_tls = time.perf_counter()
    conn = http.client.HTTPSConnection(HOST, timeout=timeout)
    conn.sock = sock
    conn.request("GET", path)
    resp = conn.getresponse()
    t_first = time.perf_counter()
    body = resp.read()
    t_end = time.perf_counter()
    conn.close()
    return {"status": resp.status, "bytes": len(body), "connect": t_conn - t0, "tls": t_tls - t0,
            "ttfb": t_first - t0, "total": t_end - t0}


def run_pass(streams: int, n: int, z: int) -> dict:
    side = max(1, int(n ** 0.5))
    paths = [f"{VOL}/{z}/{Y0 + i // side}/{X0 + i % side}" for i in range(n)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(streams) as ex:
        res = list(ex.map(timed_get, paths))
    wall = time.perf_counter() - t0
    ok = [r for r in res if r["status"] == 200]
    nbytes = sum(r["bytes"] for r in ok)

    def med(k):
        v = sorted(r[k] for r in ok)
        return round(v[len(v) // 2] * 1000, 1) if v else None

    return {"streams": streams, "chunks_ok": len(ok), "chunks_missing": n - len(ok), "MiB": round(nbytes / 2**20, 1),
            "wall_s": round(wall, 2), "MB_per_s": round(nbytes / wall / 1e6, 2),
            "median_ms": {k: med(k) for k in ("connect", "tls", "ttfb", "total")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--chunks", type=int, default=16)
    args = ap.parse_args()
    out = {"host": socket.gethostname(), "when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "passes": [run_pass(s, args.chunks, Z0 + i) for i, s in enumerate(args.streams)]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
