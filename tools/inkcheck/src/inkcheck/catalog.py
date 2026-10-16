"""Resolve scroll volumes and derived data from the open-data catalog (metadata.json)."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import requests

from .volume import OPEN_DATA_HTTP, default_cache_dir

CATALOG_URL = f"{OPEN_DATA_HTTP}/metadata.json"
BUCKET = "s3://vesuvius-challenge-open-data/"

# The volume the PHerc. 343 positive control was measured on. The reference mesh and the letter boxes are in voxels
# of this scan, so it is pinned by id and URL instead of picked from whatever the live catalog lists for PHerc0343.
REFERENCE_SAMPLE = "PHerc0343"
REFERENCE_VOLUME_ID = "20250521140437"
REFERENCE_VOLUME_URL = BUCKET + "PHerc0343/volumes/20250521140437-8.640um-1.2m-116keV-masked.zarr/"
REFERENCE_VOXEL_UM = 8.64


def load_catalog(path: Path | None = None, refresh: bool = False) -> dict:
    path = path or default_cache_dir() / "metadata.json"
    if refresh or not path.exists():
        r = requests.get(CATALOG_URL, timeout=120)
        r.raise_for_status()
        raw = r.content
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return json.loads(path.read_text())


@dataclass
class VolumeInfo:
    sample: str
    volume_id: str
    long_id: str
    url: str
    voxel_um: float
    energy_kev: float | None
    shape: tuple[int, int, int] | None
    surface_prediction_url: str | None


def _first_path(entry: dict) -> str | None:
    for o in entry.get("origins", []):
        if o.get("path"):
            return BUCKET + o["path"]
    return None


def volumes(catalog: dict, sample: str) -> list[VolumeInfo]:
    s = catalog["samples"][sample]
    out = []
    for vid, v in s["volumes"].items():
        props = v.get("properties") or {}
        url = pred = None
        for d in v.get("data", []):
            if d.get("type") == "ome-zarr" and url is None:
                url = _first_path(d)
            elif d.get("type") == "surface-prediction-zarr" and pred is None:
                pred = _first_path(d)
        out.append(
            VolumeInfo(
                sample=sample,
                volume_id=vid,
                long_id=v.get("long_id", vid),
                url=url,
                voxel_um=props.get("pixel_size_um"),
                energy_kev=props.get("energy_keV"),
                shape=tuple(props["shape"]) if props.get("shape") else None,
                surface_prediction_url=pred,
            )
        )
    return out


def find_volume(catalog: dict, sample: str, volume_id: str | None = None) -> VolumeInfo:
    """Pick a volume of a sample by id prefix; without an id, the only/coarsest-resolution one."""
    vols = volumes(catalog, sample)
    if volume_id:
        hits = [v for v in vols if v.volume_id.startswith(volume_id) or v.long_id.startswith(volume_id)]
        if len(hits) != 1:
            raise KeyError(f"{sample}: volume {volume_id!r} matches {[v.volume_id for v in hits]}")
        return hits[0]
    if len(vols) == 1:
        return vols[0]
    return max(vols, key=lambda v: v.voxel_um or 0)


def segments(catalog: dict, sample: str) -> dict:
    return catalog["samples"][sample].get("segments", {})


def reference_volume(catalog: dict | None = None) -> VolumeInfo:
    """The pinned reference volume (PHerc. 343); raises if the catalog no longer maps its id to the same URL and voxel size."""
    vi = find_volume(catalog if catalog is not None else load_catalog(), REFERENCE_SAMPLE, REFERENCE_VOLUME_ID)
    if vi.url != REFERENCE_VOLUME_URL or vi.voxel_um != REFERENCE_VOXEL_UM:
        raise RuntimeError(f"catalog maps {REFERENCE_SAMPLE} {REFERENCE_VOLUME_ID} to {vi.url} ({vi.voxel_um} um), "
                           f"pinned {REFERENCE_VOLUME_URL} ({REFERENCE_VOXEL_UM} um)")
    return vi
