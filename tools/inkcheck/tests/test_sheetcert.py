"""sheetcert on synthetic layered CT: passes on a sheet, fails in the gap, along the sheet, and on
two fused sheets."""

import numpy as np

from inkcheck import sheetcert as sc
from inkcheck.surface import Surface


class FakeCT:
    """Planes normal to ``n``: sheets of ``thick`` voxels every ``period`` voxels, fibre texture."""

    scale = 1

    def __init__(self, period=30.0, thick=10.0, fused_at=None, seed=0):
        self.n = np.array([0.15, 0.95, 0.25])
        self.n /= np.linalg.norm(self.n)
        self.period, self.thick, self.fused_at = period, thick, fused_at
        self.rng = np.random.default_rng(seed)

    def read_box(self, lo, hi):
        idx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(float)
        d = np.tensordot(self.n, idx, axes=1)
        ph = np.mod(d, self.period)
        sheet = np.minimum(ph, self.period - ph) <= self.thick / 2
        if self.fused_at is not None:  # fill the gap next to the sheet at d ~ fused_at
            sheet |= np.abs(d - (self.fused_at + self.period / 2)) <= self.period / 2
        v = np.where(sheet, 150.0, 50.0)
        v += 25 * np.sin(idx[2] * 0.9) * sheet  # fibre-like texture inside the sheet
        v += self.rng.normal(0, 6, v.shape)
        return np.clip(v, 1, 255).astype(np.uint8)


def flat_surface(ct, d0, n=12, step=6.0):
    t = np.cross(ct.n, [1.0, 0, 0])
    t /= np.linalg.norm(t)
    b = np.cross(ct.n, t)
    c = np.array([200.0, 200.0, 200.0])
    c = c + (d0 - c @ ct.n) * ct.n
    i, j = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2, indexing="ij")
    zyx = c + step * (i[..., None] * t + j[..., None] * b)
    return Surface(zyx=zyx, step=step)


def run(ct, d0, bp=None):
    surf = flat_surface(ct, d0)
    prof = sc.sample_profiles(surf, ct, tile_nodes=12, workers=1)
    return sc.summarise(sc.evaluate(prof, bp or sc.BandParams(), voxel_um=4.8))


def test_on_sheet_passes_and_nulls_fail():
    ct = FakeCT()
    d0 = 300.0  # a sheet centre (multiple of the period)
    s = run(ct, d0)
    assert s["F"] > 0.9
    assert s["T"] < 0.1  # along the sheet there are no gaps
    assert s["H"] < 0.1  # half a spacing off lands in the gap
    assert abs(s["spacing_um_median_vox"] - 30) <= 1
    assert s["full_spacing"] > 0.9  # wrap-blind by design
    assert sc.certificate(s, bar=0.5)


def test_in_gap_fails():
    ct = FakeCT()
    assert run(ct, 315.0)["F"] < 0.1


def test_fused_sheets_fail_width():
    ct = FakeCT(period=30.0, thick=10.0, fused_at=300.0)
    s = run(ct, 300.0, sc.BandParams(w_max_um=90.0))  # fused band ~ 40 voxels = 192 um
    assert s["F"] < 0.1
