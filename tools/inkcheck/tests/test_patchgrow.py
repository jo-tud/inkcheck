"""patchgrow on a synthetic prediction: it must stay on its sheet and refuse where sheets touch."""

import numpy as np

from inkcheck import patchgrow as pg
from inkcheck.surface import Surface
from inkcheck.volume import from_level, to_level  # prediction voxel centres


class FakePred:
    """Two tilted parallel planes, ``gap`` voxels apart along their normal, as a binary volume."""

    def __init__(self, gap: float, shape=(300, 300, 300)):
        self.n = np.array([0.2, 0.9, 0.3])
        self.n /= np.linalg.norm(self.n)
        self.c = np.array(shape) / 2
        self.gap = gap
        self.shape = shape

    def dist(self, idx):
        return (idx - self.c[:, None, None, None]).transpose(1, 2, 3, 0) @ self.n

    def read(self, z0, z1, y0, y1, x0, x1):
        idx = np.mgrid[z0:z1, y0:y1, x0:x1].astype(float)
        d = self.dist(idx)
        return ((np.abs(d) <= 1.0) | (np.abs(d - self.gap) <= 1.0)).astype(np.uint8)


def test_stays_on_sheet():
    pred = FakePred(gap=15)
    gp = pg.GrowParams(half_nodes=12)
    zyx, st = pg.grow(pred, from_level(pred.c, 4), gp, pred_scale=4)
    assert zyx is not None
    s = Surface(zyx=zyx, step=20.0)
    pts = to_level(zyx[s.valid], 4)
    d = (pts - pred.c) @ pred.n
    assert s.valid.sum() > 0.8 * (2 * 12 + 1) ** 2
    assert np.abs(d).max() < 1.0  # never jumped to the plane 15 voxels away
    assert pg.tears(s) == 0.0


def test_refuses_touching_sheets():
    pred = FakePred(gap=4)
    zyx, st = pg.grow(pred, from_level(pred.c, 4), pg.GrowParams(half_nodes=12), pred_scale=4)
    assert zyx is None or np.isfinite(zyx[..., 0]).sum() <= 1
