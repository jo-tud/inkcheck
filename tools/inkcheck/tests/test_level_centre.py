"""Level L > 0 is read at voxel centres.

Organiser level L voxel k is the mean of level-0 voxels 2^L k ... 2^L k + 2^L - 1, so its
centre is level-0 2^L k + (2^L - 1)/2. On a linear field the block mean equals the field at the
block centre and trilinear sampling is exact, so every level must give back f(p) at level-0 p;
the old corner rule (p / 2^L) gives f(p + (2^L - 1)/2).
"""

import numpy as np
import pytest
from scipy import ndimage

from inkcheck import render, sheetcert
from inkcheck.surface import Surface
from inkcheck.volume import ArrayMeta, Volume, ZarrLevel, from_level, to_level

GRAD = np.array([0.5, 0.3, 0.2])  # field per level-0 voxel along z, y, x
N = 128


class FakeLevel(ZarrLevel):
    def __init__(self, data):
        self.data = data
        self.meta = ArrayMeta(data.shape, (32, 32, 32), data.dtype, 0, ".", None)
        self.workers = 2

    def _chunk(self, cz, cy, cx):
        out = np.zeros((32, 32, 32), self.data.dtype)
        s = self.data[cz * 32:(cz + 1) * 32, cy * 32:(cy + 1) * 32, cx * 32:(cx + 1) * 32]
        out[:s.shape[0], :s.shape[1], :s.shape[2]] = s
        return out


def field(p):
    return 10.0 + np.asarray(p, float) @ GRAD


def level_volume(level):
    a = field(np.indices((N, N, N), dtype=np.float64).transpose(1, 2, 3, 0))
    for _ in range(level):  # 2x2x2 block mean, as the organiser pyramid (no rounding: float data)
        z, y, x = a.shape
        a = a.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean((1, 3, 5))
    v = Volume.__new__(Volume)
    v.array, v.level, v.scale = FakeLevel(a.astype(np.float64)), level, 2 ** level
    return v


def plane(step=4.0, K=4):
    n = np.array([1.0, 2.0, 2.0]) / 3
    u = np.cross(n, [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    i, j = np.meshgrid(np.arange(-K, K + 1), np.arange(-K, K + 1), indexing="ij")
    zyx = np.array([64.3, 63.8, 64.55]) + (i * step)[..., None] * u + (j * step)[..., None] * v
    return Surface(zyx=zyx, step=step), n


@pytest.mark.parametrize("level", [1, 2])
def test_sample_reads_voxel_centres(level):
    pts = np.random.default_rng(level).uniform(30, 98, size=(500, 3))
    got = level_volume(level).sample(pts)
    np.testing.assert_allclose(got, field(pts), atol=1e-4)  # corner rule: off by 0.5 (L1), 1.5 (L2)


@pytest.mark.parametrize("level", [1, 2])
def test_render_reads_voxel_centres(level):
    surf, n = plane()
    nn = np.broadcast_to(n, surf.zyx.shape).copy()
    offs = np.arange(-8, 8.01, 1.0)
    lay, ok = render.render_region(surf, nn, level_volume(level), 0, 33, 0, 33, offs)
    p, _ = render.pixel_geometry(surf, nn, 0, 33, 0, 33)
    want = field(p[None] + offs[:, None, None, None] * n)
    np.testing.assert_allclose(lay[:, ok], want[:, ok], atol=1e-3)


def test_sheetcert_profiles_read_voxel_centres():
    level = 2
    surf, n = plane()
    sp = sheetcert.SampleParams(long_radius=6, window=4, smooth=1)
    prof = sheetcert.sample_profiles(surf, level_volume(level), sp=sp, workers=1)
    s = 2 ** level
    rc = prof["rc"]
    off = np.arange(-6, 7) * s  # level-0 offsets along the level-scaled normal
    want = field(surf.zyx[rc[:, 0], rc[:, 1]][:, None, :] + off[None, :, None] * n)  # plane() normals are +n
    np.testing.assert_allclose(prof["normal_s1"], want, atol=1e-3)


def test_level0_is_plain_trilinear():
    """Level 0 is unchanged: Volume.sample = map_coordinates on the array itself, bit for bit."""
    rng = np.random.default_rng(0)
    data = rng.integers(1, 256, size=(64, 64, 64), dtype=np.uint8)
    v = Volume.__new__(Volume)
    v.array, v.level, v.scale = FakeLevel(data), 0, 1
    pts = rng.uniform(4, 60, size=(300, 3))
    lo = np.floor(pts.min(0)).astype(int) - 2
    hi = np.ceil(pts.max(0)).astype(int) + 3
    box = data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    want = ndimage.map_coordinates(box, (pts - lo).T, order=1, mode="nearest")
    assert v.sample(pts).tobytes() == want.astype(np.float32).tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_scale1_helpers_equal_old_expressions(dtype):
    """At level 0 the helpers return exactly the old p / 1 and p * 1, dtype included."""
    p = (np.random.default_rng(1).uniform(0, 500, size=(50, 3)) * 1).astype(dtype)
    for got, old in ((to_level(p, 1), p / 1), (from_level(p, 1), p * 1)):
        assert got.dtype == old.dtype and got.tobytes() == old.tobytes()
