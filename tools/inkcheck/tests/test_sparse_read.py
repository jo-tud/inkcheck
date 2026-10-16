"""The stencil-chunk reader must give bit-identical trilinear samples to the bbox reader."""

import numpy as np
from scipy import ndimage

from inkcheck.volume import ArrayMeta, ZarrLevel, linear_stencil_chunks


class FakeLevel(ZarrLevel):
    """In-memory ZarrLevel: chunk reads come from a numpy array, no network or disk."""

    def __init__(self, data: np.ndarray, chunks):
        self.data = data
        self.meta = ArrayMeta(data.shape, tuple(chunks), data.dtype, 0, ".", None)
        self.workers = 2
        self.fetched: list[tuple[int, int, int]] = []

    def _chunk(self, cz, cy, cx):
        self.fetched.append((cz, cy, cx))
        cs = self.meta.chunks
        out = np.zeros(cs, self.data.dtype)
        src = self.data[cz * cs[0]:(cz + 1) * cs[0], cy * cs[1]:(cy + 1) * cs[1], cx * cs[2]:(cx + 1) * cs[2]]
        out[:src.shape[0], :src.shape[1], :src.shape[2]] = src
        return out


def _sample(level, pts, sparse):
    lo = np.floor(pts.min(0)).astype(int) - 2
    hi = np.ceil(pts.max(0)).astype(int) + 3
    only = linear_stencil_chunks(pts, level.meta.chunks) if sparse else None
    box = level.read(*(v for d in range(3) for v in (lo[d], hi[d])), only=only).astype(np.float32)
    return ndimage.map_coordinates(box, (pts - lo).T, order=1, mode="nearest")


def test_stencil_read_is_bit_identical_and_smaller():
    rng = np.random.default_rng(0)
    data = rng.integers(1, 256, size=(64, 96, 96), dtype=np.uint8)
    level = FakeLevel(data, (16, 16, 16))
    # a curved sheet: a thin shell of points, including exact chunk boundaries and integer coords
    u, v = np.meshgrid(np.linspace(0, np.pi, 120), np.linspace(5, 58, 60), indexing="ij")
    sheet = np.stack([v.ravel(), 48 + 40 * np.cos(u.ravel()), 48 + 40 * np.sin(u.ravel())], 1)
    edges = np.array([[16.0, 31.0, 47.0], [15.999, 32.0, 63.5], [32.0, 48.0, 64.0], [47.5, 15.0, 31.99]])
    pts = np.concatenate([sheet, edges, sheet + rng.normal(0, 1.5, sheet.shape)])

    dense = _sample(level, pts, sparse=False)
    n_dense = len(level.fetched)
    level.fetched.clear()
    sparse = _sample(level, pts, sparse=True)
    n_sparse = len(level.fetched)

    assert np.array_equal(dense.view(np.uint32), sparse.view(np.uint32))
    assert n_sparse < 0.8 * n_dense


def test_stencil_includes_next_chunk_only_across_a_boundary():
    assert linear_stencil_chunks(np.array([[5.0, 5.0, 5.0]]), (16, 16, 16)) == {(0, 0, 0)}
    assert linear_stencil_chunks(np.array([[15.2, 5.0, 5.0]]), (16, 16, 16)) == {(0, 0, 0), (1, 0, 0)}
    assert len(linear_stencil_chunks(np.array([[15.5, 31.5, 47.5]]), (16, 16, 16))) == 8
