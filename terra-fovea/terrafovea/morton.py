"""Z-order (Morton) indexing.

Cells inside a tile are stored in Z-order.  Two properties make the whole
variable-resolution engine exact and cheap:

  1. The Z-index of a cell at level L is the finest Z-index shifted right by 2L bits.
  2. The four children of any parent are four consecutive entries.

So coarsening a tile is `array.reshape(-1, 4)` followed by a reduction, and
refining is `np.repeat(array, 4)`.  No resampling, no interpolation, no overlap:
alignment errors are impossible by construction.
"""
import numpy as np


def interleave(ix: np.ndarray, iy: np.ndarray, bits: int = 8) -> np.ndarray:
    ix = ix.astype(np.int64); iy = iy.astype(np.int64)
    out = np.zeros_like(ix)
    for b in range(bits):
        out |= ((ix >> b) & 1) << (2 * b)
        out |= ((iy >> b) & 1) << (2 * b + 1)
    return out


def deinterleave(m: np.ndarray, bits: int = 8):
    m = m.astype(np.int64)
    ix = np.zeros_like(m); iy = np.zeros_like(m)
    for b in range(bits):
        ix |= ((m >> (2 * b)) & 1) << b
        iy |= ((m >> (2 * b + 1)) & 1) << b
    return ix, iy


def tables(n: int):
    """LUT[iy, ix] -> Z-index for an n x n tile, plus the inverse (ix, iy) per Z-index."""
    iy, ix = np.mgrid[0:n, 0:n]
    lut = interleave(ix, iy).astype(np.int32)
    inv_x, inv_y = deinterleave(np.arange(n * n))
    return lut, inv_x.astype(np.int32), inv_y.astype(np.int32)
