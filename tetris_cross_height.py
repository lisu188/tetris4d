"""Certify that a CROSS landing existed throughout a recorded interval."""
from functools import lru_cache
from operator import index

import numpy as np

import sparse_tetris_4d_optimized as game


HEIGHT = game.HEIGHT
HEIGHTS = np.fromiter((i.bit_length() for i in range(1 << HEIGHT)), np.uint8)
_CELL_COUNTS = np.fromiter((i.bit_count() for i in range(1 << HEIGHT)), np.uint8)


def _board(columns):
    packed = np.asarray(columns)
    if packed.shape != (game.AREA,) or packed.dtype.kind not in 'iu':
        raise ValueError('Expected one integer mask per column')
    if packed.dtype != np.uint16 and (packed.min() < 0 or packed.max() >= 1 << HEIGHT):
        raise ValueError('Column masks must fit the board height')
    return packed.astype(np.uint16, copy=False)


@lru_cache(maxsize=256)
def _before_clear(mask):
    """Map a post-clear upper height to a pre-clear upper height."""
    remaining = [z + 1 for z in range(HEIGHT) if not (mask & (1 << z))]
    # Every column occupied every deleted layer. Surviving cells keep their order.
    levels = [max(mask.bit_length(), h) for h in [0, *remaining]]
    return np.array(levels + [HEIGHT] * mask.bit_count(), dtype=np.uint8)


def _upper(end_heights, masks):
    bounds = np.arange(HEIGHT + 1, dtype=np.uint8)
    for mask in reversed(masks):
        bounds = _before_clear(mask)[bounds]
    return bounds[end_heights]


def _has_cross(heights):
    grid = heights.reshape(game.FOURTH, game.DEPTH, game.WIDTH)
    for shape in game.SHAPES['CROSS']:
        sx, sy, sz, sw = shape.extents
        nw, ny, nx = game.FOURTH - sw + 1, game.DEPTH - sy + 1, game.WIDTH - sx + 1
        fits = np.ones((nw, ny, nx), dtype=bool)
        for offset, _, bottom in shape.columns:
            w, rem = divmod(offset, game.WIDTH * game.DEPTH)
            y, x = divmod(rem, game.WIDTH)
            fits &= grid[w:w+nw, y:y+ny, x:x+nx] <= HEIGHT - sz + bottom
        if fits.any():
            return True
    return False


def _compact(columns, level):
    low = (1 << level) - 1
    return (columns & low) | ((columns >> (level + 1)) << level)


def _compact_mask(columns, mask):
    for level in reversed(range(mask.bit_length())):
        if mask & (1 << level):
            columns = _compact(columns, level)
    return columns


def cross_survives_interval(start_board, end_board, cleared_layers, *, clear_masks=None, added_cells=None):
    """True proves positive CROSS mobility on every board, including both endpoints.

    Supply the number of cleared layers, and optionally the chronological, nonzero
    precompaction bitmasks from the clearing placements. Without masks, handle zero
    or one cleared layer. If supplied, added_cells must be the exact increase in the
    game's cumulative cells_placed counter. False means the observations do not certify.
    """
    start, end = _board(start_board), _board(end_board)
    count = index(cleared_layers)
    if count < 0:
        raise ValueError('Cleared-layer count must be nonnegative')
    additions = int(_CELL_COUNTS[end].sum()) - int(_CELL_COUNTS[start].sum()) + count * game.AREA
    if additions < 0 or (added_cells is not None and index(added_cells) != additions):
        return False

    end_heights = HEIGHTS[end]
    if clear_masks is not None:
        masks = tuple(index(m) for m in clear_masks)
        if any(not 0 < m < 1 << HEIGHT for m in masks) or sum(m.bit_count() for m in masks) != count:
            raise ValueError('Clear masks must be nonzero and match the number of cleared layers')
        survivors = start
        for mask in masks:
            survivors = _compact_mask(survivors, mask)
        return not (survivors & ~end).any() and _has_cross(_upper(end_heights, masks))
    if count == 0:
        return not (start & ~end).any() and _has_cross(end_heights)
    if count != 1:
        return False

    # At the only clear all initially empty cells in that layer had to be filled.
    # Every initial cell outside the layer must survive at its compacted position.
    feasible = False
    for z in range(HEIGHT):
        bit = 1 << z
        missing = game.AREA - np.count_nonzero(start & bit)
        if missing > additions or (_compact(start, z) & ~end).any():
            continue
        feasible = True
        if not _has_cross(_before_clear(bit)[end_heights]):
            return False
    return feasible
