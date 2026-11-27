"""Find a two-drop layer clear that reopens a physical CROSS landing."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import chain

import sparse_tetris_4d_optimized as game


@lru_cache(maxsize=None)
def _slices(piece):
    return [(rotation, dz, tuple((x, y, w) for x, y, z, w in shape.cells if z == dz))
            for rotation, shape in enumerate(game.SHAPES[piece]) for dz, _ in shape.levels]


def _in_bounds(shape, x, y, w):
    sx, sy, _, sw = shape.extents
    return 0 <= x <= game.WIDTH - sx and 0 <= y <= game.DEPTH - sy and 0 <= w <= game.FOURTH - sw


def _landing(heights, shape, base):
    return max(heights[base + offset] - bottom for offset, _, bottom in shape.columns)


def _aligned(piece, level, holes, heights, min_cells, *, first):
    """Translate a slice into original holes. Heights can only rise before the first clear."""
    hole_xyz = [game.column_coordinates(i) for i in holes]
    for rotation, dz, cells in _slices(piece):
        shape = game.SHAPES[piece][rotation]
        z = level - dz
        if not min_cells <= len(cells) <= len(holes) or not 0 <= z <= game.HEIGHT - shape.extents[2]:
            continue
        ax, ay, aw = cells[0]
        for hx, hy, hw in hole_xyz:
            x, y, w = hx - ax, hy - ay, hw - aw
            if not _in_bounds(shape, x, y, w):
                continue
            covered = frozenset(game.column_index(x + dx, y + dy, w + dw) for dx, dy, dw in cells)
            if covered <= holes:
                height = _landing(heights, shape, game.column_index(x, y, w))
                if height == z or (not first and height < z):
                    yield covered, (rotation, x, y, w), z, height


def _supports(current, visible, reply, reply_z, heights):
    """First drops whose column tops can supply missing support for a particular reply."""
    rotation, x, y, w = reply
    base = game.column_index(x, y, w)
    for offset, _, bottom in game.SHAPES[visible][rotation].columns:
        sx, sy, sw = game.column_coordinates(base + offset)
        needed_height = reply_z + bottom
        for r, shape in enumerate(game.SHAPES[current]):
            for col, bits, _ in shape.columns:
                dx, dy, dw = game.column_coordinates(col)
                fx, fy, fw = sx - dx, sy - dy, sw - dw
                z = needed_height - bits.bit_length()
                if 0 <= z <= game.HEIGHT - shape.extents[2] and _in_bounds(shape, fx, fy, fw):
                    if _landing(heights, shape, game.column_index(fx, fy, fw)) == z:
                        yield r, fx, fy, fw


def _search_layers(board, current, visible, context):
    maxima = [max(len(cells) for _, _, cells in _slices(piece)) for piece in (current, visible)]
    layers, first_specs = [], set()
    scan_rest = False
    for level, count in enumerate(context.fill_counts):
        missing = game.AREA - count
        if not 0 < missing <= sum(maxima):
            continue
        holes = frozenset(i for i, col in enumerate(board) if not col & (1 << level))
        if any(context.heights[i] > level for i in holes):
            continue
        replies = defaultdict(list)
        for covered, spec, z, height in _aligned(visible, level, holes, context.heights,
                                                  max(1, missing - maxima[0]), first=False):
            replies[covered].append((spec, z))
            if covered == holes:
                if height == z:
                    # A first drop away from every hole and every support may still leave the reply legal.
                    scan_rest = True
                else:
                    first_specs.update(_supports(current, visible, spec, z, context.heights))
        if not replies:
            continue
        for covered, spec, _, _ in _aligned(current, level, holes, context.heights,
                                             max(1, missing - maxima[1]), first=True):
            if holes - covered in replies:
                first_specs.add(spec)
        for options in replies.values():
            options.sort(key=lambda entry: _engine_order(entry[0]))
        layers.append((level, holes, replies))
    return layers, first_specs, scan_rest


def _engine_order(spec):
    rotation, x, y, w = spec
    return rotation, w, y, x


def _try_first(board, current, visible, context, layers, spec):
    rotation, x, y, w = spec
    shape = game.SHAPES[current][rotation]
    base = game.column_index(x, y, w)
    z = _landing(context.heights, shape, base)
    if z + shape.extents[2] > game.HEIGHT:
        return None
    if any(context.fill_counts[z + dz] + n == game.AREA for dz, n in shape.levels):
        return None
    raised = {base + offset: z + bits.bit_length() for offset, bits, _ in shape.columns}
    first = None
    for level, holes, replies in layers:
        fill = frozenset(game.column_index(x + dx, y + dy, w + dw)
                         for dx, dy, dz, dw in shape.cells if z + dz == level)
        for second_spec, wanted_z in replies.get(holes - fill, ()):
            rr, rx, ry, rw = second_spec
            second_base = game.column_index(rx, ry, rw)
            # Original support never decreases. Only the first shape's columns can change it.
            actual_z = max(raised.get(second_base + col, context.heights[second_base + col]) - bottom
                           for col, _, bottom in game.SHAPES[visible][rr].columns)
            if actual_z != wanted_z:
                continue
            if first is None:
                first = game.make_move(board, current, *spec)
            second = game.make_move(first.board, visible, *second_spec)
            if second is None or second.layers == 0 or second.candidate.z != wanted_z:
                raise AssertionError('Candidate geometry disagrees with the rigid-drop engine')
            if game.has_legal_move(second.board, 'CROSS'):
                return first, second
    return None


def find_recovery(board, current, visible, preferred=()):
    """Find a non-clearing current drop followed by a visible clear reopening CROSS.

    Preferred first placements are (rotation, x, y, w) tuples in policy order.
    When they fail, search the other geometrically possible first placements.
    None means that this two-drop pattern is absent or CROSS already fits.
    """
    if current == 'CROSS' or visible == 'CROSS' or game.has_legal_move(board, 'CROSS'):
        return None
    context = game.evaluate_board(board)
    layers, first_specs, scan_rest = _search_layers(board, current, visible, context)
    if not layers:
        return None
    seen = set()
    for spec in chain(preferred, sorted(first_specs, key=_engine_order)):
        spec = tuple(spec)
        if spec in seen:
            continue
        rotation, x, y, w = spec
        if not 0 <= rotation < len(game.SHAPES[current]) or not _in_bounds(game.SHAPES[current][rotation], x, y, w):
            raise ValueError('Invalid preferred first placement')
        seen.add(spec)
        pair = _try_first(board, current, visible, context, layers, spec)
        if pair:
            return pair
    if scan_rest:
        for spec in game.specifications(current, 0):
            if spec not in seen:
                pair = _try_first(board, current, visible, context, layers, spec)
                if pair:
                    return pair
    return None
