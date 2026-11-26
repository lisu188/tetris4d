"""Upper bounds on filling a layer before the next CROSS, using public game state."""
from fractions import Fraction
from functools import lru_cache
from math import comb

import numpy as np

import sparse_tetris_4d_optimized as game
import tetris_cross_survival as cross


@lru_cache(maxsize=None)
def _footprints(piece):
    """Unique horizontal slices, largest first. Other levels may be ignored for an upper bound."""
    footprints = set()
    for shape in game.SHAPES[piece]:
        for level, _ in shape.levels:
            cells = [(x, y, w) for x, y, z, w in shape.cells if z == level]
            lx, ly, lw = map(min, zip(*cells))
            footprints.add(tuple(sorted((x-lx, y-ly, w-lw) for x, y, w in cells)))
    result = []
    for cells in sorted(footprints, key=lambda f: (-len(f), f)):
        mx, my, mw = map(max, zip(*cells))
        # Restrict the origin to keep each translated slice inside the board.
        row = (1 << (game.WIDTH-mx)) - 1
        valid = sum(row << game.column_index(0, y, w)
                    for w in range(game.FOURTH-mw) for y in range(game.DEPTH-my))
        result.append((tuple(game.column_index(*cell) for cell in cells), valid))
    return tuple(result)


def _capacity(holes, piece):
    for offsets, valid in _footprints(piece):
        for offset in offsets:
            valid &= holes >> offset
            if not valid:
                break
        if valid:
            return len(offsets)
    return 0


def layer_bounds(board, pieces, *, fit_footprints=True):
    """Return (level, missing cells, per-piece maxima) for layers a first clear could reach."""
    columns = np.asarray(board, dtype=np.uint16)
    if columns.shape != (game.AREA,):
        raise ValueError('Expected one packed column per board position')
    below_top = columns.copy()
    for shift in (1, 2, 4, 8):
        below_top |= below_top >> shift
    buried_levels = int(np.bitwise_or.reduce(below_top & ~columns))
    maxima = {p: len(_footprints(p)[0][0]) for p in set(pieces)}
    limit = sum(maxima[p] for p in pieces)
    bounds = []
    for z in range(game.HEIGHT):
        if buried_levels & (1 << z):
            continue
        holes = (columns & (1 << z)) == 0
        missing = int(holes.sum())
        if not missing:
            raise ValueError('Expected a board after full layers were compacted')
        if missing <= limit:
            bits = int.from_bytes(np.packbits(holes, bitorder='little').tobytes(), 'little')
            capacities = {p: _capacity(bits, p) for p in maxima} if fit_footprints else maxima
            bounds.append((z, missing, capacities))
    return bounds


def public_prefixes(counts, visible):
    """Fixed pieces and an unordered pool racing the next CROSS, after the current drop."""
    names = game.PIECE_SETS['expanded']
    whole = sum(counts.values()) // len(names)
    if (set(counts) != set(names) or visible not in names
            or any(type(n) is not int or n < 0 or n not in (whole, whole+1) for n in counts.values())):
        raise ValueError('Piece counts are inconsistent with complete expanded bags')
    remaining = tuple(p for p in names if counts[p] == whole)
    if visible not in remaining:
        raise ValueError('Visible piece was already consumed in this bag')
    if visible == 'CROSS':
        return (), ()
    if 'CROSS' in remaining:
        return (visible,), tuple(p for p in remaining if p not in (visible, 'CROSS'))
    return remaining, tuple(p for p in names if p != 'CROSS')


@lru_cache(maxsize=16)
def _subsets(size):
    index = np.arange(1 << size, dtype=np.int32)
    return ((index[:, None] >> np.arange(size)) & 1).astype(np.int16)


def forced_probability(bounds, fixed, possible):
    """Lower bound on death by the next CROSS when the board currently has zero CROSS origins."""
    if not bounds:
        return Fraction(1)
    subsets = _subsets(len(possible))
    impossible = np.ones(len(subsets), dtype=bool)
    for _, missing, capacities in bounds:
        guaranteed = sum(capacities[p] for p in fixed)
        upper = guaranteed + subsets @ np.asarray([capacities[p] for p in possible], dtype=np.int16)
        impossible &= upper < missing
    # In a uniformly shuffled bag, the CROSS rank is uniform. Conditional on its
    # rank, every preceding subset of that size is equally likely.
    totals = np.bincount(subsets[impossible].sum(axis=1), minlength=len(possible)+1)
    return sum(Fraction(int(n), (len(possible)+1) * comb(len(possible), k)) for k, n in enumerate(totals))


def analyze_drop(board, counts, current, visible, rotation, x, y, w):
    """Analyze a proposed legal drop without observing the stream or second preview."""
    if current not in game.SHAPES or not 0 <= rotation < len(game.SHAPES[current]):
        raise ValueError('Unknown current piece or rotation')
    move = game.score_candidate(game.evaluate_board(board), current, rotation, x, y, w)
    if move is None:
        raise ValueError('The proposed drop does not fit')
    counts = dict(counts)
    public_prefixes(counts, current)  # Check the observed current piece against the public bag.
    counts[current] += 1
    fixed, possible = public_prefixes(counts, visible)
    after = cross._after_board(board, move)
    report = dict(current=current, visible=visible, rotation=rotation, x=x, y=y, z=move.z, w=w,
                  cleared=move.layers, cross_before=cross.cross_count(board), cross_after=cross.cross_count(after))
    if report['cross_after'] == 0:
        bounds = layer_bounds(after, fixed+possible)
        lower = forced_probability(bounds, fixed, possible)
        report.update(potential_first_clear_layers=[z for z, missing, maxima in bounds
                                                   if sum(maxima[p] for p in (*fixed, *possible)) >= missing],
                      forced_loss_probability_lower_bound=[lower.numerator, lower.denominator],
                      max_more_placements_if_certain=len(fixed)+len(possible) if lower == 1 else None)
    return report


if __name__ == '__main__':
    import argparse
    import base64
    import gzip
    import json
    from pathlib import Path
    import zlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, help='Expanded-game evaluation snapshot, cache entry or preterminal record (.json or .json.gz)')
    parser.add_argument('--drop', nargs=4, type=int, required=True, metavar=('ROT', 'X', 'Y', 'W'))
    args = parser.parse_args()
    opener = gzip.open if args.snapshot.suffix == '.gz' else open
    with opener(args.snapshot, 'rt') as stream:
        record = json.load(stream)
    state = record.get('snapshot', record.get('data', record))
    if state['version'] != 1 or state['piece_set'] != 'expanded' or sum(state['piece_counts'].values()) != state['pieces']:
        parser.error('Expected a version 1 expanded-game snapshot with complete placement counts')
    packed = zlib.decompress(base64.b64decode(state['board'], validate=True))
    board = tuple(map(int, np.frombuffer(packed, dtype='<u2')))
    print(json.dumps(analyze_drop(board, state['piece_counts'], state['current'], state['next_piece'], *args.drop), indent=2))
