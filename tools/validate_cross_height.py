"""Reproduce the CROSS interval certificate audit from the registered training games."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
import tetris_cross_height as certificate
import tetris_game_library as library


def digest(data):
    return hashlib.sha256(data).hexdigest()


def cross_origins():
    """Independent literal hard-drop indices from the rigid cells."""
    orientations = []
    for shape in game.SHAPES['CROSS']:
        sx, sy, sz, sw = shape.extents
        origins = np.array([game.column_index(x, y, w) for w in range(game.FOURTH - sw + 1)
                            for y in range(game.DEPTH - sy + 1) for x in range(game.WIDTH - sx + 1)])
        # A cell at dz may sit on a column of height at most HEIGHT - sz + dz.
        orientations.append([(origins + game.column_index(x, y, w), game.HEIGHT - sz + z)
                             for x, y, z, w in shape.cells])
    return orientations


def actual_landing(heights, orientations):
    return any(np.logical_and.reduce([heights[ids] <= ceiling for ids, ceiling in cells]).any()
               for cells in orientations)


def replay(path, root, window, orientations):
    if digest(path.read_bytes()) != root['file_sha256']:
        raise ValueError(f"Archive SHA mismatch for seed {root['seed']}")
    record = library.load(path)
    seed = root['seed']
    assert (record.meta['id'], record.meta['seed'], record.meta['phase'], record.meta['piece_set'],
            len(record.moves), record.meta['pieces_sha256']) == (
                root['record_id'], seed, 'training', 'expanded', root['placements'], library.engine_fingerprint(game))
    shape_data = {name: [np.array(shape.columns, dtype=np.int32).T for shape in shapes]
                  for name, shapes in game.SHAPES.items()}
    stream = game.PieceStream(seed, 'expanded')
    board = np.zeros(game.AREA, dtype=np.uint16)
    exact_heights = np.array([int(i).bit_length() for i in range(1 << game.HEIGHT)], np.int16)
    heights = exact_heights[board]
    start = board.copy()
    peak = heights.copy()
    always_fits = True
    clear_masks = []
    added = 0
    counts = Counter()
    for offset, row in enumerate(record.moves, 1):
        piece_id, rotation, x, y, z, w, _, layers = map(int, row)
        piece = stream.pop()
        assert piece == game.PIECE_NAMES[piece_id]
        shape = game.SHAPES[piece][rotation]
        sx, sy, sz, sw = shape.extents
        assert x + sx <= game.WIDTH and y + sy <= game.DEPTH and w + sw <= game.FOURTH and z + sz <= game.HEIGHT
        offsets, masks, bottoms = shape_data[piece][rotation]
        ids = game.column_index(x, y, w) + offsets
        assert np.max(heights[ids] - bottoms) == z
        assert not (board[ids] & (masks << z)).any()
        board[ids] |= (masks << z).astype(np.uint16)
        added += len(shape.cells)
        clearing = int(np.bitwise_and.reduce(board))
        assert clearing.bit_count() == layers
        if clearing:
            clear_masks.append(clearing)
            board = np.array(game.clear_layers(tuple(map(int, board)))[0], dtype=np.uint16)
        heights = exact_heights[board]
        fits = actual_landing(heights, orientations)
        always_fits &= fits
        np.maximum(peak, heights, out=peak)
        if offset % window and offset < len(record.moves):
            continue

        total = sum(mask.bit_count() for mask in clear_masks)
        kind = 'zero' if total == 0 else 'one' if total == 1 else 'multiple'
        counts[f'windows_{kind}'] += 1
        inferred = certificate.cross_survives_interval(start, board, total, added_cells=added)
        known = certificate.cross_survives_interval(start, board, total, clear_masks=clear_masks, added_cells=added)
        assert (not inferred or known) and (not known or always_fits)
        assert (peak <= certificate._upper(heights, clear_masks)).all()
        if always_fits:
            counts['windows_always_fit'] += 1
        if inferred:
            counts[f'certified_unknown_{kind}'] += 1
        if known:
            counts['certified_known_masks'] += 1
        naive = np.minimum(game.HEIGHT, heights + total)
        if (peak > naive).any():
            counts['windows_violating_height_plus_count'] += 1
        if not always_fits and actual_landing(naive, orientations):
            counts['unsafe_height_plus_count_skips'] += 1
        start = board.copy()
        peak = heights.copy()
        always_fits = fits
        clear_masks = []
        added = 0
    assert digest(board.astype('<u2').tobytes()) == record.meta['final_board_sha256']
    return counts


def audit(registration, library_root):
    if not __debug__:
        raise ValueError('Run the archive verification with Python assertions enabled')
    orientations = cross_origins()
    counts = Counter(windows_zero=0, windows_one=0, windows_multiple=0,
                     certified_unknown_zero=0, certified_unknown_one=0,
                     windows_always_fit=0, certified_known_masks=0,
                     unsafe_height_plus_count_skips=0, windows_violating_height_plus_count=0)
    rows = []
    for root in registration['roots']:
        c = replay(library_root / root['file'], root, registration['window'], orientations)
        counts.update(c)
        rows.append({'seed': root['seed'], 'counts': dict(c)})
    return {'placements': sum(r['placements'] for r in registration['roots']), 'counts': dict(counts),
            'per_game': rows, 'false_certificates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'validation/cross-height-certificate.json')
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding='utf-8'))
    registration = evidence['registration']
    signed = json.dumps(registration, separators=(',', ':')).encode() + b'\n'
    if digest(signed) != evidence['registration_sha256']:
        raise ValueError('Training roster registration changed')
    actual = audit(registration, args.library)
    if actual != evidence['audit']:
        raise ValueError(f'Reproduction differs: {actual}')
    print(json.dumps({'placements': actual['placements'], **actual['counts']}, sort_keys=True))


if __name__ == '__main__':
    main()
