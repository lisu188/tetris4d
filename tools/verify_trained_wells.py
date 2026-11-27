#!/usr/bin/env python3
"""Play a trained-wells game to a placement cap, record it, and replay it.

``play`` starts from an empty board, uses only the learned linear well scorer,
and records every hard drop as seven bytes: (expanded piece index, rotation,
x, y, w, landing z, cleared layers). ``replay`` regenerates the seeded piece
stream on its own and drops each recorded placement with ``reference_drop``,
which lowers the rigid piece one Z step at a time onto occupied cells and
compacts full hyperlayers independently of the fast engine path.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as t4  # noqa: E402


def board_sha(board) -> str:
    return hashlib.sha256(struct.pack('<4096H', *board)).hexdigest()


def play(seed: int, cap: int, model: Path, record: Path, progress: Path, every: int) -> dict:
    t4.configure_backend('native')
    t4.TRAINED_WELLS_MODEL = model
    weights = t4.trained_wells_weights()
    game = t4.Game(seed, 'expanded')
    moves = bytearray()
    peak = new_holes = 0
    started = time.time()
    with progress.open('w') as log:
        while not game.game_over and game.pieces < cap:
            decision = t4.choose_trained_wells_move(game.board, game.current, game.pieces)
            if decision is None:
                break
            c, move = decision.move.candidate, decision.move
            game.commit(move)
            moves += bytes((t4.PIECE_NAMES.index(c.piece), c.rotation, c.x, c.y, c.w, c.z, move.layers))
            if game.pieces % every == 0 or game.game_over:
                heights, holes = t4.board_features(game.board)
                peak = max(peak, max(heights))
                log.write(json.dumps({'placements': game.pieces, 'layers': game.layers,
                                      'max_height': max(heights), 'holes': holes,
                                      'occupied_cells': sum(m.bit_count() for m in game.board),
                                      'seconds': round(time.time() - started, 1)}) + '\n')
                log.flush()
    with gzip.open(record, 'wb', compresslevel=9) as handle:
        handle.write(bytes(moves))
    heights, holes = t4.board_features(game.board)
    return {'seed': seed, 'cap': cap, 'placements': game.pieces, 'layers': game.layers,
            'cells_placed': game.cells_placed, 'top_out': game.game_over,
            'stop_reason': 'top_out' if game.game_over else ('piece_limit' if game.pieces >= cap else 'no_decision'),
            'final_max_height': max(heights), 'final_holes': holes,
            'occupied_cells': sum(m.bit_count() for m in game.board),
            'peak_height_at_progress_samples': peak,
            'end_has_legal_move': t4.has_legal_move(game.board, game.current),
            'board_sha256': board_sha(game.board),
            'moves_sha256': hashlib.sha256(bytes(moves)).hexdigest(),
            'weights': list(weights), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'play_seconds': round(time.time() - started, 1)}


FULL = (1 << 16) - 1


def step_drop(board, piece: str, rotation: int, x: int, y: int, w: int):
    """Drop a rigid piece one Z step at a time, then compact full levels.

    Written separately from the engine's fast placement path. Compaction is
    only rebuilt when some level is actually full.
    """
    shape = t4.SHAPES[piece][rotation]
    sx, sy, sz, sw = shape.extents
    if not (0 <= x <= t4.WIDTH - sx and 0 <= y <= t4.DEPTH - sy and 0 <= w <= t4.FOURTH - sw):
        return None
    cells = [(t4.column_index(x + dx, y + dy, w + dw), dz) for dx, dy, dz, dw in shape.cells]
    z = t4.HEIGHT  # start above the well and lower until the next step would collide
    while z > 0 and not any((board[c] >> (z - 1 + dz)) & 1 for c, dz in cells if z - 1 + dz < t4.HEIGHT):
        z -= 1
    if z + sz > t4.HEIGHT:
        return None
    columns = list(board)
    for c, dz in cells:
        if (columns[c] >> (z + dz)) & 1:
            raise AssertionError('step drop overlapped an occupied cell')
        columns[c] |= 1 << (z + dz)
    common = -1
    for mask in columns:
        common &= mask
        if not common:
            break
    if not common:
        return tuple(columns), 0, z
    retained = [level for level in range(t4.HEIGHT) if not (common >> level) & 1]
    compacted = tuple(sum(((mask >> old) & 1) << new for new, old in enumerate(retained)) for mask in columns)
    return compacted, t4.HEIGHT - len(retained), z


def play_native(seed: int, cap: int, model: Path, record: Path, progress: Path, every: int) -> dict:
    """Same policy and record format as play(), with the game loop in C.

    The seeded piece stream is still drawn in Python; the native loop calls
    the identical t4_cross_wells_weighted scorer with the same board, piece,
    turn and budget, applies the drop and compacts full levels.
    """
    import ctypes
    import numpy as np
    t4.configure_backend('native')
    t4.TRAINED_WELLS_MODEL = model
    weights = np.asarray(t4.trained_wells_weights(), dtype=np.float64)
    play_loop = t4._NATIVE.t4_play_trained_wells
    pointer = ctypes.c_void_p
    play_loop.argtypes = [pointer, pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_uint32,
                          ctypes.c_int, pointer, pointer]
    play_loop.restype = ctypes.c_int
    records = [t4.native_shapes(name) for name in t4.PIECE_NAMES]
    counts = np.array([len(r) for r in records], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    shapes = np.ascontiguousarray(np.concatenate(records), dtype=np.int32)
    index = {name: i for i, name in enumerate(t4.PIECE_NAMES)}
    sizes = np.array([t4.PIECE_SIZES[name] for name in t4.PIECE_NAMES], dtype=np.int64)
    bit_length = np.array([v.bit_length() for v in range(1 << 16)], dtype=np.int64)
    bit_count = np.array([v.bit_count() for v in range(1 << 16)], dtype=np.int64)
    stream = t4.PieceStream(seed, 'expanded')
    board = np.zeros(t4.AREA, dtype=np.uint16)
    digest, placed, layers, cells, peak = hashlib.sha256(), 0, 0, 0, 0
    top_out, started = False, time.time()
    with gzip.open(record, 'wb', compresslevel=6) as handle, progress.open('w') as log:
        while placed < cap and not top_out:
            chunk = min(every, cap - placed)
            pieces = np.fromiter((index[stream.pop()] for _ in range(chunk)), dtype=np.uint8, count=chunk)
            moves = np.zeros(chunk * 7, dtype=np.uint8)
            made = play_loop(board.ctypes.data, shapes.ctypes.data, offsets.ctypes.data, counts.ctypes.data,
                             pieces.ctypes.data, chunk, placed & 0xffffffff, t4.CROSS_WELLS_CANDIDATES,
                             weights.ctypes.data, moves.ctypes.data)
            if made < 0:
                raise RuntimeError(f'native play loop failed with code {made} after {placed} placements')
            data = moves[:made * 7].tobytes()
            handle.write(data)
            digest.update(data)
            layers += int(moves[6:made * 7:7].sum())
            cells += int(sizes[pieces[:made]].sum())
            placed += made
            if made < chunk:
                top_out = True
                following = t4.PIECE_NAMES[pieces[made]]
            heights = bit_length[board]
            peak = max(peak, int(heights.max()))
            log.write(json.dumps({'placements': placed, 'layers': layers, 'max_height': int(heights.max()),
                                  'holes': int((heights - bit_count[board]).sum()),
                                  'occupied_cells': int(bit_count[board].sum()),
                                  'seconds': round(time.time() - started, 1)}) + '\n')
            log.flush()
    if not top_out:
        following = stream.pop()
    final = tuple(int(v) for v in board)
    heights, holes = t4.board_features(final)
    return {'seed': seed, 'cap': cap, 'placements': placed, 'layers': layers, 'cells_placed': cells,
            'top_out': top_out, 'stop_reason': 'top_out' if top_out else 'piece_limit',
            'final_max_height': max(heights), 'final_holes': holes,
            'occupied_cells': sum(m.bit_count() for m in final),
            'peak_height_at_progress_samples': peak,
            'end_has_legal_move': t4.has_legal_move(final, following),
            'board_sha256': board_sha(final), 'moves_sha256': digest.hexdigest(),
            'weights': weights.tolist(), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'engine': 'native-loop', 'play_seconds': round(time.time() - started, 1)}


def replay(seed: int, record: Path) -> dict:
    data = gzip.open(record, 'rb').read()
    if len(data) % 7:
        raise ValueError('Record length is not a whole number of placements.')
    stream = t4.PieceStream(seed, 'expanded')
    board, layers, max_seen, started = t4.EMPTY_BOARD, 0, 0, time.time()
    cross_checked = 0
    for i in range(0, len(data), 7):
        piece_index, rotation, x, y, w, z, cleared = data[i:i + 7]
        piece = stream.pop()
        if t4.PIECE_NAMES[piece_index] != piece:
            raise AssertionError(f'placement {i // 7 + 1}: stream drew {piece}, record has {t4.PIECE_NAMES[piece_index]}')
        n = i // 7
        dropped = step_drop(board, piece, rotation, x, y, w)
        if dropped is None:
            raise AssertionError(f'placement {n + 1}: step drop rejected the move')
        if cleared or dropped[1] or n < 2000 or n % 997 == 0:
            if t4.reference_drop(board, piece, rotation, x, y, w) != dropped:
                raise AssertionError(f'placement {n + 1}: step drop disagrees with reference_drop')
            cross_checked += 1
        board, got_layers, got_z = dropped
        if (got_z, got_layers) != (z, cleared):
            raise AssertionError(f'placement {i // 7 + 1}: reference z/layers {got_z}/{got_layers} != {z}/{cleared}')
        layers += got_layers
        if (i // 7) % 4096 == 0:
            max_seen = max(max_seen, max(m.bit_length() for m in board))
    following = stream.pop()
    return {'placements_replayed': len(data) // 7, 'layers': layers, 'board_sha256': board_sha(board),
            'moves_sha256': hashlib.sha256(data).hexdigest(), 'next_piece': following,
            'next_piece_has_legal_move': t4.has_legal_move(board, following),
            'reference_drop_cross_checks': cross_checked,
            'replay_method': 'independent one-step drop with full-level compaction; engine reference_drop cross-checked on the first 2000 placements, every 997th, and every clearing placement',
            'replay_seconds': round(time.time() - started, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('play')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--cap', type=int, default=1_000_000)
    p.add_argument('--model', type=Path, default=ROOT / 'models' / 'trained-wells.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--every', type=int, default=10_000)
    p.add_argument('--engine', choices=('python', 'native'), default='python',
                   help='native runs the game loop in C with the identical policy call')
    r = sub.add_parser('replay')
    r.add_argument('--seed', type=int, required=True)
    r.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / f'moves-{args.seed}.bin.gz'
    if args.command == 'play':
        runner = play_native if args.engine == 'native' else play
        result = runner(args.seed, args.cap, args.model, record, args.output / f'progress-{args.seed}.jsonl', args.every)
        name = f'play-{args.seed}.json'
    else:
        result = replay(args.seed, record)
        name = f'replay-{args.seed}.json'
    (args.output / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
