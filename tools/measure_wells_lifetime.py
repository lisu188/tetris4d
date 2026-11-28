#!/usr/bin/env python3
"""Measure how long trained-wells games last when nothing caps them.

Each worker plays fresh seeds with the native game loop until physical
top-out or the shared deadline (then the game is recorded as censored). Full
move records would take about 7 bytes per placement, so a game keeps
statistics instead: a streaming SHA-256 of every move, occupancy and height
histograms, the longest gap between layer clears, and counts of occupancy
excursions sampled every --sample placements. For a top-out it also saves the
last window of moves with the board and piece-stream state at its start, so
the collapse can be replayed independently.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

EXCURSIONS = (24_000, 28_000, 32_000, 36_000)
TURN_LIMIT = 4_000_000_000  # the policy's turn counter is 32-bit


def worker(job):
    seed, model, engine_root, out_dir, deadline, sample, window = job
    if time.time() >= deadline:
        return None  # budget spent before this seed started; record nothing
    sys.path.insert(0, str(engine_root))
    import ctypes
    import numpy as np
    import sparse_tetris_4d_optimized as t4
    t4.configure_backend('native')
    t4.TRAINED_WELLS_MODEL = Path(model)
    weights = np.asarray(t4.trained_wells_weights(), dtype=np.float64)
    loop = t4._NATIVE.t4_play_trained_wells
    pointer = ctypes.c_void_p
    loop.argtypes = [pointer, pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_uint32,
                     ctypes.c_int, pointer, pointer]
    loop.restype = ctypes.c_int
    records = [t4.native_shapes(name) for name in t4.PIECE_NAMES]
    counts = np.array([len(r) for r in records], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    shapes = np.ascontiguousarray(np.concatenate(records), dtype=np.int32)
    index = {name: i for i, name in enumerate(t4.PIECE_NAMES)}
    bit_length = np.array([v.bit_length() for v in range(1 << 16)], dtype=np.int64)
    bit_count = np.array([v.bit_count() for v in range(1 << 16)], dtype=np.int64)

    stream = t4.PieceStream(seed, 'expanded')
    board = np.zeros(t4.AREA, dtype=np.uint16)
    digest = hashlib.sha256()
    placed = layers = 0
    occupancy_hist = np.zeros(65, dtype=np.int64)   # bins of 1,024 cells
    height_hist = np.zeros(17, dtype=np.int64)
    excursions = {limit: 0 for limit in EXCURSIONS}
    in_excursion = {limit: False for limit in EXCURSIONS}
    max_occupancy = max_height = max_holes = 0
    last_clear = 0
    longest_gap = 0
    window_start = {'placements': 0, 'board': board.copy(), 'rng': stream.rng.getstate(), 'bag': list(stream.bag)}
    previous_window = None
    window_moves = bytearray()
    started = time.time()
    top_out = False
    progress = (out_dir / f'progress-{seed}.jsonl').open('w')
    next_report = 10_000_000
    while placed < TURN_LIMIT and time.time() < deadline:
        pieces = np.fromiter((index[stream.pop()] for _ in range(sample)), dtype=np.uint8, count=sample)
        moves = np.zeros(sample * 7, dtype=np.uint8)
        made = loop(board.ctypes.data, shapes.ctypes.data, offsets.ctypes.data, counts.ctypes.data,
                    pieces.ctypes.data, sample, placed & 0xffffffff, t4.CROSS_WELLS_CANDIDATES,
                    weights.ctypes.data, moves.ctypes.data)
        if made < 0:
            raise RuntimeError(f'seed {seed}: native loop failed with code {made} after {placed}')
        data = moves[:made * 7].tobytes()
        digest.update(data)
        window_moves += data
        cleared = moves[6:made * 7:7]
        hits = np.flatnonzero(cleared) + placed + 1
        for position in hits.tolist():
            longest_gap = max(longest_gap, position - last_clear)
            last_clear = position
        layers += int(cleared.sum())
        placed += made
        longest_gap = max(longest_gap, placed - last_clear)
        heights = bit_length[board]
        occupancy = int(bit_count[board].sum())
        height = int(heights.max())
        holes = int((heights - bit_count[board]).sum())
        occupancy_hist[occupancy // 1024] += 1
        height_hist[height] += 1
        max_occupancy, max_height, max_holes = max(max_occupancy, occupancy), max(max_height, height), max(max_holes, holes)
        for limit in EXCURSIONS:
            above = occupancy >= limit
            if above and not in_excursion[limit]:
                excursions[limit] += 1
            in_excursion[limit] = above
        if made < sample:
            top_out = True
            following = t4.PIECE_NAMES[pieces[made]]
            break
        if placed - window_start['placements'] >= window:
            previous_window = (window_start, bytes(window_moves))
            window_start = {'placements': placed, 'board': board.copy(), 'rng': stream.rng.getstate(),
                            'bag': list(stream.bag)}
            window_moves = bytearray()
        if placed >= next_report:
            progress.write(json.dumps({'placements': placed, 'layers': layers, 'max_height': height,
                                       'holes': holes, 'occupied_cells': occupancy,
                                       'seconds': round(time.time() - started, 1)}) + '\n')
            progress.flush()
            next_report += 10_000_000
    progress.close()
    result = {'seed': seed, 'placements': placed, 'top_out': top_out, 'censored': not top_out,
              'layers': layers, 'seconds': round(time.time() - started, 1),
              'moves_sha256': digest.hexdigest(),
              'board_sha256': hashlib.sha256(board.astype('<u2').tobytes()).hexdigest(),
              'max_occupancy': max_occupancy, 'max_height': max_height, 'max_holes': max_holes,
              'longest_gap_between_clears': longest_gap,
              'occupancy_excursions': {str(k): v for k, v in excursions.items()},
              'occupancy_histogram_1024': occupancy_hist.tolist(), 'height_histogram': height_hist.tolist(),
              'sample_every': sample, 'model_sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest()}
    if top_out:
        final = tuple(int(v) for v in board)
        result['next_piece'] = following
        result['next_piece_has_legal_move'] = t4.has_legal_move(final, following)
        # Keep the last one or two windows so the collapse can be replayed from a known board.
        start, moves_blob = (previous_window[0], previous_window[1] + bytes(window_moves)) if previous_window \
            else (window_start, bytes(window_moves))
        with gzip.open(out_dir / f'topout-{seed}.bin.gz', 'wb') as handle:
            handle.write(moves_blob)
        (out_dir / f'topout-{seed}.json').write_text(json.dumps({
            'seed': seed, 'window_start_placements': start['placements'],
            'board_at_window_start': [int(v) for v in start['board']],
            'rng_state_at_window_start': [start['rng'][0], list(start['rng'][1]), start['rng'][2]],
            'bag_at_window_start': start['bag'], 'moves_in_window': len(moves_blob) // 7,
            'final_placements': placed, 'next_piece': following}) + '\n')
    (out_dir / f'game-{seed}.json').write_text(json.dumps(result, indent=1) + '\n')
    return result


def replay_topout(path: Path, engine_root: Path) -> dict:
    """Re-drop a saved top-out window from its starting board and piece-stream state."""
    sys.path.insert(0, str(engine_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sparse_tetris_4d_optimized as t4
    from verify_trained_wells import step_drop
    meta = json.loads(path.read_text())
    data = gzip.open(path.with_suffix('.bin.gz')).read()
    stream = t4.PieceStream(meta['seed'], 'expanded')
    version, state, gauss = meta['rng_state_at_window_start']
    stream.rng.setstate((version, tuple(state), gauss))
    stream.bag = list(meta['bag_at_window_start'])
    board = tuple(meta['board_at_window_start'])
    for i in range(0, len(data), 7):
        piece_index, rotation, x, y, w, z, cleared = data[i:i + 7]
        piece = stream.pop()
        if t4.PIECE_NAMES[piece_index] != piece:
            raise AssertionError(f'move {i // 7}: stream drew {piece}, record has {t4.PIECE_NAMES[piece_index]}')
        dropped = step_drop(board, piece, rotation, x, y, w)
        if dropped is None or dropped[2] != z or dropped[1] != cleared:
            raise AssertionError(f'move {i // 7}: step drop disagrees with the record')
        board = dropped[0]
    following = stream.pop()
    assert following == meta['next_piece']
    end = meta['window_start_placements'] + len(data) // 7
    assert end == meta['final_placements']
    return {'seed': meta['seed'], 'replayed_from': meta['window_start_placements'], 'replayed_to': end,
            'next_piece': following, 'next_piece_has_legal_move': t4.has_legal_move(board, following)}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == 'replay-topout':
        root = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[1]
        print(json.dumps(replay_topout(Path(sys.argv[2]), root), indent=1))
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--engine-root', type=Path, default=Path(__file__).resolve().parents[1],
                    help='checkout whose engine and native library play the games')
    ap.add_argument('--seeds', required=True, help='first:last seed range, played in order')
    ap.add_argument('--hours', type=float, required=True, help='wall-clock budget; running games are censored at the end')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--sample', type=int, default=2000, help='placements between board statistics samples')
    ap.add_argument('--window', type=int, default=100_000, help='placements kept for replaying a top-out')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    first, last = (int(v) for v in args.seeds.split(':'))
    deadline = time.time() + args.hours * 3600
    meta = {'model': str(args.model), 'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'engine_root': str(args.engine_root), 'seeds': [first, last], 'hours': args.hours,
            'workers': args.workers, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (args.output / 'run.json').write_text(json.dumps(meta, indent=1) + '\n')
    jobs = [(seed, str(args.model), args.engine_root, args.output, deadline, args.sample, args.window)
            for seed in range(first, last + 1)]
    with mp.get_context('fork').Pool(args.workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(worker, jobs):
            if result is None:
                continue
            status = 'TOP-OUT' if result['top_out'] else 'censored'
            print(f"{result['seed']} {status} at {result['placements']:,} placements "
                  f"(max occupancy {result['max_occupancy']:,}, height {result['max_height']}, "
                  f"longest clear gap {result['longest_gap_between_clears']:,})", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
