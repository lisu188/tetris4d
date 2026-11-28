#!/usr/bin/env python3
"""Paired comparison of two trained-wells models on the same fresh seeds.

Both models play each seed from an empty board with the native loop for
--placements placements (or until top-out). Board statistics are sampled every
--sample placements: occupancy, stack height and holes, plus the longest gap
between layer clears. Occupancy excursions precede the rare stall failure, so
their frequency is the main stability measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

from finetune_wells_policy import engine

LIMITS = (20_000, 24_000, 28_000, 32_000)


def play(job):
    name, weights, seed, placements, sample = job
    e = engine()
    t4 = e['t4']
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    stream = t4.PieceStream(seed, 'expanded')
    board = np.zeros(t4.AREA, dtype=np.uint16)
    placed, last_clear, longest_gap, layers, top_out = 0, 0, 0, 0, False
    occupancy, heights, holes = [], [], []
    digest = hashlib.sha256()
    started = time.time()
    while placed < placements:
        pieces = np.fromiter((e['index'][stream.pop()] for _ in range(sample)), dtype=np.uint8, count=sample)
        moves = np.zeros(sample * 7, dtype=np.uint8)
        made = e['loop'](board.ctypes.data, e['shapes'].ctypes.data, e['offsets'].ctypes.data,
                         e['counts'].ctypes.data, pieces.ctypes.data, sample, placed, t4.CROSS_WELLS_CANDIDATES,
                         weights.ctypes.data, moves.ctypes.data)
        if made < 0:
            raise RuntimeError(f'native loop failed with code {made}')
        digest.update(moves[:made * 7].tobytes())
        cleared = moves[6:made * 7:7]
        for position in (np.flatnonzero(cleared) + placed + 1).tolist():
            longest_gap = max(longest_gap, position - last_clear)
            last_clear = position
        layers += int(cleared.sum())
        placed += made
        longest_gap = max(longest_gap, placed - last_clear)
        height = e['bit_length'][board]
        count = e['bit_count'][board]
        occupancy.append(int(count.sum()))
        heights.append(int(height.max()))
        holes.append(int((height - count).sum()))
        if made < sample:
            top_out = True
            break
    excursions = {}
    for limit in LIMITS:
        above = [o >= limit for o in occupancy]
        excursions[str(limit)] = sum(1 for i, a in enumerate(above) if a and (i == 0 or not above[i - 1]))
    return {'model': name, 'seed': seed, 'placements': placed, 'top_out': top_out, 'layers': layers,
            'mean_occupancy': round(statistics.mean(occupancy), 1), 'max_occupancy': max(occupancy),
            'p99_occupancy': int(np.percentile(occupancy, 99)), 'max_height': max(heights),
            'mean_height': round(statistics.mean(heights), 3), 'max_holes': max(holes),
            'longest_gap_between_clears': longest_gap, 'excursions': excursions,
            'moves_sha256': digest.hexdigest(), 'seconds': round(time.time() - started, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', action='append', required=True, metavar='NAME=PATH')
    ap.add_argument('--seeds', required=True, help='first:last')
    ap.add_argument('--placements', type=int, default=2_000_000)
    ap.add_argument('--sample', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    e = engine()
    models = {}
    for spec in args.model:
        name, path = spec.split('=', 1)
        e['t4'].TRAINED_WELLS_MODEL = Path(path)
        e['t4']._TRAINED_WELLS_WEIGHTS = None
        models[name] = {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        'weights': list(e['t4'].trained_wells_weights())}
    first, last = (int(v) for v in args.seeds.split(':'))
    jobs = [(name, m['weights'], seed, args.placements, args.sample)
            for seed in range(first, last + 1) for name, m in models.items()]
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(play, jobs)
    summary = {}
    for name in models:
        rows = [r for r in results if r['model'] == name]
        summary[name] = {
            'games': len(rows), 'top_outs': sum(r['top_out'] for r in rows),
            'placements': sum(r['placements'] for r in rows),
            'mean_occupancy': round(statistics.mean(r['mean_occupancy'] for r in rows), 1),
            'mean_of_max_occupancy': round(statistics.mean(r['max_occupancy'] for r in rows), 1),
            'worst_max_occupancy': max(r['max_occupancy'] for r in rows),
            'worst_max_height': max(r['max_height'] for r in rows),
            'worst_longest_gap': max(r['longest_gap_between_clears'] for r in rows),
            'excursions': {str(k): sum(r['excursions'][str(k)] for r in rows) for k in LIMITS},
            'layers_per_million': round(1e6 * sum(r['layers'] for r in rows) / sum(r['placements'] for r in rows), 1)}
    names = list(models)
    if len(names) == 2:
        a, b = names
        paired = []
        for seed in range(first, last + 1):
            ra = next(r for r in results if r['model'] == a and r['seed'] == seed)
            rb = next(r for r in results if r['model'] == b and r['seed'] == seed)
            paired.append(rb['max_occupancy'] - ra['max_occupancy'])
        summary['paired_max_occupancy_difference'] = {
            'models': f'{b} - {a}', 'mean': round(statistics.mean(paired), 1),
            'lower_in_b': sum(d < 0 for d in paired), 'games': len(paired)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'models': models, 'seeds': [first, last], 'placements': args.placements,
                                       'sample_every': args.sample, 'summary': summary, 'games': results},
                                      indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
