#!/usr/bin/env python3
"""Fine-tune the trained-wells scorer, including its layer-completion features.

The rare failure of the 7-feature model is a stall: low levels keep a few
open cells, clears stop and occupancy climbs until the stack reaches the
ceiling. Top-outs are far too rare to optimize directly, so this CEM run plays
long games with the native loop and minimizes their occupancy, the measurable
precursor: fitness = -(mean occupancy + 0.5 * peak occupancy), sampled every
--sample placements, with a large penalty for any top-out.

Weights use w = sinh(3 theta) / scale as in train_wells_policy.py. The search
starts from an existing model's theta and zero for the two new features.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('new_levels', 'new_holes', 'covers_ready', 'well_benefit', 'height_cost', 'roughness', 'high',
            'completed_levels', 'completion_urgency')
SCALE = np.array([1.0, 4.0, 1.0, 50.0, 100.0, 50.0, 8.0, 0.1, 0.1])
_ENGINE = {}


def to_weights(theta):
    return np.sinh(3.0 * np.asarray(theta, dtype=float)) / SCALE


def engine():
    if not _ENGINE:
        sys.path.insert(0, str(ROOT))
        import ctypes
        import sparse_tetris_4d_optimized as t4
        t4.configure_backend('native')
        loop = t4._NATIVE.t4_play_trained_wells
        pointer = ctypes.c_void_p
        loop.argtypes = [pointer, pointer, pointer, pointer, pointer, ctypes.c_int, ctypes.c_uint32,
                         ctypes.c_int, pointer, pointer]
        loop.restype = ctypes.c_int
        records = [t4.native_shapes(name) for name in t4.PIECE_NAMES]
        counts = np.array([len(r) for r in records], dtype=np.int32)
        _ENGINE.update(t4=t4, loop=loop, counts=counts,
                       offsets=np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32),
                       shapes=np.ascontiguousarray(np.concatenate(records), dtype=np.int32),
                       index={name: i for i, name in enumerate(t4.PIECE_NAMES)},
                       bit_count=np.array([v.bit_count() for v in range(1 << 16)], dtype=np.int64),
                       bit_length=np.array([v.bit_length() for v in range(1 << 16)], dtype=np.int64))
    return _ENGINE


def play(job):
    weights, seed, placements, sample = job
    e = engine()
    t4 = e['t4']
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    stream = t4.PieceStream(seed, 'expanded')
    board = np.zeros(t4.AREA, dtype=np.uint16)
    placed, occupancy, heights, top_out = 0, [], [], False
    while placed < placements:
        pieces = np.fromiter((e['index'][stream.pop()] for _ in range(sample)), dtype=np.uint8, count=sample)
        moves = np.zeros(sample * 7, dtype=np.uint8)
        made = e['loop'](board.ctypes.data, e['shapes'].ctypes.data, e['offsets'].ctypes.data,
                         e['counts'].ctypes.data, pieces.ctypes.data, sample, placed, t4.CROSS_WELLS_CANDIDATES,
                         weights.ctypes.data, moves.ctypes.data)
        if made < 0:
            raise RuntimeError(f'native loop failed with code {made}')
        placed += made
        occupancy.append(int(e['bit_count'][board].sum()))
        heights.append(int(e['bit_length'][board].max()))
        if made < sample:
            top_out = True
            break
    return {'seed': seed, 'placements': placed, 'top_out': top_out, 'mean_occupancy': float(np.mean(occupancy)),
            'max_occupancy': max(occupancy), 'max_height': max(heights)}


def fitness(results, placements):
    score = 0.0
    for r in results:
        if r['top_out']:
            score -= 1e6 * (1 - r['placements'] / placements) + 1e5
        score -= r['mean_occupancy'] + 0.5 * r['max_occupancy']
    return score / len(results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--init', type=Path, default=ROOT / 'models' / 'trained-wells.json')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--generations', type=int, default=8)
    ap.add_argument('--population', type=int, default=16)
    ap.add_argument('--elite', type=int, default=4)
    ap.add_argument('--games', type=int, default=2, help='fresh seeds per generation, shared by all candidates')
    ap.add_argument('--placements', type=int, default=300_000)
    ap.add_argument('--sample', type=int, default=2000)
    ap.add_argument('--old-std', type=float, default=0.1)
    ap.add_argument('--new-std', type=float, default=0.6)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--rng-seed', type=int, default=20261128)
    ap.add_argument('--train-seed-base', type=int, default=65_000_000)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    init = json.loads(args.init.read_text())
    theta0 = list(init['theta']) + [0.0] * (len(FEATURES) - len(init['theta']))
    mean = np.array(theta0, dtype=float)
    std = np.array([args.old_std] * 7 + [args.new_std] * 2)
    rng = np.random.default_rng(args.rng_seed)
    log = (args.output / 'generations.jsonl').open('a')
    seed_counter = args.train_seed_base
    with mp.get_context('fork').Pool(args.workers) as pool:
        for gen in range(args.generations):
            started = time.time()
            seeds = list(range(seed_counter, seed_counter + args.games))
            seed_counter += args.games
            thetas = rng.normal(mean, std, size=(args.population, len(FEATURES)))
            thetas[0] = mean
            jobs = [(to_weights(t), s, args.placements, args.sample) for t in thetas for s in seeds]
            outcomes = pool.map(play, jobs)
            scored = []
            for i in range(args.population):
                results = outcomes[i * len(seeds):(i + 1) * len(seeds)]
                scored.append((fitness(results, args.placements), i, results))
            scored.sort(key=lambda item: -item[0])
            elite = np.array([thetas[i] for _, i, _ in scored[:args.elite]])
            mean_result = next(r for _, i, r in scored if i == 0)
            mean, std = elite.mean(0), np.maximum(elite.std(0), 0.03) + 0.01
            record = {'generation': gen, 'seeds': seeds, 'placements': args.placements,
                      'fitness_quartiles': np.percentile([f for f, _, _ in scored], [0, 25, 50, 75, 100]).tolist(),
                      'best_fitness': scored[0][0], 'best_results': scored[0][2],
                      'previous_mean_fitness': next(f for f, i, _ in scored if i == 0),
                      'previous_mean_results': mean_result,
                      'top_outs': sum(r['top_out'] for _, _, rs in scored for r in rs),
                      'mean_theta': mean.tolist(), 'mean_weights': to_weights(mean).tolist(), 'std': std.tolist(),
                      'seconds': round(time.time() - started, 1)}
            log.write(json.dumps(record) + '\n')
            log.flush()
            best = scored[0][2]
            print(f"gen {gen} best {scored[0][0]:.0f} (occ {np.mean([r['mean_occupancy'] for r in best]):.0f}/"
                  f"{max(r['max_occupancy'] for r in best)}) prev-mean {record['previous_mean_fitness']:.0f} "
                  f"top-outs {record['top_outs']} {record['seconds']}s "
                  f"w7={record['mean_weights'][7]:.3g} w8={record['mean_weights'][8]:.3g}", flush=True)
    model = {'schema': 'trained-wells-linear-v2', 'feature_names': list(FEATURES),
             'weights': to_weights(mean).tolist(), 'theta': mean.tolist(), 'parameterization': 'sinh',
             'weight_formula': 'w = sinh(3*theta) / scale', 'scale': SCALE.tolist(), 'lower_score_is_better': True,
             'budget': 1536,
             'training': {'method': 'cross-entropy method fine-tuning on long-game occupancy (native loop)',
                          'initialization': f'theta of {args.init.name} for the first seven features; 0 for the two layer-completion features',
                          'fitness': '-(mean occupancy + 0.5 * peak occupancy) per game, sampled every '
                                     f'{args.sample} placements; top-outs penalized',
                          'rng_seed': args.rng_seed, 'generations': args.generations, 'population': args.population,
                          'elite': args.elite, 'games_per_generation': args.games, 'placements_per_game': args.placements,
                          'train_seed_range': [args.train_seed_base, seed_counter - 1]}}
    (args.output / 'trained-wells-v2.json').write_text(json.dumps(model, indent=2) + '\n')
    print(json.dumps(model['weights']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
