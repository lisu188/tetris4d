#!/usr/bin/env python3
"""Train the linear CROSS-well scorer by the cross-entropy method on survival.

The policy is the native candidate generator plus a learned linear score over
seven per-candidate features (see ``t4_cross_wells_weighted``). Weights start
from a zero-mean isotropic Gaussian, so no coefficient, sign or priority is
preset. Fitness is the number of placements survived from an empty board on
training seeds, capped; the cap doubles whenever most of the elite reaches it.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURES = ('new_levels', 'new_holes', 'covers_ready', 'well_benefit',
            'height_cost', 'roughness', 'high')
# Diagonal preconditioner: rough per-feature magnitudes, so that one unit of
# search noise moves every feature's contribution by a comparable amount.
SCALE = np.array([1.0, 4.0, 1.0, 50.0, 100.0, 50.0, 8.0])
SINH_GAIN = 3.0


def to_weights(theta, param):
    # 'linear': w = theta/scale. 'sinh': w = sinh(3 theta)/scale, which keeps a
    # zero-mean, sign-free start but lets the search reach near-lexicographic
    # priorities (ratios of 10^3-10^4) within a few standard deviations.
    theta = np.asarray(theta, dtype=float)
    return (np.sinh(SINH_GAIN * theta) if param == 'sinh' else theta) / SCALE


def play(args):
    weights, seed, cap = args
    import sparse_tetris_4d_optimized as t4
    t4.configure_backend('native')
    game = t4.Game(seed, 'expanded')
    while not game.game_over and game.pieces < cap:
        decision = t4.choose_trained_wells_move(game.board, game.current, game.pieces,
                                                weights=tuple(weights))
        if decision is None:
            break
        game.commit(decision.move)
    heights, holes = t4.board_features(game.board)
    return {'seed': seed, 'placements': game.pieces, 'capped': game.pieces >= cap,
            'max_height': max(heights), 'holes': holes, 'layers': game.layers}


def fitness(results, cap):
    # Survival first; for capped games prefer a lower, cleaner final board.
    total = 0.0
    for r in results:
        total += r['placements']
        if r['capped']:
            total += cap * 0.05 * (16 - r['max_height']) / 16 - r['holes']
    return total / len(results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--generations', type=int, default=30)
    ap.add_argument('--population', type=int, default=16)
    ap.add_argument('--elite', type=int, default=4)
    ap.add_argument('--seeds-per-eval', type=int, default=2)
    ap.add_argument('--cap', type=int, default=2000)
    ap.add_argument('--max-cap', type=int, default=64000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--rng-seed', type=int, default=20261126)
    ap.add_argument('--train-seed-base', type=int, default=51_000_000)
    ap.add_argument('--param', choices=('linear', 'sinh'), default='linear')
    ap.add_argument('--min-std', type=float, default=0.05)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / 'generations.jsonl').open('a')

    rng = np.random.default_rng(args.rng_seed)
    mean, std = np.zeros(len(FEATURES)), np.ones(len(FEATURES))
    cap, best = args.cap, None
    seed_counter = args.train_seed_base
    with mp.get_context('fork').Pool(args.workers) as pool:
        for gen in range(args.generations):
            started = time.time()
            # Fresh training seeds every generation, shared by the whole population.
            seeds = [seed_counter + i for i in range(args.seeds_per_eval)]
            seed_counter += args.seeds_per_eval
            thetas = rng.normal(mean, std, size=(args.population, len(FEATURES)))
            thetas[0] = mean  # re-evaluate the current mean on the new seeds
            weights = to_weights(thetas, args.param)
            jobs = [(w, s, cap) for w in weights for s in seeds]
            outcomes = pool.map(play, jobs)
            scores = []
            for i in range(args.population):
                res = outcomes[i*len(seeds):(i+1)*len(seeds)]
                scores.append((fitness(res, cap), i, res))
            scores.sort(key=lambda t: -t[0])
            elite = np.array([thetas[i] for _, i, _ in scores[:args.elite]])
            mean = elite.mean(0)
            std = np.maximum(elite.std(0), args.min_std) + 0.02
            elite_capped = sum(all(r['capped'] for r in res) for _, _, res in scores[:args.elite])
            top = scores[0]
            record = {'generation': gen, 'cap': cap, 'seeds': seeds, 'param': args.param,
                      'mean_weights': to_weights(mean, args.param).tolist(),
                      'best_fitness': top[0], 'best_theta': thetas[top[1]].tolist(),
                      'best_results': top[2], 'mean_theta': mean.tolist(), 'std': std.tolist(),
                      'elite_capped': elite_capped,
                      'fitness_quartiles': np.percentile([s for s, _, _ in scores], [0, 25, 50, 75, 100]).tolist(),
                      'seconds': round(time.time() - started, 1)}
            log.write(json.dumps(record) + '\n'); log.flush()
            print(f"gen {gen:3d} cap {cap:6d} best {top[0]:9.1f} median "
                  f"{record['fitness_quartiles'][2]:9.1f} elite_capped {elite_capped} "
                  f"{record['seconds']}s mean={np.round(mean, 2).tolist()}", flush=True)
            best = mean
            if elite_capped >= args.elite - 1 and cap < args.max_cap:
                cap *= 2
    model = {'schema': 'trained-wells-linear-v1', 'feature_names': list(FEATURES),
             'weights': to_weights(best, args.param).tolist(), 'theta': best.tolist(),
             'parameterization': args.param, 'scale': SCALE.tolist(),
             'lower_score_is_better': True,
             'training': {'method': 'cross-entropy method on capped survival from empty boards',
                          'initialization': 'zero-mean isotropic Gaussian; no preset coefficients',
                          'rng_seed': args.rng_seed, 'param': args.param, 'generations': args.generations,
                          'population': args.population, 'elite': args.elite,
                          'seeds_per_eval': args.seeds_per_eval, 'final_cap': cap,
                          'train_seed_range': [args.train_seed_base, seed_counter - 1]}}
    (args.output / 'trained-wells.json').write_text(json.dumps(model, indent=2) + '\n')
    print(json.dumps(model, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
