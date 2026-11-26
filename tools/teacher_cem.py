#!/usr/bin/env python3
"""Cross-entropy tuning of the teacher heuristic for survival, and paired actor comparisons.

The exhaustive teacher (sustained keys: target gain - 4 * future holes, then the shared-sigmoid
score) is what every strong policy here imitates, so a better teacher is the most direct lever.

  tune     Noisy cross-entropy search over the score weights (holes, layers, roughness, threshold;
           height fixed for scale). Fitness: mean placements the teacher survives from near-top-out
           library positions (capped). Every candidate plays the same tuning positions. The best mean
           is then compared with the starting weights on a disjoint held-out set (games of the other
           id parity), exhaustive screening, longer cap.
  actors   Greedy survival of actor checkpoints on the held-out set, each with its own teacher
           weights (an actor's logits include its teacher's prior), paired by position.

Positions are replayed from recorded top-out games and never recorded back into the library.
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
from pathlib import Path
import pickle
import random
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural  # noqa: E402
import train_survival_v4 as v4t  # noqa: E402

base, game = v4t.base, v4t.game
TUNED = ('holes', 'layers', 'roughness', 'threshold')
LOW = np.array([-100., 0., -100., .25])
HIGH = np.array([0., 100., 0., 6.])


def positions(count, window, per_game, seed, parity):
    """Deterministic near-top-out positions from recorded top-out games whose id parity matches."""
    root = v4t.library_root()
    games = sorted((e for e in base.library.iter_index(root)
                    if e.get('game_over') and e.get('stop_reason') == 'top_out' and e.get('piece_set') == 'expanded'
                    and e.get('placements', 0) > window[1] and int(e['id'][:8], 16) % 2 == parity),
                   key=lambda e: e['id'])
    rng = random.Random(seed)
    rng.shuffle(games)
    chosen = []
    for entry in games:
        if len(chosen) >= count:
            break
        try:
            record = base.library.load(root / entry['file'])
            total = len(record.moves)
            wanted = sorted({total - rng.randint(*window) for _ in range(per_game)})
            for index, (state, _move, row) in enumerate(base.library.replay(record, verify=False)):
                if index in wanted:
                    snapshot = copy.deepcopy(state)
                    snapshot._library_log = None
                    base.library.ignore(snapshot)
                    chosen.append({'state': snapshot, 'target': int(row[6]), 'game_id': entry['id'], 'offset': index})
                if index >= wanted[-1]:
                    break
        except ValueError:
            continue  # recorded on a different piece catalogue
    if len(chosen) < count:
        raise SystemExit(f'Only {len(chosen)} positions available for parity {parity}')
    return chosen[:count]


def configure(parameters, note):
    game.configure_network({key: float(value) for key, value in parameters.items()}, {'method': note})


@torch.no_grad()
def play(pool, cap, config, device, choose):
    """Lockstep greedy play from every position; returns placements survived (capped) per position."""
    from tetris_gpu_simulation import GpuPreparer
    episodes = []
    for number, entry in enumerate(pool):
        state = copy.deepcopy(entry['state'])
        state._library_log = None
        base.library.ignore(state)
        episodes.append(base.Episode(number, state, target=entry['target'], phase='benchmark'))
    starts = [e.state.pieces for e in episodes]
    preparer = GpuPreparer(config, device)
    active = list(range(len(episodes)))
    while active:
        result = preparer.prepare_batch([episodes[i].state for i in active], [episodes[i].target for i in active])
        live = []
        for local, i in enumerate(active):
            if result.prepared[local] is not None:
                episodes[i].target = int(result.targets[local])
                live.append(i)
        if live:
            rows = [active[local] for local in result.environment_indices]
            for i, candidate in zip(rows, result.selected_candidates(choose(result))):
                base.apply_move(episodes[i], candidate, 0, 'benchmark')
        active = [i for i in live if episodes[i].state.pieces - starts[i] < cap]
    return [e.state.pieces - s for e, s in zip(episodes, starts)]


def cached_positions(path, build):
    """Positions are expensive to replay; build once and share them (also across worker processes)."""
    path = Path(path)
    if path.exists():
        with path.open('rb') as stream:
            return pickle.load(stream)
    pool = build()
    with path.open('wb') as stream:
        pickle.dump(pool, stream)
    return pool


_WORKER = {}


def _evaluate(job):
    """Worker: teacher survival for one parameter set (module globals are per process)."""
    parameters, path, cap, screening = job
    if _WORKER.get('path') != path:
        game.configure_backend('native')
        with open(path, 'rb') as stream:
            _WORKER.update(path=path, pool=pickle.load(stream))
    configure(parameters, 'teacher-cem-candidate')
    began = time.perf_counter()
    survived = play(_WORKER['pool'], cap, teacher_config(screening), torch.device('cuda'), lambda r: r.teachers)
    return survived, time.perf_counter() - began


def teacher_config(screening):
    return neural.Config(screening=screening, shortlist=2, protect_target=True, recover_target=True)


def paired(values, reference, seed=0):
    differences = [a - b for a, b in zip(values, reference)]
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choice(differences) for _ in differences) for _ in range(4000))
    return {'mean_difference': statistics.mean(differences), 'ci95': [boots[100], boots[3899]],
            'wins': sum(d > 0 for d in differences), 'ties': sum(d == 0 for d in differences),
            'losses': sum(d < 0 for d in differences)}


def tune(args):
    start = game.load_checkpoint(args.weights)
    baseline = dict(start['parameters'])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / 'cem.jsonl').open('a', encoding='utf-8')
    tuning_path = out / 'tuning-positions.pkl'
    tuning = cached_positions(tuning_path, lambda: positions(args.tune_positions, args.window, args.per_game,
                                                             args.seed, parity=0))
    holdout_path = out / 'holdout-positions.pkl'
    holdout = cached_positions(holdout_path, lambda: positions(args.holdout_positions, args.window_holdout,
                                                               args.per_game, args.seed + 1, parity=1))
    print(f'{len(tuning)} tuning and {len(holdout)} held-out positions', flush=True)
    workers = multiprocessing.get_context('spawn').Pool(args.workers)
    rng = np.random.default_rng(args.seed)
    mean = np.array([baseline[k] for k in TUNED])
    std = np.maximum(np.abs(mean) * args.relative_std, .05)
    std[TUNED.index('threshold')] = args.threshold_std

    def evaluate_all(vectors, labels):
        sets = [{**baseline, **dict(zip(TUNED, map(float, vector)))} for vector in vectors]
        outcomes = workers.map(_evaluate, [(p, str(tuning_path), args.cap, args.screening) for p in sets])
        rows = []
        for label, parameters, (survived, seconds) in zip(labels, sets, outcomes):
            rows.append({'label': label, 'parameters': parameters, 'fitness': statistics.mean(survived),
                         'capped': sum(s >= args.cap for s in survived), 'survived': survived, 'seconds': seconds})
            log.write(json.dumps(rows[-1]) + '\n')
        log.flush()
        return rows

    history = []
    for generation in range(args.generations):
        candidates = [mean.copy()] + [np.clip(rng.normal(mean, std), LOW, HIGH) for _ in range(args.population - 1)]
        rows = evaluate_all(candidates, [f'g{generation}-{i}' for i in range(len(candidates))])
        order = np.argsort([-r['fitness'] for r in rows])[:args.elite]
        elite = np.array([candidates[i] for i in order])
        mean = elite.mean(0)
        noise = args.noise * max(0., 1 - generation / max(1, args.generations - 1))
        std = elite.std(0) + noise * np.maximum(np.abs(mean), .05)
        summary = {'generation': generation, 'mean_fitness_of_mean': rows[0]['fitness'],
                   'best_fitness': rows[order[0]]['fitness'], 'best': rows[order[0]]['parameters'],
                   'next_mean': dict(zip(TUNED, map(float, mean))), 'next_std': dict(zip(TUNED, map(float, std)))}
        history.append(summary)
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in summary.items()
                          if k in ('generation', 'mean_fitness_of_mean', 'best_fitness')}),
              json.dumps({k: round(v, 4) for k, v in summary['next_mean'].items()}), flush=True)
    final = {**baseline, **dict(zip(TUNED, map(float, mean)))}
    outcomes = workers.map(_evaluate, [(p, str(holdout_path), args.holdout_cap, 'exhaustive') for p in (baseline, final)])
    workers.close()
    results = {name: survived for name, (survived, _) in zip(('baseline', 'tuned'), outcomes)}
    for name, survived in results.items():
        print(f'holdout {name}: mean {statistics.mean(survived):.0f}', flush=True)
    comparison = paired(results['tuned'], results['baseline'])
    tuned_checkpoint = {**start, 'parameters': final,
                        'training': {'method': 'cross-entropy survival tuning (teacher_cem.py)', 'source': str(args.weights),
                                     'trained_parameters': list(TUNED), 'generations': args.generations,
                                     'population': args.population, 'seed': args.seed,
                                     'holdout_comparison': comparison}}
    base.durable_json(out / 'tuned-sigmoid.json', tuned_checkpoint)
    base.durable_json(out / 'tune-report.json', {
        'schema': 'tetris4d-teacher-cem-v1', 'baseline': baseline, 'tuned': final, 'history': history,
        'tuning': {'positions': len(tuning), 'window': args.window, 'cap': args.cap, 'screening': args.screening},
        'holdout': {'positions': len(holdout), 'window': args.window_holdout, 'cap': args.holdout_cap,
                    'screening': 'exhaustive', 'ids': [[p['game_id'], p['offset']] for p in holdout],
                    'baseline': results['baseline'], 'tuned': results['tuned'], 'comparison': comparison}})
    print('holdout tuned vs baseline:', json.dumps(comparison), flush=True)


def actors(args):
    device = torch.device('cuda')
    game.load_checkpoint(args.weights[0])
    holdout = cached_positions(args.positions_file, lambda: positions(
        args.holdout_positions, args.window_holdout, args.per_game, args.seed + 1, parity=1))
    results = []
    for checkpoint, weights in zip(args.checkpoints, args.weights):
        game.load_checkpoint(weights)
        model, _ = neural.load_checkpoint(checkpoint, device)
        model.eval()
        survived = play(holdout, args.holdout_cap, model.config, device,
                        lambda r: model(r.batch)['logits'].argmax(-1))
        results.append({'checkpoint': str(checkpoint), 'weights': str(weights), 'survived': survived,
                        'mean': statistics.mean(survived)})
        print(f'{checkpoint}: mean {results[-1]["mean"]:.0f}', flush=True)
    report = {'schema': 'tetris4d-actor-holdout-v1', 'positions': len(holdout), 'cap': args.holdout_cap,
              'ids': [[p['game_id'], p['offset']] for p in holdout], 'results': results,
              'comparisons': [paired(r['survived'], results[0]['survived']) for r in results[1:]]}
    for comparison in report['comparisons']:
        print('vs first:', json.dumps(comparison), flush=True)
    if args.output:
        base.durable_json(args.output, report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('tune', 'actors'):
        p = sub.add_parser(name)
        p.add_argument('--holdout-positions', type=int, default=128)
        p.add_argument('--window-holdout', type=int, nargs=2, default=(500, 3000))
        p.add_argument('--holdout-cap', type=int, default=6000)
        p.add_argument('--per-game', type=int, default=2)
        p.add_argument('--seed', type=int, default=20261118)
    t = sub.choices['tune']
    t.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    t.add_argument('--output-dir', type=Path, required=True)
    t.add_argument('--tune-positions', type=int, default=64)
    t.add_argument('--window', type=int, nargs=2, default=(200, 1200))
    t.add_argument('--cap', type=int, default=2000)
    t.add_argument('--screening', choices=('sampled', 'exhaustive'), default='exhaustive')
    t.add_argument('--population', type=int, default=16)
    t.add_argument('--elite', type=int, default=4)
    t.add_argument('--generations', type=int, default=8)
    t.add_argument('--relative-std', type=float, default=.3)
    t.add_argument('--threshold-std', type=float, default=.8)
    t.add_argument('--noise', type=float, default=.1, help='Szita-Lorincz extra std (relative), decaying to 0')
    t.add_argument('--workers', type=int, default=4, help='Parallel evaluation processes (each holds its own weights)')
    a = sub.choices['actors']
    a.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    a.add_argument('--weights', type=Path, nargs='+', required=True, help='Teacher weights per checkpoint')
    a.add_argument('--output', type=Path)
    a.add_argument('--positions-file', type=Path, required=True, help='Pickled held-out positions (built if missing)')
    args = parser.parse_args(argv)
    game.configure_backend('native')
    if args.command == 'tune':
        if not 2 <= args.elite <= args.population:
            parser.error('--elite must be between 2 and --population')
        tune(args)
    else:
        if len(args.checkpoints) != len(args.weights):
            parser.error('Give one --weights file per checkpoint')
        actors(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
