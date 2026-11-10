from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_tetris_4d_optimized as game
from tools.train_agent import atomic_json, curriculum, evaluate_task, summarize


def compare(checkpoint: Path, output: Path, workers: int = 4, full_games: int = 6,
            full_pieces: int = 1000, stress_games: int = 16) -> dict:
    loaded = game.load_checkpoint(checkpoint)
    parameters = loaded['parameters']
    suites = {
        'heldout_curriculum': (curriculum(10001, stress_games),
                              {'pieces': 256, 'depth': 1, 'beam': 4, 'candidates': 512, 'backend': 'numpy'}),
        'default_search_empty': ([{'seed': 20001 + i, 'kind': 'empty', 'piece_set': 'expanded'} for i in range(full_games)],
                                 {'pieces': full_pieces, 'depth': 2, 'beam': 4, 'candidates': 2048, 'backend': 'numpy'})}
    training_seeds = {s['seed'] for key in ('train_specs', 'validation_specs')
                      for s in loaded['optimizer']['config'][key]}
    test_seeds = [s['seed'] for specs, _ in suites.values() for s in specs]
    if len(test_seeds) != len(set(test_seeds)) or training_seeds.intersection(test_seeds):
        raise ValueError('Held-out test seeds must be unique and disjoint from training and validation.')
    started = time.perf_counter()
    report = {'checkpoint': str(checkpoint), 'parameters': parameters,
              'baseline_parameters': dict(game.BASELINE_PARAMETERS),
              'selection': 'Checkpoint frozen before evaluation. No optimization uses these test results.',
              'suites': {}}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for name, (specs, settings) in suites.items():
            jobs = [(label, spec, pool.submit(evaluate_task, (weights, [spec], settings)))
                    for label, weights in (('baseline', dict(game.BASELINE_PARAMETERS)), ('trained', parameters))
                    for spec in specs]
            runs = {'baseline': [], 'trained': []}
            for label, spec, job in jobs:
                result = job.result()['runs'][0]
                runs[label].append(result)
                print(json.dumps({'suite': name, 'model': label, **result}), flush=True)
            summary = {label: summarize(values) for label, values in runs.items()}
            paired = [{'seed': a['seed'], 'pieces_change': b['pieces'] - a['pieces'],
                       'holes_change': b['new_holes'] - a['new_holes'],
                       'hyperlayers_change': b['hyperlayers'] - a['hyperlayers'],
                       'fitness_change': b['fitness'] - a['fitness']}
                      for a, b in zip(runs['baseline'], runs['trained'])]
            report['suites'][name] = {'settings': settings, 'summary': summary, 'runs': runs,
                                     'paired': paired,
                                     'mean_paired_fitness_change': statistics.fmean(p['fitness_change'] for p in paired)}
            atomic_json(output, report)
    report['wall_seconds'] = round(time.perf_counter() - started, 3)
    report['limitations'] = 'Small fixed test sets; capped performance and shaped fitness do not establish indefinite survival or universal superiority.'
    atomic_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a frozen Tetris 4D checkpoint on untouched seeds.')
    parser.add_argument('--weights', type=Path, default=Path('models/trained-sigmoid.json'))
    parser.add_argument('--output', type=Path, default=Path('validation/training-heldout.json'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--games', type=int, default=6)
    parser.add_argument('--pieces', type=int, default=1000)
    parser.add_argument('--stress-games', type=int, default=16)
    args = parser.parse_args()
    if min(args.workers, args.games, args.pieces, args.stress_games) < 1:
        parser.error('Workers, games, pieces and stress-games must be positive.')
    try:
        report = compare(args.weights, args.output, args.workers, args.games, args.pieces, args.stress_games)
        print(json.dumps({name: suite['summary'] for name, suite in report['suites'].items()}, indent=2))
    except (ValueError, KeyError, OSError, AssertionError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
