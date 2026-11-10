from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_tetris_4d_optimized as game

BOUNDS = ((math.log(.05), math.log(20.0)), (math.log(.02), math.log(2.0)), (math.log(.25), math.log(6.0)))
BASE_COMMIT = 'f7d79639ca50274a0fffb770b664f1505b6788af'


def initial_board(kind: str, seed: int) -> game.Board:
    rng = random.Random(seed ^ 0x4D545249)
    if kind == 'empty':
        return game.EMPTY_BOARD
    if kind == 'rough':
        columns = [(1 << rng.randrange(4, 12)) - 1 for _ in range(game.AREA)]
        columns[rng.randrange(game.AREA)] = 0
        return tuple(columns)
    if kind not in ('pocket4', 'pocket5'):
        raise ValueError(f'Unknown curriculum: {kind}')
    size = int(kind[-1])
    ox, oy, ow = (rng.randrange(1, 16 - size) for _ in range(3))
    columns = [(1 << game.HEIGHT) - 1] * game.AREA
    for w in range(ow, ow + size):
        for y in range(oy, oy + size):
            for x in range(ox, ox + size):
                columns[game.column_index(x, y, w)] = (1 << rng.randrange(3, 9)) - 1
    columns[game.column_index(ox, oy, ow)] = 0
    return tuple(columns)


def curriculum(start: int, count: int) -> list[dict]:
    kinds = ('empty', 'rough', 'pocket4', 'pocket5')
    sets = ('expanded', 'classic', '4d')
    return [{'seed': start + i, 'kind': kinds[i % 4], 'piece_set': sets[(i // 4 + i) % 3]} for i in range(count)]


def rollout(spec: dict, pieces: int, depth: int, beam: int, candidates: int) -> dict:
    agent = game.Game(spec['seed'], spec['piece_set'])
    agent.board = initial_board(spec['kind'], spec['seed'])
    initial = game.evaluate_board(agent.board)
    initial_mass = sum(mask.bit_count() for mask in agent.board)
    evaluations = 0
    for step in range(pieces):
        if agent.game_over:
            break
        decision = game.choose_move(agent.board, agent.current, agent.next_piece, depth, beam,
                                    candidates, spec['seed'] * 8191 + step)
        if decision is None:
            agent.game_over = True
            break
        evaluations += decision.evaluations
        agent.commit(decision.move)
    summary = agent.summary()
    if summary['occupied_cells'] != initial_mass + agent.cells_placed - game.AREA * agent.layers:
        raise AssertionError('Curriculum rollout violated cell conservation.')
    new_holes = summary['holes'] - initial.holes
    fitness = agent.pieces / pieces + .25 * agent.layers - .25 * new_holes / max(1, agent.cells_placed)
    return dict(spec, pieces=agent.pieces, cells_placed=agent.cells_placed, hyperlayers=agent.layers,
                holes=summary['holes'], initial_holes=initial.holes, new_holes=new_holes,
                max_height=summary['max_height'], topped_out=agent.game_over,
                capped=not agent.game_over and agent.pieces == pieces,
                fitness=fitness, evaluations=evaluations)


def summarize(runs: list[dict]) -> dict:
    return {'mean_fitness': statistics.fmean(r['fitness'] for r in runs),
            'mean_pieces': statistics.fmean(r['pieces'] for r in runs),
            'mean_new_holes': statistics.fmean(r['new_holes'] for r in runs),
            'total_hyperlayers': sum(r['hyperlayers'] for r in runs),
            'survived_to_cap': sum(r['capped'] for r in runs), 'episodes': len(runs)}


def evaluate_task(task: tuple) -> dict:
    parameters, specs, settings = task
    game.configure_backend(settings['backend'])
    game.configure_network(parameters, {'method': 'evaluation'})
    started = time.perf_counter()
    runs = [rollout(spec, settings['pieces'], settings['depth'], settings['beam'], settings['candidates']) for spec in specs]
    return {'parameters': parameters, 'summary': summarize(runs), 'runs': runs,
            'seconds': round(time.perf_counter() - started, 4)}


def encode(parameters: dict) -> list[float]:
    return [math.log(-parameters['holes']), math.log(-parameters['roughness']), math.log(parameters['threshold'])]


def decode(vector: list[float]) -> dict:
    values = [math.exp(min(high, max(low, value))) for value, (low, high) in zip(vector, BOUNDS)]
    return dict(game.BASELINE_PARAMETERS, holes=-values[0], roughness=-values[1], threshold=values[2])


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def train(args: argparse.Namespace) -> dict:
    settings = {'pieces': args.pieces, 'depth': args.depth, 'beam': args.beam,
                'candidates': args.candidates, 'backend': args.backend}
    specs = curriculum(101, args.episodes)
    validation_specs = curriculum(1001, args.validation_episodes)
    if set(s['seed'] for s in specs) & set(s['seed'] for s in validation_specs):
        raise ValueError('Training and validation seeds overlap.')
    run_config = {'settings': settings, 'train_specs': specs, 'validation_specs': validation_specs,
                  'population': args.population, 'seed': args.seed}
    rng = random.Random(args.seed)
    mean, std = encode(game.BASELINE_PARAMETERS), [.9, .7, .5]
    history, previous_best, start_generation = [], None, 0
    if args.resume:
        previous = json.loads(args.resume.read_text(encoding='utf-8'))
        game.load_checkpoint(args.resume)
        if previous['optimizer']['config'] != run_config:
            raise ValueError('Resume must retain the original search settings, datasets, seed and population.')
        state = previous['optimizer']
        mean, std, history = state['mean'], state['std'], state['history']
        previous_best, start_generation = state['best'], state['generation']
        rng.setstate(tuple_tree(state['rng_state']))
    started = time.perf_counter()
    source = Path(game.__file__).read_bytes()
    baseline_task = (dict(game.BASELINE_PARAMETERS), specs, settings)
    workers = min(args.workers, args.population)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        baseline = pool.submit(evaluate_task, baseline_task).result()
        best = previous_best or baseline
        for generation in range(start_generation, start_generation + args.generations):
            vectors = [encode(best['parameters']), list(mean)]
            while len(vectors) < args.population:
                sample = [rng.gauss(m, s) for m, s in zip(mean, std)]
                vectors.append([min(hi, max(lo, value)) for value, (lo, hi) in zip(sample, BOUNDS)])
            tasks = [(decode(vector), specs, settings) for vector in vectors]
            results = list(pool.map(evaluate_task, tasks))
            ranking = sorted(range(len(results)), key=lambda i: results[i]['summary']['mean_fitness'], reverse=True)
            if results[ranking[0]]['summary']['mean_fitness'] > best['summary']['mean_fitness']:
                best = results[ranking[0]]
            elite = [vectors[i] for i in ranking[:max(2, args.population // 4)]]
            elite_mean = [statistics.fmean(v[j] for v in elite) for j in range(3)]
            elite_std = [statistics.pstdev(v[j] for v in elite) for j in range(3)]
            mean = [.3 * old + .7 * fresh for old, fresh in zip(mean, elite_mean)]
            std = [max(.08, .3 * old + .7 * fresh) for old, fresh in zip(std, elite_std)]
            record = {'generation': generation + 1, 'best_fitness': best['summary']['mean_fitness'],
                      'generation_mean_fitness': statistics.fmean(r['summary']['mean_fitness'] for r in results),
                      'parameters': best['parameters'], 'rollout_pieces': sum(r['pieces'] for t in results for r in t['runs']),
                      'evaluations': sum(r['evaluations'] for t in results for r in t['runs'])}
            history.append(record)
            checkpoint = {'schema': 1, 'kind': 'tetris4d-shared-sigmoid', 'well': [16, 16, 16, 16],
                          'parameters': best['parameters'],
                          'training': {'method': 'cross-entropy policy search', 'trained_parameters': ['holes', 'roughness', 'threshold'],
                                       'generation': generation + 1, 'seed': args.seed, 'status': 'not-yet-validated',
                                       'base_commit': BASE_COMMIT, 'source_sha256': hashlib.sha256(source).hexdigest()},
                          'optimizer': {'generation': generation + 1, 'mean': mean, 'std': std,
                                        'rng_state': rng.getstate(), 'history': history, 'best': best, 'config': run_config}}
            atomic_json(args.output, checkpoint)
            print(json.dumps(record), flush=True)
        validation = list(pool.map(evaluate_task, [(dict(game.BASELINE_PARAMETERS), validation_specs, settings),
                                                  (best['parameters'], validation_specs, settings)]))
    improved = validation[1]['summary']['mean_fitness'] > validation[0]['summary']['mean_fitness']
    checkpoint['training']['status'] = 'validation-improved' if improved else 'experimental-no-validation-improvement'
    checkpoint['training']['rollout_pieces'] = sum(r['rollout_pieces'] for r in history)
    checkpoint['training']['candidate_evaluations'] = sum(r['evaluations'] for r in history)
    checkpoint['training']['training_episodes'] = len(history) * args.population * args.episodes
    checkpoint['training']['fitness'] = 'pieces / episode_limit + 0.25 * hyperlayers - 0.25 * added_holes / max(1, cells_placed)'
    checkpoint['training']['note'] = 'Curriculum uses full-size boards with synthetic starting stacks; shaped reward is not proof of longer full games.'
    checkpoint['validation'] = {'baseline': validation[0], 'trained': validation[1], 'improved': improved}
    atomic_json(args.output, checkpoint)
    report = {'passed': True, 'training_baseline': baseline, 'training_champion': best,
              'validation': checkpoint['validation'], 'history': history, 'config': run_config,
              'wall_seconds': round(time.perf_counter() - started, 3),
              'training': checkpoint['training']}
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description='Train shared sigmoid output weights and threshold using seeded gameplay.')
    parser.add_argument('--generations', type=int, default=8)
    parser.add_argument('--population', type=int, default=12)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--validation-episodes', type=int, default=8)
    parser.add_argument('--pieces', type=int, default=128)
    parser.add_argument('--depth', type=int, choices=(1, 2), default=1)
    parser.add_argument('--beam', type=int, default=4)
    parser.add_argument('--candidates', type=int, default=512)
    parser.add_argument('--backend', choices=('python', 'numpy', 'auto'), default='numpy')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, default=Path('models/trained-sigmoid.json'))
    parser.add_argument('--report', type=Path, default=Path('validation/training.json'))
    args = parser.parse_args()
    if min(args.generations, args.episodes, args.validation_episodes, args.pieces, args.workers, args.beam) < 1 or args.population < 4 or args.candidates < 0:
        parser.error('Positive sizes and at least four population members are required.')
    if args.episodes > 900 or args.validation_episodes > 900:
        parser.error('At most 900 episodes per split; use additional independent runs for larger datasets.')
    try:
        game.configure_backend(args.backend)
        result = train(args)
        print(json.dumps({'output': str(args.output), 'validation': result['validation'],
                          'wall_seconds': result['wall_seconds']}, indent=2))
        return 0
    except (ValueError, OSError, KeyError, AssertionError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
