"""Calibrate the afterstate policy on historical positions and evaluate fresh, paired games."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural
import train_survival_v4 as v4

base, game = v4.base, v4.game
TOP_K = 16
POLICY_ORDER = ('actor', 'frozen', 'ppo', 'previous_td', 'td')
DEVELOPMENT_SEEDS = tuple(910001 + 2*i for i in range(8))
HOLDOUT_SEEDS = tuple(7310001 + 2*i for i in range(16))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def runtime_identity(device):
    paths = [Path(__file__), ROOT/'native/libtetris4d_native.so']
    paths += [ROOT/(name+'.py') for name in ('tetris_afterstate_td', 'tetris_neural', 'tetris_survival_v4',
              'sparse_tetris_4d_optimized', 'tetris_gpu_simulation', 'tetris_gpu_encoding',
              'tetris_gpu_spatial', 'tetris_gpu_board_features', 'tetris_survival_v5', 'train_overnight',
              'train_survival_v4', 'tetris_game_library')]
    return {'files': {path.name: base.file_hash(path) for path in paths},
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(device), 'numpy': np.__version__}


def load_model(path, device):
    model, payload = neural.load_checkpoint(path, device)
    if payload['teacher_parameters'] != game.network_parameters():
        raise ValueError('Checkpoint and candidate preparation use different teacher weights')
    if model.config.screening != 'exhaustive':
        model.config = replace(model.config, screening='exhaustive')
    return model.eval()


def population_beta(spreads):
    if not spreads or not all(math.isfinite(x) and x >= 0 for x in spreads):
        raise ValueError('Invalid afterstate value spreads')
    sigma = statistics.median(spreads)
    if sigma <= 0 or not math.isfinite(1 / sigma):
        raise ValueError('Top-16 afterstate values have no usable spread')
    return sigma, 1 / sigma


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_cached(path, identity):
    if not path.is_file():
        return None
    record = read_json(path)
    if record['identity'] != identity:
        raise ValueError(f'Cached evaluation identity differs in {path}')
    if record['data_sha256'] != digest(record['data']):
        raise ValueError(f'Cached evaluation data digest differs in {path}')
    return record['data']


def write_cache(path, identity, data):
    base.durable_json(path, {'identity': identity, 'data': data, 'data_sha256': digest(data)})


@torch.inference_mode()
def calibrate(args, device, budget, runtime):
    import tetris_afterstate_td as td
    from tetris_gpu_simulation import GpuPreparer

    report = read_json(args.positions_report)
    positions = [tuple(row) for row in report['position_ids']]
    source_hash = base.file_hash(args.source)
    if report.get('schema') != 'tetris4d-survival-benchmark-v1':
        raise ValueError('Expected a survival benchmark position report')
    if len(positions) != 32 or len(set(positions)) != 32 or report['checkpoint_sha256'] != source_hash:
        raise ValueError('Expected the 32 distinct development positions for the supplied checkpoint')
    grouped = defaultdict(set)
    for game_id, offset in positions:
        if type(offset) is not int or offset < 0:
            raise ValueError('Invalid calibration offset')
        grouped[game_id].add(offset)
    library = v4.library_root(args.library)
    files = {entry['id']: library/entry['file'] for entry in base.library.iter_index(library)
             if entry['id'] in grouped}
    if files.keys() != grouped.keys():
        raise ValueError('Calibration games missing from the library')
    model = td.from_v4(load_model(args.source, device)).to(device).eval()
    preparer = GpuPreparer(model.base.config, device)
    identity = {'source_sha256': source_hash, 'position_report_sha256': base.file_hash(args.positions_report),
                'weights_sha256': base.file_hash(args.weights), 'runtime': runtime,
                'screening': 'exhaustive', 'top_k': TOP_K}
    cache = args.cache_dir/digest(identity)
    spreads = {}
    for game_id, offsets in grouped.items():
        key = dict(identity, game_id=game_id, offsets=sorted(offsets), record_sha256=base.file_hash(files[game_id]))
        path = cache/(digest(key)+'.json')
        values = checked_cached(path, key)
        if values is None:
            if budget.expired():
                return False
            states, targets, numbers = [], [], []
            record = base.library.load(files[game_id])
            for index, (state, _, move_row) in enumerate(base.library.replay(record, verify=False)):
                # Only the board and piece stream are needed for headless replay.
                state.locked_pieces = ()
                if budget.expired():
                    return False
                if index in offsets:
                    states.append(copy.deepcopy(state))
                    targets.append(int(move_row[6]))
                    numbers.append(index)
                if len(numbers) == len(offsets):
                    break
            if set(numbers) != offsets:
                raise ValueError(f'Calibration position missing in game {game_id}')
            result = preparer.prepare_batch(states, targets)
            if list(result.environment_indices) != list(range(len(states))):
                raise ValueError('Calibration position has no legal candidate')
            pool = model.prepare_device(result, states)
            scores = model.score(pool.features)
            values = []
            for i, number in enumerate(numbers):
                valid = scores[i][pool.valid[i]]
                if len(valid) < 1:
                    raise ValueError('Empty top-16 calibration shortlist')
                values.append([number, float(valid.double().std(unbiased=False))])
            write_cache(path, key, values)
            print(f'Calibrated {len(spreads)+len(values)}/32 positions', flush=True)
        spreads.update({(game_id, number): value for number, value in values})
    ordered = [spreads[pos] for pos in positions]
    sigma, beta = population_beta(ordered)
    base.durable_json(args.output, {**identity, 'schema': 'afterstate-td-beta-v1', 'complete': True,
                                   'beta': beta, 'sigma': sigma, 'std_ddof': 0,
                                   'position_ids': positions, 'spreads': ordered})
    return True


def summarize(rows, seeds, policy, cap):
    rows = [dict(rows[s]) for s in seeds if s in rows]
    for row in rows:
        row['clears_per_1k'] = 1000 * row['clears'] / row['placements'] if row['placements'] else None
    placements = [r['placements'] for r in rows]
    total = sum(placements)
    clears_per_1k = 1000 * sum(r['clears'] for r in rows) / total if total else None
    return {'policy': policy, 'cap': cap, 'complete': len(rows) == len(seeds),
            'requested_seeds': list(seeds), 'seeds': [r['seed'] for r in rows],
            'placements': placements, 'clears': [r['clears'] for r in rows], 'games': rows,
            'clears_per_1k': clears_per_1k,
            'stop_reasons': [r['reason'] for r in rows], 'censored': sum(r['reason']=='cap' for r in rows),
            'mean': statistics.mean(placements) if rows else None,
            'median': statistics.median(placements) if rows else None, 'minimum': min(placements, default=None)}


def select_moves(model, states, targets, scoring, device, preparer):
    """Map live batch indices to their (recovered target, chosen placement)."""
    if preparer is None:
        moves = {}
        config = model.base.config if scoring else model.config
        for index, (state, target) in enumerate(zip(states, targets)):
            prepared = neural.prepare(state, target, config)
            if prepared is None:
                continue
            if scoring:
                pool = model.prepare_cpu(prepared, state)
                choice = int(pool.indices[0, model.choose(pool).item()])
            else:
                choice = int(model(neural.collate([prepared.arrays], device))['logits'].argmax(1).item())
            moves[index] = (int(prepared.arrays['targets']), prepared.options[choice])
        return moves
    result = preparer.prepare_batch(states, targets)
    if not result.environment_indices:
        return {}
    if scoring:
        pool = model.prepare_device(result, states)
        actions = pool.indices.gather(1, model.choose(pool)[:, None]).squeeze(1)
    else:
        actions = model(result.batch)['logits'].argmax(-1)
    return {i: (int(result.targets[i]), candidate)
            for i, candidate in zip(result.environment_indices, result.selected_candidates(actions))}


@torch.inference_mode()
def evaluate_fresh(model, seeds, *, policy, cap, device, top_k=TOP_K, deadline=None,
                   stop_requested=None, on_finish=None):
    """Fresh greedy lockstep evaluation. Per-game time includes concurrent lockstep peers."""
    import tetris_afterstate_td as td

    seeds = tuple(seeds)
    if top_k != TOP_K or cap < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Expected a positive cap, distinct fresh seeds and top_k=16')
    if policy not in POLICY_ORDER:
        raise ValueError('Unknown evaluation policy')
    began = time.monotonic()
    original_model = model
    was_training = model.training
    model.eval()
    try:
        if policy == 'frozen':
            model = td.from_v4(model).to(device).eval()
        scoring = policy in ('frozen', 'previous_td', 'td')
        preparer = None
        if torch.device(device).type == 'cuda':
            from tetris_gpu_simulation import GpuPreparer
            preparer = GpuPreparer(model.base.config if scoring else model.config, device)
        states = {seed: game.Game(seed, 'expanded') for seed in seeds}
        for state in states.values():
            base.library.ignore(state)
        targets, rows, live = dict.fromkeys(seeds, 1), {}, list(seeds)
        while live:
            if (deadline is not None and time.monotonic() >= deadline) or (stop_requested and stop_requested()):
                break
            current = [states[s] for s in live]
            moves = select_moves(model, current, [targets[s] for s in live], scoring, device, preparer)
            playable = [live[i] for i in moves]
            for i, (target, candidate) in moves.items():
                seed, state = live[i], current[i]
                targets[seed] = target
                move = game.materialize_move(state.board, candidate)
                state.commit(move)
                state.locked_pieces = ()  # Rendering history is unused by headless decisions.
                if move.layers:
                    following = game.next_hyperlayer_target(state.board, target)
                    if following is not None:
                        targets[seed] = following
            for seed in live:
                state = states[seed]
                if seed in playable and state.pieces < cap:
                    continue
                reason = 'cap' if state.pieces >= cap else ('top_out' if state.game_over else 'target_or_no_move')
                row = {'seed': seed, 'placements': state.pieces, 'clears': state.layers,
                       'reason': reason, 'wall_seconds': time.monotonic()-began}
                rows[seed] = row
                if on_finish:
                    on_finish(row)
            live = [seed for seed in playable if seed not in rows]
        return {**summarize(rows, seeds, policy, cap), 'wall_seconds': time.monotonic()-began}
    finally:
        original_model.train(was_training)


def development_score(result):
    if not result['complete'] or tuple(result['seeds']) != DEVELOPMENT_SEEDS:
        raise ValueError('Checkpoint selection requires completed development games')
    return result['mean'], result['median']


def paired(candidate, control):
    if (not candidate['complete'] or not control['complete'] or candidate['seeds'] != control['seeds']
            or candidate['cap'] != control['cap']):
        raise ValueError('Paired analysis requires identical completed seeds and caps')
    changes = np.asarray(candidate['placements'], dtype=float) - np.asarray(control['placements'])
    rng = np.random.default_rng(20261123)
    means = changes[rng.integers(0, len(changes), size=(10000, len(changes)))].mean(axis=1)
    low, high = np.quantile(means, [.025,.975]).tolist()
    gain = float(changes.mean())
    ratio = gain/control['mean'] if control['mean'] else None
    return {'control': control['policy'], 'cap': candidate['cap'], 'mean_difference': gain,
            'relative_gain': ratio, 'paired_bootstrap_95': [low, high], 'wins': int((changes>0).sum()),
            'ties': int((changes==0).sum()), 'passes': ratio is not None and ratio >= .05 and low > 0}


def rank(policies):
    # The final tie-break is fixed before evaluation and independent of cache/insertion order.
    return sorted(policies, key=lambda name: (-policies[name]['mean'], -policies[name]['median'],
                                              POLICY_ORDER.index(name)))


def extension_pair(results):
    order = rank(results)[:2]
    if len(order) == 2 and any(2*results[name]['censored'] >= len(results[name]['seeds']) for name in order):
        return order
    return []


def evaluate_checkpoint(checkpoint, checkpoint_hash, name, seeds, cap, split, device, budget, runtime, weights_hash, cache_dir):
    started = time.monotonic()
    identity = {'policy': name, 'checkpoint': checkpoint_hash, 'weights': weights_hash, 'runtime': runtime,
                'screening': 'exhaustive', 'top_k': TOP_K, 'cap': cap, 'split': split, 'seeds': list(seeds)}
    path = cache_dir/digest(identity)
    rows = {}
    def record(row):
        key = dict(identity, seed=row['seed'])
        write_cache(path/f"{row['seed']}.json", key, row)
        rows[row['seed']] = row
    for seed in seeds:
        row = checked_cached(path/f'{seed}.json', dict(identity, seed=seed))
        if row is not None:
            rows[seed] = row
    missing = [s for s in seeds if s not in rows]
    if missing and not budget.expired():
        start = time.monotonic()
        model = load_model(checkpoint, device)
        if base.file_hash(checkpoint) != checkpoint_hash:
            raise ValueError('Evaluation checkpoint changed during load')
        evaluate_fresh(model, missing, policy=name, cap=cap, device=device, stop_requested=budget.expired, on_finish=record)
        print(f'{name} cap {cap}: {len(rows)}/{len(seeds)} games, {time.monotonic()-start:.1f}s', flush=True)
        del model
    result = summarize(rows, seeds, name, cap)
    result['invocation_wall_seconds'] = time.monotonic()-started
    return result


def select_final(args, device, budget, runtime):
    """Consider the final update alongside the best completed scheduled development trial."""
    run = args.run_dir
    summary = read_json(run/'summary.json')
    selection = read_json(run/'selection.json')
    manifest = read_json(run/'manifest.json')
    cap = manifest['selection_pieces']
    weights_hash = base.file_hash(args.weights)
    if selection['cap'] != cap or weights_hash != manifest['signature']['weights_sha256']:
        raise ValueError('Development cap or teacher weights differ from the training manifest')
    final_path = run/'last.pt'
    final_hash = base.file_hash(final_path)
    if (not summary['complete'] or summary['stop_reason'] != 'time_budget' or not summary['updates']
            or final_hash != summary['last_sha256'] or tuple(selection['seeds']) != DEVELOPMENT_SEEDS):
        raise ValueError('Final selection requires completed training and its pinned checkpoint and development seeds')
    best = selection['best']
    if best:
        if (not selection['complete'] or best != summary['best'] or best['sha256'] != selection['best_sha256']
                or base.file_hash(run/'best.pt') != best['sha256']):
            raise ValueError('Scheduled development winner differs from its selection record')
    elif summary['best'] is not None:
        raise ValueError('Scheduled development winner is missing')
    candidate = evaluate_checkpoint(final_path, final_hash, 'td', DEVELOPMENT_SEEDS, cap, 'development',
                                    device, budget, runtime, weights_hash, args.cache_dir)
    if not candidate['complete']:
        return False
    score = development_score(candidate)
    final_wins = best is None or score > (best['mean'], best['median'])
    winner, winner_hash = (final_path, final_hash) if final_wins else (run/'best.pt', best['sha256'])
    base.durable_json(args.output, dict(complete=True, best_sha256=winner_hash, checkpoint=winner.name,
                      checkpoint_dir=str(run.resolve()), seeds=list(DEVELOPMENT_SEEDS), cap=cap,
                      final_development=candidate, final_sha256=final_hash, scheduled_best=best))
    return True


def evaluate(args, device, budget, runtime):
    split = args.split
    seeds = DEVELOPMENT_SEEDS if split == 'development' else HOLDOUT_SEEDS
    paths = {'actor': args.source, 'frozen': args.source}
    if args.ppo:
        paths['ppo'] = args.ppo
    if args.previous_td:
        paths['previous_td'] = args.previous_td
    if args.td:
        paths['td'] = args.td
    if split == 'holdout':
        if not args.ppo or not args.td or not args.selection:
            raise ValueError('Holdout requires the frozen PPO control and the selected TD checkpoint with selection.json')
        selection = read_json(args.selection)
        if selection.get('complete') is not True or selection.get('best_sha256') != base.file_hash(args.td):
            raise ValueError('Held-out checkpoint was not selected in completed development evaluation')
    weights_hash = base.file_hash(args.weights)
    hashes = {name: base.file_hash(path) for name,path in paths.items()}
    report = {'schema': 'afterstate-td-fresh-eval-v1', 'split': split, 'complete': False, 'seeds': list(seeds),
              'runtime': runtime, 'checkpoints': hashes, 'weights_sha256': weights_hash, 'top_k': TOP_K,
              'selection_sha256': base.file_hash(args.selection) if args.selection else None,
              'screening': 'exhaustive', 'policy_tie_order': list(POLICY_ORDER), 'stages': [], 'clock': 'lockstep elapsed wall seconds per seed'}
    names = list(paths)
    for cap in dict.fromkeys((args.cap, args.max_cap)):
        stage = {}
        report['stages'].append({'cap': cap, 'policies': stage})
        for name in names:
            stage[name] = evaluate_checkpoint(paths[name], hashes[name], name, seeds, cap, split,
                                              device, budget, runtime, weights_hash, args.cache_dir)
            base.durable_json(args.output, report)
            if not stage[name]['complete']:
                return False
        names = extension_pair(stage)
        if not names:
            break
    first = report['stages'][0]['policies']
    report['primary_comparison'] = None
    if 'td' in first:
        report['initial_comparisons'] = [paired(first['td'], first[name]) for name in first if name != 'td']
        final = report['stages'][-1]['policies']
        leader = next(name for name in rank(final) if name != 'td')
        common = final if 'td' in final and leader in final else first
        comparison = paired(common['td'], common[leader])
        report['primary_comparison'] = comparison
        if split == 'holdout':
            report['promotion_gate_passed'] = comparison['passes']
    report['complete'] = True
    base.durable_json(args.output, report)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    calibration = sub.add_parser('calibrate')
    calibration.add_argument('--positions-report', type=Path, required=True)
    calibration.add_argument('--library', type=Path)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--split', choices=('development','holdout'), required=True)
    evaluation.add_argument('--td', type=Path)
    evaluation.add_argument('--ppo', type=Path)
    evaluation.add_argument('--previous-td', type=Path, help='Evaluate the pinned predecessor TD checkpoint as a control')
    evaluation.add_argument('--selection', type=Path)
    evaluation.add_argument('--cap', type=int, default=64000)
    evaluation.add_argument('--max-cap', type=int, default=128000)
    final = sub.add_parser('select-final')
    final.add_argument('--run-dir', type=Path, required=True)
    for command in (calibration, evaluation):
        command.add_argument('--source', type=Path, required=True)
    for command in (calibration, evaluation, final):
        command.add_argument('--weights', type=Path, default=ROOT/'models/trained-sigmoid.json')
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--cache-dir', type=Path)
        command.add_argument('--max-seconds', type=float, default=24*3600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('max-seconds must be finite and positive')
    if args.action == 'evaluate' and not 0 < args.cap <= args.max_cap:
        parser.error('expected positive cap <= max-cap')
    args.cache_dir = args.cache_dir or args.output.parent/'evaluation-cache'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    game.configure_backend('native')
    game.load_checkpoint(args.weights)
    with base.Budget(args.max_seconds / 3600) as budget:
        runtime = runtime_identity(device)
        action = {'calibrate': calibrate, 'evaluate': evaluate, 'select-final': select_final}[args.action]
        complete = action(args, device, budget, runtime)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
