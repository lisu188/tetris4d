"""Matched Spatial V4 PPO curriculum experiment with a frozen paired greedy holdout."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_afterstate_td as ev

v4, base = ev.v4, ev.base
INCUMBENT_SHA256 = 'a6470b4ef118aea46969922559ed4f58826b82f0ae08c26ff81acdc0eac84cd9'
TRAIN_SEED = 445000
DEV = tuple(16435001 + 2*i for i in range(8))
HOLDOUT = tuple(26435001 + 2*i for i in range(72))
ARMS = {'control': 4, 'diverse': 1}
UPDATES = 160
PLACEMENTS_PER_UPDATE = 32 * 256
CAP = 64000
EVAL_BATCH = 32
OPTIONS = {'grouped': ['--group-exhaustive'],
           'grouped-actor-kl': ['--group-exhaustive', '--actor-only-kl'],
           'combined': ['--group-exhaustive', '--actor-only-kl', '--compact-rollout']}


def freeze(path, data):
    if path.exists():
        if ev.read_json(path) != data:
            raise ValueError(f'Frozen experiment record differs at {path}')
    else:
        base.durable_json(path, data)


def trainer_args(out, incumbent, arm, optimizations, *, seed=TRAIN_SEED, danger_per_game=None,
                 selection_seeds=DEV + HOLDOUT, extra=()):
    per_game = ARMS[arm] if danger_per_game is None else danger_per_game
    return v4.arguments([
        '--run-dir', str(out / arm), '--pretrained', str(incumbent), '--preserve-pretrained-critic',
        '--device', 'cuda', '--backend', 'native', '--no-gpu-defaults', '--gpu-prepare', '--coalesce-transfers',
        '--cache-rollout', '--fused-encoding', '--rollout-cache-mib', '2048', '--precision', 'fp32', '--threads', '1',
        '--seed', str(seed), '--calibration-games', '0', '--allow-uncalibrated', '--pieces', '0',
        '--danger-starts', '.25', '--distinct-danger-seeds', '--danger-pool', '32', '--danger-per-game', str(per_game),
        '--danger-window', '500', '3000', '--recover-target', '--regional-critic', '--screening', 'exhaustive',
        '--policy-mode', 'teacher-prior', '--teacher-prior-scale', '.125', '--fixed-lr', '--lr', '5e-5',
        '--critic-lr', '.0003', '--entropy', '.001', '--gae-lambda', '.999', '--ppo-epochs', '2',
        '--clip-ratio', '.1', '--target-kl', '.01', '--envs', '32', '--rollout', '256', '--batch', '32',
        '--optimizer-batch', '256', '--warmup-updates', '0', '--imitation', '0', '--secondary-weight', '0',
        '--regret-weight', '0', '--defer-selection', '--selection-seeds', *map(str, selection_seeds),
        '--selection-pieces', str(CAP), '--save-seconds', '300', '--save-every', '10', *OPTIONS[optimizations], *extra])


def protocol(incumbent, optimizations, device):
    import triton

    if base.file_hash(incumbent) != INCUMBENT_SHA256:
        raise ValueError('Incumbent checkpoint SHA-256 differs from the frozen starting player')
    runtime = ev.runtime_identity(device)
    runtime['float32_matmul_precision'] = torch.get_float32_matmul_precision()
    runtime['triton'] = triton.__version__
    runtime['compute_capability'] = list(torch.cuda.get_device_capability(device))
    training_sources = {name: base.file_hash(ROOT / name) for name in
                        ('train_survival_v2.py', 'train_survival_v3.py', 'train_survival.py',
                         'native/tetris_native.c', 'tools/run_spatial_curriculum.py')}
    model = ev.load_model(incumbent, torch.device('cpu'))
    return {'schema': 'spatial-ppo-curriculum-v1', 'incumbent_sha256': INCUMBENT_SHA256,
            'starting_model_sha256': v4.v3.model_digest(model), 'weights_sha256': base.file_hash(ROOT/'models/trained-sigmoid.json'),
            'runtime': runtime, 'training_sources': training_sources, 'training_seed': TRAIN_SEED,
            'updates_per_arm': UPDATES, 'placements_per_update': PLACEMENTS_PER_UPDATE, 'danger_per_game': ARMS,
            'danger_fraction': .25, 'optimizations': optimizations, 'development_seeds': list(DEV),
            'holdout_seeds': list(HOLDOUT), 'evaluation_cap': CAP, 'evaluation_batch_size': EVAL_BATCH,
            'development_ranking': ['mean', 'minimum', 'median', 'control_on_exact_tie'],
            'promotion': {'relative_gain_at_least': .05, 'paired_bootstrap_lower_95_above': 0}}


@contextmanager
def charge(out, phase):
    """Measure elapsed exclusive GPU reservation time, including setup, saves and evaluation."""
    path = out / 'gpu-time.json'
    ledger = ev.read_json(path) if path.exists() else []
    entry = {'phase': phase, 'start_unix': time.time(), 'seconds': None}
    ledger.append(entry)
    base.durable_json(path, ledger)
    started = time.monotonic()
    try:
        yield
    finally:
        entry['seconds'] = time.monotonic() - started
        base.durable_json(path, ledger)


def gpu_hours(out):
    ledger = ev.read_json(out / 'gpu-time.json')
    return {'measured_gpu_hours': {phase: sum(e['seconds'] for e in ledger if e['phase'] == phase
                                              and e['seconds'] is not None) / 3600
                                   for phase in ('training', 'development', 'holdout')},
            'unmeasured_interrupted_reservations': sum(e['seconds'] is None for e in ledger)}


class MatchedTrainer(v4.SurvivalTrainer):
    """Finish an entire PPO update before honoring a graceful experiment stop."""
    def __init__(self, args, stop):
        self.experiment_stop = stop
        super().__init__(args, base.Budget(float('inf')))
        self.stop_at_update = UPDATES

    def update(self, rollout, warmup):
        result = super().update(rollout, warmup)
        if self.experiment_stop.expired():
            self.budget.reason = self.experiment_stop.reason
        return result


def fit(out, incumbent, arm, plan, stop):
    root = out / arm
    last = root / 'last.pt'
    receipt = root / 'endpoint.json'
    if receipt.exists():
        endpoint = ev.read_json(receipt)
        if endpoint['checkpoint_sha256'] != base.file_hash(last) or endpoint['plan_sha256'] != ev.digest(plan):
            raise ValueError(f'Frozen checkpoint changed for {arm}')
        return endpoint
    with base.run_lock(root), charge(out, 'training'):
        trainer = MatchedTrainer(trainer_args(out, incumbent, arm, plan['optimizations']), stop)
        initialization = ev.read_json(root / 'initialization.json')
        if (initialization['imported_model_sha256'] != INCUMBENT_SHA256 or not initialization['fresh_optimizer']
                or not initialization['fresh_ppo_progress'] or initialization['model_sha256'] != plan['starting_model_sha256']):
            raise ValueError(f'Initialization differs from the incumbent for {arm}')
        pool = [{'seed': p['state'].seed, 'game_id': p['game_id'], 'offset': p['offset']} for p in trainer.danger_pool]
        counts = Counter(row['seed'] for row in pool)
        if (len(pool) != 32 or len(counts) != 32 // ARMS[arm] or set(counts.values()) != {ARMS[arm]}
                or set(counts).intersection(DEV + HOLDOUT)):
            raise ValueError(f'Unexpected danger pool for {arm}')
        freeze(root / 'curriculum.json', pool)
        if trainer.progress['placements'] != trainer.progress['runner_updates'] * PLACEMENTS_PER_UPDATE:
            raise ValueError(f'A partial training update would unmatch the placement budget for {arm}')
        if stop.expired():
            return None
        summary = trainer.run()
        if stop.expired() and summary['runner_updates'] < UPDATES:
            return None
        if (summary['runner_updates'] != UPDATES or summary['placements'] != UPDATES * PLACEMENTS_PER_UPDATE
                or not summary['optimizer_steps']):
            raise ValueError(f'Training stopped before the matched endpoint for {arm}')
        manifest = ev.read_json(root / 'manifest.json')
        if manifest['checkpoint_sha256'] != base.file_hash(last) or manifest['runner_updates'] != UPDATES:
            raise ValueError(f'Checkpoint and manifest disagree for {arm}')
        endpoint = {'arm': arm, 'checkpoint_sha256': manifest['checkpoint_sha256'], 'plan_sha256': ev.digest(plan),
                    'updates': UPDATES, 'placements': summary['placements'], 'optimizer_steps': summary['optimizer_steps']}
        freeze(receipt, endpoint)
        return endpoint


def choose(results):
    for result in results.values():
        if not result['complete'] or result['seeds'] != list(DEV) or result['cap'] != CAP:
            raise ValueError('Development requires all eight predeclared games')
    return max(ARMS, key=lambda name: tuple(results[name][key] for key in ('mean', 'minimum', 'median')))


def experiment(out, incumbent, plan, device, stop, cache=None):
    cache = out / 'evaluation-cache' if cache is None else cache
    endpoints = {}
    for arm in ARMS:
        if stop.expired():
            return False
        endpoints[arm] = fit(out, incumbent, arm, plan, stop)
        if endpoints[arm] is None:
            return False
    if (out / 'result.json').exists():
        previous = ev.read_json(out / 'result.json')
        selected = ev.read_json(out / 'selected.json')
        if (previous.get('complete') is not True or previous['plan_sha256'] != ev.digest(plan)
                or previous['selected_arm'] != selected['arm']
                or previous['inputs']['candidate'] != endpoints[selected['arm']]['checkpoint_sha256']
                or previous['inputs']['incumbent'] != INCUMBENT_SHA256):
            raise ValueError('Completed result does not match its frozen checkpoints and protocol')
        return True

    def evaluate(path, sha, split, seeds):
        with charge(out, split):
            return ev.evaluate_checkpoint(path, sha, 'ppo', seeds, CAP, split, device, stop, plan['runtime'],
                                          plan['weights_sha256'], cache, ppo_batch_size=EVAL_BATCH,
                                          group_exhaustive=True)
    development = {}
    for arm, endpoint in endpoints.items():
        result = evaluate(out / arm / 'last.pt', endpoint['checkpoint_sha256'], 'development', DEV)
        base.durable_json(out / arm / 'development.json', result)
        if not result['complete']:
            return False
        development[arm] = result
    winner = choose(development)
    selected = {'arm': winner, 'checkpoint_sha256': endpoints[winner]['checkpoint_sha256'], 'plan_sha256': ev.digest(plan),
                'development': {arm: {k: result[k] for k in ('mean', 'minimum', 'median', 'placements')}
                                for arm, result in development.items()}}
    freeze(out / 'selected.json', selected)
    final_paths = {'incumbent': (incumbent, INCUMBENT_SHA256),
                   'candidate': (out / winner / 'last.pt', selected['checkpoint_sha256'])}
    results = {}
    for label, (path, sha) in final_paths.items():
        result = evaluate(path, sha, 'holdout', HOLDOUT)
        base.durable_json(out / f'holdout-{label}.json', result)
        if not result['complete']:
            return False
        results[label] = result
    comparison = ev.paired(results['candidate'], results['incumbent'])
    retain = 'candidate' if comparison['passes'] else 'incumbent'
    base.durable_json(out/'result.json', {'complete': True, 'plan_sha256': ev.digest(plan), 'selected_arm': winner,
                      'inputs': {label: sha for label, (_, sha) in final_paths.items()},
                      'holdout_seeds': list(HOLDOUT), 'cap': CAP, 'paired': comparison,
                      'recommended_checkpoint': str(final_paths[retain][0]), 'recommended_sha256': final_paths[retain][1],
                      **gpu_hours(out)})
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--optimizations', choices=OPTIONS, default='combined')
    parser.add_argument('--cache-dir', type=Path, help='Reusable per-game evaluation cache (default: output/evaluation-cache)')
    args = parser.parse_args(argv)
    out, incumbent = args.output.resolve(), args.incumbent.resolve()
    os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
    torch.set_num_threads(1)
    v4.game.configure_backend('native')
    ev.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    device = ev.neural.device_for('cuda')
    with base.run_lock(out), v4.v2.device_lock(device), base.Budget(float('inf')) as stop:
        plan = protocol(incumbent, args.optimizations, device)
        if not (out / 'protocol.json').exists() and any(p.name != '.runner.lock' for p in out.iterdir()):
            raise ValueError('Use a new output directory or an existing curriculum experiment')
        freeze(out / 'protocol.json', plan)
        success = experiment(out, incumbent, plan, device, stop, args.cache_dir.resolve() if args.cache_dir else None)
        return 0 if success else 130


if __name__ == '__main__':
    raise SystemExit(main())
