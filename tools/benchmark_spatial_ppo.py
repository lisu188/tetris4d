"""Compare full V4 rollout and PPO updates against a built reference checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'reference': [], 'native': [], 'grouped': ['--group-exhaustive'],
    'actor_kl': ['--actor-only-kl'], 'fixed_features': ['--compact-rollout'],
    'combined': ['--group-exhaustive', '--actor-only-kl', '--compact-rollout'],
}


def fingerprint(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def measure(args):
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import train_survival_v4 as v4

    with tempfile.TemporaryDirectory() as temporary:
        options = v4.arguments([
            '--run-dir', temporary, '--device', 'cuda', '--backend', 'native', '--no-gpu-defaults',
            '--gpu-prepare', '--coalesce-transfers', '--cache-rollout', '--fused-encoding',
            '--pretrained', str(args.incumbent.resolve()), '--preserve-pretrained-critic',
            '--calibration-games', '0', '--allow-uncalibrated', '--regional-critic', '--recover-target',
            '--screening', 'exhaustive', '--fixed-lr', '--seed', str(args.seed), '--pieces', '0',
            '--envs', '32', '--rollout', '256', '--batch', '32', '--optimizer-batch', '256',
            '--rollout-cache-mib', '2048', '--threads', '1', *VARIANTS[args.variant]])
        torch.set_num_threads(1)
        v4.game.configure_backend('native')
        v4.game.load_checkpoint(options.weights)
        with v4.v2.device_lock(torch.device('cuda')), v4.base.Budget(1) as budget:
            trainer = v4.SurvivalTrainer(options, budget)
            assert trainer.calibrate()
            options.rollout = 4
            trainer.update(trainer.rollout(False), False)
            options.rollout = 256
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rollout = trainer.rollout(False)
            trainer.update(rollout, False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = json.loads(trainer.metrics.read_text().splitlines()[-1])
            prep = row['gpu_prepare']
            if row['rollout_placements'] != 8192 or not prep['enabled'] or prep.get('runtime_fallback'):
                raise ValueError('Expected a full 32 x 256 CUDA rollout')
            if '--compact-rollout' in VARIANTS[args.variant] and row['rollout_cache'].get('storage') != 'fixed_features':
                raise ValueError(f"Compact memory path fell back on this device: {row['rollout_cache']}")
            proof = {
                'choices': hashlib.sha256(json.dumps(rollout.choices).encode()).hexdigest(),
                'boards': [v4.base.board_hash(e.state.board) for e in trainer.environments],
                'model': v4.v3.model_digest(trainer.model),
                'gradients': fingerprint((n, p.grad) for n, p in trainer.model.named_parameters() if p.grad is not None),
                'adam': fingerprint((f'{name}/{key}', value) for name, p in trainer.model.named_parameters()
                                    for key, value in sorted(trainer.optimizer.state[p].items())),
                'rng': fingerprint([('cpu', torch.get_rng_state()), ('cuda', torch.cuda.get_rng_state())]),
                'kl': row['final_exact_kl'], 'stopped': row['kl_early_stop'], 'steps': row['optimizer_steps_this_rollout']}
            result = {'seed': args.seed, 'variant': args.variant, 'seconds': seconds,
                      'placements_per_second': 8192 / seconds, 'proof': proof, 'cache': row['rollout_cache'],
                      'preparation': prep, 'terminals': row['terminal_transitions'], 'truncations': row['truncated_transitions'],
                      'phases': row['phase_seconds'], 'peak_reserved_mib': row['peak_reserved_mib'],
                      'gpu': row['device'], 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                      'incumbent': trainer.imported_checkpoint_sha256,
                      'runtime': trainer.signature, 'native_sha256': v4.base.file_hash(args.source/'native/libtetris4d_native.so')}
            v4.base.durable_json(args.output, result)


def compare(args):
    if not args.reference or args.pairs < 1:
        raise ValueError('Supply a built reference checkout and a positive pair count')
    args.output.mkdir(parents=True, exist_ok=False)
    timings = {name: [] for name in VARIANTS}
    vs_native = {name: [] for name in VARIANTS if name != 'reference'}
    env = dict(os.environ, TETRIS4D_GAME_LIBRARY='off', PYTHONDONTWRITEBYTECODE='1')
    for pair in range(args.pairs):
        variants = list(VARIANTS)
        if pair % 2:
            variants.reverse()
        rows = {}
        for name in variants:
            path = args.output/f'{pair+1}-{name}.json'
            source = args.reference if name == 'reference' else ROOT
            command = [sys.executable, '-B', __file__, '--variant', name, '--source', str(source),
                       '--incumbent', str(args.incumbent), '--seed', str(args.seed + pair * 100),
                       '--output', str(path)]
            subprocess.run(command, env=env, check=True)
            rows[name] = json.loads(path.read_text())
        for name, row in rows.items():
            for key in ('seed', 'incumbent', 'gpu', 'torch', 'cuda'):
                if row[key] != rows['reference'][key]:
                    raise ValueError(f'Benchmark inputs differ for {name}: {key}')
            if row['proof'] != rows['reference']['proof']:
                differences = [key for key in row['proof'] if row['proof'][key] != rows['reference']['proof'][key]]
                raise ValueError(f'Pair {pair+1} {name} parity differs for {differences}')
            timings[name].append(rows['reference']['seconds'] / row['seconds'])
            if name in vs_native:
                vs_native[name].append(rows['native']['seconds'] / row['seconds'])
    medians = {k: statistics.median(v) for k, v in timings.items()}
    report = {'pairs': args.pairs, 'speedup_per_pair': timings, 'median_speedup': medians,
              'median_speedup_vs_native': {k: statistics.median(v) for k, v in vs_native.items()},
              'combined_passes': medians['combined'] >= 1.10}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--pairs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=418000)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--source', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.variant:
        measure(args)
    else:
        compare(args)
