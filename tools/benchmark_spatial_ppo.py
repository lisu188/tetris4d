"""Compare full V4 rollout and PPO updates against a built reference checkout."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    'reference': [], 'native': [], 'grouped': ['--group-exhaustive'],
    'actor_kl': ['--actor-only-kl'], 'fixed_features': ['--compact-rollout'],
    'combined': ['--group-exhaustive', '--actor-only-kl', '--compact-rollout'],
}
# At 32 x 256 the production raw records require 740 MiB, the compact slab
# 748.0625 MiB. Between them, the real compact guard retains raw PPO records.
BOUNDARY_CACHE_MIB = 744
RECOVERY_BOARD_SHA = 'b33102044f7dc51f228b28ea30d00af6600c46c263e52d6165ea9d3762ca2b25'


def fingerprint(values):
    digest = hashlib.sha256()
    for name, value in values:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_preparation(row, envs, steps):
    """A normal terminal/recovery row can use native preparation within a CUDA batch."""
    prep = row['gpu_prepare']
    if row['rollout_placements'] != envs * steps or not prep['enabled'] or prep.get('runtime_fallback'):
        raise ValueError('Expected a complete rollout on the requested CUDA preparation backend')
    stats = prep['preparations']
    batches, environments = stats['batch_calls'], stats['environment_calls']
    # The initial batch and each committed step prepare all environments. Multiple
    # episode resets in one step can share one extra preparation before the next draw.
    ends = row['terminal_transitions'] + row['truncated_transitions']
    if not steps + 1 <= batches <= steps + 1 + ends or environments != envs * batches:
        raise ValueError('CUDA preparation counters do not cover the full benchmark rollout')
    fallbacks = stats['cpu_fallbacks']
    per_environment = {'buried_target', 'full_target', 'clear_candidate', 'exhaustive_or_recovery'}
    if (set(fallbacks) - per_environment or sum(fallbacks.values()) >= environments):
        raise ValueError(f'Benchmark used reference-only preparation: {fallbacks}')


def boundary_episodes(v4, numbers):
    # The vertical I from this deterministic piece bag is the only legal placement.
    clearing = v4.game.Game(31, 'expanded')
    v4.base.library.ignore(clearing)
    clearing.board = (0,) + (65535,) * (v4.game.AREA - 1)
    cases = json.loads((ROOT / 'tests/fixtures/blocked-targets.json').read_text())['cases']
    case = next(c for c in cases if c['board_sha256'] == RECOVERY_BOARD_SHA)
    board = zlib.decompress(base64.b64decode(case['board_zlib_base64']))
    if hashlib.sha256(board).hexdigest() != RECOVERY_BOARD_SHA:
        raise ValueError('Recovery fixture board checksum differs')
    summary = case['summary']
    recovery = v4.game.Game(case['seed'], case['piece_set'])
    v4.base.library.ignore(recovery)
    # Reconstruct the remaining bag as well as the next two pieces.
    for _ in range(summary['pieces']):
        recovery.current, recovery.next_piece = recovery.next_piece, recovery.stream.pop()
    recovery.board = struct.unpack('<4096H', board)
    recovery.pieces, recovery.piece_counts = summary['pieces'], summary['piece_counts']
    recovery.cells_placed, recovery.layers, recovery.score = summary['cells_placed'], summary['hyperlayers'], summary['score']
    if (recovery.current, recovery.next_piece) != (case['current'], case['next_piece']) or recovery.summary() != summary:
        raise ValueError('Recovery fixture differs from the recorded game state')
    return (v4.base.Episode(numbers[0], clearing, target=15),
            v4.base.Episode(numbers[1], recovery, target=case['target']))


def boundary_events(episodes):
    if not all(e.moves for e in episodes):
        raise ValueError('Boundary states did not both commit a measured action')
    clear, recover = (e.moves[0] for e in episodes)
    if clear['clears'] != 4 or recover.get('retarget_from') != 12 or recover['target_before'] != 15:
        raise ValueError('Expected a four-layer clear and an actual 12-to-15 recovery before the first action')
    return {'clearing_first_move': clear, 'recovery_first_move': recover,
            'recovery_initial_sha256': RECOVERY_BOARD_SHA}


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
            '--rollout-cache-mib', str(BOUNDARY_CACHE_MIB if args.workload == 'boundaries' else 2048),
            '--threads', '1', *VARIANTS[args.variant]])
        torch.set_num_threads(1)
        v4.game.configure_backend('native')
        v4.game.load_checkpoint(options.weights)
        with v4.v2.device_lock(torch.device('cuda')), v4.base.Budget(1) as budget:
            trainer = v4.SurvivalTrainer(options, budget)
            if not trainer.calibrate():
                raise ValueError('Benchmark initialization failed calibration')
            options.rollout = 4
            trainer.update(trainer.rollout(False), False)
            options.rollout = 256
            boundaries = None
            if args.workload == 'boundaries':
                boundaries = boundary_episodes(v4, [e.number for e in trainer.environments[:2]])
                trainer.environments[:2] = boundaries
                trainer.prepared = trainer._prepared_gpu_batch = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            rollout = trainer.rollout(False)
            trainer.update(rollout, False)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            row = json.loads(trainer.metrics.read_text().splitlines()[-1])
            validate_preparation(row, options.envs, options.rollout)
            cache = row['rollout_cache']
            if '--compact-rollout' in VARIANTS[args.variant]:
                if boundaries:
                    if cache.get('fixed_fallback') != 'memory_bound' or rollout.fixed_cache is not None or not rollout.records:
                        raise ValueError(f'Expected the compact memory guard to retain real raw records: {cache}')
                elif cache.get('storage') != 'fixed_features':
                    raise ValueError(f'Compact memory path fell back on this device: {cache}')
            proof = {
                'choices': hashlib.sha256(json.dumps(rollout.choices).encode()).hexdigest(),
                'boards': [v4.base.board_hash(e.state.board) for e in trainer.environments],
                'model': v4.v3.model_digest(trainer.model),
                'gradients': fingerprint((n, p.grad) for n, p in trainer.model.named_parameters() if p.grad is not None),
                'adam': fingerprint((f'{name}/{key}', value) for name, p in trainer.model.named_parameters()
                                    for key, value in sorted(trainer.optimizer.state[p].items())),
                'rng': fingerprint([('cpu', torch.get_rng_state()), ('cuda', torch.cuda.get_rng_state())]),
                'kl': row['final_exact_kl'], 'stopped': row['kl_early_stop'], 'steps': row['optimizer_steps_this_rollout']}
            if boundaries:
                proof['boundaries'] = boundary_events(boundaries)
            result = {'seed': args.seed, 'variant': args.variant, 'workload': args.workload, 'seconds': seconds,
                      'placements_per_second': 8192 / seconds, 'proof': proof, 'cache': row['rollout_cache'],
                      'preparation': row['gpu_prepare'], 'terminals': row['terminal_transitions'],
                      'truncations': row['truncated_transitions'],
                      'phases': row['phase_seconds'], 'peak_reserved_mib': row['peak_reserved_mib'],
                      'gpu': row['device'], 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                      'incumbent': trainer.imported_checkpoint_sha256,
                      'runtime': trainer.signature, 'native_sha256': v4.base.file_hash(args.source/'native/libtetris4d_native.so')}
            v4.base.durable_json(args.output, result)


def compare(args):
    if not args.reference or args.pairs < 1:
        raise ValueError('Supply a built reference checkout and a positive pair count')
    args.output.mkdir(parents=True, exist_ok=False)
    names = ('reference', 'combined') if args.workload == 'boundaries' else tuple(VARIANTS)
    timings = {name: [] for name in names}
    vs_native = {name: [] for name in names if 'native' in names and name != 'reference'}
    env = dict(os.environ, TETRIS4D_GAME_LIBRARY='off', PYTHONDONTWRITEBYTECODE='1')
    for pair in range(args.pairs):
        variants = list(names)
        if pair % 2:
            variants.reverse()
        rows = {}
        for name in variants:
            path = args.output/f'{pair+1}-{name}.json'
            source = args.reference if name == 'reference' else ROOT
            command = [sys.executable, '-B', __file__, '--variant', name, '--source', str(source),
                       '--incumbent', str(args.incumbent), '--workload', args.workload, '--seed', str(args.seed + pair * 100),
                       '--output', str(path)]
            subprocess.run(command, env=env, check=True)
            rows[name] = json.loads(path.read_text())
        for name, row in rows.items():
            for key in ('seed', 'incumbent', 'workload', 'gpu', 'torch', 'cuda', 'terminals', 'truncations'):
                if row[key] != rows['reference'][key]:
                    raise ValueError(f'Benchmark inputs differ for {name}: {key}')
            if row['preparation']['preparations'] != rows['reference']['preparation']['preparations']:
                raise ValueError(f'Pair {pair+1} {name} preparation coverage differs from the reference')
            if row['proof'] != rows['reference']['proof']:
                differences = [key for key in row['proof'] if row['proof'][key] != rows['reference']['proof'][key]]
                raise ValueError(f'Pair {pair+1} {name} parity differs for {differences}')
            timings[name].append(rows['reference']['seconds'] / row['seconds'])
            if name in vs_native:
                vs_native[name].append(rows['native']['seconds'] / row['seconds'])
    medians = {k: statistics.median(v) for k, v in timings.items()}
    report = {'workload': args.workload, 'pairs': args.pairs, 'speedup_per_pair': timings, 'median_speedup': medians,
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
    parser.add_argument('--workload', choices=('opening', 'boundaries'), default='opening',
                        help='Boundaries compares reference and combined at a 744 MiB limit with a forced clear and target recovery')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--source', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.variant:
        os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
        measure(args)
    else:
        compare(args)
