"""Compare full V4 rollout and PPO updates against a built reference checkout."""
import argparse
import ast
import base64
from contextlib import ExitStack
import gzip
import hashlib
import json
import math
import os
import shutil
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


def make_late_states(library_root, output):
    """Freeze actual training trajectories before top-out. Only the board and stream are needed to resume."""
    sys.path.insert(0, str(ROOT))
    import train_survival_v4 as v4
    v4.game.configure_backend('native')
    entries = sorted(v4.base.library.iter_index(library_root), key=lambda e: e['id'])
    seen, rows = set(), []
    for entry in entries:
        if (entry.get('source') != 'survival-ppo-v4' or entry.get('phase') != 'training'
                or entry.get('stop_reason') != 'top_out' or not entry.get('game_over')
                or entry['seed'] in seen or entry['placements'] < 4096):
            continue
        record = v4.base.library.load(library_root / entry['file'])
        meta = record.meta
        identity = v4.base.library.GameLibrary.game_id(meta['pieces_sha256'], meta['seed'], meta['piece_set'], record.moves)
        if identity != entry['id'] or len(record.moves) != entry['placements']:
            raise ValueError('Late fixture game identity differs from its index')
        offsets = {len(record.moves) - distance for distance in (512, 384, 256, 128)}
        for index, (state, _move, action) in enumerate(v4.base.library.replay(record, verify=True)):
            if index in offsets:
                data = struct.pack('<4096H', *state.board)
                rows.append({'game_id': identity, 'remaining_recorded': len(record.moves) - index,
                             'summary': state.summary(), 'target': int(action[6]), 'current': state.current,
                             'next_piece': state.next_piece, 'board_sha256': hashlib.sha256(data).hexdigest(),
                             'board_zlib_base64': base64.b64encode(zlib.compress(data)).decode()})
        seen.add(entry['seed'])
        if len(rows) == 32:
            break
    if len(rows) != 32:
        raise ValueError('Eight distinct completed V4 training seeds are required for a late-game timing gate')
    output.write_text(json.dumps({'cases': rows}, separators=(',', ':')) + '\n')


def late_episodes(v4, numbers, path):
    cases = json.loads(path.read_text())['cases']
    if len(cases) != len(numbers):
        raise ValueError('Late workload must provide a board for every environment')
    episodes = []
    for case, number in zip(cases, numbers):
        raw = zlib.decompress(base64.b64decode(case['board_zlib_base64']))
        if hashlib.sha256(raw).hexdigest() != case['board_sha256']:
            raise ValueError('Late fixture board digest differs')
        summary = case['summary']
        state = v4.game.Game(summary['seed'], summary['piece_set'])
        v4.base.library.ignore(state)
        for _ in range(summary['pieces']):
            state.current, state.next_piece = state.next_piece, state.stream.pop()
        state.board = struct.unpack('<4096H', raw)
        state.pieces, state.piece_counts = summary['pieces'], summary['piece_counts']
        state.cells_placed, state.layers, state.score = summary['cells_placed'], summary['hyperlayers'], summary['score']
        if state.summary() != summary or (state.current, state.next_piece) != (case['current'], case['next_piece']):
            raise ValueError('Late fixture counters or piece stream differ from the archive')
        episodes.append(v4.base.Episode(number, state, target=case['target']))
    return episodes


def boundary_events(episodes):
    if not all(e.moves for e in episodes):
        raise ValueError('Boundary states did not both commit a measured action')
    clear, recover = (e.moves[0] for e in episodes)
    if clear['clears'] != 4 or recover.get('retarget_from') != 12 or recover['target_before'] != 15:
        raise ValueError('Expected a four-layer clear and an actual 12-to-15 recovery before the first action')
    return {'clearing_first_move': clear, 'recovery_first_move': recover,
            'recovery_initial_sha256': RECOVERY_BOARD_SHA}


def greedy_benchmark(v4, args):
    """Measure the normal fixed-batch greedy evaluator, including commit, with a per-action proof."""
    from unittest.mock import patch
    from tools import evaluate_afterstate_td as evaluate
    import torch
    device = torch.device('cuda')
    model = evaluate.load_model(args.incumbent.resolve(), device)
    original_select = evaluate.select_moves
    trace = hashlib.sha256()

    def record(model, states, *arguments, **kwargs):
        moves = original_select(model, states, *arguments, **kwargs)
        for index, (target, c) in moves.items():
            state = states[index]
            trace.update(struct.pack('<9i', state.seed, state.pieces, target, c.rotation, c.x, c.y, c.w, c.z, c.layers))
        return moves

    switches = {'two_stage': True, 'headless': True} if args.variant == 'combined' else {}
    seeds = [args.seed + 2000000 + i for i in range(8)]
    with patch.object(evaluate, 'select_moves', record):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = evaluate.evaluate_fresh(model, seeds, policy='ppo', cap=256, device=device,
                                         ppo_batch_size=32, group_exhaustive=True, **switches)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    if result['placements'] != [256] * len(seeds):
        raise ValueError('Expected a full 8 x 256 opening greedy evaluation')
    proof = {'choices_sha256': trace.hexdigest(), **{k: result[k] for k in ('seeds', 'placements', 'clears', 'stop_reasons')}}
    return elapsed, proof


def profile_rollout_step(trainer, device, output, metadata):
    """Export a single 32-environment rollout step; keep all instrumentation outside timed workers."""
    import torch
    from unittest.mock import patch
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        if activities[-1] not in torch.profiler.supported_activities():
            raise RuntimeError('CUDA profiling requires a CUDA-enabled profiler with CUPTI')
        torch.cuda.synchronize()
    trainer.args.rollout = 1
    with torch.profiler.profile(activities=activities) as capture, ExitStack() as ranges:
        capture.add_metadata_json('simulation_benchmark', json.dumps(metadata))
        # Wrap only this worker's warmed preparer, including older revisions without a final selector.
        preparer = trainer._gpu_preparer
        for owner, name in ((trainer, 'prepare_episode'), (preparer, '_prepare_group'),
                            (preparer, '_screen_shortlist'), (preparer, '_final_shortlist')):
            method = getattr(owner, name, None)
            if method is not None:
                labeled = torch.profiler.record_function('simulation/' + name.lstrip('_'))(method)
                ranges.enter_context(patch.object(owner, name, labeled))
        with torch.profiler.record_function('benchmark/rollout_step'):
            rollout = trainer.rollout(False)
            if device == 'cuda':
                torch.cuda.synchronize()
    if len(rollout.choices) != trainer.args.envs:
        raise RuntimeError('Incomplete profiling rollout step')
    if device == 'cuda':
        preparation = trainer._gpu_prepare_status
        stats = preparation.get('preparations', {})
        if (not preparation['enabled'] or preparation.get('runtime_fallback') or not stats.get('batch_calls')
                or sum(stats['cpu_fallbacks'].values()) >= stats['environment_calls']):
            raise RuntimeError('Profile did not exercise CUDA simulation for this workload')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        trace = Path(temporary) / output.name
        capture.export_chrome_trace(str(trace))
        # Newer CUPTI backends can defer the write until the observer is joined.
        if hasattr(capture, 'wait_for_exports'):
            capture.wait_for_exports()
        opener = gzip.open if trace.suffix == '.gz' else open
        with opener(trace, 'rt', encoding='utf-8') as stream:
            events = json.load(stream)['traceEvents']
        if not any(e.get('name') == 'benchmark/rollout_step' for e in events):
            raise RuntimeError('Profiler did not emit the host rollout range')
        if device == 'cuda' and not any(e.get('cat') == 'kernel' for e in events):
            raise RuntimeError('Profiler did not emit any CUDA kernels; check CUPTI availability')
        os.link(trace, output)  # Publish only a complete trace, without replacing any existing file.


def measure(args):
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    import train_survival_v4 as v4

    with tempfile.TemporaryDirectory() as temporary:
        switches = VARIANTS[args.variant]
        device = args.profile_step or 'cuda'
        if args.compare_simulation or args.profile_step:
            switches = VARIANTS['combined'].copy()
            if args.variant == 'combined':
                switches.extend(('--two-stage-exhaustive', '--headless-commit'))
        backend_flags = ['--gpu-prepare', '--coalesce-transfers', '--cache-rollout', '--fused-encoding']
        if device == 'cpu':
            backend_flags = ['--coalesce-transfers', '--cache-rollout']
        options = v4.arguments([
            '--run-dir', temporary, '--device', device, '--backend', 'native', '--no-gpu-defaults',
            *backend_flags,
            '--pretrained', str(args.incumbent.resolve()), '--preserve-pretrained-critic',
            '--calibration-games', '0', '--allow-uncalibrated', '--regional-critic', '--recover-target',
            '--screening', 'exhaustive', '--fixed-lr', '--seed', str(args.seed), '--pieces', '0',
            '--envs', '32', '--rollout', '256', '--batch', '32', '--optimizer-batch', '256',
            '--rollout-cache-mib', str(BOUNDARY_CACHE_MIB if args.workload == 'boundaries' else 2048),
            '--threads', '1', *switches])
        torch.set_num_threads(1)
        v4.game.configure_backend('native')
        v4.game.load_checkpoint(options.weights)
        with v4.v2.device_lock(torch.device(device)), v4.base.Budget(1) as budget:
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
                initial_clears = trainer.progress['clears']
                initial_recoveries = trainer.progress['target_recoveries']
            if args.workload == 'late':
                trainer.environments[:] = late_episodes(v4, [e.number for e in trainer.environments], args.late_states)
                trainer.prepared = trainer._prepared_gpu_batch = None
            if args.profile_step:
                metadata = dict(device=device, workload=args.workload, variant=args.variant, seed=args.seed,
                                environments=options.envs, rollout_steps=1, warmup_steps=4,
                                checkpoint_sha256=trainer.imported_checkpoint_sha256,
                                simulator_sha256=_sha(args.source/'tetris_gpu_simulation.py'))
                profile_rollout_step(trainer, device, args.output, metadata)
                print(f'Wrote {device.upper()} rollout trace to {args.output}')
                return
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
            if '--compact-rollout' in switches:
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
            if args.workload == 'late':
                proof['late_fixture_sha256'] = hashlib.sha256(args.late_states.read_bytes()).hexdigest()
            if boundaries:
                proof['boundaries'] = {**boundary_events(boundaries),
                                       'measured_clears': row['clears'] - initial_clears,
                                       'measured_target_recoveries': row['target_recoveries'] - initial_recoveries}
                if proof['boundaries']['measured_clears'] < 4 or proof['boundaries']['measured_target_recoveries'] < 1:
                    raise ValueError('Training telemetry did not count both measured boundary events')
            result = {'seed': args.seed, 'variant': args.variant, 'workload': args.workload, 'seconds': seconds,
                      'placements_per_second': 8192 / seconds, 'proof': proof, 'cache': row['rollout_cache'],
                      'preparation': row['gpu_prepare'], 'terminals': row['terminal_transitions'],
                      'truncations': row['truncated_transitions'],
                      'phases': row['phase_seconds'], 'peak_reserved_mib': row['peak_reserved_mib'],
                      'gpu': row['device'], 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                      'incumbent': trainer.imported_checkpoint_sha256,
                      'runtime': trainer.signature, 'native_sha256': v4.base.file_hash(args.source/'native/libtetris4d_native.so')}
            if args.compare_simulation:
                counters = getattr(trainer._gpu_preparer, 'two_stage_counts', {})
                packed_envs = sum(getattr(e.state, 'packed_board', None) is not None for e in trainer.environments)
                result['simulation'] = {'switches': switches, 'two_stage': counters, 'packed_environments': packed_envs}
                if args.variant == 'combined' and (not counters.get('screened_groups') or not packed_envs):
                    raise ValueError(f'Candidate runtime did not exercise both paths: {result["simulation"]}')
            if args.compare_simulation and args.workload == 'opening':
                # Release rollout storage before constructing an independent greedy evaluator.
                del rollout, trainer
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                greedy_seconds, greedy_proof = greedy_benchmark(v4, args)
                result['greedy_seconds'] = greedy_seconds
                result['proof']['greedy'] = greedy_proof
            v4.base.durable_json(args.output, result)


def compare(args):
    if not args.reference or args.pairs < 1:
        raise ValueError('Supply a built reference checkout and a positive pair count')
    args.output.mkdir(parents=True, exist_ok=False)
    names = ('reference', 'combined') if args.workload == 'boundaries' or args.compare_simulation else tuple(VARIANTS)
    timings = {name: [] for name in names}
    greedy_timings = []
    vs_native = {name: [] for name in names if 'native' in names and name != 'reference'}
    env = dict(os.environ, TETRIS4D_GAME_LIBRARY='off', PYTHONDONTWRITEBYTECODE='1')
    for pair in range(args.pairs):
        variants = list(names)
        if pair % 2:
            variants.reverse()
        rows = {}
        for name in variants:
            path = args.output/f'{pair+1}-{name}.json'
            source = args.reference if name == 'reference' else args.candidate
            command = [sys.executable, '-B', __file__, '--variant', name, '--source', str(source),
                       '--incumbent', str(args.incumbent), '--workload', args.workload, '--seed', str(args.seed + pair * 100),
                       '--output', str(path)]
            if args.compare_simulation:
                command.append('--compare-simulation')
            if args.late_states:
                command.extend(('--late-states', str(args.late_states.resolve())))
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
        if 'greedy_seconds' in rows['reference']:
            greedy_timings.append(rows['reference']['greedy_seconds'] / rows['combined']['greedy_seconds'])
        print(f'Pair {pair+1} {args.workload}: {timings["combined"][-1]:.3f}x; parity passed', flush=True)
    medians = {k: statistics.median(v) for k, v in timings.items()}
    report = {'workload': args.workload, 'pairs': args.pairs, 'speedup_per_pair': timings, 'median_speedup': medians,
              'median_speedup_vs_native': {k: statistics.median(v) for k, v in vs_native.items()},
              'combined_passes': medians['combined'] >= 1.10}
    if greedy_timings:
        report.update(greedy_speedup_per_pair=greedy_timings, greedy_median_speedup=statistics.median(greedy_timings))
    (args.output/'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


BASELINE_REVISION = '39292969e031226c68e53757ed4379ee5c9f2dee'
# These blobs are from the already optimized baseline commit. The generated library also
# carries the source/binary digests from tools/build_native.py.
BASELINE_FILES = {
    'native/tetris_native.c': '46684425837fa0310dc214ddd7e57c79eec994d09833f761da3333f700518af3',
    'tetris_gpu_simulation.py': 'a9f82f8978d2b3cdaaa94bf3b69c0872c5728528c205104c372197ebf32310ab',
    'tetris_neural.py': '7b17ba11a00de24e8d52e468e3805b05d9f6bfe535d15b678ad91806314b81c8',
    'sparse_tetris_4d_optimized.py': '8e4db008a170f4e91e6316f6d2fd9868260c5a6d376951c458ca1556f0cc434a',
    'train_overnight.py': '224c1f88cb3c48602d42073819ccdf2c2c4cb16c14c879a31476c1301ee36887',
}
LIBRARY = 'native/libtetris4d_native.so'
WORKLOADS = ('opening', 'boundaries', 'late')
RUNTIME_SIGNATURE_FILES = {
    'neural_sha256': 'tetris_neural.py', 'engine_sha256': 'sparse_tetris_4d_optimized.py',
    'runner_sha256': 'train_overnight.py', 'gpu_preparer_sha256': 'tetris_gpu_simulation.py',
}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_simulation(root):
    """Prepare the small, immutable simulator runtimes required by the timing and replay gates."""
    root.mkdir(parents=True, exist_ok=True)
    reference = root/'reference-3929296'
    candidate = root/'candidate'
    if reference.exists() or candidate.exists():
        raise ValueError('Use a fresh snapshot directory to preserve existing benchmark evidence')
    with tempfile.TemporaryDirectory(dir=root) as temp:
        staged = Path(temp)
        ref, new = staged/reference.name, staged/candidate.name
        files = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', BASELINE_REVISION],
                                        text=True).splitlines()
        for name in files:
            if not ((name.endswith('.py') and not name.startswith('tests/')) or name in ('models/trained-sigmoid.json', 'native/tetris_native.c')):
                continue
            path = ref/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{BASELINE_REVISION}:{name}']))
        (ref/'SOURCE_REVISION').write_text(BASELINE_REVISION + '\n')
        subprocess.run([sys.executable, '-B', str(ref/'tools/build_native.py')], check=True)
        manifest = {'base_revision': BASELINE_REVISION,
                    'code_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    'sha256': {}}
        files = [*ROOT.glob('*.py'), *(ROOT/'tools').glob('*.py'), ROOT/'models/trained-sigmoid.json',
                 ROOT/'native/tetris_native.c', ROOT/LIBRARY, ROOT/(LIBRARY + '.json')]
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            target = new/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest['sha256'][name] = _sha(target)
        (new/'SOURCE.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
        _manifest_hashes(staged, ROOT)
        ref.rename(reference)
        new.rename(candidate)


def _simulation_ast(path):
    # Interactive UI entry points do not participate in the simulator or trainer.
    exclude = {'sparse_tetris_4d_optimized.py': {'main', 'launch_demo'}, 'tetris_neural.py': {'NeuralAgent'}}.get(path.name, set())
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if getattr(node, 'name', None) not in exclude]
    return ast.dump(tree, include_attributes=False)


def _manifest_hashes(root, runtime_root):
    reference, candidate = root/'reference-3929296', root/'candidate'
    if (reference/'SOURCE_REVISION').read_text().strip() != BASELINE_REVISION:
        raise ValueError('Expected the pinned optimized Spatial PPO baseline')
    if any(_sha(reference/name) != digest for name, digest in BASELINE_FILES.items()):
        raise ValueError('Frozen reference source differs from the pinned baseline')
    manifest = json.loads((candidate/'SOURCE.json').read_text())
    if manifest['base_revision'] != BASELINE_REVISION:
        raise ValueError('Candidate snapshot used another base revision')
    for name in (*BASELINE_FILES, LIBRARY):
        if _sha(candidate/name) != manifest['sha256'][name]:
            raise ValueError(f'Frozen candidate file differs from its manifest: {name}')
        if _sha(runtime_root/name) != manifest['sha256'][name]:
            if not name.endswith('.py') or _simulation_ast(runtime_root/name) != _simulation_ast(candidate/name):
                raise ValueError(f'Current simulation runtime differs from the measured candidate: {name}')
    for source in (reference, candidate, runtime_root):
        built = json.loads((source/(LIBRARY + '.json')).read_text())
        if built['source_sha256'] != _sha(source/'native/tetris_native.c') or built['binary_sha256'] != _sha(source/LIBRARY):
            raise ValueError(f'Native build does not match its manifest in {source}')
    return reference, candidate


def _available_receipts(root):
    paths = [path for w in WORKLOADS for i in range(1, 6) for v in ('reference', 'combined')
             if (path := root/w/f'{i}-{v}.json').is_file()]
    return {str(path.relative_to(root)): _sha(path) for path in paths}


def _expected_gate(root, runtime_root, device_name, incumbent_sha):
    reference, snapshot = _manifest_hashes(root, runtime_root)
    sources = {'reference': reference, 'combined': snapshot}
    signatures = {name: {key: _sha(source / file) for key, file in RUNTIME_SIGNATURE_FILES.items()}
                  for name, source in sources.items()}
    binaries = {name: _sha(source/LIBRARY) for name, source in sources.items()}
    late_hash = _sha(root/'late-states.json')
    report = {'baseline_revision': BASELINE_REVISION, 'candidate_manifest_sha256': _sha(snapshot/'SOURCE.json'),
              'incumbent': incumbent_sha, 'device': device_name, 'parity': True, 'workloads': {},
              'receipt_sha256': _available_receipts(root)}
    passed = True
    for workload in WORKLOADS:
        directory = root / workload
        measured, evaluation = [], []
        aggregate = {variant: [] for variant in sources}
        for pair in range(1, 6):
            rows = {}
            for variant in sources:
                row = json.loads((directory/f'{pair}-{variant}.json').read_text())
                runtime = row['runtime']
                staged = variant == 'combined'
                if (row['workload'] != workload or row['variant'] != variant or row['seed'] != 418000 + (pair - 1)*100
                        or row['incumbent'] != incumbent_sha or row['gpu'] != device_name
                        or row['native_sha256'] != binaries[variant] or any(runtime[k] != h for k, h in signatures[variant].items())
                        or bool(runtime.get('two_stage_exhaustive')) != staged or bool(runtime.get('headless_commit')) != staged
                        or runtime['envs'] != 32 or runtime['rollout'] != 256
                        or not all(runtime[k] for k in ('group_exhaustive', 'compact_rollout', 'actor_only_kl'))):
                    raise ValueError(f'Unmatched benchmark inputs or source in {workload}/{pair}/{variant}')
                if staged and (row['simulation']['two_stage']['screened_groups'] < 1 or not row['simulation']['packed_environments']):
                    raise ValueError('Candidate did not exercise both fast paths')
                exact = row['proof']
                if (not all(exact.get(k) for k in ('boards', 'choices', 'model', 'gradients', 'adam', 'rng', 'steps'))
                        or len(exact['boards']) != 32 or not isinstance(exact.get('stopped'), bool) or 'kl' not in exact):
                    raise ValueError('Full PPO parity proof is incomplete')
                if workload == 'late' and exact.get('late_fixture_sha256') != late_hash:
                    raise ValueError('Late-game fixture identity differs')
                if workload == 'boundaries' and (row['cache'].get('fixed_fallback') != 'memory_bound'
                        or exact['boundaries']['measured_clears'] < 4 or exact['boundaries']['measured_target_recoveries'] < 1):
                    raise ValueError('Boundary workload missed required events')
                if workload == 'opening' and (exact['greedy'].get('placements') != [256] * 8 or not exact['greedy']['choices_sha256']):
                    raise ValueError('Greedy trace proof is incomplete')
                durations = [row['seconds']]
                if workload == 'opening':
                    durations.append(row['greedy_seconds'])
                if any(type(t) not in (float, int) or not math.isfinite(t) or t <= 0 for t in durations):
                    raise ValueError('Timing receipts must contain positive, finite wall times')
                rows[variant] = row
                aggregate[variant].append(row)
            ref, candidate = rows['reference'], rows['combined']
            if (ref['proof'] != candidate['proof'] or ref['preparation']['preparations'] != candidate['preparation']['preparations']
                    or any(ref[key] != candidate[key] for key in ('torch', 'cuda', 'terminals', 'truncations'))):
                raise ValueError(f'Pair {workload}/{pair} failed parity')
            measured.append(ref['seconds'] / candidate['seconds'])
            if workload == 'opening':
                evaluation.append(ref['greedy_seconds'] / candidate['greedy_seconds'])
        median = statistics.median(measured)
        passed = passed and median >= 1.10
        item = {'pairs': 5, 'per_pair': measured, 'median_speedup': median, 'passes': median >= 1.10}
        for variant, rows in aggregate.items():
            item[variant] = {
                'median_placements_per_second': statistics.median(8192 / row['seconds'] for row in rows),
                'median_reserved_mib': statistics.median(row['peak_reserved_mib'] for row in rows),
                'median_phase_seconds': {k: statistics.median(row['phases'][k] for row in rows)
                                         for k in rows[0]['phases']},
            }
        if evaluation:
            item['greedy'] = {'per_pair': evaluation, 'median_speedup': statistics.median(evaluation),
                              'median_placements_per_second': {
                                  variant: statistics.median(2048 / row['greedy_seconds'] for row in rows)
                                  for variant, rows in aggregate.items()}}
        report['workloads'][workload] = item
    report['enabled'] = {'two_stage': passed, 'headless': passed}
    greedy = passed and report['workloads']['opening']['greedy']['median_speedup'] >= 1.0
    report['greedy_enabled'] = {'two_stage': greedy, 'headless': greedy}
    return report


def _publish_gate(output, report):
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix='.gate-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def report_failed_gate(root, output, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('An explicit stopping reason is required')
    disabled = {'two_stage': False, 'headless': False}
    result = {'baseline_revision': BASELINE_REVISION, 'parity': False, 'failure': reason,
              'receipt_sha256': _available_receipts(root), 'workloads': {}, 'enabled': disabled, 'greedy_enabled': disabled}
    _publish_gate(output, result)


def verified_gate(perf_root, candidate_runtime_root, device, incumbent_sha):
    """Verify the recorded gate using every matched pair and the actual trainer source and GPU."""
    root = Path(perf_root)
    report = json.loads((root/'gate.json').read_text())
    if 'failure' in report:
        if (not isinstance(report['failure'], str) or not report['failure'].strip()
                or any(set(report.get(section, {})) != {'two_stage', 'headless'}
                       or any(flag is not False for flag in report[section].values())
                       for section in ('enabled', 'greedy_enabled')) or report.get('parity') is not False
                or report.get('receipt_sha256') != _available_receipts(root) or report.get('baseline_revision') != BASELINE_REVISION):
            raise ValueError('Invalid unsuccessful simulation gate')
        return report
    import torch
    name = torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else str(device)
    expected = _expected_gate(root, Path(candidate_runtime_root), name, incumbent_sha)
    if report != expected:
        raise ValueError('Performance gate differs from independently verified receipts or current runtime')
    return report


def summarize_simulation_gate(root, output, incumbent_sha):
    device = json.loads((root/'opening/1-reference.json').read_text())['gpu']
    report = _expected_gate(root, ROOT, device, incumbent_sha)
    _publish_gate(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--incumbent', type=Path)
    parser.add_argument('--make-late-states', type=Path, metavar='GAME_LIBRARY')
    parser.add_argument('--freeze-simulation', action='store_true', help='Snapshot pinned reference and current candidate in --output')
    parser.add_argument('--summarize-simulation-gate', type=Path, metavar='RECEIPT_ROOT')
    parser.add_argument('--gate-failure', metavar='REASON')
    parser.add_argument('--late-states', type=Path)
    parser.add_argument('--compare-simulation', action='store_true', help='Compare incumbent V4 simulation against staged selection and native headless commitment')
    parser.add_argument('--profile-step', choices=('cpu', 'cuda'),
                        help='Standalone one-step training rollout trace after warmup, including preparation, actor, commits and bootstrap. '
                             'Use --output TRACE.json or TRACE.json.gz and optionally --variant reference|combined (default combined).')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--pairs', type=int)
    parser.add_argument('--seed', type=int, default=418000)
    parser.add_argument('--workload', choices=('opening', 'boundaries', 'late'), default='opening',
                        help='Boundaries compares reference and combined at a 744 MiB limit with a forced clear and target recovery')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--source', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.profile_step:
        if (args.compare_simulation or args.reference or args.candidate or args.pairs is not None
                or args.freeze_simulation or args.summarize_simulation_gate
                or args.make_late_states or args.gate_failure or args.variant not in (None, 'reference', 'combined')):
            parser.error('--profile-step is standalone; choose --source and --variant reference or combined')
        if not args.output.name.endswith(('.json', '.json.gz')) or args.output.exists() or args.output.is_symlink():
            parser.error('--profile-step requires a new --output TRACE.json or TRACE.json.gz')
        args.variant = args.variant or 'combined'
    else:
        args.candidate = args.candidate or ROOT
        if args.pairs is None:
            args.pairs = 5
    if args.freeze_simulation:
        freeze_simulation(args.output)
        raise SystemExit(0)
    if args.summarize_simulation_gate:
        if args.gate_failure:
            report_failed_gate(args.summarize_simulation_gate, args.output, args.gate_failure)
        elif args.incumbent:
            summarize_simulation_gate(args.summarize_simulation_gate, args.output, _sha(args.incumbent))
        else:
            parser.error('Provide the benchmark --incumbent to independently verify the gate')
        raise SystemExit(0)
    if args.make_late_states:
        os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
        make_late_states(args.make_late_states, args.output)
        raise SystemExit(0)
    if not args.incumbent or (args.workload == 'late' and not args.late_states):
        parser.error('Provide --incumbent and, for the late workload, --late-states')
    if args.variant:
        os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
        measure(args)
    else:
        compare(args)
