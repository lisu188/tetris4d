from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import train_survival_v2 as training

base, neural, game = training.base, training.neural, training.game


def snapshot_checkpoint(source, directory):
    source, directory = Path(source).resolve(), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=directory, prefix='.snapshot-')
    os.close(handle)
    temporary = Path(name)
    try:
        base.copy_atomic(source, temporary)
        digest = base.file_hash(temporary)
        destination = directory / f'{digest}.pt'
        if destination.exists():
            if base.file_hash(destination) != digest:
                raise ValueError('Existing immutable checkpoint has changed')
        else:
            os.replace(temporary, destination)
        return destination, digest
    finally:
        temporary.unlink(missing_ok=True)


def source_manifest():
    return {name: base.file_hash(ROOT / name) for name in
            ('tetris_neural.py', 'sparse_tetris_4d_optimized.py', 'train_survival_v2.py', 'train_overnight.py')}


def run_game(seed, limit, piece_set, agent, policy, candidates, mode, action_seed, recorder=None, number=0):
    episode = base.Episode(number, game.Game(seed, piece_set), phase=f'evaluation-{policy}-{mode}')
    generator = torch.Generator(device=agent.device if agent else 'cpu').manual_seed(action_seed)
    started = time.perf_counter()
    times = {'prepare': 0., 'inference_transfer': 0., 'commit': 0.}
    clear_steps, recoveries, agreements, primary_agreements = [], 0, 0, 0
    reason = 'piece_limit'
    while episode.state.pieces < limit:
        start = time.perf_counter()
        state = episode.state
        if policy == 'sustained':
            decision = game.choose_sustained_move(state.board, state.current, episode.target, candidates, state.pieces)
            candidate = decision.move.candidate if decision else None
            times['prepare'] += time.perf_counter() - start
        else:
            config = agent.model.config if agent else neural.Config(candidates=candidates, recover_target=True)
            prepared = neural.prepare(state, episode.target, config)
            times['prepare'] += time.perf_counter() - start
            if prepared is None:
                candidate = None
            else:
                target = int(prepared.arrays['targets'])
                if target != episode.target:
                    episode.pending_retarget, episode.target = episode.target, target
                    recoveries += 1
                start = time.perf_counter()
                if agent:
                    batch = neural.collate([prepared.arrays], agent.device)
                    with torch.inference_mode():
                        prediction = agent.model(batch)
                        logp = training.legacy.categorical_logp(prediction['logits'], batch['valid'])
                        index = int(logp.argmax(-1).item()) if mode == 'greedy' else int(torch.multinomial(logp.exp(), 1, generator=generator).item())
                        winners, _ = training.legacy.utility_targets(batch['features'], batch['valid'])
                        primary_agreements += int(winners[0, index].item())
                else:
                    index = prepared.teacher
                times['inference_transfer'] += time.perf_counter() - start
                agreements += int(index == prepared.teacher)
                candidate = prepared.options[index]
        if candidate is None:
            reason = base.stopped_reason(state)
            break
        start = time.perf_counter()
        move = base.apply_move(episode, candidate, 0, f'{mode}_{policy}')
        times['commit'] += time.perf_counter() - start
        if move.layers:
            clear_steps.append(state.pieces)
        if state.pieces % 2048 == 0:
            print(json.dumps({'event': 'evaluation_progress', 'seed': seed, 'policy': policy, 'mode': mode,
                              'pieces': state.pieces, 'clears': state.layers}), flush=True)
    episode.complete, episode.stop_reason = True, reason
    if recorder:
        recorder.write(episode)
    seconds = time.perf_counter() - started
    return {**episode.state.summary(), 'policy': policy, 'mode': mode, 'action_seed': action_seed,
            'stopped_reason': reason, 'capped': reason == 'piece_limit', 'target_recoveries': recoveries,
            'teacher_agreement': agreements / episode.state.pieces if episode.state.pieces and policy != 'sustained' else None,
            'primary_utility_agreement': primary_agreements / episode.state.pieces if episode.state.pieces and agent else None,
            'clear_steps': clear_steps, 'wall_seconds': seconds,
            'pieces_per_second': episode.state.pieces / seconds, 'phase_seconds': times, 'trace_capture': False}


def compare(args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Comparison output already exists; choose a new immutable output name')
    paths = args.checkpoints
    if not paths:
        paths = [ROOT / 'models/neural-rl.pt', ROOT / 'runs/overnight-survival/best.pt', ROOT / 'runs/survival-ppo/best.pt']
        newest = sorted(ROOT.glob('survival-ppo-inspection-*.pt'))
        if newest:
            paths.append(newest[-1])
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError('All checkpoint paths must exist; supply --checkpoints explicitly for a smaller comparison')
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    device = neural.device_for(args.device)
    artifacts = output.parent / f'{output.stem}.artifacts'
    report = {'schema': 'tetris4d-checkpoint-comparison-v2', 'complete': False, 'seeds': args.seeds,
              'piece_limit': args.pieces, 'piece_set': args.piece_set, 'modes': args.modes,
              'action_seeds': args.action_seeds, 'python': platform.python_version(), 'torch': str(torch.__version__),
              'device': str(device), 'sources': source_manifest(), 'checkpoints': [], 'runs': [],
              'note': 'Paired frozen-checkpoint comparison. Baselines and learned models are distinct policies. No automatic promotion.'}
    with training.device_lock(device, args.allow_concurrent) as competing:
        report['competing_trainers'] = competing
        report['resource_snapshot'] = training.device_telemetry()
        agents = []
        for path in paths:
            snapshot, digest = snapshot_checkpoint(path, artifacts / 'checkpoints')
            agent = neural.NeuralAgent(snapshot, str(device), capture_traces=False)
            agent.model.eval()
            report['checkpoints'].append({'source': str(path), 'snapshot': str(snapshot), 'sha256': digest,
                                          'config': asdict(agent.model.config), 'progress': agent.checkpoint['progress']})
            agents.append((agent, digest))
        budgets = {a.model.config.candidates for a, _ in agents}
        if len(budgets) != 1:
            raise ValueError('Checkpoints use different candidate budgets; compare matching configurations separately')
        candidates = budgets.pop()
        recorder = training.CompressedRecorder(artifacts / 'games', args.record_games, maximum_mib=args.record_max_mib)
        recorder.context = {'evaluation_manifest': str(output), 'sources': report['sources']}
        with base.preserve_rng(), torch.inference_mode():
            for agent, _ in agents:
                prepared = neural.prepare(game.Game(args.seeds[0], args.piece_set), 1, agent.model.config)
                agent.model(neural.collate([prepared.arrays], device))
            if device.type == 'cuda':
                torch.cuda.synchronize()
        base.durable_json(output, report)
        try:
            for seed in args.seeds:
                for baseline in ('sustained', 'sustained-recovery'):
                    result = run_game(seed, args.pieces, args.piece_set, None, baseline, candidates, 'greedy', 0,
                                      recorder, len(report['runs']))
                    report['runs'].append(result)
                    base.durable_json(output, report)
                    print(json.dumps(result), flush=True)
                for agent, digest in agents:
                    for mode in args.modes:
                        for action_seed in ([0] if mode == 'greedy' else args.action_seeds):
                            result = run_game(seed, args.pieces, args.piece_set, agent, 'neural', candidates, mode,
                                              action_seed, recorder, len(report['runs']))
                            result['checkpoint_sha256'] = digest
                            report['runs'].append(result)
                            base.durable_json(output, report)
                            print(json.dumps(result), flush=True)
        except KeyboardInterrupt:
            report['interrupted'] = True
            base.durable_json(output, report)
            raise
        report['complete'] = True
        groups = {}
        for row in report['runs']:
            key = row.get('checkpoint_sha256', row['policy']) + ':' + row['mode']
            groups.setdefault(key, []).append(row)
        report['summary'] = {key: {'games': len(rows), 'mean_pieces': statistics.mean(r['pieces'] for r in rows),
                                  'minimum_pieces': min(r['pieces'] for r in rows),
                                  'median_pieces': statistics.median(r['pieces'] for r in rows),
                                  'total_clears': sum(r['hyperlayers'] for r in rows),
                                  'capped_games': sum(r['capped'] for r in rows),
                                  'pieces_per_second': sum(r['pieces'] for r in rows) / sum(r['wall_seconds'] for r in rows)}
                             for key, rows in groups.items()}
        base.durable_json(output, report)
        print(json.dumps(report['summary'], indent=2))
    return report


def archive(args):
    root, output = args.run_dir.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError('Archive already exists; choose a new immutable destination')
    if output == root or root in output.parents:
        raise ValueError('Archive destination must be outside the run directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with base.run_lock(root):
        files = [p for p in root.iterdir() if p.is_file() and not p.is_symlink() and p.name != '.runner.lock'
                 and p.suffix in ('.pt', '.json', '.jsonl', '.log')]
        if not any(p.suffix == '.pt' for p in files):
            raise ValueError('Run has no checkpoint to archive')
        if args.include_games:
            files.extend(p for p in (root / 'games').rglob('*') if p.is_file() and not p.is_symlink()
                         and (p.name.endswith('.json') or p.name.endswith('.json.gz')))
        manifest = {'schema': 'tetris4d-experiment-archive-v1', 'sources': source_manifest(), 'files': []}
        handle, name = tempfile.mkstemp(dir=output.parent, prefix='.archive-')
        os.close(handle)
        temporary = Path(name)
        try:
            with tarfile.open(temporary, 'w:gz') as package:
                for path in sorted(files):
                    relative = path.relative_to(root).as_posix()
                    before = base.file_hash(path)
                    package.add(path, arcname=relative, recursive=False)
                    if base.file_hash(path) != before:
                        raise ValueError('A file changed during archive creation')
                    manifest['files'].append({'path': relative, 'sha256': before, 'bytes': path.stat().st_size})
                raw = json.dumps(manifest, indent=2, allow_nan=False).encode()
                info = tarfile.TarInfo('archive-manifest.json')
                info.size, info.mtime = len(raw), 0
                package.addfile(info, io.BytesIO(raw))
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    result = {'archive': str(output), 'sha256': base.file_hash(output), 'files': len(files), 'originals_unchanged': True}
    base.durable_json(output.with_name(output.name + '.json'), result)
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Immutable paired policy comparisons and lossless experiment archives')
    sub = parser.add_subparsers(dest='command', required=True)
    test = sub.add_parser('compare')
    test.add_argument('--checkpoints', nargs='+', type=Path)
    test.add_argument('--seeds', nargs='+', type=int, default=[94001, 94003, 94005, 94007])
    test.add_argument('--action-seeds', nargs='+', type=int, default=[101, 102, 103])
    test.add_argument('--modes', nargs='+', choices=('greedy', 'sampled'), default=['greedy', 'sampled'])
    test.add_argument('--pieces', type=int, default=32000)
    test.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    test.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    test.add_argument('--backend', choices=('auto', 'native', 'numpy', 'python'), default='auto')
    test.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    test.add_argument('--threads', type=int, default=4)
    test.add_argument('--output', type=Path, default=ROOT / 'survival-comparison.json')
    test.add_argument('--allow-concurrent', action='store_true')
    test.add_argument('--record-games', action='store_true')
    test.add_argument('--record-max-mib', type=float, default=1024)
    save = sub.add_parser('archive')
    save.add_argument('--run-dir', type=Path, required=True)
    save.add_argument('--output', type=Path, required=True)
    save.add_argument('--include-games', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'compare':
        if args.pieces < 1 or args.threads < 1 or args.record_max_mib <= 0:
            parser.error('pieces, threads and recording quota must be positive')
        if len(set(args.seeds)) != len(args.seeds) or len(set(args.action_seeds)) != len(args.action_seeds):
            parser.error('Game seeds and action seeds must not contain duplicates')
    try:
        (compare if args.command == 'compare' else archive)(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Experiment failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
