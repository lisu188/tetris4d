"""Replay a single closure decision under paired, publicly possible future draws."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MANIFEST = '297bb0a24b982254b066ca56688aaf59b87f8d4b358a11272c39c8de46779a3a'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def public_remainder(snapshot, names):
    """Infer undrawn pieces after dealing the current piece and its preview."""
    dealt = Counter(snapshot['piece_counts'])
    require(set(dealt) == set(names) and sum(dealt.values()) == snapshot['pieces'], 'Piece counts differ')
    dealt.update((snapshot['current'], snapshot['next_piece']))
    full, partial = divmod(snapshot['pieces'] + 2, len(names))
    require(all(dealt[name] in (full, full + 1) for name in names), 'Invalid bag counts')
    return sorted(name for name in names if partial and dealt[name] == full)


def sample_draws(game, snapshot, domain, seed, repeat, horizon):
    token = f'{domain}|{seed}|{repeat}'.encode('ascii')
    stream = game.PieceStream(int.from_bytes(hashlib.sha256(token).digest(), 'big'), snapshot['piece_set'])
    stream.bag = public_remainder(snapshot, stream.names)
    require(stream.bag == sorted(snapshot['bag']), 'Public and recorded remaining bag membership differ')
    stream.rng.shuffle(stream.bag)
    bag, rng = list(stream.bag), stream.rng.getstate()
    draws = [snapshot['current'], snapshot['next_piece']] + [stream.pop() for _ in range(horizon)]
    return bag, rng, draws


def play(ev, model, snapshot, event, branch, bag, rng, draws, *, uncapped=False):
    game = ev.game
    state = game.Game(snapshot['seed'], snapshot['piece_set'])
    ev.base.library.ignore(state)
    target, _ = ev._resume(state, snapshot)
    require(ev._snapshot(state, target, 0.) == snapshot, 'Root reconstruction differs')
    state.stream.bag = list(bag)
    state.stream.rng.setstate(rng)
    # An independent stream checks every dealt piece, including those beyond the saved prefix.
    oracle = game.PieceStream(0, snapshot['piece_set'])
    oracle.bag = list(bag)
    oracle.rng.setstate(rng)
    expected = draws[:2]
    reason = 'horizon'
    step = 0
    while uncapped or step < len(draws) - 2:
        require([state.current, state.next_piece] == expected, 'Common draw sequence differs')
        prepared = ev.neural.prepare(state, target, model.config)
        if prepared is None:
            reason = 'controller_stop'
            break
        batch = ev.neural.collate([prepared.arrays], 'cpu')
        logits = ev._fixed_actor_logits(model, batch, 32)[0]
        index = int(logits.argmax())
        if step == 0:
            # Check every ranked root score against the original, unshuffled stream.
            require(len(event['rows']) == len(prepared.options) == logits.numel(), 'Root shortlist differs')
            require(all(float(logits[row['index']]) == row['logit'] for row in event['rows']), 'Root logits differ')
            require(index == event['greedy']['index'], 'Root greedy action differs')
            selected = event[branch]
            index = selected['index']
            candidate = prepared.options[index]
            require([candidate.rotation, candidate.x, candidate.y, candidate.w, candidate.z, candidate.layers]
                    == selected['move'], 'Root move identity differs')
        candidate = prepared.options[index]
        target = int(prepared.arrays['targets'])
        state.commit(game.materialize_move(state.board, candidate))
        state.locked_pieces = ()
        if candidate.layers:
            updated = game.next_hyperlayer_target(state.board, target)
            if updated is not None:
                target = updated
        if step == 0:
            from tetris_cross_survival import cross_count
            require(cross_count(state.board) == selected['cross_after'], 'Physical CROSS count differs')
        step += 1
        expected = [expected[1], oracle.pop()]
        if step <= len(draws) - 2:
            require(expected == draws[step:step + 2], 'Registered draw prefix differs')
        require([state.current, state.next_piece] == expected, 'Post-move draws differ')
        if state.game_over:
            require(not game.has_legal_move(state.board, state.current), 'Reported top-out has a legal move')
            reason = 'top_out'
            break
    return {'placements': state.pieces - snapshot['pieces'], 'clears': state.layers - snapshot['layers'],
            'end': reason, 'current': state.current, 'board_sha256': ev.base.board_hash(state.board)}


def summarize(rows):
    roots = {}
    for seed in sorted({row['seed'] for row in rows}):
        pairs = [row for row in rows if row['seed'] == seed]
        differences = [row['difference'] for row in pairs if row['difference'] is not None]
        roots[str(seed)] = {
            'pairs': len(pairs), 'resolved': len(differences),
            'wins': sum(d > 0 for d in differences), 'ties': differences.count(0),
            'losses': sum(d < 0 for d in differences),
            'mean_resolved_difference': statistics.mean(differences) if differences else None,
        }
    complete = all(root['pairs'] == root['resolved'] for root in roots.values())
    mean = statistics.mean(root['mean_resolved_difference'] for root in roots.values()) if complete else None
    return roots, mean


def run(args):
    runner_sha256 = digest(Path(__file__).read_bytes())
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    runtime = args.runtime.resolve()
    manifest_bytes = (args.run_dir / 'public-cold612-last-cross-intervention/runtime.json').read_bytes()
    require(digest(manifest_bytes) == RUNTIME_MANIFEST, 'Runtime manifest differs')
    manifest = json.loads(manifest_bytes)
    require(manifest['revision'] == plan['runtime_revision'], 'Revision differs')
    for name, checksum in manifest['files'].items():
        require(digest((runtime / name).read_bytes()) == checksum, f'Runtime file differs: {name}')

    os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(runtime))
    import torch
    from tools import evaluate_afterstate_td as ev

    require(Path(ev.__file__).resolve() == runtime / 'tools/evaluate_afterstate_td.py', 'Unexpected evaluator')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    require(ev.game.configure_backend('native') == 'native', 'Native scorer unavailable')
    ev.game.load_checkpoint(runtime / 'models/trained-sigmoid.json')
    ev.game.set_simulation_fast_paths(two_stage=True, headless=False)
    with (args.run_dir / 'training/612000-cold/last.pt').open('rb') as checkpoint:
        require(digest(checkpoint.read()) == plan['checkpoint_sha256'], 'Checkpoint differs')
        checkpoint.seek(0)
        model, meta = ev.neural.load_checkpoint(checkpoint, 'cpu')
    require(meta['teacher_parameters'] == ev.game.network_parameters(), 'Teacher weights differ')
    model.fused_encoding = False
    model.eval()

    rows = []
    with torch.inference_mode():
        for key, checksum in plan['root_sha256'].items():
            raw = (args.run_dir / 'public-cold612-last-cross' / f'{key}-last-loss.json.gz').read_bytes()
            require(digest(raw) == checksum, f'Root differs: {key}')
            root = json.loads(gzip.decompress(raw))
            snapshot, event = root['snapshot'], root['event']
            require(snapshot['seed'] == int(key) and snapshot['pieces'] + 1 == event['placement'], 'Wrong closure')
            for repeat in range(plan['replicates']):
                bag, rng, draws = sample_draws(ev.game, snapshot, plan['domain'], key, repeat, plan['horizon'])
                row = {'seed': int(key), 'replicate': repeat, 'draws_sha256': digest('|'.join(draws).encode())}
                for branch in ('greedy', 'best_preserving'):
                    row[branch] = play(ev, model, snapshot, event, branch, bag, rng, draws, uncapped=args.uncapped)
                greedy, preserving = row['greedy'], row['best_preserving']
                row['difference'] = (None if 'controller_stop' in (greedy['end'], preserving['end'])
                                     else preserving['placements'] - greedy['placements'])
                rows.append(row)
            print(f"Seed {key} completed {plan['replicates']} pairs", flush=True)
    require(not torch.cuda.is_initialized(), 'Unexpected CUDA context')
    require(digest(Path(__file__).read_bytes()) == runner_sha256, 'Runner changed during simulation')
    roots, mean = summarize(rows)
    return {'plan_sha256': digest(plan_bytes), 'runtime_revision': manifest['revision'],
            'runner_sha256': runner_sha256, 'torch': str(torch.__version__),
            'horizon': None if args.uncapped else plan['horizon'],
            ('natural_mean_difference' if args.uncapped else 'restricted_mean_difference'): mean,
            'roots': roots, 'pairs': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path, help='Root of the recorded CROSS tournament')
    parser.add_argument('--runtime', required=True, type=Path, help='Verified CPU scoring runtime from the closure replay')
    parser.add_argument('--plan', type=Path, default=ROOT / 'validation/cross-closure-suffix-plan.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--uncapped', action='store_true', help='Continue the same paired draws to physical top-out')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path')
    result = run(args)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')


if __name__ == '__main__':
    main()
