#!/usr/bin/env python3
"""Survival from fixed near-top-out positions: a fast, discriminative policy benchmark.

Full games of the current policies reach the 30,000-placement caps, so they no longer
separate policies cheaply. This benchmark replays recorded top-out games from the game
library to 500-3,000 placements (by default) before their end, and measures how many
placements a policy survives from each position (capped). Policies:

  base                 the V4 checkpoint itself (greedy or sampled)
  v5:R:m[:tau][:mode]  V5 built from the checkpoint: residual bound R, beta = m / sigma, where
                       sigma is the median per-state std of the critic's next-state values on the
                       benchmark positions (so m is in "standard deviations"), temperature tau,
                       shortlist mode (teacher-random or teacher-top)

All policies play the same positions in lockstep on CUDA. Copies of library states are
never recorded back into the library.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural  # noqa: E402
import tetris_survival_v5 as v5  # noqa: E402
import train_survival_v4 as v4t  # noqa: E402

base, game = v4t.base, v4t.game


def benchmark_states(args):
    """Deterministic set of start positions (dicts with an unrecorded `state` and its `target`)."""
    settings = SimpleNamespace(danger_window=tuple(args.window), seed=args.seed, danger_pool=args.states,
                               danger_per_game=args.per_game, piece_set='expanded')
    pool = v4t.build_danger_pool(settings)
    if len(pool) < args.states:
        raise SystemExit(f'Only {len(pool)} benchmark positions available in the game library')
    return pool


def parse_policy(text):
    if text == 'base':
        return {'name': 'base'}
    kind, *fields = text.split(':')
    if kind != 'v5' or not 2 <= len(fields) <= 4:
        raise argparse.ArgumentTypeError(f'Policy must be "base" or "v5:R:m[:tau][:mode]", got {text!r}')
    return {'name': text, 'bound': float(fields[0]), 'multiple': float(fields[1]),
            'temperature': float(fields[2]) if len(fields) > 2 else 1.,
            'mode': fields[3] if len(fields) > 3 else None}


@torch.no_grad()
def value_spread(source, pool, device):
    """Median over positions of the std of the critic's next-state values across shortlisted candidates."""
    from tetris_gpu_simulation import GpuPreparer
    probe = v5.from_v4(source, residual_bound=1., policy_temperature=1., afterstate_scale=1.).eval()
    states = [entry['state'] for entry in pool]
    result = v5.augment_device_batch(GpuPreparer(probe.config, device).prepare_batch(
        states, [entry['target'] for entry in pool]), states)
    scores = probe.score(result.batch['next_features'])
    valid = result.batch['valid'].bool()
    spreads = [float(scores[row][valid[row]].std()) for row in range(len(valid)) if int(valid[row].sum()) > 1]
    return statistics.median(spreads)


@torch.no_grad()
def play(model, pool, device, cap, sampled, seed, is_v5):
    from tetris_gpu_simulation import GpuPreparer
    episodes = []
    for number, entry in enumerate(pool):
        state = copy.deepcopy(entry['state'])
        state._library_log = None
        base.library.ignore(state)
        episodes.append(base.Episode(number, state, target=entry['target'], phase='benchmark'))
    starts = [(e.state.pieces, e.state.layers) for e in episodes]
    generators = [torch.Generator(device=device).manual_seed(seed + i) for i in range(len(episodes))]
    preparer = GpuPreparer(model.config, device)
    reasons, active, began = [None] * len(episodes), list(range(len(episodes))), time.perf_counter()
    while active:
        states = [episodes[i].state for i in active]
        result = preparer.prepare_batch(states, [episodes[i].target for i in active])
        if is_v5:
            result = v5.augment_device_batch(result, states)
        live = []
        for local, i in enumerate(active):
            if result.prepared[local] is None:
                reasons[i] = 'top_out' if episodes[i].state.game_over else 'no_move_or_target'
            else:
                episodes[i].target = int(result.targets[local])
                live.append(i)
        if live:
            rows = [active[local] for local in result.environment_indices]
            logits = model(result.batch)['logits']
            if sampled:
                probability = torch.softmax(logits.float(), dim=-1)
                actions = torch.stack([torch.multinomial(probability[r], 1, generator=generators[i])[0]
                                       for r, i in enumerate(rows)])
            else:
                actions = logits.argmax(-1)
            for i, candidate in zip(rows, result.selected_candidates(actions)):
                base.apply_move(episodes[i], candidate, 0, 'benchmark')
        active = []
        for i in live:
            if episodes[i].state.pieces - starts[i][0] >= cap:
                reasons[i] = 'cap'
            else:
                active.append(i)
    survived = [e.state.pieces - s[0] for e, s in zip(episodes, starts)]
    clears = [e.state.layers - s[1] for e, s in zip(episodes, starts)]
    return {'survived': survived, 'clears': clears, 'stop_reasons': reasons,
            'mean': statistics.mean(survived), 'median': statistics.median(survived), 'minimum': min(survived),
            'capped': sum(r == 'cap' for r in reasons), 'clears_per_1k': 1000 * sum(clears) / max(1, sum(survived)),
            'wall_seconds': time.perf_counter() - began}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='V4 checkpoint with a calibrated regional critic')
    parser.add_argument('--policies', type=parse_policy, nargs='+', default=[parse_policy('base')])
    parser.add_argument('--states', type=int, default=32)
    parser.add_argument('--per-game', type=int, default=2, help='Start positions taken from each recorded game')
    parser.add_argument('--window', type=int, nargs=2, default=(500, 3000), metavar=('MIN', 'MAX'))
    parser.add_argument('--cap', type=int, default=3000, help='Placements survived after which a position counts as capped')
    parser.add_argument('--sampled', action='store_true', help='Sample actions instead of greedy argmax')
    parser.add_argument('--seed', type=int, default=20261117, help='Benchmark position and sampling seed')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    game.configure_backend('native')
    game.load_checkpoint(args.weights)
    device = torch.device('cuda')
    source, _ = neural.load_checkpoint(args.checkpoint, device)
    source.eval()
    pool = benchmark_states(args)
    sigma = value_spread(source, pool, device) if any(p['name'] != 'base' for p in args.policies) else None
    report = {'schema': 'tetris4d-survival-benchmark-v1', 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': base.file_hash(args.checkpoint), 'positions': len(pool), 'window': list(args.window),
              'cap': args.cap, 'sampled': args.sampled, 'seed': args.seed, 'value_spread_sigma': sigma,
              'position_ids': [[p['game_id'], p['offset']] for p in pool], 'results': []}
    print(f'{len(pool)} positions; critic next-state spread sigma = {sigma}', flush=True)
    for policy in args.policies:
        if policy['name'] == 'base':
            model, is_v5 = source, False
        else:
            model = v5.from_v4(source, residual_bound=policy['bound'], policy_temperature=policy['temperature'],
                               afterstate_scale=policy['multiple'] / sigma, shortlist_mode=policy['mode']).eval()
            is_v5 = True
        result = {'policy': policy['name'], **play(model, pool, device, args.cap, args.sampled, args.seed, is_v5)}
        report['results'].append(result)
        print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in result.items()
                          if k in ('policy', 'mean', 'median', 'minimum', 'capped', 'clears_per_1k', 'wall_seconds')}),
              flush=True)
    if args.output:
        base.durable_json(args.output, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
