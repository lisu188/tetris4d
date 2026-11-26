#!/usr/bin/env python3
"""Issue #37 experiment: choose moves by the critic's value of each afterstate.

For every step, the actor's top-k shortlisted candidates are placed on a copy of
the board. The critic scores the successor once per possible next preview and
averages those predictions under the public bag distribution. Policies played in lockstep
on the same seeds:

  actor       greedy argmax of the trained policy logits (the selection policy)
  afterstate  argmax of critic(afterstate) among the actor's top-k candidates

The critic was trained on pre-move states under the current policy, so this is a
zero-retraining probe, not a tuned afterstate learner.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural  # noqa: E402
import tetris_survival_v4 as v4  # noqa: E402
import train_overnight as base  # noqa: E402

game = neural.game


def afterstate(state, target, candidate, upcoming=None):
    """(proxy state, target) after placing `candidate`; nothing is committed."""
    after, layers = game.clear_layers(game.merge_rigid_piece(state.board, game.placed_cells(candidate), candidate.piece))
    if layers:
        following = game.next_hyperlayer_target(after, target)
        target = following if following is not None else target
    counts = dict(state.piece_counts)
    counts[candidate.piece] += 1
    if upcoming is None:  # Realized successor used to replay archived V5 and TD features.
        upcoming = copy.deepcopy(state.stream).pop()
    proxy = SimpleNamespace(board=after, piece_set=state.piece_set, pieces=state.pieces + 1,
                            current=state.next_piece, next_piece=upcoming, piece_counts=counts)
    return proxy, target


def afterstate_values(model, device, proxies):
    boards = torch.tensor([p.board for p, _ in proxies], dtype=torch.int32, device=device)
    targets = torch.tensor([t for _, t in proxies], device=device)
    context = torch.from_numpy(np.stack([neural.state_context(p, t) for p, t in proxies])).to(device)
    return model.encode(boards, targets, context)[2].float()


def public_reference_values(model, device, state, target, candidates):
    if not candidates:
        return torch.empty(0, dtype=torch.float64, device=device)
    predictions = []
    for upcoming in v4.public_afterstate_previews(state):
        proxies = [afterstate(state, target, c, upcoming) for c in candidates]
        predictions.append(afterstate_values(model, device, proxies))
    return torch.stack(predictions).double().mean(dim=0)


def play(model, device, seeds, policy, top_k, cap, preparer, cpu_reference=False):
    states = [game.Game(seed, 'expanded') for seed in seeds]
    for state in states:
        base.library.ignore(state)
    targets, reasons = [1] * len(seeds), [None] * len(seeds)
    active, changed, decisions, started = list(range(len(seeds))), 0, 0, time.perf_counter()
    while active:
        result = preparer.prepare_batch([states[i] for i in active], [targets[i] for i in active])
        rows = []
        for local, i in enumerate(active):
            if result.prepared[local] is None:
                reasons[i] = 'top_out' if states[i].game_over else 'target_or_no_move'
            else:
                targets[i] = int(result.targets[local])
                rows.append(i)
        if rows:
            logits = model(result.batch)['logits']
            actions = logits.argmax(-1)
            if policy == 'afterstate':
                k = min(top_k, logits.shape[1])
                top = logits.topk(k, dim=-1).indices
                finite = torch.isfinite(logits.gather(1, top)).cpu().tolist()
                indices, options = [], []
                for row, (ranked, ok) in enumerate(zip(top.cpu().tolist(), finite)):
                    valid = [a for a, keep in zip(ranked, ok) if keep]
                    pool = result.prepared[result.environment_indices[row]].options
                    indices.append(valid)
                    options.append([pool[a] for a in valid])
                if cpu_reference:
                    values = [public_reference_values(model, device, states[i], targets[i], cands)
                              for i, cands in zip(rows, options)]
                else:
                    values = v4.afterstate_values(model, [states[i] for i in rows], [targets[i] for i in rows], options)
                chosen = [valid[int(v.argmax())] for valid, v in zip(indices, values)]
                changed += sum(best != int(a) for best, a in zip(chosen, actions.tolist()))
                decisions += len(chosen)
                actions = torch.tensor(chosen, device=device)
            for i, candidate in zip(rows, result.selected_candidates(actions)):
                move = game.materialize_move(states[i].board, candidate)
                states[i].commit(move)
                if move.layers:
                    following = game.next_hyperlayer_target(states[i].board, targets[i])
                    targets[i] = following if following is not None else targets[i]
        active = []
        for i in rows:
            if states[i].pieces >= cap:
                reasons[i] = 'piece_limit'
            else:
                active.append(i)
    pieces = [s.pieces for s in states]
    return {'policy': policy, 'placements': pieces, 'stop_reasons': reasons,
            'mean': float(np.mean(pieces)), 'median': float(np.median(pieces)), 'minimum': int(min(pieces)),
            'clears': [s.layers for s in states], 'censored': sum(r == 'piece_limit' for r in reasons),
            'changed_from_actor': (changed / decisions) if decisions else None,
            'wall_seconds': time.perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='V4 checkpoint (e.g. a run\'s best.pt)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[910001 + 2 * i for i in range(8)])
    parser.add_argument('--policies', nargs='+', choices=('actor', 'afterstate'), default=['actor', 'afterstate'])
    parser.add_argument('--top-k', type=int, default=16)
    parser.add_argument('--pieces', type=int, default=40000, help='Censor games at this many placements')
    parser.add_argument('--screening', choices=('sampled', 'exhaustive'), default=None,
                        help='Override the checkpoint screening (default: keep it)')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cpu-reference', action='store_true',
                        help='Build afterstate contexts with the CPU reference path (slow; for verification)')
    args = parser.parse_args(argv)
    if args.top_k < 2 or args.pieces < 1:
        parser.error('--top-k must be >= 2 and --pieces positive')
    game.configure_backend('native')
    game.load_checkpoint(args.weights)
    device = torch.device('cuda')
    model, _ = neural.load_checkpoint(args.checkpoint, device)
    if args.screening:
        from dataclasses import replace
        model.config = replace(model.config, screening=args.screening)
    model.eval()
    from tetris_gpu_simulation import GpuPreparer
    preparer = GpuPreparer(model.config, device)
    report = {'schema': 'tetris4d-afterstate-ranking-v2', 'preview_policy': 'public-bag-mean', 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': base.file_hash(args.checkpoint), 'seeds': args.seeds, 'top_k': args.top_k,
              'piece_cap': args.pieces, 'screening': model.config.screening, 'results': []}
    with torch.inference_mode():
        for policy in args.policies:
            result = play(model, device, args.seeds, policy, args.top_k, args.pieces, preparer, args.cpu_reference)
            report['results'].append(result)
            print(json.dumps({k: result[k] for k in ('policy', 'mean', 'median', 'minimum', 'censored',
                                                     'changed_from_actor', 'wall_seconds')}), flush=True)
    if args.output:
        base.durable_json(args.output, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
