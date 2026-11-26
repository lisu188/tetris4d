"""Compare single-game greedy and visible-preview latency on the same archived training positions."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_afterstate_td as ev
import tetris_cross_survival as cross

game = ev.game


def positions(library):
    """Take the two shortest distinct-seed expanded training histories ending in a physical loss."""
    positions, seeds = [], set()
    entries = (e for e in ev.base.library.iter_index(library) if e['phase'] == 'training' and
               e['game_over'] and e['piece_set'] == 'expanded' and e['placements'] > 132)
    for entry in sorted(entries, key=lambda e: (e['placements'], e['id'])):
        if entry['seed'] in seeds:
            continue
        seeds.add(entry['seed'])
        record = ev.base.library.load(library / entry['file'])
        state = game.Game(entry['seed'], entry['piece_set'])
        ev.base.library.ignore(state)
        steps = {4, 100, *(len(record.moves) - offset for offset in (32, 8, 2, 1))}
        for step, row in enumerate(record.moves):
            piece, rot, x, y, z, w, target, layers = map(int, row[:8])
            if game.PIECE_NAMES[piece] != state.current:
                raise ValueError(f'Stream mismatch in {entry["id"]} at {step}')
            if step in steps:
                snapshot = copy.deepcopy(state)
                ev.base.library.ignore(snapshot)
                positions.append((entry['id'], step, snapshot, target))
            placement = game.Candidate(state.current, rot, x, y, w, z, layers, 0.)
            state.board = cross._after_board(state.board, placement)
            state.pieces += 1
            state.cells_placed += game.PIECE_SIZES[state.current]
            state.piece_counts[state.current] += 1
            state.layers += layers
            state.score += game.clear_points(layers)
            state.current, state.next_piece = state.next_piece, state.stream.pop()
        if ev.base.board_hash(state.board) != record.meta['final_board_sha256']:
            raise ValueError(f'Replayed final board differs for {entry["id"]}')
        if len(seeds) == 2:
            return positions
    raise ValueError('Need two seed-distinct completed training recordings')


@torch.inference_mode()
def measure(model, device, prep, pool, preview):
    samples, triggers, choices, overrides = [], [], [], []
    for _, _, start_state, target in pool:
        state = copy.deepcopy(start_state)
        ev.base.library.ignore(state)
        if device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter_ns()
        stats = defaultdict(dict)
        move = ev.select_moves(model, [state], [target], False, device, prep, 1, preview, stats)
        if move:
            game.play_candidate(state, move[0][1])
        if device == 'cuda':
            torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
        if preview == 'preserve-last-cross':
            triggers.append(stats[0].get('last_cross_losses', 0))
            overrides.append(stats[0].get('last_cross_overrides', 0))
        else:
            triggers.append(stats[0].get('planner_triggers', 0))
        if move:
            candidate = move[0][1]
            choices.append([candidate.rotation, candidate.x, candidate.y, candidate.w])
        else:
            choices.append(None)
    return {'device': device, 'preview': preview, 'ms': samples, 'triggers': triggers, 'choices': choices,
            **({'last_cross_overrides': overrides} if preview == 'preserve-last-cross' else {}),
            'p50_ms': float(np.percentile(samples, 50)), 'p95_ms': float(np.percentile(samples, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--library', type=Path)
    parser.add_argument('--devices', nargs='+', choices=('cpu', 'cuda'), default=('cpu', 'cuda'))
    parser.add_argument('--cross-weight', type=float, default=.5)
    parser.add_argument('--two-stage', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--rescue-only', action='store_true', help='Also measure physical rescue without reply scoring.')
    parser.add_argument('--preserve-last-cross', action='store_true', help='Also measure the last-CROSS guard.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    game.configure_backend('native')
    game.load_checkpoint(args.weights)
    game.set_simulation_fast_paths(two_stage=args.two_stage, headless=args.headless)
    pool = positions(ev.v4.library_root(args.library))
    report = {'schema': 'cross-preview-latency-v1', 'checkpoint_sha256': ev.base.file_hash(args.checkpoint),
              'weights_sha256': ev.base.file_hash(args.weights), 'cross_weight': args.cross_weight,
              'two_stage': args.two_stage, 'headless': args.headless, 'torch': str(torch.__version__),
              'visible_reply_shortlist': ('native two-stage' if game.BACKEND == 'native'
                                         and hasattr(game._NATIVE, 't4_exhaustive_primary') else 'dense'),
              'scope': 'Single game, batch=1, CPU threads=1. From prepare through checked commit, '
                       'device synchronized. The two_stage flag controls outer candidate preparation. '
                       'Every position and mode runs once to warm before timed pass.',
              'positions': [[rid, step, cross.cross_count(s.board)] for rid, step, s, _ in pool], 'modes': []}
    for device in args.devices:
        model = ev.load_model(args.checkpoint, device)
        model.config = replace(model.config, cross_weight=args.cross_weight)
        model.fused_encoding = device == 'cuda'
        prep = None
        if device == 'cuda':
            from tetris_gpu_simulation import GpuPreparer
            prep = GpuPreparer(model.config, device, two_stage=args.two_stage, group_exhaustive=True)
        modes = (False, True, 'rescue-only') if args.rescue_only else (False, True)
        if args.preserve_last_cross:
            modes += ('preserve-last-cross',)
        for preview in modes:
            if prep is not None:
                prep.require_cross_mobility = preview == 'preserve-last-cross'
            measure(model, device, prep, pool, preview)
            row = measure(model, device, prep, pool, preview)
            report['modes'].append(row)
            print(f"{device} preview={preview}: p50 {row['p50_ms']:.2f}ms, p95 {row['p95_ms']:.2f}ms", flush=True)
            ev.base.durable_json(args.output, report)


if __name__ == '__main__':
    main()
