"""Reproducible teacher-to-actor spatial imitation; not a survival benchmark."""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

import tetris_neural as neural
from tetris_survival_v4 import ARCHITECTURE


def collect(seeds, snapshots_per_seed, stride, config, deadline):
    records, games = [], []
    for seed in seeds:
        state = neural.game.Game(seed, 'expanded')
        target, snapshots = 1, 0
        while not state.game_over and snapshots < snapshots_per_seed and time.monotonic() < deadline:
            prepared = neural.prepare(state, target, config)
            if prepared is None:
                break
            target = int(prepared.arrays['targets'])
            if state.pieces % stride == 0:
                records.append(prepared.arrays)
                snapshots += 1
            chosen = neural.game.materialize_move(state.board, prepared.options[prepared.teacher])
            state.commit(chosen)
        games.append({'seed': seed, 'snapshots': snapshots, 'placements': state.pieces,
                      'complete_game': state.game_over, 'collection_complete': snapshots == snapshots_per_seed})
    return records, games


@torch.no_grad()
def evaluate(model, records, temperature, device):
    if not records:
        raise ValueError('No validation snapshots were collected')
    model.eval()
    total_kl = total_top1 = 0.
    for record in records:
        batch = neural.collate([record], device)
        valid = batch['valid'].bool()
        raw = model.raw_scores(batch)[0].float().masked_fill(~valid, -torch.inf)
        teacher = batch['teacher_prior'].float().masked_fill(~valid, -torch.inf)
        target = F.softmax(teacher / temperature, dim=-1)
        total_kl += float(F.kl_div(F.log_softmax(raw, dim=-1), target, reduction='batchmean'))
        selected = int(raw[0].argmax())
        chosen = int(teacher[0].argmax())
        total_top1 += selected == chosen
    return {'snapshots': len(records), 'raw_kl': total_kl / len(records),
            'raw_teacher_top1': total_top1 / len(records)}


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    neural.game.configure_backend(args.backend)
    neural.game.load_checkpoint(args.weights)
    device = neural.device_for(args.device)
    config = neural.Config(architecture=ARCHITECTURE, spatial_actor=True, shortlist_mode=args.shortlist_mode,
                           policy_mode='teacher-prior', teacher_prior_scale=.125, recover_target=True,
                           candidates=args.candidates, shortlist=args.shortlist)
    deadline = time.monotonic() + args.max_seconds
    training, training_games = collect(args.train_seeds, args.samples_per_seed, args.stride, config, deadline)
    validation, validation_games = collect(args.validation_seeds, args.validation_samples_per_seed, args.stride,
                                           config, deadline)
    if len(training) < args.batch or not validation:
        raise ValueError('Insufficient collected teacher snapshots for supervised training')
    model = neural.create_model(config).to(device)
    actor = [parameter for name, parameter in model.named_parameters() if not name.startswith('value.')]
    optimizer = torch.optim.AdamW(actor, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    initial = evaluate(model, validation, args.temperature, device)
    best, history = float('inf'), []
    selected_weights, selected_step = None, 0
    for step in range(1, args.steps + 1):
        if time.monotonic() >= deadline:
            break
        picks = rng.integers(0, len(training), size=args.batch)
        batch = neural.collate([training[int(index)] for index in picks], device)
        valid = batch['valid'].bool()
        teacher = batch['teacher_prior'].float().masked_fill(~valid, -torch.inf)
        soft_target = F.softmax(teacher / args.temperature, dim=-1)
        model.train()
        logits = model.raw_scores(batch)[0].float().masked_fill(~valid, -torch.inf)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), soft_target, reduction='batchmean')
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite supervised actor loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor, 1., error_if_nonfinite=True)
        optimizer.step()
        if step % args.evaluate_every == 0 or step == args.steps:
            report = {'step': step, 'train_loss': float(loss.detach()),
                      **evaluate(model, validation, args.temperature, device)}
            history.append(report)
            if report['raw_kl'] < best:
                best = report['raw_kl']
                selected_step = step
                selected_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if selected_weights is None:
        raise ValueError('Time budget ended before completing a validation checkpoint')
    model.load_state_dict(selected_weights)
    selected = evaluate(model, validation, args.temperature, device)
    checkpoint = Path(args.output)
    neural.save_checkpoint(checkpoint, model, progress={'updates': selected_step, 'supervised_updates': selected_step,
                          'ppo_updates': 0, 'mode': 'spatial-imitation'},
                           extra={'algorithm': 'reconstructed-spatial-actor-distillation-v1',
                                  'critic_calibrated': False, 'value_semantics': 'uncalibrated_survival_value',
                                  'imitation_temperature': args.temperature})
    report = {'schema': 'tetris4d-spatial-pretraining-reconstructed-v1',
              'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'config': asdict(config), 'selected_step': selected_step, 'ppo_updates': 0,
              'initial_validation': initial, 'selected_validation': selected,
              'train_snapshots': len(training), 'validation_snapshots': len(validation),
              'train_games': training_games, 'validation_games': validation_games,
              'history': history, 'provenance': 'Reconstructed script and newly generated weights; not the 2026-11-16 historical checkpoint.',
              'limitation': 'KL and teacher agreement measure imitation, not uncensored survival improvement.'}
    if args.manifest:
        neural.atomic_json(Path(args.manifest), report)
    print(json.dumps(report, indent=2))
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/neural-spatial-actor-reconstructed.pt'))
    parser.add_argument('--manifest', type=Path, default=Path('validation/spatial-pretraining-reconstructed.json'))
    parser.add_argument('--weights', type=Path, default=Path('models/trained-sigmoid.json'))
    parser.add_argument('--backend', choices=('python', 'numpy', 'native'), default='native')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261116)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[6100002, 6100004, 6100006, 6100008])
    parser.add_argument('--validation-seeds', type=int, nargs='+', default=[6200001, 6200003])
    parser.add_argument('--samples-per-seed', type=int, default=256)
    parser.add_argument('--validation-samples-per-seed', type=int, default=128)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--candidates', type=int, default=2048)
    parser.add_argument('--shortlist', type=int, default=128)
    parser.add_argument('--shortlist-mode', choices=('teacher-random', 'diverse'), default='teacher-random')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--temperature', type=float, default=8.)
    parser.add_argument('--evaluate-every', type=int, default=20)
    parser.add_argument('--max-seconds', type=float, default=480.)
    args = parser.parse_args(argv)
    if min(args.threads, args.samples_per_seed, args.validation_samples_per_seed, args.stride,
           args.steps, args.batch, args.evaluate_every) <= 0 or args.temperature <= 0 or args.max_seconds <= 0:
        parser.error('Numeric limits must be positive')
    if set(args.train_seeds) & set(args.validation_seeds):
        parser.error('Training and validation seeds must not overlap')
    return args


if __name__ == '__main__':
    train(arguments())
