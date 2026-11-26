"""Collect uncensored teacher lifetimes, calibrate a V4 critic, and audit without refitting."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F

import tetris_game_library as library
import tetris_neural as neural
import train_survival_v2 as v2
from tetris_survival_v4 import ARCHITECTURE

SCALE = .001


def load_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        x = np.asarray(data['x'], dtype=np.float32)
        y = np.asarray(data['y'], dtype=np.float32)
        seeds = np.asarray(data['seed'], dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != 156 or y.shape != (len(x),) or seeds.shape != (len(x),):
        raise ValueError('Invalid spatial critic dataset schema')
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Dataset contains no complete lifetime targets or non-finite inputs')
    return x, y, seeds


def describe(predictions, targets, constant):
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    near = targets <= 1.
    return {'samples': len(targets), 'mse': float(np.mean(errors ** 2)),
            'mae_pieces': float(np.mean(np.abs(errors)) / SCALE),
            'signed_error_pieces': float(np.mean(errors) / SCALE),
            'constant_mse': float(np.mean((targets - constant) ** 2)),
            'constant_mae_pieces': float(np.mean(np.abs(targets - constant)) / SCALE),
            'near_end_samples': int(near.sum()),
            'near_end_mae_pieces': float(np.mean(np.abs(errors[near])) / SCALE) if near.any() else None}


@torch.inference_mode()
def collect_gpu(args, selector, model, device):
    """Play every seed's teacher game in lockstep with batched CUDA preparation and encoding."""
    from tetris_gpu_simulation import GpuPreparer
    preparer = GpuPreparer(selector, device)
    states = [neural.game.Game(seed, 'expanded') for seed in args.seeds]
    for state in states:
        library.tag(state, source='precalibrate_spatial', phase='teacher-continuation', default_behavior='teacher')
    targets = [1] * len(states)
    snapshots = [[] for _ in states]
    active = list(range(len(states)))
    while active:
        result = preparer.prepare_batch([states[i] for i in active], [targets[i] for i in active])
        rows = {local: row for row, local in enumerate(result.env_indices)}
        moves = result.selected_candidates(result.teachers) if rows else ()
        wanted = [local for local in rows if states[active[local]].pieces % args.stride == 0]
        if wanted:
            picks = torch.tensor([rows[local] for local in wanted], device=device)
            boards = result.batch['boards'].index_select(0, picks)
            levels = result.batch['targets'].index_select(0, picks)
            context = result.batch['context'].index_select(0, picks)
            _, pooled, _, _, _ = model.encode(boards, levels, context)
            values = torch.cat((pooled, context), 1).float().cpu().numpy()
            for local, row in zip(wanted, values):
                snapshots[active[local]].append((row.copy(), states[active[local]].pieces))
        survivors = []
        for local, env in enumerate(active):
            if local not in rows:
                continue
            targets[env] = result.targets[local]
            library.annotate(states[env], 'teacher', None, targets[env])
            states[env].commit(neural.game.materialize_move(states[env].board, moves[rows[local]]))
            if states[env].pieces < args.max_pieces and not states[env].game_over:
                survivors.append(env)
        active = survivors
    for state in states:
        library.finish(state, 'top_out' if state.game_over else
                       'piece_limit' if state.pieces >= args.max_pieces else 'planner_blocked')
    return [(seed, states[i], snapshots[i]) for i, seed in enumerate(args.seeds)]


def collect(args):
    torch.set_num_threads(args.threads)
    neural.game.configure_backend(args.backend)
    neural.game.load_checkpoint(args.weights)
    config = neural.Config(architecture=ARCHITECTURE, spatial_actor=True, recover_target=True,
                           policy_mode='teacher-prior', teacher_prior_scale=.125)
    selector = neural.Config(architecture='survival-feature-v3', recover_target=True,
                             policy_mode='teacher-prior', teacher_prior_scale=.125)
    model = neural.create_model(config).eval()
    inputs, labels, seed_rows, episodes = [], [], [], []
    device = neural.device_for(args.device)
    if device.type == 'cuda' and args.backend == 'native':
        rollouts = collect_gpu(args, selector, model.to(device), device)
    else:
        rollouts = None
    for seed in (() if rollouts is not None else args.seeds):
        state = neural.game.Game(seed, 'expanded')
        library.tag(state, source='precalibrate_spatial', phase='teacher-continuation', default_behavior='teacher')
        target, snapshots = 1, []
        while state.pieces < args.max_pieces and not state.game_over:
            prepared = neural.prepare(state, target, selector)
            if prepared is None:
                break
            target = int(prepared.arrays['targets'])
            if state.pieces % args.stride == 0:
                board = torch.tensor([state.board], dtype=torch.int32)
                levels = torch.tensor([target], dtype=torch.int64)
                context = torch.from_numpy(neural.state_context(state, target))[None]
                with torch.inference_mode():
                    _, pooled, _, _, _ = model.encode(board, levels, context)
                snapshots.append((torch.cat((pooled, context), 1)[0].numpy().copy(), state.pieces))
            library.annotate(state, 'teacher', None, target)
            state.commit(neural.game.materialize_move(state.board, prepared.options[prepared.teacher]))
        library.finish(state, 'top_out' if state.game_over else
                       'piece_limit' if state.pieces >= args.max_pieces else 'planner_blocked')
        complete = bool(state.game_over or not neural.game.has_legal_move(state.board, state.current))
        if complete:
            for row, piece in snapshots:
                inputs.append(row)
                labels.append((state.pieces - piece) * SCALE)
                seed_rows.append(seed)
        episodes.append({'seed': seed, 'pieces': state.pieces, 'snapshots': len(snapshots),
                         'natural_top_out': complete, 'labeled': len(snapshots) if complete else 0})
    for seed, state, snapshots in (rollouts or ()):
        complete = bool(state.game_over or not neural.game.has_legal_move(state.board, state.current))
        if complete:
            for row, piece in snapshots:
                inputs.append(row)
                labels.append((state.pieces - piece) * SCALE)
                seed_rows.append(seed)
        episodes.append({'seed': seed, 'pieces': state.pieces, 'snapshots': len(snapshots),
                         'natural_top_out': complete, 'labeled': len(snapshots) if complete else 0})
    if not inputs:
        raise ValueError('No naturally completed teacher games; censored games are not labeled')
    output = Path(args.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.stack(inputs).astype(np.float32),
                        y=np.asarray(labels, dtype=np.float32), seed=np.asarray(seed_rows, dtype=np.int64))
    manifest = {'schema': 'tetris4d-reconstructed-critic-data-v1', 'dataset': str(output),
                'dataset_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'episodes': episodes, 'labeled_snapshots': len(inputs),
                'value_semantics': 'teacher_remaining_successful_placements_times_0.001',
                'censored_lifetimes_excluded': True,
                'limitation': 'Samples within each game are correlated and not independent observations.'}
    if args.report:
        neural.atomic_json(Path(args.report), manifest)
    print(json.dumps(manifest, indent=2))


def fit(args):
    torch.set_num_threads(args.threads)
    neural.game.configure_backend(args.backend)
    neural.game.load_checkpoint(args.weights)
    train_x, train_y, train_seeds = load_dataset(args.train_dataset)
    val_x, val_y, val_seeds = load_dataset(args.selection_dataset)
    if set(train_seeds) & set(val_seeds):
        raise ValueError('Training and checkpoint-selection games must be disjoint')
    device = neural.device_for(args.device)
    model, original = neural.load_checkpoint(args.actor_checkpoint, device)
    if model.config.architecture != ARCHITECTURE:
        raise ValueError('Expected spatial V4 actor checkpoint')
    if original['teacher_parameters'] != neural.game.network_parameters():
        raise ValueError('Teacher policy weights do not match checkpoint')
    actor_before = v2.actor_digest(model)
    x = torch.from_numpy(train_x).to(device)
    y = torch.from_numpy(train_y).to(device)
    vx = torch.from_numpy(val_x).to(device)
    constant = float(np.mean(train_y))
    optimizer = torch.optim.Adam(model.value.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best, best_state, selected, history = float('inf'), None, None, []
    for step in range(1, args.steps + 1):
        indices = torch.from_numpy(rng.integers(0, len(x), size=args.batch)).long().to(device, non_blocking=True)
        loss = F.mse_loss(model.critic(x[indices]), y[indices])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Non-finite critic calibration loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % args.evaluate_every == 0 or step == args.steps:
            with torch.inference_mode():
                predictions = model.critic(vx).cpu().numpy()
            metrics = describe(predictions, val_y, constant)
            report = {'update': step, **metrics}
            history.append(report)
            if (metrics['mse'] < best and metrics['mse'] < metrics['constant_mse']
                    and metrics['near_end_samples']
                    and metrics['near_end_mae_pieces'] <= args.near_end_max_mae):
                best, selected = metrics['mse'], report
                best_state = {name: value.detach().cpu().clone() for name, value in model.value.state_dict().items()}
    if best_state is None:
        raise ValueError('No critic checkpoint passed both global MSE and near-end MAE selection gates')
    model.value.load_state_dict(best_state)
    if v2.actor_digest(model) != actor_before:
        raise AssertionError('Calibrating the critic changed actor parameters')
    output = Path(args.output)
    extra = dict(original.get('extra') or {})
    extra.update(critic_calibrated=True,
                 value_semantics='expected_teacher_remaining_placements_times_0.001',
                 critic_calibration={'schema': 'reconstructed-spatial-calibration-v1',
                                     'training_seeds': sorted(set(map(int, train_seeds))),
                                     'selection_seeds': sorted(set(map(int, val_seeds))),
                                     'selected_update': selected['update'], 'selected': selected,
                                     'constant_training_mean': constant,
                                     'actor_unchanged': True, 'actor_sha256': actor_before})
    model = model.cpu()
    neural.save_checkpoint(output, model, progress=original.get('progress'), extra=extra)
    report = {'schema': 'tetris4d-reconstructed-critic-fit-v1', 'output': str(output),
              'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'selected': selected,
              'actor_sha256': actor_before, 'actor_unchanged': True,
              'train_snapshots': len(x), 'selection_snapshots': len(vx),
              'constant_training_mean_placements': constant / SCALE, 'history': history,
              'limitation': 'Teacher-policy critic is not calibrated for a learned-actor continuation.'}
    if args.report:
        neural.atomic_json(Path(args.report), report)
    print(json.dumps(report, indent=2))


def audit(args):
    torch.set_num_threads(args.threads)
    x, y, seeds = load_dataset(args.dataset)
    device = neural.device_for(args.device)
    model, checkpoint = neural.load_checkpoint(args.checkpoint, device)
    if model.config.architecture != ARCHITECTURE or not checkpoint.get('extra', {}).get('critic_calibrated'):
        raise ValueError('Audit requires a calibrated Spatial V4 checkpoint')
    used = checkpoint['extra']['critic_calibration']
    if set(map(int, seeds)) & (set(used['training_seeds']) | set(used['selection_seeds'])):
        raise ValueError('Independent audit seeds overlap the fitting/selection split')
    with torch.inference_mode():
        values = model.critic(torch.from_numpy(x).to(device)).cpu().numpy()
    baseline = float(used['constant_training_mean'])
    result = {'schema': 'tetris4d-reconstructed-critic-independent-audit-v1',
              'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'seeds': sorted(set(map(int, seeds))), 'fit_modified': False,
              'metrics': describe(values, y, baseline),
              'limitation': 'Independent games, but correlated snapshots; no learned-policy lifetime claim.'}
    if args.report:
        neural.atomic_json(Path(args.report), result)
    print(json.dumps(result, indent=2))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for command in ('collect', 'fit', 'audit'):
        item = commands.add_parser(command)
        item.add_argument('--report', type=Path)
        item.add_argument('--threads', type=int, default=2)
        item.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
        if command != 'audit':
            item.add_argument('--backend', choices=('python', 'numpy', 'native'), default='native')
            item.add_argument('--weights', type=Path, default=Path('models/trained-sigmoid.json'))
        if command == 'collect':
            item.add_argument('--dataset', type=Path, required=True)
            item.add_argument('--seeds', nargs='+', type=int, required=True)
            item.add_argument('--stride', type=int, default=128)
            item.add_argument('--max-pieces', type=int, default=100000)
        elif command == 'fit':
            item.add_argument('--actor-checkpoint', type=Path, required=True)
            item.add_argument('--train-dataset', type=Path, required=True)
            item.add_argument('--selection-dataset', type=Path, required=True)
            item.add_argument('--output', type=Path, required=True)
            item.add_argument('--steps', type=int, default=2000)
            item.add_argument('--batch', type=int, default=128)
            item.add_argument('--seed', type=int, default=20261117)
            item.add_argument('--lr', type=float, default=.001)
            item.add_argument('--evaluate-every', type=int, default=25)
            item.add_argument('--near-end-max-mae', type=float, default=1000.)
        else:
            item.add_argument('--checkpoint', type=Path, required=True)
            item.add_argument('--dataset', type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    options = arguments()
    {'collect': collect, 'fit': fit, 'audit': audit}[options.operation](options)
