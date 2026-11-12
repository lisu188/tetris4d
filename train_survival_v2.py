from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

import train_overnight as base
import train_survival as legacy

neural, game = base.neural, base.game
ALGORITHM = 'survival-ppo-v2'


def tensor_summary(values, scale=1.):
    x = np.asarray(values, dtype=np.float64).reshape(-1) * scale
    if not len(x):
        return {'count': 0}
    if not np.isfinite(x).all():
        raise ValueError('Non-finite diagnostic values')
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std()), minimum=float(x.min()),
                maximum=float(x.max()), p10=float(np.quantile(x, .1)), p50=float(np.quantile(x, .5)),
                p90=float(np.quantile(x, .9)))


def regression_metrics(predicted, target, constant=None):
    p, y = np.asarray(predicted, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid critic predictions or targets')
    error = p - y
    baseline = float(y.mean()) if constant is None else float(constant)
    variance = float(y.var())
    end = y <= .256
    return {'samples': len(y), 'mae_pieces': float(np.abs(error).mean() * 1000),
            'rmse_pieces': float(np.sqrt(np.mean(error ** 2)) * 1000),
            'mse': float(np.mean(error ** 2)), 'constant_mse': float(np.mean((y - baseline) ** 2)),
            'explained_variance': float(1 - np.var(error) / variance) if variance > 1e-12 else None,
            'near_end_samples': int(end.sum()),
            'near_end_mae_pieces': float(np.abs(error[end]).mean() * 1000) if end.any() else None,
            'constant_near_end_mae_pieces': float(np.abs(y[end] - baseline).mean() * 1000) if end.any() else None,
            'predictions_pieces': tensor_summary(p, 1000), 'targets_pieces': tensor_summary(y, 1000)}


def secondary_teacher_loss(logp, batch):
    winners, _ = legacy.utility_targets(batch['features'], batch['valid'])
    prior = batch.get('teacher_prior', batch['features'][..., 13].float() * 64)
    targets = torch.softmax(prior.float().masked_fill(~winners, -torch.inf), -1)
    conditional = logp - torch.logsumexp(logp.masked_fill(~winners, -torch.inf), -1, keepdim=True)
    return -(targets * torch.where(winners, conditional, 0.)).sum(-1).mean()


def gradient_size(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = sum(g.detach().float().square().sum() for g in gradients if g is not None)
    return float(squared.sqrt().item()) if isinstance(squared, torch.Tensor) else 0.


def actor_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('value.'):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def device_telemetry():
    command = '/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi'
    try:
        result = subprocess.run([command, '--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        return {'nvidia_smi': result.stdout.strip(), 'returncode': result.returncode,
                'note': 'Snapshot, not a time-averaged utilization measurement.'}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'unavailable': type(exc).__name__}


def competing_trainers():
    found = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            pid = int(directory.name)
            if pid == os.getpid() or directory.stat().st_uid != os.getuid():
                continue
            arguments = (directory / 'cmdline').read_bytes().split(b'\0')
            arguments = [x.decode(errors='replace') for x in arguments if x]
            names = {'train_overnight.py', 'train_survival.py', 'train_survival_v2.py'}
            if any(Path(arg).name in names for arg in arguments[1:3]):
                found.append({'pid': pid, 'command': arguments[:3]})
        except (OSError, ValueError):
            continue
    return found


@contextmanager
def device_lock(device, allow_concurrent=False):
    if device.type != 'cuda':
        yield []
        return
    competitors = competing_trainers()
    if competitors and not allow_concurrent:
        raise ValueError(f'Other Tetris trainers are running: {competitors}. Stop them gracefully or explicitly use --allow-concurrent.')
    key = hashlib.sha256(str(getattr(torch.cuda.get_device_properties(device), 'uuid', torch.cuda.get_device_name(device))).encode()).hexdigest()[:16]
    user = str(getattr(os, 'getuid', lambda: os.environ.get('USERNAME', 'user'))())
    lock = base.run_lock(Path(tempfile.gettempdir()) / f'tetris4d-cuda-{user}-{key}')
    with nullcontext() if allow_concurrent else lock:
        yield competitors


class CompressedRecorder(base.Recorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.used += sum(p.stat().st_size for p in self.root.rglob('*.json.gz')) if self.root.exists() else 0

    def write(self, episode, reason=None):
        path = super().write(episode, reason)
        if path is None or not episode.complete:
            return path
        try:
            packed = gzip.compress(path.read_bytes(), mtime=0)
            target = path.with_suffix('.json.gz')
            previous = target.stat().st_size if target.exists() else 0
            handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.compress-')
            try:
                with os.fdopen(handle, 'wb') as output:
                    output.write(packed)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.used += len(packed) - self.sizes.pop(path, path.stat().st_size) - previous
            path.unlink()
            return target
        except OSError as exc:
            self.disable(f'Compression failed: {type(exc).__name__}; checkpoints remain independent.')
            return path


@dataclass
class Rollout:
    records: list
    choices: list
    labels: list
    advantages: np.ndarray
    returns: np.ndarray
    old_logp: list
    old_values: np.ndarray
    raw_advantages: np.ndarray
    terminals: np.ndarray
    truncations: np.ndarray
    timings: dict


class SurvivalTrainer(base.Trainer):
    def training_signature(self):
        names = ('optimizer_batch', 'critic_lr', 'secondary_weight', 'ppo_epochs', 'clip_ratio', 'target_kl',
                 'regret_weight', 'value_loss', 'value_beta', 'policy_mode', 'residual_scale', 'recover_target',
                 'calibration_games', 'calibration_validation_games', 'calibration_steps', 'calibration_stride',
                 'calibration_seed', 'allow_uncalibrated', 'calibration_ratio', 'gradient_every')
        return {'algorithm': ALGORITHM, 'trainer_sha256': base.file_hash(Path(__file__)),
                'runner_sha256': base.file_hash(Path(base.__file__)),
                **{name: getattr(self.args, name) for name in names}}

    def configure_model(self, resuming):
        if not resuming:
            self.model.config = replace(self.model.config, recover_target=self.args.recover_target,
                                        policy_mode=self.args.policy_mode, residual_scale=self.args.residual_scale,
                                        detach_critic=True, centered_critic=True)
            self.model.value[0].reset_parameters()
            self.model.value[-1].weight.data.zero_()
            self.model.value[-1].bias.data.zero_()
            if self.args.policy_mode == 'teacher-residual':
                self.model.rank_out.weight.data.zero_()
                self.model.rank_out.bias.data.zero_()
        self.calibration_actor = actor_digest(self.model)

    def create_optimizer(self):
        actor = [p for n, p in self.model.named_parameters() if not n.startswith(('value.', 'completion.'))]
        return torch.optim.AdamW([{'params': actor, 'lr': self.args.lr, 'name': 'actor'},
                                  {'params': list(self.model.value.parameters()), 'lr': self.args.critic_lr, 'name': 'critic'}],
                                 weight_decay=1e-4)

    def restore_extra(self, saved):
        self.calibration = saved['extra']['calibration']
        self.initial_digest = saved['extra']['initial_checkpoint_sha256']
        if self.calibration.get('episode') is not None:
            self.calibration_episode = base.restore_episode(self.calibration['episode'])

    def checkpoint_metadata(self):
        if self.calibration_episode is not None:
            self.calibration['episode'] = self.calibration_episode.snapshot()
        return {'algorithm': ALGORITHM, 'value_semantics': 'expected_remaining_placements_times_0.001',
                'completion_semantics': 'imitation_auxiliary_not_recalibrated_during_survival',
                'initial_checkpoint_sha256': getattr(self, 'initial_digest', None),
                'calibration': self.calibration, 'actor_critic_gradient_isolation': True}

    def __init__(self, args, budget):
        self.calibration = {'stage': 'collect', 'game_index': 0, 'train': [], 'validation': [], 'pending': [],
                            'completed_games': 0, 'censored_games': 0, 'fit_updates': 0, 'done': False,
                            'episode': None, 'fit_optimizer': None, 'best': None, 'best_weights': None}
        self.calibration_episode = None
        super().__init__(args, budget)
        if not hasattr(self, 'initial_digest'):
            self.initial_digest = base.file_hash(Path(self.initial_checkpoint)) if self.initial_checkpoint else None
        self.progress.setdefault('optimizer_steps', 0)
        self.progress.setdefault('kl_stopped_updates', 0)
        self.progress.setdefault('target_recoveries', 0)
        self.progress.setdefault('calibration_updates', self.calibration.get('fit_updates', 0))
        context = self.recorder.context
        self.recorder = CompressedRecorder(self.root / 'games', args.record_games, args.record_every, args.record_max_mib)
        self.recorder.context = {**context, 'algorithm': ALGORITHM, 'policy_mode': self.model.config.policy_mode}
        self.emit('survival_algorithm', algorithm=ALGORITHM, microbatch=args.batch, optimizer_batch=args.optimizer_batch,
                  policy_mode=self.model.config.policy_mode, reward='one placement * 0.001; gamma=1',
                  calibration_actor_frozen=True, critic_encoder_detached=True, **device_telemetry())

    def prepare_episode(self, episode):
        old = episode.target
        prepared = neural.prepare(episode.state, old, self.model.config)
        if prepared is not None:
            new = int(prepared.arrays['targets'])
            if new != old:
                if getattr(episode, 'pending_retarget', None) is None:
                    episode.pending_retarget = old
                episode.target = new
                self.progress['target_recoveries'] += 1
                self.emit('target_recovered', episode=episode.number, phase=episode.phase, pieces=episode.state.pieces,
                          from_target=old, to_target=new, reward_pieces=0, board_sha256=base.board_hash(episode.state.board))
        return prepared

    def _prepare_all(self):
        if self.prepared is None:
            self.prepared = [self.prepare_episode(e) for e in self.environments]

    def finish_episode(self, index, reason):
        episode = self.environments[index]
        episode.complete, episode.stop_reason = True, reason
        self.pending.append(episode)
        self.progress['finished_episodes'] += 1
        teacher_moves = sum(row['behavior'] == 'teacher' for row in episode.moves)
        self.emit('episode_end', episode=episode.number, stop_reason=reason, teacher_placements=teacher_moves,
                  autonomous_placements=episode.state.pieces - teacher_moves,
                  terminated=reason in ('top_out', 'planner_blocked'), truncated=reason == 'piece_limit',
                  **episode.state.summary())
        self.environments[index] = self.new_episode()
        self.prepared[index] = self.prepare_episode(self.environments[index])
        if self.prepared[index] is None:
            raise ValueError('No legal candidate on an empty board')

    def value_inputs(self, episodes):
        boards = torch.tensor([e.state.board for e in episodes], dtype=torch.int32, device=self.device)
        targets = torch.tensor([e.target for e in episodes], device=self.device)
        context = torch.from_numpy(np.stack([neural.state_context(e.state, e.target) for e in episodes])).to(self.device)
        with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
            encoded = self.model.encode(boards, targets, context)
        joined = torch.cat((encoded[1], context), 1)
        return encoded[2].float().cpu().numpy(), joined.float().detach().cpu()

    def calibrate(self):
        c = self.calibration
        if c['done']:
            return True
        if c['stage'] in ('failed', 'insufficient_uncensored_games'):
            self.budget.reason = 'calibration_failed'
            return False
        if self.args.calibration_games == 0:
            if not self.args.allow_uncalibrated:
                raise ValueError('Skipping calibration requires --allow-uncalibrated')
            c.update(done=True, stage='explicitly_skipped')
            self.emit('calibration_skipped', explicit=True)
            return True
        total = self.args.calibration_games + self.args.calibration_validation_games
        self.model.eval()
        while c['game_index'] < total:
            if self.budget.expired():
                return False
            if self.calibration_episode is None:
                index = c['game_index']
                self.calibration_episode = base.Episode(index, game.Game(self.args.calibration_seed + 2 * index, self.args.piece_set),
                                                        phase='calibration')
                c['pending'] = []
            episode = self.calibration_episode
            prepared = self.prepare_episode(episode)
            while prepared is not None and episode.state.pieces < self.args.pieces and not self.budget.expired():
                with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                    output = self.model(neural.collate([prepared.arrays], self.device))
                    logp = legacy.categorical_logp(output['logits'], torch.ones_like(output['logits'], dtype=torch.bool))
                    action = int(torch.multinomial(logp.exp(), 1).item())
                if episode.state.pieces % self.args.calibration_stride == 0:
                    c['pending'].append({'x': output['critic_input'][0].detach().float().cpu(), 'piece': episode.state.pieces})
                base.apply_move(episode, prepared.options[action], self.progress['updates'], 'frozen_calibration_policy')
                prepared = self.prepare_episode(episode)
                if episode.state.pieces % 512 == 0:
                    self.emit('calibration_progress', game=c['game_index'], total_games=total, pieces=episode.state.pieces)
                if time.monotonic() - self.saved_at >= self.args.save_seconds:
                    self.save('calibration_collect')
            if self.budget.expired():
                return False
            terminated = prepared is None
            episode.complete = True
            episode.stop_reason = base.stopped_reason(episode.state) if terminated else 'piece_limit'
            if terminated:
                _, joined = self.value_inputs([episode])
                c['pending'].append({'x': joined[0], 'piece': episode.state.pieces})
                destination = 'train' if c['game_index'] < self.args.calibration_games else 'validation'
                for row in c['pending']:
                    c[destination].append({'x': row['x'], 'y': (episode.state.pieces - row['piece']) * base.REWARD_SCALE,
                                           'seed': episode.state.seed})
                c['completed_games'] += 1
            else:
                c['censored_games'] += 1
            self.recorder.write(episode)
            self.emit('calibration_game', seed=episode.state.seed, pieces=episode.state.pieces,
                      terminated=terminated, truncated=not terminated, labeled=terminated,
                      split='train' if c['game_index'] < self.args.calibration_games else 'validation')
            c['game_index'] += 1
            c['pending'], c['episode'], self.calibration_episode = [], None, None
            self.save('calibration_game_complete')
        if not c['train'] or not c['validation']:
            c.update(stage='insufficient_uncensored_games', done=self.args.allow_uncalibrated)
            self.emit('calibration_failed', reason=c['stage'], completed=c['completed_games'], censored=c['censored_games'])
            self.budget.reason = None if c['done'] else 'calibration_failed'
            return c['done']
        x = torch.stack([row['x'] for row in c['train']]).to(self.device)
        y = torch.tensor([row['y'] for row in c['train']], dtype=torch.float32, device=self.device)
        vx = torch.stack([row['x'] for row in c['validation']]).to(self.device)
        vy = np.asarray([row['y'] for row in c['validation']], dtype=np.float64)
        constant = float(y.mean().item())
        optimizer = torch.optim.AdamW(self.model.value.parameters(), lr=self.args.critic_lr, weight_decay=1e-4)
        if c['fit_optimizer'] is not None:
            optimizer.load_state_dict(c['fit_optimizer'])
        elif c['fit_updates'] == 0:
            with torch.no_grad():
                hidden = (torch.sigmoid(self.model.value[0](x)) - .5).double().cpu()
                design = torch.cat((hidden, torch.ones(len(hidden), 1, dtype=torch.float64)), 1)
                ridge = torch.eye(design.shape[1], dtype=torch.float64) * 1e-3
                ridge[-1, -1] = 1e-9
                solution = torch.linalg.solve(design.T @ design + ridge, design.T @ y.double().cpu())
                self.model.value[-1].weight.copy_(solution[:-1][None].to(self.device, dtype=torch.float32))
                self.model.value[-1].bias.copy_(solution[-1:].to(self.device, dtype=torch.float32))
        def assess():
            with torch.no_grad():
                pred = self.model.critic(vx).float().cpu().numpy()
            metrics = regression_metrics(pred, vy, constant)
            if c['best'] is None or metrics['mse'] < c['best']['mse']:
                c['best'] = metrics
                c['best_weights'] = {k: v.detach().cpu().clone() for k, v in self.model.value.state_dict().items()}
            return metrics
        if c['best'] is None:
            assess()
        c['stage'] = 'fit'
        while c['fit_updates'] < self.args.calibration_steps:
            if self.budget.expired():
                c['fit_optimizer'] = optimizer.state_dict()
                return False
            indices = torch.randint(len(x), (min(self.args.optimizer_batch, len(x)),), device=self.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(self.model.critic(x[indices]), y[indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite calibration loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.value.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            c['fit_updates'] += 1
            self.progress['calibration_updates'] += 1
            if c['fit_updates'] % 25 == 0 or c['fit_updates'] == self.args.calibration_steps:
                self.emit('critic_calibration', fit_updates=c['fit_updates'], **assess())
            if time.monotonic() - self.saved_at >= self.args.save_seconds:
                c['fit_optimizer'] = optimizer.state_dict()
                self.save('critic_calibration')
        self.model.value.load_state_dict(c['best_weights'])
        metrics = c['best']
        passed = metrics['mse'] < metrics['constant_mse'] * self.args.calibration_ratio
        if metrics['near_end_samples']:
            passed &= metrics['near_end_mae_pieces'] < metrics['constant_near_end_mae_pieces']
        if actor_digest(self.model) != self.calibration_actor:
            raise AssertionError('Calibration changed actor parameters')
        report = {'algorithm': ALGORITHM, 'passed': bool(passed), 'actor_unchanged': True,
                  'actor_sha256': self.calibration_actor, 'train_samples': len(x), 'validation_samples': len(vx),
                  'completed_games': c['completed_games'], 'censored_games': c['censored_games'],
                  'criterion': f'validation MSE < {self.args.calibration_ratio} * training-mean baseline; lower near-end MAE',
                  'validation': metrics, 'fit_updates': c['fit_updates']}
        base.durable_json(self.root / 'critic-calibration.json', report)
        c.update(stage='passed' if passed else 'failed', done=bool(passed or self.args.allow_uncalibrated))
        c['train'], c['validation'], c['fit_optimizer'], c['best_weights'] = [], [], None, None
        self.emit('calibration_complete', **report, override=bool(not passed and self.args.allow_uncalibrated))
        if not c['done']:
            self.budget.reason = 'calibration_failed'
        return c['done']

    def rollout(self, warmup):
        timings = {'prepare_seconds': 0., 'inference_transfer_seconds': 0., 'commit_seconds': 0., 'bootstrap_seconds': 0.}
        started = time.perf_counter()
        self._prepare_all()
        timings['prepare_seconds'] += time.perf_counter() - started
        records, choices, labels, old = [], [], [], []
        values, next_values, rewards, terminals, truncations = [], [], [], [], []
        self.model.eval()
        for step in range(self.args.rollout):
            if self.budget.expired():
                break
            for i, prepared in enumerate(self.prepared):
                if prepared is None:
                    self.finish_episode(i, base.stopped_reason(self.environments[i].state))
            started = time.perf_counter()
            batch = neural.collate([p.arrays for p in self.prepared], self.device)
            with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                prediction = self.model(batch)
                logp = legacy.categorical_logp(prediction['logits'], batch['valid'])
                selected = [p.teacher for p in self.prepared] if warmup else torch.multinomial(logp.exp(), 1).squeeze(1).cpu().tolist()
                current_values = prediction['value'].float().cpu().numpy()
                probabilities = logp.detach().cpu()
            timings['inference_transfer_seconds'] += time.perf_counter() - started
            if values:
                continuing = ~(terminals[-1] | truncations[-1])
                next_values[-1][continuing] = current_values[continuing]
            values.append(current_values)
            terminal = np.zeros(self.args.envs, dtype=np.bool_)
            truncated = np.zeros(self.args.envs, dtype=np.bool_)
            following_values = np.zeros(self.args.envs, dtype=np.float32)
            for i, index in enumerate(selected):
                episode, prepared = self.environments[i], self.prepared[i]
                records.append(prepared.arrays)
                choices.append(index)
                labels.append(prepared.teacher)
                old.append(probabilities[i, :len(prepared.options)].clone())
                started = time.perf_counter()
                move = base.apply_move(episode, prepared.options[index], self.progress['updates'],
                                       'teacher' if warmup else 'stochastic_neural')
                timings['commit_seconds'] += time.perf_counter() - started
                self.progress['placements'] += 1
                self.progress['clears'] += move.layers
                started = time.perf_counter()
                self.prepared[i] = self.prepare_episode(episode)
                timings['prepare_seconds'] += time.perf_counter() - started
                terminal[i] = self.prepared[i] is None
                truncated[i] = episode.state.pieces >= self.args.pieces and not terminal[i]
            capped = np.flatnonzero(truncated).tolist()
            if capped:
                started = time.perf_counter()
                following_values[capped] = self.value_inputs([self.environments[i] for i in capped])[0]
                timings['bootstrap_seconds'] += time.perf_counter() - started
            rewards.append(np.full(self.args.envs, base.REWARD_SCALE, dtype=np.float32))
            terminals.append(terminal)
            truncations.append(truncated)
            next_values.append(following_values)
            for i in range(self.args.envs):
                if terminal[i] or truncated[i]:
                    self.finish_episode(i, 'piece_limit' if truncated[i] else base.stopped_reason(self.environments[i].state))
        if not records:
            return None
        final = np.flatnonzero(~(terminals[-1] | truncations[-1])).tolist()
        if final:
            started = time.perf_counter()
            next_values[-1][final] = self.value_inputs([self.environments[i] for i in final])[0]
            timings['bootstrap_seconds'] += time.perf_counter() - started
        advantage, returns = neural.gae(np.asarray(rewards), np.asarray(values), np.asarray(next_values),
                                        np.asarray(terminals), np.asarray(truncations), 1., self.args.gae_lambda)
        raw = advantage.ravel().copy()
        normalized = (raw - raw.mean()) / max(1e-6, float(raw.std()))
        return Rollout(records, choices, labels, normalized, returns.ravel(), old, np.asarray(values).ravel(),
                       raw, np.asarray(terminals).ravel(), np.asarray(truncations).ravel(), timings)

    old_batch = legacy.SurvivalTrainer.old_batch
    rollout_kl = legacy.SurvivalTrainer.rollout_kl

    def update(self, rollout, warmup):
        self.model.train()
        size = len(rollout.records)
        epoch_count = 1 if warmup else self.args.ppo_epochs
        stop, steps, finished_epochs, total_weight = False, 0, 0, 0
        totals, per_loss_norms = {}, {}
        parameters = tuple(p for n, p in self.model.named_parameters() if not n.startswith('completion.'))
        start_time = time.perf_counter()
        for epoch in range(epoch_count):
            full_kl = self.rollout_kl(rollout)
            if not warmup and full_kl > self.args.target_kl:
                stop = True
                break
            order = torch.randperm(size).tolist()
            for block in range(0, size, self.args.optimizer_batch):
                subset = order[block:block + self.args.optimizer_batch]
                self.optimizer.zero_grad(set_to_none=True)
                block_metrics = {}
                for start in range(0, len(subset), self.args.batch):
                    indices = subset[start:start + self.args.batch]
                    batch = neural.collate([rollout.records[i] for i in indices], self.device)
                    actions = torch.tensor([rollout.choices[i] for i in indices], device=self.device)
                    advantage = torch.as_tensor(rollout.advantages[indices], device=self.device)
                    returns = torch.as_tensor(rollout.returns[indices], device=self.device)
                    old = self.old_batch(rollout, indices, batch['valid'].shape[1])
                    with neural.precision_context(self.device, self.args.precision):
                        prediction = self.model(batch)
                        logp = legacy.categorical_logp(prediction['logits'], batch['valid'])
                        imitation, regret, winners = legacy.teacher_losses(logp, batch['features'], batch['valid'])
                        secondary = secondary_teacher_loss(logp, batch)
                        entropy = -(logp.exp() * torch.where(batch['valid'], logp, 0.)).sum(-1).mean()
                        critic = (F.mse_loss(prediction['value'].float(), returns) if self.args.value_loss == 'mse'
                                  else F.smooth_l1_loss(prediction['value'].float(), returns, beta=self.args.value_beta))
                        actor, clipped = legacy.clipped_policy_loss(logp, old, actions, advantage, self.args.clip_ratio)
                        components = {'actor': actor * (not warmup), 'critic': .5 * critic,
                                      'imitation': imitation * (1 if warmup else self.args.imitation),
                                      'secondary': self.args.secondary_weight * secondary,
                                      'regret': self.args.regret_weight * regret,
                                      'entropy': -self.args.entropy * entropy * (not warmup)}
                        loss = sum(components.values())
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Non-finite v2 PPO loss')
                    if not per_loss_norms and self.progress['runner_updates'] % self.args.gradient_every == 0:
                        per_loss_norms = {name: gradient_size(value, parameters) for name, value in components.items()}
                    fraction = len(indices) / len(subset)
                    self.scaler.scale(loss * fraction).backward()
                    picked = logp.argmax(-1)
                    rows = torch.arange(len(indices), device=self.device)
                    metrics = {'loss': loss, 'policy_loss': actor, 'value_loss': critic, 'tie_imitation_loss': imitation,
                               'secondary_teacher_loss': secondary, 'expected_utility_regret': regret, 'entropy': entropy,
                               'clip_fraction': clipped, 'utility_match': winners[rows, picked].float().mean(),
                               'teacher_agreement': (picked == torch.tensor([rollout.labels[i] for i in indices], device=self.device)).float().mean()}
                    for name, value in metrics.items():
                        block_metrics[name] = block_metrics.get(name, 0.) + float(value.detach().item()) * len(indices)
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.progress['updates'] += 1
                self.progress['optimizer_steps'] += 1
                steps += 1
                total_weight += len(subset)
                block_metrics['gradient_norm'] = float(norm) * len(subset)
                for name, value in block_metrics.items():
                    totals[name] = totals.get(name, 0.) + value
                with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                    local_kl = 0.
                    for start in range(0, len(subset), self.args.batch):
                        indices = subset[start:start + self.args.batch]
                        batch = neural.collate([rollout.records[i] for i in indices], self.device)
                        logp = legacy.categorical_logp(self.model(batch)['logits'], batch['valid'])
                        local_kl += float(legacy.exact_kl(self.old_batch(rollout, indices, logp.shape[1]), logp, batch['valid']).sum())
                if not warmup and local_kl / len(subset) > self.args.target_kl:
                    stop = True
                    break
            if not stop:
                finished_epochs += 1
            else:
                break
        if steps == 0:
            raise ValueError('Behaviour/current policy mismatch before optimizer update')
        final_kl = self.rollout_kl(rollout)
        self.progress['runner_updates'] += 1
        self.progress['warmup_updates' if warmup else 'survival_updates'] += 1
        self.progress['kl_stopped_updates'] += int(stop)
        diagnostics = regression_metrics(rollout.old_values, rollout.returns)
        self.emit('update', **self.progress, stage='warmup' if warmup else 'survival-ppo-v2',
                  **{k: v / total_weight for k, v in totals.items()}, per_loss_gradient_norms=per_loss_norms,
                  optimizer_steps_this_rollout=steps, completed_epochs=finished_epochs,
                  final_exact_kl=final_kl, kl_early_stop=stop, kl_check_scope='minibatch after each step; full rollout each epoch and at end',
                  rollout_placements=size, terminal_transitions=int(rollout.terminals.sum()),
                  truncated_transitions=int(rollout.truncations.sum()), critic_bootstrapped_diagnostics=diagnostics,
                  raw_advantages=tensor_summary(rollout.raw_advantages), phase_seconds={**rollout.timings, 'update': time.perf_counter() - start_time},
                  **neural.memory_report(self.device))

    def evaluate_policy(self, policy):
        runs = []
        self.model.eval()
        with base.preserve_rng(), torch.inference_mode():
            for index, seed in enumerate(self.args.selection_seeds):
                if self.budget.expired():
                    break
                episode = base.Episode(self.progress['runner_updates'] * len(self.args.selection_seeds) + index,
                                       game.Game(seed, self.args.piece_set), phase=f'selection-{policy}')
                started = time.perf_counter()
                reason, recoveries = 'piece_limit', self.progress['target_recoveries']
                while episode.state.pieces < self.args.selection_pieces:
                    if self.budget.expired():
                        reason = self.budget.reason
                        break
                    if policy == 'sustained':
                        decision = game.choose_sustained_move(episode.state.board, episode.state.current, episode.target,
                                                             self.model.config.candidates, episode.state.pieces)
                        candidate = decision.move.candidate if decision else None
                    else:
                        prepared = self.prepare_episode(episode)
                        candidate = None
                        if prepared is not None:
                            batch = neural.collate([prepared.arrays], self.device)
                            logits = self.model(batch)['logits']
                            legacy.categorical_logp(logits, batch['valid'])
                            candidate = prepared.options[int(logits[0].argmax().item())]
                    if candidate is None:
                        reason = base.stopped_reason(episode.state)
                        break
                    base.apply_move(episode, candidate, self.progress['updates'], f'greedy_{policy}')
                episode.stop_reason, episode.complete = reason, reason in ('top_out', 'planner_blocked', 'piece_limit')
                self.recorder.write(episode)
                report = {**episode.state.summary(), 'stop_reason': reason, 'evaluation_complete': episode.complete,
                          'capped': reason == 'piece_limit', 'policy': policy, 'policy_mode': self.model.config.policy_mode if policy != 'sustained' else 'legacy',
                          'target_recoveries': self.progress['target_recoveries'] - recoveries,
                          'wall_seconds': time.perf_counter() - started}
                runs.append(report)
                self.emit('selection_game', **report)
                if not episode.complete:
                    break
        complete = len(runs) == len(self.args.selection_seeds) and all(x['evaluation_complete'] for x in runs)
        return {'complete': complete, 'runs': runs, 'score': base.survival_score(runs) if complete else None,
                'signature': self.signature, 'score_definition': 'mean capped survival, then minimum, then median'}

    def save(self, reason='checkpoint'):
        started = time.perf_counter()
        summary = super().save(reason)
        manifest = {'schema': 'tetris4d-run-manifest-v2', 'algorithm': ALGORITHM,
                    'checkpoint_sha256': base.file_hash(self.last), 'runner_updates': self.progress['runner_updates'],
                    'metrics_bytes': self.metrics.stat().st_size, 'source_sha256': self.signature,
                    'policy_mode': self.model.config.policy_mode, 'calibration_stage': self.calibration['stage'],
                    'checkpoint_seconds': time.perf_counter() - started,
                    'note': 'Checkpoint and metrics offset are tied to this save; live metrics may continue.'}
        base.durable_json(self.root / 'manifest.json', manifest)
        self.emit('checkpoint_manifest', **manifest)
        return summary

    def run(self):
        self.save('started')
        if not self.calibrate():
            summary = self.save(self.budget.reason or 'calibration_incomplete')
            self.emit('run_finished', reason=self.budget.reason or 'calibration_incomplete', **self.progress)
            return summary
        return super().run()


def arguments(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--optimizer-batch', type=int, default=128)
    extra.add_argument('--critic-lr', type=float, default=3e-4)
    extra.add_argument('--secondary-weight', type=float, default=.02)
    extra.add_argument('--ppo-epochs', type=int, default=3)
    extra.add_argument('--clip-ratio', type=float, default=.2)
    extra.add_argument('--target-kl', type=float, default=.02)
    extra.add_argument('--regret-weight', type=float, default=.1)
    extra.add_argument('--value-loss', choices=('mse', 'huber'), default='mse')
    extra.add_argument('--value-beta', type=float, default=1.)
    extra.add_argument('--policy-mode', choices=('neural', 'teacher-residual'), default='teacher-residual')
    extra.add_argument('--residual-scale', type=float, default=1.)
    extra.add_argument('--recover-target', action=argparse.BooleanOptionalAction, default=True)
    extra.add_argument('--calibration-games', type=int, default=4)
    extra.add_argument('--calibration-validation-games', type=int, default=2)
    extra.add_argument('--calibration-steps', type=int, default=300)
    extra.add_argument('--calibration-stride', type=int, default=32)
    extra.add_argument('--calibration-seed', type=int, default=2300000)
    extra.add_argument('--calibration-ratio', type=float, default=.95)
    extra.add_argument('--allow-uncalibrated', action='store_true')
    extra.add_argument('--allow-concurrent', action='store_true')
    extra.add_argument('--gradient-every', type=int, default=10)
    options, remaining = extra.parse_known_args(supplied)
    if '-h' in supplied or '--help' in supplied:
        extra.print_help()
        print('V2: frozen-policy critic calibration, target recovery, teacher-residual policy, real PPO minibatches. Old runs remain separate.')
    args = base.arguments(['--run-dir', str(base.ROOT / 'runs/survival-ppo-v2'), '--rollout', '128', '--warmup-updates', '0'] + remaining)
    for key, value in vars(options).items():
        setattr(args, key, value)
    for key in ('optimizer_batch', 'critic_lr', 'value_beta', 'calibration_validation_games', 'calibration_stride', 'gradient_every'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            extra.error(f'{key} must be finite and positive')
    if not 1 <= args.ppo_epochs <= 16 or not 0 < args.clip_ratio < 1 or not 0 < args.target_kl < 1:
        extra.error('Invalid PPO epoch, clipping or KL setting')
    if not 0 <= args.residual_scale <= 4 or not 0 < args.calibration_ratio < 1:
        extra.error('Invalid residual scale or calibration threshold')
    if args.calibration_games < 0 or args.calibration_steps < 0:
        extra.error('Calibration counts must be nonnegative')
    if args.calibration_games == 0 and not args.allow_uncalibrated:
        extra.error('--calibration-games 0 requires explicit --allow-uncalibrated')
    if any(not math.isfinite(v) or v < 0 for v in (args.regret_weight, args.secondary_weight)):
        extra.error('Teacher-loss weights must be finite and nonnegative')
    if args.calibration_seed % 2 or args.calibration_seed < 2 * args.seed + 200000:
        extra.error('Calibration seeds must be even and use a separate range above training seeds')
    if args.checkpoint is None:
        for path in (base.ROOT / 'models/neural-rl.pt', base.ROOT / 'models/neural-policy.pt'):
            if path.is_file():
                args.checkpoint = path
                break
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    if args.verify_game:
        print(json.dumps(base.verify_game(args.verify_game), indent=2))
        return 0
    try:
        device = neural.device_for(args.device)
        with device_lock(device, args.allow_concurrent) as competing, base.run_lock(args.run_dir), base.Budget(args.hours) as budget:
            trainer = SurvivalTrainer(args, budget)
            trainer.emit('resource_isolation', competing_trainers=competing, override=args.allow_concurrent)
            trainer.run()
        return 2 if budget.reason == 'calibration_failed' else 0
    except KeyboardInterrupt:
        print('Interrupted. Resume the v2 run from its last completed checkpoint.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'V2 training stopped: {exc}. Existing v1 checkpoints remain unchanged.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
