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
import weakref

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


def survival_gae(rewards, values, next_values, terminated, truncated, lam, nonnegative_bootstrap=False):
    """Use the same nonnegative baseline on both sides of a survival TD error.

    The critic still trains on raw predictions so negative values receive a
    corrective gradient. Flooring only next_values would break the cancellation
    of intermediate baselines in lambda returns.
    """
    if nonnegative_bootstrap:
        values = np.maximum(values, 0.)
        next_values = np.maximum(next_values, 0.)
    return neural.gae(rewards, values, next_values, terminated, truncated, 1., lam)


def secondary_teacher_loss(logp, batch, winners=None):
    if winners is None:
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


def scalar_values(values):
    """Read exact scalar values together; callers retain Python summation order."""
    return torch.stack([value.detach() for value in values]).cpu().tolist() if values else []


def rollout_to_host(logp, values, teachers=None, coalesce=False, keep_logp_device=False):
    """Keep the original draw; pack its exactly representable index with float32 outputs."""
    drawn = (teachers if isinstance(teachers, torch.Tensor) else
             None if teachers is not None else torch.multinomial(logp.exp(), 1).squeeze(1))
    if coalesce and logp.dtype == torch.float32 and logp.shape[1] <= 2**24:
        width = 0 if keep_logp_device else logp.shape[1]
        columns = ([] if keep_logp_device else [logp.detach()]) + [values.detach().float()[:, None]]
        if drawn is not None:
            columns.append(drawn[:, None].float())
        packed = torch.cat(columns, dim=1).cpu()
        selected = list(teachers) if drawn is None else packed[:, -1].long().tolist()
        # Own these few values, rather than retaining the whole probability buffer per step.
        return selected, packed[:, width].numpy().copy(), logp.detach() if keep_logp_device else packed[:, :width]
    selected = list(teachers) if drawn is None else drawn.cpu().tolist()
    return selected, values.float().cpu().numpy(), logp.detach() if keep_logp_device else logp.detach().cpu()


def label_tensor(labels, device):
    if any(isinstance(value, torch.Tensor) for value in labels):
        return torch.stack([torch.as_tensor(value, device=device, dtype=torch.long) for value in labels])
    return torch.tensor(labels, device=device)


def device_rollout_to_host(logp, values, prepared, warmup=False):
    """Copy actions, value and exact candidate metadata once; keep behaviour logp on CUDA."""
    actions = prepared.teachers if warmup else torch.multinomial(logp.exp(), 1).squeeze(1)
    metadata = prepared.selected_metadata(actions)
    packed = torch.cat((metadata, values.detach().float().double()[:, None], actions.double()[:, None]), 1).cpu()
    candidates = prepared.candidates_from_metadata(packed[:, :7].tolist())
    return (packed[:, 8].long().tolist(), packed[:, 7].numpy().astype(np.float32),
            logp.detach(), candidates)


def cuda_available_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    return free + max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))


def clip_gradient_groups(optimizer, scope, maximum=1., coalesce=False):
    """Keep critic gradient magnitude from setting the actor's clipping scale."""
    if scope not in ('joint', 'separate') or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError('Invalid gradient clipping setting')
    groups = {group['name']: group['params'] for group in optimizer.param_groups}
    actor, critic = groups['actor'], groups['critic']
    norms = {}
    for name, parameters in (('actor', actor), ('critic', critic)):
        gradients = [p.grad for p in parameters if p.grad is not None]
        norms[name] = (torch.nn.utils.get_total_norm(gradients, error_if_nonfinite=True) if gradients
                       else torch.zeros((), device=parameters[0].device))
    if scope == 'joint':
        total = torch.nn.utils.clip_grad_norm_(actor + critic, maximum, error_if_nonfinite=True)
    else:
        torch.nn.utils.clip_grad_norm_(actor, maximum, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(critic, maximum, error_if_nonfinite=True)
        total = torch.linalg.vector_norm(torch.stack(tuple(norms.values())))
    if coalesce:
        return dict(zip(('actor_gradient_norm', 'critic_gradient_norm', 'gradient_norm'),
                        scalar_values([norms['actor'], norms['critic'], total])))
    return {'actor_gradient_norm': float(norms['actor']), 'critic_gradient_norm': float(norms['critic']),
            'gradient_norm': float(total)}


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
            names = {'train_overnight.py', 'train_survival.py', 'train_survival_v2.py', 'train_survival_v3.py'}
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
    gpu_prepare: dict | None = None


class RolloutCache:
    """Immutable inputs on the update device; minibatches retain original padding."""
    CANDIDATE_KEYS = neural.CANDIDATE_FIELDS

    @staticmethod
    def required_bytes(rollout):
        records = rollout.records
        count, maximum = len(records), max(len(row['features']) for row in records)
        total = 0
        for name in records[0]:
            value = records[0][name]
            if any(isinstance(row[name], torch.Tensor) for row in records):
                value = torch.as_tensor(value)
                dtype = value.dtype
                for row in records[1:]:
                    dtype = torch.promote_types(dtype, torch.as_tensor(row[name]).dtype)
                itemsize = torch.empty((), dtype=dtype).element_size()
            else:
                value = np.asarray(value)
                itemsize = np.result_type(*(np.asarray(row[name]).dtype for row in records)).itemsize
            shape = (maximum, *value.shape[1:]) if name in RolloutCache.CANDIDATE_KEYS else value.shape
            total += count * math.prod(shape) * itemsize
        # Padded float32 behaviour distribution, integer actions/labels, original target dtypes.
        return total + count * (maximum * 4 + 16 + rollout.advantages.dtype.itemsize + rollout.returns.dtype.itemsize)

    def __init__(self, rollout, device):
        self.rollout = rollout
        self.device = device
        self.lengths = [len(row['features']) for row in rollout.records]
        self.inputs = neural.collate(rollout.records, device)
        maximum = self.inputs['valid'].shape[1]
        if any(row.device.type != 'cpu' for row in rollout.old_logp):
            old = torch.nn.utils.rnn.pad_sequence([row.to(device=device, dtype=torch.get_default_dtype())
                                                 for row in rollout.old_logp],
                                                batch_first=True, padding_value=-torch.inf)
        else:
            old = torch.full((len(self.lengths), maximum), -torch.inf)
            for index, row in enumerate(rollout.old_logp):
                old[index, :len(row)] = row
        self.targets = {
            'old': old.to(device),
            'actions': torch.tensor(rollout.choices, device=device),
            'labels': label_tensor(rollout.labels, device),
            'advantage': torch.as_tensor(rollout.advantages, device=device),
            'returns': torch.as_tensor(rollout.returns, device=device),
        }
        self.bytes = sum(value.numel() * value.element_size() for value in (*self.inputs.values(), *self.targets.values()))

    def select(self, indices, targets=True):
        maximum = max(self.lengths[index] for index in indices)
        selected = torch.tensor(indices, dtype=torch.long, device=self.device)
        # Slice before gathering: identical shape and contiguous strides to fresh collation.
        batch = {name: (value[:, :maximum] if name in self.CANDIDATE_KEYS else value).index_select(0, selected)
                 for name, value in self.inputs.items()}
        names = self.targets if targets else ('old',)
        values = {name: (self.targets[name][:, :maximum] if name == 'old' else self.targets[name]).index_select(0, selected)
                  for name in names}
        return batch, values


class SurvivalTrainer(base.Trainer):
    algorithm = ALGORITHM

    def reference_signature(self):
        weight = getattr(self.args, 'reference_kl_weight', 0.)
        if not weight:
            return {}
        path = self.args.reference_checkpoint
        if path is None or not Path(path).is_file():
            raise ValueError('Reference KL requires an immutable --reference-checkpoint')
        return {'reference_kl_weight': weight, 'reference_checkpoint_sha256': base.file_hash(Path(path)),
                'reference_kl_direction': 'frozen_reference_to_current'}

    def training_signature(self):
        names = ('optimizer_batch', 'critic_lr', 'secondary_weight', 'ppo_epochs', 'clip_ratio', 'target_kl',
                 'regret_weight', 'value_loss', 'value_beta', 'policy_mode', 'residual_scale', 'teacher_prior_scale',
                 'preserve_initial_actor', 'gradient_clip_scope', 'recover_target',
                 'calibration_games', 'calibration_validation_games', 'calibration_steps', 'calibration_stride',
                 'calibration_seed', 'allow_uncalibrated', 'calibration_ratio', 'gradient_every')
        return {**self.reference_signature(), 'algorithm': self.algorithm, 'trainer_sha256': base.file_hash(Path(__file__)),
                'runner_sha256': base.file_hash(Path(base.__file__)),
                **{name: getattr(self.args, name) for name in names},
                'cache_rollout': getattr(self.args, 'cache_rollout', False),
                'nonnegative_bootstrap': getattr(self.args, 'nonnegative_bootstrap', False),
                'freeze_encoder': getattr(self.args, 'freeze_encoder', False),
                'rollout_cache_mib': getattr(self.args, 'rollout_cache_mib', 1024),
                'coalesce_transfers': getattr(self.args, 'coalesce_transfers', False),
                'fused_encoding': getattr(self.args, 'fused_encoding', False),
                'gpu_prepare': getattr(self.args, 'gpu_prepare', False),
                'gpu_preparer_sha256': (base.file_hash(Path(__file__).with_name('tetris_gpu_simulation.py'))
                                        if getattr(self.args, 'gpu_prepare', False) else None),
                'gpu_encoding_sha256': (base.file_hash(Path(__file__).with_name('tetris_gpu_encoding.py'))
                                        if getattr(self.args, 'fused_encoding', False) else None)}

    def policy_logp(self, logits, valid):
        if not getattr(self.args, 'coalesce_transfers', False):
            return legacy.categorical_logp(logits, valid)
        valid = valid.bool()
        # Ignore padding just as logits[valid] does, without dynamic Boolean indexing.
        checks = valid.any(-1).all() & torch.isfinite(torch.where(valid, logits, 0.)).all()
        if not bool(checks):
            raise ValueError('Invalid candidate mask or non-finite action logits.')
        return F.log_softmax(logits.float().masked_fill(~valid, -torch.inf), dim=-1)

    def configure_model(self, resuming):
        self.model.fused_encoding = getattr(self.args, 'fused_encoding', False)
        if not resuming:
            self.model.config = replace(self.model.config, recover_target=self.args.recover_target,
                                        policy_mode=self.args.policy_mode, residual_scale=self.args.residual_scale,
                                        teacher_prior_scale=self.args.teacher_prior_scale,
                                        detach_critic=True, centered_critic=True)
            self.model.value[0].reset_parameters()
            self.model.value[-1].weight.data.zero_()
            self.model.value[-1].bias.data.zero_()
            if self.args.policy_mode == 'teacher-residual' and not self.args.preserve_initial_actor:
                self.model.rank_out.weight.data.zero_()
                self.model.rank_out.bias.data.zero_()
        # Checkpoints store tensors, so reapply the encoder freeze on resume.
        if getattr(self.args, 'freeze_encoder', False):
            self.model.stem.requires_grad_(False)
            self.model.blocks.requires_grad_(False)
        self.calibration_actor = actor_digest(self.model)
        self.initial_actor_sha256 = self.calibration_actor
        if not resuming and getattr(self.args, 'reference_kl_weight', 0.):
            with base.preserve_rng():
                reference, checkpoint = neural.load_checkpoint(self.args.reference_checkpoint, self.device)
            if checkpoint['teacher_parameters'] != game.network_parameters():
                raise ValueError('Reference policy uses different candidate-pipeline weights')
            if base.file_hash(Path(self.args.reference_checkpoint)) != self.signature['reference_checkpoint_sha256']:
                raise ValueError('Reference checkpoint changed while loading')
            self.set_reference(reference)

    def set_reference(self, reference):
        # Critic-only flags do not affect candidate generation or the actor distribution.
        def actor_config(model):
            return {key: value for key, value in asdict(model.config).items()
                    if key not in ('detach_critic', 'centered_critic')}
        if actor_config(reference) != actor_config(self.model):
            raise ValueError('Reference and current policy must use the same actor and candidate configuration')
        reference.requires_grad_(False)
        reference.eval()
        reference.fused_encoding = getattr(self.args, 'fused_encoding', False)
        self.reference_policy = reference
        self.reference_actor_sha256 = actor_digest(reference)

    @torch.no_grad()
    def reference_logp(self, batch):
        return self.policy_logp(self.reference_policy(batch)['logits'], batch['valid'])

    def create_optimizer(self):
        actor = [p for n, p in self.model.named_parameters()
                 if p.requires_grad and not n.startswith(('value.', 'completion.'))]
        return torch.optim.AdamW([{'params': actor, 'lr': self.args.lr, 'name': 'actor'},
                                  {'params': list(self.model.value.parameters()), 'lr': self.args.critic_lr, 'name': 'critic'}],
                                 weight_decay=1e-4)

    def restore_extra(self, saved):
        self.calibration = saved['extra']['calibration']
        self.initial_digest = saved['extra']['initial_checkpoint_sha256']
        self.initial_actor_sha256 = saved['extra'].get('initial_actor_sha256', self.initial_actor_sha256)
        if self.calibration.get('episode') is not None:
            self.calibration_episode = base.restore_episode(self.calibration['episode'])
        if getattr(self.args, 'reference_kl_weight', 0.):
            reference = saved['extra'].get('frozen_reference')
            if not reference or reference['checkpoint_sha256'] != self.signature['reference_checkpoint_sha256']:
                raise ValueError('Missing or changed frozen reference in the resumed experiment')
            with base.preserve_rng():
                model = neural.ColumnPolicy(neural.Config(**reference['config'])).to(self.device)
            model.load_state_dict(reference['model'], strict=True)
            if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
                raise ValueError('Non-finite frozen reference parameters')
            self.set_reference(model)
            if self.reference_actor_sha256 != reference['actor_sha256']:
                raise ValueError('Frozen reference digest does not match its stored weights')

    def checkpoint_metadata(self):
        if self.calibration_episode is not None:
            self.calibration['episode'] = self.calibration_episode.snapshot()
        reference = ({'frozen_reference': {
            'checkpoint_sha256': self.signature['reference_checkpoint_sha256'],
            'actor_sha256': self.reference_actor_sha256, 'config': asdict(self.reference_policy.config),
            'model': {name: value.detach().cpu().clone() for name, value in self.reference_policy.state_dict().items()}}}
            if getattr(self, 'reference_policy', None) is not None else {})
        return {**reference, 'algorithm': self.algorithm, 'value_semantics': 'expected_remaining_placements_times_0.001',
                'completion_semantics': 'imitation_auxiliary_not_recalibrated_during_survival',
                'initial_checkpoint_sha256': getattr(self, 'initial_digest', None),
                'initial_actor_sha256': self.initial_actor_sha256,
                'calibration': self.calibration, 'critic_encoder_detached': True,
                'actor_critic_gradient_isolation': self.args.gradient_clip_scope == 'separate',
                'preserve_initial_actor': self.args.preserve_initial_actor,
                'nonnegative_bootstrap': getattr(self.args, 'nonnegative_bootstrap', False),
                'freeze_encoder': getattr(self.args, 'freeze_encoder', False),
                'gradient_clip_scope': self.args.gradient_clip_scope}

    def __init__(self, args, budget):
        if getattr(args, 'freeze_encoder', False) and self.algorithm != ALGORITHM:
            raise ValueError('--freeze-encoder requires the shared column encoder of the V2 trainer')
        self.calibration = {'stage': 'collect', 'game_index': 0, 'train': [], 'validation': [], 'pending': [],
                            'completed_games': 0, 'censored_games': 0, 'fit_updates': 0, 'done': False,
                            'episode': None, 'fit_optimizer': None, 'best': None, 'best_weights': None}
        self.calibration_episode = None
        self.reference_policy = None
        self._gpu_preparer = self._prepared_gpu_batch = None
        self._gpu_prepare_active = False
        super().__init__(args, budget)
        if not hasattr(self, 'initial_digest'):
            self.initial_digest = base.file_hash(Path(self.initial_checkpoint)) if self.initial_checkpoint else None
        self.progress.setdefault('optimizer_steps', 0)
        self.progress.setdefault('kl_stopped_updates', 0)
        self.progress.setdefault('target_recoveries', 0)
        self.progress.setdefault('calibration_updates', self.calibration.get('fit_updates', 0))
        context = self.recorder.context
        self.recorder = CompressedRecorder(self.root / 'games', args.record_games, args.record_every, args.record_max_mib)
        self.recorder.context = {**context, 'algorithm': self.algorithm, 'policy_mode': self.model.config.policy_mode}
        base.library.set_context(source=self.algorithm, policy_mode=self.model.config.policy_mode)
        self.emit('survival_algorithm', algorithm=self.algorithm, microbatch=args.batch, optimizer_batch=args.optimizer_batch,
                  policy_mode=self.model.config.policy_mode, reward='one placement * 0.001; gamma=1',
                  teacher_prior_scale=self.model.config.teacher_prior_scale, preserve_initial_actor=args.preserve_initial_actor,
                  nonnegative_bootstrap=getattr(args, 'nonnegative_bootstrap', False),
                  freeze_encoder=getattr(args, 'freeze_encoder', False),
                  gradient_clip_scope=args.gradient_clip_scope, initial_actor_sha256=self.initial_actor_sha256,
                  calibration_actor_frozen=True, critic_encoder_detached=True, **device_telemetry())

    def prepare_episode(self, episode):
        old = episode.target
        prepared = neural.prepare(episode.state, old, self.model.config)
        if prepared is not None:
            self.note_target(episode, int(prepared.arrays['targets']))
        return prepared

    def note_target(self, episode, new):
        old = episode.target
        if new != old:
            if getattr(episode, 'pending_retarget', None) is None:
                episode.pending_retarget = old
            episode.target = new
            self.progress['target_recoveries'] += 1
            self.emit('target_recovered', episode=episode.number, phase=episode.phase, pieces=episode.state.pieces,
                      from_target=old, to_target=new, reward_pieces=0, board_sha256=base.board_hash(episode.state.board))

    def gpu_prepare_plan(self):
        requested = getattr(self.args, 'gpu_prepare', False)
        plan = {'requested': requested, 'enabled': False}
        if not requested:
            return plan
        if self.device.type != 'cuda' or game.BACKEND != 'native':
            raise ValueError('--gpu-prepare requires CUDA and the native reference backend')
        # Retain owned input records for the whole rollout, plus their optional update cache.
        shortlist = self.model.config.shortlist
        per_record = 4096 * 4 + 8 + neural.CONTEXT * 4 + shortlist * (
            neural.FEATURES * 4 + neural.COLUMNS * 9 + 1 + 4 + 4) + 24
        if self.model.config.architecture in neural.SPATIAL_ARCHITECTURES:
            per_record += 6 * 4 + shortlist * 18 * 4
        per_record += shortlist * 4 * getattr(self.model, 'NEXT_FEATURES', 0)
        retained = per_record * self.args.envs * self.args.rollout
        free = cuda_available_bytes(self.device)
        limit = getattr(self.args, 'rollout_cache_mib', 1024) * 2**20
        needed = retained * (2 if getattr(self.args, 'cache_rollout', False) else 1) + 128 * 2**20
        if retained > limit or needed > free // 2:
            return {**plan, 'reason': 'memory_bound', 'estimated_retained_bytes': retained,
                    'estimated_with_cache_bytes': needed, 'free_bytes': free, 'limit_bytes': limit}
        return {**plan, 'enabled': True, 'estimated_retained_bytes': retained}

    def _prepare_all(self):
        if getattr(self, '_gpu_prepare_active', False):
            if self._prepared_gpu_batch is None:
                try:
                    if self._gpu_preparer is None:
                        from tetris_gpu_simulation import GpuPreparer
                        self._gpu_preparer = GpuPreparer(self.model.config, self.device)
                    result = self._gpu_preparer.prepare_batch([e.state for e in self.environments],
                                                              [e.target for e in self.environments])
                except torch.OutOfMemoryError:
                    # Preparation consumes no policy RNG and commits no move: continue from
                    # the same states on CPU, retaining any already collected device records.
                    self._gpu_prepare_active = False
                    self._prepared_gpu_batch = None
                    plan = getattr(self, '_gpu_prepare_status', None)
                    if plan is not None:
                        plan.update(enabled=bool(self._gpu_prepare_stats['batch_calls']),
                                    runtime_fallback='allocation_failed')
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    self.prepared = [self.prepare_episode(e) for e in self.environments]
                    return
                self._accept_gpu_batch(result)
            return
        if self.prepared is None:
            self.prepared = [self.prepare_episode(e) for e in self.environments]

    def _accept_gpu_batch(self, result, calls=1):
        stats = getattr(self, '_gpu_prepare_stats', None)
        if stats is not None:
            stats['batch_calls'] += calls
            stats['environment_calls'] += len(self.environments)
            for reason in result.cpu_fallbacks:
                if reason:
                    stats['cpu_fallbacks'][reason] = stats['cpu_fallbacks'].get(reason, 0) + 1
        self.prepared = list(result.prepared)
        self._prepared_gpu_batch = result
        for episode, prepared, target in zip(self.environments, self.prepared, result.targets):
            if prepared is not None:
                self.note_target(episode, target)

    def committed_environment(self, index):
        """Hook after environment `index` committed its move (GPU preparation active)."""

    def finish_episode(self, index, reason):
        episode = self.environments[index]
        episode.complete, episode.stop_reason = True, reason
        base.library.finish(episode.state, reason)
        self.pending.append(episode)
        self.progress['finished_episodes'] += 1
        teacher_moves = sum(row['behavior'] == 'teacher' for row in episode.moves)
        self.emit('episode_end', episode=episode.number, stop_reason=reason, teacher_placements=teacher_moves,
                  autonomous_placements=episode.state.pieces - teacher_moves,
                  terminated=reason in ('top_out', 'planner_blocked'), truncated=reason == 'piece_limit',
                  **episode.state.summary())
        self.environments[index] = self.new_episode()
        self.prepared[index] = self.prepare_episode(self.environments[index])
        self._prepared_gpu_batch = None
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

    def initialize_calibration_output(self, x, y):
        with torch.no_grad():
            hidden = (torch.sigmoid(self.model.value[0](x)) - .5).double().cpu()
            design = torch.cat((hidden, torch.ones(len(hidden), 1, dtype=torch.float64)), 1)
            ridge = torch.eye(design.shape[1], dtype=torch.float64) * 1e-3
            ridge[-1, -1] = 1e-9
            solution = torch.linalg.solve(design.T @ design + ridge, design.T @ y.double().cpu())
            self.model.value[-1].weight.copy_(solution[:-1][None].to(self.device, dtype=torch.float32))
            self.model.value[-1].bias.copy_(solution[-1:].to(self.device, dtype=torch.float32))

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
                    logp = self.policy_logp(output['logits'], torch.ones_like(output['logits'], dtype=torch.bool))
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
            self.initialize_calibration_output(x, y)
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
        report = {'algorithm': self.algorithm, 'passed': bool(passed), 'actor_unchanged': True,
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
        self.release_previous_device_rollout()
        gpu_plan = self.gpu_prepare_plan()
        if getattr(self, '_gpu_prepare_active', False) or gpu_plan['enabled']:
            self.prepared = None
        self._gpu_prepare_active = gpu_plan['enabled']
        self._prepared_gpu_batch = None
        self._gpu_prepare_stats = {'batch_calls': 0, 'environment_calls': 0, 'cpu_fallbacks': {}}
        self._gpu_prepare_status = gpu_plan
        if self._gpu_prepare_active:
            gpu_plan['preparations'] = self._gpu_prepare_stats
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
            if self._gpu_prepare_active:
                self._prepare_all()
            started = time.perf_counter()
            device_prepared = self._prepared_gpu_batch
            if device_prepared is not None and device_prepared.environment_indices != tuple(range(self.args.envs)):
                raise ValueError('Incomplete GPU environment mapping after episode reset')
            batch = (device_prepared.batch if device_prepared is not None else
                     neural.collate([p.arrays for p in self.prepared], self.device))
            with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                prediction = self.model(batch)
                logp = self.policy_logp(prediction['logits'], batch['valid'])
                teachers = (device_prepared.teachers if device_prepared is not None else
                            [p.teacher for p in self.prepared]) if warmup else None
                if device_prepared is not None and getattr(self.args, 'coalesce_transfers', False):
                    selected, current_values, probabilities, candidates = device_rollout_to_host(
                        logp, prediction['value'], device_prepared, warmup)
                else:
                    selected, current_values, probabilities = rollout_to_host(
                        logp, prediction['value'], teachers, getattr(self.args, 'coalesce_transfers', False),
                        keep_logp_device=device_prepared is not None)
                    candidates = device_prepared.selected_candidates(selected) if device_prepared is not None else None
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
                old_row = probabilities[i, :len(prepared.options)]
                old.append(old_row if device_prepared is not None else old_row.clone())
                started = time.perf_counter()
                candidate = candidates[i] if candidates is not None else prepared.options[index]
                move = base.apply_move(episode, candidate, self.progress['updates'],
                                       'teacher' if warmup else 'stochastic_neural')
                timings['commit_seconds'] += time.perf_counter() - started
                self.progress['placements'] += 1
                self.progress['clears'] += move.layers
                if self._gpu_prepare_active:
                    self.committed_environment(i)
                else:
                    started = time.perf_counter()
                    self.prepared[i] = self.prepare_episode(episode)
                    timings['prepare_seconds'] += time.perf_counter() - started
                    terminal[i] = self.prepared[i] is None
                    truncated[i] = episode.state.pieces >= self.args.pieces and not terminal[i]
            if self._gpu_prepare_active:
                started = time.perf_counter()
                self._prepared_gpu_batch = None
                self._prepare_all()
                timings['prepare_seconds'] += time.perf_counter() - started
                terminal = np.asarray([p is None for p in self.prepared], dtype=np.bool_)
                truncated = np.asarray([e.state.pieces >= self.args.pieces and not terminal[i]
                                        for i, e in enumerate(self.environments)], dtype=np.bool_)
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
        advantage, returns = survival_gae(np.asarray(rewards), np.asarray(values), np.asarray(next_values),
                                         np.asarray(terminals), np.asarray(truncations), self.args.gae_lambda,
                                         getattr(self.args, 'nonnegative_bootstrap', False))
        raw = advantage.ravel().copy()
        normalized = (raw - raw.mean()) / max(1e-6, float(raw.std()))
        return Rollout(records, choices, labels, normalized, returns.ravel(), old, np.asarray(values).ravel(),
                       raw, np.asarray(terminals).ravel(), np.asarray(truncations).ravel(), timings,
                       gpu_prepare=gpu_plan if gpu_plan['requested'] else None)

    def policy_batch(self, rollout, indices):
        cache = getattr(self, '_rollout_cache', None)
        if cache is not None and cache.rollout is rollout:
            batch, values = cache.select(indices, targets=False)
            return batch, values['old']
        batch = neural.collate([rollout.records[i] for i in indices], self.device)
        return batch, self.old_batch(rollout, indices, batch['valid'].shape[1])

    def old_batch(self, rollout, indices, size):
        cache = getattr(self, '_rollout_cache', None)
        if cache is not None and cache.rollout is rollout and size <= cache.targets['old'].shape[1]:
            selected = torch.tensor(indices, dtype=torch.long, device=self.device)
            return cache.targets['old'][:, :size].index_select(0, selected)
        rows = [rollout.old_logp[i] for i in indices]
        if any(row.device.type != 'cpu' for row in rows):
            old = torch.nn.utils.rnn.pad_sequence([row.to(device=self.device, dtype=torch.get_default_dtype())
                                                 for row in rows], batch_first=True, padding_value=-torch.inf)
            return F.pad(old, (0, size - old.shape[1]), value=-torch.inf)
        return legacy.SurvivalTrainer.old_batch(self, rollout, indices, size)

    def rollout_kl(self, rollout):
        total = 0.
        pending = []
        coalesce = getattr(self.args, 'coalesce_transfers', False)
        with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
            for start in range(0, len(rollout.records), self.args.batch):
                indices = list(range(start, min(start + self.args.batch, len(rollout.records))))
                batch, old = self.policy_batch(rollout, indices)
                current = self.policy_logp(self.model(batch)['logits'], batch['valid'])
                value = legacy.exact_kl(old, current, batch['valid']).sum()
                if coalesce:
                    pending.append(value)
                else:
                    total += float(value.item())
        for value in scalar_values(pending):
            total += value
        return total / len(rollout.records)

    @torch.no_grad()
    def reference_rollout_kl(self, rollout):
        values = []
        with neural.precision_context(self.device, self.args.precision):
            for start in range(0, len(rollout.records), self.args.batch):
                indices = list(range(start, min(start + self.args.batch, len(rollout.records))))
                batch, _ = self.policy_batch(rollout, indices)
                current = self.policy_logp(self.model(batch)['logits'], batch['valid'])
                values.append(legacy.exact_kl(self.reference_logp(batch), current, batch['valid']).sum())
        return sum(scalar_values(values)) / len(rollout.records)

    def update_inputs(self, rollout, indices):
        cache = getattr(self, '_rollout_cache', None)
        if cache is not None and cache.rollout is rollout:
            return cache.select(indices)
        batch, old = self.policy_batch(rollout, indices)
        return batch, {'old': old, 'actions': torch.tensor([rollout.choices[i] for i in indices], device=self.device),
                       'labels': label_tensor([rollout.labels[i] for i in indices], self.device),
                       'advantage': torch.as_tensor(rollout.advantages[indices], device=self.device),
                       'returns': torch.as_tensor(rollout.returns[indices], device=self.device)}

    def create_rollout_cache(self, rollout):
        self._rollout_cache = None
        requested = getattr(self.args, 'cache_rollout', False)
        telemetry = {'requested': requested, 'enabled': False, 'bytes': 0}
        if not requested:
            return telemetry
        required = RolloutCache.required_bytes(rollout)
        limit = getattr(self.args, 'rollout_cache_mib', 1024) * 2**20
        if self.device.type == 'cuda':
            free, _ = torch.cuda.mem_get_info(self.device)
            # Leave at least three quarters of currently free device memory for model work.
            limit = min(limit, free // 4)
        if required > limit:
            return {**telemetry, 'estimated_bytes': required, 'reason': 'memory_bound', 'limit_bytes': limit}
        try:
            self._rollout_cache = RolloutCache(rollout, self.device)
        except (MemoryError, torch.OutOfMemoryError):
            # Allocation precedes every optimizer step; fallback cannot replay a partial update.
            self._rollout_cache = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return {**telemetry, 'estimated_bytes': required, 'reason': 'allocation_failed'}
        return {**telemetry, 'enabled': True, 'bytes': self._rollout_cache.bytes}

    @torch.no_grad()
    def actor_diagnostics(self, rollout):
        count = min(len(rollout.records), self.args.batch, 16)
        cache = getattr(self, '_rollout_cache', None)
        if cache is not None and cache.rollout is rollout:
            batch, inputs = cache.select(list(range(count)))
            actions = inputs['actions']
        else:
            batch = neural.collate(rollout.records[:count], self.device)
            actions = torch.tensor(rollout.choices[:count], device=self.device)
        diagnostics = self.model.ranking_diagnostics(batch)
        logp = self.policy_logp(self.model(batch)['logits'], batch['valid'])
        winners, deficit = legacy.utility_targets(batch['features'], batch['valid'])
        probability = logp.exp()
        rows = torch.arange(count, device=self.device)
        digest = actor_digest(self.model)
        diagnostics.update(lower_primary_probability=float(probability.masked_fill(winners, 0.).sum(-1).mean()),
                           expected_primary_deficit=float((probability * deficit * 16).sum(-1).mean()),
                           sampled_lower_primary_fraction=float((~winners[rows, actions]).float().mean()),
                           sampled_primary_deficit=float((deficit[rows, actions] * 16).mean()),
                           actor_sha256=digest, initial_actor_sha256=self.initial_actor_sha256,
                           actor_changed_from_initial=digest != self.initial_actor_sha256)
        return diagnostics

    def update(self, rollout, warmup):
        started = time.perf_counter()
        try:
            telemetry = self.create_rollout_cache(rollout)
            result = self._update(rollout, warmup, started, telemetry)
            if rollout.gpu_prepare is not None and rollout.gpu_prepare['enabled']:
                self._previous_device_rollout = weakref.ref(rollout)
            return result
        finally:
            self._rollout_cache = None

    def release_previous_device_rollout(self):
        """Retire inputs after their optimizer update, before base.run collects its RHS."""
        reference = getattr(self, '_previous_device_rollout', None)
        previous = reference() if reference is not None else None
        if previous is not None:
            previous.records.clear()
            previous.old_logp.clear()
            previous.labels.clear()
        self._previous_device_rollout = None

    def _update(self, rollout, warmup, start_time, cache_telemetry):
        self.model.train()
        size = len(rollout.records)
        epoch_count = 1 if warmup else self.args.ppo_epochs
        stop, steps, finished_epochs, total_weight = False, 0, 0, 0
        totals, per_loss_norms = {}, {}
        parameters = tuple(p for n, p in self.model.named_parameters()
                           if p.requires_grad and not n.startswith('completion.'))
        measure_actor = self.progress['runner_updates'] % self.args.gradient_every == 0
        coalesce = getattr(self.args, 'coalesce_transfers', False)
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
                pending_metrics = []
                for start in range(0, len(subset), self.args.batch):
                    indices = subset[start:start + self.args.batch]
                    batch, inputs = self.update_inputs(rollout, indices)
                    actions, advantage, returns, old = (inputs[key] for key in ('actions', 'advantage', 'returns', 'old'))
                    with neural.precision_context(self.device, self.args.precision):
                        prediction = self.model(batch)
                        logp = self.policy_logp(prediction['logits'], batch['valid'])
                        imitation, regret, winners = legacy.teacher_losses(logp, batch['features'], batch['valid'])
                        secondary = secondary_teacher_loss(logp, batch, winners if coalesce else None)
                        entropy = -(logp.exp() * torch.where(batch['valid'], logp, 0.)).sum(-1).mean()
                        critic = (F.mse_loss(prediction['value'].float(), returns) if self.args.value_loss == 'mse'
                                  else F.smooth_l1_loss(prediction['value'].float(), returns, beta=self.args.value_beta))
                        actor, clipped = legacy.clipped_policy_loss(logp, old, actions, advantage, self.args.clip_ratio)
                        components = {'actor': actor * (not warmup), 'critic': .5 * critic,
                                      'imitation': imitation * (1 if warmup else self.args.imitation),
                                      'secondary': self.args.secondary_weight * secondary,
                                      'regret': self.args.regret_weight * regret,
                                      'entropy': -self.args.entropy * entropy * (not warmup)}
                        if getattr(self, 'reference_policy', None) is not None:
                            reference_kl = legacy.exact_kl(self.reference_logp(batch), logp, batch['valid']).mean()
                            components['reference_kl'] = self.args.reference_kl_weight * reference_kl
                        loss = sum(components.values())
                    if not bool(torch.isfinite(loss)):
                        raise ValueError('Non-finite v2 PPO loss')
                    if not per_loss_norms and measure_actor:
                        per_loss_norms = {name: gradient_size(value, parameters) for name, value in components.items()}
                    fraction = len(indices) / len(subset)
                    self.scaler.scale(loss * fraction).backward()
                    picked = logp.argmax(-1)
                    rows = torch.arange(len(indices), device=self.device)
                    metrics = {'loss': loss, 'policy_loss': actor, 'value_loss': critic, 'tie_imitation_loss': imitation,
                               'secondary_teacher_loss': secondary, 'expected_utility_regret': regret, 'entropy': entropy,
                               'clip_fraction': clipped, 'utility_match': winners[rows, picked].float().mean(),
                               'teacher_agreement': (picked == inputs['labels']).float().mean()}
                    if getattr(self, 'reference_policy', None) is not None:
                        metrics['reference_kl'] = reference_kl
                    if coalesce:
                        pending_metrics.append((len(indices), {name: value.detach() for name, value in metrics.items()}))
                    else:
                        for name, value in metrics.items():
                            block_metrics[name] = block_metrics.get(name, 0.) + float(value.detach().item()) * len(indices)
                if pending_metrics:
                    measured = iter(scalar_values([value for _, metrics in pending_metrics for value in metrics.values()]))
                    for count, metrics in pending_metrics:
                        for name in metrics:
                            block_metrics[name] = block_metrics.get(name, 0.) + next(measured) * count
                self.scaler.unscale_(self.optimizer)
                gradient_norms = clip_gradient_groups(self.optimizer, self.args.gradient_clip_scope, coalesce=coalesce)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.progress['updates'] += 1
                self.progress['optimizer_steps'] += 1
                steps += 1
                total_weight += len(subset)
                for name, value in gradient_norms.items():
                    block_metrics[name] = value * len(subset)
                for name, value in block_metrics.items():
                    totals[name] = totals.get(name, 0.) + value
                with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                    local_kl = 0.
                    pending_kl = []
                    for start in range(0, len(subset), self.args.batch):
                        indices = subset[start:start + self.args.batch]
                        batch, old = self.policy_batch(rollout, indices)
                        logp = self.policy_logp(self.model(batch)['logits'], batch['valid'])
                        value = legacy.exact_kl(old, logp, batch['valid']).sum()
                        if coalesce:
                            pending_kl.append(value)
                        else:
                            local_kl += float(value)
                    for value in scalar_values(pending_kl):
                        local_kl += value
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
        actor_diagnostics = self.actor_diagnostics(rollout) if measure_actor else {}
        reference_diagnostics = ({'final_reference_kl': self.reference_rollout_kl(rollout),
                                  'reference_kl_weight': self.args.reference_kl_weight,
                                  'reference_actor_sha256': self.reference_actor_sha256}
                                 if getattr(self, 'reference_policy', None) is not None else {})
        self.emit('update', **self.progress, stage='warmup' if warmup else self.algorithm,
                  **reference_diagnostics,
                  **{k: v / total_weight for k, v in totals.items()}, per_loss_gradient_norms=per_loss_norms,
                  gradient_clip_scope=self.args.gradient_clip_scope, actor_diagnostics=actor_diagnostics,
                  rollout_cache=cache_telemetry,
                  **({'gpu_prepare': rollout.gpu_prepare} if rollout.gpu_prepare is not None else {}),
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
                            self.policy_logp(logits, batch['valid'])
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
        manifest = {'schema': 'tetris4d-run-manifest-v2', 'algorithm': self.algorithm,
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
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--optimizer-batch', type=int, default=128)
    extra.add_argument('--critic-lr', type=float, default=3e-4)
    extra.add_argument('--secondary-weight', type=float, default=.02)
    extra.add_argument('--ppo-epochs', type=int, default=3)
    extra.add_argument('--clip-ratio', type=float, default=.2)
    extra.add_argument('--target-kl', type=float, default=.02)
    extra.add_argument('--reference-kl-weight', type=float, default=0.,
                       help='Opt-in fixed weight for KL(frozen reference || current); use a new experiment directory')
    extra.add_argument('--reference-checkpoint', type=Path,
                       help='Immutable initializer with the same actor/candidate configuration; preserved for resume')
    extra.add_argument('--regret-weight', type=float, default=.1)
    extra.add_argument('--value-loss', choices=('mse', 'huber'), default='mse')
    extra.add_argument('--value-beta', type=float, default=1.)
    extra.add_argument('--nonnegative-bootstrap', action='store_true',
                       help='Floor survival rollout baselines and bootstrap values at zero while fitting the raw critic')
    extra.add_argument('--freeze-encoder', action='store_true',
                       help='Keep the V2 column encoder fixed while optimizing the actor and critic heads')
    extra.add_argument('--policy-mode', choices=('neural', 'teacher-residual', 'teacher-prior'), default='teacher-residual')
    extra.add_argument('--residual-scale', type=float, default=1.)
    extra.add_argument('--teacher-prior-scale', type=float, default=1.,
                       help='Scale the teacher prior only in the opt-in teacher-prior policy')
    extra.add_argument('--preserve-initial-actor', action='store_true',
                       help='Retain imported actor parameters; changing policy mode still changes behavior')
    extra.add_argument('--gradient-clip-scope', choices=('joint', 'separate'), default='joint')
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
    extra.add_argument('--cache-rollout', action='store_true', help='Cache immutable PPO rollout inputs on device for each update')
    extra.add_argument('--rollout-cache-mib', type=int, default=1024, help='Maximum rollout input cache MiB; allocation safely falls back before optimization')
    extra.add_argument('--coalesce-transfers', action='store_true',
                       help='Batch scalar telemetry and rollout host transfers; retain PPO reduction and sampling order')
    extra.add_argument('--fused-encoding', action='store_true',
                       help='Use equivalent device column-feature operations without one-hot intermediates')
    extra.add_argument('--gpu-prepare', action='store_true',
                       help='Prepare rollout candidates/features on CUDA, with exact native fallbacks and bounded retained inputs')
    options, remaining = extra.parse_known_args(supplied)
    if '-h' in supplied or '--help' in supplied:
        extra.print_help()
        print('V2: frozen-policy critic calibration, target recovery, teacher-residual policy, real PPO minibatches. Old runs remain separate.')
    args = base.arguments(['--run-dir', str(base.ROOT / 'runs/survival-ppo-v2'), '--rollout', '128', '--warmup-updates', '0'] + remaining)
    for key, value in vars(options).items():
        setattr(args, key, value)
    for key in ('optimizer_batch', 'critic_lr', 'value_beta', 'calibration_validation_games', 'calibration_stride', 'gradient_every', 'rollout_cache_mib'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            extra.error(f'{key} must be finite and positive')
    if not 1 <= args.ppo_epochs <= 16 or not 0 < args.clip_ratio < 1 or not 0 < args.target_kl < 1:
        extra.error('Invalid PPO epoch, clipping or KL setting')
    if not math.isfinite(args.reference_kl_weight) or args.reference_kl_weight < 0:
        extra.error('Reference KL weight must be finite and nonnegative')
    if bool(args.reference_kl_weight) != (args.reference_checkpoint is not None):
        extra.error('Positive --reference-kl-weight and --reference-checkpoint must be supplied together')
    if not 0 <= args.residual_scale <= 4 or not 0 < args.calibration_ratio < 1:
        extra.error('Invalid residual scale or calibration threshold')
    if not math.isfinite(args.teacher_prior_scale) or not 0 <= args.teacher_prior_scale <= 4:
        extra.error('Teacher prior scale must be finite and in [0, 4]')
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
