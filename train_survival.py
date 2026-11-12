from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

import train_overnight as base

neural, game = base.neural, base.game
ALGORITHM = 'survival-ppo-v1'


def utility_targets(features, valid):
    if features.ndim != 3 or features.shape[-1] != neural.FEATURES or valid.shape != features.shape[:2]:
        raise ValueError('Candidate features do not match the trained schema.')
    valid = valid.bool()
    if not bool(valid.any(-1).all()):
        raise ValueError('Every state needs at least one valid candidate.')
    gain = torch.round(features[..., 8].float() * 16)
    holes = torch.round(features[..., 9].float() * 64)
    utility = gain - game.FUTURE_HOLE_PENALTY * holes
    best = utility.masked_fill(~valid, -torch.inf).amax(-1, keepdim=True)
    winners = valid & (utility == best)
    deficit = torch.where(valid, (best - utility) / 16, 0.)
    return winners, deficit


def categorical_logp(logits, valid):
    valid = valid.bool()
    if not bool(valid.any(-1).all()) or not bool(torch.isfinite(logits[valid]).all()):
        raise ValueError('Invalid candidate mask or non-finite action logits.')
    return F.log_softmax(logits.float().masked_fill(~valid, -torch.inf), dim=-1)


def teacher_losses(logp, features, valid):
    winners, deficit = utility_targets(features, valid)
    set_loss = -torch.logsumexp(logp.masked_fill(~winners, -torch.inf), -1).mean()
    regret = (logp.exp() * deficit).sum(-1).mean()
    return set_loss, regret, winners


def exact_kl(old_logp, new_logp, valid):
    old = torch.where(valid, old_logp, 0.)
    new = torch.where(valid, new_logp, 0.)
    probability = torch.where(valid, old_logp.exp(), 0.)
    return (probability * (old - new)).sum(-1).clamp_min(0.)


def clipped_policy_loss(new_logp, old_logp, actions, advantages, clip_ratio):
    difference = new_logp.gather(1, actions[:, None]).squeeze(1) - old_logp.gather(1, actions[:, None]).squeeze(1)
    ratio = difference.clamp(-20, 20).exp()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages)
    return -surrogate.mean(), ((ratio - 1).abs() > clip_ratio).float().mean()


@dataclass
class Rollout:
    records: list
    choices: list
    labels: list
    advantages: np.ndarray
    returns: np.ndarray
    old_logp: list


class SurvivalTrainer(base.Trainer):
    def training_signature(self):
        return {'algorithm': ALGORITHM,
                'ppo_epochs': self.args.ppo_epochs, 'clip_ratio': self.args.clip_ratio,
                'target_kl': self.args.target_kl, 'regret_weight': self.args.regret_weight,
                'trainer_sha256': base.file_hash(Path(__file__)),
                'runner_sha256': base.file_hash(Path(base.__file__))}

    def checkpoint_metadata(self):
        return {'algorithm': ALGORITHM, 'value_semantics': 'expected_remaining_placements_times_0.001',
                'completion_semantics': 'imitation_auxiliary_not_recalibrated_during_survival',
                'initial_checkpoint_sha256': getattr(self, 'initial_digest', None)}

    def __init__(self, args, budget):
        super().__init__(args, budget)
        if self.last.exists():
            saved = torch.load(self.last, map_location='cpu', weights_only=True)
            self.initial_digest = saved['extra'].get('initial_checkpoint_sha256')
        else:
            self.initial_digest = base.file_hash(Path(self.initial_checkpoint)) if self.initial_checkpoint else None
        self.progress.setdefault('optimizer_steps', 0)
        self.progress.setdefault('kl_stopped_updates', 0)
        self.emit('survival_algorithm', algorithm=ALGORITHM, initial_checkpoint_sha256=self.initial_digest,
                  teacher_loss='probability mass on all maximum primary-utility candidates',
                  reward='one successful placement; scaled by 0.001; gamma=1',
                  value_head='reset on transfer; restored on resume', ppo_epochs=args.ppo_epochs,
                  clip_ratio=args.clip_ratio, target_kl=args.target_kl)

    def rollout(self, warmup):
        old = []
        def capture(module, inputs, output):
            batch = inputs[0]
            probabilities = categorical_logp(output['logits'], batch['valid']).detach().cpu()
            for row, mask in zip(probabilities, batch['valid'].detach().cpu()):
                length = int(mask.sum())
                if not bool(mask[:length].all()) or bool(mask[length:].any()):
                    raise ValueError('Only suffix-padded candidate batches are supported.')
                old.append(row[:length].clone())
        hook = self.model.register_forward_hook(capture)
        try:
            result = super().rollout(warmup)
        finally:
            hook.remove()
        if result is None:
            return None
        if len(result[0]) != len(old):
            raise AssertionError('Behaviour probabilities and rollout states do not align.')
        return Rollout(*result, old)

    def old_batch(self, rollout, indices, size):
        result = torch.full((len(indices), size), -torch.inf, device=self.device)
        for index, source in enumerate(indices):
            row = rollout.old_logp[source]
            result[index, :len(row)] = row.to(self.device)
        return result

    def rollout_kl(self, rollout):
        total = 0.
        with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
            for start in range(0, len(rollout.records), self.args.batch):
                indices = list(range(start, min(start + self.args.batch, len(rollout.records))))
                batch = neural.collate([rollout.records[i] for i in indices], self.device)
                current = categorical_logp(self.model(batch)['logits'], batch['valid'])
                old = self.old_batch(rollout, indices, current.shape[1])
                total += float(exact_kl(old, current, batch['valid']).sum().item())
        return total / len(rollout.records)

    def update(self, rollout, warmup):
        self.model.train()
        size = len(rollout.records)
        epochs = 1 if warmup else self.args.ppo_epochs
        completed, stop_kl = 0, False
        aggregate = {}
        for epoch in range(epochs):
            measured_kl = self.rollout_kl(rollout)
            if not warmup and measured_kl > self.args.target_kl:
                stop_kl = True
                break
            self.optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(size).tolist()
            totals = {}
            for start in range(0, size, self.args.batch):
                indices = order[start:start + self.args.batch]
                batch = neural.collate([rollout.records[i] for i in indices], self.device)
                actions = torch.tensor([rollout.choices[i] for i in indices], device=self.device)
                advantage = torch.tensor(rollout.advantages[indices], device=self.device)
                returns = torch.tensor(rollout.returns[indices], device=self.device)
                old = self.old_batch(rollout, indices, batch['valid'].shape[1])
                with neural.precision_context(self.device, self.args.precision):
                    prediction = self.model(batch)
                    logp = categorical_logp(prediction['logits'], batch['valid'])
                    imitation, regret, winners = teacher_losses(logp, batch['features'], batch['valid'])
                    entropy = -(logp.exp() * torch.where(batch['valid'], logp, 0.)).sum(-1).mean()
                    critic = F.smooth_l1_loss(prediction['value'].float(), returns)
                    actor, clipped = clipped_policy_loss(logp, old, actions, advantage, self.args.clip_ratio)
                    loss = (imitation if warmup else actor + self.args.imitation * imitation) + .5 * critic
                    loss += self.args.regret_weight * regret
                    if not warmup:
                        loss -= self.args.entropy * entropy
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Non-finite PPO loss; previous checkpoint retained.')
                fraction = len(indices) / size
                self.scaler.scale(loss * fraction).backward()
                picked = logp.argmax(-1)
                rows = torch.arange(len(indices), device=self.device)
                metrics = {'loss': loss, 'policy_loss': actor, 'value_loss': critic, 'tie_imitation_loss': imitation,
                           'expected_utility_regret': regret, 'entropy': entropy, 'clip_fraction': clipped,
                           'utility_match': winners[rows, picked].float().mean()}
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach().item()) * fraction
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            completed += 1
            self.progress['updates'] += 1
            self.progress['optimizer_steps'] += 1
            totals['gradient_norm'] = float(norm)
            for key, value in totals.items():
                aggregate[key] = aggregate.get(key, 0.) + value
        if not completed:
            raise ValueError('Behaviour/current policy mismatch before the first PPO optimizer step.')
        final_kl = self.rollout_kl(rollout)
        self.progress['runner_updates'] += 1
        self.progress['warmup_updates' if warmup else 'survival_updates'] += 1
        self.progress['kl_stopped_updates'] += int(stop_kl)
        self.emit('update', **self.progress, stage='warmup' if warmup else 'survival-ppo',
                  **{key: value / completed for key, value in aggregate.items()},
                  completed_epochs=completed, final_exact_kl=final_kl, kl_early_stop=stop_kl,
                  rollout_placements=size, **neural.memory_report(self.device))

    def save(self, reason='checkpoint'):
        report = super().save(reason)
        spectra = {}
        with torch.no_grad():
            for name in ('rank1', 'rank2'):
                values = torch.linalg.svdvals(getattr(self.model, name).weight.detach().float().cpu()).square()
                energy = values / values.sum().clamp_min(1e-30)
                spectra[name] = {'top_direction_energy': float(energy[0]),
                                 'participation_rank': float(1 / energy.square().sum())}
        diagnosis = {'algorithm': ALGORITHM, 'runner_updates': self.progress['runner_updates'],
                     'checkpoint_sha256': base.file_hash(self.last), 'ranking_spectra': spectra,
                     'value_semantics': 'expected remaining placements / 1000',
                     'completion_semantics': 'imitation auxiliary; not calibrated for the survival policy',
                     'note': 'Spectra are diagnostics, not proof of stronger gameplay.'}
        base.durable_json(self.root / 'diagnostics.json', diagnosis)
        self.emit('policy_diagnostics', **diagnosis)
        return report


def arguments(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--ppo-epochs', type=int, default=3)
    extra.add_argument('--clip-ratio', type=float, default=.2)
    extra.add_argument('--target-kl', type=float, default=.02)
    extra.add_argument('--regret-weight', type=float, default=.1)
    options, remaining = extra.parse_known_args(supplied)
    defaults = ['--run-dir', str(base.ROOT / 'runs/survival-ppo'), '--rollout', '128', '--warmup-updates', '64']
    if '-h' in supplied or '--help' in supplied:
        print('Survival PPO options: --ppo-epochs 3 --clip-ratio 0.2 --target-kl 0.02 --regret-weight 0.1')
        print('Default: runs/survival-ppo; 128-step rollouts; 64 teacher warm-up updates; latest RL checkpoint when present.')
    args = base.arguments(defaults + remaining)
    for key, value in vars(options).items():
        setattr(args, key, value)
    if not 1 <= args.ppo_epochs <= 16:
        extra.error('ppo-epochs must be in [1, 16]')
    if not math.isfinite(args.clip_ratio) or not 0 < args.clip_ratio < 1:
        extra.error('clip-ratio must be in (0, 1)')
    if not math.isfinite(args.target_kl) or args.target_kl <= 0:
        extra.error('target-kl must be finite and positive')
    if not math.isfinite(args.regret_weight) or args.regret_weight < 0:
        extra.error('regret-weight must be finite and nonnegative')
    if args.checkpoint is None:
        for name in ('neural-rl.pt', 'neural-policy.pt'):
            path = base.ROOT / 'models' / name
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
        with base.run_lock(args.run_dir), base.Budget(args.hours) as budget:
            SurvivalTrainer(args, budget).run()
        return 0
    except KeyboardInterrupt:
        print('Forced interruption. Resume from the last completed checkpoint.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Survival training stopped: {error}\nExisting model files and best checkpoint remain unchanged.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
