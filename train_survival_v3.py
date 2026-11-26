"""Fresh survival PPO with fixed feature inputs and an independent positive critic."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

import train_survival_v2 as v2
from tetris_survival_v3 import ARCHITECTURE

base, neural, game = v2.base, v2.neural, v2.game
ALGORITHM = 'survival-ppo-v3'


CRITIC_SAMPLE_STRIDE = 128


def realized_critic_errors(samples, final_pieces):
    """Per-game residuals against realized survival, including late-game predictions."""
    if not samples:
        return None
    distance = np.asarray([final_pieces - piece for piece, _ in samples])
    predicted = np.asarray([value for _, value in samples])
    targets = distance * base.REWARD_SCALE
    result = v2.regression_metrics(predicted, targets)
    errors = (predicted - targets) / base.REWARD_SCALE
    result['mean_signed_error_pieces'] = float(errors.mean())
    result['by_remaining_placements'] = {}
    for label, mask in (('1-256', distance <= 256), ('257-2048', (distance > 256) & (distance <= 2048)),
                        ('2049+', distance > 2048)):
        result['by_remaining_placements'][label] = {
            'samples': int(mask.sum()), 'mae_pieces': float(np.abs(errors[mask]).mean()) if mask.any() else None}
    return result


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SurvivalTrainer(v2.SurvivalTrainer):
    algorithm = ALGORITHM

    def emit(self, event, **fields):
        if event == 'survival_algorithm':
            fields.update(critic_encoder_detached=False, actor_critic_parameter_isolation=True,
                          fixed_board_feature_inputs=True, architecture=ARCHITECTURE)
        return super().emit(event, **fields)

    def initial_policy_checkpoint(self):
        if self.args.checkpoint is not None:
            raise ValueError('V3 starts from random policy weights; --checkpoint is not accepted')
        return None

    def create_fresh_model(self):
        config = neural.Config(architecture=ARCHITECTURE, recover_target=self.args.recover_target,
                               policy_mode=self.args.policy_mode, residual_scale=self.args.residual_scale,
                               teacher_prior_scale=self.args.teacher_prior_scale,
                               detach_critic=False, centered_critic=False)
        return neural.create_model(config).to(self.device)

    def configure_model(self, resuming):
        if self.model.config.architecture != ARCHITECTURE:
            raise ValueError('V3 cannot resume a legacy architecture')
        self.model.fused_encoding = getattr(self.args, 'fused_encoding', False)
        self.calibration_actor = v2.actor_digest(self.model)
        self.initial_actor_sha256 = self.calibration_actor
        if not resuming:
            if self.initial_checkpoint is not None or self.progress['updates'] != 0:
                raise ValueError('Fresh V3 unexpectedly inherited training progress')
            proof = {'schema': 'tetris4d-fresh-initialization-v1',
                     'fresh_random_trainable_weights': True, 'inherited_checkpoint': None,
                     'inherited_optimizer': False, 'seed': self.args.seed,
                     'architecture': ARCHITECTURE, 'config': asdict(self.model.config),
                     'model_sha256': model_digest(self.model),
                     'actor_sha256': self.initial_actor_sha256,
                     'parameters': sum(parameter.numel() for parameter in self.model.parameters()),
                     'teacher_weights_sha256': base.file_hash(self.args.weights),
                     'prior': 'Fixed candidate heuristics and optional teacher logit prior; no policy weights inherited.'}
            base.durable_json(self.root/'initialization.json', proof)

    def training_signature(self):
        return {**super().training_signature(), 'algorithm': ALGORITHM,
                'v3_trainer_sha256': base.file_hash(Path(__file__)),
                'architecture_sha256': base.file_hash(Path(__file__).with_name('tetris_survival_v3.py')),
                'fresh_initialization': True, 'critic_output': 'softplus_remaining_placements_times_0.001',
                'sampled_selection_action_seed': 2701}

    def checkpoint_metadata(self):
        return {**super().checkpoint_metadata(), 'algorithm': ALGORITHM,
                'architecture': ARCHITECTURE, 'critic_encoder_detached': False,
                'actor_critic_parameter_isolation': True, 'fixed_board_feature_inputs': True,
                'completion_semantics': 'no_unsupervised_auxiliary_head',
                'fresh_random_trainable_weights': True}

    def initialize_calibration_output(self, x, y):
        # Start from the training-only constant mean, not a reused critic.
        mean = y.double().mean().clamp_min(1e-6)
        inverse_softplus = mean + torch.log(-torch.expm1(-mean))
        with torch.no_grad():
            self.model.value[-1].weight.zero_()
            self.model.value[-1].bias.fill_(float(inverse_softplus))

    def selection_ready(self):
        return self.calibration['done']

    def select(self):
        previous = self.last_evaluated_update
        super().select()
        if (self.budget.expired() or self.last_evaluated_update is None
                or self.last_evaluated_update == previous):
            return
        result = self.sampled_policy_check()
        result.update(runner_updates=self.progress['runner_updates'],
                      checkpoint_sha256=base.file_hash(self.last), selection_only=True)
        base.durable_json(self.root/f'sampled-selection-{self.progress["runner_updates"]:08d}.json', result)
        self.emit('sampled_selection_complete', **result)
        self.evaluated_at = time.monotonic()

    def sampled_policy_check(self):
        runs = []
        self.model.eval()
        with base.preserve_rng(), torch.inference_mode():
            for seed in self.args.selection_seeds:
                if self.budget.expired():
                    break
                episode = base.Episode(self.progress['runner_updates'], game.Game(seed, self.args.piece_set),
                                       phase='selection-sampled')
                generator = torch.Generator(device=self.device).manual_seed(2701 + seed)
                samples = []
                started = time.perf_counter()
                reason = 'piece_limit'
                while episode.state.pieces < self.args.selection_pieces:
                    if self.budget.expired():
                        reason = self.budget.reason
                        break
                    prepared = self.prepare_episode(episode)
                    if prepared is None:
                        reason = base.stopped_reason(episode.state)
                        break
                    batch = neural.collate([prepared.arrays], self.device)
                    prediction = self.model(batch)
                    logp = self.policy_logp(prediction['logits'], batch['valid'])
                    if episode.state.pieces % CRITIC_SAMPLE_STRIDE == 0:
                        samples.append((episode.state.pieces, float(prediction['value'][0])))
                    action = int(torch.multinomial(logp.exp(), 1, generator=generator).item())
                    base.apply_move(episode, prepared.options[action], self.progress['updates'], 'frozen_sampled_neural')
                terminated = reason in ('top_out', 'planner_blocked')
                base.library.finish(episode.state, reason)
                diagnostics = realized_critic_errors(samples, episode.state.pieces) if terminated else None
                report = {**episode.state.summary(), 'policy': 'sampled_neural', 'mode': 'sampled',
                          'action_seed': 2701 + seed, 'stop_reason': reason,
                          'evaluation_complete': terminated or reason == 'piece_limit',
                          'capped': reason == 'piece_limit', 'critic_monte_carlo': diagnostics,
                          'critic_uncensored': terminated, 'critic_samples': len(samples),
                          'wall_seconds': time.perf_counter() - started,
                          'limitation': 'One realized continuation per state; within-game samples are correlated.'}
                runs.append(report)
                self.emit('sampled_selection_game', **report)
                if not report['evaluation_complete']:
                    break
        complete = len(runs) == len(self.args.selection_seeds) and all(run['evaluation_complete'] for run in runs)
        return {'complete': complete, 'runs': runs,
                'score': base.survival_score(runs) if complete else None,
                'note': 'Frozen sampled-policy diagnostics; protected best is selected by separate greedy games.'}

    def run(self):
        self.save('started')
        if not self.calibrate():
            return self.save(self.budget.reason or 'calibration_incomplete')
        if self.best_record is None and self.selection_ready() and not self.budget.expired():
            self.save('initial_policy_calibrated')
            self.select()
        return base.Trainer.run(self)


def arguments(argv=None, *, allow_reference=False):
    supplied = list(sys.argv[1:] if argv is None else argv)
    if any(argument.split('=', 1)[0] == '--checkpoint' for argument in supplied):
        raise ValueError('V3 starts fresh; external --checkpoint is not supported')
    args = v2.arguments(['--run-dir', str(base.ROOT/'runs/survival-ppo-v3'),
                         '--lr', '0.00005', '--ppo-epochs', '2', '--clip-ratio', '.1',
                         '--target-kl', '.01', '--imitation', '0', '--secondary-weight', '0',
                         '--regret-weight', '0', '--policy-mode', 'teacher-prior',
                         '--teacher-prior-scale', '.125', *supplied])
    if args.reference_kl_weight and not allow_reference:
        raise ValueError('V3 starts fresh; frozen reference regularization is not supported')
    # V2's historical auto-discovery is deliberately bypassed, even if an old
    # checkpoint appears in models/ after argument parsing.
    args.checkpoint = None
    return args



if __name__ == '__main__':
    raise SystemExit('This trainer is retired. Use train_survival_v4.py for new runs, or the original revision to resume an older run.')
