"""PPO for Survival V5: a frozen V4 actor plus a bounded, learned next-state scorer.

The scorer and the critic start as copies of a calibrated regional V4 critic
(--source-checkpoint), so calibration is skipped by default and the initial
policy is the bounded afterstate ranking. See tetris_survival_v5.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

import tetris_survival_v5 as v5
import train_survival_v4 as v4t

v2, v3, base, neural, game = v4t.v2, v4t.v3, v4t.base, v4t.neural, v4t.game
ALGORITHM = 'survival-ppo-v5'


class SurvivalTrainer(v4t.SurvivalTrainer):
    algorithm = ALGORITHM
    architecture = v5.ARCHITECTURE

    def emit(self, event, **fields):
        if event == 'survival_algorithm':
            fields.update(architecture=v5.ARCHITECTURE, frozen_base_actor=True, next_state_features=v5.FEATURES,
                          residual_bound=self.args.residual_bound, policy_temperature=self.args.policy_temperature,
                          afterstate_scale=self.args.afterstate_scale, actor_critic_parameter_isolation=True)
            return v2.SurvivalTrainer.emit(self, event, **fields)
        return super().emit(event, **fields)

    def create_fresh_model(self):
        if self.args.source_checkpoint is None:
            raise ValueError('V5 needs --source-checkpoint: a V4 checkpoint with a calibrated regional critic')
        path = Path(self.args.source_checkpoint)
        source, checkpoint = neural.load_checkpoint(path, self.device)
        if checkpoint['teacher_parameters'] != game.network_parameters():
            raise ValueError('V5 source teacher weights differ from the runtime')
        model = v5.from_v4(source, residual_bound=self.args.residual_bound,
                           policy_temperature=self.args.policy_temperature,
                           afterstate_scale=self.args.afterstate_scale,
                           shortlist_mode=self.args.shortlist_mode, screening=self.args.screening)
        # The generic trainer zeroes value[-1] on a fresh start; configure_model restores the copy.
        self._source_critic = {k: v.detach().clone() for k, v in source.value.state_dict().items()}
        self.imported_checkpoint_sha256 = base.file_hash(path)
        return model

    def configure_model(self, resuming):
        if not resuming and getattr(self, '_source_critic', None) is not None:
            self.model.value.main.load_state_dict(self._source_critic)
        super().configure_model(resuming)

    def create_optimizer(self):
        # The base actor is frozen: only the scorer (actor) and the critic are optimized.
        return torch.optim.AdamW([{'params': list(self.model.scorer.parameters()), 'lr': self.args.lr, 'name': 'actor'},
                                  {'params': list(self.model.value.parameters()), 'lr': self.args.critic_lr,
                                   'name': 'critic'}], weight_decay=1e-4)

    def augment_batch(self, result, states):
        return v5.augment_device_batch(result, states)

    def _accept_gpu_batch(self, result, calls=1):
        return super()._accept_gpu_batch(v5.augment_device_batch(result, [e.state for e in self.environments]), calls)

    def prepare_episode(self, episode):
        prepared = super().prepare_episode(episode)
        if prepared is not None and 'next_features' not in prepared.arrays:
            v5.augment_prepared(prepared, episode.state, self.device)
        return prepared

    def training_signature(self):
        signature = super().training_signature()
        source = self.args.source_checkpoint
        signature.update(algorithm=ALGORITHM, v5_trainer_sha256=base.file_hash(Path(__file__)),
                         v5_architecture_sha256=base.file_hash(Path(v5.__file__)),
                         source_checkpoint_sha256=base.file_hash(source) if source is not None else None,
                         residual_bound=self.args.residual_bound, policy_temperature=self.args.policy_temperature,
                         afterstate_scale=self.args.afterstate_scale)
        return signature

    def checkpoint_metadata(self):
        values = super().checkpoint_metadata()
        values.update(frozen_base_actor=True, fresh_random_trainable_weights=False,
                      imported_source_sha256=self.signature.get('source_checkpoint_sha256'))
        return values


def arguments(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--source-checkpoint', type=Path,
                       help='V4 checkpoint whose actor becomes the frozen base and whose regional critic seeds V5')
    extra.add_argument('--residual-bound', type=float, default=2.,
                       help='R: the learned residual moves each logit by at most R nats from the base')
    extra.add_argument('--policy-temperature', type=float, default=.5, help='tau: logits are divided by tau')
    extra.add_argument('--afterstate-scale', type=float, default=1.,
                       help='beta: residual = R * tanh(beta * (score - mean score)), scores in thousands of placements')
    options, remaining = extra.parse_known_args(supplied)
    args = v4t.arguments(['--run-dir', str(base.ROOT / 'runs/survival-ppo-v5'), '--fixed-lr',
                          '--calibration-games', '0', '--allow-uncalibrated', *remaining])
    args.source_checkpoint = options.source_checkpoint
    args.residual_bound, args.policy_temperature = options.residual_bound, options.policy_temperature
    args.afterstate_scale = options.afterstate_scale
    neural.Config(architecture=v5.ARCHITECTURE, spatial_actor=True, residual_bound=args.residual_bound,
                  policy_temperature=args.policy_temperature, afterstate_scale=args.afterstate_scale)  # validates
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    try:
        device = neural.device_for(args.device)
        with v2.device_lock(device, args.allow_concurrent) as competing, base.run_lock(args.run_dir), \
                base.Budget(args.hours) as budget:
            trainer = SurvivalTrainer(args, budget)
            trainer.emit('resource_isolation', competing_trainers=competing, override=args.allow_concurrent)
            trainer.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f'V5 training stopped: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
