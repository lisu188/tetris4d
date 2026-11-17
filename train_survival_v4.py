"""PPO continuation for a spatial V4 policy with a fresh optimizer and progress."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys

import torch

import train_survival_v3 as v3
from tetris_survival_v4 import ARCHITECTURE

v2, base, neural, game = v3.v2, v3.base, v3.neural, v3.game
ALGORITHM = 'survival-ppo-v4'


class SurvivalTrainer(v3.SurvivalTrainer):
    algorithm = ALGORITHM

    def emit(self, event, **fields):
        if event == 'survival_algorithm':
            fields.update(architecture=ARCHITECTURE, spatial_critic=True,
                          afterstate_spatial_actor=True, actor_critic_parameter_isolation=True)
        return v2.SurvivalTrainer.emit(self, event, **fields)

    def create_fresh_model(self):
        config = neural.Config(architecture=ARCHITECTURE, spatial_actor=True,
                               shortlist_mode=self.args.shortlist_mode,
                               screening=self.args.screening,
                               recover_target=self.args.recover_target,
                               policy_mode=self.args.policy_mode,
                               residual_scale=self.args.residual_scale,
                               teacher_prior_scale=self.args.teacher_prior_scale,
                               detach_critic=False, centered_critic=False)
        self.imported_checkpoint_sha256 = None
        if self.args.pretrained is None:
            return neural.create_model(config).to(self.device)
        source = Path(self.args.pretrained)
        loaded, checkpoint = neural.load_checkpoint(source, self.device)
        # Screening only changes which placements are offered, never the weights,
        # so a sampled-mode pretrained actor may continue under exhaustive screening.
        if asdict(replace(loaded.config, screening=config.screening)) != asdict(config):
            raise ValueError('Pretrained V4 config does not match the requested actor and shortlist')
        loaded.config = config
        if checkpoint['teacher_parameters'] != game.network_parameters():
            raise ValueError('Pretrained V4 teacher weights differ from the runtime')
        if not all(bool(torch.isfinite(parameter).all()) for parameter in loaded.state_dict().values()):
            raise ValueError('Pretrained V4 contains non-finite weights')
        self.imported_checkpoint_sha256 = base.file_hash(source)
        return loaded

    def prepare_episode(self, episode):
        """Calibration and selection games prepare single states on CUDA when enabled.

        Rollout ('training') episodes keep the inherited path; their batches are
        already prepared on device. Results match the CPU reference preparation.
        """
        if (episode.phase == 'training' or not getattr(self.args, 'gpu_prepare', False)
                or self.device.type != 'cuda' or game.BACKEND != 'native'):
            return super().prepare_episode(episode)
        try:
            if getattr(self, '_gpu_preparer', None) is None:
                from tetris_gpu_simulation import GpuPreparer
                self._gpu_preparer = GpuPreparer(self.model.config, self.device)
            result = self._gpu_preparer.prepare_batch([episode.state], [episode.target])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return super().prepare_episode(episode)
        prepared = result.prepared[0]
        if prepared is not None:
            self.note_target(episode, int(result.targets[0]))
        return prepared

    def configure_model(self, resuming):
        if self.model.config.architecture != ARCHITECTURE:
            raise ValueError('V4 cannot resume a different model architecture')
        self.model.fused_encoding = getattr(self.args, 'fused_encoding', False)
        self.calibration_actor = v2.actor_digest(self.model)
        self.initial_actor_sha256 = self.calibration_actor
        if not resuming:
            if self.initial_checkpoint is not None or self.progress['updates'] != 0:
                raise ValueError('V4 accidentally inherited PPO progress')
            imported = getattr(self, 'imported_checkpoint_sha256', None)
            proof = {'schema': 'tetris4d-v4-initialization-v1',
                     'fresh_optimizer': True, 'fresh_ppo_progress': True,
                     'inherited_checkpoint': None, 'imported_model_sha256': imported,
                     'seed': self.args.seed, 'architecture': ARCHITECTURE,
                     'config': asdict(self.model.config),
                     'model_sha256': v3.model_digest(self.model),
                     'actor_sha256': self.initial_actor_sha256,
                     'parameters': sum(parameter.numel() for parameter in self.model.parameters()),
                     'teacher_weights_sha256': base.file_hash(self.args.weights)}
            base.durable_json(self.root / 'initialization.json', proof)

    def training_signature(self):
        signature = super().training_signature()
        signature.update(algorithm=ALGORITHM,
                         v4_trainer_sha256=base.file_hash(Path(__file__)),
                         spatial_architecture_sha256=base.file_hash(Path(__file__).with_name('tetris_survival_v4.py')),
                         pretrained_sha256=(base.file_hash(self.args.pretrained)
                                            if self.args.pretrained is not None else None),
                         shortlist_mode=self.args.shortlist_mode,
                         screening=self.args.screening,
                         spatial_actor=True, fresh_ppo_progress=True)
        return signature

    def checkpoint_metadata(self):
        values = super().checkpoint_metadata()
        values.update(algorithm=ALGORITHM, architecture=ARCHITECTURE,
                      fresh_random_trainable_weights=self.args.pretrained is None,
                      imported_pretrained_sha256=self.signature.get('pretrained_sha256'),
                      value_semantics='expected_current_policy_remaining_placements_times_0.001',
                      spatial_actor=True, spatial_critic=True)
        return values


GPU_FLAGS = ('--gpu-prepare', '--fused-encoding', '--coalesce-transfers', '--cache-rollout')


def gpu_defaults(supplied, disabled=False):
    """GPU-first defaults; explicit later arguments still override valued options."""
    def value(name):
        for index, argument in enumerate(supplied):
            if argument == name and index + 1 < len(supplied):
                return supplied[index + 1]
            if argument.startswith(name + '='):
                return argument.split('=', 1)[1]
        return None
    if disabled or value('--device') == 'cpu' or not torch.cuda.is_available():
        return []
    if value('--backend') not in (None, 'native'):
        return []
    return ['--device', 'cuda', '--backend', 'native', *GPU_FLAGS]


def arguments(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    extra.add_argument('--pretrained', type=Path)
    extra.add_argument('--shortlist-mode', choices=('teacher-random', 'diverse'), default='teacher-random')
    extra.add_argument('--screening', choices=('sampled', 'exhaustive'), default='sampled',
                       help="Candidate pool: 'sampled' (config.candidates placements) or every legal placement")
    extra.add_argument('--no-gpu-defaults', action='store_true',
                       help='Do not enable CUDA preparation, fused encoding, coalesced transfers and rollout caching')
    options, remaining = extra.parse_known_args(supplied)
    args = v3.arguments(['--run-dir', str(base.ROOT/'runs/survival-ppo-v4'),
                         '--gradient-clip-scope', 'separate',
                         *gpu_defaults(remaining, options.no_gpu_defaults), *remaining])
    args.pretrained = options.pretrained
    args.shortlist_mode = options.shortlist_mode
    args.screening = options.screening
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    try:
        device = neural.device_for(args.device)
        with v2.device_lock(device, args.allow_concurrent) as competing, base.run_lock(args.run_dir), base.Budget(args.hours) as budget:
            trainer = SurvivalTrainer(args, budget)
            trainer.emit('resource_isolation', competing_trainers=competing, override=args.allow_concurrent)
            trainer.run()
        return 2 if budget.reason == 'calibration_failed' else 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as error:
        print(f'V4 training stopped: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
