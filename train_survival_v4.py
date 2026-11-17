"""PPO continuation for a spatial V4 policy with a fresh optimizer and progress."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

import train_survival_v3 as v3
from tetris_survival_v4 import ARCHITECTURE

v2, base, neural, game = v3.v2, v3.base, v3.neural, v3.game
ALGORITHM = 'survival-ppo-v4'
SELECTION_SEEDS = tuple(910001 + 2 * i for i in range(8))


def phase_behaviour(phase):
    return 'frozen_sampled_neural' if phase == 'selection-sampled' else f'greedy_{phase.removeprefix("selection-")}'


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

    def lockstep_available(self):
        return (getattr(self.args, 'lockstep_selection', False) and getattr(self.args, 'gpu_prepare', False)
                and self.device.type == 'cuda' and game.BACKEND == 'native')

    def _selection_preparer(self):
        if getattr(self, '_gpu_preparer', None) is None:
            from tetris_gpu_simulation import GpuPreparer
            self._gpu_preparer = GpuPreparer(self.model.config, self.device)
        return self._gpu_preparer

    def lockstep_games(self, phase, choose, on_step=None):
        """Play every selection seed at once: one batched CUDA preparation and one
        forward pass per step (issue #36). `choose(seed_indices, logits, logp)`
        returns one action per live row. Returns (episodes, reasons, seconds)."""
        seeds = list(self.args.selection_seeds)
        episodes = [base.Episode(self.progress['runner_updates'] * len(seeds) + i,
                                 game.Game(seed, self.args.piece_set), phase=phase)
                    for i, seed in enumerate(seeds)]
        reasons = [None] * len(seeds)
        started = time.perf_counter()
        active = list(range(len(seeds)))
        preparer = self._selection_preparer()
        while active:
            if self.budget.expired():
                for i in active:
                    reasons[i] = self.budget.reason
                break
            result = preparer.prepare_batch([episodes[i].state for i in active], [episodes[i].target for i in active])
            live = []
            for local, i in enumerate(active):
                if result.prepared[local] is None:
                    reasons[i] = base.stopped_reason(episodes[i].state)
                else:
                    self.note_target(episodes[i], int(result.targets[local]))
                    live.append(i)
            if live:
                rows = [active[local] for local in result.environment_indices]
                prediction = self.model(result.batch)
                logp = self.policy_logp(prediction['logits'], result.batch['valid'])
                if on_step is not None:
                    on_step(rows, episodes, prediction)
                actions = choose(rows, prediction['logits'], logp)
                for i, candidate in zip(rows, result.selected_candidates(actions)):
                    base.apply_move(episodes[i], candidate, self.progress['updates'], phase_behaviour(phase))
            active = []
            for i in live:
                if episodes[i].state.pieces >= self.args.selection_pieces:
                    reasons[i] = 'piece_limit'
                else:
                    active.append(i)
        return episodes, reasons, time.perf_counter() - started

    def evaluate_policy(self, policy):
        if policy == 'sustained' or not self.lockstep_available():
            return super().evaluate_policy(policy)
        self.model.eval()
        recoveries = self.progress['target_recoveries']
        with base.preserve_rng(), torch.inference_mode():
            episodes, reasons, seconds = self.lockstep_games(
                f'selection-{policy}', lambda rows, logits, logp: logits.argmax(-1))
        runs = []
        for episode, reason in zip(episodes, reasons):
            episode.stop_reason, episode.complete = reason, reason in ('top_out', 'planner_blocked', 'piece_limit')
            self.recorder.write(episode)
            report = {**episode.state.summary(), 'stop_reason': reason, 'evaluation_complete': episode.complete,
                      'capped': reason == 'piece_limit', 'policy': policy,
                      'policy_mode': self.model.config.policy_mode,
                      'target_recoveries': self.progress['target_recoveries'] - recoveries,
                      'wall_seconds': seconds, 'lockstep_games': len(episodes)}
            runs.append(report)
            self.emit('selection_game', **report)
        complete = all(run['evaluation_complete'] for run in runs)
        return {'complete': complete, 'runs': runs, 'score': base.survival_score(runs) if complete else None,
                'signature': self.signature, 'score_definition': 'mean capped survival, then minimum, then median',
                'lockstep': True}

    def sampled_policy_check(self):
        if not self.lockstep_available():
            return super().sampled_policy_check()
        self.model.eval()
        seeds = list(self.args.selection_seeds)
        generators = [torch.Generator(device=self.device).manual_seed(2701 + seed) for seed in seeds]
        samples = [[] for _ in seeds]

        def record(rows, episodes, prediction):
            for row, i in enumerate(rows):
                pieces = episodes[i].state.pieces
                if pieces % 128 == 0 and len(samples[i]) < 256:
                    samples[i].append((pieces, float(prediction['value'][row])))

        def sample(rows, logits, logp):
            return torch.stack([torch.multinomial(logp[row].exp(), 1, generator=generators[i])[0]
                                for row, i in enumerate(rows)])

        with base.preserve_rng(), torch.inference_mode():
            episodes, reasons, seconds = self.lockstep_games('selection-sampled', sample, record)
        runs = []
        for i, (episode, reason) in enumerate(zip(episodes, reasons)):
            terminated = reason in ('top_out', 'planner_blocked')
            base.library.finish(episode.state, reason)
            diagnostics = None
            if terminated and samples[i]:
                targets = np.asarray([(episode.state.pieces - piece) * base.REWARD_SCALE for piece, _ in samples[i]])
                predicted = np.asarray([value for _, value in samples[i]])
                diagnostics = v2.regression_metrics(predicted, targets)
                diagnostics['mean_signed_error_pieces'] = float((predicted - targets).mean() * 1000)
            report = {**episode.state.summary(), 'policy': 'sampled_neural', 'mode': 'sampled',
                      'action_seed': 2701 + seeds[i], 'stop_reason': reason,
                      'evaluation_complete': terminated or reason == 'piece_limit',
                      'capped': reason == 'piece_limit', 'critic_monte_carlo': diagnostics,
                      'critic_uncensored': terminated, 'critic_samples': len(samples[i]),
                      'wall_seconds': seconds, 'lockstep_games': len(episodes),
                      'limitation': 'One realized continuation per state; within-game samples are correlated.'}
            runs.append(report)
            self.emit('sampled_selection_game', **report)
        complete = all(run['evaluation_complete'] for run in runs)
        return {'complete': complete, 'runs': runs, 'score': base.survival_score(runs) if complete else None,
                'note': 'Frozen sampled-policy diagnostics; protected best is selected by separate greedy games.',
                'lockstep': True}

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

    def rollout_kl(self, rollout):
        kl = super().rollout_kl(rollout)
        self._last_rollout_kl = kl
        return kl

    def adapt_learning_rate(self, kl):
        """KL-targeted actor learning rate; the critic rate is unchanged.

        PPO stays within --target-kl through early stopping; this only moves the
        actor step size towards --adaptive-kl-target when updates are far from it.
        The adjusted rate is stored in the optimizer state, so it survives resume.
        """
        target = self.args.adaptive_kl_target
        group = next(g for g in self.optimizer.param_groups if g.get('name') == 'actor')
        before = group['lr']
        if kl < target / 2:
            group['lr'] = min(before * 1.5, self.args.lr * self.args.adaptive_lr_range)
        elif kl > target * 2:
            group['lr'] = max(before / 1.5, self.args.lr / self.args.adaptive_lr_range)
        if group['lr'] != before:
            self.emit('learning_rate', actor_lr=group['lr'], previous_actor_lr=before, final_exact_kl=kl,
                      adaptive_kl_target=target, runner_updates=self.progress['runner_updates'])
        return group['lr']

    def update(self, rollout, warmup):
        self._last_rollout_kl = None
        result = super().update(rollout, warmup)
        if self.args.adaptive_lr and not warmup and self._last_rollout_kl is not None:
            self.adapt_learning_rate(float(self._last_rollout_kl))
        return result

    def training_signature(self):
        signature = super().training_signature()
        signature.update(algorithm=ALGORITHM,
                         v4_trainer_sha256=base.file_hash(Path(__file__)),
                         spatial_architecture_sha256=base.file_hash(Path(__file__).with_name('tetris_survival_v4.py')),
                         pretrained_sha256=(base.file_hash(self.args.pretrained)
                                            if self.args.pretrained is not None else None),
                         shortlist_mode=self.args.shortlist_mode,
                         screening=self.args.screening,
                         adaptive_lr=self.args.adaptive_lr,
                         lockstep_selection=self.lockstep_available(),
                         adaptive_kl_target=self.args.adaptive_kl_target if self.args.adaptive_lr else None,
                         adaptive_lr_range=self.args.adaptive_lr_range if self.args.adaptive_lr else None,
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
    extra.add_argument('--screening', choices=('sampled', 'exhaustive'), default=None,
                       help="Candidate pool: 'sampled' (config.candidates placements) or every legal placement. "
                            "Default: exhaustive with the GPU defaults, sampled otherwise (CPU exhaustive is ~1.3 s/state)")
    extra.add_argument('--sequential-selection', action='store_true',
                       help='Play selection games one at a time instead of in lockstep on CUDA (issue #36)')
    extra.add_argument('--fixed-lr', action='store_true',
                       help='Disable the KL-targeted actor learning rate (issue #33)')
    extra.add_argument('--adaptive-kl-target', type=float, default=None,
                       help='Full-rollout KL the actor step size aims for (default: half of --target-kl)')
    extra.add_argument('--adaptive-lr-range', type=float, default=16.,
                       help='Actor learning rate stays within [lr / range, lr * range]')
    extra.add_argument('--no-gpu-defaults', action='store_true',
                       help='Do not enable CUDA preparation, fused encoding, coalesced transfers and rollout caching')
    options, remaining = extra.parse_known_args(supplied)
    gpu = gpu_defaults(remaining, options.no_gpu_defaults)
    args = v3.arguments(['--run-dir', str(base.ROOT/'runs/survival-ppo-v4'),
                         '--gradient-clip-scope', 'separate',
                         '--selection-seeds', *map(str, SELECTION_SEEDS), *gpu, *remaining])
    args.pretrained = options.pretrained
    args.shortlist_mode = options.shortlist_mode
    args.screening = options.screening or ('exhaustive' if gpu else 'sampled')
    args.adaptive_lr = not options.fixed_lr
    args.lockstep_selection = not options.sequential_selection
    args.adaptive_kl_target = options.adaptive_kl_target if options.adaptive_kl_target is not None else args.target_kl / 2
    args.adaptive_lr_range = options.adaptive_lr_range
    if args.adaptive_kl_target <= 0 or args.adaptive_kl_target > args.target_kl or args.adaptive_lr_range < 1:
        raise ValueError('--adaptive-kl-target must be in (0, --target-kl] and --adaptive-lr-range >= 1')
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
