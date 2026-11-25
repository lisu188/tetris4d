"""PPO continuation for a spatial V4 policy with a fresh optimizer and progress."""
from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import torch

import train_survival_v3 as v3
from tetris_survival_v4 import ARCHITECTURE

v2, base, neural, game = v3.v2, v3.base, v3.neural, v3.game
ALGORITHM = 'survival-ppo-v4'
SELECTION_SEEDS = tuple(910001 + 2 * i for i in range(8))


def library_root(root=None):
    """Where to *read* recorded games. Disabling recording (TETRIS4D_GAME_LIBRARY=off) must not
    hide the shared library from danger starts, so fall back to the main checkout's library."""
    if root is not None:
        return Path(root)
    library = base.library
    resolved = library.default_root()
    if resolved is None:
        resolved = library._main_checkout(Path(library.__file__).resolve().parent) / 'data' / 'game-library'
    return resolved


def library_start_state(game_id, offset, root=None):
    """Replay a recorded game to `offset` placements; return an unrecorded copy and its target."""
    library = base.library
    root = library_root(root)
    entry = next((e for e in library.iter_index(root) if e['id'] == game_id), None)
    if entry is None:
        raise ValueError(f'Game {game_id} is not in the game library')
    record = library.load(root / entry['file'])
    for index, (state, _move, row) in enumerate(library.replay(record, verify=False)):
        if index == offset:
            snapshot = copy.deepcopy(state)
            snapshot._library_log = None
            library.ignore(snapshot)
            return snapshot, int(row[6])
    raise ValueError(f'Game {game_id} has no placement {offset}')


def replay_danger_positions(entry, wanted, root):
    library = base.library
    record = library.load(root / entry['file'])
    meta = record.meta
    identity = library.GameLibrary.game_id(meta['pieces_sha256'], meta['seed'], meta['piece_set'], record.moves)
    if identity != entry['id'] or meta['seed'] != entry['seed'] or meta['piece_set'] != entry['piece_set']:
        raise ValueError(f"Recorded game differs from its index entry {entry['id']}")
    total = len(record.moves)
    wanted = set(wanted)
    found = {}
    for index, (state, _move, row) in enumerate(library.replay(record, verify=False)):
        if index in wanted:
            snapshot = copy.deepcopy(state)
            snapshot._library_log = None
            library.ignore(snapshot)
            found[index] = {'kind': 'library', 'game_id': entry['id'], 'offset': index,
                            'remaining_recorded': total - index, 'state': snapshot, 'target': int(row[6])}
            if len(found) == len(wanted):
                return found
    raise ValueError(f"Recorded game {entry['id']} lacks requested danger positions")


def saved_danger_pool(run_dir):
    # The checkpoint signature is verified again by the base trainer. Its small
    # manifest normally lets us read the immutable IDs without loading weights twice.
    last = run_dir / 'last.pt'
    if not last.is_file():
        return None
    manifest = run_dir / 'manifest.json'
    if manifest.is_file():
        pairs = json.loads(manifest.read_text())['source_sha256']['danger_pool']
    else:
        pairs = torch.load(last, map_location='cpu', weights_only=True)['extra']['signature']['danger_pool']
    if not pairs:
        return []
    root = library_root()
    entries = {entry['id']: entry for entry in base.library.iter_index(root)}
    wanted = {}
    for game_id, offset in pairs:
        wanted.setdefault(game_id, []).append(offset)
    restored = {}
    for game_id, offsets in wanted.items():
        if game_id not in entries:
            raise ValueError(f'Checkpoint danger game {game_id} is missing from the library')
        restored[game_id] = replay_danger_positions(entries[game_id], offsets, root)
    return [restored[game_id][offset] for game_id, offset in pairs]


def build_danger_pool(args, root=None, *, exclude_game_ids=(), exclude_seeds=()):
    """Snapshots shortly before recorded top-outs (issue #32). Each game is replayed once;
    several offsets per game are captured, so the pool costs ~1 ms per replayed placement."""
    library = base.library
    root = library_root(root)
    low, high = args.danger_window
    rng = random.Random(args.seed + 32)
    entries = [e for e in library.iter_index(root)
               if e.get('game_over') and e.get('stop_reason') == 'top_out'
               and e.get('piece_set') == args.piece_set and e.get('placements', 0) > high
               and e['id'] not in exclude_game_ids and e.get('seed') not in exclude_seeds]
    entries.sort(key=lambda e: e['id'])
    rng.shuffle(entries)
    pool = []
    distinct = getattr(args, 'distinct_danger_seeds', False)
    used_seeds = set()
    for entry in entries:
        if len(pool) >= args.danger_pool:
            break
        if distinct and (entry.get('phase') != 'training' or entry['seed'] in used_seeds):
            continue
        try:
            total = entry['placements']
            if distinct:
                count = min(args.danger_per_game, args.danger_pool - len(pool))
                wanted = sorted(total - distance for distance in rng.sample(range(low, high + 1), count))
            else:
                wanted = sorted({total - rng.randint(low, high) for _ in range(args.danger_per_game)})
            found = replay_danger_positions(entry, wanted, root)
            pool.extend(found[offset] for offset in wanted)
            used_seeds.add(entry['seed'])
        except ValueError:
            continue  # recorded on a different engine/piece catalogue
    if distinct and len(pool) < args.danger_pool:
        raise ValueError('Insufficient distinct training seeds to fill the requested danger pool')
    return pool[:args.danger_pool]


def phase_behaviour(phase):
    return 'frozen_sampled_neural' if phase == 'selection-sampled' else f'greedy_{phase.removeprefix("selection-")}'


class SurvivalTrainer(v3.SurvivalTrainer):
    algorithm = ALGORITHM
    architecture = ARCHITECTURE

    def __init__(self, args, budget):
        self.danger_pool, self._danger_starts = [], {}
        self._danger_rng = random.Random(args.seed + 3200)
        if getattr(args, 'danger_starts', 0) > 0:
            self.danger_pool = saved_danger_pool(args.run_dir)
            if self.danger_pool is None:
                self.danger_pool = build_danger_pool(args, exclude_seeds=args.selection_seeds)
            if not self.danger_pool:
                raise ValueError('--danger-starts needs recorded top-out games in the game library')
        cache = {(p['game_id'], p['offset']): p for p in self.danger_pool}

        def rebuild(start):
            cached = cache.get((start['game_id'], start['offset']))
            if cached is not None:
                state = copy.deepcopy(cached['state'])
                base.library.ignore(state)
                return state, cached['target']
            return library_start_state(start['game_id'], start['offset'])

        base.START_BUILDERS['library'] = rebuild
        super().__init__(args, budget)

    def _next_danger_start(self):
        if self.danger_pool and self._danger_rng.random() < self.args.danger_starts:
            return self.danger_pool[self._danger_rng.randrange(len(self.danger_pool))]
        return None

    def restore_extra(self, saved):
        super().restore_extra(saved)
        if 'danger_rng' in saved['extra']:
            self._danger_rng.setstate(saved['extra']['danger_rng'])
        elif self.danger_pool:
            # Older checkpoints omit this private stream. Advance it to the
            # uninterrupted schedule for all episode IDs already allocated.
            for _ in range(self.next_episode):
                self._next_danger_start()

    def new_episode(self):
        chosen = self._next_danger_start()
        if chosen is None:
            return super().new_episode()
        number = self.next_episode
        self.next_episode += 1
        state = copy.deepcopy(chosen['state'])
        base.library.ignore(state)
        start = {k: chosen[k] for k in ('kind', 'game_id', 'offset', 'remaining_recorded')}
        self._danger_starts[number] = start
        return base.Episode(number, state, target=chosen['target'], phase='training', start=start)

    def emit(self, event, **fields):
        if event == 'episode_end':
            start = self._danger_starts.pop(fields.get('episode'), None) if hasattr(self, '_danger_starts') else None
            if start is None:
                episode = next((e for e in getattr(self, 'environments', ()) if e.number == fields.get('episode')), None)
                start = episode.start if episode is not None else None
            fields.update(start_kind=start['kind'] if start else 'fresh',
                          start_pieces=start['offset'] if start else 0,
                          own_placements=fields.get('pieces', 0) - (start['offset'] if start else 0))
        if event == 'survival_algorithm':
            fields.update(architecture=ARCHITECTURE, spatial_critic=True,
                          afterstate_spatial_actor=True, actor_critic_parameter_isolation=True)
        return v2.SurvivalTrainer.emit(self, event, **fields)

    def create_fresh_model(self):
        config = neural.Config(architecture=ARCHITECTURE, spatial_actor=True,
                               shortlist_mode=self.args.shortlist_mode,
                               screening=self.args.screening,
                               regional_critic=self.args.regional_critic,
                               recover_target=self.args.recover_target,
                               policy_mode=self.args.policy_mode,
                               residual_scale=self.args.residual_scale,
                               teacher_prior_scale=self.args.teacher_prior_scale,
                               detach_critic=False, centered_critic=False)
        self.imported_checkpoint_sha256 = None
        if self.args.pretrained is None:
            return neural.create_model(config).to(self.device)
        source = Path(self.args.pretrained)
        if self.args.preserve_pretrained_critic:
            # Saved checkpoints are replaced atomically. Hash and load one open
            # file so a concurrent replacement cannot change the imported model.
            with source.open('rb') as stream:
                imported_hash = base.stream_hash(stream)
                if imported_hash != self.signature['pretrained_sha256']:
                    raise ValueError('Pretrained checkpoint changed during initialization')
                stream.seek(0)
                loaded, checkpoint = neural.load_checkpoint(stream, self.device)
            extra = checkpoint['extra']
            calibration = extra.get('calibration', {})
            inherited = (calibration.get('stage') == 'explicitly_skipped' and
                         extra.get('signature', {}).get('preserve_pretrained_critic') is True)
            if (loaded.config != config or calibration.get('done') is not True or
                    not (calibration.get('stage') == 'passed' or inherited)):
                raise ValueError('Preserving the critic requires a matching policy and a calibrated V4 checkpoint')
        else:
            loaded, checkpoint = neural.load_checkpoint(source, self.device)
        # Screening only changes which placements are offered, never the weights, so a
        # sampled-mode pretrained actor may continue under exhaustive screening. A regional
        # critic changes only the critic input: the actor is copied, the critic starts fresh.
        if asdict(replace(loaded.config, screening=config.screening,
                          regional_critic=config.regional_critic)) != asdict(config):
            raise ValueError('Pretrained V4 config does not match the requested actor and shortlist')
        if loaded.config.regional_critic != config.regional_critic:
            model = neural.create_model(config).to(self.device)
            actor = {k: v for k, v in loaded.state_dict().items() if not k.startswith('value.')}
            missing, unexpected = model.load_state_dict(actor, strict=False)
            if unexpected or any(not k.startswith('value.') for k in missing):
                raise ValueError('Pretrained V4 actor parameters do not fit the regional-critic model')
            loaded = model
        loaded.config = config
        if checkpoint['teacher_parameters'] != game.network_parameters():
            raise ValueError('Pretrained V4 teacher weights differ from the runtime')
        if not all(bool(torch.isfinite(parameter).all()) for parameter in loaded.state_dict().values()):
            raise ValueError('Pretrained V4 contains non-finite weights')
        self.imported_checkpoint_sha256 = (imported_hash if self.args.preserve_pretrained_critic
                                          else base.file_hash(source))
        return loaded

    def _new_gpu_preparer(self):
        from tetris_gpu_simulation import GpuPreparer
        options = {'group_exhaustive': True} if getattr(self.args, 'group_exhaustive', False) else {}
        return GpuPreparer(self.model.config, self.device, **options)

    def prepare_episode(self, episode):
        """Calibration and selection games prepare single states on CUDA when enabled.

        Rollout ('training') episodes keep the inherited path; their batches are
        already prepared on device. Results match the CPU reference preparation.
        """
        # Training resets only need the None check before the next batched preparation;
        # on CUDA they use the same exact single-state path instead of CPU preparation
        # (~1.3 s per state with exhaustive screening). See issue #34.
        training_on_device = episode.phase == 'training' and getattr(self, '_gpu_prepare_active', False)
        if ((episode.phase == 'training' and not training_on_device) or not getattr(self.args, 'gpu_prepare', False)
                or self.device.type != 'cuda' or game.BACKEND != 'native'):
            return super().prepare_episode(episode)
        try:
            if getattr(self, '_gpu_preparer', None) is None:
                self._gpu_preparer = self._new_gpu_preparer()
            result = self._gpu_preparer.prepare_batch([episode.state], [episode.target])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return super().prepare_episode(episode)
        prepared = result.prepared[0]
        if prepared is not None:
            self.note_target(episode, int(result.targets[0]))
        return prepared

    def _overlap_split(self):
        if (not getattr(self.args, 'overlap_prepare', False) or not getattr(self, '_gpu_prepare_active', False)
                or len(self.environments) < 4):
            return 0
        return len(self.environments) // 2

    def committed_environment(self, index):
        """Issue #34: once the first half of the environments has committed, prepare it
        on a second CUDA preparer in a worker thread while the main thread commits and
        prepares the second half. CUDA waits release the GIL, so CPU commits overlap
        GPU preparation. The merged batch equals a single full preparation."""
        split = self._overlap_split()
        if not split or index != split - 1:
            return
        if getattr(self, '_overlap_preparer', None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._overlap_preparer = self._new_gpu_preparer()
            self._overlap_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='v4-prepare')
        states = [e.state for e in self.environments[:split]]
        targets = [e.target for e in self.environments[:split]]
        self._overlap_pending = (split, self._overlap_executor.submit(self._overlap_preparer.prepare_batch,
                                                                      states, targets))

    def _prepare_all(self):
        pending, self._overlap_pending = getattr(self, '_overlap_pending', None), None
        if pending is None:
            return super()._prepare_all()
        split, future = pending
        if not self._gpu_prepare_active or self._prepared_gpu_batch is not None:
            future.exception()  # drain; the full preparation below supersedes it
            return super()._prepare_all()
        from tetris_gpu_simulation import merge_prepared_batches
        try:
            if getattr(self, '_gpu_preparer', None) is None:
                self._gpu_preparer = self._new_gpu_preparer()
            second = self._gpu_preparer.prepare_batch([e.state for e in self.environments[split:]],
                                                      [e.target for e in self.environments[split:]])
            first = future.result()
        except torch.OutOfMemoryError:
            future.exception()
            torch.cuda.empty_cache()
            return super()._prepare_all()  # retries in one call, then falls back to CPU
        stats = getattr(self, '_gpu_prepare_stats', None)
        if stats is not None:
            stats['overlapped_steps'] = stats.get('overlapped_steps', 0) + 1
        self._accept_gpu_batch(merge_prepared_batches(first, second), calls=2)

    def augment_batch(self, result, states):
        """Hook for subclasses that add per-candidate arrays to a prepared CUDA batch."""
        return result

    def lockstep_available(self):
        return (getattr(self.args, 'lockstep_selection', False) and getattr(self.args, 'gpu_prepare', False)
                and self.device.type == 'cuda' and game.BACKEND == 'native')

    def _selection_preparer(self):
        if getattr(self, '_gpu_preparer', None) is None:
            self._gpu_preparer = self._new_gpu_preparer()
        return self._gpu_preparer

    def lockstep_games(self, phase, choose, on_step=None, *, greedy=False):
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
            result = self.augment_batch(preparer.prepare_batch([episodes[i].state for i in active],
                                                               [episodes[i].target for i in active]),
                                        [episodes[i].state for i in active])
            live = []
            for local, i in enumerate(active):
                if result.prepared[local] is None:
                    reasons[i] = base.stopped_reason(episodes[i].state)
                else:
                    self.note_target(episodes[i], int(result.targets[local]))
                    live.append(i)
            if live:
                rows = [active[local] for local in result.environment_indices]
                if greedy and on_step is None:
                    logits = self.model.actor_logits(result.batch)
                    actions = logits.argmax(-1)
                else:
                    prediction = self.model(result.batch)
                    logp = self.policy_logp(prediction['logits'], result.batch['valid'])
                    if on_step is not None:
                        on_step(rows, episodes, prediction)
                    actions = choose(rows, prediction['logits'], logp, episodes, result)
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
                f'selection-{policy}', None, greedy=True)
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
                if pieces % v3.CRITIC_SAMPLE_STRIDE == 0:
                    samples[i].append((pieces, float(prediction['value'][row])))

        def sample(rows, logits, logp, *_):
            return torch.stack([torch.multinomial(logp[row].exp(), 1, generator=generators[i])[0]
                                for row, i in enumerate(rows)])

        with base.preserve_rng(), torch.inference_mode():
            episodes, reasons, seconds = self.lockstep_games('selection-sampled', sample, record)
        runs = []
        for i, (episode, reason) in enumerate(zip(episodes, reasons)):
            terminated = reason in ('top_out', 'planner_blocked')
            base.library.finish(episode.state, reason)
            diagnostics = v3.realized_critic_errors(samples[i], episode.state.pieces) if terminated else None
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

    def selection_ready(self):
        return not self.args.defer_selection and super().selection_ready()

    def select(self):
        if self.selection_ready() and not self.budget.expired():
            # Lockstep selection emits its games only when they finish; announce the phase now.
            self.emit('selection_started', runner_updates=self.progress['runner_updates'],
                      selection_seeds=len(self.args.selection_seeds), lockstep=self.lockstep_available(),
                      selection_cap=self.args.selection_pieces)
        previous = self.last_evaluated_update
        super().select()
        if (not getattr(self.args, 'afterstate_selection', False) or self.budget.expired()
                or self.last_evaluated_update in (None, previous) or not self.lockstep_available()):
            return
        result = self.afterstate_policy_check()
        result.update(runner_updates=self.progress['runner_updates'], checkpoint_sha256=base.file_hash(self.last))
        base.durable_json(self.root / f'afterstate-selection-{self.progress["runner_updates"]:08d}.json', result)
        self.emit('afterstate_selection_complete', score=result['score'], top_k=self.args.afterstate_top_k,
                  runner_updates=self.progress['runner_updates'])

    def afterstate_policy_check(self):
        """Issue #37: greedy games choosing, among the actor's top-k, the candidate whose
        afterstate the critic values highest. Report-only; checkpoint selection is unchanged."""
        from tetris_survival_v4 import afterstate_values
        self.model.eval()

        def choose(rows, logits, logp, episodes, result):
            top = logits.topk(min(self.args.afterstate_top_k, logits.shape[1]), dim=-1).indices
            finite = torch.isfinite(logits.gather(1, top)).cpu().tolist()
            indices, options = [], []
            for row, (ranked, ok) in enumerate(zip(top.cpu().tolist(), finite)):
                indices.append([a for a, keep in zip(ranked, ok) if keep])
                pool = result.prepared[result.environment_indices[row]].options
                options.append([pool[a] for a in indices[-1]])
            values = afterstate_values(self.model, [episodes[i].state for i in rows],
                                       [episodes[i].target for i in rows], options)
            return torch.tensor([valid[int(v.argmax())] for valid, v in zip(indices, values)], device=logits.device)

        with base.preserve_rng(), torch.inference_mode():
            episodes, reasons, seconds = self.lockstep_games('selection-afterstate', choose)
        runs = []
        for episode, reason in zip(episodes, reasons):
            base.library.finish(episode.state, reason)
            runs.append({**episode.state.summary(), 'policy': 'afterstate', 'stop_reason': reason,
                         'evaluation_complete': reason in ('top_out', 'planner_blocked', 'piece_limit'),
                         'capped': reason == 'piece_limit', 'wall_seconds': seconds})
            self.emit('afterstate_selection_game', **runs[-1])
        complete = all(run['evaluation_complete'] for run in runs)
        return {'complete': complete, 'runs': runs, 'score': base.survival_score(runs) if complete else None,
                'top_k': self.args.afterstate_top_k,
                'note': 'Report-only afterstate ranking with the current critic; best checkpoint selection unchanged.'}

    def configure_model(self, resuming):
        if self.model.config.architecture != self.architecture:
            raise ValueError(f'{self.algorithm} cannot resume a different model architecture')
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
                     'seed': self.args.seed, 'architecture': self.architecture,
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
        if self.args.danger_starts:
            signature.update(danger_pool_size=self.args.danger_pool, danger_per_game=self.args.danger_per_game,
                             distinct_danger_seeds=self.args.distinct_danger_seeds)
        if self.args.defer_selection:
            signature['defer_selection'] = True
        if getattr(self.args, 'group_exhaustive', False):
            signature['group_exhaustive'] = True
        if self.args.preserve_pretrained_critic:
            signature['preserve_pretrained_critic'] = True
        signature.update(algorithm=ALGORITHM,
                         v4_trainer_sha256=base.file_hash(Path(__file__)),
                         spatial_architecture_sha256=base.file_hash(Path(__file__).with_name('tetris_survival_v4.py')),
                         pretrained_sha256=(base.file_hash(self.args.pretrained)
                                            if self.args.pretrained is not None else None),
                         shortlist_mode=self.args.shortlist_mode,
                         screening=self.args.screening,
                         adaptive_lr=self.args.adaptive_lr,
                         danger_starts=self.args.danger_starts,
                         danger_window=list(self.args.danger_window) if self.args.danger_starts else None,
                         danger_pool=([[p['game_id'], p['offset']] for p in self.danger_pool]
                                      if self.args.danger_starts else None),
                         lockstep_selection=self.lockstep_available(),
                         overlap_prepare=self.args.overlap_prepare,
                         afterstate_selection=self.args.afterstate_top_k if self.args.afterstate_selection else None,
                         adaptive_kl_target=self.args.adaptive_kl_target if self.args.adaptive_lr else None,
                         adaptive_lr_range=self.args.adaptive_lr_range if self.args.adaptive_lr else None,
                         spatial_actor=True, fresh_ppo_progress=True)
        return signature

    def checkpoint_metadata(self):
        values = super().checkpoint_metadata()
        values.update(algorithm=self.algorithm, architecture=self.architecture, danger_rng=self._danger_rng.getstate(),
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
    extra = argparse.ArgumentParser(add_help=False, allow_abbrev=False, description='Spatial V4 PPO options')
    extra.add_argument('--pretrained', type=Path)
    extra.add_argument('--preserve-pretrained-critic', action='store_true',
                       help='Keep the complete calibrated V4 critic when explicitly skipping calibration')
    extra.add_argument('--shortlist-mode', choices=('teacher-random', 'diverse', 'teacher-top'), default='teacher-random')
    extra.add_argument('--screening', choices=('sampled', 'exhaustive'), default=None,
                       help="Candidate pool: 'sampled' (config.candidates placements) or every legal placement. "
                            "Default: exhaustive with the GPU defaults, sampled otherwise (CPU exhaustive is ~1.3 s/state)")
    extra.add_argument('--regional-critic', action='store_true',
                       help='Add per-region height/hole features to the critic input (issue #38)')
    extra.add_argument('--danger-starts', type=float, default=0.,
                       help='Share of training episodes started shortly before a recorded top-out (issue #32)')
    extra.add_argument('--danger-window', type=int, nargs=2, default=(500, 3000), metavar=('MIN', 'MAX'),
                       help='Start this many placements before the recorded top-out')
    extra.add_argument('--danger-pool', type=int, default=32, help='Number of distinct start states')
    extra.add_argument('--distinct-danger-seeds', action='store_true',
                       help='Fill the pool from distinct source seeds in completed training games')
    extra.add_argument('--group-exhaustive', action='store_true',
                       help='Group exhaustive CUDA preparation by per-piece candidate count')
    extra.add_argument('--danger-per-game', type=int, default=4, help='Start states taken from each replayed game')
    extra.add_argument('--afterstate-selection', action='store_true',
                       help='Also report afterstate-ranked selection games after each selection (issue #37)')
    extra.add_argument('--afterstate-top-k', type=int, default=16,
                       help='Actor candidates whose afterstates the critic ranks')
    extra.add_argument('--overlap-prepare', action='store_true',
                       help='Overlap CPU move commits with CUDA preparation of the other half (issue #34). '
                            'Identical results; measured ~+1%% on an idle RTX 4060 Ti, so off by default')
    extra.add_argument('--defer-selection', action='store_true',
                       help='Train to the requested budget and evaluate frozen checkpoints separately')
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
    if '-h' in supplied or '--help' in supplied:
        extra.print_help()
    gpu = gpu_defaults(remaining, options.no_gpu_defaults)
    args = v3.arguments(['--run-dir', str(base.ROOT/'runs/survival-ppo-v4'),
                         '--gradient-clip-scope', 'separate',
                         '--selection-seeds', *map(str, SELECTION_SEEDS),
                         # 32 x 256 = 8,192 placements per update. Measured (exhaustive, shared GPU):
                         # 16 envs 227/s, 32 envs 247/s, 64 envs 67/s (preparation memory pressure).
                         '--envs', '32', '--rollout', '256', *gpu, *remaining])
    args.defer_selection = options.defer_selection
    args.group_exhaustive = options.group_exhaustive
    args.distinct_danger_seeds = options.distinct_danger_seeds
    args.pretrained = options.pretrained
    args.preserve_pretrained_critic = options.preserve_pretrained_critic
    if args.preserve_pretrained_critic and (args.pretrained is None or args.calibration_games != 0):
        extra.error('--preserve-pretrained-critic requires --pretrained, --calibration-games 0 and --allow-uncalibrated')
    args.shortlist_mode = options.shortlist_mode
    args.screening = options.screening or ('exhaustive' if gpu else 'sampled')
    args.adaptive_lr = not options.fixed_lr
    args.lockstep_selection = not options.sequential_selection
    args.overlap_prepare = options.overlap_prepare
    args.afterstate_selection, args.afterstate_top_k = options.afterstate_selection, options.afterstate_top_k
    if args.afterstate_top_k < 2:
        raise ValueError('--afterstate-top-k must be at least 2')
    args.regional_critic = options.regional_critic
    args.danger_starts, args.danger_window = options.danger_starts, tuple(options.danger_window)
    args.danger_pool, args.danger_per_game = options.danger_pool, options.danger_per_game
    if not 0 <= args.danger_starts < 1 or not 0 < args.danger_window[0] <= args.danger_window[1] \
            or min(args.danger_pool, args.danger_per_game) < 1:
        raise ValueError('--danger-starts must be in [0, 1) with a positive MIN <= MAX window')
    available_offsets = args.danger_window[1] - args.danger_window[0] + 1
    if args.distinct_danger_seeds and min(args.danger_pool, args.danger_per_game) > available_offsets:
        raise ValueError('Danger window is too small for distinct offsets in each source game')
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
