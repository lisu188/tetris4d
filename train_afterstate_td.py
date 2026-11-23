#!/usr/bin/env python3
"""Train a V4 afterstate value head with sampled on-policy TD(lambda)."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import time
import uuid
import zlib

import torch
from torch.nn import functional as F

import train_overnight as base
import train_survival_v4 as v4
import tetris_afterstate_td as td
from tools import evaluate_afterstate_td as evaluation

game, neural = base.game, base.neural
ALGORITHM = 'afterstate-td-lambda'
UNIT = .001
LEGACY_TRAINER_SHA256 = 'b39b02e6c7c875c1afa37848d948bbdd11ca2aaf742b4b2e79c3380fab1affd6'
AMENDMENT_FILE = 'cap-amendment.json'
BOUNDARY_FILE = 'cap-amendment-boundary.pt'


def lambda_returns(next_values, has_next, continues, lam):
    """Each afterstate earns a placement only if a *following* placement exists.

    continues links to the next row in the same episode. At the final row of
    a rollout and at a cap, bootstrap using the sampled successor in that game.
    """
    targets = torch.zeros_like(next_values)
    for t in reversed(range(len(targets))):
        future = next_values[t]
        if t + 1 < len(targets):
            future = torch.where(continues[t], future.lerp(targets[t + 1], lam), future)
        targets[t] = torch.where(has_next[t], UNIT + future, 0.)
    return targets


@dataclass
class Environment:
    state: object
    target: int
    number: int
    start: dict
    start_pieces: int
    start_layers: int


def pack_env(env):
    """Save exactly the engine state used for headless simulation, including its piece bag."""
    s = env.state
    board = struct.pack('<4096H', *s.board)
    return dict(seed=s.seed, number=env.number, target=env.target, start=env.start,
                start_pieces=env.start_pieces, start_layers=env.start_layers,
                board=board, board_hash=hashlib.sha256(board).hexdigest(), rng=s.stream.rng.getstate(),
                bag=list(s.stream.bag), current=s.current, next_piece=s.next_piece, pieces=s.pieces,
                cells_placed=s.cells_placed, piece_counts=dict(s.piece_counts), layers=s.layers,
                score=s.score, game_over=s.game_over)


def unpack_env(payload):
    if hashlib.sha256(payload['board']).hexdigest() != payload['board_hash']:
        raise ValueError('TD environment board checksum mismatch')
    s = game.Game(payload['seed'], 'expanded')
    base.library.ignore(s)
    s.board = struct.unpack('<4096H', payload['board'])
    s.stream.rng.setstate(payload['rng'])
    s.stream.bag = list(payload['bag'])
    for field in ('current', 'next_piece', 'pieces', 'cells_placed', 'layers', 'score', 'game_over'):
        setattr(s, field, payload[field])
    s.piece_counts = dict(payload['piece_counts'])
    return Environment(s, payload['target'], payload['number'], payload['start'],
                       payload['start_pieces'], payload['start_layers'])


def screening_exclusions(path):
    if path is None:
        return (), ()
    manifest = json.loads(path.read_bytes())
    schemas = ('afterstate-development-positions-v1', 'afterstate-screen-exclusions-v1')
    if manifest.get('schema') not in schemas:
        raise ValueError('Unrecognized screening position manifest')
    rows = manifest.get('positions', [])
    phases = {row['phase'] for row in rows}
    if (not {'discovery', 'confirmation'} <= phases <= {'discovery', 'confirmation', 'reserved_extension'} or
            any(type(row['seed']) is not int or not row.get('game_id') for row in rows)):
        raise ValueError('Incomplete discovery/confirmation game IDs or stream seeds')
    return sorted({r['game_id'] for r in rows}), sorted({r['seed'] for r in rows})


def danger_pool(args, calibration):
    if args.danger_starts == 0:
        return []
    screen_ids, screen_seeds = screening_exclusions(args.exclude_positions)
    excluded_ids = sorted({p[0] for p in calibration['position_ids']} | set(screen_ids))
    excluded_seeds = sorted(set(evaluation.DEVELOPMENT_SEEDS) | set(evaluation.HOLDOUT_SEEDS) | set(screen_seeds))
    identity = dict(schema=1, seed=args.seed, pool=args.danger_pool, window=list(args.danger_window),
                    per_game=args.danger_per_game, catalog=neural.fingerprint(),
                    exclude_ids=excluded_ids, exclude_seeds=excluded_seeds)
    cache = args.danger_cache or args.run_dir / 'danger-pool.pt'
    with base.run_lock(cache.parent / ('.' + cache.name + '.lock')):
        if cache.is_file():
            saved = torch.load(cache, map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError(f'Danger cache {cache} belongs to different inputs')
            rows = saved['rows']
        else:
            pool = v4.build_danger_pool(args, exclude_game_ids=set(excluded_ids),
                                        exclude_seeds=set(excluded_seeds))
            if len(pool) != args.danger_pool:
                raise ValueError(f'Requested {args.danger_pool} disjoint danger starts, found {len(pool)}')
            rows = []
            for p in pool:
                s = p['state']
                start = {key: p[key] for key in ('kind', 'game_id', 'offset', 'remaining_recorded')}
                rows.append(pack_env(Environment(s, p['target'], 0, start, s.pieces, s.layers)))
            temp = cache.with_suffix(cache.suffix + '.tmp')
            try:
                torch.save(dict(identity=identity, rows=rows), temp)
                temp.replace(cache)
            finally:
                temp.unlink(missing_ok=True)
    if (len(rows) != args.danger_pool or any(row['seed'] in excluded_seeds or
            row['start']['game_id'] in excluded_ids or
            hashlib.sha256(row['board']).hexdigest() != row['board_hash'] for row in rows)):
        raise ValueError('Invalid or overlapping training danger cache')
    return rows


class Rollout:
    """Compact, checkpointable records. A pending row owns one selected feature per environment."""
    def __init__(self, steps, envs):
        size = (steps, envs)
        self.features = torch.zeros((*size, td.FEATURES))
        self.valid = torch.zeros(size, dtype=torch.bool)
        self.has_next = torch.zeros_like(self.valid)
        self.continues = torch.zeros_like(self.valid)
        self.next_values = torch.zeros(size)
        self.pending = torch.zeros((envs, td.FEATURES))
        self.pending_valid = torch.zeros(envs, dtype=torch.bool)
        self.steps = 0

    def record(self, features, values, live, committed):
        # A committed move is the successor of the previous afterstate. At a cap
        # we obtain its value on the old board without committing it across reset.
        if self.pending_valid.any():
            t = self.steps
            self.features[t].copy_(self.pending)
            self.valid[t].copy_(self.pending_valid)
            self.has_next[t].copy_(live & self.pending_valid)
            self.continues[t].copy_(committed & self.pending_valid)
            self.next_values[t].copy_(values)
            self.steps += 1
        self.pending.copy_(features)
        self.pending_valid.copy_(committed)

    def save(self):
        return dict(features=self.features[:self.steps].clone(), valid=self.valid[:self.steps].clone(),
                    has_next=self.has_next[:self.steps].clone(), continues=self.continues[:self.steps].clone(),
                    next_values=self.next_values[:self.steps].clone(),
                    pending=self.pending, pending_valid=self.pending_valid)

    def restore(self, saved):
        self.steps = len(saved['valid'])
        if not 0 <= self.steps <= len(self.valid):
            raise ValueError('Invalid partial TD rollout length')
        for field in ('features', 'valid', 'has_next', 'continues', 'next_values'):
            getattr(self, field)[:self.steps].copy_(saved[field])
        self.pending.copy_(saved['pending'])
        self.pending_valid.copy_(saved['pending_valid'])


def provenance(args, calibration, pool):
    sources = ('train_afterstate_td.py', 'tetris_afterstate_td.py', 'tetris_survival_v4.py',
               'tetris_gpu_simulation.py', 'tetris_survival_v5.py', 'tetris_neural.py',
               'sparse_tetris_4d_optimized.py', 'tools/evaluate_afterstate_td.py')
    keys = ('seed', 'envs', 'rollout', 'batch', 'pieces', 'hours', 'lr', 'lam', 'danger_starts',
            'selection_hours', 'selection_pieces', 'device')
    extra = {key: getattr(args, key) for key in ('actor_bound', 'target_tau', 'clip_grad')
             if getattr(args, key) is not None}
    if args.warm_start:
        extra['warm_start_sha256'] = base.file_hash(args.warm_start)
    if args.exclude_positions:
        extra['exclude_positions_sha256'] = base.file_hash(args.exclude_positions)
        _, extra['excluded_screen_seeds'] = screening_exclusions(args.exclude_positions)
    return dict(algorithm=ALGORITHM, **{k: getattr(args, k) for k in keys}, **extra,
                sha256={name: base.file_hash(Path(__file__).parent / name) for name in sources},
                source_sha256=base.file_hash(args.source_checkpoint), weights_sha256=base.file_hash(args.weights),
                calibration_sha256=base.file_hash(args.calibration), beta=calibration['beta'],
                danger_ids=[[p['start']['game_id'], p['start']['offset'], p['board_hash']] for p in pool])


def cap_amendment(run_dir, manifest, original_manifest_sha256, signature):
    """Validate the one supported protocol amendment while holding both trainer locks."""
    manifest_bytes = (run_dir / 'manifest.json').read_bytes()
    if (hashlib.sha256(manifest_bytes).hexdigest() != original_manifest_sha256
            or json.loads(manifest_bytes) != manifest):
        raise ValueError('Original TD manifest digest differs')
    prior = manifest['signature']
    trainer_file = 'train_afterstate_td.py'
    trainer_sha = base.file_hash(Path(__file__))
    expected = copy.deepcopy(prior)
    expected['pieces'] = 0
    expected['sha256'][trainer_file] = trainer_sha
    if (prior['algorithm'] != ALGORITHM or type(prior['pieces']) is not int or prior['pieces'] < 1
            or prior['sha256'][trainer_file] not in (LEGACY_TRAINER_SHA256, trainer_sha)
            or signature != expected):
        raise ValueError('Only the training piece cap and supported trainer revision may change')
    if time.time() >= manifest['deadline_epoch']:
        raise ValueError('Cannot uncap an expired TD run')

    receipt_path = run_dir / AMENDMENT_FILE
    boundary = run_dir / BOUNDARY_FILE
    last = run_dir / 'last.pt'
    if not last.is_file():
        raise ValueError('An existing TD training checkpoint is required to uncap a run')
    summary_file = run_dir / 'summary.json'
    latest_summary = json.loads(summary_file.read_bytes()) if summary_file.exists() else {}
    if latest_summary.get('complete'):
        raise ValueError('Cannot uncap a completed TD run')
    existing = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
    summary = existing['interrupted_summary'] if existing is not None else latest_summary
    reason = summary.get('stop_reason', '')
    if (summary.get('complete') is not False or reason not in ('signal_1', 'signal_2', 'signal_15')
            or summary.get('algorithm') != ALGORITHM or summary.get('start_epoch') != manifest['start_epoch']
            or summary.get('deadline_epoch') != manifest['deadline_epoch']):
        raise ValueError('The original run must have a matching graceful signal-interruption summary')

    # A failed process can leave the independently copied boundary before publishing
    # its receipt. An existing boundary is always verified, never overwritten.
    checkpoint_sha = summary['last_sha256']
    if existing is None and base.file_hash(last) != checkpoint_sha:
        raise ValueError('Interrupted summary does not identify the current checkpoint')
    source = boundary if boundary.exists() or existing is not None else last
    if not source.is_file() or base.file_hash(source) != checkpoint_sha:
        raise ValueError('Cap-amendment boundary checkpoint digest differs')
    if boundary.exists() and boundary.samefile(last):
        raise ValueError('Cap-amendment boundary must be an independent checkpoint copy')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    extra, progress = saved['extra'], saved['progress']
    if (extra['signature'] != prior or extra['deadline_epoch'] != manifest['deadline_epoch']
            or extra['reason'] != reason or any(summary.get(k) != v for k, v in progress.items())):
        raise ValueError('Interrupted checkpoint and summary disagree on settings, reason, or progress')
    journal = zlib.decompress(extra['metrics_zlib'])
    if len(journal) != extra['metrics_bytes']:
        raise ValueError('Invalid cap-amendment boundary metrics journal')
    event = json.loads(journal.rstrip(b'\n').rsplit(b'\n', 1)[-1])
    if (event.get('event') != 'run_interrupted' or event.get('stop_reason') != reason
            or not event.get('session') or any(event.get(k) != v for k, v in progress.items())):
        raise ValueError('Missing final interruption event in cap-amendment boundary')
    receipt = dict(schema='afterstate-training-cap-amendment-v1',
                   original_manifest_sha256=original_manifest_sha256,
                   prior_signature=prior, effective_signature=signature,
                   start_epoch=manifest['start_epoch'], deadline_epoch=manifest['deadline_epoch'],
                   boundary_checkpoint=BOUNDARY_FILE, boundary_checkpoint_sha256=checkpoint_sha,
                   boundary_updates=progress['updates'], boundary_session=event['session'],
                   interrupted_summary=summary)
    if existing is not None:
        if receipt != existing:
            raise ValueError('Cap-amendment receipt differs from original run or interrupted checkpoint')
        return receipt
    if not boundary.exists():
        base.copy_atomic(last, boundary)
        if base.file_hash(boundary) != checkpoint_sha:
            raise ValueError('Independent boundary checkpoint copy differs')
    base.durable_json(receipt_path, receipt)
    if os.name == 'posix':
        directory_fd = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return receipt


class Trainer:
    def __init__(self, args, budget, pool, signature, manifest, amendment=None):
        self.args, self.budget, self.pool, self.signature = args, budget, pool, signature
        self.root, self.device = args.run_dir, torch.device(args.device)
        self.last, self.metrics = self.root / 'last.pt', self.root / 'metrics.jsonl'
        self.amendment = amendment
        self.amendment_sha = base.file_hash(self.root / AMENDMENT_FILE) if amendment is not None else None
        self.at_cap_boundary = False
        self.started, self.deadline = manifest['start_epoch'], manifest['deadline_epoch']
        self.rng, self.session = random.Random(args.seed), uuid.uuid4().hex[:12]
        self.excluded_seeds = set(evaluation.HOLDOUT_SEEDS) | set(evaluation.DEVELOPMENT_SEEDS)
        self.excluded_seeds.update(signature.get('excluded_screen_seeds', ()))
        self.buffer = Rollout(args.rollout, args.envs)
        self.best = None
        self.next_selection = 1
        self.last_selected_update = 0
        self.next_episode = 0
        self.training_seconds = 0.
        self.max_step_seconds = .05
        self.max_checkpoint_seconds = .05
        self.peak_allocated = self.peak_reserved = 0
        self.progress = dict(updates=0, runner_updates=0, optimizer_steps=0, placements=0,
                             clears=0, finished_episodes=0, terminal_afterstates=0, truncated_afterstates=0)
        if self.last.exists():
            self.model, saved = neural.load_checkpoint(self.last, self.device, restore_rng=True)
            extra = saved['extra']
            if amendment is not None and extra['signature'] == amendment['prior_signature']:
                self.at_cap_boundary = base.file_hash(self.last) == amendment['boundary_checkpoint_sha256']
                if not self.at_cap_boundary:
                    raise ValueError('Only the exact interrupted checkpoint may cross the cap amendment')
            if (extra['deadline_epoch'] != self.deadline or not self.at_cap_boundary and
                    (extra['signature'] != signature or extra.get('cap_amendment_sha256') != self.amendment_sha)):
                raise ValueError('TD checkpoint settings, source or fixed deadline changed')
            self.envs = [unpack_env(s) for s in extra['environments']]
            self.buffer.restore(extra['rollout'])
            self.progress = saved['progress']
            self.best, self.next_selection = extra['best'], extra['next_selection']
            self.last_selected_update = extra.get('last_selected_update', 0)
            self.next_episode = extra['next_episode']
            self.rng.setstate(extra['rng'])
            self.optimizer = self.new_optimizer()
            self.optimizer.load_state_dict(saved['optimizer'])
            self.training_seconds = extra['training_seconds']
            self.max_checkpoint_seconds = extra['max_checkpoint_seconds']
            self.max_step_seconds = extra['max_step_seconds']
            self.peak_allocated, self.peak_reserved = extra['cuda_peak_bytes']
            # The checkpoint is the transaction boundary. Recover metric rows even
            # when the filesystem lost the JSONL entirely, or retained a partial row.
            journal = zlib.decompress(extra['metrics_zlib'])
            if len(journal) != extra['metrics_bytes']:
                raise ValueError('Invalid metrics journal in TD checkpoint')
            with self.metrics.open('a+b') as stream:
                stream.seek(0)
                if stream.read(len(journal)) != journal:
                    stream.truncate(0)
                    stream.write(journal)
                stream.truncate(len(journal))
        else:
            torch.manual_seed(args.seed)
            source, saved = neural.load_checkpoint(args.source_checkpoint, self.device)
            if saved['teacher_parameters'] != game.network_parameters():
                raise ValueError('TD initializer and current heuristic weights differ')
            self.model = td.from_v4(source, actor_bound=args.actor_bound, value_scale=signature['beta'])
            if args.warm_start is not None:
                # Warm start only the scorer. All source weights, rollout state,
                # optimizer moments, RNGs and the run budget are otherwise fresh.
                with base.preserve_rng():
                    warm, initializer = neural.load_checkpoint(args.warm_start, self.device)
                if (not isinstance(warm, td.AfterstateTDPolicy) or initializer['progress'].get('updates', 0) < 1
                        or initializer['teacher_parameters'] != game.network_parameters()
                        or warm.base.config != self.model.base.config
                        or v4.v3.model_digest(warm.base) != v4.v3.model_digest(source)
                        or initializer['extra'].get('signature', {}).get(
                            'source_sha256', signature['source_sha256']) != signature['source_sha256']
                        or base.file_hash(args.warm_start) != signature['warm_start_sha256']):
                    raise ValueError('Warm start must be a trained TD checkpoint with the pinned V4 and teacher')
                self.model.value.load_state_dict(warm.value.state_dict(), strict=True)
            self.optimizer = self.new_optimizer()
            self.envs = [self.new_env() for _ in range(args.envs)]
        self.model.eval()
        if (self.model.actor_bound != args.actor_bound or
                args.actor_bound is not None and self.model.config.afterstate_scale != signature['beta']):
            raise ValueError('TD checkpoint policy bound or calibrated value scale changed')
        self.target_head = None
        if args.target_tau is not None:
            self.target_head = copy.deepcopy(self.model.value).eval().requires_grad_(False)
            if self.last.exists():
                self.target_head.load_state_dict(extra['target_head'], strict=True)
                if v4.v3.model_digest(self.target_head) != extra['target_sha256']:
                    raise ValueError('TD target head changed on resume')
        self.actor_hash = v4.v2.actor_digest(self.model.base)
        if self.last.exists() and self.actor_hash != extra['actor_sha256']:
            raise ValueError('TD frozen actor changed on resume')
        if self.last.exists() and v4.v3.model_digest(self.model.value) != extra['scorer_sha256']:
            raise ValueError('TD value head changed on resume')
        self.restore_best()
        self.preparer = None
        if self.device.type == 'cuda':
            from tetris_gpu_simulation import GpuPreparer
            self.preparer = GpuPreparer(self.model.base.config, self.device)
        self.next_save = time.monotonic() + args.save_seconds

    def new_optimizer(self):
        return torch.optim.AdamW(self.model.value.parameters(), lr=self.args.lr)

    def new_env(self):
        number = self.next_episode
        self.next_episode += 1
        if self.pool and self.rng.random() < self.args.danger_starts:
            env = unpack_env(self.rng.choice(self.pool))
            env.number = number
            return env
        seed = 2 * (self.args.seed + number)
        while seed in self.excluded_seeds:
            number = self.next_episode
            self.next_episode += 1
            seed = 2 * (self.args.seed + number)
        state = game.Game(seed, 'expanded')
        base.library.ignore(state)
        return Environment(state, 1, number, {'kind': 'fresh'}, 0, 0)

    def emit(self, event, **fields):
        row = dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   session=self.session, elapsed_seconds=time.time()-self.started, **self.progress, **fields)
        with self.metrics.open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')

    def active_games(self):
        return [dict(episode=e.number, seed=e.state.seed, pieces=e.state.pieces,
                     start_kind=e.start['kind'], own_placements=e.state.pieces-e.start_pieces,
                     clears=e.state.layers-e.start_layers) for e in self.envs]

    def sampled_rows(self, pool):
        scores = self.model.score(pool.features)
        slots = self.model.choose(pool, beta=self.signature['beta'], scores=scores)
        rows = torch.arange(len(slots), device=self.device)
        chosen = pool.features[rows, slots]
        # Only the single on-policy successor is bootstrapped. The lagged head
        # is held fixed throughout collection and never selects a different move.
        head = self.target_head if self.target_head is not None else self.model.value
        return pool.indices[rows, slots], chosen, td.score_features(head, chosen)

    @torch.no_grad()
    def choices(self):
        """Return only selected afterstate features/values on CPU and candidate handles."""
        states = [e.state for e in self.envs]
        targets = [e.target for e in self.envs]
        choices = {}
        features = torch.zeros_like(self.buffer.pending)
        values = torch.zeros(len(states))
        if self.preparer is not None:
            result = self.preparer.prepare_batch(states, targets)
            indices = list(result.environment_indices)
            if indices:
                pool = self.model.prepare_device(result, states)
                original_indices, chosen, next_values = self.sampled_rows(pool)
                features[indices] = chosen.cpu()
                values[indices] = next_values.cpu()
                for i, candidate in zip(indices, result.selected_candidates(original_indices)):
                    choices[i] = int(result.targets[i]), candidate
        else:
            for i, (state, target) in enumerate(zip(states, targets)):
                prepared = neural.prepare(state, target, self.model.base.config)
                if prepared is None:
                    continue
                pool = self.model.prepare_cpu(prepared, state)
                original_indices, chosen, next_values = self.sampled_rows(pool)
                features[i] = chosen[0].cpu()
                values[i] = next_values.item()
                choices[i] = int(prepared.arrays['targets']), prepared.options[original_indices.item()]
        return features, values, choices

    def step(self):
        features, values, choices = self.choices()
        live = torch.zeros(len(self.envs), dtype=torch.bool)
        committed = torch.zeros_like(live)
        for i, env in enumerate(self.envs):
            s = env.state
            if i in choices:
                live[i] = True
            if i not in choices or (self.args.pieces > 0 and s.pieces-env.start_pieces >= self.args.pieces):
                reason = ('piece_limit' if i in choices else 'top_out' if s.game_over else 'planner_blocked')
                if self.buffer.pending_valid[i]:
                    key = 'truncated_afterstates' if i in choices else 'terminal_afterstates'
                    self.progress[key] += 1
                self.progress['finished_episodes'] += 1
                self.emit('episode_end', episode=env.number, seed=s.seed, pieces=s.pieces, hyperlayers=s.layers,
                          own_placements=s.pieces-env.start_pieces, start_pieces=env.start_pieces,
                          start_kind=env.start['kind'], stop_reason=reason)
                self.envs[i] = self.new_env()
                continue
            env.target, candidate = choices[i]
            move = game.materialize_move(s.board, candidate)
            s.commit(move)
            # Geometry of past pieces is solely for rendering. The board is authoritative
            # for physics, value features and engine legality checks.
            s.locked_pieces = ()
            if move.layers:
                following = game.next_hyperlayer_target(s.board, env.target)
                if following is not None:
                    env.target = following
            self.progress['placements'] += 1
            self.progress['clears'] += move.layers
            committed[i] = True
        self.buffer.record(features, values, live, committed)

    def update(self):
        buf = self.buffer
        if buf.steps != self.args.rollout:
            raise ValueError('TD optimizer requires a complete rollout')
        bootstrap = buf.next_values.double() if self.target_head is not None else buf.next_values
        targets = lambda_returns(bootstrap, buf.has_next, buf.continues, self.args.lam)[buf.valid].to(
            self.device, dtype=buf.features.dtype)
        features = buf.features[buf.valid].to(self.device)
        order = torch.randperm(len(targets), device=self.device)
        squared_error = torch.zeros((), device=self.device)
        for batch in order.split(self.args.batch):
            predictions = self.model.score(features[batch])
            loss = F.mse_loss(predictions, targets[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError('TD loss is not finite')
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.value.parameters(), self.args.clip_grad,
                                               error_if_nonfinite=True)
            self.optimizer.step()
            if self.target_head is not None:
                with torch.no_grad():
                    for target, current in zip(self.target_head.parameters(), self.model.value.parameters()):
                        target.lerp_(current, self.args.target_tau)
            self.progress['optimizer_steps'] += 1
            squared_error += loss.detach() * len(batch)
        self.progress['updates'] += 1
        self.progress['runner_updates'] += 1
        self.emit('update', value_loss=float(squared_error / len(targets)), td_samples=len(targets),
                  mean_target=float(targets.mean()), active_games=self.active_games())
        buf.steps = 0

    def checkpoint(self, reason):
        began = time.monotonic()
        if self.actor_hash != v4.v2.actor_digest(self.model.base):
            raise RuntimeError('Frozen V4 actor changed during TD training')
        if self.device.type == 'cuda':
            self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated(self.device))
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved(self.device))
        journal = self.metrics.read_bytes()
        extra = dict(signature=self.signature, deadline_epoch=self.deadline, actor_sha256=self.actor_hash,
                     scorer_sha256=v4.v3.model_digest(self.model.value),
                     environments=[pack_env(e) for e in self.envs], rollout=self.buffer.save(),
                     rng=self.rng.getstate(), next_episode=self.next_episode, best=self.best,
                     next_selection=self.next_selection, last_selected_update=self.last_selected_update,
                     metrics_bytes=len(journal),
                     metrics_zlib=zlib.compress(journal, level=1), reason=reason,
                     training_seconds=self.training_seconds, max_step_seconds=self.max_step_seconds,
                     max_checkpoint_seconds=self.max_checkpoint_seconds,
                     cuda_peak_bytes=[self.peak_allocated, self.peak_reserved])
        if self.target_head is not None:
            extra['target_head'] = self.target_head.state_dict()
            extra['target_sha256'] = v4.v3.model_digest(self.target_head)
        if self.amendment_sha is not None:
            extra['cap_amendment_sha256'] = self.amendment_sha
        neural.save_checkpoint(self.last, self.model, self.optimizer, self.progress, extra)
        base.durable_json(self.root / 'active-games.json', {'games': self.active_games()})
        self.next_save = time.monotonic() + self.args.save_seconds
        elapsed = time.monotonic() - began
        self.max_checkpoint_seconds = max(self.max_checkpoint_seconds, elapsed)
        return elapsed

    def should_stop(self):
        # Leave enough time for the longest observed operation and a final atomic
        # checkpoint. The absolute wall deadline is shared by training and selection.
        if self.budget.reason is not None:
            return True
        reserve = self.max_step_seconds + self.max_checkpoint_seconds * 1.5 + .1
        remaining = min(self.deadline - time.time(), self.budget.deadline - self.budget.clock())
        if remaining <= reserve:
            self.budget.reason = self.budget.reason or 'time_budget'
        return self.budget.reason is not None

    def restore_best(self):
        if self.best:
            candidate = self.root / self.best['checkpoint']
            if base.file_hash(candidate) != self.best['sha256']:
                raise ValueError('TD selection checkpoint digest changed')
            best_path = self.root / 'best.pt'
            if not best_path.is_file() or base.file_hash(best_path) != self.best['sha256']:
                base.copy_atomic(candidate, best_path)

    def selection_due(self):
        due_epoch = self.started + self.next_selection * self.args.selection_hours * 3600
        return self.progress['updates'] > self.last_selected_update and due_epoch <= time.time() < self.deadline

    def select(self):
        slot = self.next_selection
        self.emit('selection_started', selection_slot=slot)
        self.checkpoint('selection_started')

        def stop_selection():
            if self.should_stop():
                return True
            if time.monotonic() >= self.next_save:
                self.emit('selection_heartbeat', selection_slot=slot)
                self.checkpoint('selection_in_progress')
            return False

        result = evaluation.evaluate_fresh(self.model, evaluation.DEVELOPMENT_SEEDS, policy='td',
                                           cap=self.args.selection_pieces, device=self.device,
                                           deadline=self.budget.deadline, stop_requested=stop_selection,
                                           on_finish=lambda row: self.emit('selection_game', seed=row['seed'],
                                               pieces=row['placements'], stop_reason=row['reason'], policy='td'))
        base.durable_json(self.root / f'selection-{slot}.json', result)
        if not result['complete']:
            self.emit('selection_interrupted', selection_slot=slot)
            return
        score = evaluation.development_score(result)
        promoted = self.best is None or score > (self.best['mean'], self.best['median'])
        if promoted:
            path = self.root / f'selection-{slot}.pt'
            extra = dict(actor_sha256=self.actor_hash, signature=self.signature,
                         scorer_sha256=v4.v3.model_digest(self.model.value),
                         development=result, selection_slot=slot)
            if self.amendment_sha is not None:
                extra['cap_amendment_sha256'] = self.amendment_sha
            neural.save_checkpoint(path, self.model, progress=self.progress, extra=extra)
            self.best = dict(checkpoint=path.name, sha256=base.file_hash(path), mean=score[0], median=score[1],
                             selection_slot=slot, updates=self.progress['updates'])
        self.last_selected_update = self.progress['updates']
        elapsed_slots = int((time.time() - self.started) // (self.args.selection_hours * 3600))
        self.next_selection = max(slot, elapsed_slots) + 1
        self.emit('selection_complete', best_updated=promoted, score=[score[0], result['minimum'], score[1]])
        self.checkpoint('selection_complete')
        self.restore_best()

    def run(self):
        if self.at_cap_boundary:
            self.emit('training_cap_amended', cap_amendment_sha256=self.amendment_sha,
                      previous_pieces=self.amendment['prior_signature']['pieces'], pieces=0,
                      boundary_checkpoint_sha256=self.amendment['boundary_checkpoint_sha256'],
                      boundary_session=self.amendment['boundary_session'])
        self.emit('run_start', beta=self.signature['beta'])
        self.checkpoint('run_start')
        self.at_cap_boundary = False
        while not self.should_stop():
            if self.selection_due():
                self.select()
            else:
                began = time.monotonic()
                if self.buffer.steps == self.args.rollout:
                    self.update()
                else:
                    self.step()
                elapsed = time.monotonic() - began
                self.training_seconds += elapsed
                # Cold kernel compilation can be far slower than a steady-state step.
                self.max_step_seconds = max(self.max_step_seconds * .98, elapsed, .05)
            if time.monotonic() >= self.next_save and not self.should_stop():
                self.emit('heartbeat', active_games=self.active_games())
                self.checkpoint('periodic')
        reason = self.budget.reason
        self.emit('run_finished' if reason == 'time_budget' else 'run_interrupted', stop_reason=reason)
        write_seconds = self.checkpoint(reason)
        # A signal can arrive while the final snapshot is being serialized.
        reason = self.budget.reason
        summary = dict(algorithm=ALGORITHM, complete=reason == 'time_budget', stop_reason=reason,
                       start_epoch=self.started, deadline_epoch=self.deadline, **self.progress, best=self.best,
                       source_sha256=self.signature['source_sha256'], actor_sha256=self.actor_hash,
                       last_sha256=base.file_hash(self.last), final_checkpoint_bytes=self.last.stat().st_size,
                       final_checkpoint_write_seconds=write_seconds, training_seconds=self.training_seconds,
                       training_placements_per_second=self.progress['placements']/max(self.training_seconds, 1e-9),
                       peak_cuda_allocated_bytes=self.peak_allocated, peak_cuda_reserved_bytes=self.peak_reserved)
        base.durable_json(self.root / 'selection.json', dict(complete=summary['complete'] and self.best is not None,
                          best_sha256=self.best['sha256'] if self.best else None, best=self.best,
                          seeds=list(evaluation.DEVELOPMENT_SEEDS), cap=self.args.selection_pieces))
        # Publish the completion marker after all required artifacts.
        if self.budget.reason != reason:
            summary.update(complete=False, stop_reason=self.budget.reason)
            selection_path = self.root / 'selection.json'
            selection = json.loads(selection_path.read_text())
            selection['complete'] = False
            base.durable_json(selection_path, selection)
        finished = time.time()
        elapsed = max(0., finished - self.started)
        summary.update(finish_epoch=finished, elapsed_seconds=elapsed, deadline_margin_seconds=self.deadline-finished,
                       wall_placements_per_second=self.progress['placements']/max(elapsed, 1e-9))
        base.durable_json(self.root / 'summary.json', summary)
        return summary


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, help='Initialize only the value head from a trained TD checkpoint')
    parser.add_argument('--actor-bound', type=float, help='Limit the actor logit gap the value head can overcome')
    parser.add_argument('--target-tau', type=float, help='Polyak coefficient for the bootstrap head per AdamW step')
    parser.add_argument('--clip-grad', type=float, help='Clip the trainable value head gradient norm before AdamW')
    parser.add_argument('--exclude-positions', type=Path, help='Discovery/confirmation positions.json to exclude')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path(__file__).parent/'models/trained-sigmoid.json')
    parser.add_argument('--danger-cache', type=Path)
    parser.add_argument('--prepare-danger-cache', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--hours', type=float, default=8.)
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--rollout', type=int, default=256)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=150000)
    parser.add_argument('--pieces', type=int, default=32000, help='Training placements per episode, 0 for unlimited')
    parser.add_argument('--uncap-existing-run', metavar='ORIGINAL_MANIFEST_SHA256',
                        help='Continue a gracefully interrupted capped run with --pieces 0')
    parser.add_argument('--danger-starts', type=float, default=.25)
    parser.add_argument('--danger-pool', type=int, default=32)
    parser.add_argument('--danger-window', type=int, nargs=2, default=(500, 3000))
    parser.add_argument('--danger-per-game', type=int, default=4)
    parser.add_argument('--selection-hours', type=float, default=2.)
    parser.add_argument('--selection-pieces', type=int, default=64000)
    parser.add_argument('--save-seconds', type=float, default=60.)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lam', type=float, default=.999)
    args = parser.parse_args(argv)
    args.piece_set = 'expanded'
    if not all(math.isfinite(n) and n > 0 for n in (args.envs, args.rollout, args.batch,
            args.hours, args.save_seconds, args.lr, args.selection_hours, args.selection_pieces, args.danger_pool,
            args.danger_per_game)) or args.pieces < 0 or not 0 <= args.lam <= 1 or not 0 <= args.danger_starts <= 1:
        parser.error('Invalid training ranges')
    if args.uncap_existing_run is not None:
        digest = args.uncap_existing_run
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            parser.error('--uncap-existing-run requires the lowercase SHA-256 digest of the original manifest')
        if args.pieces != 0 or args.prepare_danger_cache:
            parser.error('--uncap-existing-run requires --pieces 0 and an existing training run')
    if not 0 < args.danger_window[0] <= args.danger_window[1]:
        parser.error('Invalid danger window')
    if args.target_tau is not None and not 0 < args.target_tau <= 1:
        parser.error('--target-tau must be in (0, 1]')
    if args.actor_bound is not None and not 0 <= args.actor_bound <= 16:
        parser.error('--actor-bound must be finite and in [0, 16]')
    if args.clip_grad is not None and not (math.isfinite(args.clip_grad) and args.clip_grad > 0):
        parser.error('--clip-grad must be finite and positive')
    return args


def main(argv=None):
    args = arguments(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    game.configure_backend('native' if args.device == 'cuda' else 'numpy')
    game.load_checkpoint(args.weights)
    # Training/evaluation use the immutable library for starts. Do not write truncated
    # danger-start episodes or restarted, partial trajectories back into its index.
    base.library.configure(enabled=False)
    calibration = json.loads(args.calibration.read_text())
    if (not math.isfinite(calibration['beta']) or calibration['beta'] <= 0 or
            calibration['source_sha256'] != base.file_hash(args.source_checkpoint) or
            calibration['top_k'] != td.TOP_K):
        raise ValueError('Calibration must match the V4 checkpoint and shortlist')
    pool = danger_pool(args, calibration)
    if args.prepare_danger_cache:
        return 0
    device = torch.device(args.device)
    with v4.v2.device_lock(device), base.run_lock(args.run_dir):
        signature = provenance(args, calibration, pool)
        path = args.run_dir / 'manifest.json'
        amendment = None
        if args.uncap_existing_run and not path.is_file():
            raise ValueError('Cannot uncap a run without its original manifest')
        if path.is_file():
            manifest = json.loads(path.read_text())
            if args.uncap_existing_run:
                amendment = cap_amendment(args.run_dir, manifest, args.uncap_existing_run, signature)
            elif (args.run_dir / AMENDMENT_FILE).exists() or manifest['signature'] != signature:
                raise ValueError('TD run settings or pinned sources changed')
            summary_path = args.run_dir / 'summary.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary.get('complete'):
                    if base.file_hash(args.run_dir / 'last.pt') != summary['last_sha256']:
                        raise ValueError('Final TD checkpoint digest differs from the completed run')
                    return 0
        else:
            if any((args.run_dir / name).exists() for name in ('last.pt','best.pt','metrics.jsonl','summary.json')):
                raise ValueError('Cannot establish a new budget in a nonempty TD run')
            start = time.time()
            manifest = dict(algorithm=ALGORITHM, signature=signature, start_epoch=start,
                            deadline_epoch=start + args.hours * 3600, selection_pieces=args.selection_pieces,
                            source_sha256=signature['source_sha256'])
            base.durable_json(path, manifest)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        remaining = manifest['deadline_epoch'] - time.time()
        with base.Budget(max(0., remaining) / 3600.) as budget:
            trainer = Trainer(args, budget, pool, signature, manifest, amendment)
            summary = trainer.run()
            print(json.dumps(summary, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
