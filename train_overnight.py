from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import struct
import sys
import tempfile
import time
import uuid

import numpy as np
import torch
from torch.nn import functional as F

try:
    import tetris_neural as neural
except ImportError as exc:
    raise SystemExit('Place train_overnight.py beside tetris_neural.py in the updated tetris4d checkout, and use its PyTorch-enabled virtual environment.') from exc

game = neural.game
ROOT = Path(__file__).resolve().parent
RUN_SCHEMA = 'tetris4d-overnight-survival-v1'
REPLAY_SCHEMA = 'tetris4d-training-game-v1'
REWARD_SCALE = 0.001


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def board_hash(board) -> str:
    return hashlib.sha256(struct.pack('<4096H', *board)).hexdigest()


def durable_json(path: Path, data) -> None:
    neural.atomic_json(Path(path), data)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=destination.parent, prefix='.copy-')
    temporary = Path(name)
    try:
        with os.fdopen(handle, 'wb') as output, source.open('rb') as stream:
            shutil.copyfileobj(stream, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.runner.lock'
    stream = path.open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError('Another trainer holds this run directory; use a different --run-dir.') from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError('Another trainer holds this run directory; use a different --run-dir.') from exc
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


class Budget:
    def __init__(self, hours: float, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + hours * 3600
        self.reason = None
        self.old_handlers = {}

    def expired(self) -> bool:
        if self.reason:
            return True
        if self.clock() >= self.deadline:
            self.reason = 'time_budget'
            return True
        return False

    def handle(self, signum, frame):
        if self.reason and self.reason.startswith('signal_'):
            raise KeyboardInterrupt
        self.reason = f'signal_{signum}'
        print('Stop requested: finishing the current safe boundary and saving progress.', file=sys.stderr, flush=True)

    def __enter__(self):
        for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
            number = getattr(signal, name, None)
            if number is not None:
                if name == 'SIGHUP' and signal.getsignal(number) == signal.SIG_IGN:
                    continue
                self.old_handlers[number] = signal.signal(number, self.handle)
        return self

    def __exit__(self, *unused):
        for number, handler in self.old_handlers.items():
            signal.signal(number, handler)


@dataclass
class Episode:
    number: int
    state: object
    target: int = 1
    moves: list = field(default_factory=list)
    phase: str = 'training'
    stop_reason: str = 'in_progress'
    complete: bool = False

    def snapshot(self):
        return {'number': self.number, 'seed': self.state.seed, 'piece_set': self.state.piece_set,
                'target': self.target, 'moves': self.moves, 'phase': self.phase,
                'board_sha256': board_hash(self.state.board), 'summary': self.state.summary()}


def apply_move(episode: Episode, candidate, update: int, behavior: str):
    state, before_target = episode.state, episode.target
    move = game.materialize_move(state.board, candidate)
    cleared = neural._clear_mask(state, move)
    state.commit(move)
    if move.layers:
        following = game.next_hyperlayer_target(state.board, episode.target)
        if following is not None:
            episode.target = following
    row = {'piece_number': state.pieces, 'piece': move.piece, 'rotation': candidate.rotation,
           'x': candidate.x, 'y': candidate.y, 'z': candidate.z, 'w': candidate.w,
           'cleared_mask': cleared, 'clears': move.layers, 'target_before': before_target,
           'target_after': episode.target, 'update': update, 'behavior': behavior,
           'reward_pieces': 1, 'board_sha256': board_hash(state.board)}
    episode.moves.append(row)
    return move


def restore_episode(snapshot) -> Episode:
    state = game.Game(int(snapshot['seed']), snapshot['piece_set'])
    episode = Episode(int(snapshot['number']), state, phase=snapshot.get('phase', 'training'))
    for row in snapshot['moves']:
        if row['piece_number'] != state.pieces + 1 or row['piece'] != state.current:
            raise ValueError('Replay piece order does not match the deterministic bag.')
        if row['target_before'] != episode.target:
            raise ValueError('Replay target history does not match.')
        move = game.make_move(state.board, state.current, row['rotation'], row['x'], row['y'], row['w'])
        if move is None or move.candidate.z != row['z']:
            raise ValueError('Replay contains an invalid rigid drop.')
        apply_move(episode, move.candidate, row.get('update', 0), row.get('behavior', 'unknown'))
        actual = episode.moves[-1]
        for name in ('clears', 'cleared_mask', 'target_after', 'reward_pieces', 'board_sha256'):
            if actual[name] != row[name]:
                raise ValueError(f'Replay mismatch at piece {state.pieces}: {name}')
    if episode.target != snapshot['target'] or board_hash(state.board) != snapshot['board_sha256']:
        raise ValueError('Replay final state does not match.')
    if snapshot.get('summary') != state.summary():
        raise ValueError('Replay final counters do not match.')
    return episode


class Recorder:
    def __init__(self, root: Path, enabled: bool, every: int = 1, maximum_mib: float = 1024):
        self.root = Path(root)
        self.enabled = enabled
        self.every = every
        self.maximum = int(maximum_mib * 2**20)
        self.attempt = uuid.uuid4().hex[:12]
        self.used = 0
        self.sizes = {}
        self.files = 0
        self.error = None
        self.context = {}
        try:
            self.used = sum(p.stat().st_size for p in self.root.rglob('*.json')) if self.root.exists() else 0
        except OSError as exc:
            self.disable(f'Recording directory unavailable: {type(exc).__name__}; checkpoints remain independent.')
        self.engine_hash = file_hash(Path(game.__file__))

    def write(self, episode: Episode, reason=None):
        if not self.enabled or episode.number % self.every:
            return None
        snapshot = episode.snapshot()
        payload = {'schema': REPLAY_SCHEMA, 'run_context': self.context, 'pieces_sha256': neural.fingerprint(),
                   'engine_sha256': self.engine_hash, 'teacher_parameters': game.network_parameters(),
                   'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'attempt': self.attempt, 'complete': episode.complete,
                   'stop_reason': reason or episode.stop_reason,
                   'terminated': episode.complete and episode.stop_reason in ('top_out', 'planner_blocked'),
                   'truncated': episode.complete and episode.stop_reason == 'piece_limit',
                   'training_reward': 'one per successful placement; critic targets scaled by 0.001',
                   **snapshot}
        path = self.root / self.attempt / f'{episode.phase}-{episode.number:08d}-seed-{episode.state.seed}.json'
        text = json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n'
        size = len(text.encode('utf-8'))
        prior = self.sizes.get(path, 0)
        if self.used - prior + size > self.maximum:
            self.disable('Recording quota reached; training and checkpoint saving continue.')
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(dir=path.parent, prefix='.record-')
            temporary = Path(name)
            try:
                with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                    stream.write(text)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            self.used += size - prior
            self.sizes[path] = size
            self.files += int(prior == 0)
            return path
        except OSError as exc:
            self.disable(f'Recording disabled after {type(exc).__name__}; checkpoints remain independent.')
            return None

    def disable(self, message):
        self.enabled = False
        self.error = message
        print(message, file=sys.stderr, flush=True)


def verify_game(path: Path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema') != REPLAY_SCHEMA or data.get('pieces_sha256') != neural.fingerprint():
        raise ValueError('Unsupported recording schema or different piece catalog.')
    old = game.network_parameters()
    try:
        game.configure_network(data['teacher_parameters'])
        episode = restore_episode(data)
        reason = data['stop_reason']
        expected_terminal = data['complete'] and reason in ('top_out', 'planner_blocked')
        expected_truncated = data['complete'] and reason == 'piece_limit'
        if data['terminated'] != expected_terminal or data['truncated'] != expected_truncated:
            raise ValueError('Inconsistent termination/truncation flags.')
        if data['complete'] and reason not in ('top_out', 'planner_blocked', 'piece_limit'):
            raise ValueError('Unknown completed-game stop reason.')
        if data['complete'] and reason == 'top_out' and game.has_legal_move(episode.state.board, episode.state.current):
            raise ValueError('Recording incorrectly claims a physical top-out.')
        if data['complete'] and reason == 'planner_blocked' and not game.has_legal_move(episode.state.board, episode.state.current):
            raise ValueError('Physical top-out incorrectly labelled as planner blockage.')
        return {'passed': True, 'path': str(path), 'pieces': episode.state.pieces,
                'hyperlayers': episode.state.layers, 'complete': data['complete'], 'stop_reason': reason,
                'engine_source_matches': data['engine_sha256'] == file_hash(Path(game.__file__))}
    finally:
        game.configure_network(old)


def survival_score(runs):
    if not runs or any(not row['evaluation_complete'] for row in runs):
        return None
    lengths = [row['pieces'] for row in runs]
    return (statistics.fmean(lengths), min(lengths), statistics.median(lengths))


def stopped_reason(state):
    return 'top_out' if not game.has_legal_move(state.board, state.current) else 'planner_blocked'


@contextmanager
def preserve_rng():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    python_state = random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        random.setstate(python_state)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


class Trainer:
    def __init__(self, args, budget: Budget):
        self.args, self.budget = args, budget
        self.root = args.run_dir.resolve()
        self.device = neural.device_for(args.device)
        neural.precision_context(self.device, args.precision)
        self.last = self.root / 'last.pt'
        self.best = self.root / 'best.pt'
        self.best_info = self.root / 'best.json'
        self.baseline_path = self.root / 'sustained-baseline.json'
        self.metrics = self.root / 'metrics.jsonl'
        self.session = uuid.uuid4().hex[:12]
        self.recorder = Recorder(self.root / 'games', args.record_games, args.record_every, args.record_max_mib)
        self.pending = []
        self.prepared = None
        self.saved_at = self.evaluated_at = time.monotonic()
        self.elapsed_before = 0.0
        self.last_evaluated_update = None
        self.stop_at_update = None
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        self.signature = {key: getattr(args, key) for key in ('envs', 'rollout', 'pieces', 'seed', 'batch',
                         'precision', 'lr', 'entropy', 'imitation', 'gae_lambda', 'piece_set',
                         'warmup_updates', 'selection_pieces', 'selection_seeds')}
        self.signature.update(objective='undiscounted_successful_placements', gamma=1.0,
                              reward_scale=REWARD_SCALE, neural_sha256=file_hash(Path(neural.__file__)),
                              engine_sha256=file_hash(Path(game.__file__)))
        self.baseline = None
        self.best_record = None
        saved = None
        if self.last.exists():
            self.model, saved = neural.load_checkpoint(self.last, self.device, restore_rng=True)
            extra = saved.get('extra', {})
            self.signature['model_config'] = asdict(self.model.config)
            if extra.get('kind') != RUN_SCHEMA or extra.get('signature') != self.signature:
                raise ValueError('Run settings or source changed. Use the original settings or a new --run-dir.')
            if saved['teacher_parameters'] != game.network_parameters():
                raise ValueError('The heuristic checkpoint changed since this run.')
            self.progress = dict(saved['progress'])
            self.next_episode = extra['next_episode']
            self.environments = [restore_episode(s) for s in extra['environments']]
            self.elapsed_before = extra.get('elapsed_seconds', 0.0)
            self.last_evaluated_update = extra.get('last_evaluated_update')
            self.initial_checkpoint = extra.get('initial_checkpoint')
        else:
            initial = args.checkpoint
            if initial is None:
                candidate = ROOT / 'models/neural-policy.pt'
                initial = candidate if candidate.is_file() else None
            self.initial_checkpoint = str(initial.resolve()) if initial else None
            inherited = 0
            if initial:
                self.model, initial_saved = neural.load_checkpoint(initial, self.device)
                if initial_saved['teacher_parameters'] != game.network_parameters():
                    raise ValueError('Initial policy and current candidate-pipeline weights differ.')
                inherited = int(initial_saved['progress'].get('updates', 0))
            else:
                self.model = neural.ColumnPolicy().to(self.device)
            self.model.value[-1].weight.data.zero_()
            self.model.value[-1].bias.data.zero_()
            self.progress = {'updates': inherited, 'runner_updates': 0, 'warmup_updates': 0,
                             'survival_updates': 0, 'placements': 0, 'clears': 0,
                             'finished_episodes': 0, 'mode': 'overnight-survival'}
            self.next_episode = 0
            self.environments = [self.new_episode() for _ in range(args.envs)]
        self.signature['model_config'] = asdict(self.model.config)
        self.recorder.context = {'training_session': self.session, 'model_config': asdict(self.model.config),
                                 'objective': self.signature['objective'],
                                 'neural_sha256': self.signature['neural_sha256']}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda' and args.precision == 'fp16')
        if saved is not None:
            self.optimizer.load_state_dict(saved['optimizer'])
            self.scaler.load_state_dict(saved['extra']['scaler'])
        if args.max_updates:
            self.stop_at_update = self.progress['runner_updates'] + args.max_updates
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        if self.best_info.exists():
            info = json.loads(self.best_info.read_text(encoding='utf-8'))
            if info.get('signature') == self.signature and self.best.is_file() and info.get('checkpoint_sha256') == file_hash(self.best):
                self.best_record = info
            else:
                raise ValueError('Best-checkpoint metadata is inconsistent; preserve it and use a new run directory.')
        if self.baseline_path.exists():
            value = json.loads(self.baseline_path.read_text(encoding='utf-8'))
            if value.get('signature') == self.signature and value.get('complete'):
                self.baseline = value
        self.emit('run_start', resumed=saved is not None, initial_checkpoint=self.initial_checkpoint,
                  device=str(self.device), torch=str(torch.__version__), cuda_runtime=torch.version.cuda, model_parameters=sum(p.numel() for p in self.model.parameters()),
                  config=asdict(self.model.config), objective=self.signature['objective'],
                  hours=args.hours, recording=args.record_games)

    def new_episode(self):
        number = self.next_episode
        self.next_episode += 1
        return Episode(number, game.Game(2 * (self.args.seed + number), self.args.piece_set))

    def emit(self, event, **fields):
        row = {'event': event, 'session': self.session, 'process_id': os.getpid(),
               'elapsed_seconds': time.monotonic() - self.budget.started, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields}
        with self.metrics.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps(row, allow_nan=False), flush=True)

    def _prepare_all(self):
        if self.prepared is None:
            self.prepared = [neural.prepare(e.state, e.target, self.model.config) for e in self.environments]

    def finish_episode(self, index, reason):
        episode = self.environments[index]
        episode.complete, episode.stop_reason = True, reason
        self.pending.append(episode)
        self.progress['finished_episodes'] += 1
        self.emit('episode_end', episode=episode.number, stop_reason=reason,
                  terminated=reason in ('top_out', 'planner_blocked'), truncated=reason == 'piece_limit',
                  **episode.state.summary())
        self.environments[index] = self.new_episode()
        self.prepared[index] = neural.prepare(self.environments[index].state, 1, self.model.config)
        if self.prepared[index] is None:
            raise ValueError('No candidate on an empty board; refusing an endless reset loop.')

    def rollout(self, warmup):
        self._prepare_all()
        records, choices, labels = [], [], []
        rewards, values, next_values, terminals, truncations = [], [], [], [], []
        self.model.eval()
        for step in range(self.args.rollout):
            if self.budget.expired():
                break
            for i, prepared in enumerate(self.prepared):
                if prepared is None:
                    self.finish_episode(i, stopped_reason(self.environments[i].state))
            batch = neural.collate([p.arrays for p in self.prepared], self.device)
            with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                prediction = self.model(batch)
                logits = prediction['logits'].float()
                if not bool(torch.isfinite(logits[batch['valid'].bool()]).all()):
                    raise ValueError('Non-finite action logits; latest saved checkpoint is preserved.')
                selected = ([p.teacher for p in self.prepared] if warmup else
                            torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1).cpu().tolist())
                values.append(prediction['value'].float().cpu().numpy())
            terminal = np.zeros(self.args.envs, dtype=np.bool_)
            truncated = np.zeros(self.args.envs, dtype=np.bool_)
            for i, selected_index in enumerate(selected):
                episode, prepared = self.environments[i], self.prepared[i]
                records.append(prepared.arrays)
                choices.append(selected_index)
                labels.append(prepared.teacher)
                move = apply_move(episode, prepared.options[selected_index], self.progress['updates'],
                                  'teacher' if warmup else 'stochastic_neural')
                self.progress['placements'] += 1
                self.progress['clears'] += move.layers
                self.prepared[i] = neural.prepare(episode.state, episode.target, self.model.config)
                terminal[i] = self.prepared[i] is None
                truncated[i] = episode.state.pieces >= self.args.pieces and not terminal[i]
            rewards.append(np.full(self.args.envs, REWARD_SCALE, dtype=np.float32))
            with torch.no_grad(), neural.precision_context(self.device, self.args.precision):
                boards = torch.tensor([e.state.board for e in self.environments], dtype=torch.int32, device=self.device)
                targets = torch.tensor([e.target for e in self.environments], device=self.device)
                context = torch.from_numpy(np.stack([neural.state_context(e.state, e.target) for e in self.environments])).to(self.device)
                next_values.append(self.model.encode(boards, targets, context)[2].float().cpu().numpy())
            terminals.append(terminal)
            truncations.append(truncated)
            for i in range(self.args.envs):
                if terminal[i] or truncated[i]:
                    self.finish_episode(i, 'piece_limit' if truncated[i] else stopped_reason(self.environments[i].state))
        if not records:
            return None
        advantages, returns = neural.gae(np.asarray(rewards), np.asarray(values), np.asarray(next_values),
                                        np.asarray(terminals), np.asarray(truncations), 1.0, self.args.gae_lambda)
        advantages = advantages.ravel()
        advantages = (advantages - advantages.mean()) / max(1e-6, float(advantages.std()))
        return records, choices, labels, advantages, returns.ravel()

    def update(self, rollout, warmup):
        records, choices, labels, advantages, returns = rollout
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = total_entropy = agreement = 0.0
        for start in range(0, len(records), self.args.batch):
            stop = min(start + self.args.batch, len(records))
            batch = neural.collate(records[start:stop], self.device)
            target_actions = torch.tensor(labels[start:stop], device=self.device)
            with neural.precision_context(self.device, self.args.precision):
                prediction = self.model(batch)
                logits = prediction['logits'].float()
                distribution = torch.distributions.Categorical(logits=logits)
                entropy = distribution.entropy().mean()
                imitation = F.cross_entropy(logits, target_actions)
                critic = F.smooth_l1_loss(prediction['value'].float(), torch.tensor(returns[start:stop], device=self.device))
                if warmup:
                    loss = imitation + .5 * critic
                else:
                    selected = torch.tensor(choices[start:stop], device=self.device)
                    advantage = torch.tensor(advantages[start:stop], device=self.device)
                    loss = -(distribution.log_prob(selected) * advantage).mean() + .5 * critic
                    loss += self.args.imitation * imitation - self.args.entropy * entropy
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite loss; last checkpoint will not be overwritten.')
            fraction = (stop - start) / len(records)
            self.scaler.scale(loss * fraction).backward()
            total_loss += float(loss.detach()) * fraction
            total_entropy += float(entropy.detach()) * fraction
            agreement += float((logits.argmax(-1) == target_actions).float().mean()) * fraction
        self.scaler.unscale_(self.optimizer)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.progress['updates'] += 1
        self.progress['runner_updates'] += 1
        self.progress['warmup_updates' if warmup else 'survival_updates'] += 1
        self.emit('update', **self.progress, stage='warmup' if warmup else 'survival',
                  loss=total_loss, entropy=total_entropy, teacher_agreement=agreement,
                  gradient_norm=float(norm), rollout_placements=len(records),
                  **neural.memory_report(self.device))

    def save(self, reason='checkpoint'):
        if not all(bool(torch.isfinite(value).all()) for value in self.model.state_dict().values()):
            raise ValueError('Non-finite weights: refusing to overwrite a valid checkpoint.')
        extra = {'kind': RUN_SCHEMA, 'signature': self.signature, 'scaler': self.scaler.state_dict(),
                 'next_episode': self.next_episode, 'environments': [e.snapshot() for e in self.environments],
                 'initial_checkpoint': self.initial_checkpoint, 'last_evaluated_update': self.last_evaluated_update,
                 'elapsed_seconds': self.elapsed_before + time.monotonic() - self.budget.started,
                 'objective': 'maximize successful placements before loss; gamma=1; reward scaled by 0.001'}
        temporary = self.root / '.next-checkpoint.pt'
        neural.save_checkpoint(temporary, self.model, self.optimizer, self.progress, extra)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        if self.last.exists():
            copy_atomic(self.last, self.root / 'previous.pt')
        os.replace(temporary, self.last)
        for episode in self.pending:
            self.recorder.write(episode)
        self.pending.clear()
        for episode in self.environments:
            self.recorder.write(episode, reason)
        self.saved_at = time.monotonic()
        summary = {'schema': RUN_SCHEMA, 'session': self.session, 'reason': reason, **self.progress,
                   'last_checkpoint': str(self.last), 'best_checkpoint': str(self.best) if self.best_record else None,
                   'best_validation_score': self.best_record['score'] if self.best_record else None,
                   'recording_enabled': self.recorder.enabled, 'recording_error': self.recorder.error,
                   'recording_total_mib': self.recorder.used / 2**20,
                   'active_games': [{'episode': e.number, **e.state.summary()} for e in self.environments],
                   'elapsed_training_seconds': extra['elapsed_seconds'],
                   'note': 'A capped game is not proof of indefinite survival. Best is selected on reused validation seeds, not an independent test.'}
        durable_json(self.root / 'summary.json', summary)
        self.emit('checkpoint_saved', reason=reason, runner_updates=self.progress['runner_updates'], path=str(self.last))
        return summary

    def evaluate_policy(self, policy):
        runs = []
        self.model.eval()
        with preserve_rng(), torch.inference_mode():
            for index, seed in enumerate(self.args.selection_seeds):
                if self.budget.expired():
                    break
                number = self.progress['runner_updates'] * len(self.args.selection_seeds) + index
                episode = Episode(number, game.Game(seed, self.args.piece_set), phase=f'selection-{policy}')
                started = time.monotonic()
                reason = 'piece_limit'
                while episode.state.pieces < self.args.selection_pieces:
                    if self.budget.expired():
                        reason = self.budget.reason
                        break
                    if policy == 'sustained':
                        decision = game.choose_sustained_move(episode.state.board, episode.state.current,
                                                             episode.target, self.model.config.candidates, episode.state.pieces)
                        candidate = decision.move.candidate if decision else None
                    else:
                        prepared = neural.prepare(episode.state, episode.target, self.model.config)
                        candidate = None
                        if prepared is not None:
                            batch = neural.collate([prepared.arrays], self.device)
                            prediction = self.model(batch)
                            if not bool(torch.isfinite(prediction['logits'][batch['valid'].bool()]).all()):
                                raise ValueError('Non-finite evaluation logits.')
                            selected = int(prediction['logits'][0].argmax().item())
                            candidate = prepared.options[selected]
                    if candidate is None:
                        reason = stopped_reason(episode.state)
                        break
                    apply_move(episode, candidate, self.progress['updates'], f'greedy_{policy}')
                episode.stop_reason = reason
                episode.complete = reason in ('top_out', 'planner_blocked', 'piece_limit')
                self.recorder.write(episode)
                report = {**episode.state.summary(), 'stop_reason': reason, 'evaluation_complete': episode.complete,
                          'capped': reason == 'piece_limit', 'policy': policy,
                          'wall_seconds': time.monotonic() - started}
                runs.append(report)
                self.emit('selection_game', **report)
                if not episode.complete:
                    break
        complete = len(runs) == len(self.args.selection_seeds) and all(r['evaluation_complete'] for r in runs)
        score = survival_score(runs) if complete else None
        return {'complete': complete, 'runs': runs, 'score': score, 'signature': self.signature,
                'score_definition': 'mean capped survival pieces, then minimum, then median; no clear bonus'}

    def select(self):
        if self.progress['updates'] < 1 or self.budget.expired():
            return
        if self.baseline is None:
            baseline = self.evaluate_policy('sustained')
            if not baseline['complete']:
                durable_json(self.root / 'selection-interrupted.json', baseline)
                return
            self.baseline = baseline
            durable_json(self.baseline_path, baseline)
        if self.budget.expired():
            return
        result = self.evaluate_policy('neural')
        result.update(updates=self.progress['updates'], runner_updates=self.progress['runner_updates'],
                      checkpoint_sha256=file_hash(self.last), selection_only=True)
        durable_json(self.root / f'selection-{self.progress["runner_updates"]:08d}.json', result)
        if not result['complete']:
            return
        self.last_evaluated_update = self.progress['runner_updates']
        score = tuple(result['score'])
        improved = self.best_record is None or score > tuple(self.best_record['score'])
        if improved:
            copy_atomic(self.last, self.best)
            result['beats_sustained_on_selection'] = score > tuple(self.baseline['score'])
            durable_json(self.best_info, result)
            self.best_record = result
        self.emit('selection_complete', score=score, best_updated=improved,
                  sustained_score=self.baseline['score'], beats_sustained=score > tuple(self.baseline['score']))
        self.evaluated_at = time.monotonic()
        durable_json(self.root / 'recommended-policy.json', {
            'policy': 'neural' if self.best_record['beats_sustained_on_selection'] else 'sustained',
            'checkpoint': str(self.best) if self.best_record['beats_sustained_on_selection'] else None,
            'selection_seeds': self.args.selection_seeds, 'selection_piece_cap': self.args.selection_pieces,
            'note': 'Selection evidence only; no automatic GUI-default change and no independent held-out claim.'})

    def run(self):
        self.save('started')
        if self.progress['runner_updates'] == 0 and self.progress['updates'] > 0 and self.best_record is None:
            self.select()
        last_details = None
        while not self.budget.expired():
            if self.stop_at_update is not None and self.progress['runner_updates'] >= self.stop_at_update:
                self.budget.reason = 'update_limit'
                break
            warmup = self.progress['warmup_updates'] < self.args.warmup_updates
            rollout = self.rollout(warmup)
            if rollout is None:
                break
            self.update(rollout, warmup)
            due = self.progress['runner_updates'] % self.args.save_every == 0 or time.monotonic() - self.saved_at >= self.args.save_seconds
            ready = self.progress['warmup_updates'] >= self.args.warmup_updates
            selection_due = ready and self.last_evaluated_update != self.progress['runner_updates'] and (
                self.best_record is None or time.monotonic() - self.evaluated_at >= self.args.eval_minutes * 60)
            if due or selection_due or self.budget.expired():
                last_details = self.save()
            if selection_due and not self.budget.expired():
                self.select()
        last_details = self.save(self.budget.reason or 'finished')
        self.emit('run_finished', **{k: v for k, v in last_details.items() if k not in ('session', 'schema')})
        return last_details


def arguments(argv=None):
    parser = argparse.ArgumentParser(description='One-command resumable survival training with optional replayable JSON games.')
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/overnight-survival')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--backend', choices=('auto', 'native', 'numpy', 'python'), default='auto')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--rollout', type=int, default=64)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--pieces', type=int, default=32000)
    parser.add_argument('--seed', type=int, default=150000)
    parser.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    parser.add_argument('--warmup-updates', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--entropy', type=float, default=.001)
    parser.add_argument('--imitation', type=float, default=.05)
    parser.add_argument('--gae-lambda', type=float, default=.999)
    parser.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='fp32')
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--save-seconds', type=float, default=60)
    parser.add_argument('--eval-minutes', type=float, default=60)
    parser.add_argument('--selection-seeds', nargs='+', type=int, default=[910001, 910003])
    parser.add_argument('--selection-pieces', type=int, default=32000)
    parser.add_argument('--max-updates', type=int, default=0)
    parser.add_argument('--record-games', action='store_true')
    parser.add_argument('--record-every', type=int, default=1)
    parser.add_argument('--record-max-mib', type=float, default=1024)
    parser.add_argument('--verify-game', type=Path)
    args = parser.parse_args(argv)
    for key in ('hours', 'threads', 'envs', 'rollout', 'batch', 'pieces', 'lr', 'save_every', 'save_seconds',
                'eval_minutes', 'selection_pieces', 'record_every', 'record_max_mib'):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{key} must be finite and positive')
    if args.warmup_updates < 0 or args.max_updates < 0:
        parser.error('warmup-updates and max-updates must be nonnegative')
    if not 0 < args.gae_lambda <= 1 or not math.isfinite(args.gae_lambda):
        parser.error('gae-lambda must be in (0, 1]')
    if any(not math.isfinite(v) or v < 0 for v in (args.entropy, args.imitation)):
        parser.error('entropy and imitation must be finite and nonnegative')
    if len(set(args.selection_seeds)) != len(args.selection_seeds) or any(s % 2 == 0 for s in args.selection_seeds):
        parser.error('Selection seeds must be unique odd integers; training seeds are even.')
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    if args.verify_game:
        print(json.dumps(verify_game(args.verify_game), indent=2))
        return 0
    try:
        with run_lock(args.run_dir), Budget(args.hours) as budget:
            trainer = Trainer(args, budget)
            trainer.run()
        return 0
    except KeyboardInterrupt:
        print('Forced interruption. The last completed checkpoint remains available.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Overnight training stopped: {exc}\nNo existing best policy was automatically promoted or deleted.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
