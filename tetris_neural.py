from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import heapq
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import sparse_tetris_4d_optimized as game

ROOT = Path(__file__).resolve().parent
SCHEMA = 'tetris4d-column-policy-v1'
CHANNELS, CONTEXT, FEATURES, COLUMNS = 34, 82, 96, 16
NAMES = tuple(game.PIECE_NAMES)


@dataclass(frozen=True)
class Config:
    width: int = 32
    blocks: int = 3
    shortlist: int = 128
    candidates: int = 2048
    chunk: int = 256
    protect_target: bool = True

    def __post_init__(self):
        for name in ('width', 'blocks', 'shortlist', 'candidates', 'chunk'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.width not in (8, 16, 32, 64) or not 1 <= self.blocks <= 6:
            raise ValueError('Unsupported encoder dimensions')
        if not 2 <= self.shortlist <= 2048 or not 1 <= self.candidates <= 65536 or not 1 <= self.chunk <= 2048:
            raise ValueError('Invalid candidate, shortlist or chunk size')
        if type(self.protect_target) is not bool:
            raise ValueError('protect_target must be boolean')


def device_for(name: str) -> torch.device:
    if name not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu or cuda')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; test the installed PyTorch wheel or select cpu')
    return torch.device('cuda' if name != 'cpu' and torch.cuda.is_available() else 'cpu')


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode='w', encoding='utf-8', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def fingerprint() -> str:
    payload = {name: [list(cell) for cell in game.PIECE_CELLS[name]] for name in NAMES}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def state_context(state, target: int) -> np.ndarray:
    context = game.evaluate_board(state.board)
    names = game.PIECE_SETS[state.piece_set]
    completed = state.pieces // len(names)
    current = [float(name == state.current) for name in NAMES]
    following = [float(name == state.next_piece) for name in NAMES]
    remaining = [float(name in names and state.piece_counts[name] == completed) for name in NAMES]
    masks = np.asarray(state.board, dtype=np.int32)[None, :]
    levels = np.arange(16, dtype=np.int32)[:, None]
    buried = np.mean((((masks >> levels) & 1) == 0) & ((masks >> (levels + 1)) != 0), axis=1).tolist()
    global_values = [target / 15, context.holes / 65536, (context.height_sum - context.holes) / 65536,
                     max(context.heights) / 16, (state.pieces % len(names)) / len(names)]
    result = np.asarray(current + following + remaining + [n / game.AREA for n in context.fill_counts] + buried + global_values, dtype=np.float32)
    if result.shape != (CONTEXT,):
        raise AssertionError('Context schema mismatch')
    return result


def encode_columns(board: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    masks = board.to(torch.int32)
    levels = torch.arange(16, device=masks.device, dtype=torch.int32)
    occupied = ((masks[:, None, :] >> levels[None, :, None]) & 1).bool()
    height = (occupied * (levels + 1)[None, :, None]).amax(dim=1)
    buried = ~occupied & (levels[None, :, None] < height[:, None, :])
    accessible = height <= target.to(torch.int32)[:, None]
    features = torch.cat((occupied.float(), buried.float(), height[:, None].float() / 16,
                          accessible[:, None].float()), dim=1)
    return features.reshape(-1, CHANNELS, 16, 16, 16)


class SigmoidBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.depthwise = nn.Conv3d(width, width, 3, padding=1, groups=width, bias=False)
        self.norm1 = nn.GroupNorm(8, width)
        self.mix = nn.Conv3d(width, width, 1, bias=False)
        self.norm2 = nn.GroupNorm(8, width)
        self.activation = nn.Sigmoid()
        self.scale = nn.Parameter(torch.tensor(.1))

    def forward(self, x):
        branch = self.activation(self.norm2(self.mix(self.activation(self.norm1(self.depthwise(x))))))
        return x + self.scale * (branch - .5), branch


class ColumnPolicy(nn.Module):
    def __init__(self, config: Config = Config()):
        super().__init__()
        self.config = config
        w = config.width
        self.stem = nn.Sequential(nn.Conv3d(CHANNELS, w, 1), nn.GroupNorm(8, w), nn.Sigmoid())
        self.blocks = nn.ModuleList(SigmoidBlock(w) for _ in range(config.blocks))
        self.rank1 = nn.Linear(2 * w + CONTEXT + FEATURES, 128)
        self.rank2 = nn.Linear(128, 64)
        self.rank_out = nn.Linear(64, 1)
        self.value = nn.Sequential(nn.Linear(w + CONTEXT, 64), nn.Sigmoid(), nn.Linear(64, 1))
        self.completion = nn.Sequential(nn.Linear(w + CONTEXT, 64), nn.Sigmoid(), nn.Linear(64, 16))

    def encode(self, boards, targets, context):
        spatial = self.stem(encode_columns(boards, targets))
        activation = spatial
        for block in self.blocks:
            spatial, activation = block(spatial)
        pooled = spatial.mean(dim=(2, 3, 4))
        joined = torch.cat((pooled, context), dim=1)
        return spatial, pooled, self.value(joined).squeeze(-1), self.completion(joined), activation

    def rank_chunk(self, spatial, pooled, context, features, indices, column_mask):
        flat = spatial.flatten(2).transpose(1, 2)
        rows = torch.arange(flat.shape[0], device=flat.device)[:, None, None]
        gathered = flat[rows, indices.long()]
        mask = column_mask[..., None].to(gathered.dtype)
        local = (gathered * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        n = features.shape[1]
        x = torch.cat((local, pooled[:, None].expand(-1, n, -1), context[:, None].expand(-1, n, -1), features), dim=-1)
        h1 = torch.sigmoid(self.rank1(x))
        h2 = torch.sigmoid(self.rank2(h1))
        return self.rank_out(h2).squeeze(-1), h1, h2

    def forward(self, batch, capture=False):
        spatial, pooled, value, completion, activation = self.encode(batch['boards'], batch['targets'], batch['context'])
        pieces = []
        for start in range(0, batch['features'].shape[1], self.config.chunk):
            stop = start + self.config.chunk
            logits, _, _ = self.rank_chunk(spatial, pooled, batch['context'], batch['features'][:, start:stop],
                                           batch['indices'][:, start:stop], batch['column_mask'][:, start:stop])
            pieces.append(logits)
        logits = torch.cat(pieces, dim=1).masked_fill(~batch['valid'].bool(), -torch.inf)
        result = {'logits': logits, 'value': value, 'completion': completion}
        if capture:
            selected = logits.argmax(dim=1)
            rows = torch.arange(len(selected), device=selected.device)
            _, h1, h2 = self.rank_chunk(spatial, pooled, batch['context'], batch['features'][rows, selected][:, None],
                                         batch['indices'][rows, selected][:, None], batch['column_mask'][rows, selected][:, None])
            result.update(activation=activation, hidden1=h1[:, 0], hidden2=h2[:, 0])
        return result


@dataclass
class Prepared:
    arrays: dict
    options: tuple
    teacher: int
    evaluations: int
    fallback: bool


def _option_features(board, candidate, target, context):
    shape = game.SHAPES[candidate.piece][candidate.rotation]
    base = game.column_index(candidate.x, candidate.y, candidate.w)
    touched = [base + offset for offset, _, _ in shape.columns]
    delta = sum(candidate.z + mask.bit_length() - context.heights[base + offset] for offset, mask, _ in shape.columns)
    gain, _, future = game.sustained_candidate_metrics(board, candidate, target)
    cells = np.zeros((16, 5), dtype=np.float32)
    cells[:len(shape.cells), :4] = np.asarray(shape.cells, dtype=np.float32) / 15
    cells[:len(shape.cells), 4] = 1
    fixed = [candidate.x / 15, candidate.y / 15, candidate.z / 15, candidate.w / 15]
    fixed += [v / 16 for v in shape.extents]
    size = game.PIECE_SIZES[candidate.piece]
    fixed += [gain / 16, future / 64, candidate.layers / 16, delta / 64, (delta - size) / 64,
              (candidate.score - context.score) / 64, size / 16, target / 15]
    return np.concatenate((np.asarray(fixed, dtype=np.float32), cells.ravel())), touched


def prepare_steps(state, target: int, config: Config, candidates: int | None = None):
    budget = config.candidates if candidates is None else candidates
    if type(budget) is not int or budget < 0:
        raise ValueError('Invalid candidate budget')
    status = game.target_layer_status(state.board, target)
    if config.protect_target and (status.buried or status.filled == game.AREA):
        return None
    context = game.evaluate_board(state.board)
    piece = state.current
    missing = (i for i, mask in enumerate(state.board) if not mask & (1 << target))
    vertical = tuple(r for r, shape in enumerate(game.SHAPES[piece]) if len(shape.columns) == 1) if piece == 'I' else ()
    anchored = ((r, *game.column_coordinates(i)) for i in missing for r in vertical)
    first = itertools.chain(game.specifications(piece, budget, state.pieces), anchored)
    seen, options, keys = set(), [], []
    evaluations = 0
    fallback = False
    for iterator in (first, game.specifications(piece, 0)):
        if fallback and (options or not budget):
            break
        while specs := tuple(itertools.islice(iterator, game.BATCH_SIZE)):
            batch = game.PlacementBatch(piece, specs)
            if batch.data is not None:
                scores, z, clears = batch.scores(context)
                gains, blocked, future = game.sustained_batch_metrics(context, piece, specs, z, target)
                rows = np.flatnonzero(np.isfinite(scores) & (~blocked if config.protect_target else True))
                for row in rows:
                    spec = specs[int(row)]
                    if spec in seen:
                        continue
                    seen.add(spec)
                    candidate = clears.get(int(row)) or game.Candidate(piece, *spec, int(z[row]), 0, float(scores[row]))
                    options.append(candidate)
                    keys.append((int(gains[row]) - game.FUTURE_HOLE_PENALTY * int(future[row]), candidate.score))
            else:
                for spec in specs:
                    if spec in seen:
                        continue
                    candidate = game.score_candidate(context, piece, *spec)
                    if candidate is None:
                        continue
                    gain, blocked, future = game.sustained_candidate_metrics(state.board, candidate, target)
                    if config.protect_target and blocked:
                        continue
                    seen.add(spec)
                    options.append(candidate)
                    keys.append((gain - game.FUTURE_HOLE_PENALTY * future, candidate.score))
            evaluations += len(specs)
            yield evaluations
            if fallback and options:
                break
        if options:
            break
        fallback = True
    if not options:
        return None
    best = max(range(len(options)), key=keys.__getitem__)
    if len(options) > config.shortlist:
        top = heapq.nlargest(max(1, config.shortlist // 2), range(len(options)), key=keys.__getitem__)
        used = set(top)
        rest = [i for i in range(len(options)) if i not in used]
        rng = random.Random(state.seed * 8191 + state.pieces)
        top.extend(rng.sample(rest, config.shortlist - len(top)))
        keep = sorted(top)
    else:
        keep = list(range(len(options)))
    options = tuple(options[i] for i in keep)
    teacher = keep.index(best)
    features = np.zeros((len(options), FEATURES), dtype=np.float32)
    indices = np.zeros((len(options), COLUMNS), dtype=np.int64)
    mask = np.zeros((len(options), COLUMNS), dtype=np.bool_)
    for i, candidate in enumerate(options):
        features[i], columns = _option_features(state.board, candidate, target, context)
        indices[i, :len(columns)], mask[i, :len(columns)] = columns, True
    arrays = {'boards': np.asarray(state.board, dtype=np.int32), 'targets': np.asarray(target, dtype=np.int64),
              'context': state_context(state, target), 'features': features, 'indices': indices,
              'column_mask': mask, 'valid': np.ones(len(options), dtype=np.bool_)}
    return Prepared(arrays, options, teacher, evaluations, fallback)


def prepare(*args, **kwargs):
    iterator = prepare_steps(*args, **kwargs)
    while True:
        try:
            next(iterator)
        except StopIteration as result:
            return result.value


def collate(records: list[dict], device: torch.device) -> dict:
    if not records:
        raise ValueError('Empty batch')
    maximum = max(len(row['features']) for row in records)
    result = {}
    for name in records[0]:
        values = []
        for row in records:
            value = np.asarray(row[name])
            if name in ('features', 'indices', 'column_mask', 'valid'):
                value = np.pad(value, [(0, maximum - len(value))] + [(0, 0)] * (value.ndim - 1))
            values.append(value)
        result[name] = torch.from_numpy(np.stack(values)).to(device)
    if not bool(result['valid'].any(dim=1).all()):
        raise ValueError('Every state must have at least one valid candidate')
    return result


def save_checkpoint(path, model, optimizer=None, progress=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema': SCHEMA, 'config': asdict(model.config), 'pieces_sha256': fingerprint(),
               'teacher_parameters': game.network_parameters(), 'model': model.state_dict(),
               'optimizer': optimizer.state_dict() if optimizer is not None else None,
               'progress': progress or {'updates': 0}, 'extra': extra or {},
               'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               'rng_python': random.getstate(), 'status': 'experimental-not-promoted'}
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, device, restore_rng=False):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != SCHEMA or checkpoint.get('pieces_sha256') != fingerprint():
        raise ValueError('Checkpoint schema or piece catalog does not match this engine')
    model = ColumnPolicy(Config(**checkpoint['config']))
    model.load_state_dict(checkpoint['model'], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError('Non-finite checkpoint parameters')
    model.to(device)
    if restore_rng:
        torch.set_rng_state(checkpoint['rng_cpu'])
        random.setstate(checkpoint['rng_python'])
        if device.type == 'cuda' and checkpoint['rng_cuda']:
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    return model, checkpoint


class NeuralAgent:
    def __init__(self, path, device='auto', engine=None, allow_untrained=False):
        global game
        if engine is not None:
            game = engine
        self.device = device_for(device)
        self.model, self.checkpoint = load_checkpoint(path, self.device)
        if self.checkpoint['progress'].get('updates', 0) < 1 and not allow_untrained:
            raise ValueError('Checkpoint has no training updates; train it before gameplay')
        if self.checkpoint['teacher_parameters'] != game.network_parameters():
            raise ValueError('Heuristic parameters differ from the training candidate pipeline; use the matching JSON weights')
        self.model.eval()
        self.last_trace = None
        self.last_metrics = {}
        self.metadata = {'schema': SCHEMA, 'checkpoint_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         'device': str(self.device), 'config': asdict(self.model.config),
                         'parameters': sum(p.numel() for p in self.model.parameters()),
                         'updates': self.checkpoint['progress'].get('updates', 0), 'status': self.checkpoint['status']}
        extra = self.checkpoint.get('extra', {})
        survival = extra.get('kind') == 'tetris4d-overnight-survival-v1'
        self.metadata['value_semantics'] = extra.get('value_semantics', 'expected_remaining_placements_times_0.001' if survival else 'discounted_hyperlayers')
        self.metadata['completion_semantics'] = extra.get('completion_semantics', 'imitation_auxiliary_not_recalibrated' if survival or self.checkpoint['progress'].get('mode') == 'a2c' else 'imitation_auxiliary')

    def steps(self, state, target=1, candidates=None):
        started = time.perf_counter()
        prepared = yield from prepare_steps(state, target, self.model.config, candidates)
        if prepared is None:
            return None
        before = time.perf_counter()
        batch = collate([prepared.arrays], self.device)
        with torch.inference_mode():
            prediction = self.model(batch, capture=True)
            index = int(prediction['logits'][0].argmax().item())
            if not bool(torch.isfinite(prediction['logits'][0][batch['valid'][0]]).all()):
                raise ValueError('Neural inference produced non-finite logits')
            score = float(prediction['logits'][0, index].item())
            self.last_trace = {'activation': prediction['activation'][0].float().cpu().numpy(),
                               'hidden1': prediction['hidden1'][0].float().cpu().numpy(),
                               'hidden2': prediction['hidden2'][0].float().cpu().numpy(),
                               'completion': torch.sigmoid(prediction['completion'][0]).float().cpu().numpy(),
                               'value': float(prediction['value'][0].item()), 'logit': score,
                               'piece': state.current, 'piece_number': state.pieces + 1,
                               'shortlist': len(prepared.options), 'selected': index}
        inferred = time.perf_counter()
        candidate = prepared.options[index]
        move = game.materialize_move(state.board, candidate)
        gain, _, _ = game.sustained_candidate_metrics(state.board, candidate, target)
        self.last_metrics = {'prepare_ms': (before - started) * 1000,
                             'inference_and_trace_transfer_ms': (inferred - before) * 1000,
                             'evaluations': prepared.evaluations, 'shortlist': len(prepared.options),
                             'teacher_agreement': index == prepared.teacher, **self.metadata}
        return game.Decision(move, score, prepared.evaluations, (time.perf_counter() - started) * 1000,
                             1, 1, candidates or self.model.config.candidates, True, 'neural', target, gain, prepared.fallback)

    def choose(self, *args, **kwargs):
        iterator = self.steps(*args, **kwargs)
        while True:
            try:
                next(iterator)
            except StopIteration as result:
                return result.value

    def create_view(self, parent, legacy_class, policy):
        return NeuralView(parent, self, legacy_class, policy)


class NeuralView:
    def __init__(self, parent, agent, legacy_class, policy):
        import tkinter as tk
        from tkinter import ttk
        self.agent, self.policy, self.key = agent, policy, None
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(parent)
        self.tabs.grid(row=0, column=0, sticky='nsew')
        learned, legacy = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(learned, text='Learned sigmoid policy')
        self.tabs.add(legacy, text='Legacy heuristic')
        self.legacy = legacy_class(legacy)
        learned.rowconfigure(1, weight=1)
        learned.columnconfigure(0, weight=1)
        self.channel = tk.IntVar(value=0)
        toolbar = ttk.Frame(learned)
        toolbar.grid(row=0, column=0, sticky='ew')
        ttk.Label(toolbar, text='Encoder channel:').pack(side='left')
        self.selector = ttk.Combobox(toolbar, textvariable=self.channel, values=tuple(range(agent.model.config.width)), width=4, state='readonly')
        self.selector.pack(side='left')
        self.canvas = tk.Canvas(learned, background='#101b28', highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky='nsew')
        self.detail = tk.StringVar()
        ttk.Label(learned, textvariable=self.detail, wraplength=380).grid(row=2, column=0, sticky='ew')
        self.positions, self.values, self.items = {}, None, {}
        self.last = (None, 0)
        self.selector.bind('<<ComboboxSelected>>', lambda event: self.update(*self.last, force=True))
        self.canvas.bind('<Configure>', lambda event: self.update(*self.last, force=True))
        self.canvas.bind('<Motion>', self.motion)
        self.tabs.bind('<<NotebookTabChanged>>', lambda event: self.update(*self.last, force=True))

    def motion(self, event):
        for item in self.canvas.find_overlapping(event.x, event.y, event.x, event.y):
            if item in self.items:
                name, value = self.items[item]
                self.detail.set(f'{name}: {value:.7f} (actual sigmoid activation)')

    def update(self, move, w=0, force=False):
        self.last = move, w
        if self.tabs.index(self.tabs.select()) == 1:
            self.legacy.update(move, w)
            return
        trace = self.agent.last_trace
        width, height = max(360, self.canvas.winfo_width()), max(300, self.canvas.winfo_height())
        key = (id(trace), w, self.channel.get(), width, height, self.policy())
        if key == self.key and not force:
            return
        self.key = key
        self.canvas.delete('all')
        self.items.clear()
        self.positions.clear()
        c = self.canvas
        c.create_text(12, 16, anchor='w', text='LEARNED SIGMOID ACTIVATIONS', fill='#e9f1fa')
        if trace is None:
            c.create_text(12, 44, anchor='w', text='No neural decision yet.', fill='#e9f1fa')
            return
        self.values = trace
        c.create_text(12, 40, anchor='w', text=f"Decision {trace['piece_number']} · {trace['piece']} · W={w} · channel={self.channel.get()}", fill='#a9bad0')
        cell = min((width - 30) / 16, max(4, (height - 245) / 16))
        def grid(values, columns, y, prefix):
            for index, value in enumerate(np.asarray(values).ravel()):
                x, yy = 12 + (index % columns) * cell, y + (index // columns) * cell
                color = game.blend('#172b40', '#70e5d0', float(value))
                item = c.create_rectangle(x, yy, x + cell - 1, yy + cell - 1, fill=color, outline='')
                self.items[item] = (f'{prefix}[{index}]', float(value))
                self.positions[len(self.positions)] = (x + cell / 2, yy + cell / 2)
        grid(trace['activation'][self.channel.get(), w], 16, 58, 'encoder-sigmoid')
        y = 64 + 16 * cell
        c.create_text(12, y, anchor='nw', text='Selected candidate: hidden 128 / 64', fill='#a9bad0')
        small = min(cell, (width - 30) / 32)
        cell = small
        grid(trace['hidden1'], 32, y + 20, 'rank-hidden-128')
        grid(trace['hidden2'], 32, y + 24 + 4 * cell, 'rank-hidden-64')
        grid(trace['completion'], 16, y + 28 + 7 * cell, 'layer-completion')
        label = 'ACTIVE NEURAL POLICY' if self.policy() == 'neural' else 'Stored trace; neural policy not active'
        value_units = 'remaining placements / 1000' if self.agent.metadata['value_semantics'] == 'expected_remaining_placements_times_0.001' else 'discounted clears'
        auxiliary = 'Completion: imitation auxiliary; not recalibrated.' if 'not_recalibrated' in self.agent.metadata['completion_semantics'] else 'Completion: imitation auxiliary.'
        self.detail.set(f"{label}\nLogit {trace['logit']:.5f} · value {trace['value']:.4f} ({value_units}) · {trace['shortlist']} legal candidates\n{auxiliary}\nEncoder input is the board BEFORE that decision. Hover for values.")


def _clear_mask(state, move):
    if not move.layers:
        return 0
    columns = list(state.board)
    for x, y, z, w in game.placed_cells(move):
        columns[game.column_index(x, y, w)] |= 1 << z
    return game.full_layer_mask(tuple(columns))


def commit(state, prepared, index, target):
    move = game.materialize_move(state.board, prepared.options[index])
    cleared = _clear_mask(state, move)
    state.commit(move)
    if move.layers:
        updated = game.next_hyperlayer_target(state.board, target)
        if updated is not None:
            target = updated
    return target, move.layers, cleared


def completion_labels(start: int, events: list, horizon: int):
    mapping = list(range(16))
    result = np.zeros(16, dtype=np.float32)
    for step, mask in events:
        if step < start:
            continue
        if step >= start + horizon:
            break
        for z in range(16):
            if mask & (1 << z) and mapping[z] >= 0:
                result[mapping[z]] = 1
        mapping = [old for z, old in enumerate(mapping) if not mask & (1 << z)]
        mapping += [-1] * (16 - len(mapping))
    return result


def collect(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Dataset directory is not empty; choose a new directory to avoid mixing schemas or seeds')
    output.mkdir(parents=True, exist_ok=True)
    config = Config(shortlist=args.shortlist, candidates=args.candidates, chunk=args.chunk,
                    protect_target=not args.unrestricted)
    if args.validation_games and args.games > 10000:
        raise ValueError('Training and validation seed ranges would overlap')
    manifest = {'schema': SCHEMA, 'config': asdict(config), 'pieces_sha256': fingerprint(),
                'teacher_parameters': game.network_parameters(), 'gamma': args.gamma,
                'aux_horizon': args.aux_horizon, 'shards': [], 'games': [], 'complete': False}
    for split, count, seed_start in (('train', args.games, args.seed), ('validation', args.validation_games, args.seed + 10000)):
        for episode in range(count):
            state, target = game.Game(seed_start + episode, args.piece_set), 1
            samples, events = [], []
            started = time.perf_counter()
            for step in range(args.pieces):
                prepared = prepare(state, target, config)
                if prepared is None:
                    break
                if step % args.stride == 0:
                    record = dict(prepared.arrays, labels=np.asarray(prepared.teacher, dtype=np.int64))
                    samples.append((step, record))
                target, reward, mask = commit(state, prepared, prepared.teacher, target)
                if reward:
                    events.append((step, mask))
            capped = state.pieces == args.pieces
            for start in range(0, len(samples), args.shard_size):
                records = []
                for step, record in samples[start:start + args.shard_size]:
                    record.update(returns=np.asarray(sum(args.gamma ** (event - step) * mask.bit_count()
                                                        for event, mask in events if event >= step), dtype=np.float32),
                                  value_valid=np.asarray(not capped, dtype=np.bool_),
                                  completion=completion_labels(step, events, args.aux_horizon),
                                  aux_valid=np.asarray(not capped or state.pieces - step >= args.aux_horizon, dtype=np.bool_))
                    records.append(record)
                batch = {key: value.numpy() for key, value in collate(records, torch.device('cpu')).items()}
                batch['boards'] = batch['boards'].astype(np.uint16)
                batch['indices'] = batch['indices'].astype(np.uint16)
                path = output / f'{split}-{len(manifest["shards"]):06d}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **batch)
                manifest['shards'].append({'file': path.name, 'split': split, 'seed': state.seed,
                                           'samples': len(records), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            report = dict(state.summary(), split=split, capped=capped, target=target,
                          sample_count=len(samples), clear_steps=[step + 1 for step, _ in events],
                          wall_seconds=time.perf_counter() - started)
            manifest['games'].append(report)
            atomic_json(output / 'manifest.json', manifest)
            print(json.dumps({'event': 'demonstration_game', **report}), flush=True)
    manifest['complete'] = True
    atomic_json(output / 'manifest.json', manifest)
    return manifest


class Dataset:
    def __init__(self, directory, split='train'):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['schema'] != SCHEMA or self.manifest['pieces_sha256'] != fingerprint() or not self.manifest['complete']:
            raise ValueError('Dataset is incompatible or collection was interrupted')
        self.config = Config(**self.manifest['config'])
        self.shards = [s for s in self.manifest['shards'] if s['split'] == split]
        self.samples = [(i, j) for i, shard in enumerate(self.shards) for j in range(shard['samples'])]
        self.cache = OrderedDict()
        self.digest = hashlib.sha256((self.root / 'manifest.json').read_bytes()).hexdigest()
        if not self.samples:
            raise ValueError(f'No {split} samples in dataset')

    def __len__(self):
        return len(self.samples)

    def sample_batch(self, size):
        anchor = random.randrange(len(self.samples))
        shard, row = self.samples[anchor]
        offset = anchor - row
        count = self.shards[shard]['samples']
        return [self[offset + random.randrange(count)] for _ in range(size)]

    def __getitem__(self, index):
        shard, row = self.samples[index]
        if shard not in self.cache:
            path = self.root / self.shards[shard]['file']
            if path.parent.resolve() != self.root.resolve():
                raise ValueError('Dataset path escapes its directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.shards[shard]['sha256']:
                raise ValueError(f'Dataset checksum mismatch: {path.name}')
            with np.load(path, allow_pickle=False) as values:
                self.cache[shard] = {key: values[key] for key in values.files}
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        arrays = self.cache[shard]
        return {key: values[row].astype(np.int64) if key in ('boards', 'indices') else values[row] for key, values in arrays.items()}


def supervised_loss(prediction, batch):
    policy = F.cross_entropy(prediction['logits'].float(), batch['labels'].long())
    value_mask = batch['value_valid'].float()
    value = (F.smooth_l1_loss(prediction['value'].float(), batch['returns'].float(), reduction='none') * value_mask).sum() / value_mask.sum().clamp_min(1)
    aux_mask = batch['aux_valid'].float()
    auxiliary = (F.binary_cross_entropy_with_logits(prediction['completion'].float(), batch['completion'].float(), reduction='none').mean(1) * aux_mask).sum() / aux_mask.sum().clamp_min(1)
    return policy + .25 * value + .1 * auxiliary, {'policy_loss': policy, 'value_loss': value, 'aux_loss': auxiliary,
             'agreement': (prediction['logits'].argmax(1) == batch['labels']).float().mean()}


def precision_context(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if device.type != 'cuda':
        raise ValueError('Mixed precision is only enabled for CUDA; use fp32 on CPU')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 is unavailable on this CUDA device')
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def memory_report(device):
    if device.type == 'cpu':
        return {'device': 'cpu'}
    return {'device': torch.cuda.get_device_name(device), 'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}


def validate(model, data, device, batch_size=16, limit=128):
    model.eval()
    losses, correct, seen = [], 0, 0
    with torch.inference_mode():
        for start in range(0, min(len(data), limit), batch_size):
            rows = [data[i] for i in range(start, min(start + batch_size, len(data), limit))]
            batch = collate(rows, device)
            prediction = model(batch)
            loss, metrics = supervised_loss(prediction, batch)
            losses.append(float(loss.item()) * len(rows))
            correct += int((prediction['logits'].argmax(1) == batch['labels']).sum().item())
            seen += len(rows)
    model.train()
    return {'samples': seen, 'loss': sum(losses) / seen, 'teacher_agreement': correct / seen}


def log_record(path, report):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


def train(args):
    device = device_for(args.device)
    precision_context(device, args.precision)
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise ValueError('Output checkpoint exists; use --resume or a new output path')
    data = Dataset(args.data, 'train')
    validation = Dataset(args.data, 'validation')
    if data.manifest['teacher_parameters'] != game.network_parameters():
        raise ValueError('Dataset and loaded heuristic checkpoint differ')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    signature = {'data_sha256': data.digest, 'batch': args.batch, 'accumulate': args.accumulate,
                 'precision': args.precision, 'lr': args.lr}
    if args.resume:
        model, saved = load_checkpoint(args.resume, device, restore_rng=True)
        if saved['extra'].get('kind') != 'imitation' or saved['extra'].get('signature') != signature:
            raise ValueError('Resume must retain dataset, batch, accumulation, precision and learning rate')
        progress = dict(saved['progress'])
    else:
        model = ColumnPolicy(Config(width=args.width, blocks=args.blocks, shortlist=data.config.shortlist,
                                    candidates=data.config.candidates, chunk=args.chunk, protect_target=data.config.protect_target)).to(device)
        saved = None
        progress = {'updates': 0, 'samples': 0, 'mode': 'imitation'}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['extra']['scaler'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    started = time.perf_counter()
    for update in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for _ in range(args.accumulate):
            rows = data.sample_batch(args.batch)
            batch = collate(rows, device)
            with precision_context(device, args.precision):
                prediction = model(batch)
                loss, metrics = supervised_loss(prediction, batch)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite training loss; checkpoint not overwritten')
            scaler.scale(loss / args.accumulate).backward()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0) + float(value.detach().item()) / args.accumulate
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        progress['updates'] += 1
        progress['samples'] += args.batch * args.accumulate
        if update == 0 or (update + 1) % args.save_every == 0 or update + 1 == args.updates:
            evaluation = validate(model, validation, device, min(args.batch, 16))
            extra = {'kind': 'imitation', 'signature': signature, 'scaler': scaler.state_dict(), 'validation': evaluation}
            save_checkpoint(output, model, optimizer, progress, extra)
            log_record(output.with_suffix('.jsonl'), {'event': 'imitation_update', **progress, **totals,
                       'gradient_norm': float(norm), 'validation': evaluation, 'wall_seconds': time.perf_counter() - started,
                       **memory_report(device)})
    return {'checkpoint': str(output), **progress, 'validation': evaluation, **memory_report(device)}


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    tail = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(len(rewards) - 1, -1, -1):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        tail = delta + gamma * lam * (~(terminated[step] | truncated[step])) * tail
        advantages[step] = tail
    return advantages, advantages + values


def _snapshot(state, target, actions):
    return {'seed': state.seed, 'piece_set': state.piece_set, 'target': target, 'actions': actions}


def restore_environment(snapshot):
    state, target = game.Game(snapshot['seed'], snapshot['piece_set']), 1
    for rotation, x, y, w in snapshot['actions']:
        move = game.make_move(state.board, state.current, rotation, x, y, w)
        if move is None:
            raise ValueError('Stored rollout contains an invalid action')
        state.commit(move)
        if move.layers:
            target = game.next_hyperlayer_target(state.board, target)
    if target != snapshot['target']:
        raise ValueError('Stored target does not match the replay')
    return state, target, list(snapshot['actions'])


def finetune(args):
    device = device_for(args.device)
    precision_context(device, args.precision)
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise ValueError('Output checkpoint exists; use --resume or a different output')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, saved = load_checkpoint(args.resume or args.checkpoint, device, restore_rng=bool(args.resume))
    if saved['teacher_parameters'] != game.network_parameters():
        raise ValueError('The inference candidate pipeline requires matching heuristic weights')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and args.precision == 'fp16')
    signature = {key: getattr(args, key) for key in ('envs', 'rollout', 'pieces', 'gamma', 'gae_lambda', 'entropy', 'imitation', 'batch', 'precision', 'lr', 'piece_set')}
    if args.resume:
        if saved['extra'].get('kind') != 'a2c' or saved['extra'].get('signature') != signature:
            raise ValueError('Resume requires matching A2C settings')
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['extra']['scaler'])
        environments = [restore_environment(row) for row in saved['extra']['environments']]
        next_seed = saved['extra']['next_seed']
        progress = dict(saved['progress'])
    else:
        environments = [(game.Game(args.seed + i, args.piece_set), 1, []) for i in range(args.envs)]
        next_seed = args.seed + args.envs
        progress = {'updates': saved['progress'].get('updates', 0), 'rl_updates': 0, 'env_steps': 0, 'clears': 0, 'mode': 'a2c'}
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for iteration in range(args.updates):
        model.eval()
        records, choices, teachers = [], [], []
        rewards = np.zeros((args.rollout, args.envs), dtype=np.float32)
        values, next_values = np.zeros_like(rewards), np.zeros_like(rewards)
        terminated = np.zeros(rewards.shape, dtype=np.bool_)
        truncated = np.zeros_like(terminated)
        prepared = [prepare(state, target, model.config) for state, target, _ in environments]
        for step in range(args.rollout):
            for env in range(args.envs):
                if prepared[env] is None:
                    environments[env] = (game.Game(next_seed, args.piece_set), 1, [])
                    next_seed += 1
                    prepared[env] = prepare(environments[env][0], 1, model.config)
            batch = collate([p.arrays for p in prepared], device)
            with torch.no_grad(), precision_context(device, args.precision):
                prediction = model(batch)
                probabilities = torch.softmax(prediction['logits'].float(), dim=-1)
                selected = torch.multinomial(probabilities, 1).squeeze(1).cpu().tolist()
                values[step] = prediction['value'].float().cpu().numpy()
            post_states, post_targets = [], []
            for env, index in enumerate(selected):
                state, target, actions = environments[env]
                records.append(prepared[env].arrays)
                choices.append(index)
                teachers.append(prepared[env].teacher)
                option = prepared[env].options[index]
                actions.append([option.rotation, option.x, option.y, option.w])
                target, reward, _ = commit(state, prepared[env], index, target)
                environments[env] = state, target, actions
                rewards[step, env] = reward
                progress['clears'] += reward
                progress['env_steps'] += 1
                prepared[env] = prepare(state, target, model.config)
                terminated[step, env] = prepared[env] is None
                truncated[step, env] = state.pieces >= args.pieces and not terminated[step, env]
                post_states.append(state)
                post_targets.append(target)
            with torch.no_grad(), precision_context(device, args.precision):
                boards = torch.tensor([state.board for state in post_states], dtype=torch.int32, device=device)
                targets = torch.tensor(post_targets, device=device)
                context = torch.from_numpy(np.stack([state_context(state, target) for state, target in zip(post_states, post_targets)])).to(device)
                next_values[step] = model.encode(boards, targets, context)[2].float().cpu().numpy()
            for env in range(args.envs):
                if terminated[step, env] or truncated[step, env]:
                    state, target, actions = environments[env]
                    log_record(output.with_suffix('.jsonl'), {'event': 'rl_episode', **state.summary(),
                               'terminal': bool(terminated[step, env]), 'truncated': bool(truncated[step, env])})
                    environments[env] = (game.Game(next_seed, args.piece_set), 1, [])
                    next_seed += 1
                    prepared[env] = prepare(environments[env][0], 1, model.config)
        advantage, returns = gae(rewards, values, next_values, terminated, truncated, args.gamma, args.gae_lambda)
        advantage = advantage.ravel()
        advantage = (advantage - advantage.mean()) / max(1e-6, float(advantage.std()))
        returns = returns.ravel()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, entropy_sum = 0., 0.
        for start in range(0, len(records), args.batch):
            stop = min(start + args.batch, len(records))
            batch = collate(records[start:stop], device)
            actions = torch.tensor(choices[start:stop], device=device)
            labels = torch.tensor(teachers[start:stop], device=device)
            advantages = torch.tensor(advantage[start:stop], device=device)
            targets = torch.tensor(returns[start:stop], device=device)
            with precision_context(device, args.precision):
                prediction = model(batch)
                distribution = torch.distributions.Categorical(logits=prediction['logits'].float())
                entropy = distribution.entropy().mean()
                loss = -(distribution.log_prob(actions) * advantages).mean()
                loss += .5 * F.smooth_l1_loss(prediction['value'].float(), targets)
                loss += args.imitation * F.cross_entropy(prediction['logits'].float(), labels) - args.entropy * entropy
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite RL loss')
            scaler.scale(loss * ((stop - start) / len(records))).backward()
            loss_sum += float(loss.detach()) * (stop - start) / len(records)
            entropy_sum += float(entropy.detach()) * (stop - start) / len(records)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        progress['updates'] += 1
        progress['rl_updates'] += 1
        if iteration == 0 or (iteration + 1) % args.save_every == 0 or iteration + 1 == args.updates:
            extra = {'kind': 'a2c', 'signature': signature, 'scaler': scaler.state_dict(), 'next_seed': next_seed,
                     'environments': [_snapshot(*env) for env in environments]}
            save_checkpoint(output, model, optimizer, progress, extra)
            log_record(output.with_suffix('.jsonl'), {'event': 'a2c_update', **progress, 'loss': loss_sum,
                       'entropy': entropy_sum, 'gradient_norm': float(norm), 'rollout_clears': float(rewards.sum()),
                       'wall_seconds': time.perf_counter() - started, **memory_report(device)})
    return {'checkpoint': str(output), **progress, **memory_report(device)}


def evaluate(args):
    agent = NeuralAgent(args.checkpoint, args.device)
    runs = []
    for seed in args.seeds:
        for policy in ('sustained', 'neural'):
            state, target, events = game.Game(seed, args.piece_set), 1, []
            started = time.perf_counter()
            agreements = decisions = 0
            while state.pieces < args.pieces:
                if policy == 'neural':
                    decision = agent.choose(state, target)
                else:
                    decision = game.choose_sustained_move(state.board, state.current, target,
                                                         agent.model.config.candidates, state.pieces)
                if decision is None:
                    break
                state.commit(decision.move)
                decisions += 1
                if policy == 'neural':
                    agreements += agent.last_metrics['teacher_agreement']
                if decision.move.layers:
                    events.append(state.pieces)
                    target = game.next_hyperlayer_target(state.board, target)
            seconds = time.perf_counter() - started
            report = dict(state.summary(), policy=policy, capped=state.pieces == args.pieces, clear_steps=events,
                          wall_seconds=seconds, pieces_per_second=state.pieces / seconds,
                          clears_per_second=state.layers / seconds,
                          teacher_agreement=agreements / decisions if policy == 'neural' and decisions else None,
                          stopped_reason='piece_limit' if state.pieces == args.pieces else 'top_out' if not game.has_legal_move(state.board, state.current) else 'planner_blocked')
            runs.append(report)
            print(json.dumps(report), flush=True)
    result = {'schema': SCHEMA, 'model': agent.metadata, 'runs': runs,
              'note': 'Paired finite evaluation. Neural policy ranks an explicit shortlist; sustained ranks its full sampled pool. No automatic promotion.'}
    atomic_json(Path(args.output), result)
    return result


def smoke(args):
    device = device_for(args.device)
    config = Config(width=32, blocks=3, shortlist=32, candidates=64)
    state = game.Game(7)
    prepared = prepare(state, 1, config)
    rows = []
    for _ in range(2):
        rows.append(dict(prepared.arrays, labels=np.asarray(prepared.teacher), returns=np.asarray(1., dtype=np.float32),
                         value_valid=np.asarray(True), completion=np.zeros(16, dtype=np.float32), aux_valid=np.asarray(True)))
    batch = collate(rows, device)
    model = ColumnPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    before = model.rank_out.weight.detach().clone()
    loss, metrics = supervised_loss(model(batch), batch)
    loss.backward()
    gradients = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    if not gradients or torch.equal(before, model.rank_out.weight):
        raise AssertionError('Training smoke test failed')
    result = {'passed': True, 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
              'device': str(device), 'parameter_count': sum(p.numel() for p in model.parameters()),
              'finite_gradients': gradients, 'weight_update': True, 'loss': float(loss.detach()),
              'seconds': time.perf_counter() - start, **memory_report(device),
              'note': 'Forward/backward/optimizer smoke test, not a trained playing-strength result.'}
    if args.output:
        atomic_json(Path(args.output), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Optional GPU sigmoid policy: teacher data, imitation, A2C and paired evaluation.')
    parser.add_argument('--backend', choices=('auto', 'native', 'numpy', 'python'), default='auto')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--threads', type=int, default=4)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('collect')
    demo.add_argument('--output', type=Path, default=ROOT / 'data/neural-demonstrations')
    demo.add_argument('--games', type=int, default=8)
    demo.add_argument('--validation-games', type=int, default=2)
    demo.add_argument('--pieces', type=int, default=16000)
    demo.add_argument('--seed', type=int, default=50000)
    demo.add_argument('--stride', type=int, default=8)
    demo.add_argument('--shard-size', type=int, default=128)
    demo.add_argument('--shortlist', type=int, default=128)
    demo.add_argument('--candidates', type=int, default=2048)
    demo.add_argument('--chunk', type=int, default=256)
    demo.add_argument('--gamma', type=float, default=.9995)
    demo.add_argument('--aux-horizon', type=int, default=2048)
    demo.add_argument('--unrestricted', action='store_true')
    demo.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    fit = sub.add_parser('train')
    fit.add_argument('--data', type=Path, default=ROOT / 'data/neural-demonstrations')
    fit.add_argument('--width', type=int, default=32)
    fit.add_argument('--blocks', type=int, default=3)
    fit.add_argument('--chunk', type=int, default=256)
    fit.add_argument('--accumulate', type=int, default=4)
    fit.add_argument('--seed', type=int, default=811)
    rl = sub.add_parser('finetune')
    rl.add_argument('--checkpoint', type=Path, default=ROOT / 'models/neural-policy.pt')
    rl.add_argument('--envs', type=int, default=4)
    rl.add_argument('--rollout', type=int, default=64)
    rl.add_argument('--pieces', type=int, default=32000)
    rl.add_argument('--seed', type=int, default=80000)
    rl.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    rl.add_argument('--gamma', type=float, default=.9995)
    rl.add_argument('--gae-lambda', type=float, default=.999)
    rl.add_argument('--entropy', type=float, default=.005)
    rl.add_argument('--imitation', type=float, default=.05)
    for child, filename, learning_rate in ((fit, 'neural-policy.pt', 3e-4), (rl, 'neural-rl.pt', 1e-4)):
        child.add_argument('--output', type=Path, default=ROOT / 'models' / filename)
        child.add_argument('--resume', type=Path)
        child.add_argument('--updates', type=int, default=1000)
        child.add_argument('--batch', type=int, default=16)
        child.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
        child.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='fp32')
        child.add_argument('--lr', type=float, default=learning_rate)
        child.add_argument('--save-every', type=int, default=10)
    test = sub.add_parser('evaluate')
    test.add_argument('--checkpoint', type=Path, default=ROOT / 'models/neural-policy.pt')
    test.add_argument('--seeds', nargs='+', type=int, default=[70001, 70002, 70003, 70004])
    test.add_argument('--pieces', type=int, default=16000)
    test.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    test.add_argument('--output', type=Path, default=Path('neural-evaluation.json'))
    test.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    quick = sub.add_parser('smoke')
    quick.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    quick.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    for key in ('threads', 'games', 'pieces', 'stride', 'shard_size', 'aux_horizon', 'batch', 'updates', 'accumulate', 'save_every', 'envs', 'rollout'):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    for key in ('gamma', 'gae_lambda'):
        if hasattr(args, key) and not 0 < getattr(args, key) <= 1:
            parser.error(f'{key} must be in (0, 1]')
    if hasattr(args, 'validation_games') and args.validation_games < 1:
        parser.error('At least one separate validation game is required')
    for key in ('lr', 'entropy', 'imitation'):
        if hasattr(args, key) and (not math.isfinite(getattr(args, key)) or getattr(args, key) < 0):
            parser.error(f'{key} must be finite and nonnegative')
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    try:
        result = {'collect': collect, 'train': train, 'finetune': finetune, 'evaluate': evaluate, 'smoke': smoke}[args.command](args)
        if args.command != 'collect':
            print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Neural pipeline failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
