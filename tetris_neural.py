from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import lru_cache
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
# Per-candidate arrays (padded along the candidate axis when batched).
CANDIDATE_FIELDS = frozenset(('features', 'indices', 'column_mask', 'valid', 'teacher_prior', 'spatial_delta',
                              'next_features', 'td_features', 'td_actor_logits'))
# Architectures whose preparation includes V4 afterstate spatial descriptors.
AFTERSTATE_TD_ARCHITECTURES = ('survival-afterstate-td', 'survival-actor-regularized-td')
SPATIAL_ARCHITECTURES = ('survival-spatial-v4', 'survival-afterstate-v5', *AFTERSTATE_TD_ARCHITECTURES)
FEATURE_ARCHITECTURES = ('survival-feature-v3', *SPATIAL_ARCHITECTURES)
NAMES = tuple(game.PIECE_NAMES)


@dataclass(frozen=True)
class Config:
    width: int = 32
    blocks: int = 3
    shortlist: int = 128
    candidates: int = 2048
    chunk: int = 256
    protect_target: bool = True
    recover_target: bool = False
    policy_mode: str = 'neural'
    residual_scale: float = 1.0
    detach_critic: bool = False
    centered_critic: bool = False
    teacher_prior_scale: float = 1.0
    architecture: str = 'column-sigmoid-v1'
    spatial_actor: bool = False
    shortlist_mode: str = 'teacher-random'
    # 'sampled' screens `candidates` placements spread over all orientations;
    # 'exhaustive' screens every legal placement (all orientations x positions).
    screening: str = 'sampled'
    # Survival V4 only: append fixed per-region height/hole summaries to the critic input.
    regional_critic: bool = False
    # Survival V5 only: logits = (base + residual_bound * tanh(afterstate_scale * centred score)) / policy_temperature.
    residual_bound: float = 0.0
    policy_temperature: float = 1.0
    afterstate_scale: float = 1.0

    @property
    def screening_budget(self) -> int:
        """Placement budget passed to game.specifications (0 = every placement)."""
        return 0 if self.screening == 'exhaustive' else self.candidates

    def __post_init__(self):
        if self.architecture not in ('column-sigmoid-v1', 'survival-feature-v3', *SPATIAL_ARCHITECTURES):
            raise ValueError('Unsupported policy architecture')
        for name in ('width', 'blocks', 'shortlist', 'candidates', 'chunk'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.width not in (8, 16, 32, 64) or not 1 <= self.blocks <= 6:
            raise ValueError('Unsupported encoder dimensions')
        if not 2 <= self.shortlist <= 2048 or not 1 <= self.candidates <= 65536 or not 1 <= self.chunk <= 2048:
            raise ValueError('Invalid candidate, shortlist or chunk size')
        if any(type(getattr(self, n)) is not bool for n in ('protect_target', 'recover_target', 'detach_critic', 'centered_critic', 'spatial_actor', 'regional_critic')):
            raise ValueError('Target and critic settings must be boolean')
        if self.shortlist_mode not in ('teacher-random', 'diverse', 'teacher-top'):
            raise ValueError('Unsupported candidate shortlist mode')
        if self.screening not in ('sampled', 'exhaustive'):
            raise ValueError('Unsupported candidate screening mode')
        for name, low, high in (('residual_bound', 0., 16.), ('policy_temperature', 1e-3, 16.), ('afterstate_scale', 1e-6, 1e4)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be in [{low}, {high}]')
        if self.policy_mode not in ('neural', 'teacher-residual', 'teacher-prior'):
            raise ValueError('Unsupported policy mode')
        if not math.isfinite(self.residual_scale) or not 0 <= self.residual_scale <= 4:
            raise ValueError('residual_scale must be between 0 and 4')
        if not math.isfinite(self.teacher_prior_scale) or not 0 <= self.teacher_prior_scale <= 4:
            raise ValueError('teacher_prior_scale must be between 0 and 4')


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


def state_context(state, target: int, context=None) -> np.ndarray:
    if context is None:
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


def encode_columns(board: torch.Tensor, target: torch.Tensor, *, fused: bool = False) -> torch.Tensor:
    if (fused and board.is_cuda and board.dtype == torch.int32 and board.is_contiguous()
            and target.device == board.device and target.dtype in (torch.int32, torch.int64)
            and target.is_contiguous()):
        from tetris_gpu_encoding import encode_columns_cuda
        return encode_columns_cuda(board, target)
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
        self.fused_encoding = False
        w = config.width
        self.stem = nn.Sequential(nn.Conv3d(CHANNELS, w, 1), nn.GroupNorm(8, w), nn.Sigmoid())
        self.blocks = nn.ModuleList(SigmoidBlock(w) for _ in range(config.blocks))
        self.rank1 = nn.Linear(2 * w + CONTEXT + FEATURES, 128)
        self.rank2 = nn.Linear(128, 64)
        self.rank_out = nn.Linear(64, 1)
        self.value = nn.Sequential(nn.Linear(w + CONTEXT, 64), nn.Sigmoid(), nn.Linear(64, 1))
        self.completion = nn.Sequential(nn.Linear(w + CONTEXT, 64), nn.Sigmoid(), nn.Linear(64, 16))

    def encode(self, boards, targets, context):
        spatial = self.stem(encode_columns(boards, targets, fused=self.fused_encoding))
        activation = spatial
        for block in self.blocks:
            spatial, activation = block(spatial)
        pooled = spatial.mean(dim=(2, 3, 4))
        joined = torch.cat((pooled, context), dim=1)
        value_input = joined.detach() if self.config.detach_critic else joined
        return spatial, pooled, self.critic(value_input), self.completion(joined), activation

    def critic(self, joined):
        if self.config.centered_critic:
            hidden = torch.sigmoid(self.value[0](joined)) - .5
            return self.value[2](hidden).squeeze(-1)
        return self.value(joined).squeeze(-1)

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
        logits = torch.cat(pieces, dim=1)
        if self.config.policy_mode in ('teacher-residual', 'teacher-prior'):
            if 'teacher_prior' not in batch:
                raise ValueError('Residual policy requires exact teacher ranking features')
            if self.config.policy_mode == 'teacher-residual':
                logits = batch['teacher_prior'].float() + self.config.residual_scale * (2 * torch.sigmoid(logits.float()) - 1)
            else:
                logits = self.config.teacher_prior_scale * batch['teacher_prior'].float() + self.config.residual_scale * logits.float()
        logits = logits.masked_fill(~batch['valid'].bool(), -torch.inf)
        result = {'logits': logits, 'value': value, 'completion': completion,
                  'critic_input': torch.cat((pooled, batch['context']), dim=1).detach()}
        if capture:
            selected = logits.argmax(dim=1)
            rows = torch.arange(len(selected), device=selected.device)
            _, h1, h2 = self.rank_chunk(spatial, pooled, batch['context'], batch['features'][rows, selected][:, None],
                                         batch['indices'][rows, selected][:, None], batch['column_mask'][rows, selected][:, None])
            result.update(activation=activation, hidden1=h1[:, 0], hidden2=h2[:, 0])
        return result

    @torch.no_grad()
    def ranking_diagnostics(self, batch):
        """Summarize one bounded batch without retaining activation traces."""
        spatial, pooled, _, _, _ = self.encode(batch['boards'], batch['targets'], batch['context'])
        count = batch['valid'].sum(1, keepdim=True).double()
        sums = torch.zeros((len(count), 64), device=spatial.device, dtype=torch.float64)
        squares = torch.zeros_like(sums)
        saturated = torch.zeros((), device=spatial.device, dtype=torch.float64)
        raw = []
        for start in range(0, batch['features'].shape[1], self.config.chunk):
            stop = start + self.config.chunk
            logits, _, hidden = self.rank_chunk(spatial, pooled, batch['context'], batch['features'][:, start:stop],
                                                batch['indices'][:, start:stop], batch['column_mask'][:, start:stop])
            mask = batch['valid'][:, start:stop, None].bool()
            values = hidden.double()
            sums += (values * mask).sum(1)
            squares += (values.square() * mask).sum(1)
            saturated += (((hidden <= .001) | (hidden >= .999)) & mask).sum()
            raw.append(logits.float())
        logits = torch.cat(raw, 1)
        valid = batch['valid'].bool()
        mean = (logits.double() * valid).sum(1, keepdim=True) / count
        variance = ((logits.double() - mean).square() * valid).sum(1) / count[:, 0]
        spread = logits.masked_fill(~valid, -torch.inf).amax(1) - logits.masked_fill(~valid, torch.inf).amin(1)
        hidden_variance = (squares / count - (sums / count).square()).clamp_min(0)
        return {'states': len(count), 'h2_saturated_fraction': float(saturated / (count.sum() * 64)),
                'h2_candidate_std': float(hidden_variance.sqrt().mean()),
                'raw_logit_spread': float(spread.mean()), 'raw_logit_std': float(variance.sqrt().mean())}


def create_model(config: Config = Config()):
    if config.architecture == 'survival-feature-v3':
        from tetris_survival_v3 import SurvivalFeaturePolicy
        return SurvivalFeaturePolicy(config)
    if config.architecture in AFTERSTATE_TD_ARCHITECTURES:
        from tetris_afterstate_td import AfterstateTDPolicy
        return AfterstateTDPolicy(config)
    if config.architecture == 'survival-afterstate-v5':
        from tetris_survival_v5 import SurvivalAfterstatePolicy
        return SurvivalAfterstatePolicy(config)
    if config.architecture == 'survival-spatial-v4':
        from tetris_survival_v4 import SurvivalSpatialPolicy
        return SurvivalSpatialPolicy(config)
    return ColumnPolicy(config)


@dataclass
class Prepared:
    arrays: dict
    options: tuple
    teacher: int
    evaluations: int
    fallback: bool


@lru_cache(maxsize=1024)
def _shape_features(shape_cells):
    cells = np.zeros((16, 5), dtype=np.float32)
    cells[:len(shape_cells), :4] = np.asarray(shape_cells, dtype=np.float32) / 15
    cells[:len(shape_cells), 4] = 1
    result = cells.ravel()
    result.flags.writeable = False
    return result


def _option_features(board, candidate, target, context, metrics=None):
    shape = game.SHAPES[candidate.piece][candidate.rotation]
    base = game.column_index(candidate.x, candidate.y, candidate.w)
    touched = [base + offset for offset, _, _ in shape.columns]
    delta = sum(candidate.z + mask.bit_length() - context.heights[base + offset] for offset, mask, _ in shape.columns)
    if metrics is None:
        gain, _, future = game.sustained_candidate_metrics(board, candidate, target)
    else:
        gain, future = metrics
    fixed = [candidate.x / 15, candidate.y / 15, candidate.z / 15, candidate.w / 15]
    fixed += [v / 16 for v in shape.extents]
    size = game.PIECE_SIZES[candidate.piece]
    fixed += [gain / 16, future / 64, candidate.layers / 16, delta / 64, (delta - size) / 64,
              (candidate.score - context.score) / 64, size / 16, target / 15]
    result = np.empty(FEATURES, dtype=np.float32)
    result[:16] = fixed
    result[16:] = _shape_features(shape.cells)
    return result, touched


def _prepare_steps(state, target: int, config: Config, candidates: int | None = None):
    budget = config.screening_budget if candidates is None else candidates
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
    if len(options) > config.shortlist and config.shortlist_mode == 'teacher-top':
        keep = sorted(heapq.nlargest(config.shortlist, range(len(options)), key=keys.__getitem__))
    elif len(options) > config.shortlist:
        top = heapq.nlargest(max(1, config.shortlist // 2), range(len(options)), key=keys.__getitem__)
        used = set(top)
        rest = [i for i in range(len(options)) if i not in used]
        rng = random.Random(state.seed * 8191 + state.pieces)
        if config.shortlist_mode == 'diverse':
            buckets, diverse = set(), []
            for index in sorted(rest, key=keys.__getitem__, reverse=True):
                option = options[index]
                bucket = (option.rotation, option.x // 4, option.y // 4, option.w // 4)
                if bucket not in buckets:
                    buckets.add(bucket)
                    diverse.append(index)
                    if len(diverse) == config.shortlist - len(top):
                        break
            top.extend(diverse)
            if len(top) < config.shortlist:
                remaining = [index for index in rest if index not in set(diverse)]
                top.extend(rng.sample(remaining, config.shortlist - len(top)))
        else:
            top.extend(rng.sample(rest, config.shortlist - len(top)))
        keep = sorted(top)
    else:
        keep = list(range(len(options)))
    kept_keys = [keys[i] for i in keep]
    options = tuple(options[i] for i in keep)
    teacher = keep.index(best)
    features = np.zeros((len(options), FEATURES), dtype=np.float32)
    indices = np.zeros((len(options), COLUMNS), dtype=np.int64)
    mask = np.zeros((len(options), COLUMNS), dtype=np.bool_)
    metrics = [None] * len(options)
    # The batch gain uses column heights. On buried targets the scalar
    # feature contract also counts gaps below that height.
    if not status.buried and game.BACKEND in ('numpy', 'native'):
        specs = [(c.rotation, c.x, c.y, c.w) for c in options]
        landings = np.asarray([c.z for c in options], dtype=np.int32)
        gains, _, future = game.sustained_batch_metrics(context, piece, specs, landings, target)
        metrics = list(zip(gains, future))
    for i, candidate in enumerate(options):
        features[i], columns = _option_features(state.board, candidate, target, context, metrics[i])
        indices[i, :len(columns)], mask[i, :len(columns)] = columns, True
    arrays = {'boards': np.asarray(state.board, dtype=np.int32), 'targets': np.asarray(target, dtype=np.int64),
              'context': state_context(state, target, context), 'features': features, 'indices': indices,
              'column_mask': mask, 'valid': np.ones(len(options), dtype=np.bool_)}
    order = sorted(range(len(options)), key=lambda i: kept_keys[i], reverse=True)
    rank = np.empty(len(options), dtype=np.float32)
    rank[order] = np.arange(len(options), dtype=np.float32)
    deficits = np.asarray([kept_keys[teacher][0] - k[0] for k in kept_keys], dtype=np.float32)
    arrays['teacher_prior'] = -32 * deficits - rank / 8
    if config.architecture in SPATIAL_ARCHITECTURES:
        from tetris_survival_v4 import afterstate_descriptors
        arrays['spatial_board'], arrays['spatial_delta'] = afterstate_descriptors(state.board, options, target)
    return Prepared(arrays, options, teacher, evaluations, fallback)


def prepare_steps(state, target: int, config: Config, candidates: int | None = None):
    prepared = yield from _prepare_steps(state, target, config, candidates)
    if prepared is not None or not config.recover_target or not config.protect_target:
        return prepared
    if state.game_over or not game.has_legal_move(state.board, state.current):
        return None
    for alternative in sorted(range(game.HEIGHT), key=lambda z: (abs(z - target), z)):
        if alternative == target:
            continue
        status = game.target_layer_status(state.board, alternative)
        if status.buried or status.filled == game.AREA:
            continue
        prepared = yield from _prepare_steps(state, alternative, config, candidates)
        if prepared is not None:
            return prepared
    return None


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
    if any(torch.is_tensor(row['features']) for row in records):
        result = {}
        for name in records[0]:
            values = []
            for row in records:
                value = torch.as_tensor(row[name], device=device)
                if name in CANDIDATE_FIELDS and len(value) < maximum:
                    value = F.pad(value, (0, 0) * (value.ndim - 1) + (0, maximum - len(value)))
                values.append(value)
            result[name] = torch.stack(values)
        if not bool(result['valid'].any(dim=1).all()):
            raise ValueError('Every state must have at least one valid candidate')
        return result
    if not all(np.asarray(row['valid']).any() for row in records):
        raise ValueError('Every state must have at least one valid candidate')
    result = {}
    for name in records[0]:
        values = []
        for row in records:
            value = np.asarray(row[name])
            if name in CANDIDATE_FIELDS and len(value) < maximum:
                value = np.pad(value, [(0, maximum - len(value))] + [(0, 0)] * (value.ndim - 1))
            values.append(value)
        result[name] = torch.from_numpy(np.stack(values)).to(device)
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
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != SCHEMA or checkpoint.get('pieces_sha256') != fingerprint():
        raise ValueError('Checkpoint schema or piece catalog does not match this engine')
    model = create_model(Config(**checkpoint['config']))
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
    def __init__(self, path, device='auto', engine=None, allow_untrained=False, capture_traces=True):
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
        self.pending_trace = None
        self.capture_traces = (False if self.model.config.architecture in FEATURE_ARCHITECTURES
                               else capture_traces)
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
        afterstate_td = self.model.config.architecture in AFTERSTATE_TD_ARCHITECTURES
        # The TD actor shortlist always starts from exhaustive screening, including GUI playback.
        if afterstate_td:
            candidates = 0
        prepared = yield from prepare_steps(state, target, self.model.config, candidates)
        if prepared is None:
            return None
        if self.model.config.architecture == 'survival-afterstate-v5':
            from tetris_survival_v5 import augment_prepared
            augment_prepared(prepared, state, self.device)
        elif afterstate_td:
            prepared = self.model.prepare_gameplay(prepared, state)
        before = time.perf_counter()
        requested_target = target
        target = int(prepared.arrays['targets'])
        batch = collate([prepared.arrays], self.device)
        with torch.inference_mode():
            prediction = self.model(batch, capture=self.capture_traces is True)
            index = int(prediction['logits'][0].argmax().item())
            if not bool(torch.isfinite(prediction['logits'][0][batch['valid'][0]]).all()):
                raise ValueError('Neural inference produced non-finite logits')
            score = float(prediction['logits'][0, index].item())
            trace_info = {'logit': score, 'piece': state.current, 'piece_number': state.pieces + 1,
                          'shortlist': len(prepared.options), 'selected': index}
            if self.capture_traces is True:
                self.last_trace = self._trace(prediction, trace_info)
            elif self.capture_traces == 'lazy':
                self.pending_trace = (prepared.arrays, trace_info)
        inferred = time.perf_counter()
        candidate = prepared.options[index]
        move = game.materialize_move(state.board, candidate)
        gain, _, _ = game.sustained_candidate_metrics(state.board, candidate, target)
        if self.model.config.architecture in FEATURE_ARCHITECTURES:
            teacher_prior = float(batch['teacher_prior'][0, index]) if 'teacher_prior' in batch else 0.
            self.last_trace = {'feature_inspector': True, **trace_info,
                               'teacher_prior': teacher_prior, 'critic_value': float(prediction['value'][0]),
                               'spatial_delta': (batch['spatial_delta'][0, index].detach().cpu().tolist()
                                                 if 'spatial_delta' in batch else None)}
            if afterstate_td:
                self.last_trace['afterstate_value'] = float(prediction['value'][0])
            elif self.model.config.architecture == 'survival-afterstate-v5':
                self.last_trace['base_logit'] = float(prediction['base_logits'][0, index])
                self.last_trace['residual'] = float(prediction['raw_logits'][0, index])
            elif self.model.config.policy_mode == 'teacher-prior' and self.model.config.residual_scale:
                self.last_trace['actor_score'] = (score - self.model.config.teacher_prior_scale * teacher_prior) / self.model.config.residual_scale
            elif self.model.config.policy_mode == 'neural':
                self.last_trace['actor_score'] = score
            else:
                self.last_trace['actor_score'] = None
        self.last_metrics = {'prepare_ms': (before - started) * 1000,
                             'inference_and_trace_transfer_ms': (inferred - before) * 1000,
                             'evaluations': prepared.evaluations, 'shortlist': len(prepared.options),
                             'teacher_agreement': index == prepared.teacher,
                             'requested_target': requested_target, 'active_target': target,
                             'target_recovered': target != requested_target,
                             'trace_captured': self.capture_traces is True, **self.metadata}
        return game.Decision(move, score, prepared.evaluations, (time.perf_counter() - started) * 1000,
                             1, 1, 0 if afterstate_td else candidates or self.model.config.candidates,
                             not afterstate_td, 'neural', target, gain, prepared.fallback)

    def choose(self, *args, **kwargs):
        iterator = self.steps(*args, **kwargs)
        while True:
            try:
                next(iterator)
            except StopIteration as result:
                return result.value

    def _trace(self, prediction, info):
        return {**info, 'activation': prediction['activation'][0].float().cpu().numpy(),
                'hidden1': prediction['hidden1'][0].float().cpu().numpy(),
                'hidden2': prediction['hidden2'][0].float().cpu().numpy(),
                'completion': torch.sigmoid(prediction['completion'][0]).float().cpu().numpy(),
                'value': float(prediction['value'][0].item())}

    def get_trace(self):
        if self.pending_trace is not None:
            arrays, info = self.pending_trace
            with torch.inference_mode():
                prediction = self.model(collate([arrays], self.device), capture=True)
            self.last_trace = self._trace(prediction, info)
            self.pending_trace = None
        return self.last_trace

    def create_view(self, parent, legacy_class, policy):
        if self.model.config.architecture in FEATURE_ARCHITECTURES:
            from tetris_survival_v4 import FeatureView
            self.capture_traces = False
            return FeatureView(parent, self, legacy_class, policy)
        self.capture_traces = 'lazy'
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
        trace = self.agent.get_trace()
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


def gae(rewards, values, next_values, terminated, truncated, gamma, lam):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    tail = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(len(rewards) - 1, -1, -1):
        delta = rewards[step] + gamma * next_values[step] * (~terminated[step]) - values[step]
        tail = delta + gamma * lam * (~(terminated[step] | truncated[step])) * tail
        advantages[step] = tail
    return advantages, advantages + values


def evaluate(args):
    agent = NeuralAgent(args.checkpoint, args.device, capture_traces=False)
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
                target = decision.target if decision.target is not None else target
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
    parser = argparse.ArgumentParser(description='Evaluate a saved policy or run a neural implementation smoke test.')
    parser.add_argument('--backend', choices=('auto', 'native', 'numpy', 'python'), default='auto')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--threads', type=int, default=4)
    sub = parser.add_subparsers(dest='command', required=True)
    test = sub.add_parser('evaluate')
    test.add_argument('--checkpoint', type=Path, required=True)
    test.add_argument('--seeds', nargs='+', type=int, default=[70001, 70002, 70003, 70004])
    test.add_argument('--pieces', type=int, default=16000)
    test.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    test.add_argument('--output', type=Path, default=Path('neural-evaluation.json'))
    test.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    quick = sub.add_parser('smoke')
    quick.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    quick.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.command == 'evaluate' and args.pieces < 1:
        parser.error('--pieces must be positive')
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    game.load_checkpoint(args.weights)
    try:
        result = {'evaluate': evaluate, 'smoke': smoke}[args.command](args)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Neural pipeline failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
