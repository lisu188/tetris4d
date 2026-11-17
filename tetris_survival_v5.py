"""Survival V5: a bounded next-state scorer on top of a frozen V4 actor.

    logits_c = (base_c + R * tanh(beta * (S(f(s'_c)) - mean_c S))) / tau

base: a frozen V4 actor; its logits already include the teacher prior. S: a scorer over
exact next-state features f(s'). The piece stream is deterministic, so the afterstate of a
candidate *is* the next state. R bounds how far training can move from the base: the
unbounded V4 raw head drifted until an obvious clear lost to ordinary moves. tau sharpens
sampled play. The critic V uses the same features with separate parameters.

Every feature row is laid out as [V4 pooled (266) | extras (34) | context (82)]. The V4 pooled
block plus the context is exactly the regional V4 critic input. That lets the scorer and the
critic both start as exact copies of a calibrated V4 critic, with the extras branch zeroed. The
initial policy is then the bounded afterstate ranking measured in issue #37.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import tetris_neural as neural
import tetris_survival_v4 as v4

ARCHITECTURE = 'survival-afterstate-v5'
game = neural.game
V4_POOLED = 2 * neural.CHANNELS + v4.SPATIAL_CRITIC_FEATURES + v4.REGIONAL_CRITIC_FEATURES
SURVIVAL_NAMES = ('holes', 'hole_depth', 'columns_with_holes', 'z_transitions', 'cumulative_wells',
                  'deepest_well', 'columns_at_12', 'columns_at_14', 'diversity_x', 'diversity_y', 'diversity_w')
MOVE_NAMES = ('layers_cleared', 'eroded_cells', 'landing_height', 'target_gain', 'teacher_score_delta')
EXTRA = v4.SPATIAL_ACTOR_FEATURES + len(SURVIVAL_NAMES) + len(MOVE_NAMES)
FEATURES = V4_POOLED + EXTRA + neural.CONTEXT
PIECE_INDEX = {name: index for index, name in enumerate(game.PIECE_NAMES)}


# Exact board features --------------------------------------------------------------------------

def pooled_columns(masks, targets):
    """Mean and max over the 4096 columns of neural.encode_columns' 34 channels: [n, 68].

    Counts are exact integers and the divisors are powers of two, so the result equals the
    float32 mean/amax of the encoding bit for bit, without building the [n, 34, 4096] tensor.
    """
    masks = masks.to(torch.int32)
    levels = torch.arange(16, device=masks.device, dtype=torch.int32)[None, :, None]
    occupied = ((masks[:, None, :] >> levels) & 1).bool()
    height = v4._bit_length16(masks)
    buried = ~occupied & (levels < height[:, None, :])
    accessible = height <= targets.to(torch.int32)[:, None]
    area = float(game.AREA)
    mean = torch.cat((occupied.sum(2).float() / area, buried.sum(2).float() / area,
                      height.sum(1, keepdim=True).float() / (16 * area),
                      accessible.sum(1, keepdim=True).float() / area), dim=1)
    maximum = torch.cat((occupied.any(2).float(), buried.any(2).float(),
                         height.amax(1, keepdim=True).float() / 16, accessible.any(1, keepdim=True).float()), dim=1)
    return torch.cat((mean, maximum), dim=1)


def survival_features(masks):
    """4D Dellacherie-Thiery-style board features, each scaled to about [0, 1]: [n, 11].

    holes / hole depth / columns with holes follow the classic definitions per vertical column.
    Z transitions count filled/empty changes up each column with the floor counted as filled.
    A well is a column lower than all six horizontal neighbours (walls count as full height);
    cumulative wells sum depth*(depth+1)/2. Diversity is the fraction of the height differences
    -2..2 that occur between neighbouring columns along each axis.
    """
    masks = masks.to(torch.int32)
    n = len(masks)
    height = v4._bit_length16(masks)
    holes = height - v4._bitcount16(masks)
    levels = torch.arange(16, device=masks.device, dtype=torch.int32)[None, :, None]
    hole_cells = (((masks[:, None, :] >> levels) & 1) == 0) & (levels < height[:, None, :])
    above = v4._bitcount16(masks[:, None, :] >> (levels + 1))
    hole_depth = (hole_cells.to(torch.int32) * above).sum((1, 2))
    transitions = v4._bitcount16((masks ^ ((masks << 1) | 1)) & 0xFFFF)
    grid = height.reshape(n, 16, 16, 16)  # [n, w, y, x]: column index = (w * 16 + y) * 16 + x
    padded = F.pad(grid, (1, 1, 1, 1, 1, 1), value=16)
    lowest = torch.stack((padded[:, 1:-1, 1:-1, :-2], padded[:, 1:-1, 1:-1, 2:],
                          padded[:, 1:-1, :-2, 1:-1], padded[:, 1:-1, 2:, 1:-1],
                          padded[:, :-2, 1:-1, 1:-1], padded[:, 2:, 1:-1, 1:-1])).amin(0)
    wells = (lowest - grid).clamp_min(0).reshape(n, -1)
    diversity = []
    for dim in (3, 2, 1):  # x, y, w
        difference = torch.diff(grid, dim=dim).reshape(n, -1)
        diversity.append(torch.stack([(difference == value).any(1) for value in range(-2, 3)], 1).sum(1).float() / 5)
    area = float(game.AREA)
    return torch.stack((holes.sum(1).float() / (16 * area), hole_depth.float() / (256 * area),
                        (holes > 0).sum(1).float() / area, transitions.sum(1).float() / (16 * area),
                        (wells * (wells + 1) // 2).sum(1).float() / (136 * area), wells.amax(1).float() / 16,
                        (height >= 12).sum(1).float() / area, (height >= 14).sum(1).float() / area,
                        *diversity), dim=1)


def board_counts_reference(masks, targets):
    """Plain-torch reference for tetris_gpu_board_features.board_counts (same [n, 46] int layout)."""
    masks = masks.to(torch.int32)
    n = len(masks)
    levels = torch.arange(16, device=masks.device, dtype=torch.int32)[None, :, None]
    bits = (masks[:, None, :] >> levels) & 1
    height = v4._bit_length16(masks)
    hole = (bits == 0) & (levels < height[:, None, :])
    above = v4._bitcount16(masks[:, None, :] >> (levels + 1))
    holes = height - v4._bitcount16(masks)
    transitions = v4._bitcount16((masks ^ ((masks << 1) | 1)) & 0xFFFF)
    grid = height.reshape(n, 16, 16, 16)  # [n, w, y, x]: column index = (w * 16 + y) * 16 + x
    padded = F.pad(grid, (1, 1, 1, 1, 1, 1), value=16)
    lowest = torch.stack((padded[:, 1:-1, 1:-1, :-2], padded[:, 1:-1, 1:-1, 2:],
                          padded[:, 1:-1, :-2, 1:-1], padded[:, 1:-1, 2:, 1:-1],
                          padded[:, :-2, 1:-1, 1:-1], padded[:, 2:, 1:-1, 1:-1])).amin(0)
    wells = (lowest - grid).clamp_min(0).reshape(n, -1)
    diversity = []
    for dim in (3, 2, 1):  # x, y, w
        difference = torch.diff(grid, dim=dim).reshape(n, -1)
        present = torch.stack([(difference == value).any(1) for value in range(-2, 3)], 1).to(torch.int32)
        diversity.append((present << torch.arange(5, device=masks.device, dtype=torch.int32)).sum(1))
    columns = lambda value: value.sum(1)[:, None]
    return torch.cat((bits.sum(2), hole.sum(2), columns(height), height.amax(1)[:, None],
                      columns(height <= targets.to(device=masks.device, dtype=torch.int32)[:, None]),
                      columns(holes), (hole * above).sum((1, 2))[:, None], columns(holes > 0), columns(transitions),
                      columns(wells * (wells + 1) // 2), wells.amax(1)[:, None], columns(height >= 12),
                      columns(height >= 14), torch.stack(diversity, 1)), dim=1).to(torch.int32)


_KERNELS = None


def board_counts(masks, targets):
    """[n, 46] integer board statistics: the fused Triton kernel on CUDA, the torch reference otherwise."""
    global _KERNELS
    if masks.is_cuda:
        if _KERNELS is None:
            try:
                import tetris_gpu_board_features as kernels
                _KERNELS = kernels
            except ImportError:  # Triton unavailable
                _KERNELS = False
        if _KERNELS:
            return _KERNELS.board_counts(masks, targets)
    return board_counts_reference(masks, targets)


def _popcount5(values):
    return sum(((values >> bit) & 1) for bit in range(5)).float()


def pooled_from_counts(counts):
    """Exactly pooled_columns(masks, targets) from the integer counts: [n, 68]."""
    area = float(game.AREA)
    occupied, buried = counts[:, :16], counts[:, 16:32]
    height_sum, height_max, accessible = counts[:, 32:33], counts[:, 33:34], counts[:, 34:35]
    return torch.cat((occupied.float() / area, buried.float() / area, height_sum.float() / (16 * area),
                      accessible.float() / area, (occupied > 0).float(), (buried > 0).float(),
                      height_max.float() / 16, (accessible > 0).float()), dim=1)


def survival_from_counts(counts):
    """Exactly survival_features(masks) from the integer counts: [n, 11]."""
    area = float(game.AREA)
    c = counts
    return torch.stack((c[:, 35].float() / (16 * area), c[:, 36].float() / (256 * area), c[:, 37].float() / area,
                        c[:, 38].float() / (16 * area), c[:, 39].float() / (136 * area), c[:, 40].float() / 16,
                        c[:, 41].float() / area, c[:, 42].float() / area,
                        _popcount5(c[:, 43]) / 5, _popcount5(c[:, 44]) / 5, _popcount5(c[:, 45]) / 5), dim=1)


def context_from_counts(counts, targets, piece_rows, piece_phase):
    """Exactly tetris_survival_v4.device_afterstate_context from the integer counts: [n, 82]."""
    area = float(game.AREA)
    holes, height_sum = counts[:, 35].double(), counts[:, 32].double()
    glob = torch.stack((targets.double() / 15, holes / 65536, (height_sum - holes) / 65536,
                        counts[:, 33].double() / 16, piece_phase.double()), dim=1)
    return torch.cat((piece_rows.double(), counts[:, :16].double() / area, counts[:, 16:32].double() / area,
                      glob), dim=1).float()


def assemble(masks, targets, context, moves, counts=None):
    """One feature row per board: [V4 pooled 266 | adjacency 18, survival 11, move 5 | context 82]."""
    masks = masks.to(torch.int32)
    counts = board_counts(masks, targets) if counts is None else counts
    adjacency = v4.adjacency_statistics(masks, targets.to(torch.int32))
    pooled = torch.cat((pooled_from_counts(counts), adjacency.mean(dim=1).float(),
                        v4.regional_statistics(masks).float()), dim=1)
    extras = torch.cat((adjacency.reshape(len(masks), -1).float(), survival_from_counts(counts), moves.float()), dim=1)
    return torch.cat((pooled, extras, context.float()), dim=1)


def state_pooled(masks, targets):
    """Feature row without the context for current states (move features are zero): [n, 300]."""
    masks = masks.to(torch.int32)
    rows = assemble(masks, targets, masks.new_zeros((len(masks), neural.CONTEXT), dtype=torch.float32),
                    masks.new_zeros((len(masks), len(MOVE_NAMES)), dtype=torch.float32))
    return rows[:, :-neural.CONTEXT]


@lru_cache(maxsize=8)
def _shape_tables(device):
    rotations = max(len(game.SHAPES[name]) for name in game.PIECE_NAMES)
    width = max(len(shape.cells) for name in game.PIECE_NAMES for shape in game.SHAPES[name])
    cells = torch.zeros((len(game.PIECE_NAMES), rotations, width, 4), dtype=torch.int64)
    present = torch.zeros((len(game.PIECE_NAMES), rotations, width), dtype=torch.bool)
    for p, name in enumerate(game.PIECE_NAMES):
        for r, shape in enumerate(game.SHAPES[name]):
            cells[p, r, :len(shape.cells)] = torch.tensor(shape.cells)
            present[p, r, :len(shape.cells)] = True
    return cells.to(device), present.to(device)


def place(boards, cells, cell_mask, clears=None):
    """Merge absolute (x, y, z, w) cells and clear full layers -> (after [k,4096] int32, full [k,16]).

    `clears` (bool [k], optional) marks rows that may clear; other rows skip the full-layer pass.
    """
    boards = boards.to(torch.int32)
    cells = cells.long()
    index = ((cells[..., 3] * 16 + cells[..., 1]) * 16 + cells[..., 0]).masked_fill(~cell_mask, 0)
    bits = torch.where(cell_mask, torch.bitwise_left_shift(torch.ones_like(index), cells[..., 2]), 0)
    # Placed cells never overlap occupied bits and differ in Z within a column, so sum == OR.
    merged = boards.clone().scatter_add_(1, index, bits.to(torch.int32))
    full = torch.zeros((len(merged), 16), dtype=torch.bool, device=merged.device)
    rows = torch.arange(len(merged), device=merged.device) if clears is None else clears.nonzero().flatten()
    if len(rows):
        subset = merged.index_select(0, rows)
        levels = torch.arange(16, device=boards.device, dtype=torch.int32)
        full[rows] = ((subset[:, None, :] >> levels[None, :, None]) & 1).sum(2) == boards.shape[1]
        merged[rows] = v4._compact_full_layers(subset)
    return merged, full


@torch.no_grad()
def next_state_features(boards, targets, specs, valid, states, move, chunk=256):
    """Exact feature rows of every valid candidate's next state.

    boards [n,4096] int, targets [n], specs [n,K,>=5] (rotation, x, y, w, z, ...), valid [n,K],
    states: the n Games (for piece one-hots), move [n,K,2] = (target_gain/16, teacher_score_delta/64)
    taken from the V4 candidate features. Returns [n, K, FEATURES] float32, zero where invalid.
    """
    device = boards.device
    n, width = valid.shape
    out = torch.zeros((n, width, FEATURES), dtype=torch.float32, device=device)
    rows = valid.bool().nonzero(as_tuple=False)
    if not len(rows):
        return out
    env, slot = rows[:, 0], rows[:, 1]
    table, present = _shape_tables(device)
    pieces = torch.tensor([PIECE_INDEX[s.current] for s in states], device=device)
    spec = specs.to(device)[env, slot].long()
    piece = pieces[env]
    mask = present[piece, spec[:, 0]]
    cells = table[piece, spec[:, 0]] + torch.stack((spec[:, 1], spec[:, 2], spec[:, 4], spec[:, 3]), 1)[:, None, :]
    info = [v4.afterstate_piece_features(s) for s in states]
    env_list = env.tolist()
    piece_rows = torch.tensor([info[e][0] for e in env_list], dtype=torch.float64, device=device)
    phase = torch.tensor([info[e][1] for e in env_list], dtype=torch.float64, device=device)
    owner_targets = targets.to(device).long()[env]
    boards = boards.to(device=device, dtype=torch.int32)
    chunks = []
    for start in range(0, len(rows), chunk):
        part = slice(start, start + chunk)
        after, full = place(boards.index_select(0, env[part]), cells[part], mask[part], spec[part, 5] > 0)
        layers = full.sum(1)
        after_targets = owner_targets[part].clone()
        clearing = (layers > 0).nonzero().flatten()
        if len(clearing):  # rare: retarget exactly like train_overnight.apply_move
            for i, board in zip(clearing.tolist(), after.index_select(0, clearing).cpu().tolist()):
                following = game.next_hyperlayer_target(tuple(board), int(after_targets[i]))
                if following is not None:
                    after_targets[i] = following
        counts = board_counts(after, after_targets)
        context = context_from_counts(counts, after_targets, piece_rows[part], phase[part])
        piece_mask = mask[part]
        eroded = (full.gather(1, cells[part][..., 2].clamp(0, 15)) & piece_mask).sum(1)
        landing = (cells[part][..., 2] * piece_mask).sum(1).float() / piece_mask.sum(1).clamp_min(1)
        moves = torch.stack((layers.float() / 16, (layers * eroded).float() / 16, landing / 16,
                             move[env[part], slot[part], 0].float(), move[env[part], slot[part], 1].float()), dim=1)
        chunks.append(assemble(after, after_targets, context, moves, counts))
    out[env, slot] = torch.cat(chunks)
    return out


# Attaching next-state features to prepared batches --------------------------------------------

def _move_columns(features):
    features = torch.as_tensor(features)
    return torch.stack((features[..., 8], features[..., 13]), dim=-1)


def augment_device_batch(result, states):
    """Add `next_features` to a DevicePreparedBatch (batch and per-environment records)."""
    if not result.environment_indices:
        return result
    batch = result.batch
    live = [states[i] for i in result.environment_indices]
    features = next_state_features(batch['boards'], batch['targets'], result._option_specs, batch['valid'].bool(),
                                   live, _move_columns(batch['features']))
    batch['next_features'] = features
    for row, env in enumerate(result.environment_indices):
        prepared = result.prepared[env]
        prepared.arrays['next_features'] = features[row, :len(prepared.options)]
    return result


def augment_prepared(prepared, state, device):
    """Add `next_features` to one prepared state (CUDA GpuPrepared or CPU reference Prepared)."""
    arrays = prepared.arrays
    on_device = torch.is_tensor(arrays['features'])
    options = prepared.options
    if hasattr(options, 'specifications'):
        specs = options.specifications.to(device)
    else:
        specs = torch.tensor([(c.rotation, c.x, c.y, c.w, c.z, c.layers) for c in options], device=device)
    features = torch.as_tensor(arrays['features'], device=device)
    rows = next_state_features(torch.as_tensor(arrays['boards'], device=device).reshape(1, -1),
                               torch.as_tensor(arrays['targets'], device=device).reshape(1),
                               specs[None], torch.as_tensor(arrays['valid'], device=device).reshape(1, -1).bool(),
                               [state], _move_columns(features)[None])[0]
    arrays['next_features'] = rows if on_device else rows.cpu().numpy()
    return prepared


# Model -----------------------------------------------------------------------------------------

class _Head(nn.Module):
    """Main branch over the V4 critic block (copyable from a V4 critic) plus a zero-initialised
    branch over the extras. Indexing reaches the main branch, as V3/V4 calibration expects."""

    def __init__(self):
        super().__init__()
        width = V4_POOLED + neural.CONTEXT
        self.main = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 128), nn.SiLU(),
                                  nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.extra = nn.Sequential(nn.LayerNorm(EXTRA), nn.Linear(EXTRA, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)

    def __getitem__(self, index):
        return self.main[index]

    def forward(self, rows):
        main = torch.cat((rows[..., :V4_POOLED], rows[..., -neural.CONTEXT:]), dim=-1)
        return (self.main(main) + self.extra(rows[..., V4_POOLED:-neural.CONTEXT])).squeeze(-1)


def base_config(config):
    return replace(config, architecture=v4.ARCHITECTURE, regional_critic=True,
                   residual_bound=0., policy_temperature=1., afterstate_scale=1.)


class SurvivalAfterstatePolicy(nn.Module):
    NEXT_FEATURES = FEATURES

    def __init__(self, config: neural.Config):
        super().__init__()
        if config.architecture != ARCHITECTURE:
            raise ValueError('SurvivalAfterstatePolicy requires the survival-afterstate-v5 architecture')
        if not config.spatial_actor:
            raise ValueError('V5 builds on a spatial V4 actor')
        self.config = config
        self.base = v4.SurvivalSpatialPolicy(base_config(config))
        self.base.requires_grad_(False)
        self.scorer = _Head()
        self.value = _Head()

    @property
    def fused_encoding(self):
        return self.base.fused_encoding

    @fused_encoding.setter
    def fused_encoding(self, value):
        self.base.fused_encoding = bool(value)

    def critic(self, joined):
        return F.softplus(self.value(joined).float())

    def score(self, rows):
        """Scorer output in the critic's units (thousands of remaining placements)."""
        return F.softplus(self.scorer(rows).float())

    def encode(self, boards, targets, context):
        pooled = state_pooled(boards, targets)
        value = self.critic(torch.cat((pooled, context.float()), dim=1))
        return None, pooled, value, value.new_empty((len(value), 0)), None

    def residual(self, batch):
        valid = batch['valid'].bool()
        if 'next_features' not in batch:
            raise ValueError('V5 requires next-state features (next_features)')
        scores = self.score(batch['next_features'])
        count = valid.sum(1, keepdim=True).clamp_min(1)
        centred = scores - scores.masked_fill(~valid, 0.).sum(1, keepdim=True) / count
        residual = self.config.residual_bound * torch.tanh(self.config.afterstate_scale * centred)
        return residual.masked_fill(~valid, 0.)

    def forward(self, batch, capture=False):
        if capture:
            raise NotImplementedError('V5 uses feature diagnostics rather than activation capture')
        valid = batch['valid'].bool()
        with torch.no_grad():
            base = self.base(batch)['logits'].float()
        _, pooled, value, completion, _ = self.encode(batch['boards'], batch['targets'], batch['context'])
        residual = self.residual(batch)
        logits = ((base + residual) / self.config.policy_temperature).masked_fill(~valid, -torch.inf)
        return {'logits': logits, 'raw_logits': residual, 'base_logits': base, 'value': value,
                'completion': completion, 'critic_input': torch.cat((pooled, batch['context'].float()), dim=1).detach()}

    @torch.no_grad()
    def ranking_diagnostics(self, batch):
        valid = batch['valid'].bool()
        residual = self.residual(batch)
        values = residual[valid]
        bound = self.config.residual_bound
        return {'states': int(len(valid)), 'architecture': ARCHITECTURE, 'h2_activation': 'bounded-tanh-residual',
                'h2_saturated_fraction': float((values.abs() >= .95 * bound).float().mean()) if bound > 0 and len(values) else 0.,
                'h2_candidate_std': float(values.std()) if len(values) > 1 else 0.,
                'raw_logit_spread': float(values.max() - values.min()) if len(values) else 0.,
                'residual_bound': bound, 'policy_temperature': self.config.policy_temperature,
                'afterstate_scale': self.config.afterstate_scale}


def from_v4(source, *, residual_bound, policy_temperature, afterstate_scale, shortlist_mode=None, screening=None):
    """V5 model whose base is `source` (frozen) and whose scorer and critic copy its calibrated critic."""
    if source.config.architecture != v4.ARCHITECTURE:
        raise ValueError('V5 starts from a V4 (survival-spatial-v4) checkpoint')
    if not source.config.regional_critic:
        raise ValueError('V5 starts from a calibrated regional V4 critic (regional_critic=True)')
    config = replace(source.config, architecture=ARCHITECTURE, residual_bound=float(residual_bound),
                     policy_temperature=float(policy_temperature), afterstate_scale=float(afterstate_scale),
                     shortlist_mode=shortlist_mode or source.config.shortlist_mode,
                     screening=screening or source.config.screening)
    model = SurvivalAfterstatePolicy(config).to(next(source.parameters()).device)
    model.base.load_state_dict(source.state_dict(), strict=True)
    model.base.requires_grad_(False)
    model.scorer.main.load_state_dict(source.value.state_dict())
    model.value.main.load_state_dict(source.value.state_dict())
    return model
