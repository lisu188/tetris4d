"""Spatial survival actor/critic built on V3 fixed descriptors."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import tetris_neural as neural
from tetris_survival_v3 import ARCHITECTURE as V3_ARCHITECTURE
from tetris_survival_v3 import SurvivalFeaturePolicy

ARCHITECTURE = 'survival-spatial-v4'
AXIS_NAMES = ('X', 'Y', 'W')
STATISTIC_NAMES = ('height_abs', 'height_squared', 'cliff', 'occupancy_boundary',
                   'buried_gap_boundary', 'target_layer_boundary')
SPATIAL_CRITIC_FEATURES = 6
SPATIAL_ACTOR_FEATURES = 18
# Each statistic is a mean over 15*16*16 neighbour edges per axis, so one placement
# moves it by ~1e-4. Rescale afterstate deltas to per-edge change counts (O(1)).
SPATIAL_DELTA_SCALE = 15 * 16 * 16


def _bitcount16(value):
    value = value.to(torch.int32)
    value = value - ((value >> 1) & 0x5555)
    value = (value & 0x3333) + ((value >> 2) & 0x3333)
    return ((((value + (value >> 4)) & 0x0f0f) * 0x0101) >> 8) & 0x1f


def _bit_length16(value):
    # Exact integer bit length of 16-bit masks (no float log2 rounding).
    result = torch.zeros_like(value)
    for shift in (8, 4, 2, 1):
        high = (value >> shift) > 0
        result = result + high.to(value.dtype) * shift
        value = torch.where(high, value >> shift, value)
    return result + (value > 0).to(result.dtype)


_GPU_SPATIAL = False


def _gpu_spatial():
    """Fused Triton kernels, or None when Triton is unavailable."""
    global _GPU_SPATIAL
    if _GPU_SPATIAL is False:
        try:
            import tetris_gpu_spatial as module
        except ImportError:
            module = None
        _GPU_SPATIAL = module
    return _GPU_SPATIAL


def adjacency_statistics(boards, targets):
    if boards.ndim != 2 or boards.shape[1] != neural.game.AREA:
        raise ValueError('Expected [batch, 4096] vertical-column masks')
    if targets.ndim != 1 or targets.shape[0] != boards.shape[0]:
        raise ValueError('Expected one target level per board')
    kernels = _gpu_spatial() if boards.is_cuda else None
    if kernels is not None:
        return kernels.adjacency_statistics(boards, targets)
    return reference_adjacency_statistics(boards, targets)


def reference_adjacency_statistics(boards, targets):
    """Tensor-op reference used on CPU and for kernel parity tests."""
    masks = boards.to(torch.int32).reshape(-1, 16, 16, 16)
    heights = _bit_length16(masks)
    buried = (torch.bitwise_left_shift(torch.ones_like(heights), heights) - 1) ^ masks
    target_layers = (masks >> targets.to(torch.int32)[:, None, None, None]) & 1
    axis_results = []
    for axis in (3, 2, 1):
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        sl, sr = tuple(left), tuple(right)
        height_delta = (heights[sl] - heights[sr]).abs().float()
        occ_xor = masks[sl] ^ masks[sr]
        gap_xor = buried[sl] ^ buried[sr]
        layer_xor = target_layers[sl] ^ target_layers[sr]
        axis_results.append(torch.stack((
            height_delta.mean((1, 2, 3)) / 16,
            height_delta.square().mean((1, 2, 3)) / 256,
            (height_delta > 1).float().mean((1, 2, 3)),
            _bitcount16(occ_xor).float().mean((1, 2, 3)) / 16,
            _bitcount16(gap_xor).float().mean((1, 2, 3)) / 16,
            layer_xor.float().mean((1, 2, 3)),
        ), dim=1))
    return torch.stack(axis_results, dim=1)


def _compact_full_layers(columns):
    """Clear full Z hyperlayers of [rows, 4096] int32 masks, matching game.clear_layers."""
    counts = torch.stack([((columns >> z) & 1).sum(1) for z in range(16)], dim=1)
    full = counts == columns.shape[1]
    if not bool(full.any()):
        return columns
    for z in range(15, -1, -1):
        rows = full[:, z, None]
        low = (1 << z) - 1
        columns = torch.where(rows, (columns & low) | ((columns >> (z + 1)) << z), columns)
    return columns


@torch.no_grad()
def device_afterstate_descriptors(boards, targets, cells, cell_mask, valid, chunk_rows=2048):
    """Batched device equivalent of afterstate_descriptors.

    boards [n,4096] int, targets [n], cells [n,S,C,4] absolute (x,y,z,w) placed
    cells, cell_mask [n,S,C], valid [n,S]. Returns ([n,6], [n,S,18]) float32.
    """
    n, shortlist = valid.shape
    boards = boards.to(torch.int32)
    targets = targets.to(torch.int32)
    initial = adjacency_statistics(boards, targets)
    cells = cells.long()
    index = (cells[..., 3] * 16 + cells[..., 1]) * 16 + cells[..., 0]
    mask = cell_mask & valid[..., None]
    bits = torch.where(mask, torch.bitwise_left_shift(torch.ones_like(index), cells[..., 2]), 0).to(torch.int32)
    kernels = _gpu_spatial() if boards.is_cuda else None
    if kernels is not None:
        owners = torch.arange(n, device=boards.device, dtype=torch.int32).repeat_interleave(shortlist)
        after = kernels.afterstate_statistics(boards, owners, targets,
                                              index.masked_fill(~mask, -1).reshape(n * shortlist, -1),
                                              bits.reshape(n * shortlist, -1))
        impacts = (after.reshape(n, shortlist, SPATIAL_ACTOR_FEATURES)
                   - initial.reshape(n, 1, SPATIAL_ACTOR_FEATURES)).masked_fill(~valid[..., None], 0.)
        return initial.mean(dim=1), impacts
    index = index.masked_fill(~mask, 0)
    flat_index = index.reshape(n * shortlist, -1)
    flat_bits = bits.reshape(n * shortlist, -1)
    flat_targets = targets[:, None].expand(n, shortlist).reshape(-1)
    owners = torch.arange(n, device=boards.device)[:, None].expand(n, shortlist).reshape(-1)
    impacts = torch.empty((n * shortlist, 3, 6), dtype=torch.float32, device=boards.device)
    for start in range(0, n * shortlist, chunk_rows):
        stop = min(start + chunk_rows, n * shortlist)
        after = boards.index_select(0, owners[start:stop]).clone()
        # Placed cells never overlap occupied bits and differ in Z within a column, so sum == OR.
        after.scatter_add_(1, flat_index[start:stop], flat_bits[start:stop])
        after = _compact_full_layers(after)
        impacts[start:stop] = (adjacency_statistics(after, flat_targets[start:stop])
                               - initial.index_select(0, owners[start:stop]))
    impacts = impacts.reshape(n, shortlist, SPATIAL_ACTOR_FEATURES).masked_fill(~valid[..., None], 0.)
    return initial.mean(dim=1), impacts


@torch.no_grad()
def afterstate_descriptors(board, options, target):
    if not options:
        raise ValueError('Cannot encode an empty candidate shortlist')
    original = tuple(int(value) for value in board)
    before = torch.as_tensor(np.asarray([original], dtype=np.int32))
    levels = torch.as_tensor([int(target)], dtype=torch.int32)
    initial = adjacency_statistics(before, levels)
    afterstates = []
    for option in options:
        placed = neural.game.merge_rigid_piece(original, neural.game.placed_cells(option), option.piece)
        compacted, _ = neural.game.clear_layers(placed)
        afterstates.append(compacted)
    after = torch.from_numpy(np.asarray(afterstates, dtype=np.int32))
    targets = torch.full((len(options),), int(target), dtype=torch.int32)
    impacts = adjacency_statistics(after, targets) - initial
    return (initial.mean(dim=1)[0].numpy().astype(np.float32),
            impacts.reshape(len(options), SPATIAL_ACTOR_FEATURES).numpy().astype(np.float32))


class SurvivalSpatialPolicy(SurvivalFeaturePolicy):
    def __init__(self, config: neural.Config):
        if config.architecture != ARCHITECTURE:
            raise ValueError('SurvivalSpatialPolicy requires survival-spatial-v4 architecture')
        super().__init__(replace(config, architecture=V3_ARCHITECTURE))
        self.config = config
        self.spatial_actor = nn.Linear(SPATIAL_ACTOR_FEATURES, 64, bias=False)
        nn.init.zeros_(self.spatial_actor.weight)
        state_width = 2 * neural.CHANNELS + neural.CONTEXT + SPATIAL_CRITIC_FEATURES
        self.value = nn.Sequential(nn.LayerNorm(state_width), nn.Linear(state_width, 128), nn.SiLU(),
                                   nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def encode(self, boards, targets, context):
        spatial = neural.encode_columns(boards, targets, fused=self.fused_encoding)
        pooled = torch.cat((spatial.mean((2, 3, 4)), spatial.amax((2, 3, 4)),
                            adjacency_statistics(boards, targets).mean(dim=1)), dim=1)
        joined = torch.cat((pooled, context), dim=1)
        value = self.critic(joined)
        return spatial, pooled, value, value.new_empty((len(value), 0)), spatial

    def rank_chunk(self, spatial, pooled, context, features, indices, column_mask,
                   spatial_delta, state_hidden=None):
        local = self.local_statistics(spatial, indices, column_mask)
        candidates = self.candidate(torch.cat((features, local), dim=-1))
        if self.config.spatial_actor:
            candidates = candidates + self.spatial_actor(spatial_delta * SPATIAL_DELTA_SCALE)
        if state_hidden is None:
            state_hidden = self.actor_state(torch.cat((pooled[:, :2 * neural.CHANNELS], context), dim=1))
        state = state_hidden[:, None].expand(-1, features.shape[1], -1)
        hidden = F.silu(self.rank1(torch.cat((state, candidates), dim=-1)))
        return self.rank_out(hidden).squeeze(-1), candidates, hidden

    def raw_scores(self, batch):
        valid = batch['valid'].bool()
        column_mask = batch['column_mask'].bool() & valid[..., None]
        if not bool(valid.any(-1).all() & ((~valid) | column_mask.any(-1)).all()):
            raise ValueError('Every state needs a valid candidate with an affected column')
        if 'spatial_delta' not in batch or batch['spatial_delta'].shape[:2] != valid.shape:
            raise ValueError('V4 requires candidate afterstate spatial descriptors')
        spatial, pooled, value, completion, _ = self.encode(batch['boards'], batch['targets'], batch['context'])
        state_hidden = self.actor_state(torch.cat((pooled[:, :2 * neural.CHANNELS], batch['context']), dim=1))
        features = batch['features'].masked_fill(~valid[..., None], 0.)
        impacts = batch['spatial_delta'].masked_fill(~valid[..., None], 0.)
        scores, hidden = [], []
        for start in range(0, features.shape[1], self.config.chunk):
            stop = start + self.config.chunk
            raw, _, h = self.rank_chunk(spatial, pooled, batch['context'], features[:, start:stop],
                                        batch['indices'][:, start:stop], column_mask[:, start:stop],
                                        impacts[:, start:stop], state_hidden)
            scores.append(raw)
            hidden.append(h)
        return torch.cat(scores, 1), torch.cat(hidden, 1), pooled, value, completion

    def forward(self, batch, capture=False):
        if capture:
            raise NotImplementedError('V4 uses feature diagnostics rather than legacy CNN activations')
        raw, _, pooled, value, completion = self.raw_scores(batch)
        logits = raw
        if self.config.policy_mode in ('teacher-residual', 'teacher-prior'):
            if 'teacher_prior' not in batch:
                raise ValueError('Hybrid policy requires an exact teacher prior')
            if self.config.policy_mode == 'teacher-residual':
                logits = batch['teacher_prior'].float() + self.config.residual_scale * (2 * torch.sigmoid(raw.float()) - 1)
            else:
                logits = self.config.teacher_prior_scale * batch['teacher_prior'].float() + self.config.residual_scale * raw.float()
        logits = logits.masked_fill(~batch['valid'].bool(), -torch.inf)
        return {'logits': logits, 'raw_logits': raw, 'value': value, 'completion': completion,
                'critic_input': torch.cat((pooled, batch['context']), dim=1).detach()}

    @torch.no_grad()
    def ranking_diagnostics(self, batch):
        result = super().ranking_diagnostics(batch)
        result['architecture'] = ARCHITECTURE
        result['spatial_actor'] = self.config.spatial_actor
        return result


class FeatureView:
    def __init__(self, parent, agent, legacy_class, policy):
        import tkinter as tk
        from tkinter import ttk

        self.agent, self.policy = agent, policy
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(parent)
        self.tabs.grid(row=0, column=0, sticky='nsew')
        learned, legacy = ttk.Frame(self.tabs), ttk.Frame(self.tabs)
        self.tabs.add(learned, text='Survival feature inspector')
        self.tabs.add(legacy, text='Legacy heuristic')
        self.legacy = legacy_class(legacy)
        learned.columnconfigure(0, weight=1)
        learned.rowconfigure(0, weight=1)
        self.display = tk.Text(learned, wrap='word', font='TkFixedFont', state='disabled')
        self.display.grid(row=0, column=0, sticky='nsew')
        self.last = (None, 0)
        self.tabs.bind('<<NotebookTabChanged>>', lambda event: self.update(*self.last, force=True))

    def update(self, move, w=0, force=False):
        self.last = (move, w)
        if self.tabs.index(self.tabs.select()) == 1:
            self.legacy.update(move, w)
            return
        trace = self.agent.get_trace()
        if not trace or not trace.get('feature_inspector'):
            content = 'No feature-policy decision recorded yet.'
        else:
            config = self.agent.model.config
            raw = trace.get('actor_score')
            teacher = trace.get('teacher_prior')
            geometry = trace.get('spatial_delta')
            value = trace.get('critic_value')
            semantics = self.agent.metadata.get('value_semantics', 'not calibrated')
            parts = [
                f"Architecture: {config.architecture}",
                f"Current policy: {self.policy()}",
                f"Move: {trace['piece_number']} / {trace['piece']}",
                f"Selection: {trace['selected']} of {trace['shortlist']} candidates",
                f"Total selected logit: {trace['logit']:.6g}",
                f"Raw actor score: {raw:.6g}" if raw is not None else "Raw actor score: not decomposed",
                f"Teacher prior: {teacher:.6g}" if teacher is not None else "Teacher prior: unavailable",
                f"Scaled teacher contribution: {teacher * config.teacher_prior_scale:.6g}" if teacher is not None and config.policy_mode == 'teacher-prior' else "Teacher contribution: depends on policy mode",
                f"Critic value: {value:.6g}" if value is not None else "Critic value: unavailable",
                f"Value semantics: {semantics}",
            ]
            if geometry is not None:
                parts.append('Afterstate adjacency changes (with layer compaction):')
                for axis, offset in zip(AXIS_NAMES, range(0, SPATIAL_ACTOR_FEATURES, 6)):
                    parts.append(axis + ': ' + ', '.join(
                        f'{name}={geometry[offset + index]:+.5f}'
                        for index, name in enumerate(STATISTIC_NAMES)))
            content = '\n'.join(parts)
        self.display.configure(state='normal')
        self.display.delete('1.0', 'end')
        self.display.insert('1.0', content)
        self.display.configure(state='disabled')
