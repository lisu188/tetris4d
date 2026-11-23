"""Spatial survival actor/critic built on V3 fixed descriptors."""
from __future__ import annotations

import copy
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
# Optional critic input (issue #38): for each 4x4x4 block of (x, y, w) columns, the
# maximum height, mean height and mean holes per column, each divided by 16.
REGION_SIDE = 4
REGIONAL_CRITIC_FEATURES = 3 * (16 // REGION_SIDE) ** 3


def regional_statistics(boards):
    """[batch, 4096] column masks -> [batch, 192] fixed local-structure features."""
    masks = boards.to(torch.int32)
    heights = _bit_length16(masks)
    holes = heights - _bitcount16(masks)
    blocks = 16 // REGION_SIDE

    def regions(values):  # column index = (w * 16 + y) * 16 + x
        grid = values.float().reshape(-1, blocks, REGION_SIDE, blocks, REGION_SIDE, blocks, REGION_SIDE)
        return grid.permute(0, 1, 3, 5, 2, 4, 6).reshape(len(values), blocks ** 3, REGION_SIDE ** 3)

    height_blocks, hole_blocks = regions(heights), regions(holes)
    return torch.cat((height_blocks.amax(-1), height_blocks.mean(-1), hole_blocks.mean(-1)), dim=1) / 16


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


@torch.no_grad()
def afterstate_boards(boards, cells, cell_mask):
    """Place each row's piece and clear full layers: boards [k,4096] int, cells [k,C,4]
    absolute (x, y, z, w), cell_mask [k,C] -> [k,4096] int32, matching game.clear_layers(merge)."""
    boards = boards.to(torch.int32)
    cells = cells.long()
    index = ((cells[..., 3] * 16 + cells[..., 1]) * 16 + cells[..., 0]).masked_fill(~cell_mask, 0)
    bits = torch.where(cell_mask, torch.bitwise_left_shift(torch.ones_like(index), cells[..., 2]), 0)
    # Placed cells never overlap occupied bits and differ in Z within a column, so sum == OR.
    return _compact_full_layers(boards.clone().scatter_add_(1, index, bits.to(torch.int32)))


@torch.no_grad()
def device_afterstate_context(after, targets, piece_rows, piece_phase):
    """Exact device mirror of neural.state_context for afterstates.

    after [k,4096] int32, targets [k], piece_rows [k, 3*len(NAMES)] (current, following,
    remaining one-hots), piece_phase [k] = (pieces % len(set)) / len(set). Values are formed
    in float64 and rounded once to float32, as the CPU path does.
    """
    masks = after.to(torch.int32)
    heights = _bit_length16(masks).long()
    holes = (heights - _bitcount16(masks).long()).sum(1)
    levels = torch.arange(16, device=masks.device, dtype=torch.int32)[None, :, None]
    bits = (masks[:, None, :] >> levels) & 1
    above = (masks[:, None, :] >> (levels + 1)) != 0
    area = float(neural.game.AREA)
    fill = bits.sum(2).double() / area
    buried = ((bits == 0) & above).sum(2).double() / area
    glob = torch.stack((targets.double() / 15, holes.double() / 65536, (heights.sum(1) - holes).double() / 65536,
                        heights.amax(1).double() / 16, piece_phase.double()), dim=1)
    return torch.cat((piece_rows.double(), fill, buried, glob), dim=1).float()


def afterstate_piece_features(state):
    """(one-hot row, phase) of the state reached after placing state.current."""
    game = neural.game
    names = game.PIECE_SETS[state.piece_set]
    counts = dict(state.piece_counts)
    counts[state.current] += 1
    pieces = state.pieces + 1
    completed = pieces // len(names)
    upcoming = copy.deepcopy(state.stream).pop()  # the stream does not depend on the move
    row = ([float(n == state.next_piece) for n in neural.NAMES] + [float(n == upcoming) for n in neural.NAMES]
           + [float(n in names and counts[n] == completed) for n in neural.NAMES])
    return row, (pieces % len(names)) / len(names)


@torch.no_grad()
def afterstate_values(model, states, targets, candidates, chunk=64):
    """Critic value of every candidate's afterstate (issue #37), batched on the model's device.

    states: Games; targets: their target layers; candidates: one list of game.Candidate per
    state. Returns one 1-D float tensor of values per state.
    """
    game = neural.game
    device = next(model.parameters()).device
    owners = [s for s, options in enumerate(candidates) for _ in options]
    flat = [c for options in candidates for c in options]
    if not flat:
        return [torch.empty(0, device=device) for _ in states]
    width = max(len(game.SHAPES[c.piece][c.rotation].cells) for c in flat)
    cells = torch.zeros((len(flat), width, 4), dtype=torch.int64)
    mask = torch.zeros((len(flat), width), dtype=torch.bool)
    for row, c in enumerate(flat):
        placed = game.placed_cells(c)
        cells[row, :len(placed)] = torch.tensor(placed)
        mask[row, :len(placed)] = True
    owner = torch.tensor(owners, device=device)
    boards = torch.tensor([s.board for s in states], dtype=torch.int32, device=device).index_select(0, owner)
    after = afterstate_boards(boards, cells.to(device), mask.to(device))
    after_targets = [targets[o] for o in owners]
    clearing = [row for row, c in enumerate(flat) if c.layers]
    if clearing:  # rare: retarget exactly like train_overnight.apply_move
        host = after.index_select(0, torch.tensor(clearing, device=device)).cpu().tolist()
        for row, board in zip(clearing, host):
            following = game.next_hyperlayer_target(tuple(board), after_targets[row])
            if following is not None:
                after_targets[row] = following
    features = [afterstate_piece_features(s) for s in states]
    piece_rows = torch.tensor([features[o][0] for o in owners], dtype=torch.float64, device=device)
    phase = torch.tensor([features[o][1] for o in owners], dtype=torch.float64, device=device)
    target_tensor = torch.tensor(after_targets, device=device)
    context = device_afterstate_context(after, target_tensor, piece_rows, phase)
    values = torch.cat([model.encode(after[i:i + chunk], target_tensor[i:i + chunk], context[i:i + chunk])[2].float()
                        for i in range(0, len(flat), chunk)])
    return list(values.split([len(options) for options in candidates]))


class SurvivalSpatialPolicy(SurvivalFeaturePolicy):
    def __init__(self, config: neural.Config):
        if config.architecture != ARCHITECTURE:
            raise ValueError('SurvivalSpatialPolicy requires survival-spatial-v4 architecture')
        super().__init__(replace(config, architecture=V3_ARCHITECTURE))
        self.config = config
        self.spatial_actor = nn.Linear(SPATIAL_ACTOR_FEATURES, 64, bias=False)
        nn.init.zeros_(self.spatial_actor.weight)
        state_width = (2 * neural.CHANNELS + neural.CONTEXT + SPATIAL_CRITIC_FEATURES
                       + (REGIONAL_CRITIC_FEATURES if config.regional_critic else 0))
        self.value = nn.Sequential(nn.LayerNorm(state_width), nn.Linear(state_width, 128), nn.SiLU(),
                                   nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))

    def encode(self, boards, targets, context):
        spatial = neural.encode_columns(boards, targets, fused=self.fused_encoding)
        parts = [spatial.mean((2, 3, 4)), spatial.amax((2, 3, 4)), adjacency_statistics(boards, targets).mean(dim=1)]
        if self.config.regional_critic:
            parts.append(regional_statistics(boards).to(parts[0].dtype))
        pooled = torch.cat(parts, dim=1)
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

    def _rank_candidates(self, batch, spatial, pooled, keep_hidden):
        valid = batch['valid'].bool()
        column_mask = batch['column_mask'].bool() & valid[..., None]
        if not bool(valid.any(-1).all() & ((~valid) | column_mask.any(-1)).all()):
            raise ValueError('Every state needs a valid candidate with an affected column')
        if 'spatial_delta' not in batch or batch['spatial_delta'].shape[:2] != valid.shape:
            raise ValueError('V4 requires candidate afterstate spatial descriptors')
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
            if keep_hidden:
                hidden.append(h)
        return torch.cat(scores, 1), torch.cat(hidden, 1) if keep_hidden else None

    def raw_scores(self, batch):
        spatial, pooled, value, completion, _ = self.encode(batch['boards'], batch['targets'], batch['context'])
        raw, hidden = self._rank_candidates(batch, spatial, pooled, keep_hidden=True)
        return raw, hidden, pooled, value, completion

    def actor_logits(self, batch):
        """Rank candidates without computing the regional critic or retaining hidden activations."""
        spatial = neural.encode_columns(batch['boards'], batch['targets'], fused=self.fused_encoding)
        pooled = torch.cat((spatial.mean((2, 3, 4)), spatial.amax((2, 3, 4))), dim=1)
        raw, _ = self._rank_candidates(batch, spatial, pooled, keep_hidden=False)
        return self._policy_logits(raw, batch)

    def _policy_logits(self, raw, batch):
        logits = raw
        if self.config.policy_mode in ('teacher-residual', 'teacher-prior'):
            if 'teacher_prior' not in batch:
                raise ValueError('Hybrid policy requires an exact teacher prior')
            if self.config.policy_mode == 'teacher-residual':
                logits = batch['teacher_prior'].float() + self.config.residual_scale * (2 * torch.sigmoid(raw.float()) - 1)
            else:
                logits = self.config.teacher_prior_scale * batch['teacher_prior'].float() + self.config.residual_scale * raw.float()
        return logits.masked_fill(~batch['valid'].bool(), -torch.inf)

    def forward(self, batch, capture=False):
        if capture:
            raise NotImplementedError('V4 uses feature diagnostics rather than legacy CNN activations')
        raw, _, pooled, value, completion = self.raw_scores(batch)
        logits = self._policy_logits(raw, batch)
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
            ]
            if 'afterstate_value' in trace:
                parts.append(f"Selected afterstate: {trace['afterstate_value'] * 1000:.1f} predicted remaining placements")
            elif 'base_logit' in trace:
                parts.extend([f"Frozen V4 logit: {trace['base_logit']:.6g}",
                              f"Afterstate residual: {trace['residual']:.6g}",
                              f"Policy temperature: {config.policy_temperature:.6g}"])
            else:
                parts.extend([
                    f"Raw actor score: {raw:.6g}" if raw is not None else "Raw actor score: not decomposed",
                    f"Teacher prior: {teacher:.6g}" if teacher is not None else "Teacher prior: unavailable",
                    f"Scaled teacher contribution: {teacher * config.teacher_prior_scale:.6g}" if teacher is not None and config.policy_mode == 'teacher-prior' else "Teacher contribution: depends on policy mode",
                ])
            parts.extend([f"Critic value: {value:.6g}" if value is not None else "Critic value: unavailable",
                          f"Value semantics: {semantics}"])
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
