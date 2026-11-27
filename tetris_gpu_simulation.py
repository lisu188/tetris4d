"""Optional batched CUDA preparation with the native candidate contract.

CPU code supplies the existing Python RNG draws and commits selected moves.
Boards are uploaded once per batch. Candidate expansion/scoring, shortlisting,
context and model features run on CUDA. A single small summary transfer supplies
the counts needed by Python's exact exploration sampler. Clear candidates and
unusual fallback/recovery states use the reference preparer per environment.
Fresh output tensors may be retained for a whole rollout. This module does not
change any default backend and does not claim GPU-resident game progression.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math
import random

import numpy as np
import torch
import triton
import triton.language as tl

import sparse_tetris_4d_optimized as game
import tetris_neural as neural

MAX_GPU_CANDIDATES = 4096
MAX_GPU_SHORTLIST = 256
SUMMARY_KERNEL_LIMIT = 16384
# The expanded catalogue ranges from 2,744 to 648,000 exhaustive placements.
# Three groups avoid sorting mostly padded rows while retaining GPU parallelism.
EXHAUSTIVE_GROUP_LIMITS = (115_000, 350_000)


@triton.jit
def _board_kernel(boards, heights, counts, contexts, scores, statistics,
                  targets, weights, ticks_low, ticks_high,
                  CONTEXT: tl.constexpr, BLOCK: tl.constexpr):
    env = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    mask = tl.load(boards + env * 4096 + column).to(tl.int32)
    height = 32 - tl.extra.cuda.libdevice.clz(mask)
    tl.store(heights + env * 4096 + column, height)
    height_sum = tl.sum(height, 0)
    mass = tl.sum(tl.extra.cuda.libdevice.popc(mask), 0)
    holes = height_sum - mass
    maximum = tl.max(height, 0)
    target = tl.load(targets + env).to(tl.int32)
    filled = tl.full((), 0, tl.int32)
    buried = tl.full((), 0, tl.int32)
    for level in tl.static_range(16):
        on = (mask >> level) & 1
        n = tl.sum(on, 0)
        under = tl.sum(((on == 0) & ((mask >> (level + 1)) != 0)).to(tl.int32), 0)
        tl.store(counts + env * 16 + level, n)
        tl.store(contexts + env * CONTEXT + 45 + level, n.to(tl.float32) / 4096)
        tl.store(contexts + env * CONTEXT + 61 + level, under.to(tl.float32) / 4096)
        filled = tl.where(target == level, n, filled)
        buried = tl.where(target == level, under, buried)
    # The native base roughness is an exact integer tick sum divided by a
    # power of two. Its numerator exceeds int64; two 32-bit limbs stay exact.
    low = tl.full((), 0, tl.int64)
    high = tl.full((), 0, tl.int64)
    for axis in tl.static_range(3):
        step = 1 << (axis * 4)
        edge = ((column >> (axis * 4)) & 15) < 15
        other_mask = tl.load(boards + env * 4096 + column + step, edge, 0).to(tl.int32)
        other_height = 32 - tl.extra.cuda.libdevice.clz(other_mask)
        difference = tl.abs(other_height - height)
        lo = tl.load(ticks_low + difference, edge, 0).to(tl.int64)
        hi = tl.load(ticks_high + difference, edge, 0).to(tl.int64)
        low += tl.sum(lo, 0)
        high += tl.sum(hi, 0)
    high += low >> 32
    low &= 4294967295
    numerator = high.to(tl.float64) * 4294967296.0 + low.to(tl.float64)
    denominator = tl.load(weights + 3)
    roughness = numerator / denominator
    score = tl.load(weights) * height_sum.to(tl.float64)
    score = score + tl.load(weights + 1) * holes.to(tl.float64)
    score = score + tl.load(weights + 2) * roughness
    tl.store(scores + env, score)
    tl.store(contexts + env * CONTEXT + 77, target.to(tl.float64) / 15.0)
    tl.store(contexts + env * CONTEXT + 78, holes.to(tl.float32) / 65536)
    tl.store(contexts + env * CONTEXT + 79, mass.to(tl.float32) / 65536)
    tl.store(contexts + env * CONTEXT + 80, maximum.to(tl.float32) / 16)
    tl.store(statistics + env * 4, filled)
    tl.store(statistics + env * 4 + 1, buried)
    tl.store(statistics + env * 4 + 2, height_sum)
    tl.store(statistics + env * 4 + 3, holes)


@triton.jit
def _candidate_kernel(boards, heights, counts, statistics, shapes, templates,
                      starts, strides, inverses, piece_ids, piece_offsets,
                      sample_counts, vertical_rotations, targets, board_scores,
                      rough_delta, weights, specs_out, scores_out, gains_out,
                      future_out, delta_out, clear_out, valid_out,
                      CANDIDATES: tl.constexpr, SAMPLES: tl.constexpr, TEMPLATE_STRIDE: tl.constexpr,
                      ROTATIONS: tl.constexpr, PROTECT: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    env = tl.program_id(1)
    live = row < CANDIDATES
    piece = tl.load(piece_ids + env)
    samples = tl.load(sample_counts + piece)
    sampled = live & (row < samples)
    anchored = live & (row >= SAMPLES) & (row < SAMPLES + 4096) & (piece == 0)
    rotation = tl.load(templates + (piece * TEMPLATE_STRIDE + row) * 2, sampled, 0)
    local = tl.load(templates + (piece * TEMPLATE_STRIDE + row) * 2 + 1, sampled, 0)
    vertical = tl.load(vertical_rotations + piece)
    rotation = tl.where(anchored, vertical, rotation)
    shape = (tl.load(piece_offsets + piece) + rotation) * 182
    nx = 17 - tl.load(shapes + shape, live, 1)
    ny = 17 - tl.load(shapes + shape + 1, live, 1)
    nw = 17 - tl.load(shapes + shape + 3, live, 1)
    total = nx * ny * nw
    packet = env * ROTATIONS + rotation
    start = tl.load(starts + packet, live, 0)
    stride = tl.load(strides + packet, live, 1)
    position = (start + local * stride) % total
    position = tl.where(anchored, row - SAMPLES, position)
    x = position % nx
    y = (position // nx) % ny
    w = position // (nx * ny)
    base = x + y * 16 + w * 256
    target = tl.load(targets + env).to(tl.int32)
    old_mask = tl.load(boards + env * 4096 + base, anchored, 0)
    anchored &= ((old_mask >> target) & 1) == 0
    inverse = tl.load(inverses + packet, anchored, 1)
    sampled_count = tl.load(inverses + packet + ROTATIONS * tl.num_programs(1), anchored, 0)
    duplicate = (((position - start + total) * inverse) % total) < sampled_count
    live &= sampled | (anchored & ~duplicate)
    columns = tl.load(shapes + shape + 4, live, 0)
    extent_z = tl.load(shapes + shape + 2, live, 0)
    z = tl.full((BLOCK,), -16, tl.int32)
    for i in range(8):
        present = live & (i < columns)
        offset = tl.load(shapes + shape + 22 + i * 10, present, 0)
        bottom = tl.load(shapes + shape + 24 + i * 10, present, 0)
        old = tl.load(heights + env * 4096 + base + offset, present, 0)
        z = tl.maximum(z, tl.where(present, old - bottom, -16))
    legal = live & (z >= 0) & (z + extent_z <= 16)
    relative = target - z
    above = tl.minimum(tl.maximum(relative + 1, 0), 17)
    gain = tl.full((BLOCK,), 0, tl.int32)
    blocked = tl.full((BLOCK,), False, tl.int1)
    future = tl.full((BLOCK,), 0, tl.int32)
    delta = tl.full((BLOCK,), 0, tl.int32)
    rough = tl.full((BLOCK,), 0.0, tl.float64)
    for i in range(8):
        present = live & (i < columns)
        offset = tl.load(shapes + shape + 22 + i * 10, present, 0)
        mask = tl.load(shapes + shape + 23 + i * 10, present, 0)
        top = tl.load(shapes + shape + 25 + i * 10, present, 0) + z
        index = base + offset
        old = tl.load(heights + env * 4096 + index, present, 0)
        on_target = (relative >= 0) & (relative < 16) & (
            ((mask >> tl.minimum(tl.maximum(relative, 0), 16)) & 1) != 0)
        gap = old <= target
        gain += (present & gap & on_target).to(tl.int32)
        blocked |= present & gap & (top > target) & ~on_target
        future += tl.where(present, tl.maximum(top - tl.maximum(old, target + 1), 0), 0)
        future -= tl.extra.cuda.libdevice.popc((mask >> above).to(tl.int32))
        delta += tl.where(present, top - old, 0)
        # Preserve t4_scores' column/direction FP64 accumulation order.
        for d in tl.static_range(6):
            coordinate = (index >> ((d // 2) * 4)) & 15
            step = (-1 if d % 2 == 0 else 1) * (1 << ((d // 2) * 4))
            neighbor = index + step
            partner = tl.load(shapes + shape + 26 + i * 10 + d, present, -1)
            edge = present & legal & ((coordinate > 0) if d % 2 == 0 else (coordinate < 15))
            edge &= (partner < 0) | (neighbor > index)
            neighbor_old = tl.load(heights + env * 4096 + neighbor, edge, 0)
            partner_top = tl.load(
                shapes + shape + 25 + tl.maximum(partner, 0) * 10,
                edge & (partner >= 0), 0) + z
            neighbor_new = tl.where(partner >= 0, partner_top, neighbor_old)
            before = tl.abs(neighbor_old - old)
            after = tl.abs(neighbor_new - top)
            rough += tl.load(rough_delta + after * 17 + before, edge, 0.0)
    cleared = tl.full((BLOCK,), 0, tl.int32)
    for dz in tl.static_range(16):
        amount = tl.load(shapes + shape + 6 + dz, legal & (dz < extent_z), 0)
        count = tl.load(counts + env * 16 + z + dz, legal & (dz < extent_z), 0)
        cleared |= tl.where(legal & (amount > 0) & (count + amount == 4096), 1 << (z + dz), 0)
    size = tl.load(shapes + shape + 5, live, 0)
    score = tl.load(board_scores + env) + (tl.load(weights) + tl.load(weights + 1)) * delta.to(tl.float64)
    score = score - size.to(tl.float64) * tl.load(weights + 1)
    score = score + tl.load(weights + 2) * rough
    # Clear rows force the environment's CPU reference path before selection.
    score = tl.where(cleared != 0, 0.0, score)
    valid = legal & (~blocked if PROTECT else True)
    tl.store(scores_out + env * CANDIDATES + row, tl.where(valid, score, -float("inf")), row < CANDIDATES)
    tl.store(gains_out + env * CANDIDATES + row, gain, row < CANDIDATES)
    tl.store(future_out + env * CANDIDATES + row, future, row < CANDIDATES)
    tl.store(delta_out + env * CANDIDATES + row, delta, row < CANDIDATES)
    tl.store(clear_out + env * CANDIDATES + row, cleared, row < CANDIDATES)
    tl.store(valid_out + env * CANDIDATES + row, valid, row < CANDIDATES)
    for field in tl.static_range(6):
        value = rotation if field == 0 else x if field == 1 else y if field == 2 else w if field == 3 else z if field == 4 else 0
        tl.store(specs_out + (env * CANDIDATES + row) * 6 + field, value, row < CANDIDATES)


@triton.jit
def _clear_score_kernel(boards, owners, cell_index, cell_bits, scratch, ticks_low, ticks_high,
                        out_int, out_limbs, CELLS: tl.constexpr, BLOCK: tl.constexpr):
    """Exact afterstate statistics of a layer-clearing placement (merge, compact, score terms)."""
    row = tl.program_id(0).to(tl.int64)
    owner = tl.load(owners + row).to(tl.int64)
    column = tl.arange(0, BLOCK)
    mask = tl.load(boards + owner * BLOCK + column).to(tl.int32)
    for cell in tl.static_range(CELLS):
        index = tl.load(cell_index + row * CELLS + cell)
        bit = tl.load(cell_bits + row * CELLS + cell)
        mask = tl.where(column == index, mask | bit, mask)
    full = tl.zeros((), dtype=tl.int32)
    for level in tl.static_range(16):
        count = tl.sum((mask >> level) & 1, axis=0)
        full = full | tl.where(count == BLOCK, 1 << level, 0)
    for layer in tl.static_range(15, -1, -1):
        compacted = (mask & ((1 << layer) - 1)) | ((mask >> (layer + 1)) << layer)
        mask = tl.where(((full >> layer) & 1) != 0, compacted, mask)
    base = scratch + row * BLOCK
    tl.store(base + column, mask)
    tl.debug_barrier()
    height = 32 - tl.extra.cuda.libdevice.clz(mask)
    height_sum = tl.sum(height, 0)
    mass = tl.sum(tl.extra.cuda.libdevice.popc(mask), 0)
    low = tl.full((), 0, tl.int64)
    high = tl.full((), 0, tl.int64)
    for axis in tl.static_range(3):
        step = 1 << (axis * 4)
        edge = ((column >> (axis * 4)) & 15) < 15
        other = tl.load(base + column + step, edge, 0).to(tl.int32)
        difference = tl.abs(32 - tl.extra.cuda.libdevice.clz(other) - height)
        low += tl.sum(tl.load(ticks_low + difference, edge, 0).to(tl.int64), 0)
        high += tl.sum(tl.load(ticks_high + difference, edge, 0).to(tl.int64), 0)
    high += low >> 32
    low &= 4294967295
    tl.store(out_int + row * 3, height_sum)
    tl.store(out_int + row * 3 + 1, height_sum - mass)
    tl.store(out_int + row * 3 + 2, tl.extra.cuda.libdevice.popc(full))
    tl.store(out_limbs + row * 2, low)
    tl.store(out_limbs + row * 2 + 1, high)


@triton.jit
def _summary_kernel(valid, clears, statistics, counts, piece_ids, sample_counts,
                    targets, summary, CANDIDATES: tl.constexpr, BLOCK: tl.constexpr):
    env = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    good = tl.load(valid + env * CANDIDATES + row, row < CANDIDATES, False)
    cleared = tl.load(clears + env * CANDIDATES + row, row < CANDIDATES, 0)
    tl.store(summary + env * 5, tl.sum(good.to(tl.int32), 0))
    tl.store(summary + env * 5 + 1, tl.max((good & (cleared != 0)).to(tl.int32), 0))
    tl.store(summary + env * 5 + 2, tl.load(statistics + env * 4 + 1))
    filled = tl.load(statistics + env * 4)
    tl.store(summary + env * 5 + 3, filled)
    piece = tl.load(piece_ids + env)
    evaluations = tl.load(sample_counts + piece) + tl.where(piece == 0, 4096 - filled, 0)
    tl.store(summary + env * 5 + 4, evaluations)


@triton.jit
def _features_kernel(chosen, specs, scores, gains, future, deltas, board_scores,
                     targets, piece_ids, piece_offsets, shapes, shape_features,
                     lengths, features, indices, column_mask, option_specs, option_scores,
                     CANDIDATES: tl.constexpr, SHORTLIST: tl.constexpr, BLOCK: tl.constexpr):
    env = tl.program_id(0)
    candidate = tl.program_id(1)
    feature = tl.arange(0, BLOCK)
    length = tl.load(lengths + env)
    live = candidate < length
    row = tl.load(chosen + env * SHORTLIST + candidate)
    source = env * CANDIDATES + tl.minimum(row, CANDIDATES - 1)
    rotation = tl.load(specs + source * 6)
    x = tl.load(specs + source * 6 + 1)
    y = tl.load(specs + source * 6 + 2)
    w = tl.load(specs + source * 6 + 3)
    z = tl.load(specs + source * 6 + 4)
    layers = tl.load(specs + source * 6 + 5)
    shape_id = tl.load(piece_offsets + tl.load(piece_ids + env)) + rotation
    shape = shape_id * 182
    ext_x = tl.load(shapes + shape)
    ext_y = tl.load(shapes + shape + 1)
    ext_z = tl.load(shapes + shape + 2)
    ext_w = tl.load(shapes + shape + 3)
    size = tl.load(shapes + shape + 5)
    delta = tl.load(deltas + source)
    gain = tl.load(gains + source)
    holes = tl.load(future + source)
    score = tl.load(scores + source)
    target = tl.load(targets + env)
    value = tl.full((BLOCK,), 0.0, tl.float64)
    value = tl.where(feature == 0, x.to(tl.float64) / 15.0, value)
    value = tl.where(feature == 1, y.to(tl.float64) / 15.0, value)
    value = tl.where(feature == 2, z.to(tl.float64) / 15.0, value)
    value = tl.where(feature == 3, w.to(tl.float64) / 15.0, value)
    value = tl.where(feature == 4, ext_x.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 5, ext_y.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 6, ext_z.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 7, ext_w.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 8, gain.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 9, holes.to(tl.float64) / 64.0, value)
    value = tl.where(feature == 10, layers.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 11, delta.to(tl.float64) / 64.0, value)
    value = tl.where(feature == 12, (delta - size).to(tl.float64) / 64.0, value)
    value = tl.where(feature == 13, (score - tl.load(board_scores + env)) / 64.0, value)
    value = tl.where(feature == 14, size.to(tl.float64) / 16.0, value)
    value = tl.where(feature == 15, target.to(tl.float64) / 15.0, value)
    cell = tl.load(shape_features + shape_id * 80 + feature - 16, (feature >= 16) & (feature < 96), 0)
    value = tl.where(feature >= 16, cell.to(tl.float64), value).to(tl.float32)
    tl.store(features + (env * SHORTLIST + candidate) * 96 + feature,
             tl.where(live, value, 0), feature < 96)
    columns = tl.load(shapes + shape + 4)
    touched = live & (feature < columns)
    offset = tl.load(shapes + shape + 22 + feature * 10, touched, 0)
    base = x + y * 16 + w * 256
    tl.store(indices + (env * SHORTLIST + candidate) * 16 + feature,
             tl.where(touched, base + offset, 0).to(tl.int64), feature < 16)
    tl.store(column_mask + (env * SHORTLIST + candidate) * 16 + feature, touched, feature < 16)
    for field in tl.static_range(6):
        metadata = tl.load(specs + source * 6 + field)
        tl.store(option_specs + (env * SHORTLIST + candidate) * 6 + field, metadata)
    tl.store(option_scores + env * SHORTLIST + candidate, score)


def _sampling_packets(piece, budget, salt):
    """The exact RNG draws made by game.specifications, without row expansion."""
    packets = []
    for rotation, shape in enumerate(game.SHAPES[piece]):
        nx, ny, nw = (17 - shape.extents[i] for i in (0, 1, 3))
        total = nx * ny * nw
        take = total if not budget else min(total, budget // len(game.SHAPES[piece]) + int(rotation < budget % len(game.SHAPES[piece])))
        start, stride = 0, 1
        if take != total:
            rng = random.Random((salt + 1) * 1000003 + rotation * 9176)
            start, stride = rng.randrange(total), rng.randrange(1, total)
            while math.gcd(stride, total) != 1:
                stride = stride % (total - 1) + 1
        packets.append((start, stride, pow(stride, -1, total), take))
    return packets


class _DeviceOptions(Sequence):
    def __init__(self, piece, specifications, scores):
        self.piece, self.specifications, self.scores = piece, specifications, scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        values = self.specifications[index].cpu().tolist()
        return game.Candidate(self.piece, *map(int, values), float(self.scores[index].item()))


@dataclass
class GpuPrepared:
    arrays: dict
    options: Sequence
    teacher: torch.Tensor
    evaluations: int
    fallback: bool


@dataclass
class DevicePreparedBatch:
    prepared: tuple
    environment_indices: tuple
    batch: dict
    teachers: torch.Tensor
    cpu_fallbacks: tuple
    targets: tuple
    _option_specs: torch.Tensor | None = None
    _option_scores: torch.Tensor | None = None

    @property
    def env_indices(self):
        return self.environment_indices

    @property
    def records(self):
        return tuple(None if p is None else p.arrays for p in self.prepared)

    @property
    def evaluations(self):
        return tuple(0 if p is None else p.evaluations for p in self.prepared)

    @property
    def fallback(self):
        return tuple(False if p is None else p.fallback for p in self.prepared)

    @property
    def candidate_counts(self):
        return tuple(0 if p is None else len(p.options) for p in self.prepared)

    def selected_metadata(self, indices):
        """Device FP64 rows: rotation, x, y, w, z, layers, native score."""
        if not isinstance(indices, torch.Tensor) or indices.device.type == "cpu":
            host_indices = torch.as_tensor(indices, dtype=torch.int64).reshape(-1).tolist()
            if len(host_indices) != len(self.environment_indices):
                raise ValueError("One action is required for every nonempty environment")
            if any(index < 0 or index >= len(self.prepared[env].options)
                   for env, index in zip(self.environment_indices, host_indices)):
                raise ValueError("Selected candidate index is out of range")
        actions = torch.as_tensor(indices, device=self.teachers.device, dtype=torch.int64).reshape(-1)
        if len(actions) != len(self.environment_indices):
            raise ValueError("One action is required for every nonempty environment")
        if not len(actions):
            return torch.empty((0, 7), dtype=torch.float64, device=self.teachers.device)
        rows = torch.arange(len(actions), device=actions.device)
        integers = self._option_specs[rows, actions]
        scores = self._option_scores[rows, actions]
        return torch.cat((integers.to(torch.float64), scores[:, None]), dim=1)

    def candidates_from_metadata(self, packed):
        if isinstance(packed, torch.Tensor):
            packed = packed.tolist()
        return tuple(game.Candidate(self.prepared[e].options.piece, *map(int, row[:6]), row[6])
                     for e, row in zip(self.environment_indices, packed))

    def selected_candidates(self, indices):
        # The trainer may instead pack selected_metadata with values/logprobs
        # into its existing inference transfer, then call candidates_from_metadata.
        return self.candidates_from_metadata(self.selected_metadata(indices).cpu().tolist())

    def to_cpu_prepared(self):
        result = []
        for p in self.prepared:
            result.append(None if p is None else neural.Prepared(
                {k: v.detach().cpu().numpy() for k, v in p.arrays.items()},
                tuple(p.options), int(p.teacher.item()), p.evaluations, p.fallback))
        return tuple(result)


class GpuPreparer:
    """Per-call owned output tensors; retained rollout records remain valid."""

    def __init__(self, config=None, device="cuda", *, group_exhaustive=False):
        self.config = neural.Config() if config is None else config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GpuPreparer requires CUDA")
        if game.BACKEND != "native":
            raise ValueError("GPU preparation requires the native reference backend")
        self.names = tuple(game.PIECE_NAMES)
        self.rotations = max(map(len, game.SHAPES.values()))
        self.budget = self.config.screening_budget
        self.group_exhaustive = group_exhaustive
        self._teacher_parameters = game.network_parameters()
        self._reference_only_reason = (
            "candidate_budget_bound" if self.budget > MAX_GPU_CANDIDATES else
            "shortlist_bound" if self.config.shortlist > MAX_GPU_SHORTLIST else
            "shortlist_mode" if self.config.shortlist_mode not in ("teacher-random", "teacher-top") else
            "catalog_columns" if any(len(s.columns) > 8 for shapes in game.SHAPES.values() for s in shapes)
            else None)
        self.spatial = self.config.architecture in neural.SPATIAL_ARCHITECTURES
        # Exact CPU fallback rows skip the slow host afterstate pass; V4 descriptors
        # for every row are recomputed on device from the final options.
        self._fallback_config = (replace(self.config, architecture="survival-feature-v3")
                                 if self.spatial else self.config)
        if self._reference_only_reason is not None:
            return
        width = max(len(s.cells) for shapes in game.SHAPES.values() for s in shapes)
        table = np.zeros((len(self.names), self.rotations, width, 4), dtype=np.int32)
        present = np.zeros((len(self.names), self.rotations, width), dtype=np.bool_)
        for i, piece in enumerate(self.names):
            for r, shape in enumerate(game.SHAPES[piece]):
                table[i, r, :len(shape.cells)] = shape.cells
                present[i, r, :len(shape.cells)] = True
        self.cell_table, self.cell_present = self._device(table), self._device(present)
        offsets, shapes, cells, templates, sizes, vertical = [], [], [], [], [], []
        cursor = 0
        for piece in self.names:
            offsets.append(cursor)
            records = game.native_shapes(piece)
            shapes.append(records)
            cells.extend(neural._shape_features(s.cells) for s in game.SHAPES[piece])
            cursor += len(records)
            rows = []
            for rotation, packet in enumerate(_sampling_packets(piece, self.budget, 0)):
                rows.extend((rotation, k) for k in range(packet[3]))
            templates.append(np.asarray(rows, dtype=np.int32))
            sizes.append(len(rows))
            vertical.append(next((i for i, s in enumerate(game.SHAPES[piece]) if len(s.columns) == 1), -1))
        self.samples = max(sizes)
        self._piece_samples = tuple(sizes)
        template = np.zeros((len(self.names), self.samples, 2), dtype=np.int32)
        for i, rows in enumerate(templates):
            template[i, :len(rows)] = rows
        self.shape_offsets = self._device(np.asarray(offsets, dtype=np.int32))
        self.shapes = self._device(np.concatenate(shapes))
        self.shape_features = self._device(np.asarray(cells, dtype=np.float32))
        self.templates = self._device(template)
        self.sample_counts = self._device(np.asarray(sizes, dtype=np.int32))
        self.vertical_rotations = self._device(np.asarray(vertical, dtype=np.int32))
        self.rough_delta = self._device(game.native_roughness_deltas(game.ROUGHNESS_TABLE))
        ticks = game.ROUGHNESS_TICKS
        self.ticks_low = self._device(np.asarray([v & 0xffffffff for v in ticks], dtype=np.int64))
        self.ticks_high = self._device(np.asarray([v >> 32 for v in ticks], dtype=np.int64))
        self.weights = self._device(np.asarray((game.HEIGHT_WEIGHT, game.HOLES_WEIGHT,
                                               game.BUMP_WEIGHT, game.ROUGHNESS_DENOMINATOR), dtype=np.float64))

    def _device(self, value):
        return torch.from_numpy(np.array(value, copy=True)).to(self.device)

    def _score_clearing(self, boards, piece_ids, specifications, scores, clears, valid, chunk=2048):
        """Score layer-clearing candidates exactly on device instead of falling back to CPU.

        Matches game.score_candidate: network_score on the merged, compacted board,
        evaluated in the same float64 operation order.
        """
        clearing = valid & (clears != 0)
        if not bool(clearing.any()):
            return
        env, slot = clearing.nonzero(as_tuple=True)
        spec = specifications[env, slot].long()
        pieces = piece_ids.long().index_select(0, env)
        relative = self.cell_table[pieces, spec[:, 0]].long()
        present = self.cell_present[pieces, spec[:, 0]]
        x, y, w, z = (relative[..., i] + spec[:, j, None] for i, j in ((0, 1), (1, 2), (3, 3), (2, 4)))
        index = torch.where(present, (w * 16 + y) * 16 + x, -1).to(torch.int32)
        bits = torch.where(present, torch.bitwise_left_shift(torch.ones_like(z), z), 0).to(torch.int32)
        rows = len(env)
        integers = torch.empty((rows, 3), dtype=torch.int32, device=self.device)
        limbs = torch.empty((rows, 2), dtype=torch.int64, device=self.device)
        owners = env.to(torch.int32)
        for start in range(0, rows, chunk):
            stop = min(rows, start + chunk)
            scratch = torch.empty((stop - start, 4096), dtype=torch.int32, device=self.device)
            _clear_score_kernel[(stop - start,)](
                boards, owners[start:stop].contiguous(), index[start:stop].contiguous(),
                bits[start:stop].contiguous(), scratch, self.ticks_low, self.ticks_high,
                integers[start:stop], limbs[start:stop], CELLS=index.shape[1], BLOCK=4096, num_warps=8)
        roughness = (limbs[:, 1].to(torch.float64) * 4294967296.0 + limbs[:, 0].to(torch.float64)) / self.weights[3]
        exact = self.weights[0] * integers[:, 0].to(torch.float64)
        exact = exact + self.weights[1] * integers[:, 1].to(torch.float64)
        exact = exact + game.LAYERS_WEIGHT * integers[:, 2].to(torch.float64)
        exact = exact + self.weights[2] * roughness
        scores[env, slot] = exact
        specifications[env, slot, 5] = integers[:, 2]
        clears[env, slot] = 0  # resolved exactly; no reference fallback needed

    def _spatial_arrays(self, boards, targets, piece_ids, option_specs, valid):
        from tetris_survival_v4 import device_afterstate_descriptors
        specs = option_specs.long()
        rotation = torch.where(valid, specs[..., 0], 0)
        pieces = piece_ids.long()[:, None].expand_as(rotation)
        relative = self.cell_table[pieces, rotation]
        present = self.cell_present[pieces, rotation]
        # Shape cells are (dx, dy, dz, dw); specs are (rotation, x, y, w, z, layers).
        origin = torch.stack((specs[..., 1], specs[..., 2], specs[..., 4], specs[..., 3]), dim=-1)
        absolute = (relative + origin[..., None, :]).clamp(0, 15)
        return device_afterstate_descriptors(boards, targets, absolute, present, valid)

    def _reference_batch(self, states, targets, reason):
        references = tuple(neural.prepare(state, target, self.config) for state, target in zip(states, targets))
        envs = tuple(i for i, p in enumerate(references) if p is not None)
        if not envs:
            return DevicePreparedBatch(
                references, (), {}, torch.empty(0, dtype=torch.int64, device=self.device),
                (reason,) * len(states), targets)
        batch = neural.collate([references[i].arrays for i in envs], self.device)
        maximum = batch["features"].shape[1]
        integers = np.zeros((len(envs), maximum, 6), dtype=np.int32)
        scores = np.zeros((len(envs), maximum), dtype=np.float64)
        for row, env in enumerate(envs):
            p = references[env]
            integers[row, :len(p.options)] = [
                (c.rotation, c.x, c.y, c.w, c.z, c.layers) for c in p.options]
            scores[row, :len(p.options)] = [c.score for c in p.options]
        option_specs, option_scores = self._device(integers), self._device(scores)
        teachers = self._device(np.asarray([references[i].teacher for i in envs], dtype=np.int64))
        prepared = [None] * len(states)
        candidate_fields = neural.CANDIDATE_FIELDS
        for row, env in enumerate(envs):
            p = references[env]
            arrays = {name: value[row, :len(p.options)] if name in candidate_fields else value[row]
                      for name, value in batch.items()}
            prepared[env] = GpuPrepared(
                arrays, _DeviceOptions(states[env].current, option_specs[row, :len(p.options)],
                                       option_scores[row, :len(p.options)]),
                teachers[row], p.evaluations, p.fallback)
        return DevicePreparedBatch(
            tuple(prepared), envs, batch, teachers, (reason,) * len(states),
            tuple(targets[i] if p is None else int(p.arrays["targets"]) for i, p in enumerate(references)),
            option_specs, option_scores)

    def prepare_batch(self, states, targets):
        states, targets = tuple(states), tuple(targets)
        if len(states) != len(targets):
            raise ValueError("State and target counts differ")
        if game.BACKEND != "native":
            raise ValueError("GPU preparation requires the native reference backend")
        if game.network_parameters() != self._teacher_parameters:
            raise ValueError("Heuristic parameters changed after GPU preparer initialization")
        if any(type(target) is not int or not 0 <= target < 16 for target in targets):
            raise ValueError("Targets must be integers in [0, 15]")
        if not states:
            return DevicePreparedBatch((), (), {}, torch.empty(0, dtype=torch.int64, device=self.device), (), ())
        if self._reference_only_reason is not None:
            return self._reference_batch(states, targets, self._reference_only_reason)
        piece_ids = [self.names.index(state.current) for state in states]
        if not self.group_exhaustive or self.budget:
            return self._prepare_group(states, targets, piece_ids, self.samples)
        groups = {}
        for env, piece in enumerate(piece_ids):
            bucket = bisect_left(EXHAUSTIVE_GROUP_LIMITS, self._piece_samples[piece])
            groups.setdefault(bucket, []).append(env)
        if len(groups) == 1:
            return self._prepare_group(states, targets, piece_ids, max(self._piece_samples[p] for p in piece_ids))
        prepared = []
        for indices in groups.values():
            pieces = [piece_ids[i] for i in indices]
            width = max(self._piece_samples[p] for p in pieces)
            group = self._prepare_group([states[i] for i in indices], [targets[i] for i in indices], pieces, width)
            prepared.append((indices, group))
        return _merge_prepared_groups(prepared)

    def _prepare_group(self, states, targets, piece_ids, samples):
        n = len(states)
        # The anchored I placements follow the group's sampled region in original
        # board-column order. Removing invalid padding between them preserves ties.
        candidates = max(self.config.shortlist, samples + (4096 if 0 in piece_ids else 0))
        shortlist = min(self.config.shortlist, candidates)
        starts = np.zeros((n, self.rotations), dtype=np.int32)
        strides = np.ones_like(starts)
        inverse_take = np.zeros((2, n, self.rotations), dtype=np.int32)
        contexts = np.zeros((n, neural.CONTEXT), dtype=np.float32)
        packets_by_state = {}
        for env, state in enumerate(states):
            key = (state.current, state.pieces)
            if key not in packets_by_state:
                packets_by_state[key] = _sampling_packets(state.current, self.budget, state.pieces)
            packets = packets_by_state[key]
            starts[env, :len(packets)] = [p[0] for p in packets]
            strides[env, :len(packets)] = [p[1] for p in packets]
            inverse_take[0, env, :len(packets)] = [p[2] for p in packets]
            inverse_take[1, env, :len(packets)] = [p[3] for p in packets]
            names = game.PIECE_SETS[state.piece_set]
            completed = state.pieces // len(names)
            contexts[env, :15] = [name == state.current for name in self.names]
            contexts[env, 15:30] = [name == state.next_piece for name in self.names]
            contexts[env, 30:45] = [name in names and state.piece_counts[name] == completed for name in self.names]
            contexts[env, 81] = (state.pieces % len(names)) / len(names)
        board_values = np.asarray([s.board for s in states])
        if (board_values.shape != (n, 4096) or board_values.dtype.kind not in "iub"
                or np.any(board_values < 0) or np.any(board_values > 65535)):
            raise ValueError("Expected 4096 integer column masks in [0, 65535] per environment")
        boards = self._device(board_values.astype(np.int32, copy=False))
        device_targets = self._device(np.asarray(targets, dtype=np.int64))
        device_pieces = self._device(np.asarray(piece_ids, dtype=np.int32))
        context = self._device(contexts)
        heights = torch.empty((n, 4096), dtype=torch.int32, device=self.device)
        counts = torch.empty((n, 16), dtype=torch.int32, device=self.device)
        statistics = torch.empty((n, 4), dtype=torch.int32, device=self.device)
        board_scores = torch.empty(n, dtype=torch.float64, device=self.device)
        _board_kernel[(n,)](boards, heights, counts, context, board_scores, statistics,
                            device_targets, self.weights, self.ticks_low, self.ticks_high,
                            CONTEXT=neural.CONTEXT, BLOCK=4096, enable_fp_fusion=False, num_warps=8)
        specifications = torch.empty((n, candidates, 6), dtype=torch.int32, device=self.device)
        scores = torch.empty((n, candidates), dtype=torch.float64, device=self.device)
        gains, future, deltas, clears = [torch.empty((n, candidates), dtype=torch.int32, device=self.device) for _ in range(4)]
        valid = torch.empty((n, candidates), dtype=torch.bool, device=self.device)
        _candidate_kernel[(triton.cdiv(candidates, 128), n)](
            boards, heights, counts, statistics, self.shapes, self.templates,
            self._device(starts), self._device(strides), self._device(inverse_take),
            device_pieces, self.shape_offsets, self.sample_counts, self.vertical_rotations,
            device_targets, board_scores, self.rough_delta, self.weights, specifications,
            scores, gains, future, deltas, clears, valid, CANDIDATES=candidates,
            SAMPLES=samples, TEMPLATE_STRIDE=self.samples, ROTATIONS=self.rotations, PROTECT=self.config.protect_target,
            BLOCK=128, enable_fp_fusion=False, num_warps=4)
        self._score_clearing(boards, device_pieces, specifications, scores, clears, valid)
        if candidates <= SUMMARY_KERNEL_LIMIT:
            summary = torch.empty((n, 5), dtype=torch.int32, device=self.device)
            _summary_kernel[(n,)](valid, clears, statistics, counts, device_pieces,
                                 self.sample_counts, device_targets, summary,
                                 CANDIDATES=candidates, BLOCK=triton.next_power_of_2(candidates), num_warps=8)
        else:
            # Exhaustive pools exceed one program block; reduce with tensor ops instead.
            filled = statistics[:, 0]
            evaluated = (self.sample_counts.index_select(0, device_pieces)
                         + torch.where(device_pieces == 0, 4096 - filled, 0))
            summary = torch.stack((valid.sum(1, dtype=torch.int32),
                                   (valid & (clears != 0)).any(1).to(torch.int32),
                                   statistics[:, 1], filled, evaluated.to(torch.int32)), dim=1)
        summaries = summary.cpu().tolist()
        reasons, references, lengths, evaluations, fallbacks = [], {}, [], [], []
        for env, (number, has_clear, buried, filled, evaluated) in enumerate(summaries):
            reason = "buried_target" if buried else "full_target" if filled == 4096 else "clear_candidate" if has_clear else "exhaustive_or_recovery" if not number else None
            if reason is not None:
                references[env] = neural.prepare(states[env], targets[env], self._fallback_config)
            p = references.get(env)
            reasons.append(reason)
            lengths.append(len(p.options) if p is not None else 0 if env in references else min(number, shortlist))
            evaluations.append(p.evaluations if p is not None else evaluated)
            fallbacks.append(p.fallback if p is not None else False)
        # Stable sorts reproduce max/heapq's earliest-row tie rule.
        primary = gains - game.FUTURE_HOLE_PENALTY * future
        minimum = torch.iinfo(torch.int32).min
        primary = torch.where(valid, primary, minimum)
        by_score = torch.argsort(scores, dim=1, descending=True, stable=True)
        by_primary = torch.argsort(primary.gather(1, by_score), dim=1, descending=True, stable=True)
        ranked = by_score.gather(1, by_primary)
        half = max(1, shortlist // 2)
        top = ranked[:, :half]
        top_mask = torch.zeros_like(valid)
        top_mask.scatter_(1, top, True)
        rest_mask = valid & ~top_mask
        positions = np.zeros((n, shortlist - half), dtype=np.int64)
        for env, (number, *_rest) in enumerate(summaries):
            if number > shortlist and env not in references and self.config.shortlist_mode != "teacher-top":
                rng = random.Random(states[env].seed * 8191 + states[env].pieces)
                positions[env] = rng.sample(range(number - half), shortlist - half)
        rest_rank = torch.cumsum(rest_mask.to(torch.int32), dim=1)
        exploration = torch.searchsorted(rest_rank.contiguous(), self._device(positions).to(torch.int32) + 1)
        chosen = torch.cat((top, exploration), dim=1)
        raw_count = self._device(np.asarray([s[0] for s in summaries], dtype=np.int64))
        first = ranked[:, :shortlist]
        chosen = first if self.config.shortlist_mode == "teacher-top" else torch.where((raw_count <= shortlist)[:, None], first, chosen)
        padding = torch.arange(shortlist, device=self.device)[None, :] >= self._device(np.asarray(lengths, dtype=np.int64))[:, None]
        chosen = torch.where(padding, candidates, chosen).sort(dim=1).values
        length_tensor = self._device(np.asarray(lengths, dtype=np.int32))
        features = torch.empty((n, shortlist, neural.FEATURES), dtype=torch.float32, device=self.device)
        indices = torch.empty((n, shortlist, neural.COLUMNS), dtype=torch.int64, device=self.device)
        column_mask = torch.empty((n, shortlist, neural.COLUMNS), dtype=torch.bool, device=self.device)
        option_specs = torch.empty((n, shortlist, 6), dtype=torch.int32, device=self.device)
        option_scores = torch.empty((n, shortlist), dtype=torch.float64, device=self.device)
        _features_kernel[(n, shortlist)](
            chosen, specifications, scores, gains, future, deltas, board_scores,
            device_targets, device_pieces, self.shape_offsets, self.shapes, self.shape_features,
            length_tensor, features, indices, column_mask, option_specs, option_scores,
            CANDIDATES=candidates, SHORTLIST=shortlist, BLOCK=128, enable_fp_fusion=False, num_warps=4)
        inverse_rank = torch.empty_like(ranked)
        inverse_rank.scatter_(1, ranked, torch.arange(candidates, device=self.device)[None, :].expand(n, -1))
        safe_chosen = chosen.clamp_max(candidates - 1)
        kept_ranks = inverse_rank.gather(1, safe_chosen).masked_fill(padding, candidates + 1)
        order = torch.argsort(kept_ranks, dim=1)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(shortlist, device=self.device)[None, :].expand(n, -1))
        teachers = kept_ranks.argmin(dim=1)
        kept_primary = primary.gather(1, safe_chosen)
        best_primary = primary.gather(1, ranked[:, :1])
        deficits = (best_primary - kept_primary).to(torch.float32)
        priors = (-32 * deficits - rank.to(torch.float32) / 8).masked_fill(padding, 0)
        all_arrays = {"boards": boards, "targets": device_targets, "context": context,
                      "features": features, "indices": indices, "column_mask": column_mask,
                      "valid": ~padding, "teacher_prior": priors}
        candidate_fields = neural.CANDIDATE_FIELDS
        for env, p in references.items():
            if p is None:
                continue
            for name, value in p.arrays.items():
                if name.startswith("spatial_"):
                    continue  # recomputed on device below from the final options
                tensor = self._device(value)
                if name in candidate_fields:
                    all_arrays[name][env].zero_()
                    all_arrays[name][env, :len(p.options)].copy_(tensor)
                else:
                    all_arrays[name][env].copy_(tensor)
            teachers[env] = p.teacher
            option_specs[env, :len(p.options)].copy_(self._device(np.asarray(
                [(c.rotation, c.x, c.y, c.w, c.z, c.layers) for c in p.options], dtype=np.int32)))
            option_scores[env, :len(p.options)].copy_(self._device(np.asarray([c.score for c in p.options], dtype=np.float64)))
        if self.spatial:
            all_arrays["spatial_board"], all_arrays["spatial_delta"] = self._spatial_arrays(
                boards, device_targets, device_pieces, option_specs, ~padding)
        prepared = []
        for env, length in enumerate(lengths):
            if not length:
                prepared.append(None)
                continue
            record = {name: value[env, :length] if name in candidate_fields else value[env]
                      for name, value in all_arrays.items()}
            options = _DeviceOptions(states[env].current, option_specs[env, :length], option_scores[env, :length])
            prepared.append(GpuPrepared(record, options, teachers[env], evaluations[env], fallbacks[env]))
        environment_indices = tuple(i for i, p in enumerate(prepared) if p is not None)
        maximum = max(lengths, default=0)
        if len(environment_indices) == n:
            # These are fresh per-call allocations, so views can safely become
            # retained rollout records without another stack/gather allocation.
            batch = {name: value[:, :maximum] if name in candidate_fields else value
                     for name, value in all_arrays.items()}
            batch_teachers = teachers
            batch_option_specs = option_specs[:, :maximum]
            batch_option_scores = option_scores[:, :maximum]
        elif environment_indices:
            env_rows = torch.tensor(environment_indices, dtype=torch.int64, device=self.device)
            batch = {name: value.index_select(0, env_rows)[:, :maximum] if name in candidate_fields
                     else value.index_select(0, env_rows) for name, value in all_arrays.items()}
            batch_teachers = teachers.index_select(0, env_rows)
            batch_option_specs = option_specs.index_select(0, env_rows)[:, :maximum]
            batch_option_scores = option_scores.index_select(0, env_rows)[:, :maximum]
        else:
            batch = {}
            batch_teachers = teachers[:0]
            batch_option_specs = option_specs[:0, :0]
            batch_option_scores = option_scores[:0, :0]
        return DevicePreparedBatch(tuple(prepared), environment_indices, batch,
                                   batch_teachers, tuple(reasons),
                                   tuple(int(references[i].arrays["targets"]) if i in references and references[i] is not None
                                         else targets[i] for i in range(n)),
                                   batch_option_specs, batch_option_scores)


CANDIDATE_FIELDS = neural.CANDIDATE_FIELDS


def _pad_candidates(tensor, width):
    missing = width - tensor.shape[1]
    if missing <= 0:
        return tensor
    shape = (tensor.shape[0], missing, *tensor.shape[2:])
    return torch.cat((tensor, tensor.new_zeros(shape)), dim=1)


def merge_prepared_batches(first, second):
    """Concatenate two DevicePreparedBatch results for consecutive environment groups.

    Valid candidate slots are identical to preparing both groups in one call; padded
    (masked) slots beyond a group's own maximum are zero. Per-environment records stay
    views of each group's own allocations.
    """
    offset = len(first.prepared)
    return _merge_prepared_groups([(range(offset), first),
                                   (range(offset, offset + len(second.prepared)), second)])


def _merge_prepared_groups(groups):
    """Restore caller order for per-environment records and compact inference rows."""
    count = sum(len(indices) for indices, _ in groups)
    prepared, reasons, targets = [None] * count, [None] * count, [None] * count
    active, rows = [], []
    for indices, result in groups:
        for local, env in enumerate(indices):
            prepared[env] = result.prepared[local]
            reasons[env] = result.cpu_fallbacks[local]
            targets[env] = result.targets[local]
        if result.environment_indices:
            active.append(result)
            rows.extend(indices[i] for i in result.environment_indices)
    original = tuple(sorted(rows))
    if len(active) <= 1:
        batch = active[0] if active else groups[0][1]
        return DevicePreparedBatch(tuple(prepared), original, batch.batch, batch.teachers, tuple(reasons),
                                   tuple(targets), batch._option_specs, batch._option_scores)
    order = None
    if rows != list(original):
        order = torch.tensor(sorted(range(len(rows)), key=rows.__getitem__), device=active[0].teachers.device)
    width = max(b._option_specs.shape[1] for b in active)

    def combine(tensors, candidate_field=False):
        if candidate_field:
            tensors = [_pad_candidates(t, width) for t in tensors]
        combined = torch.cat(tensors)
        return combined if order is None else combined.index_select(0, order)

    batch = {name: combine([b.batch[name] for b in active], name in CANDIDATE_FIELDS)
             for name in active[0].batch}
    return DevicePreparedBatch(
        tuple(prepared), original, batch, combine([b.teachers for b in active]), tuple(reasons), tuple(targets),
        combine([b._option_specs for b in active], True), combine([b._option_scores for b in active], True))


def compile_offline():
    """Compile bounded sm_89 kernels without initializing CUDA or running jobs."""
    import hashlib
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    target = GPUTarget("cuda", 89, 32)
    specifications = [
        (_board_kernel,
         ["*i32", "*i32", "*i32", "*fp32", "*fp64", "*i32", "*i64", "*fp64", "*i64", "*i64"],
         {"CONTEXT": 82, "BLOCK": 4096}, 8),
        (_candidate_kernel,
         ["*i32", "*i32", "*i32", "*i32", "*i32", "*i32", "*i32", "*i32", "*i32",
          "*i32", "*i32", "*i32", "*i32", "*i64", "*fp64", "*fp64", "*fp64", "*i32",
          "*fp64", "*i32", "*i32", "*i32", "*i32", "*i1"],
         {"CANDIDATES": 6144, "SAMPLES": 2048, "TEMPLATE_STRIDE": 2048, "ROTATIONS": 192, "PROTECT": True, "BLOCK": 128}, 4),
        (_summary_kernel,
         ["*i1", "*i32", "*i32", "*i32", "*i32", "*i32", "*i64", "*i32"],
         {"CANDIDATES": 6144, "BLOCK": 8192}, 8),
        (_features_kernel,
         ["*i64", "*i32", "*fp64", "*i32", "*i32", "*i32", "*fp64", "*i64", "*i32",
          "*i32", "*i32", "*fp32", "*i32", "*fp32", "*i64", "*i1", "*i32", "*fp64"],
         {"CANDIDATES": 6144, "SHORTLIST": 128, "BLOCK": 128}, 4)]
    result = {}
    for kernel, types, constants, warps in specifications:
        names = [name for name in kernel.arg_names if name not in constants]
        compiled = triton.compile(
            ASTSource(kernel, dict(zip(names, types)), constexprs=constants), target=target,
            options={"enable_fp_fusion": False, "num_warps": warps})
        ptx = compiled.asm["ptx"].encode()
        result[kernel.__name__] = {"ptx_bytes": len(ptx), "ptx_sha256": hashlib.sha256(ptx).hexdigest()}
    return result
