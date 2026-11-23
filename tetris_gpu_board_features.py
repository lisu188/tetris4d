"""Fused per-board integer statistics for Survival V5 next-state features (CUDA, Triton).

One program per board reads its 4096 16-bit column masks once and writes COUNTS integers.
tetris_survival_v5.board_counts_reference computes the same layout in plain torch; both feed
the same float conversion, so device and reference features are identical.

Layout:
  0..15   occupied cells per level           16..31  buried cells per level (empty, below the column top)
  32      sum of column heights              33      maximum column height
  34      accessible columns (height <= target)
  35      holes (empty cells below column tops)
  36      hole depth (filled cells above each hole, summed)
  37      columns with at least one hole     38      Z transitions (floor counts as filled)
  39      cumulative wells (depth*(depth+1)/2, walls = 16)
  40      deepest well                       41/42   columns with height >= 12 / >= 14
  43..45  height-difference diversity masks along x, y, w (bit v+2 set if difference v in -2..2 occurs)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

COUNTS = 46


@triton.jit
def _height(m):
    h = tl.zeros_like(m)
    for z in tl.static_range(16):
        h = tl.where(((m >> z) & 1) != 0, z + 1, h)
    return h


@triton.jit
def _diversity(difference, valid):
    bits = 0
    for v in tl.static_range(5):
        present = tl.max(((difference == v - 2) & valid).to(tl.int32), axis=0)
        bits = bits | (present << v)
    return bits


@triton.jit
def _board_counts_kernel(masks, targets, out, COUNTS: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, 4096)
    base = masks + row * 4096
    m = tl.load(base + offsets)
    h = _height(m)
    target = tl.load(targets + row)
    o = out + row * COUNTS
    above = tl.zeros_like(m)
    depth = tl.zeros_like(m)
    for k in tl.static_range(16):
        z = 15 - k
        b = (m >> z) & 1
        tl.store(o + z, tl.sum(b, axis=0))
        hole = (b == 0) & (z < h)
        tl.store(o + 16 + z, tl.sum(hole.to(tl.int32), axis=0))
        depth += tl.where(hole, above, 0)
        above += b
    holes = h - above
    transitions = tl.zeros_like(m)
    previous = tl.zeros_like(m) + 1
    for z in tl.static_range(16):
        b = (m >> z) & 1
        transitions += (b != previous).to(tl.int32)
        previous = b
    tl.store(o + 32, tl.sum(h, axis=0))
    tl.store(o + 33, tl.max(h, axis=0))
    tl.store(o + 34, tl.sum((h <= target).to(tl.int32), axis=0))
    tl.store(o + 35, tl.sum(holes, axis=0))
    tl.store(o + 36, tl.sum(depth, axis=0))
    tl.store(o + 37, tl.sum((holes > 0).to(tl.int32), axis=0))
    tl.store(o + 38, tl.sum(transitions, axis=0))
    x = offsets % 16
    y = (offsets // 16) % 16
    w = offsets // 256
    x_low = tl.where(x > 0, _height(tl.load(base + offsets - 1, mask=x > 0, other=0)), 16)
    x_high = tl.where(x < 15, _height(tl.load(base + offsets + 1, mask=x < 15, other=0)), 16)
    y_low = tl.where(y > 0, _height(tl.load(base + offsets - 16, mask=y > 0, other=0)), 16)
    y_high = tl.where(y < 15, _height(tl.load(base + offsets + 16, mask=y < 15, other=0)), 16)
    w_low = tl.where(w > 0, _height(tl.load(base + offsets - 256, mask=w > 0, other=0)), 16)
    w_high = tl.where(w < 15, _height(tl.load(base + offsets + 256, mask=w < 15, other=0)), 16)
    lowest = tl.minimum(tl.minimum(tl.minimum(x_low, x_high), tl.minimum(y_low, y_high)), tl.minimum(w_low, w_high))
    well = tl.maximum(lowest - h, 0)
    tl.store(o + 39, tl.sum(well * (well + 1) // 2, axis=0))
    tl.store(o + 40, tl.max(well, axis=0))
    tl.store(o + 41, tl.sum((h >= 12).to(tl.int32), axis=0))
    tl.store(o + 42, tl.sum((h >= 14).to(tl.int32), axis=0))
    tl.store(o + 43, _diversity(x_high - h, x < 15))
    tl.store(o + 44, _diversity(y_high - h, y < 15))
    tl.store(o + 45, _diversity(w_high - h, w < 15))


def board_counts(masks: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[n, 4096] int column masks and [n] targets on CUDA -> [n, COUNTS] int32."""
    masks = masks.to(torch.int32).contiguous()
    targets = targets.to(device=masks.device, dtype=torch.int32).contiguous()
    out = torch.empty((len(masks), COUNTS), dtype=torch.int32, device=masks.device)
    if len(masks):
        _board_counts_kernel[(len(masks),)](masks, targets, out, COUNTS=COUNTS, num_warps=8)
    return out
