"""Fused CUDA (Triton) kernels for Spatial V4 adjacency descriptors.

The reference implementation in tetris_survival_v4 issues ~100 small tensor
operations per call, which makes single-state inference launch-bound. These
kernels compute the same [rows, 3, 6] statistics with one program per board;
the afterstate kernel also merges the candidate cells and compacts full Z
hyperlayers in registers before measuring, matching game.clear_layers.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

AREA = 4096
EDGES = 15 * 16 * 16


@triton.jit
def _bit_length(value):
    result = tl.zeros_like(value)
    for k in tl.static_range(16):
        result += (value >= (1 << k)).to(tl.int32)
    return result


@triton.jit
def _popcount16(value):
    value = value - ((value >> 1) & 0x5555)
    value = (value & 0x3333) + ((value >> 2) & 0x3333)
    return ((((value + (value >> 4)) & 0x0F0F) * 0x0101) >> 8) & 0x1F


@triton.jit
def _axis_statistics(base, offsets, masks, heights, buried, layers, target, out, STRIDE: tl.constexpr):
    valid = ((offsets // STRIDE) % 16) < 15
    neighbour = tl.load(base + offsets + STRIDE, mask=valid, other=0)
    neighbour_heights = _bit_length(neighbour)
    neighbour_buried = ((1 << neighbour_heights) - 1) ^ neighbour
    neighbour_layers = (neighbour >> target) & 1
    delta = tl.abs(heights - neighbour_heights)
    zero = tl.zeros_like(delta)
    absolute = tl.sum(tl.where(valid, delta, zero), axis=0)
    squared = tl.sum(tl.where(valid, delta * delta, zero), axis=0)
    cliffs = tl.sum(tl.where(valid & (delta > 1), 1, zero), axis=0)
    occupancy = tl.sum(tl.where(valid, _popcount16(masks ^ neighbour), zero), axis=0)
    gaps = tl.sum(tl.where(valid, _popcount16(buried ^ neighbour_buried), zero), axis=0)
    layer = tl.sum(tl.where(valid, layers ^ neighbour_layers, zero), axis=0)
    tl.store(out + 0, absolute.to(tl.float32) / 3840.0 / 16.0)
    tl.store(out + 1, squared.to(tl.float32) / 3840.0 / 256.0)
    tl.store(out + 2, cliffs.to(tl.float32) / 3840.0)
    tl.store(out + 3, occupancy.to(tl.float32) / 3840.0 / 16.0)
    tl.store(out + 4, gaps.to(tl.float32) / 3840.0 / 16.0)
    tl.store(out + 5, layer.to(tl.float32) / 3840.0)


@triton.jit
def _board_statistics(base, offsets, masks, target, out):
    heights = _bit_length(masks)
    buried = ((1 << heights) - 1) ^ masks
    layers = (masks >> target) & 1
    _axis_statistics(base, offsets, masks, heights, buried, layers, target, out, 1)
    _axis_statistics(base, offsets, masks, heights, buried, layers, target, out + 6, 16)
    _axis_statistics(base, offsets, masks, heights, buried, layers, target, out + 12, 256)


@triton.jit
def _adjacency_kernel(boards, targets, out, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    base = boards + row * BLOCK
    masks = tl.load(base + offsets)
    target = tl.load(targets + row).to(tl.int32)
    _board_statistics(base, offsets, masks, target, out + row * 18)


@triton.jit
def _afterstate_kernel(boards, owners, targets, cell_index, cell_bits, scratch, out,
                       CELLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    owner = tl.load(owners + row).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    masks = tl.load(boards + owner * BLOCK + offsets)
    for cell in tl.static_range(CELLS):
        index = tl.load(cell_index + row * CELLS + cell)
        bit = tl.load(cell_bits + row * CELLS + cell)
        masks = tl.where(offsets == index, masks | bit, masks)
    full = tl.zeros((), dtype=tl.int32)
    for z in tl.static_range(16):
        count = tl.sum((masks >> z) & 1, axis=0)
        full = full | tl.where(count == BLOCK, 1 << z, 0)
    for layer in tl.static_range(15, -1, -1):
        compacted = (masks & ((1 << layer) - 1)) | ((masks >> (layer + 1)) << layer)
        masks = tl.where(((full >> layer) & 1) != 0, compacted, masks)
    base = scratch + row * BLOCK
    tl.store(base + offsets, masks)
    tl.debug_barrier()
    target = tl.load(targets + owner).to(tl.int32)
    _board_statistics(base, offsets, masks, target, out + row * 18)


def adjacency_statistics(boards: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """[rows, 4096] int masks, [rows] targets -> [rows, 3, 6] float32 on CUDA."""
    boards = boards.to(torch.int32).contiguous()
    targets = targets.to(torch.int32).contiguous()
    out = torch.empty((len(boards), 3, 6), dtype=torch.float32, device=boards.device)
    if len(boards):
        _adjacency_kernel[(len(boards),)](boards, targets, out, BLOCK=AREA, num_warps=8)
    return out


def afterstate_statistics(boards, owners, targets, cell_index, cell_bits):
    """Statistics of each owner's board with its candidate cells merged and full layers cleared.

    boards [n, 4096] int; owners [rows] board row per candidate; targets [n];
    cell_index/cell_bits [rows, C] int32 (index -1 marks an absent cell).
    Returns [rows, 3, 6] float32.
    """
    boards = boards.to(torch.int32).contiguous()
    rows, cells = cell_index.shape
    out = torch.empty((rows, 3, 6), dtype=torch.float32, device=boards.device)
    if rows:
        scratch = torch.empty((rows, AREA), dtype=torch.int32, device=boards.device)
        _afterstate_kernel[(rows,)](boards, owners.contiguous(), targets.to(torch.int32).contiguous(),
                                    cell_index.to(torch.int32).contiguous(), cell_bits.to(torch.int32).contiguous(),
                                    scratch, out, CELLS=cells, BLOCK=AREA, num_warps=8)
    return out
