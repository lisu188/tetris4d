"""Exact device CROSS mobility for the packed 16³-column board and shortlisted moves."""
import torch
import triton
import triton.language as tl


@triton.jit
def _centers(masks, ids):
    x, y, w = ids % 16, (ids // 16) % 16, ids // 256
    ok = (x > 0) & (x < 15) & (y > 0) & (y < 15) & (w > 0) & (w < 15) & (masks < 8192)
    for step in tl.static_range(6):
        stride = 1 << (4 * (step // 2))
        shift = stride if step % 2 == 0 else -stride
        neighbor = tl.gather(masks, tl.maximum(0, tl.minimum(4095, ids + shift)), 0)
        ok = ok & (neighbor < 16384)
    return ok


@triton.jit
def _before_kernel(boards, centers, counts):
    env = tl.program_id(0)
    ids = tl.arange(0, 4096)
    masks = tl.load(boards + env * 4096 + ids)
    ok = _centers(masks, ids)
    tl.store(centers + env * 4096 + ids, ok)
    tl.store(counts + env, tl.sum(ok.to(tl.int32), 0))


@triton.jit
def _after_kernel(boards, centers, before, specs, pieces, catalog, offsets, valid, after, WIDTH: tl.constexpr, CAP: tl.constexpr):
    env, option = tl.program_id(0), tl.program_id(1)
    row = env * WIDTH + option
    if tl.load(valid + row):
        spec = specs + row * 6
        shape = catalog + (tl.load(offsets + tl.load(pieces + env)) + tl.load(spec)) * 182
        columns = tl.load(shape + 4)
        count = tl.load(before + env)
        # Seven potentially lost centers per occupied column, including moves that clear.
        if CAP == 0 or count < CAP + 7 * columns:
            origin = tl.load(spec + 1) + 16 * tl.load(spec + 2) + 256 * tl.load(spec + 3)
            z, layers = tl.load(spec + 4), tl.load(spec + 5)
            if layers == 0:
                lane = tl.arange(0, 128)
                col, role = lane // 8, lane % 8
                cid = origin + tl.load(shape + 22 + 10 * col, col < columns, 0)
                bits = tl.load(shape + 23 + 10 * col, col < columns, 0) << z
                stride = 1 << (4 * (tl.maximum(role - 1, 0) // 2))
                step = tl.where(role == 0, 0, tl.where(role % 2 == 1, stride, -stride))
                center = cid + step
                lost = (col < columns) & (role < 7) & (bits >= tl.where(role == 0, 8192, 16384))
                live = tl.load(centers + env * 4096 + center, lost & (center >= 0) & (center < 4096), 0)
                # An out-of-bounds axial step wraps only onto an invalid boundary center.
                ordered = tl.sort(tl.where(lost & live, center, 4096))
                previous = tl.gather(ordered, tl.maximum(0, lane - 1), 0)
                count = count - tl.sum(((ordered < 4096) & ((lane == 0) | (ordered != previous))).to(tl.int32), 0)
            else:
                ids = tl.arange(0, 4096)
                masks = tl.load(boards + env * 4096 + ids)
                for c in range(columns):
                    cid = origin + tl.load(shape + 22 + 10 * c)
                    bits = tl.load(shape + 23 + 10 * c) << z
                    masks |= tl.where(ids == cid, bits, 0)
                for level in tl.static_range(15, -1, -1):
                    if tl.min((masks >> level) & 1, 0):
                        masks = (masks & ((1 << level) - 1)) | ((masks >> (level + 1)) << level)
                count = tl.sum(_centers(masks, ids).to(tl.int32), 0)
        if CAP:
            count = tl.minimum(count, CAP)
        tl.store(after + row, count)
    else:
        tl.store(after + row, 0)


def candidate_counts(boards, specs, piece_ids, catalog, offsets, valid, *, capped=False):
    """Return raw board counts and exact or policy-capped after-move counts on CUDA."""
    from tetris_cross_survival import MOBILITY_CAP
    n, width = valid.shape
    centers = torch.empty_like(boards, dtype=torch.bool)
    before = torch.empty(n, dtype=torch.int32, device=boards.device)
    after = torch.empty_like(valid, dtype=torch.int32)
    _before_kernel[(n,)](boards.contiguous(), centers, before, num_warps=8)
    if width:
        _after_kernel[(n, width)](boards.contiguous(), centers, before, specs.contiguous(), piece_ids,
                                  catalog, offsets, valid.contiguous(), after, WIDTH=width, CAP=MOBILITY_CAP if capped else 0, num_warps=4)
    return before, after
