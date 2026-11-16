"""Optional fused integer-board encoding; imported only by the CUDA opt-in path."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _encode(boards, targets, output, N: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = index < N
        column = index % 4096
        channel = index // 4096 % 34
        env = index // (34 * 4096)
        mask = tl.load(boards + env * 4096 + column, live, 0).to(tl.int32) & 65535
        height = 32 - tl.extra.cuda.libdevice.clz(mask)
        level = channel % 16
        occupied = ((mask >> level) & 1) != 0
        target = tl.load(targets + env, live, 0).to(tl.int32)
        value = tl.where(channel < 16, occupied.to(tl.float32),
                         tl.where(channel < 32, (~occupied & (level < height)).to(tl.float32),
                                  tl.where(channel == 32, height.to(tl.float32) / 16.,
                                           (height <= target).to(tl.float32))))
        tl.store(output + index, value, live)


def encode_columns_cuda(boards, targets):
    if triton is None:
        raise RuntimeError('Fused CUDA encoding requires Triton from a compatible CUDA PyTorch installation')
    if (boards.device.type != 'cuda' or targets.device != boards.device
            or boards.dtype != torch.int32 or targets.dtype not in (torch.int32, torch.int64)
            or boards.ndim != 2 or boards.shape[1] != 4096 or targets.shape != (boards.shape[0],)
            or not boards.is_contiguous() or not targets.is_contiguous()):
        raise ValueError('Fused encoding requires contiguous CUDA int32 boards and one integer target per board')
    output = torch.empty((boards.shape[0], 34, 16, 16, 16), device=boards.device, dtype=torch.float32)
    if output.numel():
        _encode[(triton.cdiv(output.numel(), 256),)](boards, targets, output, output.numel(), BLOCK=256)
    return output
