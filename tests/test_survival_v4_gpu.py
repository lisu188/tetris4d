from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
import unittest

import numpy as np
import torch

import sparse_tetris_4d_optimized as game
import tetris_neural as neural
from tetris_survival_v4 import (ARCHITECTURE, _bit_length16, _compact_full_layers,
                                afterstate_descriptors, device_afterstate_descriptors)
try:
    import tetris_gpu_simulation as gpu
except ModuleNotFoundError as error:
    if error.name != "triton" and not error.name.startswith("triton."):
        raise
    gpu = None


class DeviceDescriptorCpuTests(unittest.TestCase):
    def test_bit_length_is_exact(self):
        values = torch.arange(65536, dtype=torch.int32)
        expected = torch.tensor([v.bit_length() for v in range(65536)], dtype=torch.int32)
        self.assertTrue(torch.equal(_bit_length16(values), expected))

    def test_compaction_matches_game(self):
        game.configure_backend("numpy")
        rng = np.random.default_rng(3)
        board = rng.integers(0, 65536, 4096, dtype=np.int64) | 0b1000100000000101
        expected, _ = game.clear_layers(tuple(int(v) for v in board))
        actual = _compact_full_layers(torch.as_tensor(board[None], dtype=torch.int32))
        self.assertEqual(tuple(actual[0].tolist()), expected)

    def test_batched_matches_reference_descriptors(self):
        game.configure_backend("numpy")
        config = neural.Config(architecture=ARCHITECTURE, spatial_actor=True)
        state = game.Game(4242, "expanded")
        for _ in range(12):
            prepared = neural.prepare(state, 1, config)
            if prepared is None:
                break
            state.commit(game.make_move(state.board, state.current, *[
                getattr(prepared.options[prepared.teacher], k) for k in ("rotation", "x", "y", "w")]))
        prepared = neural.prepare(state, 1, config)
        options = prepared.options
        cells = [game.placed_cells(o) for o in options]
        width = max(map(len, cells))
        table = np.zeros((1, len(options), width, 4), dtype=np.int64)
        present = np.zeros((1, len(options), width), dtype=np.bool_)
        for i, c in enumerate(cells):
            table[0, i, :len(c)] = c
            present[0, i, :len(c)] = True
        board, delta = device_afterstate_descriptors(
            torch.as_tensor([state.board], dtype=torch.int32), torch.tensor([1]),
            torch.from_numpy(table), torch.from_numpy(present), torch.ones((1, len(options)), dtype=torch.bool))
        ref_board, ref_delta = afterstate_descriptors(state.board, options, 1)
        np.testing.assert_allclose(board[0].numpy(), ref_board, atol=1e-6)
        np.testing.assert_allclose(delta[0].numpy(), ref_delta, atol=1e-6)


@unittest.skipUnless(gpu is not None and os.environ.get("TETRIS4D_TEST_CUDA") == "1",
                     "CUDA probes require an explicitly coordinated bounded slot")
class GpuPreparerSpatialTests(unittest.TestCase):
    def test_gpu_prepare_matches_cpu_spatial_arrays(self):
        game.configure_backend("native")
        config = neural.Config(architecture=ARCHITECTURE, spatial_actor=True)
        states = [game.Game(9300 + i, "expanded") for i in range(12)]
        for i, state in enumerate(states):
            state.current = game.PIECE_NAMES[i % len(game.PIECE_NAMES)]
            for _ in range(i):
                prepared = neural.prepare(state, 1, config)
                if prepared is None:
                    break
                c = prepared.options[prepared.teacher]
                state.commit(game.make_move(state.board, state.current, c.rotation, c.x, c.y, c.w))
        states[0].board = (0,) + (3,) * 4095  # forces a clear candidate via CPU fallback
        targets = [1] * len(states)
        result = gpu.GpuPreparer(config).prepare_batch(states, targets)
        for env, p in enumerate(result.prepared):
            reference = neural.prepare(states[env], targets[env], config)
            if reference is None:
                self.assertIsNone(p)
                continue
            np.testing.assert_allclose(p.arrays["spatial_board"].cpu().numpy(),
                                       reference.arrays["spatial_board"], atol=1e-6)
            np.testing.assert_allclose(p.arrays["spatial_delta"].cpu().numpy(),
                                       reference.arrays["spatial_delta"], atol=1e-6)
        self.assertEqual(result.batch["spatial_delta"].shape[-1], 18)


@unittest.skipUnless(gpu is not None and os.environ.get("TETRIS4D_TEST_CUDA") == "1",
                     "CUDA probes require an explicitly coordinated bounded slot")
class FusedKernelTests(unittest.TestCase):
    def test_adjacency_kernel_matches_reference(self):
        from tetris_survival_v4 import adjacency_statistics, reference_adjacency_statistics
        generator = torch.Generator().manual_seed(1)
        heights = torch.randint(0, 17, (64, 4096), generator=generator)
        noise = torch.randint(0, 65536, (64, 4096), generator=generator)
        boards = (((1 << heights) - 1) ^ (noise & ((1 << heights) - 1) & (noise >> 3))).to(torch.int32)
        boards[3] = 65535
        targets = torch.randint(0, 16, (64,), generator=generator)
        expected = reference_adjacency_statistics(boards, targets)
        actual = adjacency_statistics(boards.cuda(), targets.cuda()).cpu()
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

    def test_afterstate_kernel_clears_layers_like_reference(self):
        generator = torch.Generator().manual_seed(2)
        n, shortlist, width = 8, 64, 6
        boards = torch.randint(0, 256, (n, 4096), dtype=torch.int32, generator=generator)
        boards[2] |= 0x3e00
        boards[2, 7] &= ~0x0200
        cells = torch.stack([torch.randint(0, 16, (n, shortlist, width), generator=generator) for _ in range(4)], -1)
        cells[..., 2] = 14 - torch.arange(width).view(1, 1, width)
        cells[2, :, 0] = torch.tensor([7, 0, 9, 0])
        index = (cells[..., 3] * 16 + cells[..., 1]) * 16 + cells[..., 0]
        occupied = (boards.gather(1, index.view(n, -1)).view(n, shortlist, width) >> cells[..., 2]) & 1
        present = (torch.rand(n, shortlist, width, generator=generator) > .3) & (occupied == 0)
        present[2, :, 0] = True
        valid = torch.rand(n, shortlist, generator=generator) > .1
        targets = torch.randint(0, 16, (n,), generator=generator)
        expected = device_afterstate_descriptors(boards, targets, cells, present, valid)
        actual = device_afterstate_descriptors(boards.cuda(), targets.cuda(), cells.cuda(),
                                               present.cuda(), valid.cuda())
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got.cpu(), want, atol=1e-7, rtol=1e-6)
        self.assertTrue(bool((expected[1][2][valid[2]] != 0).any()))


if __name__ == "__main__":
    unittest.main()
