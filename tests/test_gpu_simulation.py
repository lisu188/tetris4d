import os
import unittest

import numpy as np
import torch

import sparse_tetris_4d_optimized as game
import tetris_neural as neural
try:
    import tetris_gpu_simulation as gpu
except ModuleNotFoundError as error:
    if error.name != "triton" and not error.name.startswith("triton."):
        raise
    gpu = None


@unittest.skipIf(gpu is None, "Optional CUDA preparation requires Triton")
class SamplingContractTests(unittest.TestCase):
    def test_rng_packets_expand_to_existing_specification_order(self):
        for piece in game.PIECE_NAMES:
            for budget in (1, 64, 2048):
                for salt in (0, 29, 103):
                    actual = []
                    for rotation, (start, stride, inverse, take) in enumerate(
                            gpu._sampling_packets(piece, budget, salt)):
                        shape = game.SHAPES[piece][rotation]
                        nx, ny, nw = (17 - shape.extents[i] for i in (0, 1, 3))
                        total = nx * ny * nw
                        self.assertEqual((stride * inverse) % total, 1)
                        for k in range(take):
                            index = (start + k * stride) % total
                            actual.append((rotation, index % nx, index // nx % ny, index // (nx * ny)))
                    with self.subTest(piece=piece, budget=budget, salt=salt):
                        self.assertEqual(actual, list(game.specifications(piece, budget, salt)))

    def test_cpu_device_rejected_before_allocations(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            gpu.GpuPreparer(device="cpu")

    def test_oversized_config_defers_device_tables(self):
        game.configure_backend("native")
        before = torch.cuda.is_initialized()
        for config, reason in (
                (neural.Config(candidates=gpu.MAX_GPU_CANDIDATES + 1), "candidate_budget_bound"),
                (neural.Config(shortlist=gpu.MAX_GPU_SHORTLIST + 1), "shortlist_bound")):
            preparer = gpu.GpuPreparer(config)
            self.assertEqual(preparer._reference_only_reason, reason)
            self.assertFalse(hasattr(preparer, "templates"))
        self.assertEqual(torch.cuda.is_initialized(), before)


@unittest.skipUnless(gpu is not None and os.environ.get("TETRIS4D_TEST_CUDA") == "1",
                     "CUDA probes require an explicitly coordinated bounded slot")
class CudaPreparationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        game.configure_backend("native")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

    def compare(self, states, targets, config):
        references = [neural.prepare(s, t, config) for s, t in zip(states, targets)]
        result = gpu.GpuPreparer(config).prepare_batch(states, targets)
        self.assertEqual(result.environment_indices, tuple(i for i, p in enumerate(references) if p is not None))
        for i, expected in enumerate(references):
            actual = result.prepared[i]
            if expected is None:
                self.assertIsNone(actual)
                continue
            self.assertEqual(tuple(actual.options), expected.options)
            self.assertEqual(actual.evaluations, expected.evaluations)
            self.assertEqual(actual.fallback, expected.fallback)
            self.assertEqual(int(actual.teacher), expected.teacher)
            self.assertEqual(result.targets[i], int(expected.arrays["targets"]))
            for name, value in expected.arrays.items():
                with self.subTest(env=i, tensor=name):
                    np.testing.assert_array_equal(actual.arrays[name].cpu().numpy(), value)
        if result.environment_indices:
            expected = neural.collate([references[i].arrays for i in result.environment_indices], torch.device("cpu"))
            for name, value in expected.items():
                torch.testing.assert_close(result.batch[name].cpu(), value, rtol=0, atol=0)
            selected = result.selected_candidates(result.teachers)
            self.assertEqual(selected, tuple(references[i].options[references[i].teacher] for i in result.environment_indices))
        return result

    def test_all_pieces_and_retained_records(self):
        config = neural.Config(candidates=2048, shortlist=128)
        states = [game.Game(9000 + i, "expanded") for i in range(16)]
        for i, state in enumerate(states):
            state.current = game.PIECE_NAMES[i % len(game.PIECE_NAMES)]
        preparer = gpu.GpuPreparer(config)
        first = preparer.prepare_batch(states, [1] * 16)
        snapshots = [{k: v.clone() for k, v in row.items()} for row in first.records]
        for state in states:
            move = game.make_move(state.board, state.current, 0, 0, 0, 0)
            if move is not None:
                state.commit(move)
        preparer.prepare_batch(states, [1] * 16)
        for actual, original in zip(first.records, snapshots):
            for name, value in original.items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
        self.compare(states, [1] * 16, config)

    def test_clear_dead_buried_and_exhaustive_fallback(self):
        states = [game.Game(9100 + i, "expanded") for i in range(5)]
        for state in states:
            state.current = "I"
        states[0].board = (0,) + (3,) * 4095
        states[1].board = (65535,) * 4096
        states[2].board = (2,) * 4096
        states[3].board = (0,) + (32767,) * 4095
        self.compare(states, [1, 15, 0, 15, 1], neural.Config(candidates=64, shortlist=16))

    def test_optional_protection_and_small_odd_shortlists(self):
        states = [game.Game(9200 + i, "expanded") for i in range(3)]
        for state, piece in zip(states, ("I", "T", "HYPER")):
            state.current = piece
        self.compare(states, [1] * 3, neural.Config(
            candidates=32, shortlist=3, protect_target=False, recover_target=False))
        self.compare(states, [1] * 3, neural.Config(
            candidates=1, shortlist=2, protect_target=False, recover_target=False))
        states[1].board = (2,) * 4096
        self.compare(states, [1, 0, 1], neural.Config(
            candidates=64, shortlist=3, protect_target=False, recover_target=False))
        result = self.compare(states[2:], [1], neural.Config(
            candidates=gpu.MAX_GPU_CANDIDATES + 1, shortlist=16))
        self.assertEqual(result.cpu_fallbacks, ("candidate_budget_bound",))
        result = self.compare(states[2:], [1], neural.Config(
            candidates=64, shortlist=gpu.MAX_GPU_SHORTLIST + 1))
        self.assertEqual(result.cpu_fallbacks, ("shortlist_bound",))


if __name__ == "__main__":
    unittest.main()
