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

    def test_merge_interleaved_and_empty_groups_with_ragged_shortlists(self):
        def batch(indices, counts):
            live = tuple(i for i, count in enumerate(counts) if count)
            width = max(counts)
            rows = torch.tensor([indices[i] for i in live], dtype=torch.int64)
            valid = torch.arange(width)[None, :] < torch.tensor([counts[i] for i in live])[:, None]
            scores = (rows[:, None] * 10 + torch.arange(width)).masked_fill(~valid, 0)
            specs = torch.zeros((len(live), width, 6), dtype=torch.int32)
            prepared = tuple(env if count else None for env, count in zip(indices, counts))
            return indices, gpu.DevicePreparedBatch(
                prepared, live, {"valid": valid, "targets": rows} if live else {}, rows,
                tuple(str(i) for i in indices), tuple(indices), specs, scores)

        groups = [batch([0, 3], [0, 0]), batch([1, 4], [2, 1]), batch([2], [3])]
        merged = gpu._merge_prepared_groups(groups)
        self.assertEqual(merged.prepared, (None, 1, 2, None, 4))
        self.assertEqual(merged.environment_indices, (1, 2, 4))
        self.assertEqual(merged.cpu_fallbacks, tuple(map(str, range(5))))
        self.assertEqual(merged.targets, tuple(range(5)))
        self.assertEqual(merged.teachers.tolist(), [1, 2, 4])
        self.assertEqual(merged.batch["targets"].tolist(), [1, 2, 4])
        self.assertEqual(merged.batch["valid"].sum(1).tolist(), [2, 3, 1])
        self.assertEqual(merged._option_scores.tolist(), [[10, 11, 0], [20, 21, 22], [40, 0, 0]])
        one = gpu._merge_prepared_groups([batch([0, 2], [0, 0]), batch([1, 3], [2, 1])])
        self.assertEqual(one.environment_indices, (1, 3))
        self.assertEqual(one.batch["targets"].tolist(), [1, 3])
        empty = gpu._merge_prepared_groups([batch([0, 2], [0, 0]), batch([1], [0])])
        self.assertEqual((empty.prepared, empty.environment_indices, empty.batch), ((None,) * 3, (), {}))

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

    def test_exhaustive_screening_all_pieces(self):
        config = neural.Config(shortlist=128, screening="exhaustive")
        states = [game.Game(9300 + i, "expanded") for i in range(len(game.PIECE_NAMES))]
        for state, piece in zip(states, game.PIECE_NAMES):
            state.current = piece
        self.compare(states, [1] * len(states), config)

    def test_grouped_exhaustive_matches_dense_through_recovery_clears_and_terminal(self):
        pieces = ["FORK", "I", "L", "HYPER", "CROSS", "I", "HOOK"]
        states = [game.Game(9500 + i, "expanded") for i in range(len(pieces))]
        for state, piece in zip(states, pieces):
            state.current = piece
        states[1].board = (0,) + (3,) * 4095      # vertical I clears two layers
        states[3].board = (65535,) * 4096          # no live row in the inference batch
        states[4].board = (0,) + (2,) * 4095       # preferred target is buried
        states[5].board = (0,) + (65535,) * 4095   # only the I fitting the shaft is legal
        targets = [1, 1, 1, 15, 0, 15, 1]
        config = neural.Config(architecture="survival-spatial-v4", spatial_actor=True,
                               recover_target=True, shortlist=17, screening="exhaustive")
        preparer = gpu.GpuPreparer(config)
        expected = preparer.prepare_batch(states, targets)
        preparer.group_exhaustive = True
        actual = preparer.prepare_batch(states, targets)
        self.assertIsNone(actual.prepared[3])
        self.assertNotEqual(actual.targets[4], targets[4])
        self.assertLess(actual.candidate_counts[5], config.shortlist)
        self.assertTrue(any(c.layers == 2 for c in actual.prepared[1].options))
        for key in ("environment_indices", "targets", "cpu_fallbacks", "evaluations", "fallback", "candidate_counts"):
            self.assertEqual(getattr(expected, key), getattr(actual, key), key)
        self.assertTrue(torch.equal(actual.teachers, expected.teachers))
        self.assertEqual(actual.batch.keys(), expected.batch.keys())
        for key in actual.batch:
            self.assertTrue(torch.equal(actual.batch[key], expected.batch[key]), key)
        valid = expected.batch["valid"]
        for key in ("_option_specs", "_option_scores"):
            self.assertTrue(torch.equal(getattr(actual, key)[valid], getattr(expected, key)[valid]), key)
        self.assertEqual(actual.selected_candidates(actual.teachers), expected.selected_candidates(expected.teachers))
        preparer.prepare_batch(states[1:3], targets[1:3])
        for row in actual.environment_indices:
            for key in actual.records[row]:
                self.assertTrue(torch.equal(actual.records[row][key], expected.records[row][key]), key)

    def test_layer_clears_scored_on_device_in_both_modes(self):
        states = [game.Game(9400 + i, "expanded") for i in range(3)]
        for state, piece in zip(states, ("I", "L", "T")):
            state.current = piece
            state.board = (0,) + (3,) * 4095
        for screening in ("sampled", "exhaustive"):
            with self.subTest(screening=screening):
                config = neural.Config(candidates=2048, shortlist=64, screening=screening,
                                       protect_target=False, recover_target=False)
                result = self.compare(states, [5] * 3, config)
                self.assertEqual(result.cpu_fallbacks, (None,) * len(states))
                if screening == "exhaustive":  # only the vertical I in column 0 clears; sampling may skip it
                    self.assertTrue(any(c.layers for p in result.prepared if p is not None for c in p.options))


if __name__ == "__main__":
    unittest.main()
