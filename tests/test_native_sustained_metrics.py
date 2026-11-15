from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


class NativeSustainedMetricsTests(unittest.TestCase):
    def setUp(self):
        self.backend = game.BACKEND
        game.configure_backend('native')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
        self.np = game._NUMPY

    def tearDown(self):
        game.configure_backend(self.backend)

    def compare(self, context, piece, specs, z, target):
        expected = game._numpy_sustained_batch_metrics(context, piece, specs, z, target)
        actual = game.sustained_batch_metrics(context, piece, specs, z, target)
        for reference, observed in zip(expected, actual):
            self.assertEqual(observed.dtype, reference.dtype)
            self.np.testing.assert_array_equal(observed, reference)
        self.assertEqual(len(actual[0]), len(specs))
        return actual

    def test_600_board_piece_target_batches_preserve_every_row(self):
        rng = random.Random(194613)
        boards = [game.EMPTY_BOARD, (65535,) * game.AREA, (32767,) * game.AREA,
                  tuple((1 << rng.randrange(17)) - 1 for _ in range(game.AREA)),
                  tuple(rng.getrandbits(16) for _ in range(game.AREA)),
                  (0,) + (2,) * (game.AREA - 1), (0,) + (3,) * (game.AREA - 1),
                  (4,) + game.EMPTY_BOARD[1:],
                  tuple(65535 if i % 5 else 0 for i in range(game.AREA)),
                  tuple(int(i % 3 != 0) for i in range(game.AREA))]
        batches = rows = blocked_rows = illegal_rows = 0
        for board in boards:
            context = game.evaluate_board(board)
            for piece in game.PIECE_NAMES:
                specs = tuple(game.specifications(piece, 129, 31))
                scores, z, _ = game.PlacementBatch(piece, specs).scores(context)
                before = scores.copy()
                illegal_rows += int((~self.np.isfinite(scores)).sum())
                for target in (0, 1, 7, 15):
                    _, blocked, _ = self.compare(context, piece, specs, z, target)
                    blocked_rows += int(blocked.sum())
                    rows += len(specs)
                    batches += 1
                    self.np.testing.assert_array_equal(scores, before)
        self.assertEqual(batches, 600)
        self.assertGreater(blocked_rows, 0)
        self.assertGreater(illegal_rows, 0)
        print(json.dumps(dict(exact_sustained_batches=batches, exact_sustained_rows=rows,
                              blocked_rows_retained=blocked_rows, illegal_rows_retained=illegal_rows)))

    def test_every_orientation_and_all_targets_at_both_boundaries(self):
        board = tuple((1 << ((i * 17 + i // 19) % 14)) - 1 for i in range(game.AREA))
        context = game.evaluate_board(board)
        for piece, shapes in game.SHAPES.items():
            specs = tuple(spec for r, shape in enumerate(shapes)
                          for spec in ((r, 0, 0, 0),
                                       (r, 16-shape.extents[0], 16-shape.extents[1], 16-shape.extents[3])))
            for start in range(0, len(specs), game.BATCH_SIZE):
                chunk = specs[start:start+game.BATCH_SIZE]
                _, z, _ = game.PlacementBatch(piece, chunk).scores(context)
                for target in range(game.HEIGHT):
                    self.compare(context, piece, chunk, z, target)

    def test_large_empty_and_noncontiguous_batches_keep_numpy_contract(self):
        context = game.evaluate_board(game.EMPTY_BOARD)
        specs = self.np.asarray(tuple(game.specifications('I', 2048, 19)), dtype=self.np.int32)
        z = self.np.zeros(len(specs), dtype=self.np.int32)
        self.compare(context, 'I', specs, z, 1)
        self.compare(context, 'I', specs[::2], z[::2], 1)
        self.compare(context, 'I', specs[:0], z[:0], 1)
        for landing in (-16, -1, 0, 15, 16):
            self.compare(context, 'I', specs[:129], self.np.full(129, landing, dtype=self.np.int32), 0)

    def test_clear_dead_buried_and_exhaustive_fallback_decisions(self):
        rotation, x, y, w = next(game.specifications('T', 1, 7))
        blocked_sample = list(game.EMPTY_BOARD)
        blocked_sample[game.column_index(x, y, w) + game.SHAPES['T'][rotation].columns[0][0]] = 3
        fixtures = [((0,) + (2,) * (game.AREA-1), 'I', 1, 64),
                    ((65535,) * game.AREA, 'I', 15, 64),
                    ((4,) + game.EMPTY_BOARD[1:], 'T', 1, 64),
                    (tuple(blocked_sample), 'T', 1, 1)]
        for board, piece, target, budget in fixtures:
            with patch.object(game, 'sustained_batch_metrics', game._numpy_sustained_batch_metrics):
                expected = game.choose_sustained_move(board, piece, target, budget, 7)
            actual = game.choose_sustained_move(board, piece, target, budget, 7)
            self.assertEqual(actual is None, expected is None)
            if actual is not None:
                self.assertEqual(actual.move, expected.move)
                self.assertEqual(actual.evaluations, expected.evaluations)
                self.assertEqual(actual.exhaustive_fallback, expected.exhaustive_fallback)
        self.assertIsNotNone(actual)
        self.assertTrue(actual.exhaustive_fallback)


if __name__ == '__main__':
    unittest.main(verbosity=2)
