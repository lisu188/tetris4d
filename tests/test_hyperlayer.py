from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tools import first_hyperlayer


class HyperlayerTests(unittest.TestCase):
    def test_status_and_validation(self):
        self.assertEqual(game.target_layer_status(game.EMPTY_BOARD), game.TargetLayerStatus(1, 0, 0))
        board = (2, 4, 6, 1) + (0,) * (game.AREA - 4)
        self.assertEqual(game.target_layer_status(board), game.TargetLayerStatus(1, 2, 1))
        for target in (-1, 16, .5, True):
            with self.assertRaises(ValueError):
                game.target_layer_status(game.EMPTY_BOARD, target)
        for board in ((0,), (-1,) + game.EMPTY_BOARD[1:], (1 << 16,) + game.EMPTY_BOARD[1:]):
            with self.assertRaises(ValueError):
                game.target_layer_status(board)

    def test_invalid_policy_arguments(self):
        for piece, candidates in (('INVALID', 1), ('O', -1)):
            with self.assertRaises(ValueError):
                game.choose_hyperlayer_move(game.EMPTY_BOARD, piece, candidates=candidates)

    def test_buried_target_is_not_top_out(self):
        board = (4,) + game.EMPTY_BOARD[1:]
        self.assertTrue(game.has_legal_move(board, 'O'))
        self.assertIsNone(game.choose_hyperlayer_move(board, 'O'))

    def test_full_and_occupied_boards(self):
        self.assertIsNone(game.choose_hyperlayer_move((2,) * game.AREA, 'O'))
        self.assertIsNone(game.choose_hyperlayer_move(((1 << 16) - 1,) * game.AREA, 'O'))

    def test_rigid_bridge_cannot_cover_unfinished_cells(self):
        board = (2,) + game.EMPTY_BOARD[1:]
        candidate = game.make_move(board, 'I', 0, 0, 0, 0).candidate
        self.assertEqual(game.target_candidate_metrics(board, candidate, 1), (0, 3))

    def test_cross_can_leave_holes_below_target(self):
        move = game.make_move(game.EMPTY_BOARD, 'CROSS', 0, 0, 0, 0)
        self.assertEqual(game.target_candidate_metrics(game.EMPTY_BOARD, move.candidate, 1), (7, 0))
        self.assertGreater(move.holes, 0)
        self.assertEqual(game.target_layer_status(move.board, 1).buried, 0)

    def test_unsampled_final_gap_and_compaction(self):
        budget, salt = 32, 71
        vertical = next(r for r, shape in enumerate(game.SHAPES['I']) if len(shape.columns) == 1)
        sampled = set(game.specifications('I', budget, salt))
        gap = next(i for i in range(game.AREA) if (vertical, *game.column_coordinates(i)) not in sampled)
        board = [2] * game.AREA
        board[gap] = 0
        before = tuple(board)
        decision = game.choose_hyperlayer_move(before, 'I', candidates=budget, salt=salt)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.move.layers, 1)
        self.assertEqual(decision.coverage_gain, 1)
        p = decision.move.candidate
        self.assertEqual(game.column_index(p.x, p.y, p.w), gap)
        self.assertEqual(game.full_layer_mask(game.merge_rigid_piece(before, game.placed_cells(decision.move), 'I')), 2)
        self.assertEqual(sum(m.bit_count() for m in decision.move.board), 3)
        self.assertEqual(before, tuple(board))
        self.assertEqual(decision.policy, 'hyperlayer')
        self.assertEqual(decision.depth, 1)
        self.assertEqual(decision.evaluations, budget + 1)

    def test_exhaustive_fallback_does_not_relax_safety(self):
        board = (2,) + game.EMPTY_BOARD[1:]
        original = game.specifications
        def specifications(piece, limit=game.DEFAULT_CANDIDATES, salt=0):
            return iter(((0, 0, 0, 0),)) if limit else original(piece, 0, salt)
        with patch.object(game, 'specifications', specifications):
            decision = game.choose_hyperlayer_move(board, 'O', candidates=1)
        self.assertIsNotNone(decision)
        self.assertTrue(decision.exhaustive_fallback)
        self.assertGreater(decision.evaluations, 1)
        self.assertEqual(game.target_candidate_metrics(board, decision.move.candidate, 1)[1], 0)

    def test_candidate_metrics_match_full_board(self):
        rng = random.Random(406)
        for name in game.PIECE_NAMES:
            for _ in range(8):
                board = tuple(2 if rng.random() < .15 else 0 for _ in range(game.AREA))
                spec = next(game.specifications(name, 1, rng.randrange(1000)))
                candidate = game.score_candidate(game.evaluate_board(board), name, *spec)
                if candidate is None:
                    continue
                gain, buried = game.target_candidate_metrics(board, candidate, 1)
                solid = game.merge_rigid_piece(board, game.placed_cells(candidate), name)
                self.assertEqual(gain, game.target_layer_status(solid).filled - game.target_layer_status(board).filled)
                self.assertEqual(buried, game.target_layer_status(solid).buried)

    def test_all_pieces_both_backends_and_determinism(self):
        original_backend = game.BACKEND
        try:
            try:
                game.configure_backend('numpy')
            except ValueError:
                self.skipTest('NumPy is unavailable')
            board = tuple(2 if i % 13 == 0 else 0 for i in range(game.AREA))
            parameters = game.network_parameters()
            for name in game.PIECE_NAMES:
                for target, before in ((1, board), (0, game.EMPTY_BOARD), (15, game.EMPTY_BOARD)):
                    game.configure_backend('python')
                    a = game.choose_hyperlayer_move(before, name, target, 128, 21)
                    game.configure_backend('numpy')
                    b = game.choose_hyperlayer_move(before, name, target, 128, 21)
                    self.assertEqual(a is None, b is None, (name, target))
                    if a is None:
                        continue
                    self.assertEqual(a.move, b.move, (name, target))
                    self.assertEqual(a.coverage_gain, b.coverage_gain)
                    self.assertEqual(a.evaluations, b.evaluations)
                    self.assertEqual(b.move, game.choose_hyperlayer_move(before, name, target, 128, 21).move)
            self.assertEqual(game.network_parameters(), parameters)
        finally:
            game.configure_backend(original_backend)

    def test_scalar_backend_rejects_numpy_dependency(self):
        original = game.BACKEND
        try:
            game.configure_backend('python')
            with patch.object(game, 'numpy_column_masks', side_effect=AssertionError('NumPy called')):
                decision = game.choose_hyperlayer_move(game.EMPTY_BOARD, 'CROSS', candidates=64)
                self.assertEqual(decision.coverage_gain, 7)
        finally:
            game.configure_backend(original)

    def test_generator_yields_and_can_be_cancelled(self):
        steps = game.hyperlayer_decision_steps(game.EMPTY_BOARD, 'O', candidates=1024)
        self.assertEqual(next(steps), game.BATCH_SIZE)
        steps.close()
        self.assertIsNotNone(game.choose_hyperlayer_move(game.EMPTY_BOARD, 'O', candidates=64))

    def test_run_game_reports_stall_without_false_game_over(self):
        with patch.object(game, 'choose_hyperlayer_move', return_value=None):
            result = game.run_game(7, 10, 2, 4, 64, 'hyperlayer')
        self.assertEqual(result['stop_reason'], 'planner_blocked')
        self.assertFalse(result['game_over'])
        self.assertFalse(result['capped'])
        self.assertEqual(result['pieces'], 0)

    def test_neural_and_literal_scores_unchanged(self):
        decision = game.choose_hyperlayer_move(game.EMPTY_BOARD, 'CROSS', candidates=64)
        move = decision.move
        self.assertTrue(math.isclose(move.score, game.network_forward(move.heights, move.holes, move.layers)[-1], abs_tol=1e-8))
        metadata = game.network_metadata()
        self.assertEqual((metadata['input_count'], metadata['sigmoid_units'], metadata['nonzero_connections']), (4098, 23040, 73218))
        self.assertEqual(game.choose_move(game.EMPTY_BOARD, 'O', 'I', 1, 4, 64).policy, 'network')

    def test_independent_verifier_detects_tampering(self):
        replay = first_hyperlayer.run(7, pieces=5, candidates=64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'replay.json'
            path.write_text(json.dumps(replay), encoding='utf-8')
            self.assertTrue(first_hyperlayer.verify(path)['passed'])
            replay['moves'][0][5] += 1
            path.write_text(json.dumps(replay), encoding='utf-8')
            with self.assertRaises(ValueError):
                first_hyperlayer.verify(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('auto', 'python', 'numpy'), default='auto')
    args, remaining = parser.parse_known_args()
    game.configure_backend(args.backend)
    unittest.main(argv=[sys.argv[0]] + remaining)
