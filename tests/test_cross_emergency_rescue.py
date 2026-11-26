import unittest
from unittest.mock import patch

import sparse_tetris_4d_optimized as game
import tetris_cross_survival as cross


class EmergencyRescueTests(unittest.TestCase):
    def test_only_vertical_i_clears_and_reopens_preview(self):
        # All centers are too high for CROSS. A vertical I in the corner completes
        # layer 14, opening the entire field to the visible piece.
        board = (1 << 10,) + (1 << 14,) * (game.AREA - 1)
        context = game.evaluate_board(board)
        failed = game.score_candidate(context, 'I', 0, 1, 0, 0)
        self.assertIsNotNone(failed)
        self.assertFalse(cross.visible_piece_fits(game.materialize_move(board, failed).board, 'CROSS'))
        for preview in ('CROSS', 'HYPER'):
            with self.subTest(preview=preview):
                self.assertFalse(game.has_legal_move(board, preview))
                action = cross.outside_shortlist_rescue(board, 'I', preview, [failed])
                move = game.materialize_move(board, action)
                self.assertTrue(game.has_legal_move(move.board, preview))
                self.assertEqual((action.x, action.y, action.z, action.w, action.layers), (0, 0, 11, 0, 1))

    def test_blocked_with_no_clear_or_unreachable_hole(self):
        board = tuple(1 << (14 + i % 2) for i in range(game.AREA))
        with patch.object(cross, '_after_board', side_effect=AssertionError('no layer can clear')):
            self.assertIsNone(cross.outside_shortlist_rescue(board, 'BRIDGE', 'CROSS', []))
        # Layer 13 lacks exactly one cell, but its only hole has a roof.
        roof = (1 << 15,) + (1 << 13,) * (game.AREA - 1)
        self.assertIsNone(cross.outside_shortlist_rescue(roof, 'I', 'CROSS', []))

    def test_buried_double_clear_is_searched_without_target_restrictions(self):
        board = [3 | (1 << 14)] * game.AREA
        board[game.column_index(7, 7, 7)] = 0
        board = tuple(board)
        self.assertFalse(cross.visible_piece_fits(board, 'CROSS'))
        action = cross.outside_shortlist_rescue(board, 'I', 'CROSS', [])
        self.assertEqual((action.x, action.y, action.z, action.w, action.layers), (7, 7, 0, 7, 2))
        self.assertTrue(game.has_legal_move(game.materialize_move(board, action).board, 'CROSS'))

    def test_visible_initially_fits_and_canonical_order_matches_game(self):
        board = [1 << 14] * game.AREA
        center = game.column_index(7, 7, 7)
        for offset in cross.NEIGHBORS:
            board[center + offset] = 1 << 11
        board = tuple(board)
        self.assertEqual(cross.cross_count(board), 1)
        context = game.evaluate_board(board)
        rotation = next(i for i, shape in enumerate(game.SHAPES['O']) if shape.extents == (2, 2, 1, 1))
        failed = game.score_candidate(context, 'O', rotation, 7, 7, 7)
        self.assertFalse(game.has_legal_move(game.materialize_move(board, failed).board, 'CROSS'))
        expected = next(a for spec in game.specifications('O', 0)
                        if (a := game.score_candidate(context, 'O', *spec)) is not None and
                        game.has_legal_move(game.materialize_move(board, a).board, 'CROSS'))
        self.assertEqual(cross.outside_shortlist_rescue(board, 'O', 'CROSS', [failed]), expected)
        # The sole physical CROSS must use the last landing itself.
        self.assertIsNone(cross.outside_shortlist_rescue(board, 'CROSS', 'CROSS', []))


if __name__ == '__main__':
    unittest.main()
