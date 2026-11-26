from collections import Counter
from fractions import Fraction
from itertools import permutations
import unittest

import sparse_tetris_4d_optimized as game
import tetris_cross_deadline as deadline


class CrossDeadlineTests(unittest.TestCase):
    def test_fillable_line_buried_hole_and_wrapped_footprint(self):
        def board_with_holes(holes):
            board = [1] * game.AREA
            for slot in holes:
                board[slot] = 0
            return board

        board = board_with_holes(range(12, 16))
        bounds = deadline.layer_bounds(board, ('I',))
        self.assertEqual(bounds, [(0, 4, {'I': 4})])
        context = game.evaluate_board(tuple(board))
        self.assertTrue(any(move is not None and move.layers == 1
                            for r in range(len(game.SHAPES['I']))
                            for move in [game.score_candidate(context, 'I', r, 12, 0, 0)]))
        self.assertEqual(deadline.forced_probability(bounds, ('I',), ()), 0)
        board[15] = 2  # A cell above the missing layer can never be passed before the first clear.
        self.assertEqual(deadline.layer_bounds(board, ('I',)), [])
        wrapped = board_with_holes((14, 15, 16, 17))
        self.assertEqual(deadline.layer_bounds(wrapped, ('I',)), [(0, 4, {'I': 1})])
        self.assertEqual(deadline.layer_bounds(wrapped, ('I',), fit_footprints=False), bounds)

    def test_bag_boundary_known_visible_and_cross_already_consumed(self):
        names = game.PIECE_SETS['expanded']
        counts = dict.fromkeys(names, 3)
        fixed, pool = deadline.public_prefixes(counts, 'I')
        self.assertEqual(fixed, ('I',))
        self.assertEqual(set(pool), set(names) - {'CROSS', 'I'})
        self.assertEqual(deadline.public_prefixes(counts, 'CROSS'), ((), ()))
        counts['CROSS'] += 1
        fixed, pool = deadline.public_prefixes(counts, 'I')
        self.assertEqual(set(fixed), set(names) - {'CROSS'})
        self.assertEqual(set(pool), set(fixed))
        with self.assertRaises(ValueError):
            deadline.public_prefixes(counts, 'CROSS')
        counts['I'] += 2
        with self.assertRaises(ValueError):
            deadline.public_prefixes(counts, 'O')

    def test_joint_bound_matches_independent_full_permutation_law(self):
        fixed, possible = ('I',), ('O', 'T', 'I', 'L')
        bounds = [(0, 5, dict(I=1, O=0, T=2, L=2)), (1, 4, dict(I=0, O=3, T=1, L=0))]
        results = Counter()
        for draw in permutations((*possible, 'CROSS')):
            pieces = fixed + draw[:draw.index('CROSS')]
            results[all(sum(maxima[p] for p in pieces) < holes for _, holes, maxima in bounds)] += 1
        expected = Fraction(results[True], sum(results.values()))
        self.assertLess(0, expected)
        self.assertLess(expected, 1)
        self.assertEqual(deadline.forced_probability(bounds, fixed, possible), expected)
        self.assertEqual(deadline.forced_probability([], (), ()), 1)

    def test_slice_capacity_matches_translating_real_shape_cells(self):
        holes = {(x, 2, 15) for x in (0, 1, 2, 3, 13, 14, 15)}
        holes.update({(1, 1, 15), (1, 3, 15), (1, 2, 14), (0, 2, 14), (12, 6, 0)})
        board = [1] * game.AREA
        for coordinates in holes:
            board[game.column_index(*coordinates)] = 0
        [(level, missing, capacities)] = deadline.layer_bounds(board, game.PIECE_NAMES)
        self.assertEqual((level, missing), (0, len(holes)))
        for piece, shapes in game.SHAPES.items():
            maximum = 0
            for shape in shapes:
                for z, n in shape.levels:
                    if n <= maximum:
                        continue
                    cells = [(x, y, w) for x, y, height, w in shape.cells if height == z]
                    x0, y0, w0 = cells[0]
                    if any(all((x+hx-x0, y+hy-y0, w+hw-w0) in holes for x, y, w in cells)
                           for hx, hy, hw in holes):
                        maximum = n
            self.assertEqual(capacities[piece], maximum, piece)

    def test_healthy_drop_uses_only_public_piece_counts_and_visible_piece(self):
        counts = dict.fromkeys(game.PIECE_NAMES, 0)
        report = deadline.analyze_drop(game.EMPTY_BOARD, counts, 'I', 'CROSS', 0, 0, 0, 0)
        self.assertGreater(report['cross_after'], 0)
        self.assertNotIn('forced_loss_probability_lower_bound', report)
        self.assertFalse(any(counts.values()))
        with self.assertRaises(ValueError):
            deadline.analyze_drop(game.EMPTY_BOARD, counts, 'I', 'I', 0, 0, 0, 0)


if __name__ == '__main__':
    unittest.main()
