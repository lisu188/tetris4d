from itertools import permutations
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_cross_closure_suffixes import public_remainder, sample_draws, summarize


class BagSamplingTests(unittest.TestCase):
    def test_public_counts_across_current_and_preview_bag_boundaries(self):
        names = ('A', 'B', 'C')
        for first in permutations(names):
            for second in permutations(names):
                stream = first + second
                for placed in range(5):
                    dealt = placed + 2
                    snapshot = {'pieces': placed, 'piece_counts': {p: stream[:placed].count(p) for p in names},
                                'current': stream[placed], 'next_piece': stream[placed + 1]}
                    end_of_bag = ((dealt + 2) // 3) * 3
                    self.assertEqual(public_remainder(snapshot, names), sorted(stream[dealt:end_of_bag]))

    def test_changing_hidden_order_cannot_change_sampled_future(self):
        from sparse_tetris_4d_optimized import PIECE_SETS, PieceStream
        from types import SimpleNamespace

        names = PIECE_SETS['expanded']
        snapshot = {'pieces': 0, 'piece_counts': dict.fromkeys(names, 0), 'piece_set': 'expanded',
                    'current': names[0], 'next_piece': names[1], 'bag': list(names[2:])}
        game = SimpleNamespace(PieceStream=PieceStream)
        first = sample_draws(game, snapshot, 'test', 42, 0, 45)
        snapshot['bag'].reverse()
        second = sample_draws(game, snapshot, 'test', 42, 0, 45)
        self.assertEqual(first, second)
        self.assertNotEqual(first, sample_draws(game, snapshot, 'test', 42, 1, 45))
        rng = random.Random()
        rng.setstate(first[1])
        self.assertEqual(first[2][2:2 + len(first[0])], first[0][::-1])
        next_bag = list(names)
        rng.shuffle(next_bag)
        self.assertEqual(first[2][len(names):2 * len(names)], next_bag[::-1])


class CoverageTests(unittest.TestCase):
    def test_controller_stop_leaves_full_cohort_estimate_undefined(self):
        roots, mean = summarize([{'seed': 2, 'difference': 12}, {'seed': 2, 'difference': None},
                                 {'seed': 3, 'difference': None}])
        self.assertIsNone(mean)
        self.assertEqual(roots['2']['resolved'], 1)
        self.assertEqual(roots['2']['pairs'], 2)
        self.assertEqual(roots['2']['mean_resolved_difference'], 12)
        self.assertIsNone(roots['3']['mean_resolved_difference'])

    def test_root_means_contribute_equally(self):
        roots, mean = summarize([{'seed': 2, 'difference': -4}, {'seed': 2, 'difference': 0},
                                 {'seed': 3, 'difference': 6}])
        self.assertEqual(mean, 2)
        self.assertEqual(roots['2']['losses'], 1)
        self.assertEqual(roots['2']['ties'], 1)
        self.assertEqual(roots['3']['wins'], 1)
