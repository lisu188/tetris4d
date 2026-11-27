from itertools import permutations
import json
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_cross_closure_suffixes import public_remainder, sample_draws, summarize
from tools.verify_cross_closure_suffixes import arm_at_boundary, indexed, object_hash, paired_stats


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


class NaturalEndpointsTests(unittest.TestCase):
    def test_horizon_and_physical_top_out_have_different_continuation_rules(self):
        capped = dict(placements=128, clears=1, end='horizon', current='I', board_sha256='board')
        natural = dict(n=128, status='physical_top_out', clears=1, current='I', board_sha256='board',
                       target=1, stream_sha256='draw-state', snapshot_sha256='state',
                       draw_chain='draws', action_chain='actions')
        boundary = {'first_pass_' + k: natural[k] for k in
                    ('target', 'stream_sha256', 'snapshot_sha256', 'draw_chain', 'action_chain')}
        boundary['first_pass_endpoint_sha256'] = object_hash(capped)
        natural.update(boundary)
        with self.assertRaisesRegex(ValueError, 'later physical endpoint'):
            arm_at_boundary(capped, boundary, natural)
        arm_at_boundary(capped, boundary, dict(natural, n=129))
        capped['end'] = 'top_out'
        boundary['first_pass_endpoint_sha256'] = object_hash(capped)
        natural.update(boundary)
        arm_at_boundary(capped, boundary, natural)
        with self.assertRaisesRegex(ValueError, 'already-finished arm changed'):
            arm_at_boundary(capped, boundary, dict(natural, n=129))

    def test_all_tracked_natural_lifetimes_reproduce_both_endpoints(self):
        directory = Path(__file__).resolve().parents[1] / 'validation'
        capped = json.loads((directory / 'cross-closure-suffix-results.json').read_text())
        natural = json.loads((directory / 'cross-closure-suffix-natural.json').read_text())
        all_changes, all_capped_changes = [], []
        self.assertEqual(set(natural['roots']), set(capped['roots']))
        for seed, row in natural['roots'].items():
            for arm in ('greedy', 'best_preserving'):
                self.assertEqual(len(row[arm]), 8)
                self.assertEqual([min(128, n) for n in row[arm]], capped['roots'][seed][arm]['placements'])
                self.assertEqual([i for i, n in enumerate(row[arm]) if n > 128],
                                 capped['roots'][seed][arm]['alive_at_horizon'])
            changes = [p - g for g, p in zip(row['greedy'], row['best_preserving'])]
            for key, value in paired_stats(changes).items():
                self.assertEqual(value, row[key])
            self.assertEqual(sum(changes), row['net_placements'])
            all_changes.extend(changes)
            all_capped_changes.extend(min(128, p) - min(128, g) for g, p in zip(row['greedy'], row['best_preserving']))
        self.assertEqual(len(all_changes), 32)
        self.assertEqual(paired_stats(all_capped_changes), natural['overall']['at_128'])
        for key, value in paired_stats(all_changes).items():
            self.assertEqual(value, natural['overall'][key])
        for row in natural['roots'].values():
            self.assertEqual((sum(all_changes) - row['net_placements']) / 24, row['mean_without_this_root'])

    def test_pair_ids_must_be_unique_and_complete(self):
        row = {'seed': 1, 'replicate': 0}
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            indexed([row, row], {(1, 0), (1, 1)})
