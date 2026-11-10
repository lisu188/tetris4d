import argparse
import json
import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_tetris_4d_optimized as game


def literal_forward(heights, holes, layers):
    values = list(map(float, heights)) + [float(holes), float(layers)]
    for a, b in game.NEIGHBORS:
        d = values[b] - values[a]
        values.extend((game.sigmoid(d - 2), game.sigmoid(-d - 2)))
    weights = [game.HEIGHT_WEIGHT] * game.AREA + [game.HOLES_WEIGHT, game.LAYERS_WEIGHT]
    weights += [game.HIDDEN_OUTPUT_WEIGHT] * (2 * len(game.NEIGHBORS))
    values.append(game.OUTPUT_BIAS + math.fsum(value * weight for value, weight in zip(values, weights)))
    return values


class NetworkStructureTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(8303)
        game.pooled_histograms.cache_clear()
        game.sigmoid_roughness.cache_clear()

    def test_shared_graph_is_lazy_and_compatible(self):
        self.assertIsInstance(game.NETWORK, game.SharedSigmoidNetwork)
        self.assertLess(sys.getsizeof(game.NETWORK), 128)
        self.assertEqual(len(game.NETWORK), 23041)
        self.assertEqual(sum(len(n.edges) for n in game.NETWORK), 73218)
        for i in (0, 1, 476, 7679, 7680, 15360, 23039):
            pair, direction = divmod(i, 2)
            a, b = game.NEIGHBORS[pair]
            sign = 1.0 if direction == 0 else -1.0
            node = game.NETWORK[i]
            self.assertEqual(node.edges, ((b, sign), (a, -sign)))
            self.assertEqual((node.bias, node.activation), (-2.0, 'sigmoid'))
        self.assertEqual(game.NETWORK[-1].activation, 'linear')
        self.assertEqual(game.NETWORK[:4], tuple(game.NETWORK[i] for i in range(4)))
        for index in (-23042, 23041):
            with self.assertRaises(IndexError):
                game.NETWORK[index]

    def test_integral_traces_match_literal_graph_exactly(self):
        boards = [(h,) * game.AREA for h in (0, 1, 8, 16)]
        boards += [tuple(self.rng.randrange(17) for _ in range(game.AREA)) for _ in range(20)]
        for heights in boards:
            holes, layers = self.rng.randrange(61441), self.rng.randrange(17)
            expected = literal_forward(heights, holes, layers)
            trace = game.network_trace(heights, holes, layers)
            self.assertEqual(trace.expand(), expected)
            self.assertEqual(game.network_forward(heights, holes, layers), expected)
            self.assertEqual(trace[-1], expected[-1])
            self.assertEqual(trace.active_count, sum(v >= .5 for v in expected[game.INPUT_COUNT:-1]))
            self.assertEqual(trace[game.INPUT_COUNT:game.INPUT_COUNT + 32], expected[game.INPUT_COUNT:game.INPUT_COUNT + 32])
            self.assertEqual([trace[i] for i in range(0, len(trace), 113)], expected[::113])

    def test_fractional_and_outside_domain_traces(self):
        for heights in ((.5,) * game.AREA, tuple(self.rng.uniform(-20, 20) for _ in range(game.AREA))):
            expected = literal_forward(heights, 3.5, .25)
            trace = game.network_trace(heights, 3.5, .25)
            self.assertIsNone(trace.histograms)
            self.assertEqual(trace.expand(), expected)
            self.assertEqual(trace.active_count, sum(v >= .5 for v in expected[game.INPUT_COUNT:-1]))

    def test_histograms_count_all_inputs_and_pairs(self):
        heights = tuple(self.rng.randrange(17) for _ in range(game.AREA))
        counts, differences = game.pooled_histograms(heights)
        self.assertEqual(sum(counts), 4096)
        self.assertEqual(sum(differences), 11520)
        self.assertEqual(counts, tuple(heights.count(h) for h in range(17)))
        expected = [0] * 17
        for a, b in game.NEIGHBORS:
            expected[abs(heights[b] - heights[a])] += 1
        self.assertEqual(differences, tuple(expected))
        roughness = math.fsum(game.ROUGHNESS_TABLE[heights[b] - heights[a] + 16] for a, b in game.NEIGHBORS)
        self.assertEqual(game.sigmoid_roughness(heights), roughness)

    def test_pooled_rounding_matches_fsum(self):
        for _ in range(40):
            counts = [self.rng.randrange(1500) for _ in range(17)]
            expected = math.fsum(game.ROUGHNESS_TABLE[16 + d] for d, count in enumerate(counts) for _ in range(count))
            actual = sum(count * tick for count, tick in zip(counts, game.ROUGHNESS_TICKS)) / game.ROUGHNESS_DENOMINATOR
            self.assertEqual(actual, expected)

    def test_bounded_cache(self):
        for h in range(17):
            game.pooled_histograms((h,) * game.AREA)
        self.assertEqual(game.pooled_histograms.cache_info().maxsize, 16)
        self.assertLessEqual(game.pooled_histograms.cache_info().currsize, 16)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            game.network_trace((0,) * (game.AREA - 1), 0, 0)
        for bad in (math.nan, math.inf, -math.inf):
            for heights, holes, layers in (((bad,) + (0,) * (game.AREA - 1), 0, 0), ((0,) * game.AREA, bad, 0), ((0,) * game.AREA, 0, bad)):
                with self.assertRaises(ValueError):
                    game.network_trace(heights, holes, layers)
        for heights in ((-1,) * game.AREA, (17,) * game.AREA, (.5,) * game.AREA):
            with self.assertRaises(ValueError):
                game.pooled_histograms(heights)
        trace = game.network_trace((0,) * game.AREA, 0, 0)
        for index in (-len(trace) - 1, len(trace)):
            with self.assertRaises(IndexError):
                trace[index]

    def test_compact_export_is_explicit_about_logical_size(self):
        metadata = json.loads(json.dumps(game.network_metadata()))
        execution = metadata['execution']
        self.assertEqual(execution['shared_sigmoid_filters'], 2)
        self.assertEqual(execution['absolute_difference_bins'], 17)
        self.assertEqual(execution['sigmoid_lookup_entries'], 33)
        self.assertEqual(execution['logical_hidden_activations'], 23040)
        self.assertEqual(execution['stored_neuron_objects'], 0)
        self.assertEqual(metadata['nonzero_connections'], 73218)
        self.assertNotIn('nodes', metadata)
        self.assertLess(len(json.dumps(metadata)), 10000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('auto', 'python', 'numpy'), default='auto')
    args, remaining = parser.parse_known_args()
    game.configure_backend(args.backend)
    unittest.main(argv=[sys.argv[0]] + remaining)
