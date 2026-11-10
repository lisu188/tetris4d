import argparse
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_tetris_4d_optimized as game
from tools import train_agent as trainer


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.original = dict(game.BASELINE_PARAMETERS)
        game.configure_network(self.original)
        self.alternative = dict(self.original, holes=-2.4, roughness=-.13, threshold=.8)

    def tearDown(self):
        game.configure_network(self.original)

    def test_parameters_round_trip(self):
        game.configure_network(self.alternative, {'method': 'test'})
        self.assertEqual(game.network_parameters(), self.alternative)
        self.assertEqual(game.network_metadata()['training'], {'method': 'test'})
        self.assertEqual(game.NETWORK[0].bias, -.8)
        self.assertEqual(game.NETWORK[-1].edges[game.AREA][1], -2.4)
        self.assertTrue(all(n.activation == 'sigmoid' for n in game.NETWORK[:-1]))
        self.assertEqual(len(game.NETWORK), 23041)

    def test_changed_weights_match_expanded_sigmoid(self):
        rng = random.Random(414)
        for threshold in (.25, .8, 2.0, 6.0):
            game.configure_network(dict(self.alternative, threshold=threshold))
            heights = tuple(rng.randrange(17) for _ in range(game.AREA))
            values = game.network_forward(heights, 427, 2)
            hidden = []
            for a, b in game.NEIGHBORS:
                d = heights[b] - heights[a]
                hidden.extend((game.sigmoid(d - threshold), game.sigmoid(-d - threshold)))
            self.assertEqual(values[game.INPUT_COUNT:-1], hidden)
            output = game.NETWORK[-1]
            expected = output.bias + math.fsum(values[source] * weight for source, weight in output.edges)
            self.assertEqual(values[-1], expected)
            self.assertAlmostEqual(values[-1], game.network_score(heights, 427, 2), places=7)

    def test_cache_invalidation_and_baseline_restore(self):
        board = tuple((1 << (i % 9)) - 1 for i in range(game.AREA))
        original = game.evaluate_board(board)
        original_trace = game.network_forward(original.heights, original.holes, 0)
        if game.BACKEND == 'numpy':
            game.numpy_tables()
            game.numpy_network_data()
        game.configure_network(self.alternative)
        self.assertEqual(game.evaluate_board.cache_info().currsize, 0)
        changed = game.evaluate_board(board)
        self.assertNotEqual(original.score, changed.score)
        game.configure_network(self.original)
        self.assertEqual(game.evaluate_board(board).score, original.score)
        self.assertEqual(game.network_forward(original.heights, original.holes, 0), original_trace)

    def test_old_traces_cannot_mix_new_parameters(self):
        trace = game.network_trace((0,) * game.AREA, 0, 0)
        game.configure_network(self.alternative)
        with self.assertRaises(RuntimeError):
            trace[game.INPUT_COUNT]
        with self.assertRaises(RuntimeError):
            trace.expand()

    def test_invalid_parameters_do_not_mutate_model(self):
        cases = [None, {}, dict(self.original, extra=3)]
        for value in (float('nan'), float('inf'), True, '3', 1, -101):
            cases.append(dict(self.original, holes=value))
        cases.extend((dict(self.original, threshold=.1), dict(self.original, threshold=7),
                      dict(self.original, layers=-1), dict(self.original, height=0)))
        for parameters in cases:
            with self.assertRaises(ValueError):
                game.configure_network(parameters)
            self.assertEqual(game.network_parameters(), self.original)

    def test_checkpoint_loading(self):
        checkpoint = {'schema': 1, 'kind': 'tetris4d-shared-sigmoid', 'well': [16] * 4,
                      'parameters': self.alternative, 'training': {'method': 'test'}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.json'
            trainer.atomic_json(path, checkpoint)
            self.assertEqual(game.load_checkpoint(path), checkpoint)
            self.assertEqual(game.network_parameters(), self.alternative)
            for key, value in (('schema', 2), ('well', [4] * 4), ('kind', 'other')):
                trainer.atomic_json(path, dict(checkpoint, **{key: value}))
                with self.assertRaises(ValueError):
                    game.load_checkpoint(path)
            path.write_text('{' + ' ' * 2_000_000)
            with self.assertRaises(ValueError):
                game.load_checkpoint(path)

    def test_curricula_use_full_4d_board_and_separate_seeds(self):
        training = trainer.curriculum(101, 8)
        validation = trainer.curriculum(1001, 8)
        heldout = trainer.curriculum(10001, 16)
        sets = [set(s['seed'] for s in split) for split in (training, validation, heldout)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for spec in training:
            board = trainer.initial_board(spec['kind'], spec['seed'])
            self.assertEqual(len(board), 4096)
            self.assertTrue(all(0 <= column < 65536 for column in board))
            self.assertEqual(game.full_layer_mask(board), 0)
            self.assertEqual(board, trainer.initial_board(spec['kind'], spec['seed']))

    def test_deterministic_rollout_and_counters(self):
        spec = {'seed': 312, 'kind': 'empty', 'piece_set': '4d'}
        game.configure_network(self.alternative)
        first = trainer.rollout(spec, 6, 1, 4, 128)
        second = trainer.rollout(spec, 6, 1, 4, 128)
        self.assertEqual(first, second)
        self.assertEqual(first['pieces'], 6)
        self.assertGreater(first['cells_placed'], 0)
        self.assertTrue(math.isfinite(first['fitness']))

    def test_parameter_search_bounds(self):
        for vector in ((-1000,) * 3, (1000,) * 3, trainer.encode(self.original)):
            parameters = trainer.decode(vector)
            game.configure_network(parameters)
            self.assertEqual(parameters['height'], self.original['height'])
            self.assertEqual(parameters['layers'], self.original['layers'])
            self.assertTrue(.25 - 1e-14 <= parameters['threshold'] <= 6)

    def test_random_state_serialization(self):
        rng = random.Random(418)
        rng.gauss(0, 1)
        state = json.loads(json.dumps(rng.getstate()))
        other = random.Random()
        other.setstate(trainer.tuple_tree(state))
        self.assertEqual([rng.gauss(0, 1) for _ in range(20)], [other.gauss(0, 1) for _ in range(20)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('auto', 'python', 'numpy'), default='auto')
    args, rest = parser.parse_known_args()
    game.configure_backend(args.backend)
    unittest.main(argv=[sys.argv[0]] + rest)
