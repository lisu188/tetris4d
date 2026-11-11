from __future__ import annotations

import argparse
from pathlib import Path
import json
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def independent_future_holes(board, candidate, target):
    columns = list(board)
    touched = set()
    for x, y, z, w in game.placed_cells(candidate):
        index = game.column_index(x, y, w)
        assert not columns[index] & (1 << z)
        columns[index] |= 1 << z
        touched.add(index)
    def count(mask):
        return sum(not mask & (1 << level) for level in range(target + 1, mask.bit_length()))
    return sum(count(columns[i]) - count(board[i]) for i in touched)


class SustainedTests(unittest.TestCase):
    def setUp(self):
        self.backend = game.BACKEND
        game.configure_backend('auto')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def tearDown(self):
        game.configure_backend(self.backend)

    def test_objective_is_frozen(self):
        self.assertEqual(game.FUTURE_HOLE_PENALTY, 4)

    def test_future_holes_match_independent_occupancy(self):
        rng = random.Random(914)
        for target in (0, 1, 5, 15):
            board = tuple((1 << rng.randrange(0, 6)) - 1 for _ in range(game.AREA))
            context = game.evaluate_board(board)
            for piece in game.PIECE_NAMES:
                for spec in game.specifications(piece, 24, target):
                    candidate = game.score_candidate(context, piece, *spec)
                    if candidate is not None:
                        gain, buried, holes = game.sustained_candidate_metrics(board, candidate, target)
                        self.assertEqual((gain, buried), game.target_candidate_metrics(board, candidate, target))
                        self.assertEqual(holes, independent_future_holes(board, candidate, target))
                        self.assertGreaterEqual(holes, 0)

    def test_scalar_and_vector_metrics_agree(self):
        if game._NUMPY is None:
            self.skipTest('Optional NumPy is absent')
        game.configure_backend('numpy')
        board = tuple((1 << ((i * 17 + i // 16) % 5)) - 1 for i in range(game.AREA))
        context = game.evaluate_board(board)
        for piece in game.PIECE_NAMES:
            specs = tuple(game.specifications(piece, 67, 97))
            scores, z, _ = game.PlacementBatch(piece, specs).scores(context)
            for target in (0, 1, 4, 15):
                gains, blocked, holes = game.sustained_batch_metrics(context, piece, specs, z, target)
                for index, spec in enumerate(specs):
                    candidate = game.score_candidate(context, piece, *spec)
                    if candidate is not None:
                        expected = game.sustained_candidate_metrics(board, candidate, target)
                        self.assertEqual((int(gains[index]), bool(blocked[index]), int(holes[index])),
                                         (expected[0], bool(expected[1]), expected[2]))

    def test_backend_decisions_keep_scores_and_stream(self):
        self.compare(32, ('python', 'numpy') if game._NUMPY is not None else ('python',), candidates=96)

    def compare(self, count, backends, candidates=2048):
        reference = []
        reports = []
        for backend in backends:
            game.configure_backend(backend)
            state, target, events = game.Game(7), 1, []
            for index in range(count):
                decision = game.choose_sustained_move(state.board, state.current, target, candidates, state.pieces)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.policy, 'sustained')
                gain, buried, _ = game.sustained_candidate_metrics(state.board, decision.move.candidate, target)
                self.assertEqual(buried, 0)
                self.assertEqual(decision.coverage_gain, gain)
                state.commit(decision.move)
                snapshot = (decision.move.candidate, state.board, state.current, state.next_piece, decision.evaluations)
                if backend == backends[0]:
                    reference.append(snapshot)
                else:
                    self.assertEqual(snapshot, reference[index])
                self.assertEqual(sum(mask.bit_count() for mask in state.board), state.cells_placed - state.layers * game.AREA)
                if decision.move.layers:
                    events.append(state.pieces)
                    target = game.next_hyperlayer_target(state.board, target)
            reports.append(dict(backend=backend, moves=state.pieces, clears=events, summary=state.summary()))
        return reports

    def test_single_clear_and_full_plane_mass(self):
        board = (0,) + (2,) * (game.AREA - 1)
        decision = game.choose_sustained_move(board, 'I', target=1, candidates=64)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.move.layers, 1)
        self.assertEqual(decision.coverage_gain, 1)
        self.assertEqual(sum(mask.bit_count() for mask in decision.move.board), 3)

    def test_invalid_inputs_and_buried_target(self):
        for target in (-1, 16, True):
            with self.assertRaises(ValueError):
                game.choose_sustained_move(game.EMPTY_BOARD, 'I', target=target)
        for count in (-1, True):
            with self.assertRaises(ValueError):
                game.choose_sustained_move(game.EMPTY_BOARD, 'I', candidates=count)
        self.assertIsNone(game.choose_sustained_move((4,) + game.EMPTY_BOARD[1:], 'I', target=1))
        self.assertIsNone(game.choose_sustained_move((2,) * game.AREA, 'I', target=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory-pieces', type=int, default=0)
    args = parser.parse_args()
    game.configure_backend('auto')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SustainedTests))
    report = dict(passed=result.wasSuccessful(), unit_tests=result.testsRun)
    if result.wasSuccessful() and args.trajectory_pieces:
        game.configure_backend('auto')
        backends = ('numpy', 'native') if game.BACKEND == 'native' else ('numpy',)
        test = SustainedTests()
        test.setUp()
        try:
            report['trajectories'] = test.compare(args.trajectory_pieces, backends)
        finally:
            test.tearDown()
    print(json.dumps(report, indent=2))
    raise SystemExit(not result.wasSuccessful())
