from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.parameters, self.training = game.network_parameters(), game.MODEL_TRAINING
        self.environment = os.environ.get('TETRIS_NATIVE_ISA')
        game.configure_backend('native')

    def tearDown(self):
        if self.environment is None:
            os.environ.pop('TETRIS_NATIVE_ISA', None)
        else:
            os.environ['TETRIS_NATIVE_ISA'] = self.environment
        game.configure_network(self.parameters, self.training)
        game.configure_backend('native')

    def test_features_match_reference_for_arbitrary_bitboards(self):
        rng = random.Random(5003)
        boards = [game.EMPTY_BOARD, (65535,) * game.AREA]
        boards.extend(tuple(rng.getrandbits(16) for _ in range(game.AREA)) for _ in range(8))
        for board in boards:
            expected_heights, expected_holes = game.board_features(board)
            context = game.evaluate_board(board)
            self.assertEqual(context.heights, expected_heights)
            self.assertEqual(context.holes, expected_holes)
            self.assertEqual(context.fill_counts, tuple(sum((m >> z) & 1 for m in board) for z in range(16)))
            for target in range(16):
                status = game.target_layer_status(board, target)
                self.assertEqual(status.buried, sum(not (m & (1 << target)) and bool(m >> (target+1)) for m in board))

    def test_scalar_and_simd_scores_are_bit_identical(self):
        rng = random.Random(31951)
        boards = [game.EMPTY_BOARD, (65535,) * game.AREA]
        boards.extend(tuple(rng.getrandbits(rng.randrange(1, 17)) for _ in range(game.AREA)) for _ in range(4))
        trained = json.loads((ROOT / 'models/trained-sigmoid.json').read_text())['parameters']
        models = [game.BASELINE_PARAMETERS, trained,
                  dict(trained, threshold=.25), dict(trained, threshold=6.)]
        count = 0
        for parameters in models:
            game.configure_network(parameters)
            for board in boards:
                for piece in game.PIECE_NAMES:
                    specs = tuple(game.specifications(piece, 129, 7))
                    game.configure_backend('numpy')
                    expected, expected_z, _ = game.PlacementBatch(piece, specs).scores(game.evaluate_board(board))
                    for isa in ('scalar', 'auto'):
                        os.environ['TETRIS_NATIVE_ISA'] = isa
                        game.configure_backend('native')
                        actual, z, _ = game.PlacementBatch(piece, specs).scores(game.evaluate_board(board))
                        self.assertTrue((expected == actual).all(), (piece, parameters, isa))
                        self.assertTrue((expected_z == z).all(), (piece, isa))
                        count += len(specs)
        print(json.dumps({'native_candidate_comparisons': count, 'maximum_absolute_error': 0.0}))

    def test_every_orientation_at_boundaries_and_simd_tails(self):
        rng = random.Random(933)
        board = tuple((1 << rng.randrange(12)) - 1 for _ in range(game.AREA))
        comparisons = 0
        for piece, shapes in game.SHAPES.items():
            specs = []
            for r, shape in enumerate(shapes):
                limits = (16-shape.extents[0], 16-shape.extents[1], 16-shape.extents[3])
                origins = [(0, 0, 0), limits]
                origins += [tuple(rng.randrange(limit+1) for limit in limits) for _ in range(7)]
                specs.extend((r, *origin) for origin in origins)
            for start in range(0, len(specs), 512):
                batch = tuple(specs[start:start+512])
                game.configure_backend('numpy')
                expected, expected_z, _ = game.PlacementBatch(piece, batch).scores(game.evaluate_board(board))
                for isa in ('scalar', 'auto'):
                    os.environ['TETRIS_NATIVE_ISA'] = isa
                    game.configure_backend('native')
                    scores, z, _ = game.PlacementBatch(piece, batch).scores(game.evaluate_board(board))
                    self.assertTrue((scores == expected).all(), (piece, start, isa))
                    self.assertTrue((z == expected_z).all())
                    comparisons += len(batch)
        print(json.dumps({'all_orientation_score_comparisons': comparisons, 'orientations': sum(map(len, game.SHAPES.values()))}))

    def test_neural_lookahead_keeps_identical_choices(self):
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
        for piece_set in ('classic', 'expanded', '4d'):
            state = game.Game(18, piece_set)
            for _ in range(24):
                game.configure_backend('numpy')
                reference = game.choose_move(state.board, state.current, state.next_piece, 2, 4, 256, state.pieces)
                game.configure_backend('native')
                actual = game.choose_move(state.board, state.current, state.next_piece, 2, 4, 256, state.pieces)
                self.assertEqual(actual.move, reference.move)
                self.assertEqual(actual.search_score, reference.search_score)
                self.assertEqual(actual.evaluations, reference.evaluations)
                state.commit(actual.move)

    def test_target_filter_keeps_lexicographic_winner(self):
        rng = random.Random(849)
        for target in (0, 1, 6, 15):
            board = tuple((1 << rng.randrange(target+2)) - 1 for _ in range(game.AREA))
            for piece in game.PIECE_NAMES:
                specs = tuple(game.specifications(piece, 127, 19))
                context = game.evaluate_board(board)
                options = []
                for i, spec in enumerate(specs):
                    candidate = game.score_candidate(context, piece, *spec)
                    if candidate is not None:
                        gain, buried = game.target_candidate_metrics(board, candidate, target)
                        if not buried:
                            options.append(((gain, candidate.score, -i), i))
                scores, _, _, gains = game.native_scores(context, piece, specs, target)
                actual = [(int(gains[i]), float(value), -i) for i, value in enumerate(scores) if math.isfinite(value)]
                self.assertEqual(max(actual) if actual else None, max(options)[0] if options else None)

    def test_multiple_layer_compaction_matches_all_column_masks(self):
        np = game._NUMPY
        for isa in ('scalar', 'auto'):
            os.environ['TETRIS_NATIVE_ISA'] = isa
            game.configure_backend('native')
            for cleared in (0, 1, 2, 0x8000, 0x5555, 0x8421, 0xffff):
                keep = [z for z in range(16) if not cleared & (1 << z)]
                for start in range(0, 65536, game.AREA):
                    columns = np.arange(start, start+game.AREA, dtype=np.uint16)
                    result = np.empty_like(columns)
                    game._native_check(game._NATIVE.t4_compact(columns.ctypes.data, result.ctypes.data, cleared, int(game._NATIVE_USE_BMI2)))
                    reference = np.zeros(game.AREA, dtype=np.uint16)
                    for new, old in enumerate(keep):
                        reference |= ((columns >> old) & 1) << new
                    self.assertTrue((reference == result).all(), (isa, cleared, start))

    def test_native_clear_candidates_use_reference_rescoring(self):
        for target in (0, 1, 8, 15):
            r = next(i for i, s in enumerate(game.SHAPES['I']) if len(s.columns) == 1)
            board = [((1 << (target+1)) - 1)] * game.AREA
            board[0] = (1 << max(0, target-3)) - 1
            board = tuple(board)
            candidate = game.score_candidate(game.evaluate_board(board), 'I', r, 0, 0, 0)
            self.assertIsNotNone(candidate)
            self.assertGreater(candidate.layers, 0)
            scores, _, clears, _ = game.native_scores(game.evaluate_board(board), 'I', ((r, 0, 0, 0),), target)
            self.assertEqual(float(scores[0]), candidate.score)
            self.assertEqual(clears[0], candidate)

    def test_invalid_specs_and_targets_are_rejected(self):
        context = game.evaluate_board(game.EMPTY_BOARD)
        for specs in ((), ((9999, 0, 0, 0),), ((0, -1, 0, 0),), ((0, 0, 16, 0),), ((0, 0, 0),), ((0, 1.5, 0, 0),), ((0, 2**40, 0, 0),)):
            with self.assertRaises(ValueError):
                game.native_scores(context, 'I', specs)
        for target in (-2, 16, True, 1.5, 2**32):
            with self.assertRaises(ValueError):
                game.native_scores(context, 'I', ((0, 0, 0, 0),), target)

    def test_checkpoint_changes_rebuild_native_delta_tables(self):
        spec = next(game.specifications('CROSS', 32, 3))
        board = (1,) + game.EMPTY_BOARD[1:]
        for threshold in (2., 5., 2.):
            parameters = dict(game.BASELINE_PARAMETERS, threshold=threshold, roughness=-.3)
            game.configure_network(parameters)
            context = game.evaluate_board(board)
            expected = game.score_candidate(context, 'CROSS', *spec)
            scores, _, _, _ = game.native_scores(context, 'CROSS', (spec,))
            self.assertEqual(float(scores[0]), expected.score)

    def test_auto_fallback_and_explicit_native_failure(self):
        with patch.object(game, '_load_native', return_value=False):
            self.assertEqual(game.configure_backend('auto'), 'numpy')
            with self.assertRaises(ValueError):
                game.configure_backend('native')
        with patch.object(game, '_load_native', side_effect=OSError('Invalid library')), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(game.configure_backend('auto'), 'numpy')
        self.assertEqual(game.configure_backend('native'), 'native')

    def test_no_arguments_keep_trained_continuous_defaults(self):
        with patch.object(game, 'launch_demo', return_value=0) as launch:
            self.assertEqual(game.main([]), 0)
        args = launch.call_args.args[0]
        self.assertEqual(args.policy, 'sustained')
        self.assertFalse(args.stop_after_clear)
        self.assertEqual(game.BACKEND, 'native')

    def test_scalar_dispatch_never_enables_optional_isa(self):
        os.environ['TETRIS_NATIVE_ISA'] = 'scalar'
        game.configure_backend('native')
        self.assertFalse(game.native_backend_info()['avx2_enabled'])
        self.assertFalse(game.native_backend_info()['bmi2_enabled'])
        os.environ['TETRIS_NATIVE_ISA'] = 'invalid'
        with self.assertRaises(ValueError):
            game.configure_backend('native')


def compare_trajectories(pieces: int) -> dict:
    game.configure_backend('numpy')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    reference = game.Game(7)
    target, expected, clear_events = 1, [], []
    for _ in range(pieces):
        decision = game.choose_hyperlayer_move(reference.board, reference.current, target, salt=reference.pieces)
        if decision is None:
            raise AssertionError('Reference planner stopped before cap.')
        reference.commit(decision.move)
        expected.append((decision.move.candidate, decision.search_score, decision.evaluations, reference.board, reference.next_piece))
        if decision.move.layers:
            clear_events.append(reference.pieces)
            target = game.next_hyperlayer_target(reference.board, target)
    game.configure_backend('native')
    actual = game.Game(7)
    target = 1
    for candidate, score, evaluations, board, next_piece in expected:
        decision = game.choose_hyperlayer_move(actual.board, actual.current, target, salt=actual.pieces)
        if decision is None or (decision.move.candidate, decision.search_score, decision.evaluations) != (candidate, score, evaluations):
            raise AssertionError(f'Move differs at piece {actual.pieces+1}')
        actual.commit(decision.move)
        if actual.board != board or actual.next_piece != next_piece:
            raise AssertionError(f'Board or piece stream differs at piece {actual.pieces}')
        if decision.move.layers:
            target = game.next_hyperlayer_target(actual.board, target)
    return {'identical_moves': actual.pieces, 'clear_events': clear_events, 'active_target': target,
            'summary': actual.summary(), 'native': game.native_backend_info()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory-pieces', type=int, default=0)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    game.configure_backend('native')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeTests))
    report = {'passed': result.wasSuccessful(), 'unit_tests': result.testsRun}
    if result.wasSuccessful() and args.trajectory_pieces:
        report['trajectory'] = compare_trajectories(args.trajectory_pieces)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(not result.wasSuccessful())
