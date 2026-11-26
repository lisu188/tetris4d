import dataclasses
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tetris_neural as neural
import train_overnight as base

game = neural.game


class SimulationFastPathsTests(unittest.TestCase):
    def setUp(self):
        previous = game.BACKEND, game.TWO_STAGE_EXHAUSTIVE, game.HEADLESS_COMMIT
        self.addCleanup(game.set_simulation_fast_paths, two_stage=previous[1], headless=previous[2])
        self.addCleanup(game.configure_backend, previous[0])
        if game.configure_backend('native') != 'native' or not hasattr(game._NATIVE, 't4_checked_apply'):
            self.skipTest('Build the current native extension to test staged simulation')
        game.set_simulation_fast_paths()

    def test_local_shortlist_override_survives_target_recovery(self):
        state = game.Game(11)
        state.current = 'I'
        state.board = (2,) + (0,) * (game.AREA - 1)
        config = neural.Config(screening='exhaustive', shortlist=4, shortlist_mode='teacher-top', recover_target=True)
        with patch.object(neural, '_native_shortlist', wraps=neural._native_shortlist) as shortlist:
            local = neural.prepare(state, 0, config, two_stage=True)
            shortlist.assert_called_once()
            self.assertFalse(game.TWO_STAGE_EXHAUSTIVE)
            shortlist.reset_mock()
            dense = neural.prepare(state, 0, config)
            shortlist.assert_not_called()
        self.assertEqual(int(local.arrays['targets']), 1)
        self.assertEqual(local.options, dense.options)
        np.testing.assert_array_equal(local.arrays['teacher_prior'], dense.arrays['teacher_prior'])

    def test_native_shortlist_ties_anchored_clear_and_recovery(self):
        boards = [
            ('BRIDGE', game.EMPTY_BOARD, 1),
            ('L', (8191, 32767, 0, 255) * (game.AREA // 4), 15),
            ('I', (0,) + (65535,) * (game.AREA - 1), 15),
            ('CROSS', (65535,) * game.AREA, 13),
        ]
        for mode in ('teacher-top', 'teacher-random'):
            config = neural.Config(shortlist_mode=mode, shortlist=17, screening='exhaustive', recover_target=True)
            for piece, board, target in boards:
                with self.subTest(piece=piece, target=target, mode=mode):
                    state = game.Game(11)
                    state.current, state.board = piece, board
                    game.set_simulation_fast_paths()
                    dense = neural.prepare(state, target, config)
                    game.set_simulation_fast_paths(two_stage=True)
                    staged = neural.prepare(state, target, config)
                    if dense is None:
                        self.assertIsNone(staged)
                        continue
                    self.assertEqual((staged.options, staged.teacher, staged.evaluations, staged.fallback),
                                     (dense.options, dense.teacher, dense.evaluations, dense.fallback))
                    self.assertEqual(set(staged.arrays), set(dense.arrays))
                    for field, value in dense.arrays.items():
                        np.testing.assert_array_equal(value, staged.arrays[field], err_msg=field)

    def test_native_and_rendered_history_and_hashes(self):
        from tools.benchmark_spatial_ppo import boundary_episodes
        import train_survival_v4 as v4
        for index in range(2):
            dense = boundary_episodes(v4, (0, 1))[index]
            staged = boundary_episodes(v4, (0, 1))[index]
            config = neural.Config(shortlist=12, candidates=128, recover_target=True)
            for step in range(25):
                game.set_simulation_fast_paths()
                prepared = neural.prepare(dense.state, dense.target, config)
                if prepared is None:
                    break
                candidate = prepared.options[prepared.teacher]
                expected = base.apply_move(dense, candidate, 1, 'parity')
                game.set_simulation_fast_paths(headless=True)
                actual = base.apply_move(staged, candidate, 1, 'parity')
                self.assertEqual(expected, actual)
                self.assertEqual(dense.moves[-1], staged.moves[-1])
                self.assertEqual(dense.state.summary(), staged.state.summary())
                self.assertEqual((dense.state.current, dense.state.next_piece), (staged.state.current, staged.state.next_piece))
                self.assertEqual(base.board_hash(staged.state.board), staged.moves[-1]['board_sha256'])
                self.assertEqual(staged.state.locked_pieces, ())
            self.assertGreater(len(staged.moves), 0)
            if index == 0:
                self.assertEqual(staged.moves[0]['cleared_mask'], 15)

    def test_rejected_candidate_is_atomic(self):
        state = game.Game(25)
        move = game.make_move(state.board, state.current, 0, 0, 0, 0)
        game.set_simulation_fast_paths(headless=True)
        for change in ({'z': 1}, {'x': 2**40}, {'rotation': -1}, {'piece': state.next_piece},
                       {'layers': 1}, {'score': move.score + 1}):
            with self.subTest(change=change):
                with self.assertRaises((ValueError, AssertionError)):
                    game.play_candidate(state, dataclasses.replace(move.candidate, **change))
                self.assertIsNone(state.packed_board)
                self.assertEqual(state.board, game.EMPTY_BOARD)
                self.assertEqual(state.pieces, 0)
        self.assertEqual(game.play_candidate(state, move.candidate), move)
        self.assertIsNotNone(state.packed_board)
        state.board = game.EMPTY_BOARD
        self.assertIsNone(state.packed_board)
