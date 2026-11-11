from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ContinuousTests(unittest.TestCase):
    def test_accessible_target_is_kept(self):
        self.assertEqual(game.next_hyperlayer_target(game.EMPTY_BOARD, 1), 1)
        self.assertEqual(game.next_hyperlayer_target((2,) + game.EMPTY_BOARD[1:], 1), 1)

    def test_buried_target_is_replaced_with_nearest_safe_target(self):
        board = (4,) + game.EMPTY_BOARD[1:]
        self.assertEqual(game.next_hyperlayer_target(board, 1), 2)
        self.assertEqual(game.target_layer_status(board, 1).buried, 1)
        self.assertEqual(game.target_layer_status(board, 2).buried, 0)

    def test_invalid_target_and_full_board(self):
        for target in (-1, 16, True, 1.5):
            with self.assertRaises(ValueError):
                game.next_hyperlayer_target(game.EMPTY_BOARD, target)
        self.assertIsNone(game.next_hyperlayer_target((65535,) * game.AREA, 1))

    def app_before_clear(self, running=True, stop=False, human=False, policy='hyperlayer'):
        app = game.TetrisApp.__new__(game.TetrisApp)
        app.game = game.Game(7)
        app.game.board = (0, 10) + (2,) * (game.AREA - 2)
        app.game.current = 'I'
        app.policy, app.target, app.human = Value(policy), Value('1'), Value(human)
        app.stop_after_clear = stop
        app.goal_complete = False
        app.running = running
        app.pending_step = False
        app.refresh_plan = Mock()
        app.planned = game.choose_hyperlayer_move(app.game.board, 'I', candidates=32).move
        self.assertEqual(app.planned.layers, 1)
        return app

    def test_autoplay_remains_running_after_a_clear(self):
        app = self.app_before_clear()
        app.place()
        self.assertTrue(app.running)
        self.assertFalse(app.goal_complete)
        self.assertEqual(app.game.layers, 1)
        self.assertEqual(app.target.get(), '2')
        app.refresh_plan.assert_called_once()
        self.assertEqual(sum(mask.bit_count() for mask in app.game.board), 4)

    def test_sustained_policy_retargets_and_continues(self):
        app = self.app_before_clear(policy='sustained')
        app.place()
        self.assertTrue(app.running)
        self.assertFalse(app.goal_complete)
        self.assertEqual(app.game.layers, 1)
        self.assertEqual(app.target.get(), '2')
        app.refresh_plan.assert_called_once()
        self.assertEqual(sum(mask.bit_count() for mask in app.game.board), 4)

    def test_single_step_does_not_start_autoplay(self):
        app = self.app_before_clear(running=False)
        app.place()
        self.assertFalse(app.running)
        self.assertFalse(app.goal_complete)
        self.assertEqual(app.target.get(), '2')

    def test_explicit_stop_on_clear_remains_available(self):
        for policy in ('hyperlayer', 'network', 'sustained'):
            app = self.app_before_clear(stop=True, policy=policy)
            app.place()
            self.assertFalse(app.running)
            self.assertTrue(app.goal_complete)
            self.assertEqual(app.target.get(), '1')

    def test_manual_play_is_not_locked_by_a_clear(self):
        app = self.app_before_clear(running=False, stop=True, human=True)
        app.place()
        self.assertFalse(app.goal_complete)
        self.assertEqual(app.target.get(), '1')

    def test_neural_only_mode_does_not_retarget(self):
        app = self.app_before_clear(policy='network')
        app.place()
        self.assertTrue(app.running)
        self.assertFalse(app.goal_complete)
        self.assertEqual(app.target.get(), '1')

    def test_no_clear_preserves_target_and_running_state(self):
        app = self.app_before_clear()
        app.game = game.Game(7)
        app.planned = game.choose_hyperlayer_move(app.game.board, app.game.current, candidates=32).move
        self.assertEqual(app.planned.layers, 0)
        app.place()
        self.assertTrue(app.running)
        self.assertFalse(app.goal_complete)
        self.assertEqual(app.target.get(), '1')

    def test_cli_continuation_is_default_but_stopping_is_explicit(self):
        for arguments, expected in (([], False), (['--stop-after-clear'], True)):
            with patch.object(game, 'launch_demo', return_value=0) as launch:
                self.assertEqual(game.main(arguments), 0)
            self.assertEqual(launch.call_args.args[0].stop_after_clear, expected)
            self.assertEqual(launch.call_args.args[0].policy, 'sustained')


class ContinuousIntegrationTests(unittest.TestCase):
    def test_legacy_policy_clears_twice_and_keeps_playing(self):
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
        result = game.run_game(7, 2600, 2, 4, 2048, 'hyperlayer')
        self.assertEqual(result['pieces'], 2600)
        self.assertEqual(result['hyperlayers'], 2)
        self.assertEqual(result['first_clear_piece'], 1251)
        self.assertEqual([e['piece'] for e in result['clear_events']], [1251, 2478])
        self.assertEqual([e['target'] for e in result['clear_events']], [1, 2])
        self.assertEqual(result['active_target'], 4)
        self.assertEqual(result['stop_reason'], 'piece_limit')
        self.assertTrue(result['capped'])
        self.assertFalse(result['game_over'])
        self.assertEqual(result['occupied_cells'], result['cells_placed'] - game.AREA * result['hyperlayers'])

    def test_explicit_first_clear_benchmark_stops(self):
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
        result = game.run_game(7, 2000, 2, 4, 2048, 'hyperlayer', stop_after_clear=True)
        self.assertEqual(result['pieces'], 1251)
        self.assertEqual(result['hyperlayers'], 1)
        self.assertEqual(result['stop_reason'], 'first_clear')
        self.assertEqual(result['active_target'], 1)
        self.assertFalse(result['capped'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('python', 'numpy', 'auto'), default='auto')
    parser.add_argument('--integration', action='store_true')
    args = parser.parse_args()
    game.configure_backend(args.backend)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContinuousTests)
    if args.integration:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ContinuousIntegrationTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
