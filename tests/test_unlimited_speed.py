from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
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


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += .0001
        return self.now


def decision():
    yield 1
    return SimpleNamespace(move=object())


def stub():
    app = game.TetrisApp.__new__(game.TetrisApp)
    app.closed = False
    app.running = True
    app.human = Value(False)
    app.speed = Value(0.0)
    app.speed_choice = Value('Unlimited')
    app.previous = 99.98
    app.progress = 0.0
    app.pending_step = False
    app.goal_complete = False
    app._in_tick = False
    app._force_draw = False
    app.planner_blocked = ''
    app.planned = app.decision = None
    app.search = decision()
    app.target = Value('1')
    app.game = SimpleNamespace(game_over=False, board=game.EMPTY_BOARD, current='I', pieces=0)
    app.root = Mock()
    app.start_button = Mock()
    app.draw = Mock()
    app.finish_plan = Mock()
    def place():
        app.game.pieces += 1
        app.pending_step = False
        app.planned = None
        app.search = decision()
    app.place = Mock(side_effect=place)
    return app


class UnlimitedTests(unittest.TestCase):
    def test_selector_contains_finite_presets_and_unlimited(self):
        self.assertEqual(tuple(game.PLAYBACK_SPEEDS), ('0.5/s', '1/s', '3/s', '6/s', '12/s', 'Unlimited'))
        self.assertEqual(game.PLAYBACK_SPEEDS['Unlimited'], 0.0)

    def test_speed_changes_preserve_pause_board_and_pending_plan(self):
        app = stub()
        app.running = False
        board, search = app.game.board, app.search
        planned = app.planned = object()
        with patch.object(game.time, 'perf_counter', return_value=101):
            app.change_speed()
            self.assertEqual(app.speed.get(), 0)
            app.speed_choice.set('12/s')
            app.change_speed()
        self.assertEqual(app.speed.get(), 12)
        self.assertEqual(app.previous, 101)
        self.assertFalse(app.running)
        self.assertIs(app.game.board, board)
        self.assertIs(app.search, search)
        self.assertIs(app.planned, planned)
        app.place.assert_not_called()

    def test_unlimited_can_commit_multiple_moves_in_one_tick(self):
        app = stub()
        with patch.object(game.time, 'perf_counter', side_effect=Clock()):
            app.tick()
        self.assertGreater(app.game.pieces, 1)
        self.assertLess(app.game.pieces, 100)
        self.assertEqual(app.root.after.call_args.args[0], 1)
        app.draw.assert_called_once()
        self.assertFalse(app._in_tick)

    def test_unfinished_search_yields_to_event_loop(self):
        app = stub()
        def pending():
            count = 0
            while True:
                count += 1
                yield count
        app.search = pending()
        with patch.object(game.time, 'perf_counter', side_effect=Clock()):
            app.tick()
        app.place.assert_not_called()
        self.assertGreater(app.evaluations, 0)
        self.assertLess(app.evaluations, 100)
        self.assertIsNotNone(app.search)
        self.assertEqual(app.root.after.call_args.args[0], 1)

    def test_paused_unlimited_plans_but_does_not_commit(self):
        app = stub()
        app.running = False
        with patch.object(game.time, 'perf_counter', side_effect=Clock()):
            app.tick()
        app.place.assert_not_called()
        self.assertIsNotNone(app.planned)
        self.assertFalse(app.running)
        self.assertGreater(app.root.after.call_args.args[0], 1)

    def test_single_step_commits_exactly_once_in_unlimited(self):
        app = stub()
        app.running = False
        app.pending_step = True
        with patch.object(game.time, 'perf_counter', side_effect=Clock()):
            app.tick()
        app.place.assert_called_once()
        self.assertFalse(app.running)
        self.assertFalse(app.pending_step)

    def test_finite_animation_still_waits_before_committing(self):
        app = stub()
        app.speed.set(3)
        app.search = None
        app.planned = object()
        with patch.object(game.time, 'perf_counter', side_effect=Clock()):
            app.tick()
        app.place.assert_not_called()
        self.assertGreater(app.progress, 0)
        self.assertLess(app.progress, 1)

    def test_stop_condition_prevents_an_extra_move(self):
        app = stub()
        def stop():
            app.game.pieces += 1
            app.running = False
            app.goal_complete = True
            app.search = None
            app.planned = None
        app.place.side_effect = stop
        with patch.object(game.time, 'perf_counter', side_effect=Clock()):
            app.tick()
        app.place.assert_called_once()
        self.assertFalse(app.running)

    def test_blocked_planner_still_pauses(self):
        app = stub()
        def blocked():
            yield 1
            return None
        app.search = blocked()
        with patch.object(game.time, 'perf_counter', side_effect=Clock()), patch.object(game, 'has_legal_move', return_value=True), patch.object(game, 'target_layer_status', return_value=SimpleNamespace(buried=0)):
            app.tick()
        self.assertFalse(app.running)
        self.assertFalse(app.game.game_over)
        self.assertIn('no target-preserving placement', app.planner_blocked)
        app.place.assert_not_called()

    def test_closed_app_never_schedules_another_tick(self):
        app = stub()
        app.closed = True
        app.tick()
        app.root.after.assert_not_called()
        app.place.assert_not_called()

    def test_speed_selector_keeps_keyboard_events(self):
        app = stub()
        app.speed_box = object()
        app.piece_set_box, app.policy_box, app.target_box = object(), object(), object()
        app.reset = Mock()
        self.assertIsNone(app.key_down(SimpleNamespace(widget=app.speed_box, keysym='r')))
        app.reset.assert_not_called()

    def test_telemetry_represents_unlimited_without_nonfinite_numbers(self):
        app = stub()
        app.frame_pacer = game.FramePacer()
        app.well = SimpleNamespace(visible_w_sections=tuple(range(16)), views=[])
        report = app.render_metrics()
        self.assertEqual(report['playback_mode'], 'unlimited')
        self.assertIsNone(report['animation_rate_per_second'])
        self.assertEqual(report['visible_w_sections'], list(range(16)))
        json.dumps(report, allow_nan=False)
        app.speed.set(6)
        self.assertEqual(app.render_metrics()['animation_rate_per_second'], 6)


def trajectory(pieces):
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    app = game.TetrisApp.__new__(game.TetrisApp)
    app.closed = False
    app.running = True
    app.pending_step = app.goal_complete = app.stop_after_clear = False
    app.policy, app.target, app.human = Value('hyperlayer'), Value('1'), Value(False)
    app.speed, app.speed_choice = Value(0.0), Value('Unlimited')
    app.follow, app.w_slice = Value(True), Value(0)
    app.lookahead = Value(True)
    app.beam, app.candidates = 4, 2048
    app.previous = game.time.perf_counter()
    app._in_tick = app._force_draw = False
    app.start_button, app.lookahead_button, app.target_box, app.root = Mock(), Mock(), Mock(), Mock()
    app.draw = Mock()
    app.game = game.Game(7)
    app.refresh_plan()
    reference = game.Game(7)
    active_target = 1
    clears = []
    original_commit = game.Game.commit
    def commit(agent, move):
        nonlocal active_target
        expected = game.choose_hyperlayer_move(reference.board, reference.current, active_target, 2048, reference.pieces)
        assert expected is not None
        assert move.candidate == expected.move.candidate
        assert move.board == expected.move.board
        assert app.decision.evaluations == expected.evaluations
        original_commit(reference, expected.move)
        result = original_commit(agent, move)
        assert agent.board == reference.board
        assert (agent.current, agent.next_piece, agent.layers) == (reference.current, reference.next_piece, reference.layers)
        if move.layers:
            clears.append(agent.pieces)
            following = game.next_hyperlayer_target(reference.board, active_target)
            if following is not None:
                active_target = following
        if agent.pieces == pieces:
            app.running = False
        return result
    ticks = 0
    with patch.object(game.Game, 'commit', commit):
        while app.running and ticks < pieces * 100:
            app.tick()
            ticks += 1
    assert app.game.pieces == pieces, (app.game.pieces, app.planner_blocked)
    assert int(app.target.get()) == active_target
    if pieces >= 2600:
        assert clears[:2] == [1251, 2478], clears
    return {'identical_moves': pieces, 'clear_events': clears, 'active_target': active_target,
            'logic_ticks': ticks, 'backend': game.BACKEND, 'game_over': app.game.game_over,
            'scope': 'Actual GUI logic methods with fake Tk variables/root and rendering mocked; not a UI speed benchmark.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory-pieces', type=int, default=0)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(UnlimitedTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    report = {'passed': True, 'unit_tests': result.testsRun}
    if args.trajectory_pieces:
        report['trajectory'] = trajectory(args.trajectory_pieces)
    print(json.dumps(report, indent=2))
