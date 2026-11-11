from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tetris_benchmark_logging import BenchmarkLog, Timing, summarize


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.log = None
        self.parameters = game.network_parameters()
        self.training = game.MODEL_TRAINING
        self.backend = game.BACKEND
        game.configure_backend('auto')
        game.load_checkpoint(ROOT / 'models' / 'trained-sigmoid.json')

    def tearDown(self):
        if self.log is not None:
            self.log.close()
        game.configure_network(self.parameters, self.training)
        game.configure_backend(self.backend)
        self.temporary.cleanup()

    def start(self, **kwargs):
        self.log = BenchmarkLog(game, self.root / 'run.jsonl', **kwargs).install()
        return self.log

    def rows(self):
        self.log.flush()
        return [json.loads(line) for line in self.log.path.read_text(encoding='utf-8').splitlines()]

    def test_metadata_is_technical_and_excludes_library_path(self):
        self.start()
        start = self.rows()[0]
        self.assertEqual(start['event'], 'session_start')
        self.assertIn('source_sha256', start['metadata'])
        self.assertIn('clock', start['metadata'])
        self.assertNotIn('library', start['metadata']['native'])
        self.assertNotIn('environment', start['metadata'])
        self.assertEqual(len(start['metadata']['parameters_sha256']), 64)

    def test_logged_moves_equal_reference_and_wrappers_are_restored(self):
        reference = []
        player = game.Game(7)
        for _ in range(40):
            decision = game.choose_hyperlayer_move(player.board, player.current, 1, 128, player.pieces)
            reference.append(decision.move)
            player.commit(decision.move)
        original = game.Game.commit
        self.start()
        player = game.Game(7)
        for expected in reference:
            decision = game.choose_hyperlayer_move(player.board, player.current, 1, 128, player.pieces)
            self.assertEqual(decision.move, expected)
            player.commit(decision.move)
        self.assertEqual(self.log.decisions, 40)
        self.assertEqual(self.log.moves, 40)
        self.assertGreater(self.log.metrics['commit_and_validation'].count, 0)
        self.log.close()
        self.assertIs(game.Game.commit, original)
        decisions = [row for row in self.rows() if row['event'] == 'decision']
        self.assertEqual(len(decisions), 40)
        self.assertTrue(all(row['work_wall_ns'] <= row['elapsed_latency_ns'] for row in decisions))

    def test_generator_suspension_is_not_search_work(self):
        self.start()
        player = game.Game(7)
        search = game.hyperlayer_decision_steps(player.board, player.current, 1, 1024, 0)
        slices = 0
        while True:
            try:
                next(search)
                slices += 1
                time.sleep(.015)
            except StopIteration:
                break
        row = [row for row in self.rows() if row['event'] == 'decision'][-1]
        self.assertGreaterEqual(slices, 2)
        self.assertGreater(row['elapsed_latency_ns'] - row['work_wall_ns'], slices * 10_000_000)

    def test_clear_event_is_flushed_and_does_not_change_game(self):
        self.start()
        player = game.Game(7)
        player.board = (0,) + (2,) * (game.AREA - 1)
        player.current = 'I'
        move = game.choose_hyperlayer_move(player.board, 'I', 1, 128, 0).move
        self.assertEqual(move.layers, 1)
        player.commit(move)
        events = [json.loads(line) for line in self.log.path.read_text(encoding='utf-8').splitlines()]
        clear = next(row for row in events if row['event'] == 'clear')
        self.assertEqual(clear['hyperlayers'], 1)
        self.assertEqual(player.layers, 1)
        self.assertEqual(len(clear['move_digest']), 64)

    def test_cache_api_and_model_reload_still_work(self):
        self.start()
        game.evaluate_board(game.EMPTY_BOARD)
        game.evaluate_board(game.EMPTY_BOARD)
        self.assertGreater(game.evaluate_board.cache_info().hits, 0)
        game.configure_network(game.BASELINE_PARAMETERS)
        self.assertEqual(game.evaluate_board.cache_info().currsize, 0)
        game.configure_backend('python')
        self.log.snapshot()

    def test_rotation_is_bounded_and_segments_remain_readable(self):
        self.start(max_bytes=9000, backups=2)
        for index in range(100):
            self.log.emit('test', index=index, data='a' * 1200)
        self.log.close()
        paths = sorted(self.root.glob('run.jsonl*'))
        self.assertEqual(len(paths), 3)
        for path in paths:
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            self.assertIn(rows[0]['event'], ('session_start', 'segment_start'))
            self.assertTrue(all(a['seq'] < b['seq'] for a, b in zip(rows, rows[1:])))
        report = summarize(paths)
        self.assertEqual(report['malformed_lines'], 0)
        self.assertTrue(report['sessions'][self.log.session]['completed'])

    def test_existing_file_is_not_overwritten(self):
        path = self.root / 'run.jsonl'
        path.write_text('retain', encoding='utf-8')
        with contextlib.redirect_stderr(io.StringIO()):
            self.start()
        self.assertFalse(self.log.enabled)
        self.assertEqual(path.read_text(encoding='utf-8'), 'retain')

    def test_permission_or_write_failure_does_not_stop_play(self):
        with patch.object(Path, 'open', side_effect=PermissionError('test')), contextlib.redirect_stderr(io.StringIO()):
            self.start()
        self.assertFalse(self.log.enabled)
        player = game.Game(7)
        self.assertIsNotNone(game.choose_hyperlayer_move(player.board, player.current, 1, 32, 0))

    def test_nonfinite_values_produce_valid_json(self):
        self.start()
        self.log.emit('test', value=float('inf'), values=[float('nan'), 1.0])
        row = self.rows()[-1]
        self.assertIsNone(row['value'])
        self.assertEqual(row['values'], [None, 1.0])

    def test_summary_handles_multiple_sessions_and_truncated_tail(self):
        self.start()
        self.log.snapshot()
        self.log.close()
        second = BenchmarkLog(game, self.root / 'second.jsonl')
        second.close()
        with second.path.open('a', encoding='utf-8') as output:
            output.write('{truncated')
        report = summarize([self.log.path, second.path])
        self.assertEqual(len(report['sessions']), 2)
        self.assertEqual(report['malformed_lines'], 1)

    def test_percentiles_use_bounded_recent_window(self):
        metric = Timing()
        for value in range(3000):
            metric.add(value * 1000000, value)
        report = metric.report()
        self.assertEqual(report['count'], 3000)
        self.assertEqual(report['recent_samples'], 2048)
        self.assertEqual(report['max_wall_ms'], 2999)
        self.assertEqual(report['recent_p50_wall_ms'], 1975)


class LaunchLoggingTests(LoggingTests):
    def test_cli_defaults_to_logging_and_can_disable_it(self):
        with patch.object(game, 'launch_demo', return_value=0), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(game.main(['--benchmark-log', str(self.root / 'cli.jsonl')]), 0)
        self.assertTrue((self.root / 'cli.jsonl').is_file())
        original = game.Game.commit
        with patch.object(game, 'launch_demo', return_value=0), patch.object(BenchmarkLog, '__init__', side_effect=AssertionError('Should not log')):
            self.assertEqual(game.main(['--no-benchmark-log']), 0)
        self.assertIs(game.Game.commit, original)

    def test_exports_do_not_create_telemetry(self):
        path = self.root / 'export.jsonl'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(game.main(['--network', '--benchmark-log', str(path)]), 0)
        self.assertFalse(path.exists())

    def test_options_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            game.main(['--no-benchmark-log', '--benchmark-log', str(self.root / 'x.jsonl')])
        self.assertEqual(caught.exception.code, 2)


def compare_trajectory(pieces):
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models' / 'trained-sigmoid.json')
    reference = []
    player, target = game.Game(7), 1
    for _ in range(pieces):
        decision = game.choose_hyperlayer_move(player.board, player.current, target, 2048, player.pieces)
        if decision is None:
            raise AssertionError('Reference planner stalled.')
        player.commit(decision.move)
        reference.append((decision.move, decision.evaluations, player.current, player.next_piece))
        if decision.move.layers:
            target = game.next_hyperlayer_target(player.board, target)
    with tempfile.TemporaryDirectory() as temporary:
        log = BenchmarkLog(game, Path(temporary) / 'trajectory.jsonl').install()
        try:
            player, target = game.Game(7), 1
            for move, evaluations, current, following in reference:
                decision = game.choose_hyperlayer_move(player.board, player.current, target, 2048, player.pieces)
                if decision is None or decision.move != move or decision.evaluations != evaluations:
                    raise AssertionError(f'Logged trajectory diverged at piece {player.pieces}.')
                player.commit(decision.move)
                if (player.current, player.next_piece) != (current, following):
                    raise AssertionError('Piece stream changed.')
                if decision.move.layers:
                    target = game.next_hyperlayer_target(player.board, target)
        finally:
            log.close()
        records = [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]
        clear_events = [row['pieces'] for row in records if row['event'] == 'clear']
        if pieces == 2600:
            assert clear_events == [1251, 2478], clear_events
        print(json.dumps({'identical_logged_moves': pieces, 'clear_events': clear_events,
                          'backend': game.BACKEND, 'summary': player.summary()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-launch', action='store_true')
    parser.add_argument('--trajectory-pieces', type=int, default=0)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LoggingTests)
    if not args.skip_launch:
        for name in ('test_cli_defaults_to_logging_and_can_disable_it', 'test_exports_do_not_create_telemetry', 'test_options_are_mutually_exclusive'):
            suite.addTest(LaunchLoggingTests(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful() and args.trajectory_pieces:
        compare_trajectory(args.trajectory_pieces)
    raise SystemExit(not result.wasSuccessful())
