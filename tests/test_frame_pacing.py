from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


class FramePacingTests(unittest.TestCase):
    def test_default_allows_first_frame(self):
        pacer = game.FramePacer()
        self.assertTrue(pacer.enabled)
        self.assertEqual(pacer.fps, 30)
        self.assertTrue(pacer.allow(10))

    def test_slow_frames_back_off_with_bounded_interval(self):
        pacer = game.FramePacer()
        pacer.complete(10, 10.08)
        self.assertAlmostEqual(pacer.interval, .16)
        self.assertFalse(pacer.allow(10.1))
        self.assertTrue(pacer.allow(10.17))
        pacer.complete(11, 11.4)
        self.assertLessEqual(pacer.interval, .2)
        self.assertGreater(pacer.next_at, 11.4)

    def test_fast_frames_recover_without_exceeding_requested_rate(self):
        pacer = game.FramePacer(30)
        pacer.complete(10, 10.09)
        for i in range(60):
            pacer.complete(20 + i, 20 + i + .001)
        self.assertAlmostEqual(pacer.interval, 1 / 30)

    def test_explicit_force_and_disable_allow_redraw(self):
        for enabled in (False, True):
            pacer = game.FramePacer(enabled=enabled)
            pacer.complete(100, 100.02)
            self.assertTrue(pacer.allow(100.021, force=True))
            self.assertEqual(pacer.allow(100.022), not enabled)

    def test_invalid_fps_is_rejected(self):
        for fps in (0, -1, 121, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                game.FramePacer(fps)

    def test_deadline_has_no_catchup_queue(self):
        pacer = game.FramePacer()
        pacer.complete(0, .001)
        self.assertTrue(pacer.allow(500))
        pacer.complete(500, 500.01)
        self.assertFalse(pacer.allow(500.011))

    def stub_app(self):
        app = game.TetrisApp.__new__(game.TetrisApp)
        app.closed = False
        app.well = object()
        app.frame_pacer = game.FramePacer()
        app.frame_pacer.next_at = 1000
        app._in_tick = False
        app._draw_id = None
        app.root = Mock()
        app.root.after.return_value = 'pending-frame'
        app._draw_now = Mock()
        return app

    def test_repeated_draws_share_one_pending_callback(self):
        app = self.stub_app()
        with patch.object(game.time, 'perf_counter', return_value=999.95):
            for _ in range(100):
                app.draw()
        app.root.after.assert_called_once()
        app._draw_now.assert_not_called()
        self.assertEqual(app.frame_pacer.skipped, 100)
        self.assertEqual(app._draw_id, 'pending-frame')

    def test_forced_redraw_cancels_pending_callback(self):
        app = self.stub_app()
        with patch.object(game.time, 'perf_counter', return_value=999.95):
            app.draw()
            app.draw(force=True)
        app.root.after_cancel.assert_called_once_with('pending-frame')
        app._draw_now.assert_called_once()
        self.assertIsNone(app._draw_id)
        self.assertEqual(app.frame_pacer.rendered, 1)

    def test_logic_tick_coalesces_requests(self):
        app = self.stub_app()
        app._in_tick = True
        for _ in range(10):
            app.draw()
        app._draw_now.assert_not_called()
        app.root.after.assert_not_called()
        self.assertEqual(app.frame_pacer.coalesced, 10)

    def test_closed_app_cannot_render_or_reschedule(self):
        app = self.stub_app()
        app.closed = True
        app._flush_draw()
        app._draw_now.assert_not_called()
        app.root.after.assert_not_called()

    def test_cli_defaults_and_override(self):
        for argv, fps, disabled in (([], 30, False), (['--ui-fps', '60', '--no-frameskip'], 60, True)):
            with patch.object(game, 'launch_demo', return_value=0) as launch, patch.object(game, 'start_benchmark_logging', return_value=None):
                self.assertEqual(game.main(argv), 0)
            args = launch.call_args.args[0]
            self.assertEqual((args.ui_fps, args.no_frameskip), (fps, disabled))
            self.assertEqual((args.policy, args.target, args.stop_after_clear), ('hyperlayer', 1, False))

    def test_cli_invalid_fps(self):
        for fps in ('0', '121'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                game.main(['--ui-fps', fps])
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
