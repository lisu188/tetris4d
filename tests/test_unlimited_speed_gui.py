from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from tkinter import ttk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tetris_benchmark_logging import BenchmarkLog


def expected_moves(count):
    state, target, results = game.Game(7), 1, []
    for _ in range(count):
        decision = game.choose_hyperlayer_move(state.board, state.current, target, 2048, state.pieces)
        state.commit(decision.move)
        results.append((decision.move.candidate, state.board, state.current, state.next_piece))
        if decision.move.layers:
            target = game.next_hyperlayer_target(state.board, target)
    return results


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def run():
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    expected = expected_moves(64)
    checks, reports = 0, []
    def check(value, message):
        nonlocal checks
        if not value:
            raise AssertionError(message)
        checks += 1
    with tempfile.TemporaryDirectory() as temporary:
        logpath = Path(temporary) / 'unlimited.jsonl'
        logger = BenchmarkLog(game, logpath, {'mode': 'unlimited-gui-regression'}).install()
        try:
            for width, height in ((1560, 960), (1120, 760)):
                root = tk.Tk()
                app = game.TetrisApp(root)
                root.geometry(f'{width}x{height}')
                root.after_cancel(app.tick_id)
                app.tick_id = None
                def ready():
                    end = time.monotonic() + 20
                    while app.search is not None and time.monotonic() < end:
                        app._advance_search(time.perf_counter() + .008)
                        root.update()
                    app.draw(force=True)
                    root.update()
                    check(app.search is None, 'Search did not finish')
                    check(not app.errors, 'Callback error')
                try:
                    ready()
                    check(isinstance(app.speed_box, ttk.Combobox), 'Speed must be a selector')
                    check(not any(isinstance(w, ttk.Scale) for w in descendants(root)), 'Speed slider remains')
                    check('Unlimited' in tuple(app.speed_box['values']), 'Unlimited option missing')
                    check(app.speed.get() == 3, 'Animated default changed unexpectedly')
                    app.speed_choice.set('Unlimited')
                    app.speed_box.event_generate('<<ComboboxSelected>>')
                    root.update()
                    check(app.speed.get() == 0 and not app.running, 'Choosing Unlimited must not start paused play')
                    before = app.game.pieces
                    app.step()
                    ready()
                    check(app.game.pieces == before + 1 and not app.running, 'Step must commit exactly one move')
                    app.human.set(True)
                    app.change_mode()
                    before = app.game.pieces
                    app.tick()
                    root.after_cancel(app.tick_id)
                    app.tick_id = None
                    check(app.game.pieces == before and not app.running, 'Unlimited auto-dropped in Human mode')
                    app.human.set(False)
                    app.change_mode()
                    for stop in (False, True):
                        app.reset()
                        app.stop_after_clear = stop
                        app.game.board = (0, 10) + (2,) * (game.AREA - 2)
                        app.game.current = 'I'
                        app.target.set('1')
                        app.refresh_plan()
                        ready()
                        check(app.planned.layers == 1, 'Clear fixture missing')
                        app.running = True
                        app.place()
                        check(app.game.layers == 1, 'Layer did not clear')
                        check(app.running == (not stop) and app.goal_complete == stop, 'Post-clear running state incorrect')
                        app.running = False
                    app.stop_after_clear = False
                    app.target.set('1')
                    app.reset()
                    ready()
                    app.speed_choice.set('Unlimited')
                    app.change_speed()
                    check(len(app.well.views) == 16, 'All 16 cubes must remain visible')
                    for view in app.well.views:
                        check(bool(view.canvas.winfo_ismapped()), 'Section cube hidden')
                    states, heartbeats = [], []
                    original = game.Game.commit
                    def commit(agent, move):
                        result = original(agent, move)
                        states.append((move.candidate, agent.board, agent.current, agent.next_piece))
                        if len(states) == len(expected):
                            app.running = False
                            root.after(0, root.quit)
                        return result
                    def heartbeat():
                        heartbeats.append(time.perf_counter())
                        if len(states) < len(expected):
                            root.after(5, heartbeat)
                    root.after(5, heartbeat)
                    timeout = root.after(60000, root.quit)
                    app.toggle()
                    app.tick_id = root.after(1, app.tick)
                    with patch.object(game.Game, 'commit', commit):
                        root.mainloop()
                    root.after_cancel(timeout)
                    check(states == expected, 'Unlimited GUI trajectory differs from headless reference')
                    check(bool(heartbeats), 'Unlimited starved Tk event callbacks')
                    check(not app.errors, 'GUI callback exception')
                    metrics = app.render_metrics()
                    check(metrics['playback_mode'] == 'unlimited', 'Unlimited telemetry missing')
                    check(metrics['animation_rate_per_second'] is None, 'Unlimited rate must be null')
                    check(metrics['visible_w_sections'] == list(range(16)), 'Cube visibility changed during play')
                    check(metrics['coalesced_requests'] > 0, 'Draw requests were not coalesced')
                    logger.snapshot('unlimited_gui_done')
                    reports.append({'size': [width, height], 'identical_moves': len(states),
                                    'heartbeat_callbacks': len(heartbeats), 'rendering': metrics})
                finally:
                    app.close()
        finally:
            logger.close()
        records = [json.loads(line) for line in logpath.read_text(encoding='utf-8').splitlines()]
        check(any(row.get('rendering', {}).get('playback_mode') == 'unlimited' for row in records), 'Log omitted playback mode')
    return {'passed': True, 'assertions': checks, 'backend': game.BACKEND,
            'runs': reports, 'environment': 'Linux virtual display; not native WSLg throughput measurements'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
