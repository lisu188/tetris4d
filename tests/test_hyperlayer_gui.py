from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def run() -> dict:
    checks = 0
    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError(message)
        checks += 1

    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    sizes = ((1560, 960), (1120, 760))
    for width, height in sizes:
        root = tk.Tk()
        app = game.TetrisApp(root)
        root.geometry(f'{width}x{height}')
        def ready():
            deadline = time.perf_counter() + 10
            while app.search is not None and time.perf_counter() < deadline:
                root.update()
                time.sleep(.005)
            root.update()
            check(app.search is None, 'Planner did not finish')
            check(not app.errors, 'Tk callback errors')
        try:
            ready()
            check(app.planned is not None, 'No initial target drop')
            check(app.decision.policy == 'hyperlayer', 'Wrong planner')
            check(app.lookahead_button.instate(['disabled']), 'Lookahead must be disabled for target search')
            check('4,096' in app.target_progress.get(), 'Target progress absent')
            check(math.isclose(app.network.values[-1], app.planned.score, abs_tol=1e-8), 'UI must show actual sigmoid score')
            for widget in (app.policy_box, app.target_box, app.start_button):
                check(widget.winfo_ismapped(), 'Planner control is not visible')
                check(widget.winfo_rooty() + widget.winfo_height() <= root.winfo_rooty() + root.winfo_height(), 'Planner control is below window')
            for w in (0, 7, 15):
                app.select_slice(w)
                check(app.w_slice.get() == w, 'W section control changed')
                check(bool(app.network.positions), 'Network heatmap missing')
            app.pending_step = True
            app.policy.set('network')
            app.change_policy()
            check(not app.pending_step and not app.running, 'Policy change must cancel pending action')
            ready()
            check(app.decision.policy == 'network', 'Neural-only mode unavailable')
            check(not app.lookahead_button.instate(['disabled']), 'Neural lookahead not restored')
            app.game.board = (4,) + game.EMPTY_BOARD[1:]
            app.game.current = 'O'
            app.policy.set('hyperlayer')
            app.change_policy()
            ready()
            check(not app.game.game_over, 'Unsafe target incorrectly reported as top-out')
            check('buried' in app.planner_blocked, 'Unsafe target explanation missing')
            check(not app.running, 'Unsafe planner must pause')
            app.toggle()
            check(not app.running, 'Blocked planner must stay paused')
            app.target.set('2')
            app.change_policy()
            ready()
            check(app.planned is not None, 'Selecting accessible target did not recover')
            app.reset()
            app.target.set('1')
            app.game.board = (0, 10) + (2,) * (game.AREA - 2)
            app.game.current = 'I'
            app.refresh_plan()
            ready()
            check(app.planned.layers == 1, 'Final target gap was not selected')
            app.running = True
            app.place()
            root.update()
            check(app.game.layers == 1, 'GUI did not commit the clear')
            check(not app.goal_complete and app.running, 'GUI paused at a clear without an explicit request')
            check(app.search is not None or app.planned is not None, 'GUI did not plan after clearing')
            check(int(app.target.get()) == 2, 'Compacted board was not retargeted')
            app.running = False
            ready()
            previous = app.game.pieces
            app.place()
            ready()
            check(app.game.pieces == previous + 1, 'Post-clear placement was not committed')
            check('continuous play' in app.target_progress.get(), 'Continuous progress status absent')
            app.reset()
            app.target.set('1')
            app.stop_after_clear = True
            app.game.board = (0,) + (2,) * (game.AREA - 1)
            app.game.current = 'I'
            app.refresh_plan()
            ready()
            app.running = True
            app.place()
            root.update()
            check(app.goal_complete and not app.running and app.search is None, 'Explicit pause on clear ignored')
            app.toggle()
            check(not app.running, 'Explicit first-clear pause was not preserved')
            app.policy.set('network')
            app.change_policy()
            ready()
            check(app.planned is not None and not app.goal_complete, 'Neural play cannot continue after clear')
            app.human.set(True)
            app.change_mode()
            root.update()
            check(app.search is None, 'Human mode must not run the planner')
            check(not app.errors, 'UI errors after state transitions')
        finally:
            app.close()
    return {'passed': True, 'assertions': checks, 'sizes': sizes, 'backend': game.BACKEND,
            'display': 'Linux Xvfb; not native Windows/macOS'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
