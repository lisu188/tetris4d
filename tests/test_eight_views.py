from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def run(with_logging=True):
    checks = 0
    def check(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    sizes = [(1560, 960), (1120, 760)]
    render_results = []
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'gui.jsonl'
        log = None
        if with_logging:
            from tetris_benchmark_logging import BenchmarkLog
            log = BenchmarkLog(game, path, {'mode': 'all-views-pacing-test'}).install()
        try:
            for width, height in sizes:
                root = tk.Tk()
                app = game.TetrisApp(root)
                root.geometry(f'{width}x{height}')
                def ready():
                    deadline = time.monotonic() + 20
                    while app.search is not None and time.monotonic() < deadline:
                        root.update()
                        time.sleep(.001)
                    app.draw(force=True)
                    root.update()
                    check(app.search is None, 'Search did not finish')
                    check(not app.errors, 'Callback error')
                try:
                    ready()
                    check(len(app.well.views) == 16, 'All 16 section cubes must exist')
                    check(not hasattr(app.well, 'page'), 'Paging must be removed')
                    for selected in (0, 7, 8, 15):
                        app.select_slice(selected)
                        root.update()
                        app.draw(force=True)
                        check([view.last_args[3] for view in app.well.views] == list(range(16)), 'Selection hid a section')
                        for w, view in enumerate(app.well.views):
                            canvas = view.canvas
                            check(bool(canvas.winfo_ismapped()), 'Canvas hidden')
                            check(canvas.winfo_rootx() >= root.winfo_rootx(), 'Canvas outside left edge')
                            check(canvas.winfo_rootx() + canvas.winfo_width() <= root.winfo_rootx() + root.winfo_width(), 'Canvas outside right edge')
                            check(canvas.winfo_rooty() + canvas.winfo_height() <= root.winfo_rooty() + root.winfo_height(), 'Canvas outside bottom edge')
                            for corner in itertools.product((0, 16), repeat=3):
                                x, y = view.project(*corner)
                                check(0 <= x < canvas.winfo_width() and 0 <= y < canvas.winfo_height(), 'Projected well clipped')
                            check(view._selected == (w == selected), 'Wrong selected section')
                        check(bool(app.network.positions), 'Activation graph missing')
                        for x, y in app.network.positions.values():
                            check(0 <= x < app.network.canvas.winfo_width() and 0 <= y < app.network.canvas.winfo_height(), 'Activation node clipped')
                    for view in app.well.views:
                        for yaw, elevation in ((0, .18), (2.5, 1.4), (5, .5)):
                            view.yaw, view.elevation = yaw, elevation
                            view._zoom(100)
                            check(view.zoom <= 1, 'Compact zoom cropped the well')
                            for corner in itertools.product((0, 16), repeat=3):
                                x, y = view.project(*corner)
                                check(0 <= x < view.canvas.winfo_width() and 0 <= y < view.canvas.winfo_height(), 'Orbit cropped a corner')
                    app.well.reset_camera()
                    fixture = game.Game(7)
                    for name in ('RING', 'HYPER'):
                        fixture.current = name
                        move = game.make_move(game.EMPTY_BOARD, name, 0, 0, 0, 7)
                        check(move is not None, 'Rigid piece fixture missing')
                        app.well.draw(fixture, move, 1.0, 0)
                        active = sum(sum(cube[-1] == -1 for cube in view.rendered_cubes) for view in app.well.views)
                        check(active == game.PIECE_SIZES[name], 'Active cells across W=7/8 not simultaneously represented')
                    fixture.current = 'I'
                    rotation = next(i for i, shape in enumerate(game.SHAPES['I']) if len(shape.columns) == 1)
                    move = game.make_move(fixture.board, 'I', rotation, 0, 0, 5)
                    app.well.draw(fixture, None)
                    counts = [view.scene_rebuilds for view in app.well.views]
                    fixture.commit(move)
                    app.well.draw(fixture, None)
                    changed = [i for i, view in enumerate(app.well.views) if view.scene_rebuilds != counts[i]]
                    check(changed == [5], 'Unchanged W sections rebuilt')
                    counts = [view.scene_rebuilds for view in app.well.views]
                    app.well.draw(fixture, None)
                    check(counts == [view.scene_rebuilds for view in app.well.views], 'Unchanged redraw rebuilt geometry')
                    app.game.board = (0, 10) + (2,) * (game.AREA - 2)
                    app.game.current = 'I'
                    app.target.set('1')
                    app.refresh_plan()
                    ready()
                    check(app.planned.layers == 1, 'Clear fixture not selected')
                    app.running = True
                    app.place()
                    check(app.running and not app.goal_complete, 'Clear stopped autoplay')
                    check(app.target.get() == '2', 'Target did not change')
                    app.running = False
                    ready()
                    app.frame_pacer.next_at = time.perf_counter() + .1
                    frames = app.frame_pacer.rendered
                    for _ in range(25):
                        app.draw()
                    check(app.frame_pacer.rendered == frames, 'Frame pacing did not defer duplicate draws')
                    check(app._draw_id is not None, 'Deferred redraw has no delivery callback')
                    app.draw(force=True)
                    check(app.frame_pacer.rendered == frames + 1, 'Forced refresh not rendered')
                    check(app._draw_id is None, 'Forced refresh left a redundant callback')
                    app.human.set(True)
                    app.change_mode()
                    ready()
                    check(not app.running and not app.pending_step, 'Manual mode restarted autoplay')
                    if log is not None:
                        log.app = app
                        log.snapshot('all_sections_test')
                    render_results.append(app.render_metrics())
                    app.frame_pacer.next_at = time.perf_counter() + .1
                    app.draw()
                    check(app._draw_id is not None, 'Close cancellation fixture absent')
                finally:
                    app.close()
                    check(app._draw_id is None, 'Closed app retained a redraw callback')
                    check(not app.errors, 'UI error on shutdown')
        finally:
            if log is not None:
                log.close()
        if log is not None:
            rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            metrics = [row for row in rows if row['event'] == 'metrics']
            last = metrics[-1]
            for phase in ('gui_all_views_submission', 'gui_well_submission', 'gui_draw_submission'):
                check(last['phases'][phase]['count'] > 0, 'Rendering timing absent')
            check(last['rendering']['visible_w_sections'] == list(range(16)), 'Final summary lost visibility metadata')
            check(last['rendering']['deferred_requests'] > 0, 'Frameskip counters absent')
            check(any(row['event'] == 'clear' for row in rows), 'Clear event absent')
            check(any(row.get('gui', {}).get('visible_w_sections') == list(range(16)) for row in metrics if row.get('gui')), 'GUI snapshot lost sections')
    return {'passed': True, 'assertions': checks, 'sizes': sizes, 'simultaneous_sections': 16,
            'paging': False, 'rendering': render_results, 'backend': game.BACKEND,
            'logging_tested': with_logging, 'environment': 'Linux Xvfb; not WSLg'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-logging', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(not args.no_logging), indent=2))
