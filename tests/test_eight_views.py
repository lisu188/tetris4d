from __future__ import annotations

import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tetris_benchmark_logging import BenchmarkLog


def run():
    checks = 0
    def check(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError(label)
        checks += 1
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    sizes = [(1560, 960), (1120, 760)]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'gui.jsonl'
        log = BenchmarkLog(game, path, {'mode': 'eight-view-gui-test'}).install()
        try:
            for width, height in sizes:
                root = tk.Tk()
                app = game.TetrisApp(root)
                root.geometry(f'{width}x{height}')
                def ready():
                    deadline = time.monotonic() + 20
                    while app.search is not None and time.monotonic() < deadline:
                        root.update()
                        time.sleep(.003)
                    root.update()
                    app.draw()
                    root.update()
                    check(app.search is None, 'Search failed to finish')
                    check(not app.errors, 'Callback error')
                try:
                    ready()
                    check(len(app.well.views) == 8, 'Expected eight 3D views')
                    covered = set()
                    for page in (0, 1):
                        app.select_slice(page * 8)
                        root.update()
                        app.draw()
                        log.snapshot('eight_views_test')
                        for i, view in enumerate(app.well.views):
                            canvas = view.canvas
                            w = page * 8 + i
                            covered.add(view.last_args[3])
                            check(view.last_args[3] == w, 'Incorrect W section')
                            check(bool(canvas.winfo_ismapped()), 'Canvas hidden')
                            check(canvas.winfo_rootx() >= root.winfo_rootx(), 'Canvas outside left edge')
                            check(canvas.winfo_rootx() + canvas.winfo_width() <= root.winfo_rootx() + root.winfo_width(), 'Canvas outside right edge')
                            check(canvas.winfo_rooty() + canvas.winfo_height() <= root.winfo_rooty() + root.winfo_height(), 'Canvas outside bottom edge')
                            for corner in itertools.product((0, 16), repeat=3):
                                x, y = view.project(*corner)
                                check(0 <= x < canvas.winfo_width() and 0 <= y < canvas.winfo_height(), 'Projected cube is clipped')
                            view._zoom(10)
                            check(view.zoom <= 1, 'Compact zoom must keep the entire well visible')
                        check(bool(app.network.positions), 'Activation graph missing')
                        for x, y in app.network.positions.values():
                            check(0 <= x < app.network.canvas.winfo_width() and 0 <= y < app.network.canvas.winfo_height(), 'Activation node clipped')
                    check(covered == set(range(16)), 'Not all W sections are accessible')
                    app.game.board = game.EMPTY_BOARD
                    app.game.locked_pieces = ()
                    app.game.current = 'RING'
                    move = game.make_move(game.EMPTY_BOARD, 'RING', 0, 0, 0, 7)
                    check(move is not None, 'Rigid ring fixture missing')
                    rendered = 0
                    for w in (0, 8):
                        app.well.draw(app.game, move, 1.0, w)
                        rendered += sum(sum(cube[-1] == -1 for cube in view.rendered_cubes) for view in app.well.views)
                    check(rendered == 8, 'All eight rigid RING cells must be present across pages')
                    app.game.board = (0, 10) + (2,) * (game.AREA - 2)
                    app.game.current = 'I'
                    app.target.set('1')
                    app.refresh_plan()
                    ready()
                    check(app.planned.layers == 1, 'No clear selected')
                    app.running = True
                    app.place()
                    check(app.running and not app.goal_complete, 'Clear stopped autoplay')
                    check(app.target.get() == '2', 'Target not updated')
                    app.running = False
                    ready()
                    app.human.set(True)
                    app.change_mode()
                    ready()
                    check(app.search is None, 'Manual mode started search')
                    app.well.reset_camera()
                    check(not app.errors, 'UI callback error after transitions')
                finally:
                    app.close()
        finally:
            log.close()
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        metrics = [row for row in rows if row['event'] == 'metrics']
        phases = metrics[-1]['phases']
        check(phases['gui_eight_views_submission']['count'] > 0, 'Eight-view timing absent')
        check(phases['gui_well_submission']['count'] > 0, 'Per-well timing absent')
        check(phases['gui_draw_submission']['count'] > 0, 'Draw timing absent')
        check(any(row['event'] == 'view_page' for row in rows), 'Page events absent')
        check(any(row['event'] == 'clear' for row in rows), 'Clear events absent')
        visible_pages = {tuple(row['gui']['visible_w_sections']) for row in metrics if row.get('gui')}
        check(tuple(range(8)) in visible_pages and tuple(range(8, 16)) in visible_pages, 'Visible W sections absent from snapshots')
        output = Path(temporary) / 'summary.json'
        subprocess.run([sys.executable, str(ROOT / 'tools/summarize_benchmark.py'), str(path), '--output', str(output)], check=True, stdout=subprocess.DEVNULL)
        check(isinstance(json.loads(output.read_text(encoding='utf-8')), dict), 'Summary wrapper failed')
    return {'passed': True, 'assertions': checks, 'sizes': sizes, 'views_per_page': 8,
            'accessible_w_sections': 16, 'backend': game.BACKEND,
            'environment': 'Linux Xvfb; not native WSLg'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
