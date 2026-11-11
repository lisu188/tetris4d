from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tkinter as tk
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game

BASELINE = '2266a4d5e01ff1df5406115c6f47970141deec44'


def load_baseline(path: Path | None = None):
    if path is None:
        source = subprocess.check_output(['git', 'show', f'{BASELINE}:sparse_tetris_4d_optimized.py'], cwd=ROOT).decode('utf-8')
    else:
        source = path.read_text(encoding='utf-8')
    module = ModuleType('tetris_render_baseline')
    module.__file__ = str(ROOT / '_render_baseline.py')
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module, source


def replay_frames(start: int, frames: int, seed: int):
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models' / 'trained-sigmoid.json')
    agent, target, snapshots = game.Game(seed), 1, []
    for step in range(start + frames + 1):
        decision = game.choose_hyperlayer_move(agent.board, agent.current, target, game.DEFAULT_CANDIDATES, step)
        if decision is None:
            raise RuntimeError(f'Planner stopped at piece {agent.pieces}, before the requested rendering fixture.')
        agent.commit(decision.move)
        if decision.move.layers:
            following = game.next_hyperlayer_target(agent.board, target)
            if following is not None:
                target = following
        if step >= start:
            snapshots.append(SimpleNamespace(**vars(agent)))
    return snapshots


def counters(panel, atlas):
    return {'scene_rebuilds': sum(v.scene_rebuilds for v in panel.views),
            'atlas_rebuilds': getattr(atlas, 'scene_rebuilds', None),
            'atlas_cell_updates': getattr(atlas, 'cell_updates', None),
            'atlas_created_cells': getattr(atlas, 'cells_created', None),
            'ownership_full_scans': getattr(panel, 'ownership_full_scans', None),
            'ownership_appends': getattr(panel, 'ownership_appends', None),
            'static_mesh_batches': sum(getattr(v, 'mesh_batches', 0) for v in panel.views)}


def measure(module, snapshots):
    root = tk.Tk()
    root.title('Tetris renderer benchmark')
    root.geometry('1100x720')
    host = tk.Frame(root)
    host.place(x=0, y=0, width=780, height=710)
    host.rowconfigure(0, weight=1)
    host.columnconfigure(0, weight=1)
    panel = module.AllWellView(host, lambda w: None)
    canvas = tk.Canvas(root, highlightthickness=0)
    canvas.place(x=790, y=0, width=300, height=710)
    atlas = module.SliceAtlas(canvas, lambda w: None)
    try:
        root.update()
        panel.draw(snapshots[0], None)
        atlas.draw(snapshots[0], None, 0)
        root.update_idletasks()
        before = counters(panel, atlas)
        rows = []
        for frame in snapshots[1:]:
            started = time.perf_counter()
            panel.draw(frame, None)
            mesh_done = time.perf_counter()
            atlas.draw(frame, None, 0)
            submitted = time.perf_counter()
            root.update_idletasks()
            painted = time.perf_counter()
            rows.append({'well_ms': (mesh_done - started) * 1000,
                         'atlas_ms': (submitted - mesh_done) * 1000,
                         'submission_ms': (submitted - started) * 1000,
                         'tk_idle_ms': (painted - submitted) * 1000,
                         'total_ms': (painted - started) * 1000})
        after = counters(panel, atlas)
        if not all(v.canvas.winfo_ismapped() for v in panel.views):
            raise AssertionError('A W section became hidden during the benchmark.')
        return {'frames': rows,
                'totals_ms': {key: sum(row[key] for row in rows) for key in rows[0]},
                'counters_delta': {k: None if v is None else v - before[k] for k, v in after.items()},
                'visible_sections': list(panel.visible_w_sections),
                'canvas_items': sum(len(v.canvas.find_all()) for v in panel.views) + len(canvas.find_all())}
    finally:
        root.destroy()


def benchmark(args):
    baseline, _ = load_baseline(args.baseline)
    frames = replay_frames(args.start, args.frames, args.seed)
    runs = {'baseline': [], 'optimized': []}
    modules = {'baseline': baseline, 'optimized': game}
    for repetition in range(args.repetitions):
        order = ('baseline', 'optimized') if repetition % 2 == 0 else ('optimized', 'baseline')
        for name in order:
            run = measure(modules[name], frames)
            runs[name].append(run)
            print(json.dumps({'repetition': repetition, 'backend': name, 'totals_ms': run['totals_ms']}), flush=True)
    medians = {name: {key: statistics.median(run['totals_ms'][key] for run in values)
                      for key in values[0]['totals_ms']} for name, values in runs.items()}
    result = {'baseline_commit': BASELINE, 'seed': args.seed, 'first_fixture_piece': frames[0].pieces,
              'last_fixture_piece': frames[-1].pieces, 'frames_per_repetition': args.frames,
              'repetitions': args.repetitions, 'visible_sections': 16, 'skipped_frames': 0,
              'platform': platform.platform(), 'python': sys.version, 'tk': tk.TkVersion,
              'fixture_backend': game.BACKEND, 'medians_ms': medians,
              'speedup': {key: medians['baseline'][key] / medians['optimized'][key] for key in medians['baseline']},
              'runs': runs,
              'scope': 'Equal 16-view plus atlas workload, fixed cameras, no active animation, actual Tk draw submission and idle painting. Excludes search, network panel, application scheduler, compositor and WSLg display latency. No frame-skipping speedup is included.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.output), 'medians_ms': medians, 'speedup': result['speedup']}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare identical all-16-view redraw workloads with the pinned renderer baseline.')
    parser.add_argument('--start', type=int, default=1200)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, default=Path('render-benchmark.json'))
    args = parser.parse_args()
    if args.start < 0 or min(args.frames, args.repetitions) < 1:
        parser.error('start must be nonnegative; frames and repetitions must be positive.')
    benchmark(args)
