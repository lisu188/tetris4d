from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tetris_benchmark_logging import BenchmarkLog


def main():
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models' / 'trained-sigmoid.json')
    with tempfile.TemporaryDirectory() as temporary:
        log = BenchmarkLog(game, Path(temporary) / 'gui.jsonl').install()
        root = tk.Tk()
        app = game.TetrisApp(root, candidates=128, policy='hyperlayer')
        root.geometry('1120x760')
        root.after(250, app.toggle)
        root.after(2500, app.close)
        try:
            root.mainloop()
        finally:
            if not app.closed:
                app.close()
            log.close()
        assert not app.errors, app.errors
        assert app.game.pieces > 0, app.game.summary()
        rows = [json.loads(line) for line in log.path.read_text(encoding='utf-8').splitlines()]
        metrics = [row for row in rows if row['event'] == 'metrics'][-1]['phases']
        for key in ('gui_tick_interval', 'gui_tick_work', 'gui_draw_submission',
                    'gui_well_submission', 'gui_atlas_submission', 'gui_network_update', 'search_active_work'):
            assert metrics[key]['count'] > 0, (key, metrics)
        assert rows[-1]['event'] == 'session_end'
        assert any(row['event'] == 'game_end' for row in rows)
        print(json.dumps({'passed': True, 'pieces': app.game.pieces, 'phases': sorted(metrics),
                          'tick_samples': metrics['gui_tick_interval']['count']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
