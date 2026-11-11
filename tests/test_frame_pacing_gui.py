from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def run(frameskip):
    root = tk.Tk()
    app = game.TetrisApp(root, candidates=128, frameskip=frameskip)
    root.geometry('1120x760')
    app.speed.set(12)
    states = []
    original = game.Game.commit
    def commit(agent, move):
        result = original(agent, move)
        states.append((move.candidate, agent.board, agent.current, agent.next_piece))
        if len(states) == 32:
            root.after(0, app.close)
        return result
    root.after(100, app.toggle)
    deadline = root.after(90000, app.close)
    start = time.perf_counter()
    try:
        with patch.object(game.Game, 'commit', commit):
            root.mainloop()
    finally:
        if not app.closed:
            root.after_cancel(deadline)
            app.close()
    assert not app.errors, app.errors
    assert len(states) == 32, (len(states), app.planner_blocked)
    return states, {'frameskip': frameskip, 'moves': len(states), 'wall_seconds': time.perf_counter() - start,
                    'rendering': app.render_metrics()}


if __name__ == '__main__':
    game.configure_backend('auto')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    paced, a = run(True)
    unpaced, b = run(False)
    assert paced == unpaced, 'Frame skipping changed committed moves, scores, boards or piece stream'
    assert a['rendering']['deferred_requests'] > 0
    assert b['rendering']['deferred_requests'] == 0
    print(json.dumps({'passed': True, 'identical_gui_moves': len(paced), 'runs': [a, b]}, indent=2))
