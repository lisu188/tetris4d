from pathlib import Path
import hashlib

p = Path('sparse_tetris_4d_optimized.py')
s = p.read_text(encoding='utf-8')
b = p.read_bytes()
assert hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest() == '1ecea0c50865663dae62c47e2b7e6a9b36eb0955'

def change(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:100], s.count(old))
    s = s.replace(old, new)

change('\n\ndef target_candidate_metrics(', '''

def next_hyperlayer_target(board: Board, preferred: int = 1) -> int | None:
    target_layer_status(board, preferred)
    for target in sorted(range(HEIGHT), key=lambda level: (abs(level - preferred), level)):
        status = target_layer_status(board, target)
        if not status.buried and status.filled < AREA:
            return target
    return None


def target_candidate_metrics(''')
change("                 policy: str = 'hyperlayer', target: int = 1) -> None:", "                 policy: str = 'hyperlayer', target: int = 1, stop_after_clear: bool = False) -> None:")
change("        self.goal_complete = False\n        self.planner_blocked = ''\n        self.target_progress", "        self.stop_after_clear = stop_after_clear\n        self.goal_complete = False\n        self.planner_blocked = ''\n        self.target_progress")
change("if self.goal_complete and self.policy.get() == 'hyperlayer' and not self.human.get():", "if self.goal_complete and not self.human.get():", 2)
change("or (self.goal_complete and self.policy.get() == 'hyperlayer'):", "or self.goal_complete:")
change("        if cleared and self.policy.get() == 'hyperlayer' and not self.human.get():\n            self.goal_complete = True\n            self.running = False", "        if cleared and not self.human.get():\n            if self.stop_after_clear:\n                self.goal_complete = True\n                self.running = False\n            elif self.policy.get() == 'hyperlayer':\n                target = next_hyperlayer_target(self.game.board, int(self.target.get()))\n                if target is not None:\n                    self.target.set(str(target))")
change("            self.target_progress.set('First clear reached. Switch planner or restart to continue.' if self.goal_complete else\n                                     f'Z={layer.target}: {layer.filled:,}/{AREA:,} filled · {layer.buried:,} buried gaps · pause on first clear')", "            behavior = 'pause on clear requested' if self.stop_after_clear else 'continuous play'\n            self.target_progress.set(f'Z={layer.target}: {layer.filled:,}/{AREA:,} filled · {layer.buried:,} buried gaps · {behavior}')")
change("'HYPERLAYER CLEARED. Autoplay paused. Select network to continue, Human to play, or R to restart.'", "'HYPERLAYER CLEARED. Paused by --stop-after-clear. Restart or change planner to play again.'")
change("    first_clear = None\n    stop_reason = 'piece_limit'", "    first_clear = None\n    active_target, clear_events = target, []\n    stop_reason = 'piece_limit'")
change("decision = choose_hyperlayer_move(game.board, game.current, target, candidates, game.pieces)", "decision = choose_hyperlayer_move(game.board, game.current, active_target, candidates, game.pieces)")
change("        if move.layers and first_clear is None:\n            first_clear = game.pieces\n        if move.layers and (stop_after_clear or policy == 'hyperlayer'):\n            stop_reason = 'first_clear'\n            break", "        if move.layers:\n            if first_clear is None:\n                first_clear = game.pieces\n            clear_events.append({'piece': game.pieces, 'layers': move.layers,\n                                 'total_layers': game.layers, 'target': active_target if policy == 'hyperlayer' else None})\n            if stop_after_clear:\n                stop_reason = 'first_clear'\n                break\n            if policy == 'hyperlayer':\n                following_target = next_hyperlayer_target(game.board, active_target)\n                if following_target is not None:\n                    active_target = following_target")
change("target=target if policy == 'hyperlayer' else None, first_clear_piece=first_clear, stop_reason=stop_reason,", "target=target if policy == 'hyperlayer' else None, active_target=active_target if policy == 'hyperlayer' else None,\n                  first_clear_piece=first_clear, clear_events=clear_events, stop_reason=stop_reason,")
change("args.piece_set, args.policy, args.target)", "args.piece_set, args.policy, args.target, args.stop_after_clear)")
change("help='Default: hyperlayer, protecting target access with depth 1 and pausing after a clear; network restores neural-only search.'", "help='Default: continuous hyperlayer search with depth 1 and automatic retargeting after clears; network uses neural-only search.'")
change("help='Stop benchmark games after their first clear; always enabled for hyperlayer.'", "help='Pause autoplay or stop benchmarks after the first clear; disabled by default.'")
p.write_text(s, encoding='utf-8')
b = p.read_bytes()
assert hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest() == '7e20001949c2497eacc4ae88a7615ba61aa95983'

p = Path('tests/test_hyperlayer_gui.py')
s = p.read_text(encoding='utf-8')
old = """            check(app.goal_complete and not app.running and app.search is None, 'GUI did not pause at first clear')
            check('CLEARED' in app.status.get(), 'Clear milestone is not displayed')
            app.toggle()
            check(not app.running, 'Completed objective must remain paused')
"""
new = """            check(not app.goal_complete and app.running, 'GUI paused at a clear without an explicit request')
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
"""
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("app.game.board = (0,) + (2,) * (game.AREA - 1)", "app.game.board = (0, 10) + (2,) * (game.AREA - 2)", 1)
p.write_text(s, encoding='utf-8')

p = Path('README.md')
s = p.read_text(encoding='utf-8')
old = 'Use Start/Space to run; the planner pauses after its first clear.'
assert old in s
s = s.replace(old, 'Use Start/Space to run. Autoplay continues after clears and automatically selects an accessible target after compaction.')
old = 'The GUI shows target coverage and buried gaps, pauses on the first clear, and distinguishes an unsafe/stalled planner from an actual top-out. Switch to `network` or Human mode to continue after clearing.'
assert old in s
s = s.replace(old, 'The GUI shows target coverage and buried gaps and continues after each clear. It keeps the current target if accessible; otherwise it selects the nearest incomplete target with no buried gaps (lower Z breaks distance ties). The board, score, piece sequence, and running state are preserved. `--stop-after-clear` explicitly requests the old first-clear pause for the GUI or benchmark. The dedicated `tools/first_hyperlayer.py` helper remains a first-clear diagnostic. A genuinely blocked planner is still distinguished from a top-out.')
p.write_text(s, encoding='utf-8')

p = Path('docs/first-hyperlayer.md')
s = p.read_text(encoding='utf-8')
old = 'Autoplay and benchmark target mode stop after the first hyperlayer clear. The GUI displays the milestone and remains paused until the user switches policy, enables Human mode, or restarts. This scope is intentional: repeated clearing and indefinite survival were not established by the research. Manual play does not enforce the planner\'s extra restrictions.'
assert old in s
s = s.replace(old, 'Autoplay and benchmarks continue after clears by default. After compaction the planner retains an accessible current target or selects the nearest incomplete, unburied target, with lower Z breaking equal-distance ties. It does not reset the board, refill cells, change the piece stream, or weaken target protection. Single-step and manual pause remain paused. An explicit `--stop-after-clear` pauses the GUI or stops benchmarks at their first clear; `tools/first_hyperlayer.py` intentionally retains its first-clear replay format. A blocked target selected manually still pauses rather than silently bypassing the protection rule.\n\nOn seed 7, the continuous default configuration reached 5,000 pieces without topping out, with clears at pieces 1,251, 2,478 and 3,880. The successive target levels were 1, 2, 4 and 5. This is one finite regression trajectory, not proof of indefinite survival; the remaining stack and holes can still grow. Benchmark reports retain `first_clear_piece` and also include `clear_events` and `active_target`.')
s = s.replace('first-clear pause, and switching back to normal play.', 'continuous post-clear retargeting, explicit first-clear pause, and switching back to normal play.')
p.write_text(s, encoding='utf-8')
