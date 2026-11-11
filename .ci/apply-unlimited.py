from pathlib import Path
import hashlib

p = Path('sparse_tetris_4d_optimized.py')
raw = p.read_bytes()
assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == '5887a9188e0adb855ea5e28b6dffd82484c45e71'
s = raw.decode('utf-8')
a = s.index('    def _draw_now(self) -> None:')
b = s.index('    def _advance_search(', a)
part = s[a:b]
changes = [
    ('        self.network.update(self.planned, w)', "        trace_move = self.planned\n        if trace_move is None and self.speed.get() == 0 and not self.human.get():\n            trace_move = self.game.last_move\n        self.network.update(trace_move, w)"),
    ('        if self.planned:\n            m, p = self.planned, self.planned.candidate', "        if trace_move:\n            m, p = trace_move, trace_move.candidate\n            label = 'LAST COMMITTED\\n' if self.planned is None else ''"),
    ("            self.features.set(f'{m.piece}", "            self.features.set(label + f'{m.piece}"),
]
for old, new in changes:
    assert part.count(old) == 1, old
    part = part.replace(old, new)
s = s[:a] + part + s[b:]
compile(s, str(p), 'exec')
p.write_text(s, encoding='utf-8')
p = Path('tests/test_unlimited_speed_gui.py')
s = p.read_text(encoding='utf-8')
old = "                    metrics = app.render_metrics()"
new = """                    app.draw(force=True)
                    root.update_idletasks()
                    check(app.planned is None, 'Stopped test must have no next selected move')
                    check(app.network.values is not None, 'Unlimited blanked the activation display')
                    last = app.game.last_move
                    expected_trace = game.network_trace(last.heights, last.holes, last.layers)
                    check(app.network.values[game.OUTPUT_INDEX] == expected_trace[game.OUTPUT_INDEX], 'Activation display does not match last committed move')
                    check('LAST COMMITTED' in app.features.get(), 'Displayed trace source is not labelled')
                    metrics = app.render_metrics()"""
assert s.count(old) == 1
p.write_text(s.replace(old, new), encoding='utf-8')
p = Path('docs/playback-speed.md')
s = p.read_text(encoding='utf-8')
s += '\nWhile Unlimited is between decisions, the activation panel retains the last committed move\'s actual trace and labels its features LAST COMMITTED. It does not draw that piece again on the board. Finite-rate and Human previews retain their previous behavior.\n'
p.write_text(s, encoding='utf-8')
