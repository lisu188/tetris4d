from pathlib import Path
import hashlib
import lzma
import subprocess


def verify(path, expected):
    data = Path(path).read_bytes()
    assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == expected, path


verify('sparse_tetris_4d_optimized.py', '5c3ee2020f3757b2ab473f1d4eebace6a41eccc0')
verify('tetris_benchmark_logging.py', '6b8be9bf602cdbc3203ef5e17ac9258d4112f2f1')
patch = lzma.decompress(Path('.ci/eight-view-only.patch.xz').read_bytes())
assert hashlib.sha256(patch).hexdigest() == '9f5df8a45a2f2bdd752e6e54a5cd84df93b56094017da5efdb7d9b2b4ce18ee6'
subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, check=True)

p = Path('tetris_benchmark_logging.py')
s = p.read_text(encoding='utf-8')
old = "                       'hyperlayers': self.app.game.layers, 'game_over': self.app.game.game_over}\n"
new = old + "                if hasattr(self.app.well, 'views'):\n                    gui.update(view_page=self.app.well.page,\n                               visible_w_sections=list(range(self.app.well.page * 8, self.app.well.page * 8 + 8)))\n"
assert s.count(old) == 1
s = s.replace(old, new)
old = "        original_tick = self.game.TetrisApp.tick\n"
new = '''        if hasattr(self.game, 'EightWellView'):
            self._timed(self.game.EightWellView, 'draw', 'gui_eight_views_submission')
            original_panel = self.game.EightWellView.draw
            @wraps(original_panel)
            def panel(view, *args, **kwargs):
                before = view.page
                result = original_panel(view, *args, **kwargs)
                if view.page != before:
                    self.emit('view_page', page=view.page, first_w=view.page * 8, last_w=view.page * 8 + 7)
                return result
            self._patch(self.game.EightWellView, 'draw', panel)
''' + old
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s, encoding='utf-8')
p = Path('README.md')
s = p.read_text(encoding='utf-8')
s += '\n\n## Eight simultaneous 3D sections\n\nNormal launch shows eight W-section boards in a 4×2 grid. Use **W 0–7** or **W 8–15** for the other page; all sixteen sections and the whole-piece preview remain accessible. Sigmoid activations, native acceleration and continued play are unchanged. Existing benchmark logs now include aggregate eight-view drawing time and the visible sections. See [view controls and telemetry](docs/eight-views.md).\n\nSummarize the latest default session, including retained rotation segments:\n\n```bash\n.venv/bin/python tools/summarize_benchmark.py --output benchmark-summary.json\n```\n'
p.write_text(s, encoding='utf-8')
verify('native/tetris_native.c', '4b5e8351abd1c433e5b31cf72ba50e9ad2533b48')
verify('models/trained-sigmoid.json', '9c2e7937cd2aef5841d83b2309715d68f9b329ef')
