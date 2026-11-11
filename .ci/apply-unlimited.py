from pathlib import Path
import hashlib

source = Path('sparse_tetris_4d_optimized.py')
raw = source.read_bytes()
assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == 'fbc183bc16d8ded049c7257286192b1bab3a6729'
s = raw.decode('utf-8')

def replace(before, after):
    global s
    assert s.count(before) == 1, (before[:100], s.count(before))
    s = s.replace(before, after)

replace('class TetrisApp:\n', "PLAYBACK_SPEEDS = {'0.5/s': .5, '1/s': 1.0, '3/s': 3.0, '6/s': 6.0, '12/s': 12.0, 'Unlimited': 0.0}\n\n\nclass TetrisApp:\n")
replace('        self.speed = tk.DoubleVar(value=3.0)\n', "        self.speed = tk.DoubleVar(value=3.0)\n        self.speed_choice = tk.StringVar(value='3/s')\n")
replace("        ttk.Scale(controls, from_=.5, to=12, variable=self.speed, length=150).pack(side='left')", "        self.speed_box = ttk.Combobox(controls, textvariable=self.speed_choice,\n                                      values=tuple(PLAYBACK_SPEEDS), state='readonly', width=10)\n        self.speed_box.pack(side='left')\n        self.speed_box.bind('<<ComboboxSelected>>', self.change_speed)")
replace('    def change_policy(self, event=None) -> None:\n', "    def change_speed(self, event=None) -> None:\n        self.speed.set(PLAYBACK_SPEEDS[self.speed_choice.get()])\n        self.previous = time.perf_counter()\n        self.draw(force=True)\n\n    def change_policy(self, event=None) -> None:\n")
replace('if event.widget in (self.piece_set_box, self.policy_box, self.target_box):', 'if event.widget in (self.piece_set_box, self.policy_box, self.target_box, self.speed_box):')
replace('        result = self.frame_pacer.report()\n', "        result = self.frame_pacer.report()\n        rate = self.speed.get()\n        result.update(playback_mode='unlimited' if rate == 0 else 'animated',\n                      animation_rate_per_second=None if rate == 0 else rate)\n")
replace("            self.status.set(f'{\"Running\" if self.running else \"Paused\"} · {mode} · Space start/pause · Enter step · [ ] W view · R restart. A clear needs 4,096 cells at one Z.')", "            pace = 'Unlimited · no animation delay' if self.speed.get() == 0 else f'{self.speed.get():g}/s animation'\n            self.status.set(f'{\"Running\" if self.running else \"Paused\"} · {pace} · {mode} · Space start/pause · Enter step · [ ] W view · R restart. A clear needs 4,096 cells at one Z.')")
start = s.index('    def tick(self) -> None:\n', s.index('class TetrisApp:'))
end = s.index('\n\ndef reference_drop(', start)
s = s[:start] + '''    def _advance_search(self, deadline: float) -> None:
        while self.search is not None and time.perf_counter() < deadline:
            try:
                self.evaluations = next(self.search)
            except StopIteration as result:
                self.decision = result.value
                self.search = None
                self.planned = self.decision.move if self.decision else None
                if self.decision is None:
                    self.running = self.pending_step = False
                    self.game.game_over = not has_legal_move(self.game.board, self.game.current)
                    if not self.game.game_over:
                        layer = target_layer_status(self.game.board, int(self.target.get()))
                        reason = f'{layer.buried} buried target gaps' if layer.buried else 'no target-preserving placement'
                        self.planner_blocked = f'PLANNER PAUSED: {reason}. Choose another target, Human, network, or restart. Not game over.'
                    self.start_button.configure(text='Paused')
                self.finish_plan()
                if self.pending_step:
                    self.place()
                break

    def tick(self) -> None:
        if self.closed:
            return
        now = time.perf_counter()
        dt = min(.08, now - self.previous)
        self.previous = now
        self._in_tick = True
        try:
            if self.running and not self.human.get() and self.speed.get() == 0:
                deadline = now + .008
                while (self.running and not self.closed and not self.game.game_over
                       and time.perf_counter() < deadline):
                    self._advance_search(deadline)
                    if not self.running or self.search is not None or self.planned is None:
                        break
                    self.place()
            else:
                was_searching = self.search is not None
                self._advance_search(time.perf_counter() + .008)
                if self.running and not was_searching and self.search is None and self.planned:
                    self.progress += dt * self.speed.get()
                    if self.progress >= 1:
                        self.place()
        finally:
            self._in_tick = False
        if self.closed:
            return
        self.draw(force=self._force_draw)
        self._force_draw = False
        elapsed = time.perf_counter() - now
        delay = 1 if self.running and not self.human.get() and self.speed.get() == 0 else max(1, 16 - int(elapsed * 1000))
        self.tick_id = self.root.after(delay, self.tick)
''' + s[end:]
compile(s, str(source), 'exec')
source.write_text(s, encoding='utf-8')

readme = Path('README.md')
s = readme.read_text(encoding='utf-8')
s += '''

## Playback speed

The speed slider is replaced with a selector: **0.5/s, 1/s, 3/s, 6/s, 12/s, Unlimited**.
The existing 3/s animated default is retained. Select **Unlimited** and press Start/Space
(or change the selection while already running) to remove animation waits. No launch
arguments are required. The /s presets are animation rates, not guaranteed throughput.

Unlimited commits every selected move in order, with the same search and validation.
It can complete multiple moves per logic callback and yields between bounded search
batches using an 8 ms cooperative work budget and a 1 ms event-loop reschedule.
An individual search batch or render can exceed that budget. Adaptive frame skipping
paces only drawing; all 16 W-section cubes remain visible. Speed changes do not reset
the board, discard the selected move, start paused play, or change the piece stream.
Pause, single-step, Human mode and explicit --stop-after-clear remain available.
Logs record playback_mode and animation_rate_per_second under rendering; Unlimited
uses a null animation rate, not a non-finite number. See docs/playback-speed.md.
'''
readme.write_text(s, encoding='utf-8')
assert hashlib.sha256(Path('native/tetris_native.c').read_bytes()).hexdigest() == '91a417d63200afda5a0a0ed6ea679b2831788ddd347af526d5fc7be0f236e09d'
