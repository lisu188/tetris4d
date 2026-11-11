# Playback speed

The speed slider has been replaced by a read-only selector with 0.5/s, 1/s, 3/s, 6/s, 12/s and Unlimited. The default remains the previous 3/s animation. Launch normally, select Unlimited, and press Start/Space. Selecting a speed during paused or Human play does not start autoplay. Selecting it during autoplay takes effect on the next logic callback without restarting or discarding a pending decision.

The finite rates control animation progress, not guaranteed achieved moves per second. Unlimited removes the per-piece animation wait. It runs the same planner, candidate enumeration, scoring, commit validation, clear handling and retargeting, repeatedly committing completed decisions while the callback's work budget allows. It is not a larger numerical animation multiplier and is not limited to one move per rendered frame.

Work yields at search-batch and move boundaries after an 8 ms cooperative callback budget. The next active Unlimited callback is scheduled with a 1 ms Tk timer to let input and display events run. A single slow search batch, commit or render can exceed that budget; Unlimited is hardware-limited, not a promise of zero UI latency or infinite throughput. Tk still owns rendering on the event-loop thread. The existing adaptive render pacer is independent: no moves or physics updates are dropped when intermediate frames are omitted.

All 16 section cubes stay on screen. Pause, Enter/Drop single-step, Human control, game over, protected-target stalls and explicit --stop-after-clear retain their existing semantics. Clears do not normally stop play. Trained weights and native C code are unchanged; an existing matching native build does not need rebuilding.

Benchmark snapshots include rendering.playback_mode (animated or unlimited) and rendering.animation_rate_per_second (null for unlimited). The existing numeric GUI speed field uses zero for Unlimited. The regular log summary command is unchanged. Compare runs with the same backend, seed, search budget, rendering settings and playback mode; rendered FPS is not simulation throughput.

Run the regression suites with:

```bash
python tests/test_unlimited_speed.py
python tests/test_unlimited_speed.py --trajectory-pieces 2600
xvfb-run -a -s '-screen 0 1600x1000x24' python tests/test_unlimited_speed_gui.py
```

The trajectory test exercises the actual GUI logic methods with a mock root/variables and no rendering, comparing every move and board to the headless reference. The GUI suite uses real Tk callbacks and tests visibility, selection, pause/step/manual controls, explicit and continuous clear handling, callback delivery and telemetry at two window sizes. Neither is a WSLg throughput benchmark.

While Unlimited is between decisions, the activation panel retains the last committed move's actual trace and labels its features LAST COMMITTED. It does not draw that piece again on the board. Finite-rate and Human previews retain their previous behavior.
