# All 16 sections and adaptive frame skipping

Normal argument-free launch displays every W section, W=0 through W=15, simultaneously in a fixed 4-by-4 grid. There are no pages or page buttons. Clicking a section, the W slider, atlas and Follow active W select the activation heatmap and outline; they never hide a section. All 16 camera resets remain available via Home. Compact cameras fit each complete well within its canvas at the supported 1120-by-760 minimum window size. At that size individual voxels are small; enlarging the window improves legibility. These are W-section boards, not a projection of the tesseract's eight boundary cells. Ordinary opaque voxel rendering can still occlude interior cells.

## Rendering, not simulation

Adaptive frame skipping is on by default with a nominal maximum scheduled redraw rate of 30 Hz. The event loop and search still execute their normal updates. Duplicate draw requests inside one logic tick are coalesced. Requests before the next redraw deadline share one pending callback; it renders the latest state, not a queue of obsolete frames. Render cost is smoothed and the interval is increased up to 200 ms when drawing is expensive. Explicit user selections and paused/manual completion can force an immediate redraw. Therefore 30 Hz is not a hard ceiling on every user-forced redraw, nor a guaranteed display FPS.

No moves, collision checks, rewards or layers are skipped. The animation may omit intermediate positions. Closing the app cancels a pending redraw. Paused/manual state stays paused, and normal autoplay still continues after clears. Headless search, native C, trained parameters and piece enumeration are unchanged.

Only W sections whose voxel occupancy or piece ownership changes invalidate cached stack geometry. Camera changes, resizing and cutaways also invalidate affected views. Unchanged sections remain on the canvas. Activation display updates are coalesced with actual rendering instead of being cleared and repainted for every search transition.

```bash
.venv/bin/python sparse_tetris_4d_optimized.py
.venv/bin/python sparse_tetris_4d_optimized.py --ui-fps 60
.venv/bin/python sparse_tetris_4d_optimized.py --no-frameskip
.venv/bin/python tools/summarize_benchmark.py --output benchmark-summary.json
```

## Telemetry

Metrics contain a top-level `rendering` object, retained even after the Tk window closes: requested FPS, adaptive interval, smoothed render cost, rendered frames, deferred and coalesced requests, scene rebuilds, unchanged-section skips, and all 16 visible W indices. `gui_draw_submission` now measures actual `_draw_now` calls rather than deferred requests; `gui_all_views_submission` replaces the eight-view panel metric. Per-well times remain inclusive children of these measurements and must not be added together. Deferred/coalesced request counts are not dropped physics updates or a count of missing display refreshes.

## Verification

The existing `tests/test_eight_views.py` entry point now checks all 16 sections, both window sizes, orbit bounds, rigid shapes spanning W=7 and W=8 at once, selective cache invalidation, first-clear continuation, deferred redraw delivery and telemetry. `tests/test_frame_pacing.py` uses deterministic clocks for pacing/CLI tests. `tests/test_frame_pacing_gui.py` compares 32 complete committed GUI moves with pacing enabled and disabled.

A local fixed-workload test replayed seed-7 boards at pieces 401 through 450 in a 640-by-600 panel, without active animation, three times. Median CPU/Tk draw submission fell from 5493.997 ms for the prior eight-view panel to 1718.025 ms for the cached sixteen-view panel; scene rebuilds fell from 400 to 111. This is not an equal-view-count comparison or a WSLg/display-FPS result. It measures caching/layout changes, not a standalone frame-skip speedup. Raw local results are in `validation/all-sections-rendering.json`.
