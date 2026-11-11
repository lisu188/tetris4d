# Eight simultaneous 3D sections

The normal no-argument launch now shows eight 3D W-section boards in a 4-by-2 grid. This integrates the UI from the supplied benchmark-logging patch without replacing the newer logging implementation on main.

Use W 0–7 and W 8–15 to switch pages; all 16 sections of the unchanged 16-by-16-by-16-by-16 board remain accessible. Clicking a section selects its W coordinate for the activation heatmap. The W slider and atlas remain available, and Follow active W follows the active origin's page. The whole-piece preview still shows a piece spanning more than one W section. Each compact view supports orbiting; zoom is capped to keep the complete well visible. Home resets all eight cameras.

Eight views means eight section boards, not eight cells per piece and not eight boundary cells of a projected tesseract. The 15 registered rigid shapes, collision rules, trained sigmoid network, default native-backend selection and continued play after clears are unchanged.

## Benchmark logging

The existing rotating JSONL recorder remains the only logger. It adds gui_eight_views_submission for the complete eight-view panel, while gui_well_submission times individual sections. These inclusive timings overlap with gui_draw_submission and must not be added together. They measure CPU/Tk submission, not displayed FPS.

GUI metrics include view_page and visible_w_sections. Page transitions produce view_page events. Existing session rotation, failure handling, source/build metadata and --no-benchmark-log are retained.

```bash
.venv/bin/python sparse_tetris_4d_optimized.py
.venv/bin/python tools/summarize_benchmark.py --output benchmark-summary.json
```

The summary helper defaults to the most recently modified default session and includes its remaining rotated segments. Explicit paths are also accepted. The original tetris_benchmark_logging.py summary entry point remains available.

## Regression checks

```bash
xvfb-run -a -s '-screen 0 1600x1000x24' python tests/test_eight_views.py
python tests/test_benchmark_logging.py --trajectory-pieces 2600
```

The GUI test checks all eight canvases at 1560-by-960 and 1120-by-760, all 16 W sections across both pages, complete projected well corners, activation heatmaps, rigid RING cells across a page boundary, post-clear continuation, and integration with the existing logger. Virtual-display tests do not establish WSLg performance or indefinite gameplay survival.
