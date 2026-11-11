# Retained atlas and batched mesh rendering

This change targets redraw CPU work, not neural search. All 16 W-section wells remain visible simultaneously. The trained sigmoid model, native C source, candidate order, rigid physics, piece stream, continuous clears, Unlimited mode, and adaptive frame pacing are unchanged. Runtime implementation remains in the main Python file with no new dependency. An existing compatible native library does not need rebuilding.

## Changes

The top-view atlas retains its canvas rectangles and updates only columns whose visible height changes. Empty columns are hidden rather than repeatedly destroyed/recreated. Section labels, selection outlines, and active-piece overlays update independently. Layout reconstruction happens only when the canvas dimensions change. Tile storage is bounded by 4,096 cells; resets and clears remove stale visible content without accumulating canvas items.

The all-well ownership map handles append-only locked-piece snapshots incrementally, including several moves coalesced into one render. Prefix object identities are verified. A clear, reset, replacement or non-append history triggers a full rebuild, preserving piece identity, colors, boundaries and the existing per-section invalidation rules. The prefix verification remains O(number of locked pieces); this is not a claim that all bookkeeping is O(1).

Face-edge offsets are precomputed once. Face depths keep the original arithmetic order and visible faces retain their original stable order. Static polygons and piece-boundary lines are submitted in one Tcl call per rebuilt mesh instead of one Python/Tcl call per primitive. Arguments are passed as Tcl lists and expanded with `{*}`; user-controlled text is not concatenated into executable script. Active-piece occlusion, landing outlines, camera orbit, cutaways and zoom behavior are unchanged.

Additional rendering telemetry reports `atlas_scene_rebuilds`, `atlas_cell_updates`, `atlas_cells_created`, `ownership_appends`, `ownership_full_scans`, and `static_mesh_batches`. These counters describe work, not display FPS.

## Matched-work benchmark

Baseline: `2266a4d5e01ff1df5406115c6f47970141deec44`. GitHub-hosted Linux/Xvfb, Python 3.12.16, Tk 8.6. Three alternating-order paired repetitions, 80 identical rendered frames per repetition, seed 7, fixed cameras, all 16 sections plus atlas, no active-piece animation. The initial piece-1201 frame is warm-up; measured frames end at piece 1281 and include the first hyperlayer clear. Every frame is rendered: no frame-skip benefit is included.

| Phase, median total for 80 frames | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| All 16 wells | 4,485.605 ms | 2,895.700 ms | 1.55x |
| Atlas | 1,294.355 ms | 32.072 ms | 40.36x |
| Combined draw submission | 5,779.960 ms | 2,924.992 ms | 1.98x |
| Submission plus Tk idle painting | 5,992.446 ms | 3,078.957 ms | 1.95x |

Medians are calculated independently for each phase, so phase medians need not add exactly. The measurement excludes search, the network panel, the application scheduler and display compositor/WSLg latency. It is not a whole-game FPS claim or a measurement of the user's machine. Raw per-frame timings are in `validation/rendering-performance.json`.

## Validation

Eight regression tests compare face geometry exactly against the pinned baseline, including fractional animation heights; compare canvas primitives and boundary tags; compare complete well scenes across cameras and cutaways; compare atlas content through updates, clears and resizes; validate batched/coalesced ownership, replacement and reset behavior; and verify safe Tcl argument quoting. A source-AST comparison verifies that this patch does not change engine/search/GUI-scheduling functions. The existing engine, all-section visibility, frame-pacing, Unlimited and hyperlayer GUI checks passed during the measured build. PR workflows provide additional backend and platform checks before merging.

## WSL use

Launch from the existing checkout without new arguments:

```bash
git switch main && git pull --ff-only
.venv/bin/python sparse_tetris_4d_optimized.py
```

Select **Unlimited** in the existing speed selector for hardware-limited play rather than the default 3/s animation pacing. This setting is separate from rendering cost. After closing a representative session:

```bash
.venv/bin/python tools/summarize_benchmark.py --output benchmark-summary.json
```

Run the matched renderer comparison on the actual WSL display:

```bash
.venv/bin/python tools/benchmark_rendering.py --output wsl-render-benchmark.json
```

The tool requires a desktop display and Git history containing the pinned baseline. `--baseline /path/to/baseline.py` supports a separately saved baseline when history is unavailable. Its default native fixture simulation runs before timed rendering. The report contains environment and raw timing data; it does not upload anything. A system without a display can use Xvfb, but that measures virtual-display behavior rather than WSLg.

References: Tk canvas `itemconfigure` and Tcl list argument expansion in the official Tcl/Tk manual (`https://www.tcl-lang.org/man/tcl8.6/TkCmd/canvas.htm`, `https://www.tcl-lang.org/man/tcl8.6.14/TclCmd/Tcl.htm`).
