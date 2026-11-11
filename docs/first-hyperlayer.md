# Target-hyperlayer planner

This integrates the previously tested first-clear prototype into the single-file engine, command-line benchmark, and Tkinter interface. The target policy is opt-in. The neural-only policy, trained checkpoint, sigmoid activation function, board dimensions, 15 piece types, rigid locking, and complete-hyperlayer compaction remain unchanged.

## Use

```bash
python sparse_tetris_4d_optimized.py --policy hyperlayer --target 1 --weights models/trained-sigmoid.json
python sparse_tetris_4d_optimized.py --benchmark --policy hyperlayer --weights models/trained-sigmoid.json --pieces 2000 --games 1 --seed 7
python tools/first_hyperlayer.py --seed 7 --output first-clear.json
python tools/first_hyperlayer.py --verify first-clear.json
```

The helper loads the existing trained checkpoint by default. The main application retains its prior baseline-weight default; pass `--weights` to reproduce the published trajectories. `--policy network` preserves the original controller. Both computation backends are supported. `python -S` with `--backend auto` works without NumPy.

## Objective and safety

For target level t, reject any placement that buries an unfinished target cell without also filling it in the same rigid placement. Choose lexicographically by `(new_target_cells, sigmoid_afterstate_score)`, keeping the first enumerated candidate on an exact tie. This is a planner constraint; it is not a new activation, a trained checkpoint, or a modification of the game's collision rules.

The default target is Z=1. The planner can accept holes below this target. Each arriving vertical I can fill an isolated unfinished target column while the no-buried-gap invariant holds. Completing a hyperlayer requires all 4,096 cells across X/Y/W at one Z level; a filled W section alone is insufficient.

Normal candidates use the existing deterministic sampler, with the zero-based move number as salt, as in the prototype. All unfinished-column vertical-I placements are added explicitly. These may add up to 4,096 evaluations beyond the configured sample budget. The planner uses depth 1, ignores neural lookahead/beam settings, and preserves the trained sigmoid score as its secondary objective.

If the sample has no safe placement, a cooperative exhaustive fallback scans batches until it finds a safe batch, selecting its best candidate. It does not silently permit a buried gap. `--candidates 0` enumerates all placements in the initial pass. The generator yields between batches so pause, mode changes, and window events remain responsive.

An already buried target is refused rather than falsely treated as reachable. The UI pauses and offers target selection, Human mode, neural-only mode, or restart. A planner stall is not a game over unless the engine independently finds no legal placement.

## UI and completion

The Planner and Target Z selectors operate on the current game. Switching them cancels any pending search/drop. The progress indicator shows global coverage out of 4,096 and the number of buried gaps, not only the currently displayed W section. Sigmoid heatmaps continue to show the actual full-board score for the selected afterstate.

Autoplay and benchmark target mode stop after the first hyperlayer clear. The GUI displays the milestone and remains paused until the user switches policy, enables Human mode, or restarts. This scope is intentional: repeated clearing and indefinite survival were not established by the research. Manual play does not enforce the planner's extra restrictions.

## Verification

The integration reproduced all seven stored research trajectories exactly: 8,660 identical moves, identical final boards, and identical game summaries. First clears occur at pieces 1,251 (seed 7), 1,238 (30001), 1,237 (30002), 1,222 (30003), 1,241 (30004), 1,225 (30005), and 1,246 (30006). The Python backend separately reproduced seed 7's entire NumPy trajectory.

The replay verifier independently reconstructs the shuffled bag, descends each rigid piece with occupied-cell sets, tests overlap/bounds, counts complete hyperlayers, compacts them, and checks conservation and final state. It shares the registered orientation catalog but does not call the engine's placement, drop, clearing, or commit functions. It validates complete or partial replays; the runner additionally exits nonzero when it fails to clear within the requested cap.

Tests include all 15 piece types, target boundaries, matching Python/NumPy choices, unsampled final-gap repair, safe exhaustive fallback, cancellation, model/physics preservation, malformed inputs, and replay tampering. UI tests cover both 1560x960 and 1120x760 on Linux Xvfb, target changes, stalled-target recovery, first-clear pause, and switching back to normal play. Native Windows/macOS GUI was not tested locally.

```bash
python tests/test_hyperlayer.py --backend python
python tests/test_hyperlayer.py --backend numpy
python -S tests/test_hyperlayer.py --backend auto
xvfb-run -a -s '-screen 0 1600x1000x24' python tests/test_hyperlayer_gui.py
```

These finite-seed results do not guarantee a first clear for every piece set, target, candidate budget, or starting board. In particular, the 4D-only set lacks I pieces, so the isolated-gap finishing mechanism is unavailable there. Z=1, the expanded bag, the shipped trained checkpoint, and 2,048 base candidates are the tested configuration. No new training or additional neural inputs were introduced.
