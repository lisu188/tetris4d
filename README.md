# Tetris 4D

A single-file Python/Tkinter falling-block game on a **16 × 16 × 16 × 16** board, with rigid pieces, a sparse sigmoid evaluator with optional trained checkpoints, and live network activation visualization.

## Run

No arguments are needed: the default is the hyperlayer planner, target Z=1, and the bundled trained sigmoid checkpoint. The checkpoint is resolved relative to the script, so launching from another working directory works. Use Start/Space to run. Autoplay continues after clears and automatically selects an accessible target after compaction.

Requires Python 3.10 or newer. The graphical interface requires Tkinter and a desktop display. Headless tests and the Python computation backend require only the standard library.

```bash
git clone https://github.com/lisu188/tetris4d.git
cd tetris4d
python sparse_tetris_4d_optimized.py
```

For optional NumPy acceleration:

```bash
python -m pip install -r requirements-numpy.txt
python sparse_tetris_4d_optimized.py --backend numpy
```

The default `--backend auto` selects NumPy when available and otherwise uses the standard-library backend. To force the latter:

```bash
python sparse_tetris_4d_optimized.py --backend python
```

On Ubuntu, Tkinter can be installed separately with `sudo apt install python3-tk`. On Windows, include Tcl/Tk when installing Python. `python -m tkinter` checks whether a graphical Tk installation is available.

## Rules

- X, Y, and W are horizontal coordinates; gravity acts along negative Z.
- Pieces move, rotate, fall, and lock as rigid objects. Their cells never settle independently.
- Rotations are available in all six coordinate planes: XY, XZ, XW, YZ, YW, and ZW.
- Clearing requires all **4,096 X/Y/W cells** at one Z level to be occupied. A filled plane in only one W section is not enough.
- Clearing removes cells in the completed hyperlayer and shifts higher hyperlayers downward. This can remove part of a previously locked piece.
- This is a placement-based variant: choose an orientation and position above the stack, then drop vertically. It does not implement wall kicks, hold, under-overhang slides, or timed input constraints.

## Pieces

The default `expanded` set contains 15 shapes: seven classic shapes and eight additional shapes whose cell centers span four dimensions.

| New piece | Cells |
| --- | ---: |
| NOVA | 5 |
| STAIR | 5 |
| FORK | 5 |
| HOOK | 6 |
| BRIDGE | 8 |
| RING | 8 |
| CROSS | 9 |
| HYPER | 16 |

Use `--piece-set classic` for the original seven or `--piece-set 4d` for only the eight new pieces. The UI also provides a piece-set selector and a gallery.

```bash
python sparse_tetris_4d_optimized.py --piece-set 4d
python sparse_tetris_4d_optimized.py --catalog --output pieces.json
```

## Clear the first hyperlayer

The target planner and trained checkpoint are selected automatically. A plain launch is sufficient:

```bash
python sparse_tetris_4d_optimized.py
```

It protects unfinished cells on Z=1, prioritizes new target coverage, and uses the loaded sigmoid score only to break ties. Vertical I placements are explicitly generated for every unfinished target column. This is a depth-1 planner, not new network training; the Lookahead control is disabled while it is selected. Use `--policy network` for neural-only search, or `--policy network --baseline` to restore the original untrained controller.

The GUI shows target coverage and buried gaps and continues after each clear. It keeps the current target if accessible; otherwise it selects the nearest incomplete target with no buried gaps (lower Z breaks distance ties). The board, score, piece sequence, and running state are preserved. `--stop-after-clear` explicitly requests the old first-clear pause for the GUI or benchmark. The dedicated `tools/first_hyperlayer.py` helper remains a first-clear diagnostic. A genuinely blocked planner is still distinguished from a top-out. Target changes preserve the current board; restarting returns to an empty board. An already buried target must be changed or abandoned; the planner never silently relaxes its protection rule.

Run an empty-start demonstration and independently verify the replay:

```bash
python tools/first_hyperlayer.py --seed 7 --output first-clear.json
python tools/first_hyperlayer.py --verify first-clear.json
```

The helper loads `models/trained-sigmoid.json` by default. The normal game also loads this checkpoint automatically; `--weights PATH` overrides it and `--baseline` disables checkpoint loading. Both Python and NumPy backends are supported.

All seven reference trajectories were reproduced by the integrated planner: first clears at 1,222–1,251 pieces, with no physics changes. These are finite first-clear tests, not guarantees of indefinite survival. See [planner details](docs/first-hyperlayer.md) and [integration results](validation/hyperlayer-integration.json).

## Controls

Enable **Human** mode for manual placement.

| Action | Control |
| --- | --- |
| Move along X/Y | Arrow keys or W/A/S/D |
| Move along W | Q / E |
| Rotate in XY, XZ, XW, YZ, YW, ZW | 1–6, respectively |
| Reverse a rotation | Shift + 1–6 |
| Drop / advance one placement | Enter |
| Start/pause autoplay; drop in Human mode | Space |
| Change displayed W section | [ / ], slider, or the overview |
| Orbit / zoom | Drag / mouse wheel |
| Reset camera / restart | Home / R |

The interface includes an orbitable 3D W section, an overview of all 16 W sections, whole-piece previews split by W, a Z cutaway, and paged activation heatmaps. Hover or pin a neuron to inspect its value and connections.

## Evaluator and search

The network has **4,098 inputs → 23,040 sigmoid units → one linear output**, plus direct input-to-output connections: **73,218 weighted connections** in total. Inputs are 4,096 vertical-column heights, the total hole count, and the number of hyperlayers cleared by a candidate placement. Hidden units compare adjacent columns along X, Y, and W. There are no ReLU activations.

This is a **hand-designed evaluator, not a trained end-to-end agent**. The original scoring coefficients are inherited from [Yiyuan Lee's Tetris evaluator](https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/); their sigmoid/4D adaptation is not claimed optimal. Piece placement generation, collision handling, and lookahead are outside the network.

Default hyperlayer search uses depth 1 and 2,048 sampled candidates plus targeted vertical-I finishing placements. Neural-only search (`--policy network`) uses depth 2, beam width 4, and 2,048 deterministically sampled candidate placements per board. `--candidates 0` enumerates every origin/orientation and can be expensive; the beam setting still limits first-ply lookahead. Use `--help` for the complete command-line interface.

Optimizations include optional batched NumPy scoring, incremental feature updates, bounded caches, retained static rendering, and changed-cell heatmap updates. Both computation backends use the same rules, weights, sampling, and tie-breaking.

## Validation

Run the embedded engine tests without a display:

```bash
python sparse_tetris_4d_optimized.py --test --backend python
python sparse_tetris_4d_optimized.py --test --backend numpy
```

Check the automatic fallback without third-party site packages:

```bash
python -S sparse_tetris_4d_optimized.py --test --backend auto
```

Run a short reproducible comparison:

```bash
python sparse_tetris_4d_optimized.py --benchmark --seed 7 --games 3 --pieces 100 --compare --output benchmark.json
```

Short capped runs are functional checks, not evidence of long-term playing strength. One cleared hyperlayer consumes 4,096 cells, while piece sizes range from 4 to 16.

Export the complete network:

```bash
python sparse_tetris_4d_optimized.py --network --output network.json
```

The application remains in `sparse_tetris_4d_optimized.py`, including its built-in tests. NumPy is optional; Tkinter is only imported when launching the interface.

## Compact shared-sigmoid network

The evaluator now uses two shared sigmoid filters, exact histogram pooling and lazy activation traces. Logical neurons and gameplay are unchanged. See [network structure and measurements](docs/network-structure.md).

`--network` exports the compact structure; `--network-expanded` reconstructs every logical neuron and connection.

## Gameplay-trained checkpoint

A reproducible cross-entropy policy search has trained three shared sigmoid parameters. The trained checkpoint is loaded automatically; this explicit form remains equivalent:

```bash
python sparse_tetris_4d_optimized.py --weights models/trained-sigmoid.json
```

On six unseen 1,000-piece games with the normal two-piece search, mean final holes decreased from 139.0 to 63.8. Both policies survived every capped game and neither cleared a hyperlayer; this demonstrates better packing in this sample, not improved survival or scoring. See [training, held-out results, and limitations](docs/training.md).

Continue the saved optimizer without replacing this checkpoint:

```bash
python tools/train_agent.py --resume models/trained-sigmoid.json --generations 20 --output models/continued.json --report validation/continued.json
```


## Native scorer for WSL/Linux

An optional strict-float C backend fuses candidate scoring and target rejection, with runtime-gated AVX2 landing and BMI2 compaction. Build with `python tools/build_native.py`; normal no-argument launch then selects it automatically. NumPy and Python remain fallbacks. See [build commands, WSL diagnostics, validation and measured performance](docs/native-backend.md).

Run `python tests/test_native.py --trajectory-pieces 2600` and `python tools/benchmark_native.py --output native-benchmark.json` to validate and measure on your machine.


## Automatic benchmark logging

Normal launches also write a local JSONL log under `logs/` (the path is printed on stderr).
No additional arguments are required. Logs record runtime/backend/ISA/build metadata,
search active work versus UI scheduling latency, board evaluation, native batch scoring,
commit/validation, activation tracing, GUI draw submission, tick intervals, cache statistics,
and clear/progress events. Phase timings are inclusive and must not be added together.
GUI draw/tick measurements are not compositor FPS. No logs are uploaded automatically.

Each session rotates at approximately 4 MiB with three backups; sessions are kept separately.
`--benchmark-log PATH` selects a new file and `--no-benchmark-log` disables instrumentation.
Tests, catalog/network exports and imported training/benchmark helpers do not log by default.
A file-system logging failure warns and disables logging without stopping gameplay.

```bash
.venv/bin/python sparse_tetris_4d_optimized.py
.venv/bin/python tetris_benchmark_logging.py logs/benchmark-*.jsonl* --output benchmark-summary.json
.venv/bin/python tools/benchmark_logging.py --pieces 1500 --repetitions 3 --output logging-overhead.json
```

The summary groups sessions; submit the summary and, for detailed investigation, the matching
JSONL segments. Logging has overhead: the paired benchmark measures it rather than subtracting
an assumed cost. Recent p50/p95/p99 values cover the last 2,048 observations per phase.
See `docs/benchmark-logging.md` for fields, privacy and timing limitations.
