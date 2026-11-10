# Tetris 4D

A single-file Python/Tkinter falling-block game on a **16 × 16 × 16 × 16** board, with rigid pieces, a fixed sparse sigmoid evaluator, and live network activation visualization.

## Run

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

Default search uses depth 2, beam width 4, and 2,048 deterministically sampled candidate placements per board. `--candidates 0` enumerates every origin/orientation and can be expensive; the beam setting still limits first-ply lookahead. Use `--help` for the complete command-line interface.

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
