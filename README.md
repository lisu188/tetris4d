# Tetris 4D

Train a sigmoid policy with PPO to maximize successful placements in a
**16 × 16 × 16 × 16** falling-block game. The repository includes the rigid-body
engine, native CPU scorer, Python/Tkinter interface, survival trainer, checkpoint
comparison and a neuron-by-neuron audit.

[Spatial V4 PPO](docs/ppo-workflow.md) is the supported learned policy training
workflow. It trains the spatial actor and regional survival critic, then selects
checkpoints from completed games. Compare frozen checkpoints on independent
seeds before choosing a policy to play.

## Install and build

CI uses Python 3.12. PPO needs NumPy and a compatible PyTorch build; CUDA is
optional. The native scorer needs a C compiler. Tkinter and a desktop display are
needed only for the UI.

```bash
git clone --depth 1 https://github.com/lisu188/tetris4d.git
cd tetris4d
python -m venv .venv
.venv/bin/python -m pip install -r requirements-neural.txt
.venv/bin/python tools/build_native.py
```

Use your platform's compatible PyTorch build for CUDA. On Ubuntu, install Tkinter
with `sudo apt install python3-tk`; on Windows, include Tcl/Tk in the Python install.
See [native builds and diagnostics](docs/native-backend.md).

## Train for survival

Start an eight-hour continuation from a local, calibrated Spatial V4 checkpoint:

```bash
.venv/bin/python -u train_survival_v4.py \
  --hours 8 --run-dir runs/survival-ppo-v4 \
  --pretrained /path/to/calibrated-v4.pt --preserve-pretrained-critic \
  --calibration-games 0 --allow-uncalibrated \
  --device cuda --backend native --screening exhaustive \
  --regional-critic --recover-target --fixed-lr --pieces 0 --selection-pieces 64000
```

The [PPO workflow](docs/ppo-workflow.md) covers new initialization, calibration,
resuming within a fixed deadline, and held-out evaluation. The bundled
`models/trained-sigmoid.json` supplies the teacher scores used to prepare moves.
Learned checkpoints are supplied as local files.

Use the [live training dashboard](docs/training-dashboard.md) for local charts,
calibration and PPO progress, and timestamped CSV/JSONL exports for later graphs.
See [training performance](docs/training-performance.md) for measured CPU
optimizations and the optional device input cache.

## Play a checkpoint

```bash
.venv/bin/python sparse_tetris_4d_optimized.py \
  --backend native \
  --neural-checkpoint runs/survival-ppo-v4/best.pt --neural-device cuda
```

Start/Space begins playback. An explicit neural checkpoint activates that policy.
Launching the game without one uses the sustained planner and bundled teacher
weights. Checkpoint selection and training do not automatically change this default.

The interface shows all 16 W sections, piece previews, a Z cutaway and learned
activations. Human mode enables manual placement.

| Action | Control |
| --- | --- |
| Move along X/Y | Arrow keys or W/A/S/D |
| Move along W | Q / E |
| Rotate in XY, XZ, XW, YZ, YW, ZW | 1–6 |
| Reverse rotation | Shift + 1–6 |
| Drop / single-step | Enter |
| Start/pause; drop in Human mode | Space |
| Orbit / zoom | Drag / mouse wheel |
| Restart | R |

See [view controls](docs/eight-views.md), [playback speed](docs/playback-speed.md)
and [logging](docs/benchmark-logging.md).

## Compare frozen models

Always supply checkpoint paths explicitly. This example evaluates the incumbent
and a completed PPO checkpoint against both sustained baselines:

```bash
.venv/bin/python tools/survival_experiments.py compare \
  --device cuda --backend native --threads 4 \
  --checkpoints /path/to/calibrated-v4.pt runs/survival-ppo-v4/best.pt \
  --seeds 960001 960003 960005 960007 --pieces 32000 \
  --modes greedy sampled --action-seeds 101 102 103 \
  --record-games --record-max-mib 128 \
  --output survival-comparison-ppo-heldout.json
```

Use seeds reserved before training. Keep greedy and sampled results separate and
average sampled repeats within each game seed. A game that reaches the piece cap
provides a lower bound on survival. See [evaluation and provenance](docs/ppo-workflow.md).

## Explain every learned unit

The [neuron audit](docs/neuron-analysis.md) reports the exact computation,
checkpoint coefficient references, activations and intervention effects of every
unit/channel. The default network has 562 named entries, including shared
convolution channels and scalar outputs. Its report distinguishes their learned
responses from unproven semantic labels.

## Check the runtime

```bash
.venv/bin/python sparse_tetris_4d_optimized.py --test --backend python
.venv/bin/python tests/test_survival_v4.py
.venv/bin/python tests/test_v4_improvements.py
.venv/bin/python tests/test_native.py --trajectory-pieces 2600
```

V4 uses the shared network, replay and PPO code exercised by the compatibility
tests. The former trainer commands can be recovered with their original source
revision for exact replay of historical experiments.

## Game rules and repository contents

Gravity acts along negative Z; X, Y and W are horizontal. Pieces rotate in six
coordinate planes and drop rigidly. A clear requires all 4,096 X/Y/W cells at one
Z level; higher layers shift down after a clear. There are no independent-cell
gravity, wall kicks, hold, under-overhang slides or timed-input constraints.
The expanded piece set has 15 shapes; `--piece-set classic` selects seven.

The checkout keeps runtime source, tests and fixtures, small validation reports,
the teacher model. Historical training outputs, demonstrations
and benchmark logs are listed in the [archive catalog](validation/archive-manifest.json).
Git history preserves their committed versions. Existing dirty and untracked
local experiments are preserved separately; they can differ from the cataloged
historical blobs. See [storage and recovery](docs/ppo-workflow.md).

## Game library

Every game played by the engine (training rollouts, calibration and selection games,
pretraining/critic collection tools and the GUI) is recorded once, compactly and
content-addressed, in `data/game-library/` of the main checkout (set
`TETRIS4D_GAME_LIBRARY=<dir>` to relocate or `=off` to disable; tests do not record).
Trainers attach the phase, behaviour policy, policy update and target layer of each
placement. Reuse recorded games from Python with `tetris_game_library.iter_games(...)`,
`replay(record)` (deterministic positions, verified final board) and
`record.remaining_placements()` (exact lifetime labels for completed games):

    python -B tetris_game_library.py stats
    python -B tetris_game_library.py verify --limit 100
    python -B tetris_game_library.py import runs/   # backfill legacy JSON recordings

