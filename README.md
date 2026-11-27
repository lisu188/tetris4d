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

## Play to 100,000 placements

The native `cross-wells` planner cleared 144 hyperlayers in 100,000 placements
on seed 7 in the standard expanded game. All 100,000 hard drops replayed
independently. Four additional seeds chosen before testing also reached the cap
with a legal next move and 144 cleared layers each. The planner uses
ordinary pieces to build landing sites where a CROSS can drop without trapping
empty cells. It reads the current board and piece and needs no checkpoint.

After building the native scorer, reproduce the run from an empty board:

```bash
.venv/bin/python sparse_tetris_4d_optimized.py --benchmark --backend native \
  --policy cross-wells --piece-set expanded --seed 7 --games 1 --pieces 100000 \
  --no-benchmark-log
```

For GUI playback, run `.venv/bin/python sparse_tetris_4d_optimized.py --policy cross-wells --seed 7`
and press Space. [Survival results and replay fingerprints](validation/cross-wells-100k.json).

## Trained policy: 1,000,000 placements

The `trained-wells` policy keeps the `cross-wells` candidate generator but
replaces its fixed priority rules with a learned linear score over seven
per-move features (newly buried levels, new holes, covering a ready CROSS well,
well progress, height cost, roughness and peak height).
`tools/train_wells_policy.py` trains the seven weights with the cross-entropy
method. It optimizes survival from empty boards, starting from a zero-mean
Gaussian with no preset coefficients, signs or priorities. Training took 5
generations of 16 candidates on fresh seeds, about 18 minutes. The bundled
`models/trained-wells.json` is the generation-4 mean.

Median survival across games is at least 1,000,000 placements. A 16-seed
evaluation set (62000001 to 62000016) was registered before play, with the
median rule fixed in advance. All 16 games reached the 1,000,000-placement cap
from an empty board, so the median is censored at the cap. The exact 95%
Clopper–Pearson lower bound on the chance that a game reaches 1,000,000 is
0.79. Five earlier verification seeds (61000001 to 61000005) also reached the
cap, giving 21 of 21 games overall, with a pooled bound of 0.84.

Every game ended with a legal next move after clearing 1,458 to 1,461
hyperlayers. Sampled every 20,000 placements, the stack never exceeded height
11. Fourteen games had zero holes at every sample. The other seven carried at
most 21 holes while clearing layers at the same rate. An independent replay
re-dropped every placement one Z step at a time from a regenerated piece stream
and matched each game's board and move digests.

```bash
.venv/bin/python tools/verify_trained_wells.py play --seed 62000001 --cap 1000000 --output runs/trained-wells-1m
.venv/bin/python tools/verify_trained_wells.py replay --seed 62000001 --output runs/trained-wells-1m
```

The benchmark CLI and GUI accept `--policy trained-wells`. Each placement takes
about 1.2 ms on one CPU core. [Training record, screening probe and replay
fingerprints](validation/trained-wells-1m.json). `tools/wells_dashboard.py RUN_DIR` serves a live
page of training and verification progress on port 8775.

### 10,000,000 placements

Median survival across games is also at least 10,000,000 placements. Sixteen
evaluation seeds (63000001 to 63000016) were registered before play, under the
same median rule. Fifteen reached the 10,000,000-placement cap. The exact 95%
lower bound on the chance that a game reaches 10,000,000 is 0.70. Every game,
including the one failure, was replayed independently move by move.

The surviving games cleared 14,643 to 14,645 hyperlayers each. Sampled every
100,000 placements, their stacks never exceeded height 11 and they carried at
most 15 holes.

Seed 63000013 was healthy at 2,700,000 placements and topped out at 2,797,644
on a CROSS. Layer clears stalled on a few open cells at the board's border in
the lowest levels. The stack rose to the ceiling, and only then did holes
cascade. The learned scorer has no feature for completing layers, and training
games of 32,000 placements are far too short to expose a hazard this rare: one
top-out in about 213 million placements, counting six 10,000,000-placement
screening games.

`--engine native` runs the game loop in C (`t4_play_trained_wells`) around the
same policy call, about 6x faster. Over all 1,000,000 placements of seed
61000001 it reproduced the Python loop's moves and board exactly.

```bash
.venv/bin/python tools/verify_trained_wells.py play --engine native --seed 63000001 --cap 10000000 --every 100000 --output runs/trained-wells-10m
.venv/bin/python tools/verify_trained_wells.py replay --seed 63000001 --output runs/trained-wells-10m
```

[10,000,000-placement evaluation, failure analysis and replay fingerprints](validation/trained-wells-10m.json).

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

## Check two-drop CROSS recovery

Given a saved board on which CROSS cannot land, this CPU tool searches for a
non-clearing drop of the current piece followed by a drop of the visible piece
that clears a layer and reopens a CROSS landing. It requires only the standard
library and the snapshot's `board`, `current` and `next_piece` fields. It accepts
evaluator JSON and gzipped snapshots, including `data` and `snapshot` wrappers.

```bash
python tools/find_two_drop_cross.py tests/fixtures/two-drop-cross.json --prefer 8 4 14 4
```

The example reproduces the archived RING and HOOK drops on seed 300040.
Omit `--prefer` for deterministic geometric order, or repeat it to try first
poses in policy order. Output includes each drop's coordinates, cleared layers
and afterboard hash. An empty `moves` array means no pair meets the predicate.

Of 49 candidate boards from 64 historical training games, 24 had a verified
two-drop recovery. On four selected roots played to physical top-out, the
witness gained 791 placements in total over greedy. A one-drop clear rule
reproduced 758 of that gain. [Validation and per-game results](validation/two-drop-cross.json)

On 13 separate historical greedy streams, three contained a two-drop recovery
across 123,799 recorded decisions. From the earliest recovery in each game,
the frozen actor survived another 375, 1,693 and 815 placements. Forcing the
two-drop recovery gave 375, 1,693 and 707. The one-drop rule chose the same
two moves in all three games. On the third board the actor restored CROSS two
moves later, and both lines placed their next CROSS on move 18. All three
opportunities came from historical selection games. [Transfer results and matched trajectories](validation/two-drop-greedy-transfer.json)

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

