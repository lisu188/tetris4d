# Tetris 4D

Train a sigmoid policy with PPO to maximize successful placements in a
**16 × 16 × 16 × 16** falling-block game. The repository includes the rigid-body
engine, native CPU scorer, Python/Tkinter interface, survival trainer, checkpoint
comparison and a neuron-by-neuron audit.

The supported training workflow is [survival PPO](docs/ppo-workflow.md).
`train_survival_v2.py` calibrates the survival critic, trains with clipped PPO and
selects checkpoints from completed games. A frozen model's measured survival,
including comparison with the sustained planners, determines its playing strength.

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

Start a new eight-hour PPO run from the bundled bootstrap weights:

```bash
.venv/bin/python -u train_survival_v2.py \
  --hours 8 --run-dir runs/survival-ppo-v2 \
  --checkpoint models/neural-rl.pt \
  --device cuda --backend native --record-games --record-max-mib 128
```

Use `--device cpu` where CUDA is unavailable. Calibration and selection are part
of the time budget. Resume the same directory with the time remaining from the
original deadline; a new invocation grants the specified hours again. Changed
source or algorithm settings require a separate run.

`models/neural-rl.pt` contains older A2C actor weights and is retained as a PPO
bootstrap, rather than a newly trained PPO result. Transfer resets the critic
because its previous clear-reward estimate has different units. The teacher
checkpoint `models/trained-sigmoid.json` also remains a runtime input.

For the current broader-exploration recipe, fresh seed ranges, calibration checks
and bounded recording settings, use [the PPO workflow](docs/ppo-workflow.md) and
[experiment details](docs/survival-longevity.md). The model has 62,485 parameters;
training does not change placement physics.

## Play a checkpoint

```bash
.venv/bin/python sparse_tetris_4d_optimized.py \
  --backend native \
  --neural-checkpoint runs/survival-ppo-v2/best.pt --neural-device cuda
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

Always supply checkpoint paths explicitly. This example evaluates the bootstrap
and a completed PPO checkpoint against both sustained baselines:

```bash
.venv/bin/python tools/survival_experiments.py compare \
  --device cuda --backend native --threads 4 \
  --checkpoints models/neural-rl.pt runs/survival-ppo-v2/best.pt \
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
.venv/bin/python tests/test_survival_v2.py
.venv/bin/python tests/test_survival_longevity.py
.venv/bin/python tests/test_survival_ppo.py
.venv/bin/python tests/test_native.py --trajectory-pieces 2600
```

PPO imports shared network, replay and trainer infrastructure from
`tetris_neural.py`, `train_overnight.py` and `train_survival.py`. Their compatibility
tests remain relevant even though the public training workflow uses PPO v2.

## Game rules and repository contents

Gravity acts along negative Z; X, Y and W are horizontal. Pieces rotate in six
coordinate planes and drop rigidly. A clear requires all 4,096 X/Y/W cells at one
Z level; higher layers shift down after a clear. There are no independent-cell
gravity, wall kicks, hold, under-overhang slides or timed-input constraints.
The expanded piece set has 15 shapes; `--piece-set classic` selects seven.

The checkout keeps runtime source, tests and fixtures, small validation reports,
the PPO bootstrap and teacher model. Historical training outputs, demonstrations
and benchmark logs are listed in the [archive catalog](validation/archive-manifest.json).
Git history preserves their committed versions. Existing dirty and untracked
local experiments are preserved separately; they can differ from the cataloged
historical blobs. See [storage and recovery](docs/ppo-workflow.md).
