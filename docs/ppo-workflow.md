# Spatial V4 PPO workflow

`train_survival_v4.py` maximizes successful placements before loss with a reward
of 0.001 per placement and gamma 1. The current model uses a spatial actor and a
regional critic with 102,830 parameters. Completed greedy selection games choose
`best.pt`. A separate paired evaluation compares the selected model with the
frozen incumbent.

## Inputs

`models/trained-sigmoid.json` supplies candidate features and teacher logits.
The learned checkpoint is a separate file. Use a calibrated Spatial V4
checkpoint with the same screening, shortlist, policy mode, prior scale, target
recovery and critic layout as the new run. Preserve its SHA-256 with the results.
For a first checkpoint, see [V4 pretraining and calibration](survival-v4.md).

## Eight-hour experiment

Install the neural requirements and build the native scorer as described in the
[README](../README.md). Begin in a new run directory. Use the recorded checkpoint
configuration and fresh training and selection seed ranges. The command below
uses the regional exhaustive teacher-prior configuration.

```bash
.venv/bin/python -u train_survival_v4.py \
  --hours 8 --run-dir runs/survival-ppo-v4 \
  --pretrained /path/to/calibrated-v4.pt --preserve-pretrained-critic \
  --calibration-games 0 --allow-uncalibrated \
  --backend native --device cuda --screening exhaustive --regional-critic \
  --recover-target --policy-mode teacher-prior --teacher-prior-scale 0.125 \
  --fixed-lr --envs 32 --rollout 256 --pieces 0 --selection-pieces 64000 \
  --seed 250000 --selection-seeds 951001 951003 951005 951007
```

`--preserve-pretrained-critic` imports an already calibrated head and starts with
a fresh PPO optimizer. For a newly pretrained actor, omit `--preserve-pretrained-critic`,
`--calibration-games 0` and `--allow-uncalibrated` to run survival critic
calibration. Use `--device cpu` for CPU training. If enough top-out games are
in the local library, `--danger-starts 0.25` starts a quarter of training
episodes from recorded late-game positions.

Initial measurement, scheduled evaluation and checkpoint writing consume part
of the eight-hour budget. `--pieces 0` lets training games continue through
top-out. Development still uses the positive `--selection-pieces` cap. The run
keeps `last.pt`, `previous.pt`, selected `best.pt`, a manifest, metrics and a
summary. `best.pt` ranks complete development games and can still be the imported
policy at update zero.

## Interrupt and resume

Run under your normal persistent terminal/session manager for long training. A
first Ctrl+C or termination signal asks the runner to save at a safe boundary.
Keep the same run directory and source/configuration when resuming. The runner
checks algorithm and source signatures and rejects an incompatible resume.

Each invocation grants its specified `--hours` anew. Record the original UTC
deadline before starting; after interruption, resume with the remaining hours.
For example, after two hours of an eight-hour allocation, use `--hours 6` with
the same other arguments. Exact resume restores the saved actor, critic,
optimizer, environment streams and calibration state from the latest save. A settings change starts a new experiment in a new directory.

Diagnostics include actor/critic gradient norms, actor hashes, candidate-score
variation, sigmoid saturation and teacher-objective deviations. These distinguish
learning activity from collapse or inactivity; strength still requires completed
games.

## Evaluate and choose the surviving model

Freeze the candidate and incumbent before the final comparison. Reserve board
seeds independently of training, critic calibration and checkpoint selection.
`951001 951003 951005 951007` above are selection seeds. The following example uses
`960001 960003 960005 960007` as final test seeds; use another untouched set if
those have already influenced an experiment.

```bash
.venv/bin/python tools/survival_experiments.py compare \
  --device cuda --backend native --threads 4 \
  --checkpoints /absolute/path/to/frozen-incumbent.pt \
                /absolute/path/to/frozen-ppo-best.pt \
  --seeds 960001 960003 960005 960007 --pieces 32000 \
  --modes greedy sampled --action-seeds 101 102 103 \
  --record-games --record-max-mib 128 \
  --output survival-comparison-ppo-heldout.json
```

Both sustained and sustained-recovery baselines are included. The tool snapshots
checkpoint inputs by hash and updates the comparison after each game. Existing
report paths are never overwritten. A partial report remains incomplete and is
not a completed comparison. The CLI requires explicit checkpoint paths;
historical local run directories are absent from a compact checkout.

Keep greedy and sampled execution separate. For sampled execution, first average
the repeated action seeds within each board seed; they are not independent board
samples. Report placements, paired differences, stop reasons, capped games,
hyperlayer clears and target recoveries. A cap is a lower bound, and a small
finite comparison cannot identify a universally longest-surviving policy.
Preserve the incumbent if the experiment does not improve the relevant measured
survival. Retain input/source hashes and actual device/library versions with the
result.

## Play and inspect the frozen model

```bash
.venv/bin/python sparse_tetris_4d_optimized.py \
  --backend native \
  --neural-checkpoint /absolute/path/to/frozen-ppo-best.pt --neural-device cuda
```

An explicit checkpoint activates neural play; normal no-checkpoint play retains
the sustained planner. The GUI uses greedy decisions. Its value and auxiliary
predictions have checkpoint-specific semantics, described in the
[neuron audit](neuron-analysis.md). That CPU-only utility produces one entry for
every shared channel and dense unit, with exact computations and sample-specific
activation/intervention evidence. Use the final immutable policy and completed
recordings with matching engine source and teacher parameters.

## Bounded local storage and historical recovery

Keep runs, demonstration shards, logs, dated model inspections and full comparison
recordings outside versioned source. A 128 MiB recording quota bounds game files;
checkpoint and metrics files use additional space. Avoid copying the environment,
native inputs or old run trees for every experiment. Preserve source revisions,
compact validation evidence, frozen deliverables and any diagnostics needed to
understand an unresolved failure.

The compact checkout preserves engine/UI/native source, PPO dependencies,
regression tests and fixtures, small validation reports, and the bundled
teacher input. In particular, `tests/fixtures/blocked-targets.json` remains a
recovery-regression fixture. Rebuilding it with
`tools/build_recovery_fixtures.py` requires the corresponding archived recorded
games; normal tests use the existing fixture without those runs.

The [archive catalog](../validation/archive-manifest.json) lists the source
commit, Git blob ID and byte size of historical tracked artifacts. The legacy A2C
bootstrap at `models/neural-rl.pt` is recoverable from
`23159fe8d860a13ab16b8d5b69f2abf868522c82` using the same Git command below.
Use each superseded trainer's original revision and config for exact replay.

Restore an exact historical blob into a new destination, never over a live run.
For example, these commands recover the cataloged PPO checkpoint from its source
revision without modifying the current checkout:

```bash
mkdir -p /absolute/path/to/new-recovery
(
  set -o noclobber
  git show 4617ca4e181800735ba702f166f69661b88edaeb:runs/survival-ppo/best.pt \
    > /absolute/path/to/new-recovery/historical-ppo-best.pt
)
git hash-object /absolute/path/to/new-recovery/historical-ppo-best.pt
```

Compare the reported blob ID to the catalog entry for that exact path. The Bash
`noclobber` setting refuses an existing destination. A shallow clone may need the
source revision first:

```bash
git fetch origin 4617ca4e181800735ba702f166f69661b88edaeb
```

A full clone still contains historical Git objects. A smaller checkout alone
does not remove that history or reclaim Windows backing-volume allocation. The
catalog and unchanged history provide recovery without copying large archived
trees back into active source.
