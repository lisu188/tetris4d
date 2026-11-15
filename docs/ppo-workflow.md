# Survival PPO workflow

`train_survival_v2.py` is the supported policy-training entrypoint. It optimizes
successful placements before loss with reward 0.001 per placement and gamma 1.
The architecture uses sigmoid activations and has 62,485 parameters at width 32
with three encoder blocks. Completed selection games choose `best.pt`; optimizer
progress or loss curves alone do not establish improved survival.

## Inputs and compatibility

The small bundled `models/neural-rl.pt` checkpoint provides initial actor weights.
Those weights originated in an earlier A2C clear-reward run. They are legitimate
PPO initialization; PPO does not require actor weights trained with PPO. Importing
them does not make that file a PPO-trained result. A new survival run reinitializes
and calibrates its critic because expected remaining placements and discounted
clears use different units.

The trainer uses `models/trained-sigmoid.json` to construct candidate teacher
features and priors. It imports the model and candidate code in `tetris_neural.py`,
shared run/checkpoint/replay infrastructure in `train_overnight.py`, and PPO loss
helpers in `train_survival.py`. These source modules remain necessary runtime
dependencies. Their compatibility tests remain part of validation. Older training
documents remain historical implementation references; the active workflow is PPO.

Keep the imported checkpoint immutable and retain its SHA256. A stronger frozen
incumbent from a completed comparison can replace the bootstrap path below.
Changing policy mode or prior scale changes behavior even if every imported actor
tensor is preserved; measure the converted policy separately.

## Eight-hour experiment

Install the neural requirements and build the native scorer as described in the
[README](../README.md). CUDA is optional; use `--device cpu` for a CPU run. Begin
in a new directory with enough free space for checkpoints, metrics and recordings.

```bash
.venv/bin/python -u train_survival_v2.py \
  --hours 8 --run-dir runs/survival-longevity \
  --checkpoint models/neural-rl.pt \
  --device cuda --backend native --threads 4 --precision fp32 \
  --policy-mode teacher-prior --teacher-prior-scale 0.125 \
  --residual-scale 1 --preserve-initial-actor \
  --gradient-clip-scope separate \
  --imitation 0 --secondary-weight 0 --regret-weight 0 \
  --warmup-updates 0 --rollout 512 \
  --seed 250000 --calibration-seed 3300000 \
  --selection-seeds 951001 951003 951005 951007 \
  --pieces 32000 --selection-pieces 32000 \
  --record-games --record-every 4 --record-max-mib 128
```

The `teacher-prior` experiment uses
`logits = 0.125 * teacher_prior + actor_logits`. Its actor can overcome immediate
teacher preferences. The bounded `teacher-residual` mode remains available; the
new recipe is an experimental setting, rather than a proven strength improvement.
See [the exploration and gradient changes](survival-longevity.md).

Calibration, initial-policy measurement and later selection consume the same
eight-hour budget. The production calibration gate remains enabled. An
uncalibrated smoke test belongs in a separate disposable run; do not transfer its
weights into this experiment or bypass a failed production gate.

The trainer saves `last.pt`, `previous.pt`, `best.pt` when selected, `summary.json`,
`manifest.json`, `critic-calibration.json`, `metrics.jsonl` and recorded games.
`best.pt` means best completed selection result in this run. It does not imply
that the model beat an independently evaluated incumbent or sustained planner.

## Interrupt and resume

Run under your normal persistent terminal/session manager for long training. A
first Ctrl+C or termination signal asks the runner to save at a safe boundary.
Keep the same run directory and source/configuration when resuming. The runner
checks algorithm and source signatures and rejects an incompatible resume.

Each invocation grants its specified `--hours` anew. Record the original UTC
deadline before starting; after interruption, resume with the remaining hours.
For example, after two hours of an eight-hour allocation, use `--hours 6` with
the same other arguments. Exact resume restores the saved actor, critic,
optimizer, environment streams and calibration state; it does not import the
bootstrap again. A settings change starts a new experiment in a new directory.

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
regression tests and fixtures, small validation reports, and the two required
model inputs. In particular, `tests/fixtures/blocked-targets.json` remains a
recovery-regression fixture. Rebuilding it with
`tools/build_recovery_fixtures.py` requires the corresponding archived recorded
games; normal tests use the existing fixture without those runs.

The [archive catalog](../validation/archive-manifest.json) records the source
commit, path, Git blob ID and byte size for every removed tracked artifact. It
describes the committed bytes, not newer dirty or untracked local experiments.
Those original local files remain preserved separately. Removing index entries
does not rewrite Git history or delete a live run.

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
