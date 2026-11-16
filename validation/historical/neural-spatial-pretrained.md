> HISTORICAL RECORD ONLY. Original V4 model files and bundle were not recovered; links below may point to missing artifacts. This is not documentation of newly trained weights.

# Spatial V4 pretrained checkpoint

`neural-spatial-pretrained.pt` is an experimental 77,870-parameter actor/critic
checkpoint produced on CPU on 2026-11-16. The actor was pretrained by supervised
teacher imitation; the critic was calibrated to complete continuations of the
embedded sustained teacher with target recovery. It has **zero PPO updates**.
The default game controller remains sustained.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| [Combined actor and critic](neural-spatial-pretrained.pt) | `fc35a53cf1ed45d729c5e5284b89e0bc9c3f5a53c2c0c268f079a17751a5fb6b` | 340,435 |
| [Actor-only stage](neural-spatial-actor.pt) | `ad759a0624968563f94a3693902f3f3a1f2c04cf3c97ab090adb732b4ea2629b` | 334,483 |

Both artifacts have the same actor parameter digest:
`2097c698abd4ebb87dea3d48c783d1406f837c80485c8bb9fc535dc78aa1d01a`.
Critic fitting did not change actor parameters, logits, candidate configuration
or the embedded teacher. The final critic digest is
`408dd969529c1fc62a05bc698a9c057e4e830b2268be17923187d338c1849f83`.

## Configuration

The architecture is `survival-spatial-v4`, with the 18-feature spatial actor
enabled, 156 critic inputs, `teacher-prior` policy mode, prior scale 0.125,
residual scale 1, 2,048 candidate budget, 128 retained candidates, the original
`teacher-random` shortlist, and target recovery enabled. It uses the bundled
`trained-sigmoid.json` teacher weights and the expanded piece set.

The six board-adjacency features and zero-initialized actor extension are
described in [Spatial V4](../docs/survival-v4.md). CPU validation used Python
3.12, PyTorch 2.10.0+cpu, NumPy 2.3.5 and the native scorer. This work did not
measure CUDA behavior or speed.

## Actor pretraining

Four training seeds (6100002, 6100004, 6100006, 6100008) supplied 1,024 snapshots.
Two separate selection seeds (6200001, 6200003) supplied 256 snapshots. Collection
retained one state every eight placements, covering 10,198 actual teacher
placements. The critic stayed frozen.

The loss was KL from a soft teacher distribution at temperature 8 to the raw
actor distribution. The explicit runtime prior was excluded from the loss.
AdamW ran 300 updates at learning rate 0.001, batch 32; selection chose update
300. Collection and fitting took 213.0 seconds in total.

| Selection-set measure | Initial actor | Selected actor |
| --- | ---: | ---: |
| Raw actor KL to teacher distribution | 1.30227 | 0.03780 |
| Raw exact teacher-choice agreement | 1.17% | 61.72% |
| Raw primary-utility agreement | 7.42% | 100% |
| Raw mean primary-utility regret | 16.355 | 0 |
| Prior-plus-actor exact teacher-choice agreement | 100% | 67.19% |
| Prior-plus-actor primary-utility agreement | 100% | 100% |

These are imitation metrics. The initial prior already chose the teacher's top
candidate. Training changes secondary ordering among placements with equal
primary utility, so exact runtime teacher agreement decreased. A lower imitation
loss does not demonstrate better survival.

The [actor manifest](../validation/spatial-pretraining.json) records the complete
recipe and measurements. The [source archive](../validation/spatial-pretraining-source.zip)
preserves the exact training Python/C and teacher JSON.

## Critic precalibration and independent audit

Training used four complete teacher games; selection used two other complete
games. Every retained label is the actual number of subsequent successful
placements before natural termination, multiplied by 0.001. Capped or interrupted
games contribute no exact-lifetime labels.

| Role | Seeds | Episode placements | Snapshots |
| --- | --- | --- | ---: |
| Training | 7200000, 7200002, 7200004, 7200006 | 15,201; 17,713; 14,275; 22,461 | 546 |
| Selection | 7300000, 7300002 | 13,398; 15,191 | 224 |
| Untouched audit | 7400001, 7400003 | 15,972; 17,031 | 259 |

All eight games ended by natural top-out. The training/selection collection used
98,239 placements and took 426.4 seconds with two CPU workers. The independent
audit added 33,003 placements and took 139.9 seconds to collect.

Adam trained only the critic for 2,000 updates, at learning rate 0.001 and batch
128, with seed 20261117. Selection chose update 225. The criterion minimized
global selection MSE among checkpoints that beat a fixed training-mean baseline
and have near-end MAE at most 1,000 placements. Near-end means at most 1,000 true
remaining placements.

| Critic measure | Selection | Untouched audit |
| --- | ---: | ---: |
| Mean absolute error, placements | 2,757 | 1,492 |
| Constant baseline MAE, placements | 3,848 | 4,187 |
| MSE / constant baseline MSE | 0.5792 | 0.1559 |
| Near-end MAE, placements | 402 | 523 |
| Constant near-end MAE, placements | 8,494 | 8,527 |
| Near-end snapshots | 16 | 16 |

The first unconstrained selection minimized average MSE but produced a 4,927-
placement near-end error. It was rejected before the independent audit. Its
[historical report](../validation/spatial-precalibration-unconstrained.json)
is retained with that rejection explicit. No fitting or selection followed the
independent audit.

The critic output is **teacher-policy remaining placements × 0.001**. It does
not estimate a calibrated lifetime under a different learned policy. Its mean
signed audit error was +1,377 placements, so a tendency to overestimate remains.
Two independent audit games are a small sample; snapshots within a game are
correlated.

The [fit report](../validation/spatial-precalibration.json),
[audit report](../validation/spatial-critic-audit.json),
[fit dataset](../validation/spatial-critic-dataset.npz), and
[audit dataset](../validation/spatial-critic-audit-dataset.npz) preserve exact
targets, completed action streams, source hashes and split provenance.

## Gameplay and late-state checks

The [matched-control gameplay smoke](../validation/spatial-pretrained-smoke.json)
contains 36 games across four held-out seeds (8100101, 8100103, 8100105, 8100107),
greedy and sampled policies, and teacher/prior controls. Every game reached the
512-piece cap, with no clears. All capped-survival differences were zero. This
verifies a functioning inference/evaluation pipeline; it does not rank long-term
playing strength or justify replacing the sustained default.

The [late-state candidate audit](../validation/spatial-pretrained-candidates.json)
checked four replay-verified states at placements 6,656, 12,032, 7,552 and 13,696.
The full prior-plus-actor policy matched the teacher's primary utility on all
four. The raw actor did so on three: on one late board it selected utility 0
while the teacher selected 2; the retained prior restored utility 2. The actor
therefore still depends on its prior, especially beyond its early-game imitation
data. This four-state audit is diagnostic, not a generalization estimate.

The exact bank is retained in
[compressed form](../validation/spatial-pretrained-candidate-states.json.gz).
The measured actor-only artifact and final combined artifact are linked through
identical actor/config/teacher hashes; the reports keep the original measured
checkpoint identity explicit.

## Use and reproduce

Play the combined checkpoint:

```sh
python sparse_tetris_4d_optimized.py --backend native \
  --neural-checkpoint models/neural-spatial-pretrained.pt --neural-device cpu
```

Start a new PPO run with these weights:

```sh
python -B train_survival_v4.py --pretrained models/neural-spatial-pretrained.pt \
  --run-dir runs/spatial-pretrained-ppo --device cpu --backend native \
  --hours 8 --cache-rollout
```

This creates a new optimizer and fresh PPO progress. It records imported
supervised provenance and, by default, recalibrates the critic under the actual
starting policy before PPO. The explicit experimental `--allow-uncalibrated`
override remains available. Use a new run directory when changing source,
configuration or initializer content.

Repeat the read-only critic audit on the retained dataset:

```sh
python tools/precalibrate_spatial.py audit \
  --dataset validation/spatial-critic-audit-dataset.npz \
  --checkpoint models/neural-spatial-pretrained.pt \
  --report runs/spatial-audit-repeat.json --threads 1
```

Recollect or refit using `tools/precalibrate_spatial.py collect`, `fit`, and
`audit`; reproduce actor pretraining with `tools/pretrain_spatial.py`. Choose new
output paths. Exact executed source and hashes are retained in the manifests and
source archives; current tools include subsequent provenance/inspection fixes.