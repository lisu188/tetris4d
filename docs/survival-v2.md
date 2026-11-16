# Survival training v2

This implements the November 12 log-audit changes. It is a new experiment, not a claim that a learned policy has already beaten sustained. Native C, rigid physics, all 16 section views, Unlimited playback defaults, trained model files and committed historical runs are preserved.

## Start an eight-hour run in WSL

Stop older training processes gracefully first. The v2 CUDA launcher checks for other Tetris trainers and holds a per-device cooperative lock. It never terminates another process. `--allow-concurrent` is an explicit override; competing processes are recorded and concurrent timing is not an isolated benchmark.

```bash
cd ~/git/tetris4d && git switch main && git pull --ff-only && mkdir -p runs/survival-ppo-v2 && (nohup .venv/bin/python -u train_survival_v2.py --hours 8 --device cuda --record-games >> runs/survival-ppo-v2/console.log 2>&1 < /dev/null &)
```

The same command resumes the v2 run, including an interrupted calibration. Windows must remain awake. The eight-hour budget includes calibration and evaluation; stopping is cooperative at safe boundaries. Do not use an old v1 run directory. The new algorithm/source signatures intentionally reject silent resume under changed semantics. Old checkpoints remain loadable for evaluation; resume an old experiment with its original source revision in a separate checkout.

```bash
tail -f ~/git/tetris4d/runs/survival-ppo-v2/console.log
```

## What is different

### Target recovery

The new policy tries the original target first, then other accessible levels by increasing distance. Bounds, rigid landing, collision rules and compaction are unchanged. Retargeting does not earn reward, place a piece, reset a game or advance its piece stream. Recovery is logged and included in replay/checkpoint state. The 35 previously recorded blocked boards are regression fixtures, reconstructed from the committed game records. Legacy checkpoints default to recovery disabled; v2 enables it. Sustained remains unchanged as a baseline, and the comparison tool includes a separate sustained-recovery baseline.

### Frozen-policy critic calibration

Before PPO, freeze the actor and collect four training games plus two distinct validation games under its sampled policy. State features are sampled every 32 placements. Only uncensored completed episodes provide exact remaining-placement targets; capped games are not treated as deaths. The critic uses centered sigmoid hidden activations and a linear output. A ridge fit initializes its output on training data, followed by up to 300 head-only AdamW updates. The actor and shared encoder are checked unchanged.

Validation selects the critic head. Its MSE must beat the training-mean constant baseline by at least 5 percent, and its near-end mean absolute error must improve where near-end samples exist. Otherwise the run saves and stops with calibration_failed, without proceeding to actor optimization. This gate is a small validation experiment, not proof of calibrated lifetime predictions on every board. `--allow-uncalibrated` explicitly bypasses a failed gate; `--calibration-games 0 --allow-uncalibrated` is intended for smoke tests or controlled ablations, not the recommended training command.

Calibration progress, data, current episode, optimizer, best critic weights and RNG state are resumable. Completed calibration produces critic-calibration.json. The value unit remains placements/1000. No additional clear or speed bonus is introduced; gamma remains 1. Capped training episodes bootstrap their final value without propagating advantages across reset boundaries.

### Teacher-guided residual policy

The default v2 policy is a hybrid, explicitly labeled teacher-residual. It retains the imported encoder and hidden ranking layers but zero-initializes the final learned correction. Its prior preserves the exact teacher's primary and secondary ordering within the shortlist, including enumeration-order ties. With zero correction, greedy selection matches the teacher. A bounded sigmoid correction can rerank nearby alternatives but cannot jump over the strong primary-utility penalty in greedy mode. Sampling is still stochastic and is not claimed identical to greedy teacher play.

Secondary teacher supervision retains preference information lost by primary-utility-only tie supervision. `--policy-mode neural` keeps an unconstrained neural actor for a separate ablation; it is not silently substituted into this default hybrid experiment. Existing original checkpoint files are not overwritten. No ReLU, GELU or SiLU is introduced, and the parameter count remains 62,485 for the standard model.

### PPO and diagnostics

`--batch 16` is the memory microbatch. `--optimizer-batch 128` is the optimizer minibatch. A default 512-placement rollout (four environments, 128 steps) permits up to twelve optimizer steps across three epochs, rather than three whole-rollout steps. The final partial minibatch uses its actual size. KL is checked after each minibatch and over the full rollout at epoch boundaries and the end. The threshold stops later updates; it is not a mathematical guarantee that a just-completed update cannot cross it.

Actor and critic use separate optimizer groups and learning rates. Critic inputs are detached from the shared encoder, so critic gradients cannot change actor features. MSE is the default critic loss; `--value-loss huber --value-beta 1` is available for a controlled comparison. Per-loss gradient norms are sampled every ten rollouts. Logs include critic target/prediction errors in pieces, explained variance, advantage distributions, terminal and truncated transition counts, primary-utility agreement, exact teacher agreement, secondary loss, real optimizer-step counts and phase timings.

The collector reuses adjacent-state values inside each frozen-policy rollout. It computes separate bootstrap values at truncations and the final boundary; terminal states bootstrap zero. It does not reuse stale predictions across optimizer updates. Use separate run directories to compare `--rollout 128`, `--rollout 512` and `--rollout 1024` rather than changing a resumed run.

### Headless inference and GUI

Headless evaluation disables activation capture and CPU trace transfers. The GUI captures the most recent decision's actual activations lazily when a frame needs the learned tab. All 16 section boards remain visible, with render-only frame skipping. Neural decisions now participate in benchmark decision counters. No GPU throughput improvement is claimed until measured in isolation.

### Files and archives

New output stays in runs/survival-ppo-v2: last.pt, previous.pt, best.pt, summary.json, critic-calibration.json, manifest.json and metrics.jsonl. A manifest ties the checkpoint checksum to source/configuration hashes and a metrics byte offset. Best-checkpoint selection remains separate from automatic deployment.

With --record-games, active prefixes are JSON and completed games are compressed JSON (.json.gz). The verifier accepts both:

```bash
.venv/bin/python train_survival_v2.py --verify-game path/to/game.json.gz
```

A recording quota failure disables recording, not training. A stopped run can be exported without deleting or changing its originals:

```bash
.venv/bin/python tools/survival_experiments.py archive --run-dir runs/survival-ppo-v2 --output experiments/survival-v2-run1.tar.gz --include-games
```

Archive creation requires the run lock, refuses an existing destination, and records file hashes. Preserve historical data; no git history rewrite is required. Prefer immutable archives to repeated live-run commits.

## Compare frozen checkpoints on common seeds

```bash
.venv/bin/python tools/survival_experiments.py compare --device cuda --checkpoints models/neural-rl.pt runs/overnight-survival/best.pt runs/survival-ppo/best.pt runs/survival-ppo-v2/best.pt --seeds 94001 94003 94005 94007 --pieces 32000 --output survival-comparison-v2.json
```

This snapshots the checkpoints by checksum, compares legacy sustained and sustained-recovery, and evaluates greedy and sampled neural execution separately. Sampled defaults use action seeds 101, 102 and 103 for each game seed. `--modes greedy` is a smaller comparison. Output is updated after every completed game; an interrupted report remains marked incomplete. Existing outputs are not overwritten. These seeds should not become training or selection seeds. A later final test needs further unused seeds.

## Frozen-initializer KL experiment and value audit

`--reference-kl-weight BETA --reference-checkpoint PATH` adds the fixed penalty
`BETA * KL(frozen_reference || current_policy)` to the minimized PPO objective.
Both distributions use the same state tensors, candidate order, teacher prior and
valid-candidate mask. The reference is an immutable policy snapshot, not teacher
action labels. Its parameters receive no gradients and never enter the optimizer.
The default weight is zero and adds no model, RNG draws, losses or checkpoint
fields. Model architecture and neuron inventory are unchanged. Source signatures
still require a new run directory when changing trainer code.

The positive weight and explicit reference checkpoint must be supplied together.
Keep the reference file immutable for the run: its checksum is part of the run
signature. Each training checkpoint also embeds the frozen weights, configuration
and actor digest; resume validates them and restores the embedded state. Candidate
or actor configuration mismatches are rejected. `reference_kl` is the raw mean
measured during optimization, `final_reference_kl` measures the full rollout after
optimization, and `reference_kl_weight` records the fixed coefficient. The existing
`final_exact_kl` remains the distinct behavior-policy PPO check. Reference KL is a
regularizer, not a guarantee against survival regression.

Start a separate controlled arm from a checksum-protected best checkpoint with
fresh optimizers, sixteen environments, rollout 512 and the existing GPU flags:

```bash
.venv/bin/python -u train_survival_v2.py \
  --run-dir runs/reference-kl-pilot --checkpoint /absolute/protected-best.pt \
  --reference-checkpoint /absolute/protected-best.pt --reference-kl-weight 0.1 \
  --device cuda --backend native --envs 16 --rollout 512 \
  --batch 16 --optimizer-batch 128 --lr 0.0001 --critic-lr 0.0003 \
  --policy-mode teacher-prior --teacher-prior-scale 0.125 --residual-scale 1 \
  --preserve-initial-actor --gradient-clip-scope separate --recover-target \
  --imitation 0 --secondary-weight 0 --regret-weight 0 --entropy 0.001 \
  --gae-lambda 0.999 --ppo-epochs 3 --clip-ratio 0.2 --target-kl 0.02 \
  --gpu-prepare --fused-encoding --coalesce-transfers --cache-rollout \
  --rollout-cache-mib 1024 --max-updates 64 --hours REMAINING_HOURS
```

This example assumes the protected best has exactly the stated actor/candidate
configuration. The fresh run resets and recalibrates the critic under the frozen
actor; that time counts against its budget. A supervisor must enforce the existing
absolute deadline (for the current experiment, 2026-11-16 18:13:32.190616 UTC).
Calculate `REMAINING_HOURS` when launching; do not restart the 24-hour clock. The
existing optimized continuation is retained as a control and recovery point.
Use the same placement budget and multiple independent training seeds when time
permits, compare paired development games, and reserve the final test seeds.
Do not increase rollout length or change the encoder in this first ablation.

The opt-in frozen-checkpoint comparison now collects bounded scalar value samples:

```bash
.venv/bin/python tools/survival_experiments.py compare \
  --checkpoints /absolute/protected-best.pt /absolute/frozen-pilot.pt \
  --seeds 955001 955003 --modes greedy sampled --action-seeds 101 102 \
  --pieces 32000 --device cuda --backend native --value-audit \
  --value-audit-stride 128 --value-audit-samples 256 \
  --output value-audit.json
```

Checkpoints are copied to checksum-addressed immutable snapshots and verified
again after comparison. Only checkpoints declaring survival-value units are
accepted for this audit. Sampled actions use a separate fixed generator; collecting
diagnostics does not change the trajectory. A completed uncensored game labels a
pre-move state with `(final_pieces - sampled_piece) * 0.001`; capped games retain
predictions with null targets and never contribute lifetime errors. Reports keep
greedy and sampled continuations separate and include per-game scalar samples,
MAE, RMSE and explained variance, bounded to at most 1024 samples per game.

The sampled audit tests the critic's training-policy objective. Greedy continuations
are an off-policy diagnostic for that critic. Each state has one realized
continuation, values are noisy, and states within a game are correlated; use
several games/action seeds before drawing conclusions. The training log's
`critic_bootstrapped_diagnostics` still compares predictions with finite-rollout
GAE targets, not realized full-episode lifetimes. An independent critic encoder
and longer rollouts remain later ablations, guided by this audit.

For a bounded paired pilot, `tools/run_reference_ablation.py` supervises baseline
and anchored arms sequentially while existing production training continues.
Both start from one protected initializer with fresh optimizers, the same training
and calibration seeds, sixteen environments and 524,288 placements per arm.
The fixed pilot coefficient is 0.1. Each arm has a three-hour limit within the
original absolute deadline; incomplete arms remain failed experiments. This
single-training-seed pilot does not automatically promote a policy.

Commit the source first, then provide `--production-artifacts`, a separate
`--output`, `--budget-ledger`, `--checkpoint`, `--weights` and the existing venv
`--python`. Without `--run` it prints the frozen plan. `--run --evaluate` also
compares initializer and final arm checkpoints on common development seeds in
greedy and sampled modes, with the lifetime audit. Reserved final-test seeds are
rejected. Recording is disabled; metrics, checkpoints and comparison data remain.
The supervisor checks both disks, a 384 MiB aggregate output bound, a private
32 MiB Triton cache and atomic checkpoint reserves. CUDA children require
3.5 GiB free initially and use a 3 GiB PyTorch allocator limit. Concurrent timing
is recorded and must not be reported as an isolated speed benchmark.

## Validation scope

The tests cover the 35 recovery states, zero-residual teacher choices, exact CPU resume, calibration actor isolation, censored games, optimizer minibatches, terminal/truncated bootstrapping, compressed replay, headless/lazy trace equivalence, neural logging and immutable archives. CUDA tests explicitly skip on CPU-only workers. Calibration tests using synthetic features establish implementation behavior, not Tetris playing strength. Longer survival and CUDA speedups require the user's paired evaluation.

## Survival exploration experiment

The opt-in `teacher-prior` mode supports scaled teacher guidance and an
unbounded linear actor correction, actor-preserving initialization and separate
actor/critic gradient clipping. See [the survival experiment](survival-longevity.md)
for the eight-hour configuration, diagnostics and independent test seeds.
