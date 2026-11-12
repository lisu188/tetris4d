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

## Validation scope

The tests cover the 35 recovery states, zero-residual teacher choices, exact CPU resume, calibration actor isolation, censored games, optimizer minibatches, terminal/truncated bootstrapping, compressed replay, headless/lazy trace equivalence, neural logging and immutable archives. CUDA tests explicitly skip on CPU-only workers. Calibration tests using synthetic features establish implementation behavior, not Tetris playing strength. Longer survival and CUDA speedups require the user's paired evaluation.
