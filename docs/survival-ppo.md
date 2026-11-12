# Survival-first PPO: weight-audit follow-up

Status: implementation and correctness checks, not a pretrained stronger policy. The committed model tensors, native C engine, rigid shapes, all 16 views, sigmoid architecture, and Unlimited/sustained GUI defaults are unchanged.

## Run under WSL

Keep the existing working CUDA PyTorch environment. No driver changes or native rebuild are needed. Run inside the repository:

```bash
.venv/bin/python -u train_survival.py --hours 8 --device cuda --record-games
```

For detached execution, including update from main:

```bash
cd ~/git/tetris4d && git switch main && git pull --ff-only && mkdir -p runs/survival-ppo && (nohup .venv/bin/python -u train_survival.py --hours 8 --device cuda --record-games >> runs/survival-ppo/console.log 2>&1 < /dev/null &)
```

The same command resumes the same run. Keep Windows awake; nohup prevents terminal hangup from ending the process but does not prevent system sleep. The time budget includes initial/periodic evaluations. Stopping is cooperative at safe boundaries, not an exact wall-clock kill.

The new default run directory is `runs/survival-ppo`; do not point it at a legacy A2C run. Both entrypoints reject algorithm/settings mismatches rather than silently changing a resumed experiment. To change settings, start another run directory. Existing source/engine/checkpoint signature checks remain active.

## Changes motivated by the audit

1. Prefer `models/neural-rl.pt` as initialization, then `models/neural-policy.pt`, otherwise a new model. `--checkpoint` overrides this. More updates do not prove that RL is stronger: the initial checkpoint is evaluated before optimization, with its exact file hash recorded. Original files are never overwritten. The critic output is reset on transfer because a clear-return estimate has different units; exact resume restores the already trained survival critic.
2. Preserve the explicit survival reward: one successful placement, internally scaled by 0.001, gamma=1. No clear or speed bonus. A game continues across clears and 128-step rollout boundaries. Truncation at the 32,000-piece episode cap bootstraps; top-out and target-planner dead ends terminate and remain distinct in reports. This is capped-policy survival, not a proof that the game can never be lost.
3. Replace one on-policy gradient pass with up to three clipped PPO full-rollout passes. The sampled behaviour log-probabilities are recorded at action selection time, not recomputed after learning. Each pass accumulates gradients in 16-board minibatches before one optimizer step. Default clip ratio 0.2. Exact masked categorical KL over the whole rollout is measured before each pass; subsequent passes stop above 0.02. This is an early-stop safeguard, not a mathematical hard KL bound: the last step can overshoot. Warm-up uses teacher actions and supervised/value losses only, never off-policy PPO.
4. Teacher regularization accepts all shortlisted candidates tied on maximum primary utility (coverage gain minus four times upper holes). Its loss is minus log probability mass assigned to this set. An additional expected utility-regret penalty discourages lower-utility alternatives. These are auxiliary optimization losses, not extra environment rewards or inference-time hard masks. They do not guarantee good future outcomes and may constrain exploration; `--imitation 0 --regret-weight 0` provides a pure PPO ablation. The legacy sigmoid tie-break is no longer a mandatory one-hot target in this trainer.
5. Record per-component losses, utility-match rate, clipping fraction, exact KL, actual optimizer-step counts, and ranking-matrix singular-energy diagnostics. Keep model size unchanged rather than compressing rank2 based on only 48 diagnostic states. The GUI now labels survival-value units and warns when layer-completion probabilities are an unrecalibrated imitation auxiliary.

The learned policy remains the existing 62,485-parameter sigmoid Conv3d reranker with its explicit 128-candidate shortlist and exact CPU legality checks. This does not implement a new architecture, force teacher choices at inference, or rank all legal placements on the GPU. The low-rank diagnostics are measurements, not a regularizer or a claim of stronger play.

## Outputs and evaluation

`last.pt` is resumable; `previous.pt` preserves the prior save; `best.pt`/`best.json` are updated only after a complete selection evaluation improves capped mean survival, with minimum/median tie-breaks. Initial and candidate evaluations reuse selection seeds and are validation, not independent held-out evidence. `recommended-policy.json` compares the best learned selection result to sustained without changing normal GUI defaults. `diagnostics.json` includes the current checkpoint SHA-256 and ranking spectra. `summary.json`, `metrics.jsonl`, and optional `games/` retain the existing logging and replay format.

Recording still supports `--record-every` and `--record-max-mib`; the default quota is 1 GiB. A quota or recording-write failure does not stop training/checkpointing. Incomplete games are recorded as valid JSON prefixes. Verify one with `python train_survival.py --verify-game path/to/game.json`.

Once `best.pt` exists, run a separate evaluation on seeds not used for training or selection:

```bash
.venv/bin/python tetris_neural.py evaluate --checkpoint runs/survival-ppo/best.pt --device cuda --seeds 940001 940003 940005 940007 --pieces 32000 --output survival-ppo-heldout.json
```

Inspect results before promoting the policy. The full checkpoint hash in the evaluation must match the checkpoint being discussed; an imitation evaluation is not evidence about the final RL checkpoint.

To play a selected policy:

```bash
.venv/bin/python sparse_tetris_4d_optimized.py --neural-checkpoint runs/survival-ppo/best.pt --neural-device cuda
```

## Reproducibility and limitations

Run `python tests/test_survival_ppo.py` and `python tests/test_overnight.py`. Tests cover PPO clipping with positive/negative advantages, valid-action masking and exact KL, tied primary objectives, gradient/weight changes, immutable source models, survival-critic initialization, exact CPU resume including optimizer/environment state, policy export, recording replay and strict configuration rejection. CUDA coverage runs only when hardware is available and otherwise explicitly skips. Existing GUI/engine suites remain required.

The settings are starting experiments. More optimizer passes can cost wall-clock throughput, and teacher regularizers can trade exploration for stability. Training fixtures and smoke updates do not show better long-game survival. No stronger checkpoint is committed by this change.

Primary methodological reference: Schulman et al., Proximal Policy Optimization Algorithms, https://arxiv.org/abs/1707.06347. Exact utility-set distillation and the integration choices above are project-specific, not claims made by that paper.
