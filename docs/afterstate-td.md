# Training an afterstate value head

The TD trainer imports an exhaustive spatial V4 actor and its calibrated regional critic. At each move the frozen actor shortlists 16 legal placements. Training samples from the shortlist with logits equal to beta times the afterstate value plus 0.125 times the teacher prior. Greedy gameplay picks the highest predicted value. The target is the number of **additional** successful placements after the chosen move, in thousands. A terminal afterstate has target zero.

Build the native backend with `python tools/build_native.py`. The timed experiment uses CUDA with Triton and the same heuristic weights that produced the source checkpoint. The three commands below assume the paths in uppercase shell variables have been set to local files or output directories.

```sh
export TETRIS4D_GAME_LIBRARY=off
python tools/evaluate_afterstate_td.py calibrate --source "$SOURCE" \
  --positions-report "$POSITIONS" --library "$GAME_LIBRARY" --output "$OUT/beta.json"

python train_afterstate_td.py --source-checkpoint "$SOURCE" --calibration "$OUT/beta.json" \
  --danger-cache "$OUT/danger-cache.pt" --run-dir "$OUT/pilot" --hours 0.08333333333333333

python train_afterstate_td.py --source-checkpoint "$SOURCE" --calibration "$OUT/beta.json" \
  --danger-cache "$OUT/danger-cache.pt" --run-dir "$OUT/production" --hours 8
```

Calibration replays 32 fixed development positions and takes the reciprocal of the median population standard deviation of the 16 value predictions. The pilot and production use separate output directories. The training danger pool uses recorded games from the main checkout's library, excluding calibration games and evaluation seeds.

Rerun a training command with the same inputs to recover its last atomic checkpoint. The manifest fixes the original deadline. Training uses 32 environments, 256 steps per rollout, a 256-row optimizer batch, one AdamW pass at 0.0003, and TD(λ) with γ = 1 and λ = 0.999. A check for the successor move supplies the bootstrap value at a truncation. The final checkpoint includes pending transitions for exact recovery.

The default training episode limit is 32,000 placements. Use `--pieces 0` for episodes that continue until top-out or the planner runs out of legal moves. To lift the limit on an existing run, stop it gracefully and resume with its original arguments, changing the limit to `--pieces 0` and adding `--uncap-existing-run ORIGINAL_MANIFEST_SHA256`. Continue to use those flags for subsequent resumes. The amendment receipt links the original manifest and interrupted checkpoint. Active games retain their piece streams, the partial rollout and optimizer continue, and the original wall deadline and greedy selection cap still apply. Games that already ended at the old limit remain recorded as truncated.

During training, run `python tools/training_dashboard.py --run-dir "$OUT/production" --data-dir "$OUT/dashboard"` for loopback telemetry. Greedy development selection uses eight fresh games on a two-hour schedule. A completed selection skips elapsed slots and requires a newer update before the next comparison. Mean placements, then median, then earliest checkpoint determine the selection.

## Experimental actor-regularized TD

Use `--warm-start "$PREVIOUS_TD"` in a fresh run directory to import the prior TD value head with a fresh optimizer and training clock. Optional `--actor-bound B` ranks shortlisted moves by the frozen actor logit plus `clamp(beta * (value - highest_valid_value), -B, 0)`. Greedy play takes the maximum. Training samples the softmax of these logits. A zero bound gives the frozen actor control. Omitting the option retains the original TD policy.

At update 40, the mean margin between the two highest predicted values on eight training boards was 0.45 placements. The `tools/screen_afterstate_td.py` discovery phase compared bounds 0.5 and 1.0 on four paired near-top-out positions capped at 1,500 additional placements. Mean survival was 557.25 and 196.5 respectively, versus 220.25 for pure TD. The paired 95% interval for the smaller bound relative to pure TD was −33 to +1,052 placements. Confirmation uses separate streams. Pass the experiment's exclusion manifest with `--exclude-positions` to keep reserved games out of training.

At `--lr 1e-5`, setting `--target-tau 0.01` moves the bootstrap head 1% toward the online head after each optimizer step. In a six-pass replay of the update-40 rollout, the mean absolute target drift fell from 132 to 20 placements. The sample contained no episode endings. Gradient clipping is independently optional via `--clip-grad`.

Related work studies policy updates constrained by a prior, the effects of value approximation error and delayed targets, and approximate policy iteration in two-dimensional Tetris. [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920), [Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html), [Approximate Modified Policy Iteration and its Application to the Game of Tetris](https://jmlr.org/papers/v16/scherrer15a.html).

## Final checkpoint and holdout

After training, `select-final` evaluates the final checkpoint on the eight development seeds and compares it with the best completed scheduled selection at the same cap. The earlier checkpoint wins a tie. If no scheduled selection finished, a completed final development evaluation supplies the checkpoint. Read its filename from the selection receipt before evaluating the 16 reserved fresh seeds. Supply `--previous-td` to include the predecessor as a fourth control alongside the V4 actor, frozen V4 afterstate critic, and PPO checkpoint. Pass the independently pinned, warm-started policy with its chosen actor bound to `--initial-td` to measure the effect of the training updates on holdout survival.

```sh
python tools/evaluate_afterstate_td.py select-final --run-dir "$OUT/production" \
  --output "$OUT/final-selection.json"
checkpoint=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$OUT/final-selection.json")
python tools/evaluate_afterstate_td.py evaluate --split holdout --source "$SOURCE" \
  --td "$OUT/production/$checkpoint" --selection "$OUT/final-selection.json" \
  --ppo "$PPO_BEST" --previous-td "$PREVIOUS_TD" --output "$OUT/holdout.json"
```

The held-out comparison uses a 64,000-placement cap and extends the leading pair to 128,000 if either has at least eight capped games. Each completed game is cached with its input identity. The report includes per-seed placements, clears, elapsed time and paired bootstrap estimates. The predefined win threshold requires at least 5% higher mean survival than the strongest control and a positive lower confidence bound. Weights and game data stay local.
