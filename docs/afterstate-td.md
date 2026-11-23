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

During training, run `python tools/training_dashboard.py --run-dir "$OUT/production" --data-dir "$OUT/dashboard"` for loopback telemetry. Greedy development selection uses eight fresh games every two hours when training has completed an update. Mean placements, then median, then earliest checkpoint determine the selection.

After production ends, evaluate the selected checkpoint against the three controls on 16 reserved fresh seeds.

```sh
python tools/evaluate_afterstate_td.py evaluate --split holdout --source "$SOURCE" \
  --td "$OUT/production/best.pt" --selection "$OUT/production/selection.json" \
  --ppo "$PPO_BEST" --output "$OUT/holdout.json"
```

The held-out comparison uses a 64,000-placement cap and extends the leading pair to 128,000 if either has at least eight capped games. Each completed game is cached with its input identity. The report includes per-seed placements, clears, elapsed time and paired bootstrap estimates. The predefined win threshold requires at least 5% higher mean survival than the strongest control and a positive lower confidence bound. Weights and game data stay local.
