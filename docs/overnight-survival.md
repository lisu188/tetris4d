# One-command overnight survival training

Place `train_overnight.py` in the repository root beside `tetris_neural.py`. This is an additive runner: it does not replace the game, the existing neural module, your recorded results, your CUDA installation, or existing model files. Its interface was reviewed against `main` at `fa22032fac74fb2d508adc7547ca314260d9341c`.

## Run

From your existing WSL checkout, with the working CUDA-enabled virtual environment:

```bash
.venv/bin/python -u train_overnight.py --hours 8 --device cuda --record-games
```

For a detached run that continues after the shell closes:

```bash
cd /home/andrz/git/tetris4d && mkdir -p runs/overnight-survival && nohup .venv/bin/python -u train_overnight.py --hours 8 --device cuda --record-games >> runs/overnight-survival/console.log 2>&1 < /dev/null &
```

The runner performs initialization, teacher warm-up, survival A2C, periodic greedy validation, checkpoint selection, recording and automatic resume. It does not require separate collect/train/finetune commands. No demonstration dataset is needed. It imports the existing model and native placement engine. No GUI is created.

The default is an eight-hour wall-time budget for this invocation, including initialization and evaluation. Stopping is cooperative: it checks between vector steps and evaluation moves, completes an in-flight optimizer step and flushes files. This is not a hard process-kill deadline. Windows must remain awake; terminal detachment cannot survive PC shutdown or forced WSL termination.

## Starting weights and resume

A valid `runs/overnight-survival/last.pt` always takes precedence and resumes automatically. Otherwise the runner initializes from an explicitly supplied `--checkpoint`, then from `models/neural-policy.pt` if present, or creates the existing 62,485-parameter sigmoid network. It does not silently choose `models/neural-rl.pt`; use `--checkpoint models/neural-rl.pt` to initialize from that file.

The initial learned model is evaluated before changing it, when it already has training updates. Its candidate/value architecture is retained, but the final value layer is reinitialized for the new return units. Policy weights are not reset. The initial checkpoint itself is never overwritten. A new run performs 128 online teacher-warm-up updates, then switches to survival A2C. Warm-up advances full teacher games across clear boundaries and does not collect a separate disk dataset. It trains imitation and a bootstrapped survival value; its updates are clearly distinguished in logs from survival updates.

Rerunning the same command resumes model, optimizer, scaler, RNG, episode seeds, exact move histories, current boards, targets and progress, for another eight-hour invocation. Existing best checkpoints are retained. Source/config mismatches fail instead of quietly changing the experiment. Recording flags and the wall-time budget may be changed on resume. Use another `--run-dir` for changed optimizer, batch, rollout, seed, warm-up or evaluation settings. Do not replace core source files while a run is in progress.

`last.pt` is saved atomically every 10 updates or approximately 60 seconds, whichever checkpoint boundary is reached first, and on graceful exit. `previous.pt` retains the preceding saved state. A forced kill can lose work since the last checkpoint; it does not erase the existing best model. One process at a time can hold a run-directory lock.

## Survival objective

Each legally committed piece contributes **one successful placement**, including the final successful placement before top-out. Clears receive no extra reward. There is no reward for attempted moves, waiting, reset, wall-clock time or frames rendered.

The critic uses reward `0.001` per successful placement and gamma **1.0**. This constant scaling expresses returns in thousands of future pieces; it does not favor clears, short games or fast hardware. A2C uses GAE lambda 0.999, advantage normalization, gradient clipping, a small entropy term and a 0.05 teacher-imitation regularizer. Therefore the optimization is regularized survival learning, not a proof of finding a globally optimal or loss-free strategy.

There are four game environments, 64 vector steps per rollout, FP32 training and memory-bounded minibatches of 16 boards by default. Rollout boundaries and hyperlayer clears do not reset games. Episodes are capped at 32,000 pieces by default. That cap is a truncation, not a loss: the next-state value is bootstrapped and the GAE trace is broken across the reset. Physical top-outs and protected-planner dead ends are recorded separately and do not bootstrap. The existing checkpoint's shortlist, candidate budget and target-protection settings are retained; this remains a hybrid candidate reranker, not unrestricted exhaustive search.

The auxiliary completion head is retained but not given misleading survival targets. The GUI's value output now measures thousands of expected future placements for these checkpoints, rather than the older clear-return units.

## Checkpoint selection

Greedy evaluations use separate odd-numbered seeds 910001 and 910003, with a common 32,000-piece evaluation cap. Training uses even-numbered seeds derived from a monotonically increasing episode index. No hidden future piece order is supplied to the model.

Selection ranks **mean capped pieces survived**, then minimum and median survival as tie-breakers. Clear count, training loss, GPU utilization and pieces per second do not decide which checkpoint is best. Equal scores retain the incumbent. This is a restricted-mean survival measure; a capped run is right-censored, not evidence that it can play forever.

The runner also evaluates `sustained` under the same piece cap and candidate budget. `best.pt` is the best fully evaluated learned checkpoint, even when every learned checkpoint is worse than `sustained`. `recommended-policy.json` keeps recommending `sustained` unless the best neural result actually beats that baseline on the selection seeds. The GUI default is never changed automatically.

Selection runs after initial-model loading, after warm-up when no incumbent exists, and approximately every 60 minutes thereafter. An interrupted or partially completed evaluation cannot promote a checkpoint. No best checkpoint exists until a learned evaluation completes. These validation seeds are reused for selection, so they are not independent held-out test results. A separate evaluation on untouched seeds is required for a playing-strength claim.

## JSON game recording

`--record-games` enables ordinary JSON files under:

```text
runs/overnight-survival/games/<invocation-id>/
```

There is one file per selected training/evaluation episode per invocation. A resumed invocation has its own directory; it does not overwrite earlier recordings. Resumed active-game recordings contain the complete move prefix from their original seeded start, so they can be replayed independently.

Files contain the piece catalog hash, engine hash, heuristic parameters, seed, piece set, episode ID, counters and a list of every committed piece with rotation, X/Y/Z/W placement, target before/after, clear mask, clear count, update number, behavior label, placement reward and resulting board hash. A training episode may span many weight updates; it is not incorrectly labelled as having one fixed policy throughout.

Active-game files are atomically refreshed at checkpoint boundaries and have `complete: false`. Finished episodes have a distinct `top_out`, `planner_blocked` or `piece_limit` reason. A recording is valid JSON even when the game is unfinished. A hard kill may leave the last recorded prefix, not all moves since that checkpoint. To avoid filling disk, recording has a default 1,024 MiB quota; reaching it or encountering a recording-only write error disables additional recording without stopping training. Checkpoint-storage failures still stop the run rather than pretending progress is safely saved.

Record fewer games or change the quota without restarting the training experiment:

```bash
.venv/bin/python train_overnight.py --hours 8 --device cuda --record-games --record-every 10 --record-max-mib 2048
```

Omit `--record-games` to disable game recording. Scalar metrics and checkpoints remain enabled. The recorder writes actual JSON, not screenshots, videos or JSONL disguised as a JSON array. `metrics.jsonl` is a separate append-only training log.

Verify one recording through deterministic piece-stream replay and the engine's rigid-placement/compaction checks:

```bash
.venv/bin/python train_overnight.py --verify-game runs/overnight-survival/games/INVOCATION/training-EPISODE-seed-SEED.json
```

This checks every move, landing height, clear mask, board hash, target transition, final counters and claimed top-out. It is engine-based verification, not a second independent implementation of the physics. JSON recordings alone reproduce games without the neural checkpoint.

## Outputs and use

| File | Meaning |
|---|---|
| `last.pt` | Latest resumable training state, not necessarily the best player |
| `previous.pt` | Previous saved training state |
| `best.pt`, `best.json` | Best fully evaluated learned checkpoint and its selection evidence |
| `sustained-baseline.json` | Reference survival on the same selection seeds |
| `selection-*.json` | Full candidate evaluation reports, including incomplete runs |
| `summary.json` | Latest progress, active games, recording status and best score |
| `metrics.jsonl` | Stage, loss, gradient norm, throughput-relevant counts and GPU memory |
| `recommended-policy.json` | Neural versus sustained recommendation on selection evidence only |
| `games/` | Optional replayable JSON episodes and active prefixes |

Keep `runs/` out of Git; recordings and checkpoints can be large. Existing committed diagnostic results are untouched.

After a full evaluation has produced `best.pt`, load it in the existing GUI:

```bash
.venv/bin/python sparse_tetris_4d_optimized.py --neural-checkpoint runs/overnight-survival/best.pt --neural-device cuda
```

Inspect progress with `tail -f runs/overnight-survival/console.log` for the detached command. A first Ctrl+C/SIGTERM requests a safe stop; a second interrupt forces termination. Restarting the same command resumes automatically.

## Verification scope

The supplied CPU tests cover exact split-run resume, actual optimizer updates, finite parameters, selection by survival, rejection of incomplete/worse evaluations, replay tampering, recording quota/write failures, locking and termination/truncation targets. A full-size model update and a 1,300-move recording/replay crossing a clear were also exercised locally.

Local integration used a compatible attached engine snapshot plus retrieved current sustained-policy functions and the checksum-verified model/candidate source from the existing repository. This environment could not clone the fresh repository, push/merge changes, execute remote CI, or run CUDA. The included tests should therefore also be run against the actual checkout, and the CUDA test is explicitly skipped when hardware is unavailable. No overnight learning improvement or guarantee of never losing is claimed.

```bash
.venv/bin/python tests/test_overnight.py
```

Termination/truncation reference: Farama's Gymnasium explanation distinguishes true termination (no bootstrap) from external truncation (bootstrap): https://farama.org/Gymnasium-Terminated-Truncated-Step-API
