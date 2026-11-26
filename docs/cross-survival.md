# Spatial V4 survival experiment

The incumbent averaged 39,718.8 placements on 72 paired seeds. The previous control endpoint averaged 37,116.8, a 6.55% decline with a paired 95% interval of −4,537 to −699 placements. Seeded piece-stream reconstruction identified CROSS as the next unplaceable piece in all 126 training top-outs across the control and diverse arms. Of 506 expanded, completed top-out recordings in the local library, 504 ended on CROSS and two on HYPER. These recordings come from 262 distinct seeds and several policies. The experiment tests whether preserving legal CROSS landings increases mean survival on fresh games. [Prior paired results](../validation/spatial-ppo-curriculum.json)

## What the weights changed

The model has 49,141 actor parameters and 53,689 critic parameters. Every parameter changed during both previous runs. The tensor audit records all 23 tensors, changed counts, relative displacement, the largest individual changes with coordinates, cosine alignment between updates, and singular-value energy for matrices. The estimates below refer to changes from the same incumbent. [Full tensor audit](../validation/cross-survival-weight-deltas.json)

| Parameter | Measured change | Interpretation |
|---|---|---|
| First candidate matrix | Control relative L2 change 8.23%. Its update direction and the diverse update had cosine 0.00145. | On 40 archived positions, restoring the entire candidate branch changed 13 control choices and 10 diverse choices. That intervention provides a reason to constrain actor changes. |
| Spatial actor matrix | Control changed by 15.94% in relative L2. One singular direction accounted for 94.84% of the squared change. | The largest change was the connection from the X-axis target-layer boundary feature to hidden unit 16, from 0.01870 to 0.00447. Restoring the entire spatial branch changed three control choices. This coefficient alone has no measured survival effect. |
| Final critic weight | Control relative L2 change 96.93%. The control and diverse update directions had cosine 0.962. | The critic predicts remaining placements in thousands. Its own parameters are isolated from the actor. Transplanting its eight tensors preserved all actor logits on the 40 audited boards. |
| Final actor bias | Control changed from 0.15029 to 0.16719. | The deployed additive teacher-prior policy adds this scalar to every candidate. In exact arithmetic it cancels from softmax and argmax. |

The 64 learned rank-unit contributions plus the teacher term reconstructed the greedy-versus-runner-up logit margin on all 40 positions for each of the three checkpoints, within 0.0000026. In the incumbent, the largest unit accounted for 4.62% of the absolute learned margin on average. The five largest accounted for 20.08%. These percentages describe these positions and cannot establish the effect of removing a unit in a complete game. [Per-unit attribution and seed provenance](../validation/cross-survival-rank-attribution.json)

In the rollout-label pilot, 24 uncensored incumbent continuations gave mean absolute errors of 11,604 placements for the original critic and 882 for the control critic. The donor also underestimated eight of twelve continuations that survived the pilot cap. The calibration gate therefore measures both critics on identical complete games, separately at each sampling temperature. Its primary unit is the game. The donor must improve final-512-placement error and limit first-quarter error growth to 5% at **both** temperatures.

## Players and training

* Direct players reuse every incumbent tensor and add a scalar times the change in CROSS mobility to each candidate logit. The tested weights are 0, 0.25, 1 and 4. Mobility is the log of one plus the legal count clipped at 16, divided by the log of 17. Zero is the original checkpoint. Direct parameter search has been evaluated in conventional Tetris. The game geometry and results here are specific to this engine. [Scherrer et al.](https://www.jmlr.org/papers/v16/scherrer15a.html)
* A learned variant freezes the original actor and trains a 2,241-parameter residual head with the 64 candidate hidden activations, two mobility values, visible preview and observed bag flags. Its zero final layer preserves the starting logits. Bounding each correction to ±0.05 protects any original top-two margin greater than 0.10. The critic still learns.
* The lookahead player uses the visible next piece. If the greedy move has at most eight remaining CROSS origins, it examines up to 16 actions and four teacher-ranked replies. If the preview already has no legal landing, it first searches the full shortlist for a rescue. Within that frontier it adds the best reply mobility to the current actor logit. Public information determines the decision.
* PPO compares full actor at temperature 1, full actor at 0.25, temperature 0.25 with a 0.05 frozen-incumbent KL penalty, and temperature 0.25 with the bounded residual. All four keep the same placement reward. Temperature applies to the sampled behavior, likelihood ratios, entropy and KL. The choice policy used in evaluation is greedy. [Schulman et al.](https://arxiv.org/abs/1707.06347) · [Policy regularization and distillation](https://arxiv.org/abs/1803.03835)

## CPU and GPU

The previous optimized training path reached a median 387.6 placements/s on the 32-environment benchmark. In the 320 measured production updates, candidate preparation accounted for 51.4% and commit for 17.6% of attributed time. The new system first screens placements by exact integer primary utility, computing FP64 secondary scores for surviving candidates, ties and seeded exploration. Dense and memory-limited paths retain the full scorer. A checked native commit combines landing validation, placement and clear compaction. The board stays packed for simulation. [Previous throughput and parity evidence](../validation/spatial-ppo-throughput.json) · [GPU top-k research](https://anilshanbhag.com/static/papers/gputopk_sigmod18.pdf) · [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

The GPU gate compares five alternating pairs on each of fresh, late, and forced clear/recovery/cache-fallback workloads. It checks shortlist/teacher scores, moves, boards, gradients, Adam state, random state and KL decisions. The new pair of fast paths needs at least 10% combined end-to-end training gain against the previously optimized implementation. The runner takes explicit flags from the hash-pinned benchmark report. Greedy evaluation throughput has its own measurement.

All 15 paired training benchmarks reproduced the previous implementation's actions, final boards, weights, gradients, Adam state, RNG and KL decisions. The separate greedy benchmark reproduced its action traces. Median paired training speedups were 1.123× on opening boards, 1.154× on forced clears/recovery, and 1.288× on archived late boards. Greedy evaluation ran at 0.729× its previous throughput. Both new paths are enabled during PPO, and greedy evaluation uses the previous simulator. In single-game tests with a diagnostic direct weight of 0.5 and both fast paths on 12 archived positions, warmed greedy and lookahead GPU latencies were 7.54 and 12.50 ms median, or 8.19 and 60.79 ms at p95. CPU medians were 42.43 and 50.16 ms. CPU and GPU chose identical actions in both modes. [Performance gate and latency measurements](../validation/cross-survival-throughput.json)

For desktop playback, `--two-stage-exhaustive` enables the exact native CPU shortlist while retaining the full rendering commit. On the same 12 positions it reduced median greedy decision time from 269.1 to 46.3 ms and median lookahead time from 779.5 to 69.4 ms. All chosen actions and triggers matched. These are single-thread, batch-one measurements on a small archived sample. The virtual-display smoke verified a selected drop and a visible feature decision with this option. Use `--backend native --neural-checkpoint PATH --two-stage-exhaustive`, adding `--neural-preview-search` for the lookahead player.

## Run and evidence

Use one RTX 4060 Ti and FP32. The cumulative time ceilings are 2 hours for preparation, 4 through calibration and direct-weight screening, 10 through training, 16 through development, and 24 through sealed evaluation. A later phase can use time saved earlier. The device ledger charges elapsed reservations and conservatively charges a lost, unclosed reservation at its full remaining allowance. Live evaluation games and sampled critic trajectories save enough state to resume. A signal requests a stop at the next safe boundary.

The trainer runs 80 updates of 8,192 placements per recipe on each of two training seeds. Eight common screening games rank recipes by their two-seed average. The lower-numbered training seed supplies the chosen recipe's endpoint. The best greedy player, best lookahead player and incumbent play up to 24 additional paired games each. Mean then median determine the winner, with single-pass play preferred on an exact tie. If the incumbent wins qualification, it remains the recommended player. Otherwise a single frozen finalist plays 144 fresh, paired games against it. Promotion requires a relative mean gain of at least 5% and a positive lower 95% paired-bootstrap bound. All games run to physical top-out. If the direct or PPO family exhausts its phase budget, the whole unmatched family is ineligible. Fully completed families and incumbent lookahead can still qualify. An incomplete qualification or final comparison leaves the incumbent in place. A separate watchdog enforces the 24-hour GPU ceiling. [Agarwal et al., evaluation under uncertainty](https://arxiv.org/abs/2108.13264)

The training and evaluation switches are checked separately. Training requires the five-pair median to clear 1.10× on all three workloads. Evaluation also requires its own five-pair median to reach 1.00×. The final report includes per-position and batched latency with warmup and batch size identified.

To generate fresh performance receipts, first build the native scorer in the candidate checkout. Set `OUT` to a fresh experiment directory and `INC` to the frozen incumbent checkpoint.

```bash
set -e
P="$OUT/perf"
python -B tools/benchmark_spatial_ppo.py --freeze-simulation --output "$P"
date +%s > "$P/gpu-smoke.start"
python -B -m unittest discover -s tests -p test_cross_survival.py
date +%s > "$P/gpu-smoke.end"
python -B tools/benchmark_spatial_ppo.py --make-late-states data/game-library --output "$P/late-states.json"
date +%s > "$P/gpu-gate.start"
for workload in opening boundaries late; do
    python -B tools/benchmark_spatial_ppo.py --compare-simulation --pairs 5 --seed 418000 \
        --workload "$workload" --reference "$P/reference-3929296" --candidate "$P/candidate" \
        --incumbent "$INC" --late-states "$P/late-states.json" --output "$P/$workload"
done
date +%s > "$P/gpu-gate.end"
python -B tools/benchmark_spatial_ppo.py --summarize-simulation-gate "$P" --incumbent "$INC" --output "$P/gate.json"
python -Bm tools.cross_run_budget "$OUT" "$P"
```

The calibration and benchmark use immutable, hashed inputs. Use the same experiment output directory for a resume. With an already built native checkout, verified checkpoints, and a completed benchmark report, run the following commands.

```bash
A="$PWD/survival-training.artifacts"
OUT="$A/cross-survival-24h"
INC="$A/v4-enhanced-run/ppo/best.pt"
DONOR="$A/spatial-ppo-improvements/experiment/control/last.pt"
export TETRIS4D_GAME_LIBRARY=off OMP_NUM_THREADS=1 TRITON_CACHE_DIR="$OUT/triton-cache"
.venv/bin/python -Bu tools/calibrate_cross_critics.py --output "$OUT" --incumbent "$INC" --donor "$DONOR"
.venv/bin/python -Bu tools/run_cross_survival.py --output "$OUT" --incumbent "$INC" --donor "$DONOR" --performance-gate "$OUT/perf/gate.json"
```

For the initial calibration on this host, set `TETRIS4D_CALIBRATION_RUNTIME` to the previous frozen Spatial PPO runtime for the calibration command alone. Pass that runtime and the byte-identical calibration driver to the main command using `--calibration-runtime` and `--calibration-driver`. The source fingerprints in each cache entry identify the runtime. The receipt verifier checks every seed and recomputes the selection decision. The experiment output includes the value-head decision, direct players, per-arm initialization and recovery manifests, per-game development records, and the GPU ledger. The selected manifest records both the checkpoint and whether to launch the player with `--neural-preview-search`.
