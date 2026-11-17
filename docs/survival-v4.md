# Spatial survival V4 — reconstructed change

## Provenance and scope

This implementation was recreated on 2026-11-16 from the existing V3 code on main at 56976f1 and user-provided historical manifests. The original local commit 3c95223, the 5.56 MB Git bundle, exact source snapshots, training datasets and V4 model weight files were not available. Consequently this is **not a byte-for-byte recovery** of that commit or its checkpoints. No newly trained weights are supplied by this pull request.

The historical actor/critic model card and two source reports are retained under validation/historical/. Their numerical results describe the earlier, inaccessible experiment; **they have not been reproduced on this branch**.

## Architecture

V4 retains V3's 34-channel board encoding, 82 context features and 164 input features per candidate. It adds six global adjacency descriptors to the independent critic, making 156 critic inputs, and adds 18 per-candidate descriptors: the change in each of six statistics along each of X, Y and W.

The six normalized statistics are absolute height difference, squared height difference, cliff frequency, occupancy-mask boundary, buried-gap-mask boundary, and target-layer occupancy boundary. Each statistic is computed across all spatial neighbor edges of an axis; the critic receives the mean across three axes, while the actor receives axis-specific afterstate-minus-before-state changes. A cliff is an adjacent-column height difference greater than one. The actor projection has 18 inputs and 64 outputs without a bias, initialized to zero. The critic adds six LayerNorm inputs and six input-to-hidden weights per 128-unit hidden layer. The complete V4 network has **77,870 parameters**, 1,932 more than V3.

Candidate afterstates are physically simulated using the same merge and hyperlayer-compaction functions as gameplay. This prevents the feature vector from describing an impossible state after a clear. The default shortlist remains deterministic teacher/random; a deterministic coarse spatially diverse option is available. The sustained heuristic remains the game default, and the explicit teacher prior is retained in hybrid neural policies.

The V3/V4 GUI uses an alternative feature inspector showing the selected teacher prior, raw actor score when decomposable, critic value and X/Y/W geometry changes. It does not mislabel V3/V4 data as CNN sigmoid activations.

## Training entry points

Prepare a native build first:

    python tools/build_native.py

Reconstruct a *new* actor checkpoint from physically generated teacher placements:

    python -B tools/pretrain_spatial.py --backend native --device cpu

The default script collects four training seeds and two separate validation seeds, one snapshot every eight placements, and trains the actor on a soft teacher-prior distribution at temperature 8. It excludes the runtime prior from its loss, leaves the critic untrained and starts with zero PPO updates. Its outputs are named with a reconstructed suffix and must not be confused with the historical checkpoint.

Collect uncensored teacher continuation datasets on separate seed groups:

    python tools/precalibrate_spatial.py collect --seeds 7200000 7200002 7200004 7200006 --dataset runs/v4/critic-train.npz
    python tools/precalibrate_spatial.py collect --seeds 7300000 7300002 --dataset runs/v4/critic-selection.npz
    python tools/precalibrate_spatial.py collect --seeds 7400001 7400003 --dataset runs/v4/critic-audit.npz

Fit a critic without changing actor parameters:

    python tools/precalibrate_spatial.py fit --actor-checkpoint models/neural-spatial-actor-reconstructed.pt --train-dataset runs/v4/critic-train.npz --selection-dataset runs/v4/critic-selection.npz --output models/neural-spatial-pretrained-reconstructed.pt

Audit independently, with training and selection seeds rejected:

    python tools/precalibrate_spatial.py audit --checkpoint models/neural-spatial-pretrained-reconstructed.pt --dataset runs/v4/critic-audit.npz

Run fresh PPO initialization (a missing pretrained file is a hard error):

    python -B train_survival_v4.py --pretrained models/neural-spatial-pretrained-reconstructed.pt --run-dir runs/v4-ppo --device cpu --backend native --hours 8 --cache-rollout

The PPO initializer imports matching V4 weights but creates a new optimizer and starts PPO progress at zero. Current-policy critic calibration is performed by the inherited trainer before PPO; the imported teacher-policy critic is **not** assumed calibrated for the actor's new policy. The optional experimental uncalibrated override requires an explicit flag. The V4 trainer rejects GPU candidate preparation until physical afterstate parity has been independently established.

## GPU training path

V4 now trains on CUDA end to end. `train_survival_v4.py` enables `--gpu-prepare`, `--fused-encoding`, `--coalesce-transfers` and `--cache-rollout` (with `--device cuda --backend native`) whenever CUDA is available; pass `--no-gpu-defaults` or `--device cpu` to opt out.

- Candidate afterstates are merged, layer-compacted and measured on device (`device_afterstate_descriptors`), so `--gpu-prepare` is supported. Exact CPU fallback rows skip the host afterstate pass; their descriptors are recomputed on device from the final options.
- `tetris_gpu_spatial.py` holds fused Triton kernels for the adjacency statistics (one launch per call instead of ~100 tensor ops), used by preparation and by the critic encoder. Single-state V4 preparation fell from 12.6 ms to 4.5 ms and forward+selection sync from 7.3 ms to 2.0 ms on an RTX 4060 Ti.
- Calibration and selection games (non-rollout phases) prepare single states on CUDA instead of the CPU reference path.
- `tools/pretrain_spatial.py` collects all seeds in lockstep through the CUDA preparer (identical records; `--cpu-collect` restores the CPU collector), keeps every snapshot resident on device and evaluates in batches. `tools/precalibrate_spatial.py` collects in lockstep on CUDA and fits/audits on `--device` (default auto).
- Diverse shortlists are not implemented by the CUDA preparer and fall back to exact CPU preparation instead of silently using teacher/random sampling.

Parity: `tests/test_survival_v4_gpu.py` checks exact integer bit lengths, layer compaction, batched-vs-reference descriptors, fused kernels and full GPU preparation against the CPU reference (set `TETRIS4D_TEST_CUDA=1`).

### Actor input scale fix

Each statistic is a mean over 3,840 neighbour edges, so one placement changes it by only ~1e-4. Fed raw into the zero-initialised, bias-free projection, the actor could not learn a material contribution from it. The actor now multiplies the 18 deltas by `SPATIAL_DELTA_SCALE = 3840` (per-placement edge-change counts, O(1)). The stored `spatial_delta` arrays and the critic inputs are unchanged. Checkpoints with non-zero spatial weights trained before this change are not equivalent; the architecture source hash already rejects resuming across it.

### Pretraining loss fix

`F.kl_div` returned NaN whenever a batch contained padded candidates (0 · −inf). The supervised loss now uses an explicitly masked KL in which padding contributes exactly zero.

### Exhaustive placement screening

`--screening exhaustive` (on `train_survival_v4.py` and `tools/pretrain_spatial.py`) screens every legal placement, meaning every orientation at every position, instead of the default `--candidates 2048` spread across orientations. That is up to ~648,000 placements per piece for the 192-orientation pieces, against ~10 positions per orientation when sampled. The network is unchanged: the teacher still shortlists `--shortlist` candidates (best plus random), and only those reach the actor. What changes is which placements the teacher can choose from.

- **Cost:** a 15-state CUDA preparation batch takes ~48 ms exhaustive vs ~37 ms sampled. The CPU reference takes ~1.3 s per state, so exhaustive training needs `--gpu-prepare`.
- **Layer clears are scored on the device** in both modes, with the same float64 order as `game.score_candidate`. Clearing states no longer fall back to the CPU.
- **Pretrained actors:** screening does not affect the weights, so a model pretrained with sampled screening can be continued with `--pretrained ... --screening exhaustive`. Any other config difference is still rejected. The mode is recorded in the run signature, so a run cannot be resumed under a different mode.
- **Selection baseline:** the sustained heuristic in selection games always uses `config.candidates` (2,048 sampled), so exhaustive runs are compared against the same baseline as sampled runs.

Parity: `tests/test_gpu_simulation.py` (`TETRIS4D_TEST_CUDA=1`) checks exhaustive GPU preparation against the CPU reference for all 15 pieces, and clearing boards in both modes. `tests/test_exhaustive_screening.py` covers the config, the enumeration count, the CLI and the pretrained continuation.

## Historical measurements (not reproduced)

The archived manifests report 1,024 training and 256 validation actor snapshots from disjoint seeds, 300 supervised steps, raw actor validation KL changing from 1.30227 to 0.03780, raw exact teacher agreement from 1.17% to 61.72%, and raw primary utility agreement from 7.42% to 100%. They report no PPO updates.

The archived model card reports critic calibration on four complete training games, two selection games and two untouched audit games. Audit MAE was 1,492 remaining placements against a constant baseline of 4,187, with near-end MAE 523 and a signed bias of +1,377 placements. Those estimates concern *teacher-policy continuation* only and are based on just two independent audit games.

The available historical gameplay smoke comprises 36 games on four seeds. All were censored at 512 placements with no clears. It validates a former early-game inference pipeline but cannot demonstrate survival gains, equality, or superiority. A later 4-state audit also found a raw-actor primary-utility failure repaired by the teacher prior. No replacement of the sustained default is justified.

## Verification and limitations

This PR adds deterministic numerical shape/bit-count tests, exact afterstate recomputation, critic/actor gradient isolation, V3 actor-initialization equivalence, masking/padding and diverse shortlist repeatability. CI runs the V4 tests and both training CLI help commands. The exact original 66-test suite, CUDA/Triton verification, full-game strength validation and pretrained artifact hashes cannot be reconstructed from the available material.

The reconstructed feature formulas and training scripts are new implementations matching the reported intent, not verified copies of their historical versions. Historic model hashes must not be assigned to new artifacts. Long-game matched-seed evaluations, new independent critic audits, and GPU benchmarks are required before declaring performance or model parity.
