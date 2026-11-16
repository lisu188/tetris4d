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

## Historical measurements (not reproduced)

The archived manifests report 1,024 training and 256 validation actor snapshots from disjoint seeds, 300 supervised steps, raw actor validation KL changing from 1.30227 to 0.03780, raw exact teacher agreement from 1.17% to 61.72%, and raw primary utility agreement from 7.42% to 100%. They report no PPO updates.

The archived model card reports critic calibration on four complete training games, two selection games and two untouched audit games. Audit MAE was 1,492 remaining placements against a constant baseline of 4,187, with near-end MAE 523 and a signed bias of +1,377 placements. Those estimates concern *teacher-policy continuation* only and are based on just two independent audit games.

The available historical gameplay smoke comprises 36 games on four seeds. All were censored at 512 placements with no clears. It validates a former early-game inference pipeline but cannot demonstrate survival gains, equality, or superiority. A later 4-state audit also found a raw-actor primary-utility failure repaired by the teacher prior. No replacement of the sustained default is justified.

## Verification and limitations

This PR adds deterministic numerical shape/bit-count tests, exact afterstate recomputation, critic/actor gradient isolation, V3 actor-initialization equivalence, masking/padding and diverse shortlist repeatability. CI runs the V4 tests and both training CLI help commands. The exact original 66-test suite, CUDA/Triton verification, full-game strength validation and pretrained artifact hashes cannot be reconstructed from the available material.

The reconstructed feature formulas and training scripts are new implementations matching the reported intent, not verified copies of their historical versions. Historic model hashes must not be assigned to new artifacts. Long-game matched-seed evaluations, new independent critic audits, and GPU benchmarks are required before declaring performance or model parity.
