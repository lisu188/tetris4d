# Optional learned sigmoid policy

This implements the GPU-capable model, data collector, imitation trainer, on-policy A2C fine-tuner, evaluation and GUI integration. It is not a pretrained winning agent. Normal launch still uses **Unlimited + sustained** and does not import PyTorch. No driver changes, downloads or training occur automatically.

## WSL commands

The existing tested CUDA-enabled PyTorch installation can be retained. The new module requires NumPy and PyTorch 2.10 or newer. Do not replace a working CUDA wheel with the CPU wheel used by CI.

```bash
git switch main && git pull --ff-only
.venv/bin/python tetris_neural.py smoke --device cuda --output neural-cuda-smoke.json
.venv/bin/python tetris_neural.py collect --output data/neural-demonstrations --games 8 --validation-games 2 --pieces 16000
.venv/bin/python tetris_neural.py train --data data/neural-demonstrations --device cuda --updates 1000 --output models/neural-policy.pt
.venv/bin/python tetris_neural.py evaluate --checkpoint models/neural-policy.pt --device cuda --output neural-evaluation.json
.venv/bin/python sparse_tetris_4d_optimized.py --neural-checkpoint models/neural-policy.pt --neural-device cuda
```

Collection uses the existing native backend when available. Its games continue through clears, with separate training seeds 50000–50007 and validation seeds 60000–60001 by default. Evaluation seeds 70001–70004 are separate. Do not tune on the evaluation seeds and continue calling them held out. The dataset is compressed, checksummed NPZ shards with no Python pickle objects. An interrupted dataset is marked incomplete and is not accepted for training. Collection must use a new or empty output directory; it is not resumable in this first implementation.

Training defaults to FP32, 16 boards per minibatch and four accumulation steps (effective batch 64). The model evaluates each board encoder once per minibatch and chunks the candidate head. Training logs include validation teacher agreement, loss, gradient norm and peak CUDA allocated/reserved memory. This is not a guaranteed 8-GB capacity/performance claim. Reduce `--batch` on OOM; do not reinstall drivers. Optional `--precision bf16` and `--precision fp16` enable CUDA autocasting; FP16 uses a gradient scaler. Start with FP32 for a correctness baseline. Resume requires the same batch, accumulation, precision, data checksum and learning rate; use a new run for changed settings.

```bash
.venv/bin/python tetris_neural.py train --data data/neural-demonstrations --device cuda --resume models/neural-policy.pt --output models/neural-policy.pt --updates 1000
```

`--updates` is additional optimizer updates on resume. Checkpoints are atomic and include model, AdamW, scaler, CPU/CUDA/Python RNG state, progress and candidate-pipeline metadata. Model files and datasets are ignored by Git. A zero-update checkpoint is rejected by the gameplay loader. A checkpoint with training updates is still experimental, not automatically considered stronger than the sustained teacher. Load only trusted local checkpoints; restricted `torch.load(weights_only=True)` is used, but this is not a security sandbox.

## Reinforcement learning

After imitation, optimize actual cleared-hyperlayer rewards:

```bash
.venv/bin/python tetris_neural.py finetune --checkpoint models/neural-policy.pt --device cuda --updates 1000 --output models/neural-rl.pt
.venv/bin/python tetris_neural.py evaluate --checkpoint models/neural-rl.pt --device cuda --output neural-rl-evaluation.json
.venv/bin/python sparse_tetris_4d_optimized.py --neural-checkpoint models/neural-rl.pt --neural-device cuda
```

This is synchronous **advantage actor-critic (A2C)**, not PPO. Four CPU engine environments are inferred together; 64 steps per environment form a rollout. Each rollout performs one on-policy optimizer update, accumulated in memory-bounded minibatches. Reward is the number of hyperlayers actually cleared. The actor uses masked categorical probabilities, GAE with gamma=0.9995 and lambda=0.999, an entropy term, value regression and a small teacher imitation anchor (0.05). The auxiliary completion head is trained from demonstrations, not assigned misleading one-step targets during A2C. No claim is made that these hyperparameters are optimal.

A rollout chunk does not end the game. Episodes continue up to 32,000 pieces or a real top-out/selected-policy dead end. Piece-limit truncation bootstraps the next-state value but breaks the GAE trace across the reset; actual termination does not bootstrap. Intermediate rollout boundaries also bootstrap. A planner dead end is reported separately from physical top-out during evaluation. Every action is still validated and committed by the original engine.

A2C resume restores model/optimizer/scaler/RNG and reconstructs in-progress environments from their recorded legal actions. That replay can add startup work; it does not silently reset long episodes.

```bash
.venv/bin/python tetris_neural.py finetune --resume models/neural-rl.pt --device cuda --output models/neural-rl.pt --updates 1000
```

## Architecture

The default has **62,485 trainable parameters**, with no ReLU, GELU or SiLU:

- Packed 4,096-column integer board is transferred once and decoded on the chosen device into `[B,34,W,Y,X]`: 16 Z occupancy channels, 16 buried-gap channels, height/16 and current-target accessibility. All 65,536 occupancy bits are retained.
- 1x1x1 stem, 34→32 channels, GroupNorm and sigmoid.
- Three 32-channel residual blocks: depthwise 3x3x3, GroupNorm, sigmoid, pointwise mixing, GroupNorm, sigmoid and a learned residual scale initialized to 0.1. The spatial output is a residual representation, not constrained to [0,1]. Displayed activation maps are the actual sigmoid branch values.
- Shared candidate head combines affected-column pooling, global pooling, 82 context features and 96 candidate features, then 128 sigmoid → 64 sigmoid → linear ranking logit.
- Linear scalar value head and 16 completion logits, transformed to sigmoid probabilities for visualization. BCE-with-logits is used for stable auxiliary training.

Context contains the current and already-visible next piece, observed remaining-bag membership, layer occupancy/buried-gap counts and basic board/target features. It never inspects the hidden piece-stream bag order. Candidate features include the complete normalized shape coordinates/presence, origin, extents and exact engine-derived deltas.

## Candidate budget and safety

CPU generation examines the configured sampled candidate pool (2,048 by default), plus vertical-I finishing anchors, and uses the legacy exhaustive fallback when necessary. Physics-illegal actions are excluded by the engine. By default, the current-target protection heuristic also excludes moves, separately from physical legality. `collect --unrestricted` creates a distinct dataset/config without that heuristic filter; it is not the shipped training default.

The learned model ranks an **explicit shortlist of up to 128**: half are the best sustained-objective candidates and half are a deterministic diverse sample of the remaining legal pool. The full teacher winner is always present. This is a hybrid reranker, not exhaustive learned ranking of all placements, and not a promise of identical moves to the teacher. Shortlist size, original candidate budget, protection mode and chunk size are saved with the checkpoint. The normal game uses the checkpoint's candidate budget for neural playback. GPU chunking does not further drop candidates from that shortlist.

The selected move's `Candidate.score` remains the legacy heuristic score for engine consistency; `Decision.search_score` is the learned logit. Raw logits are not probabilities. The GUI provides separate **Learned sigmoid policy** and **Legacy heuristic** tabs, with actual per-channel W-section maps, selected-candidate 128/64 activations, value and layer-completion probabilities. The learned trace refers to the board before its labelled decision, rather than pretending to be the current board after subsequent moves. Human mode and pause still work; all 16 section views remain visible.

Demonstration value labels are discounted returns from terminated teacher episodes. Values from capped teacher episodes are masked rather than incorrectly treating truncation as terminal. Completion targets identify the original Z levels cleared within a 2,048-step horizon, tracking subsequent compaction. Horizon-censored targets near a capped episode end are masked. If all collected games are capped, imitation will not supervise the value head; A2C bootstrapped updates still train it. Increase data variety rather than interpreting teacher imitation as proof of strong gameplay.

## Measurement and validation

Evaluation compares the saved neural policy with sustained, on identical seeds and the same original sampled-candidate budget. It reports clears, survival, piece/clear throughput, agreement, stop reason and checkpoint identity. Neural inference includes CPU preparation and GPU-to-host trace transfer; it is not a pure CUDA-kernel benchmark. There is no automatic promotion to the no-argument default. Require independent long-game results before deployment.

```bash
.venv/bin/python tests/test_neural.py --output neural-tests.json
.venv/bin/python tetris_neural.py smoke --device cuda --output neural-cuda-smoke.json
```

The test suite checks occupancy preservation, no hidden-future leakage, teacher retention, legal rigid drops, real layer compaction, masking, chunk equivalence, sigmoid gradients/optimizer updates, checkpoint roundtrip/rejection, exact CPU imitation resume, A2C resume, original-level auxiliary targets and terminal/truncation bootstrapping. CUDA checks run only when a device is present and are explicitly skipped otherwise. Linux/Xvfb tests cover GUI inference, all 16 views, both trace tabs, pause/step/Human controls and continued post-clear retargeting. CPU smoke updates do not establish GPU performance or stronger play.

Primary API references: PyTorch Conv3d (`https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html`), AMP (`https://docs.pytorch.org/docs/stable/amp.html`), CUDA timing (`https://docs.pytorch.org/docs/stable/notes/cuda.html`) and restricted checkpoint loading (`https://docs.pytorch.org/docs/stable/generated/torch.load.html`). CUDA under WSL uses the Windows driver (`https://docs.nvidia.com/cuda/wsl-user-guide/index.html`); this implementation does not install or modify it.
