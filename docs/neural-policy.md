# Optional learned sigmoid policy

Use a trained column-policy checkpoint for evaluation or playback. For new training, follow the [Spatial V4 PPO guide](ppo-workflow.md). The original demonstration collector, imitation trainer and A2C fine-tuner are available at the [archived source revision](https://github.com/lisu188/tetris4d/blob/23159fe8d860a13ab16b8d5b69f2abf868522c82/docs/neural-policy.md).

## Evaluate or play a checkpoint

Pass the same checkpoint to either command. Use `--device cpu` for CPU evaluation or `--neural-device cpu` for CPU playback.

```bash
.venv/bin/python tetris_neural.py evaluate --checkpoint /path/to/model.pt --device cuda --output runs/neural-evaluation.json
.venv/bin/python sparse_tetris_4d_optimized.py --neural-checkpoint /path/to/model.pt --neural-device cuda
```

## Architecture

The default column-policy architecture has **62,485 trainable parameters**, with no ReLU, GELU or SiLU:

- Packed 4,096-column integer board is transferred once and decoded on the chosen device into `[B,34,W,Y,X]`: 16 Z occupancy channels, 16 buried-gap channels, height/16 and current-target accessibility. All 65,536 occupancy bits are retained.
- 1x1x1 stem, 34→32 channels, GroupNorm and sigmoid.
- Three 32-channel residual blocks: depthwise 3x3x3, GroupNorm, sigmoid, pointwise mixing, GroupNorm, sigmoid and a learned residual scale initialized to 0.1. The spatial output is a residual representation, not constrained to [0,1]. Displayed activation maps are the actual sigmoid branch values.
- Shared candidate head combines affected-column pooling, global pooling, 82 context features and 96 candidate features, then 128 sigmoid → 64 sigmoid → linear ranking logit.
- Linear scalar value head and 16 completion logits, transformed to sigmoid probabilities for visualization. BCE-with-logits is used for stable auxiliary training.

Context contains the current and already-visible next piece, observed remaining-bag membership, layer occupancy/buried-gap counts and basic board/target features. It never inspects the hidden piece-stream bag order. Candidate features include the complete normalized shape coordinates/presence, origin, extents and exact engine-derived deltas.

## Candidate budget and safety

CPU generation examines the configured sampled candidate pool (2,048 by default), plus vertical-I finishing anchors, and uses the legacy exhaustive fallback when necessary. Physics-illegal actions are excluded by the engine. By default, the current-target protection heuristic also excludes moves, separately from physical legality.

The learned model ranks an **explicit shortlist of up to 128**: half are the best sustained-objective candidates and half are a deterministic diverse sample of the remaining legal pool. The full teacher winner is always present. This is a hybrid reranker, not exhaustive learned ranking of all placements, and not a promise of identical moves to the teacher. Shortlist size, original candidate budget, protection mode and chunk size are saved with the checkpoint. The normal game uses the checkpoint's candidate budget for neural playback. GPU chunking does not further drop candidates from that shortlist.

The selected move's `Candidate.score` remains the legacy heuristic score for engine consistency; `Decision.search_score` is the learned logit. Raw logits are not probabilities. The GUI provides separate **Learned sigmoid policy** and **Legacy heuristic** tabs, with actual per-channel W-section maps, selected-candidate 128/64 activations, value and layer-completion probabilities. The learned trace refers to the board before its labelled decision, rather than pretending to be the current board after subsequent moves. Human mode and pause still work; all 16 section views remain visible.

## Measurement and validation

Evaluation compares the saved neural policy with sustained, on identical seeds and the same original sampled-candidate budget. It reports clears, survival, piece/clear throughput, agreement, stop reason and checkpoint identity. Neural inference includes CPU preparation and GPU-to-host trace transfer; it is not a pure CUDA-kernel benchmark. There is no automatic promotion to the no-argument default. Require independent long-game results before deployment.

```bash
.venv/bin/python tests/test_neural.py --output neural-tests.json
.venv/bin/python tetris_neural.py smoke --device cuda --output neural-cuda-smoke.json
```

The test suite checks occupancy preservation, no hidden-future leakage, teacher retention, legal rigid drops, real layer compaction, masking, chunk equivalence, sigmoid gradients/optimizer updates, checkpoint round-trip and schema rejection, restoration of Python and PyTorch CPU random state, and terminal/truncation bootstrapping. CUDA checks run only when a device is present and are explicitly skipped otherwise. Linux/Xvfb tests cover GUI inference, all 16 views, both trace tabs, pause/step/Human controls and continued post-clear retargeting. CPU smoke updates do not establish GPU performance or stronger play.

Primary API references: PyTorch Conv3d (`https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html`), AMP (`https://docs.pytorch.org/docs/stable/amp.html`), CUDA timing (`https://docs.pytorch.org/docs/stable/notes/cuda.html`) and restricted checkpoint loading (`https://docs.pytorch.org/docs/stable/generated/torch.load.html`). CUDA under WSL uses the Windows driver (`https://docs.nvidia.com/cuda/wsl-user-guide/index.html`); this implementation does not install or modify it.
