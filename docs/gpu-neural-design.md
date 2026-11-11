# Proposed GPU-ready sigmoid policy

Status: architecture proposal, not a trained/deployed replacement. The shipped default is the sustained native/NumPy/CPU planner. No torch import, GPU driver installation or mandatory dependency is added to normal launch.

## Why a different representation

The current shared sigmoid network sees 4,096 heights, total holes and cleared layers. Boards with equal inputs but different buried occupancy receive equal scores. More optimization of the same weights cannot recover missing geometry. The new policy's upper-hole cost is an immediate heuristic fix, not a learned long-horizon model.

## Column-volume encoder

Represent the X/Y/W footprint as a 16x16x16 spatial tensor. Use 34 channels: 16 Z-occupancy planes, 16 buried-gap planes, normalized column height, and whether the current target cell is vertically accessible. Gravity Z is represented explicitly in channels, rather than mistaken for the fourth axis of a stock Conv3d. Input shape is [batch, 34, 16, 16, 16]. No voxel occupancy is discarded.

A 1x1x1 stem maps 34 channels to 32. Three residual blocks use depthwise 3x3x3 convolution (32 groups), GroupNorm, sigmoid, and pointwise mixing. A small residual scale limits early saturation. Hidden nonlinearities are sigmoid only: no ReLU, GELU or SiLU. This is a design choice to test, not a claim that sigmoid is optimal.

Encode the board once. For every candidate, pool trunk features over its affected columns and concatenate global features, exact candidate deltas, placement coordinates/orientation, current/visible-next-piece embeddings, remaining-bag membership, target Z, per-layer occupancy and buried-gap counts. Bag membership comes from observed history, not hidden future bag order. Use a shared 128 -> 64 sigmoid ranking head and a linear logit. Add a scalar value head and a 16-output sigmoid auxiliary head predicting near-future layer completion. Exact count/feature widths determine parameter count; target roughly 50,000-150,000 parameters, not millions.

The engine still supplies rigid placements and an exact legality mask. Physics-invalid actions are masked before softmax; heuristic target restrictions must be reported separately from legal actions. For deployment, rank a CPU-generated shortlist initially, increasing its size only when evaluations justify the cost. For training, batch candidates and multiple games to reuse the board encoding. Never run 2,048 full-board convolutions for one decision.

## Training and evaluation

Start with imitation of successful multi-clear trajectories, followed by long-horizon policy/value optimization. Optimize actual repeated clears and survival, not only holes or first-clear success. Episodes must continue through multiple clears; 128 empty-board moves cannot fill a 4,096-cell layer, even with 16-cell pieces. Use 16,000-32,000-piece episode limits, bootstrapped rollout chunks and proper truncation handling. An undiscounted finite-horizon objective is a simple baseline; for discounted training, gamma=0.9995 is a starting experiment rather than the usual 0.99. Potential-based shaping, if used, must be gamma*Phi(next)-Phi(current) with terminal potentials handled consistently. Do not repeatedly reward unchanged partial coverage.

Measure held-out clears, first/next-clear latency, capped survival, terminal holes, bad-action rate, candidate-ranking quality, and wall-clock pieces/second. Split by seed and evaluate from empty and mid-game states. Keep a CPU teacher/reference and deploy a learned model only after it wins on independent long games.

## GPU deployment

Use GPU first for minibatch training and multi-environment inference. Keep the existing small shared-lookup evaluator on CPU: the supplied older GUI log spent about 30.5 seconds drawing versus 0.99 seconds in active search, and only about 0.094 seconds in native batch preparation/scoring. These nested timings must not be added. The log predates the newer all-section frame pacing/renderer and is not a measurement of current Unlimited performance.

For the proposed model, transfer packed occupancy once per board, unpack on device, reuse the encoding, batch candidate heads and avoid per-candidate device synchronization. Use inference_mode for deployment. Treat torch.compile, CUDA graphs and mixed precision as measured options, not assumed wins. Benchmark warmed execution with CUDA events/synchronization plus separate end-to-end transfer timing; CPU launch timing alone is misleading. Keep exact integer collision handling on CPU and test mixed precision near decision ties separately.

CUDA under WSL2 uses the Windows NVIDIA driver. Do not install a Linux display driver inside WSL. The actual GPU/VRAM is not present in the supplied log; the CPU model does not establish GPU compatibility. Run:

```bash
.venv/bin/python tools/gpu_probe.py --output gpu-info.json
```

The probe reports NVIDIA and optional PyTorch CUDA information without installing anything or uploading data. A missing torch package or unavailable CUDA is reported, not silently presented as GPU support. AMD/Intel GPU support needs a device-specific backend assessment before installation advice.

## Primary references

- NVIDIA CUDA on WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- PyTorch Conv3d, shapes and depthwise groups: https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv3d.html
- PyTorch CUDA synchronization/timing: https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch performance tuning: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Huang and Ontanon, invalid action masking: https://arxiv.org/abs/2006.14171
- Ng, Harada and Russell, potential shaping: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- Gabillon, Ghavamzadeh and Scherrer, policy search in Tetris: https://papers.nips.cc/paper_files/paper/2013/hash/7504adad8bb96320eb3afdd4df6e1f60-Abstract.html

The sources support the components, not the untested claim that this architecture will solve 4D Tetris.
