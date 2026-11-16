# Training performance

The native backend computes sustained candidate metrics in C without pruning or
reordering the neural policy's candidate set. NumPy remains the reference path.
Candidate features reuse immutable shape encodings and compute shortlisted
metrics as a batch. Buried targets retain their scalar gap-feature behavior.
Batch collation skips padding when the candidate count already matches and
validates candidates before transferring tensors to the device.

Measured on the four saved training environments, with 2,048 candidate samples
and a 128-candidate shortlist:

| Workload | Reference | Optimized | Ratio |
| --- | ---: | ---: | ---: |
| Sustained metrics, 240 batches / 122,880 rows | 39.16 ms | 19.94 ms | 1.96x |
| Prepare 60 board/piece combinations, CPU time | 334.67 ms | 298.63 ms | 1.12x |
| Collate 100 batches of 16 states, elapsed time | 116.60 ms | 13.60 ms | 8.58x |

These are isolated measurements, not total training speedups. The metrics test
uses seven alternating repetitions. Preparation uses one Torch thread and CPU
time to exclude scheduling delays while training continues in another process.
Prepared arrays and candidate ordering matched the reference exactly. The
native regression suite also covers all orientations, targets, illegal landings,
layer clears, boundaries and exhaustive fallback.

For long CUDA rollouts, opt in to a bounded cache of immutable PPO inputs:

```bash
python train_survival_v2.py --envs 16 --batch 16 --optimizer-batch 128 \
  --cache-rollout --rollout-cache-mib 1024 # add the usual checkpoint/run settings
```

The cache is rebuilt for each update and released afterward. It preserves the
original candidate padding, shuffled sample order, losses, optimizer steps and
KL checks. If its estimate exceeds the configured limit or a quarter of free GPU
memory, training uses ordinary collation. Allocation failures also fall back
before any optimizer step. Update telemetry records whether caching was enabled.

On an RTX 4060 Ti, two fixed 128-state update repetitions averaged 247.85 ms
without the cache and 214.38 ms with it (1.16x). Model parameters, gradients and
Adam state matched the previous trainer exactly at batch 16. Larger microbatches
were faster but changed that trajectory beyond the comparison tolerance, so the
continuation retains batch 16 and optimizer batch 128. This small benchmark does
not establish full-run throughput or playing strength.

The full 16-environment preflight completed one 8,192-placement rollout and 192
optimizer steps with caching enabled. Its input cache used 700,252,160 bytes;
peak Torch allocation was 871 MiB (952 MiB reserved). All model/optimizer tensors
were finite and all sixteen saved game states replayed exactly. See
[`validation/ppo-performance.json`](../validation/ppo-performance.json).

The small benchmark reports and source hashes are retained locally under
`survival-training.artifacts/validation/`. No original training checkpoints or
telemetry are replaced during performance experiments.

The experimental GPU physics benchmark matched integer landing/clear/commit
results and non-clearing FP64 scores. Clearing candidates still require CPU
rescoring. For sixteen groups of 2,048 candidates, native scoring took 3.27 ms;
GPU scoring plus transfers took 0.99 ms. That saves about 2.28 ms within roughly
100 ms of full preparation, before integration overhead. This physics-only
measurement motivated the broader optional preparation backend below. The
production experiment still uses its frozen native CPU runtime.

## Optional GPU preparation

New runs can add `--gpu-prepare --fused-encoding --coalesce-transfers` to the
[PPO recipe](ppo-workflow.md). These flags are recorded in the run signature;
changing them requires a separate run directory. The running 24-hour experiment
keeps its frozen runtime and continues on the native CPU preparation path.

`--gpu-prepare` requires CUDA, Triton and the native reference backend. It uploads
the sixteen column boards together, then computes board statistics, candidate
landings, strict FP64 non-clearing scores, stable shortlists, teacher priors and
model inputs on the GPU. The policy consumes those tensors directly, and only
selected move metadata returns to the CPU. Exact Python RNG draws and a small
count-summary transfer preserve the original candidate order and exploration.

Selected moves still commit on the CPU, including replay hashes and piece-stream
updates. Clears, buried/full targets and exhaustive/recovery cases use the native
preparer for the affected environment. Candidate budgets above 4,096, shortlists
above 256 or catalog shapes touching more than eight columns use the reference
path. This is a GPU preparation backend; complete GPU game progression remains
follow-up work.

Rollout records own their storage through the PPO update. The trainer estimates
retained inputs and the optional input cache before collection, releases retired
records before the next rollout, and falls back to CPU preparation when memory
is insufficient. Allocation fallback does not retry policy draws or optimizer
steps. Telemetry reports the enabled path and fallback reasons.

`--fused-encoding` builds the existing 34 integer-board feature channels in one
CUDA kernel. CPU and other input layouts keep the original implementation.
`--coalesce-transfers` batches scalar diagnostics, KL reads and inference output
transfers while retaining Python accumulation order and the original sampling
draws. The flags are independent and disabled by default. Triton is imported only
when a CUDA opt-in path needs it; CPU-only PyTorch remains supported.

Exact local checks cover forty frozen preparation cases, all 65,536 column masks,
target boundaries, mixed CPU/device records, record lifetime, optional protection,
odd shortlists and fallbacks. Encoder model outputs/gradients and PPO actions,
gradients, Adam state, metrics and RNG matched the reference. Development probes
used one thread, low CPU priority and a 512 MiB Torch allocator cap alongside the
same live trainer. CUDA context memory and shared-device contention remain
outside that allocator cap. These checks establish bounded parity, not learned
survival gains or total training throughput.

Four alternating warmed pairs compared full preparation on four frozen real
boards with sixteen distinct piece/counter contexts:

| Workload | Native prepare, collate and H2D | GPU prepare and selected moves D2H | Ratio |
| --- | ---: | ---: | ---: |
| Sixteen environments, median wall time | 76.89 ms | 22.51 ms | 3.42x |

Outputs matched exactly. These measurements include shared-device contention
while training continued, constructed counter contexts, and no CPU-fallback
cases. They do not measure complete PPO rollouts or training throughput. A
separate repeated-context workload is recorded in
[`validation/gpu-preparation.json`](../validation/gpu-preparation.json).

A second comparison timed the complete short rollout and PPO update together:
sixteen environments, four steps (64 transitions), the full 32-channel/three-block
model, FP32, batch 16, optimizer batch 128 and three PPO epochs. After one warmup,
three alternating paired runs measured median wall time of **1.233 s to 0.784 s
(1.57x)**. Timing includes preparation, transfers, cache construction and KL checks;
it excludes checkpoint loading and setup. Model parameters, gradients, Adam state
and CPU/CUDA RNG matched exactly for every pair. This is a concurrent
64-transition microbenchmark; a full 8,192-transition GPU rollout and sustained
training throughput remain unmeasured. The active training process was unchanged
and continued advancing during every successful probe.
