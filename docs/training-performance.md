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
100 ms of full preparation, before integration overhead. The production backend
therefore remains native CPU; the GPU prototype is retained locally for research.
