# Archived Survival V5 checkpoints

Survival V5 checkpoints combine a frozen V4 actor with a bounded next-state scorer. The archived experiment records the training settings and input requirements for reproducing existing checkpoints. [Original V5 protocol](https://github.com/lisu188/tetris4d/blob/23159fe8d860a13ab16b8d5b69f2abf868522c82/docs/survival-v5.md)

## Why
Evidence from the V4 runs:

- **The V4 actor's raw head is unbounded and drifts under PPO.** On an obvious clearing move, the margin over the best alternative fell from 9.5 to 0.3 in a collapsed run, where clears per game went from 7 to 0. Even a healthy run took it from 5.5 to 1.8.
- **Ranking candidates by the critic's value of the resulting state beat the actor** by 19,472 vs 14,325 placements, on 8/8 seeds (#37).
- **An actor pretrained against the exhaustive teacher is already strong**: greedy play reached ≥ 29,110 placements, with 7 of 8 games capped at 30,000.

The strongest classical Tetris controllers evaluate afterstates with features such as holes, landing height, transitions, wells and eroded cells (Dellacherie–Thiery; CBMPI and the cross-entropy method).

## Policy
```
logits_c = (base_c + R * tanh(beta * (S(f(s'_c)) - mean_c S))) / tau
```
- **`base`**: a frozen V4 actor. Its logits already include the teacher prior. It is excluded from the optimizer.
- **`S`**: a scorer over exact next-state features `f(s')`. The piece stream is deterministic, so a candidate's afterstate is exactly the next state, including the current and next pieces.
- **`R`** is the checkpoint’s maximum residual adjustment in nats before temperature scaling. `tau` sets the temperature for sampled play.
- **Critic `V`**: uses the same features, with separate parameters.

## Features (`tetris_survival_v5.py`, 382 per state)
Each row is `[V4 pooled 266 | extras 34 | context 82]`.

- **V4 pooled block** (column-encoding mean/max, mean adjacency statistics, regional statistics), together with the context, equals the regional V4 critic input bit for bit.
  - So the scorer and the critic start as exact copies of a calibrated V4 critic, with the extras branch zero-initialised.
  - The initial policy is therefore the bounded afterstate ranking from #37.
- **Extras**:
  - adjacency statistics per axis (18)
  - 4D Dellacherie–Thiery survival features (11): holes, hole depth, columns with holes, Z transitions, cumulative and deepest wells over the six horizontal neighbours with walls counted as full, columns at height ≥ 12 and ≥ 14, and height-difference diversity per axis
  - move features (5): layers cleared, eroded piece cells, landing height, target gain, teacher-score delta.
- **Context**: the next state's exact context, bit-identical to `neural.state_context`.

On CUDA, one Triton program per board (`tetris_gpu_board_features.py`) computes the integer statistics behind the pooled, context and survival features. `board_counts_reference` is the torch reference, and both feed the same float conversion. Layer-full detection and compaction run only for candidates that clear.

`shortlist_mode='teacher-top'` keeps the teacher's best K candidates instead of 64 best plus 64 random.

## Benchmark (`tools/survival_benchmark.py`)
Full games now reach the 30k caps, so policies are compared on survival from fixed near-top-out positions: recorded top-out games replayed to 500–3,000 placements before their end.

```
python tools/survival_benchmark.py SOURCE.pt --states 32 --cap 5000 \
    --policies base v5:R:m[:tau][:shortlist] [--sampled]
```
`beta = m / sigma`, where `sigma` is the median spread of the critic's next-state values across shortlisted candidates. This makes `m` scale-free.

## Verification
`tests/test_survival_v5.py` checks:
- **Exactness:** pooled features vs `encode_columns`; the V5 critic at initialisation vs the V4 critic; survival features vs a plain-Python reference; next-state rows vs engine-materialised afterstates, including clears; count-based features vs the reference functions.
- **Policy:** `teacher-top`; a zero bound replays the base exactly; the residual stays within the bound.
- **CUDA, opt-in:** kernel vs reference, device vs CPU rows, `teacher-top` parity, and zero-bound greedy replaying V4's choices.
