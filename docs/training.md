# Trained shared-sigmoid checkpoint

The previous model was untrained. This is the first actual gameplay-based training pass, not additional compression or a claim that all 73,218 logical connections were trained independently.

## Use the checkpoint

```bash
python sparse_tetris_4d_optimized.py --weights models/trained-sigmoid.json
```

Without `--weights`, the original baseline remains available unchanged. Checkpoints are bounded, schema-checked JSON, not executable pickle files. Loading rebuilds sigmoid/output tables and invalidates model-dependent caches. The UI displays the loaded model's actual activations and threshold. The 16x16x16x16 board, 15 rigid pieces, sigmoid activation, and normal search defaults remain unchanged.

## Training

Cross-entropy policy search optimized three shared parameters: hole weight, roughness weight, and the sigmoid threshold (the negative hidden bias). Height and layer coefficients were held fixed, as were the +/-1 hidden stencil weights and sigmoid slope. This is a low-dimensional policy optimization, not end-to-end deep reinforcement learning.

Eight generations, 12 population members, eight seeded curriculum episodes per member, and a 128-piece cap produced **768 training rollouts, 72,259 placements, and 56,217,757 candidate evaluations**. The training seed was `20261110`. Training used depth 1 and 512 candidates, while deployment defaults remain depth 2, beam 4, and 2,048 candidates.

All curriculum boards have the full 16^4 dimensions. Four initial-state families are used: empty, rough skyline, and 4x4x4 or 5x5x5 open pockets surrounded by full columns. The nonempty starts are synthetic stress cases, not asserted to be naturally reachable from empty. After initialization the ordinary game rules apply unchanged, including hyperlayer clearing; no walls are refilled.

Fitness is:

`pieces_survived / episode_limit + 0.25 * hyperlayers - 0.25 * added_holes / max(1, cells_placed)`

Training seeds 101-108 are reused across candidates for paired comparisons. Seeds 1001-1008 are validation only. The champion is chosen using training fitness alone; validation is checked afterward. Seeds 10001-10016 and 20001-20006 are final tests and do not participate in optimization or checkpoint selection.

| Parameter | Baseline | Trained |
| --- | ---: | ---: |
| Height | -0.510066 | -0.510066 |
| Holes | -0.356630 | -0.7685650472003598 |
| Hyperlayers | 0.760666 | 0.760666 |
| Roughness | -0.184483 | -0.2629078098390759 |
| Sigmoid threshold | 2.0 | 3.9612832250311807 |

## Results on the frozen checkpoint

| Test suite | Metric | Baseline | Trained |
| --- | --- | ---: | ---: |
| Six unseen empty-board games, 1,000-piece cap, normal search | Mean holes after 1,000 pieces | 139.0 | 63.8333 |
| Same six games | Reached cap | 6/6 | 6/6 |
| Same six games | Total hyperlayers | 0 | 0 |
| Sixteen unseen curriculum starts, 256-piece cap, training search | Mean additional holes | 312.5625 | 292.3125 |
| Same sixteen starts | Mean pieces placed | 177.8125 | 177.8750 |
| Same sixteen starts | Reached cap | 10/16 | 10/16 |
| Same sixteen starts | Total hyperlayers | 10 | 10 |

The default-search runs have **54.08% fewer holes on average**, with a hole reduction on every tested seed. Survival and layer-clearing did not improve in that suite. Stress-set results are mixed per seed, including two cases with worse fitness. These small, capped tests establish improved packing on these samples, not indefinite survival, a universal improvement, or a 54% increase in playing strength.

Raw per-seed comparisons are in `validation/training-heldout.json`; optimization history and validation comparisons are in `validation/training.json`.

## Reproduce or continue

```bash
python tools/train_agent.py
python tools/evaluate_agent.py --weights models/trained-sigmoid.json
python tools/train_agent.py --resume models/trained-sigmoid.json --generations 20
```

The resume command performs 20 additional generations. The checkpoint includes the optimizer distribution, complete random state, and history; resuming preserves the original dataset, seed, population, and search settings. The trainer checks these constraints. It checkpoints atomically after each completed generation. Continuing overwrites the output checkpoint by default; choose `--output models/continued.json --report validation/continued.json` to retain the original.

Use `--help` for population, episode length, workers, and search settings. NumPy is recommended for training; `--backend python` is supported but slower. Changing search settings is a new experiment, not an exact resume.

## Software validation

The loaded checkpoint passed the existing 67,034-assertion engine suite with Python, NumPy, and auto without site packages. Ten new unit tests passed on both backends, covering parameter validation, cache rebuilding, exact expanded sigmoid traces, stale-trace rejection, checkpoint loading, deterministic curriculum rollouts, and random-state restoration. A two-generation tiny run matched a one-generation run resumed for one more generation exactly in parameters, optimizer distribution, random state, and optimization history. The trained GUI smoke test passed on a Linux virtual display; native Windows GUI was not tested locally.

Reference: Thiery and Scherrer, *Improvements on Learning Tetris with Cross Entropy*, ICGA Journal (2009), DOI 10.3233/ICG-2009-32104. The 4D curriculum and reward here are project-specific; that paper does not validate these results.
