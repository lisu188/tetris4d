# Sustained clearing

Plain launch selects Unlimited (initially paused) and the sustained planner. Press Start/Space to begin. All 16 section views, adaptive render-only frame skipping and local benchmark logging are retained. Native C and trained sigmoid parameters are unchanged; a matching existing native library requires no rebuild.

The previous planner optimizes immediate target coverage, then the sigmoid score. It protects only the current target: missing cells above that target can become buried, and subsequent target levels drift toward the height limit. The new planner retains the same legal candidate generator, 2,048 sampled budget, targeted vertical-I finishing moves and target protection. Its lexicographic objective is:

1. Maximize newly filled target cells minus 4 times newly created holes strictly ABOVE the target.
2. Break ties using the unchanged binary64 sigmoid-network score.
3. Keep original enumeration order for exact ties.

The cost is deliberately soft. A strict prohibition on new upper holes stalled on seed 7 at 1,218 pieces, before its first clear, with a CROSS piece. The soft penalty allows necessary compromises instead of turning a useful heuristic into an impossible placement rule.

For each touched column, let h be its original height, h' its new height and A its newly added occupancy mask. The number of new holes above target t is max(0, h' - max(h, t+1)) minus popcount(A >> (t+1)). This counts occupancy gaps, not visual empty space. Tests compare it with independent cell-by-cell reconstruction and compare scalar/vectorized decisions. Native scoring is called without legacy highest-gain filtering, because that filter would incorrectly remove candidates that win the new objective.

This changes placement selection, not physics or learned weights. Raw Candidate.score and activation displays still describe the actual sigmoid network, not the target-planning objective. The original policy remains available as --policy hyperlayer; --policy network keeps neural-only lookahead. The new policy is sustained, not a recurrent neural network.

## Evaluation

The penalty was frozen after development on seeds 7 and 30001. Four distinct evaluation seeds (40001-40004) used a 16,000-piece cap, identical piece bags, checkpoints and candidate budgets. The local NumPy results were 36 total clears for the old controller versus 51 for sustained clearing. Two sustained games reached the cap; no baseline game did. This small deterministic test set supports an improvement, not indefinite survival or universal superiority. See validation/sustained-policy-ci.json for the independent repository-native rerun and raw per-seed outcomes.

Run the paired comparison:

```bash
.venv/bin/python tools/evaluate_sustained.py --output sustained-evaluation.json
```

It compares complete games, not just the first clear. Separate playing strength (layers and survival) from throughput (pieces/second). Concurrent execution timings are not suitable for backend speedup claims. The first-clear diagnostic and legacy regression trajectories are unchanged.
