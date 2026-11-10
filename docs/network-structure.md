# Compact shared-sigmoid network

This is lossless structural compilation of the existing fixed evaluator, not training, approximate pruning, or a reduction of the search budget.

The expanded logical network remains **4,098 inputs -> 23,040 sigmoid activations -> one linear output**, including 73,218 weighted connections. Its 11,520 neighboring-column pairs all use the same two filters:

- `u = sigmoid(h_b - h_a - 2)`
- `v = sigmoid(h_a - h_b - 2)`

The implementation now stores those filters implicitly instead of retaining 23,041 `Node` objects and their repeated edges. `SharedSigmoidNetwork` reconstructs any requested logical node; full expansion remains available for inspection and export. This does **not** mean that the mathematical model has only two neurons.

## Exact pooling

Column heights are integers in `[0, 16]`. Let `C[k]` count neighboring pairs for which `abs(h_b - h_a) == k`. The roughness response is exactly:

`6 * sum(C[k] * (sigmoid(k - 2) + sigmoid(-k - 2) - 2 * sigmoid(-2)) for k in range(17))`

Thus the expanded hidden responses can be pooled into **17 difference bins**, using **33 shared sigmoid lookup values**. A separate 17-bin height histogram preserves the direct height contributions. The old version already used sigmoid lookup tables; this change removes graph duplication and unnecessary full activation materialization, rather than claiming to eliminate 23,040 exponential calls that were not occurring.

The pooled sums use exact integer representations of the existing binary floating-point terms and round once. This avoids changes caused by grouping repeated float additions. The tested outputs match the previous `math.fsum` evaluation exactly. Arbitrary finite, nonintegral feature inputs use the literal sigmoid fallback.

An uncached board still requires examining all neighboring heights. Search continues to use its existing local feature-delta updates. Pure-Python roughness evaluation retains the existing `math.fsum` path because it measured faster than building a histogram in Python.

## Lazy inspection

`NetworkTrace` stores the inputs, pooled counts, score, and active-unit count. It reconstructs a requested hidden activation from its neighboring heights. The UI therefore no longer allocates all 27,139 values for each selected move. All W pages, hover formulas, signed connections and pinned inspection remain available; the score always covers the entire board.

`network_forward()` remains a compatibility API returning the complete list. `--network` now exports compact metadata. Use `--network-expanded` to request the complete logical graph.

## Measurements

Compared with commit `b3f2e8b4eea26f2f3fdeb18a2c7c4666103ffc0a`, on Linux / CPython 3.13.5, NumPy backend, 20 seeded height maps, median of five repetitions:

| Operation / retained data | Previous | Compact |
| --- | ---: | ---: |
| Cold activation trace preparation used by UI | 1.514 ms | 0.389 ms |
| Full expanded trace, cold | 1.514 ms | 1.141 ms |
| Trace data reachable from the trace object | 895,680 bytes | 34,768 bytes |
| Repeated neuron/edge graph object | 11,450,469 bytes | 32-byte lazy proxy |

The graph-object row excludes shared neighbor topology, lookup tables, and caches; it is **not total process memory**. Cold runs clear feature caches, not imports. Lazy trace preparation excludes actually inspecting every hidden activation; the full-expanded row measures that separate operation. These are network measurements, not whole-game FPS or search-speed claims. Warm lazy trace preparation measured 0.021 ms. The dependency-free backend remains supported.

## Verification and reproduction

Both backends matched 2,659,622 compared activation values across 49 integral feature sets each, plus fractional fallback checks. Across classic, expanded and 4D-only piece sets, 150 compared moves retained identical choices, boards, search scores and evaluation counts. Each existing engine suite passed 67,034 assertions. UI tests covered all 27,139 logical nodes across 16 W pages on a Linux virtual display. Native Windows GUI was not tested locally.

```bash
python sparse_tetris_4d_optimized.py --test --backend python
python sparse_tetris_4d_optimized.py --test --backend numpy
python tests/test_network_structure.py --backend python
python tests/test_network_structure.py --backend numpy
python tools/benchmark_network_structure.py --backend numpy --output network-benchmark.json
python sparse_tetris_4d_optimized.py --network --output compact-network.json
python sparse_tetris_4d_optimized.py --network-expanded --output expanded-network.json
```

The benchmark loads the pinned baseline through local Git history. `--baseline /path/to/baseline.py` is available when Git history is unavailable. The application itself remains one self-contained Python file; no new runtime dependency is required.

Implementation references: [NumPy bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html) and [Python math.fsum](https://docs.python.org/3/library/math.html#math.fsum).
