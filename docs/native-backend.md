# Native scorer for WSL/Linux

The optional native backend accelerates the existing evaluator and planner without changing sigmoid responses, trained parameters, rigid-piece rules, candidate enumeration, tie breaking, or continuous play. It is not a retrained model. Python/Tkinter remains the application; the new C source is `native/tetris_native.c`.

## Build and run

Run these commands in Bash inside your existing repository checkout in WSL (Ubuntu/Debian):

```bash
git switch main && git pull --ff-only
sudo apt-get update && sudo apt-get install -y build-essential python3-venv python3-tk
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-numpy.txt
.venv/bin/python tools/build_native.py
.venv/bin/python tests/test_native.py --trajectory-pieces 2600
.venv/bin/python sparse_tetris_4d_optimized.py
```

No game arguments are required after the build: auto selects the native library, then NumPy, then Python. The native backend requires NumPy for its buffers. There is no automatic compilation or binary download at launch. No hardware-specific binaries are committed.

The builder supports Linux/WSL and macOS with GCC/Clang-compatible compilers. Native Windows Python retains the existing NumPy/Python paths; build and run this backend inside WSL instead. macOS and ARM were not tested locally. The scalar code does not require x86 instructions; AVX2 and BMI2 functions are isolated and runtime-gated on x86. A mismatched source/binary checksum or ABI is rejected; auto reports the problem and falls back, whereas `--backend native` fails explicitly.

Microsoft recommends keeping Linux-command-line projects under the Linux filesystem (for example `~/src/tetris4d`), rather than `/mnt/c`, for filesystem performance. This does not itself speed up an already resident CPU scoring kernel.

## What is implemented

- Fused native candidate scoring: no NumPy neighborhood/roughness temporary tensors or cumulative-sum output.
- Before roughness scoring, the hyperlayer path rejects illegal and target-blocking placements and finds the maximum coverage gain in the current batch. Only candidates that can win that batch's lexicographic comparison are scored. Original order resolves exact ties.
- Packed immutable native board snapshots: 4,096 uint16 columns (8 KiB), 4,096 int32 heights (16 KiB, allowing AVX2 gathers), and 16 int32 layer counts. These live in the existing bounded eight-board evaluation cache. Python remains the canonical board owner; this is not a complete persistent incremental native-state rewrite.
- Native height, hole, fill-count, and buried-gap scans; cached heights are reused for legality checks.
- Two execution paths for landing: scalar, or eight int32 candidates of the same orientation per AVX2 group. The emitted code uses `VPGATHERDD` and `VPMAXSD`, not float32 approximations. Tail candidates use scalar code.
- A 17-by-17 binary64 roughness-difference table, rebuilt by cache key when the model changes. Each candidate's edge accumulation order remains unchanged.
- BMI2 `PEXT` compaction of cleared column bits, independently gated from AVX2, with shift/mask fallback.
- Clear-producing candidates are rescored by the existing reference path. The logical sigmoid graph and activation display are unchanged.

The default flags are `-O3 -fno-fast-math -ffp-contract=off`. There is no global `-march=native`, reassociation, float32 conversion, or approximation of sigmoid. CPU feature detection checks usable AVX2 before entering the specialized function.

## Measure on your machine

```bash
.venv/bin/python tools/benchmark_native.py --pieces 1500 --repetitions 3 --output wsl-native-benchmark.json
.venv/bin/python tools/benchmark_native.py --diagnostics-only --output wsl-native-diagnostics.json
objdump -d native/libtetris4d_native.so | grep -E 'vpgatherdd|vpmaxsd|pext'
```

The benchmark compares NumPy, native scalar, and native auto on identical settings and checks matching game summaries. For strict trajectory equivalence, run the separate native test suite. It records CPU model/flags, Python/NumPy versions, native capabilities, and raw timings. Upload the JSON report for hardware-specific tuning; no credentials or API keys are needed.

Force a path for diagnosis:

```bash
TETRIS_NATIVE_ISA=scalar .venv/bin/python sparse_tetris_4d_optimized.py --backend native
TETRIS_NATIVE_ISA=avx2 .venv/bin/python sparse_tetris_4d_optimized.py --backend native
.venv/bin/python sparse_tetris_4d_optimized.py --backend numpy
```

`scalar` disables both optional instruction paths; `avx2` fails explicitly when AVX2 is unavailable. BMI2 is always checked separately. The default `auto` uses supported optional instructions, but they are not assumed to improve every CPU or workload.

## Local measurements and validation

On the local AMD EPYC 9V74 Linux environment, seed 7, 300 pieces, 2,048 candidates, five repetitions with rotating backend order:

| Operation | NumPy | Native scalar | Native auto |
| --- | ---: | ---: | ---: |
| Complete headless game, median | 2.0523 s | 0.9552 s | 0.9579 s |
| Batch preparation + scoring, 512 candidates, median | 0.6150 ms | 0.1514 ms | 0.1655 ms |

The whole-game improvement is about 2.1x. AVX2 was not faster than native scalar on this sample; the major gain is fused execution and reduced allocation. The batch test measures preparation plus scoring for a fixed synthetic board, across all 15 pieces; it excludes board evaluation and GUI work. These are not WSL-machine measurements or a whole-application FPS claim. Raw results are in `validation/native-backend.json`.

Tests cover 109,728 exact per-candidate score comparisons, all 936 orientations, boundaries and SIMD tails, four model configurations, invalid inputs, target-filter winners, and every 16-bit column mask against seven clear masks on scalar and dispatched compaction. The 2,600-move NumPy/native trajectory comparison preserves each selected candidate, score, evaluation count, resulting board and next piece, including clears at 1,251 and 2,478 and continued play. The neural-only two-ply path is compared separately on all three piece sets. The existing 67,034-assertion engine suite and a Linux virtual-display GUI smoke test also pass. An undefined-behavior-sanitized build passed the native unit suite.

To run the sanitizer locally, then restore the optimized build:

```bash
.venv/bin/python tools/build_native.py --sanitize && .venv/bin/python tests/test_native.py
.venv/bin/python tools/build_native.py
```

References: GCC optimization options (`https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html`), GCC x86 function attributes (`https://gcc.gnu.org/onlinedocs/gcc/x86-Function-Attributes.html`), and Microsoft's WSL filesystem guidance (`https://learn.microsoft.com/en-us/windows/wsl/filesystems`).
