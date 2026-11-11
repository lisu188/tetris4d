# Local benchmark logging

Running `.venv/bin/python sparse_tetris_4d_optimized.py` enables logging automatically.
The file location is printed to stderr: `logs/benchmark-<UTC>-<pid>-<session>.jsonl`,
relative to the script directory, not the current working directory. Each record is one
JSON object. The game still loads the trained hyperlayer policy and prefers a compatible
native library without arguments; no physics, model or rendering changes are made here.

## Measurements

The session header records Python/NumPy/platform, CPU model/flags where readable,
selected backend and actual AVX2/BMI2 settings, compiler flags, code/native/binary hashes,
network parameters and game configuration. It deliberately omits the native library's
absolute path, usernames, hostnames, process environment and credentials. Nothing is uploaded.
Absolute logfile paths are printed locally; an overhead report also identifies its output files.

Every completed decision records candidate count, search settings, target, coverage gain,
exhaustive fallback, active wall/process CPU time, and elapsed scheduling latency. Active
search time is measured around each `next()` of the search generator, so it excludes time
spent suspended between GUI callbacks. Elapsed latency intentionally includes that time.

Metrics snapshots are emitted about every five seconds, every 100 committed pieces,
on clears, and at shutdown. They contain inclusive times for board evaluation, native
feature scanning, native batch preparation/scoring, move materialization, commit validation,
activation trace generation, drawing the well/atlas/network, overall GUI drawing and tick work.
Cache hit/miss/size statistics are included. GUI snapshots include the actual speed, paused/Human
state, target, viewed section and cutaway. Per-phase count/total/mean/max cover the session;
p50/p95/p99 use the most recent 2,048 observations, labelled `recent_*`.

**These are inclusive timings:** a native scoring time can also be part of search time.
Do not add nested phase totals. Drawing is CPU-side Tk submission, not compositor completion.
Tick intervals include scheduling and instrumentation; they are not an FPS guarantee.
The reported serialization/write time does not include every instrumentation operation and
is not automatically subtracted. `process_time_ns` is whole-process CPU time, not thread CPU.

Clear/progress events include total pieces, cells and hyperlayers, game-over state and a
rolling SHA-256 digest of committed placements. The digest contains shapes/coordinates/layer
counts; it is a debugging fingerprint, not a proof that every board and score agrees.
GUI and headless game-end records carry final game summaries. Explicit resets create separate
game identifiers. Model/backend changes are captured by game starts and metric snapshots.

## Usage

```bash
.venv/bin/python sparse_tetris_4d_optimized.py
.venv/bin/python sparse_tetris_4d_optimized.py --benchmark --pieces 2600 --games 1 --benchmark-log run.jsonl
.venv/bin/python tetris_benchmark_logging.py logs/benchmark-*.jsonl* --output benchmark-summary.json
.venv/bin/python tools/benchmark_logging.py --pieces 1500 --repetitions 3 --output logging-overhead.json
```

`--no-benchmark-log` disables instrumentation entirely. It is mutually exclusive with
`--benchmark-log PATH`. An existing explicit path is not overwritten: logging warns and
is disabled, while the game continues. Default session filenames are unique. Importing the
game, training helpers, native benchmark tools, tests and network/catalog exports do not
activate logging unless the caller explicitly constructs and installs a recorder.

Logs are buffered and flushed every 64 records, about every five seconds, and on milestones
or shutdown. A forced process termination can lose the unflushed tail. Files rotate at
approximately 4 MiB with three backups; each new segment includes session metadata. Older
segments may have been discarded. Separate sessions are retained separately; delete unwanted
sessions manually. Rotation keeps each session bounded, not the aggregate number of sessions.
`logs/` is ignored by Git. Disk/permission failures warn and disable logging without stopping
play. Summaries tolerate truncated lines and group multiple sessions by session ID.

For profiling comparisons, use the same seed, piece set, candidate budget, weights, backend,
view and speed; do not compare a paused GUI run with a headless game. The supplied overhead
benchmark alternates logged/unlogged ordering, warms both, and rejects mismatched game summaries.
No claim is made that logging is free or that this change accelerates play.

References: Python `time` clock documentation and the logging handler documentation explain
high-resolution elapsed/CPU clocks, buffering and rotation. This recorder uses buffered text I/O
and explicit rotation rather than placing a handler in the native candidate loop.
