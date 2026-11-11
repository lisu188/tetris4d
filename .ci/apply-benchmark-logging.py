from pathlib import Path
import hashlib

p = Path('sparse_tetris_4d_optimized.py')
s = p.read_text(encoding='utf-8')
b = p.read_bytes()
assert hashlib.sha1(f'blob {len(b)}\0'.encode() + b).hexdigest() == '683d87c038777e0effae438b8ae5489b453ee727'


def replace(before, after):
    global s
    assert s.count(before) == 1, (before, s.count(before))
    s = s.replace(before, after)


replace('\n\ndef main(argv: Sequence[str] | None = None) -> int:', '''

def start_benchmark_logging(args):
    if args.no_benchmark_log:
        return None
    try:
        from tetris_benchmark_logging import BenchmarkLog
    except ImportError as error:
        print(f'Benchmark logging unavailable: {error}', file=sys.stderr)
        return None
    settings = {key: getattr(args, key) for key in ('seed', 'piece_set', 'policy', 'target', 'depth',
                 'beam', 'candidates', 'stop_after_clear', 'backend')}
    settings['mode'] = 'benchmark' if args.benchmark else 'gui'
    if args.benchmark:
        settings.update(pieces=args.pieces, games=args.games)
    recorder = BenchmarkLog(sys.modules[__name__], args.benchmark_log, settings).install()
    if recorder.enabled:
        print(f'Benchmark log: {recorder.path}', file=sys.stderr)
    return recorder


def main(argv: Sequence[str] | None = None) -> int:''')
replace("    parser.add_argument('--gui-smoke', action='store_true')\n    args = parser.parse_args(argv)", "    parser.add_argument('--gui-smoke', action='store_true')\n    logs = parser.add_mutually_exclusive_group()\n    logs.add_argument('--benchmark-log', type=Path, help='Write local benchmark JSONL here; default: a unique file under logs/. Existing files are not overwritten.')\n    logs.add_argument('--no-benchmark-log', action='store_true', help='Disable all benchmark logging and instrumentation for clean timing comparisons.')\n    args = parser.parse_args(argv)")
replace("    try:\n        configure_backend(args.backend)", "    recorder = None\n    try:\n        configure_backend(args.backend)")
replace("        elif args.benchmark:\n            runs = []", "        elif args.benchmark:\n            recorder = start_benchmark_logging(args)\n            runs = []")
replace("        else:\n            return launch_demo(args)", "        else:\n            recorder = start_benchmark_logging(args)\n            return launch_demo(args)")
replace("    except (ValueError, AssertionError, OSError) as error:\n        print(f'Error: {error}', file=sys.stderr)\n        return 1", "    except (ValueError, AssertionError, OSError) as error:\n        if recorder is not None:\n            recorder.emit('error', phase='application', exception=type(error).__name__)\n        print(f'Error: {error}', file=sys.stderr)\n        return 1\n    finally:\n        if recorder is not None:\n            recorder.close()")
p.write_text(s, encoding='utf-8')
p = Path('.gitignore')
s = p.read_text(encoding='utf-8')
if '\nlogs/\n' not in '\n' + s:
    p.write_text(s.rstrip() + '\nlogs/\n', encoding='utf-8')
p = Path('README.md')
s = p.read_text(encoding='utf-8')
s += '''

## Automatic benchmark logging

Normal launches also write a local JSONL log under `logs/` (the path is printed on stderr).
No additional arguments are required. Logs record runtime/backend/ISA/build metadata,
search active work versus UI scheduling latency, board evaluation, native batch scoring,
commit/validation, activation tracing, GUI draw submission, tick intervals, cache statistics,
and clear/progress events. Phase timings are inclusive and must not be added together.
GUI draw/tick measurements are not compositor FPS. No logs are uploaded automatically.

Each session rotates at approximately 4 MiB with three backups; sessions are kept separately.
`--benchmark-log PATH` selects a new file and `--no-benchmark-log` disables instrumentation.
Tests, catalog/network exports and imported training/benchmark helpers do not log by default.
A file-system logging failure warns and disables logging without stopping gameplay.

```bash
.venv/bin/python sparse_tetris_4d_optimized.py
.venv/bin/python tetris_benchmark_logging.py logs/benchmark-*.jsonl* --output benchmark-summary.json
.venv/bin/python tools/benchmark_logging.py --pieces 1500 --repetitions 3 --output logging-overhead.json
```

The summary groups sessions; submit the summary and, for detailed investigation, the matching
JSONL segments. Logging has overhead: the paired benchmark measures it rather than subtracting
an assumed cost. Recent p50/p95/p99 values cover the last 2,048 observations per phase.
See `docs/benchmark-logging.md` for fields, privacy and timing limitations.
'''
p.write_text(s, encoding='utf-8')
