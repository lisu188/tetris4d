from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tetris_benchmark_logging import BenchmarkLog


def normalized(result):
    return {key: value for key, value in result.items() if key not in ('wall_seconds', 'mean_decision_ms')}


def main():
    parser = argparse.ArgumentParser(description='Measure the overhead of benchmark instrumentation on matching seeded games.')
    parser.add_argument('--pieces', type=int, default=300)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--backend', choices=('auto', 'python', 'numpy', 'native'), default='auto')
    parser.add_argument('--output', type=Path, default=Path('logging-overhead.json'))
    args = parser.parse_args()
    if args.pieces < 1 or args.repetitions < 1:
        parser.error('Pieces and repetitions must be positive.')
    measurements = {'off': [], 'on': []}
    reference = None
    log_paths = []
    environment = None
    for repetition in range(args.repetitions):
        for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
            game.configure_backend(args.backend)
            game.load_checkpoint(ROOT / 'models' / 'trained-sigmoid.json')
            game.run_game(args.seed, 20, 2, 4, 2048, 'hyperlayer')
            for function in (game.evaluate_board, game.target_layer_status, game.pooled_histograms, game.sigmoid_roughness):
                function.cache_clear()
            log = None
            try:
                if enabled:
                    log = BenchmarkLog(game, settings={'mode': 'overhead_test', 'pieces': args.pieces,
                        'seed': args.seed, 'repetition': repetition, 'candidates': 2048}).install()
                    if not log.enabled:
                        raise RuntimeError('Logging failed; overhead comparison would be invalid.')
                    environment = log.metadata
                    log_paths.append(str(log.path))
                start, cpu = time.perf_counter_ns(), time.process_time_ns()
                result = game.run_game(args.seed, args.pieces, 2, 4, 2048, 'hyperlayer')
                elapsed, consumed = time.perf_counter_ns() - start, time.process_time_ns() - cpu
            finally:
                if log is not None:
                    log.close()
            summary = normalized(result)
            if reference is None:
                reference = summary
            if summary != reference:
                raise AssertionError('Logged and unlogged game summaries differ; timing comparison rejected.')
            name = 'on' if enabled else 'off'
            measurements[name].append({'wall_seconds': elapsed / 1e9, 'process_cpu_seconds': consumed / 1e9})
            print(f'logging={name} repetition={repetition} wall={elapsed/1e9:.6f}s', file=sys.stderr)
    medians = {key: statistics.median(row['wall_seconds'] for row in value) for key, value in measurements.items()}
    report = {'environment': environment, 'seed': args.seed, 'pieces': args.pieces,
              'repetitions': args.repetitions, 'raw': measurements, 'median_wall_seconds': medians,
              'overhead_percent': (medians['on'] / medians['off'] - 1) * 100,
              'summary_matches': True, 'final_summary': reference, 'logs': log_paths,
              'measurement_scope': 'Headless run_game including per-move instrumentation and in-run file flushes. Imports, warmup, logger startup and final close are excluded. Ordering alternates. Checks game summaries, not every move; use logging regression tests for trajectory checks.'}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
