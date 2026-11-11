from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def diagnostics() -> dict:
    cpu = {}
    path = Path('/proc/cpuinfo')
    if path.is_file():
        for line in path.read_text().split('\n\n', 1)[0].splitlines():
            key, _, value = line.partition(':')
            if key.strip() in ('model name', 'vendor_id', 'flags'):
                cpu[key.strip()] = value.strip()
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'machine': platform.machine(), 'numpy': game._NUMPY.__version__, 'cpu': cpu,
            'native': game.native_backend_info()}


def select(name: str) -> None:
    os.environ['TETRIS_NATIVE_ISA'] = 'scalar' if name == 'native-scalar' else 'auto'
    game.configure_backend('native' if name.startswith('native') else name)


def main() -> int:
    parser = argparse.ArgumentParser(description='Compare NumPy and strict-float native kernels on identical game settings.')
    parser.add_argument('--pieces', type=int, default=300)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('native-benchmark.json'))
    parser.add_argument('--diagnostics-only', action='store_true')
    args = parser.parse_args()
    if args.pieces < 1 or args.repetitions < 1:
        parser.error('Pieces and repetitions must be positive.')
    game.configure_backend('native')
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    report = {'environment': diagnostics(), 'seed': args.seed, 'pieces': args.pieces,
              'repetitions': args.repetitions, 'policy': 'hyperlayer', 'candidates': 2048,
              'timings_include': 'run_game wall time including search, board bookkeeping and validation; no GUI or import/build time',
              'reference_summary_matches': True, 'gameplay': {}, 'batch_prepare_and_score': {}}
    if not args.diagnostics_only:
        modes = ('numpy', 'native-scalar', 'native')
        measurements = {mode: [] for mode in modes}
        reference = None
        select('numpy')
        game.run_game(args.seed, 20, 2, 4, 2048, 'hyperlayer')
        for repetition in range(args.repetitions):
            for mode in modes[repetition % 3:] + modes[:repetition % 3]:
                select(mode)
                started = time.perf_counter()
                result = game.run_game(args.seed, args.pieces, 2, 4, 2048, 'hyperlayer')
                elapsed = time.perf_counter() - started
                measured = {k: v for k, v in result.items() if k not in ('backend', 'mean_decision_ms', 'wall_seconds')}
                if reference is None:
                    reference = measured
                if measured != reference:
                    raise AssertionError(f'Game summary differs in {mode}; performance result rejected.')
                measurements[mode].append(elapsed)
                print(f'{mode}: {elapsed:.4f}s, {result["pieces"]} pieces, {result["hyperlayers"]} clears', file=sys.stderr)
        base = statistics.median(measurements['numpy'])
        for mode, values in measurements.items():
            median = statistics.median(values)
            report['gameplay'][mode] = {'seconds': values, 'median_seconds': median,
                                       'speedup_vs_numpy': base/median}
        report['final_summary'] = reference
        board = tuple((1 << ((i*17+i//19) % 5)) - 1 for i in range(game.AREA))
        specs = {piece: tuple(game.specifications(piece, 512, args.seed)) for piece in game.PIECE_NAMES}
        for mode in modes:
            select(mode)
            context = game.evaluate_board(board)
            samples = []
            for _ in range(args.repetitions):
                start = time.perf_counter()
                for _ in range(20):
                    for piece in game.PIECE_NAMES:
                        game.PlacementBatch(piece, specs[piece]).scores(context)
                samples.append((time.perf_counter()-start)/(20*len(game.PIECE_NAMES))*1000)
            report['batch_prepare_and_score'][mode] = {'median_ms_per_512_candidates': statistics.median(samples), 'samples_ms': samples}
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
