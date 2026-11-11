from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def run_task(task):
    seed, policy, limit, backend = task
    game.configure_backend(backend)
    game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    return game.run_game(seed, limit, 2, 4, 2048, policy, 'expanded', 1, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[40001, 40002, 40003, 40004])
    parser.add_argument('--pieces', type=int, default=16000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--backend', choices=('auto', 'numpy', 'native', 'python'), default='auto')
    parser.add_argument('--output', type=Path, default=Path('sustained-evaluation.json'))
    args = parser.parse_args()
    if args.pieces < 1 or args.workers < 1:
        parser.error('pieces and workers must be positive')
    tasks = [(seed, policy, args.pieces, args.backend) for seed in args.seeds for policy in ('hyperlayer', 'sustained')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_task, tasks))
    totals = {policy: {'clears': sum(r['hyperlayers'] for r in runs if r['policy'] == policy),
                       'pieces': sum(r['pieces'] for r in runs if r['policy'] == policy),
                       'capped_games': sum(r['capped'] for r in runs if r['policy'] == policy)}
              for policy in ('hyperlayer', 'sustained')}
    for row in totals.values():
        row['clears_per_1000_played_pieces'] = 1000 * row['clears'] / row['pieces'] if row['pieces'] else 0
    result = {'source_sha256': hashlib.sha256((ROOT / 'sparse_tetris_4d_optimized.py').read_bytes()).hexdigest(),
              'python': platform.python_version(), 'platform': platform.platform(), 'future_hole_penalty': 4,
              'development_seeds': [7, 30001], 'evaluation_seeds': args.seeds, 'piece_limit': args.pieces,
              'runs': runs, 'totals': totals,
              'note': 'Paired playing-strength comparison, not a speed benchmark. Concurrent timing is not comparable to isolated runs. Finite capped games do not establish indefinite survival.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
