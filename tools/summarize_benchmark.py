from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tetris_benchmark_logging import summarize


def main() -> int:
    parser = argparse.ArgumentParser(description='Summarize explicit logs, or the latest default benchmark session.')
    parser.add_argument('logs', nargs='*', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        paths = args.logs
        if not paths:
            sessions = list((ROOT / 'logs').glob('benchmark-*.jsonl'))
            if not sessions:
                parser.error('No benchmark sessions found under logs/. Run the game first or specify log paths.')
            latest = max(sessions, key=lambda path: path.stat().st_mtime_ns)
            paths = [latest] + [path for path in latest.parent.glob(latest.name + '.*') if path.name.rsplit('.', 1)[-1].isdigit()]
        text = json.dumps(summarize(paths), indent=2, allow_nan=False)
        if args.output:
            args.output.write_text(text + '\n', encoding='utf-8')
        else:
            print(text)
        return 0
    except (OSError, ValueError) as error:
        print(f'Cannot summarize benchmark logs: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
