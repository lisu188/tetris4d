from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def build():
    game.configure_backend('auto')
    fixtures = []
    for path in sorted((ROOT / 'runs').glob('*/games/*/*.json')):
        recording = json.loads(path.read_text(encoding='utf-8'))
        if not (recording.get('complete') and recording.get('stop_reason') == 'planner_blocked'
                and recording.get('phase') == 'training'):
            continue
        state = game.Game(recording['seed'], recording['piece_set'])
        board, counts, piece_counts = [0] * game.AREA, [0] * game.HEIGHT, {name: 0 for name in game.PIECE_NAMES}
        total = layers = 0
        for row in recording['moves']:
            if row['piece'] != state.current:
                raise ValueError('Recorded piece stream mismatch')
            shape = game.SHAPES[row['piece']][row['rotation']]
            origin = game.column_index(row['x'], row['y'], row['w'])
            z = max(board[origin + offset].bit_length() - bottom for offset, _, bottom in shape.columns)
            if z != row['z'] or z + shape.extents[2] > 16:
                raise ValueError('Invalid rigid landing')
            for dx, dy, dz, dw in shape.cells:
                coordinates = row['x'] + dx, row['y'] + dy, z + dz, row['w'] + dw
                if any(not 0 <= v < 16 for v in coordinates):
                    raise ValueError('Placement out of bounds')
                x, y, level, w = coordinates
                index = game.column_index(x, y, w)
                if board[index] & (1 << level):
                    raise ValueError('Overlapping placement')
                board[index] |= 1 << level
                counts[level] += 1
            full = sum(1 << level for level, count in enumerate(counts) if count == game.AREA)
            if full != row['cleared_mask'] or full.bit_count() != row['clears']:
                raise ValueError('Layer-compaction mismatch')
            if full:
                board = list(game.clear_layers(tuple(board))[0])
                counts = [count for level, count in enumerate(counts) if not full & (1 << level)] + [0] * full.bit_count()
            total += len(shape.cells)
            layers += full.bit_count()
            piece_counts[row['piece']] += 1
            state.current, state.next_piece = state.next_piece, state.stream.pop()
        packed = struct.pack('<4096H', *board)
        digest = hashlib.sha256(packed).hexdigest()
        if digest != recording['board_sha256']:
            raise ValueError('Final recorded board hash mismatch')
        summary = recording['summary']
        if (total != summary['cells_placed'] or layers != summary['hyperlayers'] or piece_counts != summary['piece_counts']
                or total - layers * 4096 != sum(mask.bit_count() for mask in board)):
            raise ValueError('Recorded counters mismatch')
        fixtures.append({'source': path.relative_to(ROOT).as_posix(), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'seed': recording['seed'], 'piece_set': recording['piece_set'], 'target': recording['target'],
                         'current': state.current, 'next_piece': state.next_piece, 'summary': summary,
                         'board_sha256': digest, 'board_zlib_base64': base64.b64encode(zlib.compress(packed, 9)).decode()})
    target = ROOT / 'tests/fixtures/blocked-targets.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'schema': 'tetris4d-target-recovery-fixtures-v1', 'cases': fixtures}, indent=2) + '\n', encoding='utf-8')
    report = {'fixtures': len(fixtures), 'replayed_moves': sum(x['summary']['pieces'] for x in fixtures),
              'bytes': target.stat().st_size, 'checked': 'rigid landings, collisions, piece order, clear masks, final hashes and counters'}
    print(json.dumps(report))
    if len(fixtures) < 35:
        raise ValueError('Expected at least the 35 recorded blocked states from the audit')
    return report


if __name__ == '__main__':
    build()
