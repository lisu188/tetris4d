"""Find two checked physical drops from the public fields of an evaluator snapshot."""
import argparse
import base64
import gzip
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_tetris_4d_optimized as game
from tetris_two_drop import find_recovery


def load_snapshot(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as source:
        record = json.load(source)
    if not isinstance(record, dict):
        raise ValueError('Expected a JSON snapshot object')
    snapshot = record.get('snapshot', record.get('data', record))
    if not isinstance(snapshot, dict):
        raise ValueError('Expected an evaluator snapshot in the input envelope')
    current, visible = snapshot.get('current'), snapshot.get('next_piece')
    if not all(isinstance(piece, str) and piece in game.SHAPES for piece in (current, visible)):
        raise ValueError('Expected registered current and next_piece names')
    packed = snapshot.get('board')
    if not isinstance(packed, str):
        raise ValueError('Expected the base64-encoded, zlib-compressed evaluator board')
    data = zlib.decompressobj()
    raw = data.decompress(base64.b64decode(packed, validate=True), 2 * game.AREA + 1)
    if len(raw) != 2 * game.AREA or not data.eof or data.unused_data:
        raise ValueError('Expected 4096 little-endian 16-bit board columns')
    board = struct.unpack('<4096H', raw)
    if game.full_layer_mask(board):
        raise ValueError('Input contains a completed layer awaiting compaction')
    return board, current, visible


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path, help='Evaluator snapshot JSON or .json.gz, bare or wrapped in snapshot/data')
    parser.add_argument('--prefer', type=int, nargs=4, action='append', default=[],
                        metavar=('ROT', 'X', 'Y', 'W'), help='Prefer this first pose; repeat in desired order')
    args = parser.parse_args(argv)
    try:
        board, current, visible = load_snapshot(args.snapshot)
        pair = find_recovery(board, current, visible, args.prefer)
    except (EOFError, OSError, ValueError, zlib.error) as error:
        parser.error(str(error))
    result = {'current': current, 'visible': visible, 'moves': []}
    for move in pair or ():
        c = move.candidate
        result['moves'].append(dict(piece=c.piece, rotation=c.rotation, x=c.x, y=c.y, w=c.w,
                                    z=c.z, clears=c.layers,
                                    after_board_sha256=hashlib.sha256(struct.pack('<4096H', *move.board)).hexdigest()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
