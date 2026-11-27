import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
from tetris_two_drop import find_recovery
from tools.find_two_drop_cross import load_snapshot


LEVEL = 12
A = (2, 3, 4)
B = (3, 3, 4)
CROSS_CENTER = (9, 8, 7)


def pocket_board(heights, *, levels=(LEVEL,), reserve_cross=True):
    """Roof off the board except for specified shafts and one potential CROSS landing."""
    layers = sum(1 << z for z in levels)
    board = [layers | (1 << 15)] * game.AREA
    if reserve_cross:
        center = game.column_index(*CROSS_CENTER)
        # A clear under these seven columns lowers the center from 14 to 13
        # and the six neighbors from 15 to 14, just enough to accept CROSS.
        board[center] = layers | (1 << 13)
        for neighbor in game.ADJACENT[center]:
            board[neighbor] = layers | (1 << 14)
    for xyz, height in heights.items():
        board[game.column_index(*xyz)] = 1 << (height - 1)
    return tuple(board)


def spec(move):
    c = move.candidate
    return c.rotation, c.x, c.y, c.w


def physical_drops(board, piece):
    """Exhaust every engine pose without assuming where either piece must land."""
    context = game.evaluate_board(board)
    for pose in game.specifications(piece, 0):
        candidate = game.score_candidate(context, piece, *pose)
        if candidate is not None:
            yield game.materialize_move(board, candidate)


def oracle_pairs(board, current, visible):
    if game.has_legal_move(board, 'CROSS'):
        return set()
    return {(spec(first), spec(second))
            for first in physical_drops(board, current) if first.layers == 0
            for second in physical_drops(first.board, visible)
            if second.layers and game.has_legal_move(second.board, 'CROSS')}


class TwoDropCrossTests(unittest.TestCase):
    def assert_physical_pair(self, board, current, visible, result):
        self.assertIsNotNone(result)
        first, second = result
        self.assertIsInstance(first, game.Move)
        self.assertIsInstance(second, game.Move)
        self.assertEqual(first, game.make_move(board, current, *spec(first)))
        self.assertEqual(second, game.make_move(first.board, visible, *spec(second)))
        self.assertEqual(first.layers, 0)
        self.assertGreater(second.layers, 0)
        self.assertFalse(game.has_legal_move(board, 'CROSS'))
        self.assertTrue(game.has_legal_move(second.board, 'CROSS'))
        return first, second

    def test_adjacent_holes_oracle_and_preferred_order(self):
        board = pocket_board({A: 9, B: 9})
        vertical = next(i for i, shape in enumerate(game.SHAPES['I']) if len(shape.columns) == 1)
        left, right = (vertical, *A), (vertical, *B)
        self.assertEqual(oracle_pairs(board, 'I', 'I'), {(left, right), (right, left)})
        self.assertEqual(len(list(physical_drops(board, 'I'))), 2)
        for order in ((left, right), (right, left)):
            with self.subTest(preferred=order):
                pair = self.assert_physical_pair(board, 'I', 'I', find_recovery(board, 'I', 'I', iter(order)))
                self.assertEqual(tuple(map(spec, pair)), order)
        default = find_recovery(board, 'I', 'I')
        self.assertEqual(tuple(map(spec, default)), (left, right))
        self.assertEqual(find_recovery(board, 'I', 'I'), default)

    def test_archive_snapshot_cli_returns_the_verified_witness(self):
        fixture = ROOT / 'tests/fixtures/two-drop-cross.json'
        output = subprocess.check_output([
            sys.executable, '-B', '-S', str(ROOT / 'tools/find_two_drop_cross.py'),
            str(fixture), '--prefer', '8', '4', '14', '4',
        ], text=True)
        moves = json.loads(output)['moves']
        self.assertEqual([m['piece'] for m in moves], ['RING', 'HOOK'])
        self.assertEqual([m['clears'] for m in moves], [0, 1])
        self.assertEqual(moves[-1]['after_board_sha256'],
                         'e7e64cfbbbf2e7c84d661a9313c6ad64e6d3822a683a2b3d4e2a7ae1b1a33a84')

    def test_snapshot_loader_handles_checked_cache_and_archive_envelopes(self):
        fixture = ROOT / 'tests/fixtures/two-drop-cross.json'
        data = json.loads(fixture.read_text(encoding='utf-8'))
        expected = load_snapshot(fixture)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'snapshot.json.gz'
            for key in ('data', 'snapshot'):
                path.write_bytes(gzip.compress(json.dumps({key: data}).encode()))
                self.assertEqual(load_snapshot(path), expected)
            path.write_bytes(gzip.compress(json.dumps(dict(data, board='')).encode()))
            with self.assertRaises(ValueError):
                load_snapshot(path)

    def test_different_current_and_visible_shapes_on_the_expanded_board(self):
        holes = {(x, y, w): 11 for x in (2, 3) for y in (3, 4) for w in (4, 5)}
        holes[(6, 3, 4)] = 9
        board = pocket_board(holes)
        pairs = oracle_pairs(board, 'HYPER', 'I')
        self.assertEqual(len(pairs), 1)
        pair = self.assert_physical_pair(board, 'HYPER', 'I', find_recovery(board, 'HYPER', 'I'))
        self.assertEqual(tuple(move.piece for move in pair), ('HYPER', 'I'))
        self.assertIn(tuple(map(spec, pair)), pairs)

    def test_support_only_first_drop_below_the_layer(self):
        board = pocket_board({A: 5})
        before = list(physical_drops(board, 'I'))
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0].layers, 0)
        pairs = oracle_pairs(board, 'I', 'I')
        self.assertEqual(len(pairs), 1)
        first, second = self.assert_physical_pair(board, 'I', 'I', find_recovery(board, 'I', 'I'))
        self.assertIn((spec(first), spec(second)), pairs)
        self.assertEqual({z for _, _, z, _ in game.placed_cells(first)}, {5, 6, 7, 8})
        self.assertEqual({z for _, _, z, _ in game.placed_cells(second)}, {9, 10, 11, 12})

    def test_first_can_leave_the_ready_reply_completely_untouched(self):
        board = list(pocket_board({A: 9}))
        for x in range(4, 8):
            board[game.column_index(x, 1, 2)] = 1 << LEVEL
        board = tuple(board)
        first, second = self.assert_physical_pair(board, 'I', 'I', find_recovery(board, 'I', 'I'))
        self.assertTrue(game.make_move(board, 'I', *spec(second)).layers)
        self.assertTrue(all((x, y, w) != A for x, y, _, w in game.placed_cells(first)))
        self.assertIn((spec(first), spec(second)), oracle_pairs(board, 'I', 'I'))

    def test_reply_can_clear_two_layers(self):
        board = pocket_board({A: 9, B: 9}, levels=(LEVEL - 1, LEVEL))
        _, second = self.assert_physical_pair(board, 'I', 'I', find_recovery(board, 'I', 'I'))
        self.assertEqual(second.layers, 2)

    def test_buried_hole_and_premature_clear(self):
        buried = list(pocket_board({A: 9, B: 9}))
        buried[game.column_index(*B)] |= 1 << 15
        immediate = pocket_board({A: 9})
        self.assertEqual([m.layers for m in physical_drops(immediate, 'I')], [1])
        for board in (tuple(buried), immediate):
            with self.subTest(buried=board != immediate):
                self.assertFalse(oracle_pairs(board, 'I', 'I'))
                self.assertIsNone(find_recovery(board, 'I', 'I'))

    def test_rejects_x_edge_wraparound(self):
        # These last two indices are consecutive in memory, but far apart in X.
        holes = (A, B, (15, 5, 7), (0, 6, 7))
        board = pocket_board(dict.fromkeys(holes, 11))
        self.assertEqual(game.column_index(*holes[2]) + 1, game.column_index(*holes[3]))
        self.assertTrue(any(physical_drops(board, 'O')))
        self.assertFalse(oracle_pairs(board, 'O', 'O'))
        self.assertIsNone(find_recovery(board, 'O', 'O'))

    def test_requires_cross_to_be_closed_then_reopened(self):
        closed = pocket_board({A: 9, B: 9}, reserve_cross=False)
        self.assertTrue(any(second.layers for first in physical_drops(closed, 'I')
                            for second in physical_drops(first.board, 'I')))
        self.assertFalse(oracle_pairs(closed, 'I', 'I'))
        self.assertIsNone(find_recovery(closed, 'I', 'I'))
        already_open = list(pocket_board({A: 9, B: 9}))
        center = game.column_index(*CROSS_CENTER)
        already_open[center] = 1 << LEVEL
        for neighbor in game.ADJACENT[center]:
            already_open[neighbor] = (1 << LEVEL) | (1 << 13)
        self.assertTrue(game.has_legal_move(tuple(already_open), 'CROSS'))
        self.assertIsNone(find_recovery(tuple(already_open), 'I', 'I'))


if __name__ == '__main__':
    unittest.main()
