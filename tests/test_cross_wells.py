import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


# First expanded bag from the seed-7 prototype trace, compressed SHA-256
# 9d966ceebd208a4b6860437c4987e5d3252f572f00402c0b0b228d3e5f515f0a.
# Entries are (piece, catalog rotation, x, y, w, landing z).
FIRST_BAG = (
    ('B', 5, 4, 10, 1, 0), ('T', 1, 5, 11, 1, 0), ('C', 32, 5, 10, 2, 0),
    ('HOOK', 183, 5, 9, 0, 0), ('I', 3, 4, 9, 2, 0), ('O', 2, 3, 9, 3, 0),
    ('STAIR', 45, 4, 10, 3, 0), ('FORK', 29, 4, 8, 0, 0), ('CROSS', 0, 3, 9, 2, 0),
    ('S', 0, 1, 8, 1, 0), ('BRIDGE', 51, 3, 10, 4, 0), ('NOVA', 2, 2, 10, 3, 0),
    ('HYPER', 0, 0, 0, 0, 0), ('RING', 9, 1, 8, 1, 0), ('L', 42, 2, 9, 4, 0),
)


class CrossWellsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(game.configure_backend, game.BACKEND)

    def test_non_native_backend_reports_build_requirement(self):
        game.configure_backend('python')
        with self.assertRaisesRegex(ValueError, 'native.*build_native'):
            game.choose_cross_wells_move(game.EMPTY_BOARD, 'CROSS', 0)

    def test_entire_first_bag_matches_archived_choices_and_board(self):
        game.configure_backend('native')
        state = game.Game(7, 'expanded')
        for expected in FIRST_BAG:
            with self.subTest(turn=state.pieces, piece=state.current):
                decision = game.choose_cross_wells_move(state.board, state.current, state.pieces)
                self.assertIsNotNone(decision)
                candidate = decision.move.candidate
                self.assertEqual((candidate.piece, candidate.rotation, candidate.x, candidate.y,
                                  candidate.w, candidate.z), expected)
                self.assertEqual(decision.policy, 'cross-wells')
                self.assertIsNone(decision.target)
                state.commit(decision.move)
        self.assertEqual(hashlib.sha256(struct.pack('<4096H', *state.board)).hexdigest(),
                         'e8e91fe39db6944bf639f39f6a177d1906c36b82bd6e91e14d9eab2055a15f91')

    def test_only_cross_socket_at_ceiling_and_physical_top_out(self):
        game.configure_backend('native')
        ceiling = 1 << (game.HEIGHT - 1)
        center = game.column_index(14, 1, 14)
        columns = [ceiling] * game.AREA
        columns[center] = (1 << 13) - 1
        for step in (-256, -16, -1, 1, 16, 256):
            columns[center + step] = (1 << 14) - 1
        board = tuple(columns)
        decision = game.choose_cross_wells_move(board, 'CROSS', 21, budget=1)
        self.assertIsNotNone(decision)
        move = decision.move
        c = move.candidate
        self.assertEqual((c.rotation, c.x, c.y, c.w, c.z), (0, 13, 0, 13, 13))
        self.assertEqual((move.board, move.layers, c.z), game.reference_drop(board, 'CROSS', 0, 13, 0, 13))
        columns[center] |= ceiling
        self.assertFalse(game.has_legal_move(tuple(columns), 'CROSS'))
        self.assertIsNone(game.choose_cross_wells_move(tuple(columns), 'CROSS', 21, budget=1))

    def test_benchmark_default_matches_explicit_prototype_budget(self):
        base = [sys.executable, '-B', str(ROOT / 'sparse_tetris_4d_optimized.py'), '--benchmark',
                '--policy', 'cross-wells', '--backend', 'native', '--no-benchmark-log',
                '--seed', '7', '--games', '1', '--pieces', '15']
        runs = []
        for extra in ([], ['--candidates', '1536']):
            output = subprocess.run(base + extra, capture_output=True, text=True, check=True,
                                    env={**os.environ, 'TETRIS4D_GAME_LIBRARY': 'off'})
            run, = json.loads(output.stdout)['runs']
            runs.append({k: v for k, v in run.items() if k not in ('mean_decision_ms', 'wall_seconds')})
        self.assertEqual(*runs)
        expected = dict(policy='cross-wells', backend='native', candidates=1536, piece_set='expanded',
                        pieces=15, cells_placed=90, occupied_cells=90, holes=0, max_height=4,
                        hyperlayers=0, stop_reason='piece_limit', capped=True, game_over=False, target=None)
        self.assertEqual({k: runs[0][k] for k in expected}, expected)


if __name__ == '__main__':
    unittest.main()
