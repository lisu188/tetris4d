import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


class TrainedWellsTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(game.configure_backend, game.BACKEND)

    def test_bundled_model_is_finite_linear_scorer(self):
        model = json.loads(game.TRAINED_WELLS_MODEL.read_text())
        self.assertEqual(model['schema'], 'trained-wells-linear-v1')
        self.assertEqual(len(model['weights']), 7)
        self.assertTrue(all(math.isfinite(v) for v in model['weights']))
        self.assertIn('cross-entropy', model['training']['method'])

    def test_native_rejects_missing_or_non_finite_weights(self):
        game.configure_backend('native')
        board = game.evaluate_board(game.EMPTY_BOARD).native_board
        shapes = game.native_shapes('T')
        out = game._NUMPY.empty(8, dtype=game._NUMPY.int32)
        bad = game._NUMPY.array([1, 1, 1, float('nan'), 1, 1, 1], dtype=game._NUMPY.float64)
        call = game._NATIVE.t4_cross_wells_weighted
        self.assertEqual(call(board.ctypes.data, shapes.ctypes.data, len(shapes),
                              game.PIECE_NAMES.index('T'), 0, 64, bad.ctypes.data, out.ctypes.data), -1)
        self.assertEqual(call(board.ctypes.data, shapes.ctypes.data, len(shapes),
                              game.PIECE_NAMES.index('T'), 0, 64, None, out.ctypes.data), -1)

    def test_bundled_model_first_decisions_match_reference_drops_and_digest(self):
        game.configure_backend('native')
        state = game.Game(4242, 'expanded')
        record = bytearray()
        for _ in range(60):
            decision = game.choose_trained_wells_move(state.board, state.current, state.pieces)
            self.assertIsNotNone(decision)
            self.assertEqual(decision.policy, 'trained-wells')
            c = decision.move.candidate
            board, layers, z = game.reference_drop(state.board, c.piece, c.rotation, c.x, c.y, c.w)
            self.assertEqual((board, layers, z), (decision.move.board, decision.move.layers, c.z))
            record += bytes((game.PIECE_NAMES.index(c.piece), c.rotation, c.x, c.y, c.w, c.z, layers))
            state.commit(decision.move)
        self.assertFalse(state.game_over)
        # Pins the bundled models/trained-wells.json; update it if the model is retrained.
        self.assertEqual(hashlib.sha256(bytes(record)).hexdigest(),
                         'a9061216a4eef17de035c0bbd1ee739e5c5caa8d983eeba9f5c1868dcefea0bd')

    def test_benchmark_cli_accepts_trained_policy(self):
        import os
        import subprocess
        output = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'sparse_tetris_4d_optimized.py'), '--benchmark',
             '--policy', 'trained-wells', '--backend', 'native', '--no-benchmark-log',
             '--seed', '7', '--games', '1', '--pieces', '20'],
            capture_output=True, text=True, check=True, env={**os.environ, 'TETRIS4D_GAME_LIBRARY': 'off'})
        run, = json.loads(output.stdout)['runs']
        self.assertEqual((run['policy'], run['pieces'], run['stop_reason']), ('trained-wells', 20, 'piece_limit'))



class NativeLoopTests(unittest.TestCase):
    def test_native_loop_reproduces_python_loop_records(self):
        import gzip
        import tempfile
        sys.path.insert(0, str(ROOT / 'tools'))
        import verify_trained_wells as verify
        game.configure_backend('native')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records = {}
            for name, runner in (('python', verify.play), ('native', verify.play_native)):
                out = folder / name
                out.mkdir()
                result = runner(4242, 300, game.TRAINED_WELLS_MODEL, out / 'moves.bin.gz', out / 'progress.jsonl', 100)
                records[name] = (gzip.open(out / 'moves.bin.gz').read(), result['board_sha256'], result['placements'])
            self.assertEqual(records['python'], records['native'])
            self.assertEqual(records['native'][2], 300)


if __name__ == '__main__':
    unittest.main()
