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

    def test_bundled_models_are_finite_linear_scorers(self):
        # The default policy uses the v2 model; v1 stays for its 1M and 10M records.
        self.assertEqual(game.TRAINED_WELLS_MODEL.name, 'trained-wells-v2.json')
        for name, schema, count in (('trained-wells.json', 'trained-wells-linear-v1', 7),
                                    ('trained-wells-v2.json', 'trained-wells-linear-v2', 9)):
            model = json.loads((ROOT / 'models' / name).read_text())
            self.assertEqual((model['schema'], len(model['weights'])), (schema, count))
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

    def test_bundled_models_first_decisions_match_reference_drops_and_digests(self):
        # Pins both bundled models; update a digest only when that model is retrained.
        for name, digest in (('trained-wells.json', 'a9061216a4eef17de035c0bbd1ee739e5c5caa8d983eeba9f5c1868dcefea0bd'),
                             ('trained-wells-v2.json', 'bedb0fcd86803dbecfb3fc679a28cfcfeee39316706ba18b91b149afa944c27f')):
            with self.subTest(model=name):
                weights = json.loads((ROOT / 'models' / name).read_text())['weights']
                self.assertEqual(self.first_decisions_digest(weights), digest)

    def first_decisions_digest(self, weights):
        game.configure_backend('native')
        state = game.Game(4242, 'expanded')
        record = bytearray()
        for _ in range(60):
            decision = game.choose_trained_wells_move(state.board, state.current, state.pieces, weights=weights)
            self.assertIsNotNone(decision)
            self.assertEqual(decision.policy, 'trained-wells')
            c = decision.move.candidate
            board, layers, z = game.reference_drop(state.board, c.piece, c.rotation, c.x, c.y, c.w)
            self.assertEqual((board, layers, z), (decision.move.board, decision.move.layers, c.z))
            record += bytes((game.PIECE_NAMES.index(c.piece), c.rotation, c.x, c.y, c.w, c.z, layers))
            state.commit(decision.move)
        self.assertFalse(state.game_over)
        return hashlib.sha256(bytes(record)).hexdigest()

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


class LayerCompletionTests(unittest.TestCase):
    """Version-2 weights add two layer-completion features to the same scorer."""

    def setUp(self):
        self.addCleanup(game.configure_backend, game.BACKEND)
        game.configure_backend('native')
        self.v1 = json.loads((ROOT / 'models' / 'trained-wells.json').read_text())['weights']
        self.v2_model = json.loads((ROOT / 'models' / 'trained-wells-v2.json').read_text())

    def test_v2_model_is_finite_nine_weight_scorer(self):
        self.assertEqual(self.v2_model['schema'], 'trained-wells-linear-v2')
        self.assertEqual(len(self.v2_model['weights']), 9)
        self.assertTrue(all(math.isfinite(v) for v in self.v2_model['weights']))
        self.assertEqual(game.full_wells_weights(self.v1)[7:], (0.0, 0.0))

    def test_v2_finishes_the_last_open_cell_of_a_level(self):
        # Level 0 is full except one corner cell; only a vertical I can fill it without a hole.
        columns = [1] * game.AREA
        columns[0] = 0
        board = tuple(columns)
        v1 = game.choose_trained_wells_move(board, 'I', 100, weights=self.v1)
        padded = game.choose_trained_wells_move(board, 'I', 100, weights=list(self.v1) + [0.0, 0.0])
        v2 = game.choose_trained_wells_move(board, 'I', 100, weights=self.v2_model['weights'])
        self.assertEqual((v1.move.candidate, v1.move.layers), (padded.move.candidate, padded.move.layers))
        self.assertEqual(v1.move.layers, 0)
        c = v2.move.candidate
        self.assertEqual((c.x, c.y, c.w, c.z, v2.move.layers), (0, 0, 0, 0, 1))


if __name__ == '__main__':
    unittest.main()
