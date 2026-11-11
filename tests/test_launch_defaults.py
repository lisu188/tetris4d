from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game

CHECKPOINT = ROOT / 'models' / 'trained-sigmoid.json'
TRAINED = json.loads(CHECKPOINT.read_text(encoding='utf-8'))['parameters']


class LaunchDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.parameters = game.network_parameters()
        self.training = game.MODEL_TRAINING
        self.backend = game.BACKEND
        self.cwd = Path.cwd()
        game.configure_network(game.BASELINE_PARAMETERS)

    def tearDown(self):
        os.chdir(self.cwd)
        game.configure_network(self.parameters, self.training)
        game.configure_backend(self.backend)

    def launch(self, argv):
        with patch.object(game, 'launch_demo', return_value=0) as launcher:
            self.assertEqual(game.main(argv), 0)
        launcher.assert_called_once()
        return launcher.call_args.args[0]

    def test_no_arguments_load_trained_hyperlayer_defaults(self):
        with patch.object(sys, 'argv', [str(ROOT / 'sparse_tetris_4d_optimized.py')]):
            args = self.launch(None)
        self.assertEqual(args.policy, 'sustained')
        self.assertEqual(args.target, 1)
        self.assertEqual(args.weights, CHECKPOINT)
        self.assertEqual(args.piece_set, 'expanded')
        self.assertEqual(args.candidates, 2048)
        self.assertEqual(game.network_parameters(), TRAINED)

    def test_default_weights_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                args = self.launch([])
                self.assertEqual(args.weights, CHECKPOINT)
                self.assertEqual(game.network_parameters(), TRAINED)
            finally:
                os.chdir(self.cwd)

    def test_network_policy_keeps_trained_weights(self):
        args = self.launch(['--policy', 'network'])
        self.assertEqual(args.policy, 'network')
        self.assertEqual(game.network_parameters(), TRAINED)

    def test_explicit_baseline_restores_original_controller(self):
        game.load_checkpoint(CHECKPOINT)
        with patch.object(game, 'load_checkpoint', side_effect=AssertionError('Unexpected checkpoint load')):
            args = self.launch(['--policy', 'network', '--baseline'])
        self.assertTrue(args.baseline)
        self.assertEqual(args.policy, 'network')
        self.assertEqual(game.network_parameters(), game.BASELINE_PARAMETERS)
        self.assertEqual(game.MODEL_TRAINING['method'], 'none')

    def test_custom_relative_checkpoint_overrides_default(self):
        checkpoint = json.loads(CHECKPOINT.read_text(encoding='utf-8'))
        checkpoint['parameters']['threshold'] = 2.75
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                Path('custom model.json').write_text(json.dumps(checkpoint), encoding='utf-8')
                args = self.launch(['--weights', 'custom model.json'])
                self.assertEqual(args.weights, Path('custom model.json'))
                self.assertEqual(game.network_parameters(), checkpoint['parameters'])
            finally:
                os.chdir(self.cwd)

    def test_missing_checkpoint_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / 'missing.json'
            with patch.object(game, 'launch_demo') as launcher, contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(game.main(['--weights', str(missing)]), 1)
            launcher.assert_not_called()
            self.assertIn('missing.json', errors.getvalue())

    def test_baseline_and_explicit_checkpoint_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            game.main(['--baseline', '--weights', str(CHECKPOINT)])
        self.assertEqual(error.exception.code, 2)

    def test_explicit_target_and_backend_are_preserved(self):
        args = self.launch(['--target', '3', '--backend', 'python'])
        self.assertEqual(args.target, 3)
        self.assertEqual(game.BACKEND, 'python')
        self.assertEqual(args.policy, 'sustained')

    def test_benchmark_dispatch_uses_default_policy(self):
        result = {'seed': 7, 'pieces': 1, 'hyperlayers': 0}
        with patch.object(game, 'run_game', return_value=result) as run, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(game.main(['--benchmark', '--games', '1', '--pieces', '1']), 0)
        self.assertEqual(run.call_args.args[5], 'sustained')
        self.assertEqual(run.call_args.args[7], 1)
        self.assertEqual(game.network_parameters(), TRAINED)

    def test_headless_network_export_loads_default_checkpoint(self):
        with patch.object(game, 'launch_demo') as launcher, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(game.main(['--network']), 0)
        launcher.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['well'], [16, 16, 16, 16])
        self.assertEqual(game.network_parameters(), TRAINED)


if __name__ == '__main__':
    unittest.main()
