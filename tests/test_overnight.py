from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_overnight as runner


class OvernightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        runner.game.configure_backend('numpy')
        runner.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.manual_seed(81)
        model = runner.neural.ColumnPolicy(runner.neural.Config(width=8, blocks=1, candidates=16, shortlist=8))
        self.initial = self.root / 'initial.pt'
        runner.neural.save_checkpoint(self.initial, model, progress={'updates': 1})

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, name='run', updates=2, **changes):
        args = runner.arguments(['--run-dir', str(self.root / name), '--device', 'cpu',
                                 '--checkpoint', str(self.initial), '--backend', 'numpy',
                                 '--envs', '2', '--rollout', '2', '--batch', '2',
                                 '--warmup-updates', '1', '--selection-pieces', '4',
                                 '--max-updates', str(updates), '--record-games'])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def train(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            with runner.run_lock(args.run_dir):
                trainer = runner.Trainer(args, runner.Budget(1))
                summary = trainer.run()
        return trainer, summary

    def episode(self, count=6):
        episode = runner.Episode(0, runner.game.Game(46))
        config = runner.neural.Config(width=8, blocks=1, candidates=16, shortlist=8)
        for _ in range(count):
            prepared = runner.neural.prepare(episode.state, episode.target, config)
            runner.apply_move(episode, prepared.options[prepared.teacher], 5, 'teacher')
        return episode

    def test_successful_placement_reward_is_not_clear_reward(self):
        episode = self.episode(4)
        self.assertEqual(sum(m['reward_pieces'] for m in episode.moves), 4)
        self.assertEqual(episode.state.layers, 0)
        self.assertEqual(runner.REWARD_SCALE, .001)

    def test_clearing_does_not_reset_or_stop_the_game(self):
        state = runner.game.Game(7)
        state.current = 'I'
        state.board = (0,) + (2,) * (runner.game.AREA - 1)
        episode = runner.Episode(0, state)
        candidate = runner.game.choose_sustained_move(state.board, 'I', 1, 16).move.candidate
        runner.apply_move(episode, candidate, 1, 'test')
        self.assertEqual(state.pieces, 1)
        self.assertEqual(state.layers, 1)
        self.assertEqual(episode.moves[-1]['reward_pieces'], 1)
        self.assertFalse(episode.complete)
        self.assertGreater(sum(mask.bit_count() for mask in state.board), 0)

    def test_termination_and_truncation_bootstrap_differ(self):
        reward = np.array([[.001]], dtype=np.float32)
        value = np.array([[.3]], dtype=np.float32)
        following = np.array([[.7]], dtype=np.float32)
        yes, no = np.array([[True]]), np.array([[False]])
        _, terminal = runner.neural.gae(reward, value, following, yes, no, 1., .999)
        _, truncated = runner.neural.gae(reward, value, following, no, yes, 1., .999)
        self.assertAlmostEqual(float(terminal[0, 0]), .001, places=6)
        self.assertAlmostEqual(float(truncated[0, 0]), .701, places=6)

    def test_snapshot_reconstructs_exact_board_and_stream(self):
        episode = self.episode()
        restored = runner.restore_episode(episode.snapshot())
        self.assertEqual(restored.state.board, episode.state.board)
        self.assertEqual(restored.state.current, episode.state.current)
        self.assertEqual(restored.state.next_piece, episode.state.next_piece)
        self.assertEqual(restored.moves, episode.moves)

    def test_json_active_record_is_valid_and_verifiable(self):
        recorder = runner.Recorder(self.root / 'games', True)
        path = recorder.write(self.episode(), 'checkpoint')
        report = runner.verify_game(path)
        self.assertTrue(report['passed'])
        self.assertFalse(report['complete'])
        self.assertEqual(report['pieces'], 6)
        self.assertEqual(report['stop_reason'], 'checkpoint')

    def test_completed_capped_record_is_not_a_loss(self):
        episode = self.episode()
        episode.complete, episode.stop_reason = True, 'piece_limit'
        path = runner.Recorder(self.root / 'games', True).write(episode)
        report = runner.verify_game(path)
        value = json.loads(path.read_text())
        self.assertTrue(report['complete'])
        self.assertTrue(value['truncated'])
        self.assertFalse(value['terminated'])

    def test_recording_detects_tampered_piece_and_board_hash(self):
        for field, value in (('z', 99), ('board_sha256', 'invalid'), ('reward_pieces', 99)):
            with self.subTest(field=field):
                snapshot = copy.deepcopy(self.episode(2).snapshot())
                snapshot['moves'][0][field] = value
                with self.assertRaises(ValueError):
                    runner.restore_episode(snapshot)

    def test_false_game_over_is_rejected(self):
        episode = self.episode(2)
        episode.complete, episode.stop_reason = True, 'top_out'
        path = runner.Recorder(self.root / 'games', True).write(episode)
        with self.assertRaises(ValueError):
            runner.verify_game(path)

    def test_recording_quota_disables_only_recording(self):
        with contextlib.redirect_stderr(io.StringIO()):
            recorder = runner.Recorder(self.root / 'games', True, maximum_mib=.0001)
            self.assertIsNone(recorder.write(self.episode()))
        self.assertFalse(recorder.enabled)
        self.assertIsNotNone(recorder.error)

    def test_recording_failure_is_not_a_training_failure(self):
        recorder = runner.Recorder(self.root / 'games', True)
        with patch.object(runner.tempfile, 'mkstemp', side_effect=PermissionError), contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(recorder.write(self.episode()))
        self.assertFalse(recorder.enabled)

    def test_selection_ranks_survival_not_clears_or_speed(self):
        a = [{'pieces': 1000, 'hyperlayers': 100, 'evaluation_complete': True},
             {'pieces': 1000, 'hyperlayers': 100, 'evaluation_complete': True}]
        b = [{'pieces': 1200, 'hyperlayers': 0, 'evaluation_complete': True},
             {'pieces': 1100, 'hyperlayers': 0, 'evaluation_complete': True}]
        self.assertGreater(runner.survival_score(b), runner.survival_score(a))
        b[-1]['evaluation_complete'] = False
        self.assertIsNone(runner.survival_score(b))

    def test_one_command_warmup_survival_evaluation_and_recording(self):
        trainer, summary = self.train(self.args())
        self.assertEqual(summary['warmup_updates'], 1)
        self.assertEqual(summary['survival_updates'], 1)
        self.assertEqual(summary['placements'], 8)
        self.assertTrue(trainer.last.is_file())
        self.assertTrue(trainer.best.is_file())
        self.assertTrue((trainer.root / 'previous.pt').is_file())
        records = list((trainer.root / 'games').rglob('*.json'))
        self.assertGreaterEqual(len(records), 6)
        for path in records:
            self.assertTrue(runner.verify_game(path)['passed'])
        checkpoint = torch.load(trainer.last, weights_only=True)
        self.assertEqual(checkpoint['extra']['signature']['gamma'], 1)
        self.assertEqual(checkpoint['extra']['kind'], runner.RUN_SCHEMA)
        self.assertTrue(all(torch.isfinite(value).all() for value in checkpoint['model'].values()))

    def test_exact_cpu_resume_across_warmup_and_survival(self):
        continuous, _ = self.train(self.args('continuous', 3))
        split, _ = self.train(self.args('split', 2))
        resumed, _ = self.train(self.args('split', 1))
        a = torch.load(continuous.last, weights_only=True)
        b = torch.load(resumed.last, weights_only=True)
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['extra']['environments'], b['extra']['environments'])
        for key in a['model']:
            self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
        self.assertTrue(torch.equal(a['rng_cpu'], b['rng_cpu']))

    def test_changed_resume_settings_do_not_overwrite_checkpoint(self):
        trainer, _ = self.train(self.args())
        digest = runner.file_hash(trainer.last)
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            runner.Trainer(self.args(batch=1), runner.Budget(1))
        self.assertEqual(runner.file_hash(trainer.last), digest)

    def test_partial_evaluation_never_promotes(self):
        trainer, _ = self.train(self.args())
        digest = runner.file_hash(trainer.best)
        trainer.baseline = None
        with patch.object(trainer, 'evaluate_policy', return_value={'complete': False, 'score': None, 'runs': []}):
            trainer.budget.reason = None
            trainer.select()
        self.assertEqual(runner.file_hash(trainer.best), digest)

    def test_best_does_not_change_when_candidate_is_worse(self):
        trainer, _ = self.train(self.args())
        digest = runner.file_hash(trainer.best)
        trainer.budget.reason = None
        result = {'complete': True, 'score': (0, 0, 0), 'runs': [], 'signature': trainer.signature}
        with patch.object(trainer, 'evaluate_policy', return_value=result), contextlib.redirect_stdout(io.StringIO()):
            trainer.select()
        self.assertEqual(runner.file_hash(trainer.best), digest)

    def test_deadline_is_cooperative_and_signal_requests_stop(self):
        clock = [0.]
        budget = runner.Budget(1, clock=lambda: clock[0])
        self.assertFalse(budget.expired())
        clock[0] = 3600.
        self.assertTrue(budget.expired())
        self.assertEqual(budget.reason, 'time_budget')
        budget = runner.Budget(1)
        with contextlib.redirect_stderr(io.StringIO()):
            budget.handle(signal.SIGTERM, None)
        self.assertTrue(budget.expired())

    @unittest.skipUnless(hasattr(signal, 'SIGHUP'), 'SIGHUP is POSIX-specific')
    def test_nohup_inherited_ignore_is_preserved(self):
        with patch.object(runner.signal, 'getsignal', side_effect=lambda n: signal.SIG_IGN if n == signal.SIGHUP else signal.SIG_DFL):
            with patch.object(runner.signal, 'signal', return_value=signal.SIG_DFL) as change:
                with runner.Budget(1):
                    pass
        self.assertNotIn(signal.SIGHUP, [call.args[0] for call in change.call_args_list])

    def test_run_directory_lock_rejects_second_writer(self):
        with runner.run_lock(self.root / 'locked'):
            with self.assertRaises(ValueError):
                with runner.run_lock(self.root / 'locked'):
                    pass
        with runner.run_lock(self.root / 'locked'):
            pass

    def test_invalid_cli_values_and_overlapping_seed_partitions(self):
        for argv in (['--hours', 'nan'], ['--record-every', '0'], ['--selection-seeds', '2'],
                     ['--warmup-updates', '-1'], ['--gae-lambda', '2']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                runner.arguments(argv)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable in this test environment')
    def test_cuda_overnight_update(self):
        trainer, summary = self.train(self.args('cuda', 2, device='cuda'))
        self.assertEqual(summary['runner_updates'], 2)
        self.assertEqual(trainer.device.type, 'cuda')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OvernightTests))
    raise SystemExit(not result.wasSuccessful())
