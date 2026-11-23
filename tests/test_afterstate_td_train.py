"""Afterstate reward indexing, same-game boundaries and deterministic TD recovery."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_afterstate_td as train


class TDTargets(unittest.TestCase):
    def test_count_only_subsequent_placements(self):
        values = torch.tensor([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
        # 0: terminal on row 1; row 2 belongs to a new game.
        # 1: row 0 cap bootstraps on old game; 2: final rollout boundary bootstraps.
        has = torch.tensor([[True, True, True], [False, True, True], [True, True, True]])
        continues = torch.tensor([[True, False, True], [False, True, True], [True, True, True]])
        returns = train.lambda_returns(values, has, continues, 1.)
        torch.testing.assert_close(returns[0, :2], torch.tensor([.001, 3.001]))
        self.assertEqual(returns[1, 0].item(), 0)
        torch.testing.assert_close(returns[:, 2], torch.tensor([10.003, 10.002, 10.001]))
        zero = train.lambda_returns(values, has, continues, 0.)
        torch.testing.assert_close(zero, torch.where(has, values + .001, 0.))

    def test_terminal_and_cap_rows_do_not_link_to_reset_game(self):
        buffer = train.Rollout(3, 2)
        f = torch.ones_like(buffer.pending)
        yes = torch.ones(2, dtype=torch.bool)
        buffer.record(f, torch.tensor([1., 1.]), yes, yes)
        buffer.record(f*2, torch.tensor([12., 0.]), torch.tensor([True, False]), ~yes)
        self.assertEqual(buffer.steps, 1)
        self.assertEqual(buffer.valid[0].tolist(), [True, True])
        self.assertEqual(buffer.has_next[0].tolist(), [True, False])
        self.assertEqual(buffer.continues[0].tolist(), [False, False])
        self.assertFalse(buffer.pending_valid.any())


class Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        train.game.configure_backend('numpy')
        train.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / 'v4.pt'
        config = train.neural.Config(architecture='survival-spatial-v4', regional_critic=True,
                                     spatial_actor=True, shortlist=8, candidates=32, recover_target=True)
        source = train.neural.create_model(config)
        train.neural.save_checkpoint(path, source)
        calibration = self.root / 'beta.json'
        calibration.write_text(json.dumps(dict(beta=1., source_sha256=train.base.file_hash(path), top_k=16)))
        self.args = train.arguments(['--device','cpu','--run-dir',str(self.root),'--source-checkpoint',str(path),
                                     '--calibration',str(calibration),'--envs','1','--rollout','2',
                                     '--batch','2','--danger-starts','0','--pieces','4'])
        self.manifest = dict(start_epoch=0, deadline_epoch=7200)
        self.signature = train.provenance(self.args, json.loads(calibration.read_text()), [])
        self.budget = train.base.Budget(2)

    def trainer(self):
        return train.Trainer(self.args, self.budget, [], self.signature, self.manifest)

    def test_midrollout_recovery_replays_optimizer_moves_and_actor(self):
        t = self.trainer()
        t.emit('run_start')
        t.step()
        t.step()
        self.assertEqual(t.buffer.steps, 1)
        t.checkpoint('test')
        frozen = train.v4.v3.model_digest(t.model.base)
        scorer = train.v4.v3.model_digest(t.model.value)
        # The held pending feature is from the selected move before checkpoint.
        pending = t.buffer.pending.clone()
        t.step()
        t.update()
        expected_board = train.pack_env(t.envs[0])
        expected_model = train.v4.v3.model_digest(t.model.value)
        with t.metrics.open('a') as log:
            log.write('incomplete-event')
        resumed = self.trainer()
        self.assertNotIn('incomplete-event', resumed.metrics.read_text())
        torch.testing.assert_close(resumed.buffer.pending, pending)
        self.assertEqual(resumed.deadline, t.deadline)
        resumed.step()
        resumed.update()
        self.assertEqual(train.pack_env(resumed.envs[0]), expected_board)
        self.assertEqual(resumed.progress, t.progress)
        self.assertEqual(train.v4.v3.model_digest(resumed.model.base), frozen)
        self.assertEqual(train.v4.v3.model_digest(resumed.model.value), expected_model)
        self.assertNotEqual(expected_model, scorer)
        self.assertTrue(all(p.grad is None for p in resumed.model.base.parameters()))

    def test_missing_or_partly_overwritten_metric_rows_are_recovered(self):
        trainer = self.trainer()
        trainer.emit('run_start')
        trainer.emit('heartbeat')
        trainer.checkpoint('test')
        recorded = trainer.metrics.read_bytes()
        trainer.metrics.unlink()
        self.trainer()
        self.assertEqual(trainer.metrics.read_bytes(), recorded)
        trainer.metrics.write_bytes(recorded[:20] + b'partial')
        self.trainer()
        self.assertEqual(trainer.metrics.read_bytes(), recorded)

    def test_warm_start_is_head_only_and_lagged_targets_resume_exactly(self):
        original = self.trainer()
        original.emit('run_start')
        while original.buffer.steps < self.args.rollout:
            original.step()
        original.update()
        original.checkpoint('warm')
        original_value = train.v4.v3.model_digest(original.model.value)
        self.args.warm_start = original.last
        self.args.run_dir = self.root/'replacement'
        self.args.run_dir.mkdir()
        self.args.actor_bound, self.args.target_tau, self.args.clip_grad = .5, .01, 1.
        calibration = json.loads(self.args.calibration.read_text())
        self.signature = train.provenance(self.args, calibration, [])
        new = self.trainer()
        self.assertEqual(new.progress['placements'], 0)
        self.assertFalse(new.optimizer.state)
        self.assertEqual(train.v4.v3.model_digest(new.model.value), original_value)
        self.assertEqual(train.v4.v3.model_digest(new.target_head), original_value)
        self.assertTrue(all(not p.requires_grad for p in new.target_head.parameters()))
        new.emit('run_start')
        while new.buffer.steps < self.args.rollout:
            new.step()
        new.update()
        self.assertNotEqual(train.v4.v3.model_digest(new.target_head), original_value)
        self.assertNotEqual(train.v4.v3.model_digest(new.model.value), train.v4.v3.model_digest(new.target_head))
        new.checkpoint('recover')
        new.step()
        restored = self.trainer()
        self.assertEqual(train.v4.v3.model_digest(restored.target_head), train.v4.v3.model_digest(new.target_head))
        restored.step()
        self.assertEqual(train.pack_env(restored.envs[0]), train.pack_env(new.envs[0]))
        self.assertTrue(torch.equal(restored.buffer.next_values[:1], new.buffer.next_values[:1]))
        self.assertTrue(torch.equal(restored.buffer.pending, new.buffer.pending))
        with torch.no_grad():
            restored.target_head[-1].bias.add_(2.)
            pool = train.td.CandidatePool(torch.randn(2, 3, train.td.FEATURES),
                torch.ones(2, 3, dtype=torch.bool), torch.tensor([[0, 1, 2], [0, 1, 2]]),
                torch.zeros(2, 3), torch.zeros(2, 3))
            indices, chosen, bootstrap = restored.sampled_rows(pool)
            self.assertTrue(torch.equal(chosen, pool.features[torch.arange(2), indices]))
            self.assertTrue(torch.equal(bootstrap, train.td.score_features(restored.target_head, chosen)))
            self.assertTrue((bootstrap > restored.model.score(chosen)).all())

    def test_training_resets_skip_screen_streams_after_resume(self):
        path = self.root/'positions.json'
        start = 2*self.args.seed
        path.write_text(json.dumps(dict(schema='afterstate-screen-exclusions-v1',positions=[
            dict(phase='discovery', game_id='first', seed=start),
            dict(phase='confirmation', game_id='second', seed=start+4),
            dict(phase='reserved_extension', game_id='third', seed=start+6)])))
        self.args.exclude_positions = path
        self.signature = train.provenance(self.args,json.loads(self.args.calibration.read_text()),[])
        original = self.trainer()
        self.assertEqual(original.envs[0].state.seed,start+2)
        original.emit('run_start')
        original.checkpoint('recover')
        self.assertEqual(self.trainer().new_env().state.seed,start+8)

    def test_signal_during_final_save_does_not_complete_the_trial(self):
        trainer = self.trainer()
        trainer.budget.deadline = trainer.budget.clock()
        save = trainer.checkpoint

        def interrupted(reason):
            elapsed = save(reason)
            if reason == 'time_budget':
                trainer.budget.handle(signal.SIGTERM, None)
            return elapsed

        with patch.object(trainer, 'checkpoint', side_effect=interrupted):
            summary = trainer.run()
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['stop_reason'], f'signal_{signal.SIGTERM}')
        self.assertFalse(json.loads((self.root/'selection.json').read_text())['complete'])
        self.assertEqual(summary['final_checkpoint_bytes'], trainer.last.stat().st_size)
        self.assertGreater(summary['final_checkpoint_write_seconds'], 0)

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'explicit CUDA slot required')
    def test_cuda_resume_restores_optimizer_rng_and_next_update(self):
        self.check_cuda_replay()

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'explicit CUDA slot required')
    def test_cuda_warm_start_restores_lagged_head_and_next_update(self):
        initializer = self.trainer()
        initializer.emit('run_start')
        while initializer.buffer.steps < self.args.rollout:
            initializer.step()
        initializer.update()
        initializer.checkpoint('initializer')
        self.args.warm_start = initializer.last
        self.args.run_dir = self.root/'replacement'
        self.args.run_dir.mkdir()
        self.args.actor_bound, self.args.target_tau, self.args.lr = .5, .01, 1e-5
        self.check_cuda_replay()

    def check_cuda_replay(self):
        train.game.configure_backend('native')
        self.addCleanup(train.game.configure_backend, 'numpy')
        self.args.device = 'cuda'
        calibration = json.loads(self.args.calibration.read_text())
        self.signature = train.provenance(self.args, calibration, [])
        t = self.trainer()
        t.emit('run_start')
        while t.buffer.steps < self.args.rollout:
            t.step()
        t.update()
        t.step()
        t.checkpoint('test')
        rng = torch.cuda.get_rng_state().clone()

        def finish_update(trainer):
            while trainer.buffer.steps < self.args.rollout:
                trainer.step()
            trainer.update()
            target = train.v4.v3.model_digest(trainer.target_head) if trainer.target_head is not None else None
            return train.pack_env(trainer.envs[0]), train.v4.v3.model_digest(trainer.model.value), target

        expected = finish_update(t)
        resumed = self.trainer()
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), rng))
        self.assertTrue(all(state['exp_avg'].is_cuda for state in resumed.optimizer.state.values()))
        self.assertEqual(finish_update(resumed), expected)

    def test_unchanged_best_on_tie_and_incomplete_selection(self):
        trainer = self.trainer()
        trainer.started = train.time.time()
        trainer.emit('run_start')
        trainer.progress['updates'] = 1
        report = dict(complete=True, seeds=list(train.evaluation.DEVELOPMENT_SEEDS),
                      mean=12., median=10., minimum=1.)
        with patch.object(train.evaluation, 'evaluate_fresh', return_value=report):
            trainer.select()
            trainer.select()
        best = dict(trainer.best)
        self.assertEqual(best['selection_slot'], 1)
        self.assertEqual(train.base.file_hash(self.root/'best.pt'), best['sha256'])
        with patch.object(train.evaluation, 'evaluate_fresh', return_value={'complete': False}):
            trainer.select()
        self.assertEqual(trainer.best, best)
        self.assertEqual(trainer.next_selection, 3)
        _, saved = train.neural.load_checkpoint(self.root/'best.pt', torch.device('cpu'))
        self.assertEqual(saved['progress']['updates'], 1)

    def test_slow_selection_skips_missed_slots_and_waits_for_new_weights_after_resume(self):
        trainer = self.trainer()
        trainer.emit('run_start')
        trainer.progress['updates'] = 4
        period = self.args.selection_hours * 3600
        with patch.object(train.time, 'time', return_value=trainer.started + period):
            self.assertTrue(trainer.selection_due())
        report = dict(complete=True, seeds=list(train.evaluation.DEVELOPMENT_SEEDS),
                      mean=10., median=10., minimum=10.)
        # Slot 1 finished after slots 2 and 3 elapsed.
        with patch.object(train.time, 'time', return_value=trainer.started + 3.5 * period), \
                patch.object(train.evaluation, 'evaluate_fresh', return_value=report):
            trainer.select()
            self.assertEqual(trainer.next_selection, 4)
            self.assertFalse(trainer.selection_due())
        resumed = self.trainer()
        with patch.object(train.time, 'time', return_value=resumed.started + 4 * period):
            self.assertFalse(resumed.selection_due())
            resumed.progress['updates'] += 1
            self.assertTrue(resumed.selection_due())

    def test_deadline_manifest_is_reused_on_restart(self):
        args = ['--device','cpu','--run-dir',str(self.root/'bounded'),'--source-checkpoint',str(self.args.source_checkpoint),
                '--calibration',str(self.args.calibration),'--envs','1','--rollout','2','--danger-starts','0','--hours','.000001']
        self.assertEqual(train.main(args), 0)
        manifest = json.loads((self.root/'bounded/manifest.json').read_text())
        summary = json.loads((self.root/'bounded/summary.json').read_text())
        self.assertEqual(manifest['deadline_epoch'], summary['deadline_epoch'])
        self.assertTrue(summary['complete'])
        self.assertTrue((self.root/'bounded/last.pt').is_file())
        self.assertEqual(train.main(args), 0)
        self.assertEqual(summary, json.loads((self.root/'bounded/summary.json').read_text()))


if __name__ == '__main__':
    unittest.main()
