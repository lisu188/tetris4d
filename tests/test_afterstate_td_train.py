"""Afterstate reward indexing, same-game boundaries and deterministic TD recovery."""
import copy
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

    def test_deadline_slot_never_runs_even_if_monotonic_clock_is_behind(self):
        trainer = self.trainer()
        trainer.progress['updates'] = 1
        period = trainer.args.selection_hours * 3600
        trainer.deadline = trainer.started + 4 * period
        for slot in (3, 4, 5):
            trainer.next_selection = slot
            now = max(trainer.deadline + .055, trainer.started + slot * period)
            with self.subTest(slot=slot), patch.object(train.time, 'time', return_value=now):
                self.assertFalse(trainer.selection_due())
        trainer.next_selection = 4
        with patch.object(train.time, 'time', return_value=trainer.deadline + .055):
            with patch.object(trainer, 'select', side_effect=AssertionError('late selection')), \
                    patch.object(trainer, 'step', side_effect=AssertionError('late rollout')):
                self.assertTrue(trainer.run()['complete'])
        events = [json.loads(line)['event'] for line in trainer.metrics.read_text().splitlines()]
        self.assertIn('run_finished', events)
        self.assertNotIn('selection_started', events)

    def test_selection_stop_reserves_checkpoint_time_on_either_clock(self):
        trainer = self.trainer()
        trainer.deadline = 1000.
        trainer.budget = train.base.Budget(1., clock=lambda: 100.)
        trainer.max_step_seconds = trainer.max_checkpoint_seconds = 1.

        def check_clocks(*args, stop_requested, **kwargs):
            # The monotonic clock has an hour left while wall time crosses the save reserve.
            with patch.object(train.time, 'time', return_value=990.):
                self.assertFalse(stop_requested())
            with patch.object(train.time, 'time', return_value=998.):
                self.assertTrue(stop_requested())
                self.assertEqual(trainer.budget.reason, 'time_budget')
            # The monotonic limit still applies when the wall clock moves backward.
            trainer.budget.reason = None
            trainer.budget.deadline = 102.
            with patch.object(train.time, 'time', return_value=500.):
                self.assertTrue(stop_requested())
                self.assertEqual(trainer.budget.reason, 'time_budget')
            for when in (trainer.budget.deadline, trainer.budget.deadline - 10):
                trainer.budget.reason = None

                def signal_during_clock_read():
                    trainer.budget.handle(signal.SIGTERM, None)
                    return when

                with patch.object(train.time, 'time', return_value=500.), \
                        patch.object(trainer.budget, 'clock', signal_during_clock_read):
                    self.assertTrue(stop_requested())
                    self.assertEqual(trainer.budget.reason, f'signal_{signal.SIGTERM}')
            return {'complete': False}

        with patch.object(train.evaluation, 'evaluate_fresh', side_effect=check_clocks):
            trainer.select()

    def test_slow_selection_skips_missed_slots_and_waits_for_new_weights_after_resume(self):
        # The fourth scheduled slot in this fixture falls before its deadline.
        self.manifest['deadline_epoch'] = self.manifest['start_epoch'] + 5*self.args.selection_hours*3600
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

    def interrupted_at_32k(self, device):
        if device == 'cuda':
            train.game.configure_backend('native')
            self.addCleanup(train.game.configure_backend, 'numpy')
        self.args.device, self.args.pieces = device, 32000
        self.args.actor_bound, self.args.target_tau = .5, .01
        self.signature = train.provenance(self.args, json.loads(self.args.calibration.read_text()), [])
        start = train.time.time() - 60
        self.manifest = dict(algorithm=train.ALGORITHM, signature=self.signature, start_epoch=start,
                             deadline_epoch=start+self.args.hours*3600, selection_pieces=self.args.selection_pieces,
                             source_sha256=self.signature['source_sha256'])
        train.base.durable_json(self.root/'manifest.json', self.manifest)
        trainer = self.trainer()
        trainer.emit('run_start')
        while trainer.buffer.steps < self.args.rollout:
            trainer.step()
        trainer.update()
        games = dict(complete=True, seeds=list(train.evaluation.DEVELOPMENT_SEEDS),
                     mean=12., median=10., minimum=1.)
        with patch.object(train.evaluation, 'evaluate_fresh', return_value=games):
            trainer.select()
        trainer.step()
        self.assertEqual(trainer.buffer.steps, 1)
        self.assertTrue(trainer.buffer.pending_valid.all())
        # Place the live episode's counter at the real limit without simulating 32,000 moves.
        trainer.envs[0].state.pieces = trainer.progress['placements'] = 32000
        trainer.budget.handle(signal.SIGTERM, None)
        summary = trainer.run()
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['stop_reason'], f'signal_{signal.SIGTERM}')
        return trainer, train.base.file_hash(self.root/'manifest.json')

    def training_state(self, t):
        def portable(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().contiguous().numpy().tobytes()
            if isinstance(x, dict):
                return {key: portable(v) for key, v in x.items()}
            if isinstance(x, (list, tuple)):
                return type(x)(portable(v) for v in x)
            return x
        return dict(envs=[train.pack_env(e) for e in t.envs], rollout=portable(t.buffer.save()),
                    optimizer=portable(t.optimizer.state_dict()), target=train.v4.v3.model_digest(t.target_head),
                    head=train.v4.v3.model_digest(t.model.value), base=train.v4.v3.model_digest(t.model.base),
                    rng=t.rng.getstate(), cpu_rng=portable(torch.get_rng_state()), python_rng=train.random.getstate(),
                    cuda_rng=portable(torch.cuda.get_rng_state()) if t.device.type=='cuda' else None,
                    progress=dict(t.progress), next_episode=t.next_episode, best=copy.deepcopy(t.best),
                    next_selection=t.next_selection, last_selected_update=t.last_selected_update, deadline=t.deadline)

    def check_uncapped_continuation(self, device):
        original, manifest_sha = self.interrupted_at_32k(device)
        original_sha = train.base.file_hash(original.last)
        best_files = [self.root/n for n in ('best.pt','selection-1.pt','selection-1.json')]
        historical = [train.base.file_hash(p) for p in best_files]
        boundary = self.training_state(original)
        # Reference continuation changes only the cap in a still-resident trainer.
        original.args = copy.copy(self.args)
        original.args.pieces = 0
        original.step()
        self.assertEqual(original.envs[0].state.pieces, 32001)
        self.assertTrue(original.buffer.continues[1,0])
        original.update()
        expected = self.training_state(original)
        self.assertEqual(expected['progress']['truncated_afterstates'], boundary['progress']['truncated_afterstates'])

        flags = ['--source-checkpoint', str(self.args.source_checkpoint), '--calibration', str(self.args.calibration),
                 '--envs','1','--rollout','2','--batch','2','--danger-starts','0','--device',device,
                 '--actor-bound','.5','--target-tau','.01','--run-dir',str(self.root),'--pieces','0',
                 '--uncap-existing-run',manifest_sha]
        amended = train.arguments(flags)
        effective = train.provenance(amended, json.loads(amended.calibration.read_text()), [])
        seen = []
        def stop_after_update(t):
            if not seen:
                seen.append(self.training_state(t))
                self.assertEqual(seen[0],boundary)
            if t.progress['updates'] == expected['progress']['updates']:
                self.assertEqual(self.training_state(t),expected)
                t.budget.handle(signal.SIGTERM,None)
                return True
            return False
        with patch.object(train.Trainer,'should_stop',stop_after_update):
            if device=='cpu':
                self.assertEqual(train.main(flags),0)
            else:
                # The opt-in CUDA fixture shares the device; never acquire the production trainer's run lock.
                receipt = train.cap_amendment(self.root,self.manifest,manifest_sha,effective)
                resumed = train.Trainer(amended,train.base.Budget(1),[],effective,self.manifest,receipt)
                resumed.run()
        self.assertEqual(len(seen),1)
        receipt = train.cap_amendment(self.root,self.manifest,manifest_sha,effective)
        self.assertEqual(receipt['boundary_checkpoint_sha256'],original_sha)
        archived = self.root/receipt['boundary_checkpoint']
        self.assertEqual(train.base.file_hash(archived),original_sha)
        self.assertNotEqual(archived.stat().st_ino, original.last.stat().st_ino)
        repeated = train.Trainer(amended,train.base.Budget(1),[],effective,self.manifest,receipt)
        self.assertEqual(self.training_state(repeated),expected)
        self.assertEqual(train.base.file_hash(self.root/'manifest.json'),manifest_sha)
        self.assertEqual([train.base.file_hash(p) for p in best_files],historical)
        journal = [json.loads(line) for line in repeated.metrics.read_text().splitlines()]
        self.assertEqual(sum(e['event']=='training_cap_amended' for e in journal),1)

    def test_uncapped_cpu_handoff_continues_32k_episode_and_partial_update(self):
        self.check_uncapped_continuation('cpu')

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'explicit CUDA slot required')
    def test_uncapped_cuda_handoff_continues_32k_episode_and_partial_update(self):
        self.check_uncapped_continuation('cuda')

    def test_uncap_existing_requires_only_the_training_limit_to_change(self):
        _, digest = self.interrupted_at_32k('cpu')
        self.args.pieces = 0
        effective = train.provenance(self.args,json.loads(self.args.calibration.read_text()),[])
        for field, value in [('lr', .05),('selection_pieces',0),('hours',9),('pieces',40000)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                train.cap_amendment(self.root,self.manifest,digest,{**effective,field:value})
        with self.assertRaises(ValueError):
            train.cap_amendment(self.root,self.manifest,'0'*64,effective)
        summary_path = self.root/'summary.json'
        summary = json.loads(summary_path.read_text())
        train.base.durable_json(summary_path,{**summary,'complete':True,'stop_reason':'time_budget'})
        with self.assertRaises(ValueError):
            train.cap_amendment(self.root,self.manifest,digest,effective)
        train.base.durable_json(summary_path,summary)
        with patch.object(train.time,'time',return_value=self.manifest['deadline_epoch']+1), self.assertRaises(ValueError):
            train.cap_amendment(self.root,self.manifest,digest,effective)
        self.assertFalse((self.root/'cap-amendment.json').exists())

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
