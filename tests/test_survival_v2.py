from __future__ import annotations

import base64
import contextlib
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickle
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_survival_v2 as v2
from tools import survival_experiments as experiments

neural, game, base = v2.neural, v2.game, v2.base


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.manual_seed(42)
        self.config = neural.Config(width=8, blocks=1, candidates=32, shortlist=8)
        self.model = neural.ColumnPolicy(self.config)
        self.path = self.root / 'original.pt'
        neural.save_checkpoint(self.path, self.model, progress={'updates': 10})

    def tearDown(self):
        self.tmp.cleanup()

    def test_legacy_checkpoint_tensor_count_and_sigmoid_only(self):
        model, _ = neural.load_checkpoint(self.path, torch.device('cpu'))
        self.assertEqual(set(model.state_dict()), set(self.model.state_dict()))
        self.assertFalse(any(isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)) for module in model.modules()))

    def test_headless_and_captured_actions_match(self):
        captured = neural.NeuralAgent(self.path, 'cpu', capture_traces=True)
        headless = neural.NeuralAgent(self.path, 'cpu', capture_traces=False)
        a, b = game.Game(111), game.Game(111)
        for _ in range(20):
            da, db = captured.choose(a), headless.choose(b)
            self.assertEqual(da.move.candidate, db.move.candidate)
            a.commit(da.move)
            b.commit(db.move)
        self.assertIsNotNone(captured.last_trace)
        self.assertIsNone(headless.last_trace)
        self.assertIsNone(headless.pending_trace)

    def test_lazy_trace_only_captures_requested_frame(self):
        agent = neural.NeuralAgent(self.path, 'cpu', capture_traces='lazy')
        state = game.Game(7)
        with patch.object(agent.model, 'forward', wraps=agent.model.forward) as forward:
            agent.choose(state)
            self.assertIsNone(agent.last_trace)
            self.assertIsNotNone(agent.pending_trace)
            self.assertFalse(forward.call_args.kwargs['capture'])
            trace = agent.get_trace()
            self.assertIsNotNone(trace)
            self.assertEqual(forward.call_count, 2)
            self.assertIs(agent.get_trace(), trace)
            self.assertEqual(forward.call_count, 2)

    def test_zero_residual_reproduces_teacher(self):
        config = replace(self.config, policy_mode='teacher-residual', recover_target=True)
        model = neural.ColumnPolicy(config)
        model.rank_out.weight.data.zero_()
        model.rank_out.bias.data.zero_()
        state, target = game.Game(7), 1
        for _ in range(64):
            prepared = neural.prepare(state, target, config)
            target = int(prepared.arrays['targets'])
            with torch.no_grad():
                logits = model(neural.collate([prepared.arrays], torch.device('cpu')))['logits']
            index = int(logits.argmax())
            self.assertEqual(index, prepared.teacher)
            teacher = game.choose_sustained_move(state.board, state.current, target, config.candidates, state.pieces)
            candidate = prepared.options[index]
            self.assertEqual(replace(candidate, score=teacher.move.candidate.score), teacher.move.candidate)
            self.assertAlmostEqual(candidate.score, teacher.move.candidate.score, places=11)
            self.assertEqual(game.materialize_move(state.board, candidate).board, teacher.move.board)
            state.commit(teacher.move)
            if teacher.move.layers:
                target = game.next_hyperlayer_target(state.board, target)

    def test_critic_gradient_does_not_change_actor_encoder(self):
        model = neural.ColumnPolicy(replace(self.config, detach_critic=True, centered_critic=True))
        prepared = neural.prepare(game.Game(5), 1, model.config)
        prediction = model(neural.collate([prepared.arrays], torch.device('cpu')))
        prediction['value'].square().mean().backward()
        for name, parameter in model.named_parameters():
            if not name.startswith('value.'):
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value.parameters()))

    def test_secondary_supervision_distinguishes_primary_ties(self):
        features = torch.zeros(1, 3, neural.FEATURES)
        features[0, :, 8] = torch.tensor([4., 4., 3.]) / 16
        batch = {'features': features, 'valid': torch.tensor([[True, True, True]]),
                 'teacher_prior': torch.tensor([[0., -2., -32.]])}
        better = v2.legacy.categorical_logp(torch.tensor([[2., 0., -8.]]), batch['valid'])
        worse = v2.legacy.categorical_logp(torch.tensor([[0., 2., -8.]]), batch['valid'])
        self.assertLess(v2.secondary_teacher_loss(better, batch), v2.secondary_teacher_loss(worse, batch))

    def test_all_recorded_blocked_states_recover_without_mutation(self):
        data = json.loads((ROOT / 'tests/fixtures/blocked-targets.json').read_text())
        self.assertGreaterEqual(len(data['cases']), 35)
        for row in data['cases']:
            packed = zlib.decompress(base64.b64decode(row['board_zlib_base64']))
            self.assertEqual(hashlib.sha256(packed).hexdigest(), row['board_sha256'])
            state = game.Game(row['seed'], row['piece_set'])
            state.board = struct.unpack('<4096H', packed)
            state.current, state.next_piece = row['current'], row['next_piece']
            for name in ('pieces', 'cells_placed', 'piece_counts'):
                setattr(state, name, row['summary'][name])
            state.layers, state.score = row['summary']['hyperlayers'], row['summary']['score']
            state.game_over = False
            before = (state.board, state.summary(), pickle.dumps(state.stream))
            config = neural.Config(recover_target=False)
            self.assertIsNone(neural.prepare(state, row['target'], config), row['source'])
            self.assertTrue(game.has_legal_move(state.board, state.current))
            prepared = neural.prepare(state, row['target'], replace(config, recover_target=True))
            self.assertIsNotNone(prepared, row['source'])
            self.assertNotEqual(int(prepared.arrays['targets']), row['target'])
            candidate = prepared.options[prepared.teacher]
            move = game.materialize_move(state.board, candidate)
            self.assertEqual(sum(x.bit_count() for x in move.board), sum(x.bit_count() for x in state.board)
                             + game.PIECE_SIZES[state.current] - move.layers * game.AREA)
            self.assertEqual(before, (state.board, state.summary(), pickle.dumps(state.stream)))

    def test_retargeted_move_and_pending_target_replay(self):
        episode = base.Episode(1, game.Game(7))
        episode.pending_retarget, episode.target = 1, 0
        restored = base.restore_episode(episode.snapshot())
        self.assertEqual(restored.target, 0)
        self.assertEqual(restored.pending_retarget, 1)
        prepared = neural.prepare(episode.state, 0, self.config)
        base.apply_move(episode, prepared.options[prepared.teacher], 1, 'test')
        self.assertEqual(episode.moves[0]['retarget_from'], 1)
        restored = base.restore_episode(episode.snapshot())
        self.assertEqual(restored.snapshot(), episode.snapshot())
        changed = json.loads(json.dumps(episode.snapshot()))
        changed['moves'][0]['retarget_from'] = 5
        with self.assertRaises(ValueError):
            base.restore_episode(changed)

    def test_completed_json_recordings_are_compressed_and_replayable(self):
        episode = base.Episode(2, game.Game(7))
        prepared = neural.prepare(episode.state, 1, self.config)
        base.apply_move(episode, prepared.options[prepared.teacher], 1, 'test')
        episode.complete, episode.stop_reason = True, 'piece_limit'
        recorder = v2.CompressedRecorder(self.root / 'games', True)
        path = recorder.write(episode)
        self.assertEqual(path.suffix, '.gz')
        self.assertTrue(base.verify_game(path)['passed'])
        self.assertEqual(recorder.used, sum(p.stat().st_size for p in (self.root / 'games').rglob('*.gz')))

    def test_logger_counts_neural_decisions(self):
        from tetris_benchmark_logging import BenchmarkLog
        agent = neural.NeuralAgent(self.path, 'cpu', capture_traces=False)
        with patch.object(game, 'NEURAL_AGENT', agent):
            logger = BenchmarkLog(game, self.root / 'benchmark.jsonl').install()
            try:
                state = game.Game(7)
                for _ in range(3):
                    state.commit(agent.choose(state).move)
                self.assertEqual(logger.decisions, 3)
                self.assertEqual(logger.moves, 3)
            finally:
                logger.close()

    def test_immutable_checkpoint_snapshot_and_archive(self):
        snapshot, digest = experiments.snapshot_checkpoint(self.path, self.root / 'snapshots')
        self.assertEqual(base.file_hash(snapshot), digest)
        self.path.write_bytes(b'changed original path')
        self.assertEqual(base.file_hash(snapshot), digest)
        run = self.root / 'run'
        run.mkdir()
        (run / 'last.pt').write_bytes(snapshot.read_bytes())
        (run / 'metrics.jsonl').write_text('{"event":"test"}\n')
        output = self.root / 'archive.tar.gz'
        args = type('Arguments', (), dict(run_dir=run, output=output, include_games=True))()
        with contextlib.redirect_stdout(io.StringIO()):
            experiments.archive(args)
        with tarfile.open(output) as package:
            manifest = json.load(package.extractfile('archive-manifest.json'))
            for item in manifest['files']:
                raw = package.extractfile(item['path']).read()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), item['sha256'])
        self.assertEqual(base.file_hash(run / 'last.pt'), digest)
        with self.assertRaises(ValueError):
            experiments.archive(args)


class TrainerTests(RuntimeTests):
    def args(self, name='run', count=1, **changes):
        args = v2.arguments(['--device', 'cpu', '--run-dir', str(self.root / name), '--checkpoint', str(self.path),
                             '--envs', '2', '--rollout', '3', '--batch', '2', '--optimizer-batch', '4',
                             '--calibration-games', '0', '--allow-uncalibrated', '--max-updates', str(count),
                             '--selection-pieces', '3', '--target-kl', '0.9', '--record-games'])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def trainer(self, args):
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            return v2.SurvivalTrainer(args, base.Budget(1))

    def run_training(self, args):
        with contextlib.redirect_stdout(io.StringIO()), base.run_lock(args.run_dir):
            trainer = v2.SurvivalTrainer(args, base.Budget(1))
            trainer.run()
        return trainer

    def test_real_optimizer_minibatches_and_diagnostics(self):
        trainer = self.run_training(self.args())
        self.assertEqual(trainer.progress['placements'], 6)
        self.assertEqual(trainer.progress['optimizer_steps'], 6)
        events = [json.loads(line) for line in trainer.metrics.read_text().splitlines()]
        update = [row for row in events if row['event'] == 'update'][-1]
        self.assertIn('critic', update['per_loss_gradient_norms'])
        self.assertIn('raw_advantages', update)
        self.assertIn('prepare_seconds', update['phase_seconds'])
        self.assertEqual(update['optimizer_steps_this_rollout'], 6)

    def test_exact_v2_resume(self):
        full = self.run_training(self.args('full', 2))
        self.run_training(self.args('split', 1))
        resumed = self.run_training(self.args('split', 1))
        a, b = torch.load(full.last, weights_only=True), torch.load(resumed.last, weights_only=True)
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['extra']['environments'], b['extra']['environments'])
        for key in a['model']:
            self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
        for index, state in a['optimizer']['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][index][key]), (index, key))

    def test_incompatible_resume_is_rejected(self):
        self.run_training(self.args())
        with self.assertRaises(ValueError):
            self.trainer(self.args(optimizer_batch=2))

    def test_rollout_reuses_adjacent_values(self):
        trainer = self.trainer(self.args(rollout=4))
        with patch.object(trainer.model, 'encode', wraps=trainer.model.encode) as encode:
            rollout = trainer.rollout(False)
        self.assertEqual(encode.call_count, 5)
        self.assertEqual(len(rollout.records), 8)
        self.assertLess(trainer.rollout_kl(rollout), 1e-6)

    def test_truncation_bootstraps_without_crossing_reset(self):
        trainer = self.trainer(self.args(rollout=2, pieces=1))
        trainer.model.value[-1].weight.data.zero_()
        trainer.model.value[-1].bias.data.fill_(2.)
        with contextlib.redirect_stdout(io.StringIO()):
            rollout = trainer.rollout(False)
        self.assertTrue(rollout.truncations.all())
        self.assertFalse(rollout.terminals.any())
        np.testing.assert_allclose(rollout.returns, 2.001, atol=1e-6)

    def test_terminal_values_do_not_bootstrap(self):
        trainer = self.trainer(self.args(rollout=2))
        trainer.model.value[-1].weight.data.zero_()
        trainer.model.value[-1].bias.data.fill_(2.)
        original = trainer.prepare_episode
        with patch.object(trainer, 'prepare_episode', side_effect=lambda e: original(e) if e.state.pieces == 0 else None), contextlib.redirect_stdout(io.StringIO()):
            rollout = trainer.rollout(False)
        self.assertTrue(rollout.terminals.all())
        np.testing.assert_allclose(rollout.returns, .001, atol=1e-6)

    def test_calibration_excludes_capped_games(self):
        trainer = self.trainer(self.args(calibration_games=1, calibration_validation_games=1, allow_uncalibrated=False, pieces=2))
        before = v2.actor_digest(trainer.model)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(trainer.calibrate())
        self.assertEqual(trainer.calibration['censored_games'], 2)
        self.assertEqual(trainer.calibration['completed_games'], 0)
        self.assertEqual(trainer.calibration['train'], [])
        self.assertEqual(before, v2.actor_digest(trainer.model))
        self.assertEqual(trainer.progress['runner_updates'], 0)

    def synthetic_calibration(self, trainer):
        c = trainer.calibration
        c['game_index'] = trainer.args.calibration_games + trainer.args.calibration_validation_games
        c['completed_games'] = c['game_index']
        generator = torch.Generator().manual_seed(37)
        width = trainer.model.config.width + neural.CONTEXT
        for split, count in (('train', 160), ('validation', 80)):
            for i in range(count):
                x = torch.randn(width, generator=generator) * .1
                x[0] = i / count
                c[split].append({'x': x, 'y': .05 + 2 * float(x[0]), 'seed': 2000000 + (split == 'validation')})

    def test_critic_calibration_improves_heldout_synthetic_targets_without_actor_change(self):
        trainer = self.trainer(self.args(calibration_games=1, calibration_validation_games=1, calibration_steps=5,
                                         allow_uncalibrated=False))
        self.synthetic_calibration(trainer)
        before = v2.actor_digest(trainer.model)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(trainer.calibrate())
        self.assertEqual(before, v2.actor_digest(trainer.model))
        report = json.loads((trainer.root / 'critic-calibration.json').read_text())
        self.assertTrue(report['passed'])
        self.assertLess(report['validation']['mse'], report['validation']['constant_mse'])

    def test_calibration_fit_resumes_exactly(self):
        args = dict(calibration_games=1, calibration_validation_games=1, calibration_steps=5, allow_uncalibrated=False)
        full = self.trainer(self.args('cal-full', **args))
        self.synthetic_calibration(full)
        with contextlib.redirect_stdout(io.StringIO()):
            full.calibrate()
        split = self.trainer(self.args('cal-split', **args))
        self.synthetic_calibration(split)
        with patch.object(split.budget, 'expired', side_effect=[False, False, True]), contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(split.calibrate())
            split.save('calibration_interrupt')
        resumed = self.trainer(self.args('cal-split', **args))
        with contextlib.redirect_stdout(io.StringIO()):
            resumed.calibrate()
        for key, value in full.model.state_dict().items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)

    def test_new_checkpoint_loads_in_old_cli_agent(self):
        trainer = self.run_training(self.args())
        agent = neural.NeuralAgent(trainer.last, 'cpu', capture_traces=False)
        self.assertEqual(agent.model.config.policy_mode, 'teacher-residual')
        self.assertIsNotNone(agent.choose(game.Game(7)))

    def test_invalid_calibration_override_fails(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            v2.arguments(['--calibration-games', '0'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            v2.arguments(['--optimizer-batch', '0'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on this worker')
    def test_cuda_v2_optimizer(self):
        trainer = self.run_training(self.args('cuda', device='cuda'))
        self.assertGreater(trainer.progress['optimizer_steps'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
