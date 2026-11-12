from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_survival as ppo


class ObjectiveTests(unittest.TestCase):
    def features(self):
        features = torch.zeros(1, 4, ppo.neural.FEATURES)
        features[0, :, 8] = torch.tensor([4., 4., 3., 15.]) / 16
        return features, torch.tensor([[True, True, True, False]])

    def test_tied_teacher_placements_are_all_accepted(self):
        features, valid = self.features()
        winners, regret = ppo.utility_targets(features, valid)
        self.assertEqual(winners.tolist(), [[True, True, False, False]])
        self.assertEqual(regret.tolist(), [[0., 0., .0625, 0.]])
        loss1 = ppo.teacher_losses(ppo.categorical_logp(torch.tensor([[8., -8., -8., 100.]]), valid), features, valid)[0]
        loss2 = ppo.teacher_losses(ppo.categorical_logp(torch.tensor([[-8., 8., -8., 100.]]), valid), features, valid)[0]
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)
        self.assertLess(loss1.item(), 1e-5)

    def test_upper_holes_use_the_real_policy_penalty(self):
        features, valid = self.features()
        features[0, 0, 9] = 1 / 64
        winners, regret = ppo.utility_targets(features, valid)
        self.assertEqual(winners.tolist(), [[False, True, False, False]])
        self.assertEqual(float(regret[0, 0]), ppo.game.FUTURE_HOLE_PENALTY / 16)

    def test_bad_candidates_have_nonzero_regret(self):
        features, valid = self.features()
        logp = ppo.categorical_logp(torch.tensor([[0., 0., 6., 0.]]), valid)
        loss, regret, _ = ppo.teacher_losses(logp, features, valid)
        self.assertGreater(loss.item(), 5)
        self.assertGreater(regret.item(), .06)

    def test_clipping_handles_both_advantage_signs(self):
        old = torch.log(torch.tensor([[.5, .5], [.5, .5]]))
        new = torch.log(torch.tensor([[.9, .1], [.1, .9]]))
        loss, fraction = ppo.clipped_policy_loss(new, old, torch.tensor([0, 0]), torch.tensor([1., -1.]), .2)
        self.assertAlmostEqual(loss.item(), -.2, places=6)
        self.assertEqual(fraction.item(), 1.)

    def test_exact_kl_ignores_padding_and_keeps_gradients_finite(self):
        valid = torch.tensor([[True, True, False]])
        logits = torch.tensor([[0., 1., 99.]], requires_grad=True)
        current = ppo.categorical_logp(logits, valid)
        old = ppo.categorical_logp(torch.zeros_like(logits), valid)
        divergence = ppo.exact_kl(old, current, valid)
        self.assertGreater(divergence.item(), 0)
        self.assertEqual(ppo.exact_kl(current.detach(), current, valid).item(), 0)
        divergence.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0)

    def test_invalid_settings_and_empty_masks_fail(self):
        for arguments in (['--ppo-epochs', '0'], ['--target-kl', 'nan'], ['--clip-ratio', '1'], ['--regret-weight', '-1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                ppo.arguments(arguments)
        with self.assertRaises(ValueError):
            ppo.utility_targets(torch.zeros(1, 2, ppo.neural.FEATURES), torch.zeros(1, 2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            ppo.categorical_logp(torch.tensor([[float('nan')]]), torch.ones(1, 1, dtype=torch.bool))

    def test_default_initialization_prefers_rl_without_claiming_it_wins(self):
        with patch.object(Path, 'is_file', return_value=True):
            args = ppo.arguments([])
        self.assertEqual(args.checkpoint.name, 'neural-rl.pt')
        self.assertEqual(args.rollout, 128)
        self.assertEqual(args.run_dir.name, 'survival-ppo')
        self.assertEqual(args.ppo_epochs, 3)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        ppo.game.configure_backend('numpy')
        ppo.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.manual_seed(82)
        model = ppo.neural.ColumnPolicy(ppo.neural.Config(width=8, blocks=1, candidates=16, shortlist=8))
        self.path = self.root / 'initial.pt'
        ppo.neural.save_checkpoint(self.path, model, progress={'updates': 7})
        self.original = self.path.read_bytes()

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, name='run', count=2, **changes):
        args = ppo.arguments(['--device', 'cpu', '--run-dir', str(self.root / name), '--checkpoint', str(self.path),
                              '--envs', '2', '--rollout', '2', '--batch', '2', '--warmup-updates', '0',
                              '--max-updates', str(count), '--selection-pieces', '4', '--record-games'])
        for key, value in changes.items():
            setattr(args, key, value)
        return args

    def train(self, args):
        with contextlib.redirect_stdout(io.StringIO()), ppo.base.run_lock(args.run_dir):
            trainer = ppo.SurvivalTrainer(args, ppo.base.Budget(1))
            summary = trainer.run()
        return trainer, summary

    def test_optimizer_changes_weights_and_records_objective(self):
        trainer, result = self.train(self.args())
        self.assertEqual(result['placements'], 8)
        self.assertEqual(result['runner_updates'], 2)
        self.assertGreaterEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['updates'], 7 + result['optimizer_steps'])
        checkpoint = torch.load(trainer.last, weights_only=True)
        self.assertEqual(checkpoint['extra']['algorithm'], ppo.ALGORITHM)
        self.assertEqual(checkpoint['extra']['value_semantics'], 'expected_remaining_placements_times_0.001')
        self.assertEqual(checkpoint['extra']['initial_checkpoint_sha256'], ppo.base.file_hash(self.path))
        initial = torch.load(self.path, weights_only=True)['model']
        self.assertFalse(torch.equal(initial['rank1.weight'], checkpoint['model']['rank1.weight']))
        self.assertEqual(self.path.read_bytes(), self.original)
        diagnostics = json.loads((trainer.root / 'diagnostics.json').read_text())
        self.assertEqual(diagnostics['checkpoint_sha256'], ppo.base.file_hash(trainer.last))
        for path in (trainer.root / 'games').rglob('*.json'):
            self.assertTrue(ppo.base.verify_game(path)['passed'])

    def test_transfer_resets_value_but_keeps_actor_weights(self):
        args = self.args()
        with contextlib.redirect_stdout(io.StringIO()), ppo.base.run_lock(args.run_dir):
            trainer = ppo.SurvivalTrainer(args, ppo.base.Budget(1))
        self.assertEqual(trainer.model.value[-1].weight.count_nonzero().item(), 0)
        initial = torch.load(self.path, weights_only=True)['model']
        self.assertTrue(torch.equal(trainer.model.rank1.weight.cpu(), initial['rank1.weight']))

    def test_exact_resume_preserves_all_model_and_optimizer_state(self):
        uninterrupted, _ = self.train(self.args('full', 2))
        self.train(self.args('resumed', 1))
        resumed, _ = self.train(self.args('resumed', 1))
        a, b = torch.load(uninterrupted.last, weights_only=True), torch.load(resumed.last, weights_only=True)
        for name in a['model']:
            self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['extra']['environments'], b['extra']['environments'])
        for index, state in a['optimizer']['state'].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, b['optimizer']['state'][index][key]))
                else:
                    self.assertEqual(value, b['optimizer']['state'][index][key])

    def test_new_settings_cannot_silently_resume(self):
        self.train(self.args('run', 1))
        with self.assertRaises(ValueError):
            self.train(self.args('run', 1, ppo_epochs=2))
        with contextlib.redirect_stdout(io.StringIO()), ppo.base.run_lock(self.root / 'run'):
            with self.assertRaises(ValueError):
                ppo.base.Trainer(self.args('run', 1), ppo.base.Budget(1))

    def test_kl_guard_stops_later_epochs(self):
        args = self.args()
        with contextlib.redirect_stdout(io.StringIO()), ppo.base.run_lock(args.run_dir):
            trainer = ppo.SurvivalTrainer(args, ppo.base.Budget(1))
            rollout = trainer.rollout(False)
            with patch.object(trainer, 'rollout_kl', side_effect=[0., 1., 1.]):
                trainer.update(rollout, False)
        self.assertEqual(trainer.progress['optimizer_steps'], 1)
        self.assertEqual(trainer.progress['kl_stopped_updates'], 1)

    def test_behavior_probabilities_match_the_pre_update_model(self):
        args = self.args()
        with contextlib.redirect_stdout(io.StringIO()), ppo.base.run_lock(args.run_dir):
            trainer = ppo.SurvivalTrainer(args, ppo.base.Budget(1))
            rollout = trainer.rollout(False)
        self.assertEqual(len(rollout.records), 4)
        self.assertLess(trainer.rollout_kl(rollout), 1e-6)
        self.assertTrue(all(torch.isfinite(row).all() for row in rollout.old_logp))

    def test_warmup_is_not_off_policy_ppo(self):
        trainer, result = self.train(self.args('warmup', 1, warmup_updates=1))
        self.assertEqual(result['warmup_updates'], 1)
        self.assertEqual(result['optimizer_steps'], 1)
        self.assertEqual(result['survival_updates'], 0)
        actions = [move['behavior'] for env in trainer.environments for move in env.moves]
        self.assertEqual(set(actions), {'teacher'})

    def test_new_checkpoint_loads_in_existing_neural_agent(self):
        trainer, _ = self.train(self.args(count=1))
        agent = ppo.neural.NeuralAgent(trainer.last, 'cpu')
        self.assertIsNotNone(agent.choose(ppo.game.Game(111)))
        self.assertEqual(agent.metadata['value_semantics'], 'expected_remaining_placements_times_0.001')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable on this runner')
    def test_cuda_update(self):
        _, report = self.train(self.args('cuda', 1, device='cuda'))
        self.assertEqual(report['placements'], 4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
