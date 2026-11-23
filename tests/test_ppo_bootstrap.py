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
import train_survival_v2 as v2

neural, game, base = v2.neural, v2.game, v2.base


class ReturnTests(unittest.TestCase):
    def test_default_is_bitwise_legacy_gae_and_does_not_mutate_inputs(self):
        generator = np.random.default_rng(17)
        rewards = np.full((11, 3), .001, dtype=np.float32)
        values = generator.normal(size=rewards.shape).astype(np.float32)
        following = np.concatenate((values[1:], np.ones((1, 3), np.float32)), axis=0)
        terminal = np.zeros(rewards.shape, dtype=bool)
        truncated = terminal.copy()
        terminal[4, 0], truncated[7, 1] = True, True
        original = values.copy(), following.copy()
        expected = neural.gae(rewards, values, following, terminal, truncated, 1., .999)
        actual = v2.survival_gae(rewards, values, following, terminal, truncated, .999)
        for observed, wanted in zip(actual, expected):
            np.testing.assert_array_equal(observed, wanted)
        np.testing.assert_array_equal(values, original[0])
        np.testing.assert_array_equal(following, original[1])

    def test_bounded_lambda_returns_match_forward_return_at_mixed_resets(self):
        rewards = np.full((4, 2), .001, dtype=np.float32)
        values = np.array([[-2., 3.], [-1., -2.], [4., -3.], [-2., 5.]], dtype=np.float32)
        following = np.array([[-1., -2.], [0., -3.], [-2., 5.], [-4., 0.]], dtype=np.float32)
        terminal = np.array([[False, False], [True, False], [False, False], [False, True]])
        truncated = np.array([[False, False], [False, True], [False, False], [False, False]])
        original = values.copy(), following.copy()
        advantage, returns = v2.survival_gae(rewards, values, following, terminal, truncated, .9, True)
        expected = np.zeros_like(rewards)
        for step in range(len(rewards) - 1, -1, -1):
            for env in range(2):
                next_value = max(float(following[step, env]), 0.)
                if terminal[step, env]:
                    future = 0.
                elif truncated[step, env] or step == len(rewards) - 1:
                    future = next_value
                else:
                    future = .1 * next_value + .9 * expected[step + 1, env]
                expected[step, env] = rewards[step, env] + future
        np.testing.assert_allclose(returns, expected, atol=4e-7)
        np.testing.assert_allclose(advantage + np.maximum(values, 0.), expected, atol=4e-7)
        self.assertGreater(float(returns.min()), 0.)
        np.testing.assert_array_equal(values, original[0])
        np.testing.assert_array_equal(following, original[1])

    def test_positive_values_are_bitwise_unchanged(self):
        rewards = np.full((8, 2), .001, dtype=np.float32)
        values = np.arange(16, dtype=np.float32).reshape(8, 2) / 4
        following = np.concatenate((values[1:], np.ones((1, 2), np.float32)), axis=0)
        terminal = np.zeros(rewards.shape, dtype=bool)
        expected = v2.survival_gae(rewards, values, following, terminal, terminal, .999)
        actual = v2.survival_gae(rewards, values, following, terminal, terminal, .999, True)
        for observed, wanted in zip(actual, expected):
            np.testing.assert_array_equal(observed, wanted)


class TrainerBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.manual_seed(42)
        self.path = self.root / 'initial.pt'
        model = neural.ColumnPolicy(neural.Config(width=8, blocks=1, candidates=32, shortlist=8))
        neural.save_checkpoint(self.path, model)

    def tearDown(self):
        self.temporary.cleanup()

    def args(self, name='run', enabled=True, **changes):
        arguments = ['--device', 'cpu', '--run-dir', str(self.root / name), '--checkpoint', str(self.path),
                     '--envs', '2', '--rollout', '3', '--batch', '2', '--optimizer-batch', '4',
                     '--calibration-games', '0', '--allow-uncalibrated', '--target-kl', '.9',
                     '--imitation', '0', '--secondary-weight', '0', '--regret-weight', '0', '--entropy', '0']
        if enabled:
            arguments.append('--nonnegative-bootstrap')
        args = v2.arguments(arguments)
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def trainer(self, args):
        args.run_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            return v2.SurvivalTrainer(args, base.Budget(1))

    def negative_critic(self, trainer):
        with torch.no_grad():
            trainer.model.value[-1].weight.zero_()
            trainer.model.value[-1].bias.fill_(-2.)

    def advance(self, trainer, count=1):
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(count):
                trainer.update(trainer.rollout(False), False)
            trainer.save('bootstrap_test')

    def test_negative_predictions_produce_valid_targets_and_remain_visible(self):
        trainer = self.trainer(self.args())
        self.negative_critic(trainer)
        rollout = trainer.rollout(False)
        np.testing.assert_array_equal(rollout.old_values, -2.)
        self.assertGreater(float(rollout.returns.min()), 0.)
        self.assertLess(float(rollout.returns.max()), .004)
        np.testing.assert_array_equal(rollout.raw_advantages, rollout.returns)

    def test_disabled_option_retains_negative_bootstrap_behavior(self):
        trainer = self.trainer(self.args(enabled=False))
        self.negative_critic(trainer)
        rollout = trainer.rollout(False)
        self.assertLess(float(rollout.returns.max()), -1.99)
        np.testing.assert_array_equal(rollout.old_values, -2.)

    def test_terminal_negative_critic_does_not_reinforce_a_large_false_gain(self):
        trainer = self.trainer(self.args(rollout=2))
        self.negative_critic(trainer)
        original = trainer.prepare_episode
        with patch.object(trainer, 'prepare_episode', side_effect=lambda e: original(e) if e.state.pieces == 0 else None), contextlib.redirect_stdout(io.StringIO()):
            rollout = trainer.rollout(False)
        self.assertTrue(rollout.terminals.all())
        np.testing.assert_allclose(rollout.returns, .001, atol=1e-7)
        np.testing.assert_allclose(rollout.raw_advantages, .001, atol=1e-7)
        np.testing.assert_array_equal(rollout.old_values, -2.)

    def test_truncation_floors_negative_bootstrap_without_crossing_reset(self):
        trainer = self.trainer(self.args(rollout=2, pieces=1))
        self.negative_critic(trainer)
        with contextlib.redirect_stdout(io.StringIO()):
            rollout = trainer.rollout(False)
        self.assertTrue(rollout.truncations.all())
        self.assertFalse(rollout.terminals.any())
        np.testing.assert_allclose(rollout.returns, .001, atol=1e-7)

    def test_critic_training_keeps_a_corrective_gradient_below_zero(self):
        trainer = self.trainer(self.args(ppo_epochs=1))
        self.negative_critic(trainer)
        rollout = trainer.rollout(False)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.update(rollout, False)
        self.assertGreater(float(trainer.model.value[-1].bias.detach()), -2.)
        self.assertLess(float(trainer.model.value[-1].bias.grad), 0.)
        update = [json.loads(row) for row in trainer.metrics.read_text().splitlines()
                  if json.loads(row).get('event') == 'update'][-1]
        self.assertGreater(update['value_loss'], 3.9)
        self.assertEqual(update['critic_bootstrapped_diagnostics']['predictions_pieces']['minimum'], -2000.)
        self.assertGreater(update['critic_bootstrapped_diagnostics']['targets_pieces']['minimum'], 0.)

    def test_checkpoint_records_option_and_rejects_changed_resume(self):
        trainer = self.trainer(self.args())
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.save('bootstrap_test')
        saved = torch.load(trainer.last, weights_only=True)
        self.assertTrue(saved['extra']['signature']['nonnegative_bootstrap'])
        self.assertTrue(saved['extra']['nonnegative_bootstrap'])
        with self.assertRaisesRegex(ValueError, 'Run settings or source changed'):
            self.trainer(self.args(enabled=False))

    def test_enabled_run_resumes_exactly(self):
        full = self.trainer(self.args('full'))
        self.negative_critic(full)
        self.advance(full, 2)
        split = self.trainer(self.args('split'))
        self.negative_critic(split)
        self.advance(split)
        resumed = self.trainer(self.args('split'))
        self.advance(resumed)
        a, b = torch.load(full.last, weights_only=True), torch.load(resumed.last, weights_only=True)
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['extra']['environments'], b['extra']['environments'])
        for name, value in a['model'].items():
            self.assertTrue(torch.equal(value, b['model'][name]), name)
        for index, state in a['optimizer']['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][index][key]), (index, key))

    def test_frozen_encoder_stays_fixed_across_updates_and_resume(self):
        args = self.args(freeze_encoder=True, imitation=.1)
        trainer = self.trainer(args)
        before = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
        frozen = [name for name in before if name.startswith(('stem.', 'blocks.'))]
        self.assertTrue(frozen)
        self.advance(trainer)
        resumed = self.trainer(args)
        optimized = {id(p) for group in resumed.optimizer.param_groups for p in group['params']}
        self.advance(resumed)
        after = dict(resumed.model.named_parameters())
        for name in frozen:
            self.assertFalse(after[name].requires_grad, name)
            self.assertNotIn(id(after[name]), optimized, name)
            self.assertIsNone(after[name].grad, name)
            self.assertTrue(torch.equal(before[name], after[name]), name)
        for head in ('rank_out.weight', 'value.2.bias'):
            self.assertFalse(torch.equal(before[head], after[head]), head)
        checkpoint = torch.load(resumed.last, weights_only=True)
        self.assertTrue(checkpoint['extra']['freeze_encoder'])
        self.assertTrue(checkpoint['extra']['signature']['freeze_encoder'])
        with self.assertRaisesRegex(ValueError, 'Run settings or source changed'):
            self.trainer(self.args(imitation=.1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
