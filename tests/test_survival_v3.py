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
import train_survival_v3 as v3


class FreshTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        v3.game.configure_backend('numpy')
        v3.game.load_checkpoint(ROOT/'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.args = v3.arguments([
            '--run-dir', str(self.directory/'run'), '--device', 'cpu', '--backend', 'numpy',
            '--weights', str(ROOT/'models/trained-sigmoid.json'), '--hours', '.1',
            '--threads', '1', '--envs', '2', '--rollout', '3', '--batch', '2',
            '--optimizer-batch', '3', '--gradient-every', '100',
            '--calibration-games', '0', '--allow-uncalibrated', '--seed', '650000',
            '--calibration-seed', '5500000', '--selection-seeds', '980101',
            '--selection-pieces', '3', '--gradient-clip-scope', 'separate'])

    def trainer(self):
        return v3.SurvivalTrainer(self.args, v3.base.Budget(.1))

    def test_fresh_start_ignores_legacy_discovery_and_inherits_no_optimizer(self):
        old = self.directory/'models/neural-policy.pt'
        old.parent.mkdir()
        old.write_bytes(b'old checkpoint must never be opened')
        with patch.object(v3.base, 'ROOT', self.directory), patch.object(
                v3.neural, 'load_checkpoint', side_effect=AssertionError('Inherited checkpoint')):
            trainer = self.trainer()
        self.assertIsNone(trainer.initial_checkpoint)
        self.assertEqual(trainer.progress['updates'], 0)
        self.assertEqual(trainer.progress['placements'], 0)
        self.assertEqual(len(trainer.optimizer.state), 0)
        proof = json.loads((trainer.root/'initialization.json').read_bytes())
        self.assertTrue(proof['fresh_random_trainable_weights'])
        self.assertFalse(proof['inherited_optimizer'])
        self.assertIsNone(proof['inherited_checkpoint'])
        self.assertEqual(proof['model_sha256'], v3.model_digest(trainer.model))
        self.assertEqual(proof['parameters'], 75938)

    def test_external_checkpoint_is_rejected(self):
        for option in (['--checkpoint', 'old.pt'], ['--checkpoint=old.pt']):
            with self.assertRaisesRegex(ValueError, 'starts fresh'):
                v3.arguments(option)

    def test_frozen_reference_is_rejected_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'frozen reference regularization'):
            v3.arguments(['--reference-kl-weight', '.1', '--reference-checkpoint', 'old.pt'])

    def test_softplus_calibration_initializes_from_training_only_mean(self):
        trainer = self.trainer()
        actor = v3.v2.actor_digest(trainer.model)
        x = torch.randn(48, 150)
        y = .1 + 3*torch.sigmoid(x[:, 0])
        trainer.initialize_calibration_output(x, y)
        torch.testing.assert_close(trainer.model.critic(x), y.mean().expand_as(y), rtol=1e-6, atol=1e-6)
        before = torch.nn.functional.mse_loss(trainer.model.critic(x), y).item()
        optimizer = torch.optim.Adam(trainer.model.value.parameters(), lr=.003)
        for _ in range(60):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(trainer.model.critic(x), y)
            loss.backward()
            optimizer.step()
        after = torch.nn.functional.mse_loss(trainer.model.critic(x), y).item()
        self.assertLess(after, before*.5)
        self.assertEqual(actor, v3.v2.actor_digest(trainer.model))
        self.assertTrue(bool((trainer.model.critic(x) >= 0).all()))

    def test_real_rollout_update_and_resume_preserve_fresh_identity(self):
        trainer = self.trainer()
        self.assertTrue(trainer.calibrate())
        proof = (trainer.root/'initialization.json').read_bytes()
        rollout = trainer.rollout(False)
        self.assertEqual(len(rollout.records), 6)
        self.assertTrue(np.isfinite(rollout.returns).all())
        self.assertTrue((rollout.old_values >= 0).all())
        self.assertTrue((rollout.returns >= -1e-6).all())
        trainer.update(rollout, False)
        self.assertEqual(trainer.progress['runner_updates'], 1)
        self.assertEqual(trainer.progress['placements'], 6)
        self.assertGreater(trainer.progress['optimizer_steps'], 0)
        trainer.save('test_checkpoint')
        expected_rng = torch.rand(5)
        resumed = self.trainer()
        torch.testing.assert_close(torch.rand(5), expected_rng, rtol=0, atol=0)
        for name, value in trainer.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(resumed.progress, trainer.progress)
        self.assertEqual(resumed.optimizer.state_dict()['param_groups'], trainer.optimizer.state_dict()['param_groups'])
        self.assertEqual((trainer.root/'initialization.json').read_bytes(), proof)
        manifest = json.loads((trainer.root/'manifest.json').read_bytes())
        self.assertEqual(manifest['algorithm'], 'survival-ppo-v3')
        self.assertEqual(manifest['calibration_stage'], 'explicitly_skipped')
        self.assertEqual(resumed.model.config.architecture, 'survival-feature-v3')
        self.assertIsNone(resumed.initial_checkpoint)

    def test_initial_policy_can_be_protected_before_any_actor_update(self):
        trainer = self.trainer()
        trainer.calibrate()
        trainer.save('calibrated')
        def evaluation(policy):
            return {'complete': True, 'runs': [], 'score': [10, 9, 10] if policy == 'neural' else [8, 7, 8],
                    'signature': trainer.signature}
        with patch.object(trainer, 'evaluate_policy', side_effect=evaluation), patch.object(
                trainer, 'sampled_policy_check', return_value={'complete': True, 'runs': [], 'score': [9, 8, 9]}):
            trainer.select()
        self.assertEqual(trainer.progress['updates'], 0)
        self.assertEqual(trainer.best_record['runner_updates'], 0)
        self.assertEqual(v3.base.file_hash(trainer.best), v3.base.file_hash(trainer.last))
        self.assertTrue((trainer.root/'sampled-selection-00000000.json').is_file())

    def test_censored_sampled_game_has_no_false_monte_carlo_target_and_preserves_rng(self):
        trainer = self.trainer()
        before = torch.get_rng_state().clone()
        digest = v3.model_digest(trainer.model)
        result = trainer.sampled_policy_check()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(v3.model_digest(trainer.model), digest)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['runs']), 1)
        run = result['runs'][0]
        self.assertEqual(run['pieces'], 3)
        self.assertTrue(run['capped'])
        self.assertFalse(run['critic_uncensored'])
        self.assertIsNone(run['critic_monte_carlo'])


if __name__ == '__main__':
    unittest.main()
