from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural
from tetris_survival_v3 import ARCHITECTURE, SurvivalFeaturePolicy


class SurvivalArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        neural.game.configure_backend('numpy')
        neural.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        torch.manual_seed(31)
        self.config = neural.Config(architecture=ARCHITECTURE, chunk=3, policy_mode='teacher-prior',
                                    teacher_prior_scale=.125, recover_target=True)
        self.model = neural.create_model(self.config)
        self.batch = {'boards': torch.randint(0, 65536, (2, 4096), dtype=torch.int32),
                      'targets': torch.tensor([4, 7]), 'context': torch.randn(2, neural.CONTEXT),
                      'features': torch.randn(2, 5, neural.FEATURES),
                      'indices': torch.randint(0, 4096, (2, 5, neural.COLUMNS)),
                      'column_mask': torch.arange(neural.COLUMNS)[None, None, :].expand(2, 5, -1) < 3,
                      'valid': torch.tensor([[True] * 5, [True, True, True, False, False]]),
                      'teacher_prior': -torch.arange(5, dtype=torch.float32)[None].expand(2, -1)}

    def test_finite_logits_positive_value_and_fixed_critic_input_contract(self):
        output = self.model(self.batch)
        self.assertTrue(torch.isfinite(output['logits'][self.batch['valid']]).all())
        self.assertTrue(torch.isneginf(output['logits'][~self.batch['valid']]).all())
        self.assertTrue(torch.isfinite(output['value']).all())
        self.assertTrue((output['value'] > 0).all())
        self.assertEqual(output['completion'].shape, (2, 0))
        spatial, pooled, value, _, _ = self.model.encode(self.batch['boards'], self.batch['targets'], self.batch['context'])
        self.assertEqual(spatial.shape, (2, 34, 16, 16, 16))
        self.assertEqual(pooled.shape, (2, 68))
        joined = torch.cat((pooled, self.batch['context']), 1)
        self.assertTrue(torch.equal(joined, output['critic_input']))
        self.assertTrue(torch.equal(value, self.model.critic(joined)))
        self.assertTrue(torch.equal(spatial, neural.encode_columns(self.batch['boards'], self.batch['targets'])))

    def test_actor_and_critic_have_disjoint_gradient_paths(self):
        self.model(self.batch)['value'].sum().backward()
        for name, parameter in self.model.named_parameters():
            if not name.startswith('value.'):
                self.assertIsNone(parameter.grad, name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.value.parameters()))
        self.model.zero_grad(set_to_none=True)
        logits = self.model(self.batch)['logits']
        logits[self.batch['valid']].sum().backward()
        for name, parameter in self.model.named_parameters():
            if name.startswith('value.'):
                self.assertIsNone(parameter.grad, name)
        self.assertGreater(float(self.model.rank_out.weight.grad.abs().sum()), 0.)

    def test_actor_update_cannot_move_critic_features_or_prediction(self):
        before = self.model(self.batch)
        actor = [p for name, p in self.model.named_parameters() if not name.startswith('value.')]
        optimizer = torch.optim.AdamW(actor, lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(before['logits'], torch.tensor([0, 1])).backward()
        optimizer.step()
        after = self.model(self.batch)
        self.assertTrue(torch.equal(before['critic_input'], after['critic_input']))
        self.assertTrue(torch.equal(before['value'], after['value']))
        self.assertFalse(torch.equal(before['logits'], after['logits']))

    def test_softplus_critic_keeps_corrective_gradient_for_negative_raw_value(self):
        with torch.no_grad():
            self.model.value[-1].weight.zero_()
            self.model.value[-1].bias.fill_(-2.)
        output = self.model(self.batch)
        F.mse_loss(output['value'], torch.full((2,), 10.)).backward()
        self.assertTrue((output['value'] > 0).all())
        self.assertLess(float(self.model.value[-1].bias.grad), 0.)

    def test_candidate_permutation_only_permutates_logits(self):
        permutation = torch.tensor([2, 0, 4, 1, 3])
        candidate_keys = {'features', 'indices', 'column_mask', 'valid', 'teacher_prior'}
        permuted = {key: value[:, permutation] if key in candidate_keys else value
                    for key, value in self.batch.items()}
        a, b = self.model(self.batch), self.model(permuted)
        torch.testing.assert_close(b['logits'], a['logits'][:, permutation], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(b['value'], a['value'], rtol=0, atol=0)

    def test_arbitrary_candidate_padding_cannot_affect_valid_scores(self):
        extra = {'features': torch.full((2, 3, neural.FEATURES), float('nan')),
                 'indices': torch.full((2, 3, neural.COLUMNS), 999999),
                 'column_mask': torch.ones((2, 3, neural.COLUMNS), dtype=torch.bool),
                 'valid': torch.zeros((2, 3), dtype=torch.bool),
                 'teacher_prior': torch.full((2, 3), float('nan'))}
        padded = {key: torch.cat((value, extra[key]), 1) if key in extra else value
                  for key, value in self.batch.items()}
        a, b = self.model(self.batch), self.model(padded)
        torch.testing.assert_close(b['logits'][:, :5], a['logits'], rtol=1e-6, atol=1e-8)
        self.assertTrue(torch.isneginf(b['logits'][:, 5:]).all())
        torch.testing.assert_close(b['value'], a['value'], rtol=0, atol=0)
        loss = F.cross_entropy(b['logits'], torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_local_mean_max_ignore_masked_columns_and_handle_empty_padding(self):
        spatial = torch.zeros(1, 34, 16, 16, 16)
        flat = spatial.flatten(2)
        flat[:, :, 0], flat[:, :, 1], flat[:, :, 2] = -5., -2., -10.
        indices = torch.tensor([[[0, 1, 2, 999999], [999999, 999999, 999999, 999999]]])
        mask = torch.tensor([[[True, True, False, False], [False] * 4]])
        local = self.model.local_statistics(spatial, indices, mask)
        torch.testing.assert_close(local[0, 0, :34], torch.full((34,), -3.5))
        torch.testing.assert_close(local[0, 0, 34:], torch.full((34,), -2.))
        self.assertTrue(torch.equal(local[0, 1], torch.zeros(68)))

    def test_empty_candidate_state_or_empty_valid_candidate_is_rejected(self):
        all_padding = dict(self.batch, valid=torch.zeros_like(self.batch['valid']))
        with self.assertRaisesRegex(ValueError, 'valid candidate'):
            self.model(all_padding)
        empty_columns = dict(self.batch, column_mask=torch.zeros_like(self.batch['column_mask']))
        with self.assertRaisesRegex(ValueError, 'affected column'):
            self.model(empty_columns)

    def test_batch_matches_individual_inference_and_chunking(self):
        together = self.model(self.batch)
        for index in range(2):
            individual = self.model({key: value[index:index + 1] for key, value in self.batch.items()})
            for key in ('logits', 'value', 'critic_input'):
                torch.testing.assert_close(individual[key], together[key][index:index + 1], rtol=1e-5, atol=1e-7)
        self.model.config = replace(self.config, chunk=1)
        split = self.model(self.batch)
        torch.testing.assert_close(split['logits'], together['logits'], rtol=1e-5, atol=1e-7)

    def test_checkpoint_factory_roundtrip_and_rng_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'v3.pt'
            neural.save_checkpoint(path, self.model, progress={'updates': 1})
            expected_random = torch.rand(5)
            loaded, checkpoint = neural.load_checkpoint(path, torch.device('cpu'), restore_rng=True)
            self.assertIsInstance(loaded, SurvivalFeaturePolicy)
            self.assertEqual(checkpoint['config']['architecture'], ARCHITECTURE)
            self.assertTrue(torch.equal(torch.rand(5), expected_random))
            original, restored = self.model(self.batch), loaded(self.batch)
            for key in ('logits', 'value', 'critic_input'):
                self.assertTrue(torch.equal(original[key], restored[key]), key)

    def test_legacy_checkpoint_without_architecture_keeps_column_policy(self):
        old = neural.ColumnPolicy(neural.Config(width=8, blocks=1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            neural.save_checkpoint(path, old)
            payload = torch.load(path, weights_only=True)
            del payload['config']['architecture']
            torch.save(payload, path)
            loaded, _ = neural.load_checkpoint(path, torch.device('cpu'))
            self.assertIsInstance(loaded, neural.ColumnPolicy)
            self.assertEqual(loaded.config.architecture, 'column-sigmoid-v1')
            for name, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)

    def test_architecture_settings_and_unsupported_capture_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'architecture'):
            neural.Config(architecture='unknown')
        with self.assertRaisesRegex(ValueError, 'centered_critic'):
            neural.create_model(replace(self.config, centered_critic=True))
        with self.assertRaisesRegex(NotImplementedError, 'legacy CNN'):
            self.model(self.batch, capture=True)
        diagnostics = self.model.ranking_diagnostics(self.batch)
        self.assertEqual(diagnostics['h2_activation'], 'silu')
        self.assertIsNone(diagnostics['h2_saturated_fraction'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
