"""Contracts for reconstructed Spatial V4 inference and PPO integration."""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

import tetris_neural as neural
from tetris_survival_v3 import SurvivalFeaturePolicy
from tetris_survival_v4 import (ARCHITECTURE, SurvivalSpatialPolicy,
                                adjacency_statistics, afterstate_descriptors)


class SpatialV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **kwargs):
        return replace(neural.Config(architecture=ARCHITECTURE, spatial_actor=True,
                                    recover_target=True, policy_mode='teacher-prior',
                                    teacher_prior_scale=.125, candidates=64, shortlist=16),
                       **kwargs)

    def test_empty_adjacency_and_exact_mask_bitcounts(self):
        empty = torch.zeros((1, 4096), dtype=torch.int32)
        targets = torch.tensor([1], dtype=torch.int64)
        values = adjacency_statistics(empty, targets)
        self.assertEqual(tuple(values.shape), (1, 3, 6))
        self.assertTrue(torch.equal(values, torch.zeros_like(values)))
        empty[0, 0] = 65535
        values = adjacency_statistics(empty, targets)
        self.assertAlmostEqual(float(values[0, 0, 3]), 1 / 3840, places=7)
        self.assertAlmostEqual(float(values[0, 0, 5]), 1 / 3840, places=7)
        self.assertEqual(float(values[0, 0, 4]), 0.0)
        self.assertGreater(float(values[0, 0, 0]), 0.)

    def test_actor_critic_width_and_zero_initialized_extension(self):
        model = SurvivalSpatialPolicy(self.config())
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 77870)
        self.assertEqual(model.value[0].normalized_shape, (156,))
        self.assertTrue(torch.equal(model.spatial_actor.weight,
                                    torch.zeros_like(model.spatial_actor.weight)))
        self.assertEqual(tuple(model.actor_state[0].normalized_shape), (150,))

    def test_actual_afterstate_matches_full_recomputation(self):
        state = neural.game.Game(710042)
        config = self.config()
        record = neural.prepare(state, 1, config)
        self.assertIsNotNone(record)
        self.assertEqual(record.arrays['spatial_delta'].shape[1], 18)
        self.assertEqual(record.arrays['spatial_board'].shape, (6,))
        option = record.options[0]
        moved = neural.game.materialize_move(state.board, option)
        actual = adjacency_statistics(torch.as_tensor(np.asarray([moved.board], dtype=np.int32)),
                                      torch.as_tensor([int(record.arrays['targets'])]))
        initial = adjacency_statistics(torch.as_tensor(np.asarray([state.board], dtype=np.int32)),
                                       torch.as_tensor([int(record.arrays['targets'])]))
        delta = (actual - initial).reshape(18).numpy()
        np.testing.assert_allclose(record.arrays['spatial_delta'][0], delta, atol=1e-7)
        np.testing.assert_allclose(record.arrays['spatial_board'], initial.mean(1)[0].numpy(), atol=1e-7)

    def test_teacher_and_actor_ranking_preserved_at_initialization(self):
        state = neural.game.Game(710043)
        config = self.config()
        record = neural.prepare(state, 1, config)
        batch = neural.collate([record.arrays], torch.device('cpu'))
        v4 = SurvivalSpatialPolicy(config)
        v3 = SurvivalFeaturePolicy(replace(config, architecture='survival-feature-v3'))
        with torch.no_grad():
            for key, value in v3.state_dict().items():
                if not key.startswith('value.'):
                    v4.state_dict()[key].copy_(value)
        result4 = v4(batch)
        result3 = v3(batch)
        torch.testing.assert_close(result4['logits'], result3['logits'], atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(result4['critic_input'].shape), (1, 156))
        self.assertTrue(torch.isfinite(result4['value']).all())

    def test_padded_batch_has_finite_valid_scores(self):
        state = neural.game.Game(710044)
        record = neural.prepare(state, 1, self.config())
        shortened = {}
        candidate_keys = {'features', 'indices', 'column_mask', 'valid', 'teacher_prior', 'spatial_delta'}
        for name, value in record.arrays.items():
            shortened[name] = value[:2] if name in candidate_keys else value
        batch = neural.collate([record.arrays, shortened], torch.device('cpu'))
        result = SurvivalSpatialPolicy(self.config())(batch)
        self.assertEqual(result['logits'].shape, batch['valid'].shape)
        self.assertTrue(torch.isfinite(result['logits'][batch['valid']]).all())
        self.assertTrue(torch.isneginf(result['logits'][~batch['valid']]).all())

    def test_critic_gradient_does_not_update_actor(self):
        state = neural.game.Game(710045)
        record = neural.prepare(state, 1, self.config())
        batch = neural.collate([record.arrays], torch.device('cpu'))
        model = SurvivalSpatialPolicy(self.config())
        model.zero_grad(set_to_none=True)
        model(batch)['value'].sum().backward()
        self.assertIsNone(model.spatial_actor.weight.grad)
        self.assertIsNone(model.rank_out.weight.grad)
        self.assertIsNotNone(model.value[1].weight.grad)
        model.zero_grad(set_to_none=True)
        model(batch)['logits'][batch['valid']].sum().backward()
        self.assertIsNone(model.value[1].weight.grad)
        self.assertIsNotNone(model.rank_out.weight.grad)

    def test_deterministic_diverse_shortlist_keeps_teacher_candidate(self):
        state = neural.game.Game(710046)
        first = neural.prepare(state, 1, self.config(shortlist_mode='diverse'))
        second = neural.prepare(state, 1, self.config(shortlist_mode='diverse'))
        self.assertIsNotNone(first)
        self.assertEqual(first.options, second.options)
        self.assertEqual(first.teacher, second.teacher)
        self.assertTrue(np.array_equal(first.arrays['spatial_delta'], second.arrays['spatial_delta']))


if __name__ == '__main__':
    unittest.main()
