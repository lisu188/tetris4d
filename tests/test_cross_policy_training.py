"""Checkpoint grafts, behavior distributions and the frozen Spatial V4 residual."""
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import unittest
from unittest.mock import patch

import torch

from test_v4_improvements import TrainerHarness, v4


class CrossTrainingTests(TrainerHarness):
    def setUp(self):
        super().setUp()
        self.quiet = redirect_stdout(StringIO())
        self.quiet.__enter__()
        self.addCleanup(self.quiet.__exit__, None, None, None)
        self.inc = self.trainer('--regional-critic').model
        self.source = self.directory / 'incumbent.pt'
        self.donor = self.directory / 'donor.pt'
        v4.neural.save_checkpoint(self.source, self.inc, extra={'calibration': {'stage': 'passed', 'done': True}})
        model = v4.neural.create_model(self.inc.config)
        with torch.no_grad():
            model.value[-1].weight.fill_(.125)
            model.value[-1].bias.fill_(.25)
        self.donor_digest = v4.v3.model_digest(model.value)
        v4.neural.save_checkpoint(self.donor, model)
        self.options = ('--regional-critic', '--pretrained', str(self.source), '--preserve-pretrained-critic',
                        '--critic-checkpoint', str(self.donor), '--behavior-temperature', '.25', '--fixed-lr')

    def test_critic_only_graft_temperature_reference_and_exact_resume(self):
        opts = self.options + ('--reference-checkpoint', str(self.source), '--reference-kl-weight', '.05',
                               '--compact-rollout', '--actor-only-kl')
        trainer = self.trainer(*opts)
        self.assertFalse(trainer.optimizer.state)
        self.assertEqual(v4.v2.actor_digest(trainer.model), v4.v2.actor_digest(self.inc))
        self.assertEqual(v4.v3.model_digest(trainer.model.value), self.donor_digest)
        self.assertTrue(all(not p.requires_grad for p in trainer.reference_policy.parameters()))
        self.assertTrue(trainer.calibrate())
        rollout = trainer.rollout(False)
        batch, old = trainer.policy_batch(rollout, [0, 1])
        logits = trainer.model.actor_logits(batch)
        expected = (logits.float() / .25).log_softmax(-1)
        torch.testing.assert_close(trainer.reference_logp(batch), expected, rtol=0, atol=0)
        torch.testing.assert_close(old, expected, rtol=0, atol=0)
        with patch.object(trainer, 'reference_logp', wraps=trainer.reference_logp) as reference:
            trainer.update(rollout, False)
            trainer.reference_rollout_kl(rollout)
        self.assertEqual(sum(call.args[0]['valid'].shape[0] for call in reference.call_args_list), len(rollout.choices))
        self.assertTrue(torch.equal(rollout.reference_logp[:2, :old.shape[1]], old))
        trainer.save('test_reference_resume')
        reference_digest = v4.v2.actor_digest(trainer.reference_policy)
        self.assertEqual(reference_digest, v4.v2.actor_digest(self.inc))
        resumed = self.trainer(*opts)
        self.assertEqual(v4.v2.actor_digest(resumed.reference_policy), reference_digest)
        self.assertEqual(v4.v3.model_digest(resumed.model), v4.v3.model_digest(trainer.model))
        update = [r for line in trainer.metrics.read_text().splitlines() if (r := json.loads(line))['event'] == 'update'][-1]
        self.assertGreater(update['final_reference_kl'], 0.)
        self.assertEqual(update['reference_kl_weight'], .05)
        with self.assertRaisesRegex(ValueError, 'settings or source changed'):
            self.trainer(*opts, '--behavior-temperature', '1')

    def test_reference_cache_variable_width_masks_and_memory_fallback(self):
        trainer = self.trainer(*self.options, '--reference-checkpoint', str(self.source), '--reference-kl-weight', '.05')
        self.assertTrue(trainer.calibrate())
        rollout = trainer.rollout(False)
        for i, row in enumerate(rollout.records):
            length = 2 + i
            for key in row.keys() & v4.neural.CANDIDATE_FIELDS:
                row[key] = row[key][:length]
            row['valid'][-1] = False
            rollout.old_logp[i] = rollout.old_logp[i][:length]
        indices = [4, 0, 3]
        batch, _ = trainer.policy_batch(rollout, indices)
        expected = trainer.reference_logp(batch)
        with patch.object(trainer.args, 'rollout_cache_mib', 0):
            self.assertEqual(trainer.cache_reference_distribution(rollout, 0)['reason'], 'memory_bound')
            self.assertIsNone(rollout.reference_logp)
            torch.testing.assert_close(trainer.reference_minibatch(rollout, batch, indices), expected)
        self.assertTrue(trainer.cache_reference_distribution(rollout, 0)['enabled'])
        cached = trainer.reference_minibatch(rollout, batch, indices)
        self.assertTrue(torch.isneginf(cached[~batch['valid']]).all())
        torch.testing.assert_close(cached, expected, atol=2e-6, rtol=1e-6)

    def test_residual_identity_learned_bound_and_frozen_base(self):
        opts = self.options + ('--cross-residual', '--compact-rollout', '--actor-only-kl')
        trainer = self.trainer(*opts)
        model = trainer.model
        self.assertTrue(trainer.calibrate())
        self.assertTrue(all(p.requires_grad == n.startswith(('cross_head.', 'value.')) for n, p in model.named_parameters()))
        self.assertEqual(v4.v2.actor_digest(model, base_only=True), v4.v2.actor_digest(self.inc))
        rollout = trainer.rollout(False)
        fixed, _ = trainer.policy_batch(rollout, [0, 1])
        raw = v4.neural.collate([v4.neural.prepare(e.state, e.target, model.config).arrays
                                for e in trainer.environments], trainer.device)
        for batch in (raw, fixed):
            base = self.inc.actor_logits(batch)
            self.assertTrue(torch.equal(model.actor_logits(batch), base))
            self.assertTrue(torch.equal(model(batch)['logits'], base))
            current, frozen = model.actor_logits(batch, return_base=True)
            self.assertTrue(torch.equal(current, frozen) and torch.equal(frozen, base))
        trainer.update(rollout, False)
        self.assertEqual(v4.v2.actor_digest(model, base_only=True), v4.v2.actor_digest(self.inc))
        with torch.no_grad():
            self.assertGreater(float(model.cross_head[-1].weight.abs().sum()), 0.)
            model.cross_head[-1].weight.mul_(1.e5)
            valid = fixed['valid']
            adjusted, frozen = model.actor_logits(fixed, return_base=True)
            self.assertTrue(torch.equal(frozen, self.inc.actor_logits(fixed)))
            delta = adjusted[valid] - frozen[valid]
            self.assertLessEqual(float(delta.abs().max()), .05001)
        trainer.save('test_residual_resume')
        resumed = self.trainer(*opts)
        self.assertEqual(v4.v3.model_digest(resumed.model), v4.v3.model_digest(model))
        self.assertTrue(all(p.requires_grad == n.startswith(('cross_head.', 'value.'))
                            for n, p in resumed.model.named_parameters()))

    def test_direct_weight_reorders_only_by_mobility_and_preserves_legacy_load(self):
        trainer = self.trainer(*self.options, '--cross-weight', '4')
        model = trainer.model
        self.assertFalse(any(p.requires_grad for n, p in model.named_parameters() if not n.startswith('value.')))
        batch = v4.neural.collate([v4.neural.prepare(e.state, e.target, model.config).arrays
                                  for e in trainer.environments], trainer.device)
        # Supply contrasting values on a full shortlist to also test near ties and padding.
        batch['cross_mobility'][..., 0] = .5
        batch['cross_mobility'][..., 1] = torch.linspace(0., 1., batch['valid'].shape[1])
        expected = self.inc.actor_logits(batch) + 4 * (batch['cross_mobility'][..., 1] - .5)
        adjusted, base = model.actor_logits(batch, return_base=True)
        torch.testing.assert_close(adjusted, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(base, self.inc.actor_logits(batch)))
        payload = torch.load(self.source, weights_only=True)
        for key in ('cross_weight', 'cross_residual'):
            payload['config'].pop(key)
        old = self.directory / 'legacy-v4.pt'
        torch.save(payload, old)
        loaded, _ = v4.neural.load_checkpoint(old, torch.device('cpu'))
        self.assertTrue(torch.equal(loaded.actor_logits(batch), self.inc.actor_logits(batch)))
        with self.assertRaisesRegex(ValueError, 'CROSS weight'):
            replace(model.config, cross_residual=True)


if __name__ == '__main__':
    unittest.main()
