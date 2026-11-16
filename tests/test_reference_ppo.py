"""Reference regularization and frozen-policy remaining-life diagnostics."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_survival_v2 as existing
from tools import survival_experiments as experiments

v2, neural, game, base = existing.v2, existing.neural, existing.game, existing.base


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing.RuntimeTests.setUpClass.__func__(cls)

    setUp = existing.RuntimeTests.setUp
    tearDown = existing.RuntimeTests.tearDown
    trainer = existing.TrainerTests.trainer
    run_training = existing.TrainerTests.run_training

    def args(self, name='run', count=1, **changes):
        options = dict(policy_mode='neural', recover_target=False, preserve_initial_actor=True,
                       reference_kl_weight=1., reference_checkpoint=self.path)
        options.update(changes)
        return existing.TrainerTests.args(self, name, count, **options)

    def test_masked_reference_kl_direction_and_gradient(self):
        valid = torch.tensor([[True, True, False], [True, False, False]])
        reference = torch.tensor([[.9, .1, 0.], [1., 0., 0.]])
        current = torch.tensor([[0., 2., 1000.], [3., -1000., 1000.]], requires_grad=True)
        logp = v2.legacy.categorical_logp(current, valid)
        reference_logp = torch.where(valid, reference.clamp_min(1e-30).log(), -torch.inf)
        kl = v2.legacy.exact_kl(reference_logp, logp, valid).sum()
        expected = (.9 * (.9 / logp.exp()[0, 0]).log() + .1 * (.1 / logp.exp()[0, 1]).log())
        torch.testing.assert_close(kl, expected)
        kl.backward()
        torch.testing.assert_close(current.grad, logp.detach().exp() - reference)
        after = v2.legacy.categorical_logp(current.detach() - .2 * current.grad, valid)
        self.assertLess(float(v2.legacy.exact_kl(reference_logp, after, valid).sum()), float(kl.detach()))

    def test_reference_is_frozen_excluded_from_optimizer_and_persisted(self):
        trainer = self.run_training(self.args())
        optimizer_ids = {id(parameter) for group in trainer.optimizer.param_groups for parameter in group['params']}
        for parameter in trainer.reference_policy.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
            self.assertNotIn(id(parameter), optimizer_ids)
        self.assertEqual(v2.actor_digest(self.model), v2.actor_digest(trainer.reference_policy))
        self.assertNotEqual(v2.actor_digest(trainer.model), v2.actor_digest(trainer.reference_policy))
        saved = torch.load(trainer.last, weights_only=True)
        self.assertEqual(saved['extra']['frozen_reference']['actor_sha256'], v2.actor_digest(self.model))
        events = [json.loads(line) for line in trainer.metrics.read_text().splitlines()]
        update = next(row for row in events if row['event'] == 'update')
        self.assertGreaterEqual(update['reference_kl'], 0.)
        self.assertGreaterEqual(update['final_reference_kl'], 0.)
        self.assertEqual(update['reference_kl_weight'], 1.)
        self.assertIn('reference_kl', update['per_loss_gradient_norms'])

    def test_reference_construction_preserves_rng_and_actor(self):
        control = self.trainer(self.args('control', reference_kl_weight=0., reference_checkpoint=None))
        cpu_rng = torch.get_rng_state().clone()
        import random
        python_rng = random.getstate()
        experimental = self.trainer(self.args('reference'))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        for name, value in control.model.state_dict().items():
            self.assertTrue(torch.equal(value, experimental.model.state_dict()[name]), name)
        self.assertNotIn('reference_kl_weight', control.signature)
        self.assertNotIn('frozen_reference', control.checkpoint_metadata())

    def test_reference_resume_is_exact(self):
        full = self.run_training(self.args('full', 2))
        self.run_training(self.args('split', 1))
        resumed = self.run_training(self.args('split', 1))
        a, b = torch.load(full.last, weights_only=True), torch.load(resumed.last, weights_only=True)
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['extra']['environments'], b['extra']['environments'])
        self.assertTrue(torch.equal(a['rng_cpu'], b['rng_cpu']))
        for name in a['model']:
            self.assertTrue(torch.equal(a['model'][name], b['model'][name]), name)
            self.assertTrue(torch.equal(a['extra']['frozen_reference']['model'][name],
                                        b['extra']['frozen_reference']['model'][name]), name)
        for index, state in a['optimizer']['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][index][key]) if torch.is_tensor(value)
                                else value == b['optimizer']['state'][index][key])

    def test_fixed_coefficient_changes_actual_ppo_update(self):
        weak = self.run_training(self.args('weak', reference_kl_weight=.1))
        strong = self.run_training(self.args('strong', reference_kl_weight=10.))
        self.assertEqual(weak.progress, strong.progress)
        self.assertTrue(any(not torch.equal(value, strong.model.state_dict()[name])
                            for name, value in weak.model.state_dict().items() if not name.startswith('value.')))
        self.assertEqual(v2.actor_digest(weak.reference_policy), v2.actor_digest(strong.reference_policy))

    def test_reference_configuration_and_changed_source_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'same actor and candidate'):
            self.trainer(self.args('wrong-config', recover_target=True))
        self.run_training(self.args('run'))
        self.model.rank_out.bias.data.add_(1.)
        neural.save_checkpoint(self.path, self.model, progress={'updates': 10})
        with self.assertRaisesRegex(ValueError, 'settings or source changed'):
            self.trainer(self.args('run'))

    def test_corrupted_embedded_reference_is_rejected(self):
        trainer = self.run_training(self.args())
        saved = torch.load(trainer.last, weights_only=True)
        saved['extra']['frozen_reference']['model']['rank_out.bias'].add_(1.)
        torch.save(saved, trainer.last)
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.trainer(self.args())

    def test_reference_options_require_explicit_positive_pair(self):
        for options in (['--reference-kl-weight', 'nan'], ['--reference-kl-weight', '-1'],
                        ['--reference-kl-weight', '1'], ['--reference-checkpoint', str(self.path)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                v2.arguments(options)

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1' and torch.cuda.is_available(),
                         'CUDA smoke requires an explicitly coordinated bounded slot')
    def test_cuda_reference_ppo_smoke(self):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.25, 512 * 2**20 / total))
        old_backend = game.BACKEND
        try:
            game.configure_backend('native')
            trainer = self.run_training(self.args('cuda', device='cuda', backend='native',
                       cache_rollout=True, gpu_prepare=True, fused_encoding=True, coalesce_transfers=True))
            self.assertEqual(set(trainer.model.state_dict()), set(self.model.state_dict()))
            self.assertEqual(sum(p.numel() for p in trainer.model.parameters()),
                             sum(p.numel() for p in self.model.parameters()))
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in trainer.model.state_dict().values()))
            self.assertEqual(v2.actor_digest(trainer.reference_policy), v2.actor_digest(self.model))
            self.assertNotEqual(v2.actor_digest(trainer.model), v2.actor_digest(self.model))
            self.assertTrue(all(parameter.grad is None and not parameter.requires_grad
                                for parameter in trainer.reference_policy.parameters()))
            events = [json.loads(line) for line in trainer.metrics.read_text().splitlines()]
            update = next(row for row in events if row['event'] == 'update')
            self.assertTrue(update['gpu_prepare']['enabled'])
            self.assertGreater(update['gpu_prepare']['preparations']['batch_calls'], 0)
            self.assertTrue(torch.isfinite(torch.tensor(update['final_reference_kl'])))
            self.assertEqual(update['reference_kl_weight'], 1.)
            self.assertLess(torch.cuda.max_memory_allocated(), 512 * 2**20)
        finally:
            game.configure_backend(old_backend)


class ValueAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        existing.RuntimeTests.setUpClass.__func__(cls)

    setUp = existing.RuntimeTests.setUp
    tearDown = existing.RuntimeTests.tearDown

    def test_true_remaining_placements_and_censoring(self):
        samples = [{'piece': 0, 'predicted_value': .010}, {'piece': 8, 'predicted_value': .002}]
        audit = experiments.value_audit(samples, 10, True, 'sampled')
        self.assertEqual([sample['remaining_pieces'] for sample in audit['samples']], [10, 2])
        self.assertAlmostEqual(audit['diagnostics']['mae_pieces'], 0.)
        self.assertTrue(audit['on_policy_for_sampled_critic'])
        capped = experiments.value_audit(samples, 10, False, 'sampled')
        self.assertIsNone(capped['diagnostics'])
        self.assertTrue(all(sample['remaining_pieces'] is None for sample in capped['samples']))
        self.assertFalse(experiments.value_audit(samples, 10, True, 'greedy')['on_policy_for_sampled_critic'])

    def test_diagnostics_do_not_change_sampled_trajectory_and_remain_bounded(self):
        agent = neural.NeuralAgent(self.path, 'cpu', capture_traces=False)
        first = experiments.run_game(71, 5, 'expanded', agent, 'neural', 32, 'sampled', 13)
        audited = experiments.run_game(71, 5, 'expanded', agent, 'neural', 32, 'sampled', 13,
                                       value_audit_stride=1, value_audit_samples=2)
        for key in ('pieces', 'hyperlayers', 'stopped_reason', 'clear_steps', 'target_recoveries'):
            self.assertEqual(first[key], audited[key], key)
        self.assertEqual(len(audited['critic_monte_carlo']['samples']), 2)
        self.assertFalse(audited['critic_monte_carlo']['uncensored'])

    def test_actual_completed_game_labels_pre_move_state(self):
        agent = neural.NeuralAgent(self.path, 'cpu', capture_traces=False)
        original = neural.prepare
        with patch.object(neural, 'prepare', side_effect=lambda state, target, config:
                          original(state, target, config) if state.pieces < 3 else None):
            report = experiments.run_game(71, 9, 'expanded', agent, 'neural', 32, 'sampled', 13,
                                          value_audit_stride=1)
        audit = report['critic_monte_carlo']
        self.assertTrue(audit['uncensored'])
        self.assertEqual([sample['remaining_pieces'] for sample in audit['samples']], [3, 2, 1])
        self.assertEqual(audit['diagnostics']['samples'], 3)

    def test_audit_rejects_other_value_units(self):
        options = type('Options', (), dict(output=self.root / 'audit.json', checkpoints=[self.path],
                        threads=1, backend='numpy', weights=ROOT / 'models/trained-sigmoid.json',
                        device='cpu', seeds=[71], pieces=5, piece_set='expanded', modes=['sampled'],
                        action_seeds=[13], allow_concurrent=False, value_audit=True,
                        value_audit_stride=128, value_audit_samples=256))()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'survival checkpoints'):
            experiments.compare(options)

    def test_complete_comparison_keeps_modes_and_censored_samples_separate(self):
        neural.save_checkpoint(self.path, self.model, progress={'updates': 10},
                               extra={'value_semantics': 'expected_remaining_placements_times_0.001'})
        output = self.root / 'audit.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code = experiments.main(['compare', '--checkpoints', str(self.path), '--seeds', '71',
                   '--action-seeds', '13', '--modes', 'greedy', 'sampled', '--pieces', '3',
                   '--device', 'cpu', '--backend', 'numpy', '--threads', '1', '--output', str(output),
                   '--value-audit', '--value-audit-stride', '1', '--value-audit-samples', '2'])
        self.assertEqual(code, 0)
        report = json.loads(output.read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['value_audit']['max_samples_per_game'], 2)
        summaries = report['critic_monte_carlo_summary']
        self.assertEqual({key.rsplit(':', 1)[-1] for key in summaries}, {'greedy', 'sampled'})
        for value in summaries.values():
            self.assertEqual(value['censored_games'], 1)
            self.assertEqual(value['uncensored_games'], 0)
            self.assertIsNone(value['diagnostics'])
        for row in report['checkpoints']:
            self.assertEqual(base.file_hash(Path(row['snapshot'])), row['sha256'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
