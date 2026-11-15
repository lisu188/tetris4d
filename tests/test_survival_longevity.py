from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_survival_v2 as support

v2, neural, game = support.v2, support.neural, support.game


class LongevityTests(unittest.TestCase):
    # Share fixtures and helpers without inheriting and rerunning the v2 suite.
    setUpClass = classmethod(support.RuntimeTests.setUpClass.__func__)
    setUp = support.RuntimeTests.setUp
    tearDown = support.RuntimeTests.tearDown
    args = support.TrainerTests.args
    trainer = support.TrainerTests.trainer
    run_training = support.TrainerTests.run_training

    def batch(self, model):
        prepared = neural.prepare(game.Game(5), 1, model.config)
        return neural.collate([prepared.arrays], torch.device('cpu'))

    def test_new_training_flags_and_prior_scale_bounds(self):
        defaults = v2.arguments(['--device', 'cpu'])
        self.assertEqual(defaults.teacher_prior_scale, 1.)
        self.assertFalse(defaults.preserve_initial_actor)
        self.assertEqual(defaults.gradient_clip_scope, 'joint')
        args = v2.arguments(['--device', 'cpu', '--policy-mode', 'teacher-prior',
                             '--teacher-prior-scale', '.5', '--preserve-initial-actor',
                             '--gradient-clip-scope', 'separate'])
        self.assertEqual(args.policy_mode, 'teacher-prior')
        self.assertEqual(args.teacher_prior_scale, .5)
        self.assertTrue(args.preserve_initial_actor)
        self.assertEqual(args.gradient_clip_scope, 'separate')
        for scale in (0., 4.):
            neural.Config(policy_mode='teacher-prior', teacher_prior_scale=scale)
        for scale in (-.1, 4.1, float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                neural.Config(policy_mode='teacher-prior', teacher_prior_scale=scale)

    def test_existing_policy_modes_ignore_prior_scale(self):
        for mode in ('neural', 'teacher-residual'):
            with self.subTest(mode=mode):
                model = neural.ColumnPolicy(replace(self.config, policy_mode=mode, teacher_prior_scale=0.))
                batch = self.batch(model)
                with torch.no_grad():
                    before = model(batch)['logits']
                    model.config = replace(model.config, teacher_prior_scale=4.)
                    after = model(batch)['logits']
                self.assertTrue(torch.equal(before, after))

    def test_legacy_checkpoint_without_prior_scale_loads_identically(self):
        positional = neural.Config(8, 1, 8, 32, 256, True, False, 'teacher-residual', 1., True, True)
        self.assertTrue(positional.detach_critic)
        self.assertTrue(positional.centered_critic)
        self.assertEqual(positional.teacher_prior_scale, 1.)
        saved = torch.load(self.path, map_location='cpu', weights_only=True)
        saved['config'].pop('teacher_prior_scale', None)
        legacy_path = self.root / 'legacy.pt'
        torch.save(saved, legacy_path)
        loaded, _ = neural.load_checkpoint(legacy_path, torch.device('cpu'))
        self.assertEqual(loaded.config.teacher_prior_scale, 1.)
        self.assertEqual(set(loaded.state_dict()), set(self.model.state_dict()))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)
        batch = self.batch(loaded)
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(batch)['logits'], loaded(batch)['logits']))
        self.assertEqual(sum(p.numel() for p in neural.ColumnPolicy().parameters()), 62485)

    def test_teacher_prior_policy_can_override_a_primary_teacher_preference(self):
        model = neural.ColumnPolicy(replace(self.config, policy_mode='teacher-prior',
                                            teacher_prior_scale=1., residual_scale=1.))
        batch = self.batch(model)
        for key in ('features', 'indices', 'column_mask', 'valid', 'teacher_prior'):
            batch[key] = batch[key][:, :2].clone()
        self.assertEqual(batch['features'].shape[1], 2)
        batch['features'].zero_()
        batch['features'][0, :, 0] = torch.tensor([0., 1.])
        batch['features'][0, :, 8] = torch.tensor([4., 3.]) / 16
        batch['teacher_prior'][0] = torch.tensor([0., -32.])
        with torch.no_grad():
            for layer in (model.rank1, model.rank2, model.rank_out):
                layer.weight.zero_()
                layer.bias.zero_()
            model.rank1.weight[0, -neural.FEATURES] = 20.
            model.rank2.weight[0, 0] = 20.
            model.rank2.bias[0] = -10.
            model.rank_out.weight[0, 0] = 100.
            new_logits = model(batch)['logits']
            model.config = replace(model.config, policy_mode='teacher-residual')
            bounded_logits = model(batch)['logits']
        winners, _ = v2.legacy.utility_targets(batch['features'], batch['valid'])
        self.assertEqual(winners.tolist(), [[True, False]])
        self.assertEqual(int(new_logits.argmax()), 1)
        self.assertEqual(int(bounded_logits.argmax()), 0)

    def test_imported_actor_preservation_is_opt_in_for_bounded_residual(self):
        normal = self.trainer(self.args('normal', preserve_initial_actor=False))
        self.assertEqual(int(torch.count_nonzero(normal.model.rank_out.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(normal.model.rank_out.bias)), 0)
        for name, mode, preserve in (('preserved', 'teacher-residual', True),
                                     ('unbounded', 'teacher-prior', False)):
            with self.subTest(mode=mode, preserve=preserve):
                trainer = self.trainer(self.args(name, policy_mode=mode, preserve_initial_actor=preserve))
                self.assertEqual(v2.actor_digest(self.model), v2.actor_digest(trainer.model))

    def clipped_step(self, scope, critic_scale):
        actor = torch.nn.Parameter(torch.tensor([1., 2.]))
        critic = torch.nn.Parameter(torch.tensor([3., 4.]))
        optimizer = torch.optim.AdamW([{'params': [actor], 'name': 'actor'},
                                       {'params': [critic], 'name': 'critic'}], lr=.01, weight_decay=0.)
        actor.grad = torch.tensor([.3, .4])
        critic.grad = torch.tensor([5., -12.]) * critic_scale
        norms = v2.clip_gradient_groups(optimizer, scope, maximum=1.)
        actor_gradient, critic_gradient = actor.grad.clone(), critic.grad.clone()
        optimizer.step()
        return actor.detach(), actor_gradient, critic_gradient, optimizer.state[actor], norms

    def test_separate_clipping_makes_actor_update_independent_of_critic_scale(self):
        a = self.clipped_step('separate', 1.)
        b = self.clipped_step('separate', 100.)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))
        for key in a[3]:
            self.assertTrue(torch.equal(a[3][key], b[3][key]), key)
        self.assertAlmostEqual(a[4]['actor_gradient_norm'], .5, places=6)
        self.assertAlmostEqual(a[4]['critic_gradient_norm'], 13., places=5)
        self.assertAlmostEqual(a[4]['gradient_norm'], (13. ** 2 + .5 ** 2) ** .5, places=5)
        self.assertLessEqual(float(a[2].norm()), 1.000001)
        self.assertLessEqual(float(b[2].norm()), 1.000001)
        joint = self.clipped_step('joint', 1.)
        self.assertLess(float(joint[1].norm()), float(a[1].norm()))

    def test_joint_clipping_matches_legacy_global_clipping(self):
        _, actor_gradient, critic_gradient, _, norms = self.clipped_step('joint', 1.)
        actor = torch.nn.Parameter(torch.zeros(2))
        critic = torch.nn.Parameter(torch.zeros(2))
        actor.grad = torch.tensor([.3, .4])
        critic.grad = torch.tensor([5., -12.])
        legacy_norm = torch.nn.utils.clip_grad_norm_((actor, critic), 1., error_if_nonfinite=True)
        self.assertTrue(torch.equal(actor.grad, actor_gradient))
        self.assertTrue(torch.equal(critic.grad, critic_gradient))
        self.assertAlmostEqual(norms['gradient_norm'], float(legacy_norm), places=5)

    def test_survival_update_changes_actor_and_policy_probabilities(self):
        trainer = self.trainer(self.args(policy_mode='teacher-prior', teacher_prior_scale=0.,
                                         preserve_initial_actor=True, gradient_clip_scope='separate', lr=.003,
                                         imitation=0., regret_weight=0., secondary_weight=0., entropy=0.))
        rollout = trainer.rollout(False)
        batch = neural.collate(rollout.records, torch.device('cpu'))
        with torch.no_grad():
            before = v2.legacy.categorical_logp(trainer.model(batch)['logits'], batch['valid'])
        digest = v2.actor_digest(trainer.model)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.update(rollout, False)
        with torch.no_grad():
            after = v2.legacy.categorical_logp(trainer.model(batch)['logits'], batch['valid'])
        self.assertNotEqual(digest, v2.actor_digest(trainer.model))
        self.assertTrue(torch.isfinite(after[batch['valid']]).all())
        self.assertGreater(float((after[batch['valid']] - before[batch['valid']]).abs().max()), 1e-7)
        events = [json.loads(line) for line in trainer.metrics.read_text().splitlines()]
        update = [row for row in events if row['event'] == 'update'][-1]
        self.assertGreater(update['actor_gradient_norm'], 0.)
        self.assertGreater(update['critic_gradient_norm'], 0.)
        self.assertEqual(update['optimizer_steps_this_rollout'], 6)
        diagnostics = update['actor_diagnostics']
        self.assertGreater(diagnostics['states'], 0)
        self.assertLessEqual(diagnostics['states'], 16)
        for key in ('h2_saturated_fraction', 'lower_primary_probability', 'sampled_lower_primary_fraction'):
            self.assertGreaterEqual(diagnostics[key], 0., key)
            self.assertLessEqual(diagnostics[key], 1.000001, key)
        for key in ('h2_candidate_std', 'raw_logit_spread', 'raw_logit_std',
                    'expected_primary_deficit', 'sampled_primary_deficit'):
            self.assertGreaterEqual(diagnostics[key], 0., key)
        self.assertEqual(diagnostics['actor_sha256'], v2.actor_digest(trainer.model))
        self.assertEqual(diagnostics['initial_actor_sha256'], digest)
        self.assertTrue(diagnostics['actor_changed_from_initial'])

    def test_actor_diagnostics_ignore_padded_candidates(self):
        trainer = self.trainer(self.args(policy_mode='teacher-prior', teacher_prior_scale=.5,
                                         preserve_initial_actor=True, gradient_clip_scope='separate'))
        prepared = neural.prepare(game.Game(5), 1, trainer.model.config)
        records = [{key: value.copy() for key, value in prepared.arrays.items()} for _ in range(2)]
        self.assertGreater(len(records[1]['valid']), 2)
        records[1]['valid'][2:] = False
        rollout = SimpleNamespace(records=records, choices=[0, 0])
        before = trainer.actor_diagnostics(rollout)
        records[1]['features'][2:] = 10000.
        records[1]['teacher_prior'][2:] = 10000.
        after = trainer.actor_diagnostics(rollout)
        for key, value in before.items():
            if isinstance(value, float):
                self.assertAlmostEqual(value, after[key], places=7, msg=key)
            else:
                self.assertEqual(value, after[key], key)

    def test_new_settings_resume_exactly_on_cpu(self):
        settings = dict(policy_mode='teacher-prior', teacher_prior_scale=.5,
                        preserve_initial_actor=True, gradient_clip_scope='separate')
        full = self.run_training(self.args('full', 2, **settings))
        self.run_training(self.args('split', 1, **settings))
        split = self.run_training(self.args('split', 1, **settings))
        a = torch.load(full.last, map_location='cpu', weights_only=True)
        b = torch.load(split.last, map_location='cpu', weights_only=True)
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['extra']['environments'], b['extra']['environments'])
        self.assertEqual(a['extra']['initial_actor_sha256'], v2.actor_digest(self.model))
        self.assertEqual(a['extra']['initial_actor_sha256'], b['extra']['initial_actor_sha256'])
        self.assertEqual(full.initial_actor_sha256, split.initial_actor_sha256)
        self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
        for key in a['model']:
            self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
        for index, state in a['optimizer']['state'].items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, b['optimizer']['state'][index][key]), (index, key))

    def test_changed_new_training_settings_reject_resume(self):
        settings = dict(policy_mode='teacher-prior', teacher_prior_scale=.5,
                        preserve_initial_actor=True, gradient_clip_scope='separate')
        self.run_training(self.args(**settings))
        for changed in ({'teacher_prior_scale': 1.}, {'preserve_initial_actor': False},
                        {'gradient_clip_scope': 'joint'}, {'policy_mode': 'teacher-residual'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.trainer(self.args(**(settings | changed)))


if __name__ == '__main__':
    unittest.main()
