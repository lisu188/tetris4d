"""Spatial V4 fixed features, collection and PPO parity."""
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_survival_v4 as v4


class FixedRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        v4.game.configure_backend('numpy')
        v4.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def compare_updates(self, device, force_fallback=False, early_stop=False):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(StringIO()), patch.dict(
                os.environ, TETRIS4D_GAME_LIBRARY='off'):
            results = []
            for mode in ('raw', 'fixed'):
                args = v4.arguments([
                    '--no-gpu-defaults', '--device', device, '--backend', 'numpy', '--threads', '1',
                    '--run-dir', str(Path(root) / mode), '--envs', '2', '--rollout', '2', '--batch', '2',
                    '--optimizer-batch', '2', '--ppo-epochs', '2', '--fixed-lr', '--regional-critic',
                    '--calibration-games', '0', '--allow-uncalibrated', '--pieces', '2', '--seed', '775001',
                    '--target-kl', '1e-10' if early_stop else '.5'])
                args.compact_rollout = args.actor_only_kl = mode == 'fixed'
                if force_fallback:
                    args.rollout_cache_mib = 0
                trainer = v4.SurvivalTrainer(args, v4.base.Budget(.1))
                self.assertTrue(trainer.calibrate())
                rollout = trainer.rollout(False)
                old = trainer.policy_batch(rollout, list(range(4)))[1]
                if mode == 'fixed' and not force_fallback:
                    self.assertEqual(rollout.records, [])
                    self.assertEqual(rollout.old_logp, [])
                    self.assertNotIn('boards', rollout.fixed_cache.inputs)
                if mode == 'fixed' and force_fallback:
                    self.assertIsNone(rollout.fixed_cache)
                    self.assertEqual(rollout.fixed_fallback, 'memory_bound')
                events = []
                trainer.emit = lambda event, **fields: events.append(fields)
                trainer.update(rollout, False)
                results.append((trainer, rollout, old, events[-1], torch.get_rng_state(),
                                torch.cuda.get_rng_state() if device == 'cuda' else None))
            raw, compact = results
            self.assertEqual(raw[1].choices, compact[1].choices)
            for field in ('advantages', 'returns', 'old_values', 'raw_advantages', 'terminals', 'truncations'):
                np.testing.assert_array_equal(getattr(raw[1], field), getattr(compact[1], field))
            torch.testing.assert_close(raw[2], compact[2], rtol=0, atol=0)
            for key in ('final_exact_kl', 'kl_early_stop', 'optimizer_steps_this_rollout', 'actor_diagnostics'):
                self.assertEqual(raw[3][key], compact[3][key])
            self.assertEqual(early_stop, raw[3]['kl_early_stop'])
            self.assertTrue(torch.equal(raw[4], compact[4]))
            if device == 'cuda':
                self.assertTrue(torch.equal(raw[5], compact[5]))
            for (n, a), (_, b) in zip(raw[0].model.named_parameters(), compact[0].model.named_parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0, msg=n)
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=n)
                for key in raw[0].optimizer.state[a]:
                    torch.testing.assert_close(raw[0].optimizer.state[a][key], compact[0].optimizer.state[b][key],
                                               rtol=0, atol=0, msg=f'{n}/{key}')
            if not force_fallback:
                compact[0].release_previous_device_rollout()
                self.assertIsNone(compact[1].fixed_cache)

    def test_actions_gradients_adam_kl_and_rng(self):
        self.compare_updates('cpu')

    def test_compact_cache_preserves_kl_early_stop(self):
        self.compare_updates('cpu', early_stop=True)

    def test_allocation_guard_falls_back_before_sampling_or_optimization(self):
        self.compare_updates('cpu', force_fallback=True, early_stop=True)

    @unittest.skipUnless(os.environ.get('RUN_TETRIS_CUDA_CACHE_TESTS') == '1' and torch.cuda.is_available(),
                         'CUDA parity requires an explicitly isolated GPU window')
    def test_cuda_actions_gradients_adam_kl_and_rng(self):
        self.compare_updates('cuda')

    def test_mixed_lengths_padding_and_invalid_column_detection(self):
        model = v4.neural.create_model(v4.neural.Config(architecture=v4.ARCHITECTURE, regional_critic=True,
                                                         spatial_actor=True, shortlist=8, candidates=40, chunk=3,
                                                         policy_mode='teacher-prior'))
        records = [v4.neural.prepare(v4.game.Game(seed), 1, model.config).arrays for seed in (101, 202)]
        records[0] = {k: a[:2] if k in v4.neural.CANDIDATE_FIELDS else a for k, a in records[0].items()}
        batch = v4.neural.collate(records, torch.device('cpu'))
        fixed = model.fixed_inputs(batch)
        old = model(batch)['logits'].log_softmax(1).detach()
        cache = v4.v2.FixedRolloutCache(fixed, 5, 8, 10**6)
        cache.append(fixed, old, torch.zeros(2, dtype=torch.int64), [2, 8])
        for indices in ([0], [1], [1, 0]):
            raw = v4.neural.collate([records[i] for i in indices], torch.device('cpu'))
            compact, values = cache.select(indices)
            for name in ('logits', 'value', 'critic_input'):
                torch.testing.assert_close(model(raw)[name], model(compact)[name], rtol=0, atol=0)
            torch.testing.assert_close(model.actor_logits(compact), model(raw)['logits'], rtol=0, atol=0)
            self.assertTrue(torch.isneginf(values['old'][~compact['valid']]).all())
        trainer = v4.SurvivalTrainer.__new__(v4.SurvivalTrainer)
        trainer.model, trainer.device = model, torch.device('cpu')
        trainer.args = SimpleNamespace(envs=2, rollout=3, rollout_cache_mib=2)
        before = torch.get_rng_state()
        with patch.object(v4.v2, 'FixedRolloutCache', side_effect=torch.OutOfMemoryError()):
            cache, original, reason = trainer.start_fixed_cache(batch)
        self.assertIsNone(cache)
        self.assertIs(original, batch)
        self.assertEqual(reason, 'allocation_failed')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        batch['column_mask'][0, 0] = False
        with self.assertRaisesRegex(ValueError, 'valid candidate'):
            model.fixed_inputs(batch)

if __name__ == '__main__':
    unittest.main()
