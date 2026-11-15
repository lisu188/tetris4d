"""Opt-in transfer batching preserves sampled actions, PPO math and failure guards."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import train_survival_v2 as v2
import test_rollout_cache as fixtures


class GpuTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device='cpu', cached=True, enabled=False):
        trainer, rollout = fixtures.RolloutCacheTests().fixture(device, cached)
        trainer.args.coalesce_transfers = enabled
        return trainer, rollout

    def policy_checks(self, device='cpu'):
        old, _ = self.fixture(device, enabled=False)
        new, _ = self.fixture(device, enabled=True)
        valid = torch.tensor([[True, False, True], [True, True, False]], device=device)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            logits = torch.tensor([[1., float('nan'), -2.], [.25, 3., float('inf')]],
                                  dtype=dtype, device=device, requires_grad=True)
            expected = old.policy_logp(logits, valid)
            actual = new.policy_logp(logits, valid)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            a = torch.autograd.grad(expected[valid].sum(), logits, retain_graph=True)[0]
            b = torch.autograd.grad(actual[valid].sum(), logits)[0]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for logits, mask in ((torch.zeros(2, 3, device=device), torch.zeros_like(valid)),
                             (torch.full((2, 3), float('nan'), device=device), valid),
                             (torch.full((2, 3), float('inf'), device=device), valid)):
            for trainer in (old, new):
                with self.assertRaisesRegex(ValueError, 'Invalid candidate mask'):
                    trainer.policy_logp(logits, mask)

    def sampling_checks(self, device='cpu'):
        valid = torch.tensor([[True, False, True, True], [True, True, False, False]], device=device)
        logits = torch.tensor([[.2, -1., .7, .1], [.4, .3, 1., 2.]], device=device)
        logp = v2.legacy.categorical_logp(logits, valid)
        values = torch.tensor([.123, -12.75], device=device)
        for teachers in (None, [2, 1]):
            for dtype in (torch.float32, torch.float64):  # float64 deliberately uses fallback.
                torch.manual_seed(998)
                expected = v2.rollout_to_host(logp.to(dtype), values, teachers, False)
                rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
                torch.manual_seed(998)
                actual = v2.rollout_to_host(logp.to(dtype), values, teachers, True)
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                if cuda_rng is not None:
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state()))
                self.assertEqual(actual[2].device.type, 'cpu')
                if dtype == torch.float32:
                    self.assertTrue(actual[1].flags.owndata)

    def update_checks(self, device='cpu'):
        for cached, warmup in ((False, False), (True, False), (True, True)):
            old, rollout = self.fixture(device, cached, False)
            new, _ = self.fixture(device, cached, True)
            torch.manual_seed(777)
            old.update(rollout, warmup)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(777)
            new.update(rollout, warmup)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            fixtures.RolloutCacheTests().assert_state_equal(old, new)
            for left, right in zip(old.model.parameters(), new.model.parameters()):
                if left.grad is not None:
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            a, b = copy.deepcopy(old.events[-1][1]), copy.deepcopy(new.events[-1][1])
            for values in (a, b):
                values.pop('phase_seconds')
                values.pop('peak_allocated_mib', None)
                values.pop('peak_reserved_mib', None)
            self.assertEqual(a, b)
            self.assertIsNone(new._rollout_cache)

    def test_scalar_reads_preserve_mixed_dtypes_and_python_sum_order(self):
        values = [torch.tensor(1e16, dtype=torch.float64), torch.tensor(1., dtype=torch.float32),
                  torch.tensor(-1e16, dtype=torch.float64), torch.tensor(.125, dtype=torch.bfloat16)]
        expected = [float(value.item()) for value in values]
        actual = v2.scalar_values(values)
        self.assertEqual(actual, expected)
        self.assertEqual(sum(actual), sum(expected))
        self.assertEqual(v2.scalar_values([]), [])

    def test_policy_validation_padding_and_gradients_exact(self):
        self.policy_checks()

    def test_packed_rollout_actions_values_probabilities_rng_exact(self):
        self.sampling_checks()

    def test_model_gradients_adam_metrics_kl_and_rng_exact(self):
        self.update_checks()

    def test_default_routes_to_legacy_and_cli_is_explicit(self):
        trainer, _ = self.fixture(enabled=False)
        with patch.object(v2.legacy, 'categorical_logp', return_value='legacy') as legacy:
            self.assertEqual(trainer.policy_logp(None, None), 'legacy')
            legacy.assert_called_once_with(None, None)
        self.assertFalse(v2.arguments(['--rollout', '3']).coalesce_transfers)
        args = v2.arguments(['--rollout', '3', '--coalesce-transfers'])
        self.assertTrue(args.coalesce_transfers)
        self.assertEqual(args.rollout, 3)
        self.assertFalse(args.gpu_prepare)
        self.assertFalse(args.fused_encoding)
        args = v2.arguments(['--gpu-prepare', '--fused-encoding'])
        self.assertTrue(args.gpu_prepare and args.fused_encoding)
        trainer.args.fused_encoding = True
        before = v2.actor_digest(trainer.model)
        trainer.configure_model(True)
        self.assertTrue(trainer.model.fused_encoding)
        self.assertEqual(v2.actor_digest(trainer.model), before)

    def test_kl_stop_and_nonfinite_loss_guard_preserved(self):
        for enabled in (False, True):
            trainer, rollout = self.fixture(enabled=enabled)
            with patch.object(trainer, 'rollout_kl', return_value=1.):
                with self.assertRaisesRegex(ValueError, 'Behaviour/current policy mismatch'):
                    trainer.update(rollout, False)
            self.assertEqual(trainer.progress['optimizer_steps'], 0)
            self.assertIsNone(trainer._rollout_cache)
            rollout.returns[0] = np.nan
            with self.assertRaisesRegex(ValueError, 'Non-finite v2 PPO loss'):
                trainer.update(rollout, False)
            self.assertEqual(trainer.progress['optimizer_steps'], 0)

    def tensor_cache_checks(self, device='cpu'):
        trainer, rollout = self.fixture(device)
        expected = v2.RolloutCache(rollout, trainer.device)
        required = v2.RolloutCache.required_bytes(rollout)
        rollout.records = [{name: torch.as_tensor(value, device=device) for name, value in row.items()}
                           for row in rollout.records]
        rollout.old_logp = [row.to(device) for row in rollout.old_logp]
        rollout.labels = [torch.tensor(value, device=device) for value in rollout.labels]
        with patch.object(v2.np, 'asarray', side_effect=AssertionError('device input copied to NumPy')):
            self.assertEqual(v2.RolloutCache.required_bytes(rollout), required)
        actual = v2.RolloutCache(rollout, trainer.device)
        self.assertEqual(actual.bytes, expected.bytes)
        for left, right in zip(expected.select([2, 0, 3]), actual.select([2, 0, 3])):
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        rollout.records[0] = {name: value.cpu().numpy() for name, value in rollout.records[0].items()}
        rollout.labels[0] = 0
        self.assertEqual(v2.RolloutCache.required_bytes(rollout), required)
        mixed = v2.RolloutCache(rollout, trainer.device)
        for left, right in zip(expected.select([2, 0, 3]), mixed.select([2, 0, 3])):
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)

    def test_tensor_record_cache_estimate_and_padding_exact(self):
        self.tensor_cache_checks()

    def test_gpu_prepare_memory_bound_before_any_preparation(self):
        trainer, _ = self.fixture()
        trainer.args.gpu_prepare = True
        trainer.args.envs, trainer.args.rollout = 16, 512
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            trainer.gpu_prepare_plan()
        trainer.device = torch.device('cuda')
        trainer.args.rollout_cache_mib = 1024
        with patch.object(v2.game, 'BACKEND', 'native'), patch.object(torch.cuda, 'mem_get_info') as memory:
            memory.return_value = (8 * 2**30, 12 * 2**30)
            self.assertTrue(trainer.gpu_prepare_plan()['enabled'])
            memory.return_value = (256 * 2**20, 12 * 2**30)
            result = trainer.gpu_prepare_plan()
            self.assertFalse(result['enabled'])
            self.assertEqual(result['reason'], 'memory_bound')
            self.assertTrue(result['requested'])
            memory.return_value = (8 * 2**30, 12 * 2**30)
            trainer.args.rollout_cache_mib = 1
            self.assertFalse(trainer.gpu_prepare_plan()['enabled'])

    def test_batched_preparer_rollout_reset_labels_and_retained_records(self):
        results = []
        for enabled, oom_call in ((False, 0), (True, 0), (True, 2)):
            trainer, template = self.fixture(enabled=True)
            trainer.args.envs, trainer.args.rollout, trainer.args.pieces = 2, 2, 1
            trainer.args.gae_lambda = .999
            trainer.budget = SimpleNamespace(expired=lambda: False)
            trainer.progress.update(placements=0, clears=0, target_recoveries=0)
            trainer.environments = [SimpleNamespace(number=i, target=1, phase='training',
                                     state=SimpleNamespace(pieces=0, board=[0] * 4096)) for i in range(2)]
            trainer.prepared = trainer._prepared_gpu_batch = None
            trainer._gpu_prepare_active = False
            trainer.gpu_prepare_plan = lambda: {'requested': enabled, 'enabled': enabled}
            def prepare(episode):
                row = {name: value.copy() for name, value in template.records[episode.number].items()}
                row['targets'] = np.asarray(episode.target, dtype=np.int64)
                row['features'][:, 0] += episode.state.pieces * .01
                return v2.neural.Prepared(row, tuple(range(len(row['features']))), 0, 12, False)
            trainer.prepare_episode = prepare
            calls = [0]
            def batch_prepare(states, targets):
                calls[0] += 1
                if calls[0] == oom_call:
                    raise torch.OutOfMemoryError('synthetic preparation allocation')
                prepared = []
                for i, (state, target) in enumerate(zip(states, targets)):
                    p = prepare(SimpleNamespace(number=i, state=state, target=target))
                    arrays = {name: torch.as_tensor(value).clone() for name, value in p.arrays.items()}
                    prepared.append(v2.neural.Prepared(arrays, p.options, torch.tensor(p.teacher), p.evaluations, i == 0))
                return SimpleNamespace(prepared=tuple(prepared), targets=tuple(targets),
                    environment_indices=(0, 1), teachers=torch.zeros(2, dtype=torch.long), cpu_fallbacks=('clear', ''),
                    batch=v2.neural.collate([p.arrays for p in prepared], trainer.device),
                    selected_metadata=lambda indices: torch.stack([torch.tensor([index, 0, 0, 0, 0, 0, 0],
                                                                                dtype=torch.float64) for index in indices]),
                    candidates_from_metadata=lambda rows: tuple(int(row[0]) for row in rows),
                    selected_candidates=lambda indices: tuple(p.options[index] for p, index in zip(prepared, indices)))
            trainer._gpu_preparer = SimpleNamespace(prepare_batch=batch_prepare)
            def finish(index, reason):
                episode = trainer.environments[index]
                episode.state.pieces = 0
                trainer.prepared[index] = prepare(episode)
                trainer._prepared_gpu_batch = None
            trainer.finish_episode = finish
            trainer.value_inputs = lambda episodes: (np.zeros(len(episodes), dtype=np.float32), None)
            actions = []
            def apply(episode, candidate, update, behavior):
                episode.state.pieces += 1
                actions.append((episode.number, candidate, behavior))
                return SimpleNamespace(layers=0)
            torch.manual_seed(777)
            with patch.object(v2.base, 'apply_move', side_effect=apply):
                rollout = trainer.rollout(False)
            snapshot = {name: torch.as_tensor(value).clone() for name, value in rollout.records[0].items()}
            batch_prepare([e.state for e in trainer.environments], [1, 1])
            for name, value in snapshot.items():
                torch.testing.assert_close(torch.as_tensor(rollout.records[0][name]), value, rtol=0, atol=0)
            results.append((trainer, rollout, actions, torch.get_rng_state().clone()))
        old = results[0]
        for new in results[1:]:
            self.assertEqual(old[2], new[2])
            self.assertTrue(torch.equal(old[3], new[3]))
            self.assertEqual(old[1].choices, new[1].choices)
            for name in ('advantages', 'returns', 'old_values', 'raw_advantages', 'terminals', 'truncations'):
                np.testing.assert_array_equal(getattr(old[1], name), getattr(new[1], name))
        for trainer, rollout, _, _ in results:
            torch.manual_seed(888)
            trainer.update(rollout, False)
        for new in results[1:]:
            fixtures.RolloutCacheTests().assert_state_equal(old[0], new[0])
        self.assertEqual(results[2][1].gpu_prepare['runtime_fallback'], 'allocation_failed')

    def test_retired_device_inputs_release_only_after_successful_update(self):
        trainer, rollout = self.fixture()
        rollout.gpu_prepare = {'enabled': True}
        with patch.object(trainer, '_update', side_effect=ValueError('before optimizer')):
            with self.assertRaisesRegex(ValueError, 'before optimizer'):
                trainer.update(rollout, False)
        self.assertEqual(len(rollout.records), 4)
        self.assertIsNone(getattr(trainer, '_previous_device_rollout', None))
        torch.manual_seed(777)
        trainer.update(rollout, False)
        self.assertIsInstance(trainer._previous_device_rollout, weakref.ReferenceType)
        self.assertEqual(len(rollout.records), 4)
        trainer.release_previous_device_rollout()
        self.assertEqual((rollout.records, rollout.old_logp, rollout.labels), ([], [], []))
        self.assertEqual(len(rollout.returns), 4)

    @unittest.skipUnless(os.environ.get('RUN_TETRIS_CUDA_TRANSFER_TESTS') == '1' and torch.cuda.is_available(),
                         'CUDA checks require an explicitly authorized bounded GPU probe')
    def test_cuda_actions_validation_model_adam_and_metrics_exact(self):
        self.policy_checks('cuda')
        self.sampling_checks('cuda')
        self.tensor_cache_checks('cuda')
        self.update_checks('cuda')

    def _cuda_16env_pair(self, optimized_first=False):
        directory = Path(os.environ['TETRIS4D_GPU_INTEGRATION_FIXTURES'])
        checkpoint = Path(os.environ['TETRIS4D_GPU_INTEGRATION_CHECKPOINT'])
        manifest = json.loads((directory / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), manifest['checkpoint']['sha256'])
        reference_path = Path(manifest['reference_sources']['tetris_neural.py']['path']).parent
        # Load the original trainer and all four dependencies in their own module
        # objects; restore the process import table before running the new trainer.
        names = ('train_survival_v2', 'train_survival', 'train_overnight', 'tetris_neural', 'sparse_tetris_4d_optimized')
        original = {name: sys.modules.pop(name, None) for name in names}
        before_path = sys.path[:]
        try:
            sys.path.insert(0, str(reference_path))
            reference = importlib.import_module('train_survival_v2')
        finally:
            sys.path[:] = before_path
            for name, module in original.items():
                sys.modules.pop(name, None)
                if module is not None:
                    sys.modules[name] = module
        cases = json.loads((directory / 'cases.json').read_text())
        wanted = [f'frozen-{i}' for i in range(4)] + ['single-clear', 'double-clear',
                  'full-target-recovery', 'topout', 'cap-piece-count']
        chosen = [next(row for row in cases if row['label'] == name) for name in wanted]
        chosen += [row for row in cases if row['label'].startswith('empty-')][:16 - len(chosen)]
        results = []
        with np.load(directory / 'arrays.npz', allow_pickle=False) as arrays:
            order = ((v2, True), (reference, False)) if optimized_first else ((reference, False), (v2, True))
            for module, optimized in order:
                print('GPU integration: ' + ('optimized' if optimized else 'frozen reference') + ' rollout', flush=True)
                module.game.configure_backend('native')
                module.game.load_checkpoint(Path(__file__).resolve().parents[1] / 'models/trained-sigmoid.json')
                fixture, _ = self.fixture('cpu')
                trainer = module.SurvivalTrainer.__new__(module.SurvivalTrainer)
                trainer.__dict__.update(fixture.__dict__)
                trainer.device = torch.device('cuda')
                trainer.model, saved = module.neural.load_checkpoint(checkpoint, trainer.device)
                self.assertEqual((trainer.model.config.width, trainer.model.config.blocks), (32, 3))
                self.assertEqual(saved['teacher_parameters'], module.game.network_parameters())
                trainer.model.fused_encoding = optimized
                trainer.args.coalesce_transfers = trainer.args.gpu_prepare = optimized
                trainer.args.fused_encoding = optimized
                trainer.args.lr, trainer.args.critic_lr = 1e-4, 3e-4
                trainer.args.batch, trainer.args.optimizer_batch, trainer.args.ppo_epochs = 16, 128, 3
                trainer.args.target_kl, trainer.args.rollout_cache_mib = .02, 1024
                trainer.args.envs, trainer.args.rollout, trainer.args.pieces = 16, 4, 32000
                trainer.args.seed, trainer.args.piece_set, trainer.args.gae_lambda = 250000, 'expanded', .999
                trainer.optimizer = trainer.create_optimizer()
                trainer.optimizer.load_state_dict(saved['optimizer'])
                trainer.scaler = torch.amp.GradScaler('cuda', enabled=False)
                trainer.scaler.load_state_dict(saved['extra'].get('scaler', {}))
                trainer.progress = copy.deepcopy(saved['progress'])
                trainer.initial_actor_sha256 = module.actor_digest(trainer.model)
                trainer.environments = []
                for index, row in enumerate(chosen):
                    state = module.game.Game(row['seed'], row['piece_set'])
                    state.board = tuple(map(int, arrays[row['prefix'] + 'state_board']))
                    if row['label'] == 'full-target-recovery':
                        # The prepare-only negative case contains a preexisting full
                        # layer; use a physically valid buried target for committed moves.
                        state.board = (2,) + (1,) * 4095
                    state.stream = module.game.PieceStream(row['seed'], row['piece_set'])
                    for _ in range(row['pieces'] + 2):
                        state.stream.pop()
                    state.current, state.next_piece = row['current'], row['next_piece']
                    state.pieces, state.piece_counts = row['pieces'], dict(row['piece_counts'])
                    trainer.environments.append(module.base.Episode(index, state, row['requested_target']))
                trainer.next_episode, trainer.pending, trainer.prepared = 16, [], None
                trainer._gpu_preparer = trainer._prepared_gpu_batch = None
                trainer._gpu_prepare_active = False
                trainer.budget = SimpleNamespace(expired=lambda: False)
                trainer.events = []
                trainer.emit = lambda event, _trainer=trainer, **fields: _trainer.events.append((event, fields))
                torch.manual_seed(8831)
                torch.cuda.synchronize()
                started = time.perf_counter()
                rollout = trainer.rollout(False)
                torch.cuda.synchronize()
                times = {'rollout_seconds': time.perf_counter() - started}
                self.assertEqual(len(rollout.records), 64)
                results.append((trainer, rollout, torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(), optimized, times))
        print('GPU integration: compare inputs, actions, states, GAE and RNG', flush=True)
        old = next(row for row in results if not row[4])
        new = next(row for row in results if row[4])
        self.assertEqual(old[1].choices, new[1].choices)
        self.assertEqual([int(x) for x in old[1].labels], [int(x) for x in new[1].labels])
        for expected, actual in zip(old[1].records, new[1].records):
            for name in expected:
                torch.testing.assert_close(torch.as_tensor(expected[name]), torch.as_tensor(actual[name]).cpu(), rtol=0, atol=0, msg=name)
        for expected, actual in zip(old[1].old_logp, new[1].old_logp):
            torch.testing.assert_close(expected.cpu(), actual.cpu(), rtol=0, atol=0)
        for name in ('advantages', 'returns', 'old_values', 'raw_advantages', 'terminals', 'truncations'):
            np.testing.assert_array_equal(getattr(old[1], name), getattr(new[1], name), err_msg=name)
        for expected, actual in zip(old[0].environments, new[0].environments):
            self.assertEqual(expected.snapshot(), actual.snapshot())
        self.assertEqual(old[0].progress, new[0].progress)
        self.assertTrue(torch.equal(old[2], new[2]) and torch.equal(old[3], new[3]))
        self.assertTrue(new[1].gpu_prepare['enabled'])
        self.assertNotIn('runtime_fallback', new[1].gpu_prepare)
        print('GPU integration: optimizer, gradient and metric parity', flush=True)
        for trainer, rollout, _, _, _, times in results:
            torch.manual_seed(9942)
            torch.cuda.synchronize()
            started = time.perf_counter()
            trainer.update(rollout, False)
            torch.cuda.synchronize()
            times['update_seconds'] = time.perf_counter() - started
            times['total_seconds'] = times['rollout_seconds'] + times['update_seconds']
            trainer.integration_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
        fixtures.RolloutCacheTests().assert_state_equal(old[0], new[0])
        for expected, actual in zip(old[0].model.parameters(), new[0].model.parameters()):
            if expected.grad is not None:
                torch.testing.assert_close(expected.grad, actual.grad, rtol=0, atol=0)
        for expected, actual in zip(old[0].integration_rng, new[0].integration_rng):
            self.assertTrue(torch.equal(expected, actual))
        a, b = copy.deepcopy(old[0].events[-1][1]), copy.deepcopy(new[0].events[-1][1])
        for metrics in (a, b):
            for name in ('phase_seconds', 'peak_allocated_mib', 'peak_reserved_mib', 'gpu_prepare'):
                metrics.pop(name, None)
        self.assertEqual(a, b)
        print('GPU integration: exact64-transition16env frozen-source parity passed', flush=True)
        def digest(rows):
            h = hashlib.sha256()
            for name, value in rows:
                h.update(str(name).encode())
                if value is None:
                    h.update(b'none')
                else:
                    h.update(str(value.dtype).encode())
                    h.update(str(tuple(value.shape)).encode())
                    h.update(value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        optimizer = new[0].optimizer.state_dict()
        return {'order': ['optimized', 'reference'] if optimized_first else ['reference', 'optimized'],
                'reference': old[5], 'optimized': new[5], 'exact_parity': True, 'transitions': 64,
                'optimizer_steps': new[0].events[-1][1]['optimizer_steps_this_rollout'],
                'completed_epochs': new[0].events[-1][1]['completed_epochs'],
                'final_exact_kl': new[0].events[-1][1]['final_exact_kl'],
                'model_sha256': digest(new[0].model.state_dict().items()),
                'gradient_sha256': digest((name, p.grad) for name, p in new[0].model.named_parameters()),
                'adam_sha256': digest((f'{i}.{key}', value) for i, row in optimizer['state'].items() for key, value in row.items()),
                'cpu_rng_sha256': hashlib.sha256(new[0].integration_rng[0].numpy().tobytes()).hexdigest(),
                'cuda_rng_sha256': hashlib.sha256(new[0].integration_rng[1].cpu().numpy().tobytes()).hexdigest()}

    @unittest.skipUnless(os.environ.get('RUN_TETRIS_CUDA_TRANSFER_TESTS') == '1'
                         and os.environ.get('TETRIS4D_GPU_INTEGRATION_FIXTURES') and torch.cuda.is_available(),
                         'Integrated CUDA check requires reviewed native fixtures and an authorized probe')
    def test_cuda_16env_rollout_and_optimizer_parity(self):
        benchmark = os.environ.get('TETRIS4D_GPU_INTEGRATION_BENCHMARK') == '1'
        pairs = []
        for index in range(4 if benchmark else 1):
            pair = self._cuda_16env_pair(optimized_first=bool(index % 2))
            pair.update(index=index, warmup=benchmark and index == 0)
            pairs.append(pair)
        if benchmark:
            for key in ('model_sha256', 'gradient_sha256', 'adam_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256'):
                self.assertEqual(len({row[key] for row in pairs}), 1, key)
            measured = pairs[1:]
            medians = {mode: {phase: statistics.median(row[mode][phase] for row in measured)
                             for phase in ('rollout_seconds', 'update_seconds', 'total_seconds')}
                       for mode in ('reference', 'optimized')}
            report = {'schema': 'tetris4d-gpu-integrated-microbenchmark-v1', 'passed': True,
                      'scope': 'Concurrent64-transition microbenchmark; not8192-transition training throughput.',
                      'timing': 'CUDA-synchronized wall; excludes checkpoint/setup; includes preparation, host copies, cache construction and full KL checks.',
                      'settings': {'envs': 16, 'rollout_steps': 4, 'transitions': 64, 'batch': 16,
                                   'optimizer_batch': 128, 'ppo_epochs': 3, 'precision': 'fp32', 'width': 32, 'blocks': 3},
                      'pairs': pairs, 'measured_pairs': 3, 'median_seconds': medians,
                      'observed_concurrent_ratio': {phase: medians['reference'][phase] / medians['optimized'][phase]
                                                    for phase in medians['reference']}}
            path = Path(os.environ['TETRIS4D_GPU_INTEGRATION_BENCHMARK_OUTPUT'])
            with path.open('x') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write('\n')


if __name__ == '__main__':
    unittest.main()
