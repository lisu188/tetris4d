"""The timing gate distinguishes row-level recovery from a CPU rollout."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_spatial_ppo import BOUNDARY_CACHE_MIB, boundary_episodes, boundary_events, validate_preparation


class PreparationGateTests(unittest.TestCase):
    def row(self, batches=257, terminals=0, fallbacks=None):
        return {
            'rollout_placements': 8192,
            'terminal_transitions': terminals, 'truncated_transitions': 0,
            'gpu_prepare': {'enabled': True, 'preparations': {
                'batch_calls': batches, 'environment_calls': 32 * batches,
                'cpu_fallbacks': fallbacks or {}}},
        }

    def test_failed_gate_is_off_and_binds_all_available_receipts(self):
        import json
        from tempfile import TemporaryDirectory
        from tools.benchmark_spatial_ppo import report_failed_gate, verified_gate
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'opening').mkdir()
            receipt = root/'opening/1-reference.json'
            receipt.write_text('{"seconds": 4}')
            gate = root/'gate.json'
            report_failed_gate(root, gate, 'preparation time budget exhausted')
            report = verified_gate(root, root, 'cpu', 'unused')
            self.assertEqual(report['enabled'], {'two_stage': False, 'headless': False})
            self.assertFalse(report['parity'])
            report['enabled']['headless'] = True
            gate.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                verified_gate(root, root, 'cpu', 'unused')
            report_failed_gate(root, gate, 'preparation time budget exhausted')
            receipt.write_text('{"seconds": 3}')
            with self.assertRaises(ValueError):
                verified_gate(root, root, 'cpu', 'unused')

    def test_complete_rollout_and_per_environment_recovery(self):
        validate_preparation(self.row(), 32, 256)
        # Three terminal rows across two steps caused two new batches.
        validate_preparation(self.row(259, 3, {'exhaustive_or_recovery': 3}), 32, 256)
        # A recovered target can continue the same episode.
        validate_preparation(self.row(fallbacks={'buried_target': 1}), 32, 256)

    def test_profile_is_standalone_and_never_overwrites(self):
        import subprocess
        from tempfile import TemporaryDirectory
        from tools import benchmark_spatial_ppo as benchmark
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing.json'
            output.write_text('previous report')
            cmd = [sys.executable, '-B', benchmark.__file__, '--profile-step', 'cpu', '--incumbent', 'unused']
            for extra, message in ((['--output', str(output)], 'requires a new --output'),
                                   (['--output', str(Path(tmp)/'new.json'), '--pairs', '2'], 'is standalone'),
                                   (['--output', str(Path(tmp)/'new.json'), '--compare-simulation'], 'is standalone')):
                with self.subTest(extra=extra):
                    result = subprocess.run(cmd + extra, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)
            self.assertEqual(output.read_text(), 'previous report')

    def test_one_cpu_profile_exports_host_ranges_without_leaking_wrappers(self):
        import gzip
        import json
        from tempfile import TemporaryDirectory
        import torch
        from tools.benchmark_spatial_ppo import profile_rollout_step

        def shortlist():
            return torch.arange(4).sum().item()

        preparer = SimpleNamespace(_screen_shortlist=shortlist)
        trainer = SimpleNamespace(args=SimpleNamespace(envs=32, rollout=256), _gpu_preparer=preparer)

        def rollout(warmup):
            self.assertFalse(warmup)
            self.assertEqual(trainer.args.rollout, 1)
            self.assertEqual(preparer._screen_shortlist(), 6)
            return SimpleNamespace(choices=range(32))

        trainer.rollout = rollout
        with TemporaryDirectory() as tmp:
            trace = Path(tmp)/'profiles'/'step.json.gz'
            profile_rollout_step(trainer, 'cpu', trace, {'device': 'cpu', 'rollout_steps': 1})
            report = json.loads(gzip.decompress(trace.read_bytes()))
            self.assertEqual(report['simulation_benchmark'], {'device': 'cpu', 'rollout_steps': 1})
            names = {e.get('name') for e in report['traceEvents']}
            self.assertTrue({'benchmark/rollout_step', 'simulation/screen_shortlist', 'aten::sum'} <= names)
            self.assertIs(preparer._screen_shortlist, shortlist)

    def test_host_only_trace_cannot_be_published_as_cuda(self):
        from tempfile import TemporaryDirectory
        import torch
        from tools.benchmark_spatial_ppo import profile_rollout_step
        cpu, cuda = torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA
        host_only = torch.profiler.profile(activities=[cpu])
        trainer = SimpleNamespace(args=SimpleNamespace(envs=32), _gpu_preparer=None,
                                  _gpu_prepare_status=dict(enabled=True, preparations=dict(
                                      batch_calls=1, environment_calls=32, cpu_fallbacks={})),
                                  rollout=lambda warmup: SimpleNamespace(choices=range(32)))
        with TemporaryDirectory() as tmp, patch.object(torch.profiler, 'supported_activities', return_value={cpu, cuda}), \
                patch.object(torch.cuda, 'synchronize'), patch.object(torch.profiler, 'profile', return_value=host_only):
            trace = Path(tmp)/'cuda.json'
            with self.assertRaisesRegex(RuntimeError, 'did not emit any CUDA kernels'):
                profile_rollout_step(trainer, 'cuda', trace, {'device': 'cuda'})
            self.assertFalse(trace.exists())

    def test_enabled_flag_alone_does_not_prove_cuda_work(self):
        fallback = self.row()
        fallback['gpu_prepare']['runtime_fallback'] = 'allocation_failed'
        incomplete = self.row()
        incomplete['gpu_prepare']['preparations']['environment_calls'] -= 1
        for row in (fallback, incomplete, self.row(256), self.row(258),
                    self.row(fallbacks={'candidate_budget_bound': 1}),
                    self.row(fallbacks={'buried_target': 8224})):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_preparation(row, 32, 256)

    def test_boundary_fixtures_drive_real_commits_and_fit_the_guard_window(self):
        import torch
        import train_survival_v4 as v4

        previous_backend = v4.game.BACKEND
        v4.game.configure_backend('numpy')
        self.addCleanup(v4.game.configure_backend, previous_backend)
        trainer = v4.SurvivalTrainer.__new__(v4.SurvivalTrainer)
        trainer.model = v4.neural.create_model(v4.neural.Config(
            architecture=v4.ARCHITECTURE, spatial_actor=True, regional_critic=True,
            policy_mode='teacher-prior', recover_target=True, screening='exhaustive'))
        trainer.device = torch.device('cpu')
        trainer.progress = {'target_recoveries': 0}
        trainer.emit = lambda *args, **kwargs: None
        trainer.args = SimpleNamespace(envs=32, rollout=256, rollout_cache_mib=BOUNDARY_CACHE_MIB,
                                       cache_rollout=True, compact_rollout=True, gpu_prepare=True)
        episodes = boundary_episodes(v4, (0, 1))
        rows = []
        for episode in episodes:
            prepared = trainer.prepare_episode(episode)
            rows.append(prepared.arrays)
            v4.base.apply_move(episode, prepared.options[prepared.teacher], 0, 'validation')
        events = boundary_events(episodes)
        self.assertEqual(events['clearing_first_move']['cleared_mask'], 15)
        self.assertEqual(events['recovery_first_move']['retarget_from'], 12)
        self.assertEqual(trainer.progress['target_recoveries'], 1)
        raw = v4.neural.collate(rows, trainer.device)
        cache, same_batch, reason = trainer.start_fixed_cache(raw)
        self.assertIsNone(cache)
        self.assertIs(same_batch, raw)
        self.assertEqual(reason, 'memory_bound')
        # Exercise the existing raw-preparation byte estimate without a CUDA context.
        trainer.device = torch.device('cuda')
        with patch.object(v4.game, 'BACKEND', 'native'), patch.object(v4.v2, 'cuda_available_bytes', return_value=8 * 2**30):
            self.assertTrue(trainer.gpu_prepare_plan()['enabled'])
