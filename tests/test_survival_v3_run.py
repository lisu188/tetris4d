"""CPU-only checks of the fresh-run budget, provenance and safe supervision."""
from pathlib import Path
import contextlib
import io
import json
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_survival_v3 as runner


class Clock:
    def __init__(self, epoch=90., step=20.):
        self.epoch, self.elapsed, self.step = epoch, 0., step

    def sleep(self, unused):
        self.epoch += self.step
        self.elapsed += self.step


class Child:
    pid = 42

    def __init__(self, exits_after_signal=True):
        self.returncode, self.signals = None, []
        self.exits_after_signal = exits_after_signal

    def poll(self):
        if self.signals and self.exits_after_signal:
            self.returncode = 0
        return self.returncode

    def send_signal(self, number):
        self.signals.append(number)


class FreshRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='tetris-v3-controller-test-', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'architecture-training.artifacts'
        self.source = self.output/'source'
        for name in runner.RUNTIME:
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'reviewed runtime')
        self.weights = self.root/'fixed-teacher.json'
        self.weights.write_text('{}')
        self.args = types.SimpleNamespace(source=self.source, output=self.output,
            cache=self.root/'private-cache', weights=self.weights, python=Path(sys.executable),
            seed=650000, calibration_seed=5500000, selection_seeds=[980101, 980103],
            run=False, plan_sha256=None)

    def git(self, command, **kwargs):
        return {'branch': 'redesign/survival-policy-v3\n', 'status': '', 'rev-parse': 'a'*40+'\n'}[command[3]]

    def config(self):
        with patch.object(runner.subprocess, 'check_output', side_effect=self.git):
            return runner.configuration(self.args)

    def prepare_supervisor(self):
        config = self.config()
        config.update(start_epoch=0., deadline_epoch=100.)
        for name in ('logs', 'control', 'run'):
            (self.output/name).mkdir()
        initialization = {'schema': 'tetris4d-fresh-initialization-v1', 'fresh_random_trainable_weights': True,
                          'inherited_checkpoint': None, 'inherited_optimizer': False,
                          'seed': config['seed'], 'teacher_weights_sha256': config['fixed_heuristic_prior']['sha256']}
        (self.output/'run/initialization.json').write_text(json.dumps(initialization))
        state = {'state': 'running', 'configuration': {}}
        return config, state

    @contextlib.contextmanager
    def mock_supervisor(self, process, clock, **overrides):
        with contextlib.ExitStack() as stack:
            patches = {'frozen': Mock(), 'storage': Mock(return_value={}), 'gpu_status': Mock(return_value={}),
                       'process_start': Mock(return_value='123')}
            patches.update(overrides)
            for name, value in patches.items():
                stack.enter_context(patch.object(runner, name, value))
            stack.enter_context(patch.object(runner.subprocess, 'Popen', return_value=process))
            stack.enter_context(patch.object(runner.time, 'time', side_effect=lambda: clock.epoch))
            stack.enter_context(patch.object(runner.time, 'monotonic', side_effect=lambda: clock.elapsed))
            stack.enter_context(patch.object(runner.time, 'sleep', side_effect=clock.sleep))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def test_default_plan_is_read_only_and_starts_no_budget_or_child(self):
        config = self.config()
        before = set(self.output.iterdir())
        with patch.object(runner, 'arguments', return_value=self.args), \
             patch.object(runner, 'configuration', return_value=config), \
             patch.object(runner, 'storage', return_value={}), \
             patch.object(runner.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runner.main([]), 0)
        value = json.loads(out.getvalue())
        self.assertIsNone(value['configuration']['start_epoch'])
        self.assertIsNone(value['configuration']['deadline_epoch'])
        self.assertEqual(value['plan_sha256'], runner.plan_hash(config))
        self.assertEqual(set(self.output.iterdir()), before)
        self.assertFalse(self.args.cache.exists())
        launch.assert_not_called()

    def test_reviewed_plan_digest_rejects_changed_settings(self):
        self.args.plan_sha256 = '0'*64
        with patch.object(runner, 'arguments', return_value=self.args), \
             patch.object(runner, 'configuration', return_value=self.config()), \
             patch.object(runner, 'run') as launch:
            with self.assertRaisesRegex(ValueError, 'plan digest'):
                runner.main([])
        launch.assert_not_called()

    def test_command_is_fresh_16_environment_run_with_remaining_hours(self):
        config = self.config()
        command = runner.training_command(config, 7.25)
        for flag, expected in [('--hours', '7.25'), ('--envs', '16'), ('--rollout', '1024'),
                ('--optimizer-batch', '256'), ('--gpu-budget-mib', '6144'), ('--gpu-headroom-mib', '512'),
                ('--record-max-mib', '16'), ('--record-every', '4'), ('--calibration-steps', '600')]:
            self.assertEqual(command[command.index(flag)+1], expected)
        self.assertNotIn('--checkpoint', command)
        self.assertNotIn('--record-games', command)
        self.assertNotIn('--allow-concurrent', command)
        self.assertNotIn('--preserve-initial-actor', command)
        self.assertIn('--gpu-prepare', command)
        self.assertIn('--worker', command)
        with self.assertRaises(ValueError):
            runner.training_command(config, 8.01)

    def test_dirty_runtime_or_main_branch_cannot_launch(self):
        for branch, dirty in [('main\n', ''), ('redesign/v3\n', ' M train_survival_v3.py')]:
            def git(command, **kwargs):
                return {'branch': branch, 'status': dirty, 'rev-parse': 'a'*40}[command[3]]
            with patch.object(runner.subprocess, 'check_output', side_effect=git):
                with self.assertRaises(ValueError):
                    runner.configuration(self.args)

    def test_source_native_and_fixed_prior_are_all_pinned(self):
        config = self.config()
        pinned = {Path(row['path']).name for row in config['files']}
        self.assertTrue({'train_survival_v3.py', 'tetris_survival_v3.py', 'tetris_gpu_simulation.py',
                         'tetris_gpu_encoding.py', 'libtetris4d_native.so', self.weights.name}.issubset(pinned))
        with patch.object(runner.subprocess, 'check_output', return_value=config['commit']):
            runner.frozen(config)
            (self.source/'native/libtetris4d_native.so').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                runner.frozen(config)

    def test_commit_change_and_configuration_change_are_rejected(self):
        config = self.config()
        with patch.object(runner.subprocess, 'check_output', return_value='b'*40):
            with self.assertRaisesRegex(ValueError, 'commit changed'):
                runner.frozen(config)
        path = self.root/'frozen.json'
        path.write_text('{}')
        proof = runner.proof(path)
        path.write_text('{"modified": true}')
        with self.assertRaisesRegex(ValueError, 'configuration changed'):
            runner.frozen(config, proof)

    def test_seed_and_path_isolation(self):
        for seeds in ([960001], [980100], [980101, 980101]):
            self.args.selection_seeds = seeds
            with self.assertRaisesRegex(ValueError, 'development seeds'):
                self.config()
        self.args.selection_seeds = [980101, 980103]
        self.args.calibration_seed = 100
        with self.assertRaisesRegex(ValueError, 'seed ranges'):
            self.config()
        self.args.calibration_seed = 5500000
        self.args.output = self.root/'elsewhere'
        with self.assertRaisesRegex(ValueError, 'isolated output/source'):
            self.config()

    def test_storage_excludes_only_source_and_reserves_two_atomic_copies(self):
        config = self.config()
        run = self.output/'run'
        run.mkdir()
        checkpoint = run/'last.pt'
        with checkpoint.open('wb') as stream:
            stream.truncate(4*runner.MIB)
        with (self.source/'large-offline-source').open('wb') as stream:
            stream.truncate(512*runner.MIB)
        usage = types.SimpleNamespace(total=1000*runner.MIB, used=100*runner.MIB, free=900*runner.MIB)
        with patch.object(runner.os.path, 'ismount', side_effect=lambda p: str(p) == '/mnt/c'), \
             patch.object(runner.shutil, 'disk_usage', return_value=usage):
            result = runner.storage(config)
            self.assertEqual(result['owned_output_bytes'], 4*runner.MIB)
            self.assertEqual(result['checkpoint_reserve_bytes'], 16*runner.MIB)
            with (self.output/'too-large').open('wb') as stream:
                stream.truncate(236*runner.MIB)
            with self.assertRaisesRegex(ValueError, 'atomic checkpoint reserve'):
                runner.storage(config)

    def test_private_cache_and_windows_volume_guards(self):
        config = self.config()
        self.args.cache.mkdir()
        with (self.args.cache/'oversized-cache').open('wb') as stream:
            stream.truncate(64*runner.MIB)
        with self.assertRaisesRegex(ValueError, 'cache bound'):
            runner.storage(config)
        (self.args.cache/'oversized-cache').unlink()
        with patch.object(runner.os.path, 'ismount', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Windows backing'):
                runner.storage(config)

    def test_storage_rejects_symlink_escapes(self):
        config = self.config()
        (self.output/'escape').symlink_to(self.weights)
        with self.assertRaisesRegex(ValueError, 'escape'):
            runner.storage(config)

    def test_ninety_percent_future_space_guard(self):
        config = self.config()
        usage = types.SimpleNamespace(total=1000*runner.MIB, used=850*runner.MIB, free=150*runner.MIB)
        with patch.object(runner.os.path, 'ismount', side_effect=lambda p: str(p) == '/mnt/c'), \
             patch.object(runner.shutil, 'disk_usage', return_value=usage):
            with self.assertRaisesRegex(ValueError, 'disk headroom'):
                runner.storage(config)

    def test_existing_run_cannot_restart(self):
        config = self.config()
        (self.output/'control').mkdir()
        (self.output/'control/status.json').write_text('{}')
        with patch.object(runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'automatic resume'):
                runner.run(config)
        launch.assert_not_called()

    def test_eight_hour_ledger_starts_at_run_and_includes_preparation(self):
        config = self.config()
        captured = {}
        def supervise(configuration, state, requests, monotonic):
            captured.update(config=configuration, state=state, monotonic=monotonic)
            return 0
        with patch.object(runner, 'frozen'), patch.object(runner, 'storage', return_value={}), \
             patch.object(runner, 'supervise', side_effect=supervise), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(config, start_epoch=1000., start_monotonic=50.), 0)
        ledger = runner.read(self.output/'control/budget-ledger.json')
        self.assertEqual(ledger['deadline_epoch'], 29800.)
        self.assertEqual(captured['monotonic'], 50.)
        self.assertTrue(captured['state']['fresh_run'])
        self.assertIn(captured['state']['budget'], captured['config']['files'])
        self.assertEqual(self.args.cache.stat().st_mode & 0o777, 0o700)

    def test_deadline_sends_one_sigint_then_finishes_after_checkpoint(self):
        config, state = self.prepare_supervisor()
        process, clock = Child(), Clock()
        with self.mock_supervisor(process, clock):
            self.assertEqual(runner.supervise(config, state, [], 0.), 0)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(state['state'], 'finished')
        self.assertEqual(state['stop_reason'], 'eight_hour_budget')
        self.assertLess(float(state['command'][state['command'].index('--hours')+1]), 8.)
        self.assertEqual(state['initialization'], runner.proof(self.output/'run/initialization.json'))

    def test_storage_failure_stops_child_without_restart(self):
        config, state = self.prepare_supervisor()
        process, clock = Child(), Clock()
        storage = Mock(side_effect=[{}, ValueError('output reserve reached'), {}])
        with self.mock_supervisor(process, clock, storage=storage):
            self.assertEqual(runner.supervise(config, state, [], 0.), 2)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(state['state'], 'failed')
        self.assertIn('output reserve', state['error'])

    def test_grace_timeout_never_forces_kill(self):
        config, state = self.prepare_supervisor()
        process, clock = Child(exits_after_signal=False), Clock(step=601.)
        with self.mock_supervisor(process, clock):
            self.assertEqual(runner.supervise(config, state, [], 0.), 2)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(state['state'], 'blocked')
        self.assertTrue(state['child_still_running'])
        self.assertIn('no forced kill', state['error'])

    def test_signal_is_forwarded_once_as_sigint(self):
        config, state = self.prepare_supervisor()
        config['deadline_epoch'] = 10000.
        process, clock, requests = Child(), Clock(), []
        original = clock.sleep
        def sleep(unused):
            original(unused)
            requests.extend([signal.SIGTERM, signal.SIGTERM])
        with self.mock_supervisor(process, clock), patch.object(runner.time, 'sleep', side_effect=sleep):
            self.assertEqual(runner.supervise(config, state, requests, 0.), 0)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(state['state'], 'stopped')

    def test_unexpected_reporting_failure_still_checkpoints_child(self):
        config, state = self.prepare_supervisor()
        process, clock = Child(), Clock()
        with self.mock_supervisor(process, clock), patch.object(runner, 'stats', side_effect=RuntimeError('bad report')):
            self.assertEqual(runner.supervise(config, state, [], 0.), 2)
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertFalse(state['child_still_running'])
        self.assertEqual(state['state'], 'failed')

    def test_fresh_initialization_proof_rejects_transferred_optimizer(self):
        config, _ = self.prepare_supervisor()
        path = self.output/'run/initialization.json'
        value = runner.read(path)
        value['inherited_optimizer'] = True
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fresh neural weights'):
            runner.initialization_proof(path, config)

    def test_cuda_allocator_bound_and_startup_headroom(self):
        total = 8192*runner.MIB
        cuda = types.SimpleNamespace(is_available=lambda: True,
            get_device_properties=lambda device: types.SimpleNamespace(total_memory=total),
            set_per_process_memory_fraction=Mock(), mem_get_info=Mock(return_value=(7000*runner.MIB, total)))
        runner.cuda_budget(types.SimpleNamespace(cuda=cuda), 6144*runner.MIB, 512*runner.MIB)
        cuda.set_per_process_memory_fraction.assert_called_once_with(.75, 0)
        cuda.mem_get_info.return_value = (6500*runner.MIB, total)
        with self.assertRaisesRegex(ValueError, 'Insufficient free GPU'):
            runner.cuda_budget(types.SimpleNamespace(cuda=cuda), 6144*runner.MIB, 512*runner.MIB)

    def test_periodic_gpu_free_headroom_guard(self):
        config = self.config()
        with patch.object(runner.subprocess, 'check_output', return_value='7000, 8192\n'):
            self.assertEqual(runner.gpu_status(config, starting=True)['free_bytes'], 7000*runner.MIB)
        with patch.object(runner.subprocess, 'check_output', return_value='500, 8192\n'):
            with self.assertRaisesRegex(ValueError, 'GPU free memory'):
                runner.gpu_status(config)

    def test_worker_imports_only_v3_after_setting_allocator_bound(self):
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace()}), \
             patch.object(runner, 'cuda_budget', return_value={}) as budget, \
             patch('runpy.run_path') as execute, patch.object(sys, 'argv', ['worker']), \
             patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
            runner.worker(['--gpu-budget-mib', '6144', '--gpu-headroom-mib', '512', '--', '--seed', '650000'])
            budget.assert_called_once()
            self.assertEqual(sys.argv, [str(runner.SOURCE/'train_survival_v3.py'), '--seed', '650000'])
            self.assertEqual(sys.path[0], str(runner.SOURCE))
            execute.assert_called_once_with(str(runner.SOURCE/'train_survival_v3.py'), run_name='__main__')
            with self.assertRaisesRegex(ValueError, 'checkpoint transfer'):
                runner.worker(['--gpu-budget-mib', '6144', '--gpu-headroom-mib', '512', '--', '--checkpoint=old.pt'])

    def test_statistics_read_latest_complete_metric_and_active_max(self):
        config = self.config()
        run = self.output/'run'
        run.mkdir()
        (run/'summary.json').write_text(json.dumps({'runner_updates': 4, 'placements': 65536,
            'active_games': [{'episode': 1, 'pieces': 600}, {'episode': 2, 'pieces': 700}]}))
        (run/'metrics.jsonl').write_text('{"event":"update","loss":0.1}\n{"partial":')
        value = runner.stats(config)
        self.assertEqual(value['current_max_survived_pieces'], 700)
        self.assertEqual(value['latest_event']['event'], 'update')
        self.assertEqual(value['placements'], 65536)

    def test_reports_every_five_minutes_and_rechecks_inputs_every_thirty_seconds(self):
        config, state = self.prepare_supervisor()
        config['deadline_epoch'] = 10000.
        clock = Clock(step=30.)
        class FiniteChild(Child):
            def poll(self):
                if clock.elapsed >= 650:
                    self.returncode = 0
                return self.returncode
        process, checks = FiniteChild(), Mock()
        with self.mock_supervisor(process, clock, frozen=checks), patch.object(runner, 'emit') as emit:
            # An early successful process exit still does not claim an eight-hour completion.
            self.assertEqual(runner.supervise(config, state, [], 0.), 2)
        reports = [call.kwargs['elapsed_seconds'] for call in emit.call_args_list
                   if call.args[1] == 'training_stats']
        self.assertEqual(reports, [0., 300., 600.])
        self.assertGreaterEqual(checks.call_count, 23)
        self.assertEqual(state['state'], 'failed')
        self.assertIn('before the eight-hour budget', state['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
