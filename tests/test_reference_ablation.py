"""Placement budgets, frozen inputs and bounded supervision for the paired pilot."""
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
from tools import run_reference_ablation as pilot


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tetris-ablation-test-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.production = self.root/'production.artifacts'
        self.output = self.root/'reference-ablation.artifacts'
        self.production.mkdir()
        for name in pilot.RUNTIME:
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'runtime')
        self.checkpoint = self.production/'best.pt'
        self.checkpoint.write_bytes(b'protected-actor')
        self.weights = self.root/'teacher.json'
        self.weights.write_text('{}')
        self.ledger = self.production/'ledger.json'
        self.ledger.write_text(json.dumps({'start_epoch': 1000., 'deadline_epoch': 87400.}))
        self.args = types.SimpleNamespace(source=self.source, production_artifacts=self.production,
            output=self.output, seed=350000, calibration_seed=4300000,
            selection_seeds=[970001, 970003, 970005, 970007], budget_ledger=self.ledger,
            checkpoint=self.checkpoint, weights=self.weights, python=Path(sys.executable),
            cache=self.root/'cache', updates=64, beta=.1, arm_hours=3., evaluate=True,
            global_limit_mib=384, action_seed=1701)

    def config(self):
        def git(command, **kwargs):
            return {'branch': 'improve/reference-pilot\n', 'status': '', 'rev-parse': 'a'*40+'\n'}[command[3]]
        with patch.object(pilot.subprocess, 'check_output', side_effect=git), patch.object(pilot.time, 'time', return_value=2000.):
            return pilot.configuration(self.args)

    def test_paired_arms_differ_only_in_output_and_reference_penalty(self):
        config = self.config()
        baseline = pilot.arm_command(config, 'baseline', 3)
        anchored = pilot.arm_command(config, 'anchored', 3)
        self.assertNotIn('--reference-checkpoint', baseline)
        self.assertNotIn('--reference-kl-weight', baseline)
        self.assertEqual(anchored[-4:], ['--reference-checkpoint', config['initializer'], '--reference-kl-weight', '0.1'])
        anchored = anchored[:-4]
        for command in (baseline, anchored):
            command[command.index('--run-dir')+1] = '<arm>'
        self.assertEqual(baseline, anchored)
        self.assertEqual(config['placements_per_arm'], 524288)
        self.assertNotIn('--record-games', baseline)
        self.assertIn('--allow-concurrent', baseline)
        self.assertEqual(config['deadline_epoch'], 87400.)

    def test_holdout_and_overlapping_output_guards(self):
        self.args.selection_seeds.append(960001)
        with self.assertRaisesRegex(ValueError, 'Reserved final'):
            self.config()
        self.args.selection_seeds.pop()
        self.args.output = self.production/'pilot'
        with self.assertRaisesRegex(ValueError, 'separate'):
            self.config()

    def test_frozen_sources_reject_mutation(self):
        config = self.config()
        pilot.frozen(config)
        (self.source/'train_survival_v2.py').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Frozen input'):
            pilot.frozen(config)

    def test_storage_counts_both_experiments_and_private_cache(self):
        config = self.config()
        self.output.mkdir()
        with (self.output/'bounded-sparse-output').open('wb') as stream:
            stream.truncate(353*pilot.MIB)
        with self.assertRaisesRegex(ValueError, 'Combined'):
            pilot.storage(config)
        (self.output/'bounded-sparse-output').unlink()
        self.args.cache.mkdir()
        with (self.args.cache/'bounded-sparse-cache').open('wb') as stream:
            stream.truncate(32*pilot.MIB)
        with self.assertRaisesRegex(ValueError, 'cache limit'):
            pilot.storage(config)

    def test_existing_pilot_cannot_restart_or_resume(self):
        config = self.config()
        self.output.mkdir()
        (self.output/'status.json').write_text('{}')
        with patch.object(pilot.subprocess, 'Popen') as child:
            with self.assertRaisesRegex(ValueError, 'already'):
                pilot.run(config)
            child.assert_not_called()

    def test_exact_arm_budget_and_fresh_calibration_proof(self):
        config = self.config()
        root = self.output/'baseline'
        root.mkdir(parents=True)
        (root/'last.pt').write_bytes(b'final-weights')
        summary = {'reason': 'update_limit', 'runner_updates': 64, 'placements': 524288}
        manifest = {'runner_updates': 64, 'checkpoint_sha256': pilot.proof(root/'last.pt')['sha256'],
                    'calibration_stage': 'passed', 'source_sha256': {'envs': 16, 'rollout': 512,
                    'seed': config['seed'], 'calibration_seed': config['calibration_seed']}}
        calibration = {'passed': True, 'actor_unchanged': True, 'actor_sha256': 'same-initial-actor'}
        for name, value in [('summary', summary), ('manifest', manifest), ('critic-calibration', calibration)]:
            (root/(name+'.json')).write_text(json.dumps(value))
        (root/'metrics.jsonl').write_text(json.dumps({'event': 'run_start', 'resumed': False,
                                                    'initial_checkpoint': config['initializer']})+'\n')
        self.assertEqual(pilot.validate_arm(config, 'baseline')['placements'], 524288)
        summary['placements'] -= 1
        (root/'summary.json').write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'exact placement'):
            pilot.validate_arm(config, 'baseline')

    def test_deadline_sends_one_safe_signal_without_forced_kill(self):
        config = self.config()
        self.output.mkdir()
        class Child:
            pid = 42
            returncode = None
            signals = []
            def poll(self):
                if self.signals:
                    self.returncode = 0
                return self.returncode
            def send_signal(self, number):
                self.signals.append(number)
        process = Child()
        state = {'phases': {}, 'configuration': {}}
        with patch.object(pilot, 'frozen'), patch.object(pilot, 'storage', return_value={}), \
             patch.object(pilot.subprocess, 'Popen', return_value=process), \
             patch.object(pilot.time, 'time', return_value=100.), patch.object(pilot.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'budget'):
                pilot.child(config, state, 'baseline', ['trainer'], 99., [])
        self.assertEqual(process.signals, [signal.SIGINT])
        self.assertEqual(state['phases']['baseline']['state'], 'incomplete')

    def test_evaluation_keeps_modes_truth_audit_and_reserved_seeds_separate(self):
        config = self.config()
        command = pilot.evaluation_command(config)
        self.assertIn('--value-audit', command)
        self.assertEqual(command[command.index('--modes')+1:command.index('--action-seeds')], ['greedy', 'sampled'])
        self.assertNotIn('--record-games', command)
        self.assertFalse(pilot.FINAL_SEEDS.intersection(config['selection_seeds']))

    def test_cuda_worker_budget_and_free_headroom(self):
        total = 8192*pilot.MIB
        cuda = types.SimpleNamespace(is_available=lambda: True,
            get_device_properties=lambda device: types.SimpleNamespace(total_memory=total),
            set_per_process_memory_fraction=Mock(), mem_get_info=Mock(return_value=(4096*pilot.MIB, total)))
        report = pilot.cuda_budget(types.SimpleNamespace(cuda=cuda), 3072*pilot.MIB, 512*pilot.MIB)
        cuda.set_per_process_memory_fraction.assert_called_once_with(.375, 0)
        self.assertEqual(report['gpu_budget_bytes'], 3072*pilot.MIB)
        cuda.mem_get_info.return_value = (3500*pilot.MIB, total)
        with self.assertRaisesRegex(ValueError, 'Insufficient free GPU'):
            pilot.cuda_budget(types.SimpleNamespace(cuda=cuda), 3072*pilot.MIB, 512*pilot.MIB)
        for name in ('baseline', 'anchored'):
            command = pilot.arm_command(self.config(), name, 3)
            self.assertIn('--worker', command)
            self.assertEqual(command[command.index('--gpu-budget-mib')+1], '3072')
        self.assertIn('--worker', pilot.evaluation_command(self.config()))

    def test_configuration_proof_is_checked_before_launch(self):
        config = self.config()
        self.output.mkdir()
        path = self.output/'configuration.json'
        path.write_text(json.dumps(config))
        state = {'configuration': pilot.proof(path), 'phases': {}}
        path.write_text('{}')
        with patch.object(pilot.subprocess, 'Popen') as child:
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                pilot.child(config, state, 'baseline', ['trainer'], 100., [])
            child.assert_not_called()

    def test_worker_sets_source_import_path_and_exact_trainer_arguments(self):
        script = pilot.SOURCE/'train_survival_v2.py'
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace()}), \
             patch.object(pilot, 'cuda_budget', return_value={'gpu_budget_bytes': 3072*pilot.MIB}) as budget, \
             patch('runpy.run_path') as execute, patch.object(sys, 'argv', ['worker']), \
             patch.object(sys, 'path', list(sys.path)), contextlib.redirect_stdout(io.StringIO()):
            pilot.worker(['--script', str(script), '--gpu-budget-mib', '3072',
                          '--gpu-headroom-mib', '512', '--', '--seed', '350000'])
            budget.assert_called_once()
            self.assertEqual(sys.path[0], str(pilot.SOURCE))
            self.assertEqual(sys.argv, [str(script), '--seed', '350000'])
            execute.assert_called_once_with(str(script), run_name='__main__')


if __name__ == '__main__':
    unittest.main(verbosity=2)
