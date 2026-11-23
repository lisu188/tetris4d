from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import continue_survival_v3 as continuation

runner = continuation.runner


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.parent, self.output = root/'parent', root/'continued'
        self.start = 1_000_000
        self.deadline = self.start + runner.HOURS*3600
        self.now = self.start + 3*3600
        self.argv = ['--parent-output', str(self.parent), '--source', str(self.output/'source'),
                     '--output', str(self.output), '--cache', str(root/'cache')]
        for output in (self.parent, self.output):
            for name in (*runner.RUNTIME, *continuation.EXTRA_RUNTIME):
                path = output/'source'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(output.name)

        def git(command, **kwargs):
            return {'branch': 'feature\n', 'status': '', 'rev-parse': Path(command[2]).parent.name}[command[3]]

        for mock in (patch.object(runner.subprocess, 'check_output', side_effect=git),
                     patch.object(runner.time, 'time', return_value=self.now)):
            mock.start()
            self.addCleanup(mock.stop)
        weights = root/'weights.json'
        weights.write_text('{}')
        options = runner.arguments(['--source', str(self.parent/'source'), '--output', str(self.parent),
                                    '--weights', str(weights), '--cache', str(root/'parent-cache')])
        original = runner.configuration(options)
        control = self.parent/'control'
        runner.write(control/'budget-ledger.json', {'start_epoch': self.start, 'deadline_epoch': self.deadline})
        budget = runner.proof(control/'budget-ledger.json')
        original['files'].append(budget)
        original.update(start_epoch=self.start, deadline_epoch=self.deadline)
        runner.write(control/'configuration.json', original)
        self.status = {'state': 'stopped', 'child_still_running': False,
                       'controller_pid': 999999991, 'child_pid': 999999992,
                       'controller_start_ticks': '123', 'child_start_ticks': '456',
                       'configuration': runner.proof(control/'configuration.json'), 'budget': budget}
        runner.write(control/'status.json', self.status)
        parent_run = self.parent/'run'
        parent_run.mkdir()
        for name in ('last.pt', 'initialization.json', 'manifest.json', 'metrics.jsonl'):
            (parent_run/name).write_text('{}')

    def config(self):
        return continuation.configuration(continuation.arguments(self.argv))

    def test_plan_preserves_original_deadline_and_worker_without_writing(self):
        with patch.object(runner, 'storage', return_value={}), patch.object(continuation, 'migrate') as adopt, \
             patch.object(runner.subprocess, 'Popen') as launch, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(continuation.main(self.argv), 0)
        plan = json.loads(out.getvalue())
        self.assertEqual(plan['remaining_hours'], 5)
        self.assertEqual(plan['configuration']['deadline_epoch'], self.deadline)
        command = plan['command_preview']
        self.assertEqual(command[command.index('--envs')+1], '4')
        self.assertEqual(command[command.index('--hours')+1], '5')
        self.assertIn(str(self.parent/'source/tools/run_survival_v3.py'), command)
        self.assertEqual(command[command.index('--run-dir')+1], str(self.output/'run'))
        self.assertEqual(set(self.output.iterdir()), {self.output/'source'})
        adopt.assert_not_called()
        launch.assert_not_called()

    def test_expired_running_or_changed_parent_is_rejected(self):
        with patch.object(runner.time, 'time', return_value=self.deadline), self.assertRaisesRegex(ValueError, 'expired'):
            self.config()
        with patch.object(runner, 'process_start', return_value='123'), self.assertRaisesRegex(ValueError, 'still running'):
            self.config()
        config = self.config()
        (self.parent/'source/train_survival_v3.py').write_text('modified parent runtime')
        with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
            continuation.run(config)
        self.assertFalse((self.output/'control').exists())

    def test_run_copies_budget_and_supervises_only_remaining_time_once(self):
        config = self.config()

        def adopt(parent, dest):
            self.assertEqual(parent, self.parent/'run')
            self.assertEqual(dest, self.output/'run')
            dest.mkdir()
            for name in ('migration.json', 'initialization.json'):
                (dest/name).write_text('{}')
            return {}

        def supervise(plan, state, requests, start_monotonic):
            self.assertFalse(state['fresh_run'])
            self.assertEqual(state['deadline_epoch'], self.deadline)
            self.assertEqual(state['active_environments'], 4)
            self.assertEqual(start_monotonic, 12000 - (self.now - self.start))
            runner.frozen(plan, state['configuration'])
            return 0

        storage = {'owned_output_bytes': 0, 'checkpoint_reserve_bytes': runner.MIB}
        with patch.object(continuation, 'migrate', side_effect=adopt) as migrated, \
             patch.object(runner.time, 'monotonic', return_value=12000), \
             patch.object(runner, 'storage', return_value=storage), \
             patch.object(runner, 'supervise', side_effect=supervise) as launched, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(continuation.run(config), 0)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                continuation.run(config)
        self.assertEqual((self.parent/'control/budget-ledger.json').read_bytes(),
                         (self.output/'control/budget-ledger.json').read_bytes())
        migrated.assert_called_once()
        launched.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
