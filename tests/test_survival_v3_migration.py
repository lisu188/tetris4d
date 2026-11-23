from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_survival_v3 as training
from tools.migrate_survival_v3 import migrate, _record


def same_tree(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            same_tree(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            same_tree(test, a, b)
    else:
        test.assertEqual(left, right)


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        training.game.configure_backend('numpy')
        training.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.parent = self.root / 'parent'
        self.destination = self.root / 'destination'
        with contextlib.redirect_stdout(io.StringIO()), training.base.run_lock(self.parent):
            self.trainer = training.SurvivalTrainer(self.args(self.parent, 16), training.base.Budget(.1))
            self.trainer.calibrate()
            self.trainer.update(self.trainer.rollout(False), False)
            self.trainer.save('migration_test_parent')
        shutil.copyfile(self.parent / 'last.pt', self.parent / 'best.pt')
        common = {'signature': self.trainer.signature, 'complete': True, 'runs': [], 'score': [3, 3, 3]}
        best = {**common, 'checkpoint_sha256': _record(self.parent / 'best.pt')['sha256'],
                'runner_updates': self.trainer.progress['runner_updates'], 'beats_sustained_on_selection': False}
        (self.parent / 'best.json').write_text(json.dumps(best))
        (self.parent / 'sustained-baseline.json').write_text(json.dumps(common))
        (self.parent / 'selection-00000001.json').write_text(json.dumps(best))
        self.before = {path.name: _record(path) for path in self.parent.iterdir() if path.is_file()}

    def args(self, path, envs):
        return training.arguments(['--run-dir', str(path), '--device', 'cpu', '--backend', 'numpy',
                                   '--weights', str(ROOT / 'models/trained-sigmoid.json'), '--threads', '1',
                                   '--envs', str(envs), '--rollout', '2', '--batch', '4', '--optimizer-batch', '8',
                                   '--ppo-epochs', '1', '--seed', '650000', '--calibration-seed', '5500000',
                                   '--calibration-games', '0', '--allow-uncalibrated', '--selection-seeds', '980101',
                                   '--selection-pieces', '3', '--gradient-clip-scope', 'separate'])

    def test_only_environment_signature_and_active_snapshot_list_change(self):
        proof = migrate(self.parent, self.destination)
        for name in ('last.pt', 'best.pt'):
            source = torch.load(self.parent / name, weights_only=True)
            actual = torch.load(self.destination / name, weights_only=True)
            expected = copy.deepcopy(source)
            expected['extra']['signature']['envs'] = 4
            expected['extra']['environments'] = expected['extra']['environments'][:4]
            same_tree(self, expected, actual)
            self.assertGreater(len(actual['optimizer']['state']), 0)
        self.assertEqual(proof['signature_after'], {**proof['signature_before'], 'envs': 4})
        for name, record in self.before.items():
            self.assertEqual(_record(self.parent / name), record)
        self.assertFalse((self.destination / 'previous.pt').exists())
        self.assertFalse((self.destination / 'games').exists())
        self.assertEqual((self.parent / 'initialization.json').read_bytes(),
                         (self.destination / 'initialization.json').read_bytes())
        self.assertEqual(json.loads((self.destination / 'migration.json').read_text()), proof)

    def test_resume_retains_rng_optimizer_episode_identity_and_best_metadata(self):
        source = torch.load(self.parent / 'last.pt', weights_only=True)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Run settings or source changed'):
            training.SurvivalTrainer(self.args(self.parent, 4), training.base.Budget(.1))
        migrate(self.parent, self.destination)
        torch.set_rng_state(source['rng_cpu'])
        expected_draw = torch.rand(8)
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = training.SurvivalTrainer(self.args(self.destination, 4), training.base.Budget(.1))
        self.assertTrue(torch.equal(torch.rand(8), expected_draw))
        self.assertEqual(len(resumed.environments), 4)
        self.assertEqual(resumed.progress, source['progress'])
        self.assertEqual(resumed.next_episode, source['extra']['next_episode'])
        self.assertEqual(resumed.last_evaluated_update, source['extra']['last_evaluated_update'])
        same_tree(self, resumed.calibration, source['extra']['calibration'])
        same_tree(self, resumed.optimizer.state_dict(), source['optimizer'])
        same_tree(self, resumed.model.state_dict(), source['model'])
        for index, episode in enumerate(resumed.environments):
            self.assertEqual(episode.snapshot(), source['extra']['environments'][index])
        next_episode = resumed.new_episode()
        self.assertEqual(next_episode.number, source['extra']['next_episode'])
        self.assertEqual(next_episode.state.seed, 2 * (resumed.args.seed + source['extra']['next_episode']))
        self.assertIsNotNone(resumed.best_record)
        self.assertIsNotNone(resumed.baseline)
        best = json.loads((self.destination / 'best.json').read_text())
        self.assertEqual(best['checkpoint_sha256'], _record(self.destination / 'best.pt')['sha256'])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Run settings or source changed'):
            training.SurvivalTrainer(self.args(self.destination, 8), training.base.Budget(.1))

    def test_twelve_parked_games_remain_recoverable_without_false_episode_end(self):
        before_metrics = (self.parent / 'metrics.jsonl').read_bytes()
        proof = migrate(self.parent, self.destination)
        parked = proof['parked_environments']
        self.assertEqual(parked['checkpoint'], _record(self.parent / 'last.pt'))
        self.assertEqual(parked['indices'], list(range(4, 16)))
        source = torch.load(Path(parked['checkpoint']['path']), weights_only=True)
        for index, metadata in zip(parked['indices'], parked['episodes']):
            snapshot = source['extra']['environments'][index]
            episode = training.base.restore_episode(snapshot)
            self.assertEqual(episode.snapshot(), snapshot)
            self.assertEqual(metadata['board_sha256'], snapshot['board_sha256'])
            self.assertFalse(episode.complete)
        metrics = (self.destination / 'metrics.jsonl').read_bytes()
        self.assertTrue(metrics.startswith(before_metrics))
        added = [json.loads(line) for line in metrics[len(before_metrics):].splitlines() if line.strip()]
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]['event'], 'environment_count_migration')
        self.assertEqual(added[0]['completed_episodes_added'], 0)

    def test_stop_before_first_selection_can_resume_without_inventing_a_best(self):
        for name in ('best.pt', 'best.json', 'sustained-baseline.json', 'selection-00000001.json'):
            (self.parent/name).unlink()
        migrate(self.parent, self.destination)
        self.assertFalse((self.destination/'best.pt').exists())
        self.assertIsNone(json.loads((self.destination/'summary.json').read_text())['best_checkpoint'])
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = training.SurvivalTrainer(self.args(self.destination, 4), training.base.Budget(.1))
        self.assertIsNone(resumed.best_record)
        self.assertIsNone(resumed.baseline)
        self.assertEqual(len(resumed.environments), 4)

    def test_existing_destination_locked_source_and_bad_best_hash_are_rejected(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            migrate(self.parent, self.destination)
        self.destination.rmdir()
        with training.base.run_lock(self.parent), self.assertRaisesRegex(ValueError, 'still running'):
            migrate(self.parent, self.destination)
        self.assertFalse(self.destination.exists())
        data = json.loads((self.parent / 'best.json').read_text())
        data['checkpoint_sha256'] = '0' * 64
        (self.parent / 'best.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'hash does not match'):
            migrate(self.parent, self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
