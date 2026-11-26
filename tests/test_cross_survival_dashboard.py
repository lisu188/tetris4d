import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import cross_survival_dashboard as dashboard


class ScreeningProgressTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        assigned = patch.object(dashboard, 'ROOT', self.root)
        assigned.start()
        self.addCleanup(assigned.stop)
        self.seeds = [101, 103]
        self.plan = dict(screening=self.seeds, recipes=['residual'], training_seeds=[72], direct_weights=[],
                         teacher='teacher', runtime={}, batch=32, greedy_enabled={'two_stage': True, 'headless': True},
                         updates=80, placements_per_update=8192, phase_hours={'development': 6})
        self.protocol = dashboard.digest(self.plan)
        self.arm = '72-residual'
        self.player = dict(path='ppo.pt', sha256='residual-hash', mode='greedy')
        incumbent = dict(path='inc.pt', sha256='incumbent-hash', mode='greedy')
        self.write('protocol.json', self.plan)
        self.write('direct-players.json', {'protocol': self.protocol, 'players': {'incumbent': incumbent}})
        self.write('training/72-residual/endpoint.json', dict(protocol=self.protocol, recipe='residual', seed=72,
                   updates=80, placements=655360, player=self.player))
        self.write('screening/incumbent.json', dict(complete=True, cap=0, protocol=self.protocol, requested_seeds=self.seeds,
                   player=incumbent, games=[dict(seed=s, placements=40, reason='top_out') for s in self.seeds]))
        self.key = dict(policy='ppo', checkpoint=self.player['sha256'], weights='teacher', runtime={}, screening='exhaustive',
                        top_k=16, cap=0, greedy_actor='fixed-fp32-v1', batch_size=32, piece_set='expanded',
                        group_exhaustive=True, two_stage=True, headless=True)
        self.folder = Path('evaluation-cache') / dashboard.digest(self.key)

    def write(self, path, value):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def cache(self, seed, partial, row):
        key = {**self.key, ('partial_seed' if partial else 'seed'): seed}
        self.write(self.folder / f'{seed}{".partial" if partial else ""}.json',
                   dict(identity=key, data=row, data_sha256=dashboard.digest(row)))

    def ledger(self, phase, label, seconds=None):
        return dict(phase=phase, label=label, seconds=seconds, start_unix=1, allowance_seconds=10)

    def test_waiting_partial_then_official_dev_score(self):
        row = dashboard.public_view()['ppo_screening'][0]
        self.assertEqual((row['status'], row['requested'], row['mean_placements']), ('Waiting', 2, None))
        row1 = dict(seed=101, placements=30, reason='top_out', decisions={'decisions': 30, 'residual_changes': 2})
        self.cache(101, False, row1)
        self.cache(103, True, dict(seed=103, pieces=12, decisions={'decisions': 12, 'residual_changes': 1}))
        # A different stage can share this checkpoint's cache directory.
        self.cache(205, True, dict(seed=205, pieces=987650))
        label = 'screening/' + self.arm
        self.write('gpu-time.json', [self.ledger('development', label)])
        view = dashboard.public_view()
        self.assertEqual(view['phase'], 'PPO development screening · 72-residual')
        row = view['ppo_screening'][0]
        self.assertEqual((row['status'], row['finished_games'], row['observed_moves'], row['residual_changes']),
                         ('Running', 1, 42, 3))
        self.assertEqual(row['live_range'], [12, 12])
        self.assertIsNone(row['mean_placements'])
        self.assertIsNone(row['change_pct'])
        self.assertNotIn('987650', json.dumps(view))
        second = dict(seed=103, placements=10, reason='top_out', decisions={'decisions': 10})
        self.write('screening/72-residual.json', dict(protocol=self.protocol, cap=0, complete=True, requested_seeds=self.seeds,
                   player=self.player, games=[row1, second], mean=20.0, median=20.0))
        self.write('gpu-time.json', [self.ledger('development', label, seconds=10), self.ledger('development', 'qualification/private')])
        self.write('qualification/private.json', {'sealed': 'DO NOT READ'})
        requests = []
        reader = dashboard.read
        def record_read(path, *args):
            requests.append(path)
            return reader(path, *args)
        with patch.object(dashboard, 'read', side_effect=record_read):
            view = dashboard.public_view()
        self.assertEqual(view['phase'], 'Sealed qualification')
        row = view['ppo_screening'][0]
        self.assertEqual((row['status'], row['finished_games'], row['natural_games'], row['active_snapshots']),
                         ('Complete', 2, 2, 0))
        self.assertEqual((row['mean_placements'], row['median_placements'], row['change_pct']), (20, 20, -50))
        self.assertEqual(view['recipe_scores'], [dict(recipe='residual', mean=20, median=20)])
        self.assertFalse(any('qualification' in path.parts or '205' in path.name for path in requests))
        self.assertNotIn('DO NOT READ', json.dumps(view))

    def test_nonphysical_finishes_do_not_score(self):
        for reason, expected in [('target_or_no_move', 'Controller stop'), ('cap', 'Capped')]:
            with self.subTest(reason=reason):
                games = [dict(seed=101, placements=30, reason='top_out'), dict(seed=103, placements=10, reason=reason)]
                self.write('screening/72-residual.json', dict(protocol=self.protocol, cap=0, complete=True,
                           requested_seeds=self.seeds, player=self.player, games=games, mean=20, median=20))
                view = dashboard.public_view()
                row = view['ppo_screening'][0]
                self.assertEqual((row['status'], row['finished_games'], row['natural_games']), (expected, 2, 1))
                self.assertEqual((row['mean_placements'], row['median_placements'], row['change_pct']), (None, None, None))
                self.assertEqual(view['recipe_scores'], [])

    def test_recipe_scores_require_every_training_seed(self):
        self.plan['training_seeds'].append(73)
        self.seeds.append(105)
        protocol = dashboard.digest(self.plan)
        self.write('protocol.json', self.plan)
        for name in ['direct-players.json', 'training/72-residual/endpoint.json']:
            self.write(name, dict(json.loads((self.root / name).read_text()), protocol=protocol))
        incumbent = json.loads((self.root / 'screening/incumbent.json').read_text())
        incumbent['games'].append(dict(seed=105, placements=40, reason='top_out'))
        self.write('screening/incumbent.json', dict(incumbent, protocol=protocol, requested_seeds=self.seeds))
        def write_screen(training_seed, placements):
            games = [dict(seed=s, placements=p, reason='top_out') for s, p in zip(self.seeds, placements)]
            self.write(f'screening/{training_seed}-residual.json', dict(protocol=protocol, cap=0, complete=True,
                       requested_seeds=self.seeds, player=self.player, games=games))
        write_screen(72, [1, 4, 100])
        self.assertEqual(dashboard.public_view()['recipe_scores'], [])
        self.write('training/73-residual/endpoint.json', dict(protocol=protocol, recipe='residual', seed=73,
                   updates=80, placements=655360, player=self.player))
        write_screen(73, [10, 19, 52])
        self.assertEqual(dashboard.public_view()['recipe_scores'], [dict(recipe='residual', mean=31, median=11.5)])

    def test_mismatched_endpoint_and_mismatched_cache_are_rejected(self):
        self.cache(101, True, dict(seed=101, pieces=1))
        f = self.root / self.folder / '101.partial.json'
        row = json.loads(f.read_text())
        row['data']['pieces'] = 99
        self.write(self.folder / f.name, row)
        self.write('gpu-time.json', [self.ledger('development', 'screening/72-residual')])
        with self.assertRaisesRegex(ValueError, 'Mismatched screening cache'):
            dashboard.public_view()
        self.write('training/72-residual/endpoint.json', {'protocol': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'Mismatched PPO endpoint'):
            dashboard.public_view()
