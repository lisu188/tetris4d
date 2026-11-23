"""Exercise fresh-game policy selection, held-out gating, paired analysis and resumption."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_afterstate_td as td
from tools import evaluate_afterstate_td as ev


def report(name, values, cap=64, seeds=(1,3,5,7)):
    rows = {s: {'seed': s, 'placements': v, 'clears': 0, 'reason': 'cap' if v == cap else 'top_out',
                'wall_seconds': 1.} for s,v in zip(seeds, values)}
    return ev.summarize(rows, seeds, name, cap)


class EvaluationTests(unittest.TestCase):
    def test_beta_is_inverse_median_development_dispersion(self):
        self.assertEqual(ev.population_beta([2., 4., 8., 0.]), (3., 1/3.))
        for spreads in ([], [0., 0.], [1., float('nan')], [float('inf')], [-1.]):
            with self.assertRaises(ValueError):
                ev.population_beta(spreads)

    def test_paired_rule_and_top_two_extension(self):
        td_result = report('td', [64,64,45,43])
        frozen = report('frozen', [48,48,40,37])
        ppo = report('ppo', [12,22,18,45])
        self.assertEqual(ev.extension_pair({'ppo':ppo, 'td':td_result, 'frozen':frozen}), ['td','frozen'])
        self.assertEqual(ev.extension_pair({'ppo':ppo, 'frozen':frozen}), [])
        self.assertTrue(ev.paired(td_result, frozen)['passes'])
        self.assertFalse(ev.paired(frozen, td_result)['passes'])
        with self.assertRaises(ValueError):
            ev.paired(td_result, report('invalid', [1,2,3,4], cap=128))
        with self.assertRaises(ValueError):
            ev.paired(td_result, report('invalid', [4,3,2,1], seeds=(7,5,3,1)))
        with self.assertRaises(ValueError):
            ev.development_score(td_result)
        dev = report('td', [1,2,3,4,5,6,7,8], seeds=ev.DEVELOPMENT_SEEDS)
        self.assertEqual(ev.development_score(dev), (4.5,4.5))

    def test_policy_tie_break_and_pooled_clear_rate(self):
        tied = {name: report(name, [64,64,63,63]) for name in ('td','ppo','frozen','actor')}
        self.assertEqual(ev.rank(tied), ['actor','frozen','ppo','td'])
        self.assertEqual(ev.extension_pair(tied), ['actor','frozen'])
        tied['ppo'] = report('ppo', [64,64,64,64])
        self.assertEqual(ev.extension_pair(tied), ['ppo','actor'])
        rows = {1:{'seed':1,'placements':100,'clears':4,'reason':'top_out','wall_seconds':1},
                3:{'seed':3,'placements':300,'clears':2,'reason':'cap','wall_seconds':2}}
        result = ev.summarize(rows, (1,3), 'actor', 300)
        self.assertEqual(result['clears_per_1k'], 15)
        self.assertEqual(result['games'][0]['clears_per_1k'], 40)
        self.assertAlmostEqual(result['games'][1]['clears_per_1k'], 20/3)

    def test_cache_rejects_different_checkpoint_or_corrupt_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'game.json'
            key, row = {'sha256': 'source', 'seed': 17}, {'placements': 19}
            ev.write_cache(path, key, row)
            self.assertEqual(ev.checked_cached(path, key), row)
            with self.assertRaisesRegex(ValueError, 'identity'):
                ev.checked_cached(path, dict(key, sha256='other'))
            saved = ev.read_json(path)
            saved['data']['placements'] += 1
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'digest'):
                ev.checked_cached(path, key)

    def test_cpu_greedy_controls_and_td_use_real_fresh_games_and_deadline(self):
        torch.set_num_threads(1)
        ev.game.configure_backend('numpy')
        ev.game.load_checkpoint(ROOT/'models/trained-sigmoid.json')
        source = ev.neural.create_model(ev.neural.Config(
            architecture='survival-spatial-v4', spatial_actor=True, regional_critic=True,
            policy_mode='teacher-prior', teacher_prior_scale=.125, screening='exhaustive', shortlist=16))
        trained = td.from_v4(source)
        results = {}
        select = ev.select_moves
        def observed(model, states, *args):
            self.assertTrue(all(state.locked_pieces == () for state in states))
            return select(model, states, *args)
        for policy, model in [('actor',source),('frozen',source),('td',trained),('ppo',source)]:
            with self.subTest(policy=policy), patch.object(ev, 'select_moves', side_effect=observed):
                rows = []
                result = ev.evaluate_fresh(model, [11,13], policy=policy, cap=2, device='cpu', on_finish=rows.append)
                self.assertTrue(result['complete'])
                self.assertEqual(result['placements'], [2,2])
                self.assertEqual(result['censored'], 2)
                self.assertEqual(result['seeds'], [r['seed'] for r in rows])
                self.assertTrue(model.training)
                results[policy] = result
        self.assertTrue(all(r['clears_per_1k'] == 0 for r in result['games']))
        self.assertEqual(result['clears_per_1k'], 0)
        self.assertEqual(results['td']['clears'], results['frozen']['clears'])
        self.assertEqual(results['ppo']['clears'], results['actor']['clears'])
        interrupted = ev.evaluate_fresh(trained, [11,13], policy='td', cap=3, device='cpu', deadline=0.)
        self.assertFalse(interrupted['complete'])
        self.assertEqual(interrupted['games'], [])

    def test_cached_development_and_holdout_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.write_text('checkpoint')
            args = SimpleNamespace(source=source, weights=source, td=None, ppo=None, split='development',
                selection=None, cap=2, max_cap=4, cache_dir=root/'cache', output=root/'dev.json')
            budget = SimpleNamespace(expired=lambda: False)
            calls = []
            def run(model, seeds, *, policy, cap, on_finish, **kwargs):
                calls.append((policy, cap, len(seeds)))
                for seed in seeds:
                    on_finish({'seed':seed, 'placements':cap, 'clears':0, 'reason':'cap', 'wall_seconds':.1})
            with patch.object(ev, 'load_model'), patch.object(ev, 'evaluate_fresh', side_effect=run):
                self.assertTrue(ev.evaluate(args, 'cpu', budget, {}))
                self.assertEqual(len(calls), 4)
                self.assertTrue(ev.evaluate(args, 'cpu', budget, {}))
                self.assertEqual(len(calls), 4)
                args.split, args.output = 'holdout', root/'held.json'
                with self.assertRaisesRegex(ValueError, 'Holdout requires'):
                    ev.evaluate(args, 'cpu', budget, {})
                args.td = args.ppo = source
                args.selection = root/'selection.json'
                args.selection.write_text(json.dumps({'complete': True, 'best_sha256': 'wrong'}))
                with self.assertRaisesRegex(ValueError, 'not selected'):
                    ev.evaluate(args, 'cpu', budget, {})
                self.assertFalse(args.output.exists())
                args.selection.write_text(json.dumps({'complete': True, 'best_sha256': ev.base.file_hash(source)}))
                self.assertTrue(ev.evaluate(args, 'cpu', budget, {}))
                held = ev.read_json(args.output)
                self.assertFalse(set(held['seeds']).intersection(ev.DEVELOPMENT_SEEDS))
                self.assertFalse(held['promotion_gate_passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
