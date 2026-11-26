"""Regression coverage for seed isolation, promotion, and a hard interrupted GPU reservation."""

import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from tools import run_cross_survival as run


class CrossExperiment(unittest.TestCase):
    def test_critic_gate_weights_games_equally_and_checks_both_temperatures(self):
        temperatures = run.calibration.TEMPERATURES

        def row(seed):
            return {
                'seed': seed,
                'reason': 'top_out',
                'mae': {key: {'incumbent': 100.0, 'donor': 90.0} for key in ('early', 'all', 'late')},
            }

        reports = {str(t): [row(s) for s in run.calibration.SEEDS] for t in temperatures}
        self.assertEqual(run.calibration.select(reports)['selected'], 'donor')
        reports['0.25'][0]['mae']['early']['donor'] = 230.0
        self.assertEqual(run.calibration.select(reports)['selected'], 'incumbent')
        reports['0.25'].pop()
        with self.assertRaises(ValueError):
            run.calibration.select(reports)

    def test_reserves_final_hours_and_charges_aborted_process_conservatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = run.base.Budget(float('inf'))
            run.base.durable_json(
                root / 'gpu-time.json', [dict(phase='training', seconds=None, allowance_seconds=720.0)]
            )
            with run.meter.reservation(root, 'development', 'test', stop):
                self.assertFalse(stop.expired())
                self.assertLessEqual(stop.deadline - stop.started, 16 * 3600 - 720)
            rows = run.meter.ledger(root)
            self.assertEqual(rows[0]['seconds'], 720.0)
            self.assertEqual(rows[-1]['status'], 'closed')
            self.assertEqual(run.meter.allowance([dict(seconds=16 * 3600)], 'holdout'), 8 * 3600)

    def test_watchdog_enforces_total_limit_during_unresponsive_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run.base.durable_json(root / 'gpu-time.json', [dict(phase='holdout', seconds=24 * 3600 - 1.5)])
            script = """from tools import cross_run_budget as m
from pathlib import Path
import sys,time
p=Path(sys.argv[1])
with m.exclusive_session(p,m.ev.base.Budget(1)):
    time.sleep(10)
"""
            proc = subprocess.run([sys.executable, '-B', '-c', script, tmp], timeout=15)
            self.assertEqual(proc.returncode, -9)
            with run.meter.exclusive_session(root, run.base.Budget(1)) as available:
                self.assertFalse(available)

    def test_incomplete_training_cannot_promote_an_unmatched_recipe(self):
        players = {'incumbent': run.player('/unused', 'inc'), 'weight-1': run.player('/unused-weight', 'w1')}

        def train(out, incumbent, donor, plan, seed, recipe, stop):
            stop.reason = 'time_budget'
            return None

        def score(out, plan, label, candidate, seeds, split, phase, device, stop):
            self.assertIsNone(stop.reason)
            # The proposed direct player remains eligible and the incumbent wins on qualification.
            value = 100 if label == 'incumbent' else 99
            return dict(complete=True, mean=value, median=value, minimum=value)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run, 'direct_players', return_value=players
        ), patch.object(run, 'train_arm', side_effect=train), patch.object(
            run, 'evaluate', side_effect=score
        ):
            out = Path(tmp)
            self.assertEqual(run.run(out, None, None, {}, 'cpu', run.base.Budget(1)), 'complete')
            result = run.ev.read_json(out / 'result.json')
            self.assertFalse(result['all_workloads_complete'])
            self.assertIn('PPO', result['omitted'])
            self.assertFalse(result['promoted'])

    def test_two_training_seeds_choose_one_primary_endpoint_before_paired_holdout(self):
        players = {
            'incumbent': run.player('/unused-incumbent', 'inc'),
            'weight-1': run.player('/unused-weight', 'w1'),
        }
        calls = []

        def train(out, incumbent, donor, plan, seed, recipe, stop):
            return run.player('/unused', f'{seed}-{recipe}')

        def evaluate(out, plan, label, candidate, seeds, split, phase, device, stop):
            self.assertTrue(set(run.TRAIN).isdisjoint(seeds))
            calls.append((split, label))
            value = 100.0
            if label.endswith('-anchored'):
                value = 111.0
            if label.endswith('-cold'):
                # Single largest endpoint must not win the two-seed recipe gate.
                value = 118.0 if label.startswith(str(run.TRAIN[0])) else 80.0
            if label.endswith('-lookahead'):
                value = 110.0
            vals = [int(value)] * len(seeds)
            return dict(
                complete=True,
                seeds=list(seeds),
                cap=0,
                policy=label,
                placements=vals,
                mean=value,
                median=value,
                bottom_decile=value,
                longest=value,
                minimum=value,
            )

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run, 'direct_players', return_value=players
        ), patch.object(run, 'train_arm', side_effect=train), patch.object(
            run, 'evaluate', side_effect=evaluate
        ):
            out = Path(tmp)
            self.assertEqual(run.run(out, None, None, {}, 'cpu', run.base.Budget(1)), 'complete')
            result = run.ev.read_json(out / 'result.json')
        primary = f'{run.TRAIN[0]}-anchored'
        self.assertEqual(result['qualified_winner'], primary)
        self.assertEqual(len(result['pairs']), 144)
        self.assertTrue(result['promoted'])
        self.assertEqual([label for split, label in calls if split == 'holdout'], ['incumbent', primary])
        self.assertEqual(len(set(run.EVALUATION)), len(run.EVALUATION))

    def test_ppo_screen_timeout_omits_entire_family_and_qualifies_cached_players(self):
        players = {label: run.player('/unused', label) for label in ('incumbent', 'weight-1')}
        first, second = (f'{seed}-plain' for seed in run.TRAIN)
        plan = dict(runtime='fixture', teacher='teacher', greedy_enabled=dict(two_stage=False, headless=False))

        def train(out, incumbent, donor, plan, seed, recipe, stop):
            return run.player('/unused', f'{seed}-{recipe}')

        with tempfile.TemporaryDirectory() as tmp, patch.object(run, 'direct_players', return_value=players), \
                patch.object(run, 'train_arm', side_effect=train), \
                patch.object(run.ev, 'load_model', side_effect=AssertionError('unexpected fresh evaluation')):
            out = Path(tmp)
            # Every development reservation is exhausted. Valid per-seed caches can still be read.
            run.base.durable_json(out / 'gpu-time.json', [dict(phase='development', seconds=16 * 3600)])

            def cache(label, seeds, length):
                actor = label.removesuffix('-lookahead')
                identity = dict(policy='ppo', checkpoint=actor, weights='teacher', runtime='fixture',
                                screening='exhaustive', top_k=run.ev.TOP_K, cap=0,
                                greedy_actor='fixed-fp32-v1', batch_size=run.BATCH, piece_set='expanded',
                                group_exhaustive=True)
                if label.endswith('-lookahead'):
                    identity['preview_search'] = 'one-visible-cross-8-16x4-v1'
                root = out / 'evaluation-cache' / run.ev.digest(identity)
                for seed in seeds:
                    run.ev.write_cache(root / f'{seed}.json', dict(identity, seed=seed),
                                       dict(seed=seed, placements=length, clears=0, reason='top_out'))

            for label, value in (('incumbent', 100), ('weight-1', 110), ('incumbent-lookahead', 105),
                                 ('weight-1-lookahead', 101), (first, 900)):
                cache(label, run.SCREEN, value)
            cache(second, run.SCREEN[:-1], 1000)
            for label, value in (('incumbent', 120), ('weight-1', 110), ('incumbent-lookahead', 115)):
                cache(label, run.QUALIFY, value)
            self.assertEqual(run.run(out, None, None, plan, 'cpu', run.base.Budget(1)), 'complete')
            screened = run.ev.read_json(out / 'screened.json')
            self.assertEqual(screened['omitted'], {'PPO': f'PPO screening incomplete at {second}'})
            self.assertIsNone(screened['recipe'])
            self.assertEqual(screened['recipe_scores'], {})
            self.assertEqual(screened['fast'], 'weight-1')
            selected = run.ev.read_json(out / 'selected.json')
            self.assertEqual(selected['name'], 'incumbent')
            self.assertEqual(set(selected['qualification']), {'incumbent', 'weight-1', 'incumbent-lookahead'})
            self.assertEqual(run.ev.read_json(out / 'qualification/incumbent.json')['seeds'], list(run.QUALIFY))
            self.assertFalse(run.ev.read_json(out / 'result.json')['all_workloads_complete'])
            self.assertFalse((out / 'holdout').exists())

    def test_ppo_screen_signal_or_absolute_deadline_stops_before_qualification(self):
        for reason in ('signal_15', 'time_budget'):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                stop = run.base.Budget(1)
                players = {'incumbent': run.player('/unused', 'incumbent')}
                arm = f'{run.TRAIN[0]}-{run.RECIPES[0]}'

                def score(out, plan, label, candidate, seeds, split, phase, device, stop):
                    if label == 'incumbent':
                        return dict(complete=True, mean=1, median=1)
                    self.assertEqual((label, split), (arm, 'screening'))
                    stop.reason = reason
                    if reason == 'time_budget':
                        stop.deadline = 0
                    return dict(complete=False)

                with patch.object(run, 'direct_players', return_value=players), \
                        patch.object(run, 'train_arm', return_value=run.player('/unused', 'ppo')), \
                        patch.object(run, 'evaluate', side_effect=score):
                    self.assertEqual(run.run(Path(tmp), None, None, {}, 'cpu', stop),
                                     f'PPO screening incomplete at {arm}')
                self.assertEqual(stop.reason, reason)
                self.assertFalse((Path(tmp) / 'budget-skips.json').exists())


if __name__ == '__main__':
    unittest.main()
