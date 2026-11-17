"""Issues #31-#38: V4 training defaults, throughput, selection and experiments."""
from pathlib import Path
import os
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
import train_survival_v4 as v4


def original_validation(board):
    return not any(not isinstance(v, int) or v < 0 or v >= 1 << game.HEIGHT for v in board)


class EngineSpeedupTests(unittest.TestCase):
    def test_fast_column_validation_accepts_exactly_the_original_set(self):
        rng = random.Random(7)
        base = [rng.getrandbits(16) for _ in range(game.AREA)]
        tails = (0, 65535, 65536, -1, 1.0, True, False, np.int64(3), np.uint16(3), '1', None, 2 ** 70)
        for tail in tails:
            for container in (tuple, list):
                board = container(base[:-1] + [tail])
                with self.subTest(tail=repr(tail), container=container.__name__):
                    self.assertEqual(game._valid_columns(board), original_validation(board))

    def test_evaluate_board_results_unchanged(self):
        for backend in ('python', 'native'):
            try:
                game.configure_backend(backend)
            except ValueError:
                continue
            rng = random.Random(11)
            board = tuple(rng.getrandbits(7) for _ in range(game.AREA))
            evaluation = game.evaluate_board.__wrapped__(board)
            heights, holes = game.board_features(board)
            with self.subTest(backend=backend):
                self.assertEqual(evaluation.heights, tuple(heights))
                self.assertTrue(all(type(h) is int for h in evaluation.heights))
                self.assertEqual(evaluation.holes, holes)
                with self.assertRaises(ValueError):
                    game.evaluate_board.__wrapped__(board[:-1] + (1 << 16,))
        game.configure_backend('numpy')


class ArgumentTests(unittest.TestCase):
    def test_exhaustive_screening_is_the_gpu_default_only(self):
        self.assertEqual(v4.arguments(['--no-gpu-defaults']).screening, 'sampled')
        with patch.object(v4, 'gpu_defaults', return_value=['--gpu-prepare']):
            self.assertEqual(v4.arguments([]).screening, 'exhaustive')
            self.assertEqual(v4.arguments(['--screening', 'sampled']).screening, 'sampled')

    def test_selection_defaults_to_eight_lockstep_seeds(self):
        args = v4.arguments(['--no-gpu-defaults'])
        self.assertEqual(args.selection_seeds, list(v4.SELECTION_SEEDS))
        self.assertEqual(len(set(args.selection_seeds)), 8)
        self.assertTrue(args.lockstep_selection)
        self.assertFalse(v4.arguments(['--no-gpu-defaults', '--sequential-selection']).lockstep_selection)
        self.assertEqual(v4.arguments(['--no-gpu-defaults', '--selection-seeds', '910001']).selection_seeds, [910001])

    def test_adaptive_lr_defaults_and_validation(self):
        args = v4.arguments(['--no-gpu-defaults'])
        self.assertTrue(args.adaptive_lr)
        self.assertAlmostEqual(args.adaptive_kl_target, args.target_kl / 2)
        self.assertFalse(v4.arguments(['--no-gpu-defaults', '--fixed-lr']).adaptive_lr)
        with self.assertRaises(ValueError):
            v4.arguments(['--no-gpu-defaults', '--adaptive-kl-target', '1'])


class TrainerTests(unittest.TestCase):
    backend, device = 'numpy', 'cpu'

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        game.configure_backend(cls.backend)
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def arguments(self, *extra):
        return v4.arguments([
            '--no-gpu-defaults', '--run-dir', str(self.directory / 'run'), '--device', self.device,
            '--backend', self.backend, '--weights', str(ROOT / 'models/trained-sigmoid.json'), '--hours', '.1',
            '--threads', '1', '--envs', '2', '--rollout', '3', '--batch', '2', '--optimizer-batch', '3',
            '--gradient-every', '100', '--calibration-games', '0', '--allow-uncalibrated', '--seed', '650000',
            '--calibration-seed', '5500000', '--selection-pieces', '3', *extra])

    def trainer(self, *extra):
        return v4.SurvivalTrainer(self.arguments(*extra), v4.base.Budget(.1))

    def actor_lr(self, trainer):
        return next(g['lr'] for g in trainer.optimizer.param_groups if g['name'] == 'actor')

    def test_learning_rate_moves_towards_the_kl_target_within_bounds(self):
        trainer = self.trainer()
        base, critic = trainer.args.lr, next(g['lr'] for g in trainer.optimizer.param_groups if g['name'] == 'critic')
        trainer.adapt_learning_rate(trainer.args.adaptive_kl_target)
        self.assertEqual(self.actor_lr(trainer), base)
        trainer.adapt_learning_rate(0.)
        self.assertAlmostEqual(self.actor_lr(trainer), base * 1.5)
        for _ in range(40):
            trainer.adapt_learning_rate(0.)
        self.assertAlmostEqual(self.actor_lr(trainer), base * trainer.args.adaptive_lr_range)
        for _ in range(80):
            trainer.adapt_learning_rate(1.)
        self.assertAlmostEqual(self.actor_lr(trainer), base / trainer.args.adaptive_lr_range)
        self.assertEqual(next(g['lr'] for g in trainer.optimizer.param_groups if g['name'] == 'critic'), critic)

    def test_real_update_adapts_and_resume_keeps_the_rate(self):
        trainer = self.trainer()
        self.assertTrue(trainer.calibrate())
        rollout = trainer.rollout(False)
        trainer.update(rollout, False)
        self.assertIsNotNone(trainer._last_rollout_kl)
        adapted = self.actor_lr(trainer)
        self.assertNotEqual(adapted, trainer.args.lr)  # a 3-step rollout barely moves the policy
        trainer.save('test_checkpoint')
        resumed = self.trainer()
        self.assertEqual(self.actor_lr(resumed), adapted)

    def test_fixed_lr_never_changes(self):
        trainer = self.trainer('--fixed-lr')
        trainer.calibrate()
        trainer.update(trainer.rollout(False), False)
        self.assertEqual(self.actor_lr(trainer), trainer.args.lr)


class DangerStartTests(TrainerTests):
    def setUp(self):
        super().setUp()
        import json
        library = v4.base.library
        self.root = self.directory / 'library'
        library.configure(self.root)
        self.addCleanup(library.configure, None, False)
        state = game.Game(4242, 'expanded')
        library.tag(state, phase='training', source='test')
        target = 1
        for _ in range(60):
            decision = game.choose_sustained_move(state.board, state.current, target, 64, state.pieces)
            library.annotate(state, 'teacher', 0, target)
            state.commit(decision.move)
        library.finish(state, 'top_out')
        index = self.root / 'index.jsonl'
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        for row in rows:
            row['game_over'] = True
        index.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        patcher = patch.object(library, 'default_root', return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_pool_offsets_and_unrecorded_copies(self):
        args = self.arguments('--danger-starts', '.5', '--danger-window', '5', '20', '--danger-pool', '3')
        pool = v4.build_danger_pool(args, self.root)
        self.assertEqual(len(pool), 3)
        for start in pool:
            self.assertTrue(5 <= start['remaining_recorded'] <= 20)
            self.assertEqual(start['state'].pieces, start['offset'])
            self.assertTrue(start['state']._library_log.ignore)
            state, target = v4.library_start_state(start['game_id'], start['offset'], self.root)
            self.assertEqual(state.board, start['state'].board)
            self.assertEqual((state.current, target), (start['state'].current, start['target']))

    def test_trainer_starts_restores_and_labels_danger_episodes(self):
        trainer = self.trainer('--danger-starts', '.99', '--danger-window', '5', '20', '--danger-pool', '4')
        starts = [e.start for e in trainer.environments]
        self.assertTrue(any(starts))
        self.assertTrue(trainer.calibrate())
        trainer.update(trainer.rollout(False), False)
        trainer.save('test_checkpoint')
        resumed = self.trainer('--danger-starts', '.99', '--danger-window', '5', '20', '--danger-pool', '4')
        for before, after in zip(trainer.environments, resumed.environments):
            self.assertEqual(after.start, before.start)
            self.assertEqual(after.state.board, before.state.board)
            self.assertEqual(after.state.pieces, before.state.pieces)
        emitted = []
        with patch.object(v4.v2.SurvivalTrainer, 'emit', lambda self, event, **f: emitted.append(f)):
            for episode in resumed.environments:
                resumed.emit('episode_end', episode=episode.number, pieces=episode.state.pieces + 7)
        for episode, event in zip(resumed.environments, emitted):
            if episode.start is None:
                self.assertEqual((event['start_kind'], event['own_placements']), ('fresh', episode.state.pieces + 7))
            else:
                self.assertEqual(event['start_kind'], 'library')
                self.assertEqual(event['start_pieces'], episode.start['offset'])
                self.assertEqual(event['own_placements'], episode.state.pieces + 7 - episode.start['offset'])

    def test_fresh_runs_never_touch_the_library(self):
        trainer = self.trainer()
        self.assertEqual(trainer.danger_pool, [])
        self.assertTrue(all(e.start is None for e in trainer.environments))


@unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'CUDA probes require TETRIS4D_TEST_CUDA=1')
class LockstepSelectionTests(TrainerTests):
    backend, device = 'native', 'cuda'

    def arguments(self, *extra):
        args = super().arguments('--selection-seeds', '910001', '910003', '910005', '--selection-pieces', '60', *extra)
        args.gpu_prepare = True
        return args

    def games(self, trainer, evaluate):
        before = torch.get_rng_state().clone()
        result = evaluate()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        return [(run['pieces'], run['score'], run['stop_reason']) for run in result['runs']], result

    def test_lockstep_matches_sequential_greedy_and_sampled_games(self):
        lockstep, sequential = self.trainer(), self.trainer('--sequential-selection')
        sequential.model.load_state_dict(lockstep.model.state_dict())
        self.assertTrue(lockstep.lockstep_available())
        self.assertFalse(sequential.lockstep_available())
        for name in ('evaluate_policy', 'sampled_policy_check'):
            call = (lambda t: (lambda: t.evaluate_policy('neural'))) if name == 'evaluate_policy' else (
                lambda t: t.sampled_policy_check)
            fast, fast_result = self.games(lockstep, call(lockstep))
            slow, slow_result = self.games(sequential, call(sequential))
            with self.subTest(name=name):
                self.assertEqual(fast, slow)
                self.assertTrue(fast_result['lockstep'])
                self.assertEqual(fast_result['score'], slow_result['score'])


if __name__ == '__main__':
    unittest.main()
