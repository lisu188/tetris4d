"""Issues #31-#38: V4 training defaults, throughput, selection and experiments."""
from pathlib import Path
import json
import os
import random
import sys
import tempfile
from types import SimpleNamespace
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

    def test_fast_height_validation_accepts_exactly_the_original_set(self):
        def original(heights):
            return not any(not isinstance(h, int) or not 0 <= h <= game.HEIGHT for h in heights)
        rng = random.Random(9)
        base = [rng.randint(0, 16) for _ in range(game.AREA)]
        for tail in (0, 16, 17, -1, 1.0, True, np.int64(3), None, 2 ** 70):
            heights = tuple(base[:-1] + [tail])
            with self.subTest(tail=repr(tail)):
                self.assertEqual(game._valid_heights(heights), original(heights))
        game.configure_backend('numpy')
        heights = tuple(base)
        self.assertEqual(game.pooled_histograms.__wrapped__(heights)[0],
                         tuple(int(heights.count(h)) for h in range(game.HEIGHT + 1)))
        with self.assertRaises(ValueError):
            game.pooled_histograms.__wrapped__(heights[:-1] + (17,))

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

    def test_overlap_prepare_is_opt_in(self):
        self.assertFalse(v4.arguments(['--no-gpu-defaults']).overlap_prepare)
        self.assertTrue(v4.arguments(['--no-gpu-defaults', '--overlap-prepare']).overlap_prepare)

    def test_selection_defaults_to_eight_lockstep_seeds(self):
        args = v4.arguments(['--no-gpu-defaults'])
        self.assertEqual(args.selection_seeds, list(v4.SELECTION_SEEDS))
        self.assertEqual(len(set(args.selection_seeds)), 8)
        self.assertTrue(args.lockstep_selection)
        self.assertFalse(v4.arguments(['--no-gpu-defaults', '--sequential-selection']).lockstep_selection)
        self.assertEqual(v4.arguments(['--no-gpu-defaults', '--selection-seeds', '910001']).selection_seeds, [910001])

    def test_throughput_defaults_keep_8192_placements_per_update(self):
        args = v4.arguments(['--no-gpu-defaults'])
        self.assertEqual((args.envs, args.rollout), (32, 256))
        self.assertEqual(v4.arguments(['--no-gpu-defaults', '--envs', '16', '--rollout', '512']).envs, 16)

    def test_adaptive_lr_defaults_and_validation(self):
        args = v4.arguments(['--no-gpu-defaults'])
        self.assertTrue(args.adaptive_lr)
        self.assertAlmostEqual(args.adaptive_kl_target, args.target_kl / 2)
        self.assertFalse(v4.arguments(['--no-gpu-defaults', '--fixed-lr']).adaptive_lr)
        with self.assertRaises(ValueError):
            v4.arguments(['--no-gpu-defaults', '--adaptive-kl-target', '1'])


class TrainerHarness(unittest.TestCase):
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


class TrainerTests(TrainerHarness):

    def test_uncapped_rollouts_keep_playing_across_old_placement_limit(self):
        trainer = self.trainer('--pieces', '0')
        original = list(trainer.environments)
        for episode in original:
            episode.state.pieces = 31999
        for _ in range(2):
            rollout = trainer.rollout(False)
            self.assertFalse(rollout.terminals.any())
            self.assertFalse(rollout.truncations.any())
        self.assertEqual([e.state.pieces for e in original], [32005, 32005])
        self.assertTrue(all(a is b for a, b in zip(original, trainer.environments)))
        self.assertEqual(trainer._gpu_prepare_active, self.device == 'cuda')

    def test_negative_training_cap_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.arguments('--pieces', '-1')

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


class RegionalCriticTests(TrainerHarness):
    def test_calibrated_pretrained_model_survives_full_start_skip_and_resume(self):
        source = self.trainer('--regional-critic').model
        with torch.no_grad():
            source.value[-1].weight.fill_(.25)
            source.value[-1].bias.fill_(.5)
        path = self.directory / 'calibrated.pt'
        v4.neural.save_checkpoint(path, source, extra={'calibration': {'stage': 'passed', 'done': True}})
        options = ('--regional-critic', '--pretrained', str(path))
        default = self.trainer(*options)
        self.assertEqual(default.model.value[-1].weight.count_nonzero().item(), 0)
        self.assertEqual(default.model.value[-1].bias.count_nonzero().item(), 0)

        options += ('--preserve-pretrained-critic',)
        trainer = self.trainer(*options)
        self.assertTrue(trainer.calibrate())
        self.assertEqual(trainer.calibration['stage'], 'explicitly_skipped')
        self.assertFalse(trainer.optimizer.state)
        self.assertEqual(trainer.progress['updates'], 0)
        self.assertEqual(trainer.model.config, source.config)
        self.assertEqual(v4.v3.model_digest(trainer.model), v4.v3.model_digest(source))
        trainer.update(trainer.rollout(False), False)
        trained_digest = v4.v3.model_digest(trainer.model)
        self.assertNotEqual(trained_digest, v4.v3.model_digest(source))
        trainer.save('test_resume')
        resumed = self.trainer(*options)
        self.assertEqual(v4.v3.model_digest(resumed.model), trained_digest)
        self.assertEqual(resumed.progress['runner_updates'], 1)
        with self.assertRaisesRegex(ValueError, 'Run settings'):
            self.trainer(*options[:-1])

        fresh = self.trainer('--run-dir', str(self.directory / 'next'), '--regional-critic',
                             '--pretrained', str(trainer.last), '--preserve-pretrained-critic')
        self.assertEqual(v4.v3.model_digest(fresh.model), trained_digest)
        self.assertEqual(fresh.progress['updates'], 0)

    def test_preserving_a_critic_requires_matching_calibrated_weights_and_skip(self):
        source = self.trainer().model
        path = self.directory / 'source.pt'
        with self.assertRaises(SystemExit):
            self.arguments('--preserve-pretrained-critic')
        with self.assertRaises(SystemExit):
            self.arguments('--pretrained', str(path), '--preserve-pretrained-critic', '--calibration-games', '2')
        for stage in ('failed', 'explicitly_skipped'):
            v4.neural.save_checkpoint(path, source, extra={'calibration': {'stage': stage, 'done': True}})
            with self.assertRaisesRegex(ValueError, 'calibrated V4'):
                self.trainer('--pretrained', str(path), '--preserve-pretrained-critic')
        v4.neural.save_checkpoint(path, source, extra={'calibration': {'stage': 'passed', 'done': True}})
        with self.assertRaisesRegex(ValueError, 'matching policy'):
            self.trainer('--pretrained', str(path), '--preserve-pretrained-critic', '--regional-critic')

    def test_atomic_replacement_keeps_imported_weights_bound_to_recorded_hash(self):
        model = self.trainer().model
        path, replacement = self.directory / 'best.pt', self.directory / 'new-best.pt'
        calibration = {'calibration': {'stage': 'passed', 'done': True}}
        v4.neural.save_checkpoint(path, model, extra=calibration)
        original_hash = v4.base.file_hash(path)
        original_model = v4.v3.model_digest(model)
        with torch.no_grad():
            model.value[-1].bias.add_(1)
        v4.neural.save_checkpoint(replacement, model, extra=calibration)
        load = v4.neural.load_checkpoint

        def rotate_and_load(stream, device):
            replacement.replace(path)
            return load(stream, device)

        with patch.object(v4.neural, 'load_checkpoint', side_effect=rotate_and_load):
            trainer = self.trainer('--pretrained', str(path), '--preserve-pretrained-critic')
        self.assertNotEqual(v4.base.file_hash(path), original_hash)
        self.assertEqual(trainer.signature['pretrained_sha256'], original_hash)
        self.assertEqual(trainer.imported_checkpoint_sha256, original_hash)
        self.assertEqual(v4.v3.model_digest(trainer.model), original_model)

    def test_regional_statistics_match_python_reference(self):
        from tetris_survival_v4 import regional_statistics
        rng = random.Random(5)
        boards = [tuple(rng.getrandbits(rng.randint(0, 16)) for _ in range(game.AREA)) for _ in range(2)]
        actual = regional_statistics(torch.tensor(boards, dtype=torch.int32)).numpy()
        for row, board in enumerate(boards):
            expected = []
            groups = {}
            for w in range(16):
                for y in range(16):
                    for x in range(16):
                        mask = board[game.column_index(x, y, w)]
                        height = mask.bit_length()
                        groups.setdefault((w // 4, y // 4, x // 4), []).append((height, height - bin(mask).count('1')))
            keys = sorted(groups)
            expected = ([max(h for h, _ in groups[k]) for k in keys] + [np.mean([h for h, _ in groups[k]]) for k in keys]
                        + [np.mean([o for _, o in groups[k]]) for k in keys])
            np.testing.assert_allclose(actual[row], np.asarray(expected) / 16, rtol=1e-6, atol=1e-6)

    def test_actor_unchanged_and_critic_input_extended(self):
        from dataclasses import replace
        from tetris_survival_v4 import REGIONAL_CRITIC_FEATURES
        trainer = self.trainer()
        plain = trainer.model
        regional = v4.neural.create_model(replace(plain.config, regional_critic=True))
        actor = {k: v for k, v in plain.state_dict().items() if not k.startswith('value.')}
        regional.load_state_dict(actor, strict=False)
        states = [game.Game(700 + i, 'expanded') for i in range(2)]
        batch = v4.neural.collate([v4.neural.prepare(s, 1, plain.config).arrays for s in states], torch.device('cpu'))
        with torch.no_grad():
            a, b = plain(batch), regional(batch)
        torch.testing.assert_close(a['logits'], b['logits'], rtol=0, atol=0)
        self.assertEqual(b['critic_input'].shape[1] - a['critic_input'].shape[1], REGIONAL_CRITIC_FEATURES)

    def test_regional_run_updates_and_adopts_a_plain_pretrained_actor(self):
        plain = self.trainer().model
        path = self.directory / 'pretrained.pt'
        v4.neural.save_checkpoint(path, plain)
        trainer = self.trainer('--regional-critic', '--pretrained', str(path))
        self.assertTrue(trainer.model.config.regional_critic)
        for name, value in plain.state_dict().items():
            if not name.startswith('value.'):
                torch.testing.assert_close(trainer.model.state_dict()[name], value, rtol=0, atol=0)
        self.assertTrue(trainer.calibrate())
        trainer.update(trainer.rollout(False), False)
        self.assertEqual(trainer.progress['runner_updates'], 1)


class AfterstateTests(TrainerHarness):
    def test_batched_afterstates_match_the_cpu_reference_exactly(self):
        import importlib.util
        from tetris_survival_v4 import afterstate_values, afterstate_boards, device_afterstate_context, \
            afterstate_piece_features
        spec = importlib.util.spec_from_file_location('afterstate_ranking', ROOT / 'tools/afterstate_ranking.py')
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        model = self.trainer('--regional-critic').model.eval()
        states, targets = [], []
        for seed, moves in ((501, 40), (503, 90)):
            state, target = game.Game(seed, 'expanded'), 1
            for _ in range(moves):
                state.commit(game.choose_sustained_move(state.board, state.current, target, 64, state.pieces).move)
            states.append(state)
            targets.append(target)
        clearing = game.Game(505, 'expanded')
        clearing.current, clearing.board = 'I', (0,) + (3,) * (game.AREA - 1)
        states.append(clearing)
        targets.append(5)
        config = v4.neural.Config(candidates=4096, shortlist=24, protect_target=False, recover_target=False)
        options = [list(v4.neural.prepare(s, t, config).options) for s, t in zip(states, targets)]
        options[-1].append(game.make_move(clearing.board, 'I', next(r for r, sh in enumerate(game.SHAPES['I'])
                                                                    if sh.extents[2] == 4), 0, 0, 0).candidate)
        self.assertTrue(any(c.layers for c in options[-1]))
        expected_values = []
        for state, target, cands in zip(states, targets, options):
            proxies = [tool.afterstate(state, target, c) for c in cands]
            boards = torch.tensor([p.board for p, _ in proxies], dtype=torch.int32)
            cells = [game.placed_cells(c) for c in cands]
            width = max(map(len, cells))
            cell_tensor = torch.zeros((len(cands), width, 4), dtype=torch.int64)
            mask = torch.zeros((len(cands), width), dtype=torch.bool)
            for row, placed in enumerate(cells):
                cell_tensor[row, :len(placed)] = torch.tensor(placed)
                mask[row, :len(placed)] = True
            after = afterstate_boards(torch.tensor([state.board] * len(cands), dtype=torch.int32), cell_tensor, mask)
            torch.testing.assert_close(after, boards, rtol=0, atol=0)
            row, phase = afterstate_piece_features(state)
            context = device_afterstate_context(after, torch.tensor([t for _, t in proxies]),
                                                torch.tensor([row] * len(cands)), torch.tensor([phase] * len(cands)))
            reference = torch.from_numpy(np.stack([v4.neural.state_context(p, t) for p, t in proxies]))
            torch.testing.assert_close(context, reference, rtol=0, atol=0)
            with torch.no_grad():
                expected_values.append(model.encode(boards, torch.tensor([t for _, t in proxies]), reference)[2].float())
        actual = afterstate_values(model, states, targets, options, chunk=5)
        for got, want in zip(actual, expected_values):
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


class DangerStartTests(TrainerHarness):
    def setUp(self):
        super().setUp()
        library = v4.base.library
        self.root = self.directory / 'library'
        library.configure(self.root)
        self.addCleanup(library.configure, None, False)
        self.record(4242)
        patcher = patch.object(library, 'default_root', return_value=self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def record(self, seed, phase='training', salt=0):
        library = v4.base.library
        state = game.Game(seed, 'expanded')
        library.tag(state, phase=phase, source='test')
        target = 1
        for _ in range(60):
            decision = game.choose_sustained_move(state.board, state.current, target, 64, state.pieces + salt)
            library.annotate(state, 'teacher', 0, target)
            state.commit(decision.move)
        library.finish(state, 'top_out')
        index = self.root / 'index.jsonl'
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        for row in rows:
            row['game_over'] = True
        index.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return rows[-1]['id']

    def test_distinct_danger_sources_filter_seed_phase_and_game_id(self):
        self.record(4242, salt=13)
        self.record(4333)
        self.record(4337)
        self.record(4335, phase='selection-sampled')
        excluded = self.record(4336)
        options = ('--distinct-danger-seeds', '--danger-per-game', '1', '--danger-starts', '.5',
                   '--danger-window', '5', '20', '--selection-seeds', '4337')
        args = self.arguments(*options, '--danger-pool', '2')
        pool = v4.build_danger_pool(args, self.root, exclude_game_ids=[excluded], exclude_seeds=[4337])
        self.assertCountEqual([start['state'].seed for start in pool], [4242, 4333])
        trainer = self.trainer(*options, '--danger-pool', '3')
        self.assertCountEqual([start['state'].seed for start in trainer.danger_pool], [4242, 4333, 4336])
        with self.assertRaisesRegex(ValueError, 'Insufficient distinct'):
            self.trainer(*options, '--danger-pool', '4')

    def test_checkpoint_restores_future_start_types_sources_and_offsets(self):
        def schedule(trainer, count):
            draws = []
            for _ in range(count):
                episode = trainer.new_episode()
                v4.base.library.ignore(episode.state)
                draws.append((episode.number, episode.state.seed, episode.start))
            return draws

        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                options = ('--danger-starts', '.5', '--danger-window', '5', '20', '--danger-pool', '4',
                           '--run-dir', str(self.directory / ('legacy' if legacy else 'current')))
                trainer = self.trainer(*options)
                schedule(trainer, 53)
                if legacy:
                    metadata = trainer.checkpoint_metadata()
                    del metadata['danger_rng']
                    with patch.object(trainer, 'checkpoint_metadata', return_value=metadata):
                        trainer.save('test_legacy')
                else:
                    # A restored stream must honor its stored position even if episode count alone differs.
                    trainer._danger_rng.getrandbits(37)
                    trainer.save('test_current')
                want_state = trainer._danger_rng.getstate()
                expected = schedule(trainer, 100)
                self.assertTrue(any(draw[2] is None for draw in expected))
                self.assertGreater(len({draw[2]['offset'] for draw in expected if draw[2]}), 1)
                resumed = self.trainer(*options)
                self.assertEqual(resumed._danger_rng.getstate(), want_state)
                self.assertEqual(schedule(resumed, 100), expected)

    def test_resume_keeps_saved_danger_sources_when_library_grows(self):
        options = ('--danger-starts', '.75', '--danger-window', '5', '20', '--danger-pool', '1',
                   '--distinct-danger-seeds', '--danger-per-game', '1')
        original = self.trainer(*options)
        original.save('before_library_grows')
        start = original.danger_pool[0]
        for seed in range(4400, 4410):
            self.record(seed)
            new_choice = v4.build_danger_pool(original.args, self.root)[0]
            if new_choice['game_id'] != start['game_id']:
                break
        self.assertNotEqual(new_choice['game_id'], start['game_id'])
        resumed = self.trainer(*options)
        recovered = resumed.danger_pool[0]
        for key in ('game_id', 'offset', 'target', 'remaining_recorded'):
            self.assertEqual(recovered[key], start[key])
        self.assertEqual(recovered['state'].board, start['state'].board)
        self.assertEqual(resumed.signature, original.signature)
        for _ in range(10):
            a, b = original.new_episode(), resumed.new_episode()
            self.assertEqual((a.number, a.start, a.state.board), (b.number, b.start, b.state.board))
            v4.base.library.ignore(a.state)
            v4.base.library.ignore(b.state)

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


class CriticSamplingTests(TrainerHarness):
    def test_long_sampled_games_use_terminal_window_and_exclude_censored_targets(self):
        trainer = self.trainer('--selection-seeds', '831', '833', '--selection-pieces', '64000')
        one_option = SimpleNamespace(arrays={}, options=[0])

        def batch(rows):
            return {'valid': torch.ones((rows, 1), dtype=torch.bool)}

        def prepare_batch(states, targets):
            prepared = [None if state.game_over else one_option for state in states]
            live = [i for i, p in enumerate(prepared) if p is not None]
            return SimpleNamespace(prepared=prepared, environment_indices=live, targets=targets,
                                   batch=batch(len(live)), selected_candidates=lambda actions: [0] * len(actions))

        def predict(inputs):
            return {'logits': torch.zeros_like(inputs['valid'], dtype=torch.float32),
                    'value': torch.zeros(len(inputs['valid']))}

        def advance(episode, *_args):
            # Simulate sample boundaries and one intervening observation without expensive game physics.
            state = episode.state
            state.pieces += 127 if state.pieces % 128 else 1
            state.game_over = state.seed == 831 and state.pieces == 51200

        for lockstep in (False, True):
            with (self.subTest(lockstep=lockstep),
                  patch.object(trainer, 'lockstep_available', return_value=lockstep),
                  patch.object(trainer, '_selection_preparer', return_value=SimpleNamespace(prepare_batch=prepare_batch)),
                  patch.object(trainer, 'prepare_episode', lambda e: None if e.state.game_over else one_option),
                  patch.object(trainer.model, 'forward', side_effect=predict),
                  patch.object(v4.neural, 'collate', lambda rows, device: batch(len(rows))),
                  patch.object(v4.base, 'apply_move', advance)):
                result = trainer.sampled_policy_check()
            self.assertTrue(result['complete'])
            finished, capped = result['runs']
            self.assertEqual((finished['pieces'], finished['critic_samples']), (51200, 400))
            self.assertEqual((capped['pieces'], capped['critic_samples']), (64000, 500))
            self.assertTrue(capped['capped'])
            self.assertIsNone(capped['critic_monte_carlo'])
            errors = finished['critic_monte_carlo']
            self.assertAlmostEqual(errors['mean_signed_error_pieces'], -25664.)
            for name, count, mae in [('1-256', 2, 192.), ('257-2048', 14, 1216.), ('2049+', 384, 26688.)]:
                group = errors['by_remaining_placements'][name]
                self.assertEqual(group['samples'], count)
                self.assertAlmostEqual(group['mae_pieces'], mae)


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

    def test_merged_halves_equal_one_full_preparation(self):
        import tetris_gpu_simulation as gpu
        states = [game.Game(8800 + i, 'expanded') for i in range(6)]
        for i, state in enumerate(states):
            state.current = game.PIECE_NAMES[(3 * i) % len(game.PIECE_NAMES)]
        states[4].board = (65535,) * game.AREA  # unpreparable: exercises the partial-batch path
        targets = [1, 1, 1, 1, 15, 1]
        for screening in ('sampled', 'exhaustive'):
            config = v4.neural.Config(architecture=v4.ARCHITECTURE, spatial_actor=True, screening=screening)
            full = gpu.GpuPreparer(config).prepare_batch(states, targets)
            merged = gpu.merge_prepared_batches(gpu.GpuPreparer(config).prepare_batch(states[:3], targets[:3]),
                                                gpu.GpuPreparer(config).prepare_batch(states[3:], targets[3:]))
            with self.subTest(screening=screening):
                self.assertEqual(merged.environment_indices, full.environment_indices)
                self.assertEqual((merged.targets, merged.cpu_fallbacks), (full.targets, full.cpu_fallbacks))
                torch.testing.assert_close(merged.teachers, full.teachers, rtol=0, atol=0)
                valid = full.batch['valid'].bool()
                self.assertEqual(merged.batch.keys(), full.batch.keys())
                for name, value in full.batch.items():
                    other = merged.batch[name]
                    self.assertEqual(other.shape, value.shape, name)
                    if name in gpu.CANDIDATE_FIELDS:
                        torch.testing.assert_close(other[valid], value[valid], rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(other, value, rtol=0, atol=0)
                self.assertEqual(merged.selected_candidates(full.teachers), full.selected_candidates(full.teachers))

    def test_overlapped_preparation_gives_identical_rollouts(self):
        for screening in ('sampled', 'exhaustive'):
            extra = ('--envs', '4', '--rollout', '6', '--screening', screening)
            overlapped, plain = self.trainer(*extra, '--overlap-prepare'), self.trainer(*extra)
            plain.model.load_state_dict(overlapped.model.state_dict())
            results = []
            for trainer in (overlapped, plain):
                trainer.calibrate()
                torch.manual_seed(4321)  # both trainers sample actions from identical RNG streams
                results.append([trainer.rollout(False) for _ in range(2)])
            with self.subTest(screening=screening):
                self.assertGreater(overlapped._gpu_prepare_stats.get('overlapped_steps', 0), 0)
                self.assertNotIn('overlapped_steps', plain._gpu_prepare_stats)
                for fast, slow in zip(*results):
                    self.assertEqual(fast.choices, slow.choices)
                    self.assertEqual(fast.labels, slow.labels)
                    for name in ('returns', 'old_values', 'terminals', 'truncations', 'raw_advantages'):
                        np.testing.assert_array_equal(getattr(fast, name), getattr(slow, name), err_msg=name)
                    for left, right in zip(fast.records, slow.records):
                        for name, value in right.items():
                            torch.testing.assert_close(left[name], value, rtol=0, atol=0)
                self.assertEqual([e.state.board for e in overlapped.environments],
                                 [e.state.board for e in plain.environments])

    def test_afterstate_selection_reports_complete_games(self):
        trainer = self.trainer('--afterstate-selection', '--afterstate-top-k', '4')
        before = torch.get_rng_state().clone()
        result = trainer.afterstate_policy_check()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertTrue(result['complete'])
        self.assertEqual([run['pieces'] for run in result['runs']], [60, 60, 60])
        self.assertEqual(result['top_k'], 4)

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
