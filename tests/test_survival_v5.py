"""Survival V5: exact next-state features, anchoring to the frozen V4 actor, trainer wiring."""
from pathlib import Path
import importlib.util
import os
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game
import tetris_neural as neural
import tetris_survival_v4 as v4
import tetris_survival_v5 as v5
import train_survival_v5 as trainer_v5

_spec = importlib.util.spec_from_file_location('afterstate_ranking', ROOT / 'tools/afterstate_ranking.py')
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def random_boards(count, seed):
    rng = random.Random(seed)
    boards = []
    for _ in range(count):
        top = rng.randint(0, 16)
        boards.append(tuple(rng.getrandbits(rng.randint(0, top)) if rng.random() < .9 else 0 for _ in range(game.AREA)))
    return boards


def v4_regional_model(seed=3):
    torch.manual_seed(seed)
    config = neural.Config(architecture=v4.ARCHITECTURE, spatial_actor=True, regional_critic=True,
                           policy_mode='teacher-prior', teacher_prior_scale=.125, recover_target=True)
    model = neural.create_model(config).eval()
    with torch.no_grad():  # a non-trivial critic, as after calibration
        for parameter in model.value.parameters():
            parameter.add_(torch.randn_like(parameter) * .05)
    return model


def reference_survival(board):
    heights = [mask.bit_length() for mask in board]
    holes = [h - bin(mask).count('1') for mask, h in zip(board, heights)]
    depth = transitions = 0
    for mask, h in zip(board, heights):
        for z in range(h):
            if not mask >> z & 1:
                depth += bin(mask >> (z + 1)).count('1')
        previous = 1
        for z in range(16):
            bit = mask >> z & 1
            transitions += bit != previous
            previous = bit
    def height(x, y, w):
        return heights[game.column_index(x, y, w)] if 0 <= x < 16 and 0 <= y < 16 and 0 <= w < 16 else 16
    wells = []
    diversity = [set(), set(), set()]
    for w in range(16):
        for y in range(16):
            for x in range(16):
                h = height(x, y, w)
                lowest = min(height(x - 1, y, w), height(x + 1, y, w), height(x, y - 1, w),
                             height(x, y + 1, w), height(x, y, w - 1), height(x, y, w + 1))
                wells.append(max(0, lowest - h))
                for axis, (dx, dy, dw) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):
                    if x + dx < 16 and y + dy < 16 and w + dw < 16:
                        difference = height(x + dx, y + dy, w + dw) - h
                        if -2 <= difference <= 2:
                            diversity[axis].add(difference)
    area = game.AREA
    return [sum(holes) / (16 * area), depth / (256 * area), sum(h > 0 for h in holes) / area,
            transitions / (16 * area), sum(d * (d + 1) // 2 for d in wells) / (136 * area), max(wells) / 16,
            sum(h >= 12 for h in heights) / area, sum(h >= 14 for h in heights) / area,
            *(len(values) / 5 for values in diversity)]


class FeatureExactnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def test_pooled_columns_equal_encode_columns_mean_and_max(self):
        boards = torch.tensor(random_boards(6, 1) + [(0,) * game.AREA, (65535,) * game.AREA], dtype=torch.int32)
        targets = torch.tensor([0, 3, 7, 15, 1, 9, 4, 0])
        spatial = neural.encode_columns(boards, targets)
        expected = torch.cat((spatial.mean((2, 3, 4)), spatial.amax((2, 3, 4))), dim=1)
        torch.testing.assert_close(v5.pooled_columns(boards, targets), expected, rtol=0, atol=0)

    def test_state_block_and_initial_critic_equal_the_v4_regional_critic(self):
        source = v4_regional_model()
        model = v5.from_v4(source, residual_bound=2., policy_temperature=.5, afterstate_scale=1.).eval()
        boards = torch.tensor(random_boards(5, 2), dtype=torch.int32)
        targets = torch.tensor([1, 4, 9, 0, 15])
        context = torch.rand((5, neural.CONTEXT))
        with torch.no_grad():
            _, pooled, value, _, _ = source.encode(boards, targets, context)
            _, pooled5, value5, _, _ = model.encode(boards, targets, context)
            scores = model.score(torch.cat((pooled5, context), dim=1))
        torch.testing.assert_close(pooled5[:, :v5.V4_POOLED], pooled, rtol=0, atol=0)
        torch.testing.assert_close(value5, value, rtol=0, atol=0)
        torch.testing.assert_close(scores, value, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in model.base.parameters()))

    def test_survival_features_match_python_reference(self):
        boards = random_boards(3, 4)
        actual = v5.survival_features(torch.tensor(boards, dtype=torch.int32))
        expected = torch.tensor([reference_survival(board) for board in boards], dtype=torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_next_state_rows_match_materialized_afterstates(self):
        states, targets = [], []
        for seed, moves in ((601, 35), (603, 70)):
            state, target = game.Game(seed, 'expanded'), 1
            for _ in range(moves):
                state.commit(game.choose_sustained_move(state.board, state.current, target, 64, state.pieces).move)
            states.append(state)
            targets.append(target)
        clearing = game.Game(605, 'expanded')
        clearing.current, clearing.board = 'I', (0,) + (3,) * (game.AREA - 1)
        states.append(clearing)
        targets.append(0)
        config = neural.Config(architecture=v5.ARCHITECTURE, spatial_actor=True, shortlist=24, screening='exhaustive',
                               protect_target=False, recover_target=False)
        cleared = 0
        for state, target in zip(states, targets):
            prepared = neural.prepare(state, target, config)
            v5.augment_prepared(prepared, state, torch.device('cpu'))
            rows = torch.as_tensor(prepared.arrays['next_features'])
            for index, candidate in enumerate(prepared.options):
                proxy, after_target = tool.afterstate(state, target, candidate)
                cells = game.placed_cells(candidate)
                merged = game.merge_rigid_piece(state.board, cells, candidate.piece)
                full = [z for z in range(16) if all(mask >> z & 1 for mask in merged)]
                moves = torch.tensor([[len(full) / 16, len(full) * sum(c[2] in full for c in cells) / 16,
                                       sum(c[2] for c in cells) / len(cells) / 16,
                                       prepared.arrays['features'][index][8], prepared.arrays['features'][index][13]]])
                expected = v5.assemble(torch.tensor([proxy.board], dtype=torch.int32), torch.tensor([after_target]),
                                       torch.from_numpy(neural.state_context(proxy, after_target))[None], moves)[0]
                with self.subTest(state=state.seed, candidate=index):
                    torch.testing.assert_close(rows[index], expected, rtol=0, atol=1e-7)
                cleared += candidate.layers > 0
        self.assertGreater(cleared, 0)

    def test_teacher_top_keeps_the_best_teacher_candidates(self):
        state = game.Game(611, 'expanded')
        for _ in range(20):
            state.commit(game.choose_sustained_move(state.board, state.current, 1, 64, state.pieces).move)
        common = dict(architecture=v5.ARCHITECTURE, spatial_actor=True, candidates=512, shortlist=16)
        top = neural.prepare(state, 1, neural.Config(**common, shortlist_mode='teacher-top'))
        mixed = neural.prepare(state, 1, neural.Config(**common, shortlist_mode='teacher-random'))
        self.assertEqual(len(top.options), 16)
        self.assertEqual(top.options[top.teacher], mixed.options[mixed.teacher])
        best_half = {c for c, prior in zip(mixed.options, mixed.arrays['teacher_prior']) if prior > -1}
        self.assertTrue(best_half <= set(top.options))


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def batch(self, config, seeds=(621, 623)):
        records = []
        for seed in seeds:
            state = game.Game(seed, 'expanded')
            for _ in range(15):
                state.commit(game.choose_sustained_move(state.board, state.current, 1, 64, state.pieces).move)
            prepared = neural.prepare(state, 1, config)
            records.append(v5.augment_prepared(prepared, state, torch.device('cpu')).arrays)
        return neural.collate(records, torch.device('cpu'))

    def test_zero_bound_replays_the_base_and_the_residual_is_bounded(self):
        source = v4_regional_model()
        for bound, temperature in ((0., 1.), (0., .5), (2., .5)):
            model = v5.from_v4(source, residual_bound=bound, policy_temperature=temperature, afterstate_scale=3.).eval()
            batch = self.batch(model.config)
            with torch.no_grad():
                out = model(batch)
                base = source(batch)['logits']
            valid = batch['valid'].bool()
            scaled = out['logits'][valid] * temperature
            with self.subTest(bound=bound, temperature=temperature):
                if bound == 0:
                    torch.testing.assert_close(out['logits'][valid], base[valid] / temperature, rtol=0, atol=0)
                self.assertTrue(bool(((scaled - base[valid]).abs() <= bound + 1e-5).all()))
                self.assertTrue(bool(torch.isinf(out['logits'][~valid]).all()))


class TrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = v4_regional_model()
        self.source_path = self.directory / 'source.pt'
        neural.save_checkpoint(self.source_path, self.source)

    def trainer(self, *extra):
        args = trainer_v5.arguments([
            '--no-gpu-defaults', '--run-dir', str(self.directory / 'run'), '--device', 'cpu', '--backend', 'numpy',
            '--weights', str(ROOT / 'models/trained-sigmoid.json'), '--hours', '.1', '--threads', '1',
            '--envs', '2', '--rollout', '3', '--batch', '2', '--optimizer-batch', '3', '--gradient-every', '100',
            '--seed', '650000', '--selection-pieces', '3', '--recover-target', '--policy-mode', 'teacher-prior',
            '--teacher-prior-scale', '.125', '--source-checkpoint', str(self.source_path), *extra])
        return trainer_v5.SurvivalTrainer(args, trainer_v5.base.Budget(.1))

    def test_fresh_update_resume_keeps_base_frozen_and_critic_seeded(self):
        trainer = self.trainer()
        model = trainer.model
        for name, value in self.source.value.state_dict().items():
            torch.testing.assert_close(model.value.main.state_dict()[name], value, rtol=0, atol=0)
        optimized = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        self.assertFalse(any(id(p) in optimized for p in model.base.parameters()))
        base_before = {k: v.clone() for k, v in model.base.state_dict().items()}
        self.assertTrue(trainer.calibrate())
        trainer.update(trainer.rollout(False), False)
        for name, value in base_before.items():
            torch.testing.assert_close(model.base.state_dict()[name], value, rtol=0, atol=0)
        trainer.save('test_checkpoint')
        resumed = self.trainer()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(resumed.progress['runner_updates'], 1)

    def test_selection_and_sampled_checks_run(self):
        trainer = self.trainer('--selection-seeds', '910001')
        trainer.calibrate()
        self.assertTrue(trainer.evaluate_policy('neural')['complete'])
        self.assertTrue(trainer.sampled_policy_check()['complete'])


@unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'CUDA probes require TETRIS4D_TEST_CUDA=1')
class CudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        game.configure_backend('native')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def states(self):
        states = [game.Game(631 + 2 * i, 'expanded') for i in range(4)]
        for i, state in enumerate(states):
            for _ in range(10 + 7 * i):
                state.commit(game.choose_sustained_move(state.board, state.current, 1, 64, state.pieces).move)
        clearing = game.Game(641, 'expanded')
        clearing.current, clearing.board = 'I', (0,) + (3,) * (game.AREA - 1)
        return states + [clearing], [1, 1, 1, 1, 0]

    def test_device_rows_and_teacher_top_match_the_cpu_reference(self):
        import tetris_gpu_simulation as gpu
        states, targets = self.states()
        for mode in ('teacher-top', 'teacher-random'):
            config = neural.Config(architecture=v5.ARCHITECTURE, spatial_actor=True, shortlist=32,
                                   screening='exhaustive', shortlist_mode=mode, protect_target=False, recover_target=False)
            result = v5.augment_device_batch(gpu.GpuPreparer(config).prepare_batch(states, targets), states)
            for env, (state, target) in enumerate(zip(states, targets)):
                reference = v5.augment_prepared(neural.prepare(state, target, config), state, torch.device('cpu'))
                actual = result.prepared[env]
                with self.subTest(mode=mode, env=env):
                    self.assertEqual(tuple(actual.options), reference.options)
                    torch.testing.assert_close(actual.arrays['next_features'].cpu(),
                                               torch.as_tensor(reference.arrays['next_features']), rtol=1e-6, atol=1e-6)

    def test_zero_bound_greedy_replays_v4_choices(self):
        import tetris_gpu_simulation as gpu
        source = v4_regional_model().cuda()
        model = v5.from_v4(source, residual_bound=0., policy_temperature=.5, afterstate_scale=1.).eval()
        states, targets = self.states()
        states, targets = states[:4], targets[:4]
        preparer = gpu.GpuPreparer(model.config)
        for _ in range(5):
            result = v5.augment_device_batch(preparer.prepare_batch(states, targets), states)
            with torch.no_grad():
                ours = model(result.batch)['logits'].argmax(-1)
                theirs = source(result.batch)['logits'].argmax(-1)
            torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
            for env, candidate in zip(result.environment_indices, result.selected_candidates(ours)):
                states[env].commit(game.materialize_move(states[env].board, candidate))


if __name__ == '__main__':
    unittest.main()
