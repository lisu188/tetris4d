"""Frozen curriculum arms, complete-game reuse and paired promotion decisions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_spatial_curriculum as run

ev = run.ev


def completed(seeds, placements):
    rows = {seed: {'seed': seed, 'placements': n, 'clears': 0, 'reason': 'top_out', 'wall_seconds': 1}
            for seed, n in zip(seeds, placements)}
    return ev.summarize(rows, seeds, 'ppo', run.CAP)


class EvaluationCacheTests(unittest.TestCase):
    def test_bounded_greedy_queue_matches_games_resumed_without_peers(self):
        ev.game.configure_backend('numpy')
        ev.game.load_checkpoint(run.ROOT / 'models/trained-sigmoid.json')
        config = ev.neural.Config(architecture='survival-spatial-v4', spatial_actor=True, shortlist=4, candidates=16)
        model = ev.neural.create_model(config).eval()
        torch.set_num_threads(1)
        def play(seeds):
            result = ev.evaluate_fresh(model, seeds, policy='ppo', cap=3, device='cpu', ppo_batch_size=2)
            self.assertTrue(result['complete'])
            return [(r['seed'], r['placements'], r['reason'], r['clears']) for r in result['games']]
        together = play([17, 19, 23])
        self.assertEqual(together, play([17]) + play([19]) + play([23]))

    def test_reuses_each_complete_seed_across_splits_orders_and_interruption(self):
        stop = SimpleNamespace(expired=lambda: False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'source.pt'
            checkpoint.write_bytes(b'checkpoint')
            sha = ev.base.file_hash(checkpoint)
            requested = []
            def fresh(model, seeds, *, on_finish, **kwargs):
                self.assertEqual(kwargs['ppo_batch_size'], 4)
                requested.append(list(seeds))
                for seed in seeds[:1] if len(requested) == 1 else seeds:
                    on_finish({'seed': seed, 'placements': seed, 'reason': 'top_out', 'clears': 0, 'wall_seconds': .1})
            def compare(seeds, split='development', cap=64):
                return ev.evaluate_checkpoint(checkpoint, sha, 'ppo', seeds, cap, split, 'cpu', stop, {},
                                              'teacher', root/'cache', ppo_batch_size=4, group_exhaustive=True)
            with patch.object(ev, 'load_model', return_value=object()), patch.object(ev, 'evaluate_fresh', side_effect=fresh):
                self.assertFalse(compare([3, 5])['complete'])
                self.assertEqual(compare([5, 7, 3], 'holdout')['placements'], [5, 7, 3])
                self.assertTrue(compare([3, 5])['complete'])
                self.assertTrue(compare([3], cap=128)['complete'])
                self.assertEqual(requested, [[3, 5], [5, 7], [3]])
            path = next((root/'cache').rglob('7.json'))
            saved = ev.read_json(path)
            saved['data']['seed'] = 8
            ev.write_cache(path, saved['identity'], saved['data'])
            with self.assertRaisesRegex(ValueError, 'seed 7'):
                compare([7])

    def test_fixed_shapes_greedy_uses_actor_and_masks_padded_columns(self):
        class Actor:
            config = SimpleNamespace(shortlist=5)
            def actor_logits(self, batch):
                self.shape = tuple(batch['valid'].shape)
                self.values = batch['features']
                return batch['features'][:, :, 0].masked_fill(~batch['valid'], -torch.inf)
        actor = Actor()
        batch = {'boards': torch.tensor([[7], [8]]), 'valid': torch.tensor([[True, True], [True, False]]),
                 'features': torch.tensor([[[-2.], [-1.]], [[-2.], [100.]]])}
        choices = ev._fixed_actor_choices(actor, batch, 4)
        self.assertEqual(actor.shape, (4, 5))
        self.assertEqual(choices.tolist(), [1, 0])
        self.assertTrue(torch.equal(actor.values[0], actor.values[3]))

    def test_single_cpu_board_reuses_only_geometry_in_strict_actor_shapes(self):
        neural = ev.neural
        cases = ((128, 128, 'neural', 0., False),
                 (19, 11, 'teacher-prior', 4., False),
                 (3, 64, 'teacher-residual', 0., True))
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(374)
            for width, chunk, policy, weight, residual in cases:
                with self.subTest(width=width, policy=policy):
                    config = neural.Config(architecture='survival-spatial-v4', spatial_actor=True,
                                           regional_critic=True, shortlist=128, chunk=chunk,
                                           policy_mode=policy, cross_weight=weight, cross_residual=residual)
                    model = neural.create_model(config).eval()
                    if residual:
                        # Use a nonzero learned correction as it would be after training.
                        model.cross_head[-1].weight.normal_()
                    valid = torch.ones(1, width, dtype=torch.bool)
                    valid[:, 1::5] = False
                    columns = torch.rand(1, width, neural.COLUMNS) > .5
                    columns[:, :, 0] = True
                    indices = torch.randint(4096, columns.shape)
                    indices[~columns] = -1
                    batch = dict(boards=torch.randint(65536, (1, 4096), dtype=torch.int32),
                                 targets=torch.tensor([9]), context=torch.randn(1, neural.CONTEXT),
                                 indices=indices, column_mask=columns, valid=valid,
                                 features=torch.randn(1, width, neural.FEATURES),
                                 spatial_delta=torch.randn(1, width, model.spatial_actor.in_features),
                                 teacher_prior=torch.randn(1, width), cross_mobility=torch.rand(1, width, 2))
                    # Raw reference duplicates the board before encoding, as the original evaluator did.
                    full = {}
                    for name, value in batch.items():
                        if name in neural.CANDIDATE_FIELDS:
                            shape = (1, config.shortlist - width, *value.shape[2:])
                            value = torch.cat((value, value.new_zeros(shape)), dim=1)
                        full[name] = value.expand(32, *value.shape[1:]).contiguous()
                    calls = []
                    def capture_state_input(module, inputs):
                        calls.append(inputs[0].clone())
                    hook = model.actor_state.register_forward_pre_hook(capture_state_input)
                    try:
                        expected = model.actor_logits(full, return_base=True)
                        with patch.object(model, 'fixed_inputs', wraps=model.fixed_inputs) as cache:
                            actual = ev._fixed_actor_logits(model, batch, 32, return_base=True)
                            cache.assert_called_once_with(batch, actor_only=True)
                    finally:
                        hook.remove()
                    self.assertEqual(calls[0].shape[0], 32)
                    self.assertTrue(torch.equal(calls[0], calls[1]))
                    for result, reference in zip(actual, expected):
                        self.assertTrue(torch.equal(result, reference[:1, :width]))
                        self.assertEqual(result.argmax().item(), reference[0].argmax().item())

    def test_fixed_actor_keeps_original_path_for_real_peers_training_and_subclasses(self):
        class OtherActor(ev.SurvivalSpatialPolicy):
            pass
        config = ev.neural.Config(architecture='survival-spatial-v4', shortlist=2)
        for cls, rows, size, training, grad in (
                (ev.SurvivalSpatialPolicy, 2, 3, False, False),
                (ev.SurvivalSpatialPolicy, 1, 1, False, False),
                (ev.SurvivalSpatialPolicy, 1, 2, True, False),
                (ev.SurvivalSpatialPolicy, 1, 2, False, True),
                (OtherActor, 1, 2, False, False)):
            with self.subTest(cls=cls.__name__, rows=rows, size=size, training=training, grad=grad):
                model = cls(config).train(training)
                batch = {'valid': torch.ones(rows, 2, dtype=torch.bool)}
                with torch.set_grad_enabled(grad), patch.object(model, 'fixed_inputs') as cache, \
                     patch.object(model, 'actor_logits', side_effect=lambda fixed: fixed['valid'].float()):
                    self.assertEqual(tuple(ev._fixed_actor_logits(model, batch, size).shape), (rows, 2))
                    cache.assert_not_called()


class CurriculumTests(unittest.TestCase):
    def test_completed_arm_resumes_from_its_pinned_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arm = root/'control'
            arm.mkdir()
            source = root/'incumbent.pt'
            source.touch()
            path = arm/'last.pt'
            path.write_bytes(b'endpoint')
            sha = ev.base.file_hash(path)
            plan = {'optimizations': 'combined', 'starting_model_sha256': 'starting-model'}
            (arm/'initialization.json').write_text(json.dumps(dict(imported_model_sha256=run.INCUMBENT_SHA256,
                         fresh_optimizer=True, fresh_ppo_progress=True, model_sha256='starting-model')))
            (arm/'manifest.json').write_text(json.dumps(dict(runner_updates=run.UPDATES, checkpoint_sha256=sha)))
            pool = [{'game_id': str(i), 'offset': j, 'state': SimpleNamespace(seed=i)} for i in range(8) for j in range(4)]
            trainer = SimpleNamespace(danger_pool=pool, progress={'runner_updates': 50, 'placements': 50*8192},
                         run=lambda: {'runner_updates': run.UPDATES, 'placements': run.UPDATES*8192, 'optimizer_steps': 10240})
            stop = SimpleNamespace(expired=lambda: False)
            with patch.object(run, 'MatchedTrainer', return_value=trainer) as build:
                first = run.fit(root, source, 'control', plan, stop)
                self.assertEqual(run.fit(root, source, 'control', plan, stop), first)
                build.assert_called_once()
                path.write_bytes(b'different-endpoint')
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    run.fit(root, source, 'control', plan, stop)

    def test_arm_settings_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'incumbent.pt'
            path.touch()
            controls = [run.trainer_args(root, path, arm, 'combined') for arm in run.ARMS]
        left, right = controls
        self.assertEqual((left.danger_per_game, right.danger_per_game), (4, 1))
        self.assertTrue(left.distinct_danger_seeds and right.distinct_danger_seeds)
        self.assertEqual({k for k in vars(left) if getattr(left, k) != getattr(right, k)}, {'run_dir', 'danger_per_game'})
        self.assertEqual((left.seed, left.envs * left.rollout, left.pieces, left.danger_starts), (445000, 8192, 0, .25))
        self.assertTrue(left.defer_selection and left.group_exhaustive and left.compact_rollout and left.actor_only_kl)
        self.assertEqual(set(run.DEV).intersection(run.HOLDOUT), set())
        self.assertEqual(len(set(left.selection_seeds)), 80)
        tied = completed(run.DEV, [30000]*8)
        self.assertEqual(run.choose({'diverse': tied, 'control': tied}), 'control')
        self.assertEqual(run.choose({'control': tied, 'diverse': completed(run.DEV, [30001]*8)}), 'diverse')
        with self.assertRaises(ValueError):
            run.choose({'control': dict(tied, complete=False), 'diverse': tied})

    def test_selects_endpoints_before_holdout_and_only_promotes_a_passing_policy(self):
        for gain in (1000, 3000):
            with self.subTest(gain=gain), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'incumbent.pt'
                source.touch()
                for name in run.ARMS:
                    (root/name).mkdir()
                plan = {'runtime': {}, 'weights_sha256': 'teacher'}
                calls = []
                def fit(_out, _source, arm, _plan, _stop):
                    return {'checkpoint_sha256': arm + '-sha'}
                def evaluate(path, sha, mode, seeds, cap, split, *args, **kwargs):
                    self.assertEqual((mode, cap, kwargs['ppo_batch_size']), ('ppo', run.CAP, run.EVAL_BATCH))
                    calls.append((sha, split))
                    if split == 'development':
                        self.assertEqual(tuple(seeds), run.DEV)
                        n = 31000 if sha == 'diverse-sha' else 30000
                    else:
                        self.assertTrue((root/'selected.json').is_file())
                        self.assertEqual(tuple(seeds), run.HOLDOUT)
                        n = 30000 + (gain if sha == 'diverse-sha' else 0)
                    return completed(seeds, [n]*len(seeds))
                with patch.object(run, 'fit', side_effect=fit), patch.object(ev, 'evaluate_checkpoint', side_effect=evaluate):
                    stop = SimpleNamespace(expired=lambda: False)
                    self.assertTrue(run.experiment(root, source, plan, 'cpu', stop))
                    self.assertTrue(run.experiment(root, source, plan, 'cpu', stop))
                self.assertEqual(calls, [('control-sha', 'development'), ('diverse-sha', 'development'),
                                         (run.INCUMBENT_SHA256, 'holdout'), ('diverse-sha', 'holdout')])
                result = json.loads((root/'result.json').read_text())
                self.assertEqual(result['paired']['passes'], gain == 3000)
                self.assertEqual(result['recommended_sha256'], 'diverse-sha' if gain == 3000 else run.INCUMBENT_SHA256)
                self.assertEqual(result['unmeasured_interrupted_reservations'], 0)


if __name__ == '__main__':
    unittest.main()
