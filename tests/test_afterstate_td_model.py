"""Afterstate TD policy, feature parity and normal checkpoint gameplay."""
import os
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_afterstate_td as td
import tetris_neural as neural
import tetris_survival_v4 as v4
from tools.afterstate_ranking import afterstate

game = neural.game


def regional_source():
    torch.manual_seed(55)
    config = neural.Config(architecture=v4.ARCHITECTURE, spatial_actor=True, regional_critic=True,
                           policy_mode='teacher-prior', teacher_prior_scale=.125, screening='exhaustive',
                           protect_target=False, shortlist=24)
    return neural.create_model(config).eval()


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def test_features_match_materialized_afterstates_including_clears(self):
        source = regional_source()
        model = td.from_v4(source)
        ordinary = game.Game(683, 'expanded')
        clearing = game.Game(685, 'expanded')
        clearing.board, clearing.current = (0,) + (3,) * (game.AREA - 1), 'I'
        saw_clear = False
        for state, target in ((ordinary, 1), (clearing, 0)):
            prepared = neural.prepare(state, target, model.config)
            pool = model.prepare_cpu(prepared, state)
            logits = source(neural.collate([prepared.arrays], torch.device('cpu')))['logits'][0]
            indices = sorted(range(len(logits)), key=lambda i: -float(logits[i].detach()))[:16]
            self.assertEqual(pool.indices.tolist(), [indices])
            self.assertTrue(pool.valid.all())
            proxies, targets = [], []
            for index in indices:
                candidate = prepared.options[index]
                saw_clear |= bool(candidate.layers)
                proxy, next_target = afterstate(state, target, candidate)
                proxies.append(proxy)
                targets.append(next_target)
            context = torch.stack([torch.from_numpy(neural.state_context(p, t)) for p, t in zip(proxies, targets)])
            _, pooled, values, _, _ = source.encode(torch.tensor([p.board for p in proxies], dtype=torch.int32),
                                                     torch.tensor(targets), context)
            torch.testing.assert_close(pool.features[0], torch.cat((pooled, context), dim=1), rtol=0, atol=0)
            torch.testing.assert_close(model.score(pool.features)[0], values)
            self.assertEqual(model.choose(pool).item(), int(values.argmax()))
        self.assertTrue(saw_clear)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        model.score(pool.features).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.value.parameters()))

    def test_masked_exploration_and_stable_ties(self):
        model = td.from_v4(regional_source())
        state = game.Game(53, 'expanded')
        prepared = neural.prepare(state, 1, model.config)
        for p in model.base.parameters():
            p.data.zero_()
        prepared.arrays['teacher_prior'][:] = 0
        pool = model.prepare_cpu(prepared, state)
        self.assertEqual(pool.indices.tolist(), [list(range(td.TOP_K))])
        pool.valid[:, 3:] = False
        pool.teacher_prior[:, 1:3] = torch.tensor([-4., -9.])
        for p in model.value.parameters():
            p.data.zero_()
        self.assertEqual(model.choose(pool).item(), 0)
        values = model.score(pool.features)
        priors = (2 * values + .125 * pool.teacher_prior).masked_fill(~pool.valid, -torch.inf)
        actual_rng = torch.Generator().manual_seed(1003)
        reference_rng = torch.Generator().manual_seed(1003)
        for _ in range(32):
            reference = torch.multinomial(priors.softmax(1), 1, generator=reference_rng).item()
            self.assertEqual(model.choose(pool, beta=2., generator=actual_rng).item(), reference)

    def test_actor_regularization_limits_logit_loss_and_uses_one_prior(self):
        model = td.from_v4(regional_source(), actor_bound=.5, value_scale=2.)
        values = torch.tensor([[1., 1.3, 2., 99.], [4., 5., 6., 99.]])
        valid = torch.tensor([[True, True, True, False], [True, True, True, False]])
        actor = torch.tensor([[3., 2.4, 2.35, 88.], [1., .6, -.5, 88.]])
        pool = td.CandidatePool(values[..., None], valid, torch.arange(4).expand_as(valid),
                                torch.tensor([[-16., -2., 0., 0.]]).expand_as(actor), actor)
        # The TD winner lies outside the allowed actor gap on both boards.
        expected = torch.tensor([[2.5, 1.9, 2.35, -torch.inf], [.5, .1, -.5, -torch.inf]])
        torch.testing.assert_close(model.ranking_logits(values, valid, actor), expected)
        self.assertEqual(model.choose(pool, scores=values).tolist(), [0, 0])
        rng = torch.Generator().manual_seed(43)
        ref_rng = torch.Generator().manual_seed(43)
        for _ in range(64):
            actual = model.choose(pool, beta=2., generator=rng, scores=values)
            reference = torch.multinomial(expected.softmax(1), 1, generator=ref_rng).squeeze(1)
            torch.testing.assert_close(actual, reference)
        # A slightly weaker actor candidate can win; adding a constant to every value changes nothing.
        closer = replace(pool, actor_logits=actor.clone())
        closer.actor_logits[0, 2] = 2.75
        self.assertEqual(model.choose(closer, scores=values + 100.).tolist(), [2, 0])
        model.configure_policy(actor_bound=0., value_scale=2.)
        self.assertEqual(model.choose(closer, scores=values).tolist(), [0, 0])
        model.configure_policy()
        self.assertEqual(model.choose(closer, scores=values).tolist(), [2, 2])
        for invalid in (float('nan'), -1., float('inf')):
            with self.assertRaises(ValueError):
                model.configure_policy(actor_bound=invalid)

    def test_actor_regularized_checkpoint_playback_and_value_units(self):
        model = td.from_v4(regional_source(), actor_bound=.5, value_scale=400.)
        state = game.Game(49, 'expanded')
        prepared = neural.prepare(state, 1, model.config)
        pool = model.prepare_cpu(prepared, state)
        values = model.score(pool.features).detach()
        logits = model.ranking_logits(values, pool.valid, pool.actor_logits)
        slot = int(logits[0].argmax())
        idx = pool.indices[0, slot].item()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'anchored-td.pt'
            neural.save_checkpoint(path, model, progress={'updates': 1})
            agent = neural.NeuralAgent(path, 'cpu')
            self.assertEqual(agent.model.actor_bound, .5)
            self.assertEqual(agent.model.config.afterstate_scale, 400.)
            decision = agent.choose(state)
        self.assertEqual(decision.move, game.materialize_move(state.board, prepared.options[idx]))
        self.assertAlmostEqual(decision.search_score, logits[0, slot].item(), places=6)
        self.assertAlmostEqual(agent.get_trace()['afterstate_value'], values[0, slot].item(), places=6)

    def test_checkpoint_agent_commits_greedy_move(self):
        model = td.from_v4(regional_source())
        state = game.Game(49, 'expanded')
        prepared = neural.prepare(state, 1, model.config)
        pool = model.prepare_cpu(prepared, state)
        index = pool.indices[0, model.choose(pool).item()].item()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'td.pt'
            neural.save_checkpoint(path, model, progress={'updates': 1})
            agent = neural.NeuralAgent(path, 'cpu')
            decision = agent.choose(state, candidates=8)  # the GUI normally supplies a finite screening budget
        self.assertEqual(decision.evaluations, prepared.evaluations)
        self.assertFalse(decision.sampled)
        self.assertEqual(decision.move, game.materialize_move(state.board, prepared.options[index]))
        self.assertAlmostEqual(decision.search_score, float(model.score(pool.features).detach().max()), places=6)
        self.assertEqual(agent.get_trace()['afterstate_value'], agent.get_trace()['logit'])
        state.commit(decision.move)
        self.assertEqual(state.pieces, 1)


@unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'CUDA probes require TETRIS4D_TEST_CUDA=1')
class CudaTests(unittest.TestCase):
    def test_cuda_and_cpu_shortlist_features_and_action_match(self):
        from tetris_gpu_simulation import GpuPreparer

        game.configure_backend('native')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
        source = regional_source()
        cpu = td.from_v4(source)
        gpu = td.from_v4(source.cuda())
        states = [game.Game(693, 'expanded'), game.Game(695, 'expanded')]
        states[1].board, states[1].current = (0,) + (3,) * (game.AREA - 1), 'I'
        targets = [1, 0]
        result = GpuPreparer(gpu.config).prepare_batch(states, targets)
        pool = gpu.prepare_device(result, states)
        reference_pools = []
        for row, env in enumerate(result.environment_indices):
            prepared = neural.prepare(states[env], targets[env], cpu.config)
            expected = cpu.prepare_cpu(prepared, states[env])
            reference_pools.append(expected)
            valid = pool.valid[row]
            actual_candidates = [result.prepared[env].options[i] for i in pool.indices[row, valid].tolist()]
            self.assertEqual(actual_candidates, [prepared.options[i] for i in expected.indices[0].tolist()])
            torch.testing.assert_close(pool.features[row, valid].cpu(), expected.features[0], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(pool.teacher_prior[row, valid].cpu(), expected.teacher_prior[0])
            torch.testing.assert_close(pool.actor_logits[row, valid].cpu(), expected.actor_logits[0],
                                       rtol=1e-5, atol=1e-5)
        for bound in (None, 0., .5, 1.):
            cpu.configure_policy(bound, value_scale=400.)
            gpu.configure_policy(bound, value_scale=400.)
            self.assertEqual(gpu.choose(pool).tolist(), [cpu.choose(p).item() for p in reference_pools])


if __name__ == '__main__':
    unittest.main()
