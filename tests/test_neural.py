from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
import tetris_neural as n

game = n.game


def train_args(root, **values):
    fields = dict(device='cpu', precision='fp32', output=root / 'model.pt', resume=None, data=root / 'data',
                  seed=123, batch=2, accumulate=2, updates=2, width=8, blocks=1, chunk=8,
                  lr=3e-4, save_every=2)
    fields.update(values)
    return SimpleNamespace(**fields)


def rl_args(root, **values):
    fields = dict(device='cpu', precision='fp32', output=root / 'rl.pt', resume=None, checkpoint=root / 'model.pt',
                  seed=900, envs=2, rollout=3, pieces=5, gamma=.9995, gae_lambda=.999, entropy=.005,
                  imitation=.05, batch=2, lr=1e-4, updates=1, save_every=1, piece_set='expanded')
    fields.update(values)
    return SimpleNamespace(**fields)


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        game.configure_backend('auto')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def fixture(self, count=3):
        state = game.Game(7)
        config = n.Config(width=8, blocks=1, shortlist=8, candidates=32, chunk=3)
        records, target = [], 1
        for _ in range(count):
            prepared = n.prepare(state, target, config)
            record = dict(prepared.arrays, labels=np.asarray(prepared.teacher), returns=np.asarray(1., dtype=np.float32),
                          value_valid=np.asarray(True), completion=np.zeros(16, dtype=np.float32), aux_valid=np.asarray(True))
            records.append(record)
            target, _, _ = n.commit(state, prepared, prepared.teacher, target)
        return config, records

    def test_encoder_preserves_every_voxel_and_buried_gap(self):
        rng = np.random.default_rng(71)
        board = rng.integers(0, 65536, (2, 4096), dtype=np.int32)
        targets = np.asarray([1, 9])
        result = n.encode_columns(torch.from_numpy(board), torch.from_numpy(targets)).numpy()
        self.assertEqual(result.shape, (2, 34, 16, 16, 16))
        flat = result.reshape(2, 34, 4096)
        for b in range(2):
            for i in range(4096):
                mask = int(board[b, i])
                self.assertEqual(sum(int(flat[b, z, i]) << z for z in range(16)), mask)
                self.assertEqual(flat[b, 32, i], mask.bit_length() / 16)
                self.assertEqual(flat[b, 33, i], mask.bit_length() <= targets[b])
                for z in (0, 1, 9, 15):
                    self.assertEqual(flat[b, 16 + z, i], not(mask & (1 << z)) and z < mask.bit_length())

    def test_observation_does_not_read_hidden_future_bag_order(self):
        a, b = game.Game(7), game.Game(7)
        b.stream.bag.reverse()
        np.testing.assert_array_equal(n.state_context(a, 1), n.state_context(b, 1))
        config = n.Config(width=8, blocks=1, shortlist=8, candidates=32)
        aa, bb = n.prepare(a, 1, config), n.prepare(b, 1, config)
        self.assertEqual(aa.options, bb.options)
        for key in aa.arrays:
            np.testing.assert_array_equal(aa.arrays[key], bb.arrays[key])

    def test_shortlist_contains_teacher_and_only_rigid_safe_actions(self):
        config = n.Config(width=8, blocks=1, shortlist=8, candidates=96)
        state, target = game.Game(31), 1
        for _ in range(32):
            prepared = n.prepare(state, target, config)
            expected = game.choose_sustained_move(state.board, state.current, target, 96, state.pieces)
            self.assertEqual(game.materialize_move(state.board, prepared.options[prepared.teacher]), expected.move)
            self.assertLessEqual(len(prepared.options), config.shortlist)
            for candidate in prepared.options:
                self.assertEqual(game.sustained_candidate_metrics(state.board, candidate, target)[1], 0)
                move = game.materialize_move(state.board, candidate)
                reference = game.reference_drop(state.board, candidate.piece, candidate.rotation, candidate.x, candidate.y, candidate.w)
                self.assertEqual((move.board, move.layers, candidate.z), reference)
            target, _, _ = n.commit(state, prepared, prepared.teacher, target)

    def test_complete_hyperlayer_commit_preserves_mass(self):
        state = game.Game(7)
        state.board = (0,) + (2,) * (game.AREA - 1)
        state.current = 'I'
        prepared = n.prepare(state, 1, n.Config(shortlist=8, candidates=64))
        move = game.materialize_move(state.board, prepared.options[prepared.teacher])
        self.assertEqual(move.layers, 1)
        self.assertEqual(sum(m.bit_count() for m in move.board), 3)
        self.assertEqual(n._clear_mask(state, move), 2)

    def test_full_network_gradients_and_optimizer_update(self):
        config, records = self.fixture()
        model = n.ColumnPolicy(config)
        self.assertFalse(any(isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU)) for m in model.modules()))
        batch = n.collate(records, torch.device('cpu'))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.rank_out.weight.detach().clone()
        loss, _ = n.supervised_loss(model(batch), batch)
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(model.rank_out.weight, before))

    def test_padded_candidates_cannot_win_and_chunking_agrees(self):
        config, rows = self.fixture(2)
        rows[1] = {key: value[:2] if key in ('features', 'indices', 'column_mask', 'valid') else value for key, value in rows[1].items()}
        rows[1]['labels'] = np.asarray(0)
        batch = n.collate(rows, torch.device('cpu'))
        torch.manual_seed(7)
        model = n.ColumnPolicy(config).eval()
        single = n.ColumnPolicy(n.Config(**{**asdict(config), 'chunk': 128})).eval()
        single.load_state_dict(model.state_dict())
        with torch.inference_mode():
            a, b = model(batch), single(batch)
        torch.testing.assert_close(a['logits'], b['logits'], atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isneginf(a['logits'][1, 2:]).all())
        self.assertLess(int(a['logits'][1].argmax()), 2)
        rows[1]['valid'][:] = False
        with self.assertRaises(ValueError):
            n.collate(rows, torch.device('cpu'))

    def test_checkpoint_roundtrip_and_schema_rejection(self):
        config, records = self.fixture()
        model = n.ColumnPolicy(config)
        batch = n.collate(records, torch.device('cpu'))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.pt'
            n.save_checkpoint(path, model, progress={'updates': 1})
            restored, saved = n.load_checkpoint(path, torch.device('cpu'))
            with torch.no_grad():
                torch.testing.assert_close(model(batch)['logits'], restored(batch)['logits'], atol=0, rtol=0)
            saved['schema'] = 'wrong'
            torch.save(saved, path)
            with self.assertRaises(ValueError):
                n.load_checkpoint(path, torch.device('cpu'))

    def test_untrained_checkpoint_cannot_silently_control_game(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'new.pt'
            n.save_checkpoint(path, n.ColumnPolicy(n.Config(width=8, blocks=1)))
            with self.assertRaises(ValueError):
                n.NeuralAgent(path, 'cpu')

    def test_actual_inference_validates_native_commit_and_trace(self):
        config, records = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.pt'
            n.save_checkpoint(path, n.ColumnPolicy(config), progress={'updates': 1})
            agent = n.NeuralAgent(path, 'cpu')
            state = game.Game(7)
            decision = agent.choose(state, 1)
            self.assertEqual(decision.policy, 'neural')
            state.commit(decision.move)
            self.assertEqual(state.cells_placed, sum(m.bit_count() for m in state.board))
            for key in ('activation', 'hidden1', 'hidden2', 'completion'):
                self.assertTrue(np.all((agent.last_trace[key] >= 0) & (agent.last_trace[key] <= 1)))
            self.assertEqual(agent.last_trace['activation'].shape, (config.width, 16, 16, 16))

    def test_completion_labels_track_original_levels_through_compaction(self):
        labels = n.completion_labels(10, [(9, 1), (11, 2), (12, 2), (20, 1)], 4)
        np.testing.assert_array_equal(np.flatnonzero(labels), [1, 2])

    def test_gae_bootstraps_truncation_but_not_terminal_or_reset(self):
        rewards = np.asarray([[1., 1.], [100., 100.]])
        values = np.zeros_like(rewards)
        next_values = np.asarray([[10., 10.], [0., 0.]])
        terminated = np.asarray([[True, False], [True, True]])
        truncated = np.asarray([[False, True], [False, False]])
        advantages, returns = n.gae(rewards, values, next_values, terminated, truncated, .9, 1.)
        self.assertEqual(returns[0, 0], 1.)
        self.assertEqual(returns[0, 1], 10.)
        self.assertEqual(returns[1, 0], 100.)

    def test_dataset_training_and_resume_match_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            demo = SimpleNamespace(output=root / 'data', games=1, validation_games=1, pieces=8, seed=100,
                                   stride=1, shard_size=4, shortlist=8, candidates=32, chunk=8,
                                   gamma=.9995, aux_horizon=4, unrestricted=False, piece_set='expanded')
            n.collect(demo)
            dataset = n.Dataset(root / 'data')
            self.assertEqual(len(dataset), 8)
            n.train(train_args(root, updates=4, output=root / 'full.pt'))
            n.train(train_args(root, updates=2, output=root / 'part.pt'))
            n.train(train_args(root, updates=2, output=root / 'part.pt', resume=root / 'part.pt'))
            full, _ = n.load_checkpoint(root / 'full.pt', torch.device('cpu'))
            part, _ = n.load_checkpoint(root / 'part.pt', torch.device('cpu'))
            for key in full.state_dict():
                torch.testing.assert_close(full.state_dict()[key], part.state_dict()[key], atol=0, rtol=0)
            with self.assertRaises(ValueError):
                n.train(train_args(root, resume=root / 'part.pt', batch=3))
            n.train(train_args(root, updates=1))
            result = n.finetune(rl_args(root))
            self.assertEqual(result['env_steps'], 6)
            result = n.finetune(rl_args(root, resume=root / 'rl.pt'))
            self.assertEqual(result['env_steps'], 12)
            self.assertEqual(result['rl_updates'], 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA device not present; covered by user-run CUDA smoke test')
    def test_cuda_forward_backward_and_cpu_inference_tolerance(self):
        config, records = self.fixture(2)
        cpu = n.ColumnPolicy(config)
        gpu = n.ColumnPolicy(config).cuda()
        gpu.load_state_dict(cpu.state_dict())
        a = cpu(n.collate(records, torch.device('cpu')))
        batch = n.collate(records, torch.device('cuda'))
        b = gpu(batch)
        torch.testing.assert_close(a['logits'], b['logits'].cpu(), atol=2e-4, rtol=2e-4)
        n.supervised_loss(b, batch)[0].backward()
        torch.cuda.synchronize()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in gpu.parameters()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NeuralTests))
    report = {'passed': result.wasSuccessful(), 'tests': result.testsRun, 'skipped': len(result.skipped),
              'torch': str(torch.__version__), 'cuda_available': torch.cuda.is_available(), 'engine_backend': game.BACKEND,
              'note': 'Smoke training and correctness checks do not establish playing-strength improvement.'}
    if args.output:
        n.atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(not result.wasSuccessful())
