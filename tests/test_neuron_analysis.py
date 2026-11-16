"""Synthetic CPU checks of the bounded neuron audit; no production models needed."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools import analyze_neurons as audit

neural, game, replay = audit.neural, audit.game, audit.replay


class NeuronAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        game.configure_backend('numpy')

    def setUp(self):
        torch.manual_seed(102)
        self.config = neural.Config(width=8, blocks=1, shortlist=4, candidates=8, chunk=2)
        self.model = neural.ColumnPolicy(self.config).eval()
        self.state = game.Game(94)
        prepared = neural.prepare(self.state, 1, self.config)
        self.batch = neural.collate([prepared.arrays], torch.device('cpu'))

    def test_inventory_and_input_names(self):
        groups = audit.inventory(neural.Config())
        self.assertEqual(sum(group['units'] for group in groups), 562)
        self.assertEqual(sum(group['units'] for group in groups if group['spatial']), 224)
        self.assertEqual(neural.ColumnPolicy().rank1.in_features, 242)
        self.assertEqual(tuple(map(len, audit.input_schema().values())), (34, 82, 96))
        keys = set(neural.ColumnPolicy().state_dict())
        for group in groups:
            for index in range(group['units']):
                description = audit.computation(group['id'], index, neural.Config())
                self.assertTrue(description['formula'])
                self.assertTrue(all(row.split('[')[0] in keys for row in description['checkpoint_tensor_rows']))

    def test_all_policy_modes_and_centering_match_runtime(self):
        formulas = {'neural': 'final_score = raw_actor_score',
                    'teacher-residual': 'final_score = teacher_prior + 0.75 * (2 * sigmoid(raw_actor_score) - 1)',
                    'teacher-prior': 'final_score = 0.125 * teacher_prior + 0.75 * raw_actor_score'}
        with torch.inference_mode():
            for mode in ('neural', 'teacher-residual', 'teacher-prior'):
                for centered in (False, True):
                    self.model.config = replace(self.config, policy_mode=mode,
                                                teacher_prior_scale=.125, residual_scale=.75, centered_critic=centered)
                    description = audit.policy_description(self.model.config)
                    self.assertEqual(description['policy_mode'], mode)
                    self.assertEqual(description['final_score_formula'], formulas[mode])
                    self.assertEqual(description['teacher_prior_formula'], 'teacher_prior = -32 * deficit - rank / 8')
                    parts = audit.forward_parts(self.model, self.batch)
                    prediction = self.model(self.batch)
                    torch.testing.assert_close(audit.policy_logits(self.model, self.batch, parts['score']), prediction['logits'])
                    torch.testing.assert_close(parts['value'], prediction['value'])
                    torch.testing.assert_close(parts['completion'], prediction['completion'].sigmoid())
                    for name in ('rank1', 'rank2', 'score', 'value_hidden', 'value', 'completion_hidden', 'completion'):
                        fast = audit.ablate(self.model, self.batch, parts, name, 0)
                        full = audit.forward_parts(self.model, self.batch, (name, 0))
                        for key in ('score', 'value', 'completion'):
                            torch.testing.assert_close(fast[key], full[key])

    def test_known_rank_unit_changes_top_choice_in_all_modes(self):
        batch = {name: value.clone() for name, value in self.batch.items()}
        for name in ('features', 'indices', 'column_mask', 'valid', 'teacher_prior'):
            batch[name] = batch[name][:, :2].clone()
        batch['features'].zero_()
        batch['features'][0, 1, 0] = 1.
        batch['teacher_prior'][0] = torch.tensor([0., -.1])
        with torch.no_grad():
            for layer in (self.model.rank1, self.model.rank2, self.model.rank_out):
                layer.weight.zero_()
                layer.bias.zero_()
            self.model.rank1.weight[0, -neural.FEATURES] = 8.
            self.model.rank1.bias[0] = -4.
            self.model.rank2.weight[0, 0] = 8.
            self.model.rank2.bias[0] = -4.
            self.model.rank_out.weight[0, 0] = 4.
        with torch.inference_mode():
            for mode in ('neural', 'teacher-residual', 'teacher-prior'):
                self.model.config = replace(self.config, policy_mode=mode)
                before = audit.forward_parts(self.model, batch)
                after = audit.ablate(self.model, batch, before, 'rank2', 0)
                self.assertEqual(int(audit.policy_logits(self.model, batch, before['score']).argmax()), 1)
                effect = audit.intervention_effect(self.model, batch, before, after)
                self.assertEqual(effect['top1_changed_fraction'], 1.)
                self.assertGreater(effect['probability_total_variation'], 0.)
                exact = self.model.rank_out.weight[0, 0] * (.5 - before['traces']['rank2'][..., 0])
                torch.testing.assert_close(after['score'] - before['score'], exact)

    def test_head_interventions_cannot_change_actions(self):
        with torch.inference_mode():
            before = audit.forward_parts(self.model, self.batch)
            for name in ('value_hidden', 'value', 'completion_hidden', 'completion'):
                after = audit.ablate(self.model, self.batch, before, name, 0)
                effect = audit.intervention_effect(self.model, self.batch, before, after)
                self.assertEqual(effect['mean_absolute_score_change'], 0.)
                self.assertEqual(effect['top1_changed_fraction'], 0.)
                self.assertEqual(effect['probability_total_variation'], 0.)
            after = audit.ablate(self.model, self.batch, before, 'value', 0)
            self.assertAlmostEqual(float(after['value'][0]), 0.)
            effect = audit.intervention_effect(self.model, self.batch, before, after)
            self.assertAlmostEqual(effect['signed_value_change'], -float(before['value'][0]))

    def fixture(self, root):
        checkpoint = root / 'synthetic.pt'
        # Checkpoint saving must not initialize a CUDA RNG on a GPU-capable host.
        with patch.object(torch.cuda, 'is_available', return_value=False):
            neural.save_checkpoint(checkpoint, self.model,
                                   extra={'value_semantics': 'expected_remaining_placements_times_0.001'})
        episode = replay.Episode(0, game.Game(94))
        for step in range(2):
            prepared = neural.prepare(episode.state, episode.target, self.config)
            replay.apply_move(episode, prepared.options[prepared.teacher], step, 'synthetic_teacher')
        episode.complete, episode.stop_reason = True, 'piece_limit'
        recorder = replay.Recorder(root / 'games', True, maximum_mib=1)
        recorded = recorder.write(episode)
        return SimpleNamespace(checkpoint=checkpoint, checkpoint_sha256=audit.sha256(checkpoint),
                               games=[recorded], output_dir=root / 'report', limit=2, threads=1, backend='numpy')

    def test_end_to_end_small_audit_with_prose_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory(prefix='tetris4d-neuron-test-') as temporary:
            args = self.fixture(Path(temporary))
            checkpoint_hash, game_hash = audit.sha256(args.checkpoint), audit.sha256(args.games[0])
            with contextlib.redirect_stdout(io.StringIO()):
                result = audit.analyze(args)
            self.assertEqual(len(result['samples']), 2)
            self.assertEqual(len(result['units']), 362)
            self.assertTrue(result['baseline_forward_parity_passed'])
            self.assertTrue(result['recordings'][0]['verified'])
            self.assertTrue(all(row['explanation'] for row in result['units']))
            self.assertTrue(all(len(row['strongest_contexts']) == 2 for row in result['units']))
            self.assertTrue(all('absolute_value_change_in_placements' in row['intervention'] for row in result['units']))
            text = (args.output_dir / 'neurons.md').read_text()
            self.assertIn('## Per-unit explanations', text)
            self.assertIn(result['policy']['final_score_formula'], text)
            self.assertIn(f"Policy mode: `{result['policy']['policy_mode']}`", text)
            for coordinates in result['input_schema'].values():
                for index, meaning in enumerate(coordinates):
                    self.assertIn(f'| {index} | {meaning} |', text)
            self.assertLess(sum(path.stat().st_size for path in args.output_dir.iterdir()), audit.MAX_REPORT_BYTES)
            self.assertEqual(audit.sha256(args.checkpoint), checkpoint_hash)
            self.assertEqual(audit.sha256(args.games[0]), game_hash)
            with self.assertRaisesRegex(ValueError, 'new output'):
                audit.analyze(args)

    def test_reject_hash_and_corrupt_replay(self):
        with tempfile.TemporaryDirectory(prefix='tetris4d-neuron-test-') as temporary:
            args = self.fixture(Path(temporary))
            correct = args.checkpoint_sha256
            args.checkpoint_sha256 = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                audit.analyze(args)
            args.checkpoint_sha256 = correct
            data = json.loads(args.games[0].read_text())
            data['moves'][0]['board_sha256'] = '0' * 64
            args.games[0].write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'Replay mismatch'):
                audit.analyze(args)
            self.assertFalse(args.output_dir.exists())

    def test_incomplete_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory(prefix='tetris4d-neuron-test-') as temporary:
            args = self.fixture(Path(temporary))
            data = json.loads(args.games[0].read_text())
            data['complete'] = False
            args.games[0].write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'complete recording'):
                audit.read_recording(args.games[0])
            args.games[0].write_text('{"bad": NaN}')
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                audit.read_recording(args.games[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
