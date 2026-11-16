from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural

game = neural.game


def scalar_features(board, candidate, target, context, metrics=None):
    shape = game.SHAPES[candidate.piece][candidate.rotation]
    base = game.column_index(candidate.x, candidate.y, candidate.w)
    columns = [base + offset for offset, _, _ in shape.columns]
    delta = sum(candidate.z + mask.bit_length() - context.heights[base + offset]
                for offset, mask, _ in shape.columns)
    gain, _, future = game.sustained_candidate_metrics(board, candidate, target)
    cells = np.zeros((16, 5), dtype=np.float32)
    cells[:len(shape.cells), :4] = np.asarray(shape.cells, dtype=np.float32) / 15
    cells[:len(shape.cells), 4] = 1
    size = game.PIECE_SIZES[candidate.piece]
    fixed = [candidate.x / 15, candidate.y / 15, candidate.z / 15, candidate.w / 15]
    fixed += [v / 16 for v in shape.extents]
    fixed += [gain / 16, future / 64, candidate.layers / 16, delta / 64,
              (delta - size) / 64, (candidate.score - context.score) / 64, size / 16, target / 15]
    return np.concatenate((np.asarray(fixed, dtype=np.float32), cells.ravel())), columns


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        game.configure_backend('numpy')
        game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')

    def compare(self, state, target, config):
        with patch.object(neural, '_option_features', scalar_features):
            expected = neural.prepare(state, target, config)
        actual = neural.prepare(state, target, config)
        self.assertEqual(actual is None, expected is None)
        if actual is None:
            return
        self.assertEqual(actual.options, expected.options)
        self.assertEqual((actual.teacher, actual.evaluations, actual.fallback),
                         (expected.teacher, expected.evaluations, expected.fallback))
        for name in actual.arrays:
            np.testing.assert_array_equal(actual.arrays[name], expected.arrays[name], err_msg=name)

    def test_all_pieces_and_clear_features_match_scalar_contract(self):
        config = neural.Config(candidates=129, shortlist=64)
        for name in game.PIECE_NAMES:
            state = game.Game(314)
            state.current = name
            self.compare(state, 1, config)
        state = game.Game(314)
        state.current = 'I'
        state.board = (0,) + (2,) * (game.AREA - 1)
        self.compare(state, 1, config)

    def test_unprotected_buried_target_preserves_scalar_gap_features(self):
        state = game.Game(171)
        state.board = tuple(2 if i % 7 else 0 for i in range(game.AREA))
        self.compare(state, 0, neural.Config(candidates=129, shortlist=64, protect_target=False))

    def test_cached_shapes_are_read_only_and_outputs_do_not_alias(self):
        shape = game.SHAPES['I'][0]
        cached = neural._shape_features(shape.cells)
        self.assertFalse(cached.flags.writeable)
        state = game.Game(9)
        prepared = neural.prepare(state, 1, neural.Config(candidates=32, shortlist=8))
        before = cached.copy()
        prepared.arrays['features'][:, 16:] = 42
        np.testing.assert_array_equal(cached, before)

    def test_collate_preserves_padding_dtypes_and_rejects_invalid_state(self):
        prepared = neural.prepare(game.Game(9), 1, neural.Config(candidates=32, shortlist=8))
        records = [prepared.arrays, {k: v[:3] if k in ('features', 'indices', 'column_mask', 'valid', 'teacher_prior')
                                   else v for k, v in prepared.arrays.items()}]
        batch = neural.collate(records, torch.device('cpu'))
        for name in records[0]:
            values = []
            for row in records:
                value = np.asarray(row[name])
                if name in ('features', 'indices', 'column_mask', 'valid', 'teacher_prior'):
                    value = np.pad(value, [(0, 8-len(value))] + [(0, 0)] * (value.ndim-1))
                values.append(value)
            self.assertTrue(torch.equal(batch[name], torch.from_numpy(np.stack(values))), name)
        invalid = dict(records[1], valid=np.zeros(3, dtype=np.bool_))
        with self.assertRaisesRegex(ValueError, 'at least one valid'):
            neural.collate([records[0], invalid], torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'Empty batch'):
            neural.collate([], torch.device('cpu'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
