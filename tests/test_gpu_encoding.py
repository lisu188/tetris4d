"""Exact optional encoder and mixed CPU/device rollout record contracts."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tetris_neural as neural


class TensorRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def records(self):
        rng = np.random.default_rng(710)
        return [{'boards': rng.integers(0, 65536, 4096, dtype=np.int32),
                 'targets': np.asarray(i, dtype=np.int64),
                 'context': rng.normal(size=neural.CONTEXT).astype(np.float32),
                 'features': rng.normal(size=(n, neural.FEATURES)).astype(np.float32),
                 'indices': rng.integers(0, 4096, (n, neural.COLUMNS), dtype=np.int64),
                 'column_mask': rng.integers(0, 2, (n, neural.COLUMNS)).astype(np.float32),
                 'valid': np.ones(n, dtype=bool),
                 'teacher_prior': rng.normal(size=n).astype(np.float32)}
                for i, n in enumerate((2, 5, 3))]

    def check_collation(self, device):
        records = self.records()
        expected = neural.collate(records, torch.device(device))
        mixed = [{k: torch.as_tensor(v, device=device) for k, v in row.items()}
                 if i % 2 == 0 else row for i, row in enumerate(records)]
        actual = neural.collate(mixed, torch.device(device))
        self.assertEqual(actual.keys(), expected.keys())
        for key in actual:
            self.assertEqual(actual[key].dtype, expected[key].dtype, key)
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
        saved = {k: v.clone() for k, v in actual.items()}
        mixed[0]['features'].zero_()
        for key in saved:
            torch.testing.assert_close(actual[key], saved[key], rtol=0, atol=0)
        mixed[0]['valid'].zero_()
        with self.assertRaisesRegex(ValueError, 'Every state must have at least one valid candidate'):
            neural.collate(mixed, torch.device(device))

    def test_cpu_mixed_records_padding_dtype_and_ownership(self):
        self.check_collation('cpu')

    def test_cpu_fused_request_uses_reference_without_triton_import(self):
        board = torch.arange(4096, dtype=torch.int32).reshape(1, -1)
        target = torch.tensor([7])
        expected = neural.encode_columns(board, target)
        with patch.dict(sys.modules, {'tetris_gpu_encoding': None}):
            actual = neural.encode_columns(board, target, fused=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(neural.ColumnPolicy(neural.Config(width=8, blocks=1)).fused_encoding)

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA_ENCODING') == '1'
                         and torch.cuda.is_available(), 'explicit bounded CUDA encoding probe required')
    def test_cuda_all_masks_targets_mixed_records_model_and_gradients_exact(self):
        self.check_collation('cuda')
        masks = torch.arange(65536, device='cuda', dtype=torch.int32).reshape(16, 4096)
        for targets in (list(range(-1, 15)), [0, 1, 15, 16, 17, -20, 65536, 2**32, 2**32+15,
                                              -2**32, 2**31, -2**31, 7, 8, 9, 10]):
            target = torch.tensor(targets, device='cuda', dtype=torch.int64)
            torch.testing.assert_close(neural.encode_columns(masks, target, fused=True),
                                       neural.encode_columns(masks, target), rtol=0, atol=0)
        records = self.records()[:2]
        batch = neural.collate(records, torch.device('cuda'))
        torch.manual_seed(811)
        model = neural.ColumnPolicy(neural.Config(width=8, blocks=1)).cuda()
        def evaluate(fused):
            model.zero_grad(set_to_none=True)
            model.fused_encoding = fused
            output = model(batch)
            loss = output['value'].sum() + output['completion'].square().mean() + output['logits'].sum()
            loss.backward()
            return {k: v.detach().clone() for k, v in output.items()}, [p.grad.clone() for p in model.parameters()]
        before, ga = evaluate(False)
        after, gb = evaluate(True)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
