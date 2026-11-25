"""The timing gate distinguishes row-level recovery from a CPU rollout."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark_spatial_ppo import validate_preparation


class PreparationGateTests(unittest.TestCase):
    def row(self, batches=257, terminals=0, fallbacks=None):
        return {
            'rollout_placements': 8192,
            'terminal_transitions': terminals, 'truncated_transitions': 0,
            'gpu_prepare': {'enabled': True, 'preparations': {
                'batch_calls': batches, 'environment_calls': 32 * batches,
                'cpu_fallbacks': fallbacks or {}}},
        }

    def test_complete_rollout_and_per_environment_recovery(self):
        validate_preparation(self.row(), 32, 256)
        # Three terminal rows across two steps caused two new batches.
        validate_preparation(self.row(259, 3, {'exhaustive_or_recovery': 3}), 32, 256)
        # A recovered target can continue the same episode.
        validate_preparation(self.row(fallbacks={'buried_target': 1}), 32, 256)

    def test_enabled_flag_alone_does_not_prove_cuda_work(self):
        fallback = self.row()
        fallback['gpu_prepare']['runtime_fallback'] = 'allocation_failed'
        incomplete = self.row()
        incomplete['gpu_prepare']['preparations']['environment_calls'] -= 1
        for row in (fallback, incomplete, self.row(256), self.row(258),
                    self.row(fallbacks={'candidate_budget_bound': 1}),
                    self.row(fallbacks={'buried_target': 8224})):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_preparation(row, 32, 256)
