from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from pathlib import Path

import torch

import sparse_tetris_4d_optimized as game
import tetris_neural as neural
import train_survival_v4 as v4
from tetris_survival_v4 import ARCHITECTURE


def placements(piece):
    return sum((game.WIDTH - s.extents[0] + 1) * (game.DEPTH - s.extents[1] + 1) * (game.FOURTH - s.extents[3] + 1)
               for s in game.SHAPES[piece])


class ScreeningConfigTests(unittest.TestCase):
    def test_budget_and_validation(self):
        self.assertEqual(neural.Config().screening, 'sampled')
        self.assertEqual(neural.Config(candidates=300).screening_budget, 300)
        self.assertEqual(neural.Config(screening='exhaustive').screening_budget, 0)
        with self.assertRaisesRegex(ValueError, 'screening'):
            neural.Config(screening='everything')

    def test_zero_budget_enumerates_every_placement_once(self):
        for piece in game.PIECE_NAMES:
            specs = list(game.specifications(piece, 0, 17))
            with self.subTest(piece=piece):
                self.assertEqual(len(specs), placements(piece))
                self.assertEqual(len(set(specs)), len(specs))

    def test_old_checkpoints_default_to_sampled(self):
        payload = neural.asdict(neural.Config())
        del payload['screening']
        self.assertEqual(neural.Config(**payload).screening, 'sampled')


class TrainerScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        game.configure_backend('python')

    def test_cli_flag_reaches_signature_and_defaults_to_sampled(self):
        self.assertEqual(v4.arguments(['--no-gpu-defaults']).screening, 'sampled')
        self.assertEqual(v4.arguments(['--no-gpu-defaults', '--screening', 'exhaustive']).screening, 'exhaustive')

    def trainer(self, **options):
        args = v4.arguments(['--no-gpu-defaults', '--device', 'cpu'])
        for name, value in options.items():
            setattr(args, name, value)
        trainer = v4.SurvivalTrainer.__new__(v4.SurvivalTrainer)
        trainer.args, trainer.device = args, torch.device('cpu')
        return trainer

    def test_sampled_pretrained_actor_continues_under_exhaustive_screening(self):
        sampled = self.trainer(pretrained=None).create_fresh_model()
        self.assertEqual(sampled.config.screening, 'sampled')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'pretrained.pt'
            neural.save_checkpoint(path, sampled)
            model = self.trainer(pretrained=path, screening='exhaustive').create_fresh_model()
            self.assertEqual(model.config.screening, 'exhaustive')
            self.assertEqual(model.config.architecture, ARCHITECTURE)
            for name, value in sampled.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                self.trainer(pretrained=path, shortlist_mode='diverse').create_fresh_model()

    def test_selection_baseline_keeps_sampled_budget(self):
        # The sustained heuristic baseline reads config.candidates, never screening_budget.
        source = Path(v4.v2.__file__).read_text()
        self.assertIn('self.model.config.candidates, episode.state.pieces', source)
        self.assertEqual(neural.Config(screening='exhaustive').candidates, 2048)


if __name__ == '__main__':
    unittest.main()
