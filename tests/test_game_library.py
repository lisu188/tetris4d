from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import gc
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import sparse_tetris_4d_optimized as game
import tetris_game_library as library


def play(seed, placements, annotate=True):
    state = game.Game(seed, 'expanded')
    library.tag(state, phase='unit', source='test')
    rows = []
    for _ in range(placements):
        decision = game.choose_sustained_move(state.board, state.current, 1, 64, state.pieces)
        move = decision.move
        if annotate:
            library.annotate(state, 'teacher', 7, 1)
        rows.append({'piece_number': state.pieces + 1, 'piece': move.piece, 'rotation': move.candidate.rotation,
                     'x': move.candidate.x, 'y': move.candidate.y, 'z': move.candidate.z, 'w': move.candidate.w,
                     'target_before': 1, 'update': 7, 'behavior': 'teacher'})
        state.commit(move)
    return state, rows


class GameLibraryTests(unittest.TestCase):
    def setUp(self):
        game.configure_backend('numpy')
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        library.configure(self.root)

    def tearDown(self):
        library.configure(enabled=False)
        self.directory.cleanup()

    def index(self):
        return [json.loads(line) for line in (self.root / 'index.jsonl').read_text().splitlines()]

    def test_default_is_disabled_under_unittest(self):
        with mock.patch.dict(os.environ, {'TETRIS4D_GAME_LIBRARY': ''}):
            self.assertIsNone(library.default_root())
        with mock.patch.dict(os.environ, {'TETRIS4D_GAME_LIBRARY': 'off'}):
            self.assertIsNone(library.default_root())

    def test_finish_stores_replayable_annotated_game_once(self):
        state, _ = play(4101, 24)
        game_id = library.finish(state, 'piece_limit')
        self.assertIsNotNone(game_id)
        self.assertIsNone(library.finish(state, 'piece_limit'))  # idempotent
        record = library.load(self.root / self.index()[0]['file'])
        self.assertEqual(record.meta['stop_reason'], 'piece_limit')
        self.assertEqual(record.meta['phase'], 'unit')
        self.assertEqual(len(record.moves), 24)
        self.assertTrue(np.all(record.update == 7))
        self.assertEqual(record.meta['behaviors'], ['teacher'])
        self.assertTrue(np.all(record.moves[:, 6] == 1))
        self.assertIsNone(record.remaining_placements())
        boards = [s.board for s, _, _ in library.replay(record)]
        self.assertEqual(len(boards), 24)
        # A deterministic repeat is indexed as a duplicate and not stored again.
        again, _ = play(4101, 24)
        self.assertEqual(library.finish(again, 'piece_limit'), game_id)
        entries = self.index()
        self.assertEqual([e['duplicate'] for e in entries], [False, True])
        self.assertEqual(len(list((self.root / 'games').rglob('*.npz'))), 1)
        # Replays are not re-recorded.
        self.assertEqual(len(self.index()), 2)

    def test_discarded_and_reset_games_are_recorded(self):
        state, _ = play(4102, 5, annotate=False)
        del state
        gc.collect()
        entries = self.index()
        self.assertEqual(entries[-1]['stop_reason'], 'released')
        self.assertEqual(entries[-1]['placements'], 5)
        state, _ = play(4103, 3, annotate=False)
        state.reset()
        self.assertEqual(self.index()[-1]['stop_reason'], 'reset')
        self.assertEqual(self.index()[-1]['phase'], 'unit')

    def test_legacy_import_matches_live_recording(self):
        state, rows = play(4104, 12)
        live = library.finish(state, 'piece_limit')
        legacy = {'schema': 'tetris4d-training-game-v1', 'seed': 4104, 'piece_set': 'expanded', 'number': 3,
                  'phase': 'training', 'stop_reason': 'piece_limit', 'complete': True, 'moves': rows,
                  'run_context': {'algorithm': 'unit-test'}}
        path = self.root / 'legacy.json.gz'
        path.write_bytes(gzip.compress(json.dumps(legacy).encode()))
        self.assertEqual(library.import_legacy(path), live)
        self.assertTrue(self.index()[-1]['duplicate'])


if __name__ == '__main__':
    unittest.main()
