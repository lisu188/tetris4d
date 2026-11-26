"""Interval certificates for the rigid CROSS piece; run with NumPy installed."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sparse_tetris_4d_optimized as game
import tetris_cross_height as cert


class CrossHeightTests(unittest.TestCase):
    def test_no_clear(self):
        first = np.zeros(game.AREA, np.uint16)
        last = first.copy()
        last[37] = 1
        self.assertTrue(cert.cross_survives_interval(first, last, 0, added_cells=1))
        self.assertFalse(cert.cross_survives_interval(last, first, 0))
        self.assertFalse(cert.cross_survives_interval(first, last, 0, added_cells=2))
        self.assertFalse(cert.cross_survives_interval(first, np.full_like(last, 65535), 0))

    def test_unknown_location_with_rigid_placement(self):
        for level, expected in ((0, True), (14, False)):
            with self.subTest(level=level):
                before = np.full(game.AREA, 1 << level, dtype=np.uint16)
                before[0] = (1 << max(0, level - 3)) - 1
                move = game.make_move(tuple(map(int, before)), 'I', 2, 0, 0, 0)
                self.assertEqual(move.layers, 1)
                end = np.asarray(move.board, np.uint16)
                self.assertEqual(cert._has_cross(cert.HEIGHTS[before]), expected)
                self.assertTrue(cert._has_cross(np.minimum(game.HEIGHT, cert.HEIGHTS[end] + 1)))
                for masks in ([1 << level], None):
                    self.assertEqual(cert.cross_survives_interval(before, end, 1, clear_masks=masks,
                                                                  added_cells=4), expected)

    def test_impossible_single_clear(self):
        start = np.zeros(game.AREA, np.uint16)
        start[0] = 3
        self.assertFalse(cert.cross_survives_interval(start, np.zeros_like(start), 1,
                                                      added_cells=game.AREA - 2))

    def test_unobserved_clear_requires_all_feasible_levels(self):
        blank = np.zeros(game.AREA, np.uint16)
        self.assertTrue(cert.cross_survives_interval(blank, blank, 1, clear_masks=[1]))
        self.assertFalse(cert.cross_survives_interval(blank, blank, 1, clear_masks=[1 << 15]))
        self.assertFalse(cert.cross_survives_interval(blank, blank, 1))

    def test_known_multilayer_histories(self):
        blank = np.zeros(game.AREA, np.uint16)
        for masks in ((3,), (1, 1)):
            self.assertTrue(cert.cross_survives_interval(blank, blank, 2, clear_masks=masks))
        self.assertFalse(cert.cross_survives_interval(blank, blank, 2))
        with self.assertRaises(ValueError):
            cert.cross_survives_interval(blank, blank, 1, clear_masks=[3])
        with self.assertRaises(ValueError):
            cert.cross_survives_interval(np.full(game.AREA, -1), blank, 0)

    def test_reverse_envelope_dominates_every_packed_column(self):
        # Check individual, simultaneous, and ordered clears against independent removal
        # of the set rows from each possible 16-bit column. Intermediate additions
        # can only increase each column until the next clearing placement.
        for masks in ((1,), (1 << 14,), (3,), (0x8021,), (1 << 11, 2), (3, 1 << 9)):
            columns = np.arange(1 << game.HEIGHT, dtype=np.uint16)
            initial_heights = cert.HEIGHTS[columns]
            for mask in masks:
                columns |= mask
                keep = [i for i in range(game.HEIGHT) if not (mask >> i) & 1]
                columns = sum(((columns >> z) & 1) << i for i, z in enumerate(keep))
            self.assertTrue((cert._upper(cert.HEIGHTS[columns], masks) >= initial_heights).all())


if __name__ == '__main__':
    unittest.main()
