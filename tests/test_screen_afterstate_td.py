"""Development game separation and predeclared challenger screening criteria."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import screen_afterstate_td as screen


class TournamentTests(unittest.TestCase):
    def test_positions_use_disjoint_nonreserved_streams_and_replayable_offsets(self):
        rows=[dict(seed=100+i//2,id=f'game-{i}',file=f'{i}.npz',placements=900,
                   game_over=True,piece_set='expanded',stop_reason='top_out') for i in range(24)]
        selected=screen.choose_positions(rows,{'game-1'},{101},8,9)
        self.assertEqual(selected,screen.choose_positions(list(reversed(rows)),{'game-1'},{101},8,9))
        self.assertFalse({r['seed'] for r in selected}.intersection({100,101}))
        self.assertEqual(len({r['seed'] for r in selected}),8)
        self.assertTrue(all(80<=r['recorded_remaining']<=240 and r['offset']+r['recorded_remaining']==900
                            for r in selected))
        manifest=dict(positions=[dict(**r,phase='discovery' if i<4 else 'confirmation')
                                 for i,r in enumerate(selected)],excluded_game_ids=[],excluded_stream_seeds=[])
        screen.validate_splits(manifest)
        manifest['positions'][7]['seed']=manifest['positions'][0]['seed']
        with self.assertRaisesRegex(ValueError,'overlap'):
            screen.validate_splits(manifest)

    def test_discovery_selects_complete_mean_then_median_then_smaller_bound(self):
        params=dict(complete=True,mean=4.,median=3.)
        selected={'bound-1':dict(params),'bound-0.5':dict(params)}
        self.assertEqual(screen.select_challenger(selected),'bound-0.5')
        selected['bound-1']['median']=4.
        self.assertEqual(screen.select_challenger(selected),'bound-1')
        selected['bound-0.5']['mean']=5.
        self.assertEqual(screen.select_challenger(selected),'bound-0.5')
        selected['bound-1']['complete']=False
        with self.assertRaises(ValueError):
            screen.select_challenger(selected)


if __name__=='__main__':
    unittest.main(verbosity=2)
