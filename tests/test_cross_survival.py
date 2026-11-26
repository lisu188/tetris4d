import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import sparse_tetris_4d_optimized as game
import tetris_cross_survival as cross
import tetris_neural as neural
from tools import evaluate_afterstate_td as ev


def frontier():
    board = [1 << 14] * game.AREA
    center = game.column_index(7, 7, 7)
    for offset in cross.NEIGHBORS:
        board[center + offset] = 1 << 11
    return tuple(board)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        game.configure_backend('numpy')

    def test_exact_absolute_count_including_boundaries_and_bit_thresholds(self):
        board = frontier()
        self.assertEqual(cross.cross_count(game.EMPTY_BOARD), 14**3)
        self.assertEqual(cross.cross_count(board), 1)
        for piece in game.PIECE_NAMES:
            self.assertEqual(cross.visible_piece_fits(board, piece), game.has_legal_move(board, piece))
        for bit, expected in [(12, 1), (13, 0)]:
            changed = list(board)
            changed[game.column_index(7, 7, 7)] = 1 << bit
            self.assertEqual(cross.cross_count(changed), expected)
        rng = random.Random(718)
        board = tuple(1 << rng.randrange(10, 16) for _ in range(4096))
        heights = tuple(x.bit_length() for x in board)
        shape = game.SHAPES['CROSS'][0]
        count = sum(max(heights[game.column_index(x, y, w) + col] - bottom for col, _, bottom in shape.columns)
                    + shape.extents[2] <= 16 for _, x, y, w in game.specifications('CROSS', 0))
        self.assertEqual(cross.cross_count(board), count)

    def test_local_deltas_equal_game_compaction_and_full_recounts(self):
        rng = random.Random(2026)
        board = tuple(1 << rng.randrange(9, 16) for _ in range(4096))
        context = game.evaluate_board(board)
        for piece in ('I', 'NOVA', 'CROSS', 'HYPER'):
            options = [game.score_candidate(context, piece, *spec) for spec in game.specifications(piece, 38)]
            options = [c for c in options if c is not None]
            before, after = cross.candidate_cross_counts(board, options)
            self.assertEqual(before, cross.cross_count(board))
            self.assertEqual(after.tolist(), [cross.cross_count(game.materialize_move(board, c).board) for c in options])
        board, move = self.clear_fixture()
        self.assertEqual(move.layers, 2)
        before, after = cross.candidate_cross_counts(board, (move,))
        self.assertEqual(before, cross.cross_count(board))
        self.assertEqual(int(after[0]), cross.cross_count(game.materialize_move(board, move).board))
        self.assertGreater(after[0], before)

    @staticmethod
    def clear_fixture():
        # Two full layers below a suspended top are eligible only after the vertical I lands.
        board = [3 | 1 << 14] * 4096
        board[game.column_index(7, 7, 7)] = 0
        board = tuple(board)
        return board, game.score_candidate(game.evaluate_board(board), 'I', 2, 7, 7, 7)

    def test_planner_finds_rescue_at_last_index_without_stream_access(self):
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.current, state.next_piece, state.board = 'I', 'CROSS', frontier()
        state.stream = object()  # any hidden draw or bag access raises
        context = game.evaluate_board(state.board)
        kills = game.score_candidate(context, 'I', 2, 7, 7, 7)
        safe = game.score_candidate(context, 'I', 0, 0, 0, 0)
        self.assertEqual(cross.candidate_cross_counts(state.board, (kills, safe))[1].tolist(), [0, 1])
        config = neural.Config(architecture='survival-spatial-v4', protect_target=False)
        options = [kills] * 127 + [safe]
        stats = {}
        self.assertEqual(cross.preview_choice(state, 1, options, np.arange(128, 0, -1), config, stats=stats), 127)
        self.assertEqual(stats, {'planner_triggers': 1, 'immediate_losses': 1, 'immediate_rescues': 1,
                                 'planner_overrides': 1})
        self.assertEqual(cross.preview_choice(state, 1, [kills] * 4, [4, 3, 2, 1], config), 0)

    def test_rescue_only_keeps_safe_greedy_and_takes_first_safe_tie_on_terminal_move(self):
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.current, state.next_piece, state.board = 'I', 'CROSS', frontier()
        state.stream = object()
        context = game.evaluate_board(state.board)
        kills = game.score_candidate(context, 'I', 2, 7, 7, 7)
        safe = [game.score_candidate(context, 'I', 0, x, 0, 0) for x in (0, 1)]
        for candidate, expected in ((kills, False), (safe[0], True), (safe[1], True)):
            self.assertEqual(game.has_legal_move(game.materialize_move(state.board, candidate).board,
                                                 state.next_piece), expected)
        options = [kills] * 126 + safe
        config = neural.Config(architecture='survival-spatial-v4', protect_target=False)
        with patch.object(cross, '_visible_replies', side_effect=AssertionError('Unexpected reply scoring')):
            stats = {}
            self.assertEqual(cross.preview_choice(state, 1, options, [1.] * 126 + [0., 0.],
                                                  config, stats=stats, rescue_only=True), 126)
            self.assertEqual(stats, dict(planner_triggers=1, immediate_losses=1,
                                         immediate_rescues=1, planner_overrides=1))
            stats = {}
            self.assertEqual(cross.preview_choice(state, 1, safe, [0., 1.], config,
                                                  stats=stats, rescue_only=True), 1)
            self.assertEqual(stats, {})
            self.assertEqual(cross.preview_choice(state, 1, [kills] * 2, [0., 1.],
                                                  config, rescue_only=True), 1)

    def test_reply_bonus_uses_normalized_mobility_and_actor_order_ties(self):
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.current, state.next_piece, state.board = 'I', 'CROSS', frontier()
        context = game.evaluate_board(state.board)
        options = [game.score_candidate(context, 'I', 0, k, 0, 0) for k in range(2)]
        config = neural.Config(architecture='survival-spatial-v4', protect_target=False)
        def choose(logits, second_reply_origins=1):
            with patch.object(cross, '_visible_replies', return_value=(options[0],)) as replies, patch.object(
                    cross, 'candidate_cross_counts', side_effect=[(1, np.array([0])), (1, np.array([second_reply_origins]))]):
                choice = cross.preview_choice(state, 1, options, logits, config)
                return choice, replies.call_count
        self.assertEqual(choose([1., .9]), (1, 2))
        self.assertEqual(choose([1., .5]), (0, 2))
        self.assertEqual(choose([float(cross.MOBILITY_VALUES[1]), 0.]), (0, 2))
        # A best possible tie needs no reply search. Any remaining headroom still requires one.
        self.assertEqual(choose([1., 0.], 16), (0, 1))
        self.assertEqual(choose([1., .001], 16), (1, 2))

    def test_uncapped_terminal_and_live_target_block_have_distinct_reasons(self):
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.current, state.next_piece, state.board = 'I', 'CROSS', frontier()
        model = neural.create_model(neural.Config(architecture='survival-spatial-v4'))
        lethal = game.score_candidate(game.evaluate_board(state.board), 'I', 2, 7, 7, 7)
        snapshot = ev._snapshot(state, 1, 0.)
        for moves, reason, decisions in [({0: (1, lethal)}, 'top_out', 1), ({}, 'target_or_no_move', 0)]:
            with patch.object(ev, 'select_moves', return_value=moves):
                result = ev.evaluate_fresh(model, [37], policy='ppo', cap=0, device='cpu', partials={37: snapshot})
            self.assertEqual(result['stop_reasons'], [reason])
            self.assertEqual(result['target_or_no_move'], int(reason == 'target_or_no_move'))
            self.assertEqual(result['decision_counts']['decisions'], decisions)
            self.assertEqual(result['move_latency']['samples'], decisions)

    def test_uncapped_partial_resume_exact_board_target_and_stream(self):
        torch.set_num_threads(1)
        config = neural.Config(architecture='survival-spatial-v4', spatial_actor=True, screening='sampled',
                               candidates=32, shortlist=4, protect_target=False)
        model = neural.create_model(config)
        saved = {}
        count = 0
        def expired():
            nonlocal count
            count += 1
            return count > 3
        def store(snapshot):
            saved[snapshot['seed']] = snapshot
        with patch('builtins.print'):
            result = ev.evaluate_fresh(model, [13], policy='ppo', cap=0, device='cpu', stop_requested=expired,
                                       on_partial=store, ppo_batch_size=1)
        self.assertFalse(result['complete'])
        self.assertEqual(saved[13]['pieces'], 3)
        baseline = game.Game(13)
        ev.base.library.ignore(baseline)
        target, _ = ev._resume(baseline, saved[13])
        restored_draws = [baseline.stream.pop() for _ in range(70)]
        uninterrupted = game.Game(13)
        ev.base.library.ignore(uninterrupted)
        for _ in range(3):
            uninterrupted.stream.pop()
        self.assertEqual(restored_draws, [uninterrupted.stream.pop() for _ in range(70)])
        self.assertEqual(target, saved[13]['target'])
        following = ev.evaluate_fresh(model, [13], policy='ppo', cap=5, device='cpu',
                                      ppo_batch_size=1, partials=saved)
        reference = ev.evaluate_fresh(model, [13], policy='ppo', cap=5, device='cpu', ppo_batch_size=1)
        for key in ('placements', 'clears', 'stop_reasons'):
            self.assertEqual(following[key], reference[key])
        self.assertEqual(ev._snapshot(baseline, target, 0.)['board'], saved[13]['board'])

    def test_rescue_only_paired_greedy_at_first_loss_and_across_resume(self):
        model = neural.create_model(neural.Config(architecture='survival-spatial-v4'))
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.board, state.current, state.next_piece = frontier(), 'I', 'I'
        state.stream.bag = ['I', 'CROSS']
        start = ev._snapshot(state, 1, 0.)

        def select(model, states, targets, scoring, device, preparer, preview_search=False, decision_stats=None):
            state, = states
            context = game.evaluate_board(state.board)
            if state.pieces == 0:
                options = [game.score_candidate(context, 'I', 0, 0, 0, 0)]
            elif state.pieces == 1:
                options = [game.score_candidate(context, 'I', 2, 7, 7, 7),
                           game.score_candidate(context, 'I', 0, 0, 0, 4)]
            elif state.pieces == 2:
                options = [game.score_candidate(context, 'CROSS', 0, 6, 6, 6)]
            else:
                return {}
            self.assertTrue(all(candidate is not None for candidate in options))
            i = cross.preview_choice(state, targets[0], options, [1., 0.], model.config,
                                     stats=decision_stats[0], rescue_only=True) if preview_search else 0
            return {0: (targets[0], options[i])}

        def run(snapshot=start, *, cap=0, mode='rescue-only', pause=False):
            saved = []
            stop = iter((False, True)).__next__ if pause else None
            with patch.object(ev, 'select_moves', side_effect=select):
                result = ev.evaluate_fresh(model, [37], policy='ppo', cap=cap, device='cpu', preview_search=mode,
                                           partials={37: snapshot}, stop_requested=stop, on_partial=saved.append)
            return saved[0] if pause else result['games'][0]

        pre = run(pause=True)
        post = run(pre, pause=True)
        self.assertEqual((pre['pieces'], pre['first_greedy_loss_placement']), (1, None))
        self.assertEqual((post['pieces'], post['first_greedy_loss_placement']), (2, 2))
        for snapshot in (start, pre, post):
            row = run(snapshot)
            self.assertEqual((row['placements'], row['reason']), (3, 'target_or_no_move'))
            self.assertEqual(row['paired_greedy'], dict(placements=2, reason='top_out', physical_top_out=True))
        with self.assertRaisesRegex(ValueError, 'paired greedy loss'):
            run({k: v for k, v in post.items() if k != 'first_greedy_loss_placement'})

        for cap, reason, physical in [(0, 'top_out', True), (1, 'cap', False), (2, 'cap', True)]:
            greedy, rescue = run(cap=cap, mode=False), run(cap=cap)
            self.assertNotIn('paired_greedy', greedy)
            self.assertEqual(greedy['reason'], reason)
            self.assertEqual(rescue['paired_greedy'], dict(placements=greedy['placements'], reason=reason,
                                                           physical_top_out=physical))
        with patch.object(ev, 'select_moves', return_value={}):
            blocked = ev.evaluate_fresh(model, [37], policy='ppo', cap=0, device='cpu', preview_search='rescue-only')
        self.assertEqual(blocked['games'][0]['paired_greedy'],
                         dict(placements=0, reason='target_or_no_move', physical_top_out=False))

    def test_disk_evaluation_partial_reused_then_removed_after_final_atomic_record(self):
        torch.set_num_threads(1)
        config = neural.Config(architecture='survival-spatial-v4', spatial_actor=True, candidates=32,
                               shortlist=4, protect_target=False)
        model = neural.create_model(config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'source.pt'
            checkpoint.write_bytes(b'test source')
            sha = ev.base.file_hash(checkpoint)
            calls = 0

            def interrupt():
                nonlocal calls
                calls += 1
                return calls > 3

            def run(budget, cache, preview_search=True):
                with patch.object(ev, 'load_model', return_value=model):
                    return ev.evaluate_checkpoint(checkpoint, sha, 'ppo', [41], 5, 'test', 'cpu', budget,
                                                  {}, 'teacher', cache, ppo_batch_size=1, preview_search=preview_search)
            partial = run(SimpleNamespace(expired=interrupt), root / 'resumed')
            self.assertFalse(partial['complete'])
            active = next((root / 'resumed').rglob('41.partial.json'))
            record = ev.read_json(active)
            self.assertEqual(record['data']['pieces'], 2)
            self.assertEqual(record['data']['decisions']['decisions'], 2)
            never = SimpleNamespace(expired=lambda: False)
            resumed = run(never, root / 'resumed')
            full = run(never, root / 'fresh')
            self.assertFalse(active.exists())
            for key in ('decision_counts', 'placements', 'clears', 'stop_reasons'):
                self.assertEqual(resumed[key], full[key])
            self.assertEqual(resumed['move_latency']['samples'], 5)
            self.assertTrue(active.with_name('41.json').is_file())
            for mode in (False, 'rescue-only'):
                result = run(never, root / 'resumed', mode)
                self.assertTrue(result['complete'])
                self.assertEqual('paired_greedy' in result['games'][0], mode == 'rescue-only')
                if mode == 'rescue-only':
                    self.assertEqual(result['games'][0]['paired_greedy'],
                                     dict(placements=5, reason='cap', physical_top_out=False))
                    with patch.object(ev, 'load_model', side_effect=AssertionError('Completed game must be cached')):
                        cached = ev.evaluate_checkpoint(checkpoint, sha, 'ppo', [41], 5, 'test', 'cpu', never, {},
                                                        'teacher', root / 'resumed', ppo_batch_size=1,
                                                        preview_search='rescue-only')
                    self.assertEqual(result['games'], cached['games'])
            self.assertEqual(len(list((root / 'resumed').rglob('41.json'))), 3)

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1', 'requires coordinated CUDA slot')
    def test_cuda_exact_delta_clears_padded_and_fused_feature_contract(self):
        from tetris_gpu_cross import candidate_counts
        from tetris_gpu_simulation import GpuPreparer
        game.configure_backend('native')
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.current, state.next_piece, state.board = 'I', 'CROSS', frontier()
        config = neural.Config(architecture='survival-spatial-v4', candidates=48, shortlist=16,
                               protect_target=False, cross_weight=.5)
        prep = GpuPreparer(config)
        prepared = prep.prepare_batch([state], [1])
        options = tuple(prepared.prepared[0].options)
        np.testing.assert_array_equal(prepared.batch['cross_mobility'][0].cpu().numpy(),
                                      cross.mobility_features(state.board, options))
        board, clear = self.clear_fixture()
        context = game.evaluate_board(state.board)
        choices = [game.score_candidate(context, 'I', 2, 7, 7, 7), game.score_candidate(context, 'I', 0, 0, 0, 0)]
        def spec(c):
            return [c.rotation, c.x, c.y, c.w, c.z, c.layers]
        packed = torch.tensor([[spec(c) for c in choices], [spec(clear), [9999]*6]], device='cuda', dtype=torch.int32)
        masks = torch.tensor([state.board, board], device='cuda', dtype=torch.int32)
        valid = torch.tensor([[True,True],[True,False]], device='cuda')
        ids = torch.tensor([prep.names.index('I')]*2, device='cuda', dtype=torch.int32)
        before, after = candidate_counts(masks, packed, ids, prep.shapes, prep.shape_offsets, valid)
        self.assertEqual(before.tolist(), [cross.cross_count(state.board), cross.cross_count(board)])
        self.assertEqual(after.tolist(), [[0, 1], [int(cross.candidate_cross_counts(board, [clear])[1][0]), 0]])


if __name__ == '__main__':
    unittest.main()
