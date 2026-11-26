import base64
import hashlib
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

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

    def test_last_cross_tensor_uses_valid_moves_and_stable_actor_order(self):
        # Each row covers a different guard outcome; the padding has the highest score and positive mobility.
        logits = torch.tensor([[12., 8., 8., 99.]]).repeat(6, 1)
        valid = torch.tensor([[True, True, True, False]]).repeat(6, 1)
        valid[-1] = False
        before = torch.tensor([1., 1., 1., 0., 1., 0.])[:, None].expand_as(logits)
        after = torch.tensor([[0., 1., 1., 1.], [0., 0., 1., 1.], [0., 0., 0., 1.],
                              [0., 1., 0., 1.], [1., 0., 1., 1.], [0., 1., 1., 1.]])
        choice, threats, overrides = cross.preserve_last_cross(logits, valid, torch.stack((before, after), -1))
        self.assertEqual(choice.tolist(), [1, 2, 0, 0, 0, -1])
        self.assertEqual(threats.tolist(), [True, True, True, False, False, False])
        self.assertEqual(overrides.tolist(), [True, True, False, False, False, False])

    def test_last_cross_device_path_uses_only_selected_metadata(self):
        # No state, next piece, RNG or candidate shortlist is accessible on the host in this path.
        model = neural.create_model(neural.Config(architecture='survival-spatial-v4', shortlist=4))
        batch = dict(valid=torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
                     cross_mobility=torch.tensor([[[1., 0.], [1., 1.], [0., 1.], [0., 1.]]]))
        result = SimpleNamespace(batch=batch, environment_indices=(1,), targets=[0, 3],
                                 selected_metadata=lambda a: torch.tensor([[int(a.item()), 0., 0., 0., 0., 0., 0.]]),
                                 candidates_from_metadata=lambda rows: [int(r[0]) for r in rows])
        prep = SimpleNamespace(prepare_batch=lambda states, targets: result)
        stats = {1: {}}
        with patch.object(ev, '_fixed_actor_logits', return_value=torch.tensor([[8., 3., 90., 80.]])):
            moves = ev.select_moves(model, [object(), object()], [0, 3], False, 'cpu', prep, 2,
                                    preview_search='preserve-last-cross', decision_stats=stats)
        self.assertEqual(moves, {1: (3, 1)})
        self.assertEqual(stats[1], dict(last_cross_losses=1, last_cross_overrides=1))
        result.selected_candidates = lambda a: [int(a.item())]
        with patch.object(ev, '_fixed_actor_logits', return_value=torch.tensor([[8., 3., 90., 80.]])):
            self.assertEqual(ev.select_moves(model, [object(), object()], [0, 3], False, 'cpu', prep, 2,
                                             preview_search='preserve-last-cross'), moves)

    def test_last_cross_gameplay_preserves_scoring_config_and_uses_exhaustive_shortlist(self):
        state = game.Game(36)
        config = neural.Config(architecture='survival-spatial-v4', candidates=48, shortlist=4, protect_target=False)
        model = neural.create_model(config)
        saved = dict(progress={'updates': 1}, teacher_parameters=game.network_parameters(), status='test')
        with tempfile.NamedTemporaryFile() as checkpoint, patch.object(neural, 'load_checkpoint', return_value=(model, saved)):
            agent = neural.NeuralAgent(checkpoint.name, 'cpu', preview_search='preserve-last-cross')
        self.assertEqual(agent.candidate_budget, 0)
        prepared = neural.prepare(state, 1, config)
        self.assertNotIn('cross_mobility', prepared.arrays)
        def prepared_steps(*args):
            self.assertIs(args[2], config)
            self.assertEqual(args[3], 0)
            yield from ()
            return prepared
        prediction = dict(logits=torch.tensor([[3., 2., 1., 0.]]), value=torch.zeros(1))
        with patch.object(neural, 'prepare_steps', side_effect=prepared_steps), patch.object(model, 'forward', return_value=prediction), \
             patch.object(cross, 'mobility_features', return_value=np.asarray([[1, 0], [1, 1], [1, 1], [1, 0]], np.float32)), \
             patch.object(cross, 'preview_choice', side_effect=AssertionError('guard cannot inspect preview')):
            chosen = agent.choose(state)
        self.assertEqual(chosen.move, game.materialize_move(state.board, prepared.options[1]))
        self.assertIs(model.config, config)

    def test_last_cross_custom_preparer_uses_actual_board_and_shortlist(self):
        model = neural.create_model(neural.Config(architecture='survival-spatial-v4', shortlist=4))
        board = frontier()
        # A custom preparer omitted mobility. The fallback uses only the public board and real options.
        state = SimpleNamespace(board=board)
        options = ('greedy', 'safe')
        result = SimpleNamespace(batch={'valid': torch.tensor([[True, True, False, False]])},
                                 environment_indices=(0,), targets=[1], prepared=[SimpleNamespace(options=options)],
                                 selected_candidates=lambda a: [options[int(a.item())]])
        preparer = SimpleNamespace(prepare_batch=lambda states, targets: result)
        with patch.object(cross, 'mobility_features', return_value=np.asarray([[1, 0], [1, 1]], np.float32)) as geometry, \
             patch.object(ev, '_fixed_actor_logits', return_value=torch.tensor([[2., 1., 99., 99.]])):
            self.assertEqual(ev.select_moves(model, [state], [1], False, 'cpu', preparer, 1,
                                             preview_search='preserve-last-cross'), {0: (1, 'safe')})
        geometry.assert_called_once_with(board, options)

    def test_exhaustive_rescue_returns_an_outside_candidate_through_both_preparers(self):
        torch.set_num_threads(1)
        model = neural.create_model(neural.Config(architecture='survival-spatial-v4', shortlist=2, protect_target=False))
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.board, state.current, state.next_piece = frontier(), 'I', 'CROSS'
        failed = game.score_candidate(game.evaluate_board(state.board), 'I', 2, 7, 7, 7)
        self.assertFalse(game.has_legal_move(game.materialize_move(state.board, failed).board, 'CROSS'))
        arrays = {'targets': np.asarray(1)}
        prepared = SimpleNamespace(arrays=arrays, options=(failed,))
        batch = {'valid': torch.tensor([[True]])}
        device_result = SimpleNamespace(
            batch=batch, environment_indices=[0], targets=[1], prepared=[prepared],
            _option_specs=torch.tensor([[[failed.rotation, failed.x, failed.y, failed.w, failed.z, failed.layers]]]),
            _option_scores=torch.tensor([[failed.score]], dtype=torch.float64))
        device_preparer = SimpleNamespace(prepare_batch=lambda *args: device_result)
        chosen = []
        with patch.object(neural, 'prepare', return_value=prepared), patch.object(neural, 'collate', return_value=batch), \
             patch.object(ev, '_fixed_actor_logits', return_value=torch.tensor([[0.]])):
            for preparer in (None, device_preparer):
                stats = {0: {}}
                _, candidate = ev.select_moves(model, [state], [1], False, 'cpu', preparer, 1,
                                               preview_search='exhaustive-rescue', decision_stats=stats)[0]
                self.assertNotEqual(candidate, failed)
                self.assertTrue(game.has_legal_move(game.materialize_move(state.board, candidate).board, 'CROSS'))
                self.assertEqual((stats[0]['outside_rescue_searches'], stats[0]['outside_rescues']), (1, 1))
                chosen.append(candidate)
        self.assertEqual(chosen[0], chosen[1])

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

    def test_policy_counts_skip_saturated_afterboards_including_clears(self):
        boundary = cross.MOBILITY_CAP + cross.MAX_CROSS_LOSS
        for before in (boundary - 1, boundary):
            cells = [1 << 14] * 4096
            centers = [game.column_index(x, y, w) for x in range(2, 14, 3)
                       for y in range(2, 14, 3) for w in range(2, 14, 3)]
            for center in centers[:before]:
                cells[center] = 1 << 12
                for step in cross.NEIGHBORS[1:]:
                    cells[center + step] = 1 << 13
            board = tuple(cells)
            move = game.score_candidate(game.evaluate_board(board), 'I', 0, 0, 0, 0)
            expected = cross.cross_count(game.materialize_move(board, move).board)
            with patch.object(cross, '_after_count', wraps=cross._after_count) as count:
                initial, after = cross.candidate_cross_counts(board, (move,), capped=True)
            self.assertEqual(initial, before)
            self.assertEqual(after.tolist(), [min(expected, cross.MOBILITY_CAP)])
            self.assertEqual(count.call_count, int(before < boundary))
        cells = [1] * 4096
        cells[game.column_index(7, 7, 7)] = 0
        board = tuple(cells)
        clear = game.score_candidate(game.evaluate_board(board), 'I', 2, 7, 7, 7)
        self.assertEqual(clear.layers, 1)
        self.assertEqual(cross.cross_count(game.materialize_move(board, clear).board), 14**3)
        with patch.object(cross, '_after_count', side_effect=AssertionError('unnecessary recount')):
            np.testing.assert_array_equal(cross.mobility_features(board, (clear,)), np.ones((1, 2), np.float32))

    @unittest.skipUnless(os.environ.get('TETRIS4D_TEST_CUDA') == '1' or os.environ.get('TRITON_INTERPRET') == '1',
                         'requires CUDA or the Triton interpreter')
    def test_device_capped_counts_match_full_board_and_keep_padding_zero(self):
        from tetris_gpu_cross import candidate_counts
        device = 'cpu' if os.environ.get('TRITON_INTERPRET') == '1' else 'cuda'
        game.configure_backend('native')
        high = [1] * 4096
        high[game.column_index(7, 7, 7)] = 0
        scarce, rare_clear = self.clear_fixture()
        groups = []
        for board in (tuple(high), frontier(), scarce):
            context = game.evaluate_board(board)
            moves = [game.score_candidate(context, 'I', 2, 7, 7, 7),
                     game.score_candidate(context, 'I', 0, 0, 0, 0)]
            groups.append((board, [m for m in moves if m is not None]))
        self.assertTrue(rare_clear.layers)
        self.assertTrue(groups[0][1][0].layers)
        specs = torch.zeros((3, 3, 6), dtype=torch.int32, device=device)
        valid = torch.zeros((3, 3), dtype=torch.bool, device=device)
        expected = np.zeros((3, 3), dtype=np.int32)
        for env, (board, moves) in enumerate(groups):
            for i, m in enumerate(moves):
                specs[env, i] = torch.tensor((m.rotation, m.x, m.y, m.w, m.z, m.layers), device=device)
                valid[env, i] = True
                expected[env, i] = cross.cross_count(game.materialize_move(board, m).board)
        boards = torch.tensor([board for board, _ in groups], dtype=torch.int32, device=device)
        catalog = torch.tensor(game.native_shapes('I'), device=device)
        ids = torch.zeros(3, dtype=torch.int32, device=device)
        offsets = torch.zeros(1, dtype=torch.int32, device=device)
        for capped in (False, True):
            before, after = candidate_counts(boards, specs, ids, catalog, offsets, valid, capped=capped)
            self.assertEqual(before.tolist(), [cross.cross_count(b) for b, _ in groups])
            np.testing.assert_array_equal(after.cpu().numpy(), np.minimum(expected, cross.MOBILITY_CAP) if capped else expected)

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
        for moves, cap, headless, reason, decisions in [
            ({0: (2, lethal)}, 0, False, 'top_out', 1), ({0: (2, lethal)}, 0, True, 'top_out', 1),
            ({}, 0, False, 'target_or_no_move', 0), ({0: (2, lethal)}, 1, False, 'cap', 1)
        ]:
            with patch.object(ev, 'select_moves', return_value=moves):
                result = ev.evaluate_fresh(model, [37], policy='ppo', cap=cap, headless=headless,
                                           device='cpu', partials={37: snapshot})
            self.assertEqual(result['stop_reasons'], [reason])
            self.assertEqual(result['target_or_no_move'], int(reason == 'target_or_no_move'))
            self.assertEqual(result['decision_counts']['decisions'], decisions)
            self.assertEqual(result['move_latency']['samples'], decisions)
            transition = result['games'][0].get('terminal_transition')
            if reason == 'top_out':
                self.assertEqual((transition['requested_target'], transition['selected_target']), (1, 2))
                packed = zlib.decompress(base64.b64decode(transition['before_board_u16le_zlib'], validate=True))
                before = tuple(map(int, np.frombuffer(packed, dtype='<u2')))
                self.assertEqual(before, state.board)
                replay = game.materialize_move(before, game.Candidate(**transition['candidate']))
                after = np.asarray(replay.board, dtype='<u2').tobytes()
                self.assertEqual(hashlib.sha256(after).hexdigest(), transition['after_board_sha256'])
                self.assertEqual(transition['blocked_piece'], 'CROSS')
                self.assertFalse(game.has_legal_move(replay.board, transition['blocked_piece']))
            else:
                self.assertIsNone(transition)

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
            for mode in (False, 'rescue-only', 'exhaustive-rescue', 'preserve-last-cross'):
                result = run(never, root / 'resumed', mode)
                self.assertTrue(result['complete'])
                self.assertEqual('paired_greedy' in result['games'][0], mode in ('rescue-only', 'exhaustive-rescue'))
                self.assertEqual('paired_shortlist_rescue' in result['games'][0], mode == 'exhaustive-rescue')
                self.assertEqual('outside_rescues' in result['decision_counts'], mode == 'exhaustive-rescue')
                self.assertEqual('last_cross_overrides' in result['decision_counts'], mode == 'preserve-last-cross')
                if mode in ('rescue-only', 'exhaustive-rescue'):
                    self.assertEqual(result['games'][0]['paired_greedy'],
                                     dict(placements=5, reason='cap', physical_top_out=False))
                    with patch.object(ev, 'load_model', side_effect=AssertionError('Completed game must be cached')):
                        cached = ev.evaluate_checkpoint(checkpoint, sha, 'ppo', [41], 5, 'test', 'cpu', never, {},
                                                        'teacher', root / 'resumed', ppo_batch_size=1,
                                                        preview_search=mode)
                    self.assertEqual(result['games'], cached['games'])
            self.assertEqual(len(list((root / 'resumed').rglob('41.json'))), 5)

    def test_outside_rescue_paired_endpoints_survive_pause_after_first_override(self):
        model = neural.create_model(neural.Config(architecture='survival-spatial-v4', protect_target=False))
        state = game.Game(37)
        ev.base.library.ignore(state)
        state.board, state.current, state.next_piece = frontier(), 'I', 'CROSS'
        state.stream.bag = ['I']
        start = ev._snapshot(state, 1, 0.)

        def select(model, states, targets, scoring, device, preparer, preview_search=False, decision_stats=None):
            s = states[0]
            spec = (2, 7, 7, 7) if s.current == 'I' else (0, 6, 6, 6)
            move = game.score_candidate(game.evaluate_board(s.board), s.current, *spec)
            self.assertIsNotNone(move)
            _, outside = cross.rescue_decision(s, targets[0], [move], [0.], model.config, stats=decision_stats[0],
                                               exhaustive=preview_search == 'exhaustive-rescue')
            return {0: (targets[0], outside or move)}

        def run(snapshot, mode, **kwargs):
            with patch.object(ev, 'select_moves', side_effect=select):
                return ev.evaluate_fresh(model, [37], policy='ppo', cap=2, device='cpu',
                                         partials={37: snapshot}, preview_search=mode, **kwargs)

        baseline = run(start, 'rescue-only')['games'][0]
        saved = []
        run(start, 'exhaustive-rescue', on_partial=saved.append, stop_requested=iter((False, True)).__next__)
        self.assertEqual((saved[0]['first_greedy_loss_placement'], saved[0]['first_shortlist_rescue_loss_placement']), (1, 1))
        full, resumed = (run(p, 'exhaustive-rescue')['games'][0] for p in (start, saved[0]))
        self.assertEqual((baseline['reason'], baseline['placements']), ('top_out', 1))
        for row in (full, resumed):
            self.assertEqual(row['placements'], 2)
            self.assertEqual(row['paired_shortlist_rescue'], dict(placements=1, reason='top_out', physical_top_out=True))
            self.assertEqual(row['paired_greedy'], row['paired_shortlist_rescue'])
            self.assertEqual(row['decisions']['outside_rescues'], 1)
        with self.assertRaisesRegex(ValueError, 'paired shortlist_rescue loss'):
            run({k: v for k, v in saved[0].items() if k != 'first_shortlist_rescue_loss_placement'}, 'exhaustive-rescue')

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
