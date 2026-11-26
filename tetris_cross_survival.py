"""Exact CROSS mobility and a one-visible-piece geometric survival planner."""
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

import sparse_tetris_4d_optimized as game


NEIGHBORS = (0, -1, 1, -16, 16, -256, 256)
MOBILITY_VALUES = (np.log1p(np.arange(17, dtype=np.float64)) / np.log(17)).astype(np.float32)
PREVIEW_THRESHOLD = 8


def _center_mask(board):
    grid = np.asarray(board, dtype=np.uint16).reshape(16, 16, 16)
    # Vertical center mask 0b111 has bottom 0. Each face neighbor has mask 0b010, bottom 1.
    ok = grid[1:-1, 1:-1, 1:-1] < 8192
    for axis in range(3):
        for adjacent in (slice(None, -2), slice(2, None)):
            slices = [slice(1, -1)] * 3
            slices[axis] = adjacent
            ok &= grid[tuple(slices)] < 16384
    centers = np.zeros_like(grid, dtype=bool)
    centers[1:-1, 1:-1, 1:-1] = ok
    return centers.ravel()


def cross_count(board):
    return int(_center_mask(board).sum())


def _after_board(board, candidate):
    columns = list(board)
    base = game.column_index(candidate.x, candidate.y, candidate.w)
    for offset, bits, _ in game.SHAPES[candidate.piece][candidate.rotation].columns:
        columns[base + offset] |= bits << candidate.z
    return game.clear_layers(tuple(columns))[0] if candidate.layers else tuple(columns)


def _after_count(board, candidate, centers, count):
    if candidate.layers:
        return cross_count(_after_board(board, candidate))
    if count == 0:
        return 0
    lost = set()
    origin = game.column_index(candidate.x, candidate.y, candidate.w)
    for offset, bits, _ in game.SHAPES[candidate.piece][candidate.rotation].columns:
        raised = bits << candidate.z
        if raised < 8192:
            continue
        col = origin + offset
        if centers[col]:
            lost.add(col)
        if raised >= 16384:
            for step in NEIGHBORS[1:]:
                center = col + step
                if 0 <= center < 4096 and centers[center]:
                    lost.add(center)
    return count - len(lost)


def candidate_cross_counts(board, options):
    centers = _center_mask(board)
    before = int(centers.sum())
    after = np.fromiter((_after_count(board, option, centers, before) for option in options), np.int32, len(options))
    return before, after


def mobility_features(board, options):
    before, after = candidate_cross_counts(board, options)
    features = np.empty((len(after), 2), np.float32)
    features[:, 0] = MOBILITY_VALUES[min(before, 16)]
    features[:, 1] = MOBILITY_VALUES[after.clip(max=16)]
    return features


@lru_cache(maxsize=256)
def _shape_windows(shape):
    sx, sy, sz, sw = shape.extents
    windows = []
    for offset, _, bottom in shape.columns:
        x, y, w = game.column_coordinates(offset)
        windows.append(((slice(w, w + 17 - sw), slice(y, y + 17 - sy), slice(x, x + 17 - sx)),
                        1 << (16 - sz + bottom)))
    return tuple(windows)


def visible_piece_fits(board, piece):
    """All physical rotations and origins using packed column masks, including buried targets."""
    grid = np.asarray(board, dtype=np.int32).reshape(16, 16, 16)
    for shape in game.SHAPES[piece]:
        windows = _shape_windows(shape)
        slices, limit = windows[0]
        valid = grid[slices] < limit
        for slices, limit in windows[1:]:
            valid &= grid[slices] < limit
        if valid.any():
            return True
    return False


def _visible_replies(state, target, after, current, config):
    import tetris_neural as neural
    counts = dict(state.piece_counts)
    counts[state.current] += 1
    # Teacher preparation uses only public board, committed piece counts and visible piece.
    # The missing preview is explicitly unknown. This view contains no stream or random state.
    view = SimpleNamespace(board=after, current=state.next_piece, next_piece=None, piece_set=state.piece_set,
                           pieces=state.pieces + 1, piece_counts=counts, seed=state.seed, game_over=False)
    if current.layers:
        following = game.next_hyperlayer_target(after, target)
        if following is not None:
            target = following
    prepared = neural.prepare(view, target, config)
    if prepared is None:
        return ()
    order = np.argsort(-prepared.arrays['teacher_prior'], kind='stable')
    return tuple(prepared.options[i] for i in order[:4])


@lru_cache(maxsize=32)
def _preview_witnesses(piece):
    shape = game.SHAPES[piece][0]
    return tuple(tuple((game.column_index(x, y, w) + offset, 1 << (16 - shape.extents[2] + bottom))
                       for offset, _, bottom in shape.columns)
                 for x, y, w in ((0, 0, 0), (6, 6, 6), (10, 3, 10), (3, 10, 3)))


_SAMPLE_CENTERS = tuple(game.column_index(x, y, w)
                        for w in (3, 8, 12) for y in (3, 8, 12) for x in (3, 8, 12))


def _ordinary_afterstate(board, action, preview, normalized_count):
    """Prove the early return cheaply on low boards; a failed witness falls back to exact search."""
    if action.layers:
        return False
    base = game.column_index(action.x, action.y, action.w)
    raised = {base + offset: board[base + offset] | (bits << action.z)
              for offset, bits, _ in game.SHAPES[action.piece][action.rotation].columns}
    if normalized_count is None:
        successes = 0
        for center in _SAMPLE_CENTERS:
            if (raised.get(center, board[center]) < 8192 and
                    all(raised.get(center + step, board[center + step]) < 16384 for step in NEIGHBORS[1:])):
                successes += 1
                if successes > PREVIEW_THRESHOLD:
                    break
        if successes <= PREVIEW_THRESHOLD:
            return False
    elif normalized_count <= MOBILITY_VALUES[PREVIEW_THRESHOLD]:
        return False
    return preview == 'CROSS' or any(all(raised.get(col, board[col]) < limit for col, limit in witness)
                                     for witness in _preview_witnesses(preview))


def preview_choice(state, target, options, logits, config, normalized_after=None, stats=None):
    """Return a shortlist index. The physical rescue scan covers every retained action.

    Outside immediate danger and the <=8 afterstate frontier, preserve the actor's choice.
    Within the frontier add the best normalized CROSS mobility of four teacher replies
    to the original actor logit. Break ties by the actor ordering.
    Prioritize physical preview survival; keep the greedy move if all actions lose immediately.
    """
    order = np.argsort(-np.asarray(logits)[:len(options)], kind='stable')
    greedy = int(order[0])
    chosen = options[greedy]
    known = normalized_after[greedy] if normalized_after is not None else None
    if _ordinary_afterstate(state.board, chosen, state.next_piece, known):
        return greedy
    centers = _center_mask(state.board)
    before = int(centers.sum())
    count = _after_count(state.board, chosen, centers, before)
    first_board = _after_board(state.board, chosen)
    alive = count > 0 if state.next_piece == 'CROSS' else visible_piece_fits(first_board, state.next_piece)
    if alive and count > PREVIEW_THRESHOLD:
        return greedy
    if stats is not None:
        stats['planner_triggers'] = 1
        stats['immediate_losses'] = int(not alive)
    candidates = []
    for i in order if not alive else order[:16]:
        i = int(i)
        action = options[i]
        board = first_board if i == greedy else _after_board(state.board, action)
        if state.next_piece == 'CROSS':
            after_count = count if i == greedy else _after_count(state.board, action, centers, before)
            playable = after_count > 0
        else:
            playable = visible_piece_fits(board, state.next_piece)
        if playable:
            candidates.append((i, action, board))
    if not candidates:
        return greedy
    if stats is not None:
        stats['immediate_rescues'] = int(not alive)
    # An immediate rescue scans the whole shortlist. Expand at most 16 surviving actions,
    # in the original actor order, even when the first surviving move ranks below 16.
    reply_config = replace(config, architecture='survival-feature-v3', screening='exhaustive',
                           shortlist=4, shortlist_mode='teacher-top', cross_residual=False, cross_weight=0.)
    best, best_key = candidates[0][0], None
    for i, action, board in candidates[:16]:
        replies = _visible_replies(state, target, board, action, reply_config)
        reply_count = int(candidate_cross_counts(board, replies)[1].max()) if replies else -1
        reply_bonus = float(MOBILITY_VALUES[min(reply_count, 16)]) if replies else 0.
        key = float(logits[i]) + reply_bonus
        if best_key is None or key > best_key:
            best, best_key = i, key
    if stats is not None:
        stats['planner_overrides'] = int(best != greedy)
    return best
