FUTURE_HOLE_PENALTY = 4


def sustained_candidate_metrics(board: Board, candidate: Candidate, target: int) -> tuple[int, int, int]:
    gain, buried = target_candidate_metrics(board, candidate, target)
    base = column_index(candidate.x, candidate.y, candidate.w)
    future_holes = 0
    for offset, mask, _ in SHAPES[candidate.piece][candidate.rotation].columns:
        old_height = board[base + offset].bit_length()
        added = mask << candidate.z
        new_height = added.bit_length()
        future_holes += max(0, new_height - max(old_height, target + 1)) - (added >> (target + 1)).bit_count()
    return gain, buried, future_holes


@lru_cache(maxsize=16)
def sustained_above_counts(piece: str):
    return _NUMPY.asarray([[sum(count for level, count in shape.levels if level >= threshold)
                            for threshold in range(HEIGHT + 2)] for shape in SHAPES[piece]], dtype=_NUMPY.int32)


def sustained_batch_metrics(context: BoardEvaluation, piece: str, specs, z, target: int):
    np = _NUMPY
    raw = np.asarray(specs, dtype=np.int32)
    rotations = raw[:, 0]
    offsets, _, tops, present, _, _ = numpy_shape_data(piece)
    present = present[rotations]
    tops = tops[rotations]
    indices = (raw[:, 1] + WIDTH * raw[:, 2] + SLICE_AREA * raw[:, 3])[:, None] + offsets[rotations]
    heights = context.native_heights if context.native_heights is not None else np.asarray(context.heights, dtype=np.int16)
    old = heights[indices]
    relative = target - z
    on_target = ((numpy_column_masks(piece)[rotations] >> np.clip(relative, 0, HEIGHT)[:, None]) & 1).astype(bool)
    on_target &= (relative >= 0)[:, None] & (relative < HEIGHT)[:, None] & present
    gaps = old <= target
    gains = np.sum(gaps & on_target, axis=1)
    blocked = np.any(gaps & (z[:, None] + tops > target) & ~on_target & present, axis=1)
    new_spans = np.maximum(z[:, None] + tops - np.maximum(old, target + 1), 0)
    added_above = sustained_above_counts(piece)[rotations, np.clip(target + 1 - z, 0, HEIGHT + 1)]
    future_holes = np.sum(np.where(present, new_spans, 0), axis=1) - added_above
    return gains, blocked, future_holes


def sustained_decision_steps(board: Board, piece: str, target: int = 1,
                             candidates: int = DEFAULT_CANDIDATES,
                             salt: int = 0) -> Generator[int, None, Decision | None]:
    if piece not in SHAPES or type(candidates) is not int or candidates < 0:
        raise ValueError('Unknown piece or invalid candidate budget.')
    status = target_layer_status(board, target)
    if status.buried or status.filled == AREA:
        return None
    if BACKEND == 'auto':
        configure_backend()
    started = time.perf_counter()
    context, evaluations, best, best_key, best_gain = evaluate_board(board), 0, None, None, 0
    missing = (i for i, mask in enumerate(board) if not mask & (1 << target))
    vertical = tuple(r for r, shape in enumerate(SHAPES[piece]) if len(shape.columns) == 1) if piece == 'I' else ()
    anchored = ((r, *column_coordinates(i)) for i in missing for r in vertical)
    initial = itertools.chain(specifications(piece, candidates, salt), anchored)
    fallback = False
    for iterator in (initial, specifications(piece, 0)):
        if fallback and (best is not None or not candidates):
            break
        while specs := tuple(itertools.islice(iterator, BATCH_SIZE)):
            batch = PlacementBatch(piece, specs)
            if batch.data is not None:
                np = _NUMPY
                scores, z, clears = batch.scores(context)
                gains, blocked, future_holes = sustained_batch_metrics(context, piece, specs, z, target)
                valid = np.flatnonzero(np.isfinite(scores) & ~blocked)
                if len(valid):
                    utility = gains - FUTURE_HOLE_PENALTY * future_holes
                    high = int(utility[valid].max())
                    valid = valid[utility[valid] == high]
                    index = int(valid[np.argmax(scores[valid])])
                    key = (high, float(scores[index]))
                    if best_key is None or key > best_key:
                        best_key, best_gain = key, int(gains[index])
                        best = clears.get(index) or Candidate(piece, *specs[index], int(z[index]), 0, key[1])
            else:
                for spec in specs:
                    candidate = score_candidate(context, piece, *spec)
                    if candidate is None:
                        continue
                    gain, buried, future_holes = sustained_candidate_metrics(board, candidate, target)
                    key = (gain - FUTURE_HOLE_PENALTY * future_holes, candidate.score)
                    if not buried and (best_key is None or key > best_key):
                        best_key, best, best_gain = key, candidate, gain
            evaluations += len(specs)
            yield evaluations
            if fallback and best is not None:
                break
        if best is not None:
            move = materialize_move(board, best)
            return Decision(move, move.score, evaluations, (time.perf_counter() - started) * 1000,
                            1, 1, candidates, bool(candidates and candidates < placement_count(piece)),
                            'sustained', target, best_gain, fallback)
        fallback = True
    return None


def choose_sustained_move(*args, **kwargs) -> Decision | None:
    steps = sustained_decision_steps(*args, **kwargs)
    while True:
        try:
            next(steps)
        except StopIteration as result:
            return result.value
