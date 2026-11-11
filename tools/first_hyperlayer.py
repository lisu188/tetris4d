from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game


def run(seed: int = 7, target: int = 1, pieces: int = 2000,
        candidates: int = game.DEFAULT_CANDIDATES, piece_set: str = 'expanded') -> dict:
    if pieces < 1:
        raise ValueError('Piece limit must be positive.')
    game.target_layer_status(game.EMPTY_BOARD, target)
    agent = game.Game(seed, piece_set)
    started = time.perf_counter()
    moves, snapshots, evaluations, peak_height = [], [], 0, 0
    stop_reason = 'piece_limit'
    for turn in range(pieces):
        decision = game.choose_hyperlayer_move(agent.board, agent.current, target, candidates, turn)
        if decision is None:
            stop_reason = 'top_out' if not game.has_legal_move(agent.board, agent.current) else 'planner_blocked'
            agent.game_over = stop_reason == 'top_out'
            break
        move, c = decision.move, decision.move.candidate
        peak_height = max(peak_height, c.z + game.SHAPES[c.piece][c.rotation].extents[2])
        evaluations += decision.evaluations
        solid = game.merge_rigid_piece(agent.board, game.placed_cells(move), move.piece)
        if game.target_layer_status(solid, target).buried:
            raise AssertionError('The target acquired an inaccessible gap.')
        moves.append([c.piece, c.rotation, c.x, c.y, c.w, c.z, c.layers])
        agent.commit(move)
        if agent.pieces % 100 == 0 or agent.layers:
            state = game.target_layer_status(agent.board, target)
            snapshots.append(dict(agent.summary(), target_filled=state.filled, buried_target=state.buried))
        if agent.layers:
            stop_reason = 'first_clear'
            break
        if agent.game_over:
            stop_reason = 'top_out'
            break
    return {'schema': 'tetris4d-first-clear-v1', 'seed': seed, 'mode': 'target', 'target': target,
            'piece_set': piece_set, 'candidates': candidates, 'backend': game.BACKEND,
            'parameters': game.network_parameters(), 'sampler_salt': 'zero-based move index',
            'summary': agent.summary(), 'stop_reason': stop_reason,
            'seconds': round(time.perf_counter() - started, 4), 'peak_height': peak_height,
            'evaluations': evaluations, 'moves': moves, 'final_board': agent.board, 'checkpoints': snapshots,
            'engine_sha256': hashlib.sha256(Path(game.__file__).read_bytes()).hexdigest()}


def verify(path: Path) -> dict:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Replay exceeds the 16 MiB limit.')
    data = json.loads(path.read_text(encoding='utf-8'))
    target = data['target']
    if data.get('mode') != 'target' or not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < game.HEIGHT:
        raise ValueError('Expected a target-planner replay with a valid Z level.')
    piece_set = data.get('piece_set', data['summary'].get('piece_set', 'expanded'))
    names = game.PIECE_SETS[piece_set]
    if not isinstance(data['moves'], list) or len(data['moves']) > 20000:
        raise ValueError('Invalid or oversized move sequence.')
    rng, bag, occupied = random.Random(data['seed']), [], set()
    placed = cleared = score = peak = prior_coverage = checks = 0
    events = []

    def check(condition, label):
        nonlocal checks
        if not condition:
            raise ValueError(f'Invalid replay: {label}')
        checks += 1

    for turn, record in enumerate(data['moves'], 1):
        check(isinstance(record, list) and len(record) == 7, 'move record')
        name, rotation, x, y, w, expected_z, expected_layers = record
        check(all(isinstance(v, int) and not isinstance(v, bool) for v in record[1:]), 'integer coordinates')
        if not bag:
            bag = list(names)
            rng.shuffle(bag)
        check(bag.pop() == name, f'piece sequence at {turn}')
        check(name in game.SHAPES and 0 <= rotation < len(game.SHAPES[name]), 'rotation')
        cells = game.SHAPES[name][rotation].cells
        check(len(cells) == len(set(cells)) == game.PIECE_SIZES[name], 'rigid shape')
        z = game.HEIGHT + max(c[2] for c in cells) + 1
        while True:
            down = {(x + dx, y + dy, z - 1 + dz, w + dw) for dx, dy, dz, dw in cells}
            if any(c[2] < 0 for c in down) or down & occupied:
                break
            z -= 1
        check(z == expected_z, f'independent landing at {turn}')
        added = {(x + dx, y + dy, z + dz, w + dw) for dx, dy, dz, dw in cells}
        check(not occupied & added, f'collision at {turn}')
        check(all(0 <= v < 16 for cell in added for v in cell), 'bounds')
        occupied |= added
        placed += len(added)
        peak = max(peak, max(c[2] for c in added) + 1)
        counts = Counter(c[2] for c in occupied)
        complete = sorted(level for level, count in counts.items() if count == game.AREA)
        check(len(complete) == expected_layers, f'independent clear count at {turn}')
        on_target = {(cx, cy, cw) for cx, cy, cz, cw in occupied if cz == target}
        above = {(cx, cy, cw) for cx, cy, cz, cw in occupied if cz > target}
        check(not above.difference(on_target), f'buried target at {turn}')
        check(len(on_target) >= prior_coverage, 'target coverage decreased')
        prior_coverage = len(on_target)
        if complete:
            check(turn == len(data['moves']), 'first-clear replay must stop at its first clear')
            events.append({'piece_number': turn, 'piece': name, 'cleared_z': complete,
                           'cells_before': len(occupied), 'full_counts': [counts[level] for level in complete]})
            occupied = {(cx, cy, cz - sum(level < cz for level in complete), cw)
                        for cx, cy, cz, cw in occupied if cz not in complete}
            cleared += len(complete)
            score += (0, 100, 300, 500, 800)[len(complete)] if len(complete) <= 4 else 800 + 400 * (len(complete) - 4)
        check(len(occupied) == placed - game.AREA * cleared, 'cell conservation')
    columns = [0] * game.AREA
    for x, y, z, w in occupied:
        columns[(w * game.DEPTH + y) * game.WIDTH + x] |= 1 << z
    check(columns == data['final_board'], 'final board')
    check(placed == data['summary']['cells_placed'], 'total cells')
    check(cleared == data['summary']['hyperlayers'], 'total hyperlayers')
    check(len(data['moves']) == data['summary']['pieces'], 'total pieces')
    check(score == data['summary']['score'], 'game score')
    return {'passed': True, 'seed': data['seed'], 'moves': len(data['moves']), 'checks': checks,
            'hyperlayers': cleared, 'clear_events': events, 'final_mass': len(occupied),
            'max_height_before_clear': peak, 'replay_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description='Run the integrated target planner and independently verify its first-clear replay.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--target', type=int, choices=range(game.HEIGHT), default=1)
    parser.add_argument('--pieces', type=int, default=2000)
    parser.add_argument('--candidates', type=int, default=game.DEFAULT_CANDIDATES)
    parser.add_argument('--piece-set', choices=tuple(game.PIECE_SETS), default='expanded')
    parser.add_argument('--backend', choices=('auto', 'python', 'numpy'), default='auto')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/trained-sigmoid.json')
    parser.add_argument('--output', type=Path, default=Path('first-clear.json'))
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.pieces < 1 or args.candidates < 0:
        parser.error('Piece limit must be positive; candidates must be nonnegative.')
    try:
        if args.verify:
            result = verify(args.verify)
        else:
            game.configure_backend(args.backend)
            game.load_checkpoint(args.weights)
            replay = run(args.seed, args.target, args.pieces, args.candidates, args.piece_set)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(replay, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            result = {key: value for key, value in replay.items() if key not in ('moves', 'final_board', 'checkpoints')}
            result['verification'] = verify(args.output)
            if replay['summary']['hyperlayers'] < 1:
                print(json.dumps(result, indent=2))
                return 1
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, AssertionError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
