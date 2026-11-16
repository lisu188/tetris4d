"""Persistent library of every played game, stored compactly for reuse by later training.

Every ``Game`` records its placements through a small hook in the engine (``Game.reset``
and ``Game.commit``), so trainers, tools and the GUI need no per-site plumbing. Callers
may enrich records: ``tag`` attaches phase/source metadata to a game, ``annotate`` labels
the next placement (behaviour policy, policy update, target layer) and ``finish`` stores
a game with an explicit stop reason. Unfinished games are stored when they are reset,
garbage collected or when the process exits.

Layout (default ``<main checkout>/data/game-library``; override with
``TETRIS4D_GAME_LIBRARY=<dir>``, disable with ``TETRIS4D_GAME_LIBRARY=off``)::

    games/<id[:2]>/<id>.npz   one file per distinct game (content addressed)
    index.jsonl               one line per played game; repeats are marked duplicate

A game's id hashes the piece catalogue, seed, piece set and placements, so identical
plays (e.g. deterministic selection games) are stored once. ``replay`` reconstructs
every position deterministically through the engine and verifies the final board.
Recording is off by default under unittest/pytest unless configured explicitly.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import weakref

import numpy as np

SCHEMA = 'tetris4d-game-library-v1'
UNKNOWN = 255
_COLUMNS = ('piece', 'rotation', 'x', 'y', 'z', 'w', 'target', 'clears')

_engine = None          # engine module whose Game objects are observed
_library = None         # active GameLibrary or None
_configured = False     # explicit configure() overrides the test/env defaults
_live = weakref.WeakSet()
_context: dict = {}


def _main_checkout(start: Path) -> Path:
    for directory in (start, *start.parents):
        marker = directory / '.git'
        if marker.is_dir():
            return directory
        if marker.is_file():
            text = marker.read_text(encoding='utf-8', errors='replace').strip()
            if text.startswith('gitdir:'):
                gitdir = Path(text.split(':', 1)[1].strip())
                if not gitdir.is_absolute():
                    gitdir = (directory / gitdir).resolve()
                # <main>/.git/worktrees/<name> -> <main>
                if gitdir.parent.name == 'worktrees':
                    return gitdir.parent.parent.parent
            return directory
    return start


def default_root():
    value = os.environ.get('TETRIS4D_GAME_LIBRARY', '').strip()
    if value.lower() in ('0', 'off', 'false', 'none', 'disabled'):
        return None
    if value:
        return Path(value).expanduser()
    if 'unittest' in sys.modules or 'pytest' in sys.modules:
        return None
    return _main_checkout(Path(__file__).resolve().parent) / 'data' / 'game-library'


def _engine_module():
    global _engine
    if _engine is None:
        import sparse_tetris_4d_optimized as engine
        install(engine)
    return _engine


def _board_sha256(board) -> str:
    return hashlib.sha256(np.asarray(board, dtype='<u2').tobytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix='.tmp-')
    try:
        with os.fdopen(handle, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class GameLibrary:
    def __init__(self, root):
        self.root = Path(root)
        self.error = None
        self.stored = self.duplicates = 0

    def path(self, game_id: str) -> Path:
        return self.root / 'games' / game_id[:2] / f'{game_id}.npz'

    @staticmethod
    def game_id(fingerprint, seed, piece_set, moves) -> str:
        digest = hashlib.sha256(f'{SCHEMA}|{fingerprint}|{seed}|{piece_set}|'.encode())
        digest.update(np.ascontiguousarray(moves[:, :6], dtype=np.uint8).tobytes())
        return digest.hexdigest()

    def store(self, meta: dict, moves, update, behavior):
        """Store one game (idempotent) and append an index line; returns its id."""
        self.root.mkdir(parents=True, exist_ok=True)
        moves = np.asarray(moves, dtype=np.uint8).reshape(-1, len(_COLUMNS))
        game_id = self.game_id(meta['pieces_sha256'], meta['seed'], meta['piece_set'], moves)
        meta = {**meta, 'schema': SCHEMA, 'id': game_id, 'placements': int(len(moves)),
                'columns': list(_COLUMNS)}
        target = self.path(game_id)
        duplicate = target.exists()
        if not duplicate:
            buffer = io.BytesIO()
            np.savez_compressed(buffer, moves=moves, update=np.asarray(update, dtype=np.int32),
                                behavior=np.asarray(behavior, dtype=np.uint8),
                                meta=np.asarray(json.dumps(meta, separators=(',', ':'), allow_nan=False)))
            _atomic_bytes(target, buffer.getvalue())
        line = {key: meta.get(key) for key in (
            'id', 'seed', 'piece_set', 'placements', 'game_over', 'stop_reason', 'phase', 'source',
            'run_dir', 'session', 'episode', 'recorded_utc', 'behaviors')}
        line.update(file=str(target.relative_to(self.root)), duplicate=duplicate)
        data = (json.dumps(line, separators=(',', ':'), allow_nan=False) + '\n').encode()
        descriptor = os.open(self.root / 'index.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)
        self.duplicates += duplicate
        self.stored += not duplicate
        return game_id


class _Log:
    """Per-game placement log kept on the engine Game object."""
    __slots__ = ('moves', 'update', 'behavior', 'behaviors', 'pending', 'meta', 'done', 'ignore', '__weakref__')

    def __init__(self):
        self.moves = bytearray()
        self.update = array('i')
        self.behavior = bytearray()
        self.behaviors = []
        self.pending = None
        self.meta = {}
        self.done = False
        self.ignore = False

    def code(self, name):
        if name is None:
            return UNKNOWN
        if name not in self.behaviors:
            if len(self.behaviors) >= UNKNOWN:
                return UNKNOWN
            self.behaviors.append(name)
        return self.behaviors.index(name)


def install(engine):
    """Attach recording hooks to an engine module (called lazily by Game)."""
    global _engine
    _engine = engine
    engine._RECORDING = sys.modules[__name__]
    _activate()
    return engine._RECORDING


def _activate():
    global _library
    if not _configured:
        root = default_root()
        _library = GameLibrary(root) if root is not None else None


def configure(root=None, enabled=True):
    """Explicitly enable recording to ``root`` (or the default root) or disable it."""
    global _library, _configured
    _configured = True
    if not enabled:
        _library = None
        return None
    root = Path(root) if root is not None else default_root()
    _library = GameLibrary(root) if root is not None else None
    return _library


def active():
    _engine_module()
    return _library


def set_context(**values):
    """Process-wide metadata merged into every subsequently stored game."""
    _context.update({k: v for k, v in values.items() if v is not None})


def _log(game, create=False):
    log = getattr(game, '_library_log', None)
    if log is None and create:
        log = _Log()
        game._library_log = log
    return log


def tag(game, **values):
    """Attach metadata (phase, source, episode, default_behavior, ...) to a game."""
    log = _log(game, create=True)
    log.meta.update({k: v for k, v in values.items() if v is not None})


def annotate(game, behavior=None, update=None, target=None):
    """Label the next placement committed on ``game``."""
    _log(game, create=True).pending = (behavior, update, target)


def ignore(game):
    """Do not store this game (pure replays of games that are already recorded)."""
    _log(game, create=True).ignore = True


# Engine hooks -------------------------------------------------------------------------

def start(game):
    log = _log(game)
    if log is None:
        game._library_log = _Log()
    else:  # reset(): keep tags for the replacement game
        meta, ignored = log.meta, log.ignore
        game._library_log = _Log()
        game._library_log.meta, game._library_log.ignore = meta, ignored
    _live.add(game)


def moved(game, move):
    if _library is None:
        return
    log = _log(game, create=True)
    candidate = move.candidate
    behavior, update, target = log.pending or (None, None, None)
    log.pending = None
    names = _engine.PIECE_NAMES
    log.moves.extend((names.index(move.piece), candidate.rotation, candidate.x, candidate.y,
                      candidate.z, candidate.w, UNKNOWN if target is None else int(target),
                      int(move.layers)))
    log.update.append(-1 if update is None else int(update))
    log.behavior.append(log.code(behavior if behavior is not None else log.meta.get('default_behavior')))
    if game.game_over:
        finish(game, 'top_out')


def retire(game, reason):
    log = _log(game)
    if log is not None and not log.done and len(log.update):
        finish(game, reason)


def finish(game, stop_reason=None, **meta):
    """Store ``game`` now with an explicit stop reason; later hooks will not store it again."""
    log = _log(game)
    if log is None or log.done or log.ignore or not len(log.update):
        if log is not None and meta:
            log.meta.update(meta)
        return None
    log.done = True
    library = _library
    if library is None:
        return None
    engine = _engine_module()
    try:
        record = {**_context, **{k: v for k, v in log.meta.items() if k != 'default_behavior'}, **meta,
                  'seed': int(game.seed), 'piece_set': game.piece_set,
                  'pieces_sha256': engine_fingerprint(engine), 'engine_file': Path(engine.__file__).name,
                  'game_over': bool(game.game_over), 'stop_reason': stop_reason or ('top_out' if game.game_over else 'unknown'),
                  'final_board_sha256': _board_sha256(game.board), 'hyperlayers': int(game.layers),
                  'score': int(game.score), 'behaviors': list(log.behaviors),
                  'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'process_id': os.getpid()}
        moves = np.frombuffer(bytes(log.moves), dtype=np.uint8).reshape(-1, len(_COLUMNS))
        return library.store(record, moves, np.asarray(log.update, dtype=np.int32),
                             np.frombuffer(bytes(log.behavior), dtype=np.uint8))
    except (OSError, ValueError, TypeError) as error:
        library.error = f'{type(error).__name__}: {error}'
        print(f'Game library recording failed: {library.error}', file=sys.stderr, flush=True)
        return None


def flush_all(reason='process_exit'):
    for game in list(_live):
        try:
            retire(game, reason)
        except Exception:  # never fail interpreter shutdown
            pass


def released(game):
    """Called from Game.__del__: store an unfinished game that is being discarded."""
    if not sys.is_finalizing():
        retire(game, 'released')


_FINGERPRINTS = {}


def engine_fingerprint(engine=None):
    engine = engine or _engine_module()
    if id(engine) in _FINGERPRINTS:
        return _FINGERPRINTS[id(engine)]
    catalogue = {name: [sorted(shape.cells) for shape in engine.SHAPES[name]] for name in engine.PIECE_NAMES}
    payload = json.dumps({'names': list(engine.PIECE_NAMES), 'sets': {k: list(v) for k, v in engine.PIECE_SETS.items()},
                          'shapes': catalogue}, sort_keys=True).encode()
    _FINGERPRINTS[id(engine)] = hashlib.sha256(payload).hexdigest()
    return _FINGERPRINTS[id(engine)]


import atexit  # noqa: E402  (registered after definitions)
atexit.register(flush_all)


# Reuse ------------------------------------------------------------------------------------

@dataclass
class GameRecord:
    meta: dict
    moves: np.ndarray      # [placements, 8]: piece, rotation, x, y, z, w, target, clears
    update: np.ndarray     # policy update per placement (-1 unknown)
    behavior: np.ndarray   # index into meta['behaviors'] (255 unknown)
    path: Path

    def remaining_placements(self):
        """Exact remaining successful placements after each position, or None if censored."""
        if not self.meta.get('game_over'):
            return None
        return np.arange(len(self.moves), 0, -1, dtype=np.int64)


def load(path) -> GameRecord:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['meta']))
        if meta.get('schema') != SCHEMA:
            raise ValueError(f'Unsupported game library schema in {path}')
        return GameRecord(meta, data['moves'], data['update'], data['behavior'], Path(path))


def iter_index(root=None, unique=True):
    root = Path(root) if root is not None else default_root()
    seen = set()
    index = root / 'index.jsonl'
    if not index.exists():
        return
    with index.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            entry = json.loads(line)
            if unique and entry['id'] in seen:
                continue
            seen.add(entry['id'])
            yield entry


def iter_games(root=None, **filters):
    """Yield GameRecords whose index entries match all keyword filters (e.g. phase='training')."""
    root = Path(root) if root is not None else default_root()
    for entry in iter_index(root):
        if all(entry.get(key) == value for key, value in filters.items()):
            yield load(root / entry['file'])


def replay(record: GameRecord, verify=True):
    """Yield (state, move, row) before each placement; the state is committed after resuming.

    The yielded state is the live engine Game: treat it as read-only. The replayed game
    itself is never re-recorded.
    """
    engine = _engine_module()
    if record.meta.get('pieces_sha256') != engine_fingerprint(engine):
        raise ValueError('Recorded piece catalogue differs from this engine')
    state = engine.Game(int(record.meta['seed']), record.meta['piece_set'])
    ignore(state)
    names = engine.PIECE_NAMES
    for index, row in enumerate(record.moves):
        piece = names[int(row[0])]
        if piece != state.current:
            raise ValueError(f'Placement {index}: recorded piece {piece} but stream gives {state.current}')
        move = engine.make_move(state.board, piece, int(row[1]), int(row[2]), int(row[3]), int(row[5]))
        if move is None or move.candidate.z != int(row[4]) or move.layers != int(row[7]):
            raise ValueError(f'Placement {index} does not replay on this engine')
        yield state, move, row
        state.commit(move)
    if verify and _board_sha256(state.board) != record.meta['final_board_sha256']:
        raise ValueError('Replayed final board differs from the recording')


# Backfill of legacy recordings -------------------------------------------------------------

def import_legacy(path, library=None):
    """Import a tetris4d-training-game-v1 JSON(.gz) recording; returns the stored id."""
    library = library or active()
    if library is None:
        raise ValueError('Game library is disabled')
    path = Path(path)
    raw = path.read_bytes()
    text = gzip.decompress(raw).decode('utf-8') if path.suffix == '.gz' else raw.decode('utf-8')
    data = json.loads(text)
    if data.get('schema') != 'tetris4d-training-game-v1':
        raise ValueError(f'Not a legacy training recording: {path}')
    engine = _engine_module()
    state = engine.Game(int(data['seed']), data['piece_set'])
    ignore(state)
    log = _Log()
    names = engine.PIECE_NAMES
    for row in data['moves']:
        move = engine.make_move(state.board, state.current, int(row['rotation']), int(row['x']), int(row['y']), int(row['w']))
        if move is None or move.candidate.z != int(row['z']):
            raise ValueError(f'Legacy recording {path} does not replay at placement {state.pieces}')
        log.moves.extend((names.index(move.piece), move.candidate.rotation, move.candidate.x, move.candidate.y,
                          move.candidate.z, move.candidate.w, int(row.get('target_before', UNKNOWN)) & 255,
                          int(move.layers)))
        log.update.append(int(row.get('update', -1)))
        log.behavior.append(log.code(row.get('behavior')))
        state.commit(move)
    context = data.get('run_context') or {}
    meta = {'seed': int(data['seed']), 'piece_set': data['piece_set'], 'pieces_sha256': engine_fingerprint(engine),
            'engine_file': Path(engine.__file__).name, 'game_over': bool(state.game_over),
            'stop_reason': data.get('stop_reason') or ('top_out' if state.game_over else 'unknown'),
            'final_board_sha256': _board_sha256(state.board), 'hyperlayers': int(state.layers),
            'score': int(state.score), 'behaviors': list(log.behaviors),
            'phase': data.get('phase'), 'episode': data.get('number'),
            'source': context.get('algorithm', 'legacy-recording'), 'session': context.get('training_session'),
            'recorded_utc': data.get('recorded_utc'), 'imported_from': str(path.resolve()),
            'imported_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'complete': bool(data.get('complete'))}
    moves = np.frombuffer(bytes(log.moves), dtype=np.uint8).reshape(-1, len(_COLUMNS))
    return library.store(meta, moves, np.asarray(log.update, dtype=np.int32),
                         np.frombuffer(bytes(log.behavior), dtype=np.uint8))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--root', type=Path, help='Library directory (default: TETRIS4D_GAME_LIBRARY or data/game-library)')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('stats', help='Summarize recorded games')
    check = commands.add_parser('verify', help='Replay games and verify their final boards')
    check.add_argument('--limit', type=int, default=0)
    legacy = commands.add_parser('import', help='Backfill legacy JSON(.gz) game recordings')
    legacy.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args(argv)
    library = configure(args.root) if args.root else configure()
    if library is None:
        parser.error('Game library is disabled (TETRIS4D_GAME_LIBRARY=off)')
    root = library.root
    if args.command == 'stats':
        entries = list(iter_index(root, unique=False))
        unique = {entry['id']: entry for entry in entries}
        size = sum(p.stat().st_size for p in (root / 'games').rglob('*.npz')) if (root / 'games').exists() else 0
        summary = {'root': str(root), 'played_games': len(entries), 'distinct_games': len(unique),
                   'placements': sum(e['placements'] for e in unique.values()),
                   'game_over': sum(bool(e.get('game_over')) for e in unique.values()),
                   'by_phase': dict(Counter(str(e.get('phase')) for e in entries)),
                   'by_source': dict(Counter(str(e.get('source')) for e in entries)),
                   'bytes': size}
        print(json.dumps(summary, indent=2))
    elif args.command == 'verify':
        checked = 0
        for entry in iter_index(root):
            record = load(root / entry['file'])
            for _ in replay(record):
                pass
            checked += 1
            if args.limit and checked >= args.limit:
                break
        print(json.dumps({'verified_games': checked}))
    else:
        files = []
        for path in args.paths:
            files.extend(sorted(p for p in path.rglob('*') if p.name.endswith(('.json', '.json.gz')))
                         if path.is_dir() else [path])
        imported, failed = 0, []
        for path in files:
            try:
                import_legacy(path, library)
                imported += 1
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
                failed.append({'path': str(path), 'error': str(error)[:200]})
        print(json.dumps({'imported': imported, 'stored': library.stored, 'duplicates': library.duplicates,
                          'failed': len(failed), 'failures': failed[:20]}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
