from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

SCHEMA = 'tetris4d-benchmark-v1'


def _safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] / 1_000_000


class Timing:
    def __init__(self):
        self.count = self.wall_ns = self.cpu_ns = self.max_ns = 0
        self.recent = deque(maxlen=2048)

    def add(self, wall_ns, cpu_ns=0):
        self.count += 1
        self.wall_ns += wall_ns
        self.cpu_ns += cpu_ns
        self.max_ns = max(self.max_ns, wall_ns)
        self.recent.append(wall_ns)

    def report(self):
        return {'count': self.count, 'total_wall_ms': self.wall_ns / 1_000_000,
                'total_process_cpu_ms': self.cpu_ns / 1_000_000,
                'mean_wall_ms': self.wall_ns / self.count / 1_000_000 if self.count else None,
                'max_wall_ms': self.max_ns / 1_000_000,
                'recent_samples': len(self.recent),
                'recent_p50_wall_ms': _percentile(self.recent, .50),
                'recent_p95_wall_ms': _percentile(self.recent, .95),
                'recent_p99_wall_ms': _percentile(self.recent, .99)}


class BenchmarkLog:
    def __init__(self, game, path=None, settings=None, max_bytes=4 * 1024 * 1024, backups=3):
        self.game = game
        self.session = uuid.uuid4().hex
        self.started_ns = time.perf_counter_ns()
        self.seq = 0
        self.metrics = {}
        self.patches = []
        self.stream = None
        self.enabled = True
        self.closed = False
        self.in_snapshot = False
        self.bytes = self.pending = self.io_ns = 0
        self.decisions = self.moves = 0
        self.last_flush_ns = self.last_snapshot_ns = self.started_ns
        self.last_tick_ns = None
        self.app = None
        self.games = {}
        self.game_serial = 0
        self.settings = dict(settings or {})
        self.max_bytes, self.backups = max(256, int(max_bytes)), max(0, int(backups))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        root = Path(game.__file__).resolve().parent
        self.path = Path(path) if path is not None else root / 'logs' / f'benchmark-{stamp}-{os.getpid()}-{self.session[:8]}.jsonl'
        self.metadata = self._environment(root)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open('x', encoding='utf-8', buffering=65536)
            self.emit('session_start', metadata=self.metadata, settings=self.settings,
                      timing_semantics='Inclusive phase timings overlap; do not sum them. Search work excludes generator suspension. GUI draw is CPU submission, not display FPS. Percentiles cover the last 2048 observations per phase.',
                      max_bytes=self.max_bytes, backups=self.backups)
            self.flush()
        except OSError as error:
            self._disable(error)

    def _environment(self, root):
        cpu = {}
        try:
            for line in Path('/proc/cpuinfo').read_text(encoding='utf-8').split('\n\n', 1)[0].splitlines():
                key, _, value = line.partition(':')
                if key.strip() in ('model name', 'vendor_id', 'flags'):
                    cpu[key.strip()] = value.strip()
        except OSError:
            pass
        native = self._backend()
        build = {}
        for suffix in ('.so.json', '.dylib.json'):
            try:
                data = json.loads((root / 'native' / ('libtetris4d_native' + suffix)).read_text(encoding='utf-8'))
                build = {key: data.get(key) for key in ('abi', 'compiler', 'source_sha256', 'binary_sha256', 'sanitized')}
                build['compiler_flags'] = [arg for arg in data.get('command', []) if str(arg).startswith('-') and arg != '-o']
                break
            except (OSError, ValueError, TypeError):
                continue
        parameters = self.game.network_parameters()
        return {'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'platform': platform.platform(), 'machine': platform.machine(), 'cpu': cpu,
                'numpy': getattr(getattr(self.game, '_NUMPY', None), '__version__', None),
                'native': native, 'build': build,
                'source_sha256': _digest(self.game.__file__),
                'native_source_sha256': _digest(root / 'native' / 'tetris_native.c'),
                'logger_sha256': _digest(__file__), 'parameters': parameters,
                'parameters_sha256': hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest(),
                'clock': {'wall': 'perf_counter_ns', 'cpu': 'process_time_ns',
                          'wall_resolution_seconds': time.get_clock_info('perf_counter').resolution}}

    def _backend(self):
        info = getattr(self.game, 'native_backend_info', lambda: {})()
        keys = ('available', 'abi', 'isa', 'avx2_available', 'bmi2_available', 'avx2_enabled', 'bmi2_enabled')
        return {'selected': self.game.BACKEND, **{key: info[key] for key in keys if key in info}}

    def _disable(self, error):
        if self.enabled:
            print(f'Benchmark logging disabled ({type(error).__name__}); gameplay continues.', file=sys.stderr)
        self.enabled = False
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError:
                pass
        self.stream = None

    def _line(self, event, payload):
        self.seq += 1
        return json.dumps(_safe({'schema': SCHEMA, 'session': self.session, 'seq': self.seq,
                                  'event': event, 'elapsed_ns': time.perf_counter_ns() - self.started_ns,
                                  **payload}), separators=(',', ':'), allow_nan=False) + '\n'

    def emit(self, event, **payload):
        if not self.enabled or self.closed:
            return
        begin = time.perf_counter_ns()
        try:
            text = self._line(event, payload)
            size = len(text.encode('utf-8'))
            if self.bytes and self.bytes + size > self.max_bytes:
                self.stream.close()
                for number in range(self.backups, 0, -1):
                    destination = self.path.with_name(self.path.name + f'.{number}')
                    source = self.path if number == 1 else self.path.with_name(self.path.name + f'.{number - 1}')
                    if source.exists():
                        source.replace(destination)
                self.stream = self.path.open('w', encoding='utf-8', buffering=65536)
                self.bytes = 0
                header = self._line('segment_start', {'metadata': self.metadata, 'settings': self.settings})
                self.stream.write(header)
                self.bytes = len(header.encode('utf-8'))
                text = self._line(event, payload)
                size = len(text.encode('utf-8'))
            self.stream.write(text)
            self.bytes += size
            self.pending += 1
            if self.pending >= 64 or time.perf_counter_ns() - self.last_flush_ns >= 5_000_000_000:
                self.flush()
        except (OSError, ValueError, TypeError) as error:
            self._disable(error)
        finally:
            self.io_ns += time.perf_counter_ns() - begin

    def flush(self):
        if not self.enabled or self.closed or self.stream is None:
            return
        try:
            self.stream.flush()
            self.pending = 0
            self.last_flush_ns = time.perf_counter_ns()
        except OSError as error:
            self._disable(error)

    def observe(self, name, wall_ns, cpu_ns=0):
        if self.enabled and not self.in_snapshot:
            metric = self.metrics.get(name)
            if metric is None:
                metric = self.metrics[name] = Timing()
            metric.add(wall_ns, cpu_ns)

    def snapshot(self, reason='periodic'):
        if not self.enabled or self.in_snapshot:
            return
        self.in_snapshot = True
        try:
            caches = {}
            for name in ('evaluate_board', 'target_layer_status', 'pooled_histograms', 'sigmoid_roughness'):
                function = getattr(self.game, name, None)
                if hasattr(function, 'cache_info'):
                    caches[name] = function.cache_info()._asdict()
            gui = None
            if self.app is not None and not self.app.closed:
                gui = {'running': self.app.running, 'human': self.app.human.get(),
                       'speed': self.app.speed.get(), 'policy': self.app.policy.get(),
                       'target': int(self.app.target.get()), 'w_slice': self.app.w_slice.get(),
                       'cutoff': self.app.cutoff.get(), 'pieces': self.app.game.pieces,
                       'hyperlayers': self.app.game.layers, 'game_over': self.app.game.game_over}
                if hasattr(self.app.well, 'views'):
                    gui.update(visible_w_sections=list(self.app.well.visible_w_sections))
            rendering = self.app.render_metrics() if self.app is not None and hasattr(self.app, 'render_metrics') else None
            self.emit('metrics', reason=reason, backend=self._backend(), gui=gui, rendering=rendering,
                      decisions=self.decisions, moves=self.moves, caches=caches,
                      write_and_serialization_ms=self.io_ns / 1_000_000,
                      phases={key: value.report() for key, value in self.metrics.items()})
            self.last_snapshot_ns = time.perf_counter_ns()
            self.flush()
        finally:
            self.in_snapshot = False

    def _periodic(self):
        if time.perf_counter_ns() - self.last_snapshot_ns >= 5_000_000_000:
            self.snapshot()

    def _patch(self, owner, name, wrapper):
        original = getattr(owner, name)
        for attribute in ('cache_clear', 'cache_info', 'cache_parameters'):
            if hasattr(original, attribute):
                setattr(wrapper, attribute, getattr(original, attribute))
        self.patches.append((owner, name, original))
        setattr(owner, name, wrapper)

    def _timed(self, owner, name, label):
        original = getattr(owner, name)
        @wraps(original)
        def call(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            start, cpu = time.perf_counter_ns(), time.process_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self.observe(label, time.perf_counter_ns() - start, time.process_time_ns() - cpu)
        self._patch(owner, name, call)

    def _search(self, name):
        original = getattr(self.game, name)
        @wraps(original)
        def steps(*args, **kwargs):
            iterator = original(*args, **kwargs)
            latency_start = time.perf_counter_ns()
            wall_total = cpu_total = slices = 0
            try:
                while True:
                    start, cpu = time.perf_counter_ns(), time.process_time_ns()
                    try:
                        value = next(iterator)
                    except StopIteration as finished:
                        wall_total += time.perf_counter_ns() - start
                        cpu_total += time.process_time_ns() - cpu
                        slices += 1
                        decision = finished.value
                        self.decisions += 1
                        self.observe('search_active_work', wall_total, cpu_total)
                        latency = time.perf_counter_ns() - latency_start
                        self.observe('search_elapsed_latency', latency)
                        self.emit('decision', index=self.decisions,
                                  backend=self._backend(), policy='hyperlayer' if name.startswith('hyperlayer') else 'network',
                                  piece=args[1] if len(args) > 1 else kwargs.get('piece'),
                                  work_wall_ns=wall_total, process_cpu_ns=cpu_total,
                                  elapsed_latency_ns=latency, slices=slices, found=decision is not None,
                                  evaluations=decision.evaluations if decision else None,
                                  coverage_gain=decision.coverage_gain if decision else None,
                                  target=decision.target if decision else None,
                                  depth=decision.depth if decision else None,
                                  beam=decision.beam if decision else None,
                                  candidates=decision.candidates if decision else None,
                                  sampled=decision.sampled if decision else None,
                                  exhaustive_fallback=decision.exhaustive_fallback if decision else None)
                        self._periodic()
                        return decision
                    except Exception as error:
                        self.emit('error', phase=name, exception=type(error).__name__)
                        self.flush()
                        raise
                    wall_total += time.perf_counter_ns() - start
                    cpu_total += time.process_time_ns() - cpu
                    slices += 1
                    yield value
            finally:
                iterator.close()
        self._patch(self.game, name, steps)

    def _game_state(self, agent):
        key = id(agent)
        if key not in self.games:
            self.game_serial += 1
            self.games[key] = {'id': self.game_serial, 'digest': hashlib.sha256()}
            self.emit('game_start', game=self.game_serial, seed=agent.seed,
                      piece_set=agent.piece_set, backend=self._backend(), parameters=self.game.network_parameters())
        return self.games[key]

    def install(self):
        if not self.enabled:
            return self
        for name, label in (('evaluate_board', 'board_evaluation'), ('native_scores', 'native_batch_prepare_and_score'),
                            ('_native_features', 'native_feature_scan'), ('materialize_move', 'move_materialization'),
                            ('network_trace', 'activation_trace'), ('target_layer_status', 'target_status')):
            if hasattr(self.game, name):
                self._timed(self.game, name, label)
        self._timed(self.game.PlacementBatch, '__init__', 'batch_preparation')
        for name in ('decision_steps', 'hyperlayer_decision_steps'):
            self._search(name)
        original_commit = self.game.Game.commit
        @wraps(original_commit)
        def commit(agent, move):
            state = self._game_state(agent)
            start, cpu = time.perf_counter_ns(), time.process_time_ns()
            result = original_commit(agent, move)
            self.observe('commit_and_validation', time.perf_counter_ns() - start, time.process_time_ns() - cpu)
            self.moves += 1
            p = move.candidate
            state['digest'].update(f'{p.piece}:{p.rotation}:{p.x}:{p.y}:{p.w}:{p.z}:{p.layers};'.encode())
            if move.layers or agent.pieces % 100 == 0 or agent.game_over:
                event = 'clear' if move.layers else 'game_over' if agent.game_over else 'progress'
                self.emit(event, game=state['id'], pieces=agent.pieces, hyperlayers=agent.layers,
                          cells_placed=agent.cells_placed, game_over=agent.game_over,
                          move_digest=state['digest'].hexdigest())
                self.snapshot(event)
            return result
        self._patch(self.game.Game, 'commit', commit)
        original_reset = self.game.Game.reset
        @wraps(original_reset)
        def reset(agent, *args, **kwargs):
            self.games.pop(id(agent), None)
            result = original_reset(agent, *args, **kwargs)
            self._game_state(agent)
            return result
        self._patch(self.game.Game, 'reset', reset)
        original_run = self.game.run_game
        @wraps(original_run)
        def run(*args, **kwargs):
            result = original_run(*args, **kwargs)
            self.emit('game_end', summary=result)
            self.snapshot('game_end')
            return result
        self._patch(self.game, 'run_game', run)
        self._timed(self.game.TetrisApp, '_draw_now', 'gui_draw_submission')
        for class_name, method, label in (('WellView', 'draw', 'gui_well_submission'),
                ('SliceAtlas', 'draw', 'gui_atlas_submission'), ('NetworkView', 'update', 'gui_network_update')):
            owner = getattr(self.game, class_name, None)
            if owner is not None and hasattr(owner, method):
                self._timed(owner, method, label)
        if hasattr(self.game, 'AllWellView'):
            self._timed(self.game.AllWellView, 'draw', 'gui_all_views_submission')
        original_tick = self.game.TetrisApp.tick
        @wraps(original_tick)
        def tick(app, *args, **kwargs):
            self.app = app
            now, cpu = time.perf_counter_ns(), time.process_time_ns()
            if self.last_tick_ns is not None:
                self.observe('gui_tick_interval', now - self.last_tick_ns)
            self.last_tick_ns = now
            try:
                return original_tick(app, *args, **kwargs)
            finally:
                self.observe('gui_tick_work', time.perf_counter_ns() - now, time.process_time_ns() - cpu)
                self._periodic()
        self._patch(self.game.TetrisApp, 'tick', tick)
        original_error = self.game.TetrisApp.callback_error
        @wraps(original_error)
        def callback_error(app, kind, value, traceback):
            self.emit('error', phase='gui_callback', exception=kind.__name__)
            self.flush()
            return original_error(app, kind, value, traceback)
        self._patch(self.game.TetrisApp, 'callback_error', callback_error)
        return self

    def close(self):
        if self.closed:
            return
        if self.app is not None:
            try:
                self.emit('game_end', summary=self.app.game.summary())
            except (ValueError, RuntimeError, AttributeError):
                pass
        self.snapshot('session_end')
        self.emit('session_end', decisions=self.decisions, moves=self.moves,
                  move_digests=[{'game': value['id'], 'digest': value['digest'].hexdigest()} for value in self.games.values()])
        self.flush()
        for owner, name, original in reversed(self.patches):
            setattr(owner, name, original)
        self.patches.clear()
        if self.stream is not None:
            try:
                self.stream.close()
            except OSError as error:
                self._disable(error)
        self.stream = None
        self.closed = True


def summarize(paths):
    sessions = {}
    malformed = 0
    for path in paths:
        with Path(path).open(encoding='utf-8') as source:
            for line in source:
                try:
                    row = json.loads(line)
                    if row.get('schema') != SCHEMA:
                        continue
                    record = sessions.setdefault(row['session'], {'metadata': {}, 'settings': {}, 'metrics': {},
                        'clears': {}, 'completed': False, 'last_metrics_seq': -1, 'events': 0})
                    record['events'] += 1
                    event = row['event']
                    if event in ('session_start', 'segment_start'):
                        record['metadata'] = row['metadata']
                        record['settings'] = row['settings']
                    elif event == 'metrics' and row['seq'] > record['last_metrics_seq']:
                        record['metrics'] = row
                        record['last_metrics_seq'] = row['seq']
                    elif event == 'clear':
                        record['clears'][str(row['seq'])] = row
                    elif event == 'session_end':
                        record['completed'] = True
                except (ValueError, KeyError, TypeError):
                    malformed += 1
    for record in sessions.values():
        record['clears'] = sorted(record['clears'].values(), key=lambda r: r['seq'])
        record.pop('last_metrics_seq', None)
    return {'schema': SCHEMA, 'sessions': sessions, 'malformed_lines': malformed,
            'note': 'Phase totals are inclusive and overlap. Recent percentiles are bounded-window samples. GUI submission/tick timing is not displayed FPS. Rotated-away events and unflushed crash tails may be absent.'}


def main():
    parser = argparse.ArgumentParser(description='Summarize Tetris4D benchmark JSONL files and their rotated segments.')
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.logs), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
