"""Local, read-only training telemetry server; no training/runtime imports."""
from __future__ import annotations
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from urllib.parse import urlsplit

MAX_BYTES = 8 * 2**20
MAX_LINE = 256 * 1024
MAX_READ = 4 * 2**20
STATIC = Path(__file__).with_suffix('')
NUMBERS = ('runner_updates', 'placements', 'optimizer_steps', 'finished_episodes', 'clears',
           'entropy', 'kl', 'policy_loss', 'value_loss', 'teacher_agreement',
           'h2_saturated_fraction', 'h2_candidate_std', 'lower_primary_probability',
           'episode', 'seed', 'episode_pieces', 'episode_clears', 'selection_game_pieces',
           'selection_mean_pieces', 'selection_minimum_pieces', 'selection_median_pieces',
           'calibration_game_index', 'calibration_game_pieces', 'calibration_total_games',
           'calibration_fit_updates', 'calibration_validation_mse', 'calibration_constant_mse')
FIELDS = ('event_id', 'event', 'timestamp_utc', 'elapsed_seconds', 'session_elapsed_seconds',
          'elapsed_seconds_basis', 'session', *NUMBERS, 'policy', 'stage', 'stop_reason',
          'calibration_split', 'calibration_terminated', 'calibration_truncated',
          'calibration_passed', 'best_updated', 'source_offset', 'source_end', 'source_line_sha256')
# Optional columns extend new rows/downloads without changing existing provenance.
OPTIONAL_NUMBERS = ('reference_kl', 'final_reference_kl', 'reference_kl_weight',
                    'critic_bootstrapped_explained_variance')
OPTIONAL_FIELDS = ('actor_changed_from_initial', 'teacher_placements', *OPTIONAL_NUMBERS)
CSV_FIELDS = (*FIELDS, *OPTIONAL_FIELDS)


def utc():
    return datetime.now(timezone.utc).isoformat()


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def read_json(path):
    if path.stat().st_size > 2 * 2**20:
        raise ValueError('metadata exceeds 2 MiB')
    def reject(value):
        raise ValueError('non-finite JSON: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


class Collector:
    def __init__(self, run_dir, data_dir, prior_data_dir=None):
        self.run_dir = Path(run_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        if self.data_dir == self.run_dir or self.data_dir.is_relative_to(self.run_dir):
            raise ValueError('Telemetry data must be outside the training run directory')
        self.metrics = self.run_dir / 'metrics.jsonl'
        self.control = self.run_dir.parent
        self.history = self.data_dir / 'history.jsonl'
        self.prior_history = None
        self.prior_proof = None
        self.manifest = self.data_dir / 'manifest.json'
        self.lock = threading.RLock()
        self.offset = 0
        self.identity = None
        self.errors = deque(maxlen=20)
        self.frozen_reason = None
        self.rows = deque(maxlen=10000)
        self.events = deque(maxlen=30)
        self.seen = set()
        self.total_rows = 0
        self.last_own_row = None
        self.completed_episodes = 0
        self.completed_record = None
        self.survival_records = []
        self.progress = {key: None for key in NUMBERS[:5]}
        self.latest = {'timestamp_utc': None, 'actor_changed_from_initial': None,
                       **{key: None for key in (*NUMBERS[5:14], *OPTIONAL_NUMBERS)}}
        self.actor_observed_offset = -1
        self.optional_observed_offsets = {key: -1 for key in OPTIONAL_NUMBERS}
        self.calibration = {'stage': None, 'completed_games': 0, 'finished_games': 0, 'censored_games': 0,
                            'total_games': None, 'current_game': None, 'current_game_index': None,
                            'current_pieces': None, 'passed': None, 'train_mse': None, 'validation_mse': None}
        self.calibration_games = {}
        self.best = None
        self.event_phase = None
        self.budget_data = {}
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.history, self.manifest):
            if path.is_symlink():
                raise ValueError('Telemetry outputs must not be symlinks')
        expected = {'schema': 'tetris4d-dashboard-history-v1', 'metrics': str(self.metrics), 'fields': list(FIELDS)}
        if prior_data_dir is not None:
            prior = Path(prior_data_dir).resolve()
            if prior == self.data_dir:
                raise ValueError('Prior history must be separate from the writable data directory')
            self.prior_history = prior/'history.jsonl'
            prior_manifest = prior/'manifest.json'
            if any(not path.is_file() or path.is_symlink() for path in (self.prior_history,prior_manifest)):
                raise ValueError('Prior history requires regular history and provenance files')
            if self.prior_history.stat().st_size > MAX_BYTES:
                raise ValueError('Prior history exceeds the 8 MiB read bound')
            if read_json(prior_manifest).get('schema') != expected['schema']:
                raise ValueError('Prior history provenance schema differs')
            self.prior_proof = {'path':str(self.prior_history),'bytes':self.prior_history.stat().st_size,
                                'sha256':hashlib.sha256(self.prior_history.read_bytes()).hexdigest()}
            expected['prior_history'] = self.prior_proof
        if self.manifest.exists():
            if read_json(self.manifest) != expected:
                raise ValueError('Existing telemetry history belongs to a different source/schema')
        elif self.history.exists():
            raise ValueError('Existing history without its provenance will not be overwritten')
        else:
            with self.manifest.open('x', encoding='utf-8') as stream:
                json.dump(expected, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        for history in (self.prior_history,self.history):
            if history is None or not history.exists():
                continue
            if history.stat().st_size > MAX_BYTES:
                self.freeze('Existing history exceeds the 8 MiB output bound')
            else:
                file_seen = set()
                with history.open('rb') as stream:
                    for line in stream:
                        try:
                            if not line.endswith(b'\n'):
                                raise ValueError('incomplete final history line')
                            row = json.loads(line)
                            if row['event_id'] in file_seen:
                                raise ValueError('duplicate durable history identity')
                            file_seen.add(row['event_id'])
                            if row['event_id'] not in self.seen:
                                if history == self.history:
                                    self.last_own_row = row
                                self.seen.add(row['event_id']); self.rows.append(row); self.apply(row)
                        except (ValueError, KeyError, TypeError) as error:
                            self.freeze('History preserved and frozen: ' + str(error)); break
            if history == self.prior_history:
                # Prior terminal events are evidence, not the new process's phase.
                self.event_phase = None
                self.actor_observed_offset = -1
                self.optional_observed_offsets = {key: -1 for key in OPTIONAL_NUMBERS}
                self.latest.update({key: None for key in OPTIONAL_NUMBERS})

    def error(self, text):
        if text not in self.errors:
            self.errors.append(text)

    def freeze(self, reason):
        self.frozen_reason = self.frozen_reason or reason
        self.error(reason)

    def artifact(self, path):
        if not path.exists():
            return {}
        try:
            if path.is_symlink():
                raise ValueError('symlink input rejected')
            return read_json(path)
        except (OSError, ValueError, TypeError) as error:
            self.error(path.name + ': ' + str(error)); return {}

    def normalize(self, raw, start, end, line):
        row = {key: None for key in CSV_FIELDS}
        event = raw.get('event')
        if not isinstance(event, str):
            raise ValueError('metric event name is missing')
        checksum = hashlib.sha256(line).hexdigest()
        row.update(event_id=hashlib.sha256((str(self.metrics)+':'+str(start)+':'+checksum).encode()).hexdigest(),
                   event=event, timestamp_utc=raw.get('utc'), session=raw.get('session'),
                   session_elapsed_seconds=number(raw.get('elapsed_seconds')),
                   source_offset=start, source_end=end, source_line_sha256=checksum)
        row['elapsed_seconds'] = row['session_elapsed_seconds']
        row['elapsed_seconds_basis'] = 'source_session_elapsed'
        if row['timestamp_utc'] and number(self.budget_data.get('start_epoch')) is not None:
            try:
                stamp = datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00')).timestamp()
                row['elapsed_seconds'] = max(0, stamp-self.budget_data['start_epoch'])
                row['elapsed_seconds_basis'] = 'utc_since_budget_start'
            except (ValueError, TypeError):
                self.error('Invalid metric UTC timestamp; using source session elapsed time')
        for key in NUMBERS:
            row[key] = number(raw.get(key))
        for key in ('policy', 'stage', 'stop_reason', 'best_updated'):
            row[key] = raw.get(key)
        row['kl'] = number(raw.get('final_exact_kl', raw.get('kl')))
        diagnostic = raw.get('actor_diagnostics') or {}
        for key in ('h2_saturated_fraction', 'h2_candidate_std', 'lower_primary_probability'):
            row[key] = number(diagnostic.get(key))
        changed = diagnostic.get('actor_changed_from_initial')
        row['actor_changed_from_initial'] = changed if type(changed) is bool else None
        row['teacher_placements'] = number(raw.get('teacher_placements'))
        for key in OPTIONAL_NUMBERS:
            row[key] = number(raw.get(key))
        critic = raw.get('critic_bootstrapped_diagnostics')
        if isinstance(critic, dict):
            row['critic_bootstrapped_explained_variance'] = number(critic.get('explained_variance'))
        if event == 'episode_end':
            row.update(episode_pieces=number(raw.get('pieces')), episode_clears=number(raw.get('hyperlayers')))
        if event == 'selection_game':
            row['selection_game_pieces'] = number(raw.get('pieces'))
        if event == 'selection_complete' and isinstance(raw.get('score'), list) and len(raw['score']) == 3:
            for key, value in zip(('selection_mean_pieces', 'selection_minimum_pieces', 'selection_median_pieces'), raw['score']):
                row[key] = number(value)
        if event in ('calibration_progress', 'calibration_game'):
            row['calibration_game_pieces'] = number(raw.get('pieces'))
            row['calibration_game_index'] = number(raw.get('game'))
            row['calibration_total_games'] = number(raw.get('total_games'))
            for source, dest in (('split','calibration_split'), ('terminated','calibration_terminated'), ('truncated','calibration_truncated')):
                row[dest] = raw.get(source)
        if event in ('critic_calibration', 'calibration_complete'):
            metrics = raw.get('validation', raw)
            row.update(calibration_fit_updates=number(raw.get('fit_updates')),
                       calibration_validation_mse=number(metrics.get('mse')),
                       calibration_constant_mse=number(metrics.get('constant_mse')),
                       calibration_passed=raw.get('passed'))
        return row

    def apply(self, row):
        event = row['event']
        self.total_rows += 1
        pieces = number(row.get('episode_pieces'))
        if (event == 'episode_end' and pieces is not None and pieces >= 0
                and row.get('policy') in (None,'neural','ppo')
                and row.get('teacher_placements') in (None,0)):
            self.completed_episodes += 1
            if self.completed_record is None or pieces > self.completed_record:
                self.completed_record = pieces
                self.survival_records.append({key:row.get(key) for key in ('timestamp_utc','seed','episode')}
                                             | {'pieces':pieces})
        self.apply_optional(row)
        for key in self.progress:
            if row.get(key) is not None:
                self.progress[key] = row[key]
        self.latest['timestamp_utc'] = row['timestamp_utc']
        for key in NUMBERS[5:14]:
            if row.get(key) is not None:
                self.latest[key] = row[key]
        cal = self.calibration
        if event == 'calibration_progress':
            cal.update(stage='collect', current_game_index=row['calibration_game_index'],
                       current_game=None if row['calibration_game_index'] is None else row['calibration_game_index']+1,
                       current_pieces=row['calibration_game_pieces'], total_games=row['calibration_total_games'])
        if event == 'calibration_game':
            self.calibration_games[row['seed']] = {key: row[key] for key in
                ('timestamp_utc','seed','calibration_game_pieces','calibration_split','calibration_terminated','calibration_truncated')}
            cal.update(stage='collect', finished_games=len(self.calibration_games),
                       completed_games=sum(game['calibration_terminated'] is True for game in self.calibration_games.values()),
                       censored_games=sum(game['calibration_truncated'] is True for game in self.calibration_games.values()),
                       current_game=None, current_game_index=None, current_pieces=None)
        if event in ('critic_calibration', 'calibration_complete'):
            cal.update(stage='fit' if event == 'critic_calibration' else 'passed' if row['calibration_passed'] else 'failed',
                       validation_mse=row['calibration_validation_mse'], constant_mse=row['calibration_constant_mse'],
                       fit_updates=row['calibration_fit_updates'])
            if event == 'calibration_complete':
                cal['passed'] = row['calibration_passed']
        if event == 'calibration_failed':
            cal.update(stage='failed', passed=False)
        if event == 'calibration_skipped':
            cal.update(stage='explicitly_skipped')
        if event == 'selection_complete' and (row['best_updated'] is True or self.best is None):
            self.best = {'mean_pieces':row['selection_mean_pieces'], 'minimum_pieces':row['selection_minimum_pieces'],
                         'median_pieces':row['selection_median_pieces'], 'observed_utc':row['timestamp_utc']}
        if event.startswith('calibration_') or event == 'critic_calibration':
            self.event_phase = 'calibration'
        if event in ('update','selection_complete') or event == 'calibration_complete' and row['calibration_passed']:
            self.event_phase = 'training'
        if event == 'selection_game':
            self.event_phase = 'selection'
        if event == 'run_finished':
            self.event_phase = 'complete'
        self.events.append({key:row[key] for key in ('event','timestamp_utc','runner_updates','episode_pieces',
            'selection_mean_pieces','calibration_game_pieces','stop_reason') if row.get(key) is not None})

    def apply_optional(self, row):
        offset = row.get('source_offset', -1)
        for key in OPTIONAL_NUMBERS:
            if (number(row.get(key)) is not None and type(offset) is int
                    and offset >= self.optional_observed_offsets[key]):
                self.latest[key] = row[key]
                self.optional_observed_offsets[key] = offset
        changed = row.get('actor_changed_from_initial')
        if type(changed) is bool and type(offset) is int and offset >= self.actor_observed_offset:
            self.latest['actor_changed_from_initial'] = changed
            self.actor_observed_offset = offset

    def append(self, row):
        if row['event_id'] in self.seen:
            return
        self.seen.add(row['event_id']); self.rows.append(row)
        self.last_own_row = row
        if self.frozen_reason:
            return
        encoded = (json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
        used = sum(path.stat().st_size for path in self.data_dir.iterdir() if path.is_file())
        if used+len(encoded) > MAX_BYTES:
            self.freeze('History frozen at the 8 MiB own-output limit; raw training metrics remain untouched')
            return
        with self.history.open('ab') as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())

    def poll(self):
        with self.lock:
            self.budget_data = self.budget()
            if not self.metrics.exists():
                return
            if self.metrics.is_symlink():
                self.freeze('Metrics symlink rejected'); return
            stat = self.metrics.stat(); identity = (stat.st_dev, stat.st_ino)
            if self.identity is None and self.last_own_row:
                last = self.last_own_row
                with self.metrics.open('rb') as stream:
                    stream.seek(last['source_offset'])
                    saved_line = stream.read(last['source_end']-last['source_offset'])
                if hashlib.sha256(saved_line).hexdigest() != last['source_line_sha256']:
                    self.freeze('Metrics source no longer matches durable history; history preserved'); return
            if self.identity is not None and (identity != self.identity or stat.st_size < self.offset):
                self.freeze('Metrics source was replaced or truncated; history preserved'); return
            self.identity = identity
            consumed = 0
            with self.metrics.open('rb') as stream:
                stream.seek(self.offset)
                while consumed < MAX_READ:
                    start = stream.tell(); line = stream.readline(MAX_LINE+1)
                    if not line:
                        break
                    if len(line) > MAX_LINE:
                        self.freeze('Metric line exceeds the bounded parser limit'); break
                    if not line.endswith(b'\n'):
                        break  # Writer has not completed this record yet.
                    self.offset = stream.tell(); consumed += len(line)
                    try:
                        raw = json.loads(line)
                        row = self.normalize(raw, start, self.offset, line)
                        if row['event_id'] not in self.seen:
                            self.apply(row); self.append(row)
                        else:
                            # Hydrate optional live fields from raw backfill; old history stays byte-identical.
                            self.apply_optional(row)
                    except (ValueError, KeyError, TypeError, AttributeError) as error:
                        self.error('Metric offset '+str(start)+': '+str(error))

    def budget(self):
        original = self.artifact(self.control/'budget-ledger.json')
        path = self.control/'budget-24h-ledger.json'
        if not path.exists():
            return original
        amended = self.artifact(path)
        start,deadline = number(amended.get('start_epoch')),number(amended.get('deadline_epoch'))
        proof = amended.get('original_8h_ledger',{})
        old_path = self.control/'budget-ledger.json'
        valid = (amended.get('schema')=='tetris4d-survival-budget-v1'
                 and amended.get('fixed',{}).get('hours')==24 and start is not None
                 and deadline==start+86400 and start==number(original.get('start_epoch'))
                 and amended.get('amendment',{}).get('authorized_total_hours')==24
                 and old_path.is_file() and not old_path.is_symlink()
                 and proof.get('path')==str(old_path) and proof.get('bytes')==old_path.stat().st_size
                 and proof.get('sha256')==hashlib.sha256(old_path.read_bytes()).hexdigest())
        if not valid:
            self.error('Amended budget is invalid; displaying the original ledger')
            return original
        return amended

    def snapshot(self):
        with self.lock:
            ledger = self.budget()
            extended = ledger.get('fixed',{}).get('hours')==24
            pipeline = self.artifact(self.control/('extended-pipeline-status.json' if extended else 'pipeline-status.json'))
            launcher = self.artifact(self.control/'status.json')
            manifest = self.artifact(self.run_dir/'manifest.json')
            summary = self.artifact(self.run_dir/'summary.json')
            calibration_report = self.artifact(self.run_dir/'critic-calibration.json')
            best = self.artifact(self.run_dir/'best.json')
            signature = manifest.get('source_sha256', {})
            cal = dict(self.calibration)
            if cal['total_games'] is None and all(type(signature.get(key)) is int for key in ('calibration_games','calibration_validation_games')):
                cal['total_games'] = signature['calibration_games']+signature['calibration_validation_games']
            if cal['stage'] is None:
                cal['stage'] = manifest.get('calibration_stage')
            if calibration_report:
                cal.update(passed=calibration_report.get('passed'), train_samples=calibration_report.get('train_samples'),
                           validation_samples=calibration_report.get('validation_samples'), criterion=calibration_report.get('criterion'),
                           validation_mse=calibration_report.get('validation',{}).get('mse'),
                           completed_games=calibration_report.get('completed_games',cal['completed_games']))
            progress = {key:self.progress[key] if self.progress[key] is not None else summary.get(key) for key in self.progress}
            best_score = best.get('score', summary.get('best_validation_score'))
            best_selection = self.best
            if isinstance(best_score,list) and len(best_score)==3:
                best_selection = dict(zip(('mean_pieces','minimum_pieces','median_pieces'),best_score))
                best_selection.update(runner_updates=best.get('runner_updates'), beats_sustained=best.get('beats_sustained_on_selection'))
            phase = self.event_phase or ('calibration' if cal['stage'] in ('collect','fit') else 'unknown')
            if not self.latest['timestamp_utc']:
                phase = 'waiting_comparison' if pipeline.get('phases',{}).get('comparison',{}).get('state')=='running' else 'waiting_training'
            if cal.get('passed') is False or launcher.get('state')=='failed' or pipeline.get('state')=='failed':
                phase = 'failed'
            elif launcher.get('state') in ('finished','budget_exhausted') or pipeline.get('phases',{}).get('training',{}).get('state')=='complete':
                phase = 'complete'
            request = self.artifact(self.control/'optimization-stop-request.json')
            if request and summary.get('reason')=='signal_2' and launcher.get('state')!='running' and pipeline.get('state')!='complete':
                phase = 'optimizing'
            elif launcher.get('state')=='running' and phase in ('unknown','waiting_training','complete'):
                phase = 'training' if manifest.get('calibration_stage')=='passed' else 'calibration'
            now = time.time(); start = number(ledger.get('start_epoch')); deadline = number(ledger.get('deadline_epoch'))
            rows = list(self.rows)
            if len(rows)>3000:
                chosen = {round(i*(len(rows)-1)/2999) for i in range(3000)}
                rows = [rows[index] for index in sorted(chosen)]
            games = summary.get('active_games', [])
            observed = datetime.fromtimestamp((self.run_dir/'summary.json').stat().st_mtime,timezone.utc).isoformat() if games else None
            active = [number(game.get('pieces')) for game in games]
            active = [pieces for pieces in active if pieces is not None and pieces >= 0]
            active_max = max(active) if active else None
            known = [pieces for pieces in (active_max,self.completed_record) if pieces is not None]
            return {'updated_utc':utc(), 'phase':phase,
                    'budget':{'start_utc':ledger.get('start_utc'),'deadline_utc':ledger.get('deadline_utc'),
                              'elapsed_seconds':None if start is None else max(0,now-start),
                              'remaining_seconds':None if deadline is None else max(0,deadline-now)},
                    'calibration':cal,'calibration_games':list(self.calibration_games.values()),'progress':progress,
                    'latest':dict(self.latest),'best_selection':best_selection,
                    'games':[{**game,'observed_utc':observed} for game in games],
                    'series':rows,'series_total_count':self.total_rows,'series_sampled':self.total_rows>len(rows),
                    'maximum_survival':{'completed_record_pieces':self.completed_record,
                                        'completed_episodes':self.completed_episodes,'records':list(self.survival_records),
                                        'active_max_pieces':active_max,'active_observed_utc':observed,
                                        'current_max_pieces':max(known) if known else None},
                    'events':list(self.events),'errors':list(self.errors),
                    'history':{'bytes':self.history.stat().st_size if self.history.exists() else 0,'limit_bytes':MAX_BYTES,
                               'frozen':self.frozen_reason is not None,'frozen_reason':self.frozen_reason,
                               'prior':self.prior_proof},
                    'source':{'metrics':str(self.metrics),'offset_bytes':self.offset,
                              'size_bytes':self.metrics.stat().st_size if self.metrics.exists() else 0,
                              'backfill_complete':not self.metrics.exists() or self.offset==self.metrics.stat().st_size}}

    def history_csv(self):
        output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=CSV_FIELDS); writer.writeheader()
        for line in self.history_lines():
            row = json.loads(line)
            writer.writerow({key:row.get(key) for key in CSV_FIELDS})
        return output.getvalue().encode('utf-8')

    def history_lines(self):
        seen = set()
        for path in (self.prior_history,self.history):
            if path is None or not path.exists():
                continue
            with path.open('rb') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if row['event_id'] not in seen:
                            seen.add(row['event_id']); yield line
                    except (ValueError,TypeError,KeyError): break


def make_server(collector, host='127.0.0.1', port=8765, static_dir=STATIC):
    if host not in ('127.0.0.1','localhost'):
        raise ValueError('The dashboard binds only to loopback 127.0.0.1')
    static_dir = Path(static_dir)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No growing access log; collection errors are visible in /api/status.

        def do_GET(self):
            try:
                host_name = urlsplit('http://'+self.headers.get('Host','')).hostname
                if host_name not in ('127.0.0.1','localhost'):
                    self.send_error(403); return
                path = urlsplit(self.path).path
                download = None
                with collector.lock:
                    if path == '/api/status':
                        body = json.dumps(collector.snapshot(),allow_nan=False).encode(); mime = 'application/json'
                    elif path == '/api/history.csv':
                        body = collector.history_csv(); mime = 'text/csv; charset=utf-8'; download = 'training-history.csv'
                    elif path == '/api/history.jsonl':
                        body = b''.join(collector.history_lines())
                        mime = 'application/x-ndjson'; download = 'training-history.jsonl'
                    elif path in ('/','/index.html','/app.js','/style.css'):
                        file = static_dir/('index.html' if path=='/' else path[1:])
                        if not file.is_file() or file.is_symlink():
                            self.send_error(404); return
                        body = file.read_bytes()
                        mime = {'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'}[file.suffix]
                    else:
                        self.send_error(404); return
                self.send_response(200); self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store')
                self.send_header('X-Content-Type-Options','nosniff')
                if download: self.send_header('Content-Disposition','attachment; filename="'+download+'"')
                self.end_headers(); self.wfile.write(body)
            except (OSError,ValueError,TypeError,KeyError) as error:
                collector.error('HTTP: '+str(error)); self.send_error(500)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--prior-data-dir',type=Path,help='Read-only prior dashboard history for a migrated run')
    parser.add_argument('--host',default='127.0.0.1',choices=('127.0.0.1','localhost'))
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--poll-seconds',type=float,default=15)
    args = parser.parse_args(argv)
    if not 1<=args.port<=65535 or not math.isfinite(args.poll_seconds) or not 1<=args.poll_seconds<=300:
        parser.error('port must be 1..65535 and poll-seconds 1..300')
    collector = Collector(args.run_dir,args.data_dir,args.prior_data_dir)
    lock_path = collector.data_dir/'collector.lock'
    if lock_path.is_symlink(): raise ValueError('Collector lock must not be a symlink')
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another collector already owns this data directory')
        collector.poll(); server = make_server(collector,args.host,args.port)
        thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        stopped = threading.Event()
        handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
        for sig in handlers: signal.signal(sig,lambda *unused:stopped.set())
        print(json.dumps({'url':f'http://127.0.0.1:{args.port}','history':str(collector.history),'poll_seconds':args.poll_seconds}),flush=True)
        try:
            while not stopped.wait(args.poll_seconds):
                try: collector.poll()
                except (OSError,ValueError,TypeError) as error: collector.error('Collection: '+str(error))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
            for sig,handler in handlers.items(): signal.signal(sig,handler)


if __name__ == '__main__':
    main()
