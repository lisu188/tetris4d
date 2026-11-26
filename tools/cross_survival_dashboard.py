"""Loopback, allowlisted read-only overview of the frozen cross-survival run."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import statistics
import time

PAGE = Path(__file__).with_name('training_dashboard') / 'cross_survival.html'
ROOT = None
PORT = 8774
MAX_JSON = 4 << 20


def read(path, limit=MAX_JSON):
    if not path.is_file() or not path.resolve().is_relative_to(ROOT):
        return None
    try:
        with path.open('rb') as f:
            raw = f.read(limit + 1)
    except FileNotFoundError:
        # A completed game replaces and removes its last partial snapshot.
        return None
    return json.loads(raw) if len(raw) <= limit else None


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def cache_row(folder, seed, base, partial):
    material = read(folder / (f'{seed}.partial.json' if partial else f'{seed}.json'), 300_000)
    if not material:
        return None
    key = dict(base, **{('partial_seed' if partial else 'seed'): seed})
    row = material['data']
    if material['identity'] != key or row['seed'] != seed or material['data_sha256'] != digest(row):
        raise ValueError('Mismatched screening cache record')
    return row


def last_update(arm):
    path = arm/'metrics.jsonl'
    if path.is_file() and path.resolve().is_relative_to(ROOT):
        with path.open('rb') as f:
            size=f.seek(0,2)
            f.seek(max(0,size-131072))
            lines=f.read(131072).splitlines(keepends=True)
        for line in reversed(lines):
            if line.endswith(b'\n') and line.startswith(b'{'):
                row=json.loads(line)
                if row.get('event')=='update':
                    return row
    return {}


def elapsed(row, now):
    if row['seconds'] is not None:
        return max(0.,row['seconds'])
    return max(0.,min(now-row['start_unix'],row['allowance_seconds']))


def screening(label, player, plan, protocol, active, previous, *, ppo=False):
    seeds = plan['screening']
    receipt = read(ROOT/'screening'/f'{label}.json') or {}
    if receipt and (receipt.get('protocol') != protocol or receipt.get('requested_seeds') != seeds or
                    receipt.get('player') != player or player is None):
        raise ValueError('Mismatched screening receipt')
    games={r['seed']:r for r in receipt.get('games',[]) if r['seed'] in seeds}
    live = []
    if player and (active or (ppo and previous and not receipt.get('complete'))):
        base = dict(policy='ppo',checkpoint=player['sha256'],weights=plan['teacher'],runtime=plan['runtime'],
                    screening='exhaustive',top_k=16,cap=0,greedy_actor='fixed-fp32-v1',batch_size=plan['batch'],
                    piece_set='expanded',group_exhaustive=True)
        for key in ('headless','two_stage'):
            if plan['greedy_enabled'][key]:
                base[key]=True
        folder=ROOT/'evaluation-cache'/digest(base)
        # Seed numbers come only from the public screen list, even when another
        # stage happens to share the per-policy cache directory.
        for seed in seeds:
            if seed not in games:
                row=cache_row(folder,seed,base,False)
                if row:
                    games[seed]=row
                else:
                    row=cache_row(folder,seed,base,True)
                    if row:
                        live.append(row)
    natural=[r for r in games.values() if r['reason']=='top_out']
    decided=sum(r.get('decisions',{}).get('decisions',0) for r in [*games.values(),*live])
    changes=sum(r.get('decisions',{}).get('direct_weight_changes',0) for r in [*games.values(),*live])
    stopped = any(r['reason'] == 'target_or_no_move' for r in games.values()) or receipt.get('planner_blocked')
    capped = any(r['reason'] == 'cap' for r in games.values()) or receipt.get('cap') or receipt.get('censored')
    accepted = (receipt.get('complete') is True and receipt.get('cap') == 0 and not stopped and not capped and
                len(natural) == len(seeds) and [r['seed'] for r in receipt['games']] == seeds)
    mean=sum(r['placements'] for r in natural)/len(natural) if accepted else None
    status=('Controller stop' if stopped else 'Capped' if capped else 'Complete' if accepted else
            'Running' if active else 'Partial' if receipt or previous else 'Waiting')
    if ppo:
        return dict(player=label, status=status, finished_games=len(games),
                    requested=len(seeds), natural_games=len(natural), active_snapshots=len(live),
                    live_range=[min(r['pieces'] for r in live), max(r['pieces'] for r in live)] if live else None,
                    mean_placements=mean,
                    median_placements=statistics.median(r['placements'] for r in natural) if accepted else None,
                    change_pct=None,
                    residual_changes=sum(r.get('decisions', {}).get('residual_changes', 0)
                                         for r in [*games.values(), *live]) if label.endswith('-residual') else None,
                    observed_moves=decided)
    return dict(player=label,status=status,natural_games=len(natural),requested=len(seeds),changed_moves=changes,
                observed_moves=decided,mean_placements=mean,change_pct=None,active_snapshots=len(live),
                live_range=[min(r['pieces'] for r in live),max(r['pieces'] for r in live)] if live else None)


def public_view():
    now=time.time()
    plan=read(ROOT/'protocol.json')
    protocol=digest(plan)
    names=['incumbent', *[f'weight-{w:g}' for w in plan['direct_weights'] if w != 0]]
    players=read(ROOT/'direct-players.json')
    if players['protocol']!=protocol:
        raise ValueError('Wrong player registry')
    ledger=read(ROOT/'gpu-time.json') or []
    session=read(ROOT/'gpu-session.json') or {}
    seconds=sum(elapsed(r,now) for r in ledger)
    if session.get('status')=='active':
        since=min(session['remaining'],max(0,now-session['start_unix']))
        booked=sum(elapsed(r,now) for r in ledger[session['first_entry']:])
        seconds+=max(0,since-booked)
    slot=next((r for r in reversed(ledger) if r.get('seconds') is None),{})
    phase,label=slot.get('phase'),slot.get('label','')
    direct=[]
    for name in names:
        key='screening/'+name
        direct.append(screening(name,players['players'][name],plan,protocol,label==key,
                                any(r.get('label')==key for r in ledger)))
    incumbent=direct[0]['mean_placements']
    for row in direct:
        if incumbent and row['mean_placements'] is not None:
            row['change_pct']=100*(row['mean_placements']/incumbent-1)
    arms=[]; observations=[]; ppo_screening=[]; recipe_scores=[]
    for recipe in plan['recipes']:
        completed_screens = []
        for seed in plan['training_seeds']:
            arm=f'{seed}-{recipe}'
            folder=ROOT/'training'/arm
            manifest=read(folder/'manifest.json') or {}
            endpoint=read(folder/'endpoint.json') or {}
            fresh=last_update(folder)
            updates=max(manifest.get('runner_updates',0),fresh.get('runner_updates',0),endpoint.get('updates',0))
            positions=fresh.get('placements') if fresh.get('runner_updates')==updates else endpoint.get('placements')
            if positions is None:
                # The trainer commits full updates at this fixed number of placements.
                positions=updates*plan['placements_per_update']
            arms.append(dict(arm=arm,recipe=recipe,seed=seed,updates=updates,placements=positions,
                             status='Complete' if endpoint.get('protocol')==protocol else
                                'Running' if phase=='training' and label==arm else 'Partial' if manifest or fresh else 'Waiting'))
            if endpoint and (endpoint.get('protocol') != protocol or endpoint.get('seed') != seed or
                             endpoint.get('recipe') != recipe or endpoint.get('player', {}).get('mode') != 'greedy'):
                raise ValueError('Mismatched PPO endpoint')
            key = 'screening/' + arm
            row = screening(arm, endpoint.get('player'), plan, protocol, phase == 'development' and label == key,
                            any(r.get('phase') == 'development' and r.get('label') == key for r in ledger), ppo=True)
            if incumbent and row['mean_placements'] is not None:
                row['change_pct'] = 100 * (row['mean_placements'] / incumbent - 1)
            ppo_screening.append(row)
            if row['status'] == 'Complete':
                completed_screens.append(row)
            for step in (10,40,80):
                receipt=read(ROOT/'weight-observations'/arm/f'{step:03}.json')
                if not receipt or receipt.get('arm')!=arm or receipt.get('updates')!=step or not receipt.get('behavior'):
                    continue
                behavior=receipt['behavior']; groups=behavior['groups']; g=groups['all']
                observations.append(dict(arm=arm,update=step,boards=g['boards'],flips=g['action_flips'],
                    danger=groups['danger']['action_flips'],ordinary=groups['ordinary']['action_flips'],
                    cross_gain=g['cross_on_flips']['sum_delta'],actor_relative_l2=receipt['groups']['base_actor']['relative_l2'],
                    critic_relative_l2=receipt['groups']['critic']['relative_l2'],residual_changes=g['residual_argmax_changes']))
        if len(completed_screens) == len(plan['training_seeds']) and completed_screens:
            recipe_scores.append(dict(recipe=recipe,
                mean=statistics.mean(r['mean_placements'] for r in completed_screens),
                median=statistics.mean(r['median_placements'] for r in completed_screens)))
    # Expose only the official recommendation field after the runner freezes it.
    result=read(ROOT/'result.json') or {}
    completed=result.get('complete') is True and result.get('protocol')==protocol
    phase_text='Between reservations'
    if completed:
        phase_text='Experiment complete'
    elif phase=='holdout':
        phase_text='Sealed holdout'
    elif label.startswith('qualification/'):
        phase_text='Sealed qualification'
    elif label in ['screening/'+n for n in names]:
        phase_text='Direct screening · '+label.split('/',1)[1]
    elif phase=='training' and label in [a['arm'] for a in arms]:
        phase_text='PPO training · '+label
    elif phase=='development' and label in ['screening/'+r['player'] for r in ppo_screening]:
        phase_text='PPO development screening · '+label.split('/',1)[1]
    elif phase=='development':
        phase_text='Development screening'
    elif phase=='calibration':
        phase_text='Calibration'
    elif phase=='preparation':
        phase_text='Preparation'
    stopped=read(ROOT/'status.json') or {}
    if stopped and not completed and not slot and session.get('status')!='active':
        phase_text='Run stopped'
    return dict(run=ROOT.name,phase=phase_text,gpu_seconds=round(seconds),budget_hours=sum(plan['phase_hours'].values()),
                updated_unix=int(now),screening_games=len(plan['screening']),target_updates=plan['updates'],
                direct=direct,arms=arms,observations=observations,ppo_screening=ppo_screening,recipe_scores=recipe_scores,
                promotion=('Challenger promoted' if result.get('promoted') else 'Incumbent retained') if completed else None)


class LocalServer(HTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(4)
        return connection, address


class Handler(BaseHTTPRequestHandler):
    server_version='LocalRun';sys_version=''
    def log_message(self,*args):
        pass

    def serve(self, body=True):
        status=200
        port = self.server.server_port
        if self.headers.get('Host') not in {f'127.0.0.1:{port}',f'localhost:{port}'}:
            status,data,kind=403,b'Forbidden','text/plain'
        elif self.path=='/':
            data,kind=self.server.page,'text/html; charset=utf-8'
        elif self.path=='/api/run':
            try:
                now=time.monotonic()
                if now>=self.server.next_read:
                    self.server.snapshot=json.dumps(public_view(),allow_nan=False,separators=(',',':')).encode()
                    self.server.next_read=now+5
                data,kind=self.server.snapshot,'application/json; charset=utf-8'
            except (OSError,KeyError,TypeError,ValueError):
                status,data,kind=503,b'{"status":"Run receipts temporarily unavailable"}','application/json'
        else:
            status,data,kind=404,b'Not found','text/plain'
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.send_header('Referrer-Policy','no-referrer');self.end_headers()
        if body:self.wfile.write(data)

    def do_GET(self):self.serve()
    def do_HEAD(self):self.serve(False)


def serve(run_dir, port, server_class=LocalServer):
    global ROOT
    ROOT = run_dir.resolve(strict=True)
    page = PAGE.read_bytes()
    snapshot = json.dumps(public_view(), allow_nan=False).encode()
    with server_class(('127.0.0.1', port), Handler) as server:
        server.page = page
        server.snapshot = snapshot
        server.next_read = time.monotonic() + 5
        server.serve_forever()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=PORT)
    args = parser.parse_args()
    serve(args.run_dir, args.port)
