#!/usr/bin/env python3
"""Live page for trained-wells training and 1M verification games.

    python tools/wells_dashboard.py RUN_DIR [--port 8775]

RUN_DIR holds cem-sinh/generations.jsonl from train_wells_policy.py and the
play/replay outputs of verify_trained_wells.py. Each --set names a directory of
games and its placement cap. Every request re-reads the files, so the page
follows running games.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time

MODEL = Path(__file__).resolve().parents[1] / 'models' / 'trained-wells.json'
COLORS = ['#e8a87c', '#7fc8a9', '#8fb3e8', '#d98fd9', '#e8d77c', '#e88f8f', '#9fd98f', '#b8b8b8']
CAP = 1_000_000


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                break  # partially written last line
    return rows


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def line_chart(series, key, label, width=900, height=260, x_max=CAP, y_floor=None):
    """series: [(name, color, rows)] with rows carrying 'placements' and key."""
    values = [r[key] for _, _, rows in series for r in rows]
    if not values:
        return '<p class="muted">No samples yet.</p>'
    pad = 52
    low = min(values) if y_floor is None else y_floor
    high = max(max(values), low + 1)
    x = lambda v: pad + v / x_max * (width - 2 * pad)
    y = lambda v: height - pad + 16 - (v - low) / (high - low) * (height - 2 * pad)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)} by placement">']
    for k in range(5):
        value = low + (high - low) * k / 4
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y(value):.1f}" y2="{y(value):.1f}" class="grid"/>'
                     f'<text x="{pad - 6}" y="{y(value) + 4:.1f}" text-anchor="end">{value:,.0f}</text>')
    for k in range(0, 11, 2):
        v = x_max * k / 10
        parts.append(f'<text x="{x(v):.1f}" y="{height - 8}" text-anchor="middle">{v / 1000:,.0f}k</text>')
    for name, color, rows in series:
        if rows:
            points = ' '.join(f'{x(r["placements"]):.1f},{y(r[key]):.1f}' for r in rows)
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.8" points="{points}">'
                         f'<title>{escape(name)}</title></polyline>')
    return ''.join(parts) + '</svg>'


def cem_chart(gens, width=900, height=240):
    if not gens:
        return '<p class="muted">No generations yet.</p>'
    pad = 52
    high = max(max(g['fitness_quartiles'][4], g['cap']) for g in gens) * 1.05
    low = 0
    x = lambda i: pad + i * (width - 2 * pad) / max(1, len(gens) - 1)
    y = lambda v: height - pad + 16 - (max(v, low) - low) / (high - low) * (height - 2 * pad)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="CEM fitness by generation">']
    for k in range(5):
        value = low + (high - low) * k / 4
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y(value):.1f}" y2="{y(value):.1f}" class="grid"/>'
                     f'<text x="{pad - 6}" y="{y(value) + 4:.1f}" text-anchor="end">{value:,.0f}</text>')
    for key, cls, idx in (('cap', 'base', None), ('median', 'mean', 2), ('best', 'best', 4)):
        pts = ' '.join(f'{x(i):.1f},{y(g["cap"] if idx is None else g["fitness_quartiles"][idx]):.1f}' for i, g in enumerate(gens))
        parts.append(f'<polyline class="{cls}" points="{pts}"/>')
    for i, g in enumerate(gens):
        parts.append(f'<text x="{x(i):.1f}" y="{height - 8}" text-anchor="middle">{g["generation"]}</text>')
    return ''.join(parts) + '</svg>'


def game_set(run: Path, title: str, directory: str, cap: int) -> str:
    verify = run / directory
    seeds = sorted({int(p.stem.split('-')[1]) for p in verify.glob('progress-*.jsonl')})
    games, series, now = [], [], time.time()
    for i, seed in enumerate(seeds):
        progress = read_rows(verify / f'progress-{seed}.jsonl')
        play = read_json(verify / f'play-{seed}.json')
        replay = read_json(verify / f'replay-{seed}.json')
        last = progress[-1] if progress else {'placements': 0, 'layers': 0, 'max_height': 0, 'holes': 0, 'occupied_cells': 0, 'seconds': 0}
        fresh = (now - (verify / f'progress-{seed}.jsonl').stat().st_mtime) < 300
        if play and play['top_out']:
            status = f'topped out at {play["placements"]:,}'
        elif play and play['placements'] >= cap:
            status = f'reached {cap:,}' + (' · replay verified' if replay and replay['board_sha256'] == play['board_sha256']
                                           and replay['moves_sha256'] == play['moves_sha256'] else ' · replay pending')
        elif fresh:
            status = 'running'
        else:
            status = 'stalled or stopped'
        rate = last['placements'] / last['seconds'] if last['seconds'] else 0
        games.append((seed, COLORS[i % len(COLORS)], last, status, rate, progress))
        series.append((str(seed), COLORS[i % len(COLORS)], progress))
    if not games:
        return f'<h2>{escape(title)}</h2><p class="muted">No games yet in {escape(directory)}.</p>'
    done = sum(1 for g in games if g[3].startswith('reached'))
    verified = sum(1 for g in games if 'replay verified' in g[3])
    lost = sum(1 for g in games if g[3].startswith('topped'))
    total = sum(g[2]['placements'] for g in games)
    running = [g for g in games if g[3] == 'running']
    eta = max(((cap - g[2]['placements']) / g[4] for g in running if g[4]), default=0)
    rows = ''.join(
        f'<tr><td><span class="dot" style="background:{color}"></span>{seed}</td>'
        f'<td><div class="bar"><div style="width:{min(100, last["placements"] / cap * 100):.1f}%;background:{color}"></div></div>'
        f'{last["placements"]:,}</td><td>{last["layers"]:,}</td><td>{last["max_height"]}</td><td>{last["holes"]}</td>'
        f'<td>{last["occupied_cells"]:,}</td><td>{rate:,.0f}/s</td><td>{escape(status)}</td></tr>'
        for seed, color, last, status, rate, _ in games)
    return f'''<h2>{escape(title)}</h2>
<div class="tiles"><div class="tile">Games at {cap:,}<b>{done} / {len(games)}</b>{lost} topped out</div>
<div class="tile">Replay verified<b>{verified} / {len(games)}</b>independent re-drop</div>
<div class="tile">Placements played<b>{total:,}</b>across these games</div>
<div class="tile">Running<b>{len(running)}</b>{f"≈ {eta / 60:.0f} min to finish" if running else "idle"}</div></div>
<div class="table"><table><tr><th>Seed</th><th>Placements</th><th>Layers</th>
<th>Height</th><th>Holes</th><th>Occupied</th><th>Rate</th><th>Status</th></tr>{rows}</table></div>
<p class="muted">Occupied cells stay bounded when layers clear as fast as cells arrive; a rising trend means the game is filling up.</p>
{line_chart(series, "occupied_cells", "Occupied cells", x_max=cap, y_floor=0)}
{line_chart(series, "max_height", "Max height", height=200, x_max=cap, y_floor=0)}'''


def page(run: Path, sets) -> str:
    gens = read_rows(run / 'cem-sinh' / 'generations.jsonl')
    model = read_json(MODEL)
    model_sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()[:12] if model else 'missing'
    selected = model['training']['selected_generation'] if model else None
    names = ['levels', 'holes', 'covers', 'benefit', 'cost', 'rough', 'high']
    gen_rows = ''.join(
        f'<tr{" class=sel" if g["generation"] == selected else ""}><td>{g["generation"]}</td><td>{g["cap"]:,}</td>'
        f'<td>{g["fitness_quartiles"][2]:,.0f}</td><td>{g["fitness_quartiles"][4]:,.0f}</td><td>{g["elite_capped"]}/4</td>'
        f'<td>{escape(" · ".join(f"{n} {w:.3g}" for n, w in zip(names, g.get("mean_weights", []))))}</td></tr>'
        for g in gens)
    sections = ''.join(game_set(run, title, directory, cap) for title, directory, cap in sets)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="10"><title>Trained wells</title><style>
:root{{color-scheme:dark}} body{{margin:0;background:#12151a;color:#e6e6e6;font:15px/1.5 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px}} h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px}}
.muted{{color:#9aa3ad}} .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0}}
.tile{{background:#1b2027;border:1px solid #2a313b;border-radius:10px;padding:12px}} .tile b{{display:block;font-size:24px;color:#e8a87c}}
svg{{width:100%;height:auto;background:#1b2027;border-radius:10px;margin-top:8px}} svg text{{fill:#9aa3ad;font-size:11px}}
.grid{{stroke:#2a313b}} .base{{fill:none;stroke:#9aa3ad;stroke-dasharray:4 4}} .mean{{fill:none;stroke:#7fc8a9;stroke-width:2}}
.best{{fill:none;stroke:#e8a87c;stroke-width:2}} table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{border-bottom:1px solid #2a313b;padding:6px;text-align:left;vertical-align:middle}} tr.sel td{{background:#22303a}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.bar{{width:120px;height:6px;background:#2a313b;border-radius:3px;display:inline-block;margin-right:8px;vertical-align:middle}}
.bar div{{height:6px;border-radius:3px}} .table{{overflow-x:auto}}
</style></head><body><main><h1>Trained wells</h1>
<p class="muted">CEM-trained linear well scorer · model {model_sha}… (generation {selected} mean) · refreshes every 10 s.</p>
{sections}
<h2>Training (CEM, sinh parameterization)</h2><p class="muted">Orange: best candidate · green: median · dashed: survival cap.
Highlighted row: the generation whose mean is the bundled model.</p>{cem_chart(gens)}
<div class="table"><table><tr><th>Gen</th><th>Cap</th><th>Median</th><th>Best</th><th>Elite capped</th><th>Mean weights</th></tr>{gen_rows}</table></div>
</main></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--port', type=int, default=8775)
    parser.add_argument('--set', action='append', metavar='TITLE=DIR:CAP',
                        help='a game set to show, e.g. "1M verification=verify-1m:1000000"; repeatable')
    args = parser.parse_args(argv)
    sets = []
    for spec in args.set or ['1M verification=verify-1m:1000000']:
        title, rest = spec.split('=', 1)
        directory, cap = rest.rsplit(':', 1)
        sets.append((title, directory, int(cap)))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                body = page(args.run, sets).encode()
                code = 200
            except Exception as error:  # noqa: BLE001 - show the problem instead of a dead page
                body = f'<pre>{escape(type(error).__name__)}: {escape(str(error))}</pre>'.encode()
                code = 500
            self.send_response(code)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            pass

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
