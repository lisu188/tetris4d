#!/usr/bin/env python3
"""Live page for a teacher_cem.py run: fitness per candidate and generation, best weights, held-out results.

    python tools/cem_dashboard.py RUN_DIR [--port 8771]

Reads cem.jsonl, tune-report.json and actors-holdout.json from RUN_DIR on every request.
"""
from __future__ import annotations

import argparse
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics

COLORS = ['#e8a87c', '#7fc8a9', '#8fb3e8', '#d98fd9', '#e8d77c', '#e88f8f', '#9fd98f', '#b8b8b8']


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                break  # partially written last line
    return rows


def chart(rows, width=900, height=320):
    if not rows:
        return '<p class="muted">No candidates evaluated yet (positions are replayed first).</p>'
    pad, fits = 48, [r['fitness'] for r in rows]
    low, high = min(fits) * .95, max(fits) * 1.05
    x = lambda i: pad + i * (width - 2 * pad) / max(1, len(rows) - 1)
    y = lambda v: height - pad - (v - low) / max(1e-9, high - low) * (height - 2 * pad)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Fitness of every candidate">']
    for k in range(5):
        value = low + (high - low) * k / 4
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y(value):.1f}" y2="{y(value):.1f}" class="grid"/>'
                     f'<text x="{pad - 6}" y="{y(value) + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    baseline = rows[0]['fitness']
    parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y(baseline):.1f}" y2="{y(baseline):.1f}" class="base"/>')
    means = [(i, r) for i, r in enumerate(rows) if r['label'].endswith('-0')]
    parts.append('<polyline class="mean" points="' + ' '.join(f'{x(i):.1f},{y(r["fitness"]):.1f}' for i, r in means) + '"/>')
    for i, r in enumerate(rows):
        generation = int(r['label'].split('-')[0][1:])
        parts.append(f'<circle cx="{x(i):.1f}" cy="{y(r["fitness"]):.1f}" r="4" fill="{COLORS[generation % len(COLORS)]}">'
                     f'<title>{escape(r["label"])}: {r["fitness"]:.0f} placements</title></circle>')
    return ''.join(parts) + '</svg>'


def page(run):
    rows = read_rows(run / 'cem.jsonl')
    report = json.loads((run / 'tune-report.json').read_text()) if (run / 'tune-report.json').exists() else None
    actors = json.loads((run / 'actors-holdout.json').read_text()) if (run / 'actors-holdout.json').exists() else None
    log = (run / 'pipeline.log').read_text(encoding='utf-8').strip().splitlines() if (run / 'pipeline.log').exists() else []
    generations = {}
    for r in rows:
        generations.setdefault(r['label'].split('-')[0], []).append(r)
    table = ''.join(
        f'<tr><td>{g}</td><td>{len(c)}</td><td>{c[0]["fitness"]:.0f}</td><td>{max(x["fitness"] for x in c):.0f}</td>'
        f'<td>{escape(", ".join(f"{k} {v:.3f}" for k, v in max(c, key=lambda x: x["fitness"])["parameters"].items() if k != "height"))}</td></tr>'
        for g, c in generations.items())
    best = max(rows, key=lambda r: r['fitness']) if rows else None
    holdout = ''
    if report:
        h = report['holdout']
        c = h['comparison']
        holdout = (f'<section><h2>Held-out teacher check</h2><p>{h["positions"]} positions, cap {h["cap"]:,}: '
                   f'baseline {statistics.mean(h["baseline"]):.0f} vs tuned {statistics.mean(h["tuned"]):.0f} placements '
                   f'(paired difference {c["mean_difference"]:+.0f}, 95% CI [{c["ci95"][0]:+.0f}, {c["ci95"][1]:+.0f}], '
                   f'wins/ties/losses {c["wins"]}/{c["ties"]}/{c["losses"]}).</p></section>')
    actor_html = ''
    if actors:
        lines = ''.join(f'<li>{escape(Path(r["checkpoint"]).name)}: mean {r["mean"]:.0f}</li>' for r in actors['results'])
        comps = ''.join(f'<li>vs first: {c["mean_difference"]:+.0f}, 95% CI [{c["ci95"][0]:+.0f}, {c["ci95"][1]:+.0f}], '
                        f'w/t/l {c["wins"]}/{c["ties"]}/{c["losses"]}</li>' for c in actors['comparisons'])
        actor_html = f'<section><h2>Re-pretrained actor (held-out)</h2><ul>{lines}{comps}</ul></section>'
    status = escape(log[-1]) if log else 'Starting'
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30"><title>Teacher tuning</title><style>
:root{{color-scheme:dark}} body{{margin:0;background:#12151a;color:#e6e6e6;font:15px/1.5 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px}} h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:18px;margin:24px 0 8px}}
.muted{{color:#9aa3ad}} .tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:16px 0}}
.tile{{background:#1b2027;border:1px solid #2a313b;border-radius:10px;padding:12px}} .tile b{{display:block;font-size:24px;color:#e8a87c}}
svg{{width:100%;height:auto;background:#1b2027;border-radius:10px}} svg text{{fill:#9aa3ad;font-size:11px}}
.grid{{stroke:#2a313b}} .base{{stroke:#9aa3ad;stroke-dasharray:4 4}} .mean{{fill:none;stroke:#7fc8a9;stroke-width:2}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td,th{{border-bottom:1px solid #2a313b;padding:6px;text-align:left}}
</style></head><body><main><h1>Teacher tuning</h1><p class="muted">{status}. Refreshes every 30 s.</p>
<div class="tiles"><div class="tile">Candidates evaluated<b>{len(rows)} / 128</b></div>
<div class="tile">Starting weights<b>{rows[0]["fitness"]:.0f}</b>placements (tuning set)</div>
<div class="tile">Best candidate<b>{best["fitness"]:.0f}</b>{escape(best["label"])}</div></div>
<h2>Fitness of every candidate</h2><p class="muted">Dots: candidates, coloured by generation. Green line: each generation's mean weights. Dashed: starting weights.</p>
{chart(rows)}<h2>Generations</h2><table><tr><th>Gen</th><th>Done</th><th>Mean weights</th><th>Best</th><th>Best weights</th></tr>{table}</table>
{holdout}{actor_html}</main></body></html>''' if rows else f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30"><title>Teacher tuning</title><style>body{{background:#12151a;color:#e6e6e6;font:15px system-ui;padding:24px}}</style></head>
<body><h1>Teacher tuning</h1><p>{status}</p><p>Replaying the tuning and held-out positions; candidates appear here as they finish.</p></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--port', type=int, default=8771)
    args = parser.parse_args(argv)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = page(args.run).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *unused):
            pass

    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
