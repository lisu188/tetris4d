"""Audit the critic gate against its pinned runtime, source and terminal per-seed game caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
import sparse_tetris_4d_optimized as game

ARTIFACTS = ROOT / 'survival-training.artifacts'
INCUMBENT = 'a6470b4ef118aea46969922559ed4f58826b82f0ae08c26ff81acdc0eac84cd9'
DONOR = '8da3b47e5a4319c7515dab18253f7b49d2f870b29c83ba3f10c03fe5ec78def8'
TEACHER = '9f8826a79bc5493e09ff5df083bb2ee4c2cd898363889a3308a01e4fa1564abb'
LEGACY_DRIVER = 'f65114c1cc2035f3a11e9d88616d749e5a43d58ec6fbe64b19ccdd8ac653d3f6'
LEGACY_METER = '1b9e08e3a8a8ad4abf3a4d497359e63153dce93abe20932fd441fce5f24f0b58'
SEEDS = tuple(48100001 + 2*i for i in range(8))
TEMPERATURES = (1., .25)
STRIDE = 128
RUNTIME_FILES = ('tools/evaluate_afterstate_td.py', 'native/libtetris4d_native.so', 'tetris_afterstate_td.py',
                 'tetris_neural.py', 'tetris_survival_v3.py', 'tetris_survival_v4.py', 'sparse_tetris_4d_optimized.py',
                 'tetris_gpu_simulation.py', 'tetris_gpu_encoding.py', 'tetris_gpu_spatial.py',
                 'tetris_gpu_board_features.py', 'tetris_survival_v5.py', 'train_overnight.py',
                 'train_survival_v4.py', 'tetris_game_library.py')


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def runtime_identity(root, device):
    files = RUNTIME_FILES
    if (root / 'tetris_gpu_cross.py').is_file():
        files += ('tetris_cross_survival.py', 'tetris_gpu_cross.py')
    name = torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else str(device)
    return dict(files={Path(path).name: file_hash(root / path) for path in files}, torch=str(torch.__version__),
                cuda=torch.version.cuda, device=name, numpy=np.__version__)


def terminal_piece(seed, placements):
    stream = game.PieceStream(seed, 'expanded')
    for _ in range(placements):
        stream.pop()
    return stream.pop()


def checked_row(path, identity, seed):
    record = json.loads(path.read_text())
    if record['identity'] != dict(identity, seed=seed) or record['data_sha256'] != digest(record['data']):
        raise ValueError(f'Cache identity or data digest differs for {path}')
    row = record['data']
    n = row['placements']
    if (row['seed'] != seed or row['reason'] != 'top_out' or type(n) is not int or n <= 0 or
            type(row['clears']) is not int or row['clears'] < 0 or row['terminal_piece'] != terminal_piece(seed, n)):
        raise ValueError(f'Expected a naturally terminated game for seed {seed}')
    if identity['script'] != LEGACY_DRIVER:
        import base64
        import zlib
        from tetris_cross_survival import visible_piece_fits
        masks = np.frombuffer(zlib.decompress(base64.b64decode(row['terminal_board'], validate=True)), dtype='<u2')
        if len(masks) != 4096 or visible_piece_fits(masks, row['terminal_piece']):
            raise ValueError(f'Claimed top-out still has a legal terminal-piece placement for seed {seed}')
    samples = row['samples']
    if len(samples) != (n + STRIDE - 1) // STRIDE:
        raise ValueError(f'Missing calibration samples for seed {seed}')
    totals = {band: [[], []] for band in ('early', 'late', 'all')}
    for j, sample in enumerate(samples):
        if (len(sample) != 3 or type(sample[0]) is not int or sample[0] != j * STRIDE or
                any(type(v) not in (float, int) or not math.isfinite(v) for v in sample[1:])):
            raise ValueError(f'Invalid calibration sample for seed {seed}')
        at, incumbent, donor = sample
        bands = ['all']
        if at * 4 < n:
            bands.append('early')
        if n - at <= 512:
            bands.append('late')
        for band in bands:
            for errors, prediction in zip(totals[band], (incumbent, donor)):
                errors.append(abs(prediction - (n - at)))
    mae = {band: dict(zip(('incumbent', 'donor'), map(statistics.mean, values))) for band, values in totals.items()}
    if row['mae'] != mae:
        raise ValueError(f'Recorded per-game MAE differs for seed {seed}')
    return mae


def _verify_budget(out):
    ledger = json.loads((out / 'gpu-time.json').read_text())
    calibrations = [r for r in ledger if r.get('phase') == 'calibration' and r.get('label') == 'critic']
    if not calibrations:
        raise ValueError('Missing closed calibration reservation for time_budget fallback')
    row = calibrations[-1]
    if (row['status'] != 'closed' or not math.isfinite(row['seconds']) or row['seconds'] < 0 or
            not math.isfinite(row['allowance_seconds']) or row['allowance_seconds'] < 0 or
            row['seconds'] < max(0., row['allowance_seconds'] - 90.)):
        raise ValueError('Calibration time_budget fallback lacks a closed reservation at its time limit')


def verify_gate(out, runtime_path, driver_path, device, *, incumbent_path=None, donor_path=None):
    out, runtime, driver = Path(out), Path(runtime_path), Path(driver_path)
    incumbent = incumbent_path or ARTIFACTS / 'v4-enhanced-run/ppo/best.pt'
    donor = donor_path or ARTIFACTS / 'spatial-ppo-improvements/experiment/control/last.pt'
    gate = json.loads((out / 'critic-gate.json').read_text())
    teacher = file_hash(runtime / 'models/trained-sigmoid.json')
    if (file_hash(incumbent), file_hash(donor), teacher) != (INCUMBENT, DONOR, TEACHER):
        raise ValueError('Calibration checkpoint or teacher hash differs')
    script = file_hash(driver)
    identity = dict(incumbent=INCUMBENT, donor=DONOR, runtime=runtime_identity(runtime, device), seeds=list(SEEDS),
                    script=script, stride=STRIDE, actor_rng_offset=2701)
    if script == LEGACY_DRIVER:
        if file_hash(driver.with_name('cross_run_budget.py')) != LEGACY_METER:
            raise ValueError('Archived calibration meter differs')
    else:
        identity['teacher'] = teacher
    if gate['identity'] != identity:
        raise ValueError('Calibration gate identity differs from supplied runtime, driver or protocol')
    if type(gate['complete']) is not bool:
        raise ValueError('Invalid calibration completion flag')
    counts, summaries = {}, {}
    for temperature in TEMPERATURES:
        key = str(temperature)
        details = dict(identity, temperature=temperature)
        folder = out / 'critic-cache' / digest(details)
        games = [checked_row(folder / f'{s}.json', details, s) for s in SEEDS if (folder / f'{s}.json').is_file()]
        counts[key] = len(games)
        if len(games) == len(SEEDS):
            mae = {band: {model: statistics.mean(game[band][model] for game in games) for model in ('incumbent', 'donor')}
                   for band in ('early', 'late', 'all')}
            summaries[key] = dict(mae=mae, passes=(mae['early']['donor'] <= mae['early']['incumbent'] * 1.05 and
                                                   mae['late']['donor'] < mae['late']['incumbent']))
    if gate['completed'] != counts or gate['complete'] != (len(summaries) == len(TEMPERATURES)):
        raise ValueError('Calibration counts differ from durable terminal games')
    if gate['complete']:
        chosen = 'donor' if all(row['passes'] for row in summaries.values()) else 'incumbent'
        if gate.get('temperatures') != summaries or gate.get('selected') != chosen or 'reason' in gate:
            raise ValueError('Calibration gate differs from the recomputed critic selection')
    else:
        if gate.get('reason') != 'time_budget' or gate.get('selected') != 'incumbent' or 'temperatures' in gate:
            raise ValueError('Incomplete calibration requires time_budget and the incumbent fallback')
        _verify_budget(out)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output', 'runtime', 'driver', 'incumbent', 'donor'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    gate = verify_gate(args.output, args.runtime, args.driver, args.device, incumbent_path=args.incumbent,
                       donor_path=args.donor)
    print(json.dumps({k: gate[k] for k in ('selected', 'complete', 'completed')}))


if __name__ == '__main__':
    main()
