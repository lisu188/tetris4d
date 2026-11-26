"""Cross-validate a small value readout using a run's completed calibration games."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import verify_cross_calibration as calibration


def load_games(run):
    gate_path = run / 'critic-gate.json'
    gate = json.loads(gate_path.read_bytes())
    protocol = json.loads((run / 'protocol.json').read_bytes())
    identity = gate['identity']
    seeds = identity['seeds']
    if (calibration.file_hash(gate_path) != protocol['critic_gate'] or not gate['complete']
            or gate['completed'] != {str(t): len(seeds) for t in calibration.TEMPERATURES}
            or len(set(seeds)) != len(seeds) or len(seeds) < 2):
        raise ValueError('Expected a complete signed, paired calibration cohort')
    games = {}
    for temperature in calibration.TEMPERATURES:
        details = dict(identity, temperature=temperature)
        folder = run / 'critic-cache' / calibration.digest(details)
        for seed in seeds:
            path = folder / f'{seed}.json'
            calibration.checked_row(path, details, seed)
            row = json.loads(path.read_bytes())['data']
            age, value, donor = np.asarray(row['samples'], dtype=np.float64).T
            inputs = np.column_stack((np.ones_like(age), value / 1000, age / 1000,
                                      np.full_like(age, temperature == 0.25)))
            games[seed, temperature] = (inputs, (row['placements'] - age) / 1000, donor)
    return games, protocol['critic_gate']


def fit(games, columns):
    # Equal total weight for every game, including the longest trajectories.
    x = np.concatenate([inputs[:, columns] / np.sqrt(len(y)) for inputs, y, _ in games])
    target = np.concatenate([y / np.sqrt(len(y)) for _, y, _ in games])
    weights, _, rank, _ = np.linalg.lstsq(x, target, rcond=None)
    if rank != len(columns):
        raise ValueError('Calibration inputs have insufficient rank')
    return weights


def measure(run):
    games, gate_sha = load_games(run)
    features = {'with_critic': [0, 1, 2, 3], 'age_and_temperature': [0, 2, 3]}
    errors = []
    for seed in sorted({s for s, _ in games}):
        training = [rows for (s, _), rows in games.items() if s != seed]
        weights = {model: fit(training, columns) for model, columns in features.items()}
        for temperature in calibration.TEMPERATURES:
            x, scaled_target, donor = games[seed, temperature]
            target = scaled_target * 1000
            predictions = {'incumbent': x[:, 1] * 1000, 'donor': donor}
            for model, columns in features.items():
                predictions[model] = np.maximum(0, x[:, columns] @ weights[model]) * 1000
            for stage, mask in {'opening': x[:, 2] == 0, 'all': slice(None), 'last_512': target <= 512}.items():
                row = dict(seed=seed, temperature=temperature, stage=stage, samples=len(target[mask]))
                for model, predictions_by_age in predictions.items():
                    predicted = predictions_by_age[mask]
                    actual = target[mask]
                    e = predicted - actual
                    row[model] = dict(mae=float(np.abs(e).mean()), bias=float(e.mean()),
                                      zero_with_over_512_left=int(((predicted <= 0) & (actual > 512)).sum()))
                errors.append(row)
    summary = {}
    for temperature in calibration.TEMPERATURES:
        by_stage = {}
        for stage in ('opening', 'all', 'last_512'):
            rows = [r for r in errors if r['temperature'] == temperature and r['stage'] == stage]
            by_stage[stage] = {name: {k: statistics.mean(r[name][k] for r in rows) for k in ('mae', 'bias')}
                               for name in ('incumbent', 'donor', *features)}
        summary[str(temperature)] = by_stage
    return dict(gate_sha256=gate_sha, inputs=['bias', 'incumbent_value_thousands', 'elapsed_thousands', 'is_cold'],
                input_columns=features, full_fit={m: fit(list(games.values()), c).tolist() for m, c in features.items()},
                held_out_summary=summary, held_out_games=errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Run containing protocol.json, critic-gate.json and critic-cache/')
    args = parser.parse_args()
    print(json.dumps(measure(args.run), indent=2, allow_nan=False))
