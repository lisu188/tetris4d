"""Measure all Spatial V4 tensor changes and their alignment at two matched endpoints."""

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_afterstate_td as ev


def describe(initial, final):
    delta = final - initial
    norm = float(initial.norm())
    magnitude = float(delta.norm())
    index = tuple(int(i) for i in torch.unravel_index(delta.abs().argmax(), delta.shape))
    result = dict(
        changed=int(delta.count_nonzero()),
        delta_l2=magnitude,
        relative_l2=magnitude / norm if norm else None,
        largest_change=dict(
            index=list(index),
            before=float(initial[index]),
            after=float(final[index]),
            signed_delta=float(delta[index]),
        ),
    )
    if delta.ndim == 2 and magnitude:
        energy = torch.linalg.svdvals(delta).square()
        result['delta_energy_in_top_direction'] = float(energy[0] / energy.sum())
        result['delta_energy_in_top_four_directions'] = float(energy[:4].sum() / energy.sum())
    return result


def compare(paths):
    weights = []
    for path in paths:
        model = ev.load_model(path, torch.device('cpu'))
        weights.append({name: t.detach().double() for name, t in model.named_parameters()})
    parent, control, diverse = weights
    if any(w.keys() != parent.keys() for w in weights):
        raise ValueError('Tensor names differ')
    rows = {}
    for key, initial in parent.items():
        c, d = control[key], diverse[key]
        if not (initial.shape == c.shape == d.shape):
            raise ValueError(f'Tensor shape differs for {key}')
        a, b = (c - initial).flatten(), (d - initial).flatten()
        denom = float(a.norm() * b.norm())
        rows[key] = dict(
            shape=list(initial.shape),
            count=initial.numel(),
            control=describe(initial, c),
            diverse=describe(initial, d),
            cosine_of_updates=float(torch.dot(a, b)) / denom if denom else None,
        )
    return dict(
        schema='spatial-v4-weight-deltas-v1',
        checksums=dict(zip(('incumbent', 'control', 'diverse'), map(ev.base.file_hash, paths))),
        tensors=rows,
        parameters=sum(row['count'] for row in rows.values()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--incumbent', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--diverse', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    ev.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    ev.base.durable_json(args.output, compare([args.incumbent, args.control, args.diverse]))


if __name__ == '__main__':
    main()
