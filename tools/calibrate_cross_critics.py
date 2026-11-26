"""Select an isolated V4 value head using paired, naturally terminated sampled games."""
import argparse
import base64
import struct
import zlib
from pathlib import Path
import os
import statistics
import sys
import time

# The initial calibration can use the previous immutable FP32 runtime while the new
# kernels are validated. Its source and checkpoint hashes travel with every result.
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT
if __name__ == '__main__':
    RUNTIME = Path(os.environ.get('TETRIS4D_CALIBRATION_RUNTIME', ROOT)).resolve()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(RUNTIME))
from tools import evaluate_afterstate_td as ev
from tools.cross_run_budget import reservation
import torch

SEEDS = tuple(48100001 + 2*i for i in range(8))
INCUMBENT = 'a6470b4ef118aea46969922559ed4f58826b82f0ae08c26ff81acdc0eac84cd9'
DONOR = '8da3b47e5a4319c7515dab18253f7b49d2f870b29c83ba3f10c03fe5ec78def8'
TEMPERATURES = (1., .25)


def errors(samples, placements):
    distances = [placements-at for at, *_ in samples]
    groups = {'early': [i for i, (at, *_) in enumerate(samples) if 4*at < placements],
              'late': [i for i, d in enumerate(distances) if 0 < d <= 512],
              'all': list(range(len(samples)))}
    return {group: {model: statistics.mean(abs(samples[i][j]-distances[i]) for i in indices)
                    for j, model in enumerate(('incumbent','donor'), 1)} if indices else None
            for group, indices in groups.items()}


def validated_row(row, seed):
    if row['seed'] != seed or row.get('reason') != 'top_out':
        raise ValueError(f'Critic calibration seed {seed} requires a natural top_out')
    return row


def cached_games(cache, identity):
    games = {}
    for seed in SEEDS:
        row = ev.checked_cached(cache / f'{seed}.json', dict(identity, seed=seed))
        if row is not None:
            games[seed] = validated_row(row, seed)
    return games


def select(reports):
    summary = {}
    for temperature in TEMPERATURES:
        games = reports[str(temperature)]
        if len(games) != len(SEEDS) or {g['seed'] for g in games} != set(SEEDS):
            raise ValueError('Critic selection requires completed matched calibration games')
        for row in games:
            validated_row(row, row['seed'])
        mae = {band: {name: statistics.mean(g['mae'][band][name] for g in games)
                      for name in ('incumbent', 'donor')} for band in ('early','late','all')}
        summary[str(temperature)] = dict(mae=mae, passes=mae['early']['donor'] <= 1.05*mae['early']['incumbent']
                                        and mae['late']['donor'] < mae['late']['incumbent'])
    return dict(selected='donor' if all(s['passes'] for s in summary.values()) else 'incumbent', temperatures=summary)


@torch.inference_mode()
def sample(model, donor, temperature, device, cache, identity, stop):
    from tetris_gpu_simulation import GpuPreparer
    base, game = ev.base, ev.game
    rows = cached_games(cache, identity)
    if len(rows) == len(SEEDS) or stop.expired():
        return list(rows.values())
    prep = GpuPreparer(model.config, device, group_exhaustive=True)
    # Each game owns its piece stream and action RNG. Partial boards and samples
    # permit an exact restart without repeating statistically expensive trajectories.
    states = [game.Game(seed, 'expanded') for seed in SEEDS]
    for state in states:
        base.library.ignore(state)
    generators = [torch.Generator(device=device).manual_seed(2701+s) for s in SEEDS]
    samples = [[] for _ in SEEDS]
    targets = [1]*len(SEEDS)
    live = {i for i, seed in enumerate(SEEDS) if seed not in rows}
    partial = cache/'active.json'
    previous = ev.checked_cached(partial, identity)
    if previous:
        for i in live:
            saved = previous.get(str(SEEDS[i]))
            if saved is None:
                continue
            state = states[i]
            for _ in range(saved['pieces']):
                state.piece_counts[state.current] += 1
                state.current, state.next_piece = state.next_piece, state.stream.pop()
            for key in ('pieces','score','cells_placed','layers','game_over'):
                setattr(state, key, saved[key])
            state.board = struct.unpack('<4096H', zlib.decompress(base64.b64decode(saved['board'])))
            generators[i].set_state(torch.tensor(list(base64.b64decode(saved['action_rng'])), dtype=torch.uint8))
            targets[i], samples[i] = saved['target'], saved['samples']

    def persist():
        active = {}
        for i in live:
            state = states[i]
            active[str(SEEDS[i])] = dict(pieces=state.pieces, score=state.score, cells_placed=state.cells_placed,
                layers=state.layers, game_over=state.game_over, target=targets[i], samples=samples[i],
                board=base64.b64encode(zlib.compress(struct.pack('<4096H', *state.board))).decode(),
                action_rng=base64.b64encode(generators[i].get_state().numpy().tobytes()).decode())
        ev.write_cache(partial, identity, active)

    began = time.monotonic()
    tick = 0
    while live and not stop.expired():
        current = sorted(live)
        result = prep.prepare_batch([states[i] for i in current], [targets[i] for i in current])
        played = [current[r] for r in result.environment_indices]
        if played:
            batch = result.batch
            # Fix reduction/GEMM shapes even when some peers finish or hit cache.
            fixed = {}
            n, width = batch['valid'].shape
            for key, tensor in batch.items():
                if key in ev.neural.CANDIDATE_FIELDS and width < model.config.shortlist:
                    dims = (0, 0) * (tensor.ndim - 2) + (0, model.config.shortlist - width)
                    tensor = torch.nn.functional.pad(tensor, dims)
                if n < len(SEEDS):
                    tensor = torch.cat((tensor, tensor[:1].expand(len(SEEDS)-n, *tensor.shape[1:])))
                fixed[key] = tensor.contiguous()
            logits = model.actor_logits(fixed)[:n]
            probabilities = (logits/temperature).softmax(-1)
            actions = torch.stack([torch.multinomial(probabilities[j], 1, generator=generators[i])[0]
                                   for j, i in enumerate(played)])
            if any(states[i].pieces % 128 == 0 for i in played):
                _, pooled = model._board_features(fixed['boards'], fixed['targets'])
                joined = torch.cat((pooled, fixed['context']), dim=1)
                values = (torch.stack([model.critic(joined), donor.critic(joined)], 1)
                          / base.REWARD_SCALE).cpu().tolist()
                for j, i in enumerate(played):
                    if states[i].pieces % 128 == 0:
                        samples[i].append([states[i].pieces, *values[j]])
            for r, candidate in zip(result.environment_indices, result.selected_candidates(actions)):
                i = current[r]
                state = states[i]
                targets[i] = int(result.targets[r])
                move = game.materialize_move(state.board, candidate)
                state.commit(move)
                state.locked_pieces = ()
                if move.layers:
                    new_target = game.next_hyperlayer_target(state.board, targets[i])
                    if new_target is not None:
                        targets[i] = new_target
        for i in live.difference(played):
            seed, state = SEEDS[i], states[i]
            row = dict(seed=seed, placements=state.pieces, clears=state.layers, terminal_piece=state.current,
                       reason='top_out' if state.game_over else 'target_or_no_move',
                       terminal_board=base64.b64encode(zlib.compress(struct.pack('<4096H', *state.board))).decode(),
                       samples=samples[i], mae=errors(samples[i], state.pieces))
            validated_row(row, seed)
            ev.write_cache(cache/f'{seed}.json', dict(identity, seed=seed), row)
            rows[seed] = row
        live = set(played)
        tick += 1
        if tick % 512 == 0:
            persist()
            base.durable_json(cache/'progress.json', dict(temperature=temperature, completed=len(rows),
                              active={SEEDS[i]:states[i].pieces for i in live}, seconds=time.monotonic()-began))
    if live:
        persist()
    elif partial.exists():
        partial.unlink()  # Every trajectory now has its durable terminal record.
    return [rows[s] for s in SEEDS if s in rows]


def calibrate(out, incumbent, donor_path, device, stop, identity):
    reports = {}
    model = donor = None
    for temperature in TEMPERATURES:
        details = dict(identity, temperature=temperature)
        cache = out / 'critic-cache' / ev.digest(details)
        games = cached_games(cache, details)
        if len(games) == len(SEEDS) or stop.expired():
            reports[str(temperature)] = list(games.values())
            continue
        if model is None:
            model, donor = [ev.load_model(p, device) for p in (incumbent, donor_path)]
            model.fused_encoding = True
        reports[str(temperature)] = sample(model, donor, temperature, device, cache, details, stop)
    complete = all(len(rows) == len(SEEDS) for rows in reports.values())
    result = dict(identity=identity, complete=complete, completed={t:len(rows) for t,rows in reports.items()})
    if complete:
        result.update(select(reports))
    else:
        result.update(selected='incumbent', reason=stop.reason or 'incomplete calibration')
    ev.base.durable_json(out / 'critic-gate.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output','incumbent','donor'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
    ev.game.configure_backend('native')
    ev.game.load_checkpoint(RUNTIME/'models/trained-sigmoid.json')
    torch.set_num_threads(1)
    device = ev.neural.device_for('cuda')
    for path, expected in ((args.incumbent, INCUMBENT), (args.donor, DONOR)):
        if ev.base.file_hash(path) != expected:
            raise ValueError(f'Unexpected checkpoint {path}')
    with ev.v4.v2.device_lock(device), ev.base.Budget(float('inf')) as stop, reservation(out, 'calibration', 'critic', stop):
        runtime = ev.runtime_identity(device)
        identity = dict(incumbent=INCUMBENT, donor=DONOR, runtime=runtime, seeds=list(SEEDS),
                        script=ev.base.file_hash(Path(__file__)), teacher=ev.base.file_hash(RUNTIME/'models/trained-sigmoid.json'),
                        stride=128, actor_rng_offset=2701)
        result = calibrate(out, args.incumbent, args.donor, device, stop, identity)
        print({k:v for k,v in result.items() if k != 'identity'}, flush=True)
        return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
