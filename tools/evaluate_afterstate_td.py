"""Calibrate the afterstate policy on historical positions and evaluate fresh, paired games."""
from __future__ import annotations

import argparse
import base64
import copy
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zlib

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_neural as neural
import train_survival_v4 as v4

base, game = v4.base, v4.game
TOP_K = 16
POLICY_ORDER = ('actor', 'frozen', 'ppo', 'previous_td', 'initial_td', 'td')
DEVELOPMENT_SEEDS = tuple(910001 + 2*i for i in range(8))
HOLDOUT_SEEDS = tuple(7310001 + 2*i for i in range(16))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def runtime_identity(device):
    paths = [Path(__file__), ROOT/'native/libtetris4d_native.so']
    paths += [ROOT/(name+'.py') for name in ('tetris_afterstate_td', 'tetris_neural', 'tetris_survival_v3', 'tetris_survival_v4',
              'sparse_tetris_4d_optimized', 'tetris_gpu_simulation', 'tetris_gpu_encoding',
              'tetris_gpu_spatial', 'tetris_gpu_board_features', 'tetris_survival_v5', 'train_overnight',
              'train_survival_v4', 'tetris_game_library', 'tetris_cross_survival', 'tetris_gpu_cross')]
    return {'files': {path.name: base.file_hash(path) for path in paths},
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else str(device), 'numpy': np.__version__}


def load_model(path, device):
    model, payload = neural.load_checkpoint(path, device)
    if payload['teacher_parameters'] != game.network_parameters():
        raise ValueError('Checkpoint and candidate preparation use different teacher weights')
    if model.config.screening != 'exhaustive':
        model.config = replace(model.config, screening='exhaustive')
    return model.eval()


def population_beta(spreads):
    if not spreads or not all(math.isfinite(x) and x >= 0 for x in spreads):
        raise ValueError('Invalid afterstate value spreads')
    sigma = statistics.median(spreads)
    if sigma <= 0 or not math.isfinite(1 / sigma):
        raise ValueError('Top-16 afterstate values have no usable spread')
    return sigma, 1 / sigma


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_cached(path, identity):
    if not path.is_file():
        return None
    record = read_json(path)
    if record['identity'] != identity:
        raise ValueError(f'Cached evaluation identity differs in {path}')
    if record['data_sha256'] != digest(record['data']):
        raise ValueError(f'Cached evaluation data digest differs in {path}')
    return record['data']


def write_cache(path, identity, data):
    base.durable_json(path, {'identity': identity, 'data': data, 'data_sha256': digest(data)})


@torch.inference_mode()
def calibrate(args, device, budget, runtime):
    import tetris_afterstate_td as td
    from tetris_gpu_simulation import GpuPreparer

    report = read_json(args.positions_report)
    positions = [tuple(row) for row in report['position_ids']]
    source_hash = base.file_hash(args.source)
    if report.get('schema') != 'tetris4d-survival-benchmark-v1':
        raise ValueError('Expected a survival benchmark position report')
    if len(positions) != 32 or len(set(positions)) != 32 or report['checkpoint_sha256'] != source_hash:
        raise ValueError('Expected the 32 distinct development positions for the supplied checkpoint')
    grouped = defaultdict(set)
    for game_id, offset in positions:
        if type(offset) is not int or offset < 0:
            raise ValueError('Invalid calibration offset')
        grouped[game_id].add(offset)
    library = v4.library_root(args.library)
    files = {entry['id']: library/entry['file'] for entry in base.library.iter_index(library)
             if entry['id'] in grouped}
    if files.keys() != grouped.keys():
        raise ValueError('Calibration games missing from the library')
    model = td.from_v4(load_model(args.source, device)).to(device).eval()
    preparer = GpuPreparer(model.base.config, device)
    identity = {'source_sha256': source_hash, 'position_report_sha256': base.file_hash(args.positions_report),
                'weights_sha256': base.file_hash(args.weights), 'runtime': runtime,
                'screening': 'exhaustive', 'top_k': TOP_K}
    cache = args.cache_dir/digest(identity)
    spreads = {}
    for game_id, offsets in grouped.items():
        key = dict(identity, game_id=game_id, offsets=sorted(offsets), record_sha256=base.file_hash(files[game_id]))
        path = cache/(digest(key)+'.json')
        values = checked_cached(path, key)
        if values is None:
            if budget.expired():
                return False
            states, targets, numbers = [], [], []
            record = base.library.load(files[game_id])
            for index, (state, _, move_row) in enumerate(base.library.replay(record, verify=False)):
                # Only the board and piece stream are needed for headless replay.
                state.locked_pieces = ()
                if budget.expired():
                    return False
                if index in offsets:
                    states.append(copy.deepcopy(state))
                    targets.append(int(move_row[6]))
                    numbers.append(index)
                if len(numbers) == len(offsets):
                    break
            if set(numbers) != offsets:
                raise ValueError(f'Calibration position missing in game {game_id}')
            result = preparer.prepare_batch(states, targets)
            if list(result.environment_indices) != list(range(len(states))):
                raise ValueError('Calibration position has no legal candidate')
            pool = model.prepare_device(result, states)
            scores = model.score(pool.features)
            values = []
            for i, number in enumerate(numbers):
                valid = scores[i][pool.valid[i]]
                if len(valid) < 1:
                    raise ValueError('Empty top-16 calibration shortlist')
                values.append([number, float(valid.double().std(unbiased=False))])
            write_cache(path, key, values)
            print(f'Calibrated {len(spreads)+len(values)}/32 positions', flush=True)
        spreads.update({(game_id, number): value for number, value in values})
    ordered = [spreads[pos] for pos in positions]
    sigma, beta = population_beta(ordered)
    base.durable_json(args.output, {**identity, 'schema': 'afterstate-td-beta-v1', 'complete': True,
                                   'beta': beta, 'sigma': sigma, 'std_ddof': 0,
                                   'position_ids': positions, 'spreads': ordered})
    return True


DECISION_COUNTERS = ('decisions', 'planner_triggers', 'immediate_losses', 'immediate_rescues',
                     'planner_overrides', 'direct_weight_changes', 'residual_changes')
# Count the base choice losing an existing CROSS landing, and the subset avoided by the guard.
LAST_CROSS_COUNTERS = ('last_cross_losses', 'last_cross_overrides')
LATENCY_SCOPE = ('Wall time from batched preparation through the last game commit in the lockstep, '
                 'including CPU and device work and warmup. One sample per committed game move. '
                 'Nearest-rank percentiles at 0.1 ms resolution.')


def _latency(histogram):
    values = sorted((int(tenths), count) for tenths, count in histogram.items())
    total = sum(n for _, n in values)
    result = {'scope': LATENCY_SCOPE, 'samples': total, 'p50_ms': None, 'p95_ms': None}
    for p in (50, 95):
        seen = 0
        for tenths, count in values:
            seen += count
            if seen >= math.ceil(total * p / 100):
                result[f'p{p}_ms'] = tenths / 10
                break
    return result


def summarize(rows, seeds, policy, cap):
    rows = [dict(rows[s]) for s in seeds if s in rows]
    for row in rows:
        row['clears_per_1k'] = 1000 * row['clears'] / row['placements'] if row['placements'] else None
    placements = [r['placements'] for r in rows]
    total = sum(placements)
    clears_per_1k = 1000 * sum(r['clears'] for r in rows) / total if total else None
    latency, counters = Counter(), Counter()
    for row in rows:
        latency.update(row.get('batch_latency_histogram_0p1ms', {}))
        counters.update(row.get('decisions', {}))
    keys = DECISION_COUNTERS + tuple(k for k in LAST_CROSS_COUNTERS if k in counters)
    return {'policy': policy, 'cap': cap, 'decision_counts': {k: counters[k] for k in keys},
            'move_latency': _latency(latency), 'complete': len(rows) == len(seeds),
            'requested_seeds': list(seeds), 'seeds': [r['seed'] for r in rows],
            'placements': placements, 'clears': [r['clears'] for r in rows], 'games': rows,
            'clears_per_1k': clears_per_1k,
            'stop_reasons': [r['reason'] for r in rows],
            'target_or_no_move': sum(r['reason'] == 'target_or_no_move' for r in rows),
            'top_outs': sum(r['reason'] == 'top_out' for r in rows), 'censored': sum(r['reason']=='cap' for r in rows),
            'mean': statistics.mean(placements) if rows else None,
            'median': statistics.median(placements) if rows else None, 'minimum': min(placements, default=None)}


def _fixed_actor_logits(model, batch, size, *, return_base=False):
    """Give each cached greedy game the same FP32 kernel shapes, independent of its peers."""
    rows, width = batch['valid'].shape
    fixed = {}
    for name, value in batch.items():
        if name in neural.CANDIDATE_FIELDS and width < model.config.shortlist:
            pad = (0, 0) * (value.ndim - 2) + (0, model.config.shortlist - width)
            value = torch.nn.functional.pad(value, pad)
        if rows < size:
            value = torch.cat((value, value[:1].expand(size - rows, *value.shape[1:])))
        fixed[name] = value.contiguous()
    if return_base:
        adjusted, base_logits = model.actor_logits(fixed, return_base=True)
        return adjusted[:rows, :width], base_logits[:rows, :width]
    return model.actor_logits(fixed)[:rows, :width]


def _fixed_actor_choices(model, batch, size):
    return _fixed_actor_logits(model, batch, size).argmax(-1)


class _HostOptions:
    def __init__(self, piece, rows):
        self.piece, self.rows = piece, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return game.Candidate(self.piece, *map(int, row[:6]), float(row[6]))


def select_moves(model, states, targets, scoring, device, preparer, ppo_batch_size=None, preview_search=False, decision_stats=None):
    """Map live batch indices to their (recovered target, chosen placement)."""
    from tetris_cross_survival import mobility_features, preserve_last_cross, preview_choice
    guard = preview_search == 'preserve-last-cross'
    if preview_search and (scoring or model.config.architecture != 'survival-spatial-v4'):
        raise ValueError('The selected move mode requires a spatial V4 actor')
    compare = decision_stats is not None and (model.config.cross_weight or model.config.cross_residual)
    change_key = 'direct_weight_changes' if model.config.cross_weight else 'residual_changes'

    def actor(batch):
        if ppo_batch_size:
            return _fixed_actor_logits(model, batch, ppo_batch_size, return_base=bool(compare))
        if compare:
            return model.actor_logits(batch, return_base=True)
        return model(batch)['logits']

    if preparer is None:
        moves = {}
        config = model.base.config if scoring else model.config
        for index, (state, target) in enumerate(zip(states, targets)):
            prepared = neural.prepare(state, target, config)
            if prepared is None:
                continue
            if guard and 'cross_mobility' not in prepared.arrays:
                prepared.arrays['cross_mobility'] = mobility_features(state.board, prepared.options)
            if scoring:
                pool = model.prepare_cpu(prepared, state)
                choice = int(pool.indices[0, model.choose(pool).item()])
            else:
                batch = neural.collate([prepared.arrays], device)
                values = actor(batch)
                if compare:
                    logits, base_logits = values
                    decision_stats[index][change_key] = int((logits.argmax(1) != base_logits.argmax(1)).item())
                else:
                    logits = values
                if guard:
                    choices, threat, override = preserve_last_cross(logits, batch['valid'], batch['cross_mobility'])
                    choice = int(choices.item())
                    if decision_stats is not None:
                        decision_stats[index].update(last_cross_losses=int(threat.item()), last_cross_overrides=int(override.item()))
                elif preview_search:
                    choice = preview_choice(state, int(prepared.arrays['targets']), prepared.options,
                                            logits[0].detach().cpu().numpy(), config,
                                            prepared.arrays['cross_mobility'][:, 1] if 'cross_mobility' in prepared.arrays else None,
                                            decision_stats[index] if decision_stats is not None else None,
                                            rescue_only=preview_search == 'rescue-only')
                else:
                    choice = int(logits.argmax(1).item())
            moves[index] = (int(prepared.arrays['targets']), prepared.options[choice])
        return moves
    result = preparer.prepare_batch(states, targets)
    if not result.environment_indices:
        return {}
    if preview_search or compare:
        batch = result.batch
        if guard and 'cross_mobility' not in batch:
            # A custom preparer can supply ordinary V4 features without mobility.
            # Recompute from its actual final shortlist and the public board.
            mobility = np.zeros((*batch['valid'].shape, 2), dtype=np.float32)
            valid_rows = batch['valid'].detach().cpu().numpy().astype(bool)
            for row, index in enumerate(result.environment_indices):
                mobility[row, valid_rows[row]] = mobility_features(states[index].board, result.prepared[index].options)
            batch = {**batch, 'cross_mobility': torch.as_tensor(mobility, device=batch['valid'].device)}
        values = actor(batch)
        if compare:
            logits, base_logits = values
            changed = (logits.argmax(1) != base_logits.argmax(1))
        else:
            logits = values
        if guard or not preview_search:
            if guard:
                choices, threat, override = preserve_last_cross(logits, batch['valid'], batch['cross_mobility'])
            else:
                choices = logits.argmax(1)
            if decision_stats is None:
                return {i: (int(result.targets[i]), c) for i, c in
                        zip(result.environment_indices, result.selected_candidates(choices))}
            columns = [result.selected_metadata(choices)]
            keys = []
            if compare:
                columns.append(changed[:, None].double())
                keys.append(change_key)
            if guard:
                columns.extend((threat[:, None].double(), override[:, None].double()))
                keys.extend(LAST_CROSS_COUNTERS)
            host = torch.cat(columns, dim=1).cpu().tolist()
            candidates = result.candidates_from_metadata([row[:7] for row in host])
            for i, row in zip(result.environment_indices, host):
                decision_stats[i].update(dict(zip(keys, map(int, row[7:]))))
            return {i: (int(result.targets[i]), c) for i, c in zip(result.environment_indices, candidates)}
        fields = [result._option_specs.double(), result._option_scores[..., None], logits.double()[..., None]]
        if 'cross_mobility' in batch:
            fields.append(batch['cross_mobility'][..., 1:2].double())
        if compare:
            fields.append(changed[:, None, None].expand(-1, logits.shape[1], 1).double())
        packed = torch.cat(fields, -1).detach().cpu().numpy()
        moves = {}
        for row, i in enumerate(result.environment_indices):
            data = packed[row, :len(result.prepared[i].options)]
            options = _HostOptions(states[i].current, data)
            target = int(result.targets[i])
            if compare:
                decision_stats[i][change_key] = int(data[0, -1])
            choice = preview_choice(states[i], target, options, data[:, 7], model.config,
                                    data[:, 8] if 'cross_mobility' in batch else None,
                                    decision_stats[i] if decision_stats is not None else None,
                                    rescue_only=preview_search == 'rescue-only')
            moves[i] = target, options[choice]
        return moves
    if scoring:
        pool = model.prepare_device(result, states)
        actions = pool.indices.gather(1, model.choose(pool)[:, None]).squeeze(1)
    elif ppo_batch_size:
        actions = _fixed_actor_choices(model, result.batch, ppo_batch_size)
    else:
        actions = model(result.batch)['logits'].argmax(-1)
    return {i: (int(result.targets[i]), candidate)
            for i, candidate in zip(result.environment_indices, result.selected_candidates(actions))}


def _snapshot(state, target, wall_seconds):
    rng_version, rng_state, gauss = state.stream.rng.getstate()
    packed = np.asarray(state.board, dtype='<u2').tobytes()
    return {'version': 1, 'seed': state.seed, 'piece_set': state.piece_set, 'target': target,
            'board': base64.b64encode(zlib.compress(packed, 1)).decode('ascii'),
            'pieces': state.pieces, 'cells_placed': state.cells_placed, 'layers': state.layers, 'score': state.score,
            'piece_counts': dict(state.piece_counts), 'current': state.current, 'next_piece': state.next_piece,
            'bag': list(state.stream.bag), 'rng': [rng_version, list(rng_state), gauss],
            'wall_seconds': wall_seconds}


def _resume(state, data):
    if data['version'] != 1 or (data['seed'], data['piece_set']) != (state.seed, state.piece_set):
        raise ValueError('Invalid partial game identity')
    board = np.frombuffer(zlib.decompress(base64.b64decode(data['board'], validate=True)), dtype='<u2')
    if len(board) != 4096 or not 0 <= data['target'] < 16 or sum(data['piece_counts'].values()) != data['pieces']:
        raise ValueError('Invalid partial game state')
    state.board = tuple(map(int, board))
    for name in ('pieces', 'layers', 'score', 'cells_placed', 'current', 'next_piece'):
        setattr(state, name, data[name])
    state.piece_counts = dict(data['piece_counts'])
    state.stream.bag = list(data['bag'])
    version, rng, gauss = data['rng']
    state.stream.rng.setstate((version, tuple(rng), gauss))
    return data['target'], data['wall_seconds']


def _paired_greedy(placements, reason, first_loss, cap):
    if first_loss is None:
        return {'placements': placements, 'reason': reason, 'physical_top_out': reason == 'top_out'}
    # The ordinary evaluator gives the cap precedence even if the last placement is fatal.
    return {'placements': first_loss, 'reason': 'cap' if first_loss == cap else 'top_out', 'physical_top_out': True}


@torch.inference_mode()
def evaluate_fresh(model, seeds, *, policy, cap, device, top_k=TOP_K, deadline=None,
                   stop_requested=None, on_finish=None, ppo_batch_size=None, group_exhaustive=False,
                   preview_search=False, two_stage=False, headless=False, partials=None, on_partial=None,
                   progress_seconds=60.):
    """Fresh greedy lockstep evaluation. Per-game time includes concurrent lockstep peers."""
    import tetris_afterstate_td as td

    seeds = tuple(seeds)
    if top_k != TOP_K or cap < 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Expected a nonnegative cap, distinct fresh seeds and top_k=16')
    if policy not in POLICY_ORDER:
        raise ValueError('Unknown evaluation policy')
    if ppo_batch_size is not None and (policy != 'ppo' or type(ppo_batch_size) is not int or ppo_batch_size < 1
                                       or not hasattr(model, 'actor_logits')):
        raise ValueError('Fixed-shape greedy evaluation requires a Spatial PPO actor and a positive batch size')
    began = time.monotonic()
    original_model = model
    was_training = model.training
    model.eval()
    previous_flags = game.TWO_STAGE_EXHAUSTIVE, game.HEADLESS_COMMIT
    game.set_simulation_fast_paths(two_stage=two_stage, headless=headless)
    try:
        if policy == 'frozen':
            model = td.from_v4(model).to(device).eval()
        scoring = policy in ('frozen', 'previous_td', 'initial_td', 'td')
        preparer = None
        if torch.device(device).type == 'cuda':
            from tetris_gpu_simulation import GpuPreparer
            preparer = GpuPreparer(model.base.config if scoring else model.config, device, group_exhaustive=group_exhaustive,
                                   two_stage=two_stage, require_cross_mobility=preview_search == 'preserve-last-cross')
        states = {seed: game.Game(seed, 'expanded') for seed in seeds}
        for state in states.values():
            base.library.ignore(state)
        capacity = ppo_batch_size or len(seeds)
        targets, rows, live = dict.fromkeys(seeds, 1), {}, list(seeds[:capacity])
        elapsed = dict.fromkeys(seeds, 0.)
        rescue_only = preview_search == 'rescue-only'
        counters = DECISION_COUNTERS + (LAST_CROSS_COUNTERS if preview_search == 'preserve-last-cross' else ())
        first_greedy_loss = {}
        decisions = {s: Counter() for s in seeds}
        latency = {s: Counter() for s in seeds}
        for seed, data in (partials or {}).items():
            targets[seed], elapsed[seed] = _resume(states[seed], data)
            decisions[seed].update(data.get('decisions', {}))
            if rescue_only:
                first = data.get('first_greedy_loss_placement')
                if ((first is not None) != bool(decisions[seed]['immediate_losses']) or
                        (first is not None and (type(first) is not int or not 1 <= first <= states[seed].pieces))):
                    raise ValueError(f'Invalid paired greedy loss in partial for seed {seed}')
                if first is not None:
                    first_greedy_loss[seed] = first
            latency[seed].update(data.get('batch_latency_histogram_0p1ms', {}))
        entered = dict.fromkeys(live, began)
        next_seed = len(live)
        next_progress = began + progress_seconds
        def save_live():
            if on_partial:
                for seed in live:
                    state = states[seed]
                    if state.pieces:
                        snapshot = _snapshot(state, targets[seed], elapsed[seed] + time.monotonic() - entered[seed])
                        snapshot.update(decisions=dict(decisions[seed]), batch_latency_histogram_0p1ms=dict(latency[seed]))
                        if rescue_only:
                            snapshot['first_greedy_loss_placement'] = first_greedy_loss.get(seed)
                        on_partial(snapshot)
        while live:
            if (deadline is not None and time.monotonic() >= deadline) or (stop_requested and stop_requested()):
                save_live()
                break
            if time.monotonic() >= next_progress:
                save_live()
                counts = [states[s].pieces for s in live]
                print(f'{policy} {len(rows)}/{len(seeds)} games complete, live placement range {min(counts)}–{max(counts)}', flush=True)
                next_progress = time.monotonic() + progress_seconds
            current = [states[s] for s in live]
            options = (ppo_batch_size,) if ppo_batch_size else ()
            stats = defaultdict(dict)
            telemetry = {}
            if preview_search or model.config.cross_weight or model.config.cross_residual:
                telemetry = {'preview_search': preview_search, 'decision_stats': stats}
            move_started = time.monotonic_ns()
            moves = select_moves(model, current, [targets[s] for s in live], scoring, device, preparer, *options, **telemetry)
            playable = [live[i] for i in moves]
            for i, (target, candidate) in moves.items():
                seed, state = live[i], current[i]
                targets[seed] = target
                if headless:
                    game.play_candidate(state, candidate)
                else:
                    state.commit(game.materialize_move(state.board, candidate))
                state.locked_pieces = ()  # Rendering history is unused by headless decisions.
                if candidate.layers:
                    following = game.next_hyperlayer_target(state.board, target)
                    if following is not None:
                        targets[seed] = following
            tenths = str((time.monotonic_ns() - move_started + 50000) // 100000)
            for i in moves:
                seed = live[i]
                decisions[seed].update(stats[i])
                decisions[seed]['decisions'] += 1
                if rescue_only and stats[i].get('immediate_losses'):
                    first_greedy_loss.setdefault(seed, states[seed].pieces)
                if not preview_search or preview_search == 'preserve-last-cross':
                    decisions[seed]['immediate_losses'] += int(states[seed].game_over)
                latency[seed][tenths] += 1
            for seed in live:
                state = states[seed]
                if seed in playable and not state.game_over and (cap == 0 or state.pieces < cap):
                    continue
                reason = 'cap' if cap and state.pieces >= cap else ('top_out' if state.game_over else 'target_or_no_move')
                row = {'seed': seed, 'placements': state.pieces, 'clears': state.layers,
                       'reason': reason, 'wall_seconds': elapsed[seed] + time.monotonic() - entered[seed],
                       'device': str(device), 'decisions': {k: decisions[seed][k] for k in counters},
                       'batch_latency_histogram_0p1ms': dict(latency[seed]), 'move_latency': _latency(latency[seed])}
                if rescue_only:
                    row['paired_greedy'] = _paired_greedy(state.pieces, reason, first_greedy_loss.get(seed), cap)
                rows[seed] = row
                if on_finish:
                    on_finish(row)
            live = [seed for seed in playable if seed not in rows]
            added = seeds[next_seed:next_seed + capacity - len(live)]
            live.extend(added)
            entered.update(dict.fromkeys(added, time.monotonic()))
            next_seed += len(added)
        return {**summarize(rows, seeds, policy, cap), 'wall_seconds': time.monotonic()-began}
    finally:
        original_model.train(was_training)
        game.set_simulation_fast_paths(two_stage=previous_flags[0], headless=previous_flags[1])


def development_score(result):
    if not result['complete'] or tuple(result['seeds']) != DEVELOPMENT_SEEDS:
        raise ValueError('Checkpoint selection requires completed development games')
    return result['mean'], result['median']


def paired(candidate, control):
    if (not candidate['complete'] or not control['complete'] or candidate['seeds'] != control['seeds']
            or candidate['cap'] != control['cap']):
        raise ValueError('Paired analysis requires identical completed seeds and caps')
    changes = np.asarray(candidate['placements'], dtype=float) - np.asarray(control['placements'])
    rng = np.random.default_rng(20261123)
    means = changes[rng.integers(0, len(changes), size=(10000, len(changes)))].mean(axis=1)
    low, high = np.quantile(means, [.025,.975]).tolist()
    gain = float(changes.mean())
    ratio = gain/control['mean'] if control['mean'] else None
    return {'control': control['policy'], 'cap': candidate['cap'], 'mean_difference': gain,
            'relative_gain': ratio, 'paired_bootstrap_95': [low, high], 'wins': int((changes>0).sum()),
            'ties': int((changes==0).sum()), 'passes': ratio is not None and ratio >= .05 and low > 0}


def rank(policies):
    # The final tie-break is fixed before evaluation and independent of cache/insertion order.
    return sorted(policies, key=lambda name: (-policies[name]['mean'], -policies[name]['median'],
                                              POLICY_ORDER.index(name)))


def extension_pair(results):
    order = rank(results)[:2]
    if len(order) == 2 and any(2*results[name]['censored'] >= len(results[name]['seeds']) for name in order):
        return order
    return []


def evaluate_checkpoint(checkpoint, checkpoint_hash, name, seeds, cap, split, device, budget, runtime,
                        weights_hash, cache_dir, *, ppo_batch_size=None, group_exhaustive=False,
                        preview_search=False, two_stage=False, headless=False, progress_seconds=60.):
    started = time.monotonic()
    identity = {'policy': name, 'checkpoint': checkpoint_hash, 'weights': weights_hash, 'runtime': runtime,
                'screening': 'exhaustive', 'top_k': TOP_K, 'cap': cap}
    if ppo_batch_size is None:
        identity.update(split=split, seeds=list(seeds))
    else:
        if (name != 'ppo' or type(ppo_batch_size) is not int or ppo_batch_size < 1
                or not seeds or cap < 0 or len(set(seeds)) != len(seeds)):
            raise ValueError('Expected distinct seeds and a positive greedy PPO evaluation batch size')
        identity.update(greedy_actor='fixed-fp32-v1', batch_size=ppo_batch_size, piece_set='expanded')
    if group_exhaustive:
        identity['group_exhaustive'] = True
    if preview_search == 'preserve-last-cross':
        identity['move_guard'] = 'preserve-existing-cross-landing-v1'
    elif preview_search:
        identity['preview_search'] = ('one-visible-physical-rescue-paired-v2' if preview_search == 'rescue-only'
                                      else 'one-visible-cross-8-16x4-v1')
    if two_stage:
        identity['two_stage'] = True
    if headless:
        identity['headless'] = True
    path = cache_dir/digest(identity)
    rows = {}
    def validate(row, seed):
        if (row['seed'] != seed or row['placements'] < 0 or (cap and row['placements'] > cap) or row['clears'] < 0
                or row['reason'] not in ('cap', 'top_out', 'target_or_no_move')
                or (row['reason'] == 'cap') != bool(cap and row['placements'] == cap)):
            raise ValueError(f'Invalid completed evaluation for seed {seed}')
        if preview_search == 'rescue-only':
            paired = row.get('paired_greedy', {})
            n = paired.get('placements')
            losses = row['decisions']['immediate_losses']
            if (type(n) is not int or type(paired.get('physical_top_out')) is not bool
                    or not 0 <= n <= row['placements'] or (losses and not n)
                    or paired != _paired_greedy(row['placements'], row['reason'], n if losses else None, cap)):
                raise ValueError(f'Invalid paired greedy evaluation for seed {seed}')

    def record(row):
        validate(row, row['seed'])
        key = dict(identity, seed=row['seed'])
        write_cache(path/f"{row['seed']}.json", key, row)
        rows[row['seed']] = row
        (path/f"{row['seed']}.partial.json").unlink(missing_ok=True)

    def record_partial(snapshot):
        seed = snapshot['seed']
        write_cache(path/f'{seed}.partial.json', dict(identity, partial_seed=seed), snapshot)
    for seed in seeds:
        row = checked_cached(path/f'{seed}.json', dict(identity, seed=seed))
        if row is not None:
            validate(row, seed)
            rows[seed] = row
    missing = [s for s in seeds if s not in rows]
    if missing and not budget.expired():
        start = time.monotonic()
        model = load_model(checkpoint, device)
        if base.file_hash(checkpoint) != checkpoint_hash:
            raise ValueError('Evaluation checkpoint changed during load')
        if ppo_batch_size and torch.device(device).type == 'cuda':
            model.fused_encoding = True
        partials = {}
        for seed in missing:
            snapshot = checked_cached(path/f'{seed}.partial.json', dict(identity, partial_seed=seed))
            if snapshot is not None:
                partials[seed] = snapshot
        evaluate_fresh(model, missing, policy=name, cap=cap, device=device, stop_requested=budget.expired,
                       on_finish=record, ppo_batch_size=ppo_batch_size, group_exhaustive=group_exhaustive,
                       preview_search=preview_search, two_stage=two_stage, headless=headless, partials=partials,
                       on_partial=record_partial, progress_seconds=progress_seconds)
        print(f'{name} cap {cap}: {len(rows)}/{len(seeds)} games, {time.monotonic()-start:.1f}s', flush=True)
        del model
    result = summarize(rows, seeds, name, cap)
    result['invocation_wall_seconds'] = time.monotonic()-started
    return result


def select_final(args, device, budget, runtime):
    """Consider the final update alongside the best completed scheduled development trial."""
    run = args.run_dir
    summary = read_json(run/'summary.json')
    selection = read_json(run/'selection.json')
    manifest = read_json(run/'manifest.json')
    cap = manifest['selection_pieces']
    weights_hash = base.file_hash(args.weights)
    if selection['cap'] != cap or weights_hash != manifest['signature']['weights_sha256']:
        raise ValueError('Development cap or teacher weights differ from the training manifest')
    final_path = run/'last.pt'
    final_hash = base.file_hash(final_path)
    if (not summary['complete'] or summary['stop_reason'] != 'time_budget' or not summary['updates']
            or final_hash != summary['last_sha256'] or tuple(selection['seeds']) != DEVELOPMENT_SEEDS):
        raise ValueError('Final selection requires completed training and its pinned checkpoint and development seeds')
    best = selection['best']
    if best:
        if (not selection['complete'] or best != summary['best'] or best['sha256'] != selection['best_sha256']
                or base.file_hash(run/'best.pt') != best['sha256']):
            raise ValueError('Scheduled development winner differs from its selection record')
    elif summary['best'] is not None:
        raise ValueError('Scheduled development winner is missing')
    candidate = evaluate_checkpoint(final_path, final_hash, 'td', DEVELOPMENT_SEEDS, cap, 'development',
                                    device, budget, runtime, weights_hash, args.cache_dir)
    if not candidate['complete']:
        return False
    score = development_score(candidate)
    final_wins = best is None or score > (best['mean'], best['median'])
    winner, winner_hash = (final_path, final_hash) if final_wins else (run/'best.pt', best['sha256'])
    base.durable_json(args.output, dict(complete=True, best_sha256=winner_hash, checkpoint=winner.name,
                      checkpoint_dir=str(run.resolve()), seeds=list(DEVELOPMENT_SEEDS), cap=cap,
                      final_development=candidate, final_sha256=final_hash, scheduled_best=best))
    return True


def evaluate(args, device, budget, runtime):
    split = args.split
    seeds = DEVELOPMENT_SEEDS if split == 'development' else HOLDOUT_SEEDS
    paths = {'actor': args.source, 'frozen': args.source}
    if args.ppo:
        paths['ppo'] = args.ppo
    if args.previous_td:
        paths['previous_td'] = args.previous_td
    if args.initial_td:
        paths['initial_td'] = args.initial_td
    if args.td:
        paths['td'] = args.td
    if split == 'holdout':
        if not args.ppo or not args.td or not args.selection:
            raise ValueError('Holdout requires the frozen PPO control and the selected TD checkpoint with selection.json')
        selection = read_json(args.selection)
        if selection.get('complete') is not True or selection.get('best_sha256') != base.file_hash(args.td):
            raise ValueError('Held-out checkpoint was not selected in completed development evaluation')
    weights_hash = base.file_hash(args.weights)
    hashes = {name: base.file_hash(path) for name,path in paths.items()}
    report = {'schema': 'afterstate-td-fresh-eval-v1', 'split': split, 'complete': False, 'seeds': list(seeds),
              'runtime': runtime, 'checkpoints': hashes, 'weights_sha256': weights_hash, 'top_k': TOP_K,
              'selection_sha256': base.file_hash(args.selection) if args.selection else None,
              'screening': 'exhaustive', 'policy_tie_order': list(POLICY_ORDER), 'stages': [], 'clock': 'lockstep elapsed wall seconds per seed'}
    report['options'] = {key: getattr(args, key, False) for key in
                         ('preview_search', 'two_stage', 'headless', 'group_exhaustive')}
    names = list(paths)
    for cap in dict.fromkeys((args.cap, args.max_cap)):
        stage = {}
        report['stages'].append({'cap': cap, 'policies': stage})
        for name in names:
            options = {}
            for key in ('two_stage', 'headless', 'group_exhaustive'):
                if getattr(args, key, False):
                    options[key] = True
            if name == 'ppo' and getattr(args, 'preview_search', False):
                options['preview_search'] = args.preview_search
            stage[name] = evaluate_checkpoint(paths[name], hashes[name], name, seeds, cap, split,
                                              device, budget, runtime, weights_hash, args.cache_dir, **options)
            base.durable_json(args.output, report)
            if not stage[name]['complete']:
                return False
        names = extension_pair(stage)
        if not names:
            break
    first = report['stages'][0]['policies']
    report['primary_comparison'] = None
    if 'td' in first:
        report['initial_comparisons'] = [paired(first['td'], first[name]) for name in first if name != 'td']
        final = report['stages'][-1]['policies']
        leader = next(name for name in rank(final) if name != 'td')
        common = final if 'td' in final and leader in final else first
        comparison = paired(common['td'], common[leader])
        report['primary_comparison'] = comparison
        if split == 'holdout':
            report['promotion_gate_passed'] = comparison['passes']
    report['complete'] = True
    base.durable_json(args.output, report)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    calibration = sub.add_parser('calibrate')
    calibration.add_argument('--positions-report', type=Path, required=True)
    calibration.add_argument('--library', type=Path)
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--split', choices=('development','holdout'), required=True)
    evaluation.add_argument('--td', type=Path)
    evaluation.add_argument('--ppo', type=Path)
    evaluation.add_argument('--previous-td', type=Path, help='Evaluate the pinned predecessor TD checkpoint as a control')
    evaluation.add_argument('--initial-td', type=Path, help='Evaluate the TD policy as initialized before training')
    evaluation.add_argument('--selection', type=Path)
    evaluation.add_argument('--cap', type=int, default=64000)
    evaluation.add_argument('--max-cap', type=int, default=128000)
    preview = evaluation.add_mutually_exclusive_group()
    preview.add_argument('--preview-search', action='store_true', help='Use one visible piece for PPO survival planning')
    preview.add_argument('--rescue-only', dest='preview_search', action='store_const', const='rescue-only',
                         help='Keep the PPO move whenever the visible next piece still fits.')
    preview.add_argument('--preserve-last-cross', dest='preview_search', action='store_const', const='preserve-last-cross',
                         help='Keep an existing CROSS landing when a shortlisted move can preserve one.')
    evaluation.add_argument('--two-stage', action='store_true')
    evaluation.add_argument('--group-exhaustive', action='store_true')
    evaluation.add_argument('--headless', action='store_true')
    final = sub.add_parser('select-final')
    final.add_argument('--run-dir', type=Path, required=True)
    for command in (calibration, evaluation):
        command.add_argument('--source', type=Path, required=True)
    for command in (calibration, evaluation, final):
        command.add_argument('--weights', type=Path, default=ROOT/'models/trained-sigmoid.json')
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--cache-dir', type=Path)
        command.add_argument('--max-seconds', type=float, default=24*3600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error('max-seconds must be finite and positive')
    if args.action == 'evaluate' and not 0 <= args.cap <= args.max_cap:
        parser.error('expected nonnegative cap <= max-cap')
    if args.action == 'evaluate' and args.cap == 0:
        args.max_cap = 0
    if args.action == 'evaluate' and args.preview_search and not args.ppo:
        parser.error('the selected move mode requires --ppo')
    args.cache_dir = args.cache_dir or args.output.parent/'evaluation-cache'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    game.configure_backend('native')
    game.load_checkpoint(args.weights)
    with base.Budget(args.max_seconds / 3600) as budget:
        runtime = runtime_identity(device)
        action = {'calibrate': calibrate, 'evaluate': evaluate, 'select-final': select_final}[args.action]
        complete = action(args, device, budget, runtime)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
