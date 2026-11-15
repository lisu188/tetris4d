"""Bounded CPU activation and intervention audit of an immutable ColumnPolicy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

import tetris_neural as neural
import train_overnight as replay

game = neural.game
MAX_JSON_BYTES = 32 * 2**20
MAX_CHECKPOINT_BYTES = 64 * 2**20
MAX_REPORT_BYTES = 8 * 2**20


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_recording(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f'Recording missing or larger than 32 MiB: {path}')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f'Decompressed recording exceeds 32 MiB: {path}')
    def invalid_constant(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    data = json.loads(payload, parse_constant=invalid_constant)
    if (data.get('schema') != replay.REPLAY_SCHEMA or not data.get('complete')
            or data.get('pieces_sha256') != neural.fingerprint()):
        raise ValueError('A complete recording with the current schema and piece catalog is required')
    if not isinstance(data.get('moves'), list) or not 1 <= len(data['moves']) <= 100000:
        raise ValueError('Recording must contain 1 to 100000 placements')
    reason = data.get('stop_reason')
    if (reason not in ('top_out', 'planner_blocked', 'piece_limit')
            or data.get('terminated') != (reason in ('top_out', 'planner_blocked'))
            or data.get('truncated') != (reason == 'piece_limit')):
        raise ValueError('Inconsistent completed-game termination flags')
    return data


def inventory(config):
    """A channel shares its weights across 4096 sites; it is not one spatial neuron."""
    groups = [('stem', config.width, 'encoder channel', True)]
    for block in range(config.blocks):
        groups += [(f'block{block}.local', config.width, 'depthwise sigmoid channel', True),
                   (f'block{block}.mix', config.width, 'mixing sigmoid channel', True)]
    groups += [('rank1', 128, 'candidate hidden unit', True),
               ('rank2', 64, 'candidate hidden unit', True),
               ('score', 1, 'raw candidate score output', False),
               ('value_hidden', 64, 'critic hidden sigmoid unit before optional centering', True),
               ('value', 1, 'critic scalar output', False),
               ('completion_hidden', 64, 'auxiliary hidden sigmoid unit', True),
               ('completion', 16, 'auxiliary probability output for original height', True)]
    return [{'id': name, 'units': size, 'role': role, 'spatial': name.startswith(('stem', 'block')),
             'sigmoid': sigmoid} for name, size, role, sigmoid in groups]


def input_schema():
    board = ([f'occupied at height {z}' for z in range(16)]
             + [f'buried empty cell at height {z}' for z in range(16)]
             + ['column height / 16', 'column height <= target'])
    context = ([f'current piece is {p}' for p in neural.NAMES]
               + [f'next piece is {p}' for p in neural.NAMES]
               + [f'{p} remains in bag' for p in neural.NAMES]
               + [f'fill fraction at height {z}' for z in range(16)]
               + [f'buried empty fraction at height {z}' for z in range(16)]
               + ['target / 15', 'holes / 65536', 'occupied cells / 65536',
                  'maximum height / 16', 'bag progress fraction'])
    candidate = ['x / 15', 'y / 15', 'z / 15', 'w / 15']
    candidate += [f'{axis} extent / 16' for axis in 'xyzw']
    candidate += ['target gain / 16', 'future holes / 64', 'clears / 16',
                  'height-sum delta / 64', '(height-sum delta - piece size) / 64',
                  'teacher score delta / 64', 'piece size / 16', 'target / 15']
    candidate += [f'piece cell {cell}: {field}' for cell in range(16)
                  for field in ('x / 15', 'y / 15', 'z / 15', 'w / 15', 'present')]
    assert (len(board), len(context), len(candidate)) == (34, 82, 96)
    return {'board_channels': board, 'context': context, 'candidate_features': candidate}


def computation(name, index, config):
    """Exact structural role and coefficient locations in the frozen state_dict."""
    width = config.width
    if name == 'stem':
        formula = f'stem[{index}](w,y,x) = sigmoid(GroupNorm8(Conv1x1(board34))[{index},w,y,x])'
        tensors = [f'stem.0.weight[{index},:,0,0,0]', f'stem.0.bias[{index}]',
                   f'stem.1.weight[{index}]', f'stem.1.bias[{index}]']
        detail = 'Mixes the 34 defined input channels at each column; GroupNorm statistics couple all sites and channels within its group.'
    elif name.startswith('block'):
        block, stage = name.split('.')
        number = int(block[5:])
        prefix = f'blocks.{number}'
        if stage == 'local':
            formula = f'block{number}.local[{index}] = sigmoid(GroupNorm8(depthwiseConv3x3x3(input))[{index}])'
            tensors = [f'{prefix}.depthwise.weight[{index},0,:,:,:]',
                       f'{prefix}.norm1.weight[{index}]', f'{prefix}.norm1.bias[{index}]']
            detail = 'The kernel reads this input channel in a zero-padded 3x3x3 neighborhood; normalization then couples sites and channels in the same group.'
        else:
            formula = f'block{number}.mix[{index}] = sigmoid(GroupNorm8(Conv1x1(local))[{index}]); output[{index}] = input[{index}] + scale*(mix[{index}] - 0.5)'
            tensors = [f'{prefix}.mix.weight[{index},:,0,0,0]',
                       f'{prefix}.norm2.weight[{index}]', f'{prefix}.norm2.bias[{index}]', f'{prefix}.scale']
            detail = f'Mixes all {width} local channels at a column, normalizes and adds the centered branch through the block residual scale.'
    elif name in ('rank1', 'rank2'):
        inputs = f'[local{width}, pooled{width}, context82, candidate96]' if name == 'rank1' else 'rank1[0:128]'
        formula = f'{name}[{index}] = sigmoid({name}.weight[{index}] dot {inputs} + {name}.bias[{index}])'
        tensors = [f'{name}.weight[{index},:]', f'{name}.bias[{index}]']
        detail = (f'The same candidate-specific unit is evaluated for every legal shortlisted move; local{width} averages touched columns and pooled{width} averages all 4096 columns.'
                  if name == 'rank1' else 'Combines the 128 first-layer candidate activations; it has no predefined detector meaning.')
    elif name == 'score':
        formula = 'raw_score = rank_out.weight[0] dot rank2[0:64] + rank_out.bias[0]'
        tensors = ['rank_out.weight[0,:]', 'rank_out.bias[0]']
        detail = 'One shared scalar scorer is evaluated for every candidate; policy_mode transforms this score and may add the teacher prior before choice.'
    else:
        value = name.startswith('value')
        hidden = name.endswith('_hidden')
        prefix = 'value' if value else 'completion'
        if hidden:
            formula = f'{name}[{index}] = sigmoid({prefix}.0.weight[{index}] dot [pooled{width},context82] + {prefix}.0.bias[{index}])'
            tensors = [f'{prefix}.0.weight[{index},:]', f'{prefix}.0.bias[{index}]']
            detail = ('One state-level critic hidden unit; centered_critic subtracts 0.5 after this activation.' if value
                      else 'One state-level hidden unit of the inherited auxiliary predictor; its output does not enter candidate scores.')
        else:
            inputs = 'value_hidden[0:64] - 0.5' if config.centered_critic else 'value_hidden[0:64]'
            formula = (f'value = value.2.weight[0] dot ({inputs}) + value.2.bias[0]' if value else
                       f'completion[{index}] = sigmoid(completion.2.weight[{index}] dot completion_hidden[0:64] + completion.2.bias[{index}])')
            tensors = [f'{prefix}.2.weight[{index},:]', f'{prefix}.2.bias[{index}]']
            detail = ('State value in the checkpoint metadata units; it does not directly enter move ranking.' if value else
                      f'Auxiliary probability that the layer initially at height {index} clears within the imitation dataset horizon; not recalibrated by survival training.')
    return {'formula': formula, 'checkpoint_tensor_rows': tensors, 'structural_role': detail}


def neutralize(tensor, intervention, name, value=.5, spatial=False):
    if intervention is None or intervention[0] != name:
        return tensor
    result = tensor.clone()
    if spatial:
        result[:, intervention[1]] = value
    else:
        result[..., intervention[1]] = value
    return result


def rank_input(spatial, pooled, batch):
    flat = spatial.flatten(2).transpose(1, 2)
    gathered = flat[torch.arange(len(flat))[:, None, None], batch['indices'].long()]
    mask = batch['column_mask'][..., None].to(gathered.dtype)
    local = (gathered * mask).sum(2) / mask.sum(2).clamp_min(1)
    count = batch['features'].shape[1]
    return torch.cat((local, pooled[:, None].expand(-1, count, -1),
                      batch['context'][:, None].expand(-1, count, -1), batch['features']), -1)


def forward_parts(model, batch, intervention=None):
    """Explicit traced forward, checked against the runtime forward for every state."""
    traces = {}
    x = model.stem(neural.encode_columns(batch['boards'], batch['targets']))
    traces['stem'] = x
    x = neutralize(x, intervention, 'stem', spatial=True)
    for i, block in enumerate(model.blocks):
        local = block.activation(block.norm1(block.depthwise(x)))
        traces[f'block{i}.local'] = local
        local = neutralize(local, intervention, f'block{i}.local', spatial=True)
        mixed = block.activation(block.norm2(block.mix(local)))
        traces[f'block{i}.mix'] = mixed
        mixed = neutralize(mixed, intervention, f'block{i}.mix', spatial=True)
        x = x + block.scale * (mixed - .5)
    pooled = x.mean((2, 3, 4))
    joined = torch.cat((pooled, batch['context']), 1)
    h1 = torch.sigmoid(model.rank1(rank_input(x, pooled, batch)))
    h1 = neutralize(h1, intervention, 'rank1')
    h2 = torch.sigmoid(model.rank2(h1))
    h2 = neutralize(h2, intervention, 'rank2')
    score = neutralize(model.rank_out(h2), intervention, 'score', 0.).squeeze(-1)
    hidden = torch.sigmoid(model.value[0](joined))
    hidden = neutralize(hidden, intervention, 'value_hidden')
    critic = model.value[2](hidden - .5 if model.config.centered_critic else hidden)
    critic = neutralize(critic, intervention, 'value', 0.).squeeze(-1)
    auxiliary_hidden = torch.sigmoid(model.completion[0](joined))
    auxiliary_hidden = neutralize(auxiliary_hidden, intervention, 'completion_hidden')
    completion = torch.sigmoid(model.completion[2](auxiliary_hidden))
    completion = neutralize(completion, intervention, 'completion')
    traces.update(rank1=h1, rank2=h2, score=score[..., None], value_hidden=hidden,
                  value=critic[:, None], completion_hidden=auxiliary_hidden, completion=completion)
    return {'traces': traces, 'score': score, 'value': critic, 'completion': completion}


def policy_logits(model, batch, score):
    if model.config.policy_mode == 'teacher-residual':
        score = batch['teacher_prior'].float() + model.config.residual_scale * (2 * torch.sigmoid(score.float()) - 1)
    elif model.config.policy_mode == 'teacher-prior':
        score = model.config.teacher_prior_scale * batch['teacher_prior'].float() + model.config.residual_scale * score.float()
    return score.masked_fill(~batch['valid'].bool(), -torch.inf)


def ablate(model, batch, baseline, name, index):
    """Reuse unaffected branches; never retain a stack of intervention tensors."""
    if name.startswith(('stem', 'block')):
        return forward_parts(model, batch, (name, index))
    traces = baseline['traces']
    result = {k: baseline[k] for k in ('score', 'value', 'completion')}
    if name in ('rank1', 'rank2', 'score'):
        h1 = neutralize(traces['rank1'], (name, index), 'rank1')
        h2 = torch.sigmoid(model.rank2(h1)) if name == 'rank1' else traces['rank2']
        h2 = neutralize(h2, (name, index), 'rank2')
        result['score'] = neutralize(model.rank_out(h2), (name, index), 'score', 0.).squeeze(-1)
    elif name.startswith('value'):
        hidden = neutralize(traces['value_hidden'], (name, index), 'value_hidden')
        value = model.value[2](hidden - .5 if model.config.centered_critic else hidden)
        result['value'] = neutralize(value, (name, index), 'value', 0.).squeeze(-1)
    else:
        hidden = neutralize(traces['completion_hidden'], (name, index), 'completion_hidden')
        value = torch.sigmoid(model.completion[2](hidden))
        result['completion'] = neutralize(value, (name, index), 'completion')
    return result


class UnitStats:
    def __init__(self, group, index, config):
        self.row = {'id': f'{group["id"]}[{index}]', 'group': group['id'], 'index': index,
                    'role': group['role'], 'spatial': group['spatial'],
                    'computation': computation(group['id'], index, config)}
        self.sigmoid = group['sigmoid']
        self.count = self.saturated = 0
        self.total = self.squares = 0.
        self.minimum, self.maximum = math.inf, -math.inf
        self.contexts, self.effects = [], []

    def observe(self, values, sample_id):
        values = values.double().flatten()
        if not bool(torch.isfinite(values).all()):
            raise ValueError('Non-finite activation')
        mean, maximum = float(values.mean()), float(values.max())
        self.count += values.numel()
        self.total += float(values.sum())
        self.squares += float(values.square().sum())
        self.minimum, self.maximum = min(self.minimum, float(values.min())), max(self.maximum, maximum)
        if self.sigmoid:
            self.saturated += int(((values <= .001) | (values >= .999)).sum())
        self.contexts.append({'sample': sample_id, 'mean_activation': mean,
                              'maximum_activation': maximum, 'maximum_flat_index': int(values.argmax())})
        self.contexts.sort(key=lambda row: (-row['mean_activation'], row['sample']))
        del self.contexts[3:]

    def finish(self):
        mean = self.total / self.count
        activation = {'observations': self.count, 'mean': mean,
                      'std': math.sqrt(max(0., self.squares / self.count - mean * mean)),
                      'minimum': self.minimum, 'maximum': self.maximum}
        if self.sigmoid:
            activation['saturated_fraction'] = self.saturated / self.count
        effects = {key: sum(row[key] for row in self.effects) / len(self.effects)
                   for key in self.effects[0]}
        effects['maximum_absolute_score_change'] = max(row['maximum_absolute_score_change'] for row in self.effects)
        effects['top1_changes'] = sum(row['top1_changed_fraction'] for row in self.effects)
        delta = effects['signed_value_change']
        value_effect = (f"increased critic value by {delta:.5g}" if delta > 0 else
                        f"decreased critic value by {-delta:.5g}" if delta < 0 else 'left critic value unchanged')
        scope = 'all 4096 sites of this shared channel' if self.row['spatial'] else 'this unit for every applicable candidate/state'
        self.row.update(activation=activation, strongest_contexts=self.contexts, intervention=effects,
                        explanation=f"{self.row['computation']['formula']}. {self.row['computation']['structural_role']} "
                        f"The intervention replaced {scope} with the fixed baseline. "
                        f"The highest-scoring candidate changed in {int(effects['top1_changes'])}/{len(self.effects)} sampled states; "
                        f"mean absolute candidate-rank change was {effects['mean_absolute_rank_change']:.5g}. "
                        f"The intervention {value_effect} on average (checkpoint value units), "
                        f"and changed completion probabilities by {effects['mean_absolute_completion_probability_change']:.5g} on average in absolute value. "
                        'This describes the measured intervention, without assigning a semantic detector label.')
        return self.row


def intervention_effect(model, batch, before, after):
    valid = batch['valid'][0].bool()
    original = policy_logits(model, batch, before['score'])[0, valid].double()
    changed = policy_logits(model, batch, after['score'])[0, valid].double()
    if not bool(torch.isfinite(changed).all()):
        raise ValueError('Non-finite intervention score')
    score_delta = (changed - original).abs()
    winner = int(original.argmax())
    probability, changed_probability = original.softmax(0), changed.softmax(0)
    # Stable tie handling matches argmax's first-index convention.
    rank = torch.argsort(torch.argsort(-original, stable=True), stable=True)
    changed_rank = torch.argsort(torch.argsort(-changed, stable=True), stable=True)
    auxiliary_delta = (after['completion'] - before['completion']).abs()
    return {'mean_absolute_score_change': float(score_delta.mean()),
            'maximum_absolute_score_change': float(score_delta.max()),
            'mean_absolute_raw_score_change': float((after['score'][0, valid] - before['score'][0, valid]).abs().mean()),
            'top1_changed_fraction': float(int(changed.argmax()) != winner),
            'mean_absolute_rank_change': float((changed_rank - rank).abs().double().mean()),
            'original_winner_rank_after': float(changed_rank[winner] + 1),
            'original_winner_probability_change': float(changed_probability[winner] - probability[winner]),
            'probability_total_variation': float((changed_probability - probability).abs().sum() / 2),
            'signed_value_change': float((after['value'] - before['value']).mean()),
            'absolute_value_change': float((after['value'] - before['value']).abs().max()),
            'mean_absolute_completion_probability_change': float(auxiliary_delta.mean())}


def check_candidates(state, prepared):
    arrays = prepared.arrays
    count = len(prepared.options)
    if (not 1 <= count <= 2048 or arrays['features'].shape != (count, neural.FEATURES)
            or arrays['indices'].shape != (count, neural.COLUMNS)
            or arrays['column_mask'].shape != (count, neural.COLUMNS)
            or arrays['valid'].shape != (count,) or not bool(arrays['valid'].all())
            or not 0 <= prepared.teacher < count or not 0 <= int(arrays['targets']) < 16
            or not np.array_equal(arrays['boards'], state.board)
            or np.any(arrays['indices'] < 0) or np.any(arrays['indices'] >= game.AREA)):
        raise ValueError('Invalid prepared candidate schema')
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Non-finite prepared features')
    for candidate in prepared.options:
        move = game.make_move(state.board, state.current, candidate.rotation, candidate.x, candidate.y, candidate.w)
        if move is None or move.candidate.z != candidate.z:
            raise ValueError('Prepared candidate is not a legal rigid drop')


def collect_samples(paths, checkpoint, model, limit):
    """Verify every move, keeping only bounded, globally spaced pre-decision states."""
    files = []
    engine_hash = sha256(Path(game.__file__))
    for path in paths:
        data = read_recording(path)
        if data['teacher_parameters'] != checkpoint['teacher_parameters'] or data['engine_sha256'] != engine_hash:
            raise ValueError('Replay teacher parameters or engine source differ from the checkpoint/current source')
        files.append({'path': str(path), 'sha256': sha256(path), 'placements': len(data['moves']),
                      'seed': data['seed'], 'stop_reason': data['stop_reason']})
    total = sum(row['placements'] for row in files)
    wanted = set(np.linspace(0, total - 1, min(limit, total), dtype=np.int64).tolist())
    samples, offset = [], 0
    for source_index, source in enumerate(files):
        path = Path(source['path'])
        data = read_recording(path)
        episode = replay.Episode(data['number'], game.Game(data['seed'], data['piece_set']))
        for index, row in enumerate(data['moves']):
            state = episode.state
            if row['piece_number'] != state.pieces + 1 or row['piece'] != state.current:
                raise ValueError('Replay piece order differs from deterministic bag')
            if 'retarget_from' in row:
                status = game.target_layer_status(state.board, row['target_before'])
                if row['retarget_from'] != episode.target or status.buried or status.filled == game.AREA:
                    raise ValueError('Invalid replay target recovery')
                episode.pending_retarget, episode.target = episode.target, row['target_before']
            if row['target_before'] != episode.target:
                raise ValueError('Replay target history mismatch')
            if offset + index in wanted:
                prepared = neural.prepare(state, episode.target, model.config)
                if prepared is None:
                    raise ValueError(f'Audited model cannot prepare sampled position {path}:{index}')
                check_candidates(state, prepared)
                samples.append({'arrays': prepared.arrays, 'description': {
                    'id': len(samples), 'recording': source_index, 'placements_before': state.pieces,
                    'seed': state.seed, 'piece': state.current, 'next_piece': state.next_piece,
                    'recorded_target': episode.target, 'effective_target': int(prepared.arrays['targets']),
                    'holes': game.evaluate_board(state.board).holes,
                    'maximum_height': max(game.evaluate_board(state.board).heights),
                    'board_sha256': replay.board_hash(state.board), 'candidates': len(prepared.options),
                    'teacher_candidate': prepared.teacher}})
            move = game.make_move(state.board, state.current, row['rotation'], row['x'], row['y'], row['w'])
            if move is None or move.candidate.z != row['z']:
                raise ValueError('Invalid replay rigid drop')
            replay.apply_move(episode, move.candidate, row.get('update', 0), row.get('behavior', 'unknown'))
            actual = episode.moves.pop()
            for name in ('clears', 'cleared_mask', 'target_after', 'reward_pieces', 'board_sha256'):
                if actual[name] != row[name]:
                    raise ValueError(f'Replay mismatch at placement {state.pieces}: {name}')
        state = episode.state
        if data.get('pending_retarget') is not None:
            status = game.target_layer_status(state.board, data['target'])
            if data['pending_retarget'] != episode.target or status.buried or status.filled == game.AREA:
                raise ValueError('Invalid pending target recovery')
            episode.target = data['target']
        if (episode.target != data['target'] or replay.board_hash(state.board) != data['board_sha256']
                or state.summary() != data['summary']):
            raise ValueError('Replay final board/counters mismatch')
        legal = game.has_legal_move(state.board, state.current)
        if (data['stop_reason'] == 'top_out' and legal) or (data['stop_reason'] == 'planner_blocked' and not legal):
            raise ValueError('Replay physical termination label is incorrect')
        if sha256(path) != source['sha256']:
            raise ValueError('Recording changed during verification')
        source['verified'] = True
        offset += len(data['moves'])
    return files, samples


def markdown(report):
    lines = ['# Neuron audit', '', f"Checkpoint SHA256: `{report['checkpoint']['sha256']}`", '',
             f"{len(report['units'])} learned channels/units; {len(report['samples'])} sampled states. CPU, {report['environment']['backend']} backend.", '',
             'A convolution channel shares a learned operation across 4,096 spatial sites. '
             'Its intervention replaces the entire channel. Hidden units have no assigned semantic name.', '',
             'Each sigmoid is clamped to 0.5; raw score and value outputs are clamped to zero. '
             'Completion output interventions set one probability to 0.5. Effects concern the listed states and candidates, '
             'and do not establish survival effects. These are structural baselines, not claims about typical neutral activity. '
             'Contexts are ordered by mean activation.', '',
             f"Value: {report['semantics']['value']}", '', f"Completion: {report['semantics']['completion']}", '',
             'Scores include the configured teacher prior. Rank 1 is best; ties follow candidate index. '
             'Constant score shifts can have zero effect on ranking and choice. Δ value uses the checkpoint value units.', '',
             '| Unit | Activation mean ± std | Saturated | Mean absolute score change | Top choice changes | Mean rank change | Probability TV | Absolute value change | Completion change | Strongest sampled contexts |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in report['units']:
        a, e = row['activation'], row['intervention']
        saturated = f"{a['saturated_fraction']:.1%}" if 'saturated_fraction' in a else '—'
        contexts = ', '.join(str(context['sample']) for context in row['strongest_contexts'])
        lines.append(f"| `{row['id']}` | {a['mean']:.5g} ± {a['std']:.3g} | {saturated} | "
                     f"{e['mean_absolute_score_change']:.4g} | {int(e['top1_changes'])}/{len(report['samples'])} | "
                     f"{e['mean_absolute_rank_change']:.4g} | {e['probability_total_variation']:.4g} | "
                     f"{e['absolute_value_change']:.4g} | {e['mean_absolute_completion_probability_change']:.4g} | {contexts} |")
    lines += ['', '## Sample contexts', '', '| ID | Recording | Seed | Placements before | Piece | Target | Maximum height | Holes | Candidates |',
              '|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for sample in report['samples']:
        lines.append('| ' + ' | '.join(str(sample[key]) for key in ('id', 'recording', 'seed', 'placements_before',
                     'piece', 'effective_target', 'maximum_height', 'holes', 'candidates')) + ' |')
    lines += ['', '## Per-unit explanations', '']
    for row in report['units']:
        lines.append(f"- **`{row['id']}`**: {row['explanation']}")
    lines += ['', 'Full JSON records every unit, input feature definitions, source hashes, sampling policy, '
              'activation extrema and intervention metrics. No semantic detector label is inferred from activation alone.', '']
    return '\n'.join(lines)


def analyze(args):
    started = time.monotonic()
    checkpoint_path = args.checkpoint.resolve(strict=True)
    paths = [path.resolve(strict=True) for path in args.games]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate recordings are not allowed')
    if checkpoint_path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError('Checkpoint exceeds 64 MiB; export weights only before auditing')
    if sha256(checkpoint_path) != args.checkpoint_sha256:
        raise ValueError('Checkpoint SHA256 does not match the supplied immutable identity')
    if args.output_dir.exists():
        raise ValueError('Choose a new output directory; existing reports are never overwritten')
    torch.set_num_threads(args.threads)
    game.configure_backend(args.backend)
    source_files = [Path(__file__).resolve(), Path(neural.__file__), Path(game.__file__), Path(replay.__file__)]
    native_info = game.native_backend_info()
    if game.BACKEND == 'native':
        library = Path(native_info['library'])
        source_files += [library, library.with_suffix(library.suffix + '.json'), library.with_name('tetris_native.c')]
    sources = [{'path': str(path), 'sha256': sha256(path)} for path in source_files]
    model, checkpoint = neural.load_checkpoint(checkpoint_path, torch.device('cpu'))
    model.eval()
    game.configure_network(checkpoint['teacher_parameters'])
    groups = inventory(model.config)
    stats = {group['id']: [UnitStats(group, index, model.config) for index in range(group['units'])] for group in groups}
    files, samples = collect_samples(paths, checkpoint, model, args.limit)
    with torch.inference_mode():
        for sample in samples:
            batch = neural.collate([sample['arrays']], torch.device('cpu'))
            baseline = forward_parts(model, batch)
            runtime = model(batch)
            # Refuse a misleading audit if a future runtime changes the traced calculation.
            for actual, expected in ((policy_logits(model, batch, baseline['score']), runtime['logits']),
                                     (baseline['value'], runtime['value']),
                                     (baseline['completion'], torch.sigmoid(runtime['completion']))):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            sample['description']['greedy_candidate'] = int(runtime['logits'].argmax())
            sample['description']['value'] = float(runtime['value'][0])
            for group in groups:
                name = group['id']
                values = baseline['traces'][name]
                for index, unit in enumerate(stats[name]):
                    activation = values[0, index] if group['spatial'] else values[..., index]
                    unit.observe(activation, sample['description']['id'])
                    after = ablate(model, batch, baseline, name, index)
                    unit.effects.append(intervention_effect(model, batch, baseline, after))
            print(json.dumps({'audited_states': sample['description']['id'] + 1, 'total_states': len(samples)}), flush=True)
    if sha256(checkpoint_path) != args.checkpoint_sha256 or any(sha256(row['path']) != row['sha256'] for row in files):
        raise ValueError('An immutable input changed during the audit')
    if any(sha256(row['path']) != row['sha256'] for row in sources):
        raise ValueError('Audit/runtime source changed during the audit')
    value_semantics = checkpoint.get('extra', {}).get('value_semantics')
    if value_semantics is None:
        value_semantics = ('expected_remaining_placements_times_0.001' if checkpoint.get('extra', {}).get('kind')
                           == 'tetris4d-overnight-survival-v1' else 'discounted_hyperlayers')
    survival_value = value_semantics == 'expected_remaining_placements_times_0.001'
    units = [unit.finish() for group in groups for unit in stats[group['id']]]
    if survival_value:
        for unit in units:
            unit['intervention']['absolute_value_change_in_placements'] = 1000 * unit['intervention']['absolute_value_change']
    report = {'schema': 'tetris4d-neuron-audit-v1', 'checkpoint': {'path': str(checkpoint_path),
              'sha256': args.checkpoint_sha256, 'config': asdict(model.config),
              'parameters': sum(parameter.numel() for parameter in model.parameters())},
              'environment': {'device': 'cpu', 'backend': game.BACKEND, 'native': native_info, 'threads': args.threads,
                              'torch': str(torch.__version__), 'numpy': str(np.__version__)},
              'sources': sources,
              'recordings': files, 'groups': groups, 'input_schema': input_schema(),
              'sampling': {'method': 'equally spaced pre-decision indices over recordings in CLI order',
                           'requested_states': args.limit, 'maximum_states': 64,
                           'activation_weighting': 'every spatial site/candidate; scalar heads once per state',
                           'strongest_contexts': 'top three states by mean unit activation; ties by sample ID'},
              'intervention': {'sigmoid_baseline': .5, 'raw_score_and_value_baseline': 0.,
                               'completion_probability_baseline': .5, 'scope': 'all sites/candidates for one unit in one sampled state',
                               'baseline_meaning': 'fixed structural baseline, not measured typical neutral activity',
                               'effect_averaging': 'states weighted equally; scores/ranks averaged over valid candidates',
                               'convolution_site_order': '[w, y, x], x varies fastest; 4096 sites per channel',
                               'limitation': 'artificial midpoint intervention; no proof of semantic meaning or causal effect on game survival'},
              'semantics': {'value_code': value_semantics,
                            'value': ('Expected remaining successful placements / 1000; an estimate, not a guarantee.' if survival_value
                                      else 'Discounted future hyperlayer clears; this checkpoint does not use survival value units.'),
                            'completion': 'Inherited imitation auxiliary: output z targets whether the layer initially at height z clears within the dataset auxiliary horizon (collector default 2048 placements), tracking original layers through compaction. Not next-move completion or fill percentage. Survival training does not recalibrate this head; probabilities can be stale as the encoder changes.'},
              'samples': [sample['description'] for sample in samples], 'units': units,
              'residual_scales': [float(block.scale.detach()) for block in model.blocks],
              'notes': ['Residual scales and weights are parameters, not extra neurons.',
                        'The raw candidate score is transformed by policy_mode and the teacher prior before selection.',
                        'Critic and completion heads do not enter move ranking. Their hidden/output interventions have no direct score effect.',
                        'A channel inventory is not a census of spatial activations or proof of a human-readable detector.'],
              'baseline_forward_parity_passed': True, 'parity_tolerance': {'rtol': 1e-6, 'atol': 1e-7},
              'elapsed_seconds': time.monotonic() - started}
    json_text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    markdown_text = markdown(report)
    if len(json_text.encode()) + len(markdown_text.encode()) > MAX_REPORT_BYTES:
        raise ValueError('Report exceeds the 8 MiB combined output bound')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'neurons.json').write_text(json_text, encoding='utf-8')
    (args.output_dir / 'neurons.md').write_text(markdown_text, encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--games', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=16)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--backend', choices=('native', 'numpy', 'python'), default='native')
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 64 or not 1 <= args.threads <= 4 or not 1 <= len(args.games) <= 8:
        parser.error('limit must be 1..64, threads 1..4, and recordings 1..8')
    if len(args.checkpoint_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.checkpoint_sha256):
        parser.error('checkpoint-sha256 must be a lowercase SHA256 digest')
    report = analyze(args)
    print(json.dumps({'units': len(report['units']), 'states': len(report['samples']),
                      'report': str(args.output_dir / 'neurons.md'), 'seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
