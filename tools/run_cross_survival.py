"""Run matched V4 survival interventions and seal a single challenger before holdout."""

import argparse
from collections import Counter
from dataclasses import replace
from itertools import product
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tools import calibrate_cross_critics as calibration
from tools import cross_run_budget as meter
from tools import run_spatial_curriculum as spatial

ev, v4, base = spatial.ev, spatial.v4, spatial.base
SCREEN = tuple(48200001 + 2 * i for i in range(8))
QUALIFY = tuple(48300001 + 2 * i for i in range(24))
HOLDOUT = tuple(48400001 + 2 * i for i in range(144))
TRAIN = (611000, 612000)
EVALUATION = calibration.SEEDS + SCREEN + QUALIFY + HOLDOUT
RECIPES = ('plain', 'cold', 'anchored', 'residual')
WEIGHTS = (0.25, 1.0, 4.0)
UPDATES = 80
BATCH = 32


def player(path, sha, mode='greedy'):
    return dict(path=str(path), sha256=sha, mode=mode)


def read_protocol(out, incumbent, donor, device, gate, calibration_runtime, calibration_driver):
    head = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    files = (
        'tools/run_cross_survival.py',
        'tools/cross_run_budget.py',
        'tools/run_spatial_curriculum.py',
        'tools/calibrate_cross_critics.py',
        'tools/verify_cross_calibration.py',
        'tools/benchmark_spatial_ppo.py',
        'native/tetris_native.c',
        'train_survival.py',
        'train_survival_v2.py',
        'train_survival_v3.py',
        'train_overnight.py',
        'tetris_cross_survival.py',
        'tetris_gpu_cross.py',
    )
    from tools.verify_cross_calibration import verify_gate
    from tools.benchmark_spatial_ppo import verified_gate

    critic_gate = verify_gate(
        out, calibration_runtime, calibration_driver, device, incumbent_path=incumbent, donor_path=donor
    )
    for path, expected in [(incumbent, calibration.INCUMBENT), (donor, calibration.DONOR)]:
        if base.file_hash(path) != expected:
            raise ValueError(f'Checkpoint identity changed: {path}')
    m = ev.load_model(incumbent, torch.device('cpu'))
    critic = ev.load_model(donor, torch.device('cpu')) if critic_gate['selected'] == 'donor' else m
    recorded = {row['seed'] for row in base.library.iter_index(v4.library_root()) if 'seed' in row}
    if not recorded or recorded.intersection(EVALUATION) or len(set(EVALUATION)) != len(EVALUATION):
        raise ValueError('Experiment needs a real archive and disjoint fresh evaluation seed lists')
    if gate.name != 'gate.json':
        raise ValueError('Expected perf/gate.json together with its benchmark receipts')
    performance = verified_gate(gate.parent, ROOT, device, calibration.INCUMBENT)
    enabled = performance['enabled']
    if set(enabled) != {'two_stage', 'headless'} or any(type(v) is not bool for v in enabled.values()):
        raise ValueError('Expected explicit boolean benchmark decisions')
    return dict(
        schema='cross-survival-v1',
        revision=head,
        runtime=ev.runtime_identity(device),
        source={f: base.file_hash(ROOT / f) for f in files},
        teacher=base.file_hash(ROOT / 'models/trained-sigmoid.json'),
        incumbent=calibration.INCUMBENT,
        donor=calibration.DONOR,
        base_actor=v4.v2.actor_digest(m),
        starting_model=v4.v3.model_digest(m),
        critic=v4.v3.model_digest(critic.value),
        critic_selected=critic_gate['selected'],
        critic_gate=base.file_hash(out / 'critic-gate.json'),
        performance_gate=base.file_hash(gate),
        enabled=enabled,
        greedy_enabled=performance['greedy_enabled'],
        screening=list(SCREEN),
        qualification=list(QUALIFY),
        holdout=list(HOLDOUT),
        training_seeds=list(TRAIN),
        recipes=list(RECIPES),
        direct_weights=[0.0, *WEIGHTS],
        phase_hours=meter.PHASE_HOURS,
        timeout_policy='omit the entire unmatched direct or PPO family and continue with complete families; holdout requires completed qualification',
        updates=UPDATES,
        placements_per_update=spatial.PLACEMENTS_PER_UPDATE,
        cap=0,
        batch=BATCH,
        residual_logit_bound=0.05,
        temperatures={'plain': 1.0, 'others': 0.25},
        anchored_reference_kl=0.05,
        training_precision='fp32',
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_compute_capability=list(torch.cuda.get_device_capability(device)),
        torch_threads=torch.get_num_threads(),
        primary_seed=TRAIN[0],
        promotion={'relative_gain': 0.05, 'lower_bound': 0.0, 'bootstrap_pairs': 10000},
    )


def direct_players(out, incumbent):
    players = {'incumbent': player(incumbent, calibration.INCUMBENT)}
    plain, payload = ev.neural.load_checkpoint(incumbent, torch.device('cpu'))
    original = v4.v3.model_digest(plain)
    metadata = {
        k: payload['extra'][k]
        for k in ('kind', 'value_semantics', 'completion_semantics')
        if k in payload['extra']
    }
    for weight in WEIGHTS:
        name = f'weight-{weight:g}'
        path = out / 'players' / (name + '.pt')
        config = replace(plain.config, cross_weight=weight)
        if not path.exists():
            plain.config = config
            ev.neural.save_checkpoint(
                path,
                plain,
                progress={'updates': payload['progress']['updates']},
                extra={**metadata, 'parent_checkpoint_sha256': calibration.INCUMBENT, 'cross_weight': weight},
            )
        model, saved = ev.neural.load_checkpoint(path, torch.device('cpu'))
        if (
            model.config != config
            or v4.v3.model_digest(model) != original
            or saved['extra']['parent_checkpoint_sha256'] != calibration.INCUMBENT
        ):
            raise ValueError(f'Direct-weight player differs from its frozen parent: {path}')
        players[name] = player(path, base.file_hash(path))
    return players


def train_arm(out, incumbent, donor, plan, seed, recipe, stop):
    label = f'{seed}-{recipe}'
    root = out / 'training' / label
    endpoint = root / 'endpoint.json'
    if endpoint.exists():
        data = ev.read_json(endpoint)
        if (
            data['protocol'] != ev.digest(plan)
            or base.file_hash(root / 'last.pt') != data['player']['sha256']
        ):
            raise ValueError(f'Invalid endpoint {label}')
        return data['player']
    flags = []
    if recipe != 'plain':
        flags += ['--behavior-temperature', '.25']
    if recipe == 'anchored':
        flags += ['--reference-checkpoint', str(incumbent), '--reference-kl-weight', '.05']
    if recipe == 'residual':
        flags.append('--cross-residual')
    if plan['critic_selected'] == 'donor':
        flags += ['--critic-checkpoint', str(donor)]
    if plan['enabled']['two_stage']:
        flags.append('--two-stage-exhaustive')
    if plan['enabled']['headless']:
        flags.append('--headless-commit')
    options = spatial.trainer_args(
        root.parent,
        incumbent,
        label,
        'combined',
        seed=seed,
        danger_per_game=4,
        selection_seeds=EVALUATION,
        extra=flags,
    )
    with base.run_lock(root), meter.reservation(out, 'training', label, stop):
        if stop.expired():
            return None
        trainer = spatial.MatchedTrainer(options, stop)
        trainer.stop_at_update = UPDATES
        proof = ev.read_json(root / 'initialization.json')
        if (
            not proof['fresh_optimizer']
            or not proof['fresh_ppo_progress']
            or proof['imported_model_sha256'] != plan['incumbent']
            or proof.get('base_actor_sha256', proof['actor_sha256']) != plan['base_actor']
            or (
                proof['critic_sha256'] != plan['critic']
                if 'critic_sha256' in proof
                else proof['model_sha256'] != plan['starting_model'] or plan['critic_selected'] != 'incumbent'
            )
        ):
            raise ValueError(f'Unexpected starting model for {label}')
        pool = [
            {'seed': p['state'].seed, 'game_id': p['game_id'], 'offset': p['offset']}
            for p in trainer.danger_pool
        ]
        counts = Counter(p['seed'] for p in pool)
        if len(counts) != 8 or set(counts.values()) != {4} or set(counts).intersection(EVALUATION):
            raise ValueError(f'Unmatched danger curriculum {label}')
        spatial.freeze(out / 'training' / f'curriculum-{seed}.json', pool)
        expected = spatial.PLACEMENTS_PER_UPDATE
        if trainer.progress['placements'] != trainer.progress['runner_updates'] * expected:
            raise ValueError(f'Partial update in {label}')
        summary = trainer.run()
        if stop.expired() and summary['runner_updates'] < UPDATES:
            return None
        if (
            summary['runner_updates'] != UPDATES
            or summary['placements'] != UPDATES * expected
            or not summary['optimizer_steps']
        ):
            raise ValueError(f'Unmatched endpoint for {label}')
        # Every fresh game uses seed 2*(training seed + episode number). Archive
        # starts consume some numbers too; checking the whole interval is conservative.
        upper = seed + trainer.next_episode
        if seed == TRAIN[0] and upper > TRAIN[1]:
            raise ValueError('Fresh game seed intervals collided between training replicates')
        if any(
            row['seed'] % 2 == 0 and seed <= row['seed'] // 2 < upper
            for row in base.library.iter_index(v4.library_root())
            if 'seed' in row
        ):
            raise ValueError('Fresh training seed interval overlaps an archived game')
        sha = base.file_hash(root / 'last.pt')
        manifest = ev.read_json(root / 'manifest.json')
        if manifest['runner_updates'] != UPDATES or manifest['checkpoint_sha256'] != sha:
            raise ValueError(f'Invalid recovery manifest for {label}')
        candidate = player(root / 'last.pt', sha)
        spatial.freeze(
            endpoint,
            dict(
                protocol=ev.digest(plan),
                player=candidate,
                seed=seed,
                recipe=recipe,
                updates=UPDATES,
                placements=UPDATES * expected,
                optimizer_steps=summary['optimizer_steps'],
                allocated_episodes=trainer.next_episode,
            ),
        )
        return candidate


def evaluate(out, plan, label, candidate, seeds, split, phase, device, stop):
    path = out / split / (label + '.json')
    with meter.reservation(out, phase, f'{split}/{label}', stop):
        result = ev.evaluate_checkpoint(
            Path(candidate['path']),
            candidate['sha256'],
            'ppo',
            seeds,
            0,
            split,
            device,
            stop,
            plan['runtime'],
            plan['teacher'],
            out / 'evaluation-cache',
            ppo_batch_size=BATCH,
            group_exhaustive=True,
            preview_search=candidate['mode'] == 'lookahead',
            two_stage=plan['greedy_enabled']['two_stage'],
            headless=plan['greedy_enabled']['headless'],
        )
    if result['placements']:
        result.update(
            longest=max(result['placements']),
            bottom_decile=float(np.quantile(result['placements'], 0.10)),
            planner_blocked=sum(reason == 'target_or_no_move' for reason in result['stop_reasons']),
        )
    result.update(player=candidate, protocol=ev.digest(plan))
    base.durable_json(path, result)
    if result['complete'] and (result['seeds'] != list(seeds) or result['censored'] or result['cap'] != 0):
        raise ValueError(f'Invalid uncapped comparison {split}/{label}')
    if result.get('planner_blocked'):
        raise ValueError(f"{split}/{label} ended {result['planner_blocked']} games before physical collision")
    return result


def best(labels, scores):
    # Stable input order prefers the incumbent, then greedy, on exact ties.
    return max(labels, key=lambda name: (scores[name]['mean'], scores[name]['median']))


def run(out, incumbent, donor, plan, device, stop):
    omitted = {}

    def skip_timed_family(name, reason):
        if stop.reason != 'time_budget' or time.monotonic() >= stop.deadline:
            return False
        omitted[name] = reason
        base.durable_json(out / 'budget-skips.json', omitted)
        stop.reason = None
        return True

    receipt = out / 'direct-players.json'
    if receipt.exists():
        direct_record = ev.read_json(receipt)
        players = direct_record['players']
        if direct_record['protocol'] != ev.digest(plan) or any(
            base.file_hash(Path(p['path'])) != p['sha256'] for p in players.values()
        ):
            raise ValueError('Derived players differ from frozen screening inputs')
    else:
        with meter.reservation(out, 'calibration', 'derive direct players', stop):
            players = direct_players(out, incumbent)
            spatial.freeze(receipt, dict(protocol=ev.digest(plan), players=players))
    screen = {}

    def score(label, seeds, split, phase):
        return evaluate(out, plan, label, players[label], seeds, split, phase, device, stop)

    for name in list(players):
        screen[name] = score(name, SCREEN, 'screening', 'calibration')
        if not screen[name]['complete']:
            if skip_timed_family('direct', f'direct screening incomplete at {name}'):
                break
            return f'direct screening incomplete at {name}'
    direct = None if 'direct' in omitted else best(list(players), screen)
    for recipe in RECIPES:
        for seed in TRAIN:
            label = f'{seed}-{recipe}'
            candidate = train_arm(out, incumbent, donor, plan, seed, recipe, stop)
            if not candidate:
                if skip_timed_family('PPO', f'training incomplete at {label}'):
                    break
                return f'training incomplete at {label}'
            players[label] = candidate
        if 'PPO' in omitted:
            break
    recipes = {}
    winner = None
    if 'PPO' not in omitted:
        for recipe, seed in product(RECIPES, TRAIN):
            label = f'{seed}-{recipe}'
            screen[label] = score(label, SCREEN, 'screening', 'development')
            if not screen[label]['complete']:
                reason = f'PPO screening incomplete at {label}'
                if skip_timed_family('PPO', reason):
                    break
                return reason
        else:
            recipes = {
                r: {k: statistics.mean(screen[f'{s}-{r}'][k] for s in TRAIN) for k in ('mean', 'median')}
                for r in RECIPES
            }
            winner = best(RECIPES, recipes)
    if not screen.get('incumbent', {}).get('complete'):
        screen['incumbent'] = score('incumbent', SCREEN, 'screening', 'development')
        if not screen['incumbent']['complete']:
            return 'incumbent screening incomplete'
    eligible = [label for label in (direct, f'{TRAIN[0]}-{winner}' if winner else None) if label]
    fast = best(['incumbent', *eligible], screen)
    challengers = [name for name in players if name.startswith('weight-')] if direct else []
    if winner:
        challengers.append(f'{TRAIN[0]}-{winner}')
    challenger = best(challengers, screen) if challengers else None
    searches = []
    for label in (['incumbent', challenger] if challenger else ['incumbent']):
        name = label + '-lookahead'
        players[name] = dict(players[label], mode='lookahead')
        searches.append(name)
        screen[name] = score(name, SCREEN, 'screening', 'development')
        if not screen[name]['complete']:
            return f'lookahead screening incomplete at {name}'
    search = best(searches, screen)
    spatial.freeze(
        out / 'screened.json',
        dict(
            recipe_scores=recipes,
            omitted=omitted,
            recipe=winner,
            fast=fast,
            strongest_greedy_challenger=challenger,
            lookahead=search,
        ),
    )
    qualified = {}
    for label in dict.fromkeys(('incumbent', fast, search)):
        qualified[label] = score(label, QUALIFY, 'qualification', 'development')
        if not qualified[label]['complete']:
            return f'qualification incomplete at {label}'
    fast = best(['incumbent', fast], qualified)
    selected = best(qualified, qualified)
    spatial.freeze(
        out / 'selected.json',
        dict(
            protocol=ev.digest(plan),
            name=selected,
            player=players[selected],
            greedy=players[fast],
            lookahead=players[search],
            qualification={k: {m: r[m] for m in ('mean', 'median', 'minimum')} for k, r in qualified.items()},
        ),
    )
    result = dict(
        protocol=ev.digest(plan),
        qualified_winner=selected,
        greedy=players[fast],
        lookahead=players[search],
        omitted=omitted,
        all_workloads_complete=not omitted,
    )
    if selected == 'incumbent':
        result.update(
            complete=True,
            promoted=False,
            reason='incumbent won qualification',
            recommended=players['incumbent'],
        )
    else:
        final = {}
        for label in ('incumbent', selected):
            final[label] = score(label, HOLDOUT, 'holdout', 'holdout')
            if not final[label]['complete']:
                return f'sealed holdout incomplete at {label}'
        paired = ev.paired(final[selected], final['incumbent'])
        result.update(
            complete=True,
            paired=paired,
            distribution={
                label: {key: rows[key] for key in ('mean', 'median', 'bottom_decile', 'longest', 'minimum')}
                for label, rows in final.items()
            },
            pairs=[
                dict(seed=seed, incumbent=a, finalist=b, difference=b - a)
                for seed, a, b in zip(
                    HOLDOUT, final['incumbent']['placements'], final[selected]['placements']
                )
            ],
            promoted=paired['passes'],
            recommended=players[selected if paired['passes'] else 'incumbent'],
        )
    spatial.freeze(out / 'result.json', result)
    return 'complete'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'incumbent', 'donor', 'performance-gate'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--calibration-runtime', type=Path, default=ROOT)
    p.add_argument('--calibration-driver', type=Path, default=ROOT / 'tools/calibrate_cross_critics.py')
    args = p.parse_args(argv)
    out, incumbent, donor = (p.resolve() for p in (args.output, args.incumbent, args.donor))
    torch.set_num_threads(1)
    os.environ['TETRIS4D_GAME_LIBRARY'] = 'off'
    ev.game.configure_backend('native')
    ev.game.load_checkpoint(ROOT / 'models/trained-sigmoid.json')
    device = ev.neural.device_for('cuda')
    with base.run_lock(out / 'pipeline'), v4.v2.device_lock(device), base.Budget(float('inf')) as stop:
        meter.import_preparation(out, args.performance_gate.resolve().parent)
        with meter.exclusive_session(out, stop) as available:
            if not available:
                reason = '24-hour GPU ceiling reached'
            else:
                plan = read_protocol(
                    out,
                    incumbent,
                    donor,
                    device,
                    args.performance_gate.resolve(),
                    args.calibration_runtime.resolve(),
                    args.calibration_driver.resolve(),
                )
                spatial.freeze(out / 'protocol.json', plan)
                reason = (
                    run(out, incumbent, donor, plan, device, stop)
                    if not (out / 'result.json').is_file()
                    else 'complete'
                )
                if reason == 'complete' and ev.read_json(out / 'result.json')['protocol'] != ev.digest(plan):
                    raise ValueError('Completed result belongs to a different protocol')
        base.durable_json(
            out / 'status.json',
            dict(
                reason=reason,
                stop=stop.reason,
                updated_unix=time.time(),
                recommended=(
                    ev.read_json(out / 'result.json')['recommended']
                    if reason == 'complete'
                    else player(incumbent, calibration.INCUMBENT)
                ),
                gpu_hours=sum(meter.charged(row) for row in meter.ledger(out)) / 3600,
            ),
        )
        print(reason, flush=True)
        return 0 if reason == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
