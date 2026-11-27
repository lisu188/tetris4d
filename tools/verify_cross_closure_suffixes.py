"""Verify the registered capped and natural closure reports against the archived replay receipts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BRANCHES = ('greedy', 'best_preserving')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, expected):
    result = {(row['seed'], row['replicate']): row for row in rows}
    check(len(rows) == len(result) and set(result) == expected, 'Incomplete, duplicate or unexpected future')
    return result


def arm_at_boundary(capped, boundary, natural):
    check(object_hash(capped) == boundary['first_pass_endpoint_sha256'], 'Capped endpoint digest differs')
    check(all(natural[key] == value for key, value in boundary.items()), 'Continuation boundary differs')
    check(natural['status'] == 'physical_top_out', 'Missing natural endpoint')
    if capped['end'] == 'top_out':
        check((natural['n'], natural['clears'], natural['current'], natural['board_sha256']) ==
              (capped['placements'], capped['clears'], capped['current'], capped['board_sha256']),
              'An already-finished arm changed')
        for field in ('target', 'stream_sha256', 'snapshot_sha256', 'draw_chain', 'action_chain'):
            check(natural[field] == boundary['first_pass_' + field], 'Already-finished state differs')
    else:
        check(capped['end'] == 'horizon' and natural['n'] > capped['placements'],
              'Censored arm did not continue to a later physical endpoint')


def paired_stats(differences):
    return {'wins': sum(d > 0 for d in differences), 'ties': differences.count(0),
            'losses': sum(d < 0 for d in differences), 'mean_difference': statistics.mean(differences)}


def report(run_dir):
    capped_dir = run_dir / 'public-cold612-paired-suffixes'
    natural_dir = run_dir / 'public-cold612-paired-suffixes-natural'
    capped = read(capped_dir / 'result.json')
    repeated = read(capped_dir / 'verified.json')
    registration = read(natural_dir / 'plan.json')
    boundary = read(natural_dir / 'boundary-verification.json')
    natural = read(natural_dir / 'result.json')
    compact = read(ROOT / 'validation/cross-closure-suffix-results.json')
    plan_path = ROOT / 'validation/cross-closure-suffix-plan.json'
    plan = read(plan_path)
    capped_hash = file_hash(capped_dir / 'result.json')
    plan_hash = file_hash(natural_dir / 'plan.json')
    check(capped['pairs'] == repeated['pairs'] and capped_hash == compact['independent_repeat_of_result_sha256']
          and file_hash(capped_dir / 'verified.json') == compact['full_result_sha256'], 'Capped replays differ')
    check(capped['plan_sha256'] == repeated['plan_sha256'] == registration['original_plan_sha256'] == file_hash(plan_path)
          and registration['original_runner_sha256'] == capped['runner_sha256']
          and registration['runtime_revision'] == plan['runtime_revision'] == capped['runtime_revision']
          and registration['roots_sha256'] == plan['root_sha256']
          and registration['checkpoint_sha256'] == plan['checkpoint_sha256'], 'Registered inputs differ')
    check(natural['plan_sha256'] == boundary['plan_sha256'] == plan_hash and
          capped_hash == natural['source_result_sha256'] == registration['original_result_sha256'] == boundary['source_result_sha256']
          and file_hash(natural_dir / 'continue.py') == natural['runner_sha256'] == boundary['runner_sha256'],
          'Natural replay sources differ')
    check(natural['status'] == 'complete' and natural['all_pairs_natural'] and natural['controller_stops'] == 0,
          'Incomplete natural replay')
    seeds = sorted(map(int, plan['root_sha256']))
    keys = {(s, i) for s in seeds for i in range(plan['replicates'])}
    first, final, cohort = (indexed(rows, keys) for rows in (capped['pairs'], natural['rows'], registration['cohort']))
    check(set(boundary['boundaries']) == {f'{s}/{i}/{a}' for s, i in keys for a in BRANCHES},
          'Missing or extra replay boundary')
    roots, endpoints = {}, []
    all_deltas, cap_deltas = [], []
    arm_sums, end_pieces, censored, capped_pairs = Counter(), Counter(), Counter(), Counter()
    for seed in seeds:
        lengths = {a: [] for a in BRANCHES}
        deltas = []
        root_caps = {a: [] for a in BRANCHES}
        for i in range(plan['replicates']):
            pair, outcome, registered = first[seed, i], final[seed, i], cohort[seed, i]
            check(pair['draws_sha256'] == registered['draws_sha256'], 'Future prefix differs')
            portable = {'seed': seed, 'replicate': i, 'draws_sha256': pair['draws_sha256']}
            for arm in BRANCHES:
                b, f = pair[arm], outcome[arm]
                check(b['end'] == registered[arm], 'Registered boundary reason differs')
                arm_at_boundary(b, boundary['boundaries'][f'{seed}/{i}/{arm}'], f)
                check(b['placements'] == compact['roots'][str(seed)][arm]['placements'][i], 'Published capped length differs')
                if b['end'] == 'horizon':
                    check(b['placements'] == plan['horizon'], 'Horizon offset differs')
                    root_caps[arm].append(i)
                    censored[arm] += 1
                arm_sums[arm] += f['n']
                end_pieces[f['current']] += 1
                lengths[arm].append(f['n'])
                portable[arm] = dict(placements=f['n'], clears=f['clears'], end='top_out',
                                     current=f['current'], board_sha256=f['board_sha256'])
            stopped = sum(pair[arm]['end'] == 'horizon' for arm in BRANCHES)
            capped_pairs[('neither', 'one', 'both')[stopped]] += 1
            endpoints.append(portable)
            delta = lengths['best_preserving'][-1] - lengths['greedy'][-1]
            cap_delta = pair['best_preserving']['placements'] - pair['greedy']['placements']
            check(delta == outcome['exact_difference'] and cap_delta == pair['difference'], 'Pair arithmetic differs')
            deltas.append(delta)
            cap_deltas.append(cap_delta)
        for arm, caps in root_caps.items():
            check(caps == compact['roots'][str(seed)][arm]['alive_at_horizon'], 'Published capped endings differ')
        roots[str(seed)] = {**lengths, **paired_stats(deltas), 'net_placements': sum(deltas)}
        all_deltas.extend(deltas)
    total = sum(all_deltas)
    for root in roots.values():
        root['mean_without_this_root'] = (total - root['net_placements']) / (len(keys) - plan['replicates'])
    stats = paired_stats(all_deltas)
    check(stats['mean_difference'] == natural['all_32_mean'] and total == natural['paired_natural_sum']
          and [stats[k] for k in ('wins', 'ties', 'losses')] == natural['wins_ties_losses']
          and statistics.mean(cap_deltas) == compact['overall']['mean_paired_restricted_difference'],
          'Published aggregate differs')
    return {
        'scope': 'Four eligible final closures among eight original public 612000-cold games. Eight sampled legal futures per board. Both continuations use the same greedy actor after the root action.',
        'replicate_order': list(range(plan['replicates'])),
        'natural_registration_utc': registration['registered_utc'],
        'provenance': {'checkpoint_sha256': plan['checkpoint_sha256'], 'runtime_revision': plan['runtime_revision'],
                       'natural_registration_sha256': plan_hash, 'capped_registration_sha256': file_hash(plan_path),
                       'capped_result_sha256': capped_hash, 'repeated_capped_result_sha256': file_hash(capped_dir / 'verified.json'),
                       'boundary_verification_sha256': file_hash(natural_dir / 'boundary-verification.json'),
                       'natural_result_sha256': file_hash(natural_dir / 'result.json'), 'natural_runner_sha256': natural['runner_sha256']},
        'endpoints_sha256': object_hash(endpoints),
        'coverage': {'source_boards': len(seeds), 'paired_futures': len(keys), 'natural_physical_endings': sum(end_pieces.values()),
                     'terminal_piece_counts': dict(end_pieces), 'already_physical_at_128_or_earlier': 2 * len(keys) - sum(censored.values()),
                     'continued_after_128': dict(censored), 'pairs_by_arms_alive_at_128': dict(capped_pairs),
                     'natural_censored': 0, 'controller_stops': 0},
        'overall': {**stats, 'net_placements': total, 'median_difference': statistics.median(all_deltas),
                    'mean_greedy': arm_sums['greedy'] / len(keys), 'mean_best_preserving': arm_sums['best_preserving'] / len(keys),
                    'at_128': paired_stats(cap_deltas)},
        'roots': roots,
        'limits': 'The four boards were selected where the original actor lost its final CROSS landing and a shortlist alternative existed. Futures on one board share a decision. These data estimate the conditional effect of preserving once under the specified draws. They provide no unbiased estimate of whole-game survival or deployment benefit.'
    }


def verify_simulation(replay, run_dir, report_data):
    capped = read(run_dir / 'public-cold612-paired-suffixes/verified.json')
    check(replay['plan_sha256'] == capped['plan_sha256'] and
          replay['runtime_revision'] == capped['runtime_revision'], 'Simulation inputs differ')
    keys = {(row['seed'], row['replicate']) for row in capped['pairs']}
    rows = indexed(replay['pairs'], keys)
    if replay['horizon'] is None:
        fields = ('seed', 'replicate', 'draws_sha256', *BRANCHES)
        endpoints = [{k: rows[key][k] for k in fields} for key in sorted(keys)]
        check(object_hash(endpoints) == report_data['endpoints_sha256'], 'Replayed natural endpoints differ')
    else:
        check(replay['horizon'] == capped['horizon'] and rows == indexed(capped['pairs'], keys),
              'Replayed capped endpoints differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='Optional output from the capped or uncapped CPU simulator')
    args = parser.parse_args()
    expected = read(ROOT / 'validation/cross-closure-suffix-natural.json')
    check(report(args.run_dir) == expected, 'Tracked natural evidence differs')
    if args.replay:
        verify_simulation(read(args.replay), args.run_dir, expected)
    print('Verified 32 paired futures across four boards, at the 128-placement boundary and physical top-out.')


if __name__ == '__main__':
    main()
