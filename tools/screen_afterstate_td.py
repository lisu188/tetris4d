"""Paired, two-stage development tournament on distinct recorded top-out streams."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tetris_afterstate_td as td
import train_afterstate_td as training
from tools import evaluate_afterstate_td as ev

base, game = training.base, training.game
PHASES = ('discovery', 'confirmation')
BOUNDS = (.5, 1.)
DEFAULT_SEED = 2026112301


def read(path):
    return json.loads(Path(path).read_text())


def choose_positions(entries, excluded_ids, excluded_seeds, count, seed):
    """One deterministic position per eligible stream, without inspecting any policy outcome."""
    excluded = set(excluded_seeds) | {r['seed'] for r in entries if r['id'] in excluded_ids}
    records = sorted(entries, key=lambda r: ev.digest([seed, r['id']]))
    positions = []
    for r in records:
        if (r['seed'] in excluded or r['id'] in excluded_ids or r.get('piece_set') != 'expanded'
                or r.get('stop_reason') != 'top_out' or not r.get('game_over') or r['placements'] <= 240):
            continue
        remaining = 80 + int(ev.digest(['offset',seed,r['id']])[:8],16) % 161
        positions.append(dict(seed=r['seed'], game_id=r['id'], file=r['file'],
                              offset=r['placements']-remaining, recorded_remaining=remaining))
        excluded.add(r['seed'])
        if len(positions) == count:
            return positions
    raise ValueError('Insufficient independent development streams')


def save_position(path, game_record, entry):
    record = base.library.load(game_record)
    if (record.meta['seed'] != entry['seed'] or len(record.moves) != entry['offset']+entry['recorded_remaining']
            or not 0 <= int(record.moves[entry['offset'],6]) < game.HEIGHT):
        raise ValueError('Incompatible recorded game or unknown target')
    for i, (state, _, move) in enumerate(base.library.replay(record, verify=False)):
        state.locked_pieces = ()
        if i == entry['offset']:
            target = int(move[6])
            env = training.Environment(state, target, 0,
                {'kind':'library','game_id':entry['game_id'],'offset':i,
                 'remaining_recorded':entry['recorded_remaining']}, i, state.layers)
            payload = training.pack_env(env)
            temp = path.with_suffix('.tmp')
            torch.save(payload, temp)
            os.replace(temp, path)
            return
    raise ValueError('Recorded start not found')


def stage(args):
    snapshot = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    signature = snapshot['extra']['signature']
    calibration = read(args.beta)
    if signature['source_sha256'] != calibration['source_sha256'] or signature['beta'] != calibration['beta']:
        raise ValueError('Calibration and pinned TD training signature differ')
    n_train = snapshot['extra']['next_episode']
    excluded_seeds = set(ev.HOLDOUT_SEEDS) | set(ev.DEVELOPMENT_SEEDS)
    excluded_seeds.update(2*(signature['seed']+i) for i in range(n_train))
    excluded_ids = {row[0] for row in signature['danger_ids']} | {row[0] for row in calibration['position_ids']}
    entries = list(base.library.iter_index(args.library))
    starts = choose_positions(entries, excluded_ids, excluded_seeds, args.discovery+args.confirmation, args.seed)
    full = []
    for i, item in enumerate(starts):
        phase = PHASES[int(i>=args.discovery)]
        src = args.library/item['file']
        packed = args.output/f"{item['game_id']}-{item['offset']}.pt"
        if not packed.exists():
            save_position(packed, src, item)
        data = torch.load(packed, map_location='cpu', weights_only=True)
        if data['start']['game_id'] != item['game_id'] or data['pieces'] != item['offset']:
            raise ValueError('Position cache conflicts with selected source')
        full.append({**item,'phase':phase,'record_sha256':base.file_hash(src),
                     'snapshot':packed.name,'sha256':base.file_hash(packed),'board_sha256':data['board_hash']})
        print(f"Prepared {i+1}/{len(starts)} {phase} seed={item['seed']} offset={item['offset']}",flush=True)
    base.durable_json(args.output/'positions.json', dict(schema='afterstate-development-positions-v1', seed=args.seed,
        checkpoint_sha256=base.file_hash(args.checkpoint), calibration_sha256=base.file_hash(args.beta),
        index_sha256=base.file_hash(args.library/'index.jsonl'), topout_window=[80,240],
        excluded_game_ids=sorted(excluded_ids),excluded_stream_seeds=sorted(excluded_seeds), positions=full))


def validate_splits(manifest):
    rows=manifest['positions']
    seeds=[p['seed'] for p in rows]
    ids=[p['game_id'] for p in rows]
    if (set(p['phase'] for p in rows)!=set(PHASES) or len(set(ids))!=len(ids) or len(set(seeds))!=len(seeds)
            or set(seeds).intersection((*ev.HOLDOUT_SEEDS,*ev.DEVELOPMENT_SEEDS,*manifest['excluded_stream_seeds']))
            or set(ids).intersection(manifest['excluded_game_ids'])):
        raise ValueError('Development splits overlap, include a reserved stream, or omit a phase')


def select_challenger(results):
    """Selection is by mean, median, then smaller intervention in a complete discovery round."""
    names = [f'bound-{b:g}' for b in BOUNDS]
    if any(not results[n]['complete'] for n in names):
        raise ValueError('Discovery results are incomplete')
    return max(names, key=lambda n: (results[n]['mean'],results[n]['median'],-float(n.split('-')[1])))


def inputs(args, positions):
    sources = [ROOT/n for n in ('tetris_afterstate_td.py','tetris_neural.py','sparse_tetris_4d_optimized.py',
        'tetris_survival_v4.py','tetris_survival_v5.py','tetris_gpu_board_features.py','tetris_gpu_simulation.py',
        'tetris_gpu_spatial.py','tetris_gpu_encoding.py','tools/evaluate_afterstate_td.py','tools/screen_afterstate_td.py',
        'native/libtetris4d_native.so')]
    frozen=ROOT/'screen-runtime.json'
    if frozen.exists():
        runtime=read(frozen)
        if any(base.file_hash(ROOT/p)!=expected for p,expected in runtime['sha256'].items()):
            raise ValueError('Frozen screening runtime changed')
        revision=runtime['revision']
    else:
        revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    return dict(v4_sha256=base.file_hash(args.v4), td_sha256=base.file_hash(args.checkpoint),
        ppo_sha256=base.file_hash(args.ppo) if args.ppo else None, weights_sha256=base.file_hash(args.weights),
        calibration_sha256=base.file_hash(args.beta), beta=read(args.beta)['beta'],
        source_sha256={str(p.relative_to(ROOT)):base.file_hash(p) for p in sources},
        revision=revision, runtime_receipt_sha256=base.file_hash(frozen) if frozen.exists() else None,
        torch=str(torch.__version__),cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(), seeds=[p['seed'] for p in positions])


def play(model, scoring, batch, cap, budget, done):
    from tetris_gpu_simulation import GpuPreparer
    gpu = torch.device('cuda')
    preparer = GpuPreparer(model.base.config if scoring else model.config, gpu)
    states = [e.state for e in batch]
    targets = [e.target for e in batch]
    initial = [s.pieces for s in states]
    clears = [s.layers for s in states]
    live = list(range(len(states)))
    iteration = 0
    began = time.monotonic()
    while live:
        if budget.expired():
            return
        if iteration % 64 == 0 and torch.cuda.mem_get_info()[0] < 768 * 2**20:
            raise RuntimeError('Shared GPU free memory dropped below the 768 MiB reserve')
        iteration += 1
        moves = ev.select_moves(model,[states[i] for i in live],[targets[i] for i in live],scoring,gpu,preparer)
        again = []
        for local, index in enumerate(live):
            state = states[index]
            if local in moves:
                target,candidate = moves[local]
                move = game.materialize_move(state.board,candidate)
                state.commit(move)
                state.locked_pieces = ()
                following = game.next_hyperlayer_target(state.board,target) if move.layers else None
                targets[index] = target if following is None else following
                if state.pieces-initial[index] < cap:
                    again.append(index)
                    continue
                reason = 'cap'
            else:
                reason = 'top_out' if state.game_over else 'target_or_no_move'
            done(dict(seed=state.seed, placements=state.pieces-initial[index], clears=state.layers-clears[index],
                      reason=reason,wall_seconds=time.monotonic()-began))
        live = again


def starting_env(args, pos):
    if args.phase == 'fresh':
        state = game.Game(pos['seed'],'expanded')
        base.library.ignore(state)
        return training.Environment(state,1,0,{'kind':'fresh'},0,0)
    path = args.output/pos['snapshot']
    if base.file_hash(path) != pos['sha256']:
        raise ValueError('Pinned starting board changed')
    return training.unpack_env(torch.load(path,map_location='cpu',weights_only=True))


def load_policies(args, device):
    if not args.v4:
        raise ValueError('Comparison requires the calibrated frozen V4 control')
    source = ev.load_model(args.v4,device)
    learner = ev.load_model(args.checkpoint,device)
    base_weights = learner.base.state_dict()
    if (source.state_dict().keys()!=base_weights.keys()
            or not all(torch.equal(v,base_weights[k]) for k,v in source.state_dict().items())):
        raise ValueError('Pinned TD checkpoint and frozen V4 do not share the identical actor')
    models = {'v4-actor':source, 'v4-frozen':td.from_v4(source).eval(), 'pure-td':learner}
    if args.ppo:
        models['ppo'] = ev.load_model(args.ppo,device)
    return models


@torch.inference_mode()
def tournament(args):
    stage_id = read(args.output/'positions.json')
    validate_splits(stage_id)
    if (stage_id['checkpoint_sha256']!=base.file_hash(args.checkpoint)
            or stage_id['calibration_sha256']!=base.file_hash(args.beta)):
        raise ValueError('Position staging and tournament inputs differ')
    # A small bounded experiment shares the device with the already-running trainer.
    gpu=torch.device('cuda')
    free,total=torch.cuda.mem_get_info()
    if free < 1792*2**20:
        raise RuntimeError(f'Tournament requires 1792 MiB free at launch; found {free//2**20}')
    torch.cuda.set_per_process_memory_fraction(min(1.,1024*2**20/total),device=0)
    phase=args.phase
    positions = ([p for p in stage_id['positions'] if p['phase']==phase] if phase!='fresh' else
                 [dict(seed=s) for s in ev.DEVELOPMENT_SEEDS[:4]])
    if not positions or set(p['seed'] for p in positions).intersection(ev.HOLDOUT_SEEDS):
        raise ValueError('Invalid development position split')
    cap = args.cap or dict(discovery=1500,confirmation=2500,fresh=32000)[phase]
    proof = inputs(args,positions)
    policies = ['v4-actor','v4-frozen','pure-td']
    discovery_path = args.output/'discovery.json'
    if phase=='discovery':
        policies += [f'bound-{b:g}' for b in BOUNDS]
        chosen = None
    else:
        disc = read(discovery_path)
        if (not disc['complete'] or disc['inputs']['td_sha256']!=proof['td_sha256']
                or disc['inputs']['v4_sha256']!=proof['v4_sha256'] or
                disc['inputs']['calibration_sha256']!=proof['calibration_sha256']
                or disc['inputs']['source_sha256']!=proof['source_sha256']):
            raise ValueError('Discovery selection is incomplete or belongs to another runtime')
        chosen = disc['selected_challenger']
        policies.append(chosen)
        if args.ppo:
            policies.append('ppo')
    if phase in ('confirmation','fresh') and not args.ppo:
        raise ValueError('Confirmation and fresh comparisons require the prior best PPO checkpoint')
    models=load_policies(args,gpu)
    report_path = discovery_path if phase=='discovery' else args.output/f'{phase}-{len(positions)}-{cap}.json'
    report=dict(schema='afterstate-td-development-tournament-v1',phase=phase, complete=False, cap=cap,
        inputs=proof, positions=positions, policies={}, selected_challenger=chosen, batch=args.batch,
        discovered_from_sha256=None if phase=='discovery' else base.file_hash(discovery_path),
        discovery_order='mean capped incremental placements, median, smaller bound',
        gpu_start_free_bytes=free,gpu_torch_allocator_limit_bytes=1024*2**20)
    seeds=[p['seed'] for p in positions]
    with base.Budget(args.max_seconds/3600) as budget:
        for name in policies:
            scoring = name not in ('v4-actor','ppo')
            if name.startswith('bound-'):
                model=models['pure-td'].configure_policy(actor_bound=float(name.split('-')[1]),value_scale=proof['beta'])
            else:
                model=models[name]
                if name=='pure-td':
                    model.configure_policy()
            key=dict(schema=report['schema'],phase=phase,cap=cap,policy=name,positions=positions,
                     models=[proof['td_sha256'],proof['v4_sha256'],proof['ppo_sha256']],calibration=proof['calibration_sha256'],
                     runtime=proof['source_sha256'], torch=proof['torch'],cuda=proof['cuda'],device=proof['device'],
                     weights=proof['weights_sha256'],batch=args.batch)
            cache=args.output/'outcomes'/ev.digest(key)
            rows={}
            def finished(row):
                ev.write_cache(cache/f"{row['seed']}.json",dict(key_sha256=ev.digest(key),seed=row['seed']),row)
                rows[row['seed']]=row
            for start in range(0,len(positions),args.batch):
                missing=[]
                for pos in positions[start:start+args.batch]:
                    seed=pos['seed']
                    row=ev.checked_cached(cache/f'{seed}.json',dict(key_sha256=ev.digest(key),seed=seed))
                    if row is None:
                        missing.append(starting_env(args,pos))
                    else:
                        rows[seed]=row
                if missing and not budget.expired():
                    play(model,scoring,missing,cap,budget,finished)
                report['policies'][name]=ev.summarize(rows,seeds,name,cap)
                base.durable_json(report_path,report)
            summary=report['policies'][name]
            print(f"{phase} {name}: {len(rows)}/{len(seeds)} done; mean={summary['mean']}, capped={summary['censored']}",flush=True)
            if not summary['complete']:
                return 2
    results=report['policies']
    chosen=chosen or select_challenger(results)
    rivals=[p for p in policies if p!=chosen and not p.startswith('bound-')]
    report.update(complete=True,selected_challenger=chosen,
                  comparisons=[ev.paired(results[chosen],results[name]) for name in rivals],
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.monotonic()-budget.started)
    base.durable_json(report_path,report)
    return 0


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('stage','discovery','confirmation','fresh'))
    parser.add_argument('--checkpoint',type=Path,required=True,help='Immutable early TD snapshot shared by all challengers')
    parser.add_argument('--beta',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--v4',type=Path)
    parser.add_argument('--ppo',type=Path)
    parser.add_argument('--weights',type=Path,default=ROOT/'models/trained-sigmoid.json')
    parser.add_argument('--batch',type=int,choices=(4,8),default=4)
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--discovery',type=int,default=4)
    parser.add_argument('--confirmation',type=int,default=8)
    parser.add_argument('--max-seconds',type=int,default=5400)
    parser.add_argument('--cap',type=int,help='Capped additional placements at each start')
    return parser.parse_args(argv)


def main():
    args=arguments()
    if args.max_seconds<=0 or (args.cap is not None and args.cap<1):
        raise ValueError('Caps and time budgets must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    game.configure_backend('native')
    game.load_checkpoint(args.weights)
    if args.phase=='stage':
        if not args.library or min(args.discovery,args.confirmation)<2:
            raise ValueError('Staging needs a library and >=2 independent streams per phase')
        stage(args)
        return 0
    return tournament(args)


if __name__ == '__main__':
    raise SystemExit(main())
