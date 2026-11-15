#!/usr/bin/env python3
"""Bounded, opt-in GPU parity probe alongside the immutable production trainer.

CPU reference fixtures are generated in a separate process using frozen SOURCE.
The default invocation checks inputs only. No action manages/signals a trainer.
Concurrent timings are explicitly labelled contention measurements.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

def artifact_root(source):
    return source.parent if source.parent.name.endswith('.artifacts') else source/'survival-training.artifacts'


SOURCE=Path(__file__).resolve().parents[1]
ART=artifact_root(SOURCE)
REFERENCE=ART/'source'
VALIDATION=ART/'gpu-validation'
RUN=ART/'run-16env'
FROZEN=ART/'checkpoints/pre-optimization-4env.pt'
FROZEN_SHA='015c29d308071967bf8a2dd4587f29ab214d2f58be7e1333452249e06b7fcca5'
sys.dont_write_bytecode=True


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def info(path):
    path=Path(path);return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
def write_new(path,value):
    path=Path(path);require(not path.is_symlink(),'Symlink output')
    with path.open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
def contained(path,root):
    path=Path(path);require(path.is_absolute() and path==path.resolve() and path.is_relative_to(root.resolve()),'Noncanonical/outside output path')
    return path


def probe_inputs(suite):
    names=['tools/benchmark_gpu_training.py','tetris_neural.py','sparse_tetris_4d_optimized.py','tetris_gpu_encoding.py']
    if suite in ('prepare','steady','preparation-contracts','integration','benchmark'):names+=['tetris_gpu_simulation.py']
    if suite=='preparation-contracts':names+=['tests/test_gpu_simulation.py']
    if suite=='encoding':names+=['tests/test_gpu_encoding.py']
    if suite in ('transfers','integration','benchmark'):names+=['train_survival_v2.py','train_survival.py','train_overnight.py','tests/test_gpu_transfers.py']
    proofs={name:info(SOURCE/name) for name in names}
    reference_names=['tetris_neural.py','sparse_tetris_4d_optimized.py'] if suite=='steady' else []
    if suite in ('integration','benchmark'):reference_names+=['train_survival_v2.py','train_survival.py','train_overnight.py','tetris_neural.py','sparse_tetris_4d_optimized.py']
    proofs.update({'reference/'+name:info(REFERENCE/name) for name in reference_names})
    return proofs


def metric_tail(path):
    """Bounded observation only; ignore a concurrently written partial line."""
    path=Path(path)
    try:
        with path.open('rb') as handle:
            size=path.stat().st_size;offset=max(0,size-16384);handle.seek(offset);data=handle.read(16384)
        lines=data.split(b'\n')
        if offset:lines=lines[1:]
        events=[]
        for line in reversed(lines[:-1]):
            try:row=json.loads(line)
            except (ValueError,UnicodeError):continue
            events.append({key:row[key] for key in ('event','utc','runner_updates','seed','pieces','game','games') if key in row})
            if len(events)==3:break
        return {'bytes':size,'recent_events':events}
    except OSError:return {'unavailable':True}


def trainer_snapshot():
    manifest=read(RUN/'manifest.json');summary=read(RUN/'summary.json');status=read(ART/'status.json')
    child=status.get('child',{});pid=child.get('pid',-1)
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        alive=stat[0]!='Z' and stat[19]==str(child.get('start_ticks'))
    except (OSError,IndexError):alive=False
    require(alive and status.get('state')=='running','Production trainer is not confirmed running')
    return {'observed_epoch':time.time(),'pid':pid,'start_ticks':child['start_ticks'],
            'runner_updates':summary['runner_updates'],'placements':summary['placements'],
            'manifest_runner_updates':manifest['runner_updates'],'summary_sha256':sha(RUN/'summary.json'),'metrics_tail':metric_tail(RUN/'metrics.jsonl')}


def progress_ok(before,after):
    require((before['pid'],before['start_ticks'])==(after['pid'],after['start_ticks']),'Trainer identity changed during probe')
    require(after['runner_updates']>=before['runner_updates'] and after['placements']>=before['placements'],'Trainer counters decreased')
    return after['runner_updates']>before['runner_updates'] and after['placements']>before['placements']


def gpu_telemetry():
    cmd='/usr/lib/wsl/lib/nvidia-smi'
    out=subprocess.check_output([cmd,'--query-gpu=memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows=[tuple(float(x.strip()) for x in line.split(',')) for line in out.strip().splitlines()]
    require(len(rows)==1,'Probe requires the one known GPU')
    return {'free_mib':rows[0][0],'total_mib':rows[0][1],'utilization_percent':rows[0][2]}


def reference_worker(directory):
    try:os.nice(10)
    except OSError:pass
    # Import only the unchanged, production reference in this private process.
    sys.path.insert(0,str(REFERENCE))
    import numpy as np
    import torch
    import tetris_neural as neural
    import sparse_tetris_4d_optimized as game
    from dataclasses import asdict,replace
    torch.set_num_threads(1);game.configure_backend('native');game.load_checkpoint(ART.parent/'models/trained-sigmoid.json')
    require(sha(FROZEN)==FROZEN_SHA,'Frozen pre-optimization snapshot changed')
    saved=torch.load(FROZEN,map_location='cpu',weights_only=True)
    cases=[];arrays={};config=neural.Config(**saved['config'])
    def add(label,board,piece='I',target=1,seed=700001,pieces=0,counts=None,next_piece=None,cfg=None):
        state=game.Game(seed,'expanded');state.board=tuple(map(int,board));state.current=piece
        state.next_piece=next_piece or game.PIECE_NAMES[(game.PIECE_NAMES.index(piece)+1)%len(game.PIECE_NAMES)]
        state.pieces=pieces;state.piece_counts=counts or {name:0 for name in game.PIECE_NAMES}
        cfg=cfg or config;prepared=neural.prepare(state,target,cfg)
        i=len(cases);prefix=f'c{i:03d}_';arrays[prefix+'state_board']=np.asarray(state.board,dtype=np.uint16)
        row={'label':label,'seed':seed,'piece_set':'expanded','current':state.current,'next_piece':state.next_piece,
             'pieces':pieces,'piece_counts':state.piece_counts,'requested_target':target,'config':asdict(cfg),'prefix':prefix,
             'none':prepared is None}
        if prepared is not None:
            for key,value in prepared.arrays.items():arrays[prefix+key]=np.asarray(value)
            row.update(teacher=prepared.teacher,evaluations=prepared.evaluations,fallback=prepared.fallback,
                       options=[[c.rotation,c.x,c.y,c.w,c.z,c.layers,c.score] for c in prepared.options])
        cases.append(row)
    for piece in game.PIECE_NAMES:add('empty-'+piece,game.EMPTY_BOARD,piece)
    # Every orientation is represented in the primitive reference below; these
    # ragged states exercise real shortlist/context generation for every piece.
    ragged=tuple((1<<((x+2*y+3*w)%9))-1 for w in range(16) for y in range(16) for x in range(16))
    for piece in game.PIECE_NAMES:add('ragged-'+piece,ragged,piece,target=9)
    # Reconstruct the four frozen real boards literally, checking every move's
    # piece stream/collision and the final packed-board checksum. No model work.
    for index,snap in enumerate(saved['extra']['environments']):
        board=np.zeros(4096,dtype=np.uint16);stream=game.PieceStream(snap['seed'],snap['piece_set'])
        current,next_piece=stream.pop(),stream.pop()
        for move in snap['moves']:
            require(move['piece']==current,'Frozen piece stream mismatch')
            base=game.column_index(move['x'],move['y'],move['w'])
            for offset,mask,_ in game.SHAPES[current][move['rotation']].columns:
                added=mask<<move['z'];require(not int(board[base+offset])&added,'Frozen replay collision')
                board[base+offset]|=added
            if move['cleared_mask']:
                compact=np.empty_like(board)
                game._native_check(game._NATIVE.t4_compact(board.ctypes.data,compact.ctypes.data,move['cleared_mask'],int(game._NATIVE_USE_BMI2)))
                board=compact
            current,next_piece=next_piece,stream.pop()
        require(hashlib.sha256(board.tobytes()).hexdigest()==snap['board_sha256'],'Frozen final board mismatch')
        add('frozen-'+str(index),board,current,snap['target'],snap['seed'],snap['summary']['pieces'],snap['summary']['piece_counts'],next_piece)
    for mask,label in ((1,'single-clear'),(3,'double-clear')):
        board=[mask]*4096;board[0]=0;add(label,board,target=0)
    add('full-target-recovery',[1]*4096,target=0)
    add('topout',[65535]*4096,target=15)
    add('cap-piece-count',game.EMPTY_BOARD,'HYPER',pieces=32000)
    holes=[65535]*4096
    for offset,_,_ in game.SHAPES['O'][0].columns:holes[offset]=0
    add('exhaustive-fallback',holes,'O',target=0,cfg=replace(config,candidates=1,recover_target=False))
    # A small literal primitive matrix covers every catalog orientation and
    # both coordinate boundaries, including rigid drops/clears/target metrics.
    orientation=[]
    for piece in game.PIECE_NAMES:
        specs=[]
        for rotation,shape in enumerate(game.SHAPES[piece]):
            specs.extend(((rotation,0,0,0),(rotation,16-shape.extents[0],16-shape.extents[1],16-shape.extents[3])))
        context=game.evaluate_board(ragged);raw=np.asarray(specs,dtype=np.int32)
        scores,z,clears,_=game.native_scores(context,piece,raw)
        gains,blocked,future=game.sustained_batch_metrics(context,piece,raw,z,9)
        prefix='orientation_'+piece+'_'
        for key,value in [('specs',raw),('scores',scores),('z',z),('gains',gains),('blocked',blocked),('future',future)]:arrays[prefix+key]=value
        orientation.append({'piece':piece,'rotations':len(game.SHAPES[piece]),'specifications':len(specs),'clear_indices':list(clears)})
    directory=Path(directory);directory.mkdir()
    np.savez_compressed(directory/'arrays.npz',**arrays)
    write_new(directory/'cases.json',cases)
    require(not torch.cuda.is_initialized(),'CPU fixture generation initialized CUDA')
    manifest={'schema':'tetris4d-gpu-parity-fixtures-v1','checkpoint':info(FROZEN),'cases':len(cases),
              'orientation_matrix':orientation,'reference_sources':{name:info(REFERENCE/name) for name in ('tetris_neural.py','sparse_tetris_4d_optimized.py')},
              'files':[info(directory/'arrays.npz'),info(directory/'cases.json')],'cuda_initialized':False,
              'scope':'Prepare/shortlist fixtures plus literal all-orientation primitive references; no training or GPU throughput claim.'}
    require(sum(p.stat().st_size for p in directory.iterdir())<=8*2**20,'CPU fixtures exceed8MiB bound')
    write_new(directory/'manifest.json',manifest)


def load_fixtures(directory):
    import numpy as np
    directory=Path(directory);manifest=read(directory/'manifest.json')
    for proof in manifest['files']:require(info(proof['path'])==proof,'CPU fixture changed')
    return manifest,read(directory/'cases.json'),np.load(directory/'arrays.npz',allow_pickle=False)


def compare_arrays(expected,actual,label):
    import numpy as np
    if hasattr(actual,'detach'):actual=actual.detach().cpu().numpy()
    actual=np.asarray(actual);expected=np.asarray(expected)
    require(actual.shape==expected.shape,'Shape mismatch: '+label)
    require(actual.dtype==expected.dtype,'Dtype mismatch: '+label)
    np.testing.assert_array_equal(actual,expected,err_msg=label)


def cuda_suite(options,output,torch,stream,telemetry):
    import importlib.util,io,unittest
    names={
        'encoding':('test_gpu_encoding.py','TensorRecordTests',('test_cuda_all_masks_targets_mixed_records_model_and_gradients_exact',),'TETRIS4D_TEST_CUDA_ENCODING'),
        'transfers':('test_gpu_transfers.py','GpuTransferTests',('test_cuda_actions_validation_model_adam_and_metrics_exact',),'RUN_TETRIS_CUDA_TRANSFER_TESTS'),
        'benchmark':('test_gpu_transfers.py','GpuTransferTests',('test_cuda_16env_rollout_and_optimizer_parity',),'RUN_TETRIS_CUDA_TRANSFER_TESTS'),
        'integration':('test_gpu_transfers.py','GpuTransferTests',('test_cuda_actions_validation_model_adam_and_metrics_exact','test_cuda_16env_rollout_and_optimizer_parity'),'RUN_TETRIS_CUDA_TRANSFER_TESTS'),
        'preparation-contracts':('test_gpu_simulation.py','CudaPreparationContractTests',('test_all_pieces_and_retained_records','test_clear_dead_buried_and_exhaustive_fallback','test_optional_protection_and_small_odd_shortlists'),'TETRIS4D_TEST_CUDA')}
    filename,classname,methods,flag=names[options.suite]
    os.environ[flag]='1';sys.path.insert(0,str(SOURCE/'tests'))
    if options.suite in ('integration','benchmark'):
        require(sha(FROZEN)==FROZEN_SHA,'Frozen integration checkpoint changed')
        _,_,fixture_arrays=load_fixtures(options.fixtures);fixture_arrays.close()
        os.environ['TETRIS4D_GPU_INTEGRATION_FIXTURES']=str(options.fixtures)
        os.environ['TETRIS4D_GPU_INTEGRATION_CHECKPOINT']=str(FROZEN)
    for key in ('TETRIS4D_GPU_INTEGRATION_BENCHMARK','TETRIS4D_GPU_INTEGRATION_BENCHMARK_OUTPUT'):os.environ.pop(key,None)
    if options.suite=='benchmark':
        os.environ['TETRIS4D_GPU_INTEGRATION_BENCHMARK']='1'
        os.environ['TETRIS4D_GPU_INTEGRATION_BENCHMARK_OUTPUT']=str(output.with_suffix('.timing.json'))
    spec=importlib.util.spec_from_file_location('bounded_gpu_'+options.suite,SOURCE/'tests'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    log=io.StringIO();started=time.monotonic()
    with torch.cuda.stream(stream):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite([getattr(module,classname)(method) for method in methods]))
    stream.synchronize()
    require(result.wasSuccessful() and result.testsRun==len(methods) and not result.skipped,'CUDA suite did not pass: '+log.getvalue())
    write_new(output,{'passed':True,'suite':options.suite,'tests':len(methods),'wall_seconds':time.monotonic()-started,
                     'test_source':info(SOURCE/'tests'/filename),'test_output':log.getvalue(),
                     'timing_report':info(output.with_suffix('.timing.json')) if options.suite=='benchmark' else None,
                     'allocator_limit_mib':options.memory_mib,'stream_priority':stream.priority,
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                     'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'gpu_before':telemetry,'gpu_after':gpu_telemetry(),
                     'scope':'Exact CUDA contract test under concurrent training; no performance claim.'})


def steady_prepare(options,output,torch,stream,game,neural,GpuPreparer,telemetry):
    import importlib.util,statistics
    manifest,cases,arrays=load_fixtures(options.fixtures)
    frozen=[row for row in cases if row['label'].startswith('frozen-')]
    require(len(frozen)==4 and all(not r['none'] for r in frozen),'Expected four real nonempty frozen states')
    game.configure_backend('native');game.load_checkpoint(ART.parent/'models/trained-sigmoid.json')
    def workload(unique):
        states=[];targets=[]
        for index,row in enumerate(frozen*4):
            state=game.Game(row['seed'],row['piece_set'])
            for key in ('current','next_piece','pieces'):setattr(state,key,row[key])
            state.piece_counts=dict(row['piece_counts'])
            # These are explicitly constructed counter contexts on four real
            # boards, not sixteen distinct observed production environments.
            extra=index//4 if unique else 0
            state.pieces+=extra;state.piece_counts[state.current]+=extra
            state.board=tuple(map(int,arrays[row['prefix']+'state_board']))
            states.append(state);targets.append(row['requested_target'])
        return states,targets
    workloads=[('frozen4x4',*workload(False)),('four_boards_distinct_counters',*workload(True))]
    require(sha(SOURCE/'sparse_tetris_4d_optimized.py')==sha(REFERENCE/'sparse_tetris_4d_optimized.py'),'Reference engine changed')
    spec=importlib.util.spec_from_file_location('gpu_benchmark_frozen_neural',REFERENCE/'tetris_neural.py')
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    refconfig=reference.Config(**frozen[0]['config']);config=neural.Config(**frozen[0]['config'])
    preparer=GpuPreparer(config,torch.device('cuda'));reports=[];started=time.monotonic()
    with torch.cuda.stream(stream),torch.inference_mode():
        retained=preparer.prepare_batch(workloads[0][1],workloads[0][2])
        retained_values=[(value,value.clone()) for value in retained.batch.values()]
        retained_values.extend((value,value.clone()) for row in retained.records if row is not None for value in row.values() if isinstance(value,torch.Tensor))
        preparer.prepare_batch(workloads[1][1],workloads[1][2]);stream.synchronize()
        require(all(torch.equal(value,saved) for value,saved in retained_values),'Later preparation overwrote retained rollout records')
        for label,states,targets in workloads:
            def cpu():
                prepared=[reference.prepare(state,target,refconfig) for state,target in zip(states,targets)]
                require(all(p is not None for p in prepared),'Steady CPU fixture unexpectedly empty')
                return reference.collate([p.arrays for p in prepared],torch.device('cuda'))
            def gpu():
                result=preparer.prepare_batch(states,targets)
                require(not any(result.cpu_fallbacks),'Steady GPU workload entered a CPU fallback')
                # The trainer needs only its chosen moves; this one packed
                # transfer is included in the GPU path, never full options D2H.
                result.selected_candidates(result.teachers)
                return result.batch
            cpu_batch=cpu();gpu_batch=gpu();stream.synchronize()
            require(set(cpu_batch)==set(gpu_batch),'Steady output schema differs')
            for key in cpu_batch:compare_arrays(cpu_batch[key].cpu().numpy(),gpu_batch[key],key)
            cpu();gpu();stream.synchronize()
            wall={'cpu_prepare_collate_h2d':[],'gpu_prepare_device_selected_d2h':[]};events={key:[] for key in wall}
            for repeat in range(max(2,options.steps//4)):
                if time.monotonic()-started>=min(options.seconds,30):break
                order=tuple(wall) if repeat%2==0 else tuple(reversed(wall))
                for name in order:
                    stream.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                    begin=time.perf_counter();a.record(stream);result=cpu() if name.startswith('cpu') else gpu()
                    b.record(stream);b.synchronize();wall[name].append((time.perf_counter()-begin)*1000)
                    events[name].append(a.elapsed_time(b))
                time.sleep(.02)
            require(len(wall['cpu_prepare_collate_h2d'])>=2,'Need at least two paired warmed samples per workload')
            medians={key:statistics.median(value) for key,value in wall.items()}
            reports.append({'name':label,'environments':16,'distinct_real_boards':4,
                            'distinct_piece_counter_keys':len({(state.current,state.pieces) for state in states}),
                            'paired_samples':len(wall['cpu_prepare_collate_h2d']),'wall_ms':wall,'cuda_event_interval_ms':events,
                            'median_wall_ms':medians,'observed_concurrent_ratio':medians['cpu_prepare_collate_h2d']/medians['gpu_prepare_device_selected_d2h']})
    write_new(output,{'passed':True,'suite':'steady','workloads':reports,'retained_records_unchanged':True,
                     'wall_seconds':time.monotonic()-started,'allocator_limit_mib':options.memory_mib,'stream_priority':stream.priority,
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                     'fixture_manifest':info(options.fixtures/'manifest.json'),'frozen_reference':info(REFERENCE/'tetris_neural.py'),
                     'gpu_before':telemetry,'gpu_after':gpu_telemetry(),
                     'scope':'Warmed full preparation+collation/H2D versus direct device batch+selected moves D2H. Four real boards with repeated or constructed distinct counter contexts; CPU fallback excluded. Shared-device contention; CUDA event intervals include host launch gaps. No uncontended or training-throughput speedup claim.'})


def cuda_worker(options,output):
    import numpy as np
    import torch
    sys.path.insert(0,str(SOURCE))
    import tetris_neural as neural
    import sparse_tetris_4d_optimized as game
    torch.set_num_threads(1)
    try:os.nice(10)
    except OSError:pass
    free_before=gpu_telemetry();require(free_before['free_mib']>=options.memory_mib+2048,'Insufficient GPU reserve; probe declined')
    torch.cuda.set_per_process_memory_fraction(float(options.memory_mib*2**20/torch.cuda.get_device_properties(0).total_memory),0)
    stream=torch.cuda.Stream(priority=0)
    if options.suite in ('encoding','transfers','preparation-contracts','integration','benchmark'):
        cuda_suite(options,output,torch,stream,free_before);return
    from tetris_gpu_simulation import GpuPreparer
    if options.suite=='steady':
        steady_prepare(options,output,torch,stream,game,neural,GpuPreparer,free_before);return
    manifest,cases,arrays=load_fixtures(options.fixtures)
    groups=defaultdict(list)
    for i,row in enumerate(cases):groups[json.dumps(row['config'],sort_keys=True)].append(i)
    checked=0;calls=0;timings=[];fallbacks=[];started=time.monotonic()
    game.configure_backend('native');game.load_checkpoint(ART.parent/'models/trained-sigmoid.json')
    with torch.cuda.stream(stream),torch.inference_mode():
        for config_json,indices in groups.items():
            preparer=GpuPreparer(neural.Config(**json.loads(config_json)),torch.device('cuda'))
            for start in range(0,len(indices),16):
                require(calls<options.steps and time.monotonic()-started<options.seconds,'Probe step/time bound reached before all fixtures')
                selected=indices[start:start+16];states=[];targets=[];expected=[]
                for i in selected:
                    row=cases[i];prefix=row['prefix'];state=game.Game(row['seed'],row['piece_set'])
                    for key in ('current','next_piece','pieces','piece_counts'):setattr(state,key,row[key])
                    state.board=tuple(map(int,arrays[prefix+'state_board']));states.append(state);targets.append(row['requested_target'])
                    expected.append(None if row['none'] else {key:arrays[prefix+key] for key in ('boards','targets','context','features','indices','column_mask','valid','teacher_prior')})
                before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
                before.record(stream);result=preparer.prepare_batch(states,targets);after.record(stream);after.synchronize()
                valid=[j for j,row in enumerate(expected) if row is not None]
                require(list(result.environment_indices)==valid,'Nonempty environment mapping differs')
                require(len(result.prepared)==len(selected),'Original-order prepared mapping missing')
                if valid:
                    cpu_batch=neural.collate([expected[j] for j in valid],torch.device('cpu'))
                    require(set(cpu_batch)==set(result.batch),'Prepared tensor schema differs')
                    for key,value in cpu_batch.items():compare_arrays(value.numpy(),result.batch[key],key)
                    compare_arrays(np.asarray([cases[selected[j]]['teacher'] for j in valid],dtype=np.int64),result.teachers,'teachers')
                else:require(not result.batch,'All-None batch must be empty')
                option_specs=result._option_specs.detach().cpu().numpy() if valid else None
                option_scores=result._option_scores.detach().cpu().numpy() if valid else None
                compact={env:position for position,env in enumerate(valid)}
                for j,i in enumerate(selected):
                    row=cases[i];got=result.prepared[j]
                    require((got is None)==row['none'],'None/fallback status differs: '+row['label'])
                    if got is not None:
                        require(got.teacher==row['teacher'] and got.evaluations==row['evaluations'] and got.fallback==row['fallback'],'Prepared metadata differs: '+row['label'])
                        position=compact[j];length=len(row['options'])
                        options_actual=np.concatenate((option_specs[position,:length].astype(np.float64),option_scores[position,:length,None]),axis=1)
                        compare_arrays(np.asarray(row['options'],dtype=np.float64),options_actual,'option order/score '+row['label'])
                    checked+=1
                fallbacks.extend(getattr(result,'cpu_fallbacks',()))
                calls+=1;timings.append(before.elapsed_time(after))
                time.sleep(.02)  # Small duty-cycle gap; priority alone is not GPU isolation.
    stream.synchronize()
    write_new(output,{'passed':True,'cases_checked':checked,'calls':calls,'wall_seconds':time.monotonic()-started,
                     'per_call_cuda_ms':timings,'cpu_fallback_reasons':fallbacks,'stream_priority':stream.priority,
                     'allocator_limit_mib':options.memory_mib,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                     'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'gpu_before':free_before,'gpu_after':gpu_telemetry(),
                     'fixture_manifest':info(options.fixtures/'manifest.json'),'scope':'Concurrent contention parity probe; no uncontended speedup claim.'})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-cpu',action='store_true');mode.add_argument('--run-cuda',action='store_true')
    parser.add_argument('--suite',choices=('prepare','encoding','transfers','preparation-contracts','integration','benchmark','steady'),default='prepare')
    parser.add_argument('--fixtures',type=Path,default=VALIDATION/'fixtures')
    parser.add_argument('--output',type=Path,default=VALIDATION/'prepare-cuda.json')
    parser.add_argument('--seconds',type=int,default=60);parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--memory-mib',type=int,default=512);parser.add_argument('--observe-seconds',type=int,default=180)
    parser.add_argument('--_reference-worker',type=Path);parser.add_argument('--_cuda-worker',action='store_true')
    options=parser.parse_args(argv)
    require(1<=options.seconds<=300 and 1<=options.steps<=64 and 64<=options.memory_mib<=1024,'Probe bounds invalid')
    require(0<=options.observe_seconds<=300,'Observation bound invalid')
    require(options.suite!='steady' or options.steps>=8,'Steady benchmark needs at least eight timed calls')
    options.fixtures=contained(options.fixtures,VALIDATION);options.output=contained(options.output,VALIDATION)
    if options._reference_worker:reference_worker(contained(options._reference_worker,VALIDATION));return 0
    if options._cuda_worker:cuda_worker(options,options.output);return 0
    if not options.prepare_cpu and not options.run_cuda:
        print(json.dumps({'inactive':True,'fixtures_exist':options.fixtures.exists(),'gpu_compute':False,
                          'bounds':{'seconds':options.seconds,'calls':options.steps,'allocator_mib':options.memory_mib},
                          'note':'GPU context/compiler allocations are outside the allocator cap; low priority does not guarantee zero contention.'}));return 0
    VALIDATION.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_CACHE_DIR=str(VALIDATION/'triton-cache'))
    if options.prepare_cpu:
        require(not options.fixtures.exists(),'CPU fixtures already exist; preserve/reuse them')
        env['CUDA_VISIBLE_DEVICES']=''
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--_reference-worker',str(options.fixtures)],env=env,check=True,timeout=120)
        print(json.dumps({'prepared':True,'manifest':str(options.fixtures/'manifest.json')}));return 0
    require(not options.output.exists() and not options.output.with_suffix('.supervisor.json').exists(),'Probe report/supervisor already exists; do not duplicate/overwrite')
    if options.suite=='benchmark':require(not options.output.with_suffix('.timing.json').exists(),'Benchmark timing report already exists')
    before=trainer_snapshot();telemetry=gpu_telemetry();inputs_before=probe_inputs(options.suite)
    require(telemetry['free_mib']>=options.memory_mib+2048,'Insufficient GPU reserve; probe declined before CUDA child startup')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--_cuda-worker','--fixtures',str(options.fixtures),
             '--output',str(options.output),'--suite',options.suite,'--seconds',str(options.seconds),'--steps',str(options.steps),'--memory-mib',str(options.memory_mib)]
    # subprocess.run can terminate only this own probe child on timeout.
    try:
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=options.seconds+45)
    except subprocess.TimeoutExpired as error:
        def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
        result=types.SimpleNamespace(returncode=124,stdout=decoded(error.stdout),stderr=decoded(error.stderr)+'\nOwn probe timed out; subprocess.run terminated only its own child.')
    inputs_after=probe_inputs(options.suite)
    deadline=time.monotonic()+options.observe_seconds;after=trainer_snapshot();advanced=progress_ok(before,after)
    while not advanced and time.monotonic()<deadline:
        time.sleep(min(5,max(0,deadline-time.monotonic())));after=trainer_snapshot();advanced=progress_ok(before,after)
    supervisor={'command':command,'exit_code':result.returncode,'stdout_tail':result.stdout[-2000:],'stderr_tail':result.stderr[-4000:],
                'trainer_before':before,'trainer_after':after,'trainer_progress_confirmed':advanced,
                'trainer_process_unchanged':True,'probe_helper':info(Path(__file__).resolve()),
                'source_inputs':inputs_before,'source_inputs_unchanged':inputs_before==inputs_after,
                'note':'No trainer signal/control API is present. Concurrent timings include contention.'}
    write_new(options.output.with_suffix('.supervisor.json'),supervisor)
    require(inputs_before==inputs_after,'Probe source inputs changed during own child; inspect retained reports')
    require(result.returncode==0,'Own probe failed; inspect retained supervisor report')
    require(advanced,'Trainer remained live but no complete new PPO update was observed; inspect metrics_tail for planned selection/progress')
    print(json.dumps({'passed':True,'report':str(options.output),'trainer_progress_confirmed':True}));return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError,AssertionError) as error:
        print(json.dumps({'passed':False,'error':str(error)}),flush=True);raise SystemExit(2)
