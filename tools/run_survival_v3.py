"""Plan or supervise one fresh, bounded eight-hour survival-policy v3 run.

Without --run this prints a read-only plan. Training never resumes or restarts
automatically. The fixed heuristic prior is inherited; neural weights are not.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

SOURCE = Path(__file__).resolve().parents[1]
MIB = 2**20
HOURS = 8
RUNTIME = ('train_survival_v3.py', 'tetris_survival_v3.py', 'train_survival_v2.py',
           'train_overnight.py', 'train_survival.py', 'tetris_neural.py',
           'sparse_tetris_4d_optimized.py', 'tetris_gpu_simulation.py',
           'tetris_gpu_encoding.py', 'tools/run_survival_v3.py',
           'native/tetris_native.c', 'native/libtetris4d_native.so',
           'native/libtetris4d_native.so.json')
FINAL_SEEDS = {960001, 960003, 960005, 960007}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(epoch=None):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time() if epoch is None else epoch))


def read(path):
    return json.loads(Path(path).read_text())


def proof(path):
    path = Path(path)
    require(path == path.resolve() and not path.is_symlink() and path.is_file(), 'Unsafe input: ' + str(path))
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Changing input: ' + str(path))
    return {'path': str(path), 'sha256': digest, 'bytes': after.st_size}


def write(path, value, exclusive=False):
    path = Path(path)
    require(not path.is_symlink(), 'Symlink output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.survival-v3-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def tree_bytes(root, exclude=()):
    """Count owned ordinary files without following links or crossing mounts."""
    root = Path(root)
    require(root == root.resolve() and not root.is_symlink(), 'Storage root escape: ' + str(root))
    if not root.exists():
        return 0
    require(root.is_dir() and root.stat().st_uid == os.getuid() and not root.is_mount(), 'Unsafe storage root')
    device, total = root.stat().st_dev, 0
    excluded = set(map(Path, exclude))
    def inaccessible(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=inaccessible):
        for name in dirs + files:
            path = Path(directory)/name
            try:
                state = path.stat()
                require(not path.is_symlink() and state.st_dev == device and state.st_uid == os.getuid(),
                        'Storage ownership/symlink/mount escape: ' + str(path))
                if path in excluded:
                    require(path.is_dir(), 'Only source directories can be excluded')
                    dirs.remove(name)
                elif name in files:
                    require(path.is_file(), 'Nonregular storage file: ' + str(path))
                    total += state.st_size
            except FileNotFoundError:
                pass  # Trainer checkpoints are replaced atomically.
    return total


def storage(config):
    output, source, cache = (Path(config[k]) for k in ('output', 'source', 'cache'))
    require(source == output/'source', 'Only the source worktree is excluded from the output bound')
    owned = tree_bytes(output, exclude=(source,))
    cache_bytes = tree_bytes(cache)
    require(cache_bytes < config['cache_limit_bytes'], 'Private Triton cache bound reached')
    largest = 0
    run = output/'run'
    if run.exists():
        for path in run.rglob('*.pt'):
            try:
                if path.is_file() and not path.is_symlink():
                    largest = max(largest, path.stat().st_size)
            except FileNotFoundError:
                pass
    reserve = 2*largest + 8*MIB
    require(owned + reserve < config['output_limit_bytes'], 'Output cap including atomic checkpoint reserve reached')
    require(os.path.ismount('/mnt/c'), 'Windows backing volume is unavailable')
    disks = {}
    future = reserve + config['cache_limit_bytes']
    for name in ('/', '/mnt/c'):
        usage = shutil.disk_usage(name)
        require(usage.used + future < .9*usage.total and usage.free > future,
                'Insufficient disk headroom: ' + name)
        disks[name] = {'free_bytes': usage.free, 'total_bytes': usage.total,
                       'used_fraction': usage.used/usage.total}
    return {'owned_output_bytes': owned, 'cache_bytes': cache_bytes,
            'checkpoint_reserve_bytes': reserve, 'output_limit_bytes': config['output_limit_bytes'],
            'cache_limit_bytes': config['cache_limit_bytes'], 'disks': disks}


def frozen(config, configuration_proof=None):
    if configuration_proof is not None:
        require(proof(configuration_proof['path']) == configuration_proof, 'Frozen controller configuration changed')
    current = subprocess.check_output(['git', '-C', config['source'], 'rev-parse', 'HEAD'], text=True).strip()
    require(current == config['commit'], 'Frozen source commit changed')
    for item in config['files']:
        require(proof(item['path']) == item, 'Frozen input changed: ' + item['path'])


def configuration(args):
    source, output, cache = (Path(getattr(args, key)).absolute() for key in ('source', 'output', 'cache'))
    require(all(p == p.resolve() and not p.is_symlink() for p in (source, output, cache)), 'Noncanonical source/output/cache')
    require(source == output/'source', 'Source must be the isolated output/source worktree')
    require(cache.is_relative_to(Path('/tmp')) and cache != Path('/tmp'), 'Private cache must be under /tmp')
    require(not output.is_relative_to(cache) and not cache.is_relative_to(output), 'Cache and output overlap')
    require(args.seed >= 0 and args.seed % 2 == 0 and args.calibration_seed % 2 == 0
            and args.calibration_seed >= 2*args.seed + 200000, 'Invalid separate training/calibration seed ranges')
    require(len(set(args.selection_seeds)) == len(args.selection_seeds)
            and all(s > 0 and s % 2 for s in args.selection_seeds)
            and not FINAL_SEEDS.intersection(args.selection_seeds), 'Invalid or reserved development seeds')
    branch = subprocess.check_output(['git', '-C', str(source), 'branch', '--show-current'], text=True).strip()
    require(branch not in ('', 'main', 'master'), 'Training source must be an isolated branch')
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', *RUNTIME], text=True)
    require(not dirty.strip(), 'Commit the reviewed runtime before preparing the run')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    files = [proof(source/name) for name in RUNTIME]
    weights = proof(args.weights.resolve())
    files += [weights, proof(args.python.resolve())]
    return {'schema': 'tetris4d-survival-v3-controller-v1', 'source': str(source), 'output': str(output),
            'commit': commit, 'branch': branch, 'files': files, 'python': str(args.python.absolute()),
            'weights': weights['path'], 'fixed_heuristic_prior': weights,
            'initialization': 'all trainable neural weights initialized afresh; no checkpoint or optimizer transfer',
            'hours': HOURS, 'start_epoch': None, 'deadline_epoch': None,
            'seed': args.seed, 'calibration_seed': args.calibration_seed, 'selection_seeds': args.selection_seeds,
            'envs': 16, 'rollout': 1024, 'gpu_budget_bytes': 4096*MIB,
            'gpu_free_headroom_bytes': 512*MIB, 'cache': str(cache), 'cache_limit_bytes': 64*MIB,
            'output_limit_bytes': 256*MIB,
            'protocol': 'One fresh eight-hour run including calibration and selection; no automatic resume, restart or promotion. Fixed heuristic prior remains an explicit hybrid component.'}


def plan_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def training_command(config, hours):
    require(0 < hours <= HOURS, 'Invalid remaining training budget')
    arguments = ['--hours', format(hours, '.12g'), '--run-dir', str(Path(config['output'])/'run'),
        '--weights', config['weights'], '--device', 'cuda', '--backend', 'native', '--threads', '4',
        '--precision', 'fp32', '--envs', '16', '--rollout', '1024', '--batch', '32', '--optimizer-batch', '256',
        '--lr', '5e-5', '--critic-lr', '3e-4', '--ppo-epochs', '2', '--clip-ratio', '.1', '--target-kl', '.01',
        '--entropy', '.001', '--gae-lambda', '.999', '--imitation', '0', '--secondary-weight', '0',
        '--regret-weight', '0', '--warmup-updates', '0', '--seed', str(config['seed']),
        '--calibration-seed', str(config['calibration_seed']), '--calibration-games', '4',
        '--calibration-validation-games', '2', '--calibration-steps', '600', '--calibration-stride', '32',
        '--calibration-ratio', '.95', '--selection-seeds', *map(str, config['selection_seeds']),
        '--selection-pieces', '32000', '--pieces', '32000', '--eval-minutes', '60', '--save-seconds', '60',
        '--save-every', '10', '--record-every', '4', '--record-max-mib', '16', '--cache-rollout',
        '--rollout-cache-mib', '2048', '--gpu-prepare', '--fused-encoding', '--coalesce-transfers',
        '--recover-target', '--policy-mode', 'teacher-prior', '--teacher-prior-scale', '.125',
        '--residual-scale', '1', '--gradient-clip-scope', 'separate']
    return [config['python'], '-B', '-u', str(Path(config['source'])/'tools/run_survival_v3.py'),
            '--worker', '--gpu-budget-mib', '4096', '--gpu-headroom-mib', '512', '--', *arguments]


def gpu_status(config, starting=False):
    executable = shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
    text = subprocess.check_output([executable, '--query-gpu=memory.free,memory.total',
                                    '--format=csv,noheader,nounits', '--id=0'], text=True, timeout=15)
    free, total = (int(value.strip())*MIB for value in text.strip().split(','))
    required = config['gpu_free_headroom_bytes'] + (config['gpu_budget_bytes'] if starting else 0)
    require(free >= required, 'Insufficient GPU free memory headroom')
    return {'free_bytes': free, 'total_bytes': total, 'required_free_bytes': required}


def stats(config):
    root = Path(config['output'])/'run'
    result = {}
    summary = root/'summary.json'
    if summary.is_file() and summary.stat().st_size <= MIB:
        try:
            value = read(summary)
            keys = ('reason', 'runner_updates', 'placements', 'optimizer_steps', 'episodes',
                    'best_validation_score', 'elapsed_training_seconds', 'recording_total_mib')
            result.update({k: value[k] for k in keys if k in value})
            games = value.get('active_games', [])[:16]
            result['active_games'] = [{k: row[k] for k in ('episode', 'pieces', 'max_height', 'holes') if k in row}
                                      for row in games]
            result['current_max_survived_pieces'] = max((row.get('pieces', 0) for row in games), default=0)
        except (OSError, ValueError, TypeError):
            pass
    metrics = root/'metrics.jsonl'
    if metrics.is_file():
        with metrics.open('rb') as stream:
            size = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, size - 131072))
            lines = stream.read().decode('utf-8', errors='replace').splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
                result['latest_event'] = {k: v for k, v in row.items() if v is None or isinstance(v, (str, int, float, bool))}
                break
            except ValueError:
                continue
    return result


def emit(config, event, **values):
    row = {'event': event, 'utc': utc(), **values}
    text = json.dumps(row, allow_nan=False)
    path = Path(config['output'])/'control/events.jsonl'
    require(path == path.resolve() and not path.is_symlink(), 'Unsafe controller event path')
    with path.open('a') as stream:
        stream.write(text + '\n')
        stream.flush()
    print(text, flush=True)


def process_start(pid):
    text = Path('/proc')/str(pid)/'stat'
    return text.read_text().rsplit(')', 1)[1].split()[19]


def initialization_proof(path, config):
    item = proof(path)
    value = read(path)
    require(value.get('schema') == 'tetris4d-fresh-initialization-v1'
            and value.get('fresh_random_trainable_weights') is True
            and value.get('inherited_checkpoint') is None and value.get('inherited_optimizer') is False
            and value.get('seed') == config['seed']
            and value.get('teacher_weights_sha256') == config['fixed_heuristic_prior']['sha256'],
            'Initialization metadata does not prove fresh neural weights and the reviewed fixed prior')
    require(proof(path) == item, 'Initialization metadata changed while reading')
    return item


@contextmanager
def stop_requests():
    requested, previous = [], {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, lambda signum, frame: requested.append(signum))
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _supervise(config, state, requests, start_monotonic, child):
    output = Path(config['output'])
    frozen(config, state['configuration'])
    state['storage'] = storage(config)
    state['gpu'] = gpu_status(config, starting=True)
    remaining = min(config['deadline_epoch']-time.time(), HOURS*3600-(time.monotonic()-start_monotonic))
    require(remaining > 0 and not requests, 'No remaining eight-hour training budget')
    command = training_command(config, remaining/3600)
    state['command'] = command
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', TRITON_CACHE_DIR=config['cache'])
    with (output/'logs/trainer.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=config['source'], env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        child['process'] = process
        state.update(child_pid=process.pid, child_start_ticks=process_start(process.pid))
        write(output/'control/status.json', state)
        sent, failure, next_check, next_report = None, None, 0., 0.
        initialization = None
        while process.poll() is None:
            now, monotonic = time.time(), time.monotonic()
            if monotonic >= next_check:
                try:
                    frozen(config, state['configuration'])
                    state['storage'] = storage(config)
                    state['gpu'] = gpu_status(config)
                    initial = output/'run/initialization.json'
                    if initial.exists():
                        current = initialization_proof(initial, config)
                        require(initialization is None or current == initialization, 'Fresh initialization metadata changed')
                        initialization = current
                        state['initialization'] = current
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                    failure = str(error)
                state.update(updated_utc=utc(now), elapsed_seconds=monotonic-start_monotonic,
                             remaining_seconds=max(0., min(config['deadline_epoch']-now,
                                                         HOURS*3600-(monotonic-start_monotonic))), stats=stats(config))
                write(output/'control/status.json', state)
                next_check = monotonic + 30
            if monotonic >= next_report:
                emit(config, 'training_stats', elapsed_seconds=state.get('elapsed_seconds', 0.),
                     remaining_seconds=state.get('remaining_seconds'), stats=state.get('stats', {}))
                next_report = monotonic + 300
            reason = failure or ('signal_'+str(requests[0]) if requests else None)
            if reason is None and (now >= config['deadline_epoch'] or monotonic-start_monotonic >= HOURS*3600):
                reason = 'eight_hour_budget'
            if reason and sent is None:
                require(process_start(process.pid) == state['child_start_ticks'], 'Child process identity changed')
                state.update(state='stopping', stop_reason=reason, stop_requested_utc=utc(now))
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                sent = monotonic
                emit(config, 'stop_requested', reason=reason, signal='SIGINT')
                write(output/'control/status.json', state)
            if sent is not None and monotonic-sent >= 600:
                state.update(state='blocked', error='Child exceeded 600-second graceful checkpoint allowance; no forced kill',
                             child_still_running=True)
                write(output/'control/status.json', state)
                return 2
            time.sleep(2)
        state.update(exit_code=process.returncode, finished_utc=utc(), child_still_running=False, stats=stats(config))
    if failure:
        state.update(state='failed', error=failure)
    elif process.returncode != 0:
        state.update(state='failed', error='Trainer exited unsuccessfully')
    elif initialization is None:
        state.update(state='failed', error='Trainer exited without fresh initialization proof')
    elif requests:
        state['state'] = 'stopped'
    elif (state.get('stop_reason') != 'eight_hour_budget' and time.time() < config['deadline_epoch']
          and time.monotonic()-start_monotonic < HOURS*3600):
        state.update(state='failed', error='Trainer exited before the eight-hour budget without an authorized stop')
    else:
        state['state'] = 'finished'
    try:
        frozen(config, state['configuration'])
        state['storage'] = storage(config)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        state.update(state='failed', error=str(error))
    emit(config, 'controller_finished', state=state['state'], exit_code=process.returncode, stats=state['stats'])
    write(output/'control/status.json', state)
    return 0 if state['state'] in ('finished', 'stopped') else 2


def supervise(config, state, requests, start_monotonic):
    """An unexpected reporting failure must still request a graceful checkpoint."""
    child = {}
    try:
        return _supervise(config, state, requests, start_monotonic, child)
    except BaseException as error:
        process = child.get('process')
        if process is None:
            raise
        state.update(state='failed', error=str(error))
        if process.poll() is None:
            if not state.get('stop_requested_utc'):
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                state.update(stop_requested_utc=utc(), stop_reason='controller_error')
            until = time.monotonic() + 600
            while process.poll() is None and time.monotonic() < until:
                time.sleep(2)
            if process.poll() is None:
                state.update(state='blocked', child_still_running=True,
                             error=str(error)+'; child exceeded graceful allowance, no forced kill')
            else:
                state.update(child_still_running=False, exit_code=process.returncode)
        try:
            write(Path(config['output'])/'control/status.json', state)
        except OSError:
            print('Controller could not persist failure status: ' + str(error), file=sys.stderr, flush=True)
        return 2


def run(config, start_epoch=None, start_monotonic=None):
    start_epoch = time.time() if start_epoch is None else start_epoch
    start_monotonic = time.monotonic() if start_monotonic is None else start_monotonic
    output = Path(config['output'])
    control = output/'control'
    require(control == control.resolve() and not control.is_symlink(), 'Unsafe controller directory')
    for name in ('run', 'logs', 'dashboard'):
        path = output/name
        require(path == path.resolve() and not path.is_symlink(), 'Unsafe runtime directory: ' + str(path))
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control/'controller.lock'
    require(lock_path == lock_path.resolve() and not lock_path.is_symlink(), 'Unsafe controller lock path')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any((control/name).exists() for name in ('configuration.json', 'status.json', 'budget-ledger.json')),
                'Run was already prepared/started; automatic resume is forbidden')
        require(not (output/'run').exists() or not any((output/'run').iterdir()), 'Fresh run directory contains existing data')
        frozen(config)
        storage(config)
        cache = Path(config['cache'])
        require(not cache.exists(), 'Private cache already exists; use a new empty path')
        cache.mkdir(mode=0o700)
        (output/'logs').mkdir(exist_ok=True)
        (output/'dashboard').mkdir(exist_ok=True)
        config = {**config, 'files': list(config['files']), 'start_epoch': start_epoch,
                  'deadline_epoch': start_epoch+HOURS*3600}
        write(control/'configuration.json', config, exclusive=True)
        ledger = {'schema': 'tetris4d-survival-v3-budget-v1', 'start_epoch': start_epoch,
                  'deadline_epoch': config['deadline_epoch'], 'hours': HOURS,
                  'budget_scope': 'fresh run, including calibration and selection; supersedes previous run budgets'}
        write(control/'budget-ledger.json', ledger, exclusive=True)
        state = {'schema': config['schema'], 'state': 'running', 'started_utc': utc(start_epoch),
                 'start_epoch': start_epoch, 'deadline_epoch': config['deadline_epoch'],
                 'configuration': proof(control/'configuration.json'), 'budget': proof(control/'budget-ledger.json'),
                 'controller_pid': os.getpid(), 'controller_start_ticks': process_start(os.getpid()),
                 'fresh_run': True, 'automatically_restarted': False, 'protocol': config['protocol']}
        # Also pin the new ledger; it must never be rewritten to extend this run.
        config['files'].append(state['budget'])
        write(control/'configuration.json', config)
        state['configuration'] = proof(control/'configuration.json')
        write(control/'status.json', state)
        emit(config, 'controller_started', deadline_epoch=config['deadline_epoch'], commit=config['commit'])
        try:
            with stop_requests() as requests:
                return supervise(config, state, requests, start_monotonic)
        except BaseException as error:
            state.update(state='failed', error=str(error), updated_utc=utc())
            write(control/'status.json', state)
            raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run', action='store_true', help='Explicitly start one new eight-hour run')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=SOURCE.parent)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--weights', type=Path, default=SOURCE/'models/trained-sigmoid.json')
    parser.add_argument('--cache', type=Path, default=Path('/tmp')/f'tetris4d-survival-v3-training-{os.getuid()}')
    parser.add_argument('--seed', type=int, default=650000)
    parser.add_argument('--calibration-seed', type=int, default=5500000)
    parser.add_argument('--selection-seeds', type=int, nargs='+', default=[980101, 980103])
    parser.add_argument('--plan-sha256', help='Require the exact previously reviewed stable configuration digest')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    start_epoch, start_monotonic = time.time(), time.monotonic()
    config = configuration(args)
    digest = plan_hash(config)
    require(args.plan_sha256 is None or args.plan_sha256 == digest, 'Reviewed plan digest changed')
    if args.run:
        return run(config, start_epoch, start_monotonic)
    print(json.dumps({'configuration': config, 'plan_sha256': digest, 'storage': storage(config),
                      'command_preview': training_command(config, HOURS),
                      'note': 'Read-only plan; no budget started and no child launched.'}, indent=2))
    return 0


def cuda_budget(torch, budget_bytes, headroom_bytes):
    require(torch.cuda.is_available(), 'Fresh training worker requires CUDA')
    total = torch.cuda.get_device_properties(0).total_memory
    require(0 < budget_bytes < total and headroom_bytes >= 0, 'Invalid CUDA allocator bound')
    torch.cuda.set_per_process_memory_fraction(budget_bytes/total, 0)
    free, observed_total = torch.cuda.mem_get_info(0)
    require(observed_total == total and free >= budget_bytes+headroom_bytes,
            'Insufficient free GPU memory for 4GiB allocator plus512MiB headroom')
    return {'gpu_budget_bytes': budget_bytes, 'gpu_free_headroom_bytes': headroom_bytes,
            'free_bytes_at_worker_start': free, 'total_bytes': total}


def worker(argv):
    import runpy
    import torch
    parser = argparse.ArgumentParser(description='Set CUDA allocator bound before importing the v3 trainer')
    parser.add_argument('--gpu-budget-mib', type=int, choices=(4096,), required=True)
    parser.add_argument('--gpu-headroom-mib', type=int, choices=(512,), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    report = cuda_budget(torch, args.gpu_budget_mib*MIB, args.gpu_headroom_mib*MIB)
    print(json.dumps({'event': 'survival_v3_worker_gpu_budget', **report}), flush=True)
    supplied = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    require(not any(item.split('=', 1)[0] == '--checkpoint' for item in supplied), 'Fresh v3 worker forbids checkpoint transfer')
    script = SOURCE/'train_survival_v3.py'
    sys.argv = [str(script), *supplied]
    sys.path.insert(0, str(SOURCE))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    try:
        if sys.argv[1:2] == ['--worker']:
            worker(sys.argv[2:])
        else:
            raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({'state': 'rejected', 'error': str(error)}), file=sys.stderr)
        raise SystemExit(2)
