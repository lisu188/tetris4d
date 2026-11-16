"""Run one bounded paired PPO pilot while the existing GPU run continues.

Each arm starts from the same protected actor with fresh optimizer/calibration.
This is a single training seed pilot; it never promotes or replaces a policy.
Importing or omitting --run never starts a child process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
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
FINAL_SEEDS = {960001, 960003, 960005, 960007}
RUNTIME = ('train_survival_v2.py', 'train_overnight.py', 'train_survival.py',
           'tetris_neural.py', 'sparse_tetris_4d_optimized.py', 'tetris_gpu_simulation.py',
           'tetris_gpu_encoding.py', 'tools/survival_experiments.py',
           'tools/run_reference_ablation.py', 'native/tetris_native.c',
           'native/libtetris4d_native.so', 'native/libtetris4d_native.so.json')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def proof(path):
    path = Path(path)
    require(path == path.resolve() and path.is_file() and not path.is_symlink(), 'Unsafe input: ' + str(path))
    before = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Changing input: ' + str(path))
    return {'path': str(path), 'sha256': digest, 'bytes': after.st_size}


def frozen(config, configuration_proof=None):
    if configuration_proof is not None:
        require(proof(configuration_proof['path']) == configuration_proof, 'Frozen pilot configuration changed')
    for item in config['files']:
        require(proof(item['path']) == item, 'Frozen input changed: ' + item['path'])


def write(path, value, exclusive=False):
    path = Path(path)
    require(not path.is_symlink(), 'Symlink output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.reference-pilot-', dir=path.parent)
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


def tree_bytes(root):
    root = Path(root)
    require(root == root.resolve() and not root.is_symlink(), 'Output root escape: ' + str(root))
    if not root.exists():
        return 0
    require(root.is_dir() and root.stat().st_uid == os.getuid() and not root.is_mount(), 'Unsafe output root')
    device, total = root.stat().st_dev, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            try:
                state = path.stat()
                require(not path.is_symlink() and state.st_dev == device and state.st_uid == os.getuid(),
                        'Output ownership/symlink/mount escape: ' + str(path))
                if name in files:
                    total += state.st_size
            except FileNotFoundError:
                pass  # Concurrent atomic checkpoint replacements are expected.
    return total


def storage(config, planned=0, pilot_training=True):
    roots = [Path(config[key]) for key in ('production_artifacts', 'output', 'cache')]
    sizes = {str(path): tree_bytes(path) for path in roots}
    require(sizes[config['cache']] < config['cache_limit_bytes'], 'Private pilot Triton cache limit reached')
    largest = 0
    for root in roots[:2]:
        for path in root.rglob('*.pt'):
            try:
                if path.is_file() and not path.is_symlink():
                    largest = max(largest, path.stat().st_size)
            except FileNotFoundError:
                pass
    # Production and pilot may save concurrently, each creating two transient copies.
    reserve = max(32*MIB, 2*(largest + 4*MIB)) * (2 if pilot_training else 1)
    total = sum(sizes.values())
    require(total + planned + reserve < config['global_limit_bytes'], 'Combined pilot/production storage reserve reached')
    require(os.path.ismount('/mnt/c'), 'Windows backing volume is unavailable')
    disks = {}
    for name in ('/', '/mnt/c'):
        usage = shutil.disk_usage(name)
        future = planned + reserve + config['cache_limit_bytes']
        require(usage.used + future < .9*usage.total and usage.free > future, 'Insufficient disk headroom: ' + name)
        disks[name] = {'free_bytes': usage.free, 'total_bytes': usage.total}
    return {'bytes_by_root': sizes, 'total_bytes': total, 'planned_bytes': planned,
            'checkpoint_reserve_bytes': reserve, 'limit_bytes': config['global_limit_bytes'], 'disks': disks}


def worker_command(config, script, arguments):
    return [config['python'], '-B', '-u', str(Path(config['source'])/'tools/run_reference_ablation.py'),
            '--worker', '--script', str(Path(config['source'])/script), '--gpu-budget-mib', '3072',
            '--gpu-headroom-mib', '512', '--', *arguments]


def arm_command(config, name, hours):
    command = ['--run-dir', str(Path(config['output'])/name), '--checkpoint', config['initializer'],
               '--weights', config['weights'], '--hours', format(hours, '.12g'),
               '--device', 'cuda', '--backend', 'native', '--threads', '4', '--precision', 'fp32',
               '--envs', '16', '--rollout', '512', '--batch', '16', '--optimizer-batch', '128',
               '--lr', '.0001', '--critic-lr', '.0003', '--entropy', '.001', '--gae-lambda', '.999',
               '--ppo-epochs', '3', '--clip-ratio', '.2', '--target-kl', '.02',
               '--policy-mode', 'teacher-prior', '--teacher-prior-scale', '.125', '--residual-scale', '1',
               '--preserve-initial-actor', '--gradient-clip-scope', 'separate', '--recover-target',
               '--imitation', '0', '--secondary-weight', '0', '--regret-weight', '0', '--warmup-updates', '0',
               '--seed', str(config['seed']), '--calibration-seed', str(config['calibration_seed']),
               '--calibration-games', '4', '--calibration-validation-games', '2', '--calibration-steps', '300',
               '--calibration-stride', '32', '--calibration-ratio', '.95', '--gradient-every', '10',
               '--selection-seeds', *map(str, config['selection_seeds']), '--pieces', '32000',
               '--selection-pieces', '32000', '--eval-minutes', '60', '--save-every', '10',
               '--save-seconds', '60', '--max-updates', str(config['updates']),
               '--cache-rollout', '--rollout-cache-mib', '1024', '--gpu-prepare', '--fused-encoding',
               '--coalesce-transfers', '--allow-concurrent']
    if name == 'anchored':
        command += ['--reference-checkpoint', config['initializer'], '--reference-kl-weight', str(config['beta'])]
    return worker_command(config, 'train_survival_v2.py', command)


def evaluation_command(config):
    output = Path(config['output'])
    return worker_command(config, 'tools/survival_experiments.py',
           ['compare', '--checkpoints', config['initializer'], str(output/'baseline/last.pt'),
            str(output/'anchored/last.pt'), '--seeds', *map(str, config['selection_seeds']),
            '--modes', 'greedy', 'sampled', '--action-seeds', str(config['action_seed']),
            '--pieces', '32000', '--piece-set', 'expanded', '--device', 'cuda', '--backend', 'native',
            '--threads', '4', '--weights', config['weights'], '--allow-concurrent', '--value-audit',
            '--value-audit-stride', '128', '--value-audit-samples', '256', '--output', str(output/'comparison.json')])


def configuration(args):
    source, production, output = (Path(getattr(args, key)).absolute() for key in ('source', 'production_artifacts', 'output'))
    require(all(p == p.resolve() and not p.is_symlink() for p in (source, production, output)), 'Noncanonical source/output')
    require(not output.is_relative_to(production) and not production.is_relative_to(output),
            'Pilot output must be separate from the active production artifacts')
    require(output.is_relative_to(production.parent), 'Pilot output must remain in the project')
    require(args.seed >= 0 and args.seed % 2 == 0 and args.calibration_seed >= 0 and args.calibration_seed % 2 == 0,
            'Training and calibration seeds must be nonnegative and even')
    require(len(args.selection_seeds) == len(set(args.selection_seeds)) and all(s > 0 and s % 2 for s in args.selection_seeds),
            'Development seeds must be unique positive odd integers')
    require(not FINAL_SEEDS.intersection(args.selection_seeds), 'Reserved final test seeds cannot tune the pilot')
    require(0 < args.updates <= 64 and 0 < args.beta and math.isfinite(args.beta), 'Invalid bounded pilot settings')
    require(0 < args.arm_hours <= 3 and math.isfinite(args.arm_hours), 'Each arm is bounded to at most three hours')
    ledger_path = args.budget_ledger.resolve()
    ledger = read(ledger_path)
    require(ledger['deadline_epoch'] == ledger['start_epoch'] + 86400, 'Pilot must retain the original total24h deadline')
    require(time.time() < ledger['deadline_epoch'], 'Original training deadline already passed')
    branch = subprocess.check_output(['git', '-C', str(source), 'branch', '--show-current'], text=True).strip()
    require(branch not in ('', 'main', 'master'), 'Pilot source must be an isolated branch')
    require(not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', *RUNTIME], text=True).strip(),
            'Commit the pilot runtime before launch')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    cache = args.cache.absolute()
    require(cache == cache.resolve() and cache.is_relative_to(Path('/tmp')) and not cache.is_symlink(), 'Cache must be private under /tmp')
    require(args.global_limit_mib == 384, 'Reviewed aggregate output bound is384MiB')
    files = [proof(source/name) for name in RUNTIME]
    files += [proof(ledger_path), proof(args.checkpoint.resolve()), proof(args.weights.resolve())]
    checkpoint = files[-2]
    files.append(proof(args.python.resolve()))
    return {'schema': 'tetris4d-reference-ablation-v1', 'source': str(source), 'commit': commit, 'branch': branch,
            'production_artifacts': str(production), 'output': str(output), 'python': str(args.python.absolute()),
            'weights': str(args.weights.resolve()), 'source_initializer': checkpoint,
            'initializer': str(output/'initializer.pt'), 'files': files,
            'start_epoch': ledger['start_epoch'], 'deadline_epoch': ledger['deadline_epoch'],
            'seed': args.seed, 'calibration_seed': args.calibration_seed, 'selection_seeds': args.selection_seeds,
            'action_seed': args.action_seed, 'updates': args.updates, 'placements_per_arm': args.updates*16*512,
            'beta': args.beta, 'arm_hours': args.arm_hours, 'evaluate': args.evaluate,
            'gpu_budget_bytes': 3072*MIB, 'gpu_free_headroom_bytes': 512*MIB,
            'cache': str(cache), 'cache_limit_bytes': 32*MIB, 'global_limit_bytes': 384*MIB,
            'storage_rationale': 'Agent operational cap expanded from256 to384MiB across both experiments; uncertain private preflight files remain protected. No replay recording. Original volume90percent guards remain.',
            'protocol': 'Baseline then anchored PPO from the same frozen actor, training seed and fresh Adam/calibration;16envs,512rollout,3epochs. Existing production GPU training continues; its degraded actor is not the paired baseline. One training seed is a pilot, not evidence of general improvement. No automatic promotion.'}


@contextmanager
def stop_requests():
    requested = []
    previous = {}
    for number in (signal.SIGINT, signal.SIGTERM):
        previous[number] = signal.signal(number, lambda signum, frame: requested.append(signum))
    try:
        yield requested
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def child(config, state, name, command, deadline, requests):
    output = Path(config['output'])
    frozen(config, state['configuration'])
    training = name in ('baseline', 'anchored')
    state['storage'] = storage(config, pilot_training=training)
    phase = {'state': 'running', 'started_utc': utc(), 'command': command, 'deadline_epoch': deadline}
    state['phases'][name] = phase
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', TRITON_CACHE_DIR=config['cache'])
    with (output/(name+'.log')).open('xb') as log:
        process = subprocess.Popen(command, cwd=config['source'], env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        phase['pid'] = process.pid
        write(output/'status.json', state)
        sent, failure, next_check = None, None, 0.
        while process.poll() is None:
            now = time.time()
            if now >= next_check:
                try:
                    frozen(config, state['configuration'])
                    state['storage'] = storage(config, pilot_training=training)
                except (OSError, ValueError) as error:
                    failure = str(error)
                next_check = now + 30
                phase['updated_utc'] = utc()
                write(output/'status.json', state)
            reason = failure or ('signal_'+str(requests[0]) if requests else None) or ('deadline' if now >= deadline else None)
            if reason and sent is None:
                phase['stop_reason'] = reason
                try:
                    process.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    pass
                sent = now
            if sent is not None and now-sent > 600:
                phase.update(state='blocked', error='Child exceeded graceful stop allowance; no forced kill', child_still_running=True)
                write(output/'status.json', state)
                raise ValueError(phase['error'])
            time.sleep(2)
        phase.update(state='complete' if process.returncode == 0 and sent is None else 'incomplete',
                     exit_code=process.returncode, finished_utc=utc())
        write(output/'status.json', state)
        require(phase['state'] == 'complete', name+' did not complete within its reviewed budget')


def validate_arm(config, name):
    root = Path(config['output'])/name
    summary, manifest = read(root/'summary.json'), read(root/'manifest.json')
    calibration = read(root/'critic-calibration.json')
    require(summary['reason'] == 'update_limit' and summary['runner_updates'] == config['updates']
            and summary['placements'] == config['placements_per_arm'], 'Pilot arm did not complete the exact placement budget')
    require(manifest['runner_updates'] == config['updates'] and manifest['calibration_stage'] == 'passed'
            and proof(root/'last.pt')['sha256'] == manifest['checkpoint_sha256'], 'Pilot checkpoint/calibration proof differs')
    require(calibration.get('passed') is True and calibration.get('actor_unchanged') is True,
            'Pilot actor changed during critic calibration')
    signature = manifest['source_sha256']
    require(signature['envs'] == 16 and signature['rollout'] == 512 and signature['seed'] == config['seed']
            and signature['calibration_seed'] == config['calibration_seed'], 'Pilot stream settings changed')
    require(signature.get('reference_kl_weight', 0.) == (config['beta'] if name == 'anchored' else 0.),
            'Pilot reference coefficient changed')
    if name == 'anchored':
        require(signature['reference_checkpoint_sha256'] == config['source_initializer']['sha256'], 'Reference actor changed')
    events = (json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines())
    first = next(row for row in events if row.get('event') == 'run_start')
    require(first['resumed'] is False and first['initial_checkpoint'] == config['initializer'], 'Arm reused a previous run')
    return {'checkpoint': proof(root/'last.pt'), 'manifest': proof(root/'manifest.json'),
            'summary': proof(root/'summary.json'), 'calibration': proof(root/'critic-calibration.json'),
            'initial_actor_sha256': calibration['actor_sha256'], 'placements': summary['placements'],
            'fresh_run': True, 'fresh_optimizer_and_critic_calibration': True}


def run(config):
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (output/'configuration.json').exists() and not (output/'status.json').exists(),
                'Pilot was already prepared/started; inspect it before any retry')
        require(not any((output/name).exists() for name in ('baseline', 'anchored', 'initializer.pt', 'comparison.json')),
                'Pilot output contains previous experiment data')
        frozen(config)
        storage(config)
        cache = Path(config['cache'])
        cache.mkdir(mode=0o700, parents=False, exist_ok=True)
        initializer = Path(config['initializer'])
        with initializer.open('xb') as target:
            target.write(Path(config['source_initializer']['path']).read_bytes())
            target.flush()
            os.fsync(target.fileno())
        require(proof(initializer)['sha256'] == config['source_initializer']['sha256'], 'Initializer snapshot changed')
        config['files'].append(proof(initializer))
        write(output/'configuration.json', config, exclusive=True)
        state = {'schema': config['schema'], 'state': 'running', 'started_utc': utc(),
                 'configuration': proof(output/'configuration.json'), 'phases': {}, 'protocol': config['protocol']}
        write(output/'status.json', state)
        try:
            with stop_requests() as requests:
                for name in ('baseline', 'anchored'):
                    remaining = config['deadline_epoch']-time.time()
                    require(remaining > 0 and not requests, 'Original budget or stop request prevents another arm')
                    hours = min(config['arm_hours'], remaining/3600)
                    child(config, state, name, arm_command(config, name, hours),
                          min(config['deadline_epoch'], time.time()+hours*3600), requests)
                    state['phases'][name]['proof'] = validate_arm(config, name)
                    write(output/'status.json', state)
                require(state['phases']['baseline']['proof']['initial_actor_sha256'] ==
                        state['phases']['anchored']['proof']['initial_actor_sha256'], 'Paired arms started from different actors')
                if config['evaluate']:
                    planned = sum((output/name).stat().st_size for name in ('initializer.pt', 'baseline/last.pt', 'anchored/last.pt')) + 8*MIB
                    storage(config, planned=planned, pilot_training=False)
                    child(config, state, 'evaluation', evaluation_command(config), config['deadline_epoch'], requests)
                    comparison = read(output/'comparison.json')
                    require(comparison['complete'] and len(comparison['runs']) == len(config['selection_seeds'])*8,
                            'Frozen comparison did not complete all paired games')
                    state['phases']['evaluation']['report'] = proof(output/'comparison.json')
            require(time.time() <= config['deadline_epoch'], 'Pilot completed after the original deadline')
            frozen(config, state['configuration'])
            state.update(state='complete', finished_utc=utc(), finished_epoch=time.time(), automatically_promoted=False,
                         limitation='One training seed and reused development seeds; confirmation on new seeds is still required.')
            write(output/'status.json', state)
            return 0
        except BaseException as error:
            state.update(state='failed', error=str(error), finished_utc=utc())
            write(output/'status.json', state)
            raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Explicitly launch; otherwise print a read-only plan')
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--production-artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-ledger', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--seed', type=int, default=350000)
    parser.add_argument('--calibration-seed', type=int, default=4300000)
    parser.add_argument('--selection-seeds', type=int, nargs='+', default=[970001, 970003, 970005, 970007])
    parser.add_argument('--action-seed', type=int, default=1701)
    parser.add_argument('--updates', type=int, default=64)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--arm-hours', type=float, default=3.)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--cache', type=Path, default=Path('/tmp/tetris4d-reference-ablation-cache'))
    parser.add_argument('--global-limit-mib', type=int, default=384)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    config = configuration(args)
    if args.run:
        return run(config)
    print(json.dumps({'configuration': config, 'storage': storage(config),
                      'commands': {name: arm_command(config, name, config['arm_hours']) for name in ('baseline', 'anchored')},
                      'evaluation_command': evaluation_command(config) if config['evaluate'] else None}, indent=2))
    return 0


def cuda_budget(torch, budget_bytes, headroom_bytes):
    require(torch.cuda.is_available(), 'Pilot worker requires CUDA')
    total = torch.cuda.get_device_properties(0).total_memory
    require(0 < budget_bytes < total and headroom_bytes >= 0, 'Invalid CUDA allocator budget')
    torch.cuda.set_per_process_memory_fraction(budget_bytes/total, 0)
    free, observed_total = torch.cuda.mem_get_info(0)
    require(observed_total == total and free >= budget_bytes+headroom_bytes,
            'Insufficient free GPU memory for3GiB pilot budget plus512MiB headroom')
    return {'gpu_budget_bytes': budget_bytes, 'gpu_free_headroom_bytes': headroom_bytes,
            'free_bytes_at_worker_start': free, 'total_bytes': total}


def worker(argv):
    import runpy
    import torch
    parser = argparse.ArgumentParser(description='Bound CUDA memory before importing the trainer/evaluator')
    parser.add_argument('--script', type=Path, required=True)
    parser.add_argument('--gpu-budget-mib', type=int, choices=(3072,), required=True)
    parser.add_argument('--gpu-headroom-mib', type=int, choices=(512,), required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    allowed = {SOURCE/'train_survival_v2.py', SOURCE/'tools/survival_experiments.py'}
    require(args.script == args.script.resolve() and args.script in allowed, 'Unapproved pilot worker script')
    report = cuda_budget(torch, args.gpu_budget_mib*MIB, args.gpu_headroom_mib*MIB)
    print(json.dumps({'event': 'pilot_worker_gpu_budget', **report}), flush=True)
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    sys.argv = [str(args.script), *arguments]
    # run_path on a file does not add its repository to Python's module path.
    sys.path.insert(0, str(SOURCE))
    runpy.run_path(str(args.script), run_name='__main__')


if __name__ == '__main__':
    try:
        if sys.argv[1:2] == ['--worker']:
            worker(sys.argv[2:])
        else:
            raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({'state': 'rejected', 'error': str(error)}), file=sys.stderr)
        raise SystemExit(2)
