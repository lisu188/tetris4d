"""Explicit four-environment continuation of a stopped V3 run; never reset its clock."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from tools import run_survival_v3 as runner
from tools.migrate_survival_v3 import migrate, migration_names

EXTRA_RUNTIME = ('tools/continue_survival_v3.py', 'tools/migrate_survival_v3.py')


def stopped(parent):
    status = runner.read(parent/'control/status.json')
    runner.require(status.get('state') in ('stopped', 'finished', 'failed')
                   and status.get('child_still_running') is False, 'Parent controller has not stopped safely')
    for prefix in ('controller', 'child'):
        pid, ticks = status.get(prefix+'_pid'), status.get(prefix+'_start_ticks')
        runner.require(type(pid) is int and pid > 0 and ticks is not None, 'Missing parent process identity')
        try:
            current = runner.process_start(pid)
        except FileNotFoundError:
            continue
        runner.require(current != str(ticks), 'Parent '+prefix+' process is still running')
    return status


def configuration(args):
    parent, output = args.parent_output.absolute(), args.output.absolute()
    runner.require(parent == parent.resolve() and output == output.resolve()
                   and not parent.is_symlink() and not output.is_symlink(), 'Unsafe continuation paths')
    runner.require(not parent.is_relative_to(output) and not output.is_relative_to(parent), 'Parent/output overlap')
    status = stopped(parent)
    original = runner.read(parent/'control/configuration.json')
    runner.frozen(original, status['configuration'])
    ledger = runner.read(parent/'control/budget-ledger.json')
    start, deadline = ledger['start_epoch'], ledger['deadline_epoch']
    runner.require(deadline == start + 8*3600 and start == original['start_epoch']
                   and deadline == original['deadline_epoch'], 'Original eight-hour budget differs')
    runner.require(start <= time.time() < deadline, 'Original training budget expired or has a future start')
    runner.require(original['envs'] == 16, 'This migration requires a sixteen-environment parent')
    options = SimpleNamespace(source=args.source, output=output, cache=args.cache, envs=4,
        python=args.python, weights=Path(original['weights']), seed=original['seed'],
        calibration_seed=original['calibration_seed'], selection_seeds=original['selection_seeds'])
    config = runner.configuration(options)
    cache = Path(config['cache'])
    runner.require(not cache.is_relative_to(parent) and not parent.is_relative_to(cache), 'Parent/cache overlap')
    dirty = subprocess.check_output(['git', '-C', config['source'], 'status', '--porcelain', '--', *EXTRA_RUNTIME], text=True)
    runner.require(not dirty.strip(), 'Commit continuation/migration tools before planning')
    inputs = [parent/'control/configuration.json', parent/'control/status.json', parent/'control/budget-ledger.json']
    inputs += [parent/'run'/name for name in sorted(migration_names(parent/'run'))]
    config['files'] += [runner.proof(Path(config['source'])/name) for name in EXTRA_RUNTIME]
    config['files'] += [runner.proof(path) for path in inputs]
    config.update(schema='tetris4d-survival-v3-four-environment-continuation-v1',
        start_epoch=start, deadline_epoch=deadline, parent_output=str(parent),
        runtime_configuration=original, runtime_configuration_proof=status['configuration'],
        initialization='Continue the same model, Adam, RNG and calibration; no new neural initialization',
        protocol='Explicit continuation with four active environments; twelve remain recoverable in the immutable parent checkpoint. Original eight-hour deadline is unchanged; no automatic restart.')
    return config


def run(config):
    output, parent = Path(config['output']), Path(config['parent_output'])
    runner.require(time.time() < config['deadline_epoch'], 'Original training budget expired')
    stopped(parent)
    runner.frozen(config)
    retained = {Path(item['path']).name: item['bytes'] for item in config['files']
                if Path(item['path']).parent == parent/'run'}
    runner.require(retained.keys() == migration_names(parent/'run'), 'Parent migration inventory changed')
    runner.storage(config, additional_bytes=sum(retained.values()),
                   checkpoint_bytes=max(retained['last.pt'], retained.get('best.pt', 0)))
    control = output/'control'
    runner.require(control == control.resolve() and not control.is_symlink(), 'Unsafe continuation control path')
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control/'controller.lock'
    runner.require(lock_path == lock_path.resolve() and not lock_path.is_symlink(), 'Unsafe continuation lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.require(not any((control/name).exists() for name in ('configuration.json', 'status.json', 'budget-ledger.json'))
                       and not (output/'run').exists(), 'Continuation already exists; no automatic restart')
        cache = Path(config['cache'])
        runner.require(not cache.exists(), 'Continuation requires a new private cache')
        cache.mkdir(mode=0o700)
        for name in ('logs', 'dashboard'):
            path = output/name
            runner.require(path == path.resolve() and not path.is_symlink(), 'Unsafe continuation directory')
            path.mkdir(exist_ok=True)
        migration = migrate(parent/'run', output/'run')
        runner.frozen(config)
        source_ledger = parent/'control/budget-ledger.json'
        with (control/'budget-ledger.json').open('xb') as stream:
            stream.write(source_ledger.read_bytes()); stream.flush(); os.fsync(stream.fileno())
        copied = runner.proof(control/'budget-ledger.json')
        runner.require(copied['sha256'] == runner.proof(source_ledger)['sha256'], 'Original ledger copy differs')
        config = {**config, 'files': list(config['files']), 'migration': migration,
                  'migration_proof': runner.proof(output/'run/migration.json')}
        config['files'] += [copied, config['migration_proof'], runner.proof(output/'run/initialization.json')]
        runner.write(control/'configuration.json', config, exclusive=True)
        state = {'schema': config['schema'], 'state': 'running', 'fresh_run': False,
                 'continuation': True, 'active_environments': 4, 'parked_environments': 12,
                 'started_utc': runner.utc(config['start_epoch']), 'continuation_utc': runner.utc(),
                 'start_epoch': config['start_epoch'], 'deadline_epoch': config['deadline_epoch'],
                 'configuration': runner.proof(control/'configuration.json'), 'budget': copied,
                 'migration': config['migration_proof'], 'controller_pid': os.getpid(),
                 'controller_start_ticks': runner.process_start(os.getpid()),
                 'automatically_restarted': False, 'protocol': config['protocol']}
        runner.write(control/'status.json', state)
        runner.emit(config, 'continuation_started', active_environments=4, parked_environments=12,
                    deadline_epoch=config['deadline_epoch'], runtime_commit=config['runtime_configuration']['commit'])
        # The shared supervisor's monotonic budget also includes the elapsed parent run.
        start_monotonic = time.monotonic() - (time.time() - config['start_epoch'])
        try:
            with runner.stop_requests() as requests:
                return runner.supervise(config, state, requests, start_monotonic)
        except BaseException as error:
            state.update(state='failed', error=str(error), updated_utc=runner.utc())
            runner.write(control/'status.json', state)
            raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--parent-output', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=SOURCE.parent)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--cache', type=Path, default=Path('/tmp')/f'tetris4d-survival-v3-four-env-{os.getuid()}')
    parser.add_argument('--plan-sha256')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    config = configuration(args)
    digest = runner.plan_hash(config)
    runner.require(args.plan_sha256 is None or digest == args.plan_sha256, 'Reviewed continuation plan changed')
    if args.run:
        return run(config)
    print(json.dumps({'configuration': config, 'plan_sha256': digest,
        'remaining_hours': (config['deadline_epoch']-time.time())/3600,
        'command_preview': runner.training_command(config, (config['deadline_epoch']-time.time())/3600),
        'storage': runner.storage(config), 'note': 'Read-only; no migration, child or budget restart.'}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({'state': 'rejected', 'error': str(error)}), file=sys.stderr)
        raise SystemExit(2)
