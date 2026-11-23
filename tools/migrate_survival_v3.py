"""Explicit metadata-only 16-to-4 environment migration for a stopped V3 run."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile



def _record(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError(f'Expected an owned regular file: {path}')
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f'Source changed while hashing: {path}')
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': before.st_size}


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


@contextmanager
def _stopped_run(parent):
    import fcntl
    path = parent / '.runner.lock'
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != os.getuid():
        raise ValueError('Parent run requires its existing owned runner lock')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Parent trainer is still running') from error
        yield
    finally:
        os.close(descriptor)


def _episode(index, snapshot):
    return {'index': index, 'episode': snapshot['number'], 'seed': snapshot['seed'],
            'pieces': snapshot['summary']['pieces'], 'board_sha256': snapshot['board_sha256']}


def _checkpoint(path, active_envs):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    extra = payload.get('extra', {})
    signature = extra.get('signature', {})
    if payload.get('config', {}).get('architecture') != 'survival-feature-v3' or extra.get('algorithm') != 'survival-ppo-v3':
        raise ValueError('Only a V3 survival checkpoint can be migrated')
    environments = extra.get('environments', [])
    if signature.get('envs') != 16 or len(environments) != 16:
        raise ValueError('Expected exactly sixteen source environments')
    if extra['next_episode'] <= max(snapshot['number'] for snapshot in environments):
        raise ValueError('Source next_episode would reuse an existing episode')
    signature_before = copy.deepcopy(signature)
    snapshots = list(environments)
    extra['signature'] = {**signature, 'envs': active_envs}
    extra['environments'] = environments[:active_envs]
    return payload, signature_before, snapshots


def migration_names(parent):
    names = {'last.pt', 'initialization.json', 'metrics.jsonl'}
    optional = {'critic-calibration.json', 'recommended-policy.json', 'summary.json', 'manifest.json',
                'best.pt', 'best.json', 'sustained-baseline.json'}
    names.update(path.name for path in parent.iterdir() if
                 path.name in optional or (path.suffix == '.json' and
                                           path.name.startswith(('selection-', 'sampled-selection-'))))
    return names


def migrate(parent_run: Path, destination_run: Path, active_envs=4):
    """Preserve learning state and park twelve games in the immutable parent.

    The parent must remain retained and stopped. The proof identifies its exact
    checkpoint hash and parked snapshot indices; no game is marked completed or
    truncated. The original eight-hour deadline belongs to the controller and
    must be carried forward separately by the caller.
    """
    import torch

    if active_envs != 4 or type(active_envs) is not int:
        raise ValueError('This migration supports only the explicit sixteen-to-four transition')
    parent = Path(parent_run).absolute()
    destination = Path(destination_run).absolute()
    if parent.resolve(strict=True) != parent or not parent.is_dir():
        raise ValueError('Parent run must be a canonical directory without symlinks')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Destination run already exists')
    if destination == parent or destination.is_relative_to(parent) or parent.is_relative_to(destination):
        raise ValueError('Parent and destination run directories must be independent')
    if destination.parent.resolve() != destination.parent:
        raise ValueError('Destination parent must not contain symlinks')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = None
    with _stopped_run(parent):
        names = migration_names(parent)
        parent_files = {name: _record(parent / name) for name in sorted(names)}
        payload, signature_before, environments = _checkpoint(parent / 'last.pt', active_envs)
        signature_after = payload['extra']['signature']
        best_info = baseline = None
        if ('best.json' in names) != ('best.pt' in names):
            raise ValueError('Best checkpoint and selection evidence must be retained together')
        if 'sustained-baseline.json' in names:
            baseline = json.loads((parent / 'sustained-baseline.json').read_text())
            if baseline.get('signature') != signature_before or not baseline.get('complete'):
                raise ValueError('Baseline requires completed selection evidence with the source signature')
        if 'best.pt' in names:
            best_payload, best_signature, _ = _checkpoint(parent / 'best.pt', active_envs)
            best_info = json.loads((parent / 'best.json').read_text())
            if (baseline is None or not best_info.get('complete') or best_signature != signature_before
                    or best_info.get('signature') != signature_before):
                raise ValueError('Best checkpoint requires matching completed selection evidence')
            if best_info.get('checkpoint_sha256') != parent_files['best.pt']['sha256']:
                raise ValueError('Best checkpoint hash does not match its selection evidence')
        stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-migration-', dir=destination.parent))
        try:
            for name in sorted(names - {'last.pt', 'best.pt'}):
                (stage / name).write_bytes((parent / name).read_bytes())
            torch.save(payload, stage / 'last.pt')
            if best_info is not None:
                torch.save(best_payload, stage / 'best.pt')
                best_info['signature'] = copy.deepcopy(signature_after)
                best_info['checkpoint_sha256'] = _record(stage / 'best.pt')['sha256']
                _json(stage / 'best.json', best_info)
            if baseline is not None:
                baseline['signature'] = copy.deepcopy(signature_after)
                _json(stage / 'sustained-baseline.json', baseline)
            if (stage / 'recommended-policy.json').exists():
                recommendation = json.loads((stage / 'recommended-policy.json').read_text())
                if recommendation.get('checkpoint'):
                    recommendation['checkpoint'] = str(destination / 'best.pt')
                _json(stage / 'recommended-policy.json', recommendation)
            if (stage / 'summary.json').exists():
                summary = json.loads((stage / 'summary.json').read_text())
                summary.update(reason='environment_count_migration', last_checkpoint=str(destination / 'last.pt'),
                               best_checkpoint=str(destination / 'best.pt') if best_info else None,
                               active_games=[{'episode': snapshot['number'], **snapshot['summary']}
                                             for snapshot in environments[:active_envs]],
                               active_environment_count=active_envs, parked_environment_count=12)
                _json(stage / 'summary.json', summary)
            event = {'event': 'environment_count_migration', 'utc': datetime.now(timezone.utc).isoformat(),
                     'from_envs': 16, 'to_envs': active_envs, 'parked_envs': 12,
                     'parent_checkpoint': parent_files['last.pt'],
                     'runner_updates': payload['progress']['runner_updates'],
                     'placements': payload['progress']['placements'], 'completed_episodes_added': 0,
                     'note': 'Twelve unfinished games remain recoverable in the retained parent checkpoint.'}
            with (stage / 'metrics.jsonl').open('ab') as stream:
                if parent_files['metrics.jsonl']['bytes'] and not (parent / 'metrics.jsonl').read_bytes().endswith(b'\n'):
                    stream.write(b'\n')
                stream.write((json.dumps(event, allow_nan=False) + '\n').encode())
            if (stage / 'manifest.json').exists():
                manifest = json.loads((stage / 'manifest.json').read_text())
                if (manifest.get('checkpoint_sha256') != parent_files['last.pt']['sha256']
                        or manifest.get('source_sha256') != signature_before):
                    raise ValueError('Manifest does not match latest checkpoint')
                manifest.update(checkpoint_sha256=_record(stage / 'last.pt')['sha256'],
                                source_sha256=copy.deepcopy(signature_after), metrics_bytes=(stage / 'metrics.jsonl').stat().st_size)
                _json(stage / 'manifest.json', manifest)
            if (stage / 'initialization.json').read_bytes() != (parent / 'initialization.json').read_bytes():
                raise AssertionError('Fresh-initialization proof changed')
            for name, record in parent_files.items():
                if _record(parent / name) != record:
                    raise ValueError(f'Parent input changed during migration: {name}')
            destination_files = {name: {**_record(stage / name), 'path': str(destination / name)} for name in sorted(names)}
            proof = {'schema': 'tetris4d-survival-v3-environment-migration-v1',
                     'parent_run': str(parent), 'destination_run': str(destination),
                     'parent_files': parent_files, 'destination_files': destination_files,
                     'signature_before': signature_before, 'signature_after': copy.deepcopy(signature_after),
                     'active_environments': [_episode(i, snapshot) for i, snapshot in enumerate(environments[:active_envs])],
                     'parked_environments': {'checkpoint': parent_files['last.pt'], 'indices': list(range(active_envs, 16)),
                                             'episodes': [_episode(i, environments[i]) for i in range(active_envs, 16)]},
                     'preserved': {name: True for name in ('model', 'optimizer', 'scaler', 'RNG', 'counters', 'calibration',
                                                         'next_episode', 'last_evaluated_update', 'initialization_proof')},
                     'source_lock_verified': True, 'parent_must_remain_retained_and_immutable': True,
                     'budget': 'Caller must preserve the original start and deadline; this migration grants no extra time.'}
            _json(stage / 'migration.json', proof)
            for path in stage.iterdir():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            if destination.exists():
                raise ValueError('Destination appeared during migration')
            os.rename(stage, destination)
            stage = None
            return proof
        finally:
            if stage is not None:
                for path in stage.iterdir():
                    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                        raise ValueError('Unexpected staging entry; preserving it for inspection')
                    path.unlink()
                stage.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--destination-run', type=Path, required=True)
    parser.add_argument('--active-envs', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(migrate(args.parent_run, args.destination_run, args.active_envs), indent=2))


if __name__ == '__main__':
    main()
