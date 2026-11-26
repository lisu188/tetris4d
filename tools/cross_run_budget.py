"""Exclusive GPU reservation accounting for the 24-hour survival experiment."""

from contextlib import contextmanager
from pathlib import Path
import os
import subprocess
import sys
import time

from tools import evaluate_afterstate_td as ev

# Each phase may spend unused time from earlier phases. Later phases stay reserved.
PHASE_HOURS = dict(preparation=2, calibration=2, training=6, development=6, holdout=8)


def ledger(root):
    path = Path(root) / 'gpu-time.json'
    return ev.read_json(path) if path.exists() else []


def import_preparation(root, perf):
    """Debit separately locked GPU checks once, including interrupted work with an end receipt."""
    path = root / 'gpu-time.json'
    with ev.base.run_lock(root / 'meter'):
        rows = ledger(root)
        for prefix, label in [
            ('gpu-smoke', 'CUDA exact parity smoke'),
            ('gpu-gate', 'formal CUDA checks and benchmark'),
        ]:
            started = float((perf / (prefix + '.start')).read_text())
            ended = float((perf / (prefix + '.end')).read_text())
            if not 0 < started <= ended <= time.time() + 1.0:
                raise ValueError(f'Invalid preparation timing for {prefix}')
            found = [r for r in rows if r['phase'] == 'preparation' and r['label'] == label]
            if found:
                if len(found) != 1 or (found[0]['start_unix'], found[0]['end_unix']) != (started, ended):
                    raise ValueError(f'Conflicting preparation receipt {label}')
                continue
            if any(
                r['seconds'] is None
                or (r.get('start_unix', ended) < ended and r.get('end_unix', started) > started)
                for r in rows
            ):
                raise ValueError('GPU reservation overlap during preparation')
            rows.append(
                dict(
                    phase='preparation',
                    label=label,
                    start_unix=started,
                    end_unix=ended,
                    seconds=ended - started,
                    status='closed',
                )
            )
        ev.base.durable_json(path, rows)


def allowance(entries, phase):
    ceiling = 0
    for name, hours in PHASE_HOURS.items():
        ceiling += hours * 3600
        if phase == name:
            break
    else:
        raise ValueError(f'Unknown phase {phase}')
    return max(0.0, ceiling - sum(row['seconds'] for row in entries))


def charged(row):
    return row['allowance_seconds'] if row['seconds'] is None else row['seconds']


def settle_session(root, ticket, now, elapsed):
    """Account for time holding the CUDA lock in between phase reservations."""
    rows = ledger(root)
    metered = sum(charged(row) for row in rows[ticket['first_entry'] :])
    missing = max(0.0, elapsed - metered)
    if missing:
        rows.append(
            dict(
                phase='overhead',
                label='exclusive setup and transitions',
                seconds=missing,
                end_unix=now,
                status='closed',
            )
        )
        ev.base.durable_json(root / 'gpu-time.json', rows)
    ev.base.durable_json(root / 'gpu-session.json', dict(ticket, status='closed', end_unix=now))


@contextmanager
def exclusive_session(root, stop):
    """Account for the whole device-lock interval and enforce its absolute outer limit."""
    root.mkdir(parents=True, exist_ok=True)
    journal = root / 'gpu-session.json'
    if journal.exists():
        previous = ev.read_json(journal)
        if previous['status'] == 'active':
            # A killed process cannot report teardown. Charge through the current
            # wall time, capped by the original upper bound, and never double-count
            # the phase-level charge for the same seconds.
            elapsed = max(0.0, min(time.time() - previous['start_unix'], previous['remaining']))
            settle_session(root, previous, time.time(), elapsed)
    rows = ledger(root)
    remaining = max(0.0, sum(PHASE_HOURS.values()) * 3600 - sum(charged(r) for r in rows))
    if remaining <= 1.0:
        stop.reason = 'time_budget'
        yield False
        return
    began = time.monotonic()
    ticket = dict(status='active', start_unix=time.time(), first_entry=len(rows), remaining=remaining)
    ev.base.durable_json(journal, ticket)
    # The monitor is independent of the training interpreter and native/CUDA calls.
    # Closing stdin (including on parent death) cancels it. Kill only this job.
    monitor = subprocess.Popen(
        [
            sys.executable,
            '-IS',
            '-c',
            'import os,select,signal,sys,time\n'
            'parent=int(sys.argv[1]); timeout=max(0.,float(sys.argv[2])-time.monotonic())\n'
            'if not select.select([sys.stdin], [], [], timeout)[0] and os.getppid()==parent:\n'
            '    os.kill(parent,signal.SIGKILL)',
            str(os.getpid()),
            str(began + remaining - 0.5),
        ],
        stdin=subprocess.PIPE,
    )
    original = stop.deadline
    stop.deadline = min(original, began + max(0.0, remaining - 90.0))
    try:
        yield True
    finally:
        monitor.stdin.close()
        monitor.wait(timeout=3)
        settle_session(root, ticket, time.time(), time.monotonic() - began)
        stop.deadline = original


@contextmanager
def reservation(root, phase, label, stop):
    """Caller holds the device lock. Count a lost reservation at its full ceiling."""
    path = root / 'gpu-time.json'
    with ev.base.run_lock(root / 'meter'):
        entries = ledger(root)
        for row in entries:
            if row['seconds'] is None:
                row.update(seconds=row['allowance_seconds'], status='lost-reservation-upper-bound')
        available = allowance(entries, phase)
        original = stop.deadline
        began = time.monotonic()
        row = dict(
            phase=phase,
            label=label,
            start_unix=time.time(),
            allowance_seconds=available,
            seconds=None,
            status='active',
        )
        entries.append(row)
        ev.base.durable_json(path, entries)
        # Leave time for a safe stop, a checkpoint write, and device teardown.
        stop.deadline = min(original, began + max(available - 90.0, 0.0))
        try:
            yield
        finally:
            elapsed = time.monotonic() - began
            row.update(seconds=elapsed, end_unix=time.time(), status='closed')
            ev.base.durable_json(path, entries)
            stop.deadline = original


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Import the measured CUDA preparation time before running calibration.'
    )
    parser.add_argument('output', type=Path)
    parser.add_argument('performance_directory', type=Path)
    args = parser.parse_args()
    import_preparation(args.output.resolve(), args.performance_directory.resolve())
