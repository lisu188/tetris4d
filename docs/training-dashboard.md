# Live training dashboard

The dashboard reads an existing survival run. It displays calibration, PPO
progress, completed episodes, checkpoint selection and the original wall-clock
budget when a launcher ledger is available. It uses the Python standard library
and browser SVG; no PyTorch import, package installation or external service is
needed. Run the server on Linux or WSL.

Start it in a separate terminal while training continues:

```bash
python tools/training_dashboard.py \
  --run-dir runs/survival-ppo-v2 \
  --data-dir survival-dashboard.artifacts/history \
  --port 8765
```

Open <http://localhost:8765>. The server listens on loopback. Keep its terminal
open while viewing the run; stopping the dashboard leaves the trainer running.
On Windows with WSL localhost forwarding enabled, the same address opens in the
Windows browser.

The page refreshes every 15 seconds. Calibration games, stochastic training
episodes and frozen greedy selection games have different purposes and appear
separately. Missing telemetry is left empty. Pausing the page stops browser
refreshes; the collector continues recording history.

**Maximum survived pieces** shows the running record of completed PPO training
episodes as a step chart, plus a separate hollow marker for the latest reported
active game. The headline is the larger of those two observed values. Calibration,
greedy selection, baseline and explicitly teacher-assisted episodes do not enter
the completed record. Horizontal steps retain the known record; they do not
represent additional completed games. The record and episode count are calculated
before chart sampling and survive history restarts and the 10000-event view limit.
Environment cards reflect the reported count, including sixteen-environment runs.

An explicit optimization stop is labeled as a requested pause. The wall-clock
budget continues during that pause. A valid amended 24-hour ledger takes priority
over the original eight-hour ledger.

## Retain data for later graphs

`metrics.jsonl` in the training directory remains the original event log. The
dashboard backfills those events and keeps normalized timestamped history in
`--data-dir`. Restart with the same directories to continue collecting without
duplicating events. Leave that history with the run when preserving results.

For a migrated run, first stop the previous dashboard collector, then pass its
unchanged data directory with `--prior-data-dir`. For example:

```bash
python tools/training_dashboard.py --run-dir survival-training.artifacts/run-16env \
  --data-dir survival-training.artifacts/dashboard-16env \
  --prior-data-dir survival-training.artifacts/dashboard
```

The prior history is read only and pinned by its checksum in the new manifest.
The new directory stores only new events. Charts and downloads combine both
histories and exclude repeated event identities. Each input history is bounded
to 8 MiB, and the new collector retains its own separate 8 MiB output limit.

The page provides **CSV** and **JSONL** downloads at:

- `/api/history.csv`
- `/api/history.jsonl`

These exports include recorded timestamps, elapsed time and available metrics.
Blank values mean the metric was absent from that event, rather than zero.
Online episode survival reflects the changing stochastic training policy; use
completed frozen selection games to judge checkpoint strength.

The dashboard keeps its own history bounded to 8 MiB and reports collection
errors on the page. It does not modify checkpoints, calibration, trainer state
or the training deadline.
