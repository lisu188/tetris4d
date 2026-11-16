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

Policy signals distinguish PPO update KL from KL against the frozen initial
policy. The reference KL chart uses the measurement after the complete PPO
update, and its coefficient is displayed as β. Critic explained variance is
labeled as fit to **bootstrapped GAE targets**; this metric does not validate
actual remaining lifetime. These optional columns are also included in CSV
downloads. Existing history and provenance files remain unchanged on restart.

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

## Explore a neuron audit

Pass an existing audit report when starting the dashboard:

```bash
python tools/training_dashboard.py --run-dir runs/survival-ppo-v2 \
  --data-dir survival-dashboard.artifacts/history \
  --neuron-audit neuron-report.artifacts/audit/neurons.json
```

Open <http://localhost:8765/neurons>, or use the dashboard's **Neuron audit** link.
The explorer groups units by layer, colors them by a selected activation or
intervention measurement, and shows a selected unit's formula, activation range,
strongest sampled contexts and measured move/value effects. Filters and search
help inspect individual units. The checkpoint checksum and sampled-state count
identify the report being viewed.

The audit describes a frozen checkpoint over its sampled replay states. Colors
show measured values; they do not establish semantic detector labels or effects
on survival. Changing a selected move under an artificial midpoint intervention
does not demonstrate that the unit improves or harms full-game performance.

The server validates the report schema, recorded provenance fields,
group counts and finite activation/intervention measurements before listening.
It reads a regular file of at most 8 MiB once and serves the exact validated
snapshot from `/api/neuron-audit`. Replacing that file requires restarting the
dashboard to display the replacement. The report is independent of live training
and is never appended to training history. Without `--neuron-audit`, the explorer
shows that no audit is configured and the API returns an explicit JSON 404.
Requests cannot select a different local audit file.
