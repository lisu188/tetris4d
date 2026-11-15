# Explain a frozen policy

`tools/analyze_neurons.py` produces `neurons.md` and `neurons.json` for every learned
channel and dense unit of a frozen checkpoint. It uses the existing CPU PyTorch
environment and supplied completed game recordings; it does not play new games,
train weights, or use CUDA.

```sh
.venv/bin/python tools/analyze_neurons.py \
  --checkpoint /absolute/path/to/frozen-policy.pt \
  --checkpoint-sha256 THE_EXPECTED_SHA256 \
  --games /absolute/path/to/completed-game.json.gz \
  --output-dir /absolute/path/to/new-neuron-report \
  --backend native --threads 1 --limit 16
```

Use the frozen policy selected after training and its recorded SHA256. Keep the
training checkpoint and original recordings. The report directory must be new.
Recordings may come from another policy with the same piece catalog, engine source
and teacher parameters: the audit evaluates the supplied checkpoint on those
boards, and does not imply that the checkpoint visited them during play.

For width 32 with three blocks, the inventory has 562 entries: 32 stem channels,
192 block channels (two sets of 32 per block), 128 and 64 candidate hidden units,
one candidate score, 64 critic hidden units and one critic output, and 64 auxiliary
hidden units and 16 auxiliary outputs. Residual scales and individual weights are
parameters. A convolution channel applies shared weights at 4,096 spatial sites;
the report summarizes those activations and does not treat them as one spatial
neuron. The candidate ranker receives 242 inputs: 32 local features, 32 pooled
features, 82 context values and 96 candidate features. All input coordinate
definitions are included in JSON and in coordinate tables in the Markdown report.
These coordinates are defined inputs, rather than learned neurons.

Each entry reports activation range, mean, standard deviation, sigmoid saturation
and its three strongest sampled contexts. Contexts are ranked by mean activation,
not by playing quality. Convolution statistics include every spatial site; ranking
statistics include every valid candidate. Context IDs point to the recording,
seed, pre-decision placement number, piece, target and board hash.

The utility then clamps one sigmoid unit to 0.5 across all its spatial sites or
candidates. For a centered critic this becomes zero after centering. It clamps a
raw score or value output to zero, and an auxiliary output probability to 0.5.
These fixed midpoint interventions are reproducible; they are artificial inputs
to later layers. They are structural baselines, and do not establish typical
neutral activity or a natural missing-neuron state.

For every intervention the report measures candidate score changes, ranking
changes, whether the highest-scoring move changed, probability total variation,
and value/auxiliary prediction changes. The report states the actual policy mode,
numeric score formula, teacher prior (`-32 * deficit - rank / 8`) and greedy or
sampled selection rule. A common shift to all scores can change their values
without changing probabilities or ordering. Effects are averaged equally across
sampled states. The critic and auxiliary heads do not choose moves, so their own
unit interventions cannot directly change candidate scores.
Every unit also has a prose explanation of its measured choice/ranking effects
and whether its intervention increased or decreased the critic prediction. Its
exact computation and checkpoint tensor row references are included, so the
frozen coefficients can be inspected without duplicating all weights. GroupNorm
uses statistics across sites and channels within a group: a local convolution
kernel does not make the whole normalized operation strictly local.

Survival checkpoints label their value as expected remaining successful placements
divided by 1,000. For example, 12 estimates 12,000 further placements. Older clear
reward checkpoints have different units; the report reads checkpoint metadata.
The 16 completion outputs come from imitation training. Output `z` targets whether
the layer initially at height `z` clears within the dataset's auxiliary horizon
(collector default: 2,048 placements), tracking layer identity through compaction.
They do not measure current fill or next-move completion. Survival training does
not recalibrate this auxiliary head, and changing encoder features can make its
probabilities stale.

Sampling is deterministic and equally spaced over the supplied recordings in CLI
order. The default is 16 states, with a hard maximum of 64 and eight recordings.
The utility verifies every recorded move, final board/counters, termination flags,
piece catalog and immutable input hashes; it also checks all sampled candidate
drops and traced-forward agreement within relative tolerance 1e-6 and absolute
tolerance 1e-7. Each
recording is limited to 32 MiB compressed and decompressed, checkpoints to 64 MiB,
and combined reports to 8 MiB. Only one recording and one state's activation maps
are processed at a time; selected prepared states are retained without full
activation dumps. The CPU runtime grows with states, channels and candidate count.

A high activation or a large intervention effect is evidence for those sampled
inputs. It does not establish a global label such as “hole detector,” nor a causal
effect on survival. The report explains exact structural roles and observed
responses while leaving unproven semantic interpretations unnamed.
