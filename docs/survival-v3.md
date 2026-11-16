# Fresh survival PPO V3

V3 replaces the learned board encoder with fixed board and placement features.
The actor and critic have independent parameters, so actor updates cannot change
the critic's input representation. All 75,938 trainable parameters start afresh;
neither policy checkpoints nor optimizer state are inherited. The fixed candidate
heuristics and teacher logit prior are retained and recorded explicitly.

## Motivation and architecture

On development seed 970003, the protected policy survived 13,371 greedy placements
and the stopped production policy at update 465 survived 8,549. Sampled games
survived 9,600 and 8,597 respectively. Their sampled critic MAEs were 292 and
2,741 pieces. This is one seed, not a statistical ranking. The pilot's greedy
game reached 12,487; its sampled game was interrupted and has no final score.
High explained variance on bootstrapped targets did not imply accurate remaining
lifetime predictions. Logit spread grew substantially; measured sigmoid
saturation was low. V3 addresses representation interference and calibration;
changing the activation alone is not an established remedy.

The architecture discriminator is `survival-feature-v3`. Fixed 34-channel board
features are pooled with global mean and maximum, then joined to 82 context
features (150 state inputs). For each placement, 96 candidate features join 34
masked local means and 34 local maxima (164 candidate inputs).

The actor uses a 64-unit state projection, a 128-to-64 candidate MLP and a 64-unit
ranking layer, with LayerNorm and SiLU. Its small initialized output is added to
the recorded teacher prior. The independent critic is a 128-to-64 MLP followed
by a softplus output. Values represent remaining placements times 0.001. There
is no unused auxiliary completion head. The actor has 47,989 parameters and the
critic 27,949. Fixed pooling is a testable hypothesis, not a demonstrated win.

## Training and evaluation

`train_survival_v3.py` rejects external `--checkpoint` arguments and bypasses
legacy model auto-discovery. `initialization.json` records initial neural hashes,
parameter count, seed and heuristic provenance before calibration. Explicit
same-run resume remains supported by the trainer with source/config checks.

Critic calibration uses four complete training games and two disjoint validation
games. Its output starts at the training-only mean and the actor remains frozen
during calibration. Failure to validate stops the run. The calibrated update-zero
policy is evaluated and protected before PPO changes it. Each selection round
also runs frozen sampled-policy games with Monte Carlo critic errors. Censored
games do not receive fabricated remaining-lifetime targets. Samples within a game
are correlated. Protected-best selection uses separate greedy games.

The bounded runner uses 16 GPU environments, 1,024 steps per environment, two PPO
epochs, actor learning rate 5e-5, critic rate 3e-4, clipping 0.1, KL target 0.01,
entropy 0.001, gamma 1 and GAE lambda 0.999. Local KL limits do not bound cumulative
policy drift. The run retains periodic checkpoints and metrics, including frozen
policy checks, rather than treating training reward as proof of improvement.

## Launching the eight-hour experiment

Use an isolated committed source worktree at `<output>/source`. Build its native
backend with `python -B tools/build_native.py`. Then inspect the read-only plan:

```sh
python -B tools/run_survival_v3.py --python /path/to/venv/bin/python \
  --weights /path/to/models/trained-sigmoid.json
```

Add `--run --plan-sha256 <printed-plan-hash>` to execute that exact plan. The new
eight-hour clock starts at launch and includes calibration and evaluation. It
does not inherit an older run's deadline. The runner pins source, executable,
native backend and heuristic hashes; caps Torch allocation at 6 GiB; bounds
output/private caches; checks Linux and Windows disk headroom; and records status
every 30 seconds with progress reports every five minutes. It requests a graceful
checkpoint on deadline or guard failure and never automatically restarts.

Budget and launcher state are in `control/`, training data in `run/`, and the
continuous log in `logs/trainer.log`. Point the dashboard collector's control
directory at `control/`. Keep previous-run histories and audits separate.

## Compatibility and validation

Legacy checkpoints retain their original architecture by default. V3 supports
headless inference, batching and checkpoint loading. The legacy CNN activation
trace interface is explicitly unsupported; an older 562-unit neuron audit does
not describe V3. Loading the protected update-zero checkpoint through NeuralAgent
requires explicit `allow_untrained=True`, preserving the existing safeguard.

CPU tests cover masking, permutation and batching, independent gradients,
positive differentiable values, fresh initialization, calibration, PPO updates,
checkpoint/RNG resume, censored sampled diagnostics and bounded-run lifecycle.
CUDA parity, memory and throughput must be checked separately on the target GPU.

Relevant primary sources: [SiLU and Tetris](https://arxiv.org/abs/1702.03118),
[Phasic Policy Gradient](https://arxiv.org/abs/2009.04416),
[GAE](https://arxiv.org/abs/1506.02438), and
[PPO reference implementation](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py).
