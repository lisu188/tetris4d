# Survival policy experiment

> Historical workflow. Use [Spatial V4 PPO](ppo-workflow.md) to train. The [archived source revision](https://github.com/lisu188/tetris4d/blob/23159fe8d860a13ab16b8d5b69f2abf868522c82/docs/survival-longevity.md) contains the commands below.

This experiment trains for successful placements with the existing reward of
0.001 per placement, gamma 1, and the audited survival PPO trainer. Completed
selection games determine the best checkpoint. The sigmoid architecture still
has 62,485 parameters, and rigid placement, collisions and hyperlayer clearing
use the existing engine.

## Exploration and initialization

The optional `teacher-prior` mode combines a scaled teacher prior with the
linear actor output:

```text
logits = teacher_prior_scale * teacher_prior + residual_scale * actor_logits
```

The existing `teacher-residual` mode bounds its correction. The new mode can
learn to choose a legal move with lower immediate teacher utility when that
improves survival. Prior scale 0.125 is an experimental starting setting.

`--preserve-initial-actor` retains the imported actor tensors, including the
ranking head. Changing the policy mode or prior scale changes action
probabilities. Keep the original checkpoint frozen as the incumbent and
measure the converted policy separately. Existing policy modes and command
defaults retain their previous behavior.

`--gradient-clip-scope separate` clips actor and critic gradients independently.
Periodic diagnostics report both gradient norms, hidden-layer saturation and
candidate variation, raw-logit spread, lower-primary action probabilities and
actor hashes. These measure learning behavior; completed games measure strength.

## Eight-hour run

Use an immutable checkpoint chosen after the paired comparison. Keep older
experiments in their original directories. With the existing environment:

```bash
.venv/bin/python -u train_survival_v2.py \
  --hours 8 --run-dir runs/survival-longevity \
  --checkpoint /absolute/path/to/frozen-checkpoint.pt \
  --device cuda --backend native --threads 4 --precision fp32 \
  --policy-mode teacher-prior --teacher-prior-scale 0.125 \
  --residual-scale 1 --preserve-initial-actor \
  --gradient-clip-scope separate \
  --imitation 0 --secondary-weight 0 --regret-weight 0 \
  --warmup-updates 0 --rollout 512 \
  --seed 250000 --calibration-seed 3300000 \
  --selection-seeds 951001 951003 951005 951007 \
  --pieces 32000 --selection-pieces 32000 \
  --record-games --record-every 4 --record-max-mib 128
```

The budget includes critic calibration and selection. Each invocation grants
the requested time; after an interruption, resume with the time remaining from
the original deadline. A matching existing run resumes its checkpoint and
rejects changed algorithm settings or sources. Record the imported
`train_survival.py` hash along with the other source and input hashes.

The normal calibration gate stays enabled. A separate short mechanism smoke
may explicitly skip calibration in a disposable run; its weights must remain
separate from the calibrated experiment. Do not bypass a failed production
calibration gate.

## Evaluation

The comparison seeds `94001 94003 94005 94007` are used to compare the existing
generations. The new selection seeds above may choose checkpoints during
training. Reserve `960001 960003 960005 960007` for the later frozen-checkpoint
test against the incumbent, sustained and sustained-recovery.

Compare greedy gameplay and sampled training behavior separately. Average
sampled repetitions within each game seed. Report capped survival, paired
differences, stop reasons, clears and recovery counts; a piece-limit survivor
provides a lower bound. Preserve the best measured incumbent if the new policy
does not improve on it. Model deployment and GUI defaults remain separate
from training and selection.
