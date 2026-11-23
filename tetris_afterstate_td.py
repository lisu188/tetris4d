"""TD policy on the afterstates shortlisted by a frozen exhaustive-screening V4 actor."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F

import tetris_neural as neural
import tetris_survival_v4 as v4
import tetris_survival_v5 as v5

ARCHITECTURE = 'survival-afterstate-td'
TOP_K = 16
FEATURES = v5.V4_POOLED + neural.CONTEXT
TEACHER_SCALE = .125


@dataclass
class CandidatePool:
    features: torch.Tensor     # [live environments, up to TOP_K, FEATURES]
    valid: torch.Tensor
    indices: torch.Tensor      # original prepared-list indices in frozen actor rank order
    teacher_prior: torch.Tensor


def score_features(head, features):
    """Expected remaining placements in thousands, using the source V4 critic convention."""
    return F.softplus(head(features).squeeze(-1).float())


class AfterstateTDPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config.architecture != ARCHITECTURE or not config.spatial_actor or not config.regional_critic
                or config.screening != 'exhaustive'):
            raise ValueError('Afterstate TD requires an exhaustive spatial actor and regional critic')
        self.config = config
        self.base = v4.SurvivalSpatialPolicy(replace(config, architecture=v4.ARCHITECTURE))
        self.value = copy.deepcopy(self.base.value)
        self.base.requires_grad_(False)

    def score(self, features):
        return score_features(self.value, features)

    @torch.no_grad()
    def _prepare(self, batch, specs, states):
        logits = self.base(batch)['logits']
        # Preserve preparation order for tied actor scores, including ties at the shortlist cutoff.
        indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :TOP_K]
        valid = batch['valid'].gather(1, indices)
        selected_specs = specs.gather(1, indices[..., None].expand(-1, -1, specs.shape[-1]))
        features = v5.next_state_features(batch['boards'], batch['targets'], selected_specs, valid, states, move=None)
        return CandidatePool(features, valid, indices, batch['teacher_prior'].gather(1, indices))

    def prepare_device(self, result, states):
        """Prepare the live rows of a GPU batch; states include the original, possibly terminal rows."""
        if not result.environment_indices:
            device = result.teachers.device
            indices = torch.empty((0, 0), device=device, dtype=torch.long)
            return CandidatePool(torch.empty((0, 0, FEATURES), device=device), indices.bool(), indices, indices.float())
        return self._prepare(result.batch, result._option_specs, [states[i] for i in result.environment_indices])

    def prepare_cpu(self, prepared, state):
        """Build one pool from CPU engine candidates, executing the value features on the model's device."""
        device = next(self.value.parameters()).device
        batch = neural.collate([prepared.arrays], device)
        specs = torch.tensor([(c.rotation, c.x, c.y, c.w, c.z, c.layers) for c in prepared.options], device=device)
        return self._prepare(batch, specs[None], [state])

    @torch.no_grad()
    def choose(self, pool, beta=None, generator=None):
        """Return local pool slots, greedy by value for evaluation or teacher-guided on-policy samples."""
        if pool.valid.shape[0] == 0:
            return pool.indices.new_empty(0)
        scores = self.score(pool.features)
        if beta is None:
            return scores.masked_fill(~pool.valid, -torch.inf).argmax(1)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Exploration beta must be finite and nonnegative')
        logits = (beta * scores + TEACHER_SCALE * pool.teacher_prior).masked_fill(~pool.valid, -torch.inf)
        return torch.multinomial(logits.softmax(1), 1, generator=generator).squeeze(1)

    def prepare_gameplay(self, prepared, state):
        """Expose the same top-K rows to NeuralAgent, keeping the engine moves and teacher identity aligned."""
        pool = self.prepare_cpu(prepared, state)
        kept = pool.indices[0].tolist()
        arrays = {key: value[kept] if key in neural.CANDIDATE_FIELDS else value for key, value in prepared.arrays.items()}
        arrays['td_features'] = pool.features[0].cpu().numpy()
        return replace(prepared, arrays=arrays, options=tuple(prepared.options[i] for i in kept),
                       teacher=kept.index(prepared.teacher) if prepared.teacher in kept else -1)

    def forward(self, batch, capture=False):
        """Gameplay picks the highest-valued afterstate; the inspector displays that selected value."""
        if capture:
            raise ValueError('Afterstate TD uses the feature inspector')
        scores = self.score(batch['td_features']).masked_fill(~batch['valid'], -torch.inf)
        return {'logits': scores, 'value': scores.amax(1)}


def from_v4(source):
    if source.config.architecture != v4.ARCHITECTURE:
        raise ValueError('Afterstate TD needs a regional V4 source checkpoint')
    device = next(source.parameters()).device
    model = AfterstateTDPolicy(replace(source.config, architecture=ARCHITECTURE, screening='exhaustive')).to(device)
    model.base.load_state_dict(source.state_dict())
    model.value.load_state_dict(source.value.state_dict())
    return model
