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

ARCHITECTURE, ACTOR_ARCHITECTURE = neural.AFTERSTATE_TD_ARCHITECTURES
TOP_K = 16
FEATURES = v5.V4_POOLED + neural.CONTEXT
TEACHER_SCALE = .125


@dataclass
class CandidatePool:
    features: torch.Tensor     # [live environments, up to TOP_K, FEATURES]
    valid: torch.Tensor
    indices: torch.Tensor      # original prepared-list indices in frozen actor rank order
    teacher_prior: torch.Tensor
    actor_logits: torch.Tensor


def score_features(head, features):
    """Expected remaining placements in thousands, using the source V4 critic convention."""
    return F.softplus(head(features).squeeze(-1).float())


class AfterstateTDPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config.architecture not in neural.AFTERSTATE_TD_ARCHITECTURES or not config.spatial_actor or not config.regional_critic
                or config.screening != 'exhaustive'):
            raise ValueError('Afterstate TD requires an exhaustive spatial actor and regional critic')
        self.config = config
        self.base = v4.SurvivalSpatialPolicy(replace(config, architecture=v4.ARCHITECTURE,
                                                    residual_bound=0., afterstate_scale=1.))
        self.value = copy.deepcopy(self.base.value)
        self.base.requires_grad_(False)

    @property
    def actor_bound(self):
        return self.config.residual_bound if self.config.architecture == ACTOR_ARCHITECTURE else None

    def configure_policy(self, actor_bound=None, value_scale=1.):
        """Use pure TD (None) or allow value estimates to overcome a bounded frozen-actor logit gap."""
        self.config = replace(self.config,
                              architecture=ARCHITECTURE if actor_bound is None else ACTOR_ARCHITECTURE,
                              residual_bound=0. if actor_bound is None else actor_bound,
                              afterstate_scale=1. if actor_bound is None else value_scale)
        return self

    def score(self, features):
        return score_features(self.value, features)

    @torch.no_grad()
    def _prepare(self, batch, specs, states):
        logits = self.base.actor_logits(batch)
        # Preserve preparation order for tied actor scores, including ties at the shortlist cutoff.
        indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :TOP_K]
        valid = batch['valid'].gather(1, indices)
        selected_specs = specs.gather(1, indices[..., None].expand(-1, -1, specs.shape[-1]))
        features = v5.next_state_features(batch['boards'], batch['targets'], selected_specs, valid, states, move=None)
        return CandidatePool(features, valid, indices, batch['teacher_prior'].gather(1, indices),
                             logits.gather(1, indices))

    def prepare_device(self, result, states):
        """Prepare the live rows of a GPU batch; states include the original, possibly terminal rows."""
        if not result.environment_indices:
            device = result.teachers.device
            indices = torch.empty((0, 0), device=device, dtype=torch.long)
            return CandidatePool(torch.empty((0, 0, FEATURES), device=device), indices.bool(), indices,
                                 indices.float(), indices.float())
        return self._prepare(result.batch, result._option_specs, [states[i] for i in result.environment_indices])

    def prepare_cpu(self, prepared, state):
        """Build one pool from CPU engine candidates, executing the value features on the model's device."""
        device = next(self.value.parameters()).device
        batch = neural.collate([prepared.arrays], device)
        specs = torch.tensor([(c.rotation, c.x, c.y, c.w, c.z, c.layers) for c in prepared.options], device=device)
        return self._prepare(batch, specs[None], [state])

    def ranking_logits(self, values, valid, actor_logits):
        """Score legal candidates; a value preference can cost at most actor_bound nats of actor logit."""
        values = values.masked_fill(~valid, -torch.inf)
        if self.actor_bound is None:
            return values
        if self.actor_bound == 0:
            return actor_logits.masked_fill(~valid, -torch.inf)
        gap = self.config.afterstate_scale * (values - values.amax(1, keepdim=True))
        return (actor_logits + gap.clamp(min=-self.actor_bound, max=0.)).masked_fill(~valid, -torch.inf)

    @torch.no_grad()
    def choose(self, pool, beta=None, generator=None, scores=None):
        """Return local slots. Passing beta enables sampling; anchored policies use their saved value scale."""
        if pool.valid.shape[0] == 0:
            return pool.indices.new_empty(0)
        values = self.score(pool.features) if scores is None else scores
        if beta is None:
            return self.ranking_logits(values, pool.valid, pool.actor_logits).argmax(1)
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Exploration beta must be finite and nonnegative')
        if self.actor_bound is None:
            logits = (beta * values + TEACHER_SCALE * pool.teacher_prior).masked_fill(~pool.valid, -torch.inf)
        else:
            # Frozen actor logits already include the teacher prior.
            logits = self.ranking_logits(values, pool.valid, pool.actor_logits)
        return torch.multinomial(logits.softmax(1), 1, generator=generator).squeeze(1)

    def prepare_gameplay(self, prepared, state):
        """Expose the same top-K rows to NeuralAgent, keeping the engine moves and teacher identity aligned."""
        pool = self.prepare_cpu(prepared, state)
        kept = pool.indices[0].tolist()
        arrays = {key: value[kept] if key in neural.CANDIDATE_FIELDS else value for key, value in prepared.arrays.items()}
        arrays['td_features'] = pool.features[0].cpu().numpy()
        if self.actor_bound is not None:
            arrays['td_actor_logits'] = pool.actor_logits[0].cpu().numpy()
        return replace(prepared, arrays=arrays, options=tuple(prepared.options[i] for i in kept),
                       teacher=kept.index(prepared.teacher) if prepared.teacher in kept else -1)

    def forward(self, batch, capture=False):
        """Return gameplay ranking logits and the raw critic estimate for the selected afterstate."""
        if capture:
            raise ValueError('Afterstate TD uses the feature inspector')
        values = self.score(batch['td_features'])
        logits = self.ranking_logits(values, batch['valid'], batch.get('td_actor_logits'))
        chosen = logits.argmax(1, keepdim=True)
        return {'logits': logits, 'value': values.gather(1, chosen).squeeze(1)}


def from_v4(source, actor_bound=None, value_scale=1.):
    if source.config.architecture != v4.ARCHITECTURE:
        raise ValueError('Afterstate TD needs a regional V4 source checkpoint')
    device = next(source.parameters()).device
    model = AfterstateTDPolicy(replace(source.config, architecture=ARCHITECTURE, screening='exhaustive')).to(device)
    model.base.load_state_dict(source.state_dict())
    model.value.load_state_dict(source.value.state_dict())
    return model.configure_policy(actor_bound=actor_bound, value_scale=value_scale)
