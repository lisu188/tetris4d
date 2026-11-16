"""Compact survival reranker with fixed board features and separate value weights."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

import tetris_neural as neural

ARCHITECTURE = 'survival-feature-v3'
STATE_FEATURES = 2 * neural.CHANNELS + neural.CONTEXT
CANDIDATE_FEATURES = neural.FEATURES + 2 * neural.CHANNELS


class SurvivalFeaturePolicy(nn.Module):
    """Learn candidate preferences without a spatial convolution or shared critic.

    The fixed 34-channel encoding preserves the legacy occupancy/gap contract.
    Global and affected-column mean/max statistics describe that encoding; all
    trainable actor and critic parameters belong to separate modules. A teacher
    prior, when requested explicitly by config, remains an external heuristic.
    """

    def __init__(self, config: neural.Config):
        super().__init__()
        if config.architecture != ARCHITECTURE:
            raise ValueError('SurvivalFeaturePolicy requires survival-feature-v3 architecture')
        if config.centered_critic:
            raise ValueError('V3 uses a softplus critic; centered_critic must be false')
        self.config = config
        self.fused_encoding = False
        self.actor_state = nn.Sequential(nn.LayerNorm(STATE_FEATURES), nn.Linear(STATE_FEATURES, 64), nn.SiLU())
        self.candidate = nn.Sequential(nn.LayerNorm(CANDIDATE_FEATURES), nn.Linear(CANDIDATE_FEATURES, 128), nn.SiLU(),
                                       nn.Linear(128, 64), nn.SiLU())
        self.rank1 = nn.Linear(128, 64)
        self.rank_out = nn.Linear(64, 1)
        self.value = nn.Sequential(nn.LayerNorm(STATE_FEATURES), nn.Linear(STATE_FEATURES, 128), nn.SiLU(),
                                   nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.rank_out.weight, gain=.01)
        nn.init.zeros_(self.rank_out.bias)

    def critic(self, joined):
        # Smoothly positive without the dead negative branch of a hard clamp.
        return F.softplus(self.value(joined).squeeze(-1).float())

    def encode(self, boards, targets, context):
        spatial = neural.encode_columns(boards, targets, fused=self.fused_encoding)
        pooled = torch.cat((spatial.mean(dim=(2, 3, 4)), spatial.amax(dim=(2, 3, 4))), dim=1)
        # Keep this order identical to inherited value_inputs: pooled then context.
        joined = torch.cat((pooled, context), dim=1)
        value = self.critic(joined)
        completion = value.new_empty((len(value), 0))
        return spatial, pooled, value, completion, spatial

    @staticmethod
    def local_statistics(spatial, indices, column_mask):
        flat = spatial.flatten(2).transpose(1, 2)
        mask = column_mask.bool()
        safe_indices = indices.long().masked_fill(~mask, 0)
        rows = torch.arange(len(flat), device=flat.device)[:, None, None]
        gathered = flat[rows, safe_indices]
        expanded = mask[..., None]
        count = expanded.sum(2).clamp_min(1)
        mean = gathered.masked_fill(~expanded, 0.).sum(2) / count
        maximum = gathered.masked_fill(~expanded, -torch.inf).amax(2)
        maximum = torch.where(expanded.any(2), maximum, 0.)
        return torch.cat((mean, maximum), dim=-1)

    def rank_chunk(self, spatial, pooled, context, features, indices, column_mask, state_hidden=None):
        local = self.local_statistics(spatial, indices, column_mask)
        candidates = self.candidate(torch.cat((features, local), dim=-1))
        if state_hidden is None:
            state_hidden = self.actor_state(torch.cat((pooled, context), dim=1))
        state = state_hidden[:, None].expand(-1, features.shape[1], -1)
        hidden = F.silu(self.rank1(torch.cat((state, candidates), dim=-1)))
        return self.rank_out(hidden).squeeze(-1), candidates, hidden

    def raw_scores(self, batch):
        valid = batch['valid'].bool()
        column_mask = batch['column_mask'].bool() & valid[..., None]
        if not bool(valid.any(-1).all() & ((~valid) | column_mask.any(-1)).all()):
            raise ValueError('Every state needs a valid candidate with an affected column')
        spatial, pooled, value, completion, _ = self.encode(batch['boards'], batch['targets'], batch['context'])
        state_hidden = self.actor_state(torch.cat((pooled, batch['context']), dim=1))
        # Padding may contain arbitrary values; remove it before learned operations.
        features = batch['features'].masked_fill(~valid[..., None], 0.)
        scores, hidden = [], []
        for start in range(0, features.shape[1], self.config.chunk):
            stop = start + self.config.chunk
            raw, _, h = self.rank_chunk(spatial, pooled, batch['context'], features[:, start:stop],
                                         batch['indices'][:, start:stop], column_mask[:, start:stop], state_hidden)
            scores.append(raw)
            hidden.append(h)
        return torch.cat(scores, 1), torch.cat(hidden, 1), pooled, value, completion

    def forward(self, batch, capture=False):
        if capture:
            raise NotImplementedError('V3 has no legacy CNN activation trace; use headless inference')
        raw, _, pooled, value, completion = self.raw_scores(batch)
        logits = raw
        if self.config.policy_mode in ('teacher-residual', 'teacher-prior'):
            if 'teacher_prior' not in batch:
                raise ValueError('Hybrid policy requires exact teacher ranking features')
            if self.config.policy_mode == 'teacher-residual':
                logits = batch['teacher_prior'].float() + self.config.residual_scale * (2 * torch.sigmoid(raw.float()) - 1)
            else:
                logits = self.config.teacher_prior_scale * batch['teacher_prior'].float() + self.config.residual_scale * raw.float()
        logits = logits.masked_fill(~batch['valid'].bool(), -torch.inf)
        return {'logits': logits, 'value': value, 'completion': completion,
                'critic_input': torch.cat((pooled, batch['context']), dim=1).detach()}

    @torch.no_grad()
    def ranking_diagnostics(self, batch):
        raw, hidden, _, _, _ = self.raw_scores(batch)
        valid = batch['valid'].bool()
        count = valid.sum(1, keepdim=True).float()
        mean = raw.masked_fill(~valid, 0.).sum(1, keepdim=True) / count
        variance = ((raw - mean).square().masked_fill(~valid, 0.).sum(1) / count[:, 0])
        spread = raw.masked_fill(~valid, -torch.inf).amax(1) - raw.masked_fill(~valid, torch.inf).amin(1)
        hmean = hidden.masked_fill(~valid[..., None], 0.).sum(1) / count
        hvar = (hidden - hmean[:, None]).square().masked_fill(~valid[..., None], 0.).sum(1) / count
        return {'states': len(raw), 'architecture': ARCHITECTURE,
                'h2_activation': 'silu', 'h2_saturated_fraction': None,
                'h2_candidate_std': float(hvar.clamp_min(0.).sqrt().mean()),
                'raw_logit_spread': float(spread.mean()), 'raw_logit_std': float(variance.clamp_min(0.).sqrt().mean())}
