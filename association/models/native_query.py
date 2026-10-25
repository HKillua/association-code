"""Native-event position residuals with matched mean/query pooling controls.

The query arm independently implements QuITE's one-layer summary mechanism;
anonymous source sharing, empty-patch exclusion and offline reconstruction are
our task adaptations. Source permutation invariance is conditional on fixed b.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from association.models.sequence_position import EARTH_RADIUS_M, tangent_position

SCOPE = 'native-query-position-v1'
TIME_UNIT_SECONDS = 20.
FEATURE_WIDTH = 9


class _TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, width - 1)

    def forward(self, seconds):
        tau = seconds.unsqueeze(-1) / TIME_UNIT_SECONDS
        return torch.cat((self.linear(tau), torch.sin(self.periodic(tau))), -1)


def _physical_features(values, valid, origin):
    """Relative visible positions and circular course, with explicit masks.

    Invalid values are sanitized before geometry/log/trigonometry. No reference
    or absolute geographic identity is an input to the learned projection.
    """
    safe = torch.where(valid, values, torch.zeros_like(values))
    latitude = torch.where(valid[:, 0], safe[:, 0], origin[0])
    longitude = torch.where(valid[:, 1], safe[:, 1], origin[1])
    north = (latitude - origin[0]) * (EARTH_RADIUS_M * math.pi / 180.)
    dlon = torch.remainder(longitude - origin[1] + 180., 360.) - 180.
    east = dlon * (EARTH_RADIUS_M * math.pi / 180.) * torch.cos(
        torch.deg2rad((latitude + origin[0]) / 2.))
    course = torch.deg2rad(safe[:, 3])
    physical = torch.stack((north / 1000., east / 1000.,
                            torch.log1p(safe[:, 2] / 10.),
                            course.sin(), course.cos()), -1)
    physical_mask = valid[:, [0, 1, 2, 3, 3]]
    physical = torch.where(physical_mask, physical, torch.zeros_like(physical))
    return torch.cat((physical, valid.to(physical.dtype)), -1).float()


class NativeQueryPosition(nn.Module):
    """T1 mean+MLP and T2 query attention on the same native observations.

    ``events`` is a list of anonymous source dictionaries: times[N] in physical
    seconds, values[N,4] in latitude/longitude/speed/course, valid[N,4]. Event IDs
    and source names are deliberately not learned inputs. Upstream assembly is
    responsible for deduplicating references to the same original event.
    """
    def __init__(self, pooling='mean', hidden_dim=64, patches=8, heads=4):
        super().__init__()
        if (pooling not in ('mean', 'query') or hidden_dim < 2 or patches < 1
                or heads < 1 or hidden_dim % heads):
            raise ValueError('mean/query pooling and compatible positive dimensions required')
        self.pooling, self.hidden_dim, self.patches = pooling, hidden_dim, patches
        self.motion_observation_policy = 'canonical_all'
        # Construct common modules before the differing pooler so a common seed
        # gives identical initial tensors wherever the architecture coincides.
        self.value_embedding = nn.Linear(FEATURE_WIDTH, hidden_dim)
        self.time_embedding = _TimeEmbedding(hidden_dim)
        self.patch_query = nn.Parameter(torch.empty(patches, hidden_dim))
        nn.init.normal_(self.patch_query, std=.02)
        self.patch_norm = nn.LayerNorm(hidden_dim)
        self.base_projection = nn.Linear(FEATURE_WIDTH, hidden_dim)
        self.decoder = nn.MultiheadAttention(hidden_dim, heads, dropout=0., batch_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if pooling == 'query':
            self.pooler = nn.MultiheadAttention(hidden_dim, heads, dropout=0., batch_first=True)
        else:
            # Active 64->128->64 MLP: 16,576 parameters vs attention's 16,640.
            self.pooler = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
                                        nn.Linear(2 * hidden_dim, hidden_dim))

    def _pack_events(self, events, base, query):
        origin = base[0, :2]
        groups, counts = [], []
        for source in events:
            values = torch.as_tensor(source['values'], device=base.device, dtype=torch.float64).detach()
            valid = torch.as_tensor(source['valid'], device=base.device, dtype=torch.bool)
            times = torch.as_tensor(source['times'], device=base.device, dtype=torch.float64).detach()
            if (values.ndim != 2 or values.shape[1] != 4 or valid.shape != values.shape
                    or times.shape != (len(values),)):
                raise ValueError('source requires values/valid[N,4] and times[N]')
            observed = valid.any(-1)
            if (not torch.isfinite(values[valid]).all() or not torch.isfinite(times[observed]).all()
                    or (values[:, 0][valid[:, 0]].abs() > 90.).any()
                    or (values[:, 2][valid[:, 2]] < 0.).any()):
                raise ValueError('valid native observations must have finite physical values/times')
            # Rows with no valid attribute are padding, including arbitrary NaNs.
            values, valid, times = values[observed], valid[observed], times[observed]
            features = _physical_features(values, valid, origin)
            encoded = self.value_embedding(features) + self.time_embedding((times - query[0]).float())
            span = query[-1] - query[0]
            if len(query) == 1:
                index = torch.zeros(len(times), device=base.device, dtype=torch.long)
            else:
                index = torch.floor((times - query[0]) / span * self.patches).long()
                index = index.clamp(0, self.patches - 1)
            source_groups = [encoded[index == patch] for patch in range(self.patches)]
            groups.extend(source_groups)
            counts.append([len(group) for group in source_groups])
        count = torch.tensor(counts, device=base.device, dtype=torch.long).reshape(len(events), self.patches)
        longest = max((len(group) for group in groups), default=0)
        packed = base.new_zeros((len(groups), longest, self.hidden_dim), dtype=torch.float32)
        visible = torch.zeros((len(groups), longest), device=base.device, dtype=torch.bool)
        for row, group in enumerate(groups):
            packed[row, :len(group)] = group
            visible[row, :len(group)] = True
        return packed, visible, count

    def forward(self, events, base_states, query_seconds):
        device = self.output.weight.device
        base = torch.as_tensor(base_states, device=device, dtype=torch.float64).detach()
        query = torch.as_tensor(query_seconds, device=device, dtype=torch.float64).detach()
        if (base.ndim != 2 or base.shape[1] != 4 or not len(base)
                or query.shape != (len(base),) or not torch.isfinite(base).all()
                or not torch.isfinite(query).all() or (base[:, 0].abs() > 90.).any()
                or (base[:, 2] < 0.).any()
                or (len(query) > 1 and not (query[1:] > query[:-1]).all())):
            raise ValueError('finite physical base[Q,4] and strictly increasing query_seconds[Q] required')
        packed, visible, counts = self._pack_events(events, base, query)
        patch_valid = counts > 0
        active = patch_valid.any(-1)
        diagnostics = dict(source_count=len(events), active_source_count=int(active.sum()),
                           event_count=int(counts.sum()), patch_counts=counts,
                           patch_valid=patch_valid, source_valid=active)
        if not active.any():
            # Exact fallback also permits a zero-supervision/empty minibatch to
            # backpropagate zero without creating an artificial source token.
            zero = sum(parameter.sum() * 0. for parameter in self.parameters())
            residual = base.new_zeros((len(base), 2)) + zero
            return dict(states=tangent_position(base, residual), base_states=base,
                        residual_m=residual, diagnostics=diagnostics)

        summary = self.patch_query.unsqueeze(0).expand(len(events), -1, -1).reshape(-1, 1, self.hidden_dim)
        if self.pooling == 'query':
            tokens = torch.cat((summary, packed), 1)
            key_valid = torch.cat((torch.ones((len(tokens), 1), dtype=torch.bool, device=device), visible), 1)
            pooled = self.pooler(summary, tokens, tokens, key_padding_mask=~key_valid,
                                 need_weights=False)[0][:, 0]
        else:
            mean = packed.sum(1) / visible.sum(1, keepdim=True).clamp_min(1)
            pooled = self.pooler(mean + summary[:, 0])
        patches = self.patch_norm(summary[:, 0] + pooled).reshape(len(events), self.patches, self.hidden_dim)
        patches = torch.where(patch_valid[..., None], patches, torch.zeros_like(patches))
        base_features = _physical_features(base, torch.ones_like(base, dtype=torch.bool), base[0, :2])
        q = self.base_projection(base_features) + self.time_embedding((query - query[0]).float())
        q = q.unsqueeze(0).expand(int(active.sum()), -1, -1)
        evidence = self.decoder(q, patches[active], patches[active],
                                key_padding_mask=~patch_valid[active], need_weights=False)[0]
        per_source = self.decoder_norm(q + evidence)
        # Accumulate in float64; only genuine sources vote. Reordering anonymous
        # sources leaves this symmetric mean unchanged up to floating arithmetic.
        fused = per_source.double().mean(0).float()
        residual = self.output(fused).double() * 1000.
        states = tangent_position(base, residual)
        return dict(states=states, base_states=base, residual_m=residual, diagnostics=diagnostics)
