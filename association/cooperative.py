"""Native-time cooperative completion, identity and physical fusion candidate.

This is an independent research model, not an exact TMTAFF reproduction. No
reference state or identity label is accepted by its forward interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .timed_models import continuous_time_encoding


@dataclass(frozen=True)
class CooperativeConfig:
    length: int = 64
    max_observations: int = 128
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 256
    dropout: float = .1
    head_hidden: int = 64
    gate_floor: float = 1e-3
    support_time_policy: str = 'legacy'

    def __post_init__(self):
        for name in ('length', 'max_observations', 'd_model', 'nhead',
                     'num_layers', 'dim_feedforward', 'head_hidden'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_observations < 2 or self.d_model % self.nhead:
            raise ValueError('need >=2 native slots and d_model divisible by nhead')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        if not math.isfinite(self.gate_floor) or not 0 < self.gate_floor < 1:
            raise ValueError('gate_floor must be in (0,1)')
        if self.support_time_policy not in ('legacy', 'train_absmax_v1'):
            raise ValueError('support_time_policy must be legacy or train_absmax_v1')


def _head(inputs, hidden, outputs):
    # No dropout: ordinary-gate and utility controls have identical capacity.
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Linear(hidden, outputs))


class CooperativeModel(nn.Module):
    input_adapter = 'native_time_v1'

    def __init__(self, config: CooperativeConfig, normalizer: dict, *, support_time_scale=None):
        super().__init__()
        self.config = config
        mean = torch.as_tensor(normalizer['mean'], dtype=torch.float32)
        scale = torch.as_tensor(normalizer['std'], dtype=torch.float32)
        if (mean.shape != (4,) or scale.shape != (4,) or not torch.isfinite(mean).all()
                or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('normalizer needs four finite means and positive scales')
        self.register_buffer('state_mean', mean.clone())
        self.register_buffer('state_std', scale.clone())
        scaled_support = config.support_time_policy == 'train_absmax_v1'
        if support_time_scale is None and not scaled_support:
            support_time_scale = 1.
        if (isinstance(support_time_scale, bool)
                or not isinstance(support_time_scale, (int, float))
                or not math.isfinite(support_time_scale) or support_time_scale < 1
                or support_time_scale > torch.finfo(torch.float32).max
                or (not scaled_support and support_time_scale != 1)):
            raise ValueError('support_time_scale must be explicit finite >=1 for train_absmax_v1, or 1 for legacy')
        # No new persistent key for historical checkpoints. The buffer carries
        # no learnable parameters, consumes no RNG, and moves with the model.
        self.register_buffer('support_time_scale', torch.tensor(support_time_scale, dtype=torch.float32),
                             persistent=scaled_support)
        d = config.d_model
        self.input_projection = nn.Linear(4, d)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, config.nhead, config.dim_feedforward, config.dropout, batch_first=True),
            config.num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d, config.nhead, config.dim_feedforward, config.dropout, batch_first=True),
            config.num_layers)
        self.state_projection = nn.Linear(d, 4)
        self.role_embedding = nn.Embedding(2, d)
        self.query_pool = nn.MultiheadAttention(d, config.nhead, dropout=0., batch_first=True)
        # Target/helper summaries, product, signed difference, and 4 support
        # features per track. Only these MLP features use the optional training
        # scale; encoded physical tau and temporal sin/cos inputs stay intact.
        self.utility_head = _head(4 * d + 8, config.head_hidden, 4)
        self.ordinary_gate = _head(4 * d + 8, config.head_hidden, 4)
        identity_width = 2 * d + 8 + 4
        self.identity_base = _head(identity_width, config.head_hidden, 1)
        self.identity_with_utility = _head(identity_width, config.head_hidden, 1)
        # Transformer containers clone a prototype. Distinct depths must not
        # retain identical sampled initial matrices.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.config.support_time_policy == 'train_absmax_v1':
            incoming = state_dict.get(prefix + 'support_time_scale')
            expected = self.support_time_scale
            if (not isinstance(incoming, Tensor) or incoming.shape != ()
                    or not incoming.is_floating_point() or not torch.isfinite(incoming).all()
                    or not torch.equal(incoming.to(expected), expected)):
                raise ValueError('checkpoint support_time_scale disagrees with its declared training scale')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _validate_inputs(self, values: Tensor, observed: Tensor):
        m, q = self.config.max_observations, self.config.length
        if values.ndim != 3 or tuple(values.shape[1:]) != (m + q, 5):
            raise ValueError('values must be [N,max_observations+length,5]')
        if (observed.shape != values.shape[:2] or observed.dtype != torch.bool
                or values.device != observed.device or not values.is_floating_point()):
            raise ValueError('observed must be a boolean mask on the values device')
        if values.device != self.state_mean.device or values.dtype != self.state_mean.dtype:
            raise ValueError('inputs must use the model device and floating dtype')
        mask = observed[:, :m]
        if observed[:, m:].any() or (mask.sum(1) < 2).any():
            raise ValueError('each fragment needs >=2 observations and no observed query rows')
        if not torch.isfinite(values[:, :m][mask]).all():
            raise ValueError('visible native values and times must be finite')
        for row in range(len(values)):
            times = values[row, :m, 4][mask[row]]
            if (times[1:] <= times[:-1]).any():
                raise ValueError('visible native times must be strictly increasing')
        queries = values[:, m:, 4]
        if not torch.isfinite(queries).all() or (queries[:, 1:] <= queries[:, :-1]).any():
            raise ValueError('query times must be finite and strictly increasing')
        return mask

    def encode(self, values: Tensor, observed: Tensor) -> dict[str, Tensor]:
        mask = self._validate_inputs(values, observed)
        m, d = self.config.max_observations, self.config.d_model
        # Sanitize before any arithmetic; query states are never indexed.
        states = torch.where(mask[..., None], values[:, :m, :4], 0.)
        times = torch.where(mask, values[:, :m, 4], 0.)
        queries = values[:, m:, 4]
        if len(values) == 0:
            memory = values.new_empty((0, m, d))
        else:
            embedded = self.input_projection(states) + continuous_time_encoding(times, d)
            embedded = torch.where(mask[..., None], embedded, 0.)
            memory = self.encoder(embedded, src_key_padding_mask=~mask)
            memory = torch.where(mask[..., None], memory, 0.)
        pooled = memory.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        return dict(memory=memory, observed=mask, times=times, queries=queries, pooled=pooled)

    @staticmethod
    def _support(encoded):
        mask, times = encoded['observed'], encoded['times']
        first = times.masked_fill(~mask, float('inf')).amin(1)
        last = times.masked_fill(~mask, -float('inf')).amax(1)
        count = mask.sum(1).to(times.dtype)
        return first, last, count

    def summarize(self, encoded) -> Tensor:
        if len(encoded['memory']) == 0:
            return encoded['memory'].new_empty((0, self.config.length, self.config.d_model))
        queries = continuous_time_encoding(encoded['queries'], self.config.d_model)
        result, _ = self.query_pool(queries, encoded['memory'], encoded['memory'],
                                   key_padding_mask=~encoded['observed'], need_weights=False)
        return result

    @staticmethod
    def _pairs(encoded, pairs) -> Tensor:
        pairs = torch.as_tensor(pairs, device=encoded['memory'].device)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype not in (torch.int32, torch.int64):
            raise ValueError('pairs must be integer [E,2]')
        pairs = pairs.long()
        if (pairs < 0).any() or (pairs >= len(encoded['memory'])).any() or (pairs[:, 0] == pairs[:, 1]).any():
            raise ValueError('pairs must index distinct encoded tracks')
        return pairs

    def _query_support(self, encoded):
        first, last, count = self._support(encoded)
        queries = encoded['queries']
        if self.config.support_time_policy == 'train_absmax_v1':
            first, last, queries = (value / self.support_time_scale for value in (first, last, queries))
        return torch.stack((queries - first[:, None], last[:, None] - queries,
                            (last - first)[:, None].expand_as(queries),
                            count.log1p()[:, None].expand_as(queries)), dim=-1)

    def _directional_features(self, encoded, pairs, summaries):
        target, helper = summaries[pairs[:, 0]], summaries[pairs[:, 1]]
        support = self._query_support(encoded)
        left, right = support[pairs[:, 0]], support[pairs[:, 1]]
        return torch.cat((target, helper, target * helper, target - helper, left, right), dim=-1)

    def utility(self, encoded, pairs, summaries=None) -> Tensor:
        pairs = self._pairs(encoded, pairs)
        if len(pairs) == 0:
            return encoded['memory'].new_empty((0, 2, self.config.length, 4))
        if not torch.equal(encoded['queries'][pairs[:, 0]], encoded['queries'][pairs[:, 1]]):
            raise ValueError('utility pairs must share physical query times')
        summaries = self.summarize(encoded) if summaries is None else summaries
        expected = (len(encoded['memory']), self.config.length, self.config.d_model)
        if tuple(summaries.shape) != expected:
            raise ValueError('summaries must be [N,Q,D]')
        left = self.utility_head(self._directional_features(encoded, pairs, summaries))
        right = self.utility_head(self._directional_features(encoded, pairs.flip(1), summaries))
        return torch.stack((left, right), dim=1)

    def score_pairs(self, encoded, pairs, use_utility=False) -> Tensor:
        pairs = self._pairs(encoded, pairs)
        left, right = encoded['pooled'][pairs[:, 0]], encoded['pooled'][pairs[:, 1]]
        first, last, count = self._support(encoded)
        if self.config.support_time_policy == 'train_absmax_v1':
            first, last = first / self.support_time_scale, last / self.support_time_scale
        support = torch.stack((first, last, last - first, count.log1p()), dim=-1)
        sl, sr = support[pairs[:, 0]], support[pairs[:, 1]]
        base = torch.cat(((left - right).abs(), left * right, (sl - sr).abs(), sl * sr), dim=-1)
        if use_utility:
            # Identity optimization cannot redefine the frozen utility meaning.
            scores = self.utility(encoded, pairs).detach().tanh().mean(dim=(1, 2))
            head = self.identity_with_utility
        else:
            scores = base.new_zeros((len(pairs), 4))
            head = self.identity_base
        return head(torch.cat((base, scores), dim=-1)).squeeze(-1)

    def _route_weights(self, encoded, members, routing, identity_logits, utility):
        q = self.config.length
        weights = encoded['memory'].new_ones((len(members), q))
        if len(members) == 1 or routing == 'plain':
            return weights
        pairs = torch.stack((members[0].expand(len(members) - 1), members[1:]), dim=1)
        if routing in ('identity', 'utility'):
            logits = self.score_pairs(encoded, pairs) if identity_logits is None else identity_logits
            if tuple(logits.shape) != (len(members) - 1,):
                raise ValueError('identity_logits must be [K-1] in helper order')
            if not torch.isfinite(logits).all():
                raise ValueError('identity_logits must be finite')
            identity = logits.sigmoid()[:, None].expand(-1, q)
        if routing == 'identity':
            helper = identity
        elif routing == 'utility':
            values = self.utility(encoded, pairs)[:, 0] if utility is None else utility
            if tuple(values.shape) != (len(members) - 1, q, 4):
                raise ValueError('utility must be [K-1,Q,4] in anchor<-helper order')
            if not torch.isfinite(values).all():
                raise ValueError('utility must be finite before routing clipping')
            helper = identity * values.mean(-1).clamp(0., 1.)
        elif routing == 'support':
            first, last, _ = self._support(encoded)
            query = encoded['queries'][members[0]]
            helper = ((query[None, :] >= first[members[1:], None])
                      & (query[None, :] <= last[members[1:], None])).to(weights.dtype)
        else:  # learned: a separate, capacity-matched ordinary gate.
            summaries = self.summarize(encoded)
            helper = self.ordinary_gate(self._directional_features(encoded, pairs, summaries)).sigmoid().mean(-1)
        if helper.device != weights.device or not torch.isfinite(helper).all():
            raise ValueError('route weights must be finite on the encoded device')
        weights[1:] = self.config.gate_floor + (1. - self.config.gate_floor) * helper
        return weights

    def decode_group(self, encoded, members, routing='plain', identity_logits=None, utility=None) -> Tensor:
        if routing not in ('plain', 'identity', 'utility', 'support', 'learned'):
            raise ValueError('unknown fusion routing')
        members = torch.as_tensor(members, device=encoded['memory'].device)
        if members.ndim != 1 or members.dtype not in (torch.int32, torch.int64) or len(members) == 0:
            raise ValueError('members must be a nonempty integer vector; first is anchor')
        members = members.long()
        if ((members < 0).any() or (members >= len(encoded['memory'])).any()
                or len(members.unique()) != len(members)):
            raise ValueError('group members must be unique valid track indices')
        queries = encoded['queries'][members]
        if not torch.equal(queries, queries[:1].expand_as(queries)):
            raise ValueError('group tracks must share the same physical query times')
        weights = self._route_weights(encoded, members, routing, identity_logits, utility)
        memories, biases = [], []
        for position, member in enumerate(members):
            valid = encoded['observed'][member]
            n = int(valid.sum())
            role = 0 if position == 0 else 1
            memories.append(encoded['memory'][member, valid] + self.role_embedding.weight[role])
            # Each member receives total prior mass w, regardless of track length.
            bias = weights[position].log() - math.log(n)
            biases.append(bias[:, None].expand(-1, n))
        memory = torch.cat(memories, dim=0)[None]
        attention_bias = torch.cat(biases, dim=1)
        query = continuous_time_encoding(queries[:1], self.config.d_model)
        hidden = self.decoder(query, memory, memory_mask=attention_bias)
        raw = self.state_projection(hidden[0]) * self.state_std + self.state_mean
        return torch.stack((raw[:, 0].clamp(-90., 90.), torch.remainder(raw[:, 1] + 180., 360.) - 180.,
                            F.softplus(raw[:, 2]), torch.remainder(raw[:, 3], 360.)), dim=-1)

    def forward(self, values, observed, members=None, routing='plain', identity_logits=None, utility=None):
        encoded = self.encode(values, observed)
        if members is None:
            members = list(range(len(values)))
        return self.decode_group(encoded, members, routing, identity_logits, utility)
