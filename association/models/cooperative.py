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

from association.models.timed import continuous_time_encoding


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
    model_variant: str = 'legacy'
    speed_output_policy: str = 'physical_softplus'
    heading_output_policy: str = 'learned_absolute'

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
        if self.model_variant not in ('legacy', 'support_residual_v2'):
            raise ValueError('unknown cooperative model_variant')
        if self.model_variant == 'support_residual_v2' and self.support_time_policy != 'train_absmax_v1':
            raise ValueError('support_residual_v2 requires train_absmax_v1 support scaling')
        if self.speed_output_policy not in ('physical_softplus', 'scaled_log'):
            raise ValueError('speed_output_policy must be physical_softplus or scaled_log')
        if self.speed_output_policy == 'scaled_log' and self.model_variant != 'support_residual_v2':
            raise ValueError('scaled_log speed requires support_residual_v2')
        if self.heading_output_policy not in ('learned_absolute', 'observation_residual'):
            raise ValueError('heading_output_policy must be learned_absolute or observation_residual')
        if self.heading_output_policy == 'observation_residual' and self.model_variant != 'support_residual_v2':
            raise ValueError('observation_residual heading requires support_residual_v2')


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
        from association.data.cooperative import validate_cooperative_representation
        representation = validate_cooperative_representation(normalizer, config)
        if representation is not None:
            for key in ('log_speed_mean', 'log_speed_std', 'speed_bias'):
                self.register_buffer(key, torch.tensor(representation[key], dtype=torch.float32))
            if config.speed_output_policy == 'scaled_log':
                # Preserve the original FP32 zero-residual speed exactly. This
                # policy-specific state also prevents old weights from being
                # silently reinterpreted in a different output coordinate.
                self.register_buffer('speed_reference', F.softplus(self.speed_bias))
            if config.heading_output_policy == 'observation_residual':
                # Same-shaped absolute/residual heads have different meaning.
                # A constant marker forbids silent reinterpretation without
                # consuming RNG or changing the input representation.
                self.register_buffer('heading_base_version', torch.tensor(1, dtype=torch.int64))
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
        self.input_projection = nn.Linear(5 if representation is not None else 4, d)
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
        if representation is not None:
            self.support_projection = nn.Linear(5, d, bias=False)
            self.local_identity = _head(2 * d + 4, config.head_hidden, 1)
            self.fusion_projection = nn.Linear(d, d, bias=False)
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
        if representation is not None:
            with torch.no_grad():
                self.state_projection.weight[:3].zero_()
                self.state_projection.bias[:3].zero_()
                if config.heading_output_policy == 'observation_residual':
                    self.state_projection.weight[3].zero_()
                    self.state_projection.bias[3].zero_()
                self.local_identity[-1].weight.zero_()
                self.local_identity[-1].bias.zero_()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.config.speed_output_policy != 'scaled_log' and prefix + 'speed_reference' in state_dict:
            raise ValueError('checkpoint scaled_log speed disagrees with declared speed_output_policy')
        if self.config.heading_output_policy == 'observation_residual':
            incoming = state_dict.get(prefix + 'heading_base_version')
            if (not isinstance(incoming, Tensor) or incoming.shape != () or incoming.dtype != torch.int64
                    or not torch.equal(incoming.to(self.heading_base_version.device), self.heading_base_version)):
                raise ValueError('checkpoint heading_base_version disagrees with declared heading_output_policy')
        elif prefix + 'heading_base_version' in state_dict:
            raise ValueError('checkpoint residual heading disagrees with declared heading_output_policy')
        if self.config.support_time_policy == 'train_absmax_v1':
            incoming = state_dict.get(prefix + 'support_time_scale')
            expected = self.support_time_scale
            if (not isinstance(incoming, Tensor) or incoming.shape != ()
                    or not incoming.is_floating_point() or not torch.isfinite(incoming).all()
                    or not torch.equal(incoming.to(expected), expected)):
                raise ValueError('checkpoint support_time_scale disagrees with its declared training scale')
        if self.config.model_variant == 'support_residual_v2':
            keys = ('state_mean', 'state_std', 'log_speed_mean', 'log_speed_std', 'speed_bias')
            if self.config.speed_output_policy == 'scaled_log':
                keys += ('speed_reference',)
            for key in keys:
                incoming, expected = state_dict.get(prefix + key), getattr(self, key)
                if (not isinstance(incoming, Tensor) or incoming.shape != expected.shape
                        or not incoming.is_floating_point() or not torch.isfinite(incoming).all()
                        or not torch.equal(incoming.to(expected), expected)):
                    raise ValueError(f'checkpoint {key} disagrees with its declared representation')
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
        native_positions = native_headings = None
        if self.config.model_variant == 'support_residual_v2':
            native_positions = states[..., :2] * self.state_std[:2] + self.state_mean[:2]
            native_positions = torch.where(mask[..., None], native_positions, 0.)
            # v2 speed is already z_logspeed, never invert it using physical moments.
            physical_heading = states[..., 3] * self.state_std[3] + self.state_mean[3]
            if self.config.heading_output_policy == 'observation_residual':
                native_headings = torch.where(mask, physical_heading, 0.)
            heading = physical_heading * (math.pi / 180.)
            states = torch.cat((states[..., :3], heading.sin()[..., None], heading.cos()[..., None]), dim=-1)
            states = torch.where(mask[..., None], states, 0.)
        if len(values) == 0:
            memory = values.new_empty((0, m, d))
        else:
            embedded = self.input_projection(states) + continuous_time_encoding(times, d)
            embedded = torch.where(mask[..., None], embedded, 0.)
            memory = self.encoder(embedded, src_key_padding_mask=~mask)
            memory = torch.where(mask[..., None], memory, 0.)
        pooled = memory.sum(1) / mask.sum(1).clamp_min(1)[:, None]
        result = dict(memory=memory, observed=mask, times=times, queries=queries, pooled=pooled)
        if native_positions is not None:
            result['native_positions'] = native_positions
        if native_headings is not None:
            result['native_headings'] = native_headings
        return result

    def support_features(self, encoded):
        """Nearest visible support, including internal gaps; no reference mask."""
        delta = encoded['queries'][..., None] - encoded['times'][:, None, :]
        valid = encoded['observed'][:, None, :]
        left = valid & (delta >= 0)
        right = valid & (delta <= 0)
        has_left, has_right = left.any(-1), right.any(-1)
        dl = delta.masked_fill(~left, float('inf')).amin(-1)
        dr = (-delta).masked_fill(~right, float('inf')).amin(-1)
        dl = torch.where(has_left, dl / self.support_time_scale, 0.)
        dr = torch.where(has_right, dr / self.support_time_scale, 0.)
        return torch.stack((dl, dr, has_left.to(delta.dtype), has_right.to(delta.dtype),
                            (has_left & has_right).to(delta.dtype)), dim=-1)

    def position_base(self, encoded, members):
        """Visible-token interpolation inside support and endpoint hold outside."""
        members = torch.as_tensor(members, device=encoded['memory'].device, dtype=torch.long)
        times, valid = encoded['times'][members], encoded['observed'][members]
        query = encoded['queries'][members]
        positions = encoded['native_positions'][members]
        expanded = times[:, None, :].expand(-1, self.config.length, -1)
        left_valid = valid[:, None, :] & (expanded <= query[..., None])
        right_valid = valid[:, None, :] & (expanded >= query[..., None])
        left = expanded.masked_fill(~left_valid, -float('inf')).argmax(-1)
        right = expanded.masked_fill(~right_valid, float('inf')).argmin(-1)
        first = times.masked_fill(~valid, float('inf')).argmin(-1)
        last = times.masked_fill(~valid, -float('inf')).argmax(-1)
        left = torch.where(left_valid.any(-1), left, first[:, None])
        right = torch.where(right_valid.any(-1), right, last[:, None])
        tl, tr = times.gather(1, left), times.gather(1, right)
        pl = positions.gather(1, left[..., None].expand(-1, -1, 2))
        pr = positions.gather(1, right[..., None].expand(-1, -1, 2))
        width = tr-tl
        alpha = torch.where(width > 0, (query-tl)/torch.where(width > 0, width, 1.), 0.)
        latitude = pl[..., 0] + alpha*(pr[..., 0]-pl[..., 0])
        longitude_delta = torch.remainder(pr[..., 1]-pl[..., 1]+180., 360.)-180.
        longitude = torch.remainder(pl[..., 1]+alpha*longitude_delta+180., 360.)-180.
        return torch.stack((latitude, longitude), dim=-1)

    def heading_base(self, encoded, members):
        """Visible native headings, shortest signed arc, endpoint hold outside.

        Half turns follow the sign of the native degree difference, as in the
        data adapter. A small dtype-relative tolerance absorbs normalization
        roundoff; query-state slots and helpers never supply a base.
        """
        members = torch.as_tensor(members, device=encoded['memory'].device, dtype=torch.long)
        times, valid = encoded['times'][members], encoded['observed'][members]
        query, headings = encoded['queries'][members], encoded['native_headings'][members]
        expanded = times[:, None, :].expand(-1, self.config.length, -1)
        left_valid = valid[:, None, :] & (expanded <= query[..., None])
        right_valid = valid[:, None, :] & (expanded >= query[..., None])
        left = expanded.masked_fill(~left_valid, -float('inf')).argmax(-1)
        right = expanded.masked_fill(~right_valid, float('inf')).argmin(-1)
        first = times.masked_fill(~valid, float('inf')).argmin(-1)
        last = times.masked_fill(~valid, -float('inf')).argmax(-1)
        left = torch.where(left_valid.any(-1), left, first[:, None])
        right = torch.where(right_valid.any(-1), right, last[:, None])
        tl, tr = times.gather(1, left), times.gather(1, right)
        hl, hr = headings.gather(1, left), headings.gather(1, right)
        width = tr-tl
        alpha = torch.where(width > 0, (query-tl)/torch.where(width > 0, width, 1.), 0.)
        raw_delta = hr-hl
        delta = torch.remainder(raw_delta+180., 360.)-180.
        # Native degrees are [0,360]. Standardize/invert in FP32 can turn a
        # true +180 difference into 180.00003 and otherwise reverse the arc.
        half_turn = (delta.abs()-180.).abs() <= 8*torch.finfo(delta.dtype).eps*360.
        delta = torch.where(half_turn, torch.where(raw_delta > 0, 180., -180.), delta)
        return torch.remainder(hl+alpha*delta, 360.)

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
        score = head(torch.cat((base, scores), dim=-1)).squeeze(-1)
        if self.config.model_variant == 'support_residual_v2':
            features, available = self.local_pair_features(encoded, pairs)
            score = score + self.local_identity(features).squeeze(-1) * available.to(score.dtype)
        return score

    def local_pair_features(self, encoded, pairs):
        """At most four visible token pairs per direction; exchange symmetric.

        Native times increase with slot order. Stable sorting flattened time
        differences therefore gives the specified time/slot tie order. Reverse
        sorting uses the transpose, then both directional summaries are averaged.
        Edges are chunked so full-scene scoring does not allocate E*M*M at once.
        """
        pairs = self._pairs(encoded, pairs)
        d = self.config.d_model
        if not len(pairs):
            return (encoded['memory'].new_empty((0, 2*d+4)),
                    encoded['observed'].new_empty((0,)))
        first, last, _ = self._support(encoded)
        outputs, availability = [], []
        for chunk in pairs.split(64):
            left, right = chunk[:, 0], chunk[:, 1]
            tl, tr = encoded['times'][left], encoded['times'][right]
            ml, mr = encoded['observed'][left], encoded['observed'][right]
            el, er = encoded['memory'][left], encoded['memory'][right]
            lo, hi = torch.maximum(first[left], first[right]), torch.minimum(last[left], last[right])
            overlap = lo <= hi
            inside_l = ml & (tl >= lo[:, None]) & (tl <= hi[:, None])
            inside_r = mr & (tr >= lo[:, None]) & (tr <= hi[:, None])
            early_l = last[left] < first[right]
            endpoint_l = torch.where(early_l[:, None], tl == last[left, None], tl == first[left, None]) & ml
            endpoint_r = torch.where(early_l[:, None], tr == first[right, None], tr == last[right, None]) & mr
            candidates_l = torch.where(overlap[:, None], inside_l, endpoint_l)
            candidates_r = torch.where(overlap[:, None], inside_r, endpoint_r)
            allowed = candidates_l[:, :, None] & candidates_r[:, None, :]
            delta = (tl[:, :, None]-tr[:, None, :]).abs()
            distance = delta.masked_fill(~allowed, float('inf'))
            gap = (lo-hi).clamp_min(0.) / self.support_time_scale

            def direction(dist, memory_l, memory_r):
                n, width_l, width_r = dist.shape
                k = min(4, width_l*width_r)
                chosen = torch.argsort(dist.reshape(n, -1), dim=-1, stable=True)[:, :k]
                selected_dt = dist.reshape(n, -1).gather(1, chosen)
                valid = torch.isfinite(selected_dt)
                li, ri = chosen // width_r, chosen % width_r
                a = memory_l.gather(1, li[..., None].expand(-1, -1, d))
                b = memory_r.gather(1, ri[..., None].expand(-1, -1, d))
                count = valid.sum(-1)
                denominator = count.clamp_min(1).to(a.dtype)
                valid_f = valid[..., None].to(a.dtype)
                difference = ((a-b).abs()*valid_f).sum(1)/denominator[:, None]
                product = ((a*b)*valid_f).sum(1)/denominator[:, None]
                mean_dt = torch.where(valid, selected_dt, 0.).sum(-1)/denominator/self.support_time_scale
                extra = torch.stack((mean_dt, overlap.to(a.dtype), count.to(a.dtype)/4., gap), dim=-1)
                return torch.cat((difference, product, extra), dim=-1), count > 0

            forward, available = direction(distance, el, er)
            reverse, _ = direction(distance.transpose(1, 2), er, el)
            outputs.append((forward+reverse)*.5)
            availability.append(available)
        return torch.cat(outputs), torch.cat(availability)

    def attribute_route_weights(self, encoded, members, routing, identity_logits=None, utility=None,
                                summaries=None):
        """New fusion's four routes share identity factors and positive floor."""
        members = torch.as_tensor(members, device=encoded['memory'].device, dtype=torch.long)
        q = self.config.length
        weights = encoded['memory'].new_ones((len(members), q, 4))
        if len(members) == 1 or routing == 'plain':
            return weights
        if routing not in ('scalar-u', 'attribute-u', 'learned-attribute'):
            raise ValueError('unknown support_residual_v2 fusion routing')
        pairs = torch.stack((members[0].expand(len(members)-1), members[1:]), dim=1)
        logits = self.score_pairs(encoded, pairs) if identity_logits is None else identity_logits
        if tuple(logits.shape) != (len(members)-1,) or not torch.isfinite(logits).all():
            raise ValueError('identity_logits must be finite [K-1] in helper order')
        identity = logits.sigmoid()[:, None, None]
        if routing == 'learned-attribute':
            summaries = self.summarize(encoded) if summaries is None else summaries
            gates = self.ordinary_gate(self._directional_features(encoded, pairs, summaries)).sigmoid()
        else:
            values = self.utility(encoded, pairs, summaries=summaries)[:, 0] if utility is None else utility
            if tuple(values.shape) != (len(members)-1, q, 4) or not torch.isfinite(values).all():
                raise ValueError('utility must be finite [K-1,Q,4] in anchor<-helper order')
            gates = values.mean(-1, keepdim=True).expand(-1, -1, 4) if routing == 'scalar-u' else values
            gates = gates.clamp(0., 1.)
        helper = identity*gates
        if helper.device != weights.device or not torch.isfinite(helper).all():
            raise ValueError('route weights must be finite on the encoded device')
        weights[1:] = self.config.gate_floor + (1.-self.config.gate_floor)*helper
        return weights

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
        routes = (('plain', 'scalar-u', 'attribute-u', 'learned-attribute')
                  if self.config.model_variant == 'support_residual_v2'
                  else ('plain', 'identity', 'utility', 'support', 'learned'))
        if routing not in routes:
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
        if self.config.model_variant == 'support_residual_v2':
            return self._decode_residual_group(encoded, members, routing, identity_logits, utility)
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

    def _decode_residual_group(self, encoded, members, routing, identity_logits, utility):
        anchor = members[0]
        visible = encoded['observed'][anchor]
        memory = (encoded['memory'][anchor, visible] + self.role_embedding.weight[0])[None]
        query = continuous_time_encoding(encoded['queries'][members[:1]], self.config.d_model)
        support = self.support_features(encoded)[members[:1]]
        query = query + self.support_projection(support)
        hidden = self.decoder(query, memory)[0]
        if len(members) > 1:
            summaries = self.summarize(encoded)
            weights = self.attribute_route_weights(encoded, members, routing, identity_logits,
                                                   utility, summaries=summaries)[1:]
            differences = summaries[members[1:]] - summaries[anchor][None]
            context = torch.einsum('kqf,kqd->qfd', weights, differences)
            context = context / (1.+weights.sum(0))[..., None]
            attribute_hidden = hidden[:, None, :] + self.fusion_projection(context)
        else:
            attribute_hidden = hidden[:, None, :].expand(-1, 4, -1)
        residual = (attribute_hidden*self.state_projection.weight[None]).sum(-1) + self.state_projection.bias
        if not torch.isfinite(residual).all():
            raise ValueError('cooperative residual output is nonfinite')
        base = self.position_base(encoded, members[:1])[0]
        latitude = (base[:, 0]+self.state_std[0]*residual[:, 0]).clamp(-90., 90.)
        longitude = torch.remainder(base[:, 1]+self.state_std[1]*residual[:, 1]+180., 360.)-180.
        if self.config.speed_output_policy == 'scaled_log':
            speed = self.speed_reference * torch.exp(self.log_speed_std * residual[:, 2])
        else:
            speed = F.softplus(residual[:, 2]+self.speed_bias)
        heading_base = (self.heading_base(encoded, members[:1])[0]
                        if self.config.heading_output_policy == 'observation_residual' else self.state_mean[3])
        course = torch.remainder(self.state_std[3]*residual[:, 3]+heading_base, 360.)
        prediction = torch.stack((latitude, longitude, speed, course), dim=-1)
        if not torch.isfinite(prediction).all():
            raise ValueError('cooperative physical output is nonfinite')
        return prediction

    def forward(self, values, observed, members=None, routing='plain', identity_logits=None, utility=None):
        encoded = self.encode(values, observed)
        if members is None:
            members = list(range(len(values)))
        return self.decode_group(encoded, members, routing, identity_logits, utility)
