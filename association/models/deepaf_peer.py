"""DeepAF P/V core, adapted to visible fixed-group position reconstruction.

Source: Cui et al., Aerospace 2025, 12, 194, sections 3.1--3.3 and 4.1.
The two independent encoder/decoder networks retain measurement differences,
their own chronological estimated history, and equation (12) cross attention.
This is not a reproduction of the original radar association/track lifecycle.

Explicit adaptation choices are fixed-group candidates, a visible spherical
local chart, real query intervals, OLS bootstrap, a masked history of five,
dropout zero, velocity units of 10 m/s, and closed-loop TBPTT of one step.
Only loss construction reads truth. Inference returns ordinary Mean speed and
course, independently of the internal DeepAF-V Cartesian velocity estimate.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from association.models.sequence_position import EARTH_RADIUS_M, sequence_inputs

SCOPE = 'deepaf-pv-position-peer-v1'
ROUTING_SEMANTICS = 'fixed056-pv-closed-loop-tbptt1-v1'
LOSS_SEMANTICS = 'pv-desired-increment-mse-v1'
MODEL_CONFIG = dict(hidden_size=32, num_layers=6, num_heads=4,
                    feedforward_size=64, history_length=5,
                    velocity_scale_mps=10.0, dropout=0.0)
INPUT_KEYS = ('base_states', 'mean_states', 'measurements_m', 'candidate_valid',
              'query_seconds', 'initial_velocity_mps', 'origin_latlon',
              'base_positions_m', 'helper_support_count')


def _basis(origin_latlon):
    lat, lon = torch.deg2rad(origin_latlon).unbind(-1)
    slat, clat, slon, clon = lat.sin(), lat.cos(), lon.sin(), lon.cos()
    radial = torch.stack((clat * clon, clat * slon, slat), -1)
    north = torch.stack((-slat * clon, -slat * slon, clat), -1)
    east = torch.stack((-slon, clon, torch.zeros_like(lon)), -1)
    return radial, north, east


def positions_to_local(position_latlon, origin_latlon):
    """Spherical logarithm in the origin's north/east tangent plane, metres.

    Arguments use normal PyTorch broadcasting and must have a final dimension
    of two. The visible origin defines one common chart for the entire group.
    This azimuthal-equidistant chart is invertible inside the antipodal cut.
    An exactly/nearly antipodal point (within 1e-10 radians) is ambiguous and
    raises instead of silently clipping or choosing an arbitrary direction.
    """
    if (position_latlon.shape[-1:] != (2,) or origin_latlon.shape[-1:] != (2,)
            or position_latlon.dtype != torch.float64 or origin_latlon.dtype != torch.float64
            or position_latlon.device != origin_latlon.device
            or not torch.isfinite(position_latlon).all() or not torch.isfinite(origin_latlon).all()
            or (position_latlon[..., 0].abs() > 90.).any()
            or (origin_latlon[..., 0].abs() > 90.).any()):
        raise ValueError('finite float64 geographic positions and origin are required')
    radial, north, east = _basis(origin_latlon)
    target = _basis(position_latlon)[0]
    cosine = (target * radial).sum(-1)
    tangent = torch.stack(((target * north).sum(-1), (target * east).sum(-1)), -1)
    sine = torch.linalg.vector_norm(tangent, dim=-1)
    if ((cosine < 0.) & (sine <= 1e-10)).any():
        raise ValueError('DeepAF local chart is undefined at the antipodal cut')
    angle = torch.atan2(sine, cosine)
    denominator = torch.where(sine > 1e-12, sine, torch.ones_like(sine))
    factor = torch.where(sine > 1e-12, angle / denominator, torch.ones_like(sine))
    result = EARTH_RADIUS_M * tangent * factor[..., None]
    # Identical positions have an exact zero chart coordinate, without changing
    # gradients through the geometric expression.
    same = (position_latlon == origin_latlon).all(-1)
    return torch.where(same[..., None], result - result.detach(), result)


def local_to_positions(position_m, origin_latlon):
    """Spherical exponential, inverse to positions_to_local inside its cut.

    Network predictions may leave the injective chart; the exponential remains
    a finite geographic output, without clipping a large prediction. Such
    excursions are exposed by ``chart_excursion_count`` in a rollout.
    """
    if (position_m.shape[-1:] != (2,) or origin_latlon.shape[-1:] != (2,)
            or position_m.dtype != torch.float64 or origin_latlon.dtype != torch.float64
            or position_m.device != origin_latlon.device
            or not torch.isfinite(position_m).all() or not torch.isfinite(origin_latlon).all()
            or (origin_latlon[..., 0].abs() > 90.).any()):
        raise ValueError('finite float64 local metres and geographic origin are required')
    radial, north, east = _basis(origin_latlon)
    angular = position_m / EARTH_RADIUS_M
    angle = torch.linalg.vector_norm(angular, dim=-1)
    tangent = angular[..., :1] * north + angular[..., 1:] * east
    vector = angle.cos()[..., None] * radial + torch.sinc(angle / math.pi)[..., None] * tangent
    latitude = torch.atan2(vector[..., 2], torch.linalg.vector_norm(vector[..., :2], dim=-1))
    pole = (vector[..., :2] == 0.).all(-1)
    meridian = torch.deg2rad(origin_latlon[..., 1])
    longitude = torch.atan2(torch.where(pole, meridian.sin(), vector[..., 1]),
                            torch.where(pole, meridian.cos(), vector[..., 0]))
    result = torch.rad2deg(torch.stack((latitude, longitude), -1))
    result = torch.stack((result[..., 0], torch.remainder(result[..., 1] + 180., 360.) - 180.), -1)
    zero = (position_m == 0.).all(-1)
    exact_origin = origin_latlon + (result - result.detach())
    return torch.where(zero[..., None], exact_origin, result)


def deepaf_inputs(motion, members, *, device='cpu'):
    """Build one group from observable physical motion only; no identities/GT.

    The existing fixed095 and ordinary Mean are independently reconstructed.
    The anchor's observable interpolation/extrapolation is always available;
    helpers are available only where their ordinary support mask permits.
    """
    members = np.asarray(members)
    states = np.asarray(motion['states'], np.float64)
    if (members.ndim != 1 or not len(members) or not np.issubdtype(members.dtype, np.integer)
            or len(set(members.tolist())) != len(members) or (members < 0).any()
            or (members >= len(states)).any()):
        raise ValueError('distinct integer anchor/helper indices are required')
    members = np.r_[members[:1], np.sort(members[1:])]
    base, _, mean = sequence_inputs(motion, members, device=device)
    origin = base[0, :2].clone()
    positions = torch.as_tensor(states[members, :, :2], dtype=torch.float64, device=device)
    support = torch.as_tensor(np.asarray(motion['support'])[members], dtype=torch.bool, device=device).clone()
    helper_count = support[1:].sum(0)
    support[0] = True
    velocity = torch.as_tensor(np.asarray(motion['ols_velocity_mps'])[members[0]],
                               dtype=torch.float64, device=device)
    times = torch.as_tensor(motion['query_seconds'], dtype=torch.float64, device=device)
    if velocity.shape != (2,) or not torch.isfinite(velocity).all():
        raise ValueError('finite visible anchor OLS velocity in north/east m/s is required')
    return dict(base_states=base, mean_states=mean,
                measurements_m=positions_to_local(positions, origin[None, None, :]),
                candidate_valid=support, query_seconds=times,
                initial_velocity_mps=velocity, origin_latlon=origin,
                base_positions_m=positions_to_local(base[:, :2], origin[None, :]),
                helper_support_count=helper_count)


def deepaf_batch_inputs(rows):
    """Pad only candidate-set length; retain all groups and all their queries."""
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError('a nonempty list of DeepAF group inputs is required')
    if any(set(row) != set(INPUT_KEYS) for row in rows):
        raise ValueError('DeepAF group input fields changed')
    q = len(rows[0]['base_states'])
    device = rows[0]['base_states'].device
    if any(len(row['base_states']) != q or row['base_states'].device != device for row in rows):
        raise ValueError('batched groups require the same query length and device')
    maximum = max(row['measurements_m'].shape[0] for row in rows)
    measurements = torch.zeros((len(rows), maximum, q, 2), dtype=torch.float64, device=device)
    candidate_valid = torch.zeros((len(rows), maximum, q), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        length = row['measurements_m'].shape[0]
        measurements[i, :length] = row['measurements_m']
        candidate_valid[i, :length] = row['candidate_valid']
    result = {key: torch.stack([row[key] for row in rows]) for key in INPUT_KEYS
              if key not in ('measurements_m', 'candidate_valid')}
    result.update(measurements_m=measurements, candidate_valid=candidate_valid)
    return result


def mahalanobis_scores(query, key):
    """Paper Eq.12 exactly: 2 q^T k - q^T q - k^T k = -||q-k||^2.

    Applied to projected attention heads; no covariance parameter, clipping,
    sqrt(d) divisor, relation bias, or separately fitted temperature is added.
    """
    return (2. * torch.matmul(query, key.transpose(-2, -1))
            - query.square().sum(-1, keepdim=True)
            - key.square().sum(-1).unsqueeze(-2))


class _Attention(nn.Module):
    def __init__(self, hidden, heads, *, mahalanobis=False):
        super().__init__()
        self.heads, self.width, self.mahalanobis = heads, hidden // heads, mahalanobis
        self.query, self.key, self.value, self.output = (nn.Linear(hidden, hidden) for _ in range(4))

    def forward(self, query, memory, key_valid):
        batch = len(query)
        project = lambda layer, value: layer(value).reshape(batch, -1, self.heads, self.width).transpose(1, 2)
        q, k, v = project(self.query, query), project(self.key, memory), project(self.value, memory)
        scores = (mahalanobis_scores(q, k) if self.mahalanobis
                  else torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.width))
        weights = scores.masked_fill(~key_valid[:, None, None, :], -torch.inf).softmax(-1)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(batch, query.shape[1], -1)
        return self.output(context)


class _EncoderUnit(nn.Module):
    def __init__(self, hidden, heads, feedforward):
        super().__init__()
        self.attention = _Attention(hidden, heads)
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, feedforward), nn.ReLU(), nn.Linear(feedforward, hidden))

    def forward(self, value, valid):
        value = self.norm1(value + self.attention(value, value, valid))
        value = self.norm2(value + self.feedforward(value))
        return torch.where(valid[..., None], value, torch.zeros_like(value))


class _DecoderUnit(nn.Module):
    def __init__(self, hidden, heads, feedforward):
        super().__init__()
        self.attention = _Attention(hidden, heads)
        self.cross_attention = _Attention(hidden, heads, mahalanobis=True)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(hidden) for _ in range(3))
        self.feedforward = nn.Sequential(nn.Linear(hidden, feedforward), nn.ReLU(), nn.Linear(feedforward, hidden))

    def forward(self, value, memory, history_valid, candidate_valid):
        value = self.norm1(value + self.attention(value, value, history_valid))
        value = self.norm2(value + self.cross_attention(value, memory, candidate_valid))
        value = self.norm3(value + self.feedforward(value))
        return torch.where(history_valid[..., None], value, torch.zeros_like(value))


class _DeepAFNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, ff, h = config['hidden_size'], config['feedforward_size'], config['num_heads']
        self.candidate_embedding = nn.Sequential(nn.Linear(2, ff), nn.ReLU(), nn.Linear(ff, d))
        self.history_embedding = nn.Sequential(nn.Linear(2, ff), nn.ReLU(), nn.Linear(ff, d))
        self.encoder = nn.ModuleList([_EncoderUnit(d, h, ff) for _ in range(config['num_layers'])])
        self.decoder = nn.ModuleList([_DecoderUnit(d, h, ff) for _ in range(config['num_layers'])])
        positions = torch.arange(config['history_length'], dtype=torch.float32)[:, None]
        rates = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.) / d))
        encoding = torch.zeros(config['history_length'], d)
        encoding[:, 0::2], encoding[:, 1::2] = (positions * rates).sin(), (positions * rates).cos()
        self.register_buffer('history_encoding', encoding)
        self.output = nn.Linear(d, 2)

    def forward(self, candidates, candidate_valid, history, history_valid):
        safe_candidates = torch.where(candidate_valid[..., None], candidates, torch.zeros_like(candidates))
        safe_history = torch.where(history_valid[..., None], history, torch.zeros_like(history))
        memory = self.candidate_embedding(safe_candidates)
        for layer in self.encoder:
            memory = layer(memory, candidate_valid)
        value = self.history_embedding(safe_history) + self.history_encoding[None, :, :]
        for layer in self.decoder:
            value = layer(value, memory, history_valid, candidate_valid)
        # The newest historical estimate is always the final valid token.
        # This is a direct linear head: no learned/handcoded carry skip.
        return self.output(value[:, -1])


class DeepAFPeer(nn.Module):
    def __init__(self, *, hidden_size=32, num_layers=6, num_heads=4,
                 feedforward_size=64, history_length=5, velocity_scale_mps=10.0, dropout=0.0):
        super().__init__()
        config = dict(hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
                      feedforward_size=feedforward_size, history_length=history_length,
                      velocity_scale_mps=velocity_scale_mps, dropout=dropout)
        if config != MODEL_CONFIG or any(isinstance(value, bool) for value in config.values()):
            raise ValueError('the declared full DeepAF P/V peer configuration is fixed')
        self.configuration = config
        self.motion_observation_policy = 'canonical_all'
        self.routing_semantics = ROUTING_SEMANTICS
        self.position_network = _DeepAFNetwork(config)
        self.velocity_network = _DeepAFNetwork(config)
        # Independently initialize both networks and all depths, without
        # borrowing or freezing any of our existing position-model weights.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _validate_batch(self, batch):
        if not isinstance(batch, dict) or set(batch) != set(INPUT_KEYS):
            raise ValueError('only declared visible DeepAF inputs may enter the forward path')
        base, mean = batch['base_states'], batch['mean_states']
        if (base.ndim != 3 or base.shape[-1] != 4 or not base.shape[0] or not base.shape[1]
                or base.dtype != torch.float64 or mean.shape != base.shape or mean.dtype != torch.float64
                or not torch.isfinite(base).all() or not torch.isfinite(mean).all()
                or not torch.equal(base[..., 2:], mean[..., 2:])):
            raise ValueError('finite physical batch [B,Q,4] and exact Mean speed/course required')
        b, q, _ = base.shape
        if any(not isinstance(v, torch.Tensor) or v.device != base.device for v in batch.values()):
            raise ValueError('DeepAF inputs must be tensors on the same device')
        measurements, mask = batch['measurements_m'], batch['candidate_valid']
        if (measurements.ndim != 4 or measurements.shape[0] != b or measurements.shape[2:] != (q, 2)
                or not measurements.shape[1] or measurements.dtype != torch.float64
                or mask.shape != measurements.shape[:-1] or mask.dtype != torch.bool
                or not mask[:, 0].all() or not torch.isfinite(measurements[mask]).all()):
            raise ValueError('finite live measurements with an always-visible anchor are required')
        shapes = dict(query_seconds=(b, q), initial_velocity_mps=(b, 2), origin_latlon=(b, 2),
                      base_positions_m=(b, q, 2), helper_support_count=(b, q))
        for key, shape in shapes.items():
            value = batch[key]
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('invalid DeepAF batch field: ' + key)
            if key != 'helper_support_count' and value.dtype != torch.float64:
                raise ValueError('physical DeepAF input must be float64: ' + key)
        if q > 1 and not (batch['query_seconds'][:, 1:] > batch['query_seconds'][:, :-1]).all():
            raise ValueError('DeepAF query times must be strictly increasing')

    @torch.no_grad()
    def forward_batch(self, batch, *, cache=True):
        """Synchronously roll all B groups through every Q, with own feedback.

        Gradients deliberately stop at each update. Training replays this same
        detached cache in large independent batches; dropout is fixed to zero.
        No target, truth mask, identity, or source label is accepted here.
        """
        if type(cache) is not bool:
            raise ValueError('cache must be a boolean')
        self._validate_batch(batch)
        b, q, _ = batch['base_states'].shape
        scale, length = self.configuration['velocity_scale_mps'], self.configuration['history_length']
        times = batch['query_seconds']
        intervals = times[:, 1:] - times[:, :-1]
        initial_interval = intervals[:, :1] if q > 1 else times.new_full((b, 1), 20.)
        delta = torch.cat((initial_interval, intervals), 1)
        previous_v = batch['initial_velocity_mps'].clone()
        previous_p = -previous_v * delta[:, :1]
        history_p = torch.zeros((b, length, 2), dtype=torch.float32, device=times.device)
        history_p[:, -1] = (previous_v / scale).float()
        history_v = history_p.clone()
        history_valid = torch.zeros((b, length), dtype=torch.bool, device=times.device)
        history_valid[:, -1] = True
        valid = batch['candidate_valid']
        measurements = torch.where(valid[..., None], batch['measurements_m'],
                                   torch.zeros_like(batch['measurements_m']))
        saved = {key: [] for key in ('candidates', 'candidate_valid', 'p_history', 'v_history', 'history_valid')} if cache else None
        positions, predecessors, p_velocity, v_velocity = [], [], [], []
        for k in range(q):
            candidate_valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=times.device), valid[:, :, k]), 1)
            measured_velocity = (measurements[:, :, k] - previous_p[:, None]) / delta[:, k, None, None]
            candidates = torch.cat((previous_v[:, None], measured_velocity), 1)
            candidates = torch.where(candidate_valid[..., None], candidates, torch.zeros_like(candidates))
            candidates = (candidates / scale).float()
            if cache:
                for key, value in (('candidates', candidates), ('candidate_valid', candidate_valid),
                                   ('p_history', history_p), ('v_history', history_v), ('history_valid', history_valid)):
                    saved[key].append(value.clone())
            p_normalized = self.position_network(candidates, candidate_valid, history_p, history_valid)
            v_normalized = self.velocity_network(candidates, candidate_valid, history_v, history_valid)
            current_p_velocity, current_v = p_normalized.double() * scale, v_normalized.double() * scale
            current_p = previous_p + current_p_velocity * delta[:, k, None]
            predecessors.append(previous_p)
            positions.append(current_p)
            p_velocity.append(current_p_velocity)
            v_velocity.append(current_v)
            previous_p, previous_v = current_p, current_v
            history_p = torch.cat((history_p[:, 1:], p_normalized[:, None]), 1)
            history_v = torch.cat((history_v[:, 1:], v_normalized[:, None]), 1)
            history_valid = torch.cat((history_valid[:, 1:], torch.ones((b, 1), dtype=torch.bool, device=times.device)), 1)
        position_m = torch.stack(positions, 1)
        if not torch.isfinite(position_m).all() or not torch.isfinite(torch.stack(v_velocity, 1)).all():
            raise FloatingPointError('DeepAF closed-loop rollout produced nonfinite physical predictions')
        position = local_to_positions(position_m, batch['origin_latlon'][:, None])
        states = torch.cat((position, batch['mean_states'][..., 2:]), -1)
        result = dict(states=states, base_states=batch['base_states'], mean_states=batch['mean_states'],
                      base_positions_m=batch['base_positions_m'],
                      positions_m=position_m, previous_positions_m=torch.stack(predecessors, 1),
                      p_velocity_mps=torch.stack(p_velocity, 1), v_velocity_mps=torch.stack(v_velocity, 1),
                      residual_m=position_m - batch['base_positions_m'],
                      origin_latlon=batch['origin_latlon'], delta_seconds=delta,
                      helper_support_count=batch['helper_support_count'],
                      chart_excursion_count=(torch.linalg.vector_norm(position_m, dim=-1) >= math.pi * EARTH_RADIUS_M).sum(1))
        if cache:
            result['cache'] = {key: torch.stack(values, 1).detach() for key, values in saved.items()}
        return result

    def forward(self, inputs, *, cache=True):
        """Single-group convenience API; production trainers/evaluators batch."""
        result = self.forward_batch(deepaf_batch_inputs([inputs]), cache=cache)
        return {key: ({name: value[0] for name, value in item.items()} if key == 'cache' else item[0])
                for key, item in result.items()}

    def replay_batch(self, rollout, start=0, stop=None):
        """Recompute one B-major/Q-minor cache slice with parameter gradients."""
        if 'cache' not in rollout:
            raise ValueError('DeepAF gradient replay requires the detached rollout cache')
        cache = rollout['cache']
        if cache['candidates'].ndim != 4:
            raise ValueError('replay_batch requires a batched rollout')
        count = cache['candidates'].shape[0] * cache['candidates'].shape[1]
        stop = count if stop is None else stop
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= count:
            raise ValueError('invalid DeepAF flattened replay interval')
        values = {key: tensor.flatten(0, 1)[start:stop].detach() for key, tensor in cache.items()}
        return dict(p_normalized=self.position_network(values['candidates'], values['candidate_valid'],
                                                       values['p_history'], values['history_valid']),
                    v_normalized=self.velocity_network(values['candidates'], values['candidate_valid'],
                                                       values['v_history'], values['history_valid']))


@torch.no_grad()
def deepaf_supervision(rollout, target, valid, *, velocity_scale_mps=10.0):
    """Prepare P desired-increment and V finite-difference targets, loss only.

    P desired = (truth_k - detached predicted previous position) / actual dt.
    V uses two adjacent valid truth positions; no interpolation across invalid
    holes. At q0 the forward difference q1-q0 is used if both are valid. With
    one query there is no V target. Both vector MSEs use the same 10 m/s unit.
    """
    if (velocity_scale_mps != MODEL_CONFIG['velocity_scale_mps']
            or target.shape != rollout['states'].shape or target.ndim != 3
            or target.dtype != torch.float64 or valid.shape != target.shape or valid.dtype != torch.bool
            or target.device != rollout['states'].device or valid.device != target.device):
        raise ValueError('matching physical target/valid [B,Q,4] and fixed velocity scale required')
    mask_p = valid[..., :2].all(-1)
    if not torch.isfinite(target[..., :2][mask_p]).all():
        raise ValueError('valid DeepAF position targets must be finite')
    origin = rollout['origin_latlon'][:, None, :]
    safe_target = torch.where(mask_p[..., None], target[..., :2], origin)
    truth_m = positions_to_local(safe_target, origin)
    delta = rollout['delta_seconds']
    p_target = (truth_m - rollout['previous_positions_m'].detach()) / delta[..., None] / velocity_scale_mps
    p_target = torch.where(mask_p[..., None], p_target, torch.zeros_like(p_target))
    v_target = torch.zeros_like(p_target)
    mask_v = torch.zeros_like(mask_p)
    if target.shape[1] > 1:
        v_target[:, 1:] = (truth_m[:, 1:] - truth_m[:, :-1]) / delta[:, 1:, None] / velocity_scale_mps
        mask_v[:, 1:] = mask_p[:, 1:] & mask_p[:, :-1]
        v_target[:, 0], mask_v[:, 0] = v_target[:, 1].clone(), mask_v[:, 1].clone()
    v_target = torch.where(mask_v[..., None], v_target, torch.zeros_like(v_target))
    p_count, v_count = mask_p.sum(1), mask_v.sum(1)
    batch = len(target)
    p_weight = mask_p.double() / p_count.clamp_min(1)[:, None] / batch
    v_weight = mask_v.double() / v_count.clamp_min(1)[:, None] / batch
    return dict(p_target=p_target.flatten(0, 1), v_target=v_target.flatten(0, 1),
                p_weight=p_weight.flatten(), v_weight=v_weight.flatten(),
                p_valid=mask_p.flatten(), v_valid=mask_v.flatten(),
                valid_position_points=p_count.cpu().tolist(), valid_velocity_points=v_count.cpu().tolist())


def deepaf_objectives(prediction, supervision, *, start=0, stop=None):
    """An additive loss chunk already normalized over the complete B groups.

    Sum each chunk's ``total.backward()`` before one optimizer update; do not
    divide by B or number of chunks again. Empty-supervision chunks retain a
    zero-gradient path through every P/V parameter.
    """
    count = len(supervision['p_target'])
    stop = count if stop is None else stop
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= count:
        raise ValueError('invalid DeepAF supervised replay interval')
    p, v = prediction['p_normalized'], prediction['v_normalized']
    if p.shape != (stop - start, 2) or v.shape != p.shape or not torch.isfinite(p).all() or not torch.isfinite(v).all():
        raise ValueError('finite DeepAF P/V normalized predictions required')
    p_squared = (p.double() - supervision['p_target'][start:stop]).square().sum(-1)
    v_squared = (v.double() - supervision['v_target'][start:stop]).square().sum(-1)
    p_loss = (p_squared * supervision['p_weight'][start:stop]).sum()
    v_loss = (v_squared * supervision['v_weight'][start:stop]).sum()
    losses = dict(p_mse=p_loss, v_mse=v_loss, total=p_loss + v_loss)
    diagnostics = dict(valid_position_points=supervision['valid_position_points'],
                       valid_velocity_points=supervision['valid_velocity_points'],
                       p_sse=float((p_squared.detach() * supervision['p_valid'][start:stop]).sum()),
                       v_sse=float((v_squared.detach() * supervision['v_valid'][start:stop]).sum()))
    return losses, diagnostics
