"""Generic sequence position-recovery controls, not a named paper reproduction.

Both backbones see the same observable evidence and start at the fixed .95
physical control. Their independent north/east residuals may change every query,
including singleton and unsupported queries. Speed/course remain ordinary Mean.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from association.models.evidence_fusion import physical_errors
from association.models.support_gate import gate_inputs, mix_gate_states

SCOPE = 'sequence-position-v1'
ROUTING_SEMANTICS = 'warm095-allquery-tangent-residual-position-v1'
FEATURE_WIDTH = 76
EARTH_RADIUS_M = 6371000.
BACKBONES = ('lstm', 'transformer')


def _signed_log(value):
    return np.sign(value) * np.log1p(np.abs(value))


def _local_delta(position, origin):
    """Visible local north/east displacement in metres; no reference involved."""
    delta = np.asarray(position, np.float64) - np.asarray(origin, np.float64)
    raw = delta[..., 1].copy()
    delta[..., 1] = (raw + 180.) % 360. - 180.
    delta[..., 1][(delta[..., 1] == -180.) & (raw > 0.)] = 180.
    delta *= EARTH_RADIUS_M * np.pi / 180.
    delta[..., 1] *= np.cos(np.deg2rad((position[..., 0] + origin[..., 0]) / 2.))
    return delta


def sequence_inputs(motion, members, *, device='cpu'):
    """Return base[Q,4], visible features[Q,76], and ordinary Mean[Q,4].

    56 pooled helper features are augmented by ten anchor reliability features,
    speed/sin-course/cos-course, two Anchor-to-Mean displacements, three real-time
    features, and two anchor displacements from its first query. Fixed log/scales
    are unit conventions, never validation-fitted statistics. No absolute origin
    or identity is fed to either encoder. Helper order is canonicalized upstream.
    """
    anchor, mean, pooled, eligible = gate_inputs(motion, members, device='cpu')
    node = int(np.asarray(members)[0])
    states = anchor.numpy()
    times = np.asarray(motion['query_seconds'], np.float64)
    q = len(states)
    if (times.shape != (q,) or not np.isfinite(times).all()
            or (q > 1 and not np.all(np.diff(times) > 0))):
        raise ValueError('sequence position requires finite increasing real query seconds')
    if (np.abs(states[:, 0]) > 90.).any() or (np.abs(mean.numpy()[:, 0]) > 90.).any():
        raise ValueError('sequence position requires geographic latitude within [-90,90]')
    reliable = np.column_stack((
        np.log1p(np.asarray(motion['nearest_seconds'][node]) / 20.),
        motion['support'][node], motion['exact_observation'][node],
        np.full(q, np.log1p(motion['native_count'][node])),
        np.full(q, np.log1p(motion['span_seconds'][node] / 20.)),
        np.full(q, np.log1p(motion['fit_rmse_m'][node] / 1000.)),
        np.broadcast_to(_signed_log(np.asarray(motion['ols_velocity_mps'][node]) / 10.), (q, 2)),
        np.broadcast_to(_signed_log((np.asarray(motion['endpoint_velocity_mps'][node])
                                    - motion['ols_velocity_mps'][node]) / 10.), (q, 2))))
    course = np.deg2rad(states[:, 3])
    attributes = np.column_stack((np.log1p(states[:, 2] / 10.), np.sin(course), np.cos(course)))
    elapsed = times - times[0]
    spacing = np.r_[0., np.diff(times)]
    time_features = np.column_stack((np.log1p(elapsed / 20.), np.log1p(spacing / 20.),
                                    np.full(q, np.log1p(elapsed[-1] / 20.))))
    features = np.column_stack((pooled.numpy(), reliable, attributes,
        _signed_log(_local_delta(mean.numpy()[:, :2], states[:, :2]) / 1000.),
        time_features, _signed_log(_local_delta(states[:, :2], states[:1, :2]) / 1000.)))
    if features.shape != (q, FEATURE_WIDTH) or not np.isfinite(features).all():
        raise ValueError('sequence position evidence must be finite [Q,76]')
    # Use precisely the float32 gate/logit arithmetic of the frozen initial head.
    initial = torch.sigmoid(torch.full((q,), math.log(19.), dtype=torch.float32))
    base = mix_gate_states(anchor, mean, initial, eligible)
    return (base.to(device), torch.as_tensor(features, dtype=torch.float32, device=device), mean.to(device))


def tangent_position(base_states, residual_m):
    """Normalize ECEF base+tangent; two independent directions, not an A-M line.

    The map stays in the hemisphere around the physical base (angular correction
    approaches 90 degrees). This geometric parameterization has no hard clipping
    or narrow tanh radius. At a zero residual we preserve exact base values using
    a value-only correction, retaining the geographic map's gradient.
    """
    if (base_states.ndim != 2 or base_states.shape[-1] != 4 or base_states.dtype != torch.float64
            or residual_m.shape != (len(base_states), 2)
            or not torch.isfinite(base_states).all() or not torch.isfinite(residual_m).all()
            or (base_states[:, 0].abs() > 90.).any()):
        raise ValueError('finite geographic base[Q,4] and tangent residual[Q,2] required')
    residual_m = residual_m.to(dtype=torch.float64, device=base_states.device)
    lat, lon = torch.deg2rad(base_states[:, :2]).unbind(-1)
    slat, clat, slon, clon = lat.sin(), lat.cos(), lon.sin(), lon.cos()
    base = torch.stack((clat * clon, clat * slon, slat), -1)
    north = torch.stack((-slat * clon, -slat * slon, clat), -1)
    east = torch.stack((-slon, clon, torch.zeros_like(lon)), -1)
    vector = base + (residual_m[:, :1] * north + residual_m[:, 1:] * east) / EARTH_RADIUS_M
    vector = vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    latitude = torch.atan2(vector[:, 2], torch.linalg.vector_norm(vector[:, :2], dim=-1))
    # Longitude has no geometric meaning exactly at a pole. Choose the base
    # meridian there and keep atan2 away from (0,0), including its unused branch.
    pole = (vector[:, :2] == 0).all(-1)
    longitude = torch.atan2(torch.where(pole, slon, vector[:, 1]),
                            torch.where(pole, clon, vector[:, 0]))
    position = torch.rad2deg(torch.stack((latitude, longitude), -1))
    position = torch.stack((position[:, 0], torch.remainder(position[:, 1] + 180., 360.) - 180.), -1)
    zero = (residual_m == 0).all(-1)
    exact_with_gradient = base_states[:, :2] + (position - position.detach())
    position = torch.where(zero[:, None], exact_with_gradient, position)
    return torch.cat((position, base_states[:, 2:]), -1)


class SequencePosition(nn.Module):
    def __init__(self, *, backbone='lstm', hidden_size=64, num_layers=2, initial_gate=.95):
        super().__init__()
        if (backbone not in BACKBONES or type(hidden_size) is not int or hidden_size != 64
                or type(num_layers) is not int or num_layers != 2
                or isinstance(initial_gate, bool) or initial_gate != .95):
            raise ValueError('fixed lstm/transformer, hidden64, two layers and .95 base required')
        self.backbone = backbone
        self.motion_observation_policy = 'canonical_all'
        self.routing_semantics = ROUTING_SEMANTICS
        self.input_projection = nn.Sequential(nn.Linear(FEATURE_WIDTH, hidden_size),
                                              nn.LayerNorm(hidden_size), nn.SiLU())
        if backbone == 'lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                                   batch_first=True, dropout=0.)
        else:
            layer = nn.TransformerEncoderLayer(hidden_size, 4, dim_feedforward=256,
                dropout=0., activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.output = nn.Linear(hidden_size, 2)
        # Independently initialize cloned Transformer depths. These rules are
        # common to both arms where modules coincide; final physical outputs match.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, base_states, features, mean_states):
        if (features.shape != (len(base_states), FEATURE_WIDTH) or features.dtype != torch.float32
                or not torch.isfinite(features).all() or mean_states.shape != base_states.shape
                or mean_states.dtype != torch.float64 or not torch.isfinite(mean_states).all()
                or not torch.equal(base_states[:, 2:], mean_states[:, 2:])):
            raise ValueError('matching visible features, physical base and ordinary Mean required')
        embedded = self.input_projection(features).unsqueeze(0)
        encoded = self.encoder(embedded)
        if self.backbone == 'lstm':
            encoded = encoded[0]
        residual_m = self.output(encoded.squeeze(0)).to(torch.float64) * 1000.
        states = tangent_position(base_states, residual_m)
        if not torch.isfinite(states).all():
            raise FloatingPointError('sequence position produced nonfinite states')
        return dict(states=states, base_states=base_states, mean_states=mean_states,
                    residual_m=residual_m)


def sequence_objectives(result, target, valid):
    """Position-only losses, per-anchor mean over every valid query, not just R."""
    if target.shape != result['states'].shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('one anchor target with per-attribute validity required')
    mask = valid[:, :2].all(-1)
    if not torch.isfinite(target[:, :2][mask]).all():
        raise ValueError('valid target positions must be finite')
    safe = torch.cat((torch.where(mask[:, None], target[:, :2], torch.zeros_like(target[:, :2])),
                      torch.zeros_like(target[:, 2:])), -1)
    error = physical_errors(result['states'], safe)[:, 0] / 1000.
    if not torch.isfinite(error[mask]).all():
        raise FloatingPointError('nonfinite valid position error')
    zero = result['residual_m'].sum() * 0.
    losses = {name: value[mask].mean() + zero if mask.any() else zero
              for name, value in (('linear', error), ('log1p', torch.log1p(error)))}
    diagnostics = dict(valid_position_points=int(mask.sum()),
        linear_sum=float(error[mask].detach().sum()), log1p_sum=float(torch.log1p(error[mask]).detach().sum()))
    return losses, diagnostics
