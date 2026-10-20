"""Symmetric native-motion evidence and a residual association score.

The geometric base remains float64. The small learned network is float32; its
residual is converted to float64 before addition so zero residual reproduces the
native OLS ranking without creating float32 distance ties. Scores are not
calibrated probabilities, particularly under class-balanced training.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from association.data.motion_evidence import EARTH_RADIUS_M


FEATURE_NAMES = (
    'log_rms_distance_km', 'log_median_distance_km', 'log_p95_distance_km',
    'log_min_distance_km', 'log_centered_displacement_rms_km',
    'log_joint_support_rms_km', 'has_joint_support',
    'log_ols_velocity_difference_mps', 'log_speed_difference_rms',
    'mean_circular_heading_difference_turn',
    'log_trend_disagreement_min_mps', 'log_trend_disagreement_max_mps',
    'joint_support_fraction', 'log_nearest_mean_min_minutes',
    'log_nearest_mean_max_minutes', 'log_native_count_min', 'log_native_count_max',
    'log_span_min_minutes', 'log_span_max_minutes',
    'log_fit_rmse_min_km', 'log_fit_rmse_max_km', 'same_source',
)
GEOMETRY_FEATURE_COUNT = 7
VARIANTS = ('geometry_only', 'full_evidence')
FIXED_OLS_DISTANCE_M = 14214.358367922307
FIXED_OLS_SCORE = -float(np.log1p(FIXED_OLS_DISTANCE_M/1000.))


def pair_features(*, states, support, nearest_seconds, native_count, span_seconds,
                  fit_rmse_m, ols_velocity_mps, endpoint_velocity_mps, sources, pairs):
    """Accept only prediction/input evidence; no identity, label, or target API.

    Every feature is invariant to exchanging pair endpoints. Empty pair sets
    survive. No-overlap pairs have zero joint RMS plus an explicit absence bit.
    Relative position variation uses Earth-centered Cartesian displacement;
    velocities use the shared evidence builder's local north/east axes.
    """
    state = np.asarray(states, np.float64)
    support = np.asarray(support)
    nearest = np.asarray(nearest_seconds, np.float64)
    pairs = np.asarray(pairs)
    sources = np.asarray(sources)
    if (state.ndim != 3 or state.shape[-1] != 4 or not state.shape[1]
            or support.shape != state.shape[:2] or support.dtype != bool
            or nearest.shape != support.shape or not np.isfinite(state).all()
            or not np.isfinite(nearest).all() or (nearest < 0).any()
            or (state[..., 2] < 0).any()):
        raise ValueError('finite [N,Q,4] predictions and native-only support required')
    n = len(state)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu'
            or sources.shape != (n,) or (pairs < 0).any() or (pairs >= n).any()
            or (pairs[:, 0] == pairs[:, 1]).any()):
        raise ValueError('valid distinct-endpoint pairs and one source per node required')
    scalar = []
    for name, value in (('native_count', native_count), ('span_seconds', span_seconds),
                        ('fit_rmse_m', fit_rmse_m)):
        value = np.asarray(value, np.float64)
        if value.shape != (n,) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f'invalid {name}')
        scalar.append(value)
    velocity = np.asarray(ols_velocity_mps, np.float64)
    endpoint = np.asarray(endpoint_velocity_mps, np.float64)
    if any(value.shape != (n, 2) or not np.isfinite(value).all() for value in (velocity, endpoint)):
        raise ValueError('finite physical velocity evidence required')
    if not len(pairs):
        return np.empty((0, len(FEATURE_NAMES)), np.float64), np.empty(0, np.float64)
    # Canonicalize the arithmetic order as well as the mathematical feature:
    # circular wrapping can otherwise differ by a few float64 ulps on reversal.
    left, right = np.sort(pairs, axis=1).T
    a, b = state[left], state[right]
    lat1, lat2 = np.deg2rad(a[..., 0]), np.deg2rad(b[..., 0])
    dlon = np.deg2rad((a[..., 1]-b[..., 1]+180.) % 360.-180.)
    hav = np.clip(np.sin((lat1-lat2)/2)**2
                  + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    distance = 2*EARTH_RADIUS_M*np.arctan2(np.sqrt(hav), np.sqrt(1-hav))
    rms = np.sqrt(np.mean(distance**2, axis=1))
    radians = np.deg2rad(state[..., :2])
    xyz = EARTH_RADIUS_M*np.stack((
        np.cos(radians[..., 0])*np.cos(radians[..., 1]),
        np.cos(radians[..., 0])*np.sin(radians[..., 1]),
        np.sin(radians[..., 0])), axis=-1)
    displacement = xyz[left]-xyz[right]
    centered = displacement-displacement.mean(1, keepdims=True)
    centered_rms = np.sqrt(np.mean(np.sum(centered**2, axis=-1), axis=1))
    joint = support[left] & support[right]
    joint_count = joint.sum(1)
    joint_rms = np.sqrt((distance**2*joint).sum(1)/np.maximum(joint_count, 1))
    speed_difference = np.sqrt(np.mean((np.log1p(a[..., 2])-np.log1p(b[..., 2]))**2, axis=1))
    heading = np.abs((a[..., 3]-b[..., 3]+180.) % 360.-180.).mean(1)/180.
    columns = [np.log1p(x/1000.) for x in (
        rms, np.median(distance, axis=1), np.quantile(distance, .95, axis=1),
        distance.min(1), centered_rms, joint_rms)]
    columns += [(joint_count > 0).astype(float),
                np.log1p(np.linalg.norm(velocity[left]-velocity[right], axis=1)),
                speed_difference, heading]

    def minmax(value, scale=1.):
        columns.extend((np.log1p(np.minimum(value[left], value[right])/scale),
                        np.log1p(np.maximum(value[left], value[right])/scale)))

    minmax(np.linalg.norm(velocity-endpoint, axis=1))
    columns.append(joint_count/state.shape[1])
    minmax(nearest.mean(1), 60.)
    minmax(scalar[0])
    minmax(scalar[1], 60.)
    minmax(scalar[2], 1000.)
    columns.append((sources[left] == sources[right]).astype(float))
    features = np.column_stack(columns)
    if features.shape != (len(pairs), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise FloatingPointError('nonfinite or inconsistent pair evidence')
    return features, -np.log1p(rms/1000.)


def features_from_motion(motion, pairs):
    """Allowlist evidence fields before calling the label-free feature API."""
    return pair_features(**{key: motion[key] for key in (
        'states', 'support', 'nearest_seconds', 'native_count', 'span_seconds',
        'fit_rmse_m', 'ols_velocity_mps', 'endpoint_velocity_mps', 'sources')}, pairs=pairs)


class EvidenceAssociation(nn.Module):
    def __init__(self, variant='full_evidence'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('unknown evidence association variant')
        self.variant = variant
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        mask = torch.ones(len(FEATURE_NAMES), dtype=torch.float32)
        if variant == 'geometry_only':
            mask[GEOMETRY_FEATURE_COUNT:] = 0.
        self.register_buffer('feature_mask', mask)

    def forward(self, normalized_features, base_scores):
        if (normalized_features.ndim != 2 or normalized_features.shape[1] != len(FEATURE_NAMES)
                or base_scores.shape != (len(normalized_features),)):
            raise ValueError('feature and base-score shapes differ')
        residual = self.network(normalized_features.to(torch.float32)*self.feature_mask).squeeze(-1)
        return base_scores.to(torch.float64)+residual.to(torch.float64)
