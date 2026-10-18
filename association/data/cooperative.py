"""Candidate-only representations; physical datasets and labels stay unchanged."""
from __future__ import annotations

import math

import numpy as np

from association.training.common import fit_normalizer, normalize_scene


REPRESENTATION_VERSION = 'support_residual_v2'


def validate_cooperative_representation(normalizer, model_config):
    representation = normalizer.get('cooperative_representation')
    if model_config.model_variant == 'legacy':
        if representation is not None:
            raise ValueError('legacy model cannot consume a cooperative v2 normalizer')
        return None
    if not isinstance(representation, dict) or representation.get('version') != REPRESENTATION_VERSION:
        raise ValueError('support_residual_v2 requires its frozen cooperative representation')
    for key in ('log_speed_mean', 'log_speed_std', 'speed_bias', 'typical_speed'):
        value = representation.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or abs(value) > np.finfo(np.float32).max):
            raise ValueError(f'cooperative representation needs finite float32 {key}')
    count = representation.get('observed_point_count')
    if (isinstance(count, bool) or not isinstance(count, int) or count <= 0
            or count != normalizer.get('observed_point_count')):
        raise ValueError('cooperative representation population differs from native normalizer')
    if (representation['log_speed_mean'] < 0 or representation['log_speed_std'] <= 0
            or representation['typical_speed'] <= 0
            or not isinstance(representation.get('log_speed_constant'), bool)):
        raise ValueError('invalid log-speed population statistics')
    try:
        expected_speed = math.expm1(representation['log_speed_mean'])
    except OverflowError as error:
        raise ValueError('log-speed initialization exceeds the physical numeric range') from error
    expected_speed = expected_speed if expected_speed > 0 else 1e-6
    expected_bias = expected_speed + math.log(-math.expm1(-expected_speed))
    if (not math.isclose(representation['typical_speed'], expected_speed, rel_tol=1e-12, abs_tol=1e-12)
            or not math.isclose(representation['speed_bias'], expected_bias, rel_tol=1e-12, abs_tol=1e-12)
            or (representation['log_speed_constant'] and representation['log_speed_std'] != 1.)):
        raise ValueError('speed initialization disagrees with its training log-speed statistics')
    return representation


def fit_cooperative_normalizer(train_scenes, model_config, check=None):
    """Fit on already assembled T_core native tokens, never targets or V."""
    scenes = list(train_scenes)
    if check is not None:
        check()
    if model_config.model_variant == 'legacy':
        return fit_normalizer(scenes)
    if not scenes or any(scene.get('split') != 'T_core' for scene in scenes):
        raise ValueError('cooperative v2 representation requires T_core scenes only')
    normalizer = fit_normalizer(scenes)
    count, mean, m2 = 0, 0., 0.
    for scene in scenes:
        if check is not None:
            check()
        speed = np.asarray(scene['native_values'], dtype=np.float64)[scene['native_observed'], 2]
        if not np.isfinite(speed).all() or (speed < 0).any():
            raise ValueError('native speeds must be finite and nonnegative before log1p')
        log_speed = np.log1p(speed)
        if not len(log_speed):
            continue
        local_mean = float(log_speed.mean())
        local_m2 = float(np.square(log_speed-local_mean).sum())
        combined = count + len(log_speed)
        delta = local_mean-mean
        m2 += local_m2 + delta*delta*(count*len(log_speed)/combined)
        mean += delta*(len(log_speed)/combined)
        count = combined
    if count == 0:
        raise ValueError('no visible native speed population')
    std = math.sqrt(max(m2/count, 0.))
    constant = std < 1e-6
    typical = math.expm1(mean)
    typical = typical if typical > 0 else 1e-6
    bias = typical + math.log(-math.expm1(-typical))
    normalizer['cooperative_representation'] = dict(
        version=REPRESENTATION_VERSION, log_speed_mean=mean,
        log_speed_std=1. if constant else std, log_speed_constant=constant,
        speed_bias=bias, typical_speed=typical, observed_point_count=count)
    validate_cooperative_representation(normalizer, model_config)
    return normalizer


def normalize_cooperative_scene(scene, normalizer, model_config):
    """Keep [N,M+Q,5]; v2's visible speed slot alone becomes z_logspeed."""
    representation = validate_cooperative_representation(normalizer, model_config)
    values, observed = normalize_scene(scene, normalizer)
    if representation is None:
        return values, observed
    if scene['max_observations'] != model_config.max_observations or scene['length'] != model_config.length:
        raise ValueError('cooperative model width differs from the native scene contract')
    native_mask = np.asarray(scene['native_observed'])
    speed = np.asarray(scene['native_values'], dtype=np.float64)[native_mask, 2]
    if not np.isfinite(speed).all() or (speed < 0).any():
        raise ValueError('native speeds must be finite and nonnegative before log1p')
    transformed = (np.log1p(speed)-representation['log_speed_mean'])/representation['log_speed_std']
    if not np.isfinite(transformed).all() or (np.abs(transformed) > np.finfo(np.float32).max).any():
        raise ValueError('log-speed representation exceeds finite float32 range')
    values[:, :model_config.max_observations, 2][native_mask] = transformed.astype(np.float32)
    return values, observed
