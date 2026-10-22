"""Three fixed position controls using only visible native positions and times.

All controls share interpolation inside the original measured endpoint span.
Outside it they hold the original endpoint, or extend that endpoint with an
endpoint/ordinary-least-squares slope. This is a full-fragment offline control,
not a causal predictor. No identity, source, reference, mask or fitted threshold
from the evaluation target is accepted by the prediction function.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.support import build_input_support


METHODS = ('native_hold', 'native_endpoint', 'native_ols')
REGIONS = ('all', 'inside_support', 'outside_support', 'before_support',
           'after_support', 'exact_observation', 'inside_nonobserved')


def _short_delta(delta):
    """Shortest signed change; an exact half turn retains the raw sign."""
    result = (delta + 180.) % 360. - 180.
    return np.where(np.abs(result) == 180., np.copysign(180., delta), result)


def predict_native_positions(native_values, native_times, native_observed, query_seconds):
    """Return fixed float64 position controls on every supplied query.

    Inputs are numeric [N,M,2 or 4], real-second [N,M], a boolean prefix mask,
    and strictly increasing [Q] seconds. Every fragment needs >=2 visible
    native points, including fragments with zero or one in-support query.
    Padding and non-position state columns are ignored. Longitude interpolation
    follows each shortest arc; slopes use its cumulative signed unwrapping.
    Extrapolation is anchored at the measured first/last point, never at the
    fitted intercept. Latitude saturation is reported, not used to drop rows.
    """
    values = np.asarray(native_values)
    times = np.asarray(native_times)
    visible = np.asarray(native_observed)
    query = np.asarray(query_seconds)
    if (values.ndim != 3 or values.shape[-1] not in (2, 4)
            or values.dtype.kind not in 'iuf' or times.dtype.kind not in 'iuf'
            or times.shape != values.shape[:2] or visible.shape != times.shape
            or visible.dtype != np.bool_):
        raise ValueError('native positions/times/prefix mask must have compatible [N,M] shapes')
    sizes = visible.sum(1)
    if ((sizes < 2).any()
            or not np.array_equal(visible, np.arange(values.shape[1])[None, :] < sizes[:, None])):
        raise ValueError('each native fragment requires a visible prefix of at least two points')
    if (query.ndim != 1 or query.dtype.kind not in 'iuf' or not len(query)
            or not np.isfinite(query).all() or (np.diff(query) <= 0).any()):
        raise ValueError('query seconds must be nonempty, finite and strictly increasing')
    positions = values[..., :2].astype(np.float64, copy=False)
    if (not np.isfinite(positions[visible]).all() or not np.isfinite(times[visible]).all()
            or (np.abs(positions[..., 0][visible]) > 90.).any()
            or (np.abs(positions[..., 1][visible]) > 180.).any()):
        raise ValueError('visible positions and times must be finite with legal latitude/longitude')
    times, query = times.astype(np.float64, copy=False), query.astype(np.float64, copy=False)
    n, q = len(values), len(query)
    support = build_input_support(query, times, visible)
    inside = support['support']
    before, after = np.zeros((n, q), bool), np.zeros((n, q), bool)
    hold = np.empty((n, q, 2), dtype=np.float64)
    endpoint_slope, ols_slope = np.empty((n, 2)), np.empty((n, 2))
    half_turns = np.zeros(n, dtype=np.int64)
    for node, size in enumerate(sizes):
        t, xy = times[node, :size], positions[node, :size]
        if (np.diff(t) <= 0).any():
            raise ValueError('visible native times must be strictly increasing')
        before[node], after[node] = query < t[0], query > t[-1]
        hold[node, before[node]], hold[node, after[node]] = xy[0], xy[-1]
        # Same float64 interpolation for all three arms, including exact copies.
        for index in np.flatnonzero(inside[node]):
            right = int(np.searchsorted(t, query[index]))
            if t[right] == query[index]:
                hold[node, index] = xy[right]
            else:
                left = right-1
                fraction = (query[index]-t[left])/(t[right]-t[left])
                hold[node, index, 0] = xy[left, 0] + fraction*(xy[right, 0]-xy[left, 0])
                lon = xy[left, 1] + fraction*float(_short_delta(xy[right, 1]-xy[left, 1]))
                hold[node, index, 1] = (lon+180.) % 360.-180.
        raw_delta = np.diff(xy[:, 1])
        delta = _short_delta(raw_delta)
        half_turns[node] = (np.abs(delta) == 180.).sum()
        unwrapped = xy.copy()
        unwrapped[:, 1] = xy[0, 1] + np.r_[0., np.cumsum(delta)]
        endpoint_slope[node] = (unwrapped[-1]-unwrapped[0])/(t[-1]-t[0])
        centred_t = t-t.mean()
        ols_slope[node] = (centred_t[:, None]*(unwrapped-unwrapped.mean(0))).sum(0)/(centred_t @ centred_t)
    predictions = {METHODS[0]: hold}
    clamps = {METHODS[0]: np.zeros(n, dtype=np.int64)}
    for name, slope in zip(METHODS[1:], (endpoint_slope, ols_slope)):
        result = hold.copy()
        for node, size in enumerate(sizes):
            # Signed physical seconds relative to the original measured endpoint.
            result[node, before[node]] += (query[before[node]]-times[node, 0])[:, None]*slope[node]
            result[node, after[node]] += (query[after[node]]-times[node, size-1])[:, None]*slope[node]
        outside = ~inside
        clamps[name] = ((np.abs(result[..., 0]) > 90.) & outside).sum(1)
        result[..., 0][outside] = np.clip(result[..., 0][outside], -90., 90.)
        result[..., 1][outside] = (result[..., 1][outside]+180.) % 360.-180.
        if not np.isfinite(result).all():
            raise FloatingPointError('native motion control produced nonfinite positions')
        predictions[name] = result
    exact = support['nearest_seconds'] == 0
    regions = dict(zip(REGIONS, (np.ones((n, q), bool), inside, ~inside, before, after,
                                exact, inside & ~exact)))
    return dict(predictions=predictions, regions=regions,
        nearest_seconds=support['nearest_seconds'], native_point_counts=sizes,
        endpoint_slope_degrees_per_second=endpoint_slope,
        ols_slope_degrees_per_second=ols_slope,
        latitude_clamp_counts=clamps, native_half_turn_counts=half_turns)


def position_metrics(prediction, target, target_valid, region):
    """Reuse the repository's physical metric, declaring position-only output."""
    prediction = np.asarray(prediction, dtype=np.float64)
    shape = prediction.shape[:2]+(4,)
    states = np.full(shape, np.nan)
    states[..., :2] = prediction
    available = np.zeros(shape, bool)
    available[..., :2] = True
    return evaluate_aligned_fusion(states, target, target_valid,
        prediction_valid=available, region_mask=region)['position']
