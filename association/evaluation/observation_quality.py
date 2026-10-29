"""Offline, reference-free predictability proxies and source contributions.

These proxies include motion/model error and cannot detect constant translation
bias in a fixed local chart. They are not calibrated sensor covariances.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.grouping import _partition
from association.evaluation.native_motion import _short_delta
from association.evaluation.query_contribution import recover_group_states
from association.evaluation.source_risk import _finite, _offsets

NATIVE_KEYS = ('node_ids', 'sources', 'query_seconds', 'event_offsets', 'event_times_seconds',
               'event_values', 'event_valid', 'event_node_ids', 'event_canonical_indices')
RADIUS = 6371000.


def native_quality(*, native_cache, motion):
    """Per-fragment full OLS residual and neighboring-event interpolation proxy."""
    if set(native_cache) != set(NATIVE_KEYS):
        raise ValueError('native cache must contain physical-only canonical schema')
    for key in ('node_ids', 'sources', 'query_seconds'):
        if not np.array_equal(native_cache[key], motion[key]):
            raise ValueError('native/motion binding differs: '+key)
    n = len(native_cache['node_ids'])
    times = _finite(native_cache['event_times_seconds'], 'native times')
    values = np.asarray(native_cache['event_values'])
    valid = np.asarray(native_cache['event_valid'])
    offsets = np.asarray(native_cache['event_offsets'])
    event_nodes = np.asarray(native_cache['event_node_ids'])
    canonical = np.asarray(native_cache['event_canonical_indices'])
    if (times.ndim != 1 or values.shape != (len(times), 4) or valid.shape != values.shape
            or valid.dtype != bool or offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != len(times) or (np.diff(offsets) < 2).any()
            or event_nodes.shape != times.shape or canonical.shape != times.shape
            or canonical.dtype.kind not in 'iu' or np.shape(native_cache['sources']) != (n,)):
        raise ValueError('complete ragged native population required')
    positions = _finite(values[:, :2], 'native positions')
    if (not valid[:, :2].all() or (np.abs(positions[:, 0]) > 90).any()
            or (np.abs(positions[:, 1]) > 180).any()):
        raise ValueError('canonical native positions must all be valid')
    rmse = _finite(motion['fit_rmse_m'], 'fit_rmse_m')
    if rmse.shape != (n,) or (rmse < 0).any():
        raise ValueError('nonnegative per-node OLS RMSE required')
    ols = rmse**2
    if not np.isfinite(ols).all():
        raise ValueError('OLS proxy overflow')
    loo, known, counts = np.zeros(n), np.zeros(n, bool), np.zeros(n, np.int64)
    for node in range(n):
        lo, hi = int(offsets[node]), int(offsets[node+1])
        t, pos = times[lo:hi], positions[lo:hi]
        if ((np.diff(t) <= 0).any()
                or not np.all(event_nodes[lo:hi] == native_cache['node_ids'][node])
                or not np.array_equal(canonical[lo:hi], np.arange(hi-lo))):
            raise ValueError('canonical time/order/lineage invalid; no event deletion allowed')
        if len(t) == 2:
            continue
        unwrapped_lon = pos[0, 1]+np.r_[0., np.cumsum(_short_delta(np.diff(pos[:, 1])))]
        xy = np.column_stack((np.deg2rad(pos[:, 0]-pos[0, 0])*RADIUS,
            np.deg2rad(unwrapped_lon-unwrapped_lon[0])*RADIUS*np.cos(np.deg2rad(pos[:, 0].mean()))))
        fraction = (t[1:-1]-t[:-2])/(t[2:]-t[:-2])
        prediction = (1.-fraction[:, None])*xy[:-2]+fraction[:, None]*xy[2:]
        residual2 = np.sum((xy[1:-1]-prediction)**2, axis=1)
        gain = 1.+(1.-fraction)**2+fraction**2
        loo[node] = np.mean(residual2/gain)
        known[node], counts[node] = True, len(t)-2
    if not np.isfinite(loo).all():
        raise ValueError('LOO proxy nonfinite')
    return dict(ols_proxy_m2=ols, loo_proxy_m2=loo, loo_known=known, loo_residual_count=counts,
                native_event_count=int(len(times)), native_fragment_count=n)


def _structure(support, pack):
    support = np.asarray(support)
    available = np.asarray(pack['available'])
    mapping = np.asarray(pack['node_to_source_row'])
    counts = np.asarray(pack['supported_counts'])
    if (support.ndim != 2 or not support.shape[1] or support.dtype != bool
            or available.ndim != 2 or available.shape[1] != support.shape[1] or available.dtype != bool
            or mapping.shape != (len(support),) or mapping.dtype.kind not in 'iu'
            or (mapping < 0).any() or (mapping >= len(available)).any()
            or counts.shape != available.shape or counts.dtype.kind not in 'iu' or (counts < 0).any()):
        raise ValueError('source support/mapping/count dimensions invalid')
    offsets = _offsets(pack['group_offsets'], len(available))
    actual = np.zeros_like(counts)
    np.add.at(actual, mapping, support.astype(np.int64))
    if not np.array_equal(actual, counts) or not np.array_equal(available, counts > 0):
        raise ValueError('source support/mapping count binding differs')
    return support, available, mapping, counts, offsets


def source_proxy(*, node_proxy_m2, node_known, support, pack):
    support, available, mapping, counts, _ = _structure(support, pack)
    quality, known = _finite(node_proxy_m2, 'node_proxy_m2'), np.asarray(node_known)
    if quality.shape != (len(support),) or (quality < 0).any() or known.shape != quality.shape or known.dtype != bool:
        raise ValueError('nonnegative per-node quality and boolean known required')
    summed, known_count = np.zeros(available.shape), np.zeros(available.shape, np.int64)
    np.add.at(summed, mapping, quality[:, None]*support)
    np.add.at(known_count, mapping, known[:, None]*support)
    # Mean of fragment proxies; not a covariance reduction by independent n.
    result = summed/np.maximum(counts, 1)
    result[~available] = 0.
    return dict(quality_m2=result, known=available & (known_count == counts), available=available.copy())


def observation_weights(*, proxy, pack, nearest_seconds, support, kind, floor_m2=None):
    support, available, mapping, counts, offsets = _structure(support, pack)
    nearest = _finite(nearest_seconds, 'nearest_seconds')
    if nearest.shape != support.shape or (nearest < 0).any() or kind not in ('equal', 'nearest', 'inverse'):
        raise ValueError('valid nearest and equal/nearest/inverse policy required')
    quality = _finite(proxy['quality_m2'], 'source proxy')
    known = np.asarray(proxy['known'])
    if (quality.shape != available.shape or (quality < 0).any() or known.shape != available.shape
            or known.dtype != bool or not np.array_equal(proxy['available'], available)):
        raise ValueError('proxy source population differs')
    if kind == 'inverse' and floor_m2 not in (1., 100.):
        raise ValueError('fixed floor 1 or 100 m2 required')
    source_nearest = np.full(available.shape, np.inf)
    np.minimum.at(source_nearest, mapping, np.where(support, nearest, np.inf))
    weight = np.zeros(available.shape)
    fallback = np.zeros((len(offsets)-1, support.shape[1]), bool)
    floor_hit = np.zeros(available.shape, bool)
    for gid, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        present = available[start:stop]
        if kind == 'equal':
            mass = present.astype(float)
        elif kind == 'nearest':
            value = source_nearest[start:stop]
            mass = (present & (value == np.min(value, axis=0))).astype(float)
        else:
            missing = (present & ~known[start:stop]).any(0)
            fallback[gid] = missing
            mass = present/np.maximum(quality[start:stop], floor_m2)
            mass[:, missing] = present[:, missing]
            floor_hit[start:stop] = present & known[start:stop] & (quality[start:stop] < floor_m2) & ~missing
        total = mass.sum(0)
        weight[start:stop] = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    node_weight = weight[mapping]*support/np.maximum(counts[mapping], 1)
    if not np.isfinite(weight).all():
        raise ValueError('contribution nonfinite')
    return dict(source_weights=weight, node_weights=node_weight, floor_hit=floor_hit,
                uninformative_fallback=fallback)


def recover_observation_policy(*, states, support, groups, pack, source_weights, mean_states):
    states, mean = _finite(states, 'states'), _finite(mean_states, 'mean_states')
    support, available, mapping, counts, offsets = _structure(support, pack)
    weights = _finite(source_weights, 'source_weights')
    groups = _partition(groups, len(states))
    if (states.shape != (*support.shape, 4) or mean.shape != (len(groups), support.shape[1], 4)
            or len(offsets) != len(groups)+1 or weights.shape != available.shape
            or (weights < 0).any() or np.any(weights[~available] != 0)):
        raise ValueError('state/contribution/group binding invalid')
    for gid, group in enumerate(groups):
        rows = mapping[group]
        if not np.all((rows >= offsets[gid]) & (rows < offsets[gid+1])):
            raise ValueError('source mapping crosses supplied group')
        totals = weights[offsets[gid]:offsets[gid+1]].sum(0)
        expected = support[group].any(0).astype(float)
        if not np.allclose(totals, expected, rtol=0., atol=1e-12):
            raise ValueError('source masses must sum to one when supported')
    node_weights = weights[mapping]*support/np.maximum(counts[mapping], 1)
    result = recover_group_states(states, node_weights, groups)
    out = result['group_states']
    for gid, group in enumerate(groups):
        no_support = ~support[group].any(0)
        out[gid, no_support] = mean[gid, no_support]
    out[..., 2:] = mean[..., 2:]
    return dict(group_states=out, node_weights=node_weights, source_weights=weights,
                recovery_audit=result['audit'])
