"""Truth-blind query contributions and physical recovery on supplied partitions.

These are new controls. The historical ordinary Mean calculation is unchanged.
Temporal support and source balancing do not assert calibrated measurement quality.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.grouping import _partition


METHODS = ('source_balanced_supported', 'nearest_supported')


def _numeric(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in 'iuf' or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain finite numeric values')
    return result.astype(np.float64, copy=False)


def supported_query_weights(nearest_seconds, support, sources, groups, *, method):
    """Construct [N,Q] weights using only visible support, time and source equality.

    Supported sources have equal total mass under source_balanced_supported.
    nearest_supported copies only a supported member, breaking ties by original
    node index. Unsupported queries have zero mass and are recovered separately.
    Audit contains counts only; source names and member indices are not copied.
    """
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    support = np.asarray(support)
    nearest = _numeric(nearest_seconds, 'nearest_seconds')
    sources = np.asarray(sources, dtype=object)
    if (support.ndim != 2 or support.shape[1] == 0 or support.dtype != bool
            or nearest.shape != support.shape or (nearest < 0).any()):
        raise ValueError('finite nonnegative nearest_seconds and boolean support must be [N,Q], Q>0')
    n, q = support.shape
    if (sources.shape != (n,)
            or any(not isinstance(source, str) or not source for source in sources)):
        raise ValueError('sources must contain one nonempty source string per node')
    groups = _partition(groups, n)
    weights = np.zeros((n, q), np.float64)
    count_shape = (len(groups), q)
    member_count, source_count, minimum, maximum = [np.zeros(count_shape, np.int64) for _ in range(4)]
    queries = np.arange(q)
    for gid, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        visible = support[members]
        member_count[gid] = visible.sum(axis=0)
        # Keep original order; source strings define equality, never priority.
        source_members = [members[sources[members] == source] for source in dict.fromkeys(sources[members])]
        counts = np.stack([support[selected].sum(axis=0) for selected in source_members])
        source_count[gid] = (counts > 0).sum(axis=0)
        maximum[gid] = counts.max(axis=0)
        minimum[gid] = np.where(counts > 0, counts, len(members) + 1).min(axis=0)
        available = member_count[gid] > 0
        minimum[gid, ~available] = 0
        if method == 'source_balanced_supported':
            for selected, count in zip(source_members, counts):
                divisor = np.maximum(source_count[gid] * count, 1)
                weights[selected] = support[selected] / divisor
        else:
            selected = members[np.argmin(np.where(visible, nearest[members], np.inf), axis=0)]
            weights[selected[available], queries[available]] = 1.
    audit = dict(method=method, supported_member_count=member_count.tolist(),
                 supported_source_count=source_count.tolist(),
                 min_supported_members_per_source=minimum.tolist(),
                 max_supported_members_per_source=maximum.tolist(),
                 unequal_source_member_counts=(minimum != maximum).tolist())
    return dict(weights=weights, audit=audit)


def recover_group_states(states, weights, groups):
    """Recover [G,Q,4] physical states from replaceable nonnegative [N,Q] mass.

    Zero total mass copies the first group member's entire query state. Circular
    zero resultants copy that anchor's affected angle. A single contributing
    member is copied exactly, including its existing longitude representation.
    No identity, reference, source, support or target-validity input is accepted.
    """
    states, weights = _numeric(states, 'states'), _numeric(weights, 'weights')
    if (states.ndim != 3 or states.shape[-1] != 4 or states.shape[1] == 0
            or weights.shape != states.shape[:2] or (weights < 0).any()):
        raise ValueError('finite states[N,Q,4] and nonnegative weights[N,Q] with Q>0 required')
    n, q, _ = states.shape
    groups = _partition(groups, n)
    output = np.empty((len(groups), q, 4), np.float64)
    audit = {key: [] for key in ('zero_weight_fallback_count', 'longitude_zero_resultant_count',
                                'heading_zero_resultant_count', 'single_member_copy_count')}
    for gid, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        values, mass = states[members], weights[members]
        with np.errstate(over='ignore'):
            total = mass.sum(axis=0)
        if not np.isfinite(total).all():
            raise ValueError('group weight totals must be finite')
        available = total > 0
        normalized = np.divide(mass, total, out=np.zeros_like(mass), where=available)
        prediction = values[0].copy()
        for column in (0, 2):
            prediction[available, column] = (values[:, :, column] * normalized).sum(axis=0)[available]
        ambiguous = {}
        for column, name in ((1, 'longitude'), (3, 'heading')):
            radians = np.deg2rad(values[:, :, column])
            sine = (np.sin(radians) * normalized).sum(axis=0)
            cosine = (np.cos(radians) * normalized).sum(axis=0)
            ambiguous[column] = available & (np.hypot(sine, cosine) <= 1e-12)
            use = available & ~ambiguous[column]
            angle = np.rad2deg(np.arctan2(sine, cosine))
            angle = (angle + 180.) % 360. - 180. if column == 1 else angle % 360.
            prediction[use, column] = angle[use]
            audit[name + '_zero_resultant_count'].append(int(ambiguous[column].sum()))
        single = (mass > 0).sum(axis=0) == 1
        selected = np.argmax(mass, axis=0)
        prediction[single] = values[selected[single], np.flatnonzero(single)]
        output[gid] = prediction
        audit['zero_weight_fallback_count'].append(int((~available).sum()))
        audit['single_member_copy_count'].append(int(single.sum()))
    if not np.isfinite(output).all():
        raise FloatingPointError('query contribution produced nonfinite physical states')
    return dict(group_states=output, audit=audit)


def query_contribution(states, nearest_seconds, support, sources, groups, *, method):
    """Compose replaceable W construction and physical recovery before evaluation."""
    contribution = supported_query_weights(nearest_seconds, support, sources, groups, method=method)
    recovered = recover_group_states(states, contribution['weights'], groups)
    return dict(group_states=recovered['group_states'], weights=contribution['weights'],
                audit=dict(contribution['audit'], **recovered['audit']))
