"""Describe actual inference inputs on physical query times, without labels.

Support is distinct from distance: a query inside a fragment's endpoint span
can still be far from every real visible observation. This module only reports
the supplied inputs; it does not repair predicted groups or change predictions.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np


def build_input_support(query_times, node_times, node_visible, *, node_support=None):
    """Return per-node ``support`` and nearest real-point ``nearest_seconds``.

    Times are in seconds, not encoded tau. Queries may be in any order. Only
    visible native slots are read; padding may contain NaN. The default support
    is the closed min/max visible-time span. An explicit boolean [N,Q] support
    mask can describe aligned controls, but a node without visible input cannot
    be marked supported. Such nodes have infinite distance at every query.
    """
    query = np.asarray(query_times)
    times = np.asarray(node_times)
    visible = np.asarray(node_visible)
    if query.ndim != 1 or query.dtype.kind not in 'iuf' or not np.isfinite(query).all():
        raise ValueError('query_times must be finite numeric [Q] seconds')
    if times.ndim != 2 or times.dtype.kind not in 'iuf':
        raise ValueError('node_times must be numeric [N,M] seconds')
    if visible.shape != times.shape or visible.dtype != np.bool_:
        raise ValueError('node_visible must be boolean [N,M]')
    query = query.astype(np.float64, copy=False)
    count = len(times)
    supported = np.zeros((count, len(query)), dtype=bool)
    distance = np.full(supported.shape, np.inf, dtype=np.float64)
    has_input = visible.any(axis=1)
    for node in np.flatnonzero(has_input):
        # Index before conversion/arithmetic so hidden NaN slots remain unused.
        observed = np.asarray(times[node, visible[node]], dtype=np.float64)
        if not np.isfinite(observed).all():
            raise ValueError('visible node times must be finite')
        observed = np.sort(observed)
        supported[node] = (query >= observed[0]) & (query <= observed[-1])
        right = np.searchsorted(observed, query, side='left')
        before = observed[np.maximum(right-1, 0)]
        after = observed[np.minimum(right, len(observed)-1)]
        with np.errstate(over='ignore'):
            nearest = np.minimum(np.abs(query-before), np.abs(query-after))
        if not np.isfinite(nearest).all():
            raise ValueError('visible-point distances exceed finite seconds range')
        distance[node] = nearest
    if node_support is not None:
        supplied = np.asarray(node_support)
        if supplied.shape != supported.shape or supplied.dtype != np.bool_:
            raise ValueError('node_support must be boolean [N,Q]')
        if supplied[~has_input].any():
            raise ValueError('a node without visible input cannot have support')
        supported = supplied.copy()
    return dict(support=supported, nearest_seconds=distance)


def _indices(values, count, name):
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f'{name} must contain ordered node-index sequences')
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f'{name} node indices must be one-dimensional')
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not 0 <= value < count:
            raise ValueError(f'{name} requires valid integer node indices')
        result.append(int(value))
    if len(result) != len(set(result)):
        raise ValueError(f'{name} cannot repeat a node within one group')
    return result


def group_input_support(prepared, groups, *, eligible_groups=None):
    """Aggregate actual group members into disjoint input-support regions.

    ``members[0]`` is the anchor. Eligible subsets select the inputs a control
    actually used; they never choose a new anchor. Helpers exclude the anchor.
    A helper's support is ``helper_only_support`` only where anchor support is
    absent. Distance is the nearest real visible point of any eligible member,
    including outside its support span. No identity labels are read. Empty
    groups or empty eligible subsets remain unsupported with infinite distance.
    """
    if not isinstance(prepared, Mapping) or not {'support', 'nearest_seconds'}.issubset(prepared):
        raise ValueError('prepared input support requires support and nearest_seconds')
    support = np.asarray(prepared['support'])
    distance = np.asarray(prepared['nearest_seconds'])
    if support.ndim != 2 or support.dtype != np.bool_:
        raise ValueError('prepared support must be boolean [N,Q]')
    if (distance.shape != support.shape or distance.dtype.kind not in 'iuf'
            or np.isnan(distance).any() or (distance < 0).any()
            or not np.isfinite(distance[support]).all()):
        raise ValueError('prepared nearest_seconds must be nonnegative [N,Q], finite on support')
    if not isinstance(groups, (list, tuple)):
        raise ValueError('groups must be an ordered list of node-index groups')
    if eligible_groups is not None and (not isinstance(eligible_groups, (list, tuple))
                                       or len(eligible_groups) != len(groups)):
        raise ValueError('eligible_groups must match the ordered group population')
    shape = (len(groups), support.shape[1])
    anchor_support = np.zeros(shape, dtype=bool)
    helper_only_support = np.zeros(shape, dtype=bool)
    nearest_seconds = np.full(shape, np.inf, dtype=np.float64)
    for row, values in enumerate(groups):
        members = _indices(values, len(support), 'groups')
        eligible = members if eligible_groups is None else _indices(
            eligible_groups[row], len(support), 'eligible_groups')
        if not set(eligible).issubset(members):
            raise ValueError('eligible inputs must be members of their original group')
        if not eligible:
            continue
        anchor = members[0]
        if anchor in eligible:
            anchor_support[row] = support[anchor]
        helpers = [node for node in eligible if node != anchor]
        if helpers:
            helper_only_support[row] = support[helpers].any(axis=0) & ~anchor_support[row]
        nearest_seconds[row] = distance[eligible].min(axis=0)
    unsupported = ~(anchor_support | helper_only_support)
    return dict(anchor_support=anchor_support, helper_only_support=helper_only_support,
                unsupported=unsupported, nearest_seconds=nearest_seconds)
