"""Replaceable physical prediction and complete group-output emission.

Partitions are supplied. Identity, references and final correspondence stay in
the evaluator; every predicted group is emitted before any target is inspected.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.grouping import _partition
from association.evaluation.query_contribution import METHODS, _numeric, query_contribution


PREDICTION_METHODS = ('anchor', 'mean_supported', 'nearest_observation') + METHODS


def emit_group_outputs(group_states, groups, *, node_count):
    """Copy every supplied [G,Q,4] state into fixed-order prediction records.

    This boundary neither chooses groups nor assigns targets. It requires a
    complete partition and exactly one state per group, rejecting mismatches
    before emission. Returned lists and arrays do not alias the supplied inputs.
    """
    if (isinstance(node_count, (bool, np.bool_))
            or not isinstance(node_count, (int, np.integer)) or node_count < 0):
        raise ValueError('node_count must be a nonnegative integer')
    groups = _partition(groups, node_count)
    states = _numeric(group_states, 'group_states')
    if (states.ndim != 3 or states.shape[0] != len(groups)
            or states.shape[1] == 0 or states.shape[2] != 4):
        raise ValueError('finite group_states[G,Q,4] must match every group, Q>0')
    return [dict(node_indices=members, states=states[gid].copy())
            for gid, members in enumerate(groups)]


def predict_group_outputs(states, nearest_seconds, support, sources, groups, *, method):
    """Compose existing physical controls and all-group emission on explicit inputs.

    Historical controls call ordinary_fusion without changing summation or
    fallback rules, then retain the first anchor per group. Their weights are
    None: rewriting old Mean with new W/recovery operations would change
    floating-point results. The query controls retain their independently
    replaceable W and recovery interfaces. All methods emit every group.
    """
    if method not in PREDICTION_METHODS:
        raise ValueError(f'method must be one of {PREDICTION_METHODS}')
    states = _numeric(states, 'states')
    # Keep historical nearest's input precision when ordering distances. The
    # query controls perform their own established float64 conversion below.
    _numeric(nearest_seconds, 'nearest_seconds')
    nearest = np.asarray(nearest_seconds)
    support, sources = np.asarray(support), np.asarray(sources, dtype=object)
    if (states.ndim != 3 or states.shape[-1] != 4 or states.shape[1] == 0
            or nearest.shape != states.shape[:2] or (nearest < 0).any()
            or support.shape != states.shape[:2] or support.dtype != bool):
        raise ValueError('finite states[N,Q,4], nonnegative nearest_seconds[N,Q] '
                         'and boolean support[N,Q] required, Q>0')
    if (sources.shape != (len(states),)
            or any(not isinstance(source, str) or not source for source in sources)):
        raise ValueError('sources must contain one nonempty source string per node')
    groups = _partition(groups, len(states))
    if method in METHODS:
        result = query_contribution(states, nearest, support, sources, groups, method=method)
    else:
        controls = ordinary_fusion(states, nearest, support, groups)
        anchors = np.asarray([group[0] for group in groups], np.int64)
        result = dict(group_states=controls['predictions'][method][anchors].copy(),
                      weights=None, audit=dict(method=method, **controls['audit'][method]))
    return dict(result, group_outputs=emit_group_outputs(
        result['group_states'], groups, node_count=len(states)))
