"""Ordinary fixed-group controls and physical metrics shared with learned heads.

These functions are moved unchanged from the completed R4/R2 controls.
"""
from __future__ import annotations

import numpy as np
from association.evaluation.physical import evaluate_aligned_fusion

ARMS = ('anchor', 'nearest_observation', 'mean_all', 'mean_supported')


def ordinary_fusion(states, nearest_seconds, support, groups):
    """Prediction-only four-arm calculation: groups are supplied, never inferred from truth."""
    states = np.asarray(states, np.float64)
    nearest, support = np.asarray(nearest_seconds), np.asarray(support)
    n, q, width = states.shape
    if (width != 4 or nearest.shape != (n, q) or support.shape != (n, q)
            or support.dtype != bool or not np.isfinite(states).all()
            or not np.isfinite(nearest).all() or (nearest < 0).any()):
        raise ValueError('invalid complete states/visible-support arrays')
    flat = [node for group in groups for node in group]
    if sorted(flat) != list(range(n)) or any(group != sorted(group) or not group for group in groups):
        raise ValueError('fixed groups must partition every node once in input order')
    predictions = {arm: states.copy() for arm in ARMS}
    selected_nodes = np.empty((n, q), np.int64)
    helper_support = np.zeros((n, q), bool)
    membership = np.empty(n, np.int64)
    audit = {arm: dict(longitude_zero_resultant=0, heading_zero_resultant=0,
                      no_supported_member_fallback=0) for arm in ARMS}
    for gid, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        membership[members] = gid
        selected = members[np.argmin(nearest[members], axis=0)]
        for anchor in members:
            selected_nodes[anchor] = selected
            predictions['nearest_observation'][anchor] = states[selected, np.arange(q)]
            helper_support[anchor] = support[members[members != anchor]].any(0)
        for arm in ('mean_all', 'mean_supported'):
            weights = np.ones((len(members), q), bool) if arm == 'mean_all' else support[members]
            count = weights.sum(0)
            divisor = np.maximum(count, 1)
            mean = np.zeros((q, 4))
            for column in (0, 2):
                mean[:, column] = (states[members, :, column]*weights).sum(0)/divisor
            ambiguous = {}
            for column in (1, 3):
                radians = np.deg2rad(states[members, :, column])
                sine = (np.sin(radians)*weights).sum(0)/divisor
                cosine = (np.cos(radians)*weights).sum(0)/divisor
                ambiguous[column] = np.hypot(sine, cosine) <= 1e-12
                angle = np.rad2deg(np.arctan2(sine, cosine))
                mean[:, column] = (angle+180.) % 360.-180. if column == 1 else angle % 360.
            for anchor in members:
                output = mean.copy()
                output[count == 0] = states[anchor, count == 0]
                for column, name in ((1, 'longitude'), (3, 'heading')):
                    fallback = ambiguous[column] & (count > 0)
                    output[fallback, column] = states[anchor, fallback, column]
                    audit[arm][name+'_zero_resultant'] += int(fallback.sum())
                audit[arm]['no_supported_member_fallback'] += int((count == 0).sum())
                predictions[arm][anchor] = output
    return dict(predictions=predictions, selected_nodes=selected_nodes,
                helper_support=helper_support, membership=membership, audit=audit)


def attribute_errors(prediction, target):
    """Errors before masking, used only after all predictions have been fixed."""
    lat1, lat2 = np.deg2rad(prediction[..., 0]), np.deg2rad(target[..., 0])
    dlon = np.deg2rad((prediction[..., 1]-target[..., 1]+180.) % 360.-180.)
    a = np.clip(np.sin((lat1-lat2)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    return dict(position=2*6371000.*np.arctan2(np.sqrt(a), np.sqrt(1-a)),
                speed=np.abs(prediction[..., 2]-target[..., 2]),
                heading=np.abs((prediction[..., 3]-target[..., 3]+180.) % 360.-180.))


def metrics(prediction, target, valid, region):
    result = evaluate_aligned_fusion(prediction, target, valid,
        prediction_valid=np.ones(prediction.shape, bool), region_mask=region)
    errors = attribute_errors(prediction, target)
    for name, columns in (('position', [0, 1]), ('speed', [2]), ('heading', [3])):
        mask = valid[..., columns].all(-1) & region
        selected = errors[name][mask]
        result[name]['p95'] = float(np.percentile(selected, 95)) if len(selected) else None
    return {name: result[name] for name in ('position', 'speed', 'heading')}


def rms_haversine_scores(positions, pairs):
    """Negative RMS metres over every Q64 time; accepts no identity or masks."""
    positions = np.asarray(positions, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if positions.ndim != 3 or positions.shape[1:] != (64, 2) or not np.isfinite(positions).all():
        raise ValueError('positions must be finite [N,64,2]')
    if pairs.shape != (len(pairs), 2):
        raise ValueError('pairs must be [E,2]')
    first, second = positions[pairs[:, 0]], positions[pairs[:, 1]]
    lat1, lat2 = np.deg2rad(first[..., 0]), np.deg2rad(second[..., 0])
    dlon = np.deg2rad((first[..., 1]-second[..., 1]+180.) % 360.-180.)
    a = np.clip(np.sin((lat1-lat2)/2)**2
                + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    distances = 2*6371000.*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return -np.sqrt(np.mean(distances**2, axis=1))
