"""Temporal measurement descriptors and unsupported candidate readout.

No identity/reference input. The OLS scale is a working sensitivity proxy under
linear IID assumptions, never a calibrated covariance or physical trust score.
"""
from __future__ import annotations

import numpy as np

RADIUS = 6371000.


def slope(t, y):
    t = t-t.mean()
    return np.sum(t[:, None]*(y-y.mean(0)), axis=0)/np.sum(t*t)


def temporal_evidence(cache):
    offsets, times, values = (np.asarray(cache[k]) for k in ('event_offsets', 'event_times_seconds', 'event_values'))
    query = np.asarray(cache['query_seconds'], np.float64)
    n = len(cache['node_ids'])
    if (offsets.shape != (n+1,) or offsets[0] != 0 or offsets[-1] != len(times)
            or (np.diff(offsets) < 2).any() or values.shape != (len(times), 4)
            or not np.isfinite(times).all() or not np.isfinite(values).all()):
        raise ValueError('complete finite ragged canonical events required')
    extra, variance = np.empty((n, 4)), np.zeros((n, len(query)))
    known = np.zeros(n, bool)
    for node, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        t, pos = times[a:b], values[a:b, :2]
        if (np.diff(t) <= 0).any():
            raise ValueError('strictly increasing native times required')
        lon = np.rad2deg(np.unwrap(np.deg2rad(pos[:, 1])))
        y = np.column_stack((pos[:, 0]-pos[0, 0], lon-lon[0]))
        y *= RADIUS*np.pi/180.*np.array([1., np.cos(np.deg2rad(pos[:, 0].mean()))])
        v = slope(t, y)
        residual = y-y.mean(0)-(t-t.mean())[:, None]*v
        radius = np.linalg.norm(residual, axis=1)
        if len(t) >= 4:
            h = max(1, len(t)//4)
            errors = []
            for train, test in ((slice(None, -h), slice(-h, None)), (slice(h, None), slice(None, h))):
                predicted = y[train].mean(0)+(t[test]-t[train].mean())[:, None]*slope(t[train], y[train])
                errors.extend(np.sum((y[test]-predicted)**2, axis=1))
            block = np.sqrt(np.mean(errors))
            half = len(t)//2
            split_velocity = np.linalg.norm(slope(t[:half], y[:half])-slope(t[half:], y[half:]))
        else:
            block, split_velocity = np.sqrt(np.mean(radius*radius)), 0.
        window = min(8, len(t))
        endpoint_velocity = max(np.linalg.norm(slope(t[:window], y[:window])-v),
                                np.linalg.norm(slope(t[-window:], y[-window:])-v))
        extra[node] = np.log1p([block/1000., split_velocity, np.quantile(radius, .95)/1000., endpoint_velocity])
        if len(t) > 2:
            tc = t-t.mean(); sxx = tc@tc
            endpoint = np.where(query < t[0], 0, len(t)-1)
            d = query-t[endpoint]
            gain = 1.+2*d*tc[endpoint]/sxx+d*d/sxx
            # At interior queries this scale is not used by the zero policy.
            variance[node] = np.sum(radius*radius)/(len(t)-2)*gain
            known[node] = True
    if not np.isfinite(extra).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise FloatingPointError('nonfinite temporal descriptor or OLS sensitivity')
    return dict(node_features=extra, predictive_scale_m2=variance, scale_known=known)


def source_extra(node_features, support, pack):
    result = np.zeros((*pack['available'].shape, node_features.shape[1]))
    np.add.at(result, pack['node_to_source_row'], node_features[:, None, :]*support[..., None])
    result /= np.maximum(pack['supported_counts'], 1)[..., None]
    return result


def spherical_position(values, weights):
    active = weights > 0
    values, weights = values[active], weights[active]
    weights = weights/weights.sum()
    if len(values) == 1:
        return values[0].copy()
    latitude = float(weights@values[:, 0])
    radians = np.deg2rad(values[:, 1])
    sine, cosine = weights@np.sin(radians), weights@np.cos(radians)
    if np.hypot(sine, cosine) <= 1e-12:
        # Numeric lexicographic physical-position tie, independent of source ID.
        order = np.lexsort((values[:, 1], values[:, 0]))
        return values[order[0]].copy()
    return np.array([latitude, (np.rad2deg(np.arctan2(sine, cosine))+180.) % 360.-180.])


def unsupported_candidates(motion, groups, base_states, evidence, kind):
    if kind not in ('equal', 'nearest', 'scale'):
        raise ValueError('unknown zero-support candidate rule')
    result = np.asarray(base_states).copy()
    unknown = changed = 0
    for gid, members in enumerate(groups):
        members = np.asarray(members, np.int64)
        source = motion['sources'][members]
        classes = [members[source == name] for name in dict.fromkeys(source.tolist())]
        zero = ~motion['support'][members].any(0)
        for q in np.flatnonzero(zero):
            rule = kind
            if rule == 'scale' and not evidence['scale_known'][members].all():
                rule = 'nearest'; unknown += 1
            score = evidence['predictive_scale_m2'][:, q] if rule == 'scale' else motion['nearest_seconds'][:, q]
            masses, scores = np.zeros(len(motion['states'])), []
            choices = []
            for group in classes:
                selected = group if rule == 'equal' else group[score[group] == score[group].min()]
                choices.append(selected); scores.append(float(score[selected].min()))
            keep = np.ones(len(classes), bool) if rule == 'equal' else np.asarray(scores) == min(scores)
            for selected, use in zip(choices, keep):
                if use:
                    masses[selected] = 1./keep.sum()/len(selected)
            result[gid, q, :2] = spherical_position(motion['states'][:, q, :2], masses)
            changed += 1
    return result, dict(zero_queries_processed=changed, scale_unknown_nearest_fallback_queries=unknown)


def fit_quadratic(x, y, weight):
    w = weight/weight.sum()
    mean = w@x; std = np.sqrt(w@((x-mean)**2)); std[std < 1e-6] = 1.
    model = dict(mean=mean, std=std)
    design = quadratic_design(model, x)
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0.
    model['beta'] = np.linalg.solve(design.T@(design*w[:, None])+penalty, design.T@(w*y))
    return model


def quadratic_design(model, x):
    z = np.clip((x-model['mean'])/model['std'], -8., 8.)
    i, j = np.triu_indices(z.shape[1])
    return np.column_stack((np.ones(len(z)), z, z[:, i]*z[:, j]))


def predict(model, x):
    result = quadratic_design(model, x)@model['beta']
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite temporal score')
    return result
