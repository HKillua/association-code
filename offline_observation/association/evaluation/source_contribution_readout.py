"""140: one supervised chord-risk critic and separable contribution readouts.

The convex surrogate concerns unnormalised 3D vectors. Physical output retains
the original node latitude/circular-longitude recovery; no Bayes or radial-MAE
optimality is implied by the surrogate or its numerical duality gap.
"""
from __future__ import annotations

import numpy as np

from association.evaluation import source_risk as source
from association.evaluation.grouping import _partition
from association.evaluation.query_contribution import recover_group_states

RADIUS_M = 6371000.
RISK_FLOOR_KM2 = 1e-6
FW_MAX_STEPS = 256
FW_RELATIVE_GAP = 1e-8


def chord_targets(errors_m):
    errors = source._finite(errors_m, 'errors_m')
    if (errors < 0).any() or (errors > np.pi*RADIUS_M).any():
        raise ValueError('radial errors must be in [0, pi R] metres')
    return (2.*RADIUS_M/1000.*np.sin(errors/(2.*RADIUS_M)))**2


def fit_risk_model(x, y, sample_weight):
    """Exactly one quadratic weighted ridge, with fold-local normalisation."""
    x, y, weight = (source._finite(v, name) for v, name in
                    ((x, 'x'), (y, 'y'), (sample_weight, 'sample_weight')))
    if (x.ndim != 2 or x.shape[1] != 10 or y.shape != (len(x),)
            or weight.shape != y.shape or (weight <= 0).any() or (y < 0).any()):
        raise ValueError('x[L,10], nonnegative y[L], strictly positive weight[L] required')
    total = float(weight.sum())
    if not np.isfinite(total):
        raise FloatingPointError('nonfinite weight sum')
    normalized = weight/total if len(x) else weight
    mean = normalized@x if len(x) else np.zeros(10)
    std = np.sqrt(normalized@((x-mean)**2)) if len(x) else np.ones(10)
    std[std < 1e-6] = 1.
    design = source._design(x, mean, std, True)
    beta = np.zeros(66)
    if len(x):
        penalty = np.eye(66); penalty[0, 0] = 0.
        beta = np.linalg.solve(design.T@(design*normalized[:, None])+penalty,
                               design.T@(normalized*y))
    if not all(np.isfinite(v).all() for v in (mean, std, beta)):
        raise FloatingPointError('nonfinite fitted risk model')
    return dict(head='QUADRATIC', target_kind='squared_chord_km2', mean=mean,
        std=std, beta=beta, training_label_count=len(y), training_weight_sum=total,
        no_labels_fallback=not len(y), ridge_lambda=1.)


def predict_risk(model, x):
    if model.get('target_kind') != 'squared_chord_km2':
        raise ValueError('chord risk model required')
    return source.predict_score(model, x)


def embedding_km(states):
    states = source._finite(states, 'source_states')
    if states.ndim < 2 or states.shape[-1] != 4 or (np.abs(states[..., 0]) > 90).any():
        raise ValueError('finite source states[...,4] with legal latitude required')
    lat, lon = np.deg2rad(states[..., 0]), np.deg2rad(states[..., 1])
    return (RADIUS_M/1000.)*np.stack((np.cos(lat)*np.cos(lon),
                                    np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)


def _geometry_weights(risk, distance):
    """Simplex minimizer: exact K<=2, bounded permutation-equivariant FW."""
    k = len(risk)
    if k == 1:
        return np.ones(1), 0, True
    if k == 2:
        d = distance[0, 1]
        if d == 0.:
            chosen = risk == risk.min()
            return chosen/chosen.sum(), 0, True
        delta = risk[1]-risk[0]
        alpha = 1. if delta >= d else 0. if delta <= -d else .5+.5*delta/d
        return np.asarray([alpha, 1.-alpha]), 0, True
    weight = np.full(k, 1./k)
    threshold = FW_RELATIVE_GAP*(1.+float(np.max(np.abs(risk)))+float(distance.max()))
    steps = 0
    for _ in range(FW_MAX_STEPS):
        gradient = risk-distance@weight
        gap = float(weight@gradient-gradient.min())
        if gap <= threshold:
            break
        chosen = gradient == gradient.min()
        vertex = chosen/chosen.sum()
        direction = vertex-weight
        slope = float(gradient@direction)
        curvature = -float(direction@distance@direction)
        step = np.clip(-slope/curvature, 0., 1.) if curvature > 0 else float(slope < 0.)
        weight = weight+step*direction
        # Do not replace an unfinished result with an unreported policy.
        steps += 1
    gradient = risk-distance@weight
    gap = float(weight@gradient-gradient.min())
    return weight, steps, gap <= threshold


def contribution_weights(risk, available, group_offsets, source_states, *, kind):
    risk = source._finite(risk, 'clipped risk')
    available = np.asarray(available)
    states = source._finite(source_states, 'source_states')
    if (risk.ndim != 2 or not risk.shape[1] or available.shape != risk.shape
            or available.dtype != bool or states.shape != risk.shape+(4,)
            or (risk < 0).any() or kind not in ('hard', 'inverse', 'geometry')):
        raise ValueError('nonnegative risk[S,Q], bool available, states[S,Q,4], valid kind required')
    offsets = source._offsets(group_offsets, len(risk))
    positions = embedding_km(states)
    result = np.zeros_like(risk)
    shape = (len(offsets)-1, risk.shape[1])
    objective = np.full(shape, np.nan)
    gap = np.full(shape, np.nan)
    iterations = np.zeros(shape, np.int64)
    converged = np.zeros(shape, bool)
    defined = np.zeros(shape, bool)
    floor_hit = available & (risk < RISK_FLOOR_KM2) if kind == 'inverse' else np.zeros_like(available)
    for gid, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        for query in range(risk.shape[1]):
            rows = np.flatnonzero(available[start:stop, query])+start
            if not len(rows):
                continue
            r = risk[rows, query]
            points = positions[rows, query]
            difference = points[:, None, :]-points[None, :, :]
            distance = np.sum(difference*difference, axis=-1)
            if kind == 'geometry':
                w, steps, done = _geometry_weights(r, distance)
                iterations[gid, query], converged[gid, query] = steps, done
            elif kind == 'hard':
                choice = r == r.min()
                w = choice/choice.sum()
            else:
                mass = 1./np.maximum(r, RISK_FLOOR_KM2)
                w = mass/mass.sum()
            result[rows, query] = w
            gradient = r-distance@w
            objective[gid, query] = r@w-.5*w@distance@w
            gap[gid, query] = max(0., float(w@gradient-gradient.min()))
            defined[gid, query] = True
    if not np.isfinite(result).all() or not np.isfinite(objective[defined]).all():
        raise FloatingPointError('nonfinite contribution calculation')
    return dict(source_weights=result, floor_hit=floor_hit, objective_km2=objective,
        duality_gap_km2=gap, solver_iterations=iterations,
        solver_converged=converged, solver_defined=defined)


def recover_contributions(*, states, support, groups, pack, source_weights, mean_states):
    states, mean = source._finite(states, 'states'), source._finite(mean_states, 'mean_states')
    support = np.asarray(support)
    sw = source._finite(source_weights, 'source_weights')
    if states.ndim != 3 or states.shape[-1] != 4 or support.shape != states.shape[:2] or support.dtype != bool:
        raise ValueError('states[N,Q,4], boolean support[N,Q] required')
    if sw.ndim != 2 or not states.shape[1]:
        raise ValueError('source_weights[S,Q] and Q>0 required')
    groups = _partition(groups, len(states))
    offsets = source._offsets(pack['group_offsets'], len(sw))
    mapping = np.asarray(pack['node_to_source_row'])
    counts = np.asarray(pack['supported_counts'])
    available = np.asarray(pack['available'])
    source_groups = np.asarray(pack['source_group_indices'])
    if (len(offsets) != len(groups)+1 or sw.shape != counts.shape or sw.shape != available.shape
            or sw.shape[1:] != (states.shape[1],) or available.dtype != bool
            or counts.dtype.kind not in 'iu' or (counts < 0).any()
            or mapping.shape != (len(states),) or mapping.dtype.kind not in 'iu'
            or (mapping < 0).any() or (mapping >= len(sw)).any()
            or source_groups.shape != (len(sw),) or source_groups.dtype.kind not in 'iu'
            or mean.shape != (len(groups), states.shape[1], 4)
            or (sw < 0).any() or np.any(sw[~available] != 0.)):
        raise ValueError('invalid source pack, weights or mean shape')
    actual_counts = np.zeros_like(counts)
    for gid, group in enumerate(groups):
        rows = mapping[group]
        if (not np.all((rows >= offsets[gid]) & (rows < offsets[gid+1]))
                or not np.all(source_groups[offsets[gid]:offsets[gid+1]] == gid)):
            raise ValueError('source mapping crosses group boundary')
        np.add.at(actual_counts, rows, support[group].astype(np.int64))
        total = sw[offsets[gid]:offsets[gid+1]].sum(0)
        wanted = support[group].any(0).astype(float)
        if not np.allclose(total, wanted, rtol=0., atol=1e-10):
            raise ValueError('source masses must sum to one exactly where supported')
    if not np.array_equal(actual_counts, counts) or not np.array_equal(available, counts > 0):
        raise ValueError('source count/support binding differs')
    node_weights = sw[mapping]*support/np.maximum(counts[mapping], 1)
    output = recover_group_states(states, node_weights, groups)['group_states']
    for gid, group in enumerate(groups):
        no_support = ~support[group].any(0)
        output[gid, no_support] = mean[gid, no_support]
    output[..., 2:] = mean[..., 2:]
    return dict(source_weights=sw, node_weights=node_weights, group_states=output)
