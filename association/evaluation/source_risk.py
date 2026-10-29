"""Shared source-error scores and variable-source contribution policies.

These are position-only experimental policies. Scores are predictions of a
supervised log error, never calibrated sensor probabilities or covariances.
Physical feature construction and prediction do not accept identities/truth.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.grouping import _partition
from association.evaluation.query_contribution import recover_group_states


FEATURE_NAMES = (
    'log_nearest_minutes', 'log_fit_rmse_km', 'log_native_count',
    'log_span_minutes', 'log_supported_count', 'supported_member_fraction',
    'log_within_source_rms_km', 'log_other_source_distance_km',
    'log_other_source_ols_velocity_difference_mps', 'other_source_heading_disagreement',
)
HEADS = ('LINEAR', 'QUADRATIC', 'SHUFFLED')
POLY_I, POLY_J = np.triu_indices(len(FEATURE_NAMES))


def _finite(value, name):
    value = np.asarray(value)
    if value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
        raise ValueError(f'{name} must be finite numeric values')
    return value.astype(np.float64, copy=False)


def _position_distance(a, b):
    """Spherical radial metres, with broadcasting only on leading axes."""
    lat1, lat2 = np.deg2rad(a[..., 0]), np.deg2rad(b[..., 0])
    dlon = np.deg2rad((a[..., 1]-b[..., 1]+180.) % 360.-180.)
    hav = np.clip(np.sin((lat1-lat2)/2)**2
                  + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0., 1.)
    return 2.*6371000.*np.arctan2(np.sqrt(hav), np.sqrt(1.-hav))


def _mean_state(values, visible, anchor, *, circular_anchor=None):
    """Visible mean with separate no-support and circular-ambiguity anchors."""
    counts = visible.sum(axis=0)
    available = counts > 0
    mass = visible/np.maximum(counts, 1)
    out = anchor.copy()
    circular_anchor = anchor if circular_anchor is None else circular_anchor
    for column in (0, 2):
        out[available, column] = (values[..., column]*mass).sum(0)[available]
    audit = {}
    for column, name in ((1, 'longitude'), (3, 'heading')):
        angle = np.deg2rad(values[..., column])
        sine, cosine = (np.sin(angle)*mass).sum(0), (np.cos(angle)*mass).sum(0)
        ambiguous = available & (np.hypot(sine, cosine) <= 1e-12)
        use = available & ~ambiguous
        angle = np.rad2deg(np.arctan2(sine, cosine))
        angle = (angle+180.) % 360.-180. if column == 1 else angle % 360.
        out[use, column] = angle[use]
        out[ambiguous, column] = circular_anchor[ambiguous, column]
        audit[name+'_zero_resultant_count'] = int(ambiguous.sum())
    single = counts == 1
    if single.any():
        selected = np.argmax(visible, axis=0)
        out[single] = values[selected[single], np.flatnonzero(single)]
    return out, audit


def build_source_features(*, states, support, nearest_seconds, sources,
                          native_count, span_seconds, fit_rmse_m, ols_velocity_mps, groups):
    """Construct source-equivariant physical features on supplied partitions."""
    states = _finite(states, 'states')
    support = np.asarray(support)
    nearest = _finite(nearest_seconds, 'nearest_seconds')
    sources = np.asarray(sources, dtype=object)
    if (states.ndim != 3 or states.shape[-1] != 4 or not states.shape[1]
            or support.shape != states.shape[:2] or support.dtype != bool
            or nearest.shape != support.shape or (nearest < 0).any()):
        raise ValueError('states[N,Q,4], boolean support and nonnegative nearest[N,Q], Q>0 required')
    n, q = support.shape
    if sources.shape != (n,) or any(not isinstance(s, str) or not s for s in sources):
        raise ValueError('one nonempty source string per node required')
    groups = _partition(groups, n)
    qualities = []
    for name, value in (('native_count', native_count), ('span_seconds', span_seconds),
                        ('fit_rmse_m', fit_rmse_m)):
        value = _finite(value, name)
        if value.shape != (n,) or (value < 0).any():
            raise ValueError(f'{name} must be nonnegative [N]')
        qualities.append(value)
    native_count, span_seconds, fit_rmse_m = qualities
    velocity = _finite(ols_velocity_mps, 'ols_velocity_mps')
    if velocity.shape != (n, 2):
        raise ValueError('ols_velocity_mps must be [N,2]')

    members_by_source, source_groups, offsets = [], [], [0]
    for gid, group in enumerate(groups):
        by_source = {}
        for node in group:
            by_source.setdefault(sources[node], []).append(node)
        members_by_source.extend(np.asarray(v, np.int64) for v in by_source.values())
        source_groups.extend([gid]*len(by_source))
        offsets.append(len(members_by_source))
    count = len(members_by_source)
    features = np.zeros((count, q, len(FEATURE_NAMES)), np.float64)
    source_states = np.empty((count, q, 4), np.float64)
    supported_counts = np.empty((count, q), np.int64)
    source_velocity = np.zeros((count, q, 2), np.float64)
    node_to_source = np.empty(n, np.int64)
    member_counts = np.asarray([len(m) for m in members_by_source], np.int64)
    audit = {prefix+name+'_zero_resultant_count': 0 for prefix in ('source_', 'context_')
             for name in ('longitude', 'heading')}
    for row, members in enumerate(members_by_source):
        node_to_source[members] = row
        visible = support[members]
        counts = visible.sum(0)
        supported_counts[row] = counts
        mass = visible/np.maximum(counts, 1)
        # An unavailable source retains its first member only as finite storage;
        # it receives no weight/label and is excluded from context averaging.
        # An available source with an undefined circular mean uses the fixed
        # group anchor, matching the existing hard-policy recovery fallback.
        group_anchor = states[groups[source_groups[row]][0]]
        source_states[row], local_audit = _mean_state(
            states[members], visible, states[members[0]], circular_anchor=group_anchor)
        for key, value in local_audit.items():
            audit['source_'+key] += value
        source_velocity[row] = mass.T @ velocity[members]
        features[row, :, 0] = np.log1p((mass*nearest[members]).sum(0)/60.)
        features[row, :, 1] = np.log1p(mass.T @ fit_rmse_m[members]/1000.)
        features[row, :, 2] = np.log1p(mass.T @ native_count[members])
        features[row, :, 3] = np.log1p(mass.T @ span_seconds[members]/60.)
        features[row, :, 4] = np.log1p(counts)
        features[row, :, 5] = counts/len(members)
        distances = _position_distance(states[members], source_states[row])
        features[row, :, 6] = np.log1p(np.sqrt((distances**2*mass).sum(0))/1000.)

    available = supported_counts > 0
    for gid, group in enumerate(groups):
        rows = np.arange(offsets[gid], offsets[gid+1])
        for row in rows:
            others = rows[rows != row]
            visible = available[others]
            other_count = visible.sum(0)
            use = available[row] & (other_count > 0)
            if not use.any():
                continue
            # Only available sources enter the mean. A zero circular resultant
            # uses the declared finite group-anchor fallback even if that
            # anchor's source is unavailable; its occurrences are audited.
            context, local_audit = _mean_state(source_states[others], visible, states[group[0]])
            for key, value in local_audit.items():
                audit['context_'+key] += value
            features[row, use, 7] = np.log1p(_position_distance(source_states[row], context)[use]/1000.)
            other_velocity = (source_velocity[others]*visible[..., None]).sum(0)/np.maximum(other_count[:, None], 1)
            features[row, use, 8] = np.log1p(np.linalg.norm(source_velocity[row]-other_velocity, axis=-1)[use])
            features[row, use, 9] = 1.-np.cos(np.deg2rad(source_states[row, use, 3]-context[use, 3]))
    features[~available] = 0.
    if not np.isfinite(features).all() or not np.isfinite(source_states).all():
        raise FloatingPointError('nonfinite source features or states')
    return dict(features=features, source_states=source_states, available=available,
        group_offsets=np.asarray(offsets, np.int64), node_to_source_row=node_to_source,
        supported_counts=supported_counts, source_group_indices=np.asarray(source_groups, np.int64),
        source_member_count=member_counts, audit=audit)


def _design(x, mean, std, quadratic):
    z = np.clip((x-mean)/std, -8., 8.)
    out = [np.ones((len(z), 1)), z]
    if quadratic:
        out.append(z[:, POLY_I]*z[:, POLY_J])
    return np.concatenate(out, axis=1)


def fit_score_models(x, y, sample_weight):
    """Weighted T-only ridge; caller alone defines the training split/fold."""
    x, y, weight = (_finite(v, name) for v, name in
                    ((x, 'x'), (y, 'y'), (sample_weight, 'sample_weight')))
    if (x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or y.shape != (len(x),)
            or weight.shape != y.shape or (weight <= 0).any()):
        raise ValueError('x[L,10], y[L] and strictly positive sample_weight[L] required')
    total = float(weight.sum())
    if not np.isfinite(total):
        raise FloatingPointError('nonfinite training weight sum')
    normalized = weight/total if len(x) else weight
    mean = normalized @ x if len(x) else np.zeros(len(FEATURE_NAMES))
    std = np.sqrt(normalized @ ((x-mean)**2)) if len(x) else np.ones(len(FEATURE_NAMES))
    std[std < 1e-6] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise FloatingPointError('nonfinite training normalizer')
    models = {}
    for head in HEADS:
        quadratic = head != 'LINEAR'
        design = _design(x, mean, std, quadratic)
        permutation = np.random.default_rng(20).permutation(len(y)) if head == 'SHUFFLED' else np.arange(len(y))
        beta = np.zeros(design.shape[1])
        if len(y):
            penalty = np.eye(len(beta)); penalty[0, 0] = 0.
            beta = np.linalg.solve(design.T @ (design*normalized[:, None])+penalty,
                                   design.T @ (normalized*y[permutation]))
        if not np.isfinite(beta).all():
            raise FloatingPointError('nonfinite fitted source score')
        models[head] = dict(head=head, mean=mean.copy(), std=std.copy(), beta=beta,
            training_label_count=len(y), training_weight_sum=total,
            no_labels_fallback=not len(y), permutation=permutation)
    return models


def predict_score(model, x):
    x = _finite(x, 'x')
    head = model.get('head')
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or head not in HEADS:
        raise ValueError('valid model and x[L,10] required')
    mean, std, beta = (_finite(model[key], key) for key in ('mean', 'std', 'beta'))
    if mean.shape != (10,) or std.shape != (10,) or (std <= 0).any() or beta.shape != ((11,) if head == 'LINEAR' else (66,)):
        raise ValueError('source score model shapes or scale invalid')
    result = _design(x, mean, std, head != 'LINEAR') @ beta
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite source score prediction')
    return result


def _offsets(group_offsets, size):
    offsets = np.asarray(group_offsets)
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or not len(offsets)
            or offsets[0] != 0 or offsets[-1] != size or (np.diff(offsets) <= 0).any()):
        raise ValueError('group_offsets must delimit nonempty source groups from 0 through S')
    return offsets


def source_weights(scores, available, group_offsets, *, kind):
    """Return source-equivariant masses, normalized over available sources."""
    scores = _finite(scores, 'scores')
    available = np.asarray(available)
    if (scores.ndim != 2 or not scores.shape[1] or available.shape != scores.shape
            or available.dtype != bool or kind not in ('equal', 'soft', 'hard')):
        raise ValueError('scores[S,Q], boolean available, Q>0 and equal/soft/hard required')
    offsets = _offsets(group_offsets, len(scores))
    result = np.zeros_like(scores)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        present, values = available[start:stop], scores[start:stop]
        if kind == 'equal':
            mass = present.astype(np.float64)
        else:
            minimum = np.min(np.where(present, values, np.inf), axis=0)
            minimum = np.where(present.any(0), minimum, 0.)
            if kind == 'hard':
                mass = (present & (values == minimum)).astype(np.float64)
            else:
                with np.errstate(over='ignore', under='ignore'):
                    mass = np.exp(np.where(present, -(values-minimum), -np.inf))
        total = mass.sum(0)
        result[start:stop] = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
    return result


def recover_source_policy(*, states, support, groups, pack, scores, kind, mean_states):
    states, mean = _finite(states, 'states'), _finite(mean_states, 'mean_states')
    support = np.asarray(support)
    if states.ndim != 3 or states.shape[-1] != 4 or not states.shape[1] or support.shape != states.shape[:2] or support.dtype != bool:
        raise ValueError('states[N,Q,4] and boolean support[N,Q], Q>0 required')
    n, q = support.shape
    groups = _partition(groups, n)
    if mean.shape != (len(groups), q, 4):
        raise ValueError('mean_states must be [G,Q,4]')
    available = np.asarray(pack['available'])
    sw = source_weights(scores, available, pack['group_offsets'], kind=kind)
    offsets = _offsets(pack['group_offsets'], len(sw))
    mapping = np.asarray(pack['node_to_source_row'])
    source_groups = np.asarray(pack['source_group_indices'])
    counts = np.asarray(pack['supported_counts'])
    if (len(offsets) != len(groups)+1 or sw.shape[1] != q or mapping.shape != (n,)
            or mapping.dtype.kind not in 'iu' or (mapping < 0).any() or (mapping >= len(sw)).any()
            or source_groups.shape != (len(sw),) or source_groups.dtype.kind not in 'iu'
            or counts.shape != sw.shape or counts.dtype.kind not in 'iu' or (counts < 0).any()):
        raise ValueError('source pack mapping/counts incompatible with physical inputs')
    actual_counts = np.zeros_like(counts)
    for gid, group in enumerate(groups):
        rows = mapping[group]
        if (not np.all((rows >= offsets[gid]) & (rows < offsets[gid+1]))
                or not np.all(source_groups[offsets[gid]:offsets[gid+1]] == gid)):
            raise ValueError('source rows must belong to their supplied group')
        np.add.at(actual_counts, rows, support[group].astype(np.int64))
    if not np.array_equal(actual_counts, counts) or not np.array_equal(available, counts > 0):
        raise ValueError('source support count/availability binding differs')
    weights = sw[mapping]*support/np.maximum(counts[mapping], 1)
    output = recover_group_states(states, weights, groups)['group_states']
    for gid, group in enumerate(groups):
        no_support = ~support[group].any(0)
        output[gid, no_support] = mean[gid, no_support]
    output[..., 2:] = mean[..., 2:]
    return dict(group_states=output, node_weights=weights, source_weights=sw)


def source_labels(pack, groups, identities, target_ids, truth, valid):
    """Training-only first-anchor, unique-target log error labels."""
    identities, targets = np.asarray(identities), np.asarray(target_ids)
    if identities.ndim != 1 or targets.ndim != 1 or len(set(targets.tolist())) != len(targets):
        raise ValueError('one identity per node and unique target ids required')
    if any(not isinstance(v, str) or not v for v in list(identities)+list(targets)):
        raise ValueError('identities and target ids must be nonempty strings')
    groups = _partition(groups, len(identities))
    source_states = _finite(pack['source_states'], 'source_states')
    available = np.asarray(pack['available'])
    if (source_states.ndim != 3 or source_states.shape[-1] != 4 or not source_states.shape[1]
            or available.shape != source_states.shape[:2] or available.dtype != bool):
        raise ValueError('source states and availability invalid')
    offsets = _offsets(pack['group_offsets'], len(source_states))
    q = source_states.shape[1]
    truth, valid = np.asarray(truth), np.asarray(valid)
    if (len(offsets) != len(groups)+1 or truth.shape != (len(targets), q, 4)
            or truth.dtype.kind not in 'iuf' or valid.shape != truth.shape or valid.dtype != bool
            or not np.isfinite(truth[..., :2][valid[..., :2].all(-1)]).all()):
        raise ValueError('complete target registry, masks and finite valid positions required')
    truth = truth.astype(np.float64, copy=False)
    y = np.full(available.shape, np.nan)
    errors = np.full_like(y, np.nan)
    labeled = np.zeros_like(available)
    weight = np.zeros_like(y)
    registry = {identity: i for i, identity in enumerate(targets.tolist())}
    claimed = set()
    duplicate = unknown = full_denominator = queries = 0
    for gid, group in enumerate(groups):
        identity = identities[group[0]]
        if identity not in registry:
            unknown += 1
            continue
        if identity in claimed:
            duplicate += 1
            continue
        claimed.add(identity)
        row = registry[identity]
        valid_position = valid[row, :, :2].all(-1)
        full_denominator += int(valid_position.sum())
        start, stop = offsets[gid:gid+2]
        count = available[start:stop].sum(0)
        use = valid_position & (count >= 2)
        queries += int(use.sum())
        use_sources = available[start:stop] & use
        if use.any():
            # Invalid reference values never participate in spherical arithmetic.
            error = _position_distance(source_states[start:stop, use], truth[row, use])
            selected = available[start:stop, use]
            for local, source in enumerate(range(start, stop)):
                indices = np.flatnonzero(use)[selected[local]]
                errors[source, indices] = error[local, selected[local]]
        labeled[start:stop] = use_sources
        weight[start:stop] = use_sources/np.maximum(count, 1)
    y[labeled] = np.log1p(errors[labeled]/1000.)
    if not np.isfinite(y[labeled]).all() or not np.isfinite(weight).all():
        raise FloatingPointError('nonfinite supervised source error labels')
    audit = dict(labeled_source_query_count=int(labeled.sum()), labeled_group_query_count=queries,
        training_weight_sum=float(weight.sum()), duplicate_anchor_groups_without_training_labels=duplicate,
        unknown_anchor_groups=unknown, target_count=len(targets), first_anchor_target_count=len(claimed),
        target_without_group_anchor_count=len(targets)-len(claimed),
        all_matched_reference_valid_query_count=full_denominator,
        unlabeled_source_query_count=int(labeled.size-labeled.sum()))
    return dict(y=y, labeled=labeled, sample_weight=weight, errors_m=errors, audit=audit)
