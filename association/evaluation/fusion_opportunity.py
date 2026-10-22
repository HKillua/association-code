"""067-B: cache-only position opportunity attribution, with no new decisions.

All references are used after the frozen predictions and routes exist. Mechanism
flags may overlap; they describe evidence, not identified causes. Point errors
are Haversine metres; local tangent vectors are descriptive directions only.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.frozen_fusion_transfer import (
    LEARNED_METHODS, NUMERIC_EPSILON, _check_artifacts, _route_scene,
)
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.learned_group_oracle import _helper_evidence
from association.evaluation.system_attribution import _same

REGIONS = ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support')
CLASSES = ('A_rule_better_model_worse', 'B_rule_worse_model_better',
           'C_both_better', 'D_both_worse', 'rule_tie_only',
           'model_tie_only', 'both_tie')
TOP_K = 20


def _distribution(values):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError('067 distributions require explicitly selected finite values')
    return dict(count=int(values.size), sum=float(values.sum()),
        mean=float(values.mean()) if values.size else None,
        p95=float(np.quantile(values, .95)) if values.size else None,
        p99=float(np.quantile(values, .99)) if values.size else None,
        min=float(values.min()) if values.size else None,
        max=float(values.max()) if values.size else None)


def _change(values):
    values = np.asarray(values, np.float64)
    return dict(signed_delta=_distribution(values),
        improvement_sum_m=float(-values[values < 0].sum()),
        harm_sum_m=float(values[values > 0].sum()),
        numerical_better_count=int((values < -NUMERIC_EPSILON).sum()),
        numerical_worse_count=int((values > NUMERIC_EPSILON).sum()),
        numerical_tie_count=int((np.abs(values) <= NUMERIC_EPSILON).sum()),
        worsening_tail=_distribution(values[values > 0]))


def _classify(rule_delta, model_delta):
    rule = np.where(rule_delta < -NUMERIC_EPSILON, -1,
                    np.where(rule_delta > NUMERIC_EPSILON, 1, 0))
    model = np.where(model_delta < -NUMERIC_EPSILON, -1,
                     np.where(model_delta > NUMERIC_EPSILON, 1, 0))
    out = np.full(rule.shape, -1, np.int8)
    for index, condition in enumerate(((rule < 0) & (model > 0),
            (rule > 0) & (model < 0), (rule < 0) & (model < 0),
            (rule > 0) & (model > 0), (rule == 0) & (model != 0),
            (rule != 0) & (model == 0), (rule == 0) & (model == 0))):
        out[condition] = index
    if (out < 0).any():
        raise ValueError('067 class partition is incomplete')
    return out


def _local_vectors(candidate, anchor):
    """North/east metres around each anchor, not a replacement error metric."""
    latitude = np.deg2rad(candidate[..., 0]-anchor[..., 0]) * 6371000.
    longitude = np.deg2rad((candidate[..., 1]-anchor[..., 1]+180.) % 360.-180.)
    return np.stack((latitude, longitude*6371000.*np.cos(np.deg2rad(anchor[..., 0]))), -1)


def _point_evidence(fragment, artifact, route, groups, anchors, rule_states, model_states):
    native, support = fragment['native_states'], fragment['support']
    g, q = artifact['group_states'].shape[:2]
    target_rows = artifact['group_anchor_target_indices']
    truth = artifact['aligned_truth'][:len(artifact['target_ids'])][target_rows]
    anchor_states = native[anchors]
    rule_vector, model_vector = (_local_vectors(x, anchor_states) for x in (rule_states, model_states))
    norm2 = np.square(rule_vector).sum(-1)
    progress = np.divide((rule_vector*model_vector).sum(-1), norm2,
        out=np.full((g, q), np.nan), where=norm2 > 0)
    base = dict(helper_mass=np.zeros((g, q)), supported_helper_mass=np.zeros((g, q)),
        helper_count=np.zeros((g, q), np.int64), supported_helper_count=np.zeros((g, q), np.int64),
        model_state_change_m=attribute_errors(model_states, anchor_states)['position'],
        rule_state_change_m=attribute_errors(rule_states, anchor_states)['position'],
        signed_progress_toward_rule=progress, local_rule_displacement_m=np.sqrt(norm2),
        local_model_displacement_m=np.linalg.norm(model_vector, axis=-1),
        supported_helper_best_error_m=np.full((g, q), np.nan),
        supported_helper_worst_error_m=np.full((g, q), np.nan),
        supported_helper_mean_error_m=np.full((g, q), np.nan),
        supported_helper_better_than_anchor_count=np.zeros((g, q), np.int64),
        supported_helper_local_spread_rms_m=np.full((g, q), np.nan),
        max_supported_helper_raw_utility=np.full((g, q), np.nan),
        group_is_pure=np.full((g, q), np.nan),
        supported_helper_wrong_identity_count=np.full((g, q), np.nan))
    identities = fragment.get('identities')
    if identities is not None and np.asarray(identities).shape != (len(native),):
        raise ValueError('067 cached identities have wrong population')
    ha, hn, weights = route['helper_anchor'], route['helper_node'], route['helper_weights']
    starts = np.r_[0, np.flatnonzero(np.diff(ha)) + 1, len(ha)]
    by_anchor = {int(ha[a]): np.arange(a, b) for a, b in zip(starts[:-1], starts[1:]) if a < b}
    base_errors = attribute_errors(anchor_states, truth)['position']
    for gid, (group, anchor) in enumerate(zip(groups, anchors)):
        indices = by_anchor.get(int(anchor), np.empty(0, np.int64))
        helpers = hn[indices]
        base['helper_count'][gid] = len(helpers)
        if identities is not None:
            base['group_is_pure'][gid] = bool(np.all(identities[group] == identities[anchor]))
            base['supported_helper_wrong_identity_count'][gid] = 0
        if not len(helpers):
            continue
        supported = support[helpers]
        count = supported.sum(0)
        base['supported_helper_count'][gid] = count
        base['helper_mass'][gid] = weights[indices, :, 0].sum(0)
        base['supported_helper_mass'][gid] = (weights[indices, :, 0]*supported).sum(0)
        errors = attribute_errors(native[helpers], truth[gid])['position']
        available = count > 0
        base['max_supported_helper_raw_utility'][gid, available] = np.where(
            supported, route['utility'][indices, :, 0], -np.inf).max(0)[available]
        base['supported_helper_best_error_m'][gid, available] = np.where(supported, errors, np.inf).min(0)[available]
        base['supported_helper_worst_error_m'][gid, available] = np.where(supported, errors, -np.inf).max(0)[available]
        base['supported_helper_mean_error_m'][gid, available] = (
            np.where(supported, errors, 0).sum(0)[available]/count[available])
        base['supported_helper_better_than_anchor_count'][gid] = (
            supported & (errors < base_errors[gid]-NUMERIC_EPSILON)).sum(0)
        vectors = _local_vectors(native[helpers], anchor_states[gid])
        center = np.where(supported[..., None], vectors, 0).sum(0)/np.maximum(count[:, None], 1)
        spread = np.where(supported, np.square(vectors-center).sum(-1), 0).sum(0)/np.maximum(count, 1)
        base['supported_helper_local_spread_rms_m'][gid, available] = np.sqrt(spread[available])
        if identities is not None:
            base['supported_helper_wrong_identity_count'][gid] = (
                supported & (identities[helpers, None] != identities[anchor])).sum(0)
    return base


def _summarize(points, mask):
    anchor, rule, model = (points[name+'_error_m'][mask] for name in ('anchor', 'rule', 'model'))
    rd, md = rule-anchor, model-anchor
    classes = _classify(rd, md)
    mass = points['helper_mass'][mask]
    unchanged = points['model_state_change_m'][mask] <= NUMERIC_EPSILON
    helper = points['helper_count'][mask] > 0
    supported = points['supported_helper_count'][mask] > 0
    progress = points['signed_progress_toward_rule'][mask]
    result = dict(point_count=int(mask.sum()),
        errors={name: dict(mae=float(values.mean()) if values.size else None,
            rmse=float(np.sqrt(np.square(values).mean())) if values.size else None,
            distribution=_distribution(values)) for name, values in
            (('anchor', anchor), ('support_switch', rule), ('helper_quality', model))},
        rule_vs_anchor=_change(rd), model_vs_anchor=_change(md),
        model_minus_rule=_change(model-rule),
        model_minus_rule_squared_error_m2=_distribution(np.square(model)-np.square(rule)),
        classes={})
    for index, name in enumerate(CLASSES):
        chosen = classes == index
        result['classes'][name] = dict(point_count=int(chosen.sum()),
            rule_vs_anchor=_change(rd[chosen]), model_vs_anchor=_change(md[chosen]),
            model_minus_rule=_change((model-rule)[chosen]),
            model_minus_rule_squared_error_m2=_distribution((np.square(model)-np.square(rule))[chosen]),
            zero_helper_mass_unchanged_count=int((chosen & helper & (mass == 0) & unchanged).sum()),
            positive_mass_count=int((chosen & (mass > 0)).sum()))
    route_states = dict(no_helper=~helper, zero_mass_with_helper=helper & (mass == 0),
        positive_mass_unchanged=(mass > 0) & unchanged, positive_mass_changed=(mass > 0) & ~unchanged)
    for index, name in enumerate(CLASSES):
        result['classes'][name]['routing_gap_contributions'] = {key: dict(
            point_count=int(((classes == index) & select).sum()),
            model_minus_rule_sum_m=float((model-rule)[(classes == index) & select].sum()),
            model_minus_rule_squared_sum_m2=float((np.square(model)-np.square(rule))[(classes == index) & select].sum()))
            for key, select in route_states.items()}
    # These are overlapping evidence flags, deliberately not a cause partition.
    result['mechanism_evidence'] = dict(
        helper_present_count=int(helper.sum()), supported_helper_present_count=int(supported.sum()),
        zero_helper_mass_unchanged_count=int((helper & (mass == 0) & unchanged).sum()),
        zero_helper_mass_changed_count=int((helper & (mass == 0) & ~unchanged).sum()),
        positive_helper_mass_unchanged_count=int(((mass > 0) & unchanged).sum()),
        positive_helper_mass_count=int((mass > 0).sum()),
        beneficial_supported_helper_exists_count=int((points['supported_helper_better_than_anchor_count'][mask] > 0).sum()),
        all_supported_helpers_no_better_than_anchor_count=int((supported &
            (points['supported_helper_better_than_anchor_count'][mask] == 0)).sum()),
        supported_wrong_identity_present_count=int((points['supported_helper_wrong_identity_count'][mask] > 0).sum()),
        identity_unidentifiable_count=int(np.isnan(points['group_is_pure'][mask]).sum()),
        mixed_group_count=int((points['group_is_pure'][mask] == 0).sum()),
        positive_mass_signed_progress_toward_rule=_distribution(progress[(mass > 0) & np.isfinite(progress)]),
        rule_direction_undefined_count=int((~np.isfinite(progress)).sum()),
        positive_mass_forward_partial_rule_progress_count=int(((mass > 0) & (progress > 0) & (progress < 1)).sum()),
        distributions={name: _distribution(points[name][mask][np.isfinite(points[name][mask])]) for name in
            ('helper_mass', 'supported_helper_mass', 'supported_helper_best_error_m',
             'supported_helper_worst_error_m', 'supported_helper_mean_error_m',
             'supported_helper_local_spread_rms_m', 'max_supported_helper_raw_utility')})
    flags = dict(
        beneficial_supported_helper_zero_mass_unchanged=supported &
            (points['supported_helper_better_than_anchor_count'][mask] > 0) & (mass == 0) & unchanged,
        positive_mass_partial_progress_rule_improves=(mass > 0) & (progress > 0) &
            (progress < 1) & (rd < -NUMERIC_EPSILON),
        supported_wrong_identity=points['supported_helper_wrong_identity_count'][mask] > 0,
        pure_group=points['group_is_pure'][mask] == 1,
        supported_helpers_all_no_better=supported &
            (points['supported_helper_better_than_anchor_count'][mask] == 0))
    gap, squared_gap = model-rule, np.square(model)-np.square(rule)
    def contributions(select):
        values, squared = gap[select], squared_gap[select]
        return dict(point_count=int(select.sum()),
            model_minus_rule_sum_m=float(values.sum()),
            gross_positive_gap_m=float(values[values > 0].sum()),
            gross_negative_gap_m=float(values[values < 0].sum()),
            model_minus_rule_squared_sum_m2=float(squared.sum()))
    union = np.logical_or.reduce(tuple(flags.values()))
    result['mechanism_evidence'].update(
        gap_contribution_interpretation='Nonexclusive, noncausal colocation with error gaps; do not sum '
            'flag rows. Negative gross gaps retain their sign. The union is counted once.',
        nonexclusive_gap_contributions={name: contributions(select) for name, select in flags.items()},
        any_flag_union_gap=contributions(union),
        multiple_flag_point_count=int((np.stack(tuple(flags.values())).sum(0) > 1).sum()))
    return result


def _region_summaries(points):
    return {name: _summarize(points, points['region_'+name]) for name in REGIONS}


def analyze_scene(fragment, artifacts, routing, scene_id, component_id):
    """Use only immutable 066/056 cached arrays; return row and reusable arrays."""
    artifact = _check_artifacts(fragment, artifacts)
    if set(routing) != set(LEARNED_METHODS):
        raise ValueError('067 requires all three saved routes')
    groups = [artifact['group_node_indices'][a:b].tolist() for a, b in
              zip(artifact['group_offsets'][:-1], artifact['group_offsets'][1:])]
    anchors = np.asarray([group[0] for group in groups], np.int64)
    n, q = fragment['native_states'].shape[:2]
    if sorted(sum(groups, [])) != list(range(n)) or any(group != sorted(group) for group in groups):
        raise ValueError('067 cached groups must retain all ordered nodes exactly once')
    _same(artifact['group_states'], fragment['native_states'][anchors], '067 anchor physical inputs')
    t = len(artifact['target_ids'])
    selected = artifact['aligned_group_indices'][:t]
    unique = np.zeros(len(groups), bool)
    unique[selected[selected >= 0]] = True
    for method in LEARNED_METHODS:
        _route_scene(fragment, artifacts[method], routing[method], groups, anchors, unique)
    _, helper_supported, _ = _helper_evidence(fragment, groups)
    region = dict(all=np.ones((len(groups), q), bool),
        anchor_inside_support=fragment['support'][anchors],
        anchor_outside_helper_supported=~fragment['support'][anchors] & helper_supported,
        predicted_group_no_support=~fragment['support'][anchors] & ~helper_supported)
    target_rows = artifact['group_anchor_target_indices']
    truth, valid = artifact['aligned_truth'][:t], artifact['aligned_truth_valid'][:t]
    valid_group = artifact['group_valid'][..., :2].all(-1) & valid[target_rows, :, :2].all(-1)
    errors = {name+'_error_m': attribute_errors(artifacts[method]['group_states'], truth[target_rows])['position']
              for name, method in (('anchor', 'anchor'), ('rule', 'support_switch'), ('model', 'helper_quality'))}
    if any(not np.isfinite(value[valid_group]).all() for value in errors.values()):
        raise ValueError('067 valid common position errors must be finite')
    evidence = _point_evidence(fragment, artifact, routing['helper_quality'], groups, anchors,
        artifacts['support_switch']['group_states'], artifacts['helper_quality']['group_states'])
    dense = dict(errors, **evidence, **{'region_'+k: v for k, v in region.items()})
    dense['target_index'] = np.broadcast_to(target_rows[:, None], valid_group.shape)
    dense['group_index'] = np.broadcast_to(np.arange(len(groups))[:, None], valid_group.shape)
    dense['query_index'] = np.broadcast_to(np.arange(q), valid_group.shape)
    query = np.asarray(fragment.get('query_seconds', np.full(q, np.nan)))
    if query.shape != (q,):
        raise ValueError('067 cached query time dimension changed')
    dense['query_seconds'] = np.broadcast_to(query, valid_group.shape)
    # Selecting groups first preserves duplicates in one view and excludes them
    # from the unique target view; neither view selects by method error.
    scopes, arrays = {}, dict(target_ids=np.asarray(artifact['target_ids']))
    for scope, scope_mask in (('all_output_groups', valid_group),
                              ('unique_targets', valid_group & unique[:, None])):
        points = {key: value[scope_mask] for key, value in dense.items()}
        scopes[scope] = _region_summaries(points)
        arrays.update({scope+'__'+key: value for key, value in points.items()})
    row = dict(scene_id=str(scene_id), component_id=str(component_id),
        fixed_population_verified=True, reference_used_for_posthoc_diagnostics_only=True,
        target_count=t, output_group_count=len(groups), unique_selected_group_count=int(unique.sum()),
        duplicate_group_count=int((~unique).sum()),
        all_output_query_count=len(groups)*q, unique_selected_query_count=int(unique.sum())*q,
        unique_valid_position_point_count=int((valid_group & unique[:, None]).sum()),
        unique_reference_unavailable_query_count=int((~valid[target_rows, :, :2].all(-1) & unique[:, None]).sum()),
        all_output_reference_unavailable_query_count=int((~valid[target_rows, :, :2].all(-1)).sum()),
        scopes=scopes)
    return row, arrays


def _points(array, scope):
    prefix = scope+'__'
    return {key[len(prefix):]: value for key, value in array.items() if key.startswith(prefix)}


def _rank_contributors(rows, arrays, region, squared=False):
    """No target identity merging across scenes; keep component membership."""
    point_records, target_records = [], []
    for row, array in zip(rows, arrays):
        points = _points(array, 'unique_targets')
        mask = points['region_'+region]
        gap = (np.square(points['model_error_m'])-np.square(points['rule_error_m']) if squared
               else points['model_error_m']-points['rule_error_m'])
        indices = np.flatnonzero(mask)
        # At most 40 per-scene points suffice for exact global top/bottom 20.
        chosen = set(sorted(indices, key=lambda i: (-gap[i], int(points['target_index'][i]), int(points['query_index'][i])))[:TOP_K])
        chosen.update(sorted(indices, key=lambda i: (gap[i], int(points['target_index'][i]), int(points['query_index'][i])))[:TOP_K])
        for i in chosen:
            point_records.append(dict(scene_id=row['scene_id'], component_id=row['component_id'],
                target_id=str(array['target_ids'][points['target_index'][i]]),
                target_index=int(points['target_index'][i]), query_index=int(points['query_index'][i]),
                group_index=int(points['group_index'][i]), contribution=float(gap[i]),
                query_seconds=float(points['query_seconds'][i]) if np.isfinite(points['query_seconds'][i]) else None,
                anchor_error_m=float(points['anchor_error_m'][i]), rule_error_m=float(points['rule_error_m'][i]),
                model_error_m=float(points['model_error_m'][i]), helper_mass=float(points['helper_mass'][i]),
                supported_helper_mass=float(points['supported_helper_mass'][i]),
                evidence={name: float(points[name][i]) if np.isfinite(points[name][i]) else None
                    for name in ('signed_progress_toward_rule', 'supported_helper_best_error_m',
                        'supported_helper_worst_error_m', 'supported_helper_local_spread_rms_m',
                        'group_is_pure', 'supported_helper_wrong_identity_count',
                        'max_supported_helper_raw_utility')}))
        for target in np.unique(points['target_index'][mask]):
            selected = mask & (points['target_index'] == target)
            target_records.append(dict(scene_id=row['scene_id'], component_id=row['component_id'],
                target_id=str(array['target_ids'][target]), target_index=int(target),
                point_count=int(selected.sum()), contribution=float(gap[selected].sum()),
                positive_contribution_sum=float(gap[selected][gap[selected] > 0].sum()),
                negative_contribution_sum=float(gap[selected][gap[selected] < 0].sum())))
    def rank(records):
        tie = lambda r: (r['scene_id'], r['target_index'], r.get('query_index', -1))
        return dict(rule_better=[r for r in sorted(records, key=lambda r: (-r['contribution'], *tie(r)))
                                 if r['contribution'] > 0][:TOP_K],
                    model_better=[r for r in sorted(records, key=lambda r: (r['contribution'], *tie(r)))
                                  if r['contribution'] < 0][:TOP_K])
    return dict(units='m2' if squared else 'm', positive_means='helper_quality error exceeds support_switch',
                points=rank(point_records), scene_targets=rank(target_records))


def aggregate(rows, arrays):
    """Point pooling and component descriptions, without independence claims."""
    if not rows or len(rows) != len(arrays) or len({r['scene_id'] for r in rows}) != len(rows):
        raise ValueError('067 requires nonempty unique scene rows and matching arrays')
    for row, array in zip(rows, arrays):
        for scope in ('all_output_groups', 'unique_targets'):
            if _region_summaries(_points(array, scope)) != row['scopes'][scope]:
                raise ValueError('067 cached points differ from scene summaries')
    def pool(indices):
        scopes = {}
        for scope in ('all_output_groups', 'unique_targets'):
            source = [_points(arrays[i], scope) for i in indices]
            joined = {key: np.concatenate([a[key] for a in source]) for key in source[0]}
            scopes[scope] = _region_summaries(joined)
        return dict(scene_count=len(indices), scopes=scopes)
    result = pool(range(len(rows)))
    result.update(status='POSTHOC_DIAGNOSTIC_ONLY', source066_acceptance='REJECTED_UNCHANGED',
        numerical_tie_tolerance_m=NUMERIC_EPSILON, top_contributor_limit=TOP_K,
        quantile_method='linear', classification_names=CLASSES,
        interpretation='Helper errors use the output anchor target reference, not helper own-target reference. '
            'Numerical ties are not engineering equivalence. Mechanism evidence is nonexclusive; '
            'positive mass or partial directional progress does not identify underweighting. Truth and identities '
            'are diagnostic only. Shared-component and scene summaries are descriptive, not independent trials.',
        population={key: sum(row[key] for row in rows) for key in (
            'target_count', 'output_group_count', 'unique_selected_group_count', 'duplicate_group_count',
            'all_output_query_count', 'unique_selected_query_count', 'unique_valid_position_point_count',
            'unique_reference_unavailable_query_count', 'all_output_reference_unavailable_query_count')})
    result['components'] = {}
    for component in sorted({row['component_id'] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
        result['components'][component] = dict(pool(indices), scene_ids=[rows[i]['scene_id'] for i in indices])
    result['component_count'] = len(result['components'])
    result['scene_distribution'] = {region: [dict(scene_id=row['scene_id'], component_id=row['component_id'],
        point_count=row['scopes']['unique_targets'][region]['point_count'],
        model_minus_rule_mean_m=row['scopes']['unique_targets'][region]['model_minus_rule']['signed_delta']['mean'],
        model_minus_rule_squared_error_sum_m2=row['scopes']['unique_targets'][region]['model_minus_rule_squared_error_m2']['sum'])
        for row in rows] for region in REGIONS}
    result['top_contributors'] = {region: {kind: _rank_contributors(rows, arrays, region, squared=kind == 'squared_error')
        for kind in ('absolute_error', 'squared_error')} for region in ('all', 'anchor_outside_helper_supported')}
    return result
