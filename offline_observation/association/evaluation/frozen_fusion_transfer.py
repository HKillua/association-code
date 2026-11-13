"""062 fixed-group fusion diagnostics, after prediction and source replay.

Reference values enter only errors in this module. The support-only control and
the frozen networks run before this function. Duplicate output groups remain in
the group/routing view; primary errors use the fixed unique-target rows.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.learned_group_oracle import _helper_evidence, _visible_regions, _quantiles
from association.evaluation.system_attribution import ATTRIBUTES, _metrics, _same, pool_metrics

METHODS = ('anchor', 'mean_supported', 'attribute_hybrid', 'support_switch',
           'helper_quality', 'direct_only', 'mixed_action')
LEARNED_METHODS = METHODS[4:]
NUMERIC_EPSILON = 1e-9
COMPARISONS = tuple(dict.fromkeys(
    [(method, control) for control in ('anchor', 'support_switch') for method in METHODS
     if method != control]
    + [(method, 'mean_supported') for method in LEARNED_METHODS]
    + [('mixed_action', method) for method in ('helper_quality', 'direct_only')]))


def _changes(values, *, quantiles=True):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError('only common valid finite errors may enter change statistics')
    better, worse = values < -NUMERIC_EPSILON, values > NUMERIC_EPSILON
    out = dict(common_point_count=int(values.size), better_point_count=int(better.sum()),
        worse_point_count=int(worse.sum()), tied_point_count=int((~better & ~worse).sum()),
        improvement_sum=float(-values[values < 0].sum()),
        harm_sum=float(values[values > 0].sum()), signed_error_change_sum=float(values.sum()),
        mean_error_change=float(values.mean()) if values.size else None)
    if quantiles:
        # Strictly positive values, including sub-epsilon changes, define this
        # tail. Numerical better/worse counts above have a different definition.
        out['positive_delta_tail'] = _quantiles(values[values > 0])
    return out


def _position_difference(candidate, control, mask):
    mask = np.asarray(mask, bool)
    differences = attribute_errors(candidate, control)['position'][mask]
    a, b = candidate[..., :2][mask], control[..., :2][mask]
    return dict(point_count=int(mask.sum()),
        exact_array_equal=bool(np.array_equal(a, b, equal_nan=True)),
        coordinate_changed_point_count=int(np.any(a != b, axis=-1).sum()),
        physical_changed_point_count=int((differences > NUMERIC_EPSILON).sum()),
        max_haversine_difference_m=float(differences.max()) if differences.size else None)


def _population(artifact):
    t, g = len(artifact['target_ids']), len(artifact['group_states'])
    selected = artifact['aligned_group_indices'][:t]
    matched = int((selected >= 0).sum())
    return dict(target_identity_count=t, predicted_group_count=g,
        duplicate_prediction_group_count=g-matched, anchor_matched_target_count=matched,
        reference_unavailable_target_count=int((~artifact['aligned_truth_valid'][:t].any(axis=(1, 2))).sum()))


def _check_artifacts(fragment, artifacts):
    if set(artifacts) != set(METHODS):
        raise ValueError('062 requires all seven fixed fusion methods')
    anchor = artifacts['anchor']
    for key, source in (('target_ids', 'target_identity_ids'),
                        ('group_offsets', 'learned_assoc_group_offsets'),
                        ('group_node_indices', 'learned_assoc_group_node_indices')):
        _same(anchor[key], fragment[source], '062 source ' + key)
    for name, artifact in artifacts.items():
        if set(artifact) != set(anchor):
            raise ValueError('062 artifact fields differ between methods')
        for key in anchor:
            if key not in ('aligned_prediction', 'group_states'):
                _same(artifact[key], anchor[key], '062 fixed ' + name + '/' + key)
        for row, group in enumerate(anchor['aligned_group_indices']):
            if group >= 0:
                _same(artifact['aligned_prediction'][row], artifact['group_states'][group],
                      '062 group/row output ' + name)
    return anchor


def _route_scene(fragment, artifact, routing, groups, anchors, unique_groups):
    """Keep helper-edge queries distinct from eligible output-anchor queries."""
    n, q = fragment['native_states'].shape[:2]
    helper_anchor, helper_node = np.asarray(routing['helper_anchor']), np.asarray(routing['helper_node'])
    expected = [(a, h) for group in groups for a in group for h in group if h != a]
    actual = list(zip(helper_anchor.tolist(), helper_node.tolist()))
    if (helper_anchor.ndim != 1 or helper_node.shape != helper_anchor.shape
            or helper_anchor.dtype.kind not in 'iu' or helper_node.dtype.kind not in 'iu'
            or actual != expected):
        raise ValueError('062 routing must retain every original fragment-anchor/helper pair in order')
    weights, own = np.asarray(routing['helper_weights']), np.asarray(routing['anchor_weights'])
    utility = np.asarray(routing['utility'])
    if weights.shape != (len(expected), q, 3) or utility.shape != weights.shape or own.shape != (n, q, 3):
        raise ValueError('062 routing tensor dimensions differ from complete fixed inputs')
    if not all(np.isfinite(x).all() for x in (weights, own, utility)) or (weights < 0).any() or (own < 0).any():
        raise ValueError('062 routing weights/utilities must be finite, with nonnegative weights')
    mass = np.zeros_like(own, dtype=np.float64)
    positive = np.zeros_like(own, dtype=bool)
    np.add.at(mass, helper_anchor, weights)
    np.logical_or.at(positive, helper_anchor, weights > 0)
    if not np.allclose(own+mass, 1., atol=1e-6, rtol=0):
        raise ValueError('062 routing anchor and helper weights must sum to one')
    membership = np.empty(n, np.int64)
    for index, group in enumerate(groups):
        membership[group] = index
    _same(routing['membership'], membership, '062 routing membership')
    target_rows = artifact['group_anchor_target_indices']
    t = len(artifact['target_ids'])
    truth, truth_valid = artifact['aligned_truth'][:t], artifact['aligned_truth_valid'][:t]
    states, native = artifact['group_states'], fragment['native_states'][anchors]
    errors, base_errors = attribute_errors(states, truth[target_rows]), attribute_errors(native, truth[target_rows])
    difference = attribute_errors(states, native)
    result = {}
    for scope, select in (('all_output_groups', np.ones(len(groups), bool)),
                          ('unique_selected_groups', unique_groups)):
        chosen = anchors[select]
        include = np.isin(helper_anchor, chosen)
        eligible = np.asarray([len(group) > 1 for group in groups])[select]
        result[scope] = {}
        for index, (attribute, columns) in enumerate(ATTRIBUTES.items()):
            helper_queries = int(include.sum())*q
            output_queries, eligible_queries = len(chosen)*q, int(eligible.sum())*q
            accepted = positive[chosen, :, index]
            eligible_valid = artifact['group_valid'][select][..., list(columns)].all(-1)
            changed = difference[attribute][select] > NUMERIC_EPSILON
            common = eligible_valid & truth_valid[target_rows[select]][..., list(columns)].all(-1)
            predicted_errors = errors[attribute][select][common]
            delta = (errors[attribute]-base_errors[attribute])[select][common]
            helper_mass = float(mass[chosen, :, index].sum())
            positive_count = int((weights[include, :, index] > 0).sum())
            accepted_count = int(accepted.sum())
            result[scope][attribute] = dict(
                output_group_count=int(select.sum()), output_query_count=output_queries,
                helper_candidate_query_count=helper_queries, positive_helper_query_count=positive_count,
                positive_helper_query_fraction=positive_count/helper_queries if helper_queries else None,
                eligible_anchor_query_count=eligible_queries, accepted_anchor_query_count=accepted_count,
                accepted_anchor_query_fraction=accepted_count/eligible_queries if eligible_queries else None,
                helper_weight_sum=helper_mass,
                mean_helper_mass_per_output_query=helper_mass/output_queries if output_queries else None,
                mean_helper_mass_per_eligible_anchor_query=helper_mass/eligible_queries if eligible_queries else None,
                valid_prediction_point_count=int(eligible_valid.sum()),
                changed_state_point_count=int((eligible_valid & changed).sum()),
                positive_helper_unchanged_state_point_count=int((eligible_valid & accepted & ~changed).sum()),
                valid_reference_point_count=int(common.sum()),
                error_sum=float(predicted_errors.sum()), squared_error_sum=float(np.square(predicted_errors).sum()),
                mae=float(predicted_errors.mean()) if predicted_errors.size else None,
                error_change_vs_anchor=_changes(delta, quantiles=False))
    return result


def summarize_scene(fragment, artifacts, routing, scene_id, component_id):
    """Summarize one fully replayed scene; no grouping or model inference here.

    ``routing`` holds predict_partition dictionaries for the three learned
    methods. ``fragment['direct_only_prediction']`` is the actual N×Q×4 terminal
    output, including fragment anchors that are not system representatives.
    Returned point arrays contain errors/masks only, suitable for pooled tails.
    """
    anchor = _check_artifacts(fragment, artifacts)
    if set(routing) != set(LEARNED_METHODS):
        raise ValueError('062 requires terminal routing for all three frozen models')
    t = len(anchor['target_ids'])
    truth, truth_valid = anchor['aligned_truth'][:t], anchor['aligned_truth_valid'][:t]
    common_valid = truth_valid & anchor['aligned_prediction_valid'][:t]
    offsets, flat = anchor['group_offsets'], anchor['group_node_indices']
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    anchors = np.asarray([group[0] for group in groups], np.int64)
    selected = anchor['aligned_group_indices'][:t]
    _, supported, _ = _helper_evidence(fragment, groups)
    visible, _, _, _ = _visible_regions(fragment, anchors, selected, supported)
    regions = dict(all=np.ones(truth.shape[:2], bool), **visible)
    unique_groups = np.zeros(len(groups), bool)
    unique_groups[selected[selected >= 0]] = True
    population = _population(anchor)
    arrays = {'region_' + key: value for key, value in regions.items()}
    row = dict(scene_id=str(scene_id), component_id=str(component_id), fragment_count=len(fragment['native_states']),
        fixed_population_verified=True, methods={}, pairwise={}, routing={})
    for method, artifact in artifacts.items():
        full_observed = artifact['aligned_observed']
        full_regions = dict(all=np.ones(full_observed.shape, bool), observed=full_observed, missing=~full_observed)
        prediction = artifact['aligned_prediction'][:t]
        errors = attribute_errors(prediction, truth)
        for attr, columns in ATTRIBUTES.items():
            mask = common_valid[..., list(columns)].all(-1)
            if not np.isfinite(errors[attr][mask]).all():
                raise ValueError('062 valid physical error is nonfinite')
            arrays[method + '_' + attr + '_error'] = np.where(mask, errors[attr], np.nan)
        row['methods'][method] = dict(population=population.copy(),
            full_target_metrics=_metrics(artifact['aligned_prediction'], artifact['aligned_truth'],
                artifact['aligned_truth_valid'], artifact['aligned_prediction_valid'], full_regions),
            common_target_strata=_metrics(prediction, truth, truth_valid, anchor['aligned_prediction_valid'][:t], regions),
            position_quantiles={key: _quantiles(arrays[method+'_position_error'][mask & np.isfinite(
                arrays[method+'_position_error'])]) for key, mask in regions.items()})
    for method, control in COMPARISONS:
        row['pairwise'][method+'__minus__'+control] = {
            key: {attr: _changes((arrays[method+'_'+attr+'_error']-arrays[control+'_'+attr+'_error'])[
                mask & np.isfinite(arrays[method+'_'+attr+'_error'])]) for attr in ATTRIBUTES}
            for key, mask in regions.items()}
    for method in LEARNED_METHODS:
        row['routing'][method] = _route_scene(fragment, artifacts[method], routing[method], groups, anchors, unique_groups)
    direct = np.asarray(fragment['direct_only_prediction'])
    if direct.shape != fragment['native_states'].shape or not np.isfinite(direct).all():
        raise ValueError('062 direct-only terminal fragment prediction missing or invalid')
    _same(direct[anchors], artifacts['direct_only']['group_states'], '062 direct-only fragment/group states')
    row['direct_only_vs_anchor'] = {
        'fragment': _position_difference(direct, fragment['native_states'], np.ones(direct.shape[:2], bool)),
        'all_output_groups': _position_difference(artifacts['direct_only']['group_states'], anchor['group_states'],
                                                  anchor['group_valid'][..., :2].all(-1)),
        'unique_valid_targets': _position_difference(artifacts['direct_only']['aligned_prediction'][:t],
            anchor['aligned_prediction'][:t], common_valid[..., :2].all(-1))}
    return row, arrays


def _pool_changes(items, values):
    out = _changes(values)
    scene_values = [item['mean_error_change'] for item in items if item['mean_error_change'] is not None]
    out.update(eligible_scene_count=len(scene_values),
        scene_macro_error_change=float(np.mean(scene_values)) if scene_values else None,
        scene_median_error_change=float(np.median(scene_values)) if scene_values else None,
        better_scene_count=sum(value < -NUMERIC_EPSILON for value in scene_values),
        worse_scene_count=sum(value > NUMERIC_EPSILON for value in scene_values),
        tied_scene_count=sum(abs(value) <= NUMERIC_EPSILON for value in scene_values))
    return out


def _pool_routes(items):
    sums = ('output_group_count', 'output_query_count', 'helper_candidate_query_count', 'positive_helper_query_count',
        'eligible_anchor_query_count', 'accepted_anchor_query_count', 'helper_weight_sum',
        'valid_prediction_point_count', 'changed_state_point_count', 'positive_helper_unchanged_state_point_count',
        'valid_reference_point_count', 'error_sum', 'squared_error_sum')
    out = {key: sum(item[key] for item in items) for key in sums}
    for key, num, den in (
        ('positive_helper_query_fraction', 'positive_helper_query_count', 'helper_candidate_query_count'),
        ('accepted_anchor_query_fraction', 'accepted_anchor_query_count', 'eligible_anchor_query_count'),
        ('mean_helper_mass_per_output_query', 'helper_weight_sum', 'output_query_count'),
        ('mean_helper_mass_per_eligible_anchor_query', 'helper_weight_sum', 'eligible_anchor_query_count'),
        ('mae', 'error_sum', 'valid_reference_point_count')):
        out[key] = out[num]/out[den] if out[den] else None
    changes = [item['error_change_vs_anchor'] for item in items]
    delta = {key: sum(item[key] for item in changes) for key in changes[0] if key != 'mean_error_change'}
    delta['mean_error_change'] = delta['signed_error_change_sum']/delta['common_point_count'] if delta['common_point_count'] else None
    out['error_change_vs_anchor'] = delta
    return out


def aggregate(rows, arrays):
    """Pool point errors, never average scene quantiles; components are descriptive."""
    if not rows or len(rows) != len(arrays):
        raise ValueError('062 requires one point-array mapping per nonempty scene row')
    if len({row['scene_id'] for row in rows}) != len(rows):
        raise ValueError('062 duplicate scene rows')
    regions = tuple(rows[0]['methods']['anchor']['position_quantiles'])
    for row, points in zip(rows, arrays):
        if tuple(row['methods']['anchor']['position_quantiles']) != regions:
            raise ValueError('062 scene region definitions differ')
        for method in METHODS:
            for region in regions:
                values = points[method+'_position_error']
                actual = _quantiles(values[points['region_'+region] & np.isfinite(values)])
                if actual != row['methods'][method]['position_quantiles'][region]:
                    raise ValueError('062 pooled point population differs from its scene summary')

    def values_for(method, attr, region, indices):
        return np.concatenate([arrays[i][method+'_'+attr+'_error'][arrays[i]['region_'+region]
            & np.isfinite(arrays[i][method+'_'+attr+'_error'])] for i in indices])

    def pool(indices, all_regions=True):
        chosen = [rows[i] for i in indices]
        use_regions = regions if all_regions else ('all',)
        result = dict(scene_count=len(chosen), fragment_count=sum(row['fragment_count'] for row in chosen),
            methods={}, pairwise={})
        for method in METHODS:
            method_rows = [row['methods'][method] for row in chosen]
            result['methods'][method] = dict(
                population={key: sum(item['population'][key] for item in method_rows) for key in method_rows[0]['population']},
                full_target_metrics=pool_metrics([item['full_target_metrics'] for item in method_rows]),
                common_target_strata=pool_metrics([{r: item['common_target_strata'][r] for r in use_regions}
                                                  for item in method_rows]),
                position_quantiles={r: _quantiles(values_for(method, 'position', r, indices)) for r in use_regions})
        for method, control in COMPARISONS:
            name = method+'__minus__'+control
            result['pairwise'][name] = {region: {attr: _pool_changes(
                [row['pairwise'][name][region][attr] for row in chosen],
                values_for(method, attr, region, indices)-values_for(control, attr, region, indices))
                for attr in ATTRIBUTES} for region in use_regions}
        return result

    result = pool(list(range(len(rows))))
    result['routing'] = {method: {scope: {attr: _pool_routes([row['routing'][method][scope][attr] for row in rows])
        for attr in ATTRIBUTES} for scope in ('all_output_groups', 'unique_selected_groups')} for method in LEARNED_METHODS}
    result['direct_only_vs_anchor'] = {}
    for scope in rows[0]['direct_only_vs_anchor']:
        items = [row['direct_only_vs_anchor'][scope] for row in rows]
        differences = [item['max_haversine_difference_m'] for item in items if item['max_haversine_difference_m'] is not None]
        result['direct_only_vs_anchor'][scope] = dict(
            **{key: sum(item[key] for item in items) for key in
               ('point_count', 'coordinate_changed_point_count', 'physical_changed_point_count')},
            exact_array_equal=all(item['exact_array_equal'] for item in items),
            max_haversine_difference_m=max(differences) if differences else None)
    result['components'] = {}
    for component in sorted({row['component_id'] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
        result['components'][component] = dict(pool(indices, all_regions=False),
            scene_ids=[rows[i]['scene_id'] for i in indices])
    result.update(component_count=len(result['components']),
        numeric_change_tolerance=NUMERIC_EPSILON, quantile_method='linear',
        interpretation='Development-set descriptive fixed-group comparison; point/scene/component views '
        'are not independent samples. Positive routing does not prove changed states or improved errors.')
    return result
