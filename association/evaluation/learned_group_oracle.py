"""061: restricted Anchor/Mean choices in the frozen056 learned partition.

The caller first replays the source with060's cache verifier. Truth is used only
after ordinary candidates and representatives are fixed. No scores, partitions,
representatives, availability masks or ordinary estimates are optimized here.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.oracle_diagnostics import (
    ARMS, POPULATION_KEYS, _comparison, select_candidates,
)
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import ATTRIBUTES, _metrics, _same, pool_metrics

METHODS = ARMS
NUMERIC_EPSILON = 1e-9
QUANTILE_METHOD = 'linear'


def _quantiles(values):
    values = np.asarray(values, np.float64)
    if values.size and not np.isfinite(values).all():
        raise ValueError('valid position errors must be finite')
    return dict(point_count=int(values.size),
        p95=float(np.quantile(values, .95, method=QUANTILE_METHOD)) if values.size else None,
        p99=float(np.quantile(values, .99, method=QUANTILE_METHOD)) if values.size else None)


def _helper_evidence(fragment, groups):
    """Follow ordinary mean_supported coefficients and angular fallbacks."""
    states, support = np.asarray(fragment['native_states'], np.float64), fragment['support']
    q = states.shape[1]
    exists = np.broadcast_to(np.asarray([len(g) > 1 for g in groups])[:, None], (len(groups), q)).copy()
    supported = np.zeros_like(exists)
    participates = np.zeros(exists.shape + (4,), bool)
    for index, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        supported[index] = support[members[1:]].any(0)
        participates[index] = supported[index, :, None]
        weights = support[members]
        divisor = np.maximum(weights.sum(0), 1)
        for column in (1, 3):
            radians = np.deg2rad(states[members, :, column])
            sine = (np.sin(radians)*weights).sum(0)/divisor
            cosine = (np.cos(radians)*weights).sum(0)/divisor
            participates[index, :, column] &= np.hypot(sine, cosine) > 1e-12
    return exists, supported, participates


def _visible_regions(fragment, anchors, selected, helper_supported):
    t, q = len(selected), fragment['native_states'].shape[1]
    represented = selected >= 0
    available = np.broadcast_to(represented[:, None], (t, q))
    anchor_support, helper_support = np.zeros((t, q), bool), np.zeros((t, q), bool)
    nearest = np.full((t, q), np.inf)
    anchor_support[represented] = fragment['support'][anchors[selected[represented]]]
    helper_support[represented] = helper_supported[selected[represented]]
    nearest[represented] = fragment['nearest_seconds'][anchors[selected[represented]]]
    regions = dict(
        anchor_inside_support=available & anchor_support,
        anchor_outside_helper_supported=available & ~anchor_support & helper_support,
        predicted_group_no_support=available & ~anchor_support & ~helper_support,
        no_predicted_group=~available,
        anchor_distance_le_300s=available & (nearest <= 300),
        anchor_distance_gt_300_le_600s=available & (nearest > 300) & (nearest <= 600),
        anchor_distance_gt_600_le_1800s=available & (nearest > 600) & (nearest <= 1800),
        anchor_distance_gt_1800s=available & (nearest > 1800) & np.isfinite(nearest))
    return regions, anchor_support, helper_support, nearest


def _selection_counts(eligible, choice, nontrivial, exists, supported, participates):
    def count(mask):
        return int((eligible & mask).sum())
    return dict(eligible_point_count=int(eligible.sum()),
        mean_selected_point_count=count(choice),
        nontrivial_point_count=count(nontrivial),
        nontrivial_mean_selected_point_count=count(nontrivial & choice),
        helper_exists_point_count=count(exists),
        helper_supported_point_count=count(supported),
        helper_supported_mean_selected_point_count=count(supported & choice),
        helper_participates_any_point_count=count(participates.any(-1)),
        helper_participates_all_point_count=count(participates.all(-1)),
        helper_participates_any_mean_selected_point_count=count(participates.any(-1) & choice),
        helper_participates_all_mean_selected_point_count=count(participates.all(-1) & choice))


def analyze_scene(fragment, replay):
    """Return JSON-compatible row and new NPZ arrays from060's six-element replay.

    The ordinary three methods come from learned_assoc only. ``arrays`` contains
    new choices, masks, mappings and point errors, not a copy of source NPZs.
    """
    source_methods, source_artifacts, source_representatives, truth, truth_valid, fixed_regions = replay
    nodes, targets = fragment['identities'], fragment['target_identity_ids']
    t, q = truth.shape[:2]
    offsets, flat = (fragment['learned_assoc_' + key] for key in ('group_offsets', 'group_node_indices'))
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    anchors = np.asarray([g[0] for g in groups], np.int64)
    artifacts = {arm: source_artifacts['learned_assoc_' + arm] for arm in ARMS[:3]}
    representatives = {arm: source_representatives['learned_assoc_' + arm] for arm in ARMS[:3]}
    anchor_artifact = artifacts['anchor']
    _same(anchor_artifact['target_ids'], targets, '061 complete target registry')
    _same(anchor_artifact['group_offsets'], offsets, '061 frozen group offsets')
    _same(anchor_artifact['group_node_indices'], flat, '061 frozen group members')
    target_rows = anchor_artifact['group_anchor_target_indices']
    selected = anchor_artifact['aligned_group_indices'][:t]
    candidates = {arm: artifacts[arm]['group_states'] for arm in ARMS[:3]}
    candidate_valid = anchor_artifact['group_valid']
    for arm in ARMS[1:3]:
        for key in ('target_ids', 'aligned_target_ids', 'aligned_group_indices', 'group_offsets',
                    'group_node_indices', 'group_anchor_target_indices', 'group_valid',
                    'aligned_prediction_valid', 'aligned_truth', 'aligned_truth_valid'):
            _same(artifacts[arm][key], anchor_artifact[key], '061 fixed ordinary ' + key)
        _same(representatives[arm], representatives['anchor'], '061 fixed representative')
    selector, selector_valid, choice = select_candidates(candidates['anchor'], candidates['mean_supported'],
        truth[target_rows], truth_valid[target_rows], candidate_valid)
    errors = {arm: attribute_errors(values, truth[target_rows]) for arm, values in
              dict(candidates, oracle_selector=selector).items()}
    group_common = truth_valid[target_rows] & candidate_valid
    difference = attribute_errors(candidates['anchor'], candidates['mean_supported'])
    nontrivial = np.stack([difference[attr] > NUMERIC_EPSILON for attr in ATTRIBUTES], axis=-1)
    for index, (attr, columns) in enumerate(ATTRIBUTES.items()):
        mask = group_common[..., list(columns)].all(-1)
        expected_choice = mask & (errors['mean_supported'][attr] < errors['anchor'][attr])
        _same(choice[..., index], expected_choice, '061 strict selector ' + attr)
        for column in columns:
            expected = np.where(choice[..., index], candidates['mean_supported'][..., column],
                                candidates['anchor'][..., column])
            _same(selector[..., column], expected, '061 candidate membership ' + attr)
        minimum = np.minimum(errors['anchor'][attr], errors['mean_supported'][attr])
        if not np.allclose(errors['oracle_selector'][attr][mask], minimum[mask],
                           atol=NUMERIC_EPSILON, rtol=0):
            raise ValueError('061 selector must equal the fixed-candidate minimum')
    outputs = [dict(node_indices=members, states=selector[index], prediction_valid=selector_valid[index])
               for index, members in enumerate(groups)]
    system = evaluate_full_scene(outputs, node_identities=nodes, target_identities=targets,
        target_states=truth, target_valid=truth_valid, node_observed=fragment['support'], split='V_select',
        grouping_kind='predicted', correspondence_policy='anchor_identity_first', return_artifact=True,
        node_input_support=dict(support=fragment['support'], nearest_seconds=fragment['nearest_seconds']))
    artifacts['oracle_selector'] = system.pop('artifact')
    representatives['oracle_selector'] = representatives['anchor'].copy()
    # The selector may change state values only, including in duplicate groups.
    for key in anchor_artifact:
        if key not in ('aligned_prediction', 'group_states'):
            _same(artifacts['oracle_selector'][key], anchor_artifact[key], '061 selector population/' + key)
    population = source_methods['learned_assoc_anchor']['population']
    if {key: system[key] for key in POPULATION_KEYS} != population:
        raise ValueError('061 selector changed complete population')
    exists, supported, participates = _helper_evidence(fragment, groups)
    visible, anchor_support, helper_support, anchor_nearest = _visible_regions(fragment, anchors, selected, supported)
    regions = dict(fixed_regions, **visible)
    unique_common = truth_valid & anchor_artifact['aligned_prediction_valid'][:t]
    unique_groups = np.zeros(len(groups), bool)
    unique_groups[selected[selected >= 0]] = True
    arrays = dict(selector_mean_choice=choice, selector_choice_attribute_names=np.asarray(tuple(ATTRIBUTES)),
        diagnostic_scope=np.asarray('061 frozen056 learned groups; restricted reference-assisted selection'),
        uses_reference_for_selection=np.asarray(True), uses_truth_identity_for_grouping=np.asarray(False),
        group_common_valid=group_common, unique_common_valid=unique_common,
        candidate_nontrivial=nontrivial, helper_exists=exists, helper_supported=supported,
        helper_participation=participates, helper_position_any=participates[..., :2].any(-1),
        helper_position_both=participates[..., :2].all(-1), group_anchor_node_indices=anchors,
        group_target_indices=target_rows, unique_selected_group_indices=selected,
        group_selected_for_unique_target=unique_groups, target_ids=np.asarray(targets),
        oracle_group_states=selector, oracle_group_valid=selector_valid,
        unique_anchor_support=anchor_support, unique_helper_support=helper_support,
        unique_anchor_nearest_seconds=anchor_nearest,
        group_anchor_nearest_seconds=fragment['nearest_seconds'][anchors])
    arrays.update({'region_' + name: mask for name, mask in regions.items()})
    methods, selections = {}, {}
    for arm in ARMS:
        source = source_methods['learned_assoc_' + arm] if arm != 'oracle_selector' else dict(
            population=population, full_target_metrics=system['full_target_metrics'],
            fixed_target_strata=_metrics(artifacts[arm]['aligned_prediction'][:t], truth, truth_valid,
                                        artifacts[arm]['aligned_prediction_valid'][:t], fixed_regions))
        if source['population'] != population:
            raise ValueError('061 ordinary population differs between arms')
        prediction = artifacts[arm]['aligned_prediction'][:t]
        position_error = attribute_errors(prediction, truth)['position']
        position_mask = unique_common[..., :2].all(-1)
        arrays[arm + '_position_error'] = np.where(position_mask, position_error, np.nan)
        methods[arm] = dict(source,
            common_target_strata=_metrics(prediction, truth, truth_valid, unique_common, regions),
            position_quantiles={name: _quantiles(position_error[position_mask & mask])
                                for name, mask in regions.items()},
            diagnostic_only=arm == 'oracle_selector',
            uses_reference_for_selection=arm == 'oracle_selector', uses_truth_identity_for_grouping=False,
            identity_usage=('fixed predicted partition and first representative; reference errors select only fixed candidates'
                if arm == 'oracle_selector' else 'evaluation correspondence only; ordinary prediction unchanged'))
    for index, (attr, columns) in enumerate(ATTRIBUTES.items()):
        mask = group_common[..., list(columns)].all(-1)
        selections[attr] = {scope: _selection_counts(mask & scope_mask, choice[..., index],
            nontrivial[..., index], exists, supported, participates[..., list(columns)])
            for scope, scope_mask in dict(group=np.ones_like(mask),
                                          unique=unique_groups[:, None]).items()}
    comparisons = [('anchor', 'mean_supported')] + [(arm, 'oracle_selector') for arm in ARMS[:3]]
    pairs = {a + '__vs__' + b: _comparison(a, b, artifacts, representatives, truth, truth_valid, regions, {})
             for a, b in comparisons}
    for pair in pairs.values():
        if pair['changed_representative_target_count'] or any(
                value for row in pair['coverage_transitions'].values() for value in row.values()):
            raise ValueError('061 fixed groups must not change representatives or coverage')
    return dict(fragment_count=len(nodes), source_replay_verified=True,
        fixed_population_verified=True, restricted_selector_verified=True, methods=methods,
        selector_choices=selections, pairwise=pairs), arrays


def aggregation_arrays(arrays):
    """Only these small point-error/mask arrays need to stay resident for pooling."""
    return {key: value for key, value in arrays.items()
            if key.startswith('region_') or key.endswith('_position_error')}


def aggregate(rows, point_arrays):
    """Pool complete rows and their same-order point arrays; quantiles are not averaged."""
    if not rows or len(rows) != len(point_arrays):
        raise ValueError('one point-array mapping per nonempty scene row required')
    regions = tuple(rows[0]['methods']['anchor']['position_quantiles'])
    result = dict(scene_count=len(rows), fragment_count=sum(row['fragment_count'] for row in rows),
                  methods={}, selector_choices={}, pairwise={})
    for arm in ARMS:
        items = [row['methods'][arm] for row in rows]
        result['methods'][arm] = dict(
            population={key: sum(item['population'][key] for item in items) for key in POPULATION_KEYS},
            **{key: pool_metrics([item[key] for item in items]) for key in
               ('full_target_metrics', 'fixed_target_strata', 'common_target_strata')},
            position_quantiles={}, diagnostic_only=arm == 'oracle_selector',
            uses_reference_for_selection=arm == 'oracle_selector', uses_truth_identity_for_grouping=False)
        for region in regions:
            values = []
            for row, arrays in zip(rows, point_arrays):
                errors = arrays[arm + '_position_error']
                use = arrays['region_' + region] & np.isfinite(errors)
                part = errors[use]
                if len(part) != row['methods'][arm]['position_quantiles'][region]['point_count']:
                    raise ValueError('quantile point population differs from its scene row')
                values.append(part)
            result['methods'][arm]['position_quantiles'][region] = _quantiles(np.concatenate(values))
    for attr in ATTRIBUTES:
        result['selector_choices'][attr] = {scope: {
            key: sum(row['selector_choices'][attr][scope][key] for row in rows)
            for key in rows[0]['selector_choices'][attr][scope]} for scope in ('group', 'unique')}
    for name, first in rows[0]['pairwise'].items():
        items = [row['pairwise'][name] for row in rows]
        pooled = dict(methods={arm: pool_metrics([item['methods'][arm] for item in items])
                              for arm in first['methods']},
            coverage_transitions={attr: {key: sum(item['coverage_transitions'][attr][key] for item in items)
                                        for key in values} for attr, values in first['coverage_transitions'].items()},
            error_change_second_minus_first={},
            **{key: sum(item[key] for item in items) for key in
               ('same_representative_target_count', 'changed_representative_target_count')},
            **{key: pool_metrics([item[key] for item in items]) for key in ('newly_covered', 'lost_coverage')})
        for region, attrs in first['error_change_second_minus_first'].items():
            pooled['error_change_second_minus_first'][region] = {}
            for attr, values in attrs.items():
                changes = [item['error_change_second_minus_first'][region][attr] for item in items]
                totals = {key: sum(change[key] for change in changes) for key in values if key != 'mean_error_change'}
                count = totals['common_point_count']
                eligible = [change['mean_error_change'] for change in changes if change['mean_error_change'] is not None]
                totals.update(mean_error_change=totals['signed_error_change_sum']/count if count else None,
                    scene_macro_error_change=float(np.mean(eligible)) if eligible else None,
                    scene_median_error_change=float(np.median(eligible)) if eligible else None,
                    eligible_scene_count=len(eligible),
                    second_better_scene_count=sum(value < -NUMERIC_EPSILON for value in eligible),
                    second_worse_scene_count=sum(value > NUMERIC_EPSILON for value in eligible),
                    tied_scene_count=sum(abs(value) <= NUMERIC_EPSILON for value in eligible))
                pooled['error_change_second_minus_first'][region][attr] = totals
        result['pairwise'][name] = pooled
    return result
