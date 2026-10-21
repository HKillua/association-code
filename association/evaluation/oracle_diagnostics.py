"""058 restricted oracle diagnostics on frozen053 arrays; never a deployable method.

Identity grouping changes only the partition. The selector uses reference errors
only after Anchor/Mean predictions and first representatives have been fixed.
Its bound applies to those two candidates, not to arbitrary state estimators.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_controls import attribute_errors, ordinary_fusion
from association.evaluation.grouping import group_tracks_by_identity
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.system_attribution import (
    ATTRIBUTES, BASE, _metrics, _regions, _same, analyze_scene, pool_metrics,
)

GROUPINGS = ('predicted', 'oracle_identity')
ARMS = ('anchor', 'mean_supported', 'attribute_hybrid', 'oracle_selector')
METHODS = tuple(group + '_' + arm for group in GROUPINGS for arm in ARMS)
POPULATION_KEYS = ('target_identity_count', 'predicted_group_count',
                   'duplicate_prediction_group_count', 'anchor_matched_target_count',
                   'reference_unavailable_target_count')


def select_candidates(anchor, mean, truth, truth_valid, prediction_valid=None):
    """Choose an entire position pair or scalar attribute; ties/invalid truth use Anchor.

    Both finite candidates have the same input availability. Reference validity
    controls selection/evaluation, never prediction coverage. Returned choice
    columns are position/speed/heading, not latitude/longitude separately.
    """
    anchor, mean, truth = (np.asarray(x, np.float64) for x in (anchor, mean, truth))
    truth_valid = np.asarray(truth_valid)
    if (anchor.shape != mean.shape or anchor.shape != truth.shape or anchor.ndim != 3
            or anchor.shape[-1] != 4 or truth_valid.shape != truth.shape
            or truth_valid.dtype != bool or not np.isfinite(anchor).all()
            or not np.isfinite(mean).all()
            or not np.isfinite(truth[truth_valid]).all()):
        raise ValueError('aligned finite candidates and explicit reference masks required')
    valid = np.ones(anchor.shape, bool) if prediction_valid is None else np.asarray(prediction_valid)
    if valid.shape != anchor.shape or valid.dtype != bool:
        raise ValueError('shared candidate prediction_valid must be boolean and aligned')
    result, choice = anchor.copy(), np.zeros(anchor.shape[:-1] + (3,), bool)
    errors = [attribute_errors(values, truth) for values in (anchor, mean)]
    for index, (attr, columns) in enumerate(ATTRIBUTES.items()):
        eligible = (truth_valid[..., list(columns)] & valid[..., list(columns)]).all(-1)
        choose = eligible & (errors[1][attr] < errors[0][attr])
        choice[..., index] = choose
        for column in columns:
            result[..., column] = np.where(choose, mean[..., column], anchor[..., column])
    return result, valid.copy(), choice


def _comparison(a, b, artifacts, representatives, truth, truth_valid, regions, arrays):
    """Paired common points plus gained/lost coverage; no survivor-only ranking."""
    t, q = truth.shape[:2]
    first, second = artifacts[a], artifacts[b]
    av, bv = (r['aligned_prediction_valid'][:t] for r in (first, second))
    common = truth_valid & av & bv
    ra, rb = representatives[a], representatives[b]
    represented = (ra >= 0) & (rb >= 0)
    same, changed = represented & (ra == rb), represented & (ra != rb)
    paired_regions = dict(regions,
        same_representative=np.broadcast_to(same[:, None], (t, q)),
        changed_representative=np.broadcast_to(changed[:, None], (t, q)))
    result = dict(methods={name: _metrics(artifacts[name]['aligned_prediction'][:t],
        truth, truth_valid, common, paired_regions) for name in (a, b)},
        same_representative_target_count=int(same.sum()),
        changed_representative_target_count=int(changed.sum()),
        error_change_second_minus_first={}, coverage_transitions={})
    errors = {name: attribute_errors(artifacts[name]['aligned_prediction'][:t], truth) for name in (a, b)}
    gained, lost = np.zeros_like(common), np.zeros_like(common)
    for attr, columns in ATTRIBUTES.items():
        reference = truth_valid[..., list(columns)].all(-1)
        old = reference & av[..., list(columns)].all(-1)
        new = reference & bv[..., list(columns)].all(-1)
        gain, loss = new & ~old, old & ~new
        result['coverage_transitions'][attr] = dict(
            rescued_target_count=int((new.any(1) & ~old.any(1)).sum()),
            lost_target_count=int((old.any(1) & ~new.any(1)).sum()),
            gained_target_points=int(gain.sum()), lost_target_points=int(loss.sum()))
        for column in columns:
            gained[..., column], lost[..., column] = gain, loss
    for region, mask in paired_regions.items():
        result['error_change_second_minus_first'][region] = {}
        for attr, columns in ATTRIBUTES.items():
            use = common[..., list(columns)].all(-1) & mask
            delta = (errors[b][attr] - errors[a][attr])[use]
            better, worse = delta < -1e-9, delta > 1e-9
            result['error_change_second_minus_first'][region][attr] = dict(
                common_point_count=len(delta), second_better_point_count=int(better.sum()),
                second_worse_point_count=int(worse.sum()), tied_point_count=int((~better & ~worse).sum()),
                benefit_sum=float(-delta[better].sum()), harm_sum=float(delta[worse].sum()),
                signed_error_change_sum=float(delta.sum()),
                mean_error_change=float(delta.mean()) if len(delta) else None)
    result['newly_covered'] = _metrics(second['aligned_prediction'][:t], truth, truth_valid, gained, regions)
    result['lost_coverage'] = _metrics(first['aligned_prediction'][:t], truth, truth_valid, lost, regions)
    prefix = a + '__vs__' + b
    arrays[prefix + '_common_valid'] = common
    arrays[prefix + '_newly_covered_valid'] = gained
    arrays[prefix + '_lost_coverage_valid'] = lost
    return result


def analyze_oracles(fragment, saved, expected=None, source_names=None):
    """Replay053 exactly, then evaluate eight fixed, explicitly diagnostic arms."""
    replay, source = analyze_scene(fragment, saved, expected, source_names=source_names)
    nodes, targets = source['identities'], source['target_ids']
    n, t = len(nodes), len(targets)
    native = np.asarray(fragment['native_states'], np.float64)
    support, nearest = fragment['support'], fragment['nearest_seconds']
    truth, truth_valid = saved['aligned_truth'][:t], saved['aligned_truth_valid'][:t]
    offsets, flat = fragment['ols_group_offsets'], fragment['ols_group_node_indices']
    partitions = dict(predicted=[flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])],
                      oracle_identity=group_tracks_by_identity(nodes, node_order='input'))
    regions = _regions(source['target_input_support'], source['target_nearest_input_seconds'],
                       source['target_exact_observation'])
    arrays = {key: source[key] for key in ('query_seconds', 'node_ids', 'identities', 'sources',
        'target_ids', 'target_input_support', 'target_nearest_input_seconds', 'target_exact_observation')}
    arrays.update(native_states=native, node_support=support, node_nearest_seconds=nearest,
                  target_states=truth, target_valid=truth_valid)
    methods, artifacts, representatives, selections = {}, {}, {}, {}
    registry = {identity: row for row, identity in enumerate(targets)}
    for grouping, groups in partitions.items():
        ordinary = ordinary_fusion(native, nearest, support, groups)['predictions']
        if grouping == 'predicted':
            _same(ordinary['mean_supported'], fragment[BASE], 'ordinary mean cache')
        anchors = np.asarray([group[0] for group in groups], np.int64)
        # Ordinary predictions are completely generated before target lookup.
        candidates = {name: ordinary[name][anchors] for name in ('anchor', 'mean_supported')}
        hybrid = candidates['mean_supported'].copy()
        hybrid[..., :2] = candidates['anchor'][..., :2]
        candidates['attribute_hybrid'] = hybrid
        target_rows = np.asarray([registry[nodes[node]] for node in anchors], np.int64)
        selector, selector_valid, choice = select_candidates(candidates['anchor'], candidates['mean_supported'],
            truth[target_rows], truth_valid[target_rows])
        candidates['oracle_selector'] = selector
        arrays[grouping + '_selector_mean_choice'] = choice
        arrays[grouping + '_selector_choice_attribute_names'] = np.asarray(tuple(ATTRIBUTES))
        arrays[grouping + '_group_anchor_node_indices'] = anchors
        for arm in ARMS:
            name = grouping + '_' + arm
            outputs = [dict(node_indices=group, states=candidates[arm][i],
                            prediction_valid=selector_valid[i]) for i, group in enumerate(groups)]
            system = evaluate_full_scene(outputs, node_identities=nodes, target_identities=targets,
                target_states=truth, target_valid=truth_valid, node_observed=support, split='V_select',
                grouping_kind='oracle' if grouping == 'oracle_identity' else 'predicted',
                correspondence_policy='anchor_identity_first', return_artifact=True,
                node_input_support=dict(support=support, nearest_seconds=nearest))
            artifact = system.pop('artifact')
            # Per-group error trees repeat the target population; arrays already
            # preserve all group outputs, including duplicate/unmatched rows.
            system['all_predicted_groups'] = [{k: v for k, v in row.items() if k != 'metrics'}
                                             for row in system['all_predicted_groups']]
            system['prediction_generation'] = ('truth-error selection between fixed Anchor/Mean; diagnostic only'
                if arm == 'oracle_selector' else 'ordinary output; no reference states used')
            system['diagnostic_only'] = grouping == 'oracle_identity' or arm == 'oracle_selector'
            system['uses_reference_for_selection'] = arm == 'oracle_selector'
            system['uses_truth_identity_for_grouping'] = grouping == 'oracle_identity'
            if grouping == 'oracle_identity':
                system['identity_usage'] = 'truth identity constructs diagnostic partition; no reference states select representative'
            if arm == 'oracle_selector':
                system['identity_usage'] = ('truth identity partition' if grouping == 'oracle_identity' else 'fixed predicted partition') + \
                    '; first-input representative unchanged; reference errors select only physical candidates'
            methods[name] = dict(system=system, fixed_target_strata=_metrics(
                artifact['aligned_prediction'][:t], truth, truth_valid,
                artifact['aligned_prediction_valid'][:t], regions))
            selected = artifact['aligned_group_indices'][:t]
            rep = np.full(t, -1, np.int64)
            rep[selected >= 0] = anchors[selected[selected >= 0]]
            representatives[name] = rep
            artifacts[name] = artifact
            arrays[name + '_selected_anchor_node_index'] = rep
            for key in ('diagnostic_only', 'uses_reference_for_selection', 'uses_truth_identity_for_grouping'):
                arrays[name + '_' + key] = np.asarray(system[key])
            arrays.update({name + '_' + key: value for key, value in artifact.items()})
        anchor_artifact = artifacts[grouping + '_anchor']
        for arm in ARMS[1:]:
            _same(artifacts[grouping + '_' + arm]['aligned_prediction_valid'],
                  anchor_artifact['aligned_prediction_valid'], 'fixed-group prediction availability')
        # Counts against the fixed representative's truth, including duplicate
        # groups. Unique-target metrics above remain the primary population.
        selections[grouping] = {attr: dict(
            eligible_group_point_count=int(truth_valid[target_rows][..., list(columns)].all(-1).sum()),
            mean_selected_group_point_count=int(choice[..., index].sum()))
            for index, (attr, columns) in enumerate(ATTRIBUTES.items())}
    # Preserve exact053 output/masks/group arrays, not merely aggregate MAEs.
    for key in saved:
        _same(artifacts['predicted_mean_supported'][key], saved[key], 'replayed053 ' + key)
    comparisons = [(group + '_' + arm, group + '_oracle_selector')
                   for group in GROUPINGS for arm in ARMS[:3]]
    comparisons += [(group + '_anchor', group + '_mean_supported') for group in GROUPINGS]
    comparisons += [('predicted_' + arm, 'oracle_identity_' + arm) for arm in ARMS]
    paired = {a + '__vs__' + b: _comparison(a, b, artifacts, representatives, truth, truth_valid, regions, arrays)
              for a, b in comparisons}
    return dict(methods=methods, pairwise=paired, selector_choices=selections,
                source_replay_verified=replay['replay_verified'], fragment_count=n), arrays


def aggregate_oracles(rows):
    """Point-weighted physical errors plus equal-scene macros and signed effects."""
    result = dict(scene_count=len(rows), methods={}, pairwise={}, selector_choices={})
    for name in METHODS:
        items = [r['methods'][name] for r in rows]
        result['methods'][name] = dict(
            full_target_metrics=pool_metrics([r['system']['full_target_metrics'] for r in items]),
            fixed_target_strata=pool_metrics([r['fixed_target_strata'] for r in items]),
            population={key: sum(r['system'][key] for r in items) for key in POPULATION_KEYS})
    for name, first in rows[0]['pairwise'].items():
        items = [r['pairwise'][name] for r in rows]
        pooled = dict(methods={method: pool_metrics([r['methods'][method] for r in items])
                              for method in first['methods']},
            coverage_transitions={attr: {key: sum(r['coverage_transitions'][attr][key] for r in items)
                                        for key in values}
                                  for attr, values in first['coverage_transitions'].items()},
            error_change_second_minus_first={},
            **{key: sum(r[key] for r in items) for key in
               ('same_representative_target_count', 'changed_representative_target_count')})
        for key in ('newly_covered', 'lost_coverage'):
            pooled[key] = pool_metrics([r[key] for r in items])
        for region, attrs in first['error_change_second_minus_first'].items():
            pooled['error_change_second_minus_first'][region] = {}
            for attr, values in attrs.items():
                changes = [r['error_change_second_minus_first'][region][attr] for r in items]
                totals = {key: sum(r[key] for r in changes) for key in values if key != 'mean_error_change'}
                count = totals['common_point_count']
                eligible = [r['mean_error_change'] for r in changes if r['mean_error_change'] is not None]
                totals.update(mean_error_change=totals['signed_error_change_sum']/count if count else None,
                    scene_macro_error_change=float(np.mean(eligible)) if eligible else None,
                    eligible_scene_count=len(eligible),
                    second_better_scene_count=sum(value < -1e-9 for value in eligible),
                    second_worse_scene_count=sum(value > 1e-9 for value in eligible),
                    tied_scene_count=sum(abs(value) <= 1e-9 for value in eligible))
                pooled['error_change_second_minus_first'][region][attr] = totals
        result['pairwise'][name] = pooled
    for group in GROUPINGS:
        result['selector_choices'][group] = {attr: {
            key: sum(r['selector_choices'][group][attr][key] for r in rows)
            for key in rows[0]['selector_choices'][group][attr]} for attr in ATTRIBUTES}
    return result
