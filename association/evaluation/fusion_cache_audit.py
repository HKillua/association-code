"""067-A: array-only attribution of rejected 057→066 OLS replays.

This does not validate the learned-group transfer or change 066 acceptance. Truth
is used only after fixed output/target correspondence has been checked. No model,
checkpoint, archive, grouping algorithm or inference entry point is imported.
"""
from __future__ import annotations

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.physical import _check_valid_states
from association.evaluation.support import group_input_support
from association.evaluation.system_attribution import ATTRIBUTES, _same

# Frozen 066 limits, duplicated to avoid importing its torch inference module.
REPLAY_TOLERANCE = dict(anchor_weight_atol=1e-6, anchor_weight_rtol=1e-5,
    state_span_factor=1e-5, position_m=.25, speed=.001, heading_deg=.001)
CATEGORIES = ('non_output_fragment', 'unselected_output_group',
              'selected_invalid_reference_or_prediction', 'selected_valid_target_point')
INPUT_KEYS = ('node_ids', 'identities', 'sources', 'target_identity_ids', 'query_seconds',
              'target', 'target_valid', 'native_states', 'support', 'nearest_seconds', 'exact_observation')
SCOPES = ('all_output_groups', 'unique_selected_groups')


def _finite(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in 'uif' or not np.isfinite(value).all():
        raise ValueError('invalid finite array: ' + name)
    return value


def _stats(values):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite values entered valid-point statistics')
    return dict(count=int(values.size), sum=float(values.sum()), squared_sum=float(np.square(values).sum()),
        mean=float(values.mean()) if values.size else None,
        rmse=float(np.sqrt(np.mean(np.square(values)))) if values.size else None,
        p95=float(np.quantile(values, .95)) if values.size else None,
        p99=float(np.quantile(values, .99)) if values.size else None,
        maximum=float(values.max()) if values.size else None,
        minimum=float(values.min()) if values.size else None)


def _metrics(arrays):
    out = {}
    for scope in SCOPES:
        out[scope] = {}
        for attr in ATTRIBUTES:
            prefix = scope + '_' + attr + '_'
            previous, current = arrays[prefix+'previous_error'], arrays[prefix+'current_error']
            out[scope][attr] = dict(previous_error=_stats(previous), current_error=_stats(current),
                signed_error_change=_stats(current-previous),
                absolute_error_change=_stats(abs(current-previous)),
                state_difference=_stats(arrays[prefix+'state_difference']),
                mae_change=float(current.mean()-previous.mean()) if len(current) else None,
                rmse_change=float(np.sqrt(np.square(current).mean())-np.sqrt(np.square(previous).mean()))
                    if len(current) else None)
    return out


def _mapping(fragment, artifact, previous):
    """Validate complete original artifact, including duplicate and invalid rows."""
    native = np.asarray(fragment['native_states'])
    if native.ndim != 3 or native.shape[-1] != 4:
        raise ValueError('native states must have shape [N,Q,4]')
    n, q = native.shape[:2]
    _finite(native, (n, q, 4), 'native_states')
    offsets = np.asarray(fragment['ols_group_offsets'])
    flat = np.asarray(fragment['ols_group_node_indices'])
    if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != n or (np.diff(offsets) <= 0).any()
            or flat.shape != (n,) or flat.dtype.kind not in 'iu'
            or not np.array_equal(np.sort(flat), np.arange(n))):
        raise ValueError('original OLS groups must partition complete input nodes')
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    anchors = np.asarray([g[0] for g in groups], np.int64)
    targets = np.asarray(fragment['target_identity_ids'])
    t = len(targets)
    _same(artifact['target_ids'], targets, '067 original targets')
    truth = np.asarray(artifact['aligned_truth'])[:t]
    valid = np.asarray(artifact['aligned_truth_valid'])[:t]
    if (any(group != sorted(group) for group in groups) or not np.array_equal(anchors, np.sort(anchors))
            or len(set(targets.tolist())) != t):
        raise ValueError('067 fixed group/target ordering invalid')
    nodes = np.asarray(fragment['identities'])
    if nodes.shape != (n,) or targets.ndim != 1 or any(x not in targets for x in nodes):
        raise ValueError('067 node identities must map to declared targets')
    target_index = {identity: index for index, identity in enumerate(targets)}
    node_targets = np.asarray([target_index[x] for x in nodes], np.int64)
    if truth.shape != (t, q, 4) or valid.shape != truth.shape or valid.dtype != bool:
        raise ValueError('067 original reference shape/mask invalid')
    _check_valid_states(truth, valid, '067 original reference')
    _same(fragment['target'], truth[node_targets], '067 fragment/full-target reference')
    _same(fragment['target_valid'], valid[node_targets], '067 fragment/full-target reference mask')
    membership = np.empty(n, np.int64)
    first, duplicate = {}, []
    anchor_targets = node_targets[anchors]
    for gid, group in enumerate(groups):
        membership[group] = gid
        if anchor_targets[gid] in first:
            duplicate.append(gid)
        else:
            first[anchor_targets[gid]] = gid
    aligned_groups = np.full(t+len(duplicate), -1, np.int64)
    for target, gid in first.items():
        aligned_groups[target] = gid
    aligned_groups[t:] = duplicate
    aligned_shape = (len(aligned_groups), q, 4)
    aligned_prediction = np.full(aligned_shape, np.nan)
    aligned_valid = np.zeros(aligned_shape, bool)
    aligned_truth = np.full(aligned_shape, np.nan)
    aligned_truth_valid = np.zeros(aligned_shape, bool)
    aligned_truth[:t], aligned_truth_valid[:t] = truth, valid
    group_valid = np.asarray(artifact['group_valid'])
    if group_valid.shape != (len(groups), q, 4) or group_valid.dtype != bool:
        raise ValueError('067 original group validity shape/type invalid')
    _same(group_valid, np.ones((len(groups), q, 4), bool),
          '067 original finite full-state prediction mask')
    _check_valid_states(previous[anchors], group_valid, '067 original group states')
    target_observed = np.zeros((t, q), bool)
    np.logical_or.at(target_observed, node_targets, fragment['support'])
    observed = np.zeros(aligned_shape[:2], bool)
    observed[:t] = target_observed
    support_kind = np.full(aligned_shape[:2], 3, np.uint8)
    nearest = np.full(aligned_shape[:2], np.inf)
    support = group_input_support(fragment, groups)
    for row, gid in enumerate(aligned_groups):
        if gid < 0:
            continue
        aligned_prediction[row], aligned_valid[row] = previous[anchors[gid]], group_valid[gid]
        if row >= t:
            observed[row] = target_observed[anchor_targets[gid]]
        support_kind[row] = np.where(support['anchor_support'][gid], 0,
            np.where(support['helper_only_support'][gid], 1, 2))
        nearest[row] = support['nearest_seconds'][gid]
    actual = dict(aligned_prediction=aligned_prediction, aligned_prediction_valid=aligned_valid,
        aligned_truth=aligned_truth, aligned_truth_valid=aligned_truth_valid, aligned_observed=observed,
        target_ids=targets.astype(str), aligned_target_ids=np.concatenate((targets.astype(str),
            np.full(len(duplicate), '', dtype=str))), aligned_group_indices=aligned_groups,
        group_states=previous[anchors], group_valid=group_valid, group_anchor_target_indices=anchor_targets,
        group_offsets=offsets, group_node_indices=flat, node_group_indices=membership,
        split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
        aligned_input_support_kind=support_kind, aligned_nearest_input_seconds=nearest,
        input_support_region_names=np.array(['anchor_support', 'helper_only_support', 'unsupported', 'unmatched_target']),
        input_distance_region_names=np.array(['le_300s', 'gt_300_le_600s', 'gt_600_le_1800s', 'gt_1800s', 'no_input']))
    if set(artifact) != set(actual):
        raise ValueError('067 original OLS artifact field set differs')
    for key in actual:
        _same(artifact[key], actual[key], '067 original complete artifact/' + key)
    unique = np.zeros(len(groups), bool)
    unique[list(first.values())] = True
    return groups, anchors, membership, unique, truth, valid


def audit_source(fragment, old_fragment, old_artifact, arm, scene_id, component_id):
    """Return one original-OLS replay row plus arrays for exact pooled tails.

    ``old_artifact`` may be the complete system NPZ or its stripped ols_learned
    artifact. Expected numerical deviations are reported, not reclassified as
    accepted. Corrupt mapping, input, routing, finite-state or weight contracts
    raise. Full violation records live in the per-scene server artifact.
    """
    if arm not in ('helper_quality', 'direct_only', 'mixed_action'):
        raise ValueError('unknown frozen 057 arm')
    for key in INPUT_KEYS:
        _same(fragment[key], old_fragment[key], '067 original input/' + key)
    for key in ('ols_group_offsets', 'ols_group_node_indices'):
        _same(fragment[key], old_fragment[key], '067 original group/' + key)
    artifact = ({key.removeprefix('ols_learned_'): value for key, value in old_artifact.items()
                 if key.startswith('ols_learned_')} if 'target_ids' not in old_artifact else old_artifact)
    native = np.asarray(fragment['native_states'])
    n, q, _ = native.shape
    prefix = 'source057_' + arm + '_'
    current = _finite(fragment[prefix+'prediction_learned'], native.shape, 'current prediction')
    previous = _finite(old_fragment['ols_learned'], native.shape, 'previous prediction')
    groups, anchors, membership, unique, truth, valid = _mapping(fragment, artifact, previous)
    expected = [(a, h) for group in groups for a in group for h in group if h != a]
    routing = {}
    for key in ('helper_anchor', 'helper_node', 'membership'):
        value = np.asarray(fragment[prefix+'routing_'+key])
        _same(value, old_fragment['ols_routing_'+key], '067 routing/' + key)
        routing[key] = value
    ha, hn = routing['helper_anchor'], routing['helper_node']
    if (ha.dtype.kind not in 'iu' or hn.dtype.kind not in 'iu'
            or ha.ndim != 1 or hn.shape != ha.shape or list(zip(ha.tolist(), hn.tolist())) != expected):
        raise ValueError('067 cached helper edge order differs from original fixed groups')
    _same(routing['membership'], membership, '067 routing membership')
    weight, utility, own = {}, {}, {}
    for side, data, route in (('current', fragment, prefix+'routing_'),
                              ('previous', old_fragment, 'ols_routing_')):
        weight[side] = _finite(data[route+'helper_weights'], (len(ha), q, 3), side+' helper weights')
        utility[side] = _finite(data[route+'utility'], (len(ha), q, 3), side+' utility')
        own[side] = _finite(data[route+'anchor_weights'], (n, q, 3), side+' own weights')
        if any(((value < 0) | (value > 1)).any() for value in (weight[side], own[side])):
            raise ValueError('067 weights must be in [0,1]')
        mass = own[side].copy()
        np.add.at(mass, ha, weight[side])
        if not np.allclose(mass, 1., atol=1e-6, rtol=1e-5):
            raise ValueError('067 normalized weight mass differs from one')
    _same(fragment[prefix+'prediction_mean_supported'], old_fragment['ols_mean_supported'],
          '067 ordinary mean replay')
    limits = REPLAY_TOLERANCE
    physical = attribute_errors(current, previous)
    attrs = tuple(ATTRIBUTES)
    target_rows = np.asarray(artifact['group_anchor_target_indices'])
    gv = np.asarray(artifact['group_valid'])
    node_ids = np.asarray(fragment['node_ids'])
    query = np.asarray(fragment['query_seconds'])
    if query.shape != (q,) or not np.isfinite(query).all():
        raise ValueError('067 complete common query time array required')

    def locate(node, time, attr):
        gid, col = int(membership[node]), ATTRIBUTES[attr]
        target = int(target_rows[gid])
        is_output = int(anchors[gid]) == node
        is_selected = is_output and bool(unique[gid])
        usable = bool(gv[gid, time, list(col)].all() and valid[target, time, list(col)].all())
        category = (CATEGORIES[0] if not is_output else CATEGORIES[1] if not is_selected
                    else CATEGORIES[2] if not usable else CATEGORIES[3])
        result = dict(anchor_node=int(node), anchor_node_id=str(node_ids[node]),
            anchor_identity=str(fragment['identities'][node]), query_index=int(time),
            query_seconds=float(query[time]), attribute=attr, group_index=gid,
            group_first_node=int(anchors[gid]), group_anchor_target_index=target,
            group_anchor_target_id=str(artifact['target_ids'][target]), category=category,
            actual_output_group=is_output, unique_selected_group=is_selected,
            actual_output_attribute_evaluable=bool(is_output and usable),
            state_difference=float(physical[attr][node, time]))
        if is_output and usable:
            old_error = float(attribute_errors(previous[node:node+1, time:time+1], truth[target:target+1, time:time+1])[attr][0, 0])
            new_error = float(attribute_errors(current[node:node+1, time:time+1], truth[target:target+1, time:time+1])[attr][0, 0])
            result.update(previous_error=old_error, current_error=new_error, error_change=new_error-old_error)
        return result

    records = {}
    wbound = limits['anchor_weight_atol']+limits['anchor_weight_rtol']*abs(weight['previous'])
    ownbound = limits['anchor_weight_atol']+limits['anchor_weight_rtol']*abs(own['previous'])
    for name, mask in (
            ('helper_weight', abs(weight['current']-weight['previous']) > wbound),
            ('utility_activation_flip', (utility['current'] > 0) != (utility['previous'] > 0))):
        records[name] = []
        for edge, time, index in np.argwhere(mask):
            node, helper, attr = int(ha[edge]), int(hn[edge]), attrs[index]
            rec = locate(node, int(time), attr)
            rec.update(helper_edge_index=int(edge), helper_node=helper, helper_node_id=str(node_ids[helper]),
                previous_helper_weight=float(weight['previous'][edge, time, index]),
                current_helper_weight=float(weight['current'][edge, time, index]),
                previous_utility=float(utility['previous'][edge, time, index]),
                current_utility=float(utility['current'][edge, time, index]),
                weight_tolerance=float(wbound[edge, time, index]))
            records[name].append(rec)
    records['anchor_weight'] = []
    for node, time, index in np.argwhere(abs(own['current']-own['previous']) > ownbound):
        rec = locate(int(node), int(time), attrs[index])
        rec.update(previous_anchor_weight=float(own['previous'][node, time, index]),
                   current_anchor_weight=float(own['current'][node, time, index]),
                   weight_tolerance=float(ownbound[node, time, index]))
        records['anchor_weight'].append(rec)
    records['physical'] = []
    for attr, key in (('position', 'position_m'), ('speed', 'speed'), ('heading', 'heading_deg')):
        for node, time in np.argwhere(physical[attr] > limits[key]):
            rec = locate(int(node), int(time), attr)
            rec['physical_tolerance'] = limits[key]
            records['physical'].append(rec)
    # Retain the historical coordinate/span check as a separate finite diagnostic.
    delta = current-previous
    native_delta = native[hn]-native[ha]
    for col in (1, 3):
        delta[..., col] = (delta[..., col]+180.) % 360.-180.
        native_delta[..., col] = (native_delta[..., col]+180.) % 360.-180.
    span = np.zeros_like(native)
    np.maximum.at(span, ha, abs(native_delta))
    span_bound = limits['state_span_factor']*span + 32*np.finfo(np.float64).eps*np.maximum(abs(previous), 1.)
    records['state_span'] = []
    for node, time, col in np.argwhere(abs(delta) > span_bound):
        rec = locate(int(node), int(time), 'position' if col < 2 else 'speed' if col == 2 else 'heading')
        rec.update(coordinate_index=int(col), signed_coordinate_difference=float(delta[node, time, col]),
                   coordinate_tolerance=float(span_bound[node, time, col]))
        records['state_span'].append(rec)

    # Float64 arithmetic audit, deliberately NOT a replacement 066 tolerance.
    # The bound is fixed before data inspection, scales with actual summation
    # magnitude/length, and includes final subtraction of large predicted speeds.
    dv = native[hn, :, 2]-native[ha, :, 2]
    terms = (weight['current'][..., 1]-weight['previous'][..., 1])*dv
    expected_delta = np.zeros((n, q), np.float64)
    magnitude = np.zeros((n, q), np.float64)
    np.add.at(expected_delta, ha, terms)
    np.add.at(magnitude, ha, (abs(weight['current'][..., 1])+abs(weight['previous'][..., 1]))*abs(dv))
    helper_count = np.bincount(ha, minlength=n)
    speed_delta = current[..., 2]-previous[..., 2]
    residual = speed_delta-expected_delta
    algebra_bound = (64*np.finfo(np.float64).eps*(1+helper_count[:, None])
                     *(1+abs(current[..., 2])+abs(previous[..., 2])+magnitude))
    algebra_mask = abs(residual) > algebra_bound
    peak = np.unravel_index(np.argmax(abs(speed_delta)), speed_delta.shape)
    node, time = map(int, peak)
    peak_record = locate(node, time, 'speed')
    peak_record.update(previous_speed=float(previous[node, time, 2]), current_speed=float(current[node, time, 2]),
        native_anchor_speed=float(native[node, time, 2]), signed_speed_difference=float(speed_delta[peak]),
        weight_decomposition=float(expected_delta[peak]), residual=float(residual[peak]),
        arithmetic_bound=float(algebra_bound[peak]), helpers=[dict(helper_edge_index=int(edge),
            helper_node=int(hn[edge]), native_speed=float(native[hn[edge], time, 2]),
            native_minus_anchor=float(dv[edge, time]), previous_weight=float(weight['previous'][edge, time, 1]),
            current_weight=float(weight['current'][edge, time, 1]), contribution=float(terms[edge, time]))
            for edge in np.flatnonzero(ha == node)])
    arrays = {}
    old_error = attribute_errors(previous[anchors], truth[target_rows])
    new_error = attribute_errors(current[anchors], truth[target_rows])
    for scope, selected in (('all_output_groups', np.ones(len(groups), bool)), ('unique_selected_groups', unique)):
        for attr, columns in ATTRIBUTES.items():
            predmask = selected[:, None] & gv[..., list(columns)].all(-1)
            common = predmask & valid[target_rows][..., list(columns)].all(-1)
            key = scope+'_'+attr+'_'
            arrays[key+'previous_error'] = old_error[attr][common]
            arrays[key+'current_error'] = new_error[attr][common]
            arrays[key+'state_difference'] = physical[attr][anchors][predmask]
    counts = {}
    for kind, items in records.items():
        counts[kind] = {}
        for category in CATEGORIES:
            selected = [item for item in items if item['category'] == category]
            counts[kind][category] = dict(record_count=len(selected),
                distinct_anchor_query_attribute_count=len({(r['anchor_node'], r['query_index'], r['attribute']) for r in selected}))
    rejected_records = [record for kind in ('helper_weight', 'anchor_weight', 'physical', 'state_span')
                        for record in records[kind]]
    adopted = [record for record in rejected_records if record['category'] == CATEGORIES[3]]
    key = lambda record: (record['anchor_node'], record['query_index'], record['attribute'])
    effects = dict(mapping_categories={category: any(record['category'] == category for record in rejected_records)
                                      for category in CATEGORIES},
        mapped_unique_valid_statepoint_count=len({key(record) for record in adopted}),
        unique_valid_changed_statepoint_count=len({key(record) for record in adopted if record['state_difference'] > 0}),
        unique_valid_changed_error_point_count=len({key(record) for record in adopted if record['error_change'] != 0}),
        difference_definition='strictly nonzero stored-state physical difference / signed GT-error difference; not engineering significance')
    row = dict(scene_id=str(scene_id), component_id=str(component_id), arm=arm,
        original_grouping='OLS', learned_group_transfer_replay_claimed=False, original066_acceptance='REJECTED',
        complete_old_artifact_and_inputs_verified=True, rejection_effects=effects,
        population=dict(fragment_count=n, group_count=len(groups), unique_selected_groups=int(unique.sum()),
                        duplicate_groups=int((~unique).sum()), target_count=len(truth)),
        original_replay_tolerance=limits.copy(), violation_records=records, violation_mapping_counts=counts,
        rejected_output_replay=any(records[key] for key in ('helper_weight', 'anchor_weight', 'physical', 'state_span')),
        speed_algebra=dict(point_count=n*q, exceeds_arithmetic_bound_count=int(algebra_mask.sum()),
            maximum_absolute_residual=float(abs(residual).max(initial=0)),
            maximum_residual_to_bound=float((abs(residual)/algebra_bound).max(initial=0)),
            definition='delta speed = sum(delta normalized helper weight * (native helper speed-native anchor speed))',
            arithmetic_bound='64*float64_eps*(1+helper_count)*(1+abs(current_speed)+abs(previous_speed)+sum((abs(w_current)+abs(w_previous))*abs(native_speed_difference)))',
            is066_acceptance_gate=False), largest_speed_difference=peak_record, metrics=_metrics(arrays))
    return row, arrays


def aggregate(rows, arrays):
    """Pool original-group metrics; keep components and source arms separate."""
    if not rows or len(rows) != len(arrays):
        raise ValueError('nonempty matched rows/point arrays required')
    if len({(r['scene_id'], r['arm']) for r in rows}) != len(rows):
        raise ValueError('duplicate scene/arm cache audit')
    expected_keys = set(arrays[0])
    if any(set(a) != expected_keys for a in arrays):
        raise ValueError('inconsistent cache audit point-array fields')

    def pool(indices):
        selected = [rows[i] for i in indices]
        pooled = {key: np.concatenate([arrays[i][key] for i in indices]) for key in expected_keys}
        counts = {kind: {category: {key: sum(row['violation_mapping_counts'][kind][category][key]
            for row in selected) for key in ('record_count', 'distinct_anchor_query_attribute_count')}
            for category in CATEGORIES} for kind in selected[0]['violation_mapping_counts']}
        return dict(scene_count=len(selected), rejected_replay_count=sum(bool(r['rejected_output_replay']) for r in selected),
            violation_mapping_counts=counts, metrics=_metrics(pooled),
            rejected_replay_count_by_mapping_category={category: sum(row['rejection_effects']['mapping_categories'][category]
                for row in selected) for category in CATEGORIES},
            rejected_replays_with_unique_valid_mapped_violation=sum(row['rejection_effects']['mapped_unique_valid_statepoint_count'] > 0 for row in selected),
            rejected_replays_with_unique_valid_effect=dict(
                state_changed=sum(row['rejection_effects']['unique_valid_changed_statepoint_count'] > 0 for row in selected),
                truth_error_changed=sum(row['rejection_effects']['unique_valid_changed_error_point_count'] > 0 for row in selected)),
            speed_algebra=dict(point_count=sum(r['speed_algebra']['point_count'] for r in selected),
                exceeds_arithmetic_bound_count=sum(r['speed_algebra']['exceeds_arithmetic_bound_count'] for r in selected),
                maximum_absolute_residual=max(r['speed_algebra']['maximum_absolute_residual'] for r in selected),
                maximum_residual_to_bound=max(r['speed_algebra']['maximum_residual_to_bound'] for r in selected)),
            largest_speed_examples=[dict(scene_id=r['scene_id'], component_id=r['component_id'],
                **r['largest_speed_difference']) for r in sorted(selected,
                key=lambda r: r['largest_speed_difference']['state_difference'], reverse=True)[:10]])
    arms = sorted({r['arm'] for r in rows})
    return dict(scope='067-A cached original057 OLS replay only; no bound on learned056-group transfer',
        original066_acceptance='REJECTED', reenactment_or_acceptance_change=False,
        replay_count=len(rows), scene_count=len({r['scene_id'] for r in rows}),
        rejected_replay_count=sum(bool(r['rejected_output_replay']) for r in rows),
        rejected_replay_count_by_mapping_category={category: sum(row['rejection_effects']['mapping_categories'][category]
            for row in rows) for category in CATEGORIES},
        rejected_replays_with_unique_valid_mapped_violation=sum(row['rejection_effects']['mapped_unique_valid_statepoint_count'] > 0 for row in rows),
        rejected_replays_with_unique_valid_effect=dict(
            state_changed=sum(row['rejection_effects']['unique_valid_changed_statepoint_count'] > 0 for row in rows),
            truth_error_changed=sum(row['rejection_effects']['unique_valid_changed_error_point_count'] > 0 for row in rows)),
        replay_category_counts_nonexclusive=True,
        original_replay_tolerance=REPLAY_TOLERANCE.copy(),
        by_arm={arm: pool([i for i, row in enumerate(rows) if row['arm'] == arm]) for arm in arms},
        by_component={component: {arm: pool([i for i, row in enumerate(rows)
            if row['component_id'] == component and row['arm'] == arm]) for arm in arms
            if any(row['component_id'] == component and row['arm'] == arm for row in rows)}
            for component in sorted({r['component_id'] for r in rows})},
        interpretation='all_output_groups repeats representative target truth for duplicate groups; unique_selected_groups uses the fixed first group once per target. Missing/invalid references never enter errors. Quantiles pool points; components are descriptive, not independent-test evidence.')
