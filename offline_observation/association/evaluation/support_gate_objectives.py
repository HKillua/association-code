"""069 read-only position objectives on the complete frozen 068 population.

No model, checkpoint, archive or training imports. Scene-target records retain
their original identity correspondence; shared identities are never deduplicated
across scenes. Missing predictions/references remain in population accounting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.system_attribution import _same

ARMS = ('anchor', 'mean_supported', 'attribute_hybrid', 'support_switch', 'learned_gate')
METHODS = ('initial', 'final', 'anchor', 'mean_supported', 'attribute_hybrid', 'support_switch')
REGIONS = ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support', 'no_predicted_group')
PAIRS = (('final', 'initial'), ('final', 'support_switch'), ('initial', 'support_switch'))
WEIGHTINGS = ('point_equal', 'scene_target_equal', 'scene_equal_point_mean', 'scene_equal_target_mean')
OBJECTIVES = ('mae', 'mse', 'log1p')
POINT_ERROR_ATOL_M = 1e-9


def _read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _artifact(data, arm):
    return {key[len(arm)+1:]: value for key, value in data.items() if key.startswith(arm+'_')}


def _fixed(a, b, label, excluded=()):
    if set(a) != set(b):
        raise ValueError('069 cached fields changed: '+label)
    for key in a:
        if key not in excluded:
            _same(a[key], b[key], '069 '+label+'/'+key)


def _structure(fragment, base):
    """Check the original ordered partition and first representative, without regrouping."""
    ids, targets = fragment['identities'], fragment['target_identity_ids']
    n, t, q = len(ids), len(targets), len(fragment['query_seconds'])
    if len(set(targets)) != t or len(set(fragment['node_ids'])) != n:
        raise ValueError('069 target/node IDs must retain unique source rows')
    offsets, flat = fragment['learned_assoc_group_offsets'], fragment['learned_assoc_group_node_indices']
    if (offsets.ndim != 1 or len(offsets) < 2 or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0)
            or flat.shape != (n,) or flat.dtype.kind not in 'iu' or not np.array_equal(np.sort(flat), np.arange(n))):
        raise ValueError('069 cached groups must partition the original complete nodes')
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    if any(group != sorted(group) for group in groups) or [g[0] for g in groups] != sorted(g[0] for g in groups):
        raise ValueError('069 cached group order or first representative changed')
    anchors = np.array([g[0] for g in groups], np.int64)
    _same(fragment['group_anchor_node_indices'], anchors, '069 original representatives')
    _same(base['target_ids'], targets, '069 complete target registry')
    _same(base['group_offsets'], offsets, '069 group offsets')
    _same(base['group_node_indices'], flat, '069 group members')
    lookup = {str(value): i for i, value in enumerate(targets)}
    if any(str(value) not in lookup for value in ids):
        raise ValueError('069 node identity missing from complete target registry')
    anchor_targets = np.array([lookup[str(ids[i])] for i in anchors], np.int64)
    _same(base['group_anchor_target_indices'], anchor_targets, '069 anchor target rows')
    selected = np.full(t, -1, np.int64)
    duplicates, membership = [], np.empty(n, np.int64)
    for gid, (group, target) in enumerate(zip(groups, anchor_targets)):
        membership[group] = gid
        if selected[target] >= 0:
            duplicates.append(gid)
        else:
            selected[target] = gid
    aligned = np.r_[selected, duplicates].astype(np.int64)
    _same(base['node_group_indices'], membership, '069 node membership')
    _same(base['aligned_group_indices'], aligned, '069 fixed first-group alignment including duplicates')
    _same(base['aligned_target_ids'], np.r_[targets, np.full(len(duplicates), '', dtype=targets.dtype)],
          '069 duplicate outputs have no independent target row')
    shape = (t+len(duplicates), q, 4)
    if (base['aligned_truth'].shape != shape or base['aligned_truth_valid'].shape != shape
            or base['aligned_truth_valid'].dtype != bool or base['aligned_prediction_valid'].shape != shape
            or base['aligned_prediction_valid'].dtype != bool or base['group_valid'].shape != (len(groups), q, 4)
            or base['group_valid'].dtype != bool or not base['group_valid'].all()):
        raise ValueError('069 original truth/prediction masks must retain complete finite output population')
    if base['aligned_truth_valid'][t:].any() or not np.isnan(base['aligned_truth'][t:]).all():
        raise ValueError('069 duplicate rows must not acquire truth')
    if not np.isfinite(base['aligned_truth'][base['aligned_truth_valid']]).all():
        raise ValueError('069 reference values declared valid must be finite')
    for key in ('support', 'exact_observation'):
        if fragment[key].shape != (n, q) or fragment[key].dtype != bool:
            raise ValueError('069 support/exact input masks must remain boolean')
    if fragment['native_states'].shape != (n, q, 4) or not np.isfinite(fragment['native_states']).all():
        raise ValueError('069 native motion must be finite')
    supported = np.array([fragment['support'][g[1:]].any(0) for g in groups])
    group_regions = np.where(fragment['support'][anchors], 0, np.where(supported, 1, 2))
    expected_kind = np.full((len(aligned), q), 3, np.uint8)
    expected_kind[aligned >= 0] = group_regions[aligned[aligned >= 0]]
    _same(base['aligned_input_support_kind'], expected_kind, '069 observable support regions')
    _same(fragment['group_eligible'], group_regions == 1, '069 group opportunity mask')
    regions = dict(all=np.ones((t, q), bool), **{name: expected_kind[:t] == i for i, name in enumerate(REGIONS[1:])})
    for region in REGIONS:
        _same(fragment['region_'+region], regions[region], '069 cached region/'+region)
    population = dict(target_identity_count=t, predicted_group_count=len(groups),
        duplicate_prediction_group_count=len(duplicates), anchor_matched_target_count=int((selected >= 0).sum()),
        reference_unavailable_target_count=int((~base['aligned_truth_valid'][:t].any(axis=(1, 2))).sum()))
    return groups, selected, regions, population


def _check_artifact(fragment, artifact, base, arm):
    _fixed(artifact, base, 'fixed system/'+arm, excluded=('aligned_prediction', 'group_states'))
    states = artifact['group_states']
    if states.shape != base['group_valid'].shape or not np.isfinite(states).all():
        raise ValueError('069 every output group must retain finite states, including duplicates')
    _same(fragment[arm+'_group_states'], states, '069 fragment/system group states/'+arm)
    aligned = np.full_like(artifact['aligned_prediction'], np.nan)
    validity = np.zeros_like(artifact['aligned_prediction_valid'])
    for row, group in enumerate(base['aligned_group_indices']):
        if group >= 0:
            aligned[row], validity[row] = states[group], base['group_valid'][group]
    _same(artifact['aligned_prediction'], aligned, '069 group-to-output mapping/'+arm)
    _same(artifact['aligned_prediction_valid'], validity, '069 group-to-output validity/'+arm)
    t = len(base['target_ids'])
    common = (base['aligned_truth_valid'][:t, :, :2] & validity[:t, :, :2]).all(-1)
    errors = attribute_errors(aligned[:t], base['aligned_truth'][:t])['position']
    cached = fragment[arm+'_position_error']
    _same(np.isfinite(cached), common, '069 saved error mask/'+arm)
    if cached.shape != errors.shape or not np.isfinite(errors[common]).all() or not np.allclose(
            errors[common], cached[common], rtol=0., atol=POINT_ERROR_ATOL_M):
        raise ValueError('069 reconstructed Haversine error differs from saved error/'+arm)
    return np.where(common, errors, np.nan)


def load_scene(initial_dir, final_dir, sid):
    """Validate full output artifacts, then return complete unique-target arrays."""
    sid = str(sid)
    fragments, systems = {}, {}
    for endpoint, directory in (('initial', initial_dir), ('final', final_dir)):
        directory = Path(directory)
        fragments[endpoint] = _read(directory/'predictions'/f'states-{sid}.npz')
        systems[endpoint] = _read(directory/'system-predictions'/f'states-{sid}.npz')
    old, new = fragments['initial'], fragments['final']
    excluded = tuple(key for key in old if key.startswith('learned_gate_'))+('group_gate', 'group_logits')
    _fixed(old, new, 'endpoint physical inputs', excluded=excluded)
    base = _artifact(systems['initial'], 'anchor')
    groups, selected, regions, population = _structure(old, base)
    errors, model_states = {}, {}
    for endpoint in ('initial', 'final'):
        fragment, system = fragments[endpoint], systems[endpoint]
        if set(system) != {arm+'_'+key for arm in ARMS for key in base}:
            raise ValueError('069 requires exactly all five system arms')
        for arm in ARMS:
            artifact = _artifact(system, arm)
            values = _check_artifact(fragment, artifact, base, arm)
            if arm != 'learned_gate':
                _fixed(artifact, _artifact(systems['initial'], arm), 'endpoint ordinary/'+arm)
                errors[arm] = values
            else:
                errors[endpoint] = values
                model_states[endpoint] = artifact['group_states']
                _same(artifact['group_states'][..., 2:], _artifact(system, 'mean_supported')['group_states'][..., 2:],
                      '069 learned speed and heading remain Mean')
                outside = ~fragment['group_eligible']
                _same(artifact['group_states'][..., :2][outside], base['group_states'][..., :2][outside],
                      '069 position outside opportunity remains Anchor')
        gate, logits = fragment['group_gate'], fragment['group_logits']
        if (gate.shape != fragment['group_eligible'].shape or logits.shape != gate.shape
                or not np.isfinite(gate).all() or not np.isfinite(logits).all()
                or (gate < 0).any() or (gate > 1).any()):
            raise ValueError('069 saved route must remain finite and bounded')
    valid = np.isfinite(errors['initial'])
    for method in METHODS:
        _same(np.isfinite(errors[method]), valid, '069 common unique target point mask/'+method)
    delta = attribute_errors(model_states['final'], model_states['initial'])['position']
    group_select = np.zeros(len(groups), bool)
    group_select[selected[selected >= 0]] = True
    comparison = {}
    for scope, use in (('all_output_groups', np.ones(len(groups), bool)), ('unique_selected_groups', group_select)):
        a, b = model_states['initial'][use], model_states['final'][use]
        comparison[scope] = dict(query_count=int(delta[use].size),
            position_coordinate_changed_count=int(np.any(a[..., :2] != b[..., :2], axis=-1).sum()),
            position_haversine_changed_count=int((delta[use] > POINT_ERROR_ATOL_M).sum()),
            max_position_change_m=float(delta[use].max()) if delta[use].size else None,
            speed_exact_equal=bool(np.array_equal(a[..., 2], b[..., 2])),
            heading_exact_equal=bool(np.array_equal(a[..., 3], b[..., 3])))
    t = len(base['target_ids'])
    return dict(scene_id=sid, target_ids=base['target_ids'], errors={method: errors[method] for method in METHODS},
        regions=regions, population=population, model_state_comparison=comparison,
        query_seconds=old['query_seconds'], truth=base['aligned_truth'][:t], truth_valid=base['aligned_truth_valid'][:t],
        selected_group_by_target=selected, groups=groups,
        group_anchor_target_indices=base['group_anchor_target_indices'],
        native_inputs={key: old[key] for key in ('node_ids', 'identities', 'native_states', 'support',
                                               'nearest_seconds', 'query_seconds')},
        fixed_cache_verified=True, reconstructed_position_errors_verified=True)


def _objectives(values):
    return np.stack((values, np.square(values), np.log1p(values/1000.)), axis=-1)


def _metric(values, count):
    if not count:
        return dict(unit_count=0, mae=None, mse=None, rmse=None, log1p=None)
    mean = np.asarray(values).mean(axis=0)
    return dict(unit_count=int(count), **{name: float(value) for name, value in zip(OBJECTIVES, mean)},
                rmse=float(np.sqrt(mean[1])))


def _region(rows, region):
    methods, units = {}, {}
    counts = dict(point_count=0, scene_target_record_count=0, nonempty_scene_count=0)
    for method in METHODS:
        pooled, targets, scene_points, scene_targets = [], [], [], []
        for row in rows:
            error = row['errors'][method]
            mask = row['regions'][region] & np.isfinite(error)
            points = _objectives(error[mask])
            target_rows = [_objectives(error[i, use]).mean(0) for i, use in enumerate(mask) if use.any()]
            pooled.append(points)
            targets.extend(target_rows)
            if len(points):
                scene_points.append(points.mean(0))
                scene_targets.append(np.mean(target_rows, axis=0))
        units[method] = dict(point_equal=np.concatenate(pooled),
            scene_target_equal=np.asarray(targets).reshape(-1, 3),
            scene_equal_point_mean=np.asarray(scene_points).reshape(-1, 3),
            scene_equal_target_mean=np.asarray(scene_targets).reshape(-1, 3))
        methods[method] = {name: _metric(value, len(value)) for name, value in units[method].items()}
        point_errors = units[method]['point_equal'][:, 0]
        methods[method]['point_equal'].update(
            p95=float(np.quantile(point_errors, .95, method='linear')) if len(point_errors) else None,
            p99=float(np.quantile(point_errors, .99, method='linear')) if len(point_errors) else None)
    counts.update(point_count=len(units['initial']['point_equal']),
                  scene_target_record_count=len(units['initial']['scene_target_equal']),
                  nonempty_scene_count=len(units['initial']['scene_equal_point_mean']))
    paired = {}
    for candidate, control in PAIRS:
        name = candidate+'__minus__'+control
        paired[name] = {}
        for weighting in WEIGHTINGS:
            a, b = units[candidate][weighting], units[control][weighting]
            if a.shape != b.shape:
                raise ValueError('069 paired aggregation units disagree')
            delta = a-b
            paired[name][weighting] = dict(unit_count=len(delta),
                **{objective: dict(signed_sum=float(delta[:, i].sum()),
                    mean_difference=float(delta[:, i].mean()) if len(delta) else None,
                    negative_unit_count=int((delta[:, i] < 0).sum()),
                    positive_unit_count=int((delta[:, i] > 0).sum()),
                    exact_tie_unit_count=int((delta[:, i] == 0).sum())) for i, objective in enumerate(OBJECTIVES)},
                rmse_difference=(methods[candidate][weighting]['rmse']-methods[control][weighting]['rmse']) if len(delta) else None)
    return dict(**counts, methods=methods, paired=paired)


def summarize(rows, components):
    """Fixed support strata, four explicitly different weighting populations."""
    ids = [row['scene_id'] for row in rows]
    if not ids or len(set(ids)) != len(ids) or set(components) != set(ids):
        raise ValueError('069 requires one complete nonempty scene list and exact component registry')
    for row in rows:
        shape = row['errors']['initial'].shape
        if len(shape) != 2 or len(row['target_ids']) != shape[0] or len(set(row['target_ids'])) != shape[0]:
            raise ValueError('069 complete scene-target rows required')
        valid = np.isfinite(row['errors']['initial'])
        for method in METHODS:
            error = row['errors'][method]
            _same(np.isfinite(error), valid, '069 summarize paired masks/'+method)
            if (error[valid] < 0).any():
                raise ValueError('069 physical errors cannot be negative')
        for region in REGIONS:
            mask = row['regions'][region]
            if mask.shape != shape or mask.dtype != bool:
                raise ValueError('069 support stratum masks must remain boolean and aligned')
        _same(row['regions']['all'], np.ones(shape, bool), '069 full region must retain all targets')
        if not (sum(row['regions'][name].astype(np.int8) for name in REGIONS[1:]) == 1).all():
            raise ValueError('069 visible support strata must partition the full population')
    population = {key: sum(row['population'][key] for row in rows) for key in rows[0]['population']}
    result = dict(status='CACHE_DIAGNOSTIC_COMPLETE', scene_count=len(rows), scene_ids=ids,
        component_count=len(set(components.values())), population=population,
        scene_target_record_count=sum(len(row['target_ids']) for row in rows),
        distinct_identity_string_count=len({str(identity) for row in rows for identity in row['target_ids']}),
        regions={region: _region(rows, region) for region in REGIONS},
        per_scene=[dict(scene_id=row['scene_id'], component_id=components[row['scene_id']],
            population=row['population'], model_state_comparison=row['model_state_comparison'],
            regions={region: _region([row], region) for region in REGIONS}) for row in rows],
        components={})
    for component in sorted(set(components.values())):
        selected = [row for row in rows if components[row['scene_id']] == component]
        result['components'][component] = dict(scene_count=len(selected), scene_ids=[r['scene_id'] for r in selected],
            regions={region: _region(selected, region) for region in REGIONS})
    result['definitions'] = dict(
        point_equal='Every valid unique-target query point has equal weight; duplicates excluded after fixed correspondence.',
        scene_target_equal='Average per-(scene,target) means with equal weight; same identity in different scenes stays a separate record.',
        scene_equal_point_mean='Equal nonempty scenes, each summarized by its valid-point mean.',
        scene_equal_target_mean='Equal nonempty scenes, each summarized by equal-weight nonempty target means.',
        rmse='Square root of MSE under the named weighting; never an average of target/scene RMSE values.',
        log1p='log1p(Haversine metres/1000), same scalar transform as068; V output-representative weighting is not its T sampler.',
        missing='No covered/error point invented; full target population and empty strata/target counts retained separately.',
        paired='Candidate minus control; negative means smaller error/objective. Counts use strict zero, not engineering thresholds.',
        components='Descriptive shared-identity components, not independent replications or global identity-level estimates.',
        cache_error_tolerance_m=POINT_ERROR_ATOL_M)
    return result


def verify_source_summaries(rows, initial_summary, final_summary):
    """Bind point metrics/population to068; use its existing aggregate tolerances."""
    ids = [row['scene_id'] for row in rows]
    for endpoint, source in (('initial', initial_summary), ('final', final_summary)):
        if source['scene_ids'] != ids or source['status'] != 'COMPLETED':
            raise ValueError('069 source summary must describe complete scenes in fixed order')
        for arm in ARMS:
            method = endpoint if arm == 'learned_gate' else arm
            source_method = source['methods'][arm]
            actual_pop = {key: sum(row['population'][key] for row in rows) for key in rows[0]['population']}
            if source_method['population'] != actual_pop:
                raise ValueError('069 complete population differs from068 summary')
            for region in REGIONS:
                values = np.concatenate([row['errors'][method][row['regions'][region] &
                         np.isfinite(row['errors'][method])] for row in rows])
                expected = source_method['common_target_strata'][region]['position']
                if len(values) != expected['intersection_count']:
                    raise ValueError('069 point population differs from068 summary')
                actual = dict(mae=float(values.mean()) if len(values) else None,
                    rmse=float(np.sqrt(np.square(values).mean())) if len(values) else None)
                for metric, value in actual.items():
                    old = expected[metric]
                    if (value is None and old is not None) or (value is not None and (
                            old is None or not np.isclose(value, old, rtol=1e-10, atol=1e-9))):
                        raise ValueError('069 reconstructed metric differs from068 summary/'+metric)
    return dict(source_population_verified=True, source_position_metrics_verified=True,
                aggregate_rtol=1e-10, aggregate_atol=1e-9)
