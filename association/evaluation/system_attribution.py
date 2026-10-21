"""054 cached-output counterfactuals. No model, fitting, or archive access.

Identity labels only audit frozen groups and align evaluation rows. They never
repair a group, choose a lower-error representative, or generate a prediction.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import numpy as np

from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.fusion_controls import attribute_errors

BASE = 'ols_mean_supported'
INDEPENDENT = 'fragment_ols'
ANCHOR = 'group_anchor_ols'
ATTRIBUTES = {'position': (0, 1), 'speed': (2,), 'heading': (3,)}
REASONS = ('reference_unavailable', 'covered', 'no_input_nodes',
           'wrong_merge_no_representative', 'selected_prediction_unavailable',
           'no_valid_time_intersection')
COUNT_KEYS = ('available_target_count', 'available_prediction_count', 'intersection_count',
              'missing_prediction_count', 'prediction_without_target_count',
              'available_target_trajectory_count', 'covered_target_trajectory_count',
              'missing_target_trajectory_count')


def _same(actual, expected, name):
    a, b = np.asarray(actual), np.asarray(expected)
    equal = (a.shape == b.shape and
             (np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc' and b.dtype.kind in 'fc'
              else np.array_equal(a, b)))
    if not equal:
        raise ValueError(f'cached replay mismatch: {name}')


def assert_metrics(actual, expected, name='metrics'):
    """Compare shared physical metrics, retaining source-only p95 diagnostics."""
    for region, rows in expected.items():
        for attr in ATTRIBUTES:
            for key in (*COUNT_KEYS, 'mae', 'mse', 'rmse', 'coverage', 'trajectory_coverage'):
                if key not in rows[attr]:
                    continue
                a, b = actual[region][attr][key], rows[attr][key]
                if a is None or b is None:
                    equal = a is b
                elif key in COUNT_KEYS:
                    equal = a == b
                else:
                    equal = math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)
                if not equal:
                    raise ValueError(f'cached replay mismatch: {name}/{region}/{attr}/{key}')


def pool_metrics(scenes):
    """Point-weighted errors and separate equal-scene macros, including nulls."""
    result = {}
    if not scenes:
        raise ValueError('cannot pool an empty scene population')
    if any(set(scene) != set(scenes[0]) for scene in scenes):
        raise ValueError('region sets differ across scenes')
    for region in scenes[0]:
        result[region] = {}
        for attr in ATTRIBUTES:
            rows = [s[region][attr] for s in scenes]
            out = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
            n = out['intersection_count']
            for k in ('mae', 'mse'):
                out[k] = sum(r[k] * r['intersection_count'] for r in rows
                             if r['intersection_count']) / n if n else None
            out['rmse'] = math.sqrt(out['mse']) if n else None
            for k, num, den in (('coverage', 'intersection_count', 'available_target_count'),
                               ('trajectory_coverage', 'covered_target_trajectory_count',
                                'available_target_trajectory_count')):
                out[k] = out[num] / out[den] if out[den] else None
            for k in ('mae', 'coverage', 'trajectory_coverage'):
                values = [r[k] for r in rows if r[k] is not None]
                out['scene_macro_' + k] = float(np.mean(values)) if values else None
                out['scene_macro_' + k + '_scene_count'] = len(values)
            out['scene_count'] = len(rows)
            out['unit'] = rows[0]['unit']
            result[region][attr] = out
    return result


def _metrics(prediction, truth, truth_valid, valid, regions):
    return {name: evaluate_aligned_fusion(prediction, truth, truth_valid,
                prediction_valid=valid, region_mask=mask) for name, mask in regions.items()}


def _regions(support, nearest, exact=None):
    regions = dict(all=np.ones(support.shape, bool), inside_support=support,
                   outside_support=~support,
                   far_gt_1800s=(nearest > 1800) & np.isfinite(nearest),
                   distance_le_300s=nearest <= 300,
                   distance_gt_300_le_600s=(nearest > 300) & (nearest <= 600),
                   distance_gt_600_le_1800s=(nearest > 600) & (nearest <= 1800),
                   no_input=~np.isfinite(nearest))
    if exact is not None:
        regions.update(exact_observation=exact, not_exact_observation=~exact)
    return regions


def _attribution(nodes, targets, groups, artifact, native_valid):
    counts, rows_by_attr = {}, {}
    t = len(targets)
    for attr, columns in ATTRIBUTES.items():
        truth = artifact['aligned_truth_valid'][:t, :, list(columns)].all(axis=-1)
        pred = artifact['aligned_prediction_valid'][:t, :, list(columns)].all(axis=-1)
        group_valid = artifact['group_valid'][..., list(columns)].all(axis=-1)
        native = native_valid[..., list(columns)].all(axis=-1)
        primary, overlap, records = Counter({k: 0 for k in REASONS}), Counter(), []
        point_loss = Counter(no_input_nodes=0, wrong_merge_no_representative=0,
                             selected_prediction_unavailable_at_point=0)
        for row, identity in enumerate(targets):
            own = np.flatnonzero(nodes == identity).tolist()
            anchors = [g for g, members in enumerate(groups) if nodes[members[0]] == identity]
            foreign = [g for g, members in enumerate(groups)
                       if any(nodes[i] == identity for i in members) and nodes[members[0]] != identity]
            mixed = [g for g, members in enumerate(groups)
                     if any(nodes[i] == identity for i in members) and len(set(nodes[members])) > 1]
            selected = anchors[0] if anchors else None
            covered = bool((truth[row] & pred[row]).any())
            if not truth[row].any():
                reason = 'reference_unavailable'
            elif covered:
                reason = 'covered'
            elif not own:
                reason = 'no_input_nodes'
            elif selected is None:
                reason = 'wrong_merge_no_representative'
            elif not pred[row].any():
                reason = 'selected_prediction_unavailable'
            else:
                reason = 'no_valid_time_intersection'
            flags = dict(has_input=bool(own), has_mixed_group=bool(mixed),
                has_foreign_anchor_group=bool(foreign), has_duplicate_anchor_groups=len(anchors) > 1,
                later_duplicate_has_reference_overlap=any(
                    bool((group_valid[g] & truth[row]).any()) for g in anchors[1:]),
                duplicate_could_rescue_uncovered=not covered and any(
                    bool((group_valid[g] & truth[row]).any()) for g in anchors[1:]),
                no_valid_native_prediction=bool(own) and not native[own].any())
            primary[reason] += 1
            overlap.update({k: int(v) for k, v in flags.items()})
            missed = int((truth[row] & ~pred[row]).sum())
            point_loss['no_input_nodes' if not own else 'wrong_merge_no_representative'
                       if selected is None else 'selected_prediction_unavailable_at_point'] += missed
            records.append(dict(target_id=str(identity), primary_reason=reason,
                input_node_indices=own, selected_group=selected, anchor_group_indices=anchors,
                foreign_anchor_group_indices=foreign, mixed_group_indices=mixed, flags=flags,
                reference_point_count=int(truth[row].sum()), intersection_count=int((truth[row] & pred[row]).sum()),
                missing_point_count=missed))
        available = t - primary['reference_unavailable']
        counts[attr] = dict(primary_counts=dict(primary), overlap_counts=dict(overlap),
            missing_point_causes=dict(point_loss), registry_target_count=t,
            available_target_trajectory_count=available, covered_target_trajectory_count=primary['covered'],
            missing_target_trajectory_count=available-primary['covered'], target_rows=records)
        rows_by_attr[attr] = np.asarray([REASONS.index(r['primary_reason']) for r in records], np.uint8)
    return counts, rows_by_attr


def analyze_scene(fragment, saved, expected=None, source_names=None):
    """Replay one saved053A scene, evaluate two controls, return JSON + NPZ data."""
    nodes = np.asarray(fragment['identities']).astype(str)
    targets = np.asarray(fragment['target_identity_ids']).astype(str)
    n, t = len(nodes), len(targets)
    query = np.asarray(fragment['query_seconds'])
    if query.ndim != 1 or not np.isfinite(query).all() or (np.diff(query) <= 0).any():
        raise ValueError('invalid cached query grid')
    q = len(query)
    _same(targets, saved['target_ids'], 'target registry')
    for key, value in (('split', 'V_select'), ('grouping_kind', 'predicted'),
                       ('correspondence_policy', 'anchor_identity_first')):
        _same(saved[key], value, key)
    if len(fragment['node_ids']) != n or len(set(fragment['node_ids'])) != n:
        raise ValueError('node population or order invalid')
    offsets, flat = np.asarray(fragment['ols_group_offsets']), np.asarray(fragment['ols_group_node_indices'])
    if (offsets.dtype.kind not in 'iu' or flat.dtype.kind not in 'iu' or offsets.ndim != 1
            or flat.ndim != 1 or not len(offsets) or offsets[0] != 0 or offsets[-1] != n
            or (np.diff(offsets) <= 0).any() or sorted(flat.tolist()) != list(range(n))):
        raise ValueError('cached groups must partition every input node')
    groups = [flat[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    truth, truth_valid = saved['aligned_truth'][:t], saved['aligned_truth_valid'][:t]
    if truth.shape != (t, q, 4) or truth_valid.shape != truth.shape:
        raise ValueError('complete target rows and query grid required')
    registry = {identity: i for i, identity in enumerate(targets)}
    if len(registry) != t or not set(nodes).issubset(registry):
        raise ValueError('invalid original target registry')
    for i, identity in enumerate(nodes):
        _same(fragment['target'][i], truth[registry[identity]], 'fragment truth')
        _same(fragment['target_valid'][i], truth_valid[registry[identity]], 'fragment reference mask')
    native, baseline = np.asarray(fragment['native_states']), np.asarray(fragment[BASE])
    if native.shape != (n, q, 4) or baseline.shape != native.shape:
        raise ValueError('full per-fragment physical states required')
    support, nearest = np.asarray(fragment['support']), np.asarray(fragment['nearest_seconds'])
    exact = np.asarray(fragment['exact_observation'])
    if (support.dtype != bool or exact.dtype != bool or support.shape != (n, q)
            or exact.shape != support.shape or nearest.shape != support.shape
            or np.isnan(nearest).any() or (nearest < 0).any()):
        raise ValueError('cached support/time evidence invalid')
    target_support, target_exact = np.zeros((t, q), bool), np.zeros((t, q), bool)
    target_nearest = np.full((t, q), np.inf)
    for node, identity in enumerate(nodes):
        row = registry[identity]
        target_support[row] |= support[node]
        target_exact[row] |= exact[node]
        target_nearest[row] = np.minimum(target_nearest[row], nearest[node])
    fixed_regions = _regions(target_support, target_nearest, target_exact)
    fragment_regions = _regions(support, nearest, exact)
    node_sources = np.asarray(fragment['sources']).astype(str)
    for source in sorted(set(node_sources)) if source_names is None else source_names:
        fragment_regions['source_' + str(source)] = np.broadcast_to(
            (node_sources == str(source))[:, None], (n, q))
    methods, artifacts, arrays = {}, {}, dict(query_seconds=query, node_ids=fragment['node_ids'],
        identities=nodes, sources=fragment['sources'], target_ids=targets,
        target_input_support=target_support, target_nearest_input_seconds=target_nearest,
        target_exact_observation=target_exact, node_input_support=support,
        node_nearest_input_seconds=nearest, node_exact_observation=exact,
        node_original_mean_supported=baseline,
        reason_names=np.asarray(REASONS))
    for name in (BASE, INDEPENDENT, ANCHOR):
        partition = [[i] for i in range(n)] if name == INDEPENDENT else groups
        values = baseline if name == BASE else native
        outputs = [dict(node_indices=g, states=values[g[0]], prediction_valid=(
            saved['group_valid'][i] if name == BASE else np.isfinite(values[g[0]])))
            for i, g in enumerate(partition)]
        report = evaluate_full_scene(outputs, node_identities=nodes, target_identities=targets,
            target_states=truth, target_valid=truth_valid, node_observed=support,
            split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first',
            return_artifact=True, node_input_support=dict(support=support, nearest_seconds=nearest))
        if name == ANCHOR:
            report['input_support_definition'] = (
                'candidate-group context only; helper measurements are NOT used by this anchor-only control; '
                'use fixed_target_strata for same-population cross-method comparisons')
            report['input_distance_definition'] = (
                'candidate-group nearest time, NOT evidence used by anchor-only output')
        artifact = report.pop('artifact')
        # Group states/offsets are in NPZ; avoid triplicating large per-group
        # metric trees. All full-population and group diagnostics remain.
        report['all_predicted_groups'] = [
            {k: v for k, v in g.items() if k != 'metrics'} for g in report['all_predicted_groups']]
        artifacts[name] = artifact
        selected_anchor = np.full(t, -1, np.int64)
        for row, group in enumerate(artifact['aligned_group_indices'][:t]):
            if group >= 0:
                selected_anchor[row] = partition[group][0]
        arrays[name+'_selected_anchor_node_index'] = selected_anchor
        emitted = np.full(native.shape, np.nan)
        emitted_valid = np.zeros(native.shape, bool)
        for i, group in enumerate(partition):
            emitted[group] = artifact['group_states'][i]
            emitted_valid[group] = artifact['group_valid'][i]
        # Preserve053A's own per-node fallback view for audit. Main fragment
        # comparison broadcasts each actually emitted group output for ALL arms.
        fragment_values = baseline if name == BASE else emitted
        methods[name] = dict(system=report,
            fragment=_metrics(fragment_values, fragment['target'], fragment['target_valid'],
                              np.isfinite(fragment_values), fragment_regions),
            emitted_fragment=_metrics(emitted, fragment['target'], fragment['target_valid'],
                                      emitted_valid, fragment_regions),
            fixed_target_strata=_metrics(artifact['aligned_prediction'][:t], truth, truth_valid,
                                         artifact['aligned_prediction_valid'][:t], fixed_regions))
        methods[name]['duplicate_outputs'] = {
            attr: dict(trajectory_count=int(artifact['aligned_prediction_valid'][t:, :, list(cols)].all(-1).any(-1).sum()),
                       point_count=int(artifact['aligned_prediction_valid'][t:, :, list(cols)].all(-1).sum()))
            for attr, cols in ATTRIBUTES.items()}
        arrays.update({name+'_'+key: value for key, value in artifact.items()})
    replay = artifacts[BASE]
    for key in replay:
        _same(replay[key], saved[key], key)
    _same(fragment['ols_membership'], replay['node_group_indices'], 'membership')
    if expected is not None:
        original = expected['unique_target'][BASE]
        for key in ('full_target_metrics', 'input_support_metrics', 'input_distance_metrics'):
            assert_metrics(methods[BASE]['system'][key], original[key], key)
        for key in ('target_identity_count', 'predicted_group_count', 'duplicate_prediction_group_count',
                    'anchor_matched_target_count', 'unmatched_target_ids', 'selected_group_by_target'):
            _same(methods[BASE]['system'][key], original[key], key)
        if 'fragment' in expected:
            assert_metrics(methods[BASE]['fragment'], expected['fragment'][BASE], 'fragment')
    attribution, codes = _attribution(nodes, targets, groups, replay, np.isfinite(native))
    for attr, row in attribution.items():
        full = methods[BASE]['system']['full_target_metrics']['all'][attr]
        for key in ('available_target_trajectory_count', 'covered_target_trajectory_count', 'missing_target_trajectory_count'):
            if row[key] != full[key]:
                raise ValueError(f'attribution does not reconcile {attr}/{key}')
        if sum(row['missing_point_causes'].values()) != full['missing_prediction_count']:
            raise ValueError('missing point accounting does not reconcile')
        arrays['attribution_' + attr] = codes[attr]
    paired = {}
    for a, b in combinations(sorted(artifacts), 2):
        pair = a+'__vs__'+b
        common = truth_valid & artifacts[a]['aligned_prediction_valid'][:t] & artifacts[b]['aligned_prediction_valid'][:t]
        selected_a, selected_b = arrays[a+'_selected_anchor_node_index'], arrays[b+'_selected_anchor_node_index']
        both_represented = (selected_a >= 0) & (selected_b >= 0)
        same = both_represented & (selected_a == selected_b)
        changed = both_represented & (selected_a != selected_b)
        pair_regions = {**fixed_regions,
            'same_representative': np.broadcast_to(same[:, None], (t, q)),
            'changed_representative': np.broadcast_to(changed[:, None], (t, q))}
        paired[pair] = dict(methods={name: _metrics(artifacts[name]['aligned_prediction'][:t],
            truth, truth_valid, common, pair_regions) for name in (a, b)},
            same_representative_target_count=int(same.sum()), changed_representative_target_count=int(changed.sum()))
        errors = {name: attribute_errors(artifacts[name]['aligned_prediction'][:t].astype(np.float64),
                                         truth.astype(np.float64)) for name in (a, b)}
        changes = paired[pair]['error_change_second_minus_first'] = {}
        for region, mask in pair_regions.items():
            changes[region] = {}
            for attr, cols in ATTRIBUTES.items():
                selected = common[..., list(cols)].all(-1) & mask
                delta = (errors[b][attr] - errors[a][attr])[selected]
                # Pure numerical tolerance in the metric's native unit, fixed
                # before seeing data; never selects a model or threshold.
                worse, better = delta > 1e-9, delta < -1e-9
                changes[region][attr] = dict(common_point_count=len(delta),
                    second_worse_point_count=int(worse.sum()), second_better_point_count=int(better.sum()),
                    tied_point_count=int((~worse & ~better).sum()),
                    harm_sum=float(delta[worse].sum()), benefit_sum=float(-delta[better].sum()),
                    signed_error_change_sum=float(delta.sum()))
        for attr, cols in ATTRIBUTES.items():
            arrays['delta_'+pair+'_'+attr] = np.where(common[..., list(cols)].all(-1),
                errors[b][attr]-errors[a][attr], np.nan)
        arrays['common_' + pair + '_valid'] = common
    # Paired coverage transitions retain rescued/lost identities and points;
    # errors alone are conditional and cannot decide the complete-system winner.
    transitions = {}
    for name in (INDEPENDENT, ANCHOR):
        transitions[name] = {}
        for attr, cols in ATTRIBUTES.items():
            valid = truth_valid[..., list(cols)].all(axis=-1)
            old = valid & replay['aligned_prediction_valid'][:t, :, list(cols)].all(axis=-1)
            new = valid & artifacts[name]['aligned_prediction_valid'][:t, :, list(cols)].all(axis=-1)
            transitions[name][attr] = dict(rescued_target_count=int((~old.any(1) & new.any(1)).sum()),
                lost_target_count=int((old.any(1) & ~new.any(1)).sum()),
                gained_target_points=int((new & ~old).sum()), lost_target_points=int((old & ~new).sum()))
    return dict(methods=methods, attribution=attribution, pairwise=paired,
                coverage_transitions=transitions, replay_verified=True), arrays
