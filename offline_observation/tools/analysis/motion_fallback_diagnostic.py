"""C77: describe the support and error of frozen C74 motion fallbacks on V32."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import METHODS, SPLIT, load_population, load_scene, read_reference
from tools.analysis.same_source_swap_pilot import MAX_AS, PARENTS, group_members, peak_rss_bytes, read, write


CAUSES = ('active', 'before_span_only', 'after_span_only', 'insufficient_inside_span',
          'before_span_and_insufficient', 'after_span_and_insufficient')
MOTION = ('speed', 'heading')
MASKS = ('active', 'internal_gap', 'near_endpoint', 'B_low_speed', 'C_low_speed')
INVARIANTS = ('aligned_prediction_valid', 'aligned_truth_valid', 'aligned_group_indices',
              'aligned_target_ids', 'group_offsets', 'group_node_indices',
              'group_anchor_target_indices', 'aligned_observed')
DISTANCE_BINS = ('at_most_1', 'over_1_through_2', 'over_2_through_4', 'over_4')
ORIGINAL_COUNTS = (3851, 1605, 1095780)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def support_tags(query, times, offsets, groups, support, sources):
    """Assign mutually exclusive branches and exact event distances without reference."""
    shape = (len(groups), len(query))
    tags = {name: np.zeros(shape, bool) for name in ('visible', 'internal_gap', 'near_endpoint')}
    tags.update(cause=np.zeros(shape, np.uint8), eligible=np.empty(len(groups), np.int64),
                source_count=np.empty(len(groups), np.int64), distance=np.empty(shape),
                endpoint_distance=np.empty(shape), anchor_side=np.zeros(shape, np.uint8),
                anchor_endpoint_distance=np.empty(shape))
    for index, nodes in enumerate(groups):
        events = np.sort(np.concatenate([times[offsets[n]:offsets[n+1]] for n in nodes]))
        before, after = query < events[0], query > events[-1]
        inside = ~(before | after)
        visible = support[nodes].any(axis=0)
        eligible = int(np.count_nonzero(inside & visible))
        enough = eligible >= 2
        tags['cause'][index, before] = 1 if enough else 4
        tags['cause'][index, after] = 2 if enough else 5
        if not enough:
            tags['cause'][index, inside] = 3
        tags['visible'][index] = visible
        tags['eligible'][index] = eligible
        tags['source_count'][index] = len(np.unique(sources[nodes]))
        tags['internal_gap'][index] = enough & inside & ~visible
        tags['near_endpoint'][index] = enough & inside & (
            np.minimum(query-events[0], events[-1]-query) <= query[1]-query[0])
        insertion = np.searchsorted(events, query)
        right = insertion.clip(0, len(events)-1)
        left = (insertion-1).clip(0, len(events)-1)
        tags['distance'][index] = np.minimum(np.abs(query-events[left]), np.abs(query-events[right]))
        tags['endpoint_distance'][index] = np.minimum(np.abs(query-events[0]), np.abs(query-events[-1]))
        first = nodes[0]
        start, end = times[offsets[first]], times[offsets[first+1]-1]
        tags['anchor_side'][index, query < start] = 1
        tags['anchor_side'][index, query > end] = 2
        tags['anchor_endpoint_distance'][index] = np.minimum(np.abs(query-start), np.abs(query-end))
    return tags


def frozen_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'membership', *METHODS, *MASKS}, 'unexpected frozen C74 scene schema')
        return {key: saved[key] for key in saved.files}


def read_legacy_motion(runs, sid, scene):
    nodes, sources, support, nearest, _, native, _ = scene
    path = runs/PARENTS['native']/'private'/'native-motion'/SPLIT/f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as saved:
        for key, original in (('node_ids', nodes), ('sources', sources), ('support', support),
                              ('nearest_seconds', nearest), ('query_seconds', native[0])):
            np.testing.assert_array_equal(saved[key], original)
        return saved['states'][..., 2:].astype(np.float64)


def freeze_scene(scene, saved, legacy_motion):
    nodes, sources, support, nearest, membership, native, baseline = scene
    query, times, _, offsets = native
    groups = group_members(membership)
    np.testing.assert_array_equal(saved['membership'], membership)
    np.testing.assert_array_equal(saved['A'], baseline)
    tags = support_tags(query, times, offsets, groups, support, sources)
    first_members = [members[0] for members in groups]
    np.testing.assert_array_equal(tags['anchor_side'] == 0, support[first_members])
    require(legacy_motion.shape == (len(nodes), len(query), 2) and np.isfinite(legacy_motion).all(),
            'finite original per-fragment speed and heading cache required')
    anchor_motion = legacy_motion[first_members].copy()
    anchor_motion[..., 0] *= KNOT_TO_MPS
    no_support = ~tags['visible']
    np.testing.assert_array_equal(baseline[..., 2:][no_support], anchor_motion[no_support],
                                  err_msg='old unsupported WS Mean is not exact first-member motion')
    require((tags['anchor_endpoint_distance'][no_support] >= tags['distance'][no_support]).all(),
            'closest group event must be at least as near as the held anchor endpoint')
    active = tags['cause'] == 0
    for name, reconstructed in (('active', active), ('internal_gap', tags['internal_gap']),
                                ('near_endpoint', tags['near_endpoint'])):
        require(saved[name].dtype == bool, f'{name} must be boolean')
        np.testing.assert_array_equal(saved[name], reconstructed)
    for name in ('B', 'C'):
        states = saved[name]
        require(states.shape == baseline.shape and np.isfinite(states).all(), f'invalid C74 {name} values')
        np.testing.assert_array_equal(states[..., :2], baseline[..., :2])
        np.testing.assert_array_equal(states[~active], baseline[~active])
        mask = saved[name+'_low_speed']
        require(mask.dtype == bool, f'invalid C74 {name} low-speed mask dtype')
        np.testing.assert_array_equal(mask, active & (states[..., 2] < .5))
        np.testing.assert_array_equal(states[..., 3][mask], baseline[..., 3][mask])
        tags[name+'_low_speed'] = mask
    return dict(tags, nodes=nodes, support=support, nearest=nearest, membership=membership,
                query=query, event_count=len(times))


def check_error(values, expected):
    actual = _stats(values)
    require(actual['count'] == expected['count'], 'C74 error count changed')
    for key, value in actual.items():
        if value is None:
            require(expected[key] is None, 'C74 empty-region metric changed')
        elif key != 'count':
            np.testing.assert_allclose(value, expected[key], rtol=1e-12, atol=1e-12,
                                       err_msg=f'C74 {key} changed')


def measure_scene(sid, component, frozen, saved, ev, prior):
    membership, query = frozen['membership'], frozen['query']
    np.testing.assert_array_equal(saved['membership'], membership)
    groups = [nodes.tolist() for nodes in group_members(membership)]
    outputs = {name: emit_group_outputs(saved[name], groups, node_count=len(membership)) for name in METHODS}
    reports, artifacts = evaluate_fusion_methods(
        outputs, prediction_speed_units={name: 'mps' for name in METHODS},
        node_identities=ev['identities'], target_ids=ev['target_ids'],
        target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=frozen['support'], nearest_seconds=frozen['nearest'], target_speed_unit='mps', split=SPLIT)
    base = artifacts['B']
    n = len(ev['target_ids'])
    chosen = base['aligned_group_indices'][:n]
    matched = chosen >= 0
    active = np.zeros((n, len(query)), bool)
    active[matched] = frozen['cause'][chosen[matched]] == 0
    group_indices = np.broadcast_to(chosen[:, None], active.shape)
    qindices = np.broadcast_to(np.arange(len(query)), active.shape)
    pure = np.asarray([len(np.unique(ev['identities'][members])) == 1 for members in groups])
    require(prior['scene_id'] == str(sid) and prior['component_id'] == str(component)
            and prior['node_count'] == len(membership) and prior['group_count'] == len(groups),
            'C74 scene populations changed')
    for key in ('active', 'internal_gap', 'near_endpoint'):
        mask = (frozen['cause'] == 0) if key == 'active' else frozen[key]
        require(int(mask.sum()) == prior['group_query_counts'][key], f'C74 {key} total changed')
    b, c = frozen['B_low_speed'], frozen['C_low_speed']
    low_counts = dict(B=int(b.sum()), C=int(c.sum()), both=int((b & c).sum()),
                      either=int((b | c).sum()), B_only=int((b & ~c).sum()),
                      C_only=int((c & ~b).sum()),
                      neither_active=int(np.count_nonzero((frozen['cause'] == 0) & ~(b | c))))
    require(low_counts == prior['fallback_group_query_counts'], 'C74 low-speed masks changed')
    vectors = {}
    for name in METHODS:
        report, artifact = reports[name], artifacts[name]
        for key in INVARIANTS:
            np.testing.assert_array_equal(artifact[key], base[key])
        np.testing.assert_array_equal(artifact['aligned_prediction'][..., :2],
                                      base['aligned_prediction'][..., :2])
        original = prior['methods'][name]
        require(all(report[key] == original['population'][key] for key in POPULATION_KEYS)
                and len(report['unmatched_target_ids']) == original['unmatched_target_count'],
                f'C74 {name} coverage or duplicate counts changed')
        for region in ('all', 'observed', 'missing'):
            for feature in ('position', *MOTION):
                current = report['full_target_metrics'][region][feature]
                old = original['full_target_metrics'][region][feature]
                require(all(current[key] == old[key] for key in COVERAGE_KEYS),
                        f'C74 {name} {region} {feature} availability changed')
        for column, feature in enumerate(MOTION, start=2):
            prediction = artifact['aligned_prediction'][:n, :, column]
            truth = artifact['aligned_truth'][:n, :, column]
            valid = artifact['aligned_prediction_valid'][:n, :, column] & artifact['aligned_truth_valid'][:n, :, column]
            error = prediction-truth
            if feature == 'heading':
                error = _wrap(error)
            check_error(error[valid], original['errors']['all'][feature])
            check_error(error[valid & active], original['errors']['active'][feature])
            if name != 'B':
                continue
            g, q = group_indices[valid], qindices[valid]
            require((g >= 0).all(), 'valid motion point has no primary group')
            nearest = frozen['distance'][g, q]
            qstep = (query[-1]-query[0])/(len(query)-1)
            vectors[feature] = dict(error=error[valid], prediction=prediction[valid], truth=truth[valid],
                pure=pure[g], source_count=frozen['source_count'][g], eligible=frozen['eligible'][g],
                steps=nearest/qstep, endpoint_steps=frozen['endpoint_distance'][g, q]/qstep,
                **{key: frozen[key][g, q] for key in ('cause', 'visible', 'endpoint_distance',
                                                     'anchor_side', 'anchor_endpoint_distance',
                                                     'B_low_speed', 'C_low_speed')},
                distance=nearest)
    position_points = reports['B']['full_target_metrics']['all']['position']['available_target_count']
    return dict(scene_id=str(sid), component_id=str(component), node_count=len(membership),
                group_count=len(groups), event_count=int(frozen['event_count']),
                reference_position_points=position_points,
                group_query_counts={name: int(np.count_nonzero(frozen['cause'] == code))
                                    for code, name in enumerate(CAUSES)},
                group_eligibility_counts={str(k): int(np.count_nonzero(np.minimum(frozen['eligible'], 2) == k))
                                          for k in (0, 1, 2)},
                legacy_anchor_hold_group_queries=int(np.count_nonzero(~frozen['visible'])),
                legacy_anchor_strictly_older_group_queries=int(np.count_nonzero(
                    ~frozen['visible'] & (frozen['anchor_endpoint_distance'] > frozen['distance']))),
                population=prior['methods']['B']['population'],
                unmatched_target_count=prior['methods']['B']['unmatched_target_count']), vectors


def distribution(values):
    if not len(values):
        return dict(min=None, median=None, p95=None, p99=None, max=None)
    quantiles = np.quantile(values, (0., .5, .95, .99, 1.))
    return dict(zip(('min', 'median', 'p95', 'p99', 'max'), map(float, quantiles), strict=True))


def distance_bin_masks(steps):
    return (steps <= 1, (steps > 1) & (steps <= 2), (steps > 2) & (steps <= 4), steps > 4)


def strata(vector):
    cause, steps = vector['cause'], vector['steps']
    inactive = cause != 0
    yield 'all', np.ones(len(cause), bool)
    for code, name in enumerate(CAUSES):
        yield name, cause == code
    yield 'inactive', inactive
    yield 'inactive_before', np.isin(cause, (1, 4))
    yield 'inactive_after', np.isin(cause, (2, 5))
    span_only = np.isin(cause, (1, 2))
    yield 'inactive_span_only', span_only
    yield 'inactive_span_only_near_pure', span_only & (steps <= 2) & vector['pure']
    yield 'inactive_insufficient_group', cause >= 3
    yield 'inactive_legacy_anchor_hold_before', inactive & ~vector['visible'] & (vector['anchor_side'] == 1)
    yield 'inactive_legacy_anchor_hold_after', inactive & ~vector['visible'] & (vector['anchor_side'] == 2)
    yield 'active_B_low_speed', vector['B_low_speed']
    yield 'active_C_low_speed', vector['C_low_speed']
    yield 'active_neither_low_speed', ~inactive & ~(vector['B_low_speed'] | vector['C_low_speed'])
    for label, region in (('inactive', inactive), ('active', ~inactive)):
        yield label+'_pure', region & vector['pure']
        yield label+'_mixed', region & ~vector['pure']
        yield label+'_one_source', region & (vector['source_count'] == 1)
        yield label+'_multiple_sources', region & (vector['source_count'] > 1)
        yield label+'_member_Q_support', region & vector['visible']
        yield label+'_no_member_Q_support', region & ~vector['visible']
    for k in (0, 1, 2):
        yield f'inactive_eligible_Q_{k}', inactive & (np.minimum(vector['eligible'], 2) == k)
    for label, selected in zip(DISTANCE_BINS, distance_bin_masks(steps), strict=True):
        yield 'inactive_nearest_steps_'+label, inactive & selected
        yield 'inactive_before_nearest_steps_'+label, np.isin(cause, (1, 4)) & selected
        yield 'inactive_after_nearest_steps_'+label, np.isin(cause, (2, 5)) & selected
    for label, selected in zip(DISTANCE_BINS, distance_bin_masks(vector['endpoint_steps']), strict=True):
        yield 'inactive_endpoint_steps_'+label, inactive & selected


def describe(vector, *, speed):
    absolute = np.abs(vector['error'])
    total = float(absolute.sum())
    inactive_total = float(absolute[vector['cause'] != 0].sum())
    regions = {}
    for name, mask in strata(vector):
        errors = vector['error'][mask]
        sum_absolute = float(absolute[mask].sum())
        anchor_hold = mask & ~vector['visible']
        anchor_age = vector['anchor_endpoint_distance'][anchor_hold]
        excess_age = anchor_age-vector['distance'][anchor_hold]
        item = dict(_stats(errors), absolute_error_sum=sum_absolute,
                    error_share_of_all=sum_absolute/total if total else None,
                    error_share_of_inactive=(sum_absolute/inactive_total
                                             if name.startswith('inactive') and inactive_total else None),
                    nearest_native_seconds=distribution(vector['distance'][mask]),
                    nearest_span_endpoint_seconds=distribution(vector['endpoint_distance'][mask]),
                    legacy_anchor_hold_count=int(np.count_nonzero(anchor_hold)),
                    legacy_anchor_strictly_older_count=int(np.count_nonzero(excess_age > 0)),
                    legacy_anchor_endpoint_seconds=distribution(anchor_age),
                    legacy_anchor_excess_age_seconds=distribution(excess_age))
        if speed:
            item['old_or_frozen_prediction_mps'] = distribution(vector['prediction'][mask])
            item['reference_mps'] = distribution(vector['truth'][mask])
        regions[name] = item
    require(sum(regions[name]['count'] for name in CAUSES) == regions['all']['count']
            and sum(regions[name]['count'] for name in CAUSES[1:]) == regions['inactive']['count'],
            'cause partition lost or duplicated evaluation points')
    np.testing.assert_allclose(sum(regions[name]['absolute_error_sum'] for name in CAUSES[1:]),
                               inactive_total, rtol=1e-12, atol=1e-12)
    return regions


def pool(rows, data):
    require(len(rows) == len(data) and bool(rows), 'nonempty matched scene rows and point vectors')
    def counts(key):
        return {field: sum(row[key][field] for row in rows) for field in rows[0][key]}
    summary = dict(scene_ids=[row['scene_id'] for row in rows], scene_count=len(rows),
                   component_count=len({row['component_id'] for row in rows}))
    for key in ('node_count', 'group_count', 'event_count', 'unmatched_target_count', 'reference_position_points',
                'legacy_anchor_hold_group_queries', 'legacy_anchor_strictly_older_group_queries'):
        summary[key] = sum(row[key] for row in rows)
    for key in ('group_query_counts', 'group_eligibility_counts', 'population'):
        summary[key] = counts(key)
    summary['errors'] = {}
    for feature in MOTION:
        vector = {key: np.concatenate([entry[feature][key] for entry in data]) for key in data[0][feature]}
        summary['errors'][feature] = describe(vector, speed=feature == 'speed')
    return summary


def run(runs, science, output):
    require(output != runs and output != science and science not in output.parents,
            'new output must remain outside frozen C74 artifacts')
    output.mkdir(exist_ok=False)
    (output/'private').mkdir()
    began = time.monotonic()
    record = dict(status='RUNNING', split=SPLIT, stages=[], optimizer_updates=0, predictions_generated=0,
                  affinity=sorted(os.sched_getaffinity(0)), runs=str(runs), science=str(science),
                  output=str(output), argv=sys.argv, started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', record)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        old_run = read(science/'run.json')
        require(old_run['status'] == 'COMPLETED' and old_run['exit_code'] == 0
                and old_run['split'] == SPLIT and old_run['scene_ids'] == ids
                and old_run['completed_scenes'] == len(ids)
                and tuple(old_run[key] for key in ('node_count', 'group_count', 'event_count')) == ORIGINAL_COUNTS,
                'completed original C74 population required')
        checkpoint(phase='freeze_support', scene_ids=ids)
        totals = np.zeros(3, np.int64)
        for sid in ids:
            scene = load_scene(runs, sid)
            frozen = freeze_scene(scene, frozen_arrays(science/'private'/f'{sid}.npz'),
                                  read_legacy_motion(runs, sid, scene))
            np.savez_compressed(output/'private'/f'support-{sid}.npz', **frozen)
            totals += (len(frozen['nodes']), len(frozen['eligible']), frozen['event_count'])
        require(tuple(totals) == ORIGINAL_COUNTS, 'V32 node, group or event count changed')
        record['stages'].append('all_support_tags_saved_before_identity_reference_and_prior_error_tables')
        checkpoint(phase='evaluate_frozen', node_count=int(totals[0]), group_count=int(totals[1]),
                   event_count=int(totals[2]), frozen_support_scenes=len(ids))
        old_rows = read(science/'scenes.json.gz')
        require([row['scene_id'] for row in old_rows] == ids, 'C74 published scene order changed')
        old_summary = read(science/'summary.json')
        rows, data = [], []
        for sid, old in zip(ids, old_rows, strict=True):
            with np.load(output/'private'/f'support-{sid}.npz', allow_pickle=False) as f:
                frozen = {key: f[key] for key in f.files}
            ev = read_reference(runs, sid, frozen['nodes'], frozen['query'])
            row, vectors = measure_scene(sid, components[sid], frozen,
                                         frozen_arrays(science/'private'/f'{sid}.npz'), ev, old)
            rows.append(row)
            data.append(vectors)
        unique_components = sorted(set(components.values()))
        by_component = {}
        for component in unique_components:
            indices = [i for i, row in enumerate(rows) if row['component_id'] == component]
            by_component[component] = pool([rows[i] for i in indices], [data[i] for i in indices])
        largest = min(unique_components, key=lambda key: (-by_component[key]['reference_position_points'], key))
        require(largest == old_summary['remove_largest_component']['excluded_component'],
                'C74 largest reference-position component changed')
        keep = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        write(output/'scenes.json.gz', [dict(row, errors=pool([row], [vectors])['errors'])
                                       for row, vectors in zip(rows, data, strict=True)])
        all_results = pool(rows, data)
        require(all_results['errors']['speed']['inactive']['count'] == 1680,
                'original C74 inactive speed population of 1,680 changed')
        for feature in MOTION:
            old = old_summary['all_components']['methods']['B']['errors']
            for region in ('all', 'active'):
                new = all_results['errors'][feature][region]
                require(new['count'] == old[region][feature]['count'], 'aggregate C74 B population changed')
                for key in ('mae', 'bias', 'rmse', 'p95', 'p99'):
                    np.testing.assert_allclose(new[key], old[region][feature][key], rtol=1e-12, atol=1e-12)
        write(output/'summary.json', dict(all_components=all_results, by_component=by_component,
            remove_largest_component=dict(excluded_component=largest,
                criterion='available reference position points; ties lexical component id',
                result=pool([rows[i] for i in keep], [data[i] for i in keep])),
            definitions=dict(
                evaluated_output='frozen C74 B; inactive speed/heading identical to A and C',
                cause_order=list(CAUSES), insufficient='fewer than 2 in-span Q64 with any member Q support',
                endpoint_side='strictly before earliest or after latest original group event; equality inside',
                nearest_native='minimum absolute seconds to any unchanged group native event',
                legacy_anchor_hold='with no member Q support, old A speed/heading equal the first member '
                    'cached native motion (kn converted once), whose source code clamps to a span endpoint',
                legacy_anchor_excess_age='on no-member-support rows, first member endpoint distance '
                    'minus nearest original group event distance, in seconds; active B need not use old A',
                normalized_step='scene span (Q64 last minus first) divided by 63',
                span_only_near_pure='before or after group span, at least 2 eligible group Q64 points, '
                    'nearest native event at most 2 steps away, all members match the anchor identity',
                source_count='distinct original source labels among unchanged group members',
                purity='evaluation-only identity agreement among members of primary anchor-selected group',
                error_share='sum absolute error divided by all or inactive valid B errors in same scope',
                caution='partitions are descriptive and do not identify the cause of motion error')))
        record['stages'].append('c74_frozen_invariants_and_all_active_motion_errors_reproduced')
        checkpoint(status='COMPLETED', exit_code=0, completed_scenes=len(rows),
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


def self_check():
    query = np.arange(64, dtype=float)
    times = np.array([1.1, 3., 4.6, 5.2, 7.7, 8.1, 20.1, 20.4])
    offsets = np.arange(0, 9, 2)
    membership = np.array([0, 0, 1, 2])
    sources = np.array([1, 2, 1, 2])
    support = np.zeros((4, 64), bool)
    support[0, [2, 3]] = True
    support[1, 5] = True
    support[2, 8] = True
    tags = support_tags(query, times, offsets, group_members(membership), support, sources)
    np.testing.assert_array_equal(tags['cause'][0, :8], [1, 1, 0, 0, 0, 0, 2, 2])
    np.testing.assert_array_equal(tags['cause'][1, 7:12], [4, 3, 5, 5, 5])
    np.testing.assert_array_equal(tags['cause'][2, 20:22], [4, 5])
    np.testing.assert_array_equal(tags['eligible'], [3, 1, 0])
    np.testing.assert_array_equal(tags['source_count'], [2, 1, 1])
    np.testing.assert_allclose(tags['distance'][0, :8], [1.1, .1, .9, 0, .6, .2, .8, 1.8])
    np.testing.assert_allclose(tags['endpoint_distance'][1, 7:12], [.7, .1, .9, 1.9, 2.9])
    np.testing.assert_allclose(tags['anchor_endpoint_distance'][0, 4:7], [1., 2., 3.])
    require(tags['internal_gap'][0, 4] and not tags['internal_gap'][1].any(),
            'internal gap belongs only to C74 active support')
    baseline = np.zeros((3, 64, 4))
    baseline[..., 2:] = (16.*KNOT_TO_MPS, 358.)
    legacy_motion = np.broadcast_to(np.array([16., 358.]), (4, 64, 2))
    baseline[..., 0] = 10.
    saved = dict(membership=membership, A=baseline.copy(), B=baseline.copy(), C=baseline.copy(),
                 **{key: tags[key] for key in ('internal_gap', 'near_endpoint')}, active=tags['cause'] == 0)
    for arm in ('B', 'C'):
        saved[arm][saved['active'], 2] = 1.
        saved[arm+'_low_speed'] = np.zeros((3, 64), bool)
    saved['B'][0, 2, 2] = 0.
    saved['B_low_speed'][0, 2] = True
    scene = (np.arange(4), sources, support, np.ones_like(support, dtype=float), membership,
             (query, times, np.ones((len(times), 2)), offsets), baseline)
    frozen = freeze_scene(scene, saved, legacy_motion)
    targets = np.array(['shipA', 'shipB', 'shipC'])
    truth = np.zeros((len(targets), 64, 4))
    truth[..., 0] = 10.
    truth[..., 2:] = (1., 1.)
    ev = dict(identities=np.array(['shipA', 'shipB', 'shipA', 'shipC']), target_ids=targets,
              target_states=truth, target_valid=np.ones_like(truth, dtype=bool))
    from tools.analysis.motion_readout_evaluation import evaluate_scene
    old, _ = evaluate_scene('100', 'test-component', scene[0],
                            [members.tolist() for members in group_members(membership)],
                            support, scene[3], ev, {key: saved[key] for key in METHODS},
                            {key: saved[key] for key in MASKS}, query)
    row, vectors = measure_scene('100', 'test-component', frozen, saved, ev, old)
    result = pool([row], [vectors])
    errors = result['errors']
    require(row['population']['duplicate_prediction_group_count'] == 1
            and row['unmatched_target_count'] == 1
            and errors['speed']['all']['count'] == 128
            and errors['speed']['inactive']['count'] == 124
            and errors['speed']['inactive_span_only']['count'] == 60
            and errors['speed']['inactive_span_only_near_pure']['count'] == 0
            and errors['speed']['inactive_mixed']['count'] == 60
            and errors['speed']['inactive']['legacy_anchor_strictly_older_count'] > 0
            and sum(errors['speed']['inactive_endpoint_steps_'+label]['count']
                    for label in DISTANCE_BINS) == 124
            and errors['heading']['inactive']['bias'] == -3., 'population, masks or circular errors changed')
    saved['B'][1, 2, 2] = 0.
    try:
        freeze_scene(scene, saved, legacy_motion)
    except AssertionError:
        pass
    else:
        raise AssertionError('mutated C74 fallback passed')
    print('C77 self-check passed: support branches, distance, C74 shared evaluation and mutation guard.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--science', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.runs is None or args.science is None or args.output is None or args.time_policy is None:
        parser.error('--runs, --science, --output and --time-policy are required unless --self-check')
    else:
        run(args.runs.resolve(), args.science.resolve(), args.output.resolve())
