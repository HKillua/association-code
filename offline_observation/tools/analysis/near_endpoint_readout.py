"""C81: emit one fixed near-endpoint speed control on the frozen C74 outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from tools.analysis.motion_fallback_diagnostic import (
    INVARIANTS, ORIGINAL_COUNTS, check_error, frozen_arrays, require,
)
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import SPLIT, load_population, read_reference
from tools.analysis.offline_endpoint_lineage import check_failed_publication, checked_paths, load_lineage
from tools.analysis.same_source_swap_pilot import MAX_AS, PARENTS, group_members, peak_rss_bytes, read, write


METHODS = ('B', 'NEAR')
NEAR_STEPS = 2.
READOUT_STAGE = 'all_actual_speed_outputs_and_switch_masks_saved_before_reference'
EXPECTED_SPEED_COUNTS = dict(all=71090, outside_no_support=1677, near_pure=156, near_pure_older=48)


def validate_inputs(runs, science, fallback, lineage, lineage_report):
    ids, components = load_population(runs)
    archives = checked_paths(runs, science, fallback, ids, need_archive=False)
    check_failed_publication(lineage, runs, science, fallback, ids, archives)
    prior = read(lineage_report/'run.json')
    require(prior['status'] == 'COMPLETED' and prior['exit_code'] == 0 and prior['phase'] == 'done'
            and prior['split'] == SPLIT and prior['scene_ids'] == ids and prior['completed_scenes'] == len(ids)
            and prior['stages'] == ['failed_publication_lineage_reused_without_rereading_measurements',
                                    'frozen_c74_c77_reference_and_endpoint_diagnostics_complete']
            and prior['frozen_lineage_scenes'] == len(ids) and prior['reused_raw_measurement_members'] == len(ids)
            and prior['raw_measurement_members'] == prior['raw_label_members'] == prior['raw_reference_members'] == 0
            and prior['optimizer_updates'] == prior['predictions_generated'] == 0 and prior['test_opened'] is False,
            'completed original C79 repair receipt required')
    require(all(Path(prior[key]).resolve() == path for key, path in
                (('runs', runs), ('science', science), ('fallback', fallback),
                 ('output', lineage_report), ('frozen_attempt', lineage)))
            and prior['archive_paths'] == archives, 'C79 repair lineage or input paths changed')
    return ids, components


def load_support(root, sid):
    with np.load(root/'private'/f'support-{sid}.npz', allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def build_readout(saved, frozen, lineage):
    """Build an output from frozen measurements without identity or reference."""
    base = saved['B']
    membership = frozen['membership']
    query = frozen['query']
    require(query.shape == (64,) and np.isfinite(query).all() and (np.diff(query) > 0).all(),
            'original ordered Q64 required')
    shape = (len(group_members(membership)), len(query))
    require(base.shape == (*shape, 4) and np.isfinite(base).all() and (base[..., 2] >= 0).all(),
            'finite nonnegative original B speed and states required')
    np.testing.assert_array_equal(saved['membership'], membership)
    active = frozen['cause'] == 0
    np.testing.assert_array_equal(saved['active'], active)
    for name in ('internal_gap', 'near_endpoint', 'B_low_speed', 'C_low_speed'):
        np.testing.assert_array_equal(saved[name], frozen[name])
    outside = lineage['scope']
    np.testing.assert_array_equal(outside, np.isin(frozen['cause'], (1, 2, 4, 5)))
    require(not (outside & (active | frozen['visible'])).any(), 'external support scope changed')
    np.testing.assert_array_equal(base[..., 2][outside], lineage['first_field'][outside])
    np.testing.assert_array_equal(saved['A'][outside], base[outside])
    np.testing.assert_array_equal(saved['C'][outside], base[outside])
    ages = (lineage['first_age'][outside], lineage['nearest_age'][outside])
    require(all(np.isfinite(age).all() and (age > 0).all() for age in ages)
            and (ages[0] >= ages[1]).all(), 'invalid fixed endpoint ages')
    qstep = (query[-1]-query[0])/(len(query)-1)
    near = outside & (lineage['nearest_age']/qstep <= NEAR_STEPS)
    older = outside & (lineage['first_age'] > lineage['nearest_age'])
    switch = near & older
    replacement = lineage['nearest_field'][switch]
    require(np.isfinite(replacement).all() and (replacement >= 0.).all()
            and (lineage['first_member'][switch] != lineage['nearest_member'][switch]).all(),
            'selected members and their original m/s endpoint values must be valid')
    candidate = base.copy()
    candidate[..., 2][switch] = replacement
    changed = candidate[..., 2] != base[..., 2]
    require(not (changed & ~switch).any(), 'speed changed outside the frozen switch')
    masks = dict(active=active, outside_no_support=outside, near_outside=near, first_older=older,
                 switch=switch, changed=changed, same_source=lineage['same_source'],
                 before_outside=outside & (lineage['first_side'] == 0),
                 qualified_outside=np.isin(frozen['cause'], (1, 2)))
    return dict(membership=membership, B=base, NEAR=candidate, **masks)


def blind_counts(artifact):
    keys = ('active', 'outside_no_support', 'near_outside', 'first_older', 'switch', 'changed')
    switch, changed, same = (artifact[key] for key in ('switch', 'changed', 'same_source'))
    return dict(group_queries=int(switch.size), **{key: int(artifact[key].sum()) for key in keys},
                switch_same_source=int((switch & same).sum()), switch_different_source=int((switch & ~same).sum()),
                changed_same_source=int((changed & same).sum()), changed_different_source=int((changed & ~same).sum()),
                switch_before=int((switch & artifact['before_outside']).sum()),
                switch_after=int((switch & ~artifact['before_outside']).sum()))


def evaluate_scene(sid, component, artifact, frozen, ev, c74, c77, c79):
    """Align two emitted arrays with the original evaluator and inspect speed only."""
    base, candidate = (artifact[key] for key in METHODS)
    np.testing.assert_array_equal(artifact['membership'], frozen['membership'])
    np.testing.assert_array_equal(candidate[..., [0, 1, 3]], base[..., [0, 1, 3]])
    np.testing.assert_array_equal(candidate[~artifact['switch']], base[~artifact['switch']])
    np.testing.assert_array_equal(candidate[artifact['active']], base[artifact['active']])
    np.testing.assert_array_equal(candidate[..., 2] != base[..., 2], artifact['changed'])
    groups = [members.tolist() for members in group_members(artifact['membership'])]
    outputs = {key: emit_group_outputs(artifact[key], groups, node_count=len(artifact['membership']))
               for key in METHODS}
    reports, aligned = evaluate_fusion_methods(
        outputs, prediction_speed_units={key: 'mps' for key in METHODS},
        node_identities=ev['identities'], target_ids=ev['target_ids'], target_states=ev['target_states'],
        target_valid=ev['target_valid'], support=frozen['support'], nearest_seconds=frozen['nearest'],
        target_speed_unit='mps', split=SPLIT)
    b, n = (aligned[key] for key in METHODS)
    for key in INVARIANTS:
        np.testing.assert_array_equal(b[key], n[key])
    np.testing.assert_array_equal(b['aligned_prediction'][..., [0, 1, 3]],
                                  n['aligned_prediction'][..., [0, 1, 3]])
    require(all(item['scene_id'] == sid and item['component_id'] == component for item in (c74, c77, c79)),
            'original public scene and component changed')
    expected = c74['methods']['B']
    population = {key: int(reports['B'][key]) for key in POPULATION_KEYS}
    unmatched = len(reports['B']['unmatched_target_ids'])
    require(population == expected['population'] == c77['population'] == c79['population']
            and unmatched == expected['unmatched_target_count'] == c77['unmatched_target_count']
                == c79['unmatched_target_count']
            and all(reports['B'][key] == reports['NEAR'][key]
                    for key in (*POPULATION_KEYS, 'unmatched_target_ids')), 'coverage or duplicates changed')
    coverage = {}
    for region in ('all', 'observed', 'missing'):
        coverage[region] = {}
        for feature in ('position', 'speed', 'heading'):
            baseline = reports['B']['full_target_metrics'][region][feature]
            near = reports['NEAR']['full_target_metrics'][region][feature]
            expected_feature = expected['full_target_metrics'][region][feature]
            coverage[region][feature] = {key: int(baseline[key]) for key in COVERAGE_KEYS}
            require(all(baseline[key] == near[key] == expected_feature[key] for key in COVERAGE_KEYS),
                    'original feature availability changed')
    target_count = len(ev['target_ids'])
    chosen = b['aligned_group_indices'][:target_count]
    common_valid = b['aligned_prediction_valid'][:target_count] & b['aligned_truth_valid'][:target_count]
    truth = b['aligned_truth'][:target_count]
    predictions = {key: aligned[key]['aligned_prediction'][:target_count] for key in METHODS}

    def rows(column):
        target, q = np.nonzero(common_valid[..., column])
        group = chosen[target]
        require((group >= 0).all(), 'valid predictions require a matched original group')
        return target, q, group

    target, q, g = rows(2)
    error = {key: pred[target, q, 2]-truth[target, q, 2] for key, pred in predictions.items()}
    labels = {key: artifact[key][g, q] for key in
              ('active', 'outside_no_support', 'near_outside', 'first_older', 'switch', 'changed',
               'same_source', 'before_outside', 'qualified_outside')}
    pure_group = np.asarray([len(np.unique(ev['identities'][members])) == 1 for members in groups])
    pure = pure_group[g]
    qualified_near_pure = labels['qualified_outside'] & labels['near_outside'] & pure
    labels.update(all=np.ones(len(g), bool), unselected=~labels['switch'],
                  switch_pure=labels['switch'] & pure, switch_mixed=labels['switch'] & ~pure,
                  switch_same_source=labels['switch'] & labels['same_source'],
                  switch_different_source=labels['switch'] & ~labels['same_source'],
                  switch_before=labels['switch'] & labels['before_outside'],
                  switch_after=labels['switch'] & ~labels['before_outside'],
                  near_pure=qualified_near_pure,
                  near_pure_older=qualified_near_pure & labels['first_older'])
    labels = {key: value for key, value in labels.items()
              if key not in ('first_older', 'same_source', 'before_outside', 'qualified_outside')}
    for key in ('all', 'active'):
        check_error(error['B'][labels[key]], expected['errors'][key]['speed'])
        if key == 'active':
            np.testing.assert_array_equal(error['B'][labels[key]], error['NEAR'][labels[key]])
    for ours, old in (('outside_no_support', 'all_eligible'), ('near_pure', 'near_pure'),
                      ('near_pure_older', 'near_pure_first_older')):
        prior = c79['regions'][old]['proxy_reference_errors_mps']
        check_error(error['B'][labels[ours]], prior['first_field'])
        if ours.startswith('near_pure'):
            check_error(error['NEAR'][labels[ours]], prior['nearest_field'])
    np.testing.assert_array_equal(error['B'][labels['unselected']], error['NEAR'][labels['unselected']])

    target, q, g = rows(3)
    heading = _wrap(predictions['B'][target, q, 3]-truth[target, q, 3])
    for region, mask in (('all', np.ones(len(g), bool)), ('active', artifact['active'][g, q])):
        check_error(heading[mask], expected['errors'][region]['heading'])
    require(blind_counts(artifact)['outside_no_support'] == c79['legacy_fallback_group_queries'],
            'original external group-query counts changed')
    row = dict(scene_id=sid, component_id=component, population=population, unmatched_target_count=unmatched,
               reference_position_points=coverage['all']['position']['available_target_count'],
               coverage=coverage, blind_group_queries=blind_counts(artifact))
    return row, dict(speed=error, labels=labels, heading=heading, heading_active=artifact['active'][g, q])


def speed_statistics(base, candidate):
    b, n = _stats(base), _stats(candidate)
    delta = np.abs(candidate)-np.abs(base)
    return dict(count=len(delta), B=b, NEAR=n,
                **{f'delta_{key}': n[key]-b[key] if len(delta) else None
                   for key in ('mae', 'rmse', 'p95', 'p99', 'bias')},
                B_absolute_error_sum=float(np.abs(base).sum()),
                NEAR_absolute_error_sum=float(np.abs(candidate).sum()),
                absolute_error_sum_delta=float(delta.sum()),
                improved_points=int((delta < -1e-9).sum()), worsened_points=int((delta > 1e-9).sum()),
                tied_points=int((np.abs(delta) <= 1e-9).sum()),
                max_point_harm_mps=max(0., float(delta.max())) if len(delta) else None,
                max_point_gain_mps=max(0., -float(delta.min())) if len(delta) else None)


def statistics(vectors):
    base, near = (np.concatenate([v['speed'][key] for v in vectors]) for key in METHODS)
    speed = {}
    for key in vectors[0]['labels']:
        mask = np.concatenate([v['labels'][key] for v in vectors])
        speed[key] = speed_statistics(base[mask], near[mask])
    heading = np.concatenate([v['heading'] for v in vectors])
    heading_active = np.concatenate([v['heading_active'] for v in vectors])
    return dict(speed=speed, unchanged_heading=dict(all=_stats(heading), active=_stats(heading[heading_active])))


def pool(rows, vectors):
    def counts(key):
        return {field: sum(row[key][field] for row in rows) for field in rows[0][key]}

    return dict(scene_count=len(rows), component_count=len({row['component_id'] for row in rows}),
                population=counts('population'), blind_group_queries=counts('blind_group_queries'),
                unmatched_target_count=sum(row['unmatched_target_count'] for row in rows),
                reference_position_points=sum(row['reference_position_points'] for row in rows),
                coverage={region: {feature: {key: sum(row['coverage'][region][feature][key] for row in rows)
                                             for key in COVERAGE_KEYS}
                                   for feature in ('position', 'speed', 'heading')}
                          for region in ('all', 'observed', 'missing')},
                **statistics(vectors))


def run(runs, science, fallback, lineage, lineage_report, output):
    protected = (science, fallback, lineage, lineage_report, runs/PARENTS['native'])
    require(output != runs and all(output != path and path not in output.parents and output not in path.parents
                                   for path in protected), 'new output must not overlap frozen inputs')
    require(output.parent.is_dir(), 'existing output parent required')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    started = time.monotonic()
    record = dict(status='RUNNING', split=SPLIT, stages=[], started_at=datetime.now(timezone.utc).isoformat(),
                  runs=str(runs), science=str(science), fallback=str(fallback), lineage=str(lineage),
                  lineage_report=str(lineage_report), output=str(output), optimizer_updates=0,
                  raw_measurement_members=0, raw_label_members=0, raw_reference_members=0,
                  predictions_generated=0, test_opened=False, argv=sys.argv,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', record)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = validate_inputs(runs, science, fallback, lineage, lineage_report)
        checkpoint(phase='emit_frozen_speed_readout', scene_ids=ids)
        totals = np.zeros(3, np.int64)
        for index, sid in enumerate(ids, start=1):
            frozen = load_support(fallback, sid)
            source, _ = load_lineage(lineage/'private'/f'lineage-{sid}.npz', frozen, {})
            artifact = build_readout(frozen_arrays(science/'private'/f'{sid}.npz'), frozen, source)
            np.savez_compressed(private/f'{sid}.npz', **artifact)
            totals += (len(frozen['nodes']), len(artifact['B']), int(frozen['event_count']))
            checkpoint(predictions_generated=index, frozen_prediction_scenes=index)
        require(tuple(totals) == ORIGINAL_COUNTS, 'original V32 node, group or event population changed')
        record['stages'].append(READOUT_STAGE)
        checkpoint(phase='evaluate_fixed_reference', node_count=int(totals[0]), group_count=int(totals[1]),
                   event_count=int(totals[2]))

        previous = [read(root/'scenes.json.gz') for root in (science, fallback, lineage_report)]
        require(all([row['scene_id'] for row in rows] == ids for rows in previous),
                'original public scene rows changed')
        rows, vectors = [], []
        for sid, c74, c77, c79 in zip(ids, *previous, strict=True):
            with np.load(private/f'{sid}.npz', allow_pickle=False) as saved:
                artifact = {key: saved[key] for key in saved.files}
            frozen = load_support(fallback, sid)
            ev = read_reference(runs, sid, frozen['nodes'], frozen['query'])
            row, vector = evaluate_scene(sid, components[sid], artifact, frozen, ev, c74, c77, c79)
            rows.append(row)
            vectors.append(vector)

        by_component = {component: pool([r for r in rows if r['component_id'] == component],
                                        [v for r, v in zip(rows, vectors, strict=True)
                                         if r['component_id'] == component])
                        for component in sorted(set(components.values()))}
        largest = min(by_component, key=lambda key: (-by_component[key]['reference_position_points'], key))
        require(all(read(root/'summary.json')['remove_largest_component']['excluded_component'] == largest
                    for root in (science, fallback, lineage_report)), 'original largest component changed')
        all_results = pool(rows, vectors)
        require({key: all_results['speed'][key]['count'] for key in EXPECTED_SPEED_COUNTS}
                == EXPECTED_SPEED_COUNTS, 'expected original C74/C79 global parity counts changed')
        outside = [(r, v) for r, v in zip(rows, vectors, strict=True) if r['component_id'] != largest]
        leave = pool([r for r, _ in outside], [v for _, v in outside])
        temporary = output/'scenes.tmp.json.gz'
        write(temporary, [dict(row, **statistics([v])) for row, v in zip(rows, vectors, strict=True)])
        temporary.replace(output/'scenes.json.gz')
        write(output/'summary.json', dict(all_components=all_results, by_component=by_component,
            remove_largest_component=dict(excluded_component=largest, result=leave),
            invariants=dict(position_exact=True, heading_exact=True, groups_exact=True,
                            masks_exact=True, correspondence_exact=True, coverage_exact=True,
                            duplicates_exact=True, active_speed_exact=True, unselected_speed_exact=True),
            definitions=dict(B='C74 frozen B', NEAR='B with speed replaced by original nearest member endpoint field',
                             switch='outside original group span with no member support; nearest endpoint at most '
                                    '2 original Q64 steps; first endpoint strictly farther in actual seconds',
                             selected_speed_unit='mps already converted once in frozen C79',
                             changed='numerical speed value differs after observable switch',
                             near_pure='original C79 qualified near pure stratum, evaluation-only identity',
                             errors='prediction minus frozen reference in m/s; heading shortest circular arc',
                             split='original already-seen V32 development scenes')))
        record['stages'].append('fixed_b_and_actual_near_output_evaluation_complete')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, completed_scenes=len(rows),
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


def self_check():
    from tempfile import TemporaryDirectory

    from association.data.motion_evidence import complete_native_attributes
    from association.data.physical_units import KNOT_TO_MPS
    from tools.analysis.motion_fallback_diagnostic import MASKS, freeze_scene, measure_scene, pool as c77_pool, support_tags
    from tools.analysis.motion_readout_evaluation import evaluate_scene as evaluate_c74
    from tools.analysis.offline_endpoint_lineage import endpoint_arrays, evaluate as evaluate_c79, freeze_endpoints, summarize_vectors

    query = np.arange(64, dtype=float)
    times = np.array([3., 4., 5., 2., 5., 7., 14., 16., 17., 13., 16., 20.])
    offsets = np.arange(0, len(times)+1, 3)
    values = np.column_stack((np.ones(len(times)), np.arange(len(times))*.001,
                              [30., 30., 30., 8., 8., 8., 10., 10., 10., 40., 40., 40.],
                              np.full(len(times), 33.)))
    membership = np.array([0, 0, 1, 1])
    nodes = np.array(['n0', 'n1', 'n2', 'n3'])
    sources = np.array(['a', 'b', 'a', 'a'])
    groups = group_members(membership)
    per_node_times = times.reshape(4, 3)
    support = (query >= per_node_times[:, :1]) & (query <= per_node_times[:, -1:])
    tags = support_tags(query, times, offsets, groups, support, sources)
    nearest = np.abs(query-per_node_times[..., None]).min(axis=1)
    native = (query, times, values[:, :2], offsets)
    legacy = complete_native_attributes(values.reshape(4, 3, 4), per_node_times,
        np.ones((4, 3), bool), query, np.zeros((4, 64, 2)))[..., 2:]
    base = np.zeros((2, 64, 4))
    base[..., :2] = (1., 2.)
    base[..., 2:] = legacy[[0, 2]]
    base[..., 2] *= KNOT_TO_MPS
    old = dict(membership=membership, A=base.copy(), B=base.copy(), C=base.copy(),
               active=tags['cause'] == 0, internal_gap=tags['internal_gap'], near_endpoint=tags['near_endpoint'],
               B_low_speed=np.zeros((2, 64), bool), C_low_speed=np.zeros((2, 64), bool))
    frozen = freeze_scene((nodes, sources, support, nearest, membership, native, base), old, legacy)
    source = freeze_endpoints(frozen, query, endpoint_arrays(times, values, offsets), sources, base, legacy)
    artifact = build_readout(old, frozen, source)
    np.testing.assert_array_equal(np.flatnonzero(artifact['switch'][0]), [0, 1, 8, 9])
    np.testing.assert_array_equal(np.flatnonzero(artifact['switch'][1]), [11, 12, 21, 22])
    np.testing.assert_array_equal(artifact['NEAR'][0, [0, 1, 8, 9], 2], np.full(4, 8.*KNOT_TO_MPS))
    np.testing.assert_array_equal(artifact['NEAR'][1, [11, 12, 21, 22], 2], np.full(4, 40.*KNOT_TO_MPS))
    truth = np.zeros((4, 64, 4))
    truth[..., :2] = (1., 2.)
    truth[..., 2] = 4.
    ev = dict(identities=np.array(['shipA', 'shipA', 'shipB', 'shipC']),
              target_ids=np.array(['shipA', 'shipB', 'shipC', 'missing']),
              target_states=truth, target_valid=np.ones_like(truth, bool))
    c74, _ = evaluate_c74('9', 'component', nodes, [g.tolist() for g in groups], support, nearest, ev,
                           {key: old[key] for key in ('A', 'B', 'C')},
                           {key: old[key] for key in MASKS}, query)
    c77, c77_vectors = measure_scene('9', 'component', frozen, old, ev, c74)
    c77 = dict(c77, errors=c77_pool([c77], [c77_vectors])['errors'])
    c79, c79_vectors = evaluate_c79('9', 'component', frozen, old, source, ev, c74, c77)
    c79 = dict(c79, regions=summarize_vectors(c79_vectors))
    row, vector = evaluate_scene('9', 'component', artifact, frozen, ev, c74, c77, c79)
    report = pool([row], [vector])
    speed = report['speed']
    require(speed['switch']['count'] == speed['changed']['count'] == 8 and speed['all']['count'] == 128
            and speed['switch_pure']['improved_points'] == 4 and speed['switch_mixed']['worsened_points'] == 4
            and speed['active']['delta_mae'] == 0. and speed['unselected']['delta_mae'] == 0.
            and speed['near_pure_older']['count'] == 4 and row['unmatched_target_count'] == 2,
            'actual output, invariant or mixed-group error accounting changed')
    with TemporaryDirectory() as directory:
        output = Path(directory)
        write(output/'summary.json', report)
        write(output/'scenes.json.gz', [dict(row, **statistics([vector]))])
        require(read(output/'summary.json') == report and len(read(output/'scenes.json.gz')) == 1,
                'strict JSON and GZ publication changed')
    print('C81 self-check passed: blind Q bound, field units, actual aligned outputs, pure/mixed errors and strict publication.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'science', 'fallback', 'lineage', 'lineage-report', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not all((args.runs, args.science, args.fallback, args.lineage, args.lineage_report,
                  args.output, args.time_policy)):
        parser.error('--runs, --science, --fallback, --lineage, --lineage-report, --output and --time-policy are required')
    else:
        run(*(path.resolve() for path in (args.runs, args.science, args.fallback, args.lineage,
                                          args.lineage_report, args.output)))
