"""C84: paper-dimension fusion metrics on already emitted C81 V32 outputs."""
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
from association.evaluation.paper_fusion_metrics import COUNTS, aggregate_paper_fusion
from tools.analysis.motion_fallback_diagnostic import INVARIANTS, ORIGINAL_COUNTS, check_error, require
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import SPLIT, load_population, read_reference
from tools.analysis.near_endpoint_readout import METHODS, READOUT_STAGE, load_support
from tools.analysis.same_source_swap_pilot import MAX_AS, PARENTS, group_members, peak_rss_bytes, read, write


def bind_inputs(runs, c81, fallback):
    ids, components = load_population(runs)
    record = read(c81/'run.json')
    require(record['status'] == 'COMPLETED' and record['exit_code'] == 0 and record['split'] == SPLIT
            and record['scene_ids'] == ids and record['completed_scenes'] == len(ids)
            and record['frozen_prediction_scenes'] == len(ids) and record['predictions_generated'] == len(ids)
            and record['optimizer_updates'] == 0 and record['test_opened'] is False
            and record['stages'] == [READOUT_STAGE, 'fixed_b_and_actual_near_output_evaluation_complete']
            and tuple(record[key] for key in ('node_count', 'group_count', 'event_count')) == ORIGINAL_COUNTS
            and Path(record['runs']).resolve() == runs and Path(record['output']).resolve() == c81
            and Path(record['fallback']).resolve() == fallback, 'completed original C81 and its C77 inputs required')
    support = read(fallback/'run.json')
    require(support['status'] == 'COMPLETED' and support['exit_code'] == 0 and support['split'] == SPLIT
            and support['scene_ids'] == ids and support['completed_scenes'] == len(ids)
            and Path(support['output']).resolve() == fallback, 'completed original C77 support required')
    return ids, components


def score_scene(sid, component, artifact, support, ev, expected):
    members = artifact['membership']
    np.testing.assert_array_equal(members, support['membership'])
    b, near = (artifact[key] for key in METHODS)
    np.testing.assert_array_equal(b[..., [0, 1, 3]], near[..., [0, 1, 3]])
    np.testing.assert_array_equal(b[~artifact['switch']], near[~artifact['switch']])
    np.testing.assert_array_equal(artifact['active'], support['cause'] == 0)
    np.testing.assert_array_equal(b[artifact['active']], near[artifact['active']])
    np.testing.assert_array_equal(b[..., 2] != near[..., 2], artifact['changed'])
    groups = [group.tolist() for group in group_members(members)]
    reports, arrays = evaluate_fusion_methods(
        {key: emit_group_outputs(artifact[key], groups, node_count=len(members)) for key in METHODS},
        prediction_speed_units={key: 'mps' for key in METHODS}, node_identities=ev['identities'],
        target_ids=ev['target_ids'], target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=support['support'], nearest_seconds=support['nearest'], target_speed_unit='mps', split=SPLIT)
    original, candidate = (arrays[key] for key in METHODS)
    for key in INVARIANTS:
        np.testing.assert_array_equal(original[key], candidate[key])
    np.testing.assert_array_equal(original['aligned_prediction'][..., [0, 1, 3]],
                                  candidate['aligned_prediction'][..., [0, 1, 3]])
    require(expected['scene_id'] == sid and expected['component_id'] == component, 'C81 scene order changed')
    coverage = {}
    for region in ('all', 'observed', 'missing'):
        coverage[region] = {}
        for feature in ('position', 'speed', 'heading'):
            original_coverage = {key: int(reports['B']['full_target_metrics'][region][feature][key])
                                 for key in COVERAGE_KEYS}
            require(original_coverage == expected['coverage'][region][feature] and all(
                value == reports['NEAR']['full_target_metrics'][region][feature][key]
                for key, value in original_coverage.items()), 'C81 feature availability changed')
            coverage[region][feature] = original_coverage
    population = {key: int(reports['B'][key]) for key in POPULATION_KEYS}
    unmatched = len(reports['B']['unmatched_target_ids'])
    require(population == expected['population'] and unmatched == expected['unmatched_target_count']
            and all(reports['B'][key] == reports['NEAR'][key]
                    for key in (*POPULATION_KEYS, 'unmatched_target_ids')), 'C81 target or duplicate population changed')

    paper = {key: reports[key]['paper_fusion_metrics'] for key in METHODS}
    for name, row in paper['B']['attributes'].items():
        other = paper['NEAR']['attributes'][name]
        require(all(row[key] == other[key] for key in COUNTS) and row['unit'] == other['unit'],
                'C81 paper metric population changed')
        if name != 'speed':
            require(row == other, 'C81 fixed coordinate or heading paper metric changed')
        else:
            require(row['unit'] == 'kn', 'paper speed must use explicit knots')
    nt = len(ev['target_ids'])
    aligned_groups = original['aligned_group_indices'][:nt]
    shared = original['aligned_prediction_valid'][:nt] & original['aligned_truth_valid'][:nt]
    truth = original['aligned_truth'][:nt]

    def points(column):
        target, q = np.nonzero(shared[..., column])
        group = aligned_groups[target]
        require((group >= 0).all(), 'valid point must have original matched group')
        return target, q, group

    target, q, g = points(2)
    speed = {key: arrays[key]['aligned_prediction'][target, q, 2]-truth[target, q, 2] for key in METHODS}
    for region, mask in (('all', np.ones(len(g), bool)),
                         ('active', artifact['active'][g, q]),
                         ('switch', artifact['switch'][g, q]),
                         ('changed', artifact['changed'][g, q]),
                         ('unselected', ~artifact['switch'][g, q])):
        for key in METHODS:
            check_error(speed[key][mask], expected['speed'][region][key])
    target, q, g = points(3)
    heading = _wrap(original['aligned_prediction'][target, q, 3]-truth[target, q, 3])
    heading_active = heading[artifact['active'][g, q]]
    check_error(heading, expected['unchanged_heading']['all'])
    check_error(heading_active, expected['unchanged_heading']['active'])
    return dict(scene_id=sid, component_id=component, population=population, unmatched_target_count=unmatched,
                coverage=coverage, paper_fusion_metrics=paper,
                circular_heading_point_statistics=dict(all=_stats(heading), active=_stats(heading_active)),
                reference_position_points=coverage['all']['position']['available_target_count']), (heading, heading_active)


def pool(rows, vectors):
    return dict(scene_count=len(rows), component_count=len({row['component_id'] for row in rows}),
                population={key: sum(row['population'][key] for row in rows) for key in POPULATION_KEYS},
                unmatched_target_count=sum(row['unmatched_target_count'] for row in rows),
                reference_position_points=sum(row['reference_position_points'] for row in rows),
                coverage={region: {feature: {key: sum(row['coverage'][region][feature][key] for row in rows)
                                             for key in COVERAGE_KEYS}
                                   for feature in ('position', 'speed', 'heading')}
                          for region in ('all', 'observed', 'missing')},
                paper_fusion_metrics={key: aggregate_paper_fusion([row['paper_fusion_metrics'][key] for row in rows])
                                      for key in METHODS},
                circular_heading_point_statistics={name: _stats(np.concatenate([v[column] for v in vectors]))
                    for column, name in enumerate(('all', 'active'))})


def check_population(actual, prior, vectors):
    require(all(actual[key] == prior[key] for key in
                ('scene_count', 'component_count', 'population', 'unmatched_target_count',
                 'reference_position_points', 'coverage')), 'C81 aggregate population changed')
    for column, name in enumerate(('all', 'active')):
        check_error(np.concatenate([v[column] for v in vectors]), prior['unchanged_heading'][name])


def run(runs, c81, fallback, output):
    protected = (c81, fallback, runs/PARENTS['native'])
    require(output != runs and all(output != path and path not in output.parents and output not in path.parents
                                   for path in protected), 'new output must not overlap frozen inputs')
    output.mkdir(exist_ok=False)
    began = time.monotonic()
    record = dict(status='RUNNING', split=SPLIT, started_at=datetime.now(timezone.utc).isoformat(),
                  runs=str(runs), c81=str(c81), fallback=str(fallback), output=str(output),
                  optimizer_updates=0, predictions_generated=0, archive_opened=False, test_opened=False,
                  argv=sys.argv, affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', record)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = bind_inputs(runs, c81, fallback)
        prior_rows = read(c81/'scenes.json.gz')
        prior = read(c81/'summary.json')
        require([row['scene_id'] for row in prior_rows] == ids and all(prior['invariants'].values()),
                'C81 result order or frozen invariants changed')
        checkpoint(phase='score_frozen_outputs', scene_ids=ids)
        rows, vectors = [], []
        for sid, expected in zip(ids, prior_rows, strict=True):
            with np.load(c81/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                artifact = {key: saved[key] for key in saved.files}
            support = load_support(fallback, sid)
            ev = read_reference(runs, sid, support['nodes'], support['query'])
            row, vector = score_scene(sid, components[sid], artifact, support, ev, expected)
            rows.append(row)
            vectors.append(vector)
            checkpoint(completed_scenes=len(rows))
        all_components = pool(rows, vectors)
        check_population(all_components, prior['all_components'], vectors)

        def subset(indices, reference):
            part_rows, part_vectors = [rows[i] for i in indices], [vectors[i] for i in indices]
            result = pool(part_rows, part_vectors)
            check_population(result, reference, part_vectors)
            return result

        by_component = {component: subset([i for i, row in enumerate(rows) if row['component_id'] == component],
                                          prior['by_component'][component])
                        for component in sorted(set(components.values()))}
        largest = min(by_component, key=lambda k: (-by_component[k]['reference_position_points'], k))
        remove_largest = prior['remove_largest_component']
        require(largest == remove_largest['excluded_component'], 'C81 largest original component changed')
        leave = subset([i for i, row in enumerate(rows) if row['component_id'] != largest], remove_largest['result'])
        temporary = output/'scenes.tmp.json.gz'
        write(temporary, rows)
        temporary.replace(output/'scenes.json.gz')
        write(output/'summary.json', dict(all_components=all_components, by_component=by_component,
            remove_largest_component=dict(excluded_component=largest, result=leave),
            invariants=dict(position_exact=True, heading_exact=True, masks_exact=True, correspondence_exact=True,
                            coverage_exact=True, duplicates_exact=True, c81_physical_metrics_reproduced=True),
            definitions=dict(
                population='original already-used development V32; different population from author Table 3',
                paper_scope='target-equal within scene and scene-equal across scenes, over each attribute valid intersection',
                paper_sign='reference minus prediction; raw linear longitude and heading',
                speed_units='frozen C81 prediction and cached reference m/s, reported as kn',
                paper_sd='residual sample SD; printed Eq29 preserved separately as diagnostic; author ESD implementation unknown',
                circular_heading='original C81 point-weighted wrapped heading error, prediction minus reference')))
        checkpoint(status='COMPLETED', phase='done', exit_code=0,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


def self_check():
    from tempfile import TemporaryDirectory

    group_count, queries = 2, 4
    state = np.zeros((group_count, queries, 4))
    state[..., :2] = (1., 2.)
    state[0, :, 2] = [1., 2., 3., 4.]
    state[1, :, 2] = [8., 9., 10., 11.]
    state[..., 3] = 359.
    truth = state.copy()
    truth[..., 3] = 1.
    truth[..., 2] = 1.
    valid = np.ones_like(truth, bool)
    valid[1, -1, 2] = False
    members = np.array([0, 1])
    switch = np.zeros(state.shape[:2], bool)
    switch[0, 3] = True
    candidate = state.copy()
    candidate[..., 2][switch] = 1.
    support = dict(membership=members, cause=np.where(switch, 1, 0),
                   support=~switch, nearest=np.zeros(state.shape[:2]))
    artifact = dict(membership=members, B=state, NEAR=candidate, switch=switch, changed=switch, active=~switch)
    ids = np.array(['a', 'b'])
    ev = dict(identities=ids, target_ids=ids, target_states=truth, target_valid=valid)
    baseline_report, _ = evaluate_fusion_methods({'B': emit_group_outputs(state, [[0], [1]], node_count=2)},
        prediction_speed_units={'B': 'mps'}, node_identities=ids, target_ids=ids, target_states=truth,
        target_valid=valid, support=support['support'], nearest_seconds=support['nearest'], target_speed_unit='mps')
    original = baseline_report['B']
    masks = dict(all=valid[..., 2], active=~switch & valid[..., 2],
                 switch=switch, changed=switch, unselected=~switch & valid[..., 2])
    expected = dict(scene_id='a', component_id='g', population={key: original[key] for key in POPULATION_KEYS},
                    unmatched_target_count=0, coverage={r: {f: {key: original['full_target_metrics'][r][f][key]
                                                                 for key in COVERAGE_KEYS}
                                                            for f in ('position', 'speed', 'heading')}
                                                       for r in ('all', 'observed', 'missing')},
                    speed={r: {key: _stats((artifact[key][..., 2]-truth[..., 2])[mask]) for key in METHODS}
                           for r, mask in masks.items()},
                    unchanged_heading=dict(all=_stats(np.full(8, -2.)), active=_stats(np.full(7, -2.))))
    row, vector = score_scene('a', 'g', artifact, support, ev, expected)
    aggregate = pool([row], [vector])
    raw = aggregate['paper_fusion_metrics']['B']['attributes']
    near_raw = aggregate['paper_fusion_metrics']['NEAR']['attributes']
    np.testing.assert_allclose(raw['heading']['scene_equal_target_mean']['ME'], -358.)
    np.testing.assert_allclose(raw['speed']['scene_equal_target_mean']['ME'], -4.75*3600./1852.)
    require(near_raw['speed']['scene_equal_target_mean']['ME'] > raw['speed']['scene_equal_target_mean']['ME']
            and raw['speed']['evaluated_points'] == 7 and raw['speed']['strict_complete_population']['ME'] is None
            and aggregate['circular_heading_point_statistics']['all']['mae'] == 2.,
            'same-dimension metrics or heading convention changed')
    with TemporaryDirectory() as directory:
        path = Path(directory)
        write(path/'summary.json', aggregate)
        write(path/'scenes.json.gz', [row])
        require(read(path/'summary.json') == aggregate and read(path/'scenes.json.gz') == [row],
                'strict JSON and GZ result round trip failed')
    print('C84 self-check passed: target-equal metrics, explicit knots, missing speed, circular heading and strict publication.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'c81', 'fallback', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not all((args.runs, args.c81, args.fallback, args.output, args.time_policy)):
        parser.error('--runs, --c81, --fallback, --output and --time-policy are required')
    else:
        run(*(path.resolve() for path in (args.runs, args.c81, args.fallback, args.output)))
