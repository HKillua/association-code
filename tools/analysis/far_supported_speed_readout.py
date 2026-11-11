"""C86: compare frozen speed readouts on the far side of observed support."""
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
from tools.analysis.motion_fallback_diagnostic import (
    INVARIANTS, ORIGINAL_COUNTS, check_error, distribution, frozen_arrays, require,
)
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS
from tools.analysis.motion_readout_pilot import SPLIT, read_reference
from tools.analysis.near_endpoint_readout import load_support, speed_statistics, validate_inputs
from tools.analysis.offline_endpoint_lineage import load_lineage
from tools.analysis.same_source_swap_pilot import MAX_AS, PARENTS, group_members, peak_rss_bytes, read, write
from tools.analysis.tmtaff_fusion_metric_bridge import bind_inputs, score_scene as check_c81_scene


METHODS = ('NEAR', 'NEAREST_FIELD', 'SUPPORTED_HOLD')
REGIONS = ('all', 'far', 'before', 'after')
FAR_STEPS = 4.
FAR_EFFECTIVE_COUNT = 1349


def build_readouts(c81, c74, support, lineage):
    """Use frozen speeds only. All outputs retain the original groups and support."""
    query = support['query']
    require(query.shape == (64,) and np.isfinite(query).all() and (np.diff(query) > 0).all(),
            'original physical Q64 order required')
    base = c81['NEAR']
    membership = c81['membership']
    shape = (len(group_members(membership)), len(query))
    require(base.shape == (*shape, 4) and np.isfinite(base).all() and (base[..., 2] >= 0.).all(),
            'original finite C81 outputs required')
    for previous in (c74, support):
        np.testing.assert_array_equal(previous['membership'], membership)
    np.testing.assert_array_equal(c74['B'], c81['B'])
    np.testing.assert_array_equal(c81['active'], support['cause'] == 0)
    np.testing.assert_array_equal(c74['active'], c81['active'])
    np.testing.assert_array_equal(c81['B'][~c81['switch']], base[~c81['switch']])
    np.testing.assert_array_equal(c81['B'][..., [0, 1, 3]], base[..., [0, 1, 3]])
    step = (query[-1]-query[0])/(len(query)-1)
    cause = support['cause']
    visible = support['visible']
    distance = support['distance']
    require(all(a.shape == shape for a in (cause, visible, distance)) and visible.dtype == bool
            and np.isfinite(distance).all() and (distance >= 0.).all(), 'invalid original C77 support tags')
    far = np.isin(cause, (1, 2)) & ~visible & (distance/step > FAR_STEPS)
    require(not (far & (c81['switch'] | c81['active'])).any() and lineage['scope'][far].all(),
            'far eligibility overlaps fixed C81 switch or lacks C79 lineage')
    np.testing.assert_array_equal(lineage['nearest_age'][far], distance[far])
    np.testing.assert_array_equal(lineage['first_field'][far], base[..., 2][far])
    before, after = far & (cause == 1), far & (cause == 2)
    donor = np.full(shape, -1, np.int64)
    for group in np.flatnonzero(far.any(axis=1)):
        supported = np.flatnonzero((cause[group] == 0) & visible[group])
        require(len(supported) == support['eligible'][group] and len(supported) >= 2,
                'qualified external group lost its original supported Q queries')
        donor[group, before[group]] = supported[0]
        donor[group, after[group]] = supported[-1]
    g, q = np.nonzero(far)
    d = donor[far]
    require((d >= 0).all() and np.all(np.where(before[far], d > q, d < q)),
            'donor is outside original Q support or on wrong time side')
    donor_age = np.abs(query[q]-query[d])
    require((donor_age >= distance[far]-1e-8).all(), 'donor predates the original support boundary')
    candidates = {name: base.copy() for name in METHODS[1:]}
    candidates['NEAREST_FIELD'][..., 2][far] = lineage['nearest_field'][far]
    candidates['SUPPORTED_HOLD'][..., 2][far] = c74['B'][g, d, 2]
    for candidate in candidates.values():
        require(np.isfinite(candidate).all() and (candidate[..., 2] >= 0.).all(), 'invalid donor speed')
        np.testing.assert_array_equal(candidate[..., [0, 1, 3]], base[..., [0, 1, 3]])
        np.testing.assert_array_equal(candidate[~far], base[~far])
    ages = np.full(shape, -1., np.float64)
    ages[far] = donor_age
    return dict(c81, **candidates, far=far, before=before, after=after, donor=donor,
                donor_age_seconds=ages, nearest_event_seconds=distance)


def scoped_speed(vectors):
    errors = {method: np.concatenate([item['errors'][method] for item in vectors]) for method in METHODS}
    output = {}
    for region in REGIONS:
        mask = np.concatenate([item['regions'][region] for item in vectors])
        base = errors['NEAR'][mask]
        pairs = {}
        for name in METHODS[1:]:
            stats = speed_statistics(base, errors[name][mask])
            stats['candidate'] = stats.pop('NEAR')
            stats['NEAR'] = stats.pop('B')
            stats['candidate_absolute_error_sum'] = stats.pop('NEAR_absolute_error_sum')
            stats['NEAR_absolute_error_sum'] = stats.pop('B_absolute_error_sum')
            pairs[name] = stats
        output[region] = pairs
    return output


def evaluate_scene(sid, component, artifact, support, ev, expected):
    c81, _ = check_c81_scene(sid, component, artifact, support, ev, expected)
    members = artifact['membership']
    groups = [group.tolist() for group in group_members(members)]
    reports, arrays = evaluate_fusion_methods(
        {key: emit_group_outputs(artifact[key], groups, node_count=len(members)) for key in METHODS},
        prediction_speed_units={key: 'mps' for key in METHODS}, node_identities=ev['identities'],
        target_ids=ev['target_ids'], target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=support['support'], nearest_seconds=support['nearest'], target_speed_unit='mps', split=SPLIT)
    original = arrays['NEAR']
    paper = {key: reports[key]['paper_fusion_metrics'] for key in METHODS}
    require(paper['NEAR'] == c81['paper_fusion_metrics']['NEAR'], 'frozen C81 NEAR paper parity changed')
    for name in METHODS:
        values = arrays[name]
        for key in (*INVARIANTS, 'aligned_truth'):
            np.testing.assert_array_equal(original[key], values[key])
        np.testing.assert_array_equal(original['aligned_prediction'][..., [0, 1, 3]],
                                      values['aligned_prediction'][..., [0, 1, 3]])
        report = reports[name]
        require(all(report[key] == c81['population'][key] for key in POPULATION_KEYS)
                and report['unmatched_target_ids'] == reports['NEAR']['unmatched_target_ids'],
                'C81 target coverage or duplicates changed')
        for region, features in c81['coverage'].items():
            for feature, coverage in features.items():
                require(all(report['full_target_metrics'][region][feature][key] == coverage[key]
                            for key in COVERAGE_KEYS), 'C81 reference or prediction masks changed')
        for feature, base in paper['NEAR']['attributes'].items():
            actual = paper[name]['attributes'][feature]
            require(all(actual[key] == base[key] for key in COUNTS) and actual['unit'] == base['unit'],
                    'C81 paper-metric population changed')
            if feature != 'speed':
                require(actual == base, 'fixed paper metric changed outside speed')
    nt = len(ev['target_ids'])
    common = original['aligned_truth_valid'][:nt, :, 2] & original['aligned_prediction_valid'][:nt, :, 2]
    target, q = np.nonzero(common)
    g = original['aligned_group_indices'][target]
    require((g >= 0).all(), 'evaluated speed points require original matched groups')
    truth = original['aligned_truth'][target, q, 2]
    errors = {name: arrays[name]['aligned_prediction'][target, q, 2]-truth for name in METHODS}
    check_error(errors['NEAR'], expected['speed']['all']['NEAR'])
    regions = dict(all=np.ones(len(g), bool), **{key: artifact[key][g, q] for key in REGIONS[1:]})
    far = regions['far']
    vector = dict(errors=errors, regions=regions,
                  donor_age=artifact['donor_age_seconds'][g[far], q[far]],
                  nearest_age=artifact['nearest_event_seconds'][g[far], q[far]])
    group_far = artifact['far']
    blind = {key: int(artifact[key].sum()) for key in REGIONS[1:]}
    for name in METHODS[1:]:
        changed = artifact[name][..., 2] != artifact['NEAR'][..., 2]
        require(not (changed & ~group_far).any(), 'speed changed outside frozen far support')
        blind['changed_'+name] = int(changed.sum())
    row = {key: c81[key] for key in ('scene_id', 'component_id', 'population', 'unmatched_target_count',
                                    'reference_position_points', 'coverage', 'circular_heading_point_statistics')}
    return dict(row, blind_group_queries=blind, paper_fusion_metrics=paper), vector


def pool(rows, vectors):
    return dict(scene_count=len(rows), component_count=len({row['component_id'] for row in rows}),
                population={key: sum(row['population'][key] for row in rows) for key in POPULATION_KEYS},
                unmatched_target_count=sum(row['unmatched_target_count'] for row in rows),
                reference_position_points=sum(row['reference_position_points'] for row in rows),
                blind_group_queries={key: sum(row['blind_group_queries'][key] for row in rows)
                                     for key in rows[0]['blind_group_queries']},
                coverage={region: {feature: {key: sum(row['coverage'][region][feature][key] for row in rows)
                                             for key in COVERAGE_KEYS}
                                   for feature in ('position', 'speed', 'heading')}
                          for region in ('all', 'observed', 'missing')},
                paper_fusion_metrics={name: aggregate_paper_fusion(
                    [row['paper_fusion_metrics'][name] for row in rows]) for name in METHODS},
                **readout_statistics(vectors))


def readout_statistics(vectors):
    donor_age = np.concatenate([v['donor_age'] for v in vectors])
    nearest_age = np.concatenate([v['nearest_age'] for v in vectors])
    return dict(speed=scoped_speed(vectors),
                far_support_seconds=dict(donor_age=distribution(donor_age), nearest_event_age=distribution(nearest_age),
                                         donor_minus_nearest=distribution(donor_age-nearest_age)))


def run(runs, c81, science, fallback, lineage, lineage_report, output):
    protected = (c81, science, fallback, lineage, lineage_report, runs/PARENTS['native'])
    require(output != runs and all(output != path and path not in output.parents and output not in path.parents
                                   for path in protected), 'new output must not overlap frozen inputs')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    began = time.monotonic()
    record = dict(status='RUNNING', split=SPLIT, stages=[], started_at=datetime.now(timezone.utc).isoformat(),
                  runs=str(runs), c81=str(c81), science=str(science), fallback=str(fallback),
                  lineage=str(lineage), lineage_report=str(lineage_report), output=str(output),
                  optimizer_updates=0, predictions_generated=0, archive_opened=False, test_opened=False,
                  argv=sys.argv, affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', record)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = bind_inputs(runs, c81, fallback)
        bound_ids, bound_components = validate_inputs(runs, science, fallback, lineage, lineage_report)
        require(ids == bound_ids and components == bound_components, 'frozen input scene order or component changed')
        receipt = read(c81/'run.json')
        require(all(Path(receipt[key]).resolve() == path for key, path in
                    (('science', science), ('lineage', lineage), ('lineage_report', lineage_report))),
                'C81 run did not use declared C74 and C79 inputs')
        checkpoint(phase='freeze_far_readouts', scene_ids=ids)
        totals = np.zeros(3, np.int64)
        for index, sid in enumerate(ids, start=1):
            support = load_support(fallback, sid)
            c79, _ = load_lineage(lineage/'private'/f'lineage-{sid}.npz', support, {})
            with np.load(c81/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                original = {key: saved[key] for key in saved.files}
            artifact = build_readouts(original, frozen_arrays(science/'private'/f'{sid}.npz'), support, c79)
            np.savez_compressed(private/f'{sid}.npz', **artifact)
            totals += (len(support['nodes']), len(artifact['NEAR']), int(support['event_count']))
            checkpoint(predictions_generated=index, frozen_prediction_scenes=index)
        require(tuple(totals) == ORIGINAL_COUNTS, 'original C81 V32 population changed')
        record['stages'].append('all_c86_readouts_and_donor_indices_frozen_before_reference')
        checkpoint(phase='evaluate_fixed_reference', node_count=int(totals[0]), group_count=int(totals[1]),
                   event_count=int(totals[2]))
        expected = read(c81/'scenes.json.gz')
        require([row['scene_id'] for row in expected] == ids, 'C81 published scene order changed')
        rows, vectors = [], []
        for sid, prior in zip(ids, expected, strict=True):
            support = load_support(fallback, sid)
            ev = read_reference(runs, sid, support['nodes'], support['query'])
            with np.load(private/f'{sid}.npz', allow_pickle=False) as saved:
                artifact = {key: saved[key] for key in saved.files}
            row, vector = evaluate_scene(sid, components[sid], artifact, support, ev, prior)
            rows.append(row)
            vectors.append(vector)
            checkpoint(completed_scenes=len(rows))

        def subset(indices):
            return pool([rows[i] for i in indices], [vectors[i] for i in indices])

        by_component = {component: subset([i for i, r in enumerate(rows) if r['component_id'] == component])
                        for component in sorted(set(components.values()))}
        largest = min(by_component, key=lambda k: (-by_component[k]['reference_position_points'], k))
        prior = read(c81/'summary.json')
        require(largest == prior['remove_largest_component']['excluded_component'],
                'C81 largest component changed')
        all_components = pool(rows, vectors)
        leave = subset([i for i, r in enumerate(rows) if r['component_id'] != largest])
        for result, old in ((all_components, prior['all_components']),
                            (leave, prior['remove_largest_component']['result']),
                            *((item, prior['by_component'][key]) for key, item in by_component.items())):
            require(all(result[key] == old[key] for key in
                        ('scene_count', 'component_count', 'population', 'unmatched_target_count',
                         'reference_position_points', 'coverage')), 'C81 aggregate coverage changed')
        require(all_components['speed']['far']['SUPPORTED_HOLD']['count'] == FAR_EFFECTIVE_COUNT,
                'original C77 far effective population changed')
        write(output/'scenes.json.gz', [dict(row, **readout_statistics([vector]))
                                       for row, vector in zip(rows, vectors, strict=True)])
        write(output/'summary.json', dict(all_components=all_components, by_component=by_component,
            remove_largest_component=dict(excluded_component=largest, result=leave),
            invariants=dict(position_exact=True, heading_exact=True, groups_exact=True, masks_exact=True,
                            correspondence_exact=True, coverage_exact=True, duplicates_exact=True,
                            nonfar_speed_exact=True, c81_physical_metrics_reproduced=True),
            definitions=dict(population='original already-used development V32', methods=list(METHODS),
                scope='C77 qualified before/after span, no member support, nearest original event over 4 original Q steps',
                donor='C74 B speed at earliest/latest originally supported in-span query on the same side',
                control='C79 frozen nearest endpoint field; original member order then first endpoint breaks ties',
                errors='point-pool speed prediction minus reference in m/s; paper four attributes full population only',
                no_support='original unsupported queries remain unsupported; an emitted state adds no observation')))
        record['stages'].append('fixed_reference_comparison_and_frozen_population_checks_complete')
        checkpoint(status='COMPLETED', phase='done', exit_code=0,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


def self_check():
    from tempfile import TemporaryDirectory

    query = np.arange(64, dtype=np.float64)*3.
    state = np.zeros((1, 64, 4))
    state[..., 2] = 6.
    state[0, [10, 20], 2] = [2., 4.]
    cause = np.where(query < query[10], 1, np.where(query > query[20], 2, 0))[None, :]
    active = cause == 0
    visible = np.zeros_like(active)
    visible[0, [10, 20]] = True
    distance = np.minimum(abs(query-query[10]), abs(query-query[20]))[None, :]
    membership = np.array([0])
    original = dict(B=state, NEAR=state, membership=membership, active=active,
                    switch=np.zeros_like(active), changed=np.zeros_like(active))
    support = dict(query=query, visible=visible, cause=cause, membership=membership,
                   distance=distance, eligible=np.array([2]))
    lineage = dict(scope=~active, nearest_age=distance, first_field=state[..., 2],
                   nearest_field=np.full_like(distance, 8.))
    actual = build_readouts(original, original, support, lineage)
    require(actual['far'][0, 5] and not actual['far'][0, 6] and not actual['far'][0, 24]
            and actual['far'][0, 25], 'far threshold or side classification changed')
    np.testing.assert_array_equal(actual['SUPPORTED_HOLD'][0, [5, 25], 2], [2., 4.])
    np.testing.assert_array_equal(actual['NEAREST_FIELD'][0, [5, 25], 2], [8., 8.])
    np.testing.assert_array_equal(actual['donor'][0, [5, 25]], [10, 20])
    np.testing.assert_array_equal(actual['donor_age_seconds'][0, [5, 25]], [15., 15.])
    vector = dict(errors={name: actual[name][0, :, 2]-3. for name in METHODS},
                  regions=dict(all=np.ones(64, bool), **{key: actual[key][0] for key in REGIONS[1:]}),
                  donor_age=actual['donor_age_seconds'][actual['far']], nearest_age=distance[actual['far']])
    stats = readout_statistics([vector])
    require(stats['speed']['far']['SUPPORTED_HOLD']['candidate']['mae'] == 1.
            and stats['speed']['far']['SUPPORTED_HOLD']['NEAR']['mae'] == 3., 'paired speed summary changed')
    with TemporaryDirectory() as directory:
        path = Path(directory)/'test.json.gz'
        write(path, stats)
        require(read(path) == stats, 'strict JSON round trip changed')
    print('C86 self-check passed: strict far boundary, original support, same-side speed, masks and paired JSON.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = ('runs', 'c81', 'science', 'fallback', 'lineage', 'lineage_report', 'output')
    for name in inputs:
        parser.add_argument('--'+name.replace('_', '-'), type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not args.time_policy or not all(getattr(args, name) for name in inputs):
        parser.error('all frozen input paths, --output and --time-policy are required')
    else:
        run(*(getattr(args, name).resolve() for name in inputs))
