"""C83: validate the frozen near and far speed rules on the same held scenes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.data.archive import MTADArchive
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.paper_fusion_metrics import COUNTS, aggregate_paper_fusion
from tools.analysis.far_supported_speed_readout import build_readouts as apply_far
from tools.analysis.motion_fallback_diagnostic import INVARIANTS, distribution, freeze_scene, require
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import SPLIT, readouts
from tools.analysis.near_endpoint_readout import build_readout as apply_near, speed_statistics
from tools.analysis.offline_endpoint_lineage import endpoint_arrays, freeze_endpoints
from tools.analysis.same_source_swap_pilot import MAX_AS, group_members, peak_rss_bytes, read, write


METHODS = ('B', 'NEAR', 'FAR', 'NEAR_FAR')
CONTRASTS = (('B', 'NEAR'), ('B', 'FAR'), ('B', 'NEAR_FAR'), ('FAR', 'NEAR_FAR'), ('NEAR', 'NEAR_FAR'))
REGIONS = ('all', 'near', 'far', 'near_before', 'near_after', 'far_before', 'far_after',
           'near_pure', 'near_mixed', 'far_pure', 'far_mixed', 'near_same_source', 'near_other_source')


def factorial(near, far, support, lineage):
    """Cross the two disjoint speed replacements without changing the readout base."""
    base = near['B']
    combined = far['SUPPORTED_HOLD']
    near_mask, far_mask = near['switch'], far['far']
    require(not (near_mask & far_mask).any(), 'near and far replacements must be disjoint')
    far_only = base.copy()
    far_only[..., 2][far_mask] = combined[..., 2][far_mask]
    states = dict(B=base, NEAR=near['NEAR'], FAR=far_only, NEAR_FAR=combined)
    for name, mask in (('NEAR', near_mask), ('FAR', far_mask), ('NEAR_FAR', near_mask | far_mask)):
        np.testing.assert_array_equal(states[name][..., [0, 1, 3]], base[..., [0, 1, 3]])
        np.testing.assert_array_equal(states[name][~mask], base[~mask])
    np.testing.assert_array_equal(combined[near_mask], states['NEAR'][near_mask])
    np.testing.assert_array_equal(combined[far_mask], far_only[far_mask])
    before = near['before_outside']
    labels = dict(near=near_mask, far=far_mask, near_before=near_mask & before, near_after=near_mask & ~before,
                  far_before=far['before'], far_after=far['after'],
                  near_same_source=near_mask & lineage['same_source'],
                  near_other_source=near_mask & ~lineage['same_source'])
    return dict(states, **labels, membership=support['membership'], nodes=support['nodes'],
                query=support['query'], support=support['support'], nearest=support['nearest'],
                donor=far['donor'], donor_age_seconds=far['donor_age_seconds'],
                nearest_event_seconds=far['nearest_event_seconds'])


def prepare(scene):
    nodes, sources, support, nearest, members, native, baseline = (
        scene[k] for k in ('nodes', 'sources', 'support', 'nearest', 'membership', 'native', 'baseline'))
    query, times, _, offsets = native
    outputs, masks = readouts(native, members, baseline, support)
    c74 = dict(outputs, **masks, membership=members)
    native_motion = scene['native_motion']
    frozen = freeze_scene((nodes, sources, support, nearest, members, native, baseline),
                          c74, native_motion[..., 2:])
    endpoints = endpoint_arrays(times, scene['values'], offsets)
    lineage = freeze_endpoints(frozen, query, endpoints, sources, baseline, native_motion[..., 2:])
    near = apply_near(c74, frozen, lineage)
    far = apply_far(near, c74, frozen, lineage)
    return factorial(near, far, frozen, lineage)


def evaluate_scene(sid, component, artifact, ev):
    members = artifact['membership']
    groups = group_members(members)
    reports, arrays = evaluate_fusion_methods(
        {name: emit_group_outputs(artifact[name], [g.tolist() for g in groups], node_count=len(members))
         for name in METHODS}, prediction_speed_units={name: 'mps' for name in METHODS},
        node_identities=ev['identities'], target_ids=ev['target_ids'], target_states=ev['target_states'],
        target_valid=ev['target_valid'], support=artifact['support'], nearest_seconds=artifact['nearest'],
        target_speed_unit='mps', split=SPLIT)
    original = arrays['B']
    base_report = reports['B']
    paper = {name: reports[name]['paper_fusion_metrics'] for name in METHODS}
    coverage = {region: {feature: {key: int(base_report['full_target_metrics'][region][feature][key])
                                    for key in COVERAGE_KEYS}
                         for feature in ('position', 'speed', 'heading')}
                for region in ('all', 'observed', 'missing')}
    population = {key: int(base_report[key]) for key in POPULATION_KEYS}
    unmatched = base_report['unmatched_target_ids']
    for name in METHODS[1:]:
        values, report = arrays[name], reports[name]
        for key in (*INVARIANTS, 'aligned_truth'):
            np.testing.assert_array_equal(original[key], values[key])
        np.testing.assert_array_equal(original['aligned_prediction'][..., [0, 1, 3]],
                                      values['aligned_prediction'][..., [0, 1, 3]])
        require(all(report[key] == population[key] for key in POPULATION_KEYS)
                and report['unmatched_target_ids'] == unmatched, 'fixed target coverage or duplicates changed')
        for region, features in coverage.items():
            for feature, counts in features.items():
                require(all(report['full_target_metrics'][region][feature][key] == counts[key]
                            for key in COVERAGE_KEYS), 'reference or prediction population changed')
        for feature, attr in paper['B']['attributes'].items():
            candidate = paper[name]['attributes'][feature]
            require(all(candidate[key] == attr[key] for key in COUNTS) and candidate['unit'] == attr['unit'],
                    'paper metric population changed')
            if feature != 'speed':
                require(candidate == attr, 'fixed paper metric changed outside speed')
    nt = len(ev['target_ids'])
    valid = original['aligned_truth_valid'][:nt] & original['aligned_prediction_valid'][:nt]
    target, q = np.nonzero(valid[..., 2])
    g = original['aligned_group_indices'][target]
    require((g >= 0).all(), 'valid speed point needs matched group')
    truth = original['aligned_truth'][target, q, 2]
    errors = {name: arrays[name]['aligned_prediction'][target, q, 2]-truth for name in METHODS}
    group_pure = np.asarray([len(np.unique(ev['identities'][nodes])) == 1 for nodes in groups], bool)
    pure = group_pure[g]
    masks = {name: artifact[name][g, q] for name in
             ('near', 'far', 'near_before', 'near_after', 'far_before', 'far_after',
              'near_same_source', 'near_other_source')}
    masks.update(all=np.ones(len(g), bool), near_pure=masks['near'] & pure,
                 near_mixed=masks['near'] & ~pure, far_pure=masks['far'] & pure, far_mixed=masks['far'] & ~pure)
    far = masks['far']
    target_h, q_h = np.nonzero(valid[..., 3])
    heading = _wrap(original['aligned_prediction'][target_h, q_h, 3]-original['aligned_truth'][target_h, q_h, 3])
    vector = dict(errors=errors, masks=masks, heading=heading,
                  far_donor_age=artifact['donor_age_seconds'][g[far], q[far]],
                  far_nearest_event_age=artifact['nearest_event_seconds'][g[far], q[far]])
    blind = {key: int(artifact[key].sum()) for key in
             ('near', 'far', 'near_before', 'near_after', 'far_before', 'far_after',
              'near_same_source', 'near_other_source')}
    blind.update({f'changed_{name}': int(np.count_nonzero(artifact[name][..., 2] != artifact['B'][..., 2]))
                  for name in METHODS[1:]})
    row = dict(scene_id=sid, component_id=component, paper_fusion_metrics=paper, coverage=coverage,
               population=population, unmatched_target_count=len(unmatched), blind_group_queries=blind,
               reference_position_points=coverage['all']['position']['available_target_count'])
    return row, vector


def statistics(vectors):
    errors = {name: np.concatenate([v['errors'][name] for v in vectors]) for name in METHODS}
    speed = {}
    for region in REGIONS:
        mask = np.concatenate([v['masks'][region] for v in vectors])
        speed[region] = {}
        for base, candidate in CONTRASTS:
            stats = speed_statistics(errors[base][mask], errors[candidate][mask])
            for field, original in (('baseline', 'B'), ('candidate', 'NEAR'),
                                    ('baseline_absolute_error_sum', 'B_absolute_error_sum'),
                                    ('candidate_absolute_error_sum', 'NEAR_absolute_error_sum')):
                stats[field] = stats.pop(original)
            speed[region][f'{candidate}_vs_{base}'] = stats
    donor_age = np.concatenate([v['far_donor_age'] for v in vectors])
    nearest = np.concatenate([v['far_nearest_event_age'] for v in vectors])
    return dict(speed=speed, unchanged_circular_heading=_stats(np.concatenate([v['heading'] for v in vectors])),
                far_support_seconds=dict(donor_age=distribution(donor_age), nearest_event_age=distribution(nearest),
                                         donor_minus_nearest=distribution(donor_age-nearest)))


def pool(rows, vectors):
    return dict(scene_count=len(rows), component_count=len({r['component_id'] for r in rows}),
                reference_position_points=sum(row['reference_position_points'] for row in rows),
                unmatched_target_count=sum(row['unmatched_target_count'] for row in rows),
                population={key: sum(r['population'][key] for r in rows) for key in POPULATION_KEYS},
                blind_group_queries={key: sum(r['blind_group_queries'][key] for r in rows)
                                     for key in rows[0]['blind_group_queries']},
                coverage={region: {feature: {key: sum(r['coverage'][region][feature][key] for r in rows)
                                             for key in COVERAGE_KEYS}
                                   for feature in ('position', 'speed', 'heading')}
                          for region in ('all', 'observed', 'missing')},
                paper_fusion_metrics={name: aggregate_paper_fusion(
                    [r['paper_fusion_metrics'][name] for r in rows]) for name in METHODS},
                **statistics(vectors))


def gates(all_scenes, components, leave):
    def effects(result, region, name):
        return result['speed'][region][name]

    def passes(result, region, name, key, strict=False):
        value = effects(result, region, name)[key]
        return value is not None and (value < 0 if strict else value <= 0)

    specs = {'NEAR_vs_B': ('near', (('all', 'delta_mae', True), ('all', 'delta_p99', False)),
                           ('all', 'delta_mae', False)),
             'FAR_vs_B': ('far', (('far', 'delta_mae', True), ('far', 'delta_p99', False),
                                 ('all', 'delta_mae', True), ('all', 'delta_p99', False)),
                          ('far', 'delta_mae', True)),
             'NEAR_FAR_vs_B': ('all', (('all', 'delta_mae', True), ('all', 'delta_p99', False)),
                               ('all', 'delta_mae', True))}
    verdicts = {}
    for name, (region, checks, omit) in specs.items():
        region_count = effects(all_scenes, region, name)['count']
        details = {f'{r}_{k}': passes(all_scenes, r, name, k, strict) for r, k, strict in checks}
        omit_check = passes(leave, omit[0], name, omit[1], omit[2])
        better_components = sum(passes(x, region, name, 'delta_mae', True) for x in components.values())
        passed = bool(region_count and all(details.values()) and omit_check and better_components >= 2)
        verdicts[name] = dict(passed=passed, evaluated_points=region_count, primary_checks=details,
                              remove_largest_passed=omit_check, improving_components=better_components)
    return verdicts


def run(runs, output):
    from tools.analysis.held_fusion_population import (
        load_population, load_scene, read_evaluation, validate_measurement_archive,
    )

    require(output.parent == runs and not output.exists(), 'new independent output under frozen runs required')
    output.mkdir()
    private = output/'private'
    private.mkdir()
    start = time.monotonic()
    state = dict(status='RUNNING', split=SPLIT, stages=[], argv=sys.argv, runs=str(runs), output=str(output),
                 started_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0,
                 raw_measurement_scenes=0, raw_label_scenes=0, raw_reference_scenes=0, test_opened=False,
                 affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components, archive_paths = load_population(runs)
        checkpoint(phase='freeze_predictions', scene_ids=ids, archive_paths=archive_paths)
        nodes_total = group_total = event_total = 0
        with MTADArchive(archive_paths) as archive:
            validate_measurement_archive(runs, archive)
            for index, sid in enumerate(ids, start=1):
                scene = load_scene(runs, sid, archive)
                artifact = prepare(scene)
                np.savez_compressed(private/f'{sid}.npz', **artifact)
                nodes_total += len(artifact['nodes'])
                group_total += len(artifact['B'])
                event_total += len(scene['native'][1])
                checkpoint(raw_measurement_scenes=index, frozen_prediction_scenes=index)
        require((nodes_total, group_total, event_total) == (1224, 493, 315446),
                'blind V48 preflight population changed')
        state['stages'].append('all_held_factorial_outputs_and_input_support_frozen_before_reference')
        checkpoint(phase='evaluate', node_count=nodes_total, group_count=group_total, event_count=event_total)
        rows, vectors = [], []
        for sid in ids:
            with np.load(private/f'{sid}.npz', allow_pickle=False) as saved:
                artifact = {key: saved[key] for key in saved.files}
            ev = read_evaluation(runs, sid, artifact['nodes'], artifact['query'])
            row, vector = evaluate_scene(sid, components[sid], artifact, ev)
            rows.append(row)
            vectors.append(vector)
            checkpoint(completed_scenes=len(rows))

        def subset(indices):
            return pool([rows[i] for i in indices], [vectors[i] for i in indices])

        all_scenes = pool(rows, vectors)
        by_component = {c: subset([i for i, r in enumerate(rows) if r['component_id'] == c])
                        for c in sorted(set(components.values()))}
        largest = min(by_component, key=lambda c: (-by_component[c]['reference_position_points'], c))
        leave = subset([i for i, r in enumerate(rows) if r['component_id'] != largest])
        write(output/'scenes.json.gz', [dict(row, **statistics([vector]))
                                       for row, vector in zip(rows, vectors, strict=True)])
        write(output/'summary.json', dict(all_components=all_scenes, by_component=by_component,
            remove_largest_component=dict(excluded_component=largest, result=leave),
            gate_results=gates(all_scenes, by_component, leave),
            invariants=dict(position_exact=True, heading_exact=True, groups_exact=True, masks_exact=True,
                            coverage_exact=True, correspondence_exact=True, duplicates_exact=True,
                            near_and_far_masks_disjoint=True, original_observation_support_exact=True),
            definitions=dict(population='held V48 for readout development only; historical V300 evaluation used these scenes',
                             position='fixed original 147 WS_MEAN',
                             speed='point-pool prediction minus reference in m/s; paper target/scene equal uses kn',
                             largest='component with most available reference position points, then lexical id',
                             near='C81 nearest original endpoint conditional replacement',
                             far='C86 same-side nearest C74-supported Q speed hold')))
        state['stages'].append('fixed_reference_and_four_attribute_evaluation_complete')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


def self_check():
    from tempfile import TemporaryDirectory

    shape = (1, 6)
    base = np.zeros((*shape, 4))
    base[..., :2] = (1., 2.)
    base[..., 2] = 7.
    base[..., 3] = 359.
    near_mask = np.zeros(shape, bool)
    far_mask = near_mask.copy()
    near_mask[0, 1] = far_mask[0, 5] = True
    near_value = base.copy()
    near_value[0, 1, 2] = 2.
    both = near_value.copy()
    both[0, 5, 2] = 3.
    query = np.arange(6) * 3.
    support = dict(membership=np.array([0]), nodes=np.array(['x']), query=query, support=np.ones(shape, bool),
                   nearest=np.zeros(shape))
    near = dict(B=base, NEAR=near_value, switch=near_mask, before_outside=near_mask)
    far = dict(SUPPORTED_HOLD=both, far=far_mask, before=np.zeros(shape, bool), after=far_mask,
               donor=np.zeros(shape, int), donor_age_seconds=np.zeros(shape), nearest_event_seconds=np.zeros(shape))
    art = factorial(near, far, support, dict(same_source=near_mask))
    ids = np.array(['x'])
    truth = base.copy()
    truth[..., 2] = 2.
    truth[..., 3] = 1.
    valid = np.ones_like(truth, bool)
    valid[0, 0, 2] = False
    row, vector = evaluate_scene('s', 'c', art, dict(identities=ids, target_ids=ids, target_states=truth,
                                                    target_valid=valid))
    result = pool([row], [vector])
    np.testing.assert_allclose([result['speed']['near']['NEAR_vs_B']['delta_mae'],
                                result['speed']['far']['FAR_vs_B']['delta_mae'],
                                result['speed']['all']['NEAR_FAR_vs_B']['delta_mae'],
                                result['unchanged_circular_heading']['mae']], [-5., -4., -1.8, 2.],
                               rtol=0., atol=1e-12, err_msg='factorial paired effects changed')
    with TemporaryDirectory() as directory:
        path = Path(directory)/'test.json.gz'
        write(path, result)
        require(read(path) == result, 'strict JSON round trip changed')
    print('C83 self-check passed: disjoint replacements, same masks, explicit units, paired effects and JSON.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not all((args.runs, args.output, args.time_policy)):
        parser.error('--runs, --output and --time-policy are required')
    else:
        run(args.runs.resolve(), args.output.resolve())
