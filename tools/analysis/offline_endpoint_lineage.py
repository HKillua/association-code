"""C79: trace frozen V32 endpoint speed fields and adjacent-position secants."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.legacy_dataset import validate_archive
from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.physical import EARTH_RADIUS_METRES
from tools.analysis.motion_fallback_diagnostic import (
    INVARIANTS, ORIGINAL_COUNTS, check_error, distance_bin_masks, distribution,
    freeze_scene, frozen_arrays, read_legacy_motion, require,
)
from tools.analysis.motion_readout_evaluation import POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import METHODS, SPLIT, load_population, load_scene, read_reference
from tools.analysis.same_source_swap_pilot import MAX_AS, PARENTS, group_members, peak_rss_bytes, read, write


FIELDS = ('time', 'lat', 'lon', 'vel', 'cou')
PROXIES = ('first_field', 'first_secant', 'nearest_field', 'nearest_secant')
RAW_BINDING_KEYS = ('canonical_rows', 'raw_row_indices', 'multiplicities')
FROZEN_STAGE = 'all_lineage_frozen_before_labels_references_and_prior_errors'
PUBLICATION_FAILURE = 'TypeError: Object of type int64 is not JSON serializable'


def speed_between(left, right, elapsed):
    """Spherical adjacent-position displacement divided by original seconds."""
    require(np.isfinite(elapsed).all() and (elapsed > 0.).all(), 'positive original intervals required')
    lat0, lat1 = np.deg2rad(left[..., 0]), np.deg2rad(right[..., 0])
    lon = np.deg2rad((right[..., 1]-left[..., 1]+180.) % 360.-180.)
    h = np.clip(np.sin((lat1-lat0)/2.)**2 + np.cos(lat0)*np.cos(lat1)*np.sin(lon/2.)**2, 0., 1.)
    return (2.*EARTH_RADIUS_METRES*np.arctan2(np.sqrt(h), np.sqrt(1.-h))) / elapsed


def bind_measurements(sid, raw, nodes, sources, times, values, offsets):
    """Bind distinct canonical first/last-two rows to the measurement CSV only."""
    indices = np.unique(np.r_[offsets[:-1], offsets[:-1]+1, offsets[1:]-2, offsets[1:]-1])
    expected = {}
    for row in indices:
        node = int(np.searchsorted(offsets[1:], row, side='right'))
        key = (str(nodes[node]), float(times[row]))
        require(key not in expected, 'non-unique canonical endpoint row')
        require(str(nodes[node]).startswith(f'{sid}:{sources[node]}:'), 'original node/source key mismatch')
        expected[key] = int(row)
    first_index, counts, raw_courses = {}, Counter(), {}
    keep_nodes = {key[0] for key in expected}
    for raw_index, raw_row in enumerate(raw):
        node = f"{sid}:{raw_row['source']}:{raw_row['batch']}"
        if node not in keep_nodes:
            continue
        key = (node, float(raw_row['time']))
        if key not in expected:
            continue
        observed = np.asarray([float(raw_row[field]) for field in FIELDS])
        row = expected[key]
        require(np.isfinite(observed).all() and 0. <= observed[4] <= 360.,
                'invalid raw row at a retained canonical endpoint')
        np.testing.assert_array_equal(observed[:4], np.r_[times[row], values[row, :3]],
                                      err_msg='raw time/position/speed differs from canonical endpoint')
        require(observed[4] % 360. == values[row, 3],
                'raw course differs beyond canonical modulo')
        if key in raw_courses:
            require(observed[4] == raw_courses[key], 'conflicting raw duplicates at canonical endpoint')
        else:
            raw_courses[key] = observed[4]
            first_index[key] = raw_index
        counts[key] += 1
    require(set(counts) == set(expected), 'retained canonical endpoint missing in the measurement CSV')
    raw_row_indices = np.asarray([first_index[key] for key in expected], np.int64)
    multiplicities = np.asarray([counts[key] for key in expected], np.int64)
    return dict(canonical_rows=indices, raw_row_indices=raw_row_indices, multiplicities=multiplicities,
                course_360_count=sum(bool(value == 360.) for value in raw_courses.values()),
                measurement_row_count=len(raw))


def endpoint_arrays(times, values, offsets):
    start, stop = offsets[:-1], offsets[1:]-1
    left = np.column_stack((start, stop-1))
    right = left+1
    event = np.column_stack((start, stop))
    elapsed = times[right]-times[left]
    start_kn, end_kn = values[start, 2], values[stop, 2]
    # The existing interpolation kernel holds the final value as left+(right-left).
    prior_kn = values[stop-1, 2]
    held_kn = np.column_stack((start_kn, prior_kn+(end_kn-prior_kn)))
    return dict(time=times[event], index=event, field=held_kn*KNOT_TO_MPS,
                canonical_field=values[event, 2]*KNOT_TO_MPS,
                secant=speed_between(values[left, :2], values[right, :2], elapsed),
                elapsed=elapsed)


def freeze_endpoints(frozen, query, endpoints, sources, baseline, native_motion):
    """Freeze endpoint choice without identities, reference or any score selection."""
    outside = np.isin(frozen['cause'], (1, 2, 4, 5))
    require(not (outside & frozen['visible']).any(), 'outside group span cannot have member support')
    shape = outside.shape
    result = dict(scope=outside, first_member=np.full(shape, -1, np.int64),
                  nearest_member=np.full(shape, -1, np.int64),
                  first_side=np.full(shape, -1, np.int8),
                  nearest_side=np.full(shape, -1, np.int8),
                  same_source=np.zeros(shape, bool))
    for prefix in ('first', 'nearest'):
        for key in ('field', 'canonical_field', 'secant', 'elapsed', 'age'):
            result[f'{prefix}_{key}'] = np.full(shape, np.nan)
    for group_id, members in enumerate(group_members(frozen['membership'])):
        q = np.flatnonzero(outside[group_id])
        if not len(q):
            continue
        sel = (group_id, q)
        times = endpoints['time'][members]
        age = np.abs(query[q, None]-times.reshape(-1)[None, :])
        choice = np.argmin(age, axis=1)  # ties: earlier original member, then start before end
        first_member, nearest_members = members[0], members[choice//2]
        first_sides, nearest_sides = frozen['anchor_side'][sel].astype(np.int8)-1, choice % 2
        require((first_sides >= 0).all() and np.array_equal(first_sides, nearest_sides),
                'both chosen endpoints must border the same external group side')
        result['same_source'][sel] = sources[first_member] == sources[nearest_members]
        for prefix, node, side in (('first', first_member, first_sides),
                                   ('nearest', nearest_members, nearest_sides)):
            result[f'{prefix}_member'][sel] = node
            result[f'{prefix}_side'][sel] = side
            result[f'{prefix}_age'][sel] = np.abs(query[q]-endpoints['time'][node, side])
            for key in ('field', 'canonical_field', 'secant', 'elapsed'):
                result[f'{prefix}_{key}'][sel] = endpoints[key][node, side]
            np.testing.assert_array_equal(result[f'{prefix}_field'][sel],
                                          native_motion[node, q, 0]*KNOT_TO_MPS,
                                          err_msg='held speed does not match the 127 motion cache')
        np.testing.assert_array_equal(result['first_field'][sel], baseline[group_id, q, 2],
                                      err_msg='first held field differs from unchanged C74 output')
        np.testing.assert_array_equal(result['first_age'][sel], frozen['anchor_endpoint_distance'][sel])
        np.testing.assert_array_equal(result['nearest_age'][sel], frozen['distance'][sel])
    return result


def evaluate(sid, component, frozen, saved, lineage, ev, prior_c74, prior_c77):
    groups = [members.tolist() for members in group_members(frozen['membership'])]
    methods = {name: emit_group_outputs(saved[name], groups, node_count=len(frozen['membership']))
               for name in METHODS}
    reports, artifacts = evaluate_fusion_methods(
        methods, prediction_speed_units={name: 'mps' for name in METHODS},
        node_identities=ev['identities'], target_ids=ev['target_ids'], target_states=ev['target_states'],
        target_valid=ev['target_valid'], support=frozen['support'], nearest_seconds=frozen['nearest'],
        target_speed_unit='mps', split=SPLIT)
    base = artifacts['B']
    n = len(ev['target_ids'])
    chosen = base['aligned_group_indices'][:n]
    matched = chosen >= 0
    active = np.zeros((n, len(frozen['query'])), bool)
    active[matched] = frozen['cause'][chosen[matched]] == 0
    require(prior_c74['scene_id'] == prior_c77['scene_id'] == sid and
            prior_c74['component_id'] == prior_c77['component_id'] == component,
            'prior public scene or component changed')
    for name in METHODS:
        a = artifacts[name]
        for key in INVARIANTS:
            np.testing.assert_array_equal(a[key], base[key])
        np.testing.assert_array_equal(a['aligned_prediction'][..., :2], base['aligned_prediction'][..., :2])
        expected = prior_c74['methods'][name]
        require(all(reports[name][key] == expected['population'][key] for key in POPULATION_KEYS)
                and len(reports[name]['unmatched_target_ids']) == expected['unmatched_target_count'],
                'C74 coverage, population or duplicates changed')
        for column, feature in ((2, 'speed'), (3, 'heading')):
            valid = a['aligned_prediction_valid'][:n, :, column] & a['aligned_truth_valid'][:n, :, column]
            error = a['aligned_prediction'][:n, :, column]-a['aligned_truth'][:n, :, column]
            if feature == 'heading':
                error = _wrap(error)
            check_error(error[valid], expected['errors']['all'][feature])
            check_error(error[valid & active], expected['errors']['active'][feature])
    speed_valid = base['aligned_prediction_valid'][:n, :, 2] & base['aligned_truth_valid'][:n, :, 2]
    target, q = np.nonzero(speed_valid)
    g = chosen[target]
    require((g >= 0).all(), 'valid original speed needs a matched target')
    scope = lineage['scope'][g, q]
    source_group_queries = int(lineage['scope'].sum())
    require(source_group_queries == sum(prior_c77['group_query_counts'][key] for key in
                ('before_span_only', 'after_span_only', 'before_span_and_insufficient',
                 'after_span_and_insufficient')), 'original C77 group-query population changed')
    require(int(scope.sum()) == prior_c77['errors']['speed']['inactive_no_member_Q_support']['count'],
            'original C77 fallback reference population changed')
    target, q, g = target[scope], q[scope], g[scope]
    query = frozen['query']
    q_step = (query[-1]-query[0])/(len(query)-1)
    vector = {key: value[g, q] for key, value in lineage.items() if key != 'scope'}
    vector.update(truth=base['aligned_truth'][target, q, 2],
                  span_only=np.isin(frozen['cause'][g, q], (1, 2)),
                  near_steps=vector['nearest_age']/q_step,
                  scene_number=np.full(len(g), int(sid), np.int64),
                  pure=np.asarray([len(np.unique(ev['identities'][group])) == 1 for group in groups])[g])
    np.testing.assert_array_equal(vector['first_field'], base['aligned_prediction'][target, q, 2])
    old = vector['first_field']-vector['truth']
    for region, mask in (('inactive_no_member_Q_support', np.ones(len(old), bool)),
                         ('inactive_span_only_near_pure',
                          vector['span_only'] & vector['pure'] & (vector['near_steps'] <= 2.))):
        check_error(old[mask], prior_c77['errors']['speed'][region])
    position_points = int(reports['B']['full_target_metrics']['all']['position']['available_target_count'])
    require(position_points == prior_c77['reference_position_points']
            and all(reports['B'][key] == prior_c77['population'][key] for key in POPULATION_KEYS),
            'C77 reference or population changed')
    return dict(scene_id=sid, component_id=component, reference_position_points=position_points,
                valid_speed_points=int(speed_valid.sum()), population=prior_c77['population'],
                unmatched_target_count=prior_c77['unmatched_target_count'],
                legacy_fallback_group_queries=source_group_queries), vector


def regions(v):
    near = v['near_steps'] <= 2.
    pure = v['pure']
    qualified = v['span_only']
    older = v['first_age'] > v['nearest_age']
    changed = v['first_member'] != v['nearest_member']
    yield 'all_eligible', np.ones(len(v['truth']), bool)
    yield 'before', v['first_side'] == 0
    yield 'after', v['first_side'] == 1
    yield 'qualified_span_only', qualified
    yield 'qualified_span_only_near', qualified & near
    yield 'near_pure', qualified & near & pure
    yield 'near_pure_before', qualified & near & pure & (v['first_side'] == 0)
    yield 'near_pure_after', qualified & near & pure & (v['first_side'] == 1)
    yield 'near_pure_first_older', qualified & near & pure & older
    yield 'near_pure_older_same_source', qualified & near & pure & older & v['same_source']
    yield 'near_pure_older_different_source', qualified & near & pure & older & ~v['same_source']
    yield 'pure', pure
    yield 'mixed', ~pure
    yield 'first_older', older
    yield 'changed_same_source', changed & v['same_source']
    yield 'changed_different_source', changed & ~v['same_source']
    yield 'first_is_nearest', ~changed
    for label, selected in zip(('at_most_1', 'over_1_to_2', 'over_2_to_4', 'over_4'),
                               distance_bin_masks(v['near_steps']), strict=True):
        yield 'nearest_steps_'+label, selected


def paired_error(first, second):
    delta = np.abs(second)-np.abs(first)
    return dict(mean_absolute_error_delta=float(delta.mean()) if len(delta) else None,
                median_absolute_error_delta=float(np.median(delta)) if len(delta) else None,
                improved_points=int((delta < -1e-9).sum()), worsened_points=int((delta > 1e-9).sum()),
                tied_points=int((np.abs(delta) <= 1e-9).sum()))


def field_secant(v, mask, member):
    field = v[member+'_canonical_field'][mask]
    secant = v[member+'_secant'][mask]
    positive = secant > 0.
    close = np.abs(field[positive]-secant[positive]) <= .01*secant[positive]
    held_delta = v[member+'_field'][mask]-field
    return dict(field_minus_adjacent_secant=_stats(field-secant),
                absolute_delta_mps=distribution(np.abs(field-secant)),
                positive_secants=int(positive.sum()), positive_secants_within_1_percent=int(close.sum()),
                zero_secants=int((~positive).sum()), zero_secants_zero_field=int(((~positive) & (field == 0.)).sum()),
                nonzero_hold_vs_canonical=int(np.count_nonzero(held_delta)),
                max_absolute_hold_vs_canonical_mps=float(np.max(np.abs(held_delta))) if len(held_delta) else None)


def unique_endpoint_rows(v, mask, member):
    rows = np.flatnonzero(mask)
    keys = np.column_stack((v['scene_number'][rows], v[member+'_member'][rows], v[member+'_side'][rows]))
    _, indices = np.unique(keys, axis=0, return_index=True)
    return rows[indices]


def summarize_vectors(v):
    result = {}
    for name, mask in regions(v):
        ref = v['truth'][mask]
        errors = {key: v[key][mask]-ref for key in PROXIES}
        result[name] = dict(
            count=len(ref),
            reference_speed_mps=distribution(ref),
            proxy_speed_mps={key: distribution(v[key][mask]) for key in PROXIES},
            proxy_reference_errors_mps={key: _stats(errors[key]) for key in PROXIES},
            paired=dict(first_secant_vs_first_field=paired_error(errors['first_field'], errors['first_secant']),
                        nearest_field_vs_first_field=paired_error(errors['first_field'], errors['nearest_field']),
                        nearest_secant_vs_first_secant=paired_error(errors['first_secant'], errors['nearest_secant']),
                        nearest_secant_vs_nearest_field=paired_error(errors['nearest_field'], errors['nearest_secant'])),
            first_field_to_secant=field_secant(v, mask, 'first'),
            nearest_field_to_secant=field_secant(v, mask, 'nearest'),
            distinct_first_endpoint_field_to_secant=field_secant(
                v, unique_endpoint_rows(v, mask, 'first'), 'first'),
            distinct_nearest_endpoint_field_to_secant=field_secant(
                v, unique_endpoint_rows(v, mask, 'nearest'), 'nearest'),
            first_age_seconds=distribution(v['first_age'][mask]),
            nearest_age_seconds=distribution(v['nearest_age'][mask]),
            age_difference_seconds=distribution((v['first_age']-v['nearest_age'])[mask]),
            first_interval_seconds=distribution(v['first_elapsed'][mask]),
            nearest_interval_seconds=distribution(v['nearest_elapsed'][mask]),
            changed_member=int((v['first_member'][mask] != v['nearest_member'][mask]).sum()),
            same_source=int(v['same_source'][mask].sum()))
    return result


def pool(rows, vectors):
    require(bool(rows) and len(rows) == len(vectors), 'nonempty scene rows and matching error vectors required')
    def summed(key):
        result = {}
        for field in rows[0][key]:
            values = [row[key][field] for row in rows]
            require(all(value is None for value in values) or all(value is not None for value in values),
                    'availability of a report field must match all scenes')
            result[field] = None if values[0] is None else sum(int(value) for value in values)
        return result
    return dict(scene_count=len(rows), component_count=len({row['component_id'] for row in rows}),
                population=summed('population'), raw_endpoint_binding=summed('raw_endpoint_binding'),
                **{key: sum(row[key] for row in rows) for key in
                   ('reference_position_points', 'valid_speed_points', 'unmatched_target_count',
                    'legacy_fallback_group_queries')},
                regions=summarize_vectors({key: np.concatenate([v[key] for v in vectors]) for key in vectors[0]}))


def checked_paths(runs, science, fallback, ids, *, need_archive):
    parent = runs/PARENTS['native']
    marker, original = read(parent/'native-cache.json'), read(parent/'run.json')
    paths = [str(Path(p).resolve()) for p in marker['archive_paths']]
    require(paths and len(set(paths)) == len(paths) and
            paths == [str(Path(p).resolve()) for p in original['archive_paths']],
            'archive parts must be the same ordered files recorded by 127')
    if need_archive:
        require(all(Path(p).is_file() for p in paths), 'original 127 archive parts must be regular files')
    for run in (science, fallback):
        metadata = read(run/'run.json')
        require(metadata['status'] == 'COMPLETED' and metadata['exit_code'] == 0 and
                metadata['split'] == SPLIT and metadata['scene_ids'] == ids and
                metadata['completed_scenes'] == len(ids) and
                tuple(metadata[key] for key in ('node_count', 'group_count', 'event_count')) == ORIGINAL_COUNTS,
                'complete original V32 C74 and C77 required')
    c77 = read(fallback/'run.json')
    require(Path(c77['runs']).resolve() == runs and Path(c77['science']).resolve() == science
            and Path(c77['output']).resolve() == fallback,
            'C77 does not bind the supplied original input paths')
    return paths


def check_failed_publication(attempt, runs, science, fallback, ids, paths):
    """Accept only the stage-complete attempt interrupted by this serialization failure."""
    original = read(attempt/'run.json')
    require(original['status'] == 'FAILED' and original['exit_code'] == 1
            and original['phase'] == 'evaluate_fixed_correspondence' and original['error'] == PUBLICATION_FAILURE
            and original['stages'] == [FROZEN_STAGE] and original['split'] == SPLIT
            and original['scene_ids'] == ids and original['frozen_lineage_scenes'] == len(ids)
            and original['raw_measurement_members'] == len(ids) and original['raw_label_members'] == 0
            and original['raw_reference_members'] == 0 and original['optimizer_updates'] == 0
            and original['predictions_generated'] == 0 and original['test_opened'] is False,
            'repair requires the failed serialization attempt with complete V32 measurement lineage')
    require(all(Path(original[key]).resolve() == path for key, path in
                (('runs', runs), ('science', science), ('fallback', fallback), ('output', attempt)))
            and original['archive_paths'] == paths, 'failed attempt input paths do not match')
    require({p.name for p in (attempt/'private').glob('lineage-*.npz')} ==
            {f'lineage-{sid}.npz' for sid in ids}, 'failed attempt does not contain every original frozen scene')


def freeze_measurements(runs, science, fallback, private, ids, paths, checkpoint):
    manifest = read(Path(read(runs/PARENTS['base']/'run.json')['manifest']))
    private.mkdir()
    raw_counts = []
    with MTADArchive(paths) as archive:
        validate_archive(archive, manifest)
        for sid in ids:
            scene = load_scene(runs, sid)
            nodes, sources, _, _, _, native, baseline = scene
            query, times, positions, offsets = native
            saved = frozen_arrays(science/'private'/f'{sid}.npz')
            legacy_motion = read_legacy_motion(runs, sid, scene)
            frozen = freeze_scene(scene, saved, legacy_motion)
            with np.load(fallback/'private'/f'support-{sid}.npz', allow_pickle=False) as prior:
                require(set(prior.files) == set(frozen), 'frozen C77 support schema changed')
                for key, value in frozen.items():
                    np.testing.assert_array_equal(value, prior[key], err_msg='C77 support field changed: '+key)
            cache_path = runs/PARENTS['native']/'private'/'native-events'/SPLIT/f'scene-{sid}.npz'
            with np.load(cache_path, allow_pickle=False) as cache:
                values, valid = cache['event_values'], cache['event_valid']
            np.testing.assert_array_equal(values[:, :2], positions)
            require(valid.all() and np.isfinite(values).all() and (values[:, 2] >= 0.).all(),
                    'all frozen native attributes must be legal')
            raw = archive._rows(sid, 'measurements')
            binding = bind_measurements(sid, raw, nodes, sources, times, values, offsets)
            lineage = freeze_endpoints(frozen, query, endpoint_arrays(times, values, offsets),
                                       sources, baseline, legacy_motion)
            np.savez_compressed(private/f'lineage-{sid}.npz', **lineage,
                                **{key: binding[key] for key in RAW_BINDING_KEYS})
            raw_counts.append(dict(canonical_course_360=binding['course_360_count'],
                                   scene_measurement_rows=binding['measurement_row_count']))
            checkpoint(frozen_lineage_scenes=len(raw_counts))
        require([(row['scene_id'], row['kind']) for row in archive.members_read] ==
                [(sid, 'measurements') for sid in ids], 'unexpected raw archive member read')
    return raw_counts


def load_lineage(path, frozen, extra_counts):
    with np.load(path, allow_pickle=False) as saved:
        lineage = {key: saved[key] for key in saved.files if key not in RAW_BINDING_KEYS}
        indices, raw_rows, counts = (saved[key] for key in RAW_BINDING_KEYS)
    require(indices.ndim == 1 and indices.dtype.kind in 'iu' and len(indices) > 0
            and (np.diff(indices) > 0).all() and indices[0] >= 0 and indices[-1] < frozen['event_count']
            and raw_rows.shape == counts.shape == indices.shape and raw_rows.dtype.kind in 'iu'
            and (raw_rows >= 0).all() and len(np.unique(raw_rows)) == len(raw_rows)
            and counts.dtype.kind in 'iu' and (counts > 0).all(),
            'invalid frozen endpoint row indices or raw multiplicities')
    outside = np.isin(frozen['cause'], (1, 2, 4, 5))
    require(lineage['scope'].dtype == bool and all(value.shape == outside.shape for value in lineage.values()),
            'frozen lineage query shape changed')
    np.testing.assert_array_equal(lineage['scope'], outside)
    np.testing.assert_array_equal(lineage['first_age'][outside], frozen['anchor_endpoint_distance'][outside])
    np.testing.assert_array_equal(lineage['nearest_age'][outside], frozen['distance'][outside])
    return lineage, dict(canonical_endpoint_rows=len(indices), matched_raw_rows=int(counts.sum()),
                         exact_duplicate_matches=int((counts-1).sum()), **extra_counts)


def run(runs, science, fallback, output, frozen_attempt=None):
    parent = runs/PARENTS['native']
    protected = (science, fallback, parent) if frozen_attempt is None else (science, fallback, parent, frozen_attempt)
    require(output != runs and all(output != run and run not in output.parents and output not in run.parents
                                   for run in protected), 'new output must not overlap frozen inputs')
    require(output.parent.is_dir(), 'existing parent directory required for new run')
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(status='RUNNING', split=SPLIT, stages=[], started_at=datetime.now(timezone.utc).isoformat(),
                  runs=str(runs), science=str(science), fallback=str(fallback), output=str(output),
                  optimizer_updates=0, predictions_generated=0, test_opened=False, argv=sys.argv,
                  frozen_attempt=str(frozen_attempt) if frozen_attempt is not None else None,
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None)

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', record)

    try:
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        paths = checked_paths(runs, science, fallback, ids, need_archive=frozen_attempt is None)
        checkpoint(phase='bind_endpoint_lineage', scene_ids=ids, archive_paths=paths)
        if frozen_attempt is None:
            private = output/'private'
            checkpoint(phase='freeze_raw_endpoint_lineage')
            raw_counts = freeze_measurements(runs, science, fallback, private, ids, paths, checkpoint)
            record['stages'].append(FROZEN_STAGE)
        else:
            check_failed_publication(frozen_attempt, runs, science, fallback, ids, paths)
            private = frozen_attempt/'private'
            raw_counts = [dict(canonical_course_360=None, scene_measurement_rows=None) for _ in ids]
            record['stages'].append('failed_publication_lineage_reused_without_rereading_measurements')
        checkpoint(phase='evaluate_fixed_correspondence', frozen_lineage_scenes=len(ids),
                   raw_measurement_members=len(ids) if frozen_attempt is None else 0,
                   reused_raw_measurement_members=len(ids) if frozen_attempt is not None else 0,
                   raw_label_members=0, raw_reference_members=0)
        c74_rows, c77_rows = [read(root/'scenes.json.gz') for root in (science, fallback)]
        require([row['scene_id'] for row in c74_rows] == [row['scene_id'] for row in c77_rows] == ids,
                'original public rows and scene order changed')
        rows, vectors = [], []
        for sid, prior_c74, prior_c77, extra_counts in zip(ids, c74_rows, c77_rows, raw_counts, strict=True):
            with np.load(fallback/'private'/f'support-{sid}.npz', allow_pickle=False) as saved:
                frozen = {key: saved[key] for key in saved.files}
            lineage, binding = load_lineage(private/f'lineage-{sid}.npz', frozen, extra_counts)
            ev = read_reference(runs, sid, frozen['nodes'], frozen['query'])
            row, vec = evaluate(sid, components[sid], frozen, frozen_arrays(science/'private'/f'{sid}.npz'),
                                lineage, ev, prior_c74, prior_c77)
            rows.append(dict(row, raw_endpoint_binding=binding))
            vectors.append(vec)
        largest = min(set(components.values()), key=lambda component:
                      (-sum(row['reference_position_points'] for row in rows if row['component_id'] == component),
                       component))
        require(all(read(run/'summary.json')['remove_largest_component']['excluded_component'] == largest
                    for run in (science, fallback)), 'largest original C74/C77 component changed')

        def subset(indices):
            return pool([rows[i] for i in indices], [vectors[i] for i in indices])

        result = pool(rows, vectors)
        require(result['valid_speed_points'] == 71090 and result['regions']['all_eligible']['count'] == 1677
                and result['regions']['near_pure']['count'] == 156
                and result['regions']['near_pure_first_older']['count'] == 48,
                'frozen C77 all/near/older speed populations changed')
        temporary = output/'scenes.tmp.json.gz'
        write(temporary, [dict(row, regions=summarize_vectors(v)) for row, v in zip(rows, vectors, strict=True)])
        temporary.replace(output/'scenes.json.gz')
        write(output/'summary.json', dict(all_components=result,
            by_component={component: subset([i for i, row in enumerate(rows) if row['component_id'] == component])
                          for component in sorted(set(components.values()))},
            remove_largest_component=dict(excluded_component=largest,
                result=subset([i for i, row in enumerate(rows) if row['component_id'] != largest])),
            definitions=dict(scope='C77 group-span-external inactive queries with no member support',
                comparison='diagnostic scalar proxies on fixed reference rows; emitted C74 A/B/C unchanged',
                endpoints='earliest or latest of original member events, closest in actual seconds with original member then endpoint ties',
                raw_binding='only original V32 measurement members; exact selected time, lat, lon, vel and canonical course modulo',
                repair_missing_raw_fields='canonical_course_360 and scene_measurement_rows were not saved '
                                          'in original private lineage and are null in publication repair',
                source_field_speed_units='raw field interpreted as kn and converted once under frozen Level1 policy',
                secant='adjacent-position spherical displacement over actual seconds; no propagation',
                distinct_endpoint_unit='same scene, original member and endpoint count once within each stratum',
                error='proxy minus original reference speed in m/s',
                near_pure='qualified span-external pure matched group, nearest event at most two scene Q steps',
                intervals='before uses first→second, after uses penultimate→last',
                target_context='original development V32 and C77 correspondence; repeated queries are not independent')))
        record['stages'].append('frozen_c74_c77_reference_and_endpoint_diagnostics_complete')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, completed_scenes=len(rows),
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


def self_check():
    query = np.arange(64, dtype=float)
    times = np.array([2., 3., 4., 1., 3., 4., 1., 3., 6.])
    offsets = np.arange(0, 10, 3)
    values = np.array([[0., 179.999, 10., 0.], [0., -179.999, 11., 2.], [0., -179.998, 12., 3.],
                       [1., 0., 20., 3.], [1., 0., 21., 4.], [1., .001, 22., 5.],
                       [2., 0., 30., 2.], [2., .001, 31., 3.], [2., .002, 32., 4.]])
    nodes, sources = np.array(['9:9001:1', '9:9002:2', '9:9002:3']), np.array(['9001', '9002', '9002'])
    raw = [dict(source=node.split(':')[1], batch=node.split(':')[2],
                **dict(zip(FIELDS, map(str, np.r_[times[row], values[row]]), strict=True)))
           for node, start, stop in zip(nodes, offsets[:-1], offsets[1:], strict=True)
           for row in range(start, stop)]
    raw[0]['cou'] = '360'
    raw.append(dict(raw[0]))
    binding = bind_measurements('9', raw, nodes, sources, times, values, offsets)
    require(len(binding['canonical_rows']) == 9 and binding['multiplicities'].sum() == 10
            and binding['course_360_count'] == 1, 'raw exact duplicate and course wrap check failed')
    endpoints = endpoint_arrays(times, values, offsets)
    require(220. < endpoints['secant'][0, 0] < 225. and endpoints['secant'][1, 0] == 0.,
            'date-line and zero-geometry checks failed')
    from association.data.motion_evidence import complete_native_attributes
    from tools.analysis.motion_fallback_diagnostic import MASKS, measure_scene, pool as c77_pool, support_tags
    from tools.analysis.motion_readout_evaluation import evaluate_scene
    membership = np.array([0, 0, 0])
    per_node_times = times.reshape(3, 3)
    support = (query >= per_node_times[:, :1]) & (query <= per_node_times[:, -1:])
    tags = support_tags(query, times, offsets, group_members(membership), support, sources)
    legacy = complete_native_attributes(values.reshape(3, 3, 4), per_node_times,
        np.ones((3, 3), bool), query, np.zeros((3, len(query), 2)))[..., 2:]
    baseline = np.zeros((1, len(query), 4))
    baseline[..., :2] = (1., 2.)
    baseline[..., 2:] = legacy[0]
    baseline[..., 2] *= KNOT_TO_MPS
    saved = dict(membership=membership, A=baseline.copy(), B=baseline.copy(), C=baseline.copy(),
                 active=tags['cause'] == 0, internal_gap=tags['internal_gap'], near_endpoint=tags['near_endpoint'],
                 B_low_speed=np.zeros((1, 64), bool), C_low_speed=np.zeros((1, 64), bool))
    nearest = np.abs(query-per_node_times[..., None]).min(axis=1)
    scene = (nodes, sources, support, nearest, membership, (query, times, values[:, :2], offsets), baseline)
    frozen = freeze_scene(scene, saved, legacy)
    data = freeze_endpoints(frozen, query, endpoints, sources, baseline, legacy)
    np.testing.assert_array_equal(np.flatnonzero(data['scope'][0]), np.r_[0, np.arange(7, 64)])
    np.testing.assert_array_equal(data['nearest_member'][0, [0, 7]], [1, 2])
    np.testing.assert_array_equal(data['first_side'][0, [0, 7]], [0, 1])
    np.testing.assert_array_equal(data['first_age'][0, [0, 7]], [2., 3.])
    np.testing.assert_array_equal(data['nearest_age'][0, [0, 7]], [1., 1.])
    bad = [dict(row) for row in raw]
    bad[0]['vel'] = '50'
    try:
        bind_measurements('9', bad, nodes, sources, times, values, offsets)
    except AssertionError:
        pass
    else:
        raise AssertionError('raw measurement mutation should be rejected')
    truth = np.zeros((2, 64, 4))
    truth[..., :2] = (1., 2.)
    truth[..., 2] = 1.
    ev = dict(identities=np.array(['ship']*3), target_ids=np.array(['ship', 'missing']),
              target_states=truth, target_valid=np.ones_like(truth, bool))
    c74, _ = evaluate_scene('9', 'component', nodes, [[0, 1, 2]], support, nearest, ev,
                            {key: saved[key] for key in METHODS}, {key: saved[key] for key in MASKS}, query)
    c77, c77_vectors = measure_scene('9', 'component', frozen, saved, ev, c74)
    c77 = dict(c77, errors=c77_pool([c77], [c77_vectors])['errors'])
    row, vector = evaluate('9', 'component', frozen, saved, data, ev, c74, c77)
    results = summarize_vectors(vector)
    require(row['unmatched_target_count'] == 1 and row['valid_speed_points'] == 64
            and results['all_eligible']['count'] == 58 and results['near_pure']['count'] == 3
            and results['near_pure_first_older']['count'] == 3
            and results['all_eligible']['distinct_first_endpoint_field_to_secant']
                       ['field_minus_adjacent_secant']['count'] == 2,
            'reference alignment, strata or distinct endpoints changed')

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        tmp = Path(directory).resolve()
        old = tmp/'old'
        (old/'private').mkdir(parents=True)
        artifact = old/'private/lineage-9.npz'
        np.savez_compressed(artifact, **data, **{key: binding[key] for key in RAW_BINDING_KEYS})
        runs, science, fallback = (tmp/key for key in ('runs', 'science', 'fallback'))
        paths = [str(tmp/'measurements.zip')]
        prior = dict(status='FAILED', exit_code=1, phase='evaluate_fixed_correspondence',
                     error=PUBLICATION_FAILURE, stages=[FROZEN_STAGE], split=SPLIT, scene_ids=['9'],
                     frozen_lineage_scenes=1, raw_measurement_members=1, raw_label_members=0,
                     raw_reference_members=0, optimizer_updates=0, predictions_generated=0,
                     test_opened=False, runs=str(runs), science=str(science), fallback=str(fallback),
                     output=str(old), archive_paths=paths)
        write(old/'run.json', prior)
        check_failed_publication(old, runs, science, fallback, ['9'], paths)
        write(old/'run.json', dict(prior, raw_label_members=1))
        try:
            check_failed_publication(old, runs, science, fallback, ['9'], paths)
        except ValueError:
            pass
        else:
            raise AssertionError('a failed attempt with labels in the freeze phase cannot be repaired')
        for extra in (dict(canonical_course_360=binding['course_360_count'],
                           scene_measurement_rows=binding['measurement_row_count']),
                      dict(canonical_course_360=None, scene_measurement_rows=None)):
            loaded, counts = load_lineage(artifact, frozen, extra)
            for key in data:
                np.testing.assert_array_equal(loaded[key], data[key])
            report_row = dict(row, raw_endpoint_binding=counts)
            scenes = [dict(report_row, regions=results)]
            summary = pool([report_row], [vector])
            write(tmp/'scenes.json.gz', scenes)
            write(tmp/'summary.json', summary)
            require(read(tmp/'scenes.json.gz') == scenes and read(tmp/'summary.json') == summary,
                    'strict JSON and GZ publication round trips changed')
            require(summary['raw_endpoint_binding']['matched_raw_rows'] == 10
                    and summary['raw_endpoint_binding']['exact_duplicate_matches'] == 1
                    and summary['raw_endpoint_binding']['canonical_course_360'] == extra['canonical_course_360'],
                    'saved row counts or missing-publication values changed')
    print('C79 self-check passed: raw lineage, endpoint ties, frozen evaluation and strict publication repair.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--science', type=Path)
    parser.add_argument('--fallback', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frozen-attempt', type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif not all((args.runs, args.science, args.fallback, args.output, args.time_policy)):
        parser.error('--runs, --science, --fallback, --output and --time-policy are required')
    else:
        run(args.runs.resolve(), args.science.resolve(), args.fallback.resolve(), args.output.resolve(),
            frozen_attempt=args.frozen_attempt.resolve() if args.frozen_attempt is not None else None)
