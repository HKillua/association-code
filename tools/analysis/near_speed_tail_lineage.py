"""C106: reproduce C103 tail endpoints and compare native same-window evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.physical_units import KNOT_TO_MPS
from association.data.reference_states import iter_reference_states
from association.data.targets import align_reference_states
from tools.analysis.held_fusion_population import (
    PARENTS, SPLIT, load_full_population, load_scene, read_evaluation, validate_measurement_archive,
)
from tools.analysis.near_speed_interval_diagnostic import independent_speed
from tools.analysis.near_speed_maneuver_held import GAP_MULTIPLE, endpoint_details, headings_from_positions
from tools.analysis.offline_endpoint_lineage import speed_between
from tools.analysis.same_source_swap_pilot import MAX_AS, group_members, peak_rss_bytes, read, write

C103 = 'LEVEL1-NEAR-SPEED-MANEUVER-C103-20261112-attempt-1'
STRATA = {'unchanged': (False, True), 'switch_same': (True, True), 'switch_other': (True, False)}
SCENE_MAX = 12
ROLES = ('B', 'NEAREST')


def select_cases(points, endpoints):
    speeds = {e['endpoint_id']: e['speeds'] for e in endpoints}
    result, populations = [], {}
    for name, (switch, same) in STRATA.items():
        sample = [p for p in points if p['switch'] == switch and p['donor_same_identity']['NEAREST'] == same]
        populations[name] = len(sample)
        choices = {}
        for criterion, count in (('harm', 2), ('field_error', 1), ('net5_error', 1)):
            ranked = []
            for point in sample:
                key = point['endpoint_ids']['NEAREST']
                field = abs(speeds[key]['field'] - point['reference'])
                net = abs(speeds[key]['net_5'] - point['reference'])
                score = {'harm': net - field, 'field_error': field, 'net5_error': net}[criterion]
                ranked.append((-round(score, 9), int(point['scene']), point['group'], point['query_index'], key, score))
            seen = set()
            for _, sid, group, query, key, score in sorted(ranked):
                if key in seen:
                    continue
                seen.add(key)
                chosen = choices.setdefault(key, dict(scene=str(sid), group=group, endpoint_id=key, stratum=name, reasons=[]))
                assert chosen['scene'] == str(sid) and chosen['group'] == group
                chosen['reasons'].append(dict(criterion=criterion, rank=len(seen), query_index=query, score_mps=score))
                if len(seen) == count:
                    break
        for item in choices.values():
            item['query_indices'] = sorted(p['query_index'] for p in sample if p['endpoint_ids']['NEAREST'] == item['endpoint_id'])
            result.append(item)
    assert len(points) == 1195 and list(populations.values()) == [780, 282, 133]
    return dict(split='V_select', c103_population=populations,
                scenes=sorted({x['scene'] for x in result}, key=int), cases=result)


def track(scene, node):
    _, times, _, offsets = scene['native']
    a, b = offsets[node:node+2]
    return times[a:b], scene['values'][a:b]


def relative_direction(a, b):
    return abs(float((a - b + 180.) % 360. - 180.))


def distance_speed(a, b, elapsed):
    speed = float(speed_between(a, b, np.asarray(elapsed)))
    np.testing.assert_allclose(speed, independent_speed(a, b, elapsed), rtol=0., atol=1e-8)
    return speed


def interpolation(times, positions, a, b):
    i, j = np.searchsorted(times, (a, b), side='left')
    start = i if times[i] == a else i - 1
    bracketing = times[start:j+1]
    t = np.asarray((a, b))
    # Unwrap only longitude on the bracketed span, with each increment on the short arc.
    positions = positions[start:j+1]
    lon = np.rad2deg(np.unwrap(np.deg2rad(positions[:, 1])))
    return start, int(j), np.stack((np.interp(t, bracketing, positions[:, 0]),
                                   (np.interp(t, bracketing, lon) + 180.) % 360. - 180.), axis=-1)


def other_window(times, values, a, b, original):
    intersection = max(0., min(float(times[-1]), b) - max(float(times[0]), a))
    result = dict(overlap_seconds=intersection)
    if intersection == 0.:
        result.update(status='no_overlap', distance_in_time_seconds=float(max(times[0]-b, a-times[-1], 0.)))
        return result
    if times[0] > a or times[-1] < b:
        return dict(result, status='partial_overlap')
    start, stop, bounds = interpolation(times, values[:, :2], a, b)
    elapsed = b - a
    dt = np.diff(times[start:stop+1])
    widest = float(dt.max())
    max_multiple = widest / float(np.median(np.diff(times)))
    result.update(max_bracket_seconds=widest, max_bracket_multiple=max_multiple,
                  exact_boundaries=int(times[start] == a) + int(times[stop] == b), native_intervals=len(dt))
    if max_multiple > GAP_MULTIPLE:
        return dict(result, status='gap_exceeded')
    overlaps = np.minimum(times[start+1:stop+1], b) - np.maximum(times[start:stop], a)
    np.testing.assert_allclose(overlaps.sum(), elapsed, rtol=0., atol=1e-10)
    matched = distance_speed(*bounds, elapsed)
    native_peaks = [distance_speed(l, r, d) for l, r, d in zip(values[start:stop, :2], values[start+1:stop+1, :2], dt, strict=True)]
    other_fields = values[start:stop, 2] * KNOT_TO_MPS
    heading = headings_from_positions(bounds)[0]
    origin_heading = headings_from_positions(original)[0]
    result.update(status='comparable', secant_mps=matched,
                  forward_fields_weighted_mps=float(overlaps @ other_fields / elapsed),
                  max_native_interval_mps=max(native_peaks), max_forward_field_mps=float(other_fields.max()),
                  direction_difference_deg=relative_direction(origin_heading, heading),
                  left_cross_distance_metres=distance_speed(original[0], bounds[0], 1.),
                  right_cross_distance_metres=distance_speed(original[1], bounds[1], 1.))
    return result


def reference_view(raw, identity, policy):
    rows = [r for r in raw['rows'] if r['identity_id'] == identity]
    times = np.asarray(sorted({r['time'] - policy['time_offset_seconds'] for r in rows
                               if r['time'] is not None and np.isfinite(r['time'])}))
    return dict(raw, rows=rows, issues=[]), times


def reference_window(view, identity, policy, a, b=None):
    raw, times = view
    requested = [a] if b is None else [a, b]
    answer = align_reference_states(raw, identity, requested, **policy, longitude_policy='shortest_arc')
    fields = []
    for timestamp, state, valid in zip(requested, answer['states'], answer['valid'], strict=True):
        right = int(np.searchsorted(times, timestamp))
        exact = right < len(times) and times[right] == timestamp
        left_gap = 0. if exact else float(timestamp - times[right-1]) if right else None
        right_gap = 0. if exact else float(times[right] - timestamp) if right < len(times) else None
        fields.append(dict(speed_mps=float(state[2]) if valid[2] else None, exact_row_time=bool(exact),
                           left_bracket_age_seconds=left_gap, right_bracket_age_seconds=right_gap))
    result = dict(query_rows=fields)
    if b is not None:
        in_window = times[(times > a) & (times < b)]
        largest = float(np.diff(np.r_[a, in_window, b]).max())
        comparable = answer['valid'][:, :2].all() and largest <= policy['max_gap_seconds']
        result.update(raw_interior_times=len(in_window), max_inside_gap_seconds=largest,
                      reference_secant_mps=distance_speed(*answer['states'][:, :2], b-a) if comparable else None)
    return result


def role_trace(scene, ev, raw, policy, node, side, endpoint, target, first_query):
    times, values = track(scene, node)
    origin = times[0 if side == 0 else -1]
    first = 0 if side == 0 else len(times) - 6
    t, v = times[first:first+6], values[first:first+6]
    field = float(values[0 if side == 0 else -1, 2] * KNOT_TO_MPS)
    # C103 passed a saved readout scalar to endpoint_details; its arithmetic
    # order may differ from direct conversion of the identical native row.
    np.testing.assert_allclose(field, endpoint['speeds']['field'], rtol=0., atol=2e-12)
    details = endpoint_details(times, values, side, endpoint['speeds']['field'])
    for key in ('speeds', 'windows', 'motion', 'boundary', 'gap', 'native_intervals'):
        assert details[key] == endpoint[key], (endpoint['endpoint_id'], key)
    dt = np.diff(t)
    secants = np.asarray([distance_speed(a, b, d) for a, b, d in zip(v[:-1, :2], v[1:, :2], dt, strict=True)])
    lengths = dt * secants
    fields = v[:, 2] * KNOT_TO_MPS
    headings = headings_from_positions(v[:, :2])
    id_to_roles = {target: ['target']}
    own = ev['identities'][node]
    id_to_roles.setdefault(own, []).append('supplier')
    trace_groups = []
    src = scene['sources']
    unique_src = np.unique(src)
    for identity, roles in id_to_roles.items():
        view = reference_view(raw, identity, policy)
        other = np.flatnonzero((ev['identities'] == identity) & (src != src[node]))
        sources = []
        for member in other:
            mt, mv = track(scene, member)
            sources.append(dict(source_ordinal=int(np.searchsorted(unique_src, src[member])+1),
                                fragment_ordinal=int(member+1),
                                intervals=[other_window(mt, mv, a, b, v[i:i+2, :2]) for i, (a, b) in enumerate(zip(t[:-1], t[1:], strict=True))],
                                full_window=other_window(mt, mv, t[0], t[-1], v[[0, -1], :2])))
        trace_groups.append(dict(identity_roles=roles, other_source_fragments=sources,
            reference_intervals=[reference_window(view, identity, policy, a, b) for a, b in zip(t[:-1], t[1:], strict=True)],
            reference_full_window=reference_window(view, identity, policy, t[0], t[-1])))
    others_same_src = np.flatnonzero((ev['identities'] == own) & (src == src[node]))
    external = []
    for member in others_same_src:
        if member == node:
            continue
        mt, _ = track(scene, member)
        outward = origin - mt if side == 0 else mt - origin
        if np.any(outward > 0):
            external.append(float(outward[outward > 0].min()))
    neighbor_gap = min(external, default=None)
    np.testing.assert_equal(neighbor_gap, endpoint['external_dt_seconds'])
    np.testing.assert_allclose(float(lengths.sum() / dt.sum()), details['speeds']['path_5'], atol=1e-8, rtol=0.)
    rows = [dict(left_seconds_from_endpoint=float(t[i]-origin), right_seconds_from_endpoint=float(t[i+1]-origin),
                 dt_seconds=float(d), displacement_metres=float(lengths[i]), secant_mps=float(secants[i]),
                 left_field_kn=float(v[i, 2]), right_field_kn=float(v[i+1, 2]),
                 left_field_mps=float(fields[i]), right_field_mps=float(fields[i+1]),
                 left_course_to_secant_direction_deg=relative_direction(v[i, 3], headings[i]),
                 turn_from_previous_secant_deg=relative_direction(headings[i], headings[i-1]) if i else None)
            for i, d in enumerate(dt)]
    return dict(endpoint_id=endpoint['endpoint_id'], fragment_ordinal=int(node+1),
                source_ordinal=int(np.searchsorted(unique_src, src[node])+1), same_identity_as_target=bool(own == target),
                endpoint_seconds_from_first_query=float(origin-first_query), boundary=details['boundary'],
                original_endpoint_field_kn=float(values[0 if side == 0 else -1, 2]),
                original_endpoint_field_mps=field,
                fragment_dt_median_seconds=details['fragment_dt_median_seconds'],
                external_same_source_neighbor_gap_seconds=neighbor_gap, speeds=details['speeds'],
                window=details['windows']['5'], intervals=rows, max_step_ordinal=int(np.argmax(lengths)+1),
                max_step_path_share=float(lengths.max()/lengths.sum()) if lengths.sum() else None,
                max_secant_ordinal=int(np.argmax(secants)+1), reference_and_cross_source=trace_groups)


def inspect_case(scene, ev, raw, policy, case, c103_private, blind, public_points, endpoints):
    g = case['group']
    group = group_members(scene['membership'])
    target = ev['identities'][group[g][0]]
    assert all(ev['identities'][other[0]] != target for other in group[:g])
    target_index, = np.flatnonzero(ev['target_ids'] == target)
    rows = {row['query_index']: row for row in blind['rows'] if row['group'] == g}
    q0 = case['query_indices'][0]
    selected = rows[q0]
    query = scene['query']
    roles = {}
    for role in ROLES:
        role_data = selected['roles'][role]
        endpoint = endpoints[selected['endpoint_ids'][role]]
        source = role_trace(scene, ev, raw, policy, role_data['node'], role_data['side'], endpoint, target, query[q0])
        roles[role] = source
    query_results = []
    for q in case['query_indices']:
        point = public_points[(case['scene'], g, q)]
        row = rows[q]
        assert row['endpoint_ids'] == selected['endpoint_ids'] == point['endpoint_ids']
        assert row['switch'] == point['switch'] and row['same_source'] == point['same_source']
        assert row['endpoint_ids']['NEAREST'] == case['endpoint_id']
        query_reference = float(ev['target_states'][target_index, q, 2])
        assert ev['target_valid'][target_index, q, 2] and query_reference == point['reference']
        view = reference_view(raw, target, policy)
        ref = reference_window(view, target, policy, query[q])['query_rows'][0]
        ref['c103_cached_speed_mps'] = query_reference
        ref['raw_matches_c103_speed'] = ref['speed_mps'] is not None and abs(ref['speed_mps']-query_reference) <= 1e-9
        for name, kind in (('B', 'B'), ('NEAREST', 'NEAR')):
            assert row['roles'][name]['age_seconds'] == point['age_seconds'][name]
            assert point['donor_same_identity'][name] == roles[name]['same_identity_as_target']
            np.testing.assert_equal(c103_private[kind][g, q, 2], roles[name]['speeds']['field'])
        query_results.append(dict(query_index=q, reference=ref, age_seconds=point['age_seconds']))
    return dict(case, reference_clock_offset_seconds=policy['time_offset_seconds'],
                roles=roles, selected_queries=query_results, switch=selected['switch'], same_source=selected['same_source'])


def reuse_bound_measurements(original, failed, selection):
    """Reuse exact C103 arrays already re-bound to raw members by attempt 1."""
    prior = read(failed/'run.json')
    assert prior['status'] == 'FAILED' and prior['error_type'] == 'AssertionError'
    assert prior['stages'] == ['selection_and_measurement_binding_before_reference']
    assert prior['raw_measurement_members_read'] == len(selection['scenes']) and prior['raw_label_members_read'] == 0
    assert read(failed/'selection.json') == selection
    result = {}
    for sid in selection['scenes']:
        with np.load(original/'private'/f'{sid}.npz', allow_pickle=False) as parent, np.load(
                original/'private'/f'{sid}-native.npz', allow_pickle=False) as native:
            scene = {key: parent[key] for key in ('nodes', 'query', 'membership', 'support', 'nearest')}
            scene.update(values=native['values'], sources=native['sources'],
                         native=(scene['query'], native['times'], None, native['offsets']))
            result[sid] = scene
    return result


def run(args):
    start = time.monotonic()
    args.output.mkdir(exist_ok=False)
    record = dict(status='RUNNING', idea_id='IDEA-C106-NEAR-SPEED-TAIL-LINEAGE-20261112',
                  plan_version='1.1-exact-binding-repair' if args.bound_measurements else '1.0',
                  affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  started_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0, neural_fits=0,
                  test_opened=False, new_prediction_arms=0, stages=[])
    write(args.output/'run.json', record)
    try:
        assert record['affinity'] == [27] and record['cuda_visible_devices'] == ''
        original = args.runs/C103
        c103 = read(original/'run.json')
        assert c103['status'] == 'COMPLETED' and c103['exit_code'] == 0 and c103['valid_near'] == 1195
        points, endpoints = (read(original/f'{name}.json.gz') for name in ('points', 'endpoints'))
        selection = select_cases(points, endpoints)
        assert selection == read(args.selection) and len(selection['scenes']) <= SCENE_MAX
        write(args.output/'selection.json', selection)
        ids = selection['scenes']
        full, components, archives = load_full_population(args.runs)
        c103_roster = read(original/'roster.json')
        assert set(ids) <= set(full) & set(c103_roster['scenes'])
        full_record = read(args.runs/PARENTS['full_data']/'run.json')
        assert full_record['measurement_speed_unit'] == 'kn' and full_record['reference_speed_unit'] == 'mps'
        manifest, policies = (read(Path(full_record[k])) for k in ('manifest', 'alignment'))
        assert not set(ids) & set(manifest['splits']['T_test'])
        assert all(policies[s]['max_gap_seconds'] == 600. for s in ids)
        blind = read(original/'private'/'blind-endpoints.json.gz')
        if args.bound_measurements:
            raw_scenes = reuse_bound_measurements(original, args.bound_measurements, selection)
            record.update(raw_measurement_members_read=0, reused_raw_measurement_bindings=len(ids),
                          binding_attempt=str(args.bound_measurements))
            record['stages'].append('selection_and_attempt1_measurement_binding_reused_before_reference')
        else:
            raw_scenes = {}
            with MTADArchive(archives) as archive:
                validate_measurement_archive(args.runs, archive)
                for sid in ids:
                    scene = load_scene(args.runs, sid, archive, scene_ids=ids)
                    _, times, _, offsets = scene['native']
                    with np.load(original/'private'/f'{sid}-native.npz', allow_pickle=False) as saved:
                        for key, actual in dict(times=times, offsets=offsets, values=scene['values'], sources=scene['sources']).items():
                            np.testing.assert_array_equal(saved[key], actual)
                    raw_scenes[sid] = scene
                assert [(r['scene_id'], r['kind']) for r in archive.members_read] == [(sid, 'measurements') for sid in ids]
            record['stages'].append('selection_and_measurement_binding_before_reference')
            record['raw_measurement_members_read'] = len(ids)
        record['raw_label_members_read'] = 0
        write(args.output/'run.json', record)
        public_points = {(p['scene'], p['group'], p['query_index']): p for p in points}
        endpoint_map = {r['endpoint_id']: r for r in endpoints}
        audited, reference_members = [], []
        for raw in iter_reference_states(archives, manifest, split=SPLIT, scene_ids=ids):
            sid = raw['scene_id']
            reference_members.append(sid)
            scene = raw_scenes.pop(sid)
            ev = read_evaluation(args.runs, sid, scene['nodes'], scene['query'], scene_ids=ids)
            with np.load(original/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                for key in ('nodes', 'query', 'membership', 'support', 'nearest'):
                    np.testing.assert_array_equal(saved[key], scene[key])
                for case in selection['cases']:
                    if case['scene'] == sid:
                        audited.append(inspect_case(scene, ev, raw, policies[sid], case, saved,
                                                   blind[sid], public_points, endpoint_map))
        assert reference_members == ids and len(audited) == len(selection['cases'])
        write(args.output/'cases.json.gz', audited)
        summaries = []
        for case in audited:
            key = case['roles']['NEAREST']
            groups = []
            for group in key['reference_and_cross_source']:
                cross = group['other_source_fragments']
                groups.append(dict(identity_roles=group['identity_roles'], other_source_fragments=len(cross),
                    comparable_intervals=sum(any(row['intervals'][i]['status'] == 'comparable' for row in cross) for i in range(5)),
                    comparable_full_window=sum(row['full_window']['status'] == 'comparable' for row in cross),
                    status_counts=dict(Counter(r['status'] for frag in cross for r in frag['intervals'])),
                    reference_full_window_secant_mps=group['reference_full_window']['reference_secant_mps']))
            summaries.append(dict(scene=case['scene'], group=case['group'], stratum=case['stratum'], component=components[case['scene']],
                query_indices=case['query_indices'], speeds=key['speeds'], max_step_path_share=key['max_step_path_share'],
                native_dt_seconds=[r['dt_seconds'] for r in key['intervals']],
                native_secant_mps=[r['secant_mps'] for r in key['intervals']],
                reference_speed_matches=sum(q['reference']['raw_matches_c103_speed'] for q in case['selected_queries']),
                groups=groups))
        write(args.output/'summary.json', dict(cases=summaries, selected_queries=sum(len(r['selected_queries']) for r in audited),
              cross_source_convention='same archived clock, no extrapolation, same fragment, 2.5x local cadence gap cap',
              shared_ancestry='MTAD sources and reference descend from original AIS; independent velocity sensing unverified'))
        record['stages'].append('raw_same_window_reference_and_cross_source_trace_completed')
        record.update(status='COMPLETED', exit_code=0, raw_reference_members_read=len(reference_members),
                      cases=len(audited), elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes(),
                      finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        record.update(status='FAILED', exit_code=1, error_type=type(error).__name__,
                      elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        write(args.output/'run.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--bound-measurements', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
    run(args)
