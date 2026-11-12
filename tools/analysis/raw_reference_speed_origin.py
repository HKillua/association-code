"""C109: trace position and speed divergence before reference interpolation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.data.reference_states import iter_reference_states
from association.data.targets import align_reference_states
from association.evaluation.physical import EARTH_RADIUS_METRES
from tools.analysis.held_fusion_population import PARENTS, SPLIT, load_full_population, read_evaluation
from tools.analysis.same_source_swap_pilot import MAX_AS, group_members, peak_rss_bytes, read, write


C103 = 'LEVEL1-NEAR-SPEED-MANEUVER-C103-20261112-attempt-1'
C106 = 'LEVEL1-NEAR-SPEED-TAIL-LINEAGE-C106-R1-20261112-attempt-1'
SCENES = ('1315', '2036', '2396', '3035', '3692', '4145', '4393', '4771')
ROLES = ('B', 'NEAREST')
FIELDS = ('lat', 'lon', 'vel', 'cou')
RANGES = ((-90., 90.), (-180., 180.), (0., math.inf), (0., 360.))
LIMITS = (1., 10.)


def valid(value, bounds):
    return value is not None and math.isfinite(value) and bounds[0] <= value <= bounds[1]


def finite_list(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def distance(a, b):
    def vector(point):
        latitude, longitude = np.deg2rad(point)
        return np.array((np.cos(latitude)*np.cos(longitude), np.cos(latitude)*np.sin(longitude), np.sin(latitude)))

    u, v = vector(a), vector(b)
    return float(EARTH_RADIUS_METRES * np.arctan2(np.linalg.norm(np.cross(u, v)), u @ v))


def raw_track(rows):
    groups = defaultdict(list)
    for row in rows:
        if valid(row['time'], (-math.inf, math.inf)):
            groups[row['time']].append(row)
    timestamps = np.asarray(sorted(groups))
    values = np.full((len(timestamps), 4), np.nan)
    counts = Counter(rows=len(rows), times=len(timestamps), unlocatable_rows=len(rows)-sum(map(len, groups.values())))
    valid_times_in_file_order = [row['time'] for row in rows if valid(row['time'], (-math.inf, math.inf))]
    counts['raw_file_order_descents'] = sum(a > b for a, b in zip(valid_times_in_file_order, valid_times_in_file_order[1:]))
    for index, timestamp in enumerate(timestamps):
        group = groups[timestamp]
        counts['duplicate_groups'] += len(group) > 1
        for col, bounds in enumerate(RANGES):
            field = [row['state'][col] for row in group]
            if all(valid(x, bounds) for x in field) and len(set(field)) == 1:
                values[index, col] = field[0] % 360. if col == 3 else field[0]
            else:
                counts[f'invalid_or_conflicting_{FIELDS[col]}_times'] += 1
    return timestamps, values, counts, groups


def bracket(times, states, q, max_gap):
    index = int(np.searchsorted(times, q))
    if index < len(times) and times[index] == q:
        return states[index].copy(), dict(status='exact', ordinals=[index+1], ages_seconds=[0.])
    missing = np.full(4, np.nan)
    if index == 0 or index == len(times):
        return missing, dict(status='outside', ordinals=[], ages_seconds=[])
    a, b = times[index-1:index+1]
    info = dict(ordinals=[index, index+1], ages_seconds=[float(q-a), float(b-q)], gap_seconds=float(b-a))
    if b-a > max_gap:
        return missing, dict(info, status='gap_exceeded')
    left, right = states[index-1:index+1]
    weight = (q-a)/(b-a)
    state = left + (right-left)*weight
    if np.isfinite(left[1]) and np.isfinite(right[1]) and abs(right[1]-left[1]) > 180.:
        state[1] = (left[1] + weight*((right[1]-left[1]+180.) % 360.-180.)+180.) % 360.-180.
    if np.isfinite(left[3]) and np.isfinite(right[3]):
        delta = (right[3]-left[3]+180.) % 360.-180.
        state[3] = (left[3]+weight*delta) % 360. if abs(delta) != 180. else np.nan
    return state, dict(info, status='interior')


def compare_fields(geometric, speeds, scale=1.):
    if geometric is None or any(speed is None for speed in speeds):
        return None
    geometric = float(geometric)
    scaled = [speed*scale for speed in speeds]
    return dict(geometry_minus_left_mps=geometric-scaled[0], geometry_minus_right_mps=geometric-scaled[1],
                geometry_minus_mean_mps=geometric-sum(scaled)/2.,
                geometry_above_both_mps=max(0., geometric-max(scaled)),
                geometry_below_both_mps=max(0., min(scaled)-geometric))


def raw_intervals(times, states, shift):
    dt = np.diff(times)
    np.testing.assert_allclose(np.diff(times-shift), dt, rtol=0., atol=2e-10)
    result = []
    for i, elapsed in enumerate(dt):
        pair = states[i:i+2]
        endpoints = finite_list(pair[:, 2])
        position_available = np.isfinite(pair[:, :2]).all()
        metres = distance(*pair[:, :2]) if position_available else None
        geometric = metres/elapsed if metres is not None else None
        result.append(dict(raw_ordinals=[i+1, i+2], dt_seconds=float(elapsed), displacement_metres=metres,
                           secant_mps=float(geometric) if geometric is not None else None,
                           endpoint_fields=endpoints, mps=compare_fields(geometric, endpoints),
                           if_kn=compare_fields(geometric, endpoints, KNOT_TO_MPS),
                           longitude_wrap=bool(position_available and abs(pair[0, 1]-pair[1, 1]) > 180.)))
    return result


def describe(intervals):
    paired = [row for row in intervals if row['mps'] is not None]
    return dict(intervals=len(intervals), comparable=len(paired), noncomparable=len(intervals)-len(paired),
                longitude_wraps=sum(row['longitude_wrap'] for row in intervals),
                max_secant_mps=max((row['secant_mps'] for row in intervals if row['secant_mps'] is not None), default=None),
                **{f'{unit}_{kind}_gt{int(limit)}': sum(row[unit][f'geometry_{kind}_both_mps'] > limit for row in paired)
                   for unit in ('mps', 'if_kn') for kind in ('above', 'below') for limit in LIMITS})


def audit_sample(track, raw, identity, queries, policy):
    times, states = track[:2]
    shift = policy['time_offset_seconds']
    requests = np.asarray(queries)
    expected = align_reference_states(raw, identity, requests, **policy, longitude_policy='shortest_arc')
    computed = [bracket(times-shift, states, t, policy['max_gap_seconds']) for t in requests]
    rebuilt = np.stack([row[0] for row in computed])
    np.testing.assert_array_equal(expected['valid'], np.isfinite(rebuilt))
    np.testing.assert_allclose(rebuilt, expected['states'], rtol=0., atol=2e-11)
    return rebuilt, [item[1] for item in computed]


def aligned_window(track, raw, identity, knots, policy, c106, intervals):
    start, stop = float(knots[0]), float(knots[-1])
    elapsed = stop-start
    raw_times, _, _, groups = track
    shift = policy['time_offset_seconds']
    times = raw_times-shift
    interior = times[(times > start) & (times < stop)]
    supports = np.r_[start, interior, stop]
    rebuilt, brackets = audit_sample(track, raw, identity, supports, policy)
    endpoint_fields = finite_list(rebuilt[[0, -1], 2])
    expected_fields = [entry['speed_mps'] for entry in c106['query_rows']]
    for a, b in zip(endpoint_fields, expected_fields, strict=True):
        if a is None or b is None:
            assert a == b
        else:
            np.testing.assert_allclose(a, b, rtol=0., atol=1e-9)
    if c106['reference_secant_mps'] is None:
        geometric = None
    else:
        geometric = distance(*rebuilt[[0, -1], :2])/elapsed
        np.testing.assert_allclose(geometric, c106['reference_secant_mps'], rtol=1e-10, atol=1e-8)
    path = (sum(distance(a, b) for a, b in zip(rebuilt[:-1, :2], rebuilt[1:, :2], strict=True))/elapsed
            if np.isfinite(rebuilt[:, :2]).all() and np.diff(supports).max() <= policy['max_gap_seconds'] else None)
    integrated = (float(np.diff(supports) @ ((rebuilt[:-1, 2]+rebuilt[1:, 2])/2)/elapsed)
                  if np.isfinite(rebuilt[:, 2]).all() and np.diff(supports).max() <= policy['max_gap_seconds'] else None)
    no_shift = np.stack([bracket(raw_times, track[1], t, policy['max_gap_seconds'])[0] for t in (start, stop)])
    no_shift_secant = distance(*no_shift[:, :2])/elapsed if np.isfinite(no_shift[:, :2]).all() else None
    crossed, before_and_after = [], []
    for i, row in enumerate(intervals):
        a, b = times[i:i+2]
        if a < stop and b > start:
            crossed.append(dict(row, relative_seconds=[float(a-start), float(b-start)],
                                overlap_seconds=float(min(b, stop)-max(a, start)),
                                duplicate_row_counts=[len(groups[t]) for t in raw_times[i:i+2]]))
            before_and_after.append(i)
    neighbors = []
    if before_and_after:
        for index in (before_and_after[0]-1, before_and_after[-1]+1):
            if 0 <= index < len(intervals):
                neighbors.append(dict(intervals[index], relative_seconds=(times[index:index+2]-start).tolist()))
    clipped_gap = compare_fields(geometric, endpoint_fields)
    phase = {}
    for direction in ('above', 'below'):
        key = f'geometry_{direction}_both_mps'
        phase[direction] = {str(int(limit)): dict(
            raw_crossing=any(row['mps'] is not None and row['mps'][key] > limit for row in crossed),
            aligned_window=clipped_gap[key] > limit if clipped_gap is not None else None) for limit in LIMITS}
    retained = {int(j)-1 for r in brackets for j in r['ordinals']}
    retained.update(j for row in crossed+neighbors for j in (row['raw_ordinals'][0]-1, row['raw_ordinals'][1]-1))
    private_rows = [row for j in sorted(retained) for row in groups[raw_times[j]]]
    public = dict(elapsed_seconds=elapsed, aligned_endpoint_fields_mps=endpoint_fields, aligned_secant_mps=geometric,
                  aligned_path_mps=path, aligned_integrated_field_mean_mps=integrated, aligned_position_minus_fields=clipped_gap,
                  unshifted_endpoint_fields=finite_list(no_shift[:, 2]), unshifted_secant_mps=no_shift_secant,
                  raw_inside_times=len(interior), alignment_brackets=[brackets[0], brackets[-1]],
                  phase=phase, crossing=describe(crossed), raw_crossing_intervals=crossed,
                  raw_adjacent_outside_intervals=neighbors)
    return public, private_rows


def native_window(role, native_times, offsets, query, first_query):
    node = role['fragment_ordinal']-1
    a, b = offsets[node:node+2]
    times = native_times[a:b]
    first = role['intervals'][0]['left_seconds_from_endpoint'] == 0.
    knots = times[:6] if first else times[-6:]
    endpoint = knots[0] if first else knots[-1]
    np.testing.assert_allclose(endpoint-query[first_query], role['endpoint_seconds_from_first_query'], atol=1e-8, rtol=0.)
    np.testing.assert_allclose(np.diff(knots), [item['dt_seconds'] for item in role['intervals']], atol=1e-8, rtol=0.)
    return node, knots


def trace_scene(raw, sid, cases, runs, offset_policy, rosters):
    source = runs/C103/'private'
    with np.load(source/f'{sid}.npz', allow_pickle=False) as parent, np.load(source/f'{sid}-native.npz', allow_pickle=False) as native:
        nodes, query, membership = [parent[k] for k in ('nodes', 'query', 'membership')]
        native_times, offsets = native['times'], native['offsets']
    ev = read_evaluation(runs, sid, nodes, query, scene_ids=rosters)
    groups = group_members(membership)
    by_identity = defaultdict(list)
    for row in raw['rows']:
        by_identity[row['identity_id']].append(row)
    tracks = {}
    raw_outputs, private = [], []
    for case in cases:
        target = ev['identities'][groups[case['group']][0]]
        target_idx, = np.flatnonzero(ev['target_ids'] == target)
        q = np.asarray(case['query_indices'])
        if target not in tracks:
            tracks[target] = raw_track(by_identity[target])
        trace, query_brackets = audit_sample(tracks[target], raw, target, query[q], offset_policy)
        keep = ev['target_valid'][target_idx, q]
        np.testing.assert_array_equal(np.isfinite(trace), keep)
        np.testing.assert_allclose(trace[keep], ev['target_states'][target_idx, q][keep], rtol=0., atol=1e-9)
        old_queries = [row['reference']['c103_cached_speed_mps'] for row in case['selected_queries']]
        np.testing.assert_allclose(trace[:, 2], old_queries, rtol=0., atol=1e-9)
        role_rows = {}
        for name in ROLES:
            role = case['roles'][name]
            node, knots = native_window(role, native_times, offsets, query, q[0])
            supplier = ev['identities'][node]
            identity_roles = {target: ['target']}
            identity_roles.setdefault(supplier, []).append('supplier')
            assert (target == supplier) == role['same_identity_as_target']
            previous = {tuple(row['identity_roles']): row for row in role['reference_and_cross_source']}
            windows = []
            for identity, role_names in identity_roles.items():
                if identity not in tracks:
                    tracks[identity] = raw_track(by_identity[identity])
                track = tracks[identity]
                reference_intervals = raw_intervals(*track[:2], offset_policy['time_offset_seconds'])
                archived = previous[tuple(role_names)]
                public, evidence = aligned_window(track, raw, identity, knots, offset_policy,
                                                  archived['reference_full_window'], reference_intervals)
                cross = [row['full_window']['secant_mps'] for row in archived['other_source_fragments']
                         if row['full_window']['status'] == 'comparable']
                windows.append(dict(public, identity_roles=role_names, track=dict(track[2]),
                                    entire_track=describe(reference_intervals), cross_source_full_mps=cross,
                                    raw_check_before_offset=True))
                private.append(dict(scene=sid, group=case['group'], role=name, identity_roles=role_names,
                                    original_rows=evidence))
            role_rows[name] = dict(endpoint_id=role['endpoint_id'], native_interval_mps=[r['secant_mps'] for r in role['intervals']],
                                   native_interval_dt_seconds=np.diff(knots).tolist(), native_speeds=role['speeds'],
                                   same_identity_as_target=bool(supplier == target), windows=windows)
        raw_outputs.append(dict(scene=sid, group=case['group'], endpoint_id=case['endpoint_id'], stratum=case['stratum'],
                                query_indices=q.tolist(), offset_seconds=offset_policy['time_offset_seconds'],
                                query_cache_replayed_fields=int(keep.sum()), query_brackets=query_brackets,
                                query_speeds_mps=trace[:, 2].tolist(), roles=role_rows))
    return raw_outputs, private


def run(args):
    start = time.monotonic()
    args.output.mkdir(exist_ok=False)
    record = dict(status='RUNNING', idea_id='IDEA-C109-RAW-REFERENCE-SPEED-ORIGIN-20261112', plan_version='1.1-json-scalar-repair',
                  affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  started_at=datetime.now(timezone.utc).isoformat(), optimizer_updates=0, neural_fits=0,
                  test_opened=False, raw_measurement_members_read=0, raw_label_members_read=0,
                  new_prediction_arms=0, stages=[])
    write(args.output/'run.json', record)
    try:
        assert record['affinity'] == [27] and record['cuda_visible_devices'] == ''
        original = args.runs/C106
        history = read(original/'run.json')
        assert history['status'] == 'COMPLETED' and history['reused_raw_measurement_bindings'] == 8
        selection = read(original/'selection.json')
        cases = read(original/'cases.json.gz')
        assert tuple(selection['scenes']) == SCENES and len(cases) == len(selection['cases']) == 10
        assert sum(len(case['query_indices']) for case in cases) == 17
        assert {(case['scene'], case['group'], case['endpoint_id'], tuple(case['query_indices'])) for case in cases} == {
            (case['scene'], case['group'], case['endpoint_id'], tuple(case['query_indices'])) for case in selection['cases']}
        full, _, archives = load_full_population(args.runs)
        record_151 = read(args.runs/PARENTS['full_data']/'run.json')
        assert record_151['measurement_speed_unit'] == 'kn' and record_151['reference_speed_unit'] == 'mps'
        manifest, policies = (read(Path(record_151[k])) for k in ('manifest', 'alignment'))
        assert set(SCENES) <= set(full) and not set(SCENES) & set(manifest['splits']['T_test'])
        assert all(policies[s]['max_gap_seconds'] == 600. for s in SCENES)
        write(args.output/'selection.json', selection)
        record['stages'].append('frozen_c106_population_bound_before_raw_reference')
        public, private, members = [], [], []
        for raw in iter_reference_states(archives, manifest, split=SPLIT, scene_ids=SCENES):
            sid = raw['scene_id']
            members.append(sid)
            traces, raw_rows = trace_scene(raw, sid, [case for case in cases if case['scene'] == sid], args.runs,
                                           policies[sid], SCENES)
            public.extend(traces)
            private.extend(raw_rows)
        assert tuple(members) == SCENES and len(public) == 10
        private_dir = args.output/'private'
        private_dir.mkdir()
        write(private_dir/'raw-row-bindings.json.gz', private)
        write(args.output/'cases.json.gz', public)
        summary = []
        for item in public:
            role = item['roles']['NEAREST']
            windows = [dict(identity_roles=row['identity_roles'], crossing=row['crossing'], phase=row['phase'],
                            aligned_secant_mps=row['aligned_secant_mps'], aligned_fields=row['aligned_endpoint_fields_mps'],
                            aligned_path_mps=row['aligned_path_mps'], cross_source_full_mps=row['cross_source_full_mps'],
                            raw_max=row['crossing']['max_secant_mps'], aligned_integrated_field_mean_mps=row['aligned_integrated_field_mean_mps'])
                       for row in role['windows']]
            summary.append(dict(scene=item['scene'], group=item['group'], stratum=item['stratum'],
                                query_indices=item['query_indices'], net_5=role['native_speeds']['net_5'], windows=windows))
        write(args.output/'summary.json', dict(cases=summary, query_count=sum(len(x['query_indices']) for x in public),
              query_fields_checked=sum(x['query_cache_replayed_fields'] for x in public),
              provenance='raw in this audit is the shipped MTAD reference, not mother AIS'))
        record['stages'].append('raw_neighbor_geometry_and_reference_cache_independently_replayed')
        record.update(status='COMPLETED', exit_code=0, raw_reference_members_read=len(members), cases=len(public),
                      reused_measurement_bindings=len(members), reference_identity_windows=len(private))
    except BaseException as error:
        record.update(status='FAILED', exit_code=1, error_type=type(error).__name__)
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes(),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write(args.output/'run.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
    run(args)
