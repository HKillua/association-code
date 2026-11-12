"""C100: frozen C83 near-point field alignment and fixed interval diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from association.data.archive import MTADArchive
from association.data.physical_units import KNOT_TO_MPS
from tools.analysis.held_fusion_population import load_scene, read_evaluation, validate_measurement_archive
from tools.analysis.offline_endpoint_lineage import speed_between
from tools.analysis.same_source_swap_pilot import group_members, read

C83 = 'LEVEL1-HELD-SPEED-READOUT-20261112-attempt-1'
C94 = 'LEVEL1-V300-POSITION-FAR-20261112-attempt-1'
SCENES = ('1052', '1934', '2252', '2425', '2665', '2731', '3067', '3155', '3862', '4894')
WINDOWS = (1, 3, 5)


def write(path, value):
    with path.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')


def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p95=float(np.quantile(x, .95)) if len(x) else None,
                p99=float(np.quantile(x, .99)) if len(x) else None,
                maximum=float(x.max()) if len(x) else None,
                minimum=float(x.min()) if len(x) else None)


def independent_speed(left, right, dt):
    p0, p1 = np.radians(left), np.radians(right)
    a = np.sin((p1[..., 0]-p0[..., 0])/2)**2 + np.cos(p0[..., 0])*np.cos(p1[..., 0])*np.sin((p1[..., 1]-p0[..., 1])/2)**2
    return 12742000.*np.arcsin(np.sqrt(np.clip(a, 0., 1.)))/dt


def alignment(values, times):
    dt = np.diff(times)
    if not np.isfinite(values).all() or not np.isfinite(dt).all() or not (dt > 0).all():
        raise ValueError('finite native values and positive original intervals required')
    speed = speed_between(values[:-1, :2], values[1:, :2], dt)
    other = independent_speed(values[:-1, :2], values[1:, :2], dt)
    gap = float(np.max(np.abs(speed-other)))
    np.testing.assert_allclose(speed, other, rtol=0., atol=1e-8)
    fields = values[:, 2]*KNOT_TO_MPS
    forward, backward = np.abs(fields[:-1]-speed), np.abs(fields[1:]-speed)
    before, after = np.abs(fields[1:-1]-speed[:-1]), np.abs(fields[1:-1]-speed[1:])
    return dict(intervals=len(speed), dt_seconds=stats(dt),
                left_field_vs_following_secant_gap_mps=stats(forward),
                right_field_vs_preceding_secant_gap_mps=stats(backward),
                interior_points=len(before), interior_closer_following=int((after < before).sum()),
                interior_closer_preceding=int((before < after).sum()), interior_tied=int((before == after).sum()),
                max_distance_implementation_gap_mps=gap), (dt, speed, fields, forward, backward, before, after)


def describe_endpoint(scene, ev, node, side, public_id):
    query, times, _, offsets = scene['native']
    values = scene['values']
    lo, hi = map(int, offsets[node:node+2])
    t, v = times[lo:hi], values[lo:hi]
    summary, primitive = alignment(v, t)
    dt, speed, fields, _, _, _, _ = primitive
    end = 0 if side == 0 else len(t)-1
    result = dict(endpoint_id=public_id, side='first' if side == 0 else 'last', native_points=len(t),
                  source=f"S{np.searchsorted(np.unique(scene['sources']), scene['sources'][node])+1}",
                  field_kn=float(v[end, 2]), field_mps=float(fields[end]),
                  repeated_last_field_exact=bool(fields[-1] == fields[-2]),
                  terminal_field_mps=float(fields[-1]), penultimate_field_mps=float(fields[-2]),
                  terminal_backward_secant_mps=float(speed[-1]),
                  whole_fragment_alignment=summary, fixed_windows={}, endpoint_interval_rows=[])
    # Windows run inward from the actual selected boundary; no label-selected duration.
    for k in WINDOWS:
        if len(t) <= k:
            result['fixed_windows'][str(k)] = None
            continue
        left, right = (0, k) if side == 0 else (len(t)-1-k, len(t)-1)
        elapsed = float(t[right]-t[left])
        net = float(speed_between(v[left, :2], v[right, :2], np.asarray(elapsed)))
        other = float(independent_speed(v[left, :2], v[right, :2], elapsed))
        assert abs(net-other) <= 1e-8
        path = float(np.sum(speed[left:right]*dt[left:right])/elapsed)
        result['fixed_windows'][str(k)] = dict(intervals=k, dt_seconds=elapsed, net_mps=net,
            path_mean_mps=path, net_over_path=(net/path if path else None))
    band = range(min(5, len(speed))) if side == 0 else range(max(0, len(speed)-5), len(speed))
    for i in band:
        result['endpoint_interval_rows'].append(dict(left_seconds_from_endpoint=float(t[i]-t[end]),
            right_seconds_from_endpoint=float(t[i+1]-t[end]), dt_seconds=float(dt[i]),
            displacement_metres=float(speed[i]*dt[i]), secant_mps=float(speed[i]),
            left_field_mps=float(fields[i]), right_field_mps=float(fields[i+1])))
    candidates = []
    same_time = 0
    same_identity_source_members = 0
    for other_node in range(len(offsets)-1):
        if other_node == node or scene['sources'][other_node] != scene['sources'][node] or ev['identities'][other_node] != ev['identities'][node]:
            continue
        same_identity_source_members += 1
        a, b = map(int, offsets[other_node:other_node+2])
        same_time += int((times[a:b] == t[end]).sum())
        valid = np.flatnonzero(times[a:b] < t[end] if side == 0 else times[a:b] > t[end])
        if len(valid):
            index = a+int(valid[-1] if side == 0 else valid[0])
            candidates.append((abs(float(times[index]-t[end])), other_node, index))
    result.update(other_registered_same_identity_source_fragments=same_identity_source_members,
                  same_time_external_samples=same_time, external_neighbor=None)
    if candidates:
        elapsed, other_node, index = min(candidates)
        bridge = float(speed_between(v[end, :2], values[index, :2], np.asarray(elapsed)))
        result['external_neighbor'] = dict(dt_seconds=elapsed, dt_over_within_fragment_median=float(elapsed/np.median(dt)),
            displacement_metres=bridge*elapsed, bridge_secant_mps=bridge,
            endpoint_field_gap_mps=abs(float(fields[end])-bridge),
            neighbor_field_mps=float(values[index, 2]*KNOT_TO_MPS),
            matching_candidate_fragments=len(candidates),
            caveat='nearest available other registered same-identity same-source sample; missing interval continuity is not assumed')
    return result, primitive


def comparison(rows, label, baseline):
    selected = [x for x in rows if x['diagnostic_speeds'].get(label) is not None]
    errors = [abs(x['diagnostic_speeds'][label]-x['reference']) for x in selected]
    control = [abs(x['diagnostic_speeds'][baseline]-x['reference']) for x in selected]
    delta = np.asarray(errors)-np.asarray(control)
    return dict(points=len(selected), excluded_missing=len(rows)-len(selected), error_mps=stats(errors),
                baseline_error_mps=stats(control), mean_error_delta=float(delta.mean()) if len(delta) else None,
                improved_points=int((delta < 0).sum()), worsened_points=int((delta > 0).sum()),
                tied_points=int((delta == 0).sum()), max_harm_mps=float(max(0., delta.max())) if len(delta) else None,
                max_improvement_mps=float(max(0., -delta.min())) if len(delta) else None)


def run(args):
    began = time.monotonic()
    args.output.mkdir(parents=False, exist_ok=False)
    record = dict(status='RUNNING', idea_id='IDEA-C100-NEAR-SPEED-INTERVAL-20261112', plan_version='1.0',
        started_at=datetime.now(timezone.utc).isoformat(), time_policy=args.time_policy,
        affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        runs=str(args.runs), output=str(args.output), lineage_helper=str(args.lineage_helper),
        optimizer_updates=0, neural_fits=0, new_production_prediction_arms=0, test_opened=False)
    write(args.output/'run.json', record)
    try:
        assert record['affinity'] == [27] and record['cuda_visible_devices'] == ''
        spec = importlib.util.spec_from_file_location('c100_lineage', args.lineage_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        c83 = read(args.runs/C83/'run.json')
        c94 = read(args.runs/C94/'run.json')
        assert c83['status'] == c94['status'] == 'COMPLETED' and c83['exit_code'] == c94['exit_code'] == 0
        expected = read(args.runs/C94/'summary.json')['near']['ordered_by_production_abs_error']
        assert len(expected) == 21 and sorted({x['scene'] for x in expected}, key=int) == list(SCENES)
        frozen_keys = {(x['scene'], x['group'], x['query_index']): x for x in expected}
        points, endpoints, primitive = [], [], []
        fragment_count = 0
        with MTADArchive(c83['archive_paths']) as archive:
            validate_measurement_archive(args.runs, archive)
            for sid in SCENES:
                old_points = helper.near_points(args.runs, args.runs/C83/'private', [sid])
                scene = load_scene(args.runs, sid, archive)
                with np.load(args.runs/C83/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                    artifact = {key: saved[key] for key in saved.files}
                ev = read_evaluation(args.runs, sid, artifact['nodes'], artifact['query'])
                detail = helper.describe_scene(scene, artifact, ev, old_points)
                groups = group_members(scene['membership'])
                local_ids = {}
                fragment_ids = {}
                for point, row in zip(old_points, detail, strict=True):
                    key = (sid, row['group'], row['query_index'])
                    original = frozen_keys[key]
                    for actual, want in ((row['reference'], original['reference']), (row['base'], original['speed']['B']),
                        (row['candidate'], original['near_control_speed']), (row['error_change'], original['near_control_error_change'])):
                        np.testing.assert_equal(actual, want)
                    row['diagnostic_speeds'] = dict(B_field=row['base'], NEAR_field=row['candidate'])
                    row['endpoint_ids'] = {}
                    for role, old_label in (('B', 'first_member'), ('NEAR', 'selected_member')):
                        node = int(groups[row['group']][row[old_label]['member_slot']])
                        side = 0 if row[old_label]['endpoint_side'] == 'first' else 1
                        endpoint_key = (node, side)
                        if endpoint_key not in local_ids:
                            endpoint_id = f'{sid}-E{len(local_ids)+1:02d}'
                            endpoint, raw = describe_endpoint(scene, ev, node, side, endpoint_id)
                            endpoint['scene'] = sid
                            if node not in fragment_ids:
                                fragment_ids[node] = f'{sid}-F{len(fragment_ids)+1:02d}'
                                primitive.append(raw)
                                fragment_count += 1
                            endpoint['fragment_id'] = fragment_ids[node]
                            endpoints.append(endpoint)
                            local_ids[endpoint_key] = endpoint
                        endpoint = local_ids[endpoint_key]
                        row['endpoint_ids'][role] = endpoint['endpoint_id']
                        for k in WINDOWS:
                            window = endpoint['fixed_windows'][str(k)]
                            for measure in ('net', 'path_mean'):
                                row['diagnostic_speeds'][f'{role}_{measure}_{k}'] = window[measure+'_mps'] if window else None
                    target, q = point['target'], point['query_index']
                    row['reference_adjacent_Q_position_secants'] = []
                    for left in range(max(0, q-1), min(len(scene['query'])-1, q+1)):
                        if ev['target_valid'][target, left:left+2, :2].all():
                            elapsed = float(scene['query'][left+1]-scene['query'][left])
                            secant = float(speed_between(ev['target_states'][target, left, :2],
                                ev['target_states'][target, left+1, :2], np.asarray(elapsed)))
                            row['reference_adjacent_Q_position_secants'].append(dict(left_Q=left, right_Q=left+1,
                                dt_seconds=elapsed, position_secant_mps=secant))
                    points.append(row)
                record.update(completed_measurement_scenes=len({x['scene'] for x in points}))
                write(args.output/'run.json', record)
            reads = [(x['scene_id'], x['kind']) for x in archive.members_read]
            assert reads == [(sid, 'measurements') for sid in SCENES]
        assert len(points) == 21 and {(x['scene'], x['group'], x['query_index']) for x in points} == set(frozen_keys)
        assert sum(x['error_change'] < 0 for x in points) == 16 and sum(x['error_change'] > 0 for x in points) == 5
        all_dt, all_speed, all_fields, forward, backward, before, after = [np.concatenate([x[i] for x in primitive]) for i in range(7)]
        labels = list(points[0]['diagnostic_speeds'])
        strata = dict(all=points, same_identity=[x for x in points if x['selected_member']['same_identity_as_anchor']],
            different_identity=[x for x in points if not x['selected_member']['same_identity_as_anchor']],
            pure_group=[x for x in points if x['distinct_truth_identities_in_group'] == 1],
            mixed_group=[x for x in points if x['distinct_truth_identities_in_group'] > 1],
            before_group=[x for x in points if x['side'] == 'before_group'],
            after_group=[x for x in points if x['side'] == 'after_group'])
        comparisons = {name: {label: comparison(rows, label, 'B_field') for label in labels} for name, rows in strata.items()}
        own_comparisons = {label: comparison(points, label, 'NEAR_field' if label.startswith('NEAR') else 'B_field') for label in labels}
        result = dict(status='COMPLETED', points=21, scenes=len(SCENES), groups=len({(x['scene'], x['group']) for x in points}),
            distinct_endpoints=len(endpoints), distinct_fragments=fragment_count, whole_fragment_native_intervals=len(all_dt),
            alignment=dict(left_field_vs_following_secant_gap_mps=stats(forward),
                right_field_vs_preceding_secant_gap_mps=stats(backward), interior_points=len(before),
                interior_closer_following=int((after < before).sum()), interior_closer_preceding=int((before < after).sum()),
                interior_tied=int((after == before).sum()), dt_seconds=stats(all_dt)),
            terminal_counts=dict(last_side_endpoints=sum(x['side'] == 'last' for x in endpoints),
                repeated_last_field_on_last_side=sum(x['side'] == 'last' and x['repeated_last_field_exact'] for x in endpoints),
                external_neighbor_available=sum(x['external_neighbor'] is not None for x in endpoints)),
            comparisons_to_B= comparisons, comparisons_to_same_supplier_field=own_comparisons,
            components_scope='fixed previously used V48; 21 correlated Q points, deduplicated endpoint evidence; no independent confirmation',
            units='speed m/s except explicit kn; time seconds; interval lengths actual, no fixed one-second assumption')
        write(args.output/'points.json', points)
        write(args.output/'endpoints.json', endpoints)
        write(args.output/'summary.json', result)
        record.update(status='COMPLETED', exit_code=0, completed_points=len(points), completed_endpoints=len(endpoints),
            raw_measurement_members_read=len(SCENES), raw_label_members_read=0, raw_reference_members_read=0,
            max_independent_speed_gap_mps=max(x['whole_fragment_alignment']['max_distance_implementation_gap_mps'] for x in endpoints))
    except BaseException as error:
        record.update(status='FAILED', exit_code=1, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-began,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(args.output/'run.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lineage-helper', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    run(args)
