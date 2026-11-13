"""104: bounded audit of three fixed cases in retained 103 evidence.

No archive, NPZ, checkpoint or model is opened. Outputs contain indices,
relative coordinates and derived quantities, never raw identities/coordinates.
Consistency findings are preserved; they do not identify physical ground truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from tools.analysis.diagnose_three_scenes import clean, local_delta, radial, read, write

CASES = (('3035', 23, 56), ('2819', 1, 60), ('1043', 3, 54))
QUERY_COUNT = 64
FIELDS = ('lat', 'lon', 'vel', 'cou')
LIMITATIONS = [
    'Measurements retained by 103 are canonical native observations, not untouched CSV rows.',
    'Measurement CSV to canonical processing cannot be independently reconstructed here.',
    'Original labels and reference may themselves be wrong; consistent replay does not establish truth.',
    'Three cases were selected after seeing tails; neighbors are fixed by index, not by error.',
    'No new predictor, calibration, matching, threshold or performance claim is produced.',
    'Method prediction vectors come from 103 public evidence; no independent model/NPZ replay.',
]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def compare(checks, name, actual, expected, *, atol=1e-8, rtol=1e-9, **where):
    """Record a mismatch without discarding the rest of the bounded evidence."""
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    same_shape = a.shape == b.shape
    equal = same_shape and bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
    finite = np.isfinite(a) & np.isfinite(b) if same_shape else np.array([], bool)
    checks.append(dict(check=name, matches=equal, absolute_tolerance=atol,
                       relative_tolerance=rtol,
                       max_absolute_difference=float(np.max(np.abs(a[finite]-b[finite])))
                       if same_shape and finite.any() else None, **where))
    return equal


def fact(checks, name, value, **where):
    checks.append(dict(check=name, matches=bool(value), **where))


def position_at(reference, identity, timestamp, policy):
    """Independent position-only calculation from raw rows, including duplicates.

    All finite reference times delimit support, even if their position is invalid.
    A conflict invalidates that feature; no averaging, repair or gap bridging.
    """
    groups = defaultdict(list)
    for row in reference['rows']:
        t = number(row['time'])
        if row['identity_id'] == identity and np.isfinite(t):
            groups[t-float(policy['time_offset_seconds'])].append(row)
    times = sorted(groups)
    right = int(np.searchsorted(times, timestamp))
    if right < len(times) and times[right] == timestamp:
        left, kind = right, 'exact'
    elif 0 < right < len(times):
        left = right-1
        kind = 'interior' if times[right]-times[left] <= policy['max_gap_seconds'] else 'gap_exceeded'
    else:
        return np.full(2, np.nan), dict(kind='outside_support' if times else 'no_reference_times',
            alpha=None, gap_seconds=None, endpoints=[]), []
    endpoints, points, rows_used = [], [], []
    for index in (left, right):
        rows = groups[times[index]]
        values = np.array([[number(v) for v in row['state']] for row in rows])
        usable = (np.isfinite(values) & (np.abs(values) <= [90., 180., np.inf, 360.])
                  & (values >= [-90., -180., 0., 0.]))
        same = np.all(values == values[0], axis=0)
        valid = usable.all(0) & same
        points.append(np.where(valid[:2], values[0, :2], np.nan))
        endpoints.append(dict(aligned_time_seconds=float(times[index]),
            raw_time_seconds=float(times[index]+policy['time_offset_seconds']),
            raw_row_count=len(rows), raw_row_lines=[row['line'] for row in rows],
            conflicting_fields=[field for j, field in enumerate(FIELDS)
                                if len(rows) > 1 and usable[:, j].all() and not same[j]],
            unusable_fields=[field for j, field in enumerate(FIELDS) if not usable[:, j].all()],
            position_valid=valid[:2].tolist()))
        rows_used.append(rows)
    gap = times[right]-times[left]
    alpha = (timestamp-times[left])/gap if gap else 0.
    a, b = points
    delta = b-a
    if abs(delta[1]) > 180.:
        delta[1] = (delta[1]+180.) % 360.-180.
    result = a+alpha*delta
    if np.isfinite(result[1]) and abs(b[1]-a[1]) > 180.:
        result[1] = (result[1]+180.) % 360.-180.
    if kind == 'gap_exceeded':
        result[:] = np.nan
    detail = dict(kind=kind, alpha=float(alpha), gap_seconds=float(gap), endpoints=endpoints,
                  endpoint_position_step_m=float(radial(a, b)) if np.isfinite([a, b]).all() else None)
    detail['endpoint_position_step_mps'] = detail['endpoint_position_step_m']/gap if gap and detail['endpoint_position_step_m'] is not None else None
    return result, detail, rows_used


def check_raw_rows(rows_used, checks, **where):
    seen = set()
    for rows in rows_used:
        for row in rows:
            if row['line'] in seen:
                continue
            seen.add(row['line'])
            raw = row['raw']
            fact(checks, 'reference_raw_identity_parse', raw['MMSI'].strip() == row['identity_id'],
                 reference_line=row['line'], **where)
            compare(checks, 'reference_raw_numeric_parse',
                [number(raw[field]) for field in ('time', *FIELDS)],
                [number(row['time']), *[number(v) for v in row['state']]],
                atol=0., rtol=0., reference_line=row['line'], **where)


def native_position(times, positions, query):
    """Independent least-squares/interpolation expression for canonical OLS."""
    t, xy = np.asarray(times, float), np.asarray(positions, float)
    change = np.diff(xy[:, 1])
    short = (change+180.) % 360.-180.
    short = np.where(np.abs(short) == 180., np.copysign(180., change), short)
    longitude = xy[0, 1]+np.r_[0., np.cumsum(short)]
    unwrapped = np.column_stack((xy[:, 0], longitude))
    if t[0] <= query <= t[-1]:
        right = int(np.searchsorted(t, query))
        if t[right] == query:
            return xy[right].copy()
        value = np.array([np.interp(query, t, unwrapped[:, j]) for j in range(2)])
    else:
        design = np.column_stack((t-t.mean(), np.ones(len(t))))
        slope = np.linalg.lstsq(design, unwrapped, rcond=None)[0][0]
        end = 0 if query < t[0] else -1
        value = xy[end]+(query-t[end])*slope
        value[0] = np.clip(value[0], -90., 90.)
    value[1] = (value[1]+180.) % 360.-180.
    return value


def audit_case(scene, public, private, target_index, center_query):
    sid = str(scene['scene_id'])
    if private['raw_reference']['scene_id'] != sid or private['raw_reference']['split'] != 'V_select':
        raise ValueError('Private reference scene/split is outside this case')
    points = sorted((r for r in public['points'] if r['target_index'] == target_index),
                    key=lambda r: r['query_index'])
    if not points or [r['query_index'] for r in points] != list(range(len(points))):
        raise ValueError('Case target must retain its complete contiguous query population')
    if not 0 <= center_query < len(points):
        raise ValueError('Fixed case center is missing')
    grid = np.array([r['query_seconds'] for r in points], float)
    if not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError('Case query grid must be finite and increasing')
    chosen = points[max(0, center_query-2):min(len(points), center_query+3)]
    ref_records = [r for r in private['reference_endpoints'] if r['target_index'] == target_index]
    if len(ref_records) != 1:
        raise ValueError('Case target reference mapping is ambiguous/missing')
    target = ref_records[0]['target_identity']
    reference, policy = private['raw_reference'], scene['alignment_policy']
    latitude_origin = scene['chart']['latitude_origin_degrees']
    tracks = {r['node_index']: r for r in private['input_tracks']}
    checks, rows, curve = [], [], []
    center_state, _, _ = position_at(reference, target, grid[center_query], policy)
    if not np.isfinite(center_state).all():
        raise ValueError('Fixed case center lacks finite reference position for relative chart')
    fact(checks, 'target_identity_parse_has_two_fields', len(target.split('-')) == 2)
    fact(checks, '103_reference_replay_reported', scene['reference_replay_verified'])
    fact(checks, '103_input_replay_reported', scene['raw_replay_verified'])
    alignment_report = ref_records[0]['alignment_report']
    for name in ('time_offset_seconds', 'max_gap_seconds'):
        compare(checks, 'retained_alignment_'+name, alignment_report[name], policy[name], atol=0., rtol=0.)
    fact(checks, 'retained_reference_longitude_policy', alignment_report['longitude_policy'] == 'shortest_arc')
    fact(checks, 'retained_reference_time_policy',
         alignment_report['time_policy'] == 'aligned_reference_time = raw_reference_time - time_offset_seconds')
    compare(checks, 'private_motion_query_grid', private['input_motion']['query_seconds'], grid)
    events_by_node, reference_plot_rows = defaultdict(dict), {}
    fixed_members = None
    for point in chosen:
        qi, qt = point['query_index'], point['query_seconds']
        fact(checks, 'fixed_group_across_query_window', point['group_index'] == points[center_query]['group_index'], query_index=qi)
        truth, detail, raw_rows = position_at(reference, target, qt, policy)
        check_raw_rows(raw_rows, checks, query_index=qi)
        valid = bool(np.isfinite(truth).all())
        fact(checks, 'reference_position_valid', valid == point['truth_valid'], query_index=qi)
        fact(checks, 'reference_kind', detail['kind'] == point['reference_kind'], query_index=qi)
        recorded = ref_records[0]['endpoints'][qi]
        fact(checks, 'retained_endpoint_kind', recorded['kind'] == detail['kind'], query_index=qi)
        for endpoint, side, raw in zip(detail['endpoints'], ('left', 'right'), raw_rows):
            kept = recorded[side]
            fact(checks, 'retained_endpoint_rows_equal_raw_reference', kept is not None and kept['rows'] == raw,
                 query_index=qi, side=side)
            compare(checks, 'retained_endpoint_time', kept['aligned_time'] if kept else None,
                    endpoint['aligned_time_seconds'], query_index=qi, side=side)
            for row in raw:
                p = np.array([number(v) for v in row['state'][:2]])
                reference_plot_rows[row['line']] = dict(line=row['line'],
                    time_from_center_seconds=number(row['time'])-policy['time_offset_seconds']-grid[center_query],
                    relative_NE_m=local_delta(p, center_state, latitude_origin))
        compare(checks, 'public_reference_endpoint_times',
                point['reference_endpoint_times'], [x['aligned_time_seconds'] for x in detail['endpoints']]
                if detail['endpoints'] else [None, None], query_index=qi)
        compare(checks, 'public_reference_endpoint_counts', point['reference_endpoint_row_counts'],
                [x['raw_row_count'] for x in detail['endpoints']] if detail['endpoints'] else [0, 0], query_index=qi)
        reference_ne = local_delta(truth, center_state, latitude_origin)
        source_rows = [r for r in public['query_sources'] if r['query_index'] == qi
            and r['bound_target_index'] == target_index and r['selected_unique_group']]
        if len(source_rows) != point['group_size'] or len({r['node_index'] for r in source_rows}) != len(source_rows):
            raise ValueError('Selected source population does not match fixed group size')
        members = {r['node_index'] for r in source_rows}
        if fixed_members is None:
            fixed_members = members
        fact(checks, 'fixed_members_across_query_window', members == fixed_members, query_index=qi)
        source_out, consistent = [], 0
        for source in source_rows:
            ni = source['node_index']; track = tracks[ni]
            where = dict(query_index=qi, node_index=ni)
            identity, node = track['identity_id'], track['node_id']
            raw_label = private['input_audit']['valid_raw_mmsi'][node]
            fact(checks, 'raw_label_to_identity_parse', '-'.join(raw_label.split('-')[:2]) == identity, **where)
            fact(checks, 'node_source_batch_parse', node == f"{sid}:{track['source']}:{track['batch']}", **where)
            fact(checks, 'source_mapping', source['source'] == track['source'], **where)
            fact(checks, 'fixed_group_binding', source['group_index'] == point['group_index'], **where)
            consistent += identity == target
            times, values = np.asarray(track['canonical_times'], float), np.asarray(track['canonical_values'], float)
            if len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any() or values.shape != (len(times), 4):
                raise ValueError('Canonical time/value schema is invalid')
            if not np.isfinite(values).all() or (np.abs(values[:, :2]) > [90, 180]).any():
                raise ValueError('Canonical positions are invalid')
            selected = np.asarray(track['selected_indices'], int)
            compare(checks, 'selected_times_are_original_canonical_indices', track['times'], times[selected], **where)
            compare(checks, 'selected_values_are_original_canonical_indices', track['values'], values[selected], **where)
            nearest = int(np.argmin(np.abs(times-qt)))
            compare(checks, 'nearest_native_time', source['nearest_event_time'], times[nearest], **where)
            compare(checks, 'nearest_native_distance_seconds', source['nearest_seconds'], abs(times[nearest]-qt), **where)
            fact(checks, 'visible_support', source['support'] == bool(times[0] <= qt <= times[-1]), **where)
            estimate = native_position(times, values[:, :2], qt)
            cached = np.asarray(private['input_motion']['states'][ni][qi], float)[:2]
            compare(checks, 'canonical_to_retained_query_position_degrees', estimate, cached, atol=1e-9, rtol=0., **where)
            compare(checks, 'source_query_to_bound_reference_NE_m', local_delta(estimate, truth, latitude_origin),
                    source['query_estimate_to_bound_reference_NE_m'], atol=1e-5, **where)
            compare(checks, 'source_query_to_bound_reference_radial_m', radial(estimate, truth) if valid else None,
                    source['query_estimate_to_bound_reference_error_m'], atol=1e-5, **where)
            own_ref, _, _ = position_at(reference, identity, times[nearest], policy)
            nearest_error = radial(values[nearest], own_ref) if np.isfinite(own_ref).all() else None
            compare(checks, 'nearest_native_to_own_time_reference_m', nearest_error,
                    source['nearest_event_to_own_reference_error_m'], atol=1e-5, **where)
            compare(checks, 'source_query_minus_nearest_native_NE_m', local_delta(estimate, values[nearest], latitude_origin),
                    source['query_estimate_minus_nearest_event_NE_m'], atol=1e-5, **where)
            local_events = []
            for event in range(max(0, nearest-1), min(len(times), nearest+2)):
                own_at_event, own_detail, own_rows = position_at(reference, identity, times[event], policy)
                check_raw_rows(own_rows, checks, node_index=ni, event_index=event)
                # Only the selected nearest±1 window contributes step endpoints.
                previous = event-1 if event > max(0, nearest-1) else None
                step_seconds = times[event]-times[previous] if previous is not None else None
                step_m = float(radial(values[event], values[previous])) if previous is not None else None
                item = dict(event_index=event, time_from_center_seconds=times[event]-grid[center_query],
                    nearest=event == nearest, relative_NE_m=local_delta(values[event], center_state, latitude_origin),
                    own_time_reference_kind=own_detail['kind'],
                    own_time_reference_error_m=float(radial(values[event], own_at_event)) if np.isfinite(own_at_event).all() else None,
                    previous_event_index=previous, previous_step_seconds=step_seconds, previous_step_m=step_m,
                    previous_step_mps=step_m/step_seconds if step_seconds else None)
                local_events.append(item); events_by_node[ni][event] = item
            source_out.append(dict(node_index=ni, source=source['source'], identity_matches_bound_target=identity == target,
                support=source['support'], nearest_event_index=nearest, nearest_seconds=abs(times[nearest]-qt),
                nearest_native_to_own_time_reference_error_m=nearest_error,
                source_query_to_bound_reference_error_m=float(radial(estimate, truth)) if valid else None,
                source_query_minus_nearest_native_distance_m=float(radial(estimate, values[nearest])),
                query_relative_NE_m=local_delta(estimate, center_state, latitude_origin), events=local_events))
        compare(checks, 'label_consistent_member_count', consistent, point['identity_consistent_members'], query_index=qi)
        # Method NE vectors are already derived in 103, not an independent model replay.
        methods = {name: dict(error_m=m['error_m'], error_NE_m=m['error_NE_m'],
                    relative_NE_m=reference_ne+np.asarray(m['error_NE_m'], float))
                   for name, m in point['methods'].items()}
        rows.append(dict(query_index=qi, center=qi == center_query, query_seconds=qt,
            time_from_center_seconds=qt-grid[center_query], group_index=point['group_index'],
            reference_valid=valid, reference=detail, sources=source_out, methods=methods))
        curve.append(dict(query_index=qi, time_from_center_seconds=qt-grid[center_query],
                          reference_relative_NE_m=reference_ne, methods=methods))
    return clean(dict(scene_id=sid, target_index=target_index, center_query_index=center_query,
        query_indices=[r['query_index'] for r in chosen], checks=checks,
        consistency_status='MATCHES_WITHIN_TOLERANCE' if all(c['matches'] for c in checks) else 'MISMATCH_FOUND',
        failed_check_count=sum(not c['matches'] for c in checks), rows=rows,
        chart=dict(axes=['north', 'east'], unit='m', origin='center query aligned reference; absolute coordinate withheld',
                   projection='103 fixed scene cosine latitude; relative diagnostic plane, not official radial scoring'),
        curve=dict(query=curve, reference_raw_endpoints=sorted(reference_plot_rows.values(), key=lambda r: r['time_from_center_seconds']),
                   native_events=[dict(node_index=n, events=[events[k] for k in sorted(events)]) for n, events in events_by_node.items()]),
        earliest_observable_relation='Canonical native observations and raw-reference alignment are both retained; compare their errors and steps in rows. Their physical correctness remains unknown.',
        limitations=LIMITATIONS))


def render_case(case, path):
    """Optional plotting: only already-deidentified audit output is accepted."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    query = case['curve']['query']
    tx = [r['time_from_center_seconds'] for r in query]
    ref = np.asarray([r['reference_relative_NE_m'] for r in query], float)
    axes[0, 0].plot(ref[:, 1], ref[:, 0], 'ks-', label='aligned reference')
    for column, ax in enumerate((axes[0, 1], axes[1, 0])):
        ax.plot(tx, ref[:, column], 'ks-', label='aligned reference')
    raw = case['curve']['reference_raw_endpoints']
    for column, ax in enumerate((axes[0, 1], axes[1, 0])):
        ax.scatter([r['time_from_center_seconds'] for r in raw],
                   [r['relative_NE_m'][column] for r in raw], marker='x', color='black', label='raw reference endpoints')
    for node in case['curve']['native_events']:
        events = node['events']; xy = np.asarray([e['relative_NE_m'] for e in events], float)
        label = f"canonical node {node['node_index']}"
        axes[0, 0].scatter(xy[:, 1], xy[:, 0], s=18, label=label)
        for col, ax in enumerate((axes[0, 1], axes[1, 0])):
            ax.scatter([e['time_from_center_seconds'] for e in events], xy[:, col], s=18, label=label)
    for method in query[0]['methods']:
        xy = np.asarray([r['methods'][method]['relative_NE_m'] for r in query], float)
        axes[0, 0].plot(xy[:, 1], xy[:, 0], '--o', markersize=3, label=method)
        for col, ax in enumerate((axes[0, 1], axes[1, 0])):
            ax.plot(tx, xy[:, col], '--o', markersize=3, label=method)
        axes[1, 1].plot(tx, [r['methods'][method]['error_m'] for r in query], '-o', label=method)
    axes[0, 0].set(xlabel='Relative east (m)', ylabel='Relative north (m)', aspect='equal')
    axes[0, 1].set(xlabel='Seconds from center query', ylabel='Relative north (m)')
    axes[1, 0].set(xlabel='Seconds from center query', ylabel='Relative east (m)')
    axes[1, 1].set(xlabel='Seconds from center query', ylabel='Original radial error (m)', ylim=(0., None))
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(f"Scene {case['scene_id']} target {case['target_index']} query {case['center_query_index']} ±2 | retained 103 evidence")
    fig.savefig(path, metadata={'Date': None})
    plt.close(fig)


def report(cases):
    lines = ['# 104 fixed-case chain audit', '',
        'Positions/identities are withheld. Complete numeric evidence is in case JSON. '
        'Consistency is an implementation finding, not physical truth.', '']
    for case in cases:
        lines += [f"## Scene {case['scene_id']}, target {case['target_index']}, center query {case['center_query_index']}", '',
            f"Consistency: {case['consistency_status']}; {case['failed_check_count']} mismatches. Plot: {case.get('plot_status', 'not requested')}.", '',
            '| query | reference kind / gap s / alpha | node | support | nearest native Δt s | native vs own-time ref m | query vs bound ref m | 090C m |',
            '|---|---|---|---|---|---|---|---|']
        def fmt(value):
            return 'unavailable' if value is None else f'{value:.6g}'
        for row in case['rows']:
            detail = row['reference']
            for src in row['sources']:
                lines.append(f"| {row['query_index']} | {detail['kind']} / {fmt(detail['gap_seconds'])} / {fmt(detail['alpha'])} | {src['node_index']} | {src['support']} | {fmt(src['nearest_seconds'])} | {fmt(src['nearest_native_to_own_time_reference_error_m'])} | {fmt(src['source_query_to_bound_reference_error_m'])} | {fmt(row['methods']['learned_sequence']['error_m'])} |")
        lines += ['', f"![Relative diagnostic curves](case-{case['scene_id']}.svg)", '']
    lines += ['## Limits', '', *['- '+item for item in LIMITATIONS], '']
    return '\n'.join(lines)


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); source = Path(args.source_run).resolve()
    state = dict(status='RUNNING', idea_id='IDEA-STATE-DIAG-20261027', plan_version='1.4-chain-audit',
        argv=sys.argv, start_utc=datetime.now(timezone.utc).isoformat(), source_run=str(source),
        cases=[dict(scene_id=s, target_index=t, center_query=q) for s, t, q in CASES],
        completed_cases=[], optimizer_updates=0, model_forwards=0, checkpoint_files_read=0,
        archive_files_read=0, npz_files_read=0, test_opened=False, groups_recomputed=False, threshold_reselected=False)
    write(out/'run.json', state)
    try:
        prior = read(source/'run.json')
        if (prior['status'] != 'COMPLETED' or prior['completed_scenes'] != [x[0] for x in CASES]
                or prior['test_opened'] is not False or prior['model_forwards'] != 0):
            raise ValueError('Source must be the completed bounded 103 diagnosis')
        config = prior['source_metadata']['config']
        if (config['data']['reference_longitude_policy'] != 'shortest_arc'
                or config['motion_observation_policy'] != 'canonical_all' or config['data']['length'] != QUERY_COUNT):
            raise ValueError('104 expressions apply only to the retained shortest_arc/canonical_all/Q64 contract')
        cases = []
        for sid, target, query in CASES:
            public = read(source/f'points-{sid}.json.gz')
            case_points = [r for r in public['points'] if r['target_index'] == target]
            if sorted(r['query_index'] for r in case_points) != list(range(QUERY_COUNT)):
                raise ValueError('Fixed target does not retain the declared Q64 population')
            case = audit_case(read(source/f'scene-{sid}.json.gz'), public,
                              read(source/f'private-evidence-{sid}.json.gz'), target, query)
            # Commit numeric evidence before optional plotting can fail.
            write(out/f'case-{sid}.json.gz', case)
            try:
                render_case(case, out/f'case-{sid}.svg')
                case['plot_status'] = 'COMPLETE'
            except ImportError as error:
                case['plot_status'] = 'MISSING_DEPENDENCY: '+type(error).__name__
            except Exception as error:
                case['plot_status'] = 'FAILED: '+type(error).__name__
            # gzip helper is exclusive; final metadata lives in summary/run.
            cases.append(case); state['completed_cases'].append(sid)
            state['elapsed_seconds'] = time.monotonic()-started
            write(out/'run.json', state)
        summary = dict(status='COMPLETED', cases=cases, limitations=LIMITATIONS,
                       consistency_mismatch_count=sum(c['failed_check_count'] for c in cases),
                       plots_complete=all(c['plot_status'] == 'COMPLETE' for c in cases))
        write(out/'summary.json.gz', summary)
        (out/'report.md').write_text(report(cases), encoding='utf-8')
        state.update(status='COMPLETED' if not summary['consistency_mismatch_count'] else 'COMPLETED_WITH_MISMATCHES',
            consistency_mismatch_count=summary['consistency_mismatch_count'], plots_complete=summary['plots_complete'])
    except BaseException as error:
        # A malformed private field can embed an identity/coordinate in an
        # exception message. Keep the traceback in server stderr, not public JSON.
        state.update(status='FAILED', error_type=type(error).__name__,
                     error='Schema/read failure; original details retained in server stderr.')
        raise
    finally:
        state.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                     peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     peak_rss_unit='bytes' if sys.platform == 'darwin' else 'KiB')
        write(out/'run.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run')
    parser.add_argument('--output', required=True)
    parser.add_argument('--render-case', help='Render existing deidentified case JSON to --output SVG; no raw input')
    args = parser.parse_args()
    if args.render_case:
        if args.source_run or Path(args.output).exists():
            parser.error('render-only requires a fresh output SVG and no --source-run')
        render_case(read(args.render_case), Path(args.output))
        return
    if not args.source_run:
        parser.error('--source-run is required for the bounded audit')
    def stop(signum, frame):
        raise InterruptedError(f'signal {signum}; preserve partial output, no retry')
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == '__main__':
    main()
