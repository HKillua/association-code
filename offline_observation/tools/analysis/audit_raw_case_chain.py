"""105: compare three selected raw CSV chains with retained 103/104 evidence.

This CPU diagnostic reads complete compressed members only for three fixed V
scenes. Raw strings, identities and absolute coordinates remain server-private.
No model, checkpoint, NPZ, threshold or test-set payload is read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import math
from pathlib import Path
import re
import resource
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from association.data.archive import HEADERS, MTADArchive, _name, validate_dataset_level, validate_manifest
from association.data.reference_states import REFERENCE_HEADER, _read_rows
from tools.analysis.diagnose_three_scenes import clean, radial, read, write

CASES = (('3035', 23, 56), ('2819', 1, 60), ('1043', 3, 54))
FIELDS = ('time', 'lat', 'lon', 'vel', 'cou')
MAX_UNCOMPRESSED_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ADDRESS_SPACE_BYTES = 8 * 1024 * 1024 * 1024
QUERY_COUNT = 64


def raw_rows(payload, header, keep_keys=None):
    """Retain CSV reader line numbers and original strings, including repeats."""
    try:
        content = payload.decode('utf-8-sig')
        encoding = 'utf-8-sig'
    except UnicodeDecodeError:
        content = payload.decode('gb18030')
        encoding = 'gb18030'
    reader = csv.DictReader(io.StringIO(content, newline=''), strict=True)
    if reader.fieldnames != header:
        raise ValueError('CSV header differs from declared data contract')
    rows = []
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError('CSV row has wrong column count')
        stripped = {k: v.strip() for k, v in row.items()}
        if keep_keys is None or (stripped['source'], stripped['batch']) in keep_keys:
            rows.append(dict(line=reader.line_num, raw=dict(row), stripped=stripped))
    return rows, encoding


def reference_member(archive, sid):
    found = []
    for info in archive._zip.infolist():
        name = _name(info)
        if not info.is_dir() and re.search(r'(?:^|/)真实场景/场景-'+re.escape(sid)+r'\.csv$', name):
            found.append((info, name))
    if len(found) != 1:
        raise ValueError('Reference member missing or ambiguous')
    return found[0]


def read_members(archive, sid, journal, checkpoint=None):
    """Exactly labels/measurements/reference for this sid; no other payload."""
    pairs = {kind: archive._members[kind][sid] for kind in ('labels', 'measurements')}
    pairs['reference'] = reference_member(archive, sid)
    result = {}
    for kind in ('labels', 'measurements', 'reference'):
        info, name = pairs[kind]
        entry = dict(scene_id=sid, kind=kind, member=name, compressed_bytes=info.compress_size,
                     uncompressed_bytes=info.file_size, status='PENDING')
        journal.append(entry)
        if checkpoint is not None:
            checkpoint()
        if info.file_size > MAX_UNCOMPRESSED_MEMBER_BYTES:
            entry['status'] = 'REJECTED_SIZE_LIMIT'
            if checkpoint is not None:
                checkpoint()
            raise ValueError('Fixed-scene member exceeds predeclared uncompressed byte ceiling')
        payload = archive._zip.read(info)
        result[kind] = payload
        entry.update(actual_uncompressed_bytes=len(payload), status='READ')
        if checkpoint is not None:
            checkpoint()
    return result


def node_check(sid, track, labels, measurements, wanted_events, center_time, expected_raw_mmsi):
    """Independently reconstruct the original reader's retained-node semantics."""
    source, batch = track['source'], track['batch']
    lbl = [r for r in labels if (r['stripped']['source'], r['stripped']['batch']) == (source, batch)]
    rows = [r for r in measurements if (r['stripped']['source'], r['stripped']['batch']) == (source, batch)]
    identities = {r['stripped']['mmsi'] for r in lbl}
    expected_identity = track['identity_id']
    labels_ok = (len(identities) == 1 and all(len(x.split('-')) == 3 and all(x.split('-'))
        and '-'.join(x.split('-')[:2]) == expected_identity and x == expected_raw_mmsi for x in identities))
    intervals, invalid_label_lines = [], []
    for entry in lbl:
        try:
            start, end = float(entry['stripped']['t_s']), float(entry['stripped']['t_e'])
            if not (math.isfinite(start) and math.isfinite(end) and start <= end):
                raise ValueError('invalid interval')
            intervals.append((start, end))
        except (ValueError, OverflowError):
            invalid_label_lines.append(entry['line'])
    node_ok = track['node_id'] == f'{sid}:{source}:{batch}'
    parsed, bad = [], []
    for row in rows:
        try:
            values = tuple(float(row['stripped'][field]) for field in FIELDS)
            if not all(math.isfinite(x) for x in values):
                raise ValueError('nonfinite')
        except (ValueError, OverflowError):
            bad.append(row['line'])
            continue
        parsed.append((row['line'], values))
    values = [item[1] for item in parsed]
    unique = sorted(set(values))
    timestamp_counts = Counter(x[0] for x in unique)
    conflicting_times = [t for t, n in timestamp_counts.items() if n > 1]
    illegal = [line for line, (t, lat, lon, vel, cou) in parsed
               if vel < 0 or not 0 <= cou <= 360 or not -90 <= lat <= 90 or not -180 <= lon <= 180]
    prior_times = [item[1][0] for item in parsed]
    array = np.asarray(unique, dtype=np.float64).reshape(-1, 5)
    if len(array):
        array[:, 4] %= 360.
    expected = np.column_stack((track['canonical_times'], track['canonical_values']))
    same = bool(array.shape == expected.shape and np.array_equal(array, expected))
    eligible = bool(labels_ok and node_ok and rows and not bad and not illegal and not invalid_label_lines
                    and not conflicting_times and len(timestamp_counts) >= 2)
    event_rows = []
    for index in sorted(set(wanted_events)):
        if not 0 <= index < len(expected):
            raise ValueError('104 fixed event index outside retained canonical sequence')
        wanted = expected[index].copy()
        matching = []
        same_time = []
        for line, values_at_line in parsed:
            normalized = np.asarray(values_at_line, float)
            normalized[4] %= 360.
            if values_at_line[0] == wanted[0]:
                same_time.append(line)
                if np.array_equal(normalized, wanted):
                    matching.append(line)
        dt = wanted[0]-expected[index-1, 0] if index else None
        step = float(radial(wanted[1:3], expected[index-1, 1:3])) if index else None
        event_rows.append(dict(event_index=index, time_from_center_seconds=float(wanted[0]-center_time),
            matching_raw_lines=matching, all_same_time_raw_lines=same_time,
            matching_row_count=len(matching), matched=bool(matching), previous_step_seconds=dt,
            previous_step_m=step, previous_step_mps=step/dt if dt else None,
            previous_step_source='103 retained canonical positions; raw rows separately matched'))
    result = dict(node_index=track['node_index'],
        raw_measurement_row_count=len(rows), label_row_count=len(lbl), identity_mapping_matches=labels_ok,
        node_mapping_matches=node_ok, invalid_label_interval_lines=invalid_label_lines,
        measurement_rows_outside_label_intervals=sum(not any(a <= values_at_line[0] <= b for a, b in intervals)
                                                      for _, values_at_line in parsed) if intervals else None,
        parse_invalid_lines=bad, invalid_state_lines=illegal,
        identical_duplicate_count=len(values)-len(unique), conflicting_time_count=len(conflicting_times),
        original_time_decrease_count=sum(b < a for a, b in zip(prior_times[:-1], prior_times[1:])),
        raw_reconstructed_canonical_rows=len(array), retained_canonical_rows=len(expected),
        reader_would_keep_node=eligible, full_canonical_exact=same,
        all_fixed_events_match=all(event['matched'] for event in event_rows), events=event_rows)
    private = dict(node_index=track['node_index'], source=source, batch=batch,
        identity_id=expected_identity, labels=lbl, raw_measurements=rows,
        retained_canonical_times=track['canonical_times'], retained_canonical_values=track['canonical_values'])
    return result, private


def reference_check(payload, private, target_index, case, center, sid):
    """Compare original reference rows with 103 private rows and retain neighbors."""
    raw, issues = _read_rows(payload, 'fixed reference member')
    refs = [x for x in private['reference_endpoints'] if x['target_index'] == target_index]
    if len(refs) != 1:
        raise ValueError('Fixed reference target missing or ambiguous')
    ref = refs[0]
    identity = ref['target_identity']
    current = [x for x in raw if x['identity_id'] == identity]
    retained = [x for x in private['raw_reference']['rows'] if x['identity_id'] == identity]
    raw_identical = clean(current) == clean(retained)
    times = sorted({r['time'] for r in current if r['time'] is not None and math.isfinite(r['time'])})
    offset = ref['alignment_report']['time_offset_seconds']
    center_time = next(r['query_seconds'] for r in case['rows'] if r['query_index'] == center)
    window_queries = [center] + ([55] if sid == '1043' else [])
    selected_times, endpoint_checks = set(), []
    for query in window_queries:
        endpoint = ref['endpoints'][query]
        for side in ('left', 'right'):
            item = endpoint[side]
            if item is None:
                endpoint_checks.append(dict(query_index=query, side=side, status='no_endpoint', matches=None))
                continue
            if not item['rows']:
                raise ValueError('Retained reference endpoint has no original row')
            target_time = item['rows'][0]['time']
            matches = [r for r in current if r['time'] == target_time]
            endpoint_checks.append(dict(query_index=query, side=side, status=endpoint['kind'],
                matches=(clean(matches) == clean(item['rows'])
                         and target_time-offset == item['aligned_time']), raw_row_lines=[r['line'] for r in matches]))
            if target_time in times:
                i = times.index(target_time)
                selected_times.update(times[max(i-1, 0):min(i+2, len(times))])
    selected = [r for r in current if r['time'] in selected_times]
    groups = defaultdict(list)
    for row in selected:
        groups[row['time']].append(row)
    blocks = []
    previous = None
    for timestamp in sorted(groups):
        rows = groups[timestamp]
        states = [r['state'] for r in rows]
        complete = all(isinstance(x, (float, int)) and math.isfinite(x) for state in states for x in state[:2])
        complete = complete and all(abs(state[0]) <= 90 and abs(state[1]) <= 180 for state in states)
        state = np.asarray(states[0][:2], float) if complete and all(s[:2] == states[0][:2] for s in states) else None
        dt = timestamp-previous[0] if previous is not None else None
        adjacent = (times.index(timestamp) == times.index(previous[0])+1) if previous is not None else None
        valid_step = adjacent and state is not None and previous[1] is not None
        step = float(radial(state, previous[1])) if valid_step else None
        delta = state-previous[1] if valid_step else None
        raw_lon = float(delta[1]) if delta is not None else None
        short_lon = (raw_lon+180.) % 360.-180. if raw_lon is not None else None
        if short_lon is not None and abs(short_lon) == 180.:
            short_lon = math.copysign(180., raw_lon)
        blocks.append(dict(time_from_center_seconds=timestamp-offset-center_time, raw_row_lines=[r['line'] for r in rows],
            row_count=len(rows), same_time_position_conflict=bool(complete and any(s[:2] != states[0][:2] for s in states)),
            position_available=state is not None, previous_selected_time_gap_seconds=dt,
            previous_selected_is_immediate_original_time_neighbor=adjacent,
            previous_selected_position_step_m=step, previous_selected_step_mps=step/dt if step is not None and dt else None,
            previous_adjacent_delta_lat_degrees=float(delta[0]) if delta is not None else None,
            previous_adjacent_delta_lon_raw_degrees=raw_lon,
            previous_adjacent_delta_lon_short_arc_degrees=short_lon,
            previous_adjacent_lon_crosses_180_degrees=(abs(raw_lon) > 180.) if raw_lon is not None else None,
            step_scope='observed reference rows only; not evidence of AIS mother-track filtering'))
        previous = (timestamp, state)
    result = dict(target_index=target_index, queries=window_queries, total_raw_rows_for_target=len(current),
        all_reference_raw_rows_match_103=raw_identical, endpoint_checks=endpoint_checks, selected_neighbor_times=blocks,
        global_reference_issue_count=len(issues))
    return result, dict(target_identity=identity, selected_reference_rows=selected,
                        retained_reference_endpoints={str(q): ref['endpoints'][q] for q in window_queries})


def audit_scene(sid, target, center, payloads, private, case):
    if (case['scene_id'] != sid or case['target_index'] != target or case['center_query_index'] != center
            or case['consistency_status'] != 'MATCHES_WITHIN_TOLERANCE'
            or case['query_indices'] != list(range(max(0, center-2),
                min(len(private['input_motion']['query_seconds']), center+3)))):
        raise ValueError('104 case binding differs from declared case')
    row = next(x for x in case['rows'] if x['query_index'] == center)
    indices = [x['node_index'] for x in row['sources']]
    if len(indices) != 2 or len(set(indices)) != 2:
        raise ValueError('104 center must preserve its original two nodes')
    tracks = {x['node_index']: x for x in private['input_tracks']}
    selected_keys = {(tracks[index]['source'], tracks[index]['batch']) for index in indices}
    if len(selected_keys) != 2:
        raise ValueError('Fixed nodes must identify two distinct source/batch pairs')
    labels, label_encoding = raw_rows(payloads['labels'], HEADERS['labels'], selected_keys)
    measurements, measurement_encoding = raw_rows(payloads['measurements'], HEADERS['measurements'], selected_keys)
    events = {x['node_index']: [e['event_index'] for e in x['events']] for x in case['curve']['native_events']}
    public_nodes, private_nodes = [], []
    for index in indices:
        track = tracks[index]
        raw_mmsi = private['input_audit']['valid_raw_mmsi'][track['node_id']]
        public, secret = node_check(sid, track, labels, measurements, events[index], row['query_seconds'], raw_mmsi)
        public_nodes.append(public); private_nodes.append(secret)
    ref_public, ref_private = reference_check(payloads['reference'], private, target, case, center, sid)
    mismatch = (sum(not (n['identity_mapping_matches'] and n['node_mapping_matches'] and n['reader_would_keep_node']
                         and n['full_canonical_exact'] and n['all_fixed_events_match']) for n in public_nodes)
                + int(not ref_public['all_reference_raw_rows_match_103'])
                + sum(x['matches'] is False for x in ref_public['endpoint_checks']))
    public = clean(dict(scene_id=sid, target_index=target, center_query_index=center,
        source_encodings=dict(labels=label_encoding, measurements=measurement_encoding),
        fixed_node_count=len(indices), nodes=public_nodes, reference=ref_public, mismatch_count=mismatch,
        interpretation='Reconstruction of retained source data; no assertion of real identity, true clock or true position.'))
    return public, clean(dict(scene_id=sid, target_index=target, nodes=private_nodes, reference=ref_private))


def report(cases):
    lines = ['# 105 raw CSV to canonical chain', '',
             'Fixed postselected scenes; no model, truth-repair, or physical ground-truth claim.', '']
    for case in cases:
        lines += [f"## Scene {case['scene_id']}: target {case['target_index']}, query {case['center_query_index']}", '',
                  f"Mismatch count: {case['mismatch_count']}; reference rows match 103: "
                  f"{case['reference']['all_reference_raw_rows_match_103']}", '',
                  '| node | raw rows | duplicate rows | conflicting times | sorted canonical exact | fixed events matched |',
                  '|---|---:|---:|---:|---|---|']
        for node in case['nodes']:
            lines.append(f"| {node['node_index']} | {node['raw_measurement_row_count']} | "
                f"{node['identical_duplicate_count']} | {node['conflicting_time_count']} | "
                f"{node['full_canonical_exact']} | {node['all_fixed_events_match']} |")
        lines += ['', 'Each selected raw-line mapping and time/position step is retained in the case JSON.', '',
                  '| reference query | side | status | raw rows | matches retained 103 |', '|---|---|---|---|---|']
        for row in case['reference']['endpoint_checks']:
            lines.append(f"| {row['query_index']} | {row['side']} | {row['status']} | "
                         f"{row.get('raw_row_lines', [])} | {row['matches']} |")
        lines.append('')
    lines += ['## Limits', '',
              '- Equality with retained 103 records does not validate physical truth or actual timestamp semantics.',
              '- Whether the reference member is raw AIS or transformed is unknown; source codes cannot be mapped to source1/source2 by assumption.',
              '- Three scenes were selected after observing large errors; no overall anomaly rate or generalization claim.',
              '- Complete original identity, coordinates and raw source rows stay in the server-only private directory.', '']
    return '\n'.join(lines)


def limit_address_space():
    """On Linux only, bound this process without relaxing inherited limits."""
    if not sys.platform.startswith('linux'):
        return dict(address_space_limit_bytes=None,
                    address_space_limit_scope='not set on non-Linux software fixture')
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(value for value in (soft, hard, MAX_ADDRESS_SPACE_BYTES)
                if value != resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return dict(address_space_limit_bytes=limit,
                address_space_limit_scope='per-process Linux virtual address space; not RSS')


def run(args):
    output = Path(args.output).resolve()
    for source in (Path(args.source103).resolve(), Path(args.source104).resolve()):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Output must not overlap either immutable source run')
    output.mkdir(parents=True, exist_ok=False)
    (output/'private').mkdir(mode=0o700)
    started = time.monotonic()
    state = dict(status='RUNNING', idea_id='IDEA-STATE-DIAG-20261027', plan_version='1.5-raw-csv-audit',
        start_utc=datetime.now(timezone.utc).isoformat(), argv=sys.argv,
        completed_scenes=[], read_members=[],
        optimizer_updates=0, model_forwards=0, npz_files_read=0, checkpoint_files_read=0,
        test_opened=False, groups_recomputed=False, threshold_reselected=False)
    write(output/'run.json', state)
    try:
        state.update(limit_address_space())
        write(output/'run.json', state)
        manifest = read(args.manifest); validate_manifest(manifest)
        if not {sid for sid, _, _ in CASES}.issubset(set(manifest['splits']['V_select'])):
            raise ValueError('Fixed cases must all be in V_select')
        prior = read(Path(args.source103)/'run.json')
        if (prior['status'] != 'COMPLETED' or prior['test_opened'] is not False
                or prior['completed_scenes'] != [sid for sid, _, _ in CASES]
                or prior['model_forwards'] != 0):
            raise ValueError('103 source is not completed development diagnosis')
        prior104 = read(Path(args.source104)/'run.json')
        if (prior104['status'] != 'COMPLETED' or prior104['test_opened'] is not False
                or prior104['completed_cases'] != [sid for sid, _, _ in CASES]
                or prior104['consistency_mismatch_count'] != 0):
            raise ValueError('104 source is not a completed development diagnosis')
        cases = []
        with MTADArchive(args.archive) as archive:
            validate_dataset_level(archive.dataset_level, manifest)
            for sid, target, center in CASES:
                payloads = read_members(archive, sid, state['read_members'], lambda: write(output/'run.json', state))
                private = read(Path(args.source103)/f'private-evidence-{sid}.json.gz')
                case = read(Path(args.source104)/f'case-{sid}.json.gz')
                if len(private['input_motion']['query_seconds']) != QUERY_COUNT:
                    raise ValueError('103 source must retain declared Q64')
                public, secret = audit_scene(sid, target, center, payloads, private, case)
                write(output/f'case-{sid}.json.gz', public)
                write(output/'private'/f'case-{sid}.json.gz', secret)
                cases.append(public)
                state['completed_scenes'].append(sid)
                write(output/'run.json', state)
        count = sum(case['mismatch_count'] for case in cases)
        write(output/'summary.json.gz', dict(cases=cases, mismatch_count=count,
            limitations=['Raw/reference generation and real clock/identity are not independently verified.',
                         'Official materials do not establish whether this reference CSV is raw AIS or its exact spatial/time transformations.',
                         'Source1/source2 periods do not establish a fixed mapping to source codes 9001/9002.',
                         'Original ordering and daily clock rollovers require semantic evidence; decreases do not by themselves establish midnight.',
                         'Only three postselected scenes; no population incidence or model performance claim.',
                         'Private raw identifiers and absolute positions remain server only.']))
        (output/'report.md').write_text(report(cases), encoding='utf-8')
        state.update(status='COMPLETED_WITH_MISMATCHES' if count else 'COMPLETED', mismatch_count=count)
    except BaseException as error:
        state.update(status='FAILED', error_type=type(error).__name__,
                     error='Private-input/schema failure; original traceback remains on server.')
        raise
    finally:
        state.update(end_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            peak_rss_native_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_rss_unit='bytes' if sys.platform == 'darwin' else 'KiB')
        write(output/'run.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, nargs='+', required=True)
    for name in ('manifest', 'source103', 'source104', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise InterruptedError(f'signal {signum}; preserve partial data')
    signal.signal(signal.SIGTERM, stop)
    run(args)


if __name__ == '__main__':
    main()
