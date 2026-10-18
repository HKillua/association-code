"""Bounded CPU diagnostic on existing T_core512/V_select64 lists, never a split.

No model, normalizer, training configuration or production alignment is made.
The old fixed min-labelled-reference-time hypothesis and gap600 diagnostic are
tested, not optimized. Raw exact-state matching supplies independent evidence.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).parent
ARCHIVE = Path('/Users/a1/Downloads/数据集level2.zip')
sys.path.insert(0, str(REPO))
from association.data.archive import MTADArchive, _name
from association.data.targets import align_reference_states
from association.data.reference_states import _read_rows


def finite(value):
    return isinstance(value, (int, float)) and np.isfinite(value)


def histogram_summary(histogram):
    count = sum(histogram.values())
    if not count:
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    ordered = sorted(histogram.items())
    def quantile(q):
        threshold = q * max(count-1, 0)
        consumed = 0
        for value, occurrences in ordered:
            consumed += occurrences
            if consumed > threshold:
                return value
    return {'count': count, 'min': ordered[0][0], 'median': quantile(.5),
            'p95': quantile(.95), 'max': ordered[-1][0],
            'above_600_seconds': sum(n for gap, n in ordered if gap > 600),
            'histogram': {str(gap): n for gap, n in ordered}}


def lists():
    evidence = json.loads((ROOT/'frozen_scene_lists_20261018.json').read_text())
    core, validation = (evidence['scene_ids'][role] for role in ('T_core', 'V_select'))
    profile = json.loads((REPO/'HandOver/results/NATIVE-20261017-007/profile.json').read_text())
    tsa = json.loads((REPO/'HandOver/results/COMPARE-20261017-011/tsa/files/data_audit.json').read_text())
    if (len(core) != 512 or len(validation) != 64 or len(set(core+validation)) != 576
            or core != profile['training_scene_ids'] or core != tsa['training_scene_ids']
            or validation != tsa['validation_scene_ids']
            or any(not re.fullmatch(r'\d+', sid) for sid in core+validation)):
        raise ValueError('Stored frozen allowlists disagree; no data may be read')
    return [('T_core', sid) for sid in core] + [('V_select', sid) for sid in validation]


def scene_audit(archive, sid, split, reference_member, deadline):
    started = time.monotonic()
    scene = archive.load_scene(sid)
    labels = archive.read_labels(sid)
    identities = list(dict.fromkeys(row['mmsi'].rsplit('-', 1)[0] for row in labels))
    rows, issues = _read_rows(archive._zip.read(reference_member), _name(reference_member))
    by_identity = defaultdict(list)
    by_state, by_time = defaultdict(set), defaultdict(list)
    for row in rows:
        identity, timestamp = row['identity_id'], row['time']
        if identity not in identities:
            continue
        by_identity[identity].append(row)
        if finite(timestamp) and all(finite(x) for x in row['state']):
            state = tuple(row['state'])
            by_state[identity, state].add(timestamp)
            by_time[identity, timestamp].append(state)
    numeric_times = [row['time'] for identity in identities for row in by_identity[identity]
                     if finite(row['time'])]
    report = {'scene_id': sid, 'split': split, 'valid_nodes': len(scene.tracks),
              'quarantined_nodes': scene.audit['quarantined_nodes'],
              'label_target_count': len(identities), 'reference_row_count': len(rows),
              'reference_issue_count': len(issues), 'native_point_count': sum(len(t.times) for t in scene.tracks)}
    if not numeric_times:
        return {**report, 'status': 'NO_LABELLED_REFERENCE_TIMES', 'seconds': time.monotonic()-started}, Counter()
    offset = min(numeric_times)
    if scene.tracks:
        start = min(t.times[0] for t in scene.tracks)
        stop = max(t.times[-1] for t in scene.tracks)
    else:
        start, stop = min(r['t_s'] for r in labels), max(r['t_e'] for r in labels)
    query = np.linspace(start, stop, 64)
    gap_histogram = Counter()
    exact_valid, interpolated_valid = np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)
    missing_ids = []
    for identity in identities:
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU diagnostic deadline')
        reference_rows = by_identity[identity]
        if not reference_rows:
            missing_ids.append(identity)
        times = sorted({r['time'] for r in reference_rows if finite(r['time'])})
        gap_histogram.update(float(b-a) for a, b in zip(times, times[1:]))
        # Grouping preserves each identity's complete original rows and issues;
        # it avoids repeating a whole-scene scan for every target.
        reference = dict(scene_id=sid, split=split, rows=reference_rows,
                         issues=issues, state_fields=['lat', 'lon', 'vel', 'cou'])
        exact = align_reference_states(reference, identity, query,
            time_offset_seconds=offset, max_gap_seconds=0)
        diagnostic = align_reference_states(reference, identity, query,
            time_offset_seconds=offset, max_gap_seconds=600)
        exact_valid += exact['valid'].sum(axis=0)
        interpolated_valid += diagnostic['valid'].sum(axis=0)
    source9001 = {(t.source, t.batch): t.identity_id for t in scene.tracks if t.source == '9001'}
    matches, inferred_offsets = Counter(), Counter()
    for row in archive._rows(sid, 'measurements'):
        identity = source9001.get((row['source'], row['batch']))
        if identity is None:
            continue
        timestamp = float(row['time'])
        values = tuple(float(row[key]) for key in ('lat', 'lon', 'vel', 'cou'))
        matches['raw_points_on_valid_source9001_nodes'] += 1
        state_times = by_state.get((identity, values), set())
        if len(state_times) == 1:
            implied = next(iter(state_times))-timestamp
            inferred_offsets[implied] += 1
            matches['unique_state_match_points'] += 1
            matches['unique_state_match_agrees_fixed_offset' if implied == offset
                    else 'unique_state_match_disagrees_fixed_offset'] += 1
        elif state_times:
            matches['multiple_time_state_match_points'] += 1
        else:
            matches['no_exact_state_match_points'] += 1
        candidates = by_time.get((identity, timestamp+offset), [])
        if len(candidates) == 1:
            matches['unique_reference_time_match_points'] += 1
            matches['fixed_offset_raw_four_exact_points' if candidates[0] == values
                    else 'fixed_offset_raw_four_mismatch_points'] += 1
        elif candidates:
            matches['ambiguous_reference_time_points'] += 1
        else:
            matches['no_reference_time_points'] += 1
    for key in ('unique_state_match_disagrees_fixed_offset', 'fixed_offset_raw_four_mismatch_points',
                'unique_state_match_points', 'fixed_offset_raw_four_exact_points',
                'raw_points_on_valid_source9001_nodes'):
        matches.setdefault(key, 0)
    supported = matches['unique_state_match_points'] > 0 and not matches['unique_state_match_disagrees_fixed_offset']
    report.update(status='CLOCK_HYPOTHESIS_SUPPORTED' if supported else 'CLOCK_UNRESOLVED',
        diagnostic_offset_seconds=float(offset), query_range_seconds=[float(start), float(stop)],
        query_count_per_target=64, target_query_count=64*len(identities),
        missing_reference_identity_ids=missing_ids,
        exact_only_reference_valid_per_feature=exact_valid.tolist(),
        gap600_diagnostic_reference_valid_per_feature=interpolated_valid.tolist(),
        reference_adjacent_gaps=histogram_summary(gap_histogram), raw_source9001_matches=dict(matches),
        implied_offsets_from_unique_exact_states={str(k):v for k,v in sorted(inferred_offsets.items())},
        seconds=time.monotonic()-started)
    return report, gap_histogram


def main():
    started = time.monotonic()
    deadline = started+180
    requested = lists()
    allowed = {sid:split for split,sid in requested}
    output = ROOT/'frozen_clock_supervision_audit_20261018.json'
    incremental = ROOT/'frozen_clock_supervision_scenes_20261018.jsonl'
    if output.exists() or incremental.exists():
        raise FileExistsError('This diagnostic evidence must not overwrite earlier output')
    records, accesses, failures, gaps = [], Counter(), [], Counter()
    status, projected = 'COMPLETED', None
    with incremental.open('x') as stream, MTADArchive(ARCHIVE) as archive:
        references = {}
        for member in archive._zip.infolist():
            match = re.search(r'/真实场景/场景-(\d+)\.csv$', _name(member))
            if match and match.group(1) in allowed:
                references[match.group(1)] = member
        if set(references) != set(allowed):
            raise ValueError('Requested reference CSVs are missing')
        original_read = archive._zip.read
        def guarded_read(member, *args, **kwargs):
            name = _name(member) if hasattr(member, 'filename') else member
            match = re.search(r'/(量测场景|关联表|真实场景)/[^/]+-(\d+)\.csv$', name)
            if not match or match.group(2) not in allowed:
                raise ValueError('CSV access outside the frozen T_core/V_select allowlist: '+name)
            accesses[allowed[match.group(2)]+':'+match.group(1)] += 1
            return original_read(member,*args,**kwargs)
        archive._zip.read = guarded_read
        for index,(split,sid) in enumerate(requested):
            if time.monotonic() >= deadline:
                status='DEADLINE_STOP'; break
            try:
                record, scene_gaps = scene_audit(archive,sid,split,references[sid],deadline)
                gaps.update(scene_gaps)
            except Exception as error:
                record={'scene_id':sid,'split':split,'status':'ERROR',
                        'error_type':type(error).__name__,'error':str(error)}
                failures.append(record)
            records.append(record)
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
            if len(records)==8:
                projected=(time.monotonic()-started)/8*len(requested)
                print(json.dumps({'first8_elapsed_seconds':time.monotonic()-started,
                                  'projected_total_seconds':projected}),flush=True)
                if projected>180:
                    status='PROJECTION_STOP'; break
            if len(records)%64==0:
                print(json.dumps({'scenes_completed':len(records),'elapsed_seconds':time.monotonic()-started}),flush=True)
    summary={}
    for role in ('T_core','V_select'):
        subset=[r for r in records if r['split']==role]
        valid=[r for r in subset if 'raw_source9001_matches' in r]
        matches=Counter()
        for r in valid: matches.update(r['raw_source9001_matches'])
        summary[role]={'requested_scenes':sum(s==role for s,_ in requested),'completed_scene_records':len(subset),
            'clock_hypothesis_supported_scenes':sum(r['status']=='CLOCK_HYPOTHESIS_SUPPORTED' for r in subset),
            'unresolved_scene_ids':[r['scene_id'] for r in subset if r['status']!='CLOCK_HYPOTHESIS_SUPPORTED'],
            'nonzero_offset_scenes':{r['scene_id']:r['diagnostic_offset_seconds'] for r in valid if r['diagnostic_offset_seconds']},
            'target_query_count':sum(r['target_query_count'] for r in valid),
            'exact_only_reference_valid_per_feature':np.asarray([r['exact_only_reference_valid_per_feature'] for r in valid],dtype=np.int64).reshape(-1,4).sum(axis=0).tolist(),
            'gap600_diagnostic_reference_valid_per_feature':np.asarray([r['gap600_diagnostic_reference_valid_per_feature'] for r in valid],dtype=np.int64).reshape(-1,4).sum(axis=0).tolist(),
            'raw_source9001_matches':dict(matches)}
    result={'status':status,'allowlist_evidence':str(ROOT/'frozen_scene_lists_20261018.json'),
        'requested_scene_count':576,'completed_scene_records':len(records),
        'elapsed_seconds':time.monotonic()-started,'cpu_limit_seconds':180,'thread_limit':4,
        'first8_projected_total_seconds':projected,'summary':summary,
        'reference_adjacent_gaps':histogram_summary(gaps),'failures':failures,
        'per_scene_output':str(incremental),'payload_read_counts':dict(accesses),
        'access_guard':'Every CSV read is rejected unless scene_id belongs to the existing T_core512/V_select64 lists',
        'T_test_payload_reads':0,'model_forward_calls':0,'optimizer_steps':0,
        'manifest_or_split_created':False,'production_alignment_created':False,
        'offset_policy':'Fixed old diagnostic hypothesis: min raw reference time among label-named identities; never searched by prediction/position error',
        'independent_clock_evidence':'For raw observations on valid source9001 nodes, identical raw four-state values at one unique reference time imply a clock delta; compare all such deltas to the fixed hypothesis',
        'coverage_policy':'Report exact-only and fixed max_gap600 diagnostic coverage on common physical scene64 queries; 600 is not a proposed training gap',
        'state_fields':['lat','lon','vel','cou'],
        'limits':'Conditional exact matches support but do not mathematically prove every timestamp. No error-minimizing offset/gap selection; no reference goes into a network.'}
    with output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({'output':str(output),'status':status,'completed':len(records),
                      'elapsed_seconds':result['elapsed_seconds'],'summary':summary},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
