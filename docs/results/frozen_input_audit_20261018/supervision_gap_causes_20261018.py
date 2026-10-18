"""Bounded offline supervision-mask attribution; no model or policy changes.

Use existing frozen lists and previous audit offsets/query bounds. The local
classifier labels each feature/query once, and must match production alignment
masks at gap600 and gap180; gap600 counts must also match prior audit evidence.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).parent
REPO = Path(__file__).resolve().parents[3]
ARCHIVE = Path('/Users/a1/Downloads/数据集level2.zip')
sys.path.insert(0, str(REPO))
from association.data.archive import MTADArchive, _name
from association.data.reference_states import _read_rows, STATE_FIELDS
from association.data.targets import align_reference_states, _numeric

REASONS = (
    'valid_exact', 'valid_interpolated', 'identity_has_no_reference_rows',
    'invalid_identity', 'no_numeric_reference_times',
    'before_reference_time_support', 'after_reference_time_support',
    'exact_feature_invalid', 'exact_duplicate_conflict', 'gap_exceeded',
    'interpolation_endpoint_invalid', 'interpolation_endpoint_conflict',
    'interpolation_endpoint_invalid_and_conflict', 'heading_180_ambiguity',
)
VALID_REASONS = {'valid_exact', 'valid_interpolated'}


def dump(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def usable(value, column):
    if not _numeric(value) or not math.isfinite(float(value)):
        return False
    return ((-90 <= value <= 90) if column == 0 else
            (-180 <= value <= 180) if column == 1 else
            value >= 0 if column == 2 else 0 <= value <= 360)


def canonical(reference_rows, identity, offset):
    rows = [r for r in reference_rows if r['identity_id'] == identity]
    parts = identity.split('-')
    if not rows:
        return 'identity_has_no_reference_rows', [], None, None, None
    if len(parts) != 2 or not all(parts):
        return 'invalid_identity', [], None, None, None
    groups = defaultdict(list)
    for row in rows:
        raw_time = row['time']
        if not _numeric(raw_time) or not math.isfinite(float(raw_time)):
            continue
        timestamp = float(raw_time)-offset
        if not math.isfinite(timestamp):
            continue
        groups[timestamp].append(row)
    if not groups:
        return 'no_numeric_reference_times', [], None, None, None
    times = np.asarray(sorted(groups), dtype=np.float64)
    values = np.full((len(times),4), np.nan, dtype=np.float64)
    kind = np.empty((len(times),4), dtype=object)
    lines = []
    for index, timestamp in enumerate(times):
        rows_at_time = groups[float(timestamp)]
        lines.append([r['line'] for r in rows_at_time])
        for column in range(4):
            source = [r['state'][column] for r in rows_at_time]
            # Production gives any invalid duplicate row precedence over conflict.
            if not all(usable(value,column) for value in source):
                kind[index,column] = 'invalid'
            elif any(value != source[0] for value in source[1:]):
                kind[index,column] = 'conflict'
            else:
                kind[index,column] = 'valid'
                values[index,column] = 0.0 if column == 3 and source[0] == 360 else source[0]
    return None, times, values, kind, lines


def classify(canon, query, max_gap):
    empty_reason, times, values, kind, lines = canon
    reasons = np.empty((len(query),4), dtype=object)
    context = []
    for index, timestamp in enumerate(query):
        detail = {'query_time':float(timestamp)}
        if empty_reason:
            reasons[index,:] = empty_reason
            context.append(detail)
            continue
        right = int(np.searchsorted(times,timestamp))
        if right < len(times) and times[right] == timestamp:
            detail.update(reference_times=[float(times[right])],reference_lines=lines[right])
            for column in range(4):
                reasons[index,column] = {'valid':'valid_exact', 'invalid':'exact_feature_invalid',
                                         'conflict':'exact_duplicate_conflict'}[kind[right,column]]
        elif right == 0:
            reasons[index,:] = 'before_reference_time_support'
            detail.update(reference_support=[float(times[0]),float(times[-1])])
        elif right == len(times):
            reasons[index,:] = 'after_reference_time_support'
            detail.update(reference_support=[float(times[0]),float(times[-1])])
        else:
            left = right-1
            gap = float(times[right])-float(times[left])
            detail.update(reference_times=[float(times[left]),float(times[right])],
                          reference_lines=[lines[left],lines[right]],gap_seconds=gap)
            if gap > max_gap:
                reasons[index,:] = 'gap_exceeded'
            else:
                for column in range(4):
                    endpoints = {kind[left,column],kind[right,column]}
                    if 'invalid' in endpoints and 'conflict' in endpoints:
                        reason = 'interpolation_endpoint_invalid_and_conflict'
                    elif 'invalid' in endpoints:
                        reason = 'interpolation_endpoint_invalid'
                    elif 'conflict' in endpoints:
                        reason = 'interpolation_endpoint_conflict'
                    elif column == 3 and abs((values[right,3]-values[left,3]+180)%360-180) == 180:
                        reason = 'heading_180_ambiguity'
                    else:
                        reason = 'valid_interpolated'
                    reasons[index,column] = reason
        context.append(detail)
    return reasons, context


def count_reasons(reasons):
    return {reason:[int(np.count_nonzero(reasons[:,c] == reason)) for c in range(4)] for reason in REASONS}


def verify_classifier(rows, identity, query, gap, offset=0.0):
    reference = dict(scene_id='synthetic', split='T_core', rows=rows, issues=[], state_fields=list(STATE_FIELDS))
    actual = align_reference_states(reference,identity,query,time_offset_seconds=offset,max_gap_seconds=gap)
    classified, _ = classify(canonical(rows,identity,offset),query,gap)
    expected = np.isin(classified,list(VALID_REASONS))
    if not np.array_equal(expected,actual['valid']):
        raise AssertionError('Reason classifier differs from production align_reference_states')
    if not np.isin(classified,REASONS).all():
        raise AssertionError('Unclassified query')
    return set(classified.flatten())


def synthetic_selfcheck():
    def row(line,t,lat=1,lon=2,vel=3,cou=4,identity='1-1'):
        return dict(line=line,time=t,state=[lat,lon,vel,cou],identity_id=identity)
    cases = [
        ([], '1-1', [0], 600),
        ([row(1,0,identity='invalid')],'invalid',[0],600),
        ([row(1,None)],'1-1',[0],600),
        ([row(1,0),row(2,10)],'1-1',[-1,0,5,10,11],600),
        ([row(1,0,lat=100),row(2,0,lat=2,lon=3)],'1-1',[0],600),
        ([row(1,0),row(2,1000)],'1-1',[500],600),
        ([row(1,0,lat=100),row(2,0,lon=3),row(3,10,lat=2),row(4,10,lat=3)],'1-1',[5],600),
        ([row(1,0,cou=0),row(2,10,cou=180)],'1-1',[5],600),
        ([row(1,0,lat=100),row(2,10)],'1-1',[5],600),
        ([row(1,0,cou=360),row(2,0,cou=0)],'1-1',[0],600),
    ]
    reached=set()
    for args in cases:
        reached.update(verify_classifier(*args))
    if reached != set(REASONS):
        raise AssertionError('Synthetic cause coverage incomplete: '+str(set(REASONS)-reached))
    return {'cases':len(cases),'all_reason_categories_checked':list(REASONS)}


def main():
    started=time.monotonic(); deadline=started+300
    selfcheck=synthetic_selfcheck()
    previous=json.loads((ROOT/'frozen_clock_supervision_audit_20261018.json').read_text())
    records=[json.loads(line) for line in (ROOT/'frozen_clock_supervision_scenes_20261018.jsonl').read_text().splitlines()]
    lists=json.loads((ROOT/'frozen_scene_lists_20261018.json').read_text())['scene_ids']
    requested=[(role,sid) for role in ('T_core','V_select') for sid in lists[role]]
    if (previous['status'] != 'COMPLETED' or len(records) != 576
            or [(r['split'],r['scene_id']) for r in records] != requested
            or len(set(s for _,s in requested)) != 576):
        raise ValueError('Frozen lists and completed prior audit disagree')
    allowed={sid:role for role,sid in requested}; prior={r['scene_id']:r for r in records}
    output=ROOT/'supervision_gap_causes_20261018.json'
    incremental=ROOT/'supervision_gap_causes_scenes_20261018.jsonl'
    if output.exists() or incremental.exists():
        raise FileExistsError('Refuse to overwrite prior evidence')
    reasons_total={role:{str(gap):np.zeros((len(REASONS),4),dtype=np.int64) for gap in (180,600)} for role in ('T_core','V_select')}
    valid_total={role:{str(gap):np.zeros(4,dtype=np.int64) for gap in (180,600)} for role in ('T_core','V_select')}
    complete=Counter(); query_total=Counter(); accesses=Counter(); examples=defaultdict(list)
    failures=[]; finished=[]; status='COMPLETED'; projection=None
    with incremental.open('x') as stream, MTADArchive(ARCHIVE) as archive:
        members={}
        for member in archive._zip.infolist():
            match=re.search(r'/真实场景/场景-(\d+)\.csv$',_name(member))
            if match and match.group(1) in allowed:
                if match.group(1) in members:raise ValueError('Duplicate reference member')
                members[match.group(1)]=member
        if set(members) != set(allowed):raise ValueError('Missing allowed reference member')
        original_read=archive._zip.read
        def guarded_read(member,*args,**kwargs):
            name=_name(member) if hasattr(member,'filename') else member
            match=re.search(r'/(关联表|真实场景)/[^/]+-(\d+)\.csv$',name)
            if not match or match.group(2) not in allowed:
                raise ValueError('Forbidden CSV payload read: '+name)
            accesses[allowed[match.group(2)]+':'+match.group(1)]+=1
            return original_read(member,*args,**kwargs)
        archive._zip.read=guarded_read
        for role,sid in requested:
            if time.monotonic() >= deadline:
                status='DEADLINE_STOP';break
            try:
                old=prior[sid]
                identities=list(dict.fromkeys(row['mmsi'].rsplit('-',1)[0] for row in archive.read_labels(sid)))
                if len(identities)!=old['label_target_count']:raise ValueError('Prior label identity count changed')
                query=np.linspace(*old['query_range_seconds'],old['query_count_per_target'])
                offset=old['diagnostic_offset_seconds']
                rows,issues=_read_rows(archive._zip.read(members[sid]),_name(members[sid]))
                by_identity=defaultdict(list)
                for row in rows:
                    if row['identity_id'] in identities:by_identity[row['identity_id']].append(row)
                counts={str(gap):np.zeros((len(REASONS),4),dtype=np.int64) for gap in (180,600)}
                full_valid=Counter(); any_valid=Counter(); zero_valid_target=Counter()
                for identity in identities:
                    if time.monotonic() >= deadline:raise TimeoutError('CPU diagnostic deadline')
                    subset=by_identity[identity]
                    canon=canonical(subset,identity,offset)
                    reference=dict(scene_id=sid,split=role,rows=subset,issues=issues,state_fields=list(STATE_FIELDS))
                    for gap in (180,600):
                        reasons,context=classify(canon,query,gap)
                        expected=np.isin(reasons,list(VALID_REASONS))
                        actual=align_reference_states(reference,identity,query,time_offset_seconds=offset,max_gap_seconds=gap)
                        if not np.array_equal(expected,actual['valid']):
                            bad=np.argwhere(expected != actual['valid'])
                            raise AssertionError(f'Classifier disagreement {sid} {identity} gap={gap} positions={bad[:3].tolist()}')
                        counted=count_reasons(reasons)
                        if any(sum(counted[r][c] for r in REASONS)!=len(query) for c in range(4)):
                            raise AssertionError('Causes not an exclusive exhaustive partition')
                        counts[str(gap)]+=np.asarray([counted[r] for r in REASONS])
                        full_valid[str(gap)]+=int(expected.all(axis=1).sum())
                        any_valid[str(gap)]+=int(expected.any(axis=1).sum())
                        zero_valid_target[str(gap)]+=int(not expected.any())
                        if gap==600:
                            for reason in REASONS:
                                if reason in VALID_REASONS or len(examples[reason])>=3:continue
                                positions=np.argwhere(reasons==reason)
                                if len(positions):
                                    i,c=map(int,positions[0]);detail=dict(context[i])
                                    detail.update(scene_id=sid,split=role,identity_id=identity,
                                                  feature=STATE_FIELDS[c],query_index=i,reason=reason)
                                    examples[reason].append(detail)
                scene_valid={gap:(matrix[0]+matrix[1]).tolist() for gap,matrix in counts.items()}
                if scene_valid['600'] != old['gap600_diagnostic_reference_valid_per_feature']:
                    raise AssertionError('New gap600 coverage differs from previous complete audit')
                record=dict(scene_id=sid,split=role,status='VERIFIED',label_target_count=len(identities),
                            target_query_count=len(identities)*len(query),offset_from_previous_audit=offset,
                            query_range_from_previous_audit=old['query_range_seconds'],
                            reasons_per_feature={gap:{reason:matrix[i].tolist() for i,reason in enumerate(REASONS)} for gap,matrix in counts.items()},
                            valid_per_feature=scene_valid,fully_valid_queries=dict(full_valid),
                            any_feature_valid_queries=dict(any_valid),targets_without_any_valid_query=dict(zero_valid_target))
                for gap in ('180','600'):
                    reasons_total[role][gap]+=counts[gap]
                    valid_total[role][gap]+=scene_valid[gap]
                query_total[role]+=record['target_query_count'];complete[role]+=1
            except Exception as error:
                record=dict(scene_id=sid,split=role,status='ERROR',error_type=type(error).__name__,error=str(error))
                failures.append(record);status='LOGIC_OR_DATA_FAILURE'
            finished.append(record)
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
            if failures:break
            if len(finished)==8:
                projection=(time.monotonic()-started)/8*len(requested)
                print(json.dumps({'first8_seconds':time.monotonic()-started,'projected_total_seconds':projection}),flush=True)
                if projection>300:status='PROJECTION_STOP';break
            if len(finished)%64==0:
                print(json.dumps({'scenes_completed':len(finished),'elapsed_seconds':time.monotonic()-started}),flush=True)
    summary={role:dict(completed_scenes=complete[role],target_query_count=query_total[role],
                  reasons_per_feature={gap:{reason:matrix[i].tolist() for i,reason in enumerate(REASONS)} for gap,matrix in reasons_total[role].items()},
                  valid_per_feature={gap:a.tolist() for gap,a in valid_total[role].items()},
                  fully_valid_queries={gap:sum(r['fully_valid_queries'][gap] for r in finished if r['split']==role and r['status']=='VERIFIED') for gap in ('180','600')},
                  targets_without_any_valid_query={gap:sum(r['targets_without_any_valid_query'][gap] for r in finished if r['split']==role and r['status']=='VERIFIED') for gap in ('180','600')}) for role in ('T_core','V_select')}
    result=dict(status=status,elapsed_seconds=time.monotonic()-started,wall_budget_seconds=300,
                thread_limit_environment=2,first8_projected_total_seconds=projection,requested_scenes=576,
                verified_scenes=sum(complete.values()),failures=failures,summary=summary,
                reason_order=list(REASONS),state_fields=list(STATE_FIELDS),examples_gap600=dict(examples),
                synthetic_classifier_checks=selfcheck,payload_read_counts=dict(accesses),
                T_test_payload_reads=0,measurement_payload_reads=0,model_forward_calls=0,optimizer_steps=0,
                production_policy_modified=False,offset_or_gap_optimized=False,
                previous_audit='frozen_clock_supervision_audit_20261018.json',
                previous_scene_audit='frozen_clock_supervision_scenes_20261018.jsonl',
                allowlist='frozen_scene_lists_20261018.json',per_scene_output=incremental.name,
                production_rules='association/data/targets.py:align_reference_states',
                partition_policy='First matching production branch: exact; outside reference time support; gap exceeded; endpoint invalid/conflict (combined if opposite endpoints); heading 180; valid interpolation. Categories are exclusive per feature/query.',
                support_policy='Actual numeric reference timestamps after the prior fixed offset; no inferred lifetime is used.',
                validation='Every identity at both gaps: reason-valid mask equals production align mask; every scene gap600 counts equal previous audit; every feature has exactly one reason.')
    dump(output,result)
    print(json.dumps({'output':str(output),'status':status,'verified_scenes':result['verified_scenes'],
                      'elapsed_seconds':result['elapsed_seconds'],'failures':failures},ensure_ascii=False),flush=True)
    if status!='COMPLETED':raise SystemExit(2)


if __name__=='__main__':main()
