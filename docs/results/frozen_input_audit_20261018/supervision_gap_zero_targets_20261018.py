"""Locate only the 36 flagged scenes; never rescan all 576 scene payloads."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time

import numpy as np

from supervision_gap_causes_20261018 import (
    ROOT, ARCHIVE, MTADArchive, _name, _read_rows, STATE_FIELDS,
    align_reference_states, canonical, classify, count_reasons, dump,
)


def main():
    started=time.monotonic();deadline=started+120
    causes=[json.loads(x) for x in (ROOT/'supervision_gap_causes_scenes_20261018.jsonl').read_text().splitlines()]
    old={x['scene_id']:x for x in [json.loads(x) for x in (ROOT/'frozen_clock_supervision_scenes_20261018.jsonl').read_text().splitlines()]}
    allow=json.loads((ROOT/'frozen_scene_lists_20261018.json').read_text())['scene_ids']
    allowed={sid:role for role in ('T_core','V_select') for sid in allow[role]}
    # Count equality with the full intersection proves identical masks for lat,
    # lon and vel. Equality with the union proves course is the only superset.
    mask_structure=all(r['fully_valid_queries'][gap] == r['valid_per_feature'][gap][0]
        == r['valid_per_feature'][gap][1] == r['valid_per_feature'][gap][2]
        and r['any_feature_valid_queries'][gap] == r['valid_per_feature'][gap][3]
        for r in causes for gap in ('180','600'))
    if len(causes)!=576 or not mask_structure:raise ValueError('Cannot limit read scope from stored mask evidence')
    flagged=[r for r in causes if any(r['targets_without_any_valid_query'][gap]
        or len(set(r['valid_per_feature'][gap])) != 1 for gap in ('180','600'))]
    chosen={r['scene_id']:r['split'] for r in flagged}
    if len(chosen)!=36 or not set(chosen).issubset(allowed):raise ValueError('Unexpected flagged scope')
    output=ROOT/'supervision_gap_zero_targets_20261018.json'
    if output.exists():raise FileExistsError('Preserve old evidence')
    records=[];access=Counter();failures=[];status='COMPLETED';projection=None
    with MTADArchive(ARCHIVE) as archive:
        members={}
        for member in archive._zip.infolist():
            m=re.search(r'/真实场景/场景-(\d+)\.csv$',_name(member))
            if m and m.group(1) in chosen:
                if m.group(1) in members:raise ValueError('Duplicate reference member')
                members[m.group(1)]=member
        if set(members)!=set(chosen):raise ValueError('Missing selected reference')
        original=archive._zip.read
        def guarded(member,*args,**kwargs):
            name=_name(member) if hasattr(member,'filename') else member
            m=re.search(r'/(关联表|真实场景)/[^/]+-(\d+)\.csv$',name)
            if not m or m.group(2) not in chosen:raise ValueError('Outside 36 flagged scenes: '+name)
            access[chosen[m.group(2)]+':'+m.group(1)]+=1
            return original(member,*args,**kwargs)
        archive._zip.read=guarded
        for scene_index,(sid,role) in enumerate(chosen.items()):
            if time.monotonic()>=deadline:status='DEADLINE_STOP';break
            try:
                previous=old[sid]
                identities=list(dict.fromkeys(r['mmsi'].rsplit('-',1)[0] for r in archive.read_labels(sid)))
                rows,issues=_read_rows(archive._zip.read(members[sid]),_name(members[sid]))
                by_id=defaultdict(list);identity_set=set(identities)
                for row in rows:
                    if row['identity_id'] in identity_set:by_id[row['identity_id']].append(row)
                query=np.linspace(*previous['query_range_seconds'],previous['query_count_per_target'])
                offset=previous['diagnostic_offset_seconds'];targets=[]
                for identity in identities:
                    if time.monotonic()>=deadline:raise TimeoutError('Supplement deadline')
                    subset=by_id[identity]
                    reference=dict(scene_id=sid,split=role,rows=subset,issues=issues,state_fields=list(STATE_FIELDS))
                    valid_counts={};zero_causes={}
                    for gap in (180,600):
                        result=align_reference_states(reference,identity,query,time_offset_seconds=offset,max_gap_seconds=gap)
                        counts=result['valid'].sum(axis=0).tolist();valid_counts[str(gap)]=counts
                        if not all(counts):
                            reason,_=classify(canonical(subset,identity,offset),query,gap)
                            zero_causes[str(gap)]={k:v for k,v in count_reasons(reason).items() if sum(v)}
                    targets.append(dict(identity_id=identity,valid_count=valid_counts,
                                        zero_feature_causes=zero_causes))
                summary={gap:np.asarray([r['valid_count'][gap] for r in targets],dtype=np.int64).sum(axis=0).tolist() for gap in ('180','600')}
                prior=next(x for x in flagged if x['scene_id']==sid)
                if summary!=prior['valid_per_feature']:raise AssertionError('Supplement disagrees with prior totals')
                records.append(dict(scene_id=sid,split=role,targets=targets,valid_per_feature=summary))
            except Exception as e:
                failures.append(dict(scene_id=sid,error_type=type(e).__name__,error=str(e)));status='FAILURE';break
            if len(records)==8:
                projection=(time.monotonic()-started)/8*len(chosen)
                print(json.dumps({'first8_seconds':time.monotonic()-started,'projected_seconds':projection}),flush=True)
                if projection>120:status='PROJECTION_STOP';break
    zero_features={gap:[dict(scene_id=s['scene_id'],split=s['split'],**t) for s in records for t in s['targets'] if not all(t['valid_count'][gap])] for gap in ('180','600')}
    result=dict(status=status,elapsed_seconds=time.monotonic()-started,wall_budget_seconds=120,
        thread_environment_limit=2,requested_scenes=list(chosen),completed_scenes=len(records),
        first8_projected_seconds=projection,failures=failures,records=records,
        target_with_any_zero_feature_counts={g:len(v) for g,v in zero_features.items()},
        zero_feature_targets=zero_features,payload_read_counts=dict(access),T_test_payload_reads=0,
        measurement_payload_reads=0,model_forward_calls=0,optimizer_steps=0,
        all576_stored_mask_structure_verified=mask_structure,
        scope_justification='All scenes: fully-valid intersection equals lat/lon/vel counts; any-valid union equals course count. Outside selected scenes all feature masks coincide and prior all-zero-target count is zero, so omitted scenes contain no zero-feature identity. Selected: 34 gap180 all-zero scenes plus 2 course-only scenes.',
        production_policy_modified=False,offsets_and_query_bounds='Unchanged previous complete clock audit',
        note='Per-identity counts saved only for the 36 flagged scenes; absence of zero-feature targets in other 540 scenes follows from saved intersection/union counts and target-zero counts, not inferred lifetime.')
    dump(output,result)
    print(json.dumps({'output':str(output),'status':status,'completed_scenes':len(records),
        'elapsed_seconds':result['elapsed_seconds'],'zero_feature_target_counts':result['target_with_any_zero_feature_counts']}),flush=True)
    if status!='COMPLETED':raise SystemExit(2)


if __name__=='__main__':main()
