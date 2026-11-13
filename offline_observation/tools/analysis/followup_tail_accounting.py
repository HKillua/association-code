"""146 post-science accounting of planned before/after zero-support strata."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from tools.analysis import audit_followup_outputs as independent


def run(args):
    start=time.monotonic(); output=args.output
    if output.exists(): raise FileExistsError(output)
    record=independent.read(args.run/'run.json')
    if record['status']!='COMPLETED' or record['exit_code']!=0: raise ValueError('146 must be completed')
    native=Path(record['native_run'])
    result=dict(status='RUNNING', command=sys.argv, run=str(args.run), trace_run=str(args.trace_run),
        started_at=datetime.now(timezone.utc).isoformat(), scope='read-only planned zero-support before/after; posthoc already-exposed4393 branch; no new outputs/fit/tuning', splits={})
    for split,ids,summary_file in (('T_core',record['training_scene_ids'],'T-summary.json'),('V_select',record['scene_ids'],'summary.json')):
        names=tuple(independent.read(args.run/summary_file)['all_components']['methods'])
        rows=[]; branch=None
        for sid in ids:
            payload=independent.load(args.run/'private'/f'states-{split}-{sid}.npz')
            ev=independent.load(native/'private/native-evaluation'/split/f'scene-{sid}.npz')
            motion=independent.load(native/'private/native-motion'/split/f'scene-{sid}.npz')
            events=independent.load(native/'private/native-events'/split/f'scene-{sid}.npz')
            base={k[len(names[0])+1:]:v for k,v in payload.items() if k.startswith(names[0]+'_')}
            groups=[base['group_node_indices'][a:b] for a,b in zip(base['group_offsets'][:-1],base['group_offsets'][1:])]
            n=len(ev['target_ids']); q=motion['query_seconds']
            valid=ev['target_valid'][...,:2].all(-1)&base['aligned_prediction_valid'][:n,:,:2].all(-1)
            before=np.zeros(valid.shape,bool); after=before.copy()
            for target,gid in enumerate(base['aligned_group_indices'][:n]):
                if gid<0: continue
                group=groups[gid]; zero=valid[target]&~motion['support'][group].any(0)
                a,b=events['event_offsets'][group[0]:group[0]+2]
                before[target]=zero&(q<events['event_times_seconds'][a]); after[target]=zero&(q>events['event_times_seconds'][b-1])
                if not np.array_equal(before[target]|after[target],zero): raise ValueError('zero outside anchor span')
            errors={name:independent.distance(payload[name+'_aligned_prediction'][:n],ev['target_states']) for name in names}
            component=independent.read(args.run/'scenes'/split/f'scene-{sid}.json.gz')['component_id']
            rows.append(dict(component=component,scene_id=sid,errors={side:{name:errors[name][mask] for name in names} for side,mask in (('before',before),('after',after))}))
            if split=='V_select' and sid=='4393':
                cases=independent.read(args.trace_run/'private'/f'cases-{split}-{sid}.json.gz')
                old='WS_OBS_SCALE_RTS_EQUAL'
                worst=max(cases,key=lambda c:c[old+'__error_m'])
                chosen=[c for c in cases if c['_anchor_node']==worst['_anchor_node'] and c['side']=='after']
                ti=np.asarray([c['_target_row'] for c in chosen],int); qi=np.asarray([c['_query_index'] for c in chosen],int)
                if len(chosen)!=27 or not np.all(after[ti,qi]): raise ValueError('original fixed27 branch')
                if not np.allclose(errors[old][ti,qi],[c[old+'__error_m'] for c in chosen],rtol=1e-9,atol=1e-6): raise ValueError('142 original branch binding')
                branch=dict(selection='posthoc original142 worst exposed4393 anchor, after side; fixed27, no deployment input',count=27,
                    methods={name:{**independent.stats(errors[name][ti,qi]),'maximum_m':float(errors[name][ti,qi].max())} for name in names})
        cs=sorted({r['component'] for r in rows});largest=max(cs,key=lambda c:sum(r['component']==c for r in rows))
        def block(chosen):
            return {side:{name:independent.stats(np.concatenate([r['errors'][side][name] for r in chosen])) for name in names} for side in ('before','after')}
        result['splits'][split]=dict(all_components=block(rows),by_component={c:block([r for r in rows if r['component']==c]) for c in cs},
            without_largest_component=block([r for r in rows if r['component']!=largest]),largest_component_id=largest)
        expected=1487 if split=='T_core' else 2770
        actual=sum(result['splits'][split]['all_components'][side][names[0]]['intersection_count'] for side in ('before','after'))
        if actual!=expected: raise ValueError('complete zero population')
        if branch is not None: result['original142_exposed_branch']=branch
    result.update(status='COMPLETED',elapsed_seconds=time.monotonic()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
        finished_at=datetime.now(timezone.utc).isoformat(),new_predictions=0,new_fits=0)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','peak_rss_bytes')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','trace-run','output'): parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
