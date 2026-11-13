"""Read-only156 tail/validation-horizon accounting. No new prediction or fitting."""
from __future__ import annotations
import argparse,json,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis.audit_source_risk_arrays import distances

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def run(root,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();record=json.loads((root/'run.json').read_text());native=Path(record['native_run']);result=dict(status='COMPLETED',exit_code=0,run=str(root),new_predictions=0,new_fits=0,splits={})
    for split,ids in [('T_core',record['training_scene_ids']),('V_select',record['scene_ids'])]:
        cases=[];rows=[]
        for sid in ids:
            z=load(root/'private'/f'states-{split}-{sid}.npz');prefix='WS_ZERO_NEAREST_CV_';ev=load(native/'private/native-evaluation'/split/f'scene-{sid}.npz');nt=len(ev['target_ids']);truth=ev['target_states'];pred=z[prefix+'aligned_prediction'][:nt];valid=ev['target_valid'][...,:2].all(-1)&z[prefix+'aligned_prediction_valid'][:nt,:,:2].all(-1)
            e=distances(pred,truth);use=e[valid];rows.append(dict(scene_id=sid,points=len(use),above100km=int((use>100000).sum()),sse_m2=float((use*use).sum()),max_error_m=float(use.max())))
            candidates=np.argwhere(valid)
            for target,q in candidates[np.argsort(e[valid])[-5:]]:cases.append((float(e[target,q]),sid,int(target),int(q)))
        detail=[]
        for error,sid,target,q in sorted(cases,reverse=True)[:5]:
            z=load(root/'private'/f'states-{split}-{sid}.npz');prefix='WS_ZERO_NEAREST_CV_';gid=int(z[prefix+'aligned_group_indices'][target]);offset=z[prefix+'group_offsets'];members=z[prefix+'group_node_indices'][offset[gid]:offset[gid+1]]
            m=load(native/'private/native-motion'/split/f'scene-{sid}.npz');events=load(native/'private/native-events'/split/f'scene-{sid}.npz');f=load(root/'private'/f'native-forecast-{split}-{sid}.npz');zero=not m['support'][members,q].any();near=m['nearest_seconds'][members,q];chosen=members[near==near.min()];selected=[]
            for node in chosen:
                a,b=events['event_offsets'][node:node+2];t=events['event_times_seconds'][a:b];v=events['event_values'][a:b];before=m['query_seconds'][q]<t[0];u=-t[::-1] if before else t;pos=v[::-1,:2] if before else v[:,:2];h=abs(m['query_seconds'][q]-(t[0] if before else t[-1]));cuts=sorted({min(len(t)-1,max(2,int(len(t)*r))) for r in (.25,.5,.75)}) if len(t)>=3 else []
                covered=[float(u[-1]-u[c-1]) for c in cuts];window=min(8,len(t));step=distances(pos[-window+1:],pos[-window:-1])/np.diff(u[-window:])
                selected.append(dict(node_index=int(node),native_count=len(t),direction='before' if before else 'after',extrapolation_seconds=float(h),local8_span_seconds=float(u[-1]-u[-window]),historical_max_horizon_seconds=max(covered) if covered else None,horizon_exceeds_all_validation=bool(not covered or h>max(covered)),expert_mass=f['expert_mass'][node,q].tolist(),maximum_local_native_step_rate_mps=float(step.max()) if len(step) else None))
            detail.append(dict(scene_id=sid,target_slot=target,query_slot=q,error_m=error,zero_support=zero,selected_candidates=selected))
        result['splits'][split]=dict(points=sum(r['points'] for r in rows),above100km=sum(r['above100km'] for r in rows),max_error_m=max(r['max_error_m'] for r in rows),scenes=rows,top5=detail)
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='posthoc saved CV tails and observable validation-horizon coverage; no physical anomaly cause or counterfactual improvement claim')
    out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run.resolve(),a.output.resolve())
