"""189: independent native-mean intervention plus complete T-only output audit."""
from __future__ import annotations
import argparse,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis.drift_memory_pilot import ARMS,PRIMARY,main_rows,condition_summary
from tools.analysis.history_followup_closeout import read,load,account,center
from tools.analysis.native_motion_closeout import forward_position

def reconstruct(run,r,split,sid,front,m,events,groups,count,op,oraw,raw,policy):
    assert split=='T_core';values=ARMS[front];parent=Path(r['persistence_run']);prior=load(parent/'private'/f'raw-{split}-{sid}.npz');old_policy=load(parent/'private'/f'policy-{split}-{sid}.npz');base=oraw[values[1]+'_NN'];expected=base.copy();eligible=np.zeros(count.shape,bool);native=[]
    for prefix in values[:-1]:
        for suffix in ('_MEAN','_NN'):assert np.array_equal(raw[prefix+suffix],prior[prefix+suffix])
    for node in range(len(m['states'])):
        a,b=events['event_offsets'][node:node+2];t=events['event_times_seconds'][a:b];p=events['event_values'][a:b,:2];valid=events['event_valid'][a:b,2:4].all(1);v=events['event_values'][a:b];angle=np.deg2rad(v[valid,3]);speed=v[valid,2]*1852/3600;mu=np.array([np.mean(speed*np.cos(angle)),np.mean(speed*np.sin(angle))]) if valid.sum()>=2 else None;native.append((t,p,valid,mu))
    for gid,nodes in enumerate(groups):
        first=min(native[n][0][0] for n in nodes);last=max(native[n][0][-1] for n in nodes)
        for j in np.flatnonzero((count[gid]==0)&((m['query_seconds']<first)|(m['query_seconds']>last))):
            eligible[gid,j]=True;chosen=nodes[op[values[1]+'__weights'][nodes,j]>0];mass=op[values[1]+'__weights'][chosen,j];positions=[]
            for node in chosen:
                t,p,valid,mu=native[node];edge=0 if m['query_seconds'][j]<t[0] else -1;dt=m['query_seconds'][j]-t[edge];positions.append(m['states'][node,j,:2] if mu is None or not valid[edge] else forward_position(p[edge],mu*dt))
            expected[gid,j,:2]=center(np.stack(positions),mass)
    assert np.array_equal(eligible,policy[front+'__eligible']) and np.array_equal(eligible,old_policy[front+'__eligible']);np.testing.assert_allclose(raw[values[-1]+'_NN'][...,:2],expected[...,:2],rtol=0,atol=1e-10);assert np.array_equal(raw[values[-1]+'_NN'][~eligible],base[~eligible]);return eligible,0.

def verify_conditions(run,r):
    start=time.monotonic();rows={front:[] for front in ARMS};means=0
    for sid in r['training_scene_ids']:
        n=Path(r['native_run'])/'private';m=load(n/'native-motion/T_core'/f'scene-{sid}.npz');ev=load(n/'native-evaluation/T_core'/f'scene-{sid}.npz');q=load(run/'private'/f'partitions-T_core-{sid}.npz');raw=load(run/'private'/f'raw-T_core-{sid}.npz');diag=read(run/'private'/f'diagnostics-T_core-{sid}.json.gz')
        for front in ARMS:
            groups=[np.flatnonzero(q[front]==v) for v in sorted(set(q[front].tolist()))];record=main_rows(m,ev,groups,raw,diag[front],front)
            for row in record:row['scene_id']=sid
            assert record==read(run/'private'/f'conditions-{front}-{sid}.json.gz');rows[front].extend(record);means+=diag[front]['native_mean_computations']
    expected=read(run/'condition-profiles.json');public={front:condition_summary(v) for front,v in rows.items()};assert expected['heads']==public
    assert means==r['native_mean_computations'] and r['native_likelihood_fits']==r['optimizer_updates']==r['neural_head_fits']==0 and r['scene_ids']==[] and r['evaluation_splits']==['T_core']
    assert not list((run/'private').glob('*V_select*')) and not (run/'summary.json').exists()
    result=dict(status='COMPLETED',exit_code=0,T_scenes=len(r['training_scene_ids']),V_scenes=0,native_mean_computations=means,heads={front:dict(main_eligible_points=v['all']['points'],PG_harms=v['PG_harm']['points'],pure_PG_harms=v['pure_PG_harm']['points']) for front,v in public.items()},elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='saved main-point condition/cohort reconstruction and budget/split binding; descriptor extraction shared, not independent physical diagnosis')
    dest=run/'drift-condition-accounting.json'
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(__import__('json').dumps(result,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();r=read(a.run/'run.json');assert r['status']=='COMPLETED' and r['evaluation_splits']==['T_core']
    from tools.analysis.association_context_closeout import closeout
    closeout(a.run);account(a.run,a.run/'history-followup-accounting.json',(ARMS,PRIMARY,reconstruct,dict(fits=0,updates=0,native_likelihood_fits=0)));verify_conditions(a.run,r)
