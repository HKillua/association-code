"""190 independent least-squares/geodesic actions and original T harm cohorts."""
from __future__ import annotations
import argparse,sys,time,resource,json
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis.fitted_endpoint_pilot import ARMS,PRIMARY,OLD_ARMS
from tools.analysis.history_followup_closeout import read,load,account,center,distance
from tools.analysis.native_motion_closeout import forward_position
from tools.analysis.history_tail_qualification import describe

def independent_models(m,events):
    result=[]
    for node in range(len(m['states'])):
        a,b=events['event_offsets'][node:node+2];t=events['event_times_seconds'][a:b];values=events['event_values'][a:b];xy=values[:,:2].copy();raw=np.diff(xy[:,1]);turn=(raw+180)%360-180;turn=np.where(abs(turn)==180,np.copysign(180,raw),turn);xy[:,1]=xy[0,1]+np.r_[0.,np.cumsum(turn)];X=np.column_stack([np.ones(len(t)),t-t.mean()]);mean=xy.mean(0);coeff=np.linalg.lstsq(X,xy-mean,rcond=None)[0];fitted=mean+np.column_stack([np.ones(2),t[[0,-1]]-t.mean()])@coeff;fitted[:,1]=(fitted[:,1]+180)%360-180;velocity=coeff[1]*6371000*np.pi/180*np.array([1.,np.cos(np.deg2rad(xy[:,0].mean()))]);good=events['event_valid'][a:b,2:4].all(1);theta=np.deg2rad(values[good,3]);s=values[good,2]*1852/3600;mu=np.array([np.mean(s*np.cos(theta)),np.mean(s*np.sin(theta))]) if good.sum()>=2 else np.zeros(2)
        result.append(dict(times=t[[0,-1]],native=values[[0,-1],:2].copy(),fitted=fitted,velocity=velocity,mu=mu,report_known=good[[0,-1]]&(good.sum()>=2),mean_known=bool(good.sum()>=2)))
    return result

def reconstruct(run,r,split,sid,front,m,events,groups,count,op,oraw,raw,policy):
    assert split=='T_core';short='H' if front=='HARD' else 'S';prior=load(Path(r['ablation_run'])/'private'/f'raw-{split}-{sid}.npz');old_policy=load(Path(r['ablation_run'])/'private'/f'policy-{split}-{sid}.npz');models=independent_models(m,events);saved=load(run/'private'/f'fitted-models-{split}-{sid}.npz')
    for key in saved:
        expected=np.stack([v[key] for v in models])
        if key in ('report_known','mean_known'):assert np.array_equal(saved[key],expected)
        elif key in ('native','fitted'):
            np.testing.assert_allclose(saved[key][...,0],expected[...,0],rtol=0,atol=1e-10);np.testing.assert_allclose((saved[key][...,1]-expected[...,1]+180)%360-180,0,rtol=0,atol=1e-10)
        else:np.testing.assert_allclose(saved[key],expected,rtol=0,atol=1e-10)
    for prefix in OLD_ARMS[front]:
        for suffix in ('_MEAN','_NN'):assert np.array_equal(raw[prefix+suffix],prior[prefix+suffix])
    base=prior[short+'MC_NN'];outputs={key:base.copy() for key in ('GL','FL','FM')};eligible=np.zeros(count.shape,bool);joint=eligible.copy()
    for gid,nodes in enumerate(groups):
        first=min(models[n]['times'][0] for n in nodes);last=max(models[n]['times'][1] for n in nodes)
        for j in np.flatnonzero((count[gid]==0)&((m['query_seconds']<first)|(m['query_seconds']>last))):
            eligible[gid,j]=True;chosen=nodes[op[short+'O__weights'][nodes,j]>0];mass=op[short+'O__weights'][chosen,j];values={k:[] for k in outputs};valid=True
            for node in chosen:
                model=models[node];edge=0 if m['query_seconds'][j]<model['times'][0] else 1;dt=m['query_seconds'][j]-model['times'][edge];anchor=model['fitted'][edge]
                if not model['report_known'][edge] or abs(anchor[0])>90 or not np.isfinite(anchor).all():valid=False;break
                for key,p,v in [('GL',model['native'][edge],model['velocity']),('FL',anchor,model['velocity']),('FM',anchor,model['mu'])]:values[key].append(forward_position(p,v*dt))
            if not valid:continue
            joint[gid,j]=True
            for key in outputs:outputs[key][gid,j,:2]=center(np.stack(values[key]),mass)
    assert np.array_equal(eligible,policy[front+'__eligible']) and np.array_equal(eligible,old_policy[front+'__eligible']) and np.array_equal(joint,policy[front+'__joint'])
    for key in outputs:
        actual=raw[short+key+'_NN'];np.testing.assert_allclose(actual[...,0],outputs[key][...,0],rtol=0,atol=1e-10);np.testing.assert_allclose((actual[...,1]-outputs[key][...,1]+180)%360-180,0,rtol=0,atol=1e-10);assert np.array_equal(actual[~joint],base[~joint])
    return eligible,0.

def case_account(run,r):
    start=time.monotonic();out=run/'endpoint-case-accounting.json'
    if out.exists():raise FileExistsError(out)
    allrows={front:[] for front in ARMS};models_count=means_count=0
    for sid in r['training_scene_ids']:
        native=Path(r['native_run'])/'private';ev=load(native/'native-evaluation/T_core'/f'scene-{sid}.npz');raw=load(run/'private'/f'raw-T_core-{sid}.npz');models=load(run/'private'/f'fitted-models-T_core-{sid}.npz');models_count+=len(models['times']);means_count+=int(models['mean_known'].sum())
        for front in ARMS:
            short='H' if front=='HARD' else 'S';rows=read(Path(r['ablation_run'])/'private'/f'conditions-{front}-{sid}.json.gz')
            for row in rows:
                gid,j,ti=(row[k] for k in ('group_index','query_index','target_index'));value=dict(PG_harm=row['PG_delta_m']>1000,PG_rescued=row['PG_delta_m']< -1000,singleton=row['group_nodes']==1)
                for key in ('GL','FL','FM'):value[key]=float(distance(raw[short+key+'_NN'][gid,j],ev['target_states'][ti,j])-row['old_error_m'])
                allrows[front].append(value)
    assert models_count==r['native_OLS_fits'] and means_count==r['native_mean_computations'] and r['evaluation_splits']==['T_core'] and r['scene_ids']==[] and r['optimizer_updates']==r['native_likelihood_fits']==0
    result={}
    for front,rows in allrows.items():
        result[front]={}
        for cohort,condition in [('all',lambda v:True),('PG_harm',lambda v:v['PG_harm']),('PG_rescued',lambda v:v['PG_rescued'])]:
            use=[v for v in rows if condition(v)];result[front][cohort]=dict(points=len(use),singletons=sum(v['singleton'] for v in use),methods={key:dict(delta_vs_O_m=describe([v[key] for v in use]),harms_over_1km=sum(v[key]>1000 for v in use),rescued_over_1km=sum(v[key]< -1000 for v in use)) for key in ('GL','FL','FM')})
    out.write_text(json.dumps(dict(status='COMPLETED',exit_code=0,heads=result,native_OLS_fits=models_count,native_mean_computations=means_count,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='original189 reference-aligned T eligible harm/rescue cohorts under new saved actions; no deployed reference gate or physical cause guarantee'),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();r=read(a.run/'run.json');assert r['status']=='COMPLETED' and r['evaluation_splits']==['T_core']
    from tools.analysis.association_context_closeout import closeout
    closeout(a.run);account(a.run,a.run/'history-followup-accounting.json',(ARMS,PRIMARY,reconstruct,dict(new_neural_fits=0,updates=0,native_likelihood_fits=0,native_OLS_fits=r['native_OLS_fits'])));case_account(a.run,r)
