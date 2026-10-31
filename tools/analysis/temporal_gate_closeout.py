"""193: saved T metrics plus matched training, native phase and gate accounting."""
from __future__ import annotations
import argparse,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import native_phase_gate as phase
from association.training.temporal_gate import numpy_predict
from tools.analysis.temporal_gate_pilot import ARMS,PRIMARY,OLD_ARMS
from tools.analysis.history_followup_closeout import read,load,account,center,distance
from tools.analysis.history_tail_qualification import describe


def verify_training(run,r):
    start=time.monotonic();manifest=read(Path(r['manifest']));folds=read(run/'training-folds.json');curves=read(run/'training-curves.json.gz');assert len(folds)==13 and len(curves)==39;pop=read(run/'training-population.json');assert [v['scene_id'] for v in pop]==r['fit_training_scene_ids'];assert r['optimizer_updates']==4992 and r['neural_head_fits']==39 and r['analytic_head_fits']==r['native_likelihood_fits']==r['teacher_forward_calls']==0;assert r['evaluation_splits']==['T_core'] and r['scene_ids']==[] and r['V_scenes']==0 and not (run/'summary.json').exists() and not list((run/'private').glob('*V_select*'));rows_total=0;max_gap=0.
    for f in folds:
        fold=f['fold'];data=load(run/'private'/f'training-{fold}.npz');n=len(data['w']);rows_total+=n;assert f['rows']==n and f['training_scene_ids']==list(dict.fromkeys(data['scene_ids'].tolist())) and np.all(data['w']==.5);assert all(str(manifest['components'][sid])!=f['held_component'] for sid in data['scene_ids']);assert np.array_equal(data['component_ids'],np.array([str(manifest['components'][sid]) for sid in data['scene_ids']]));assert f['frozen_risk_fold']==fold and f['updates_per_head']==128
        sample=load(run/'private'/f'sampling-{fold}.npz');assert np.array_equal(sample['draws'],np.random.default_rng(20).integers(n,size=(128,1024)));models=torch.load(run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False);initial=torch.load(run/'private'/f'initial-{fold}.pt',map_location='cpu',weights_only=False);assert sum(v.numel() for v in initial.values())==2145 and not initial['net.4.weight'].count_nonzero() and not initial['net.4.bias'].count_nonzero()
        x=data['x_ORDER'];w=data['w']/data['w'].sum();a=w@x[:,:12];block=x[:,12:].reshape(-1,4,5);b=np.sum(w[:,None,None]*block,axis=(0,1))/4;mean=np.r_[a,np.tile(b,4)];std=np.r_[np.sqrt(w@((x[:,:12]-a)**2)),np.tile(np.sqrt(np.sum(w[:,None,None]*(block-b)**2,axis=(0,1))/4),4)];std[std<1e-6]=1
        bag=np.array([v[np.lexsort(tuple(v[:,k] for k in range(4,-1,-1)))] for v in block]).reshape(n,20);assert np.array_equal(data['x_BAG'][:,12:],bag) and not data['x_MEAN'][:,12:].any()
        for k in phase.KINDS:
            assert np.array_equal(data['x_'+k][:,:12],x[:,:12]);model=models[k];assert model['updates']==128 and model['parameter_count']==2145 and sum(v.numel() for v in model['weights'].values())==2145;np.testing.assert_allclose(model['mean'],mean,atol=1e-12,rtol=0);np.testing.assert_allclose(model['std'],std,atol=1e-12,rtol=0);curve=next(v['curve'] for v in curves if v['fold']==fold and v['kind']==k);assert curve==model['curve'] and [v['update'] for v in curve]==list(range(1,129)) and all(np.isfinite(v['loss_km']) and np.isfinite(v['gradient_norm']) for v in curve)
    # Independent target binding to the original T registry, not a re-evaluation or fit.
    parent=read(Path(r['parent_run'])/'run.json');index=read(Path(parent['data_run'])/'private/index.json')['T_core'];datasets={f['fold']:load(run/'private'/f'training-{f["fold"]}.npz') for f in folds};bound=0
    for sid in parent['fit_training_scene_ids']:
        z=load(Path(r['parent_run'])/'private'/f'training-{sid}.npz');n=Path(r['native_run'])/'private';ev=load(n/'native-evaluation/T_core'/f'scene-{sid}.npz') if sid in r['training_scene_ids'] else load(Path(index[sid]['evaluation']));registry={str(v):i for i,v in enumerate(ev['target_ids'])}
        for front in ARMS:
            member=z[front+'__membership'];groups=[np.flatnonzero(member==v) for v in sorted(set(member.tolist()))];mapping={};claimed=set()
            for gid,nodes in enumerate(groups):
                key=str(ev['identities'][nodes[0]])
                if key in registry and key not in claimed:mapping[gid]=registry[key];claimed.add(key)
            for f in folds:
                d=datasets[f['fold']];ii=np.flatnonzero((d['scene_ids']==sid)&(d['fronts']==front))
                for k in ii:
                    gid,j=d['indices'][k];ti=mapping[gid];assert ev['target_valid'][ti,j,:2].all();assert np.array_equal(d['target'][k],ev['target_states'][ti,j,:2]);bound+=1
    assert bound==rows_total
    return dict(fits=39,updates=4992,folds=13,training_rows_across_folds=rows_total,target_bindings=bound,elapsed_seconds=time.monotonic()-start,scope='component exclusion, matched sampling/initial architecture, shared phase normalization, BAG multiset, curves and original reference binding; no optimizer replay or full-pipeline independence')


def reconstruct(run,r,split,sid,front,m,events,groups,count,op,oraw,raw,policy):
    assert split=='T_core';prior=load(Path(r['control_run'])/'private'/f'raw-{split}-{sid}.npz');arms=ARMS[front]
    for prefix in OLD_ARMS[front]:
        for s in ('_MEAN','_NN'):assert np.array_equal(raw[prefix+s],prior[prefix+s])
    native,node_mc,_=phase.node_context(m,events);case=phase.cases(m,native,node_mc,groups,count,op[front+'__scores']);assert np.array_equal(case['eligible'],policy[front+'__eligible']) and np.array_equal(case['indices'],policy[front+'__indices']);gid,j=case['indices'].T;manifest=read(Path(r['manifest']));models=torch.load(run/'private'/('model-leave_'+str(manifest['components'][sid])+'.pt'),map_location='cpu',weights_only=False);max_gap=0.
    for prefix,kind in zip(arms[7:],('HALF',)+phase.KINDS):
        alpha=policy[prefix+'__alpha'];assert alpha.shape==(len(gid),) and ((alpha>=0)&(alpha<=1)).all()
        if kind=='HALF':assert np.array_equal(alpha,np.full(len(alpha),.5))
        else:
            assert np.array_equal(policy[prefix+'__features'],case['x'][kind]);expected=numpy_predict(models[kind],case['x'][kind]);gap=float(np.max(np.abs(expected-alpha))) if len(alpha) else 0.;max_gap=max(gap,max_gap);assert gap<1e-5
        expected=np.array([center(np.stack((prior[arms[1]+'_NN'][g,k,:2],prior[arms[6]+'_NN'][g,k,:2])),np.array([1-a,a])) for (g,k),a in zip(case['indices'],alpha)]).reshape(-1,2);np.testing.assert_allclose(raw[prefix+'_NN'][gid,j,:2],expected,atol=1e-10,rtol=0);assert np.array_equal(raw[prefix+'_NN'][~case['eligible']],prior[arms[1]+'_NN'][~case['eligible']])
    return case['eligible'],max_gap


def old_cohorts(run,r,training):
    start=time.monotonic();heads={front:[] for front in ARMS}
    for sid in r['training_scene_ids']:
        raw=load(run/'private'/f'raw-T_core-{sid}.npz');policy=load(run/'private'/f'policy-T_core-{sid}.npz');ev=load(Path(r['native_run'])/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        for front,arms in ARMS.items():
            lookup={tuple(v):i for i,v in enumerate(policy[front+'__indices'])};old=read(Path(r['control_run'])/'private'/f'conditions-{front}-{sid}.json.gz')
            for v in old:
                gid,j=v['group_index'],v['query_index'];index=lookup[gid,j];row=dict(scene_id=sid,group_index=gid,pure=v['pure'],PG_delta_m=v['PG_delta_m']);row['heads']={}
                for prefix in arms[7:]:row['heads'][prefix]=dict(alpha=float(policy[prefix+'__alpha'][index]),delta_m=float(distance(raw[prefix+'_NN'][gid,j],ev['target_states'][v['target_index'],j])-v['old_error_m']))
                heads[front].append(row)
    public={}
    for front,rows in heads.items():
        assert len(rows)==(451 if front=='HARD' else 344);pub={}
        for name,predicate in [('all',lambda v:True),('PG_harm',lambda v:v['PG_delta_m']>1000),('PG_rescued',lambda v:v['PG_delta_m']< -1000)]:
            selected=[v for v in rows if predicate(v)];pub[name]=dict(points=len(selected),groups=len({(v['scene_id'],v['group_index']) for v in selected}),heads={prefix:dict(alpha=describe([v['heads'][prefix]['alpha'] for v in selected]),new_harm_over_1km=sum(v['heads'][prefix]['delta_m']>1000 for v in selected),rescued_over_1km=sum(v['heads'][prefix]['delta_m']< -1000 for v in selected)) for prefix in ARMS[front][7:]})
        assert pub['PG_harm']['points']==(34 if front=='HARD' else 1) and pub['PG_rescued']['points']==(3 if front=='HARD' else 33);public[front]=pub
    dest=run/'temporal-gate-accounting.json';assert not dest.exists();dest.write_text(__import__('json').dumps(dict(status='COMPLETED',exit_code=0,training=training,heads=public,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='planned original T PG harm/rescue cohort alpha and risk accounting; original descriptors shared, no physical mechanism certification'),indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args();r=read(a.run/'run.json');assert r['status']=='COMPLETED';from tools.analysis.association_context_closeout import closeout
    closeout(a.run);training=verify_training(a.run,r);account(a.run,a.run/'history-followup-accounting.json',(ARMS,PRIMARY,reconstruct,training));old_cohorts(a.run,r,training)
