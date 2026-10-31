"""178: saved-output audit, independent action/label accounting and tail strata."""
from __future__ import annotations
import argparse,json,time,resource,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.analysis.decoded_coverage_closeout import read,load,stats
from tools.analysis.audit_shared_control_arrays import position

ARMS={'HARD':('HARD','HM','HE','HZ','H1','HO'),'STRUCT':('STRUCT','SM','SE','SZ','S1','SO')}

def distance(a,b):
    aa,bb=np.broadcast_arrays(a[...,:2],b[...,:2]);return position(aa,bb,np.ones(aa.shape[:-1],bool))

def balanced(nodes,sources):
    # Independent per-source counting, no source-name ordering in the result.
    counts={s:int((sources[nodes]==s).sum()) for s in set(sources[nodes].tolist())}
    return np.array([1./len(counts)/counts[sources[i]] for i in nodes])

def center(v,w):
    if len(v)==1:return v[0,:2]
    w=w/w.sum();lat=w@v[:,0];r=np.deg2rad(v[:,1]);s=w@np.sin(r);c=w@np.cos(r)
    if np.hypot(s,c)<=1e-12:return v[np.lexsort((v[:,1],v[:,0]))[0],:2]
    return np.array([lat,np.rad2deg(np.arctan2(s,c))])

def verify_training(run,r):
    index=read(Path(r['data_run'])/'private/index.json')['T_core'];manifest=read(Path(r['manifest']));population=read(run/'training-population.json');folds=read(run/'training-folds.json');curves=read(run/'training-curves.json.gz');counts=[]
    assert r['optimizer_updates']==3328 and r['analytic_head_fits']==r['neural_head_fits']==13 and r['teacher_forward_calls']==r['frontend_optimizer_updates']==0
    for sid in r['fit_training_scene_ids']:
        z=load(run/'private'/f'training-{sid}.npz');original=sid in r['training_scene_ids'];native=Path(r['native_run'])/'private'
        m=load(native/'native-motion/T_core'/f'scene-{sid}.npz') if original else load(Path(index[sid]['motion']))
        ev=load(native/'native-evaluation/T_core'/f'scene-{sid}.npz') if original else load(Path(index[sid]['evaluation']))
        registry={str(v):i for i,v in enumerate(ev['target_ids'])}
        for front in ARMS:
            membership=z[front+'__membership'];used=np.zeros(m['support'].shape,bool);weights=np.zeros(m['support'].shape);target=np.full(m['support'].shape,np.nan);claimed=set();queries=duplicates=unknown=0
            for gid,g in enumerate(sorted(set(membership.tolist()))):
                members=np.flatnonzero(membership==g);identity=str(ev['identities'][members[0]]);count=sum(np.any(m['support'][members[m['sources'][members]==s]],axis=0) for s in set(m['sources'][members].tolist()))
                assert np.array_equal(count,z[front+'__count'][gid])
                if identity not in registry:unknown+=1;continue
                if identity in claimed:duplicates+=1;continue
                claimed.add(identity);t=registry[identity];valid=ev['target_valid'][t,:,:2].all(-1)&(count<=1);ix=np.ix_(members,np.flatnonzero(valid));used[ix]=True;weights[ix]=balanced(members,m['sources'])[:,None]
                target[ix]=np.log1p(distance(m['states'][members][:,valid],ev['target_states'][t,valid])/1000);queries+=int(valid.sum())
            assert np.array_equal(used,z[front+'__labeled']);np.testing.assert_allclose(weights,z[front+'__w'],atol=1e-14);np.testing.assert_allclose(target[used],z[front+'__y'][used],atol=1e-12)
            public=next(p for p in population if p['scene_id']==sid and p['front']==front)
            assert public['labeled_node_queries']==int(used.sum()) and public['group_queries']==queries and public['unknown_groups']==unknown and public['duplicate_groups_without_labels']==duplicates
            counts.append(dict(scene_id=sid,front=front,labels=int(used.sum()),weight_sum=float(.5*weights.sum()),component_id=str(manifest['components'][sid])))
    assert len(counts)==256
    components=sorted({str(manifest['components'][s]) for s in r['training_scene_ids']})
    assert {f['fold'] for f in folds}=={'full',*['leave_'+c for c in components]}
    for f in folds:
        rows=[a for a in counts if a['component_id']!=f['held_component']];assert f['training_scene_ids']==list(dict.fromkeys(a['scene_id'] for a in rows));assert f['labels']==sum(a['labels'] for a in rows);assert abs(f['weight_sum']-sum(a['weight_sum'] for a in rows))<1e-7
        model=torch.load(run/'private'/('model-'+f['fold']+'.pt'),map_location='cpu',weights_only=False);curve=next(c for c in curves if c['fold']==f['fold'])
        assert len(model['base']['beta'])==91 and model['dimension']==12 and model['parameter_count']==1505 and model['updates']==256
        assert sum(v.numel() for v in model['weights'].values())==1505 and curve['curve']==model['curve'] and len(curve['curve'])==256
        assert [c['update'] for c in curve['curve']]==list(range(1,257)) and all(np.isfinite(c['weighted_batch_mse']) for c in curve['curve'])
    return dict(T_scenes=128,front_views=256,label_rows=sum(a['labels'] for a in counts),weighted_group_queries=sum(a['weight_sum'] for a in counts),folds=len(folds),analytic_fits=13,neural_fits=13,optimizer_updates=3328,scope='independent first-anchor label/weight/population and leave-component scene exclusion; complete curve/checkpoint architecture/update accounting, not optimizer replay or physical truth certification')

def account(run,dest):
    if dest.exists():raise FileExistsError(dest)
    start=time.monotonic();r=read(run/'run.json');assert r['status']=='COMPLETED';result=dict(status='COMPLETED',exit_code=0,training=verify_training(run,r),splits={})
    comparisons=[(base,p) for base,arms in ARMS.items() for p in arms[1:]]+[(arms[1],arms[-1]) for arms in ARMS.values()]+[(arms[2],arms[-1]) for arms in ARMS.values()]+[('HARD','SO')]
    for split,ids,fn in [('T_core',r['training_scene_ids'],'T-summary.json'),('V_select',r['scene_ids'],'summary.json')]:
        pools={b+'__'+p:{} for b,p in comparisons};adoption={p:dict(deployed_low_queries=0,changed_position_queries=0,unsupported_positive_mass_queries=0,candidate_excess_over_1km=0,selected_error_sum_m=0.,oracle_error_sum_m=0.,labeled_queries=0) for arms in ARMS.values() for p in arms[1:]};examples={k:[] for k in pools};summary=read(run/fn)
        for sid in ids:
            native=Path(r['native_run'])/'private';m=load(native/'native-motion'/split/f'scene-{sid}.npz');ev=load(native/'native-evaluation'/split/f'scene-{sid}.npz');q=load(run/'private'/f'partitions-{split}-{sid}.npz');policy=load(run/'private'/f'policy-{split}-{sid}.npz');raw=load(run/'private'/f'raw-{split}-{sid}.npz');old=load(Path(r['parent_run'])/'private'/f'raw-{split}-{sid}.npz');oldq=load(Path(r['parent_run'])/'private'/f'partitions-{split}-{sid}.npz');saved=load(run/'private'/f'states-{split}-{sid}.npz');arms={};nt=len(ev['target_ids']);tv=ev['target_valid'][...,:2].all(-1)
            for base,prefixes in ARMS.items():
                assert np.array_equal(raw[base+'_NN'],old[base+'_NN']) and np.array_equal(raw[base+'_MEAN'],old[base+'_MEAN'])
                groups=[np.flatnonzero(q[base]==g) for g in sorted(set(q[base].tolist()))];scores=policy[base+'__scores'];registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set()
                for p in prefixes:
                    assert np.array_equal(q[p],oldq[base]) and np.array_equal(q[p+'_scores'],oldq[base+'_scores']) and np.array_equal(raw[p+'_MEAN'],old[base+'_MEAN'])
                for gid,members in enumerate(groups):
                    count=sum(np.any(m['support'][members[m['sources'][members]==s]],axis=0) for s in set(m['sources'][members].tolist()));assert np.array_equal(count,policy[base+'__count'][gid])
                    identity=str(ev['identities'][members[0]]);target=registry.get(identity);valid=np.zeros(64,bool) if target is None or identity in claimed else ev['target_valid'][target,:,:2].all(-1);claimed.add(identity)
                    for p,key in zip(prefixes[1:],('M','E','Z','1','O')):
                        weights=policy[p+'__weights'];assert np.array_equal(raw[p+'_NN'][gid,:,2:],old[base+'_NN'][gid,:,2:])
                        for j in range(64):
                            active=count[j]<=1 and (key!='Z' or count[j]==0) and (key!='1' or count[j]==1)
                            candidates=members[m['support'][members,j]] if key=='M' else members
                            w=np.zeros(len(members));selected=[]
                            if active and len(candidates):
                                selected=candidates if key=='E' else candidates[scores[candidates,j]==scores[candidates,j].min()];mass=balanced(selected,m['sources'])
                                for node,weight in zip(selected,mass):w[np.flatnonzero(members==node)[0]]=weight
                            np.testing.assert_allclose(weights[members,j],w,atol=1e-14)
                            if len(selected):
                                expected=center(m['states'][selected,j],mass);np.testing.assert_allclose(raw[p+'_NN'][gid,j,:2],expected,atol=1e-10,rtol=0)
                                a=adoption[p];a['deployed_low_queries']+=1;a['unsupported_positive_mass_queries']+=int(np.any((w>0)&~m['support'][members,j]));a['changed_position_queries']+=int(not np.array_equal(raw[p+'_NN'][gid,j,:2],old[base+'_NN'][gid,j,:2]))
                                if valid[j]:
                                    errs=distance(m['states'][members,j],ev['target_states'][target,j]);e=float(distance(raw[p+'_NN'][gid,j],ev['target_states'][target,j]));best=float(errs.min());a['labeled_queries']+=1;a['candidate_excess_over_1km']+=int(e-best>1000);a['selected_error_sum_m']+=e;a['oracle_error_sum_m']+=best
                            else:assert np.array_equal(raw[p+'_NN'][gid,j],old[base+'_NN'][gid,j])
                for p in prefixes:
                    prefix=p+'_NN_';a={k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)};gi=a['aligned_group_indices'][:nt];valid=a['aligned_prediction_valid'][:nt,:,:2].all(-1);pure=np.zeros(nt,bool);count=np.zeros((nt,64),int);anchor=np.full(nt,-1,int)
                    for t,g in enumerate(gi):
                        if g>=0:
                            members=groups[g];pure[t]=len(set(ev['identities'][members].tolist()))==1;anchor[t]=members[0];count[t]=policy[base+'__count'][g]
                    arms[p]=dict(error=position(a['aligned_prediction'][:nt],ev['target_states'],tv&valid),valid=valid,pure=pure,count=count,anchor=anchor)
            for b,p in comparisons:
                a,c=arms[b],arms[p];use=tv&a['valid']&c['valid'];d=c['error']-a['error'];key=b+'__'+p
                for label,mask in [('all',np.ones_like(use)),('zero',c['count']==0),('one',c['count']==1),('multi',c['count']>=2),('pure',np.broadcast_to(c['pure'][:,None],use.shape)),('mixed',np.broadcast_to(~c['pure'][:,None],use.shape)),('same_anchor',np.broadcast_to((a['anchor']==c['anchor'])[:,None],use.shape))]:
                    sel=use&mask;pool=pools[key].setdefault(label,dict(old=[],new=[],delta=[]))
                    for name,v in [('old',a['error']),('new',c['error']),('delta',d)]:pool[name].append(v[sel])
                for t,j in np.argwhere(use&(d>1000)):
                    examples[key].append(dict(scene_id=sid,query_index=int(j),old_error_m=float(a['error'][t,j]),new_error_m=float(c['error'][t,j]),increase_m=float(d[t,j]),new_source_support=int(c['count'][t,j]),pure=bool(c['pure'][t]),same_anchor=bool(a['anchor'][t]==c['anchor'][t])))
        physics={}
        for b,p in comparisons:
            key=b+'__'+p;out={}
            for label,values in pools[key].items():
                a,c,d=[np.concatenate(values[k]) for k in ('old','new','delta')];out[label]=dict(old=stats(a),new=stats(c),mean_delta_m=float(d.mean()) if len(d) else None,new_harm_over_1km=int((d>1000).sum()),rescued_over_1km=int((d< -1000).sum()),max_harm_m=float(max(0,d.max())) if len(d) else None)
            expected=summary['all_components']['pairwise'][p+'_NN_vs_'+b+'_NN'];assert out['all']['new_harm_over_1km']==expected['actual_harm']['new_harm_over_1km'];assert abs(out['all']['mean_delta_m']-expected['strata']['all']['candidate_minus_A0']['mean_error_change'])<1e-6
            physics[p+'_vs_'+b]=dict(strata=out,new_harm_example_count=len(examples[key]),examples=sorted(examples[key],key=lambda x:x['increase_m'],reverse=True)[:5])
        result['splits'][split]=dict(adoption=adoption,physics=physics)
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,scope='independent saved action weights/positions/domain invariance/original controls/T labels/folds/training amount and posthoc support-purity-anchor tail decomposition; no scientific retry/tuning/causal or peer certification')
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    from tools.analysis.association_context_closeout import closeout
    closeout(a.run);account(a.run,a.run/'history-candidate-accounting.json')
