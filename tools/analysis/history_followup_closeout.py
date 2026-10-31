"""180/181: independent saved metrics plus objective/native bracket accounting."""
from __future__ import annotations
import argparse,time,resource,sys,json
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.analysis.history_candidate_closeout import read,load,distance,center,balanced
from tools.analysis.audit_shared_control_arrays import position
from tools.analysis.decoded_coverage_closeout import stats

def numpy_score(model,x):
    z=np.clip((x-model['base']['mean'])/model['base']['std'],-8,8);i,j=np.triu_indices(12);base=np.column_stack([np.ones(len(x)),z,z[:,i]*z[:,j]])@model['base']['beta']
    z=np.clip((x-model['mean'])/model['std'],-8,8).astype(np.float32);w={k:v.numpy() for k,v in model['weights'].items()}
    for layer in (0,2):
        z=z@w[f'net.{layer}.weight'].T+w[f'net.{layer}.bias'];z=z/(1+np.exp(-z))
    return base+(z@w['net.4.weight'].T+w['net.4.bias']).ravel()

def verify_training(run,r):
    parent=Path(r['parent_run']);old=read(parent/'run.json');manifest=read(Path(r['manifest']));bags=[];population=[]
    for sid in old['fit_training_scene_ids']:
        z=load(parent/'private'/f'training-{sid}.npz')
        for front in ('HARD','STRUCT'):
            member=z[front+'__membership'];count=z[front+'__count'];rows=nodes_count=single=0
            for gid,label in enumerate(sorted(set(member.tolist()))):
                nodes=np.flatnonzero(member==label);qs=np.flatnonzero((count[gid]==1)&z[front+'__labeled'][nodes[0]])
                if len(nodes)==1:single+=len(qs);continue
                for j in qs:bags.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),front=front,group_index=gid,query_index=int(j)))
                rows+=len(qs);nodes_count+=len(nodes)*len(qs)
            population.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),front=front,decision_queries=rows,decision_node_rows=nodes_count,single_candidate_queries_without_updates=single))
    assert population==read(run/'training-population.json') and bags==read(run/'private/training-bags.json.gz')
    folds=read(run/'training-folds.json');curves=read(run/'training-curves.json.gz');assert len(folds)==13 and len(curves)==39
    for f in folds:
        ids=np.array([i for i,b in enumerate(bags) if b['component_id']!=f['held_component']]);sample=load(run/'private'/f'sampling-{f["fold"]}.npz');assert np.array_equal(ids,sample['training_bag_indices']);expected=np.random.default_rng(20).integers(len(ids),size=(256,128));assert np.array_equal(expected,sample['local_draws'])
        assert f['training_bags']==len(ids) and f['query_presentations_per_head']==32768 and f['training_scene_ids']==list(dict.fromkeys(bags[i]['scene_id'] for i in ids))
        original=torch.load(parent/'private'/f'model-{f["fold"]}.pt',map_location='cpu',weights_only=False);models=torch.load(run/'private'/f'model-{f["fold"]}.pt',map_location='cpu',weights_only=False)
        for kind,m in models.items():
            assert m['updates']==256 and m['parameter_count']==1505 and sum(v.numel() for v in m['weights'].values())==1505
            for key in ('mean','std'):assert np.array_equal(m[key],original[key])
            for key in ('mean','std','beta'):assert np.array_equal(m['base'][key],original['base'][key])
            c=next(c for c in curves if c['fold']==f['fold'] and c['kind']==kind);assert c['curve']==m['curve'] and [a['update'] for a in c['curve']]==list(range(1,257))
    journal=[json.loads(x) for x in (run/'training-journal.jsonl').read_text().splitlines()];assert len(journal)==3328 and r['optimizer_updates']==9984 and r['neural_head_fits']==39 and r['analytic_head_fits']==0
    return dict(bags=len(bags),candidate_rows=sum(p['decision_node_rows'] for p in population),folds=13,neural_fits=39,updates=9984,query_presentations_per_head_all_folds=13*32768,scope='independent decision bag population, component exclusion, exact shared sampling, frozen bases/normalizers, checkpoints and journals; no optimizer replay')

def selected_weights(scores,nodes,sources,active):
    chosen=(scores[nodes]==scores[nodes].min(0))&active[None,:];w=np.zeros(chosen.shape,float);classes=set(sources[nodes].tolist());sourcecount=np.zeros(chosen.shape[1])
    for s in classes:
        use=sources[nodes]==s;n=chosen[use].sum(0);sourcecount+=n>0;w[use]=chosen[use]/np.maximum(n,1)
    return w/np.maximum(sourcecount,1)

def weighted_positions(values,w):
    lat=np.sum(values[...,0]*w,axis=0);lon=np.deg2rad(values[...,1]);sin=np.sum(np.sin(lon)*w,axis=0);cos=np.sum(np.cos(lon)*w,axis=0);out=np.column_stack([lat,np.rad2deg(np.arctan2(sin,cos))])
    for j in np.flatnonzero((w.sum(0)>0)&(np.hypot(sin,cos)<=1e-12)):
        selected=values[w[:,j]>0,j,:2];out[j]=selected[np.lexsort((selected[:,1],selected[:,0]))[0]]
    return out

def native_expected(m,events,groups,count,scores,base):
    expected={k:base.copy() for k in ('R','N','B','E')};eligible=np.zeros(count.shape,bool);sourcecount=np.zeros(count.shape,int)
    for gid,nodes in enumerate(groups):
        native=[]
        for source in set(m['sources'][nodes].tolist()):
            own=nodes[m['sources'][nodes]==source];indices=np.concatenate([np.arange(*events['event_offsets'][i:i+2]) for i in own]);indices=indices[events['event_valid'][indices,:2].all(1)];times=events['event_times_seconds'][indices];values=events['event_values'][indices,:2];tt=np.unique(times);positions=np.array([center(values[times==t],np.ones(int((times==t).sum()))) for t in tt]);native.append((own,tt,positions))
        for j in np.flatnonzero(count[gid]==0):
            t=m['query_seconds'][j];items=[]
            for own,tt,positions in native:
                li=np.searchsorted(tt,t,side='left')-1;ri=np.searchsorted(tt,t,side='right')
                if li<0 or ri>=len(tt):continue
                tl,tr=tt[li],tt[ri];v=positions[[li,ri]];alpha=(t-tl)/(tr-tl);b=center(v,np.array([1-alpha,alpha]));n=center(v,np.array([float(t-tl<=tr-t),float(t-tl>=tr-t)]));best=own[scores[own,j]==scores[own,j].min()];prior=center(m['states'][best,j,:2],np.ones(len(best)));items.append((float(scores[best[0],j]),prior,n,b))
            if not items:continue
            eligible[gid,j]=True;sourcecount[gid,j]=len(items);best=min(v[0] for v in items);chosen=[v for v in items if v[0]==best]
            for key,index in [('R',1),('N',2),('B',3)]:expected[key][gid,j,:2]=center(np.stack([v[index] for v in chosen]),np.ones(len(chosen)))
            expected['E'][gid,j,:2]=center(np.stack([v[3] for v in items]),np.ones(len(items)))
    return expected,eligible,sourcecount

def account(run,dest):
    if dest.exists():raise FileExistsError(dest)
    start=time.monotonic();r=read(run/'run.json');assert r['status']=='COMPLETED';cost=r['mode']=='cost_routing'
    if cost:
        from tools.analysis.cost_routing_pilot import ARMS,PRIMARY
    else:
        from tools.analysis.native_history_bridge_pilot import ARMS,PRIMARY
    parent=Path(r['parent_run']);old=read(parent/'run.json');result=dict(status='COMPLETED',exit_code=0,training=verify_training(run,r) if cost else dict(updates=0,fits=0),splits={});max_score_gap=0.
    comparisons=list(dict.fromkeys(list(PRIMARY)+[new+'_NN_vs_'+front+'_NN' for front,arms in ARMS.items() for new in arms[2:]]))
    for split,ids,fn in [('T_core',r['training_scene_ids'],'T-summary.json'),('V_select',r['scene_ids'],'summary.json')]:
        pool={k:{} for k in comparisons};harm_changes={};eligible_total={k:0 for k in ARMS}
        for sid in ids:
            m=load(Path(r['native_run'])/'private/native-motion'/split/f'scene-{sid}.npz');ev=load(Path(r['native_run'])/'private/native-evaluation'/split/f'scene-{sid}.npz');q=load(run/'private'/f'partitions-{split}-{sid}.npz');raw=load(run/'private'/f'raw-{split}-{sid}.npz');policy=load(run/'private'/f'policy-{split}-{sid}.npz');oq=load(parent/'private'/f'partitions-{split}-{sid}.npz');op=load(parent/'private'/f'policy-{split}-{sid}.npz');oraw=load(parent/'private'/f'raw-{split}-{sid}.npz');saved=load(run/'private'/f'states-{split}-{sid}.npz');nt=len(ev['target_ids']);tv=ev['target_valid'][...,:2].all(-1);arms_info={}
            if cost:
                manifest=read(Path(r['manifest']));fold='leave_'+str(manifest['components'][sid]) if split=='T_core' else 'full';models=torch.load(run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)
            else:events=load(Path(r['native_run'])/'private/native-events'/split/f'scene-{sid}.npz')
            for front,prefixes in ARMS.items():
                groups=[np.flatnonzero(oq[front]==g) for g in sorted(set(oq[front].tolist()))];count=op[front+'__count'];base=oraw[prefixes[1]+'_NN']
                assert np.array_equal(raw[front+'_NN'],oraw[front+'_NN']) and np.array_equal(raw[prefixes[1]+'_NN'],base)
                for prefix in prefixes:
                    assert np.array_equal(q[prefix],oq[front]) and np.array_equal(q[prefix+'_scores'],oq[front+'_scores']) and np.array_equal(raw[prefix+'_MEAN'],oraw[front+'_MEAN'])
                    if prefix in prefixes[2:]:assert np.array_equal(raw[prefix+'_NN'][...,2:],base[...,2:])
                if cost:
                    for prefix,kind in zip(prefixes[2:],('MSE','COST','TAIL')):
                        score=policy[prefix+'__scores'];expected=numpy_score(models[kind],op[front+'__features'].reshape(-1,12)).reshape(score.shape);gap=float(np.max(np.abs(score-expected)));max_score_gap=max(max_score_gap,gap);assert gap<1e-4
                        for gid,nodes in enumerate(groups):
                            active=count[gid]==1;w=selected_weights(score,nodes,m['sources'],active);np.testing.assert_allclose(policy[prefix+'__weights'][nodes],w,atol=1e-12);pos=weighted_positions(m['states'][nodes],w)
                            np.testing.assert_allclose(raw[prefix+'_NN'][gid,active,:2],pos[active],atol=1e-10,rtol=0);assert np.array_equal(raw[prefix+'_NN'][gid,~active],base[gid,~active])
                    eligible=count==1
                else:
                    expected,eligible,sourcecount=native_expected(m,events,groups,count,op[front+'__scores'],base);assert np.array_equal(eligible,policy[front+'__eligible']);assert np.array_equal(sourcecount,policy[front+'__eligible_source_count'])
                    for prefix,key in zip(prefixes[2:],('R','N','E','B')):
                        np.testing.assert_allclose(raw[prefix+'_NN'][...,:2],expected[key][...,:2],atol=1e-10,rtol=0);assert np.array_equal(raw[prefix+'_NN'][~eligible],base[~eligible])
                eligible_total[front]+=int(eligible.sum())
                for prefix in prefixes:
                    name=prefix+'_NN_';a={k[len(name):]:v for k,v in saved.items() if k.startswith(name)};gi=a['aligned_group_indices'][:nt];valid=a['aligned_prediction_valid'][:nt,:,:2].all(-1);pure=np.zeros(nt,bool);counts=np.zeros((nt,64),int);use_eligible=np.zeros((nt,64),bool)
                    for t,gid in enumerate(gi):
                        if gid>=0:pure[t]=len(set(ev['identities'][groups[gid]].tolist()))==1;counts[t]=count[gid];use_eligible[t]=eligible[gid]
                    arms_info[prefix]=dict(error=position(a['aligned_prediction'][:nt],ev['target_states'],tv&valid),valid=valid,pure=pure,count=counts,eligible=use_eligible)
                a,b=arms_info[front],arms_info[prefixes[1]]
                for prefix in prefixes[2:]:
                    c=arms_info[prefix];use=tv&a['valid']&b['valid']&c['valid'];oldbad=(b['error']-a['error'])>1000;newbad=(c['error']-a['error'])>1000;v=harm_changes.setdefault(prefix,{k:0 for k in ('old_harms','new_harms','resolved','introduced','persistent','pure_old_harms','pure_new_harms')})
                    for key,mask in [('old_harms',oldbad),('new_harms',newbad),('resolved',oldbad&~newbad),('introduced',~oldbad&newbad),('persistent',oldbad&newbad),('pure_old_harms',oldbad&b['pure'][:,None]),('pure_new_harms',newbad&c['pure'][:,None])]:v[key]+=int((use&mask).sum())
            for key in comparisons:
                right,left=key.split('_vs_');a,b=arms_info[left[:-3]],arms_info[right[:-3]];use=tv&a['valid']&b['valid'];delta=b['error']-a['error']
                for scope,mask in [('all',np.ones_like(use)),('zero',b['count']==0),('one',b['count']==1),('multi',b['count']>1),('pure',np.broadcast_to(b['pure'][:,None],use.shape)),('mixed',np.broadcast_to(~b['pure'][:,None],use.shape)),('eligible',b['eligible'])]:
                    v=pool[key].setdefault(scope,dict(old=[],new=[],delta=[]));sel=use&mask
                    for field,values in [('old',a['error']),('new',b['error']),('delta',delta)]:v[field].append(values[sel])
        report={};summary=read(run/fn)
        for key,scopes in pool.items():
            report[key]={}
            for scope,values in scopes.items():
                a,b,d=[np.concatenate(values[k]) for k in ('old','new','delta')];report[key][scope]=dict(old=stats(a),new=stats(b),mean_delta_m=float(d.mean()) if len(d) else None,new_harm_over_1km=int((d>1000).sum()),rescued_over_1km=int((d< -1000).sum()),max_harm_m=float(max(0,d.max())) if len(d) else None)
            assert report[key]['all']['new_harm_over_1km']==summary['all_components']['pairwise'][key]['actual_harm']['new_harm_over_1km']
        result['splits'][split]=dict(physics=report,harm_changes_relative_original_base=harm_changes,raw_eligible_group_queries=eligible_total)
    result.update(elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,max_numpy_score_gap=max_score_gap,scope='independent main populations/positions/relations via generic audit, same-evaluator full four-attribute accounting, plus independent frozen controls, actions or native brackets, training/folds/sampling and support-purity-harm strata; not independent implementations of all attributes, physical truth, optimizer replay or causal guarantee')
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    r=read(a.run/'run.json')
    if r.get('phase')=='NO_T_CONDITION':
        from tools.analysis.native_history_bridge_pilot import ARMS
        for sid in r['training_scene_ids']:
            policy=load(a.run/'private'/f'policy-T_core-{sid}.npz');raw=load(a.run/'private'/f'raw-T_core-{sid}.npz');old=load(Path(r['parent_run'])/'private'/f'raw-T_core-{sid}.npz')
            for front,arms in ARMS.items():
                assert not policy[front+'__eligible'].any()
                for p in arms[2:]:assert np.array_equal(raw[p+'_NN'],old[arms[1]+'_NN'])
        (a.run/'history-followup-accounting.json').write_text(json.dumps(dict(status='COMPLETED',exit_code=0,T_scenes=32,V_scenes=0,scope='no T bracket condition; all new positions exact frozen178; full V metric audit inapplicable'))+'\n')
    else:
        from tools.analysis.association_context_closeout import closeout
        closeout(a.run);account(a.run,a.run/'history-followup-accounting.json')
