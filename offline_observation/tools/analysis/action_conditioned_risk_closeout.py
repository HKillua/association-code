"""195: independent saved-output/score/action audit and diagnostic opportunity accounting."""
from __future__ import annotations
import argparse,sys,time,json,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.analysis.action_conditioned_risk_pilot import ARMS,PRIMARY
from tools.analysis.history_followup_closeout import read,load,account,numpy_score,center,distance
from tools.analysis.native_motion_closeout import forward_position
from association.evaluation import action_conditioned_risk as action


def native(m,events):
    value=np.asarray(m['states']).copy();velocity=np.asarray(m['ols_velocity_mps']).copy();bounds=[];known=np.zeros(m['support'].shape,bool)
    for node,(a,b) in enumerate(zip(events['event_offsets'][:-1],events['event_offsets'][1:])):
        ts=events['event_times_seconds'][a:b];v=events['event_values'][a:b];valid=events['event_valid'][a:b,2:4].all(1);bounds.append(ts[[0,-1]])
        if valid.sum()<2:continue
        theta=np.deg2rad(v[valid,3]);speed=v[valid,2]*1852/3600;mu=np.array([np.mean(speed*np.cos(theta)),np.mean(speed*np.sin(theta))]);velocity[node]=mu
        for j,t in enumerate(m['query_seconds']):
            edge=0 if t<ts[0] else -1
            if (t<ts[0] or t>ts[-1]) and valid[edge] and not m['support'][node,j]:
                known[node,j]=True;value[node,j,:2]=forward_position(v[edge,:2],mu*(t-ts[edge]))
    return value,velocity,np.asarray(bounds),known


def reconstruct(run,r,split,sid,front,m,events,groups,count,op,oraw,raw,policy):
    assert split=='T_core';arms=ARMS[front];mc,velocity,bounds,known=native(m,events)
    prior=load(Path(r['control_run'])/'private'/f'raw-T_core-{sid}.npz')
    for prefix in arms[:7]:
        for suffix in ('_MEAN','_NN'):assert np.array_equal(raw[prefix+suffix],prior[prefix+suffix])
    x,eligible=action.features(m,groups,count,op[front+'__features'],mc,velocity,bounds,known)
    # Feature extraction shared; native motion, score algebra and decisions independently reconstructed.
    np.testing.assert_allclose(policy[front+'__features'],x,rtol=0,atol=1e-10)
    assert np.array_equal(policy[front+'__eligible'],eligible) and np.array_equal(policy[front+'__known'],known)
    manifest=read(Path(r['manifest']));model=torch.load(Path(r['parent_run'])/'private'/('model-leave_'+str(manifest['components'][sid])+'.pt'),map_location='cpu',weights_only=False)
    expected=numpy_score(model,x.reshape(-1,12)).reshape(m['support'].shape);expected[~known]=op[front+'__scores'][~known]
    score=policy[front+'__scores'];gap=float(np.max(np.abs(expected-score)));assert gap<1e-4
    for gid,group in enumerate(groups):
        nodes=np.asarray(group,int)
        for j in np.flatnonzero(eligible[gid]):
            original=nodes[op[front+'__scores'][nodes,j]==op[front+'__scores'][nodes,j].min()]
            for prefix,kind in zip(arms[7:],('CF','RM','JM')):
                pool=original if kind=='CF' else nodes;mode=[1] if kind=='RM' else [0,1]
                records=[(int(n),k,float(op[front+'__scores'][n,j] if k==0 else score[n,j])) for n in pool for k in mode]
                best=min(t[2] for t in records);chosen=[t for t in records if t[2]==best];sources={str(m['sources'][t[0]]) for t in chosen};w=np.zeros((len(m['states']),2))
                for n,k,_ in chosen:
                    own={t[0] for t in chosen if m['sources'][t[0]]==m['sources'][n]};modes=sum(t[0]==n for t in chosen);w[n,k]=1/len(sources)/len(own)/modes
                np.testing.assert_array_equal(policy[prefix+'__weights'][nodes,j],w[nodes])
                vals=np.array([m['states'][n,j,:2] if k==0 else mc[n,j,:2] for n,k,_ in chosen]);mass=np.array([w[n,k] for n,k,_ in chosen]);np.testing.assert_allclose(raw[prefix+'_NN'][gid,j,:2],center(vals,mass),rtol=0,atol=1e-10)
    for prefix in arms[7:]:assert np.array_equal(raw[prefix+'_NN'][~eligible],oraw[arms[1]+'_NN'][~eligible])
    return eligible,gap


def opportunities(run):
    began=time.monotonic();r=read(run/'run.json');assert r['evaluation_splits']==['T_core'] and r['scene_ids']==[]
    for key in ('optimizer_updates','neural_head_fits','analytic_head_fits','native_likelihood_fits'):assert r[key]==0
    pool={f:[] for f in ARMS};counts={f:dict(raw_eligible=0,unclaimed_or_invalid=0) for f in ARMS}
    for sid in r['training_scene_ids']:
        root=Path(r['native_run'])/'private';m=load(root/'native-motion/T_core'/f'scene-{sid}.npz');events=load(root/'native-events/T_core'/f'scene-{sid}.npz');ev=load(root/'native-evaluation/T_core'/f'scene-{sid}.npz');mc,_,_,_=native(m,events);q=load(run/'private'/f'partitions-T_core-{sid}.npz');pol=load(run/'private'/f'policy-T_core-{sid}.npz');op=load(Path(r['parent_run'])/'private'/f'policy-T_core-{sid}.npz');raw=load(run/'private'/f'raw-T_core-{sid}.npz');registry={str(t):i for i,t in enumerate(ev['target_ids'])};scene_rows=[]
        for front,arms in ARMS.items():
            claimed=set();groups=[np.flatnonzero(q[front]==v) for v in sorted(set(q[front].tolist()))];before=len(pool[front]);counts[front]['raw_eligible']+=int(pol[front+'__eligible'].sum())
            for gid,nodes in enumerate(groups):
                key=str(ev['identities'][nodes[0]])
                if key not in registry or key in claimed:continue
                claimed.add(key);ti=registry[key]
                for j in np.flatnonzero(pol[front+'__eligible'][gid]&ev['target_valid'][ti,:,:2].all(-1)):
                    target=ev['target_states'][ti,j];eo=distance(m['states'][nodes,j,:2],target);em=distance(mc[nodes,j,:2],target);chosen=op[arms[1]+'__weights'][nodes,j]>0
                    row=dict(scene_id=sid,front=front,group_index=gid,query_index=int(j),nodes=len(nodes),pure=len(set(ev['identities'][nodes].tolist()))==1,oracle_O=float(eo.min()),oracle_MC=float(em.min()),oracle_joint=float(min(eo.min(),em.min())),oracle_fixed=float(min(eo[chosen].min(),em[chosen].min())),best_node_sets_disjoint=not bool(set(nodes[eo==eo.min()])&set(nodes[em==em.min()])))
                    for label,prefix in [('O',arms[1]),('MC',arms[6]),('CF',arms[7]),('RM',arms[8]),('JM',arms[9])]:row[label]=float(distance(raw[prefix+'_NN'][gid,j,:2],target))
                    original=set(nodes[chosen].tolist())
                    for label,prefix in zip(('CF','RM','JM'),arms[7:]):
                        weights=pol[prefix+'__weights'][nodes,j];selected=set(nodes[weights.sum(1)>0].tolist());row[label+'_node_changed']=selected!=original;row[label+'_MC_mass']=float(weights[:,1].sum())
                    pool[front].append(row);scene_rows.append(row)
            counts[front]['unclaimed_or_invalid']+=int(pol[front+'__eligible'].sum())-(len(pool[front])-before)
        # IDs/indexed conditions remain private.
        dest=run/'private'/f'opportunity-T_core-{sid}.json.gz';assert not dest.exists();action_json=__import__('association.utils.runtime',fromlist=['write_json_gzip']);action_json.write_json_gzip(dest,scene_rows)
    summary={}
    for front,rows in pool.items():
        scopes={}
        for name,keep in [('all',lambda x:True),('pure',lambda x:x['pure']),('single_node',lambda x:x['nodes']==1),('multiple_nodes',lambda x:x['nodes']>1),('MC_old_harm',lambda x:x['MC']-x['O']>1000),('MC_old_rescue',lambda x:x['MC']-x['O']< -1000)]:
            v=[x for x in rows if keep(x)];d=dict(points=len(v),groups=len({(x['scene_id'],x['group_index']) for x in v}),best_node_sets_disjoint=sum(x['best_node_sets_disjoint'] for x in v))
            for key in ('O','MC','CF','RM','JM','oracle_O','oracle_MC','oracle_joint','oracle_fixed'):d[key+'_MAE']=float(np.mean([x[key] for x in v])) if v else None
            for key in ('CF','RM','JM'):
                d[key]=dict(node_changed=sum(x[key+'_node_changed'] for x in v),MC_mass_mean=float(np.mean([x[key+'_MC_mass'] for x in v])) if v else None,harms_vs_O=sum(x[key]-x['O']>1000 for x in v),rescues_vs_O=sum(x[key]-x['O']< -1000 for x in v))
            scopes[name]=d
        summary[front]=dict(**counts[front],scopes=scopes)
    report=dict(status='COMPLETED',heads=summary,elapsed_seconds=time.monotonic()-began,scope='post-prediction original first-claim T32 opportunity; oracle per-node states excludes tied output mixtures, not deployable or independent reference truth')
    dest=run/'action-opportunity-accounting.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    from tools.analysis.association_context_closeout import closeout
    closeout(a.run);account(a.run,a.run/'history-followup-accounting.json',(ARMS,PRIMARY,reconstruct,dict(fits=0,updates=0)));opportunities(a.run)
