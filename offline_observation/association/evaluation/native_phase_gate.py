"""Coarse native phase information and bounded readout of two motion estimates."""
from __future__ import annotations
import numpy as np
from association.evaluation import native_compatibility as nc, native_motion_interventions as ni
from association.evaluation.history_candidates import source_balanced
from association.evaluation.native_temporal import spherical_position
KINDS=('MEAN','BAG','ORDER')

def node_context(m,events):
    n,q=m['support'].shape;node_mc,_,diagnostic=ni.mean_velocity_control(m,events,[np.array([i]) for i in range(n)],m['support'].astype(int),np.ones((n,q)),m['states']);models=[]
    for node in range(n):
        a,b=events['event_offsets'][node:node+2];t=events['event_times_seconds'][a:b];xy=events['event_values'][a:b,:2];vv=events['event_values'][a:b,2:4];good=events['event_valid'][a:b,2:4].all(1);v=vv[:,0,None]*1852/3600*np.column_stack((np.cos(np.deg2rad(vv[:,1])),np.sin(np.deg2rad(vv[:,1]))));mu=v[good].mean(0) if good.sum()>=2 else np.zeros(2);spread=float(np.sqrt(np.mean(np.sum((v[good]-mu)**2,axis=1)))) if good.sum()>=2 else 0.;phase=np.zeros((4,5));bins=np.minimum((4*(t-t[0])/(t[-1]-t[0])).astype(int),3)
        for k in range(4):
            ids=np.flatnonzero(bins==k);known=ids[good[ids]]
            if len(ids)<2 or len(known)<2:continue
            report=v[known].mean(0);pos=nc.displacement(xy[ids[0]],xy[ids[-1]])/(t[ids[-1]]-t[ids[0]]);scale=max(float(np.linalg.norm(mu)),1e-12);phase[k]=[(report@mu)/scale/5,(mu[0]*report[1]-mu[1]*report[0])/scale/5,np.log1p(np.linalg.norm(pos-m['ols_velocity_mps'][node])),np.log1p(np.linalg.norm(report-pos)),1.]
        bag=phase[np.lexsort(tuple(phase[:,k] for k in range(4,-1,-1)))];models.append(dict(times=t[[0,-1]],mu=mu,spread=spread,phase=phase,bag=bag))
    return models,node_mc,diagnostic['native_mean_computations']

def cases(m,models,node_mc,groups,count,scores):
    rows=[];old=[];mc=[];features={k:[] for k in KINDS};eligible=np.zeros(count.shape,bool);sources=m['sources'];selections=[]
    for gid,nodes in enumerate(groups):
        nodes=np.asarray(nodes,int);first=min(models[i]['times'][0] for i in nodes);last=max(models[i]['times'][-1] for i in nodes)
        for j in np.flatnonzero((count[gid]==0)&((m['query_seconds']<first)|(m['query_seconds']>last))):
            eligible[gid,j]=True;chosen=nodes[scores[nodes,j]==scores[nodes,j].min()];w=source_balanced(chosen,sources);o=spherical_position(m['states'][chosen,j,:2],w);c=spherical_position(node_mc[chosen,j,:2],w);common=np.zeros(12);phases=np.zeros((4,5))
            for node,mass in zip(chosen,w):
                model=models[node];dt=m['query_seconds'][j]-model['times'][0 if m['query_seconds'][j]<first else -1];mu=model['mu'];ols=m['ols_velocity_mps'][node]
                common[:9]+=mass*np.array([np.log1p(abs(dt)/60),float(dt<0),np.log1p(m['span_seconds'][node]/60),np.log1p(m['native_count'][node]),np.log1p(m['fit_rmse_m'][node]/1000),np.log1p(np.linalg.norm(mu)),np.log1p(np.linalg.norm(ols)),np.log1p(np.linalg.norm(mu-ols)),np.log1p(model['spread'])]);phases+=mass*model['phase']
            common[9:]=[np.log1p(len(nodes)),np.log1p(len(set(sources[nodes].tolist()))),np.log1p(nc.distance(o,c)/1000)]
            bags=phases[np.lexsort(tuple(phases[:,k] for k in range(4,-1,-1)))]
            for k,v in [('MEAN',np.zeros((4,5))),('BAG',bags),('ORDER',phases)]:features[k].append(np.r_[common,v.ravel()])
            rows.append((gid,int(j)));old.append(o);mc.append(c);selections.append((chosen,w))
    return dict(indices=np.asarray(rows,int).reshape(-1,2),old=np.asarray(old).reshape(-1,2),mc=np.asarray(mc).reshape(-1,2),x={k:np.asarray(v).reshape(-1,32) for k,v in features.items()},eligible=eligible,selections=selections)

def blend(old,mc,alpha):
    old=np.asarray(old);mc=np.asarray(mc);alpha=np.asarray(alpha);out=np.empty_like(old,dtype=float)
    for i,(a,b,z) in enumerate(zip(old,mc,alpha)):out[i]=spherical_position(np.stack((a,b)),np.array([1-z,z]))
    return out

def labels(case,groups,ev):
    lookup={str(v):i for i,v in enumerate(ev['target_ids'])};claimed=set();target_by_group={}
    for gid,nodes in enumerate(groups):
        key=str(ev['identities'][nodes[0]])
        if key in lookup and key not in claimed:target_by_group[gid]=lookup[key];claimed.add(key)
    selected=[];target=[]
    for i,(gid,j) in enumerate(case['indices']):
        ti=target_by_group.get(gid)
        if ti is not None and ev['target_valid'][ti,j,:2].all():selected.append(i);target.append(ev['target_states'][ti,j,:2])
    return np.asarray(selected,int),np.asarray(target).reshape(-1,2)
