"""Declared endpoint-derivative and one-sided OU working-prior interventions."""
from __future__ import annotations
import numpy as np
from association.evaluation import native_compatibility as nc
from association.evaluation.native_temporal import spherical_position

def hermite(a,p0,p1,d0,d1,span,kind):
    delta=np.asarray(p1)-p0;d0,d1=np.asarray(d0).copy(),np.asarray(d1).copy()
    if kind=='linear':return p0+a*delta
    if kind=='limited':
        sec=delta/span
        for d in (d0,d1):
            nonzero=sec!=0;ratio=np.zeros(2);ratio[nonzero]=d[nonzero]/sec[nonzero];d[:]=np.clip(ratio,0,3)*sec
    return (2*a**3-3*a*a+1)*p0+(a**3-2*a*a+a)*span*d0+(-2*a**3+3*a*a)*p1+(a**3-a*a)*span*d1

def bridge(m,events,groups,count,scores,base,old_bridge):
    out={k:old_bridge.copy() for k in ('CL','CH','CM')};eligible=np.zeros(count.shape,bool);records=[];fallbacks=0
    for gid,nodes in enumerate(groups):
        streams=[]
        for source in set(m['sources'][nodes].tolist()):
            own=nodes[m['sources'][nodes]==source];streams.append((own,nc.stream(events,own,speed_unit='kn')))
        for j in np.flatnonzero(count[gid]==0):
            t=m['query_seconds'][j];items=[]
            for own,s in streams:
                left=np.searchsorted(s['t'],t,side='left')-1;right=np.searchsorted(s['t'],t,side='right')
                if left<0 or right>=len(s['t']):continue
                p0,p1=s['p'][left].copy(),s['p'][right].copy();p1[1]=p0[1]+(p1[1]-p0[1]+180)%360-180;span=s['t'][right]-s['t'][left];a=(t-s['t'][left])/span
                scale=nc.RADIUS*np.pi/180*np.array([1.,np.cos(np.deg2rad((p0[0]+p1[0])/2))]);valid=bool(s['known'][left] and s['known'][right] and abs(scale[1])>1e-9)
                values={}
                if valid:
                    for key,kind in [('CL','linear'),('CH','raw'),('CM','limited')]:
                        value=hermite(a,p0,p1,s['v'][left]/scale,s['v'][right]/scale,span,kind);value[1]=(value[1]+180)%360-180;values[key]=value
                    valid=all(np.isfinite(v).all() and abs(v[0])<=90 for v in values.values())
                items.append(dict(score=float(scores[own,j].min()),valid=valid,values=values,span=float(span),alpha=float(a)))
            if not items:continue
            eligible[gid,j]=True;best=min(x['score'] for x in items);chosen=[x for x in items if x['score']==best]
            if not all(x['valid'] for x in chosen):fallbacks+=1;continue
            for key in out:out[key][gid,j,:2]=spherical_position(np.stack([x['values'][key] for x in chosen]),np.ones(len(chosen)))
            records.append(dict(group_index=gid,query_index=int(j),selected_sources=len(chosen),span_seconds=[x['span'] for x in chosen],alpha=[x['alpha'] for x in chosen]))
    return out,dict(eligible=eligible),dict(eligible_queries=int(eligible.sum()),joint_domain_fallbacks=fallbacks,decisions=records)

def fit_ou(times,velocity,drift):
    times,velocity=np.asarray(times),np.asarray(velocity);dt=np.diff(times);mu=velocity.mean(0) if drift else np.zeros(2);u=velocity-mu
    if len(dt)<1 or (dt<=0).any():raise ValueError('strict native velocity times required')
    grid=np.r_[0.,np.logspace(-6,2,65)/np.median(dt)];cost=np.empty(len(grid));sigma=np.empty(len(grid));floors=0
    constant=bool(np.array_equal(velocity[1:],velocity[:-1]))
    if constant:return dict(beta=0.,mean=mu,grid_index=0,grid_size=66,constant_limit=True,variance_floor_candidates=0,profile_variance=0.)
    for k,b in enumerate(grid):
        q=dt if b==0 else -np.expm1(-2*b*dt)/(2*b);res=u[1:]-np.exp(-b*dt)[:,None]*u[:-1];variance=float(np.sum(np.sum(res*res,axis=1)/q)/(2*len(dt)));floors+=int(variance<1e-24);sigma[k]=variance;cost[k]=len(dt)*np.log(max(variance,1e-24))+np.log(q).sum()
    ix=int(np.argmin(cost));return dict(beta=float(grid[ix]),mean=mu,grid_index=ix,grid_size=66,constant_limit=False,variance_floor_candidates=floors,profile_variance=float(sigma[ix]))

def ou_displacement(v,mu,beta,dt):
    h=float(dt) if beta==0 else np.sign(dt)*(-np.expm1(-beta*abs(dt)))/beta
    return mu*dt+(v-mu)*h

def persistence(m,events,groups,count,weights,base):
    out={k:base.copy() for k in ('PH','PC','PD','PG')};models=[];native=[];fit_count=constant=boundary=floors=0
    for node in range(len(m['states'])):
        a,b=events['event_offsets'][node:node+2];t=events['event_times_seconds'][a:b];values=events['event_values'][a:b];valid=events['event_valid'][a:b,2:4].all(1);speed=values[:,2]*1852/3600;theta=np.deg2rad(values[:,3]);v=speed[:,None]*np.column_stack([np.cos(theta),np.sin(theta)]);fits={}
        if valid.sum()>=2:
            for key,drift in [('PD',False),('PG',True)]:
                fits[key]=fit_ou(t[valid],v[valid],drift);fit_count+=1;constant+=int(fits[key]['constant_limit']);boundary+=int(fits[key]['grid_index'] in (0,65));floors+=fits[key]['variance_floor_candidates']
        models.append(fits);native.append((t,values[:,:2],v,valid))
    eligible=np.zeros(count.shape,bool);unknown=0;decisions=[]
    for gid,nodes in enumerate(groups):
        first=min(native[i][0][0] for i in nodes);last=max(native[i][0][-1] for i in nodes)
        for j in np.flatnonzero((count[gid]==0)&((m['query_seconds']<first)|(m['query_seconds']>last))):
            eligible[gid,j]=True;chosen=nodes[weights[nodes,j]>0];mass=weights[chosen,j];values={k:[] for k in out};dt_values=[]
            for node in chosen:
                t,p,v,valid=native[node];edge=0 if m['query_seconds'][j]<t[0] else -1;dt=float(m['query_seconds'][j]-t[edge]);dt_values.append(dt)
                if not models[node] or not valid[edge]:
                    unknown+=1
                    for key in values:values[key].append(m['states'][node,j,:2])
                    continue
                values['PH'].append(p[edge]);values['PC'].append(nc.advance(p[edge],v[edge],dt))
                for key in ('PD','PG'):
                    model=models[node][key];delta=ou_displacement(v[edge],model['mean'],model['beta'],dt);values[key].append(nc.advance(p[edge],delta,1.))
            for key in out:out[key][gid,j,:2]=spherical_position(np.stack(values[key]),mass)
            decisions.append(dict(group_index=gid,query_index=int(j),selected_nodes=chosen.tolist(),dt_seconds=dt_values))
    private=[{k:{**v,'mean':v['mean'].tolist()} for k,v in f.items()} for f in models]
    return out,dict(eligible=eligible),dict(eligible_queries=int(eligible.sum()),native_likelihood_fits=fit_count,constant_limit_fits=constant,grid_boundary_fits=boundary,variance_floor_candidates=floors,unknown_selected_node_queries=unknown,models=private,decisions=decisions)
