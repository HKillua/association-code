"""Separate endpoint and velocity interventions on the same native short window."""
from __future__ import annotations
import numpy as np
from association.evaluation.native_temporal import slope
from association.evaluation.native_motion import _short_delta

def endpoint_line(t,y):
    """Coordinatewise pair-slope and residual medians; not Euclidean r1 regression."""
    t=np.asarray(t,float);y=np.asarray(y,float)
    if len(t)<2 or y.shape!=(len(t),2) or (np.diff(t)<=0).any() or not np.isfinite(y).all():raise ValueError('finite increasing native line')
    a,b=np.triu_indices(len(t),1);v=np.median((y[b]-y[a])/(t[b]-t[a])[:,None],axis=0)
    p=np.median(y-(t-t[-1])[:,None]*v,axis=0)
    return p,v

def native_endpoint_states(cache,original):
    q=cache['query_seconds'];output={k:original[...,:2].copy() for k in ('endpoint','velocity','joint')};records=[];clamps=0;poles=0;n2=0
    for i,(a,b) in enumerate(zip(cache['event_offsets'][:-1],cache['event_offsets'][1:])):
        t=np.asarray(cache['event_times_seconds'][a:b],float);y=cache['event_values'][a:b,:2].astype(float).copy()
        if len(t)<2 or (np.diff(t)<=0).any() or not np.isfinite(y).all():raise ValueError('native events')
        y[:,1]=y[0,1]+np.r_[0.,np.cumsum(_short_delta(np.diff(y[:,1])))];scale=6371000*np.pi/180*np.array([1.,np.cos(np.deg2rad(y[:,0].mean()))])
        if abs(scale[1])<1e-9:poles+=1;continue
        for reverse in (False,True):
            u,z=(-t[::-1],y[::-1]) if reverse else (t,y);use=q<t[0] if reverse else q>t[-1];uq=-q if reverse else q
            if len(t)==2:n2+=1;continue
            w=min(8,len(t));origin=z[-1];local=(z[-w:]-origin)*scale;endpoint,v=endpoint_line(u[-w:],local);globalv=slope(u,z)*scale
            records.append(dict(node=i,direction=int(reverse),window=w,endpoint_change_m=float(np.linalg.norm(endpoint)),velocity_change_mps=float(np.linalg.norm(v-globalv)),query_count=int(use.sum())))
            for kind,p,vel in [('endpoint',endpoint,globalv),('velocity',np.zeros(2),v),('joint',endpoint,v)]:
                pos=origin+(p+(uq[use]-u[-1])[:,None]*vel)/scale;clamps+=int((np.abs(pos[:,0])>90).sum());pos[:,0]=np.clip(pos[:,0],-90,90);pos[:,1]=(pos[:,1]+180)%360-180;output[kind][i,use]=pos
    return output,dict(direction_fits=len(records),pair_slope_counts=sum(r['window']*(r['window']-1)//2 for r in records),n2_direction_global_fallbacks=n2,pole_node_global_fallbacks=poles,latitude_saturations=clamps),records
