"""Within-fragment historical forecasting, never cross-source risk calibration."""
from __future__ import annotations
import numpy as np
from association.evaluation.native_motion import _short_delta
from association.evaluation.native_temporal import slope,spherical_position

KINDS=('global','hold','local8','cv','uniform')

def _legal(x):
    x=np.asarray(x).copy();clipped=int((np.abs(x[...,0])>90).sum());x[...,0]=np.clip(x[...,0],-90,90);x[...,1]=(x[...,1]+180)%360-180
    return x,clipped

def fragment_forecasts(times,positions,query,original):
    t=np.asarray(times,np.float64);y=np.asarray(positions,np.float64).copy();q=np.asarray(query,np.float64)
    if len(t)<2 or y.shape!=(len(t),2) or (np.diff(t)<=0).any() or not np.isfinite(t).all() or not np.isfinite(y).all():raise ValueError('finite increasing native observations required')
    y[:,1]=y[0,1]+np.r_[0.,np.cumsum(_short_delta(np.diff(y[:,1])))];out={k:np.asarray(original,np.float64).copy() for k in KINDS};mass=np.zeros((len(q),3));unknown=far=clamps=0;fold_count=0;gap=[]
    scale=6371000.*np.pi/180*np.array([1.,np.cos(np.deg2rad(y[:,0].mean()))])
    for reverse in (False,True):
        u,z=(-t[::-1],y[::-1]) if reverse else (t,y)
        use=q<t[0] if reverse else q>t[-1];uq=-q if reverse else q
        full=np.stack((np.zeros(2),slope(u,z),slope(u[-min(8,len(u)):],z[-min(8,len(u)):])))
        cuts=sorted({min(len(u)-1,max(2,int(len(u)*f))) for f in (.25,.5,.75)}) if len(u)>=3 else []
        folds=[]
        for cut in cuts:
            prefix=u[:cut];p=z[:cut];vel=np.stack((np.zeros(2),slope(prefix,p),slope(prefix[-min(8,cut):],p[-min(8,cut):])))
            folds.append((cut,vel))
        fold_count+=len(folds)
        for j in np.flatnonzero(use):
            h=uq[j]-u[-1];expert,clip=_legal(z[-1]+h*full);clamps+=clip
            out['hold'][j]=expert[0];out['local8'][j]=expert[2]
            # Same all-native, endpoint-anchored global control as original127.
            if not np.allclose(expert[1],original[j],rtol=0,atol=1e-8):raise ValueError('original OLS endpoint binding differs')
            losses=[];matched=[]
            for cut,vel in folds:
                target=u[cut-1]+h;right=int(np.searchsorted(u,target));choices={min(len(u)-1,max(cut,right)),min(len(u)-1,max(cut,right-1))}
                k=min(choices,key=lambda k:(abs(u[k]-target),u[k]));dt=u[k]-u[cut-1]
                forecast=z[cut-1]+dt*vel
                losses.append(np.sum(((forecast-z[k])*scale)**2,axis=1));matched.append(abs(dt-h))
            if losses:
                risk=np.mean(losses,axis=0);w=(risk==risk.min()).astype(float);w/=w.sum();mass[j]=w
                gap.append(float(np.mean(matched)));far+=int(all(h>u[-1]-u[c-1] for c,_ in folds))
            else:mass[j,1]=1.;unknown+=1
            out['cv'][j]=spherical_position(expert,mass[j]);out['uniform'][j]=spherical_position(expert,np.ones(3))
    return out,dict(expert_mass=mass,no_validation_queries=unknown,horizon_exceeds_all_validation_windows=far,
        validation_folds=fold_count,native_ols_fits=4+2*fold_count,mean_absolute_horizon_mismatch_seconds=float(np.mean(gap)) if gap else None,latitude_saturations=clamps)

def native_forecasts(cache,original):
    original=np.asarray(original,np.float64);out={k:original[...,:2].copy() for k in KINDS};rows=[];offsets=cache['event_offsets']
    if len(offsets)!=len(original)+1:raise ValueError('event/node count')
    for i,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        values,summary=fragment_forecasts(cache['event_times_seconds'][a:b],cache['event_values'][a:b,:2],cache['query_seconds'],original[i,:,:2])
        for k in KINDS:out[k][i]=values[k]
        rows.append(summary)
    return out,rows
