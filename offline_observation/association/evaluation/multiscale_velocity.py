"""Same-source current velocity consensus. No source quality or truth input."""
from __future__ import annotations
import numpy as np
from association.evaluation.native_temporal import slope
from association.evaluation.native_motion import _short_delta

def geometric_median(points,max_iterations=500):
    x=np.asarray(points,np.float64)
    if x.ndim!=2 or x.shape[1]!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('finite velocity vectors [W,2]')
    z=x.mean(0)
    for i in range(max_iterations):
        delta=x-z;d=np.linalg.norm(delta,axis=1);same=d<=1e-12;other=~same
        if not other.any():return z,True,i+1
        inverse=1/d[other];proposal=np.sum(x[other]*inverse[:,None],axis=0)/inverse.sum();r=np.linalg.norm(np.sum(delta[other]*inverse[:,None],axis=0));eta=same.sum()/r if r>0 else 1.
        if same.any() and r<=same.sum():return z,True,i+1
        nxt=max(0.,1-eta)*proposal+min(1.,eta)*z
        if np.linalg.norm(nxt-z)<=1e-10*(1+np.linalg.norm(z)):return nxt,True,i+1
        z=nxt
    return z,False,max_iterations

def native_multiscale(cache,original):
    original=np.asarray(original,np.float64);q=cache['query_seconds'];median=original[...,:2].copy();mean=median.copy();n=len(original)
    candidates=np.full((n,2,5,2),np.nan);lengths=np.zeros((n,5),int);aggregate=np.full((n,2,2,2),np.nan);fits=iterations=failed=poles=clamps=0
    for node,(a,b) in enumerate(zip(cache['event_offsets'][:-1],cache['event_offsets'][1:])):
        t=np.asarray(cache['event_times_seconds'][a:b],np.float64);y=np.asarray(cache['event_values'][a:b,:2],np.float64).copy()
        if len(t)<2 or (np.diff(t)<=0).any() or not np.isfinite(y).all():raise ValueError('increasing finite native events')
        y[:,1]=y[0,1]+np.r_[0.,np.cumsum(_short_delta(np.diff(y[:,1])))];windows=sorted({min(w,len(t)) for w in (8,16,32,64,len(t))});lengths[node,:len(windows)]=windows
        chart=6371000.*np.pi/180*np.array([1.,np.cos(np.deg2rad(y[:,0].mean()))])
        if abs(chart[1])<1e-9:poles+=1;continue
        for direction in (0,1):
            u,z=(-t[::-1],y[::-1]) if direction==0 else (t,y);uq=-q if direction==0 else q;use=q<t[0] if direction==0 else q>t[-1]
            velocities=np.stack([slope(u[-w:],z[-w:])*chart for w in windows]);fits+=len(windows);candidates[node,direction,:len(windows)]=velocities
            med,ok,count=geometric_median(velocities);iterations+=count;avg=velocities.mean(0);aggregate[node,direction]=[med,avg]
            if not ok:failed+=1
            # A single scale is exactly the old global control, including rounding.
            if len(windows)==1:continue
            for destination,velocity,enabled in [(median,med,ok),(mean,avg,True)]:
                if not enabled:continue
                positions=z[-1]+(uq[use]-u[-1])[:,None]*velocity/chart
                clamps+=int((np.abs(positions[:,0])>90).sum());positions[:,0]=np.clip(positions[:,0],-90,90);positions[:,1]=(positions[:,1]+180)%360-180;destination[node,use]=positions
    return dict(medianv=median,meanv=mean),dict(window_lengths=lengths,velocity_candidates_ne=candidates,aggregate_velocity_ne=aggregate),dict(native_ols_fits=fits,numerical_iterations=iterations,nonconverged_direction_global_fallbacks=failed,pole_chart_global_fallbacks=poles,latitude_saturations=clamps)
