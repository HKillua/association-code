"""Visible native velocity measures; these descriptors are not calibrated risk."""
from __future__ import annotations
import numpy as np

FIELDS=('native_count','valid_report_count','span_seconds','valid_span_fraction',
        'interval_cv','interval_max_to_median','weight_TV','time_weight_neff',
        'count_mean_speed_mps','time_mean_speed_mps','mean_measure_gap_mps',
        'report_spread_mps','iid_mean_se_working_mps')

def velocity_measures(times,values,valid):
    t=np.asarray(times,float);values=np.asarray(values);valid=np.asarray(valid)
    if t.ndim!=1 or len(t)<2 or values.shape!=(len(t),4) or valid.shape!=values.shape or valid.dtype!=np.bool_ or not np.isfinite(t).all() or (np.diff(t)<=0).any():raise ValueError('strict native times and [event,4] values/validity')
    dt=np.diff(t);good=valid[:,2:4].all(1);tt=t[good];x=values[good,2:4]
    if not np.isfinite(x).all():raise ValueError('valid report must be finite')
    n=len(tt);known=n>=2;mu=np.zeros(2);time_mu=mu.copy();w=np.empty(0);spread=0.
    if known:
        speed=x[:,0]*1852/3600;theta=np.deg2rad(x[:,1]);v=speed[:,None]*np.column_stack((np.cos(theta),np.sin(theta)));mu=v.mean(0);delta=np.diff(tt);w=np.r_[delta[0],delta[:-1]+delta[1:],delta[-1]]/(2*(tt[-1]-tt[0]));time_mu=w@v;spread=float(np.sqrt(np.mean(np.sum((v-mu)**2,axis=1))))
        # Independent interval-integral expression binds the node-weight formula.
        integral=np.sum((v[1:]+v[:-1])*.5*delta[:,None],axis=0)/(tt[-1]-tt[0]);np.testing.assert_allclose(time_mu,integral,rtol=1e-12,atol=1e-10);assert (w>0).all() and np.isclose(w.sum(),1.)
    stats=dict(native_count=len(t),valid_report_count=n,span_seconds=float(t[-1]-t[0]),valid_span_fraction=float((tt[-1]-tt[0])/(t[-1]-t[0])) if known else 0.,interval_cv=float(dt.std()/dt.mean()),interval_max_to_median=float(dt.max()/np.median(dt)),weight_TV=float(np.abs(w-1/n).sum()/2) if known else 0.,time_weight_neff=float(1/(w@w)) if known else 0.,count_mean_speed_mps=float(np.linalg.norm(mu)),time_mean_speed_mps=float(np.linalg.norm(time_mu)),mean_measure_gap_mps=float(np.linalg.norm(time_mu-mu)),report_spread_mps=spread,iid_mean_se_working_mps=spread/np.sqrt(n) if known else 0.)
    return dict(known=known,count_mean=mu,time_mean=time_mu,weights=w,stats=stats)
