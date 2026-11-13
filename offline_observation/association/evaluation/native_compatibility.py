"""Label-free same-source native kinematics; canonical input speed is explicit."""
from __future__ import annotations
import numpy as np
from association.evaluation.native_temporal import spherical_position
from association.evaluation.source_risk import _position_distance as distance

RADIUS=6371000.
NAMES=('log_native_position_distance_km','log_report_forecast_distance_km',
 'log_node_report_velocity_gap_mps','log_node_secant_velocity_gap_mps',
 'log_report_secant_gap_mps','log_native_age_minutes','log_native_span_minutes',
 'report_known','native_bracket','log_bracket_gap_minutes','log_bracket_report_gap_mps',
 'log_report_speed_mps','bracket_report_known')

def displacement(a,b):
    a,b=np.asarray(a),np.asarray(b);d=b-a;lon=(d[...,1]+180)%360-180
    return RADIUS*np.pi/180*np.stack([d[...,0],lon*np.cos(np.deg2rad((a[...,0]+b[...,0])/2))],axis=-1)

def advance(position,velocity,dt):
    """Great-circle constant tangent velocity, including signed backward time."""
    p,v=np.asarray(position),np.asarray(velocity);dt=np.asarray(dt);d=np.linalg.norm(v,axis=-1)*dt/RADIUS;bearing=np.arctan2(v[...,1],v[...,0]);lat,lon=np.deg2rad(p[...,0]),np.deg2rad(p[...,1])
    phi=np.arcsin(np.clip(np.sin(lat)*np.cos(d)+np.cos(lat)*np.sin(d)*np.cos(bearing),-1,1));lam=lon+np.arctan2(np.sin(bearing)*np.sin(d)*np.cos(lat),np.cos(d)-np.sin(lat)*np.sin(phi))
    return np.stack([np.rad2deg(phi),(np.rad2deg(lam)+180)%360-180],axis=-1)

def stream(events,nodes,*,speed_unit):
    if speed_unit not in ('kn','mps'):raise ValueError('explicit canonical speed unit required')
    idx=np.concatenate([np.arange(*events['event_offsets'][i:i+2]) for i in nodes]);idx=idx[events['event_valid'][idx,:2].all(1)]
    t=events['event_times_seconds'][idx];v=events['event_values'][idx];valid=events['event_valid'][idx];tt,inverse=np.unique(t,return_inverse=True);n=len(tt)
    if n<2:raise ValueError('native same-source stream needs >=2 times')
    counts=np.bincount(inverse);sine=np.bincount(inverse,weights=np.sin(np.deg2rad(v[:,1])));cosine=np.bincount(inverse,weights=np.cos(np.deg2rad(v[:,1])))
    pos=np.column_stack([np.bincount(inverse,weights=v[:,0])/counts,(np.rad2deg(np.arctan2(sine,cosine))+180)%360-180]);vel=np.zeros((n,2))
    # Coalescing distinct event rows is not identity deduplication.
    for k in np.flatnonzero(np.hypot(sine,cosine)<=1e-12*counts):
        pos[k]=spherical_position(v[inverse==k,:2],np.ones(int(counts[k])))
    good=valid[:,2:4].all(1);good_count=np.bincount(inverse[good],minlength=n);known=good_count>0
    speed=v[good,2]*(1852/3600 if speed_unit=='kn' else 1.);theta=np.deg2rad(v[good,3])
    for axis,component in enumerate((np.cos(theta),np.sin(theta))):vel[:,axis]=np.bincount(inverse[good],weights=speed*component,minlength=n)/np.maximum(good_count,1)
    return dict(t=tt,p=pos,v=vel,known=known)

def at_queries(s,query):
    t,p,v,known=(s[k] for k in ('t','p','v','known'));q=np.asarray(query);left=np.searchsorted(t,q,side='left')-1;right=np.searchsorted(t,q,side='right');bracket=(left>=0)&(right<len(t));nearest=np.minimum(np.searchsorted(t,q),len(t)-1);other=np.maximum(nearest-1,0)
    choose=np.abs(t[other]-q)<np.abs(t[nearest]-q);tie=(other!=nearest)&(np.abs(t[other]-q)==np.abs(t[nearest]-q));nearest=np.where(choose,other,nearest)
    anchor=p[nearest].copy();report=v[nearest].copy();report_known=known[nearest].copy();reported=advance(anchor,report,q-t[nearest])
    for j in np.flatnonzero(tie):
        ii=[other[j],nearest[j]];anchor[j]=spherical_position(p[ii],np.ones(2));report_known[j]=bool(known[ii].all());report[j]=v[ii].mean(0);reported[j]=spherical_position(advance(p[ii],v[ii],q[j]-t[ii]),np.ones(2))
    # At an exact event use an adjacent native interval, never zero dt.
    l=np.clip(left,0,len(t)-2);r=np.clip(right,1,len(t)-1);r=np.where(r<=l,l+1,r)
    secant=displacement(p[l],p[r])/(t[r]-t[l])[:,None];pairknown=known[l]&known[r];gap=np.where(bracket,t[r]-t[l],0.)
    mismatch=np.where(pairknown,np.linalg.norm(secant-(v[l]+v[r])/2,axis=1),0.)
    return dict(anchor=anchor,report_velocity=report,report_known=report_known,reported=reported,secant_velocity=secant,
        report_secant_gap=np.where(report_known,np.linalg.norm(secant-report,axis=1),0.),age=np.abs(q-t[nearest]),span=t[-1]-t[0],bracket=bracket,gap=gap,bracket_report_gap=np.where(bracket,mismatch,0.),bracket_report_known=bracket&pairknown,
        before=q<t[0],after=q>t[-1])

def descriptors(m,events,groups,*,speed_unit):
    n,q=m['support'].shape;x=np.zeros((n,q,len(NAMES)));details={}
    for gid,nodes in enumerate(groups):
        nodes=np.asarray(nodes,int);items=[]
        for source in set(m['sources'][nodes].tolist()):
            own=nodes[m['sources'][nodes]==source];s=stream(events,own,speed_unit=speed_unit);a=at_queries(s,m['query_seconds']);items.append((own,a))
            for node in own:
                reportgap=np.linalg.norm(m['ols_velocity_mps'][node]-a['report_velocity'],axis=1);secgap=np.linalg.norm(m['ols_velocity_mps'][node]-a['secant_velocity'],axis=1)
                columns=[distance(m['states'][node],a['anchor'])/1000,np.where(a['report_known'],distance(m['states'][node],a['reported'])/1000,0),np.where(a['report_known'],reportgap,0),secgap,a['report_secant_gap'],a['age']/60,np.full(q,a['span']/60),a['report_known'],a['bracket'],a['gap']/60,a['bracket_report_gap'],np.linalg.norm(a['report_velocity'],axis=1),a['bracket_report_known']]
                x[node]=np.column_stack([v if i in (7,8,12) else np.log1p(v) for i,v in enumerate(columns)])
        details[gid]=items
    if not np.isfinite(x).all():raise FloatingPointError('native context must be finite')
    return x,details
