"""Identity-free node descriptors and explicit history-candidate action sets."""
from __future__ import annotations
import numpy as np
from association.evaluation.source_risk import _position_distance
from association.evaluation.native_temporal import spherical_position

FEATURE_NAMES=('log_age_minutes','log_fit_rmse_km','log_native_count','log_span_minutes','observed_support',
 'log_age_span_ratio','log_group_distance_km','log_source_history_distance_km','log_group_velocity_difference_mps',
 'group_heading_disagreement','source_support_fraction','log_group_source_count')


def source_balanced(members,sources):
    weights=np.zeros(len(members),float)
    classes=list(dict.fromkeys(sources[members].tolist()))
    for source in classes:
        use=sources[members]==source;weights[use]=1./len(classes)/int(use.sum())
    return weights


def descriptors(motion,groups):
    states=np.asarray(motion['states'],float);support=motion['support'];sources=motion['sources'];nearest=motion['nearest_seconds'];n,q=support.shape
    x=np.zeros((n,q,len(FEATURE_NAMES)));count=np.zeros((len(groups),q),int)
    for gid,group in enumerate(groups):
        members=np.asarray(group,int);classes=list(dict.fromkeys(sources[members].tolist()))
        same_centers={s:np.array([spherical_position(states[members[sources[members]==s],j,:2],np.ones(int((sources[members]==s).sum()))) for j in range(q)]) for s in classes}
        for j in range(q):
            visible=members[support[members,j]];count[gid,j]=len(set(sources[visible].tolist()))
            context_members=visible if len(visible) else members;mass=source_balanced(context_members,sources)
            center=spherical_position(states[context_members,j,:2],mass);velocity=mass@motion['ols_velocity_mps'][context_members]
            radians=np.deg2rad(states[context_members,j,3]);sine=mass@np.sin(radians);cosine=mass@np.cos(radians)
            if np.hypot(sine,cosine)<=1e-12:
                order=np.lexsort((states[context_members,j,3],states[context_members,j,1],states[context_members,j,0]));heading=states[context_members[order[0]],j,3]
            else:heading=np.rad2deg(np.arctan2(sine,cosine))
            for node in members:
                own=members[sources[members]==sources[node]];age=nearest[node,j];span=motion['span_seconds'][node]
                x[node,j]=[np.log1p(age/60),np.log1p(motion['fit_rmse_m'][node]/1000),np.log1p(motion['native_count'][node]),np.log1p(span/60),float(support[node,j]),
                    np.log1p(age/max(span,1.)),np.log1p(_position_distance(states[node,j],center)/1000),np.log1p(_position_distance(states[node,j],same_centers[sources[node]][j])/1000),
                    np.log1p(np.linalg.norm(motion['ols_velocity_mps'][node]-velocity)),1-np.cos(np.deg2rad(states[node,j,3]-heading)),float(support[own,j].mean()),np.log1p(len(classes))]
    if not np.isfinite(x).all():raise FloatingPointError('nonfinite history features')
    return x,count


def labels(motion,groups,count,ev):
    shape=motion['support'].shape;y=np.full(shape,np.nan);w=np.zeros(shape);used=np.zeros(shape,bool);registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set();duplicate=unknown=queries=0
    for gid,group in enumerate(groups):
        identity=str(ev['identities'][group[0]])
        if identity not in registry:unknown+=1;continue
        if identity in claimed:duplicate+=1;continue
        claimed.add(identity);row=registry[identity];valid=ev['target_valid'][row,:,:2].all(-1)&(count[gid]<=1);members=np.asarray(group,int);mass=source_balanced(members,motion['sources'])
        if valid.any():
            error=_position_distance(motion['states'][members][:,valid],ev['target_states'][row,valid]);y[np.ix_(members,np.flatnonzero(valid))]=np.log1p(error/1000.)
            w[np.ix_(members,np.flatnonzero(valid))]=mass[:,None];used[np.ix_(members,np.flatnonzero(valid))]=True;queries+=int(valid.sum())
    if not np.isfinite(y[used]).all() or (w[used]<=0).any():raise FloatingPointError('invalid history labels')
    return dict(y=y,w=w,labeled=used,audit=dict(labeled_node_queries=int(used.sum()),group_queries=queries,unknown_groups=unknown,duplicate_groups_without_labels=duplicate,total_target_count=len(registry),claimed_targets=len(claimed),total_node_queries=int(np.prod(shape)),total_group_queries=int(count.size),low_support_group_queries=int((count<=1).sum()),multi_support_group_queries=int((count>1).sum())))


def readouts(motion,groups,count,scores,base):
    if scores.shape!=motion['support'].shape or not np.isfinite(scores).all():raise ValueError('finite node query scores required')
    names=('M','E','Z','1','O');outputs={k:np.asarray(base).copy() for k in names};weights={k:np.zeros(scores.shape) for k in names}
    states=motion['states'];sources=motion['sources'];support=motion['support']
    for gid,group in enumerate(groups):
        members=np.asarray(group,int)
        for j in np.flatnonzero(count[gid]<=1):
            for key in names:
                if (key=='Z' and count[gid,j]!=0) or (key=='1' and count[gid,j]!=1):continue
                eligible=members[support[members,j]] if key=='M' else members
                if not len(eligible):continue
                selected=eligible if key=='E' else eligible[scores[eligible,j]==scores[eligible,j].min()]
                mass=source_balanced(selected,sources);weights[key][selected,j]=mass
                outputs[key][gid,j,:2]=spherical_position(states[selected,j,:2],mass)
    return outputs,weights
