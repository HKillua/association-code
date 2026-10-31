"""Source-equivariant native-time bracket controls, only on zero-support queries."""
from __future__ import annotations
import numpy as np
from association.evaluation.native_temporal import spherical_position

def controls(m,events,groups,count,scores,base):
    result={k:np.asarray(base).copy() for k in ('R','N','B','E')};eligible=np.zeros(count.shape,bool);source_count=np.zeros(count.shape,int);gaps=[];decisions=[]
    for gid,group in enumerate(groups):
        nodes=np.asarray(group,int);streams={}
        for source in dict.fromkeys(m['sources'][nodes].tolist()):
            own=nodes[m['sources'][nodes]==source];times=[];values=[]
            for node in own:
                a,b=events['event_offsets'][node:node+2];use=events['event_valid'][a:b,:2].all(1)
                times.extend(events['event_times_seconds'][a:b][use]);values.extend(events['event_values'][a:b,:2][use])
            streams[source]=(own,np.asarray(times),np.asarray(values))
        for j in np.flatnonzero(count[gid]==0):
            now=m['query_seconds'][j];candidates=[]
            for source,(own,t,v) in streams.items():
                left=t[t<now];right=t[t>now]
                if not len(left) or not len(right):continue
                tl,tr=left.max(),right.min();a=(now-tl)/(tr-tl);pl=spherical_position(v[t==tl],np.ones(int((t==tl).sum())));pr=spherical_position(v[t==tr],np.ones(int((t==tr).sum())))
                endpoints=np.stack([pl,pr]);bridge=spherical_position(endpoints,np.array([1-a,a]));nearest=spherical_position(endpoints,np.array([1.,0.]) if now-tl<tr-now else np.array([0.,1.]) if now-tl>tr-now else np.ones(2))
                best=own[scores[own,j]==scores[own,j].min()];old=spherical_position(m['states'][best,j,:2],np.ones(len(best)))
                candidates.append(dict(source=source,score=float(scores[best[0],j]),R=old,N=nearest,B=bridge,gap=float(tr-tl),alpha=float(a)))
            if not candidates:continue
            eligible[gid,j]=True;source_count[gid,j]=len(candidates);best=min(c['score'] for c in candidates);selected=[c for c in candidates if c['score']==best]
            for k in ('R','N','B'):result[k][gid,j,:2]=spherical_position(np.stack([c[k] for c in selected]),np.ones(len(selected)))
            result['E'][gid,j,:2]=spherical_position(np.stack([c['B'] for c in candidates]),np.ones(len(candidates)))
            gaps.extend(c['gap'] for c in selected);decisions.append(dict(group_index=gid,query_index=int(j),eligible_sources=len(candidates),selected_sources=len(selected),mean_selected_gap_seconds=float(np.mean([c['gap'] for c in selected])),mean_selected_alpha=float(np.mean([c['alpha'] for c in selected]))))
    return result,dict(eligible=eligible,eligible_source_count=source_count),dict(eligible_group_queries=int(eligible.sum()),selected_source_decisions=len(gaps),selected_gap_seconds=gaps,decisions=decisions)
