"""Independent NumPy raw/decoded association, anchors/support and saved physics."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import audit_variable_group_outputs as parent
core=parent.core;load=parent.load


def binary(y,d):
    tp=int((y&d).sum());fp=int((~y&d).sum());fn=int((y&~d).sum());tn=int((~y&~d).sum())
    return dict(pair_count=len(y),positive_count=int(y.sum()),tp=tp,fp=fp,fn=fn,tn=tn,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
        pca=tp/(tp+fn) if tp+fn else None,pfa=fp/(tp+fp) if tp+fp else None)


def support_counts(membership,m):
    groups=sorted(set(membership));sources=np.unique(m['sources']);out=[]
    for g in groups:
        out.append(sum(np.any(m['support'][(membership==g)&(m['sources']==s)],axis=0) for s in sources))
    return np.asarray(out)


def scene(run,record,split,sid,row,saved,checks):
    q=load(run/'private'/f'partitions-{split}-{sid}.npz')
    m=load(Path(record['native_run'])/'private/native-motion'/split/f'scene-{sid}.npz')
    ev=load(Path(record['native_run'])/'private/native-evaluation'/split/f'scene-{sid}.npz')
    pairs=q['pairs'];n=len(m['states']);checks.array(pairs,np.column_stack(np.triu_indices(n,1)),'all complete pairs')
    truth=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]]
    cross=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]];items={}
    prefixes=list(dict.fromkeys(k.rsplit('_',1)[0] for k in record['methods']))
    for p in prefixes:
        checks.array(saved[p+'_MEAN']['node_group_indices'],q[p],'Mean partition')
        checks.array(saved[p+'_NN']['node_group_indices'],q[p],'NN partition')
        for kind in ('raw','decoded'):
            d=q[p+'_scores']>=1.7320103112387182 if kind=='raw' else q[p][pairs[:,0]]==q[p][pairs[:,1]]
            key=p+'_'+kind;items[key]={}
            for scope,mask in [('all',np.ones(len(pairs),bool)),('itsa',~cross),('mstsa',cross)]:
                item=binary(truth[mask],d[mask]);items[key][scope]=item
                public=row['association'][key] if scope=='all' else row['association'][key][scope]
                checks.fields(item,public['per_scene'][0],('pair_count','positive_count','tp','fp','fn','tn'),'association count')
                check_summary(public,[item],checks)
    if record['mode']=='support':
        weights=q['scores']-1.7320103112387182;w=np.zeros((n,n));w[pairs[:,0],pairs[:,1]]=weights;w+=w.T
        traces=core.read(run/'private'/f'moves-{split}-{sid}.json')
        oldanchors=np.array([np.flatnonzero(q['WS']==g)[0] for g in sorted(set(q['WS']))])
        for p in ('REF','ANCHOR','SUPPORT'):
            labels=q['WS'].copy();J=float(weights[labels[pairs[:,0]]==labels[pairs[:,1]]].sum());initial=J
            for move in traces[p]:
                node,old,new=[move[k] for k in ('node','from_label','to_label')]
                checks.exact(labels[node]==old,'move original membership');gain=float(w[node,labels==new].sum()-w[node,labels==old].sum())
                checks.exact(gain>1e-10,'positive actual J gain');checks.value(gain,move['gain'],'delta','move')
                before=support_counts(labels,m) if p=='SUPPORT' else None
                labels[node]=new;J+=gain;checks.value(J,move['J'],'delta','trajectory')
                if p!='REF':
                    anchors=np.array([np.flatnonzero(labels==g)[0] for g in sorted(set(labels))]);checks.array(anchors,oldanchors,'original anchor set')
                if p=='SUPPORT':checks.exact(np.all(support_counts(labels,m)>=before),'every query source count preserved')
            checks.array(labels[:,None]==labels[None,:],q[p][:,None]==q[p][None,:],'final relation')
            checks.value(initial,row['search'][p]['initial_J'],'delta','initial J');checks.value(J,row['search'][p]['final_J'],'delta','final J')
            checks.exact(len(traces[p])<=4*n,'bounded moves')
    return items


def check_summary(public,items,checks):
    active=[i for i in items if i['pair_count']]
    totals={k:sum(x[k] for x in items) for k in ('pair_count','tp','fp','fn','tn')}
    checks.fields(totals,public,tuple(totals),'pooled association')
    checks.value(float(np.mean([x['f1'] for x in active])) if active else None,public['macro_f1'],'macro_f1','association')
    denominator=2*totals['tp']+totals['fp']+totals['fn']
    checks.value(2*totals['tp']/denominator if denominator else 1. if active else None,public['micro_f1'],'micro_f1','association')
    for field,short in [('m_pca','pca'),('m_pfa','pfa')]:
        vals=[x[short] for x in active]
        checks.value(float(np.mean(vals)) if vals and all(x is not None for x in vals) else None,public['paper_metrics'][field],field,'association')


def summary(sub,indices,items,checks):
    for label in sub['association']:
        for scope,public in sub['association'][label].items():
            check_summary(public,[items[i][label][scope] for i in indices],checks)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent.audit(a.run.resolve(),a.output.resolve(),extra_scene=scene,extra_summary=summary)
