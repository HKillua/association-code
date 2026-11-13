"""176 T-only grid selection; all policies share candidates and error budgets."""
from __future__ import annotations
import numpy as np


def coverage(membership, identities, target_ids, target_valid):
    anchors=np.unique(membership,return_index=True)[1]
    represented={str(identities[i]) for i in anchors}
    visible=np.asarray([str(t) in represented for t in target_ids])
    valid=np.asarray(target_valid)[...,:2].all(-1)
    return dict(points=int(valid[visible].sum()),targets=int((visible&valid.any(1)).sum()))


def binary(labels,predicted):
    positive=int(labels.sum());accepted=int(predicted.sum());tp=int((labels&predicted).sum())
    return dict(tp=tp,fp=accepted-tp,fn=positive-tp,positive=positive,accepted=accepted,
                pca=tp/positive if positive else None,pfa=1-tp/accepted if accepted else None)


def aggregate(rows):
    result={}
    for kind in ('raw','decoded'):
        result[kind]={field:float(np.mean([r[kind][field] for r in rows]))
            if all(r[kind][field] is not None for r in rows) else None for field in ('pca','pfa')}
    result['coverage']={k:sum(r['coverage'][k] for r in rows) for k in ('points','targets')}
    return result


def choose(curve,limits,floor,policy):
    if policy not in ('R','D','C'):raise ValueError('explicit R/D/C required')
    eligible=[]
    for row in curve:
        m=row['metrics']
        if any(m[k][f] is None for k in ('raw','decoded') for f in ('pca','pfa')):continue
        if any(m[k]['pfa']>limits[k]+1e-12 for k in ('raw','decoded')):continue
        if policy=='C' and any(m['coverage'][k]<floor[k] for k in ('points','targets')):continue
        eligible.append(row)
    return max(eligible,key=lambda r:(r['metrics']['raw' if policy=='R' else 'decoded']['pca'],
                                    -r['metrics']['decoded']['pfa'],r['threshold'])) if eligible else None
