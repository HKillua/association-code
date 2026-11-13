"""Frozen edge embeddings with symmetric, leave-edge-out graph context.

No identities, reference states or sensor names enter this interface.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class EdgeResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(192,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,x):
        return self.network(x).squeeze(-1)


def context_cache(hidden,pairs,scores,node_count,threshold):
    h=np.asarray(hidden,np.float32);p=np.asarray(pairs);s=np.asarray(scores,float);n=int(node_count)
    if (n<0 or h.shape!=(len(p),64) or p.shape!=(n*(n-1)//2,2)
            or p.dtype.kind not in 'iu' or s.shape!=(len(p),) or not np.isfinite(h).all()
            or not np.isfinite(s).all() or not np.isfinite(threshold)):
        raise ValueError('finite 64-dimensional edges on complete unordered graph required')
    if ((p<0).any() or (p>=n).any() or (p[:,0]>=p[:,1]).any()
            or len(np.unique(p,axis=0))!=len(p)):
        raise ValueError('complete unique unordered pairs required')
    w=np.exp(-np.logaddexp(0.,-(s-float(threshold))))
    sums=np.zeros((n,64));den=np.zeros(n)
    for endpoint in (0,1):
        np.add.at(sums,p[:,endpoint],w[:,None]*h)
        np.add.at(den,p[:,endpoint],w)
    return dict(hidden=h,pairs=p,weight=w,weighted_sum=sums,weight_sum=den)


def edge_inputs(cache,indices,kind):
    idx=np.asarray(indices);h=cache['hidden'][idx]
    if kind=='OWN':return np.concatenate((h,h,h),axis=1)
    if kind!='CTX':raise ValueError('OWN or CTX required')
    p=cache['pairs'][idx];w=cache['weight'][idx];values=[]
    for endpoint in (0,1):
        den=cache['weight_sum'][p[:,endpoint]]-w
        num=cache['weighted_sum'][p[:,endpoint]]-w[:,None]*h
        values.append(np.divide(num,den[:,None],out=np.zeros_like(num),where=den[:,None]>1e-12))
    a,b=values
    return np.concatenate((h,a+b,np.abs(a-b)),axis=1).astype(np.float32)


def fit_context_normalizer(caches,kind):
    count=0;mean=np.zeros(192);m2=np.zeros(192)
    for cache in caches:
        for start in range(0,len(cache['pairs']),4096):
            x=edge_inputs(cache,np.arange(start,min(start+4096,len(cache['pairs']))),kind).astype(float)
            size=len(x)
            if not size:continue
            local=x.mean(0);delta=local-mean;total=count+size
            m2+=np.square(x-local).sum(0)+delta*delta*(count*size/total)
            mean+=delta*(size/total);count=total
    if not count:raise ValueError('nonempty T-only edge caches required')
    std=np.sqrt(np.maximum(m2/count,0.))
    return dict(mean=mean,std=np.where(std<1e-6,1.,std),pair_count=count)


def normalize(x,normalizer):
    result=np.clip((x-normalizer['mean'])/normalizer['std'],-8.,8.).astype(np.float32)
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite edge residual input')
    return result
