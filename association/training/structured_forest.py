"""Partial FY value-gap on complete upper-edge graphs; labels are T-only.

Known k is required by this training oracle. This is not a deployment decoder.
The full-label constraint reduces exactly to one spanning tree per class.
"""
from __future__ import annotations
import numpy as np
import torch


def maximum_forest(scores, pairs, node_count, k, labels=None):
    s=np.asarray(scores,dtype=float);p=np.asarray(pairs);n=int(node_count);k=int(k)
    if not (1<=k<=n) or s.shape!=(n*(n-1)//2,) or p.shape!=(len(s),2) or p.dtype.kind not in 'iu':
        raise ValueError('complete upper graph and 1<=k<=n required')
    if not np.isfinite(s).all() or (p<0).any() or (p>=n).any() or (p[:,0]>=p[:,1]).any() or len(np.unique(p,axis=0))!=len(p):
        raise ValueError('finite scores and unique complete upper edges required')
    if labels is not None:
        labels=np.asarray(labels)
        if labels.shape!=(n,) or len(np.unique(labels))!=k:raise ValueError('full labels with exactly k classes required')
    selected=np.zeros(len(s),bool);parent=np.arange(n);rank=n-k
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    count=0
    for e in np.argsort(-s,kind='stable'):
        if count==rank:break
        i,j=p[e]
        if labels is not None and labels[i]!=labels[j]:continue
        a,b=root(i),root(j)
        if a!=b:parent[b]=a;selected[e]=True;count+=1
    if count!=rank:raise ValueError('forest rank unattainable')
    return selected


def forest_contrast(scores,pairs,labels,noise):
    """Return mean edge contrast and mean noise offset for a fixed MC draw."""
    labels=np.asarray(labels);n=len(labels);k=len(np.unique(labels));s=np.asarray(scores,float)
    z=np.asarray(noise,float)
    if z.ndim!=2 or z.shape[1:]!=s.shape or not len(z) or not np.isfinite(z).all():raise ValueError('finite nonempty B x E noise required')
    differences=[];offsets=[]
    for perturbation in z:
        t=s+perturbation
        a=maximum_forest(t,pairs,n,k);b=maximum_forest(t,pairs,n,k,labels)
        d=a.astype(float)-b.astype(float)
        if np.dot(d,t)<-1e-8:raise ArithmeticError('negative optimal value gap')
        differences.append(d);offsets.append(float(np.dot(d,perturbation)))
    scale=max(n-k,1)
    return np.mean(differences,axis=0)/scale,float(np.mean(offsets))/scale


def partial_fy_loss(scores,pairs,labels,noise):
    """Autograd of optimal value gap, using envelope gradient, not closure JVP."""
    coeff,offset=forest_contrast(scores.detach().cpu().numpy(),pairs,labels,noise)
    return (scores*torch.as_tensor(coeff,device=scores.device,dtype=scores.dtype)).sum()+offset
