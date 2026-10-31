"""Matched continued log-risk regression and variable-action cost supervision."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch.nn import functional as F
from association.evaluation.local_source_evidence import ResidualScore
from association.evaluation.native_temporal import predict

KINDS=('MSE','COST','TAIL')

def objective(kind,risk,y,mask,mass):
    if kind not in KINDS or mask.ndim!=2 or not torch.all(mask.sum(1)>=2):raise ValueError('valid decision bags with >=2 actions required')
    if kind=='MSE':return ((risk-y).square()*mass).sum(1).mean()
    errors=torch.expm1(y).masked_fill(~mask,torch.inf)
    regret=(errors-errors.amin(1,keepdim=True)).masked_fill(~mask,0).clamp_min(0)
    costs=regret if kind=='COST' else regret+F.relu(regret-1.)
    n=mask.sum(1,keepdim=True);tau=(costs.sum(1,keepdim=True)-costs).masked_fill(~mask,0)/(n*(n-1))
    logp=F.log_softmax((-risk).masked_fill(~mask,-torch.inf),dim=1)
    return -(tau*torch.where(mask,logp,torch.zeros_like(logp))).sum(1).mean()

def fit_heads(bags,initial,device,*,steps=256,check=None):
    torch.manual_seed(20);offsets=np.r_[0,np.cumsum([len(b['y']) for b in bags])]
    x=np.concatenate([b['x'] for b in bags]);target=np.concatenate([b['y'] for b in bags]);w=np.concatenate([b['w'] for b in bags]);maxn=max(len(b['y']) for b in bags)
    indices=np.zeros((len(bags),maxn),np.int64);mask=np.zeros(indices.shape,bool)
    for i,(left,right) in enumerate(zip(offsets[:-1],offsets[1:])):indices[i,:right-left]=np.arange(left,right);mask[i,:right-left]=True
    z=torch.as_tensor(np.clip((x-initial['mean'])/initial['std'],-8,8),dtype=torch.float32,device=device)
    y=torch.as_tensor(target,dtype=torch.float32,device=device);mass=torch.as_tensor(w,dtype=torch.float32,device=device)
    base=torch.as_tensor(predict(initial['base'],x),dtype=torch.float32,device=device);ix=torch.as_tensor(indices,device=device);valid=torch.as_tensor(mask,device=device)
    heads={k:ResidualScore(12).to(device) for k in KINDS}
    for net in heads.values():net.load_state_dict(initial['weights'])
    opts={k:torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4) for k,net in heads.items()}
    draws=np.random.default_rng(20).integers(len(bags),size=(steps,128));curves={k:[] for k in KINDS}
    for step,draw in enumerate(draws):
        batch=torch.as_tensor(draw,device=device);padded=ix[batch];ok=valid[batch];sampley=y[padded];samplew=mass[padded]*ok
        for kind,net in heads.items():
            opts[kind].zero_grad(set_to_none=True);risk=base[padded]+net(z[padded]);loss=objective(kind,risk,sampley,ok,samplew)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite cost-routing loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opts[kind].step()
            curves[kind].append(dict(update=step+1,loss=float(loss.detach().cpu()),unclipped_gradient_norm=float(gn)))
        if check is not None:check(step+1,{k:curves[k][-1] for k in KINDS})
    models={}
    for kind,net in heads.items():
        model={k:copy.deepcopy(initial[k]) for k in ('mean','std','base','dimension','parameter_count')}
        model.update(weights={k:v.detach().cpu() for k,v in net.state_dict().items()},kind=kind,updates=steps,curve=curves[kind],score_semantics='log_error' if kind=='MSE' else 'cost_routing_score_not_calibrated_error')
        models[kind]=model
    return models,draws
