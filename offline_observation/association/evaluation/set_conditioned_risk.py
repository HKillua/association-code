"""Own-only versus other-source quality context, on a frozen residual base.

All-source permutation equivariance is an interface property, not evidence of
physical reliability or external K-source generalization.
"""
from __future__ import annotations
import numpy as np
import torch
from association.evaluation.local_source_evidence import ResidualScore, predict_residual


def conditional_features(pack, base, *, context):
    x = np.asarray(pack['features'],float)
    available = np.asarray(pack['available'],bool)
    z = np.clip((x-base['mean'])/base['std'],-8.,8.)
    second = z.copy()
    if context:
        second = np.zeros_like(z)
        for a,b in zip(pack['group_offsets'][:-1],pack['group_offsets'][1:]):
            total = (z[a:b]*available[a:b,:,None]).sum(0)
            count = available[a:b].sum(0)
            mass = count[None,:]-available[a:b]
            second[a:b] = (total[None,:,:]-z[a:b]*available[a:b,:,None])/np.maximum(mass,1)[...,None]
            second[a:b][mass == 0] = 0.
    out = np.concatenate((z,second),axis=-1)
    if not np.isfinite(out).all(): raise FloatingPointError('nonfinite conditional features')
    return out


def fit_heads(x, y, weights, frozen_prediction, *, device, steps=128):
    """The two arrays must present the same source-label rows in the same order."""
    torch.manual_seed(20)
    names = ('OWN','CONTEXT')
    net0 = ResidualScore(20); initial_weights = net0.state_dict()
    nets = {n:ResidualScore(20).to(device) for n in names}
    for net in nets.values(): net.load_state_dict(initial_weights)
    optimizers = {n:torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4) for n,net in nets.items()}
    data = {n:torch.as_tensor(x[n],dtype=torch.float32,device=device) for n in names}
    target = torch.as_tensor(y-frozen_prediction,dtype=torch.float32,device=device)
    weight = torch.as_tensor(weights/weights.sum()*len(weights),dtype=torch.float32,device=device)
    def whole(n):
        with torch.no_grad():return float((weight*(nets[n](data[n])-target)**2).mean())
    initial = {n:whole(n) for n in names};curves = {n:[] for n in names};rng=np.random.default_rng(20)
    for step in range(steps):
        ix = torch.as_tensor(rng.integers(len(y),size=4096),device=device)
        for n in names:
            optimizers[n].zero_grad(set_to_none=True)
            loss = (weight[ix]*(nets[n](data[n][ix])-target[ix])**2).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite conditional residual loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(nets[n].parameters(),1.,error_if_nonfinite=True);optimizers[n].step()
            curves[n].append(dict(update=step+1,loss=float(loss.detach()),gradient_norm=float(gn)))
    return {n:dict(weights={k:v.detach().cpu() for k,v in nets[n].state_dict().items()},parameter_count=1761,
        updates=steps,initial_mse=initial[n],final_mse=whole(n),curve=curves[n]) for n in names}


@torch.no_grad()
def predict(model, pack, *, device, context):
    x = conditional_features(pack,model['base'],context=context)
    net = ResidualScore(20).to(device);net.load_state_dict(model['head']['weights']);net.eval()
    old = predict_residual(model['base'],pack['features'].reshape(-1,10),device).reshape(pack['available'].shape)
    result = old+net(torch.as_tensor(x.reshape(-1,20),dtype=torch.float32,device=device)).cpu().numpy().reshape(old.shape)
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite set conditional score')
    return result
