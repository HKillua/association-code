"""Matched old-information and native-information corrections, frozen178 base."""
from __future__ import annotations
import numpy as np
import torch
from association.evaluation.local_source_evidence import ResidualScore

def fit(x,y,w,old_score,device,*,steps=256,progress=None):
    torch.manual_seed(20);w=w/w.sum();mean=w@x;std=np.sqrt(w@((x-mean)**2));std[std<1e-6]=1.;z=np.clip((x-mean)/std,-8,8);inputs={'OLDINFO':z.copy(),'NATIVE':z};inputs['OLDINFO'][:,12:]=0
    inputs={k:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in inputs.items()};target=torch.as_tensor(y-old_score,dtype=torch.float32,device=device);mass=torch.as_tensor(w*len(w),dtype=torch.float32,device=device)
    a=ResidualScore(25).to(device);b=ResidualScore(25).to(device);b.load_state_dict(a.state_dict());nets={'OLDINFO':a,'NATIVE':b};opts={k:torch.optim.Adam(v.parameters(),lr=.001,weight_decay=1e-4) for k,v in nets.items()};draws=np.random.default_rng(20).integers(len(x),size=(steps,4096));curves={k:[] for k in nets}
    for step,draw in enumerate(draws):
        ix=torch.as_tensor(draw,device=device)
        for key,net in nets.items():
            opts[key].zero_grad(set_to_none=True);loss=(mass[ix]*(net(inputs[key][ix])-target[ix]).square()).mean()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite native correction')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opts[key].step();curves[key].append(dict(update=step+1,loss=float(loss.detach().cpu()),gradient_norm=float(gn)))
        if progress is not None:progress(step+1)
    models={k:dict(weights={name:v.detach().cpu() for name,v in net.state_dict().items()},mean=mean,std=std,dimension=25,parameter_count=1921,kind=k,updates=steps,curve=curves[k]) for k,net in nets.items()}
    return models,draws

@torch.no_grad()
def predict(model,x,device):
    z=np.clip((x-model['mean'])/model['std'],-8,8)
    if model['kind']=='OLDINFO':z[:,12:]=0
    net=ResidualScore(25).to(device);net.load_state_dict(model['weights']);net.eval();return net(torch.as_tensor(z,dtype=torch.float32,device=device)).cpu().numpy()
