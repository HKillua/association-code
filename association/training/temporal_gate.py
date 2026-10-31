"""Matched phase-information gates trained on the actual circular position loss."""
from __future__ import annotations
import numpy as np
import torch
from association.evaluation.local_source_evidence import ResidualScore
from association.evaluation.native_phase_gate import KINDS

def physical_loss(logits,old,mc,target):
    z=torch.sigmoid(logits).double();a=torch.deg2rad(old);b=torch.deg2rad(mc);y=torch.deg2rad(target);lat=(1-z)*a[:,0]+z*b[:,0];s=(1-z)*torch.sin(a[:,1])+z*torch.sin(b[:,1]);c=(1-z)*torch.cos(a[:,1])+z*torch.cos(b[:,1]);degenerate=s.square()+c.square()<=1e-24;lon=torch.atan2(torch.where(degenerate,torch.zeros_like(s),s),torch.where(degenerate,torch.ones_like(c),c))
    choose=(old[:,0]<mc[:,0])|((old[:,0]==mc[:,0])&(old[:,1]<=mc[:,1]));fallback=torch.where(choose[:,None],a,b);lat=torch.where(degenerate,fallback[:,0],lat);lon=torch.where(degenerate,fallback[:,1],lon)
    h=torch.sin((lat-y[:,0])/2).square()+torch.cos(lat)*torch.cos(y[:,0])*torch.sin((lon-y[:,1])/2).square();h=torch.clamp(h,0,1);return 2*6371000*torch.atan2(torch.sqrt(h+1e-24),torch.sqrt(1-h+1e-24))

def fit(data,device,progress=None,steps=128):
    torch.manual_seed(20);w=data['w']/data['w'].sum();norm={};inputs={}
    x=data['x_ORDER'];common_mean=w@x[:,:12];common_std=np.sqrt(w@((x[:,:12]-common_mean)**2));blocks=x[:,12:].reshape(-1,4,5);bm=np.sum(w[:,None,None]*blocks,axis=(0,1))/4;bs=np.sqrt(np.sum(w[:,None,None]*(blocks-bm)**2,axis=(0,1))/4);mean=np.r_[common_mean,np.tile(bm,4)];std=np.r_[common_std,np.tile(bs,4)];std[std<1e-6]=1
    for k in KINDS:
        norm[k]=(mean.copy(),std.copy());z=np.clip((data['x_'+k]-mean)/std,-8,8)
        if k=='MEAN':z[:,12:]=0
        inputs[k]=torch.as_tensor(z,dtype=torch.float32,device=device)
    net=ResidualScore(32).to(device);initial={k:v.detach().clone() for k,v in net.state_dict().items()};nets={};opts={};curves={k:[] for k in KINDS}
    for k in KINDS:
        nets[k]=ResidualScore(32).to(device);nets[k].load_state_dict(initial);opts[k]=torch.optim.Adam(nets[k].parameters(),lr=.001,weight_decay=1e-4)
    old,mc,target=[torch.as_tensor(data[k],dtype=torch.float64,device=device) for k in ('old','mc','target')];mass=torch.as_tensor(w*len(w),dtype=torch.float64,device=device);draws=np.random.default_rng(20).integers(len(w),size=(steps,1024))
    for step,draw in enumerate(draws):
        ix=torch.as_tensor(draw,device=device)
        for k in KINDS:
            opts[k].zero_grad(set_to_none=True);loss=(mass[ix]*physical_loss(nets[k](inputs[k][ix]),old[ix],mc[ix],target[ix])).mean()/1000
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite physical gate loss')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(nets[k].parameters(),1.,error_if_nonfinite=True);opts[k].step();curves[k].append(dict(update=step+1,loss_km=float(loss.detach().cpu()),gradient_norm=float(gn)))
        if progress:progress(step+1)
    result={k:dict(kind=k,weights={n:v.detach().cpu() for n,v in nets[k].state_dict().items()},mean=norm[k][0],std=norm[k][1],parameter_count=sum(v.numel() for v in nets[k].parameters()),updates=steps,curve=curves[k]) for k in KINDS}
    return result,draws,{k:v.cpu() for k,v in initial.items()}

@torch.no_grad()
def predict(model,x,device):
    net=ResidualScore(32).to(device);net.load_state_dict(model['weights']);net.eval();z=np.clip((x-model['mean'])/model['std'],-8,8)
    if model['kind']=='MEAN':z[:,12:]=0
    return torch.sigmoid(net(torch.as_tensor(z,dtype=torch.float32,device=device))).cpu().numpy().astype(float)

def numpy_predict(model,x):
    z=np.clip((x-model['mean'])/model['std'],-8,8).astype(np.float32)
    if model['kind']=='MEAN':z[:,12:]=0
    w={k:v.numpy() for k,v in model['weights'].items()}
    for layer in (0,2):z=z@w[f'net.{layer}.weight'].T+w[f'net.{layer}.bias'];z=z/(1+np.exp(-z))
    z=(z@w['net.4.weight'].T+w['net.4.bias']).ravel();return 1/(1+np.exp(-z))
