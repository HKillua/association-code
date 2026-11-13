"""Query-grid temporal encoding and a shared residual score; no truth in features."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from association.evaluation.native_temporal import slope, predict
from association.evaluation.native_motion import _short_delta

LOCAL_NAMES=('log_increment_vector_disagreement','log_local_position_rmse_km',
 'log_local_velocity_disagreement','log_velocity_vector_scatter',
 'log_query_position_residual_km','log_local_global_slope_difference','local_window_defined')

def local_features(motion,pack,*,speed_unit):
    if speed_unit not in ('kn','mps'):raise ValueError('explicit speed unit')
    states=np.asarray(motion['states'],float);support=motion['support'];q=motion['query_seconds'];out=np.zeros((*support.shape,7))
    for i in range(len(states)):
        ids=np.flatnonzero(support[i])
        if len(ids)<2:continue
        y=states[i,ids,:2].copy();y[:,1]=y[0,1]+np.r_[0.,np.cumsum(_short_delta(np.diff(y[:,1])))];y-=y[0]
        y*=6371000.*np.pi/180*np.array([1.,np.cos(np.deg2rad(states[i,ids,0].mean()))])
        tt=q[ids];globalv=slope(tt,y);speed=states[i,ids,2]*(1852/3600 if speed_unit=='kn' else 1.)
        heading=np.deg2rad(states[i,ids,3]);vel=speed[:,None]*np.column_stack((np.cos(heading),np.sin(heading)))
        for j,k in enumerate(ids):
            pick=np.sort(np.argsort(np.abs(tt-q[k]),kind='stable')[:8]);t=tt[pick];z=y[pick];v=vel[pick]
            sl=slope(t,z);residual=z-z.mean(0)-(t-t.mean())[:,None]*sl
            increments=np.diff(z,axis=0)/np.diff(t)[:,None]-(v[1:]+v[:-1])/2
            x=[np.sqrt(np.mean(np.sum(increments**2,axis=1))),np.sqrt(np.mean(np.sum(residual**2,axis=1)))/1000.,
               np.linalg.norm(sl-v.mean(0)),np.sqrt(np.mean(np.sum((v-v.mean(0))**2,axis=1))),
               np.linalg.norm(y[j]-z.mean(0)-(q[k]-t.mean())*sl)/1000.,np.linalg.norm(sl-globalv)]
            out[i,k,:6]=np.log1p(x);out[i,k,6]=1.
    result=np.zeros((*pack['available'].shape,7));np.add.at(result,pack['node_to_source_row'],out*support[...,None])
    result/=np.maximum(pack['supported_counts'],1)[...,None]
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite local feature')
    return result

class ResidualScore(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x).squeeze(-1)

def fit_residual(x,y,w,base,*,device,steps=256):
    torch.manual_seed(20);w=w/w.sum();mean=w@x;std=np.sqrt(w@((x-mean)**2));std[std<1e-6]=1.
    z=torch.tensor(np.clip((x-mean)/std,-8,8),dtype=torch.float32,device=device)
    target=torch.tensor(y-predict(base,x),dtype=torch.float32,device=device)
    weight=torch.tensor(w*len(w),dtype=torch.float32,device=device);model=ResidualScore(x.shape[1]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4);rng=np.random.default_rng(20);curve=[]
    with torch.no_grad():initial=float((weight*(model(z)-target)**2).mean())
    for step in range(steps):
        ix=torch.as_tensor(rng.integers(len(x),size=4096),device=device);optimizer.zero_grad(set_to_none=True)
        loss=(weight[ix]*(model(z[ix])-target[ix])**2).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite score loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        curve.append(dict(update=step+1,weighted_batch_mse=float(loss.detach()),gradient_norm=float(gn)))
    with torch.no_grad():final=float((weight*(model(z)-target)**2).mean())
    return dict(mean=mean,std=std,base=base,weights={k:v.detach().cpu() for k,v in model.state_dict().items()},
                dimension=x.shape[1],parameter_count=sum(p.numel() for p in model.parameters()),updates=steps,initial_mse=initial,final_mse=final,curve=curve)

@torch.no_grad()
def predict_residual(model,x,device):
    net=ResidualScore(model['dimension']).to(device);net.load_state_dict(model['weights']);net.eval()
    z=torch.tensor(np.clip((x-model['mean'])/model['std'],-8,8),dtype=torch.float32,device=device)
    result=predict(model['base'],x)+net(z).cpu().numpy()
    if not np.isfinite(result).all():raise FloatingPointError('nonfinite residual score')
    return result
