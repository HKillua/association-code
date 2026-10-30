"""Shared source score: independent objective and train-only difficulty weights."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F
from association.evaluation.local_source_evidence import ResidualScore
from association.evaluation.native_temporal import predict

ARMS=('ABS_UNIFORM','ABS_BALANCED','PAIR_UNIFORM','PAIR_BALANCED')

def paired_training_data(packs,base):
    xs=[];ys=[];errors=[];weights=[];hard=[]
    for p in packs:
        shape=p['features'].shape[:2];score=predict(base,p['features'].reshape(-1,10)).reshape(shape)
        for a,b in zip(p['group_offsets'][:-1],p['group_offsets'][1:]):
            valid=p['labeled'][a:b];count=valid.sum(0);pair_count=np.maximum(count*(count-1)/2,1)
            e=np.where(valid,p['errors_m'][a:b],np.inf);s=np.where(valid,score[a:b],np.inf);best=e.min(0)
            chosen=valid&(s==s.min(0));mass=chosen/np.maximum(chosen.sum(0),1)
            selected=(mass*np.where(valid,p['errors_m'][a:b],0.)).sum(0);difficult=(count>=2)&(selected-np.where(count>=2,best,0.)>1e-6)
            for i in range(a,b):
                for j in range(i+1,b):
                    use=p['labeled'][i]&p['labeled'][j]
                    if not use.any():continue
                    xs.append(np.stack((p['features'][i,use],p['features'][j,use]),axis=1));ys.append(np.stack((p['y'][i,use],p['y'][j,use]),axis=1));errors.append(np.stack((p['errors_m'][i,use],p['errors_m'][j,use]),axis=1));weights.append(1/pair_count[use]);hard.append(difficult[use])
    if not xs:raise ValueError('no valid source pairs')
    x,y,e,w,h=map(np.concatenate,(xs,ys,errors,weights,hard));balanced=w.copy();hm=float(w[h].sum());em=float(w[~h].sum());fallback=hm<=0 or em<=0
    if not fallback:balanced[h]*=.5*(hm+em)/hm;balanced[~h]*=.5*(hm+em)/em
    return dict(x=x,y=y,error=e,weight=w,balanced_weight=balanced,hard=h),dict(pairs=len(x),query_mass=float(w.sum()),hard_query_mass=hm,easy_query_mass=em,balanced_fallback=fallback,balanced_hard_mass=float(balanced[h].sum()),balanced_easy_mass=float(balanced[~h].sum()))

def fit_factorial(packs,normalizer,*,device,steps=128):
    base=normalizer['base'];data,account=paired_training_data(packs,base);x=data['x'];mean=normalizer['mean'];std=normalizer['std'];n=len(x)
    z=torch.tensor(np.clip((x-mean)/std,-8,8),dtype=torch.float32,device=device);y=torch.tensor(data['y'],dtype=torch.float32,device=device)
    original=torch.tensor(predict(base,x.reshape(-1,10)).reshape(n,2),dtype=torch.float32,device=device);delta=data['error'][:,0]-data['error'][:,1]
    target=torch.tensor(np.where(delta==0,.5,(delta>0).astype(float)),dtype=torch.float32,device=device)
    ws={k:torch.tensor(data[key]/data[key].sum()*n,dtype=torch.float32,device=device) for k,key in [('uniform','weight'),('balanced','balanced_weight')]}
    nets={};opts={};curves={k:[] for k in ARMS}
    for name in ARMS:
        torch.manual_seed(20);nets[name]=ResidualScore(10).to(device);opts[name]=torch.optim.Adam(nets[name].parameters(),lr=.001,weight_decay=1e-4)
    def losses(name,index):
        score=original[index]+nets[name](z[index].reshape(-1,10)).reshape(-1,2)
        value=((score-y[index])**2).mean(-1) if name.startswith('ABS') else F.binary_cross_entropy_with_logits(score[:,0]-score[:,1],target[index],reduction='none')
        return (value*ws['balanced' if name.endswith('BALANCED') else 'uniform'][index]).mean()
    with torch.no_grad():initial={name:float(losses(name,slice(None))) for name in ARMS}
    rng=np.random.default_rng(20)
    for step in range(steps):
        index=torch.as_tensor(rng.integers(n,size=2048),device=device)
        for name in ARMS:
            opts[name].zero_grad(set_to_none=True);loss=losses(name,index)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(nets[name].parameters(),1.,error_if_nonfinite=True);opts[name].step();curves[name].append(dict(update=step+1,weighted_loss=float(loss.detach()),gradient_norm=float(gn)))
    with torch.no_grad():final={name:float(losses(name,slice(None))) for name in ARMS}
    models={name:dict(mean=mean.copy(),std=std.copy(),base=base,weights={k:v.detach().cpu() for k,v in nets[name].state_dict().items()},dimension=10,parameter_count=sum(p.numel() for p in nets[name].parameters()),updates=steps,initial_loss=initial[name],final_loss=final[name],curve=curves[name]) for name in ARMS}
    return models,account
