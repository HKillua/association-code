"""Task-adapted PTFM core: four-state recovery, not an author reproduction.

Adaptations are explicit: first representative + supported-helper pool, Q64
native completion, missing-helper context,208->128 projection, T-only z-score,
and Eq21 interpreted as mean(-log(w+epsilon)), consistent with Eq23.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn

SCOPE='ptfm-task-adapted-v1'
ROUTING_SEMANTICS='anchor-supported-helper-fullstate-maskcontext-v1'
LOSS_SEMANTICS='ptfm-normalized-position-speed-mse-native-circular-heading-adaptive-v1'
RESIDUAL_SCOPE='ptfm-prior095-residual-v1'
RESIDUAL_ROUTING_SEMANTICS='anchor-supported-helper-fullstate-maskcontext-prior095-residual-v1'
RESIDUAL_LOSS_SEMANTICS='ptfm-prior095-residual-normalized-position-speed-mse-native-circular-heading-adaptive-v1'
POSITION_MODES=('direct','prior095_residual')


def _short(x):
    return (np.asarray(x)+180.)%360.-180.


def local_states(states, origin):
    """Exact reversible local native units: deg offsets, native speed, deg course."""
    states=np.asarray(states,np.float64); origin=np.asarray(origin,np.float64)
    out=states.copy(); out[...,0]-=origin[0]
    out[...,1]=_short(states[...,1]-origin[1]); out[...,3]=_short(states[...,3]-origin[3])
    return out


def native_states(local, origin, *, position_base=None, position_residual_deg=None):
    """Invert local units; reflect latitude at poles and wrap longitude/course.

    Reflection is only a finite geographic output parameterization, not data
    clipping. Speed uses abs(native speed), an explicit nonnegative-domain
    adaptation. Raw output excursions and negative speed logits are reported.
    """
    lat=torch.remainder(local[:,0]+origin[0]+180.,360.)-180.
    crossed=lat.abs()>90.
    latitude=torch.where(lat>90.,180.-lat,torch.where(lat< -90.,-180.-lat,lat))
    longitude=torch.remainder(local[:,1]+origin[1]+crossed*180.+180.,360.)-180.
    heading=torch.remainder(local[:,3]+origin[3],360.)
    position=torch.stack((latitude,longitude),-1)
    if (position_base is None)!=(position_residual_deg is None):
        raise ValueError('PTFM residual inverse requires both base and residual')
    if position_base is not None:
        if (position_base.shape!=local.shape or position_base.dtype!=torch.float64
                or position_residual_deg.shape!=(len(local),2)
                or position_residual_deg.dtype!=torch.float64 or position_base.device!=local.device
                or position_residual_deg.device!=local.device
                or not torch.isfinite(position_base).all() or not torch.isfinite(position_residual_deg).all()
                or (position_base[:,0].abs()>90.).any()):
            raise ValueError('PTFM residual inverse requires finite geographic base and degree residual')
        expected_local=torch.stack((position_base[:,0]-origin[0],
            torch.remainder(position_base[:,1]-origin[1]+180.,360.)-180.),-1)+position_residual_deg
        if not torch.equal(local[:,:2],expected_local):
            raise ValueError('PTFM local position does not equal base plus degree residual')
        # Modulo arithmetic can lose a few bits even at zero. Correct only its
        # value, keeping precisely the same native-coordinate gradient/map.
        zero=(position_residual_deg==0).all(-1)
        exact=position_base[:,:2]+(position-position.detach())
        position=torch.where(zero[:,None],exact,position)
    return torch.cat((position,local[:,2:3].abs(),heading[:,None]),-1)


def ptfm_inputs(motion,members,*,device='cpu',include_prior=False):
    """Only observed/native evidence enters these two streams and context."""
    members=np.asarray(members,np.int64)
    states=np.asarray(motion['states'],np.float64); support=np.asarray(motion['support'])
    if (members.ndim!=1 or not len(members) or len(set(members.tolist()))!=len(members)
            or (members<0).any() or (members>=len(states)).any() or states.ndim!=3 or states.shape[-1]!=4
            or not np.isfinite(states).all() or support.shape!=states.shape[:2] or support.dtype!=bool):
        raise ValueError('PTFM requires finite native states, unique members and boolean support')
    anchor=states[members[0]]; helpers=np.sort(members[1:]); q=len(anchor)
    times=np.asarray(motion['query_seconds'],np.float64)
    if times.shape!=(q,) or not np.isfinite(times).all() or (np.diff(times)<=0).any():
        raise ValueError('PTFM requires actual increasing query seconds')
    count=support[helpers].sum(0); present=count>0; divisor=np.maximum(count,1)
    # Neutral numeric carrier where absent; its enhanced features are masked out.
    helper=anchor.copy()
    if len(helpers):
        for col in (0,2):
            mean=(states[helpers,:,col]*support[helpers]).sum(0)/divisor
            helper[present,col]=mean[present]
        for col in (1,3):
            angles=np.deg2rad(states[helpers,:,col]); weights=support[helpers]
            sine=(np.sin(angles)*weights).sum(0)/divisor
            cosine=(np.cos(angles)*weights).sum(0)/divisor
            usable=present & (np.hypot(sine,cosine)>1e-12)
            value=np.rad2deg(np.arctan2(sine,cosine))
            helper[usable,col]=_short(value[usable]) if col==1 else value[usable]%360.
    origin=anchor[0].copy()
    result=(torch.as_tensor(local_states(anchor,origin),dtype=torch.float64,device=device),
            torch.as_tensor(local_states(helper,origin),dtype=torch.float64,device=device),
            torch.as_tensor(np.column_stack((present,support[members[0]])),dtype=torch.float32,device=device),
            torch.as_tensor(origin,dtype=torch.float64,device=device))
    if include_prior:
        # The prior is an output bypass, never an additional encoder feature.
        # Reproduce the ordinary supported mean and float32 sigmoid arithmetic
        # of the frozen .95 control, without constructing gate-model features.
        from association.evaluation.fusion_controls import ordinary_fusion
        from association.models.support_gate import mix_gate_states
        ordered=np.r_[members[:1],helpers]
        nearest=np.asarray(motion['nearest_seconds'],np.float64)
        if nearest.shape!=support.shape:raise ValueError('PTFM prior requires nearest-query seconds')
        ordinary=ordinary_fusion(states[ordered],nearest[ordered],support[ordered],[list(range(len(ordered)))])
        mean=torch.as_tensor(ordinary['predictions']['mean_supported'][0],dtype=torch.float64)
        initial=torch.sigmoid(torch.full((q,),math.log(19.),dtype=torch.float32))
        base=mix_gate_states(torch.as_tensor(anchor,dtype=torch.float64),mean,initial,
            torch.as_tensor(~support[members[0]]&present,dtype=torch.bool))
        result=(*result,base.to(device))
    return result


class PTFM(nn.Module):
    def __init__(self,normalizer,*,position_mode='direct'):
        super().__init__()
        if position_mode not in POSITION_MODES:raise ValueError('unknown PTFM position output mode')
        self.position_mode=position_mode
        mean=np.asarray(normalizer['mean'],np.float64); scale=np.asarray(normalizer['scale'],np.float64)
        if (mean.shape!=(4,) or scale.shape!=(4,) or not np.isfinite(mean).all()
                or not np.isfinite(scale).all() or (scale<=0).any() or normalizer.get('split')!='T_core'
                or normalizer.get('truth_used') is not False):
            raise ValueError('PTFM requires training-only finite native-unit normalizer')
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float64))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float64))
        self.anchor=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU())
        self.helper=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Linear(32,64),nn.ReLU())
        self.attention=nn.Sequential(nn.Linear(144,128),nn.ReLU(),nn.Linear(128,144),nn.Sigmoid())
        self.temporal=nn.LSTM(144,64,batch_first=True)
        self.gate=nn.Linear(208,208)
        self.context_gate=nn.Linear(2,208,bias=False)
        self.residual=nn.Linear(144,208,bias=False)
        self.projection=nn.Linear(208,128)
        self.output=nn.Linear(128,4)
        self.task_weights=nn.Linear(128,3)
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)
        for name,value in self.temporal.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(value)
            else: nn.init.zeros_(value)
        # Do not add a module or consume RNG: all other initial tensors remain
        # bit-identical to the direct-output model at the same seed.
        if position_mode=='prior095_residual':
            with torch.no_grad():self.output.weight[:2].zero_();self.output.bias[:2].zero_()
        self.motion_observation_policy='canonical_all'
        self.routing_semantics=RESIDUAL_ROUTING_SEMANTICS if position_mode=='prior095_residual' else ROUTING_SEMANTICS

    def forward(self,anchor,helper,context,origin,base_states=None):
        q=len(anchor)
        if (anchor.shape!=(q,4) or helper.shape!=anchor.shape or context.shape!=(q,2) or origin.shape!=(4,)
                or anchor.dtype!=torch.float64 or helper.dtype!=torch.float64
                or not all(torch.isfinite(x).all() for x in (anchor,helper,context,origin))
                or not ((context==0)|(context==1)).all()):
            raise ValueError('invalid PTFM local streams or explicit availability context')
        residual_mode=self.position_mode=='prior095_residual'
        if residual_mode!=(base_states is not None):
            raise ValueError('PTFM physical prior must be supplied only for residual position mode')
        if base_states is not None and (base_states.shape!=anchor.shape or base_states.dtype!=torch.float64
                or base_states.device!=anchor.device or not torch.isfinite(base_states).all()
                or (base_states[:,0].abs()>90.).any()):
            raise ValueError('PTFM physical prior must be finite geographic float64 [Q,4]')
        a=self.anchor(((anchor-self.mean)/self.scale).float())
        b=self.helper(((helper-self.mean)/self.scale).float())*context[:,:1]
        time=torch.arange(q,dtype=torch.float32,device=anchor.device)[:,None]
        channels=torch.arange(16,dtype=torch.float32,device=anchor.device)[None,:]
        angle=time/torch.pow(torch.tensor(10000.,device=anchor.device),channels/16.)
        pe=torch.where((channels.long()%2)==0,angle.sin(),angle.cos())
        x=torch.cat((a,b,pe),-1); attended=x*self.attention(x)
        temporal=self.temporal(x.unsqueeze(0))[0].squeeze(0)
        joined=torch.cat((temporal,attended),-1)
        fused=joined*torch.sigmoid(self.gate(joined)+self.context_gate(context))+self.residual(attended)
        latent=self.projection(fused)
        normalized=self.output(latent).double()
        local=normalized*self.scale+self.mean
        residual=None
        if residual_mode:
            residual=normalized[:,:2]*self.scale[:2]
            prior_local=torch.stack((base_states[:,0]-origin[0],
                torch.remainder(base_states[:,1]-origin[1]+180.,360.)-180.),-1)
            local=torch.cat((prior_local+residual,local[:,2:]),-1)
        states=native_states(local,origin,position_base=base_states,position_residual_deg=residual)
        weights=torch.softmax(self.task_weights(latent),-1).double()
        if not torch.isfinite(states).all(): raise FloatingPointError('nonfinite PTFM states')
        result=dict(states=states,normalized_states=normalized,local_states=local,origin=origin,
                    task_weights=weights,helper_present=context[:,0].bool(),anchor_supported=context[:,1].bool())
        if residual_mode:result.update(base_states=base_states,position_residual_deg=residual)
        return result

    def l2(self):
        return sum(value.square().sum() for name,value in self.temporal.named_parameters() if 'weight' in name)


def objectives(model,result,target,valid,*,mode='adaptive'):
    """Paper adaptive objective with per-task valid masks and native-angle loss.

    Position is squared standardized local lat/lon distance; speed standardized
    MSE. Circular course error is computed after native inversion, then divided
    by the T-fitted course scale. A task with no labels contributes neither loss
    nor sigma. No labels are used in inference or normalizer fitting.
    """
    if mode not in ('adaptive','fixed-equal','position-only'):raise ValueError('unknown PTFM objective mode')
    if target.shape!=result['states'].shape or valid.shape!=target.shape or valid.dtype!=torch.bool:
        raise ValueError('PTFM requires per-attribute target masks')
    if not torch.isfinite(target[valid]).all(): raise ValueError('valid PTFM truth must be finite')
    zero=result['normalized_states'].sum()*0.+result['task_weights'].sum()*0.
    safe=torch.where(valid,target,torch.zeros_like(target))
    # Supervise the actual native output after geographic/speed inversion, not
    # an unwrapped latent value with a different physical interpretation.
    dlat=result['states'][:,0]-safe[:,0]
    dlon=torch.remainder(result['states'][:,1]-safe[:,1]+180.,360.)-180.
    position=(dlat/model.scale[0]).square()+(dlon/model.scale[1]).square()
    speed=((result['states'][:,2]-safe[:,2])/model.scale[2]).square()
    heading=(torch.remainder(result['states'][:,3]-safe[:,3]+180.,360.)-180.).abs()/model.scale[3]
    masks=(valid[:,:2].all(-1),valid[:,2],valid[:,3]); losses=[]; sigmas=[]; counts=[]
    for index,(error,mask) in enumerate(zip((position,speed,heading),masks)):
        counts.append(int(mask.sum()))
        loss=error[mask].mean()+zero if mask.any() else zero
        sigma=-torch.log(result['task_weights'][mask,index]+1e-8).mean()+zero if mask.any() else zero
        losses.append(loss);sigmas.append(sigma)
    if mode=='adaptive':
        coefficients=[torch.exp(-sigma) if count else zero for sigma,count in zip(sigmas,counts)]
        regularizers=[sigma if count else zero for sigma,count in zip(sigmas,counts)]
    else:
        enabled=[bool(count and (mode=='fixed-equal' or index==0)) for index,count in enumerate(counts)]
        coefficients=[zero+1./3 if use else zero for use in enabled]
        regularizers=[zero+math.log(3.) if use else zero for use in enabled]
    contributions=[coefficient*loss+regularizer for coefficient,loss,regularizer in zip(coefficients,losses,regularizers)]
    total=sum(contributions)+zero
    if not torch.isfinite(total): raise FloatingPointError('nonfinite PTFM objective')
    return total,dict(task_valid_counts=counts,task_losses=[float(x.detach()) for x in losses],
                      task_sigmas=[float(x.detach()) for x in sigmas],
                      objective_mode=mode,actual_task_coefficients=[float(x.detach()) for x in coefficients],
                      actual_task_regularizers=[float(x.detach()) for x in regularizers],
                      actual_task_contributions=[float(x.detach()) for x in contributions])
