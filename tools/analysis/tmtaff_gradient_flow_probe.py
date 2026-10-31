"""167: original physical FusionLoss gradients at immutable107, no update."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.training.tmtaff import load_complete_snapshot
from association.training.common import normalize_scene
from association.evaluation.fusion_controls import attribute_errors
from tools.analysis.tmtaff_training_physical_probe import load,timed_scene
from tools.analysis import observation_common as common


def gradient_record(fusion,objective,left,right,target,valid,*,training,clip):
    before={k:v.detach().clone() for k,v in fusion.state_dict().items()}
    objective_before={k:v.detach().clone() for k,v in objective.state_dict().items()}
    old_mode=fusion.training;fusion.requires_grad_(True);objective.requires_grad_(True);fusion.train(training)
    torch.manual_seed(20)
    x=left.detach().clone().requires_grad_(True);y=right.detach().clone().requires_grad_(True)
    result=fusion(x,y);prediction=result['states'];memory=result['fused_features']
    loss=objective(prediction,target,valid)
    position=sum(.5*torch.exp(-objective.log_variance[i])*loss['losses'][name] for i,name in enumerate(objective.feature_names[:2]))
    parameters=list(fusion.named_parameters())+[('objective.'+k,v) for k,v in objective.named_parameters()]
    names=[k for k,_ in parameters]+['input.left','input.right','activation.memory','activation.states']
    tensors=[v for _,v in parameters]+[x,y,memory,prediction]
    gradients={}
    for tag,value in [('total',loss['total']),('position',position)]:
        grad=torch.autograd.grad(value,tensors,allow_unused=True,retain_graph=tag=='total')
        records={}
        for name,g in zip(names,grad):
            if g is None:records[name]=dict(none=True,l2=0.,elements=0,zero_elements=0,max_abs=0.)
            else:
                if not torch.isfinite(g).all():raise FloatingPointError('nonfinite diagnostic gradient')
                records[name]=dict(none=False,l2=float(torch.linalg.vector_norm(g.double())),elements=g.numel(),zero_elements=int((g==0).sum()),max_abs=float(g.abs().max()))
        paramnorm=float(np.sqrt(sum(v['l2']**2 for k,v in records.items() if not k.startswith(('input.','activation.')))))
        gradients[tag]=dict(parameters=records,total_parameter_gradient_l2=paramnorm,original_clip=clip,theoretical_clip_multiplier=min(1.,clip/(paramnorm+1e-6)))
    for k,v in fusion.state_dict().items():
        if not torch.equal(v,before[k]):raise ValueError('frozen fusion mutated: '+k)
    for k,v in objective.state_dict().items():
        if not torch.equal(v,objective_before[k]):raise ValueError('frozen objective mutated: '+k)
    fusion.train(old_mode)
    return prediction.detach().cpu().numpy(),dict(mode='train_seed20' if training else 'eval',total_loss=float(loss['total'].detach()),
        losses={k:float(v.detach()) for k,v in loss['losses'].items()},variance=loss['variance'].detach().cpu().tolist(),
        valid_attribute_counts=valid.sum((0,1)).cpu().tolist(),gradients=gradients,parameter_updates=0,weights_unchanged=True)


def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'private').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-TFM-FROZEN-GRADIENT-FLOW-20261031',plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,test_opened=False,validation_opened=False,checkpoint=str(a.checkpoint),native_run=str(a.native_run),key_value_run=str(a.key_value_run))
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);snapshot=load_complete_snapshot(a.checkpoint,device=a.device,selected=False);model,fusion,objective=[snapshot[k] for k in ('model','fusion','objective')];model.requires_grad_(False).eval()
        payload=torch.load(a.checkpoint,map_location='cpu',weights_only=True);clip=float(payload['config']['gradient_clip']);del payload
        rows=[];errors={'eval':[],'train':[]}
        for sid in common.old.frozen.read(a.native_run/'native-cache.json')['scene_ids']['T_core'][:2]:
            cache=load(a.native_run/'private/native-events/T_core'/f'scene-{sid}.npz');ev=load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz');ident=ev['identities']
            pairs=[(i,j) for i in range(len(ident)) for j in range(i+1,len(ident)) if ident[i]==ident[j] and cache['sources'][i]!=cache['sources'][j]][:2]
            if len(pairs)!=2:raise ValueError('fixed original four pairs')
            scene=timed_scene(cache,ident,sid);packed,mask=normalize_scene(scene,snapshot['normalizer']);nodes=sorted({n for pair in pairs for n in pair});lookup={n:i for i,n in enumerate(nodes)}
            with torch.no_grad():
                latent=model.complete(torch.as_tensor(packed[nodes],dtype=next(model.parameters()).dtype,device=a.device),torch.as_tensor(mask[nodes],device=a.device))
            left=torch.stack([latent[lookup[i]] for i,j in pairs]);right=torch.stack([latent[lookup[j]] for i,j in pairs]);registry={str(v):i for i,v in enumerate(ev['target_ids'])};ix=[registry[str(ident[i])] for i,j in pairs]
            target=torch.as_tensor(ev['target_states'][ix],dtype=left.dtype,device=a.device);valid=torch.as_tensor(ev['target_valid'][ix],device=a.device);arms={};preds={}
            for name,training in [('eval',False),('train',True)]:
                pred,record=gradient_record(fusion,objective,left,right,target,valid,training=training,clip=clip);arms[name]=record;preds[name]=pred
                use=ev['target_valid'][ix,:,:2].all(-1);errors[name].extend(attribute_errors(pred.astype(float),ev['target_states'][ix])['position'][use].tolist())
            layer_run=Path(common.old.frozen.read(a.key_value_run/'run.json')['layer_run']);normal=load(layer_run/'private'/f'predictions-{sid}.npz')['memory_1.0']
            maxdiff=float(attribute_errors(preds['eval'].astype(float),normal.astype(float))['position'].max())
            np.savez_compressed(out/'private'/f'predictions-{sid}.npz',**preds)
            rows.append(dict(scene_id=sid,pairs=2,normal164_max_position_difference_m=maxdiff,arms=arms));check(completed_scenes=len(rows))
        position={n:dict(count=len(v),mae=float(np.mean(v)),rmse=float(np.sqrt(np.mean(np.asarray(v)**2))),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99))) for n,v in errors.items()}
        common.old.frozen.write(out/'summary.json',dict(status='COMPLETED',scenes=rows,position=position,scope='frozen107 original FusionLoss at T2; new dropout RNG, no optimizer, not original training-gradient replay or causal certification'))
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','checkpoint','key-value-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
