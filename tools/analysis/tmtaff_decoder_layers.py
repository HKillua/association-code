"""161: frozen decoder layer path and memory-scale interventions, T2 only."""
from __future__ import annotations
import argparse,sys,time,math
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from association.training.tmtaff import load_complete_snapshot
from association.training.common import normalize_scene
from association.evaluation.fusion_controls import attribute_errors
from tools.analysis.tmtaff_training_physical_probe import load,timed_scene
from tools.analysis.tmtaff_constant_path import stats
from tools.analysis import observation_common as common

@torch.no_grad()
def layer_trace(fusion,left,right,scale):
    captured={};attention={};handles=[]
    def keep(name,tuple_output=False):
        def hook(module,args,out):
            x=out[0] if tuple_output else out;captured.setdefault(name,[]).append(x[:,-1:].detach().cpu())
        return hook
    def replay(name):
        def hook(module,args):
            if name in attention:return
            query,key=args[:2];d=module.embed_dim;heads=module.num_heads;dim=d//heads;w=module.in_proj_weight.double();b=module.in_proj_bias
            bias=None if b is None else b.double();q=F.linear(query.double(),w[:d],None if bias is None else bias[:d]);k=F.linear(key.double(),w[d:2*d],None if bias is None else bias[d:2*d]);q=q.reshape(len(q),-1,heads,dim).transpose(1,2);k=k.reshape(len(k),-1,heads,dim).transpose(1,2)
            logits=q@k.transpose(-1,-2)/math.sqrt(dim);prob=torch.softmax(logits,-1);entropy=-(prob*torch.log(prob.clamp_min(1e-300))).sum(-1)
            attention[name]=dict(logit_max_abs=float(logits.abs().max()),logit_range_max=float((logits.max(-1).values-logits.min(-1).values).max()),entropy_mean=float(entropy.mean()),max_probability_mean=float(prob.max(-1).values.mean()),heads=int(heads),keys=int(key.shape[1]))
        return hook
    try:
        handles.append(fusion.fc2.register_forward_hook(lambda m,a,o:o*scale))
        for i,layer in enumerate(fusion.decoder.layers):
            for name,module,is_tuple in [('layer',layer,False),('norm1',layer.norm1,False),('cross',layer.multihead_attn,True),('norm2',layer.norm2,False),('norm3',layer.norm3,False)]:handles.append(module.register_forward_hook(keep(f'{i}_{name}',is_tuple)))
            handles.append(layer.multihead_attn.register_forward_pre_hook(replay(str(i))))
        output=fusion(left,right)['states'].detach().cpu().numpy()
        return output,{k:torch.cat(v,dim=1).numpy() for k,v in captured.items()},attention
    finally:
        for handle in handles:handle.remove()

def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'private').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-TFM-DECODER-LAYERS-20261031',plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,test_opened=False,validation_opened=False,native_run=str(a.native_run),checkpoint=str(a.checkpoint))
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('8GiB RSS')
    try:
        torch.set_num_threads(4);s=load_complete_snapshot(a.checkpoint,device=a.device,selected=False);model,fusion=s['model'],s['fusion'];model.requires_grad_(False);fusion.requires_grad_(False);rows=[];values={}
        for sid in common.old.frozen.read(a.native_run/'native-cache.json')['scene_ids']['T_core'][:2]:
            cache=load(a.native_run/'private/native-events/T_core'/f'scene-{sid}.npz');ev=load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz');identity=ev['identities'];pairs=[(i,j) for i in range(len(identity)) for j in range(i+1,len(identity)) if identity[i]==identity[j] and cache['sources'][i]!=cache['sources'][j]][:2]
            if len(pairs)!=2:raise ValueError('declared4pairs')
            scene=timed_scene(cache,identity,sid);packed,mask=normalize_scene(scene,s['normalizer']);nodes=sorted({n for pair in pairs for n in pair});lookup={n:i for i,n in enumerate(nodes)}
            with torch.inference_mode():
                latent=model.complete(torch.as_tensor(packed[nodes],dtype=next(model.parameters()).dtype,device=a.device),torch.as_tensor(mask[nodes],device=a.device));left=torch.stack([latent[lookup[i]] for i,j in pairs]);right=torch.stack([latent[lookup[j]] for i,j in pairs]);predictions={};paths={}
                for scale in (1.,0.,.01,.1):
                    name='memory_'+str(scale);prediction,capture,attention=layer_trace(fusion,left,right,scale);predictions[name]=prediction;paths[name]=dict(path={k:stats(v) for k,v in capture.items()},first_step_cross_attention_float64_replay=attention);np.savez_compressed(out/'private'/f'layers-{sid}-{scale}.npz',**capture)
            np.savez_compressed(out/'private'/f'predictions-{sid}.npz',pairs=pairs,**predictions)
            old=load(a.path_run/'private'/f'predictions-{sid}.npz');maxchange=float(attribute_errors(predictions['memory_1.0'].astype(float),old['TFM'].astype(float))['position'].max())
            registry={str(v):i for i,v in enumerate(ev['target_ids'])};indices=[registry[str(identity[i])] for i,j in pairs];truth=ev['target_states'][indices];valid=ev['target_valid'][indices,:,:2].all(-1)
            for name,p in predictions.items():values.setdefault(name,[]).extend(attribute_errors(p.astype(float),truth)['position'][valid].tolist())
            rows.append(dict(scene_id=sid,pairs=len(pairs),original154_max_position_change_m=maxchange,scales=paths));check(completed_scenes=len(rows))
        metrics={n:dict(count=len(v),mae=float(np.mean(v)),rmse=float(np.sqrt(np.mean(np.asarray(v)**2))),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99))) for n,v in values.items()}
        common.old.frozen.write(out/'summary.json',dict(status='COMPLETED',idea_id=state['idea_id'],scenes=rows,position=metrics,scope='frozen T2 layer response/memory-scale intervention; attention first-step double replay, not gradient/optimization root-cause proof'))
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','checkpoint','path-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
