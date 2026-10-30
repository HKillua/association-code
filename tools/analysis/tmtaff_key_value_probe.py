"""164: frozen key/value scale factorial. No training or deployment selection."""
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
from tools.analysis.tmtaff_decoder_layers import layer_trace
from tools.analysis.tmtaff_constant_path import stats
from tools.analysis import observation_common as common
ARMS={'NORMAL':(1.,1.),'KEY_ONLY':(.01,1.),'VALUE_ONLY':(1.,.01),'BOTH':(.01,.01)}

@torch.no_grad()
def kv_trace(fusion,left,right,key_scale,value_scale):
    handles=[]
    def change(module,args):
        q,k,v,*rest=args
        if key_scale==1. and value_scale==1.:return args
        if key_scale==value_scale and k is v:
            shared=k*key_scale;return (q,shared,shared,*rest)
        return (q,k if key_scale==1. else k*key_scale,v if value_scale==1. else v*value_scale,*rest)
    try:
        for layer in fusion.decoder.layers:handles.append(layer.multihead_attn.register_forward_pre_hook(change))
        return layer_trace(fusion,left,right,1.)
    finally:
        for handle in handles:handle.remove()

def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'private').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-TFM-KEY-VALUE-SEPARATION-20261031',plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,test_opened=False,validation_opened=False,native_run=str(a.native_run),checkpoint=str(a.checkpoint),layer_run=str(a.layer_run))
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);s=load_complete_snapshot(a.checkpoint,device=a.device,selected=False);model,fusion=s['model'],s['fusion'];model.requires_grad_(False);fusion.requires_grad_(False);rows=[];errors={};captures={n:[] for n in ARMS};preds={n:[] for n in ARMS};check(phase='frozen_T')
        for sid in common.old.frozen.read(a.native_run/'native-cache.json')['scene_ids']['T_core'][:2]:
            cache=load(a.native_run/'private/native-events/T_core'/f'scene-{sid}.npz');ev=load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz');ident=ev['identities'];pairs=[(i,j) for i in range(len(ident)) for j in range(i+1,len(ident)) if ident[i]==ident[j] and cache['sources'][i]!=cache['sources'][j]][:2]
            if len(pairs)!=2:raise ValueError('fixed4pairs')
            scene=timed_scene(cache,ident,sid);packed,mask=normalize_scene(scene,s['normalizer']);nodes=sorted({n for pair in pairs for n in pair});lookup={n:i for i,n in enumerate(nodes)};predictions={};paths={}
            with torch.inference_mode():
                latent=model.complete(torch.as_tensor(packed[nodes],dtype=next(model.parameters()).dtype,device=a.device),torch.as_tensor(mask[nodes],device=a.device));left=torch.stack([latent[lookup[i]] for i,j in pairs]);right=torch.stack([latent[lookup[j]] for i,j in pairs])
                for name,(ks,vs) in ARMS.items():
                    p,capture,attention=kv_trace(fusion,left,right,ks,vs);predictions[name]=p;captures[name].append(capture);preds[name].append(p);paths[name]=dict(path={k:stats(v) for k,v in capture.items()},first_step_cross_attention_float64_replay=attention);np.savez_compressed(out/'private'/f'path-{sid}-{name}.npz',**capture)
            np.savez_compressed(out/'private'/f'predictions-{sid}.npz',pairs=pairs,**predictions);old=load(a.layer_run/'private'/f'predictions-{sid}.npz');matches={name:float(attribute_errors(predictions[name].astype(float),old[key].astype(float))['position'].max()) for name,key in [('NORMAL','memory_1.0'),('BOTH','memory_0.01')]}
            registry={str(v):i for i,v in enumerate(ev['target_ids'])};ix=[registry[str(ident[i])] for i,j in pairs];truth=ev['target_states'][ix];valid=ev['target_valid'][ix,:,:2].all(-1)
            for name,p in predictions.items():errors.setdefault(name,[]).extend(attribute_errors(p.astype(float),truth)['position'][valid].tolist())
            rows.append(dict(scene_id=sid,pairs=len(pairs),matched161_max_position_difference_m=matches,arms=paths));check(completed_scenes=len(rows))
        response={n:dict(layers={k:float(np.ptp(np.concatenate([v[k] for v in captures[n]],axis=0).astype(float),axis=0).max()) for k in captures[n][0]},position_degrees_same_decoding_index_cross_input=np.ptp(np.concatenate(preds[n]).astype(float)[...,:2],axis=0).max(axis=0).tolist()) for n in ARMS}
        metrics={n:dict(count=len(v),mae=float(np.mean(v)),rmse=float(np.sqrt(np.mean(np.asarray(v)**2))),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99))) for n,v in errors.items()};common.old.frozen.write(out/'summary.json',dict(status='COMPLETED',scenes=rows,position=metrics,cross_input_response=response,scope='T2 key/value factorial with autoregressive feedback; first-step attention replay and decoding-index response, not unique training cause'))
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','checkpoint','layer-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
