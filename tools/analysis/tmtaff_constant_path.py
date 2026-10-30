"""154: frozen TFΜ path and zero-memory intervention, no training or V."""
from __future__ import annotations
import argparse,json,sys,time
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
from tools.analysis import observation_common as common

def stats(x):
    x=np.asarray(x);flat=x.reshape(-1,x.shape[-1]);spread=np.ptp(flat,axis=0)
    return dict(shape=list(x.shape),finite=bool(np.isfinite(x).all()),rms=float(np.sqrt(np.mean(x*x))),
        maximum_rowwise_channel_range=float(spread.max()),varying_channels=int((spread>1e-12).sum()))

@torch.no_grad()
def trace(fusion,left,right,*,zero_memory=False):
    saved={};handles=[];decoder=[]
    try:
        if zero_memory:handles.append(fusion.fc2.register_forward_hook(lambda m,a,o:torch.zeros_like(o)))
        def hook(name):
            def keep(m,a,o):
                if name=='decoder':decoder.append(o[:,-1:].detach().cpu())
                else:saved[name]=o.detach().cpu()
                if name=='projection':saved['projection_input']=a[0].detach().cpu()
            return keep
        for name,module in [('left_encoder',fusion.left_encoder),('right_encoder',fusion.right_encoder),('fc1',fusion.fc1),('fc2',fusion.fc2),('decoder',fusion.decoder),('projection',fusion.state_projection)]:handles.append(module.register_forward_hook(hook(name)))
        result=fusion(left,right)['states'].detach().cpu();saved['decoder_last']=torch.cat(decoder,dim=1)
        return result.numpy(),{k:v.numpy() for k,v in saved.items()}
    finally:
        for handle in handles:handle.remove()

@torch.no_grad()
def double_readout(fusion,latent,original):
    x=torch.as_tensor(latent,dtype=torch.float64,device=fusion.bos.device)
    raw=F.linear(x,fusion.state_projection.weight.double(),fusion.state_projection.bias.double())
    position=raw[...,:2]
    if fusion.config.output_parameterization=='train_position_affine':position=position*fusion.position_std.double()+fusion.position_mean.double()
    result=np.asarray(original,np.float64).copy();result[...,:2]=position.cpu().numpy()
    return result,raw.cpu().numpy()

def run(a):
    out=a.output.resolve();out.mkdir(exist_ok=False);(out/'private').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-TFM-CONSTANT-PATH-20261030',plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),
        optimizer_updates=0,test_opened=False,validation_opened=False,completed_scenes=0,native_run=str(a.native_run),checkpoint=str(a.checkpoint))
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('8GiB RSS')
    try:
        torch.set_num_threads(4);snapshot=load_complete_snapshot(a.checkpoint,device=a.device,selected=False);model,fusion=snapshot['model'],snapshot['fusion']
        model.requires_grad_(False);fusion.requires_grad_(False);rows=[];values={};head=fusion.state_projection
        parameter=dict(position_projection_row_l2=head.weight[:2].double().norm(dim=1).cpu().tolist(),
            projection_bias_l2=float(head.bias.double().norm()),
            decoder_norms={n:dict(l2=float(p.double().norm()),min_abs=float(p.abs().min()),max_abs=float(p.abs().max())) for n,p in fusion.named_parameters() if n.startswith('decoder.') and '.norm' in n and n.endswith('.weight')})
        ids=common.old.frozen.read(a.native_run/'native-cache.json')['scene_ids']['T_core'][:2]
        with torch.inference_mode():
            for sid in ids:
                cache=load(a.native_run/'private/native-events/T_core'/f'scene-{sid}.npz')
                with np.load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz',allow_pickle=False) as z:identities=z['identities'].copy()
                pairs=[(i,j) for i in range(len(identities)) for j in range(i+1,len(identities)) if identities[i]==identities[j] and cache['sources'][i]!=cache['sources'][j]][:2]
                if len(pairs)!=2:raise ValueError('two declared positive pairs required')
                scene=timed_scene(cache,identities,sid);packed,mask=normalize_scene(scene,snapshot['normalizer']);nodes=sorted({n for p in pairs for n in p});lookup={n:i for i,n in enumerate(nodes)}
                x=torch.as_tensor(packed[nodes],device=a.device,dtype=next(model.parameters()).dtype);visible=torch.as_tensor(mask[nodes],device=a.device);latent=model.complete(x,visible)
                left=torch.stack([latent[lookup[i]] for i,j in pairs]);right=torch.stack([latent[lookup[j]] for i,j in pairs])
                original,capture=trace(fusion,left,right);zero,zero_capture=trace(fusion,left,right,zero_memory=True);double,raw64=double_readout(fusion,capture['projection_input'],original)
                previous=load(a.probe_run/'private'/f'predictions-{sid}.npz')
                np.testing.assert_array_equal(previous['pairs'],np.asarray(pairs))
                predictions=dict(TFM=original,TFM_ZERO_MEMORY=zero,TFM_DOUBLE_PROJECTION=double,CAP128_MEAN=previous['CAP128_MEAN'])
                np.savez_compressed(out/'private'/f'predictions-{sid}.npz',pairs=np.asarray(pairs),**predictions)
                np.savez_compressed(out/'private'/f'path-{sid}.npz',**capture,projection_double=raw64)
                ev=load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz');registry={str(v):i for i,v in enumerate(ev['target_ids'])};targets=[registry[str(identities[i])] for i,j in pairs];truth=ev['target_states'][targets];valid=ev['target_valid'][targets,:,:2].all(-1)
                for n,p in predictions.items():values.setdefault(n,[]).extend(attribute_errors(np.asarray(p,np.float64),truth)['position'][valid].tolist())
                rows.append(dict(scene_id=sid,pairs=len(pairs),path={**{k:stats(v) for k,v in capture.items()},'completed':stats(latent.cpu().numpy())},
                    zero_memory_path={k:stats(v) for k,v in zero_capture.items()},
                    raw_position_range=np.ptp(capture['projection'][...,:2].reshape(-1,2),axis=0).tolist(),
                    double_position_range=np.ptp(double[...,:2].reshape(-1,2),axis=0).tolist(),
                    zero_memory_raw_max_change=float(np.max(np.abs(capture['projection']-zero_capture['projection']))),
                    zero_memory_position_max_change_m=float(attribute_errors(zero.astype(float),original.astype(float))['position'].max()),
                    original153_position_max_change_m=float(attribute_errors(original.astype(float),previous['TFM'].astype(float))['position'].max()),
                    double_readout_position_max_change_m=float(attribute_errors(double,original.astype(float))['position'].max())))
                check(completed_scenes=len(rows))
        metrics={}
        for n,v in values.items():
            v=np.asarray(v);metrics[n]=dict(count=len(v),mae=float(v.mean()),rmse=float(np.sqrt(np.mean(v*v))),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)))
        common.old.frozen.write(out/'summary.json',dict(status='COMPLETED',idea_id=state['idea_id'],scenes=rows,parameters=parameter,position=metrics,
            scope='T2 fixed4 positive pairs; operational path/intervention only, not unique optimization cause or peer ranking'))
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','checkpoint','probe-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
