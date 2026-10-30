"""153: immutable107 T8 physical probe; no training, V, raw CSV or test."""
from __future__ import annotations
import argparse,json,sys,time,resource
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.data.timed import prepare_timed_scene
from association.data.physical_units import SPEED_UNIT_POLICY
from association.data.motion_evidence import DATA,scene_motion_evidence
from association.training.common import normalize_scene
from association.training.tmtaff import load_complete_snapshot
from association.evaluation.fusion_controls import ordinary_fusion,attribute_errors
from tools.analysis import observation_common as common

IDEA='IDEA-TMTAFF-T-PHYSICAL-PROBE-20261030'
NAMES=('TFM','TFM_SWAP','TFM_SELF_LEFT','TFM_STATE_HOLD','CAP128_MEAN','CANONICAL_MEAN')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def timed_scene(cache,identities,sid):
    tracks=[];intervals=[]
    for i,(a,b) in enumerate(zip(cache['event_offsets'][:-1],cache['event_offsets'][1:])):
        times=cache['event_times_seconds'][a:b];values=cache['event_values'][a:b]
        tracks.append(dict(node_id=str(cache['node_ids'][i]),source=str(cache['sources'][i]),identity_id=str(identities[i]),canonical_times=times,canonical_values=values))
        intervals.append([times[0],times[-1]])
    native=dict(scene_id=sid,split='T_core',tracks=tracks,node_ids=cache['node_ids'],sources=cache['sources'],identities=identities,intervals=np.asarray(intervals).reshape(-1,2))
    return prepare_timed_scene(native,**DATA,query_times=cache['query_seconds'],speed_unit_policy=SPEED_UNIT_POLICY)

def run(a):
    output=a.output.resolve();output.mkdir(exist_ok=False);(output/'private').mkdir()
    started=time.monotonic();state=dict(status='RUNNING',idea_id=IDEA,plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),
        optimizer_updates=0,validation_opened=False,test_opened=False,raw_reference_read=False,completed_scenes=0,
        checkpoint=str(a.checkpoint.resolve()),native_run=str(a.native_run.resolve()),methods=NAMES)
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(output/'run.json',state)
        if state['peak_rss_bytes']>6*1024**3:raise MemoryError('6GiB resident budget')
    try:
        torch.set_num_threads(4);check(phase='load')
        snapshot=load_complete_snapshot(a.checkpoint,device=a.device,selected=False)
        model,fusion=snapshot['model'],snapshot['fusion'];model.requires_grad_(False);fusion.requires_grad_(False)
        ids=common.old.frozen.read(a.native_run/'native-cache.json')['scene_ids']['T_core'][:8]
        values={};counts={};rows=[]
        with torch.inference_mode():
            for sid in ids:
                cache=load(a.native_run/'private/native-events/T_core'/f'scene-{sid}.npz')
                # Only training identities for selecting positive input pairs; reference target is read later.
                with np.load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz',allow_pickle=False) as z:identities=z['identities'].copy()
                scene=timed_scene(cache,identities,sid);packed,mask=normalize_scene(scene,snapshot['normalizer'])
                pairs=[]
                for i in range(len(identities)):
                    for j in range(i+1,len(identities)):
                        if identities[i]==identities[j] and cache['sources'][i]!=cache['sources'][j]:pairs.append((i,j))
                        if len(pairs)==2:break
                    if len(pairs)==2:break
                if not pairs:raise ValueError('declared T scene has no cross-source positive pair')
                nodes=sorted({i for pair in pairs for i in pair});lookup={n:i for i,n in enumerate(nodes)}
                x=torch.as_tensor(packed[nodes],device=a.device,dtype=next(model.parameters()).dtype);observed=torch.as_tensor(mask[nodes],device=a.device)
                latent=model.complete(x,observed);held=model.complete_state_hold(x,observed)
                left=torch.stack([latent[lookup[i]] for i,j in pairs]);right=torch.stack([latent[lookup[j]] for i,j in pairs])
                hold_left=torch.stack([held[lookup[i]] for i,j in pairs]);hold_right=torch.stack([held[lookup[j]] for i,j in pairs])
                predictions=dict(TFM=fusion(left,right)['states'].cpu().numpy(),TFM_SWAP=fusion(right,left)['states'].cpu().numpy(),
                    TFM_SELF_LEFT=fusion(left,left)['states'].cpu().numpy(),TFM_STATE_HOLD=fusion(hold_left,hold_right)['states'].cpu().numpy())
                cap=scene_motion_evidence(scene,motion_observation_policy='selected_cap128');all_motion=load(a.native_run/'private/native-motion/T_core'/f'scene-{sid}.npz')
                all_motion['states']=all_motion['states'].copy();all_motion['states'][...,2]*=1852/3600
                for name,m in [('CAP128_MEAN',cap),('CANONICAL_MEAN',all_motion)]:
                    pred=[]
                    for i,j in pairs:
                        pred.append(ordinary_fusion(m['states'][[i,j]],m['nearest_seconds'][[i,j]],m['support'][[i,j]],[[0,1]])['predictions']['mean_supported'][0])
                    predictions[name]=np.asarray(pred)
                np.savez_compressed(output/'private'/f'predictions-{sid}.npz',pairs=np.asarray(pairs),**predictions)
                ev=load(a.native_run/'private/native-evaluation/T_core'/f'scene-{sid}.npz');registry={str(t):i for i,t in enumerate(ev['target_ids'])}
                both=np.asarray([cap['support'][i]&cap['support'][j] for i,j in pairs]);any_support=np.asarray([cap['support'][i]|cap['support'][j] for i,j in pairs])
                truth=np.asarray([ev['target_states'][registry[str(identities[i])]] for i,j in pairs]);valid=np.asarray([ev['target_valid'][registry[str(identities[i])]] for i,j in pairs])
                for region,region_mask in [('all',np.ones(both.shape,bool)),('both_supported',both),('one_supported',any_support&~both),('zero_supported',~any_support)]:
                    for name,pred in predictions.items():
                        errors=attribute_errors(pred,truth)
                        for attr,columns in [('position',[0,1]),('speed',[2]),('heading',[3])]:
                            use=valid[...,columns].all(-1)&region_mask;key=(region,name,attr)
                            values.setdefault(key,[]).extend(errors[attr][use].tolist());counts[key]=counts.get(key,0)+int(use.sum())
                ld=(latent-held).square().mean(-1).cpu().numpy();node_support=cap['support'][nodes]
                rows.append(dict(scene_id=sid,pairs=len(pairs),unique_nodes=len(nodes),both_supported_queries=int(both.sum()),zero_supported_queries=int((~any_support).sum()),
                    latent_hold_mse=float(ld.mean()),latent_hold_supported_mse=float(ld[node_support].mean()) if node_support.any() else None,
                    latent_hold_unsupported_mse=float(ld[~node_support].mean()) if (~node_support).any() else None,
                    swap_position_displacement_mean_m=float(attribute_errors(predictions['TFM_SWAP'],predictions['TFM'])['position'].mean()),
                    self_position_displacement_mean_m=float(attribute_errors(predictions['TFM_SELF_LEFT'],predictions['TFM'])['position'].mean())))
                check(completed_scenes=len(rows),phase='T_probe')
        result={}
        for (region,name,attr),v in values.items():
            x=np.asarray(v);result.setdefault(region,{}).setdefault(name,{})[attr]=dict(count=len(x),mae=float(x.mean()) if len(x) else None,rmse=float(np.sqrt(np.mean(x*x))) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,p99=float(np.quantile(x,.99)) if len(x) else None)
        common.old.frozen.write(output/'summary.json',dict(status='COMPLETED',idea_id=IDEA,scope='fixed T8 first2 positive cross-source pairs, no V generalization or new method',regions=result,scenes=rows,
            limitations=['T8 selection not representative wholeT','swap/self/statehold can be off training distribution','no new fitting; cannot identify a unique physical failure cause']))
        check(status='COMPLETED',exit_code=0,phase='completed',finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
