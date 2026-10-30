"""152: scratch full-population peer training and fixedWS V300 evaluation.

Fixed150/20 epochs, no V selection, no old weights, retry or automatic resume.
Full-data cache is explicitly m/s; PTFM core/missing and Mean fallback separate.
"""
from __future__ import annotations
import argparse,json,sys,time,resource
from collections import OrderedDict
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.models.ptfm import PTFM
from association.models import deepaf_peer as deep
from association.models.fusion_peer_inputs import sensor_segments,ptfm_segment_inputs,deepaf_visible_inputs,group_mean
from association.models.evidence_fusion import physical_errors
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.query_contribution import query_contribution
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.tmtaff import aggregate_fusion_reports
from tools.experiments.fusion_peer_training_probe import ptfm_loss
from tools.analysis import observation_common as common
from tools.analysis.temporal_source_pilot import add_harm,summarize

EPOCHS={'ptfm':150,'deepaf':20}
BATCH={'ptfm':32,'deepaf':16}
IDEA='IDEA-FUSION-PEER-FULL-TRAINING-20261030'
CONTROLS=('WS_MEAN','WS_ANCHOR','WS_NEAREST_SUPPORTED','CAP128_MEAN','CAP128_NEAREST_SUPPORTED','WS_SHARED_Q_HARD')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def row(motion,ev,group):
    members=np.asarray(group['members'],np.int64)
    segments=sensor_segments(motion,members)
    # The eligibility/index is fixed during preparation, not changed by fitting.
    keep={tuple(q) for q in group['segments']}
    segments=[s for s in segments if tuple(s['query_indices']) in keep]
    if len(segments)!=len(keep):raise ValueError('training support segments changed')
    target=group['target_index']
    return dict(motion=motion,members=members,segments=segments,target=ev['target_states'][target],valid=ev['target_valid'][target])

class Scenes:
    def __init__(self,index):self.index=index;self.cache=OrderedDict()
    def get(self,sid):
        if sid in self.cache:self.cache.move_to_end(sid);return self.cache[sid]
        entry=self.index[sid];motion=load(entry['motion']);ev=load(entry['evaluation'])
        if str(motion['state_speed_unit'])!='mps':raise ValueError('new unit cache required')
        common.old._same(motion['node_ids'],ev['node_ids'],'T node order')
        value=(motion,ev);self.cache[sid]=value
        if len(self.cache)>4:self.cache.popitem(last=False)
        return value

def normalizer(index,scenes,check):
    count=0;mean=np.zeros(4);m2=np.zeros(4)
    for sid,entry in index.items():
        motion,ev=scenes.get(sid)
        for group in entry['groups']:
            for segment in row(motion,ev,group)['segments']:
                a,b,_,_=ptfm_segment_inputs(segment);x=np.concatenate((a.numpy(),b.numpy()))
                n=len(x);mu=x.mean(0);delta=mu-mean;total=count+n
                m2+=np.sum((x-mu)**2,axis=0)+delta**2*count*n/total;mean+=delta*n/total;count=total
        check()
    if not count:raise ValueError('no training-only normalizer samples')
    return dict(mean=mean.tolist(),scale=np.maximum(np.sqrt(m2/count),1e-6).tolist(),split='T_core',truth_used=False,count=count,speed_unit='mps')

def train_batch(model,rows,method,optimizer,device):
    optimizer.zero_grad(set_to_none=True)
    valid_counts=np.zeros(4,np.int64)
    if method=='ptfm':
        total=0.
        for r in rows:
            parts=[dict(r,segment=s) for s in r['segments']]
            loss,_=ptfm_loss(model,parts,device)
            (loss/len(rows)).backward();total+=float(loss.detach())/len(rows)
            for s in r['segments']:valid_counts+=r['valid'][s['query_indices']].sum(0)
        regularizer=1e-4*model.l2();regularizer.backward();total+=float(regularizer.detach())
    else:
        inputs=deep.deepaf_batch_inputs([deepaf_visible_inputs(r['motion'],r['members'],device=device) for r in rows])
        rollout=model.forward_batch(inputs,cache=True)
        target=torch.as_tensor(np.stack([r['target'] for r in rows]),dtype=torch.float64,device=device)
        valid=torch.as_tensor(np.stack([r['valid'] for r in rows]),dtype=torch.bool,device=device)
        supervision=deep.deepaf_supervision(rollout,target,valid);total=0.
        for start in range(0,len(rows)*64,64):
            stop=min(start+64,len(rows)*64)
            loss,_=deep.deepaf_objectives(model.replay_batch(rollout,start,stop),supervision,start=start,stop=stop)
            loss['total'].backward();total+=float(loss['total'].detach())
        valid_counts[:2]=int(supervision['p_valid'].sum());valid_counts[2]=int(supervision['v_valid'].sum())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    if not np.isfinite(total):raise FloatingPointError('nonfinite training loss')
    optimizer.step()
    return dict(loss=total,gradient_norm=float(norm),valid_supervision_counts=valid_counts.tolist())

@torch.no_grad()
def predict_group(model,motion,members,method,device):
    mean=group_mean(motion,members)
    if method=='deepaf':
        prediction=model(deepaf_visible_inputs(motion,members,device=device),cache=False)['states'].cpu().numpy()
        return prediction,np.ones_like(prediction,bool),mean
    prediction=mean.copy();valid=np.zeros_like(prediction,bool)
    for s in sensor_segments(motion,members):
        q=s['query_indices'];result=model(*ptfm_segment_inputs(s,device=device))
        prediction[q]=result['states'].cpu().numpy();valid[q]=True
    return prediction,valid,mean

@torch.no_grad()
def training_probe(model,training,scenes,method,device):
    rng=np.random.default_rng(20);ids=list(training)
    sid=next(ids[i] for i in rng.permutation(len(ids)) if training[ids[i]]['groups'])
    chosen=rng.permutation(len(training[sid]['groups']))[:8]
    motion,ev=scenes.get(sid);errors=[];controls=[]
    model.eval()
    for i in chosen:
        group=training[sid]['groups'][i];r=row(motion,ev,group)
        prediction,_,mean=predict_group(model,motion,r['members'],method,device)
        target=torch.as_tensor(r['target'],dtype=torch.float64,device=device)
        valid=torch.as_tensor(r['valid'],dtype=torch.bool,device=device);safe=torch.where(valid,target,torch.zeros_like(target));use=valid[:,:2].all(-1)
        for array,destination in ((prediction,errors),(mean,controls)):
            e=physical_errors(torch.as_tensor(array,dtype=torch.float64,device=device),safe)[:,0][use]
            destination.extend(e.cpu().tolist())
    def metrics(x):
        x=np.asarray(x);return dict(points=len(x),mae_m=float(x.mean()),rmse_m=float(np.sqrt(np.mean(x*x))),p95_m=float(np.quantile(x,.95)),p99_m=float(np.quantile(x,.99)))
    return dict(scope='fixed first8 groups of planned first training batch; not validation selection',
                groups=len(chosen),position=metrics(errors),mean_control=metrics(controls))

@torch.no_grad()
def evaluate(model,a,index,output,check):
    method=a.method;candidate='WS_PTFM_PLUS_MEAN' if method=='ptfm' else 'WS_DEEPAF_PV'
    names=CONTROLS+(candidate,)+(('WS_PTFM_CORE',) if method=='ptfm' else ())
    rows=[];arrays=[];physical=[]
    model.eval()
    for sid,entry in index.items():
        motion=load(entry['motion'])
        if str(motion['state_speed_unit'])!='mps':raise ValueError('mps cache required')
        membership=np.asarray(common.old.frozen.read(a.partition_run/'private'/f'scene-{sid}.json.gz')['memberships']['WS'],np.int64)
        groups=common.old.groups_from_membership(membership,len(motion['states']));first=np.asarray([g[0] for g in groups],np.int64)
        mean=ordinary_fusion(motion['states'],motion['nearest_seconds'],motion['support'],groups)['predictions']['mean_supported'][first]
        nearest=query_contribution(motion['states'],motion['nearest_seconds'],motion['support'],motion['sources'],groups,method='nearest_supported')['group_states']
        cap_mean=ordinary_fusion(motion['cap128_states'],motion['cap128_nearest_seconds'],motion['cap128_support'],groups)['predictions']['mean_supported'][first]
        cap_near=query_contribution(motion['cap128_states'],motion['cap128_nearest_seconds'],motion['cap128_support'],motion['sources'],groups,method='nearest_supported')['group_states']
        with np.load(a.qhard_run/'private'/f'states-{sid}.npz',allow_pickle=False) as z:
            common.old._same(z['WS_SHARED_Q_HARD_node_group_indices'],membership,'147 WS membership')
            if str(z['WS_SHARED_Q_HARD_state_speed_unit'])!='mps':raise ValueError('147 canonical output required')
            hard=z['WS_SHARED_Q_HARD_group_states'].copy()
            common.old._same(mean[...,:2],z['WS_MEAN_group_states'][...,:2],'147 Mean positions')
        predicted=np.empty_like(mean);core_valid=np.empty_like(mean,bool)
        for gid,g in enumerate(groups):predicted[gid],core_valid[gid],_=predict_group(model,motion,g,method,a.device)
        raw=dict(WS_MEAN=mean,WS_ANCHOR=motion['states'][first],WS_NEAREST_SUPPORTED=nearest,
            CAP128_MEAN=cap_mean,CAP128_NEAREST_SUPPORTED=cap_near,WS_SHARED_Q_HARD=hard,**{candidate:predicted})
        if method=='ptfm':raw['WS_PTFM_CORE']=predicted
        np.savez_compressed(output/'private'/f'raw-V_select-{sid}.npz',**raw,core_valid=core_valid)
        # Prediction is fixed before the frozen V reference/identity is opened.
        ev=load(entry['evaluation'])
        outputs={n:emit_group_outputs(raw[n],groups,node_count=len(motion['states'])) for n in names}
        if method=='ptfm':
            for gid,value in enumerate(outputs['WS_PTFM_CORE']):value['prediction_valid']=core_valid[gid]
        reports,artifacts=evaluate_fusion_methods(outputs,prediction_speed_units={n:'mps' for n in names},
            node_identities=ev['identities'],target_ids=ev['target_ids'],target_states=ev['target_states'],target_valid=ev['target_valid'],
            support=motion['support'],nearest_seconds=motion['nearest_seconds'],target_speed_unit='mps')
        fragment=dict(identities=ev['identities'],query_seconds=motion['query_seconds'],support=motion['support'],nearest_seconds=motion['nearest_seconds'])
        regions=common.old.fixed_regions(fragment,ev['target_ids'])
        record=dict(scene_id=sid,component_id=entry['component_id'],methods={},pairwise={});arr=dict(methods={},pairs={})
        for n in names:record['methods'][n],arr['methods'][n]=common.qc.describe(reports[n],artifacts[n],fragment,regions)
        for n in names[len(CONTROLS):]:
            for base in CONTROLS:
                key=n+'_vs_'+base
                record['pairwise'][key],arr['pairs'][key],_=common.old.compare_pair(artifacts[base],artifacts[n],regions)
                record['pairwise'][key].update(baseline_method=base,candidate_method=n,side_labels=dict(A0=base,candidate=n))
        add_harm(record,arr)
        common.save_artifacts(output/'private'/f'states-V_select-{sid}.npz',artifacts)
        common.old.frozen.write(output/'scenes'/f'{sid}.json.gz',record)
        rows.append(record);arrays.append(arr);physical.append({n:{k:reports[n][k] for k in ('full_target_metrics','paper_fusion_metrics')} for n in names})
        check(completed_V_scenes=len(rows),phase='V_evaluation')
    result=common.summarize(rows,arrays,method_names=names,primary_pairs=[candidate+'_vs_WS_MEAN',candidate+'_vs_WS_SHARED_Q_HARD'])
    result['physical']={n:aggregate_fusion_reports([r[n] for r in physical]) for n in names}
    result.update(status='COMPLETED',idea_id=IDEA,method=method,training_epochs=EPOCHS[method],seed=20,
        limitations=['fixed sharedWS offline task adaptation, not original complete systems','single seed and developed/shared-identity V300','PTFM core and publicMean extension separated; DeepAF speed/course remain Mean'])
    common.old.frozen.write(output/'scene-summary.json.gz',rows);common.old.frozen.write(output/'summary.json',result)

def run(a):
    output=a.output.resolve();output.mkdir(exist_ok=False)
    for d in ('private','scenes'):(output/d).mkdir()
    started=time.monotonic();state=dict(status='RUNNING',idea_id=IDEA,plan_version='1.0',method=a.method,seed=20,
        started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,completed_epochs=0,completed_V_scenes=0,
        data_run=str(a.data_run.resolve()),partition_run=str(a.partition_run.resolve()),qhard_run=str(a.qhard_run.resolve()),
        state_speed_unit='mps',test_opened=False,validation_selected=False,old_weights_loaded=False,planned_epochs=EPOCHS[a.method],batch_groups=BATCH[a.method])
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-started,peak_rss_bytes=common.old.frozen.peak_rss_bytes())
        common.old.frozen.write(output/'run.json',state)
        if state['peak_rss_bytes']>12*1024**3:raise MemoryError('12GiB RSS budget')
    try:
        torch.set_num_threads(4);torch.manual_seed(20)
        ready=common.old.frozen.read(a.data_run/'run.json')
        if ready['status']!='COMPLETED' or ready['exit_code']!=0 or ready['completed_T_scenes']!=2100 or ready['completed_V_scenes']!=300:raise ValueError('full151 must be completed')
        index=common.old.frozen.read(a.data_run/'private/index.json');training=index['T_core'];scenes=Scenes(training)
        group_count=sum(len(v['groups']) for v in training.values());updates_per_epoch=sum((len(v['groups'])+BATCH[a.method]-1)//BATCH[a.method] for v in training.values())
        check(phase='normalizer',training_scenes=len(training),trainable_groups=group_count,planned_group_presentations=group_count*EPOCHS[a.method],planned_optimizer_updates=updates_per_epoch*EPOCHS[a.method])
        norm=normalizer(training,scenes,check) if a.method=='ptfm' else None
        model=(PTFM(norm,sensor_branches=True) if a.method=='ptfm' else deep.DeepAFPeer()).to(a.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(20)
        common.old.frozen.write(output/'T-probe-u0.json',training_probe(model,training,scenes,a.method,a.device))
        check(parameter_count=sum(p.numel() for p in model.parameters()),phase='training')
        sid_list=list(training);history=[]
        with (output/'training.jsonl').open('x') as journal:
            for epoch in range(1,EPOCHS[a.method]+1):
                model.train();count=updates=0;losses=[];supervision=np.zeros(4,np.int64);epoch_start=time.monotonic()
                for si in rng.permutation(len(sid_list)):
                    sid=sid_list[si];entry=training[sid];groups=entry['groups']
                    if not groups:continue
                    motion,ev=scenes.get(sid);order=rng.permutation(len(groups))
                    for start in range(0,len(order),BATCH[a.method]):
                        chosen=order[start:start+BATCH[a.method]];rows=[row(motion,ev,groups[i]) for i in chosen]
                        info=train_batch(model,rows,a.method,optimizer,a.device)
                        state['optimizer_updates']+=1;count+=len(rows);updates+=1;losses.append(info['loss']);supervision+=info['valid_supervision_counts']
                        journal.write(json.dumps(dict(epoch=epoch,update=state['optimizer_updates'],scene_id=sid,group_indices=chosen.tolist(),groups=len(rows),**info))+'\n');journal.flush()
                        if state['optimizer_updates']%25==0:check()
                if count!=group_count or updates!=updates_per_epoch:raise ValueError('incomplete epoch exposure')
                history.append(dict(epoch=epoch,groups=count,updates=updates,loss_mean=float(np.mean(losses)),valid_supervision_counts=supervision.tolist(),elapsed_seconds=time.monotonic()-epoch_start))
                common.old.frozen.write(output/'epoch-summary.json',history)
                torch.save(dict(scope='fusion-peer-full-new-unit-v1',method=a.method,model=model.state_dict(),normalizer=norm,
                    optimizer=optimizer.state_dict(),epoch=epoch,updates=state['optimizer_updates'],state_speed_unit='mps'),output/'private/last-epoch.pt')
                check(completed_epochs=epoch,training_group_presentations=count*epoch)
        check(training_completed=True,phase='V_evaluation')
        common.old.frozen.write(output/'T-probe-final.json',training_probe(model,training,scenes,a.method,a.device))
        evaluate(model,a,index['V_select'],output,check)
        check(status='COMPLETED',exit_code=0,phase='completed',finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=tuple(EPOCHS),required=True)
    for k in ('data-run','partition-run','qhard-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
