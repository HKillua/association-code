"""193: matched T-only task gate with mean, orderless blocks, ordered blocks."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import native_phase_gate as phase
from association.training import temporal_gate as gate
from association.evaluation.local_source_evidence import predict_residual
from tools.analysis import association_context_pilot as p
from tools.analysis.drift_memory_pilot import ARMS as OLD_ARMS

ARMS={f:old+(s+'HALF',s+'GM',s+'GB',s+'GT') for f,s,old in [('HARD','H',OLD_ARMS['HARD']),('STRUCT','S',OLD_ARMS['STRUCT'])]}
METHODS=tuple(k+'_'+s for v in ARMS.values() for k in v for s in ('MEAN','NN'))
PRIMARY=tuple(v[b]+'_NN_vs_'+v[a]+'_NN' for v in ARMS.values() for a,b in [(9,10),(8,10),(6,10),(1,10),(7,10),(6,8),(6,9),(1,5)])

def prepare(a,parent,out,manifest,check):
    prior=p.common.old.frozen.read(a.native_training_run/'run.json');assert prior['status']=='COMPLETED' and prior['parent_run']==str(a.parent_run);index=p.common.old.frozen.read(Path(parent['data_run'])/'private/index.json')['T_core'];components=sorted({str(manifest['components'][s]) for s in parent['training_scene_ids']});assert len(components)==12;folds=[('leave_'+c,c) for c in components]+[('full',None)];records={f:[] for f,_ in folds};models={f:torch.load(a.parent_run/'private'/f'model-{f}.pt',map_location='cpu',weights_only=False) for f,_ in folds};population=[];means=calls=0
    for i,sid in enumerate(parent['fit_training_scene_ids']):
        n=Path(parent['native_run'])/'private';known=sid in parent['training_scene_ids'];m=p.common.load_npz(n/'native-motion/T_core'/f'scene-{sid}.npz') if known else p.common.load_npz(Path(index[sid]['motion']));events=p.common.load_npz(n/'native-events/T_core'/f'scene-{sid}.npz') if known else p.common.load_npz(a.native_training_run/'private'/f'events-T_core-{sid}.npz');z=p.common.load_npz(a.parent_run/'private'/f'training-{sid}.npz');p.common.old._same(m['node_ids'],events['node_ids'],'native nodes');p.common.old._same(np.diff(events['event_offsets']),m['native_count'],'native counts');p.common.old._same(m['query_seconds'],events['query_seconds'],'native queries');native,node_mc,nmeans=phase.node_context(m,events);means+=nmeans
        ev=p.common.load_npz(n/'native-evaluation/T_core'/f'scene-{sid}.npz') if known else p.common.load_npz(Path(index[sid]['evaluation']));p.common.old._same(m['node_ids'],ev['node_ids'],'label nodes');component=str(manifest['components'][sid]);scene_pop={}
        for front in ARMS:
            groups=p.common.old.groups_from_membership(z[front+'__membership'],len(m['states']));count=z[front+'__count'];scene_pop[front]={}
            for fold,held in folds:
                if component==held:continue
                score=predict_residual(models[fold],z[front+'__features'].reshape(-1,12),a.device).reshape(m['support'].shape);calls+=1;case=phase.cases(m,native,node_mc,groups,count,score);ix,target=phase.labels(case,groups,ev);scene_pop[front][fold]=len(ix)
                if not len(ix):continue
                row=dict(old=case['old'][ix],mc=case['mc'][ix],target=target,w=np.full(len(ix),.5),scene_ids=np.repeat(sid,len(ix)),component_ids=np.repeat(component,len(ix)),fronts=np.repeat(front,len(ix)),indices=case['indices'][ix]);row.update({'x_'+k:case['x'][k][ix] for k in phase.KINDS});records[fold].append(row)
        population.append(dict(scene_id=sid,component_id=component,nodes=len(m['states']),native_mean_computations=nmeans,rows=scene_pop));check(phase='T_prepare',prepared_T_scenes=i+1,native_mean_computations=means,frozen_risk_forward_calls=calls)
    p.common.old.frozen.write(out/'training-population.json',population);results={};foldmeta=[];curves=[]
    for fold,held in folds:
        data={key:np.concatenate([v[key] for v in records[fold]]) for key in records[fold][0]};assert len(data['w']) and (data['component_ids']!=held).all();np.savez_compressed(out/'private'/f'training-{fold}.npz',**data);done=len(results)*384
        models,draws,initial=gate.fit(data,a.device,progress=lambda step:check(phase='T_fit',last_fold=fold,optimizer_updates=done+3*step) if step%16==0 else None);torch.save(models,out/'private'/f'model-{fold}.pt');torch.save(initial,out/'private'/f'initial-{fold}.pt');np.savez_compressed(out/'private'/f'sampling-{fold}.npz',draws=draws);results[fold]=models;foldmeta.append(dict(fold=fold,held_component=held,training_scene_ids=list(dict.fromkeys(data['scene_ids'].tolist())),rows=len(data['w']),weight_sum=float(data['w'].sum()),updates_per_head=128,frozen_risk_fold=fold));curves.extend(dict(fold=fold,kind=k,curve=v['curve']) for k,v in models.items());check(neural_head_fits=3*len(results),optimizer_updates=384*len(results))
    p.common.old.frozen.write(out/'training-folds.json',foldmeta);p.common.old.frozen.write(out/'training-curves.json.gz',curves);return results

def run(a):
    old=p.common.old.frozen;parent=old.read(a.parent_run/'run.json');previous=old.read(a.control_run/'run.json');assert parent['status']==previous['status']=='COMPLETED' and previous['parent_run']==str(a.parent_run) and previous['evaluation_splits']==['T_core'];out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-TEMPORAL-GATE-193',plan_version='1.0',mode='temporal_gate',parent_run=str(a.parent_run),control_run=str(a.control_run),native_training_run=str(a.native_training_run),native_run=parent['native_run'],manifest=parent['manifest'],training_scene_ids=parent['training_scene_ids'],fit_training_scene_ids=parent['fit_training_scene_ids'],scene_ids=[],evaluation_splits=['T_core'],V_scenes=0,methods=METHODS,optimizer_updates=0,neural_head_fits=0,analytic_head_fits=0,native_likelihood_fits=0,teacher_forward_calls=0,frontend_optimizer_updates=0,frozen_risk_forward_calls=0,native_mean_computations=0,reference_used_for_prediction=False,test_opened=False,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kwargs):
        state.update(kwargs,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=old.peak_rss_bytes(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated());old.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3 or state['cuda_peak_allocated_bytes']>1024**3:raise MemoryError('RSS8GiB/CUDA1GiB')
    try:
        torch.set_num_threads(4);manifest=old.read(Path(parent['manifest']));assert len(parent['fit_training_scene_ids'])==128 and len(parent['training_scene_ids'])==32;models=prepare(a,parent,out,manifest,check);check(all_models_frozen_before_evaluation=True,additional_gate_parameters=2145,full_training_models=39)
        rows=[];arrays=[]
        for sid in parent['training_scene_ids']:
            n=Path(parent['native_run'])/'private';m=p.common.load_npz(n/'native-motion/T_core'/f'scene-{sid}.npz');events=p.common.load_npz(n/'native-events/T_core'/f'scene-{sid}.npz');op=p.common.load_npz(a.parent_run/'private'/f'policy-T_core-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-T_core-{sid}.npz');raw=p.common.load_npz(a.control_run/'private'/f'raw-T_core-{sid}.npz');native,node_mc,nmeans=phase.node_context(m,events);state['native_mean_computations']+=nmeans;parts=dict(pairs=q['pairs'],scores=q['scores']);groups={};policy={};fold='leave_'+str(manifest['components'][sid])
            for front,arms in ARMS.items():
                g=p.common.old.groups_from_membership(q[front],len(m['states']));case=phase.cases(m,native,node_mc,g,op[front+'__count'],op[front+'__scores']);gid,j=case['indices'].T;np.testing.assert_allclose(case['old'],raw[arms[1]+'_NN'][gid,j,:2],rtol=0,atol=1e-10);np.testing.assert_allclose(case['mc'],raw[arms[6]+'_NN'][gid,j,:2],rtol=0,atol=1e-10);policy[front+'__eligible']=case['eligible'];policy[front+'__indices']=case['indices']
                for prefix,kind in zip(arms[7:],('HALF',)+phase.KINDS):
                    alpha=np.full(len(gid),.5) if kind=='HALF' else gate.predict(models[fold][kind],case['x'][kind],a.device);candidate=raw[arms[1]+'_NN'].copy();candidate[gid,j,:2]=phase.blend(case['old'],case['mc'],alpha);raw[prefix+'_NN']=candidate;raw[prefix+'_MEAN']=raw[front+'_MEAN'];policy[prefix+'__alpha']=alpha
                    if kind!='HALF':policy[prefix+'__features']=case['x'][kind]
                for prefix in arms:groups[prefix]=g;parts[prefix]=q[front];parts[prefix+'_scores']=q[front+'_scores']
            np.savez_compressed(out/'private'/f'raw-T_core-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'partitions-T_core-{sid}.npz',**parts);np.savez_compressed(out/'private'/f'policy-T_core-{sid}.npz',**policy);ev=p.common.evaluation(Path(parent['native_run']),'T_core',sid,m);row,arr,artifact=p.evaluate(m,ev,groups,{},raw,'T_core',method_names=METHODS);original=old.read(a.control_run/'scenes'/f'T_core-{sid}.json.gz');relation={}
            for front,arms in ARMS.items():
                for prefix in arms:
                    for kind in ('raw','decoded'):relation[prefix+'_'+kind]=original['association'][front+'_'+kind]
            row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=relation);p.common.save_artifacts(out/'private'/f'states-T_core-{sid}.npz',artifact);old.write(out/'scenes'/f'T_core-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(phase='T_core',completed_T_scenes=len(rows))
        result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id']);old.write(out/'T-summary.json',result);old.write(out/'T-scene-summary.json.gz',rows);check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-run','control-run','native-training-run','output','device'):parser.add_argument('--'+key,required=True,type=str if key=='device' else Path)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
