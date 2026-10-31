"""178: one shared history-risk model, matched supported/all-history actions."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import history_candidates as h
from association.evaluation.native_temporal import fit_quadratic
from association.evaluation.local_source_evidence import fit_residual,predict_residual
from association.models.association_context import EdgeResidual
from tools.analysis.structured_association_pilot import predict
from tools.analysis import association_context_pilot as p
from tools.analysis.association_operating_point_pilot import THRESHOLD

FRONTS=('HARD','STRUCT')
ARMS={'HARD':('HARD','HM','HE','HZ','H1','HO'),'STRUCT':('STRUCT','SM','SE','SZ','S1','SO')}
METHODS=tuple(k+'_'+f for front in FRONTS for k in ARMS[front] for f in ('MEAN','NN'))
PRIMARY=tuple(o+'_NN_vs_'+b+'_NN' for front in FRONTS for o in [ARMS[front][-1]] for b in ARMS[front][:-1])+('SO_NN_vs_HARD_NN',)


def train_models(training,components,out,device,check):
    curves=[];models={};folds=[]
    for held in components+[None]:
        fold='full' if held is None else 'leave_'+held;rows=[r for r in training if r['component_id']!=held]
        x=np.concatenate([r['x'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);w=np.concatenate([r['w'] for r in rows])
        if len(y)==0 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('finite nonempty T training required')
        base=fit_quadratic(x,y,w);model=fit_residual(x,y,w,base,device=device)
        if len(base['beta'])!=91 or model['parameter_count']!=1505 or model['updates']!=256:raise ValueError('fixed architecture/updates')
        torch.save(model,out/'private'/f'model-{fold}.pt');models[fold]=model
        folds.append(dict(fold=fold,held_component=held,training_scene_ids=list(dict.fromkeys(r['scene_id'] for r in rows)),training_views=len(rows),labels=len(y),weight_sum=float(w.sum())))
        curves.append(dict(fold=fold,**{k:model[k] for k in ('parameter_count','updates','initial_mse','final_mse','curve')}))
        check(phase='T_fit',last_fold=fold,optimizer_updates=256*len(models),analytic_head_fits=len(models),neural_head_fits=len(models))
    p.common.old.frozen.write(out/'training-curves.json.gz',curves);p.common.old.frozen.write(out/'training-folds.json',folds)
    return models


def run(a):
    parent=p.common.old.frozen.read(a.parent_run/'run.json')
    if parent['status']!='COMPLETED' or parent['optimizer_updates']!=1024:raise ValueError('completed175 required')
    for k in ('native_run','value_run','partition_run','source_risk_run','front_run','data_run'):setattr(a,k,Path(parent[k]))
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')}
    out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-HISTORY-CANDIDATE-ACTIONS-178',plan_version='1.0',mode='history_candidates',methods=METHODS,
        parent_run=str(a.parent_run),front_run=str(a.front_run),data_run=str(a.data_run),feature_names=h.FEATURE_NAMES,
        optimizer_updates=0,analytic_head_fits=0,neural_head_fits=0,teacher_forward_calls=0,frontend_optimizer_updates=0,
        test_opened=False,reference_used_for_prediction=False,started_at=datetime.now(timezone.utc).isoformat(),
        **{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=p.common.old.frozen.read(paths['manifest'])
        tids,vids=p.common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        trainids=manifest['splits']['T_core'][:128]
        if trainids[:32]!=tids or trainids!=parent['fit_training_scene_ids']:raise ValueError('fixed T128/T32 binding')
        index=p.common.old.frozen.read(a.data_run/'private/index.json')['T_core']
        saved=torch.load(a.parent_run/'private/heads.pt',map_location='cpu',weights_only=False)
        original=torch.load(Path(parent['parent_run'])/'private/heads.pt',map_location='cpu',weights_only=False)
        heads={}
        for key,weights in [('HARD',saved['heads']['HARD']),('STRUCT',original['heads']['STRUCT'])]:
            head=EdgeResidual().to(a.device);head.load_state_dict(weights);head.eval();head.requires_grad_(False);heads[key]=head
        check(training_scene_ids=tids,scene_ids=vids,fit_training_scene_ids=trainids,thresholds={k:parent['thresholds'][k] for k in FRONTS})
        training=[];population=[]
        for sid in trainids:
            native=paths['native_run']/'private'
            m=p.common.load_npz(native/'native-motion/T_core'/f'scene-{sid}.npz') if sid in tids else p.common.load_npz(Path(index[sid]['motion']))
            ev=p.common.load_npz(native/'native-evaluation/T_core'/f'scene-{sid}.npz') if sid in tids else p.common.load_npz(Path(index[sid]['evaluation']))
            cache=p.common.load_npz(a.front_run/'private'/f'front-cache-T_core-{sid}.npz')
            p.common.old._same(m['node_ids'],cache['node_ids'],'T feature nodes');p.common.old._same(m['node_ids'],ev['node_ids'],'T label nodes')
            rawscores=predict(cache,heads,saved['normalizer'],a.device);arrays={}
            for front in FRONTS:
                scores=rawscores[front]+THRESHOLD-parent['thresholds'][front]
                membership=p.partition.agglomerate(len(m['states']),cache['pairs'],scores,'WS')[0]
                if sid in tids:
                    old=p.common.load_npz(a.parent_run/'private'/f'partitions-T_core-{sid}.npz')
                    p.common.old._same(scores,old[front+'_scores'],'frozen175 scores');p.common.old._same(membership,old[front],'frozen175 groups')
                groups=p.common.old.groups_from_membership(membership,len(m['states']));x,count=h.descriptors(m,groups);z=h.labels(m,groups,count,ev);use=z['labeled']
                training.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),x=x[use],y=z['y'][use],w=.5*z['w'][use]))
                population.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),front=front,nodes=len(m['states']),groups=len(groups),weighted_group_queries=float(.5*z['w'].sum()),**z['audit']))
                arrays.update({front+'__'+k:v for k,v in dict(features=x,count=count,membership=membership,scores=scores,**z).items() if k!='audit'})
            np.savez_compressed(out/'private'/f'training-{sid}.npz',**arrays);check(phase='T_prepare',prepared_T_scenes=len(population)//2)
        p.common.old.frozen.write(out/'training-population.json',population)
        components=sorted({str(manifest['components'][s]) for s in tids})
        if len(components)!=12:raise ValueError('T32 twelve components required')
        models=train_models(training,components,out,a.device,check);del training,heads
        check(full_T_models_frozen_before_V=True)
        for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m=p.common.load_npz(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz')
                old=p.common.load_npz(a.parent_run/'private'/f'partitions-{split}-{sid}.npz');oldraw=p.common.load_npz(a.parent_run/'private'/f'raw-{split}-{sid}.npz')
                model=models['leave_'+str(manifest['components'][sid]) if split=='T_core' else 'full'];raw={};parts=dict(pairs=old['pairs'],scores=old['scores']);groups={};policies={}
                for front in FRONTS:
                    g=p.common.old.groups_from_membership(old[front],len(m['states']));x,count=h.descriptors(m,g)
                    scores=predict_residual(model,x.reshape(-1,12),a.device).reshape(m['support'].shape)
                    outputs,weights=h.readouts(m,g,count,scores,oldraw[front+'_NN'])
                    for prefix,key in zip(ARMS[front],(None,'M','E','Z','1','O')):
                        groups[prefix]=g;parts[prefix]=old[front];parts[prefix+'_scores']=old[front+'_scores']
                        raw[prefix+'_MEAN']=oldraw[front+'_MEAN'];raw[prefix+'_NN']=oldraw[front+'_NN'] if key is None else outputs[key]
                        if key is not None:policies[prefix+'__weights']=weights[key]
                    policies.update({front+'__features':x,front+'__count':count,front+'__scores':scores})
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',**parts)
                np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**policies)
                # Deployment outputs are frozen before reading this scene's evaluation reference.
                ev=p.common.evaluation(paths['native_run'],split,sid,m);row,arr,art=p.evaluate(m,ev,groups,{},raw,split,method_names=METHODS)
                pairs=old['pairs'];rec=dict(scene_id=sid,component_id=str(manifest['components'][sid]),node_ids=m['node_ids'],sources=m['sources'],pairs=pairs,
                    labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]],joint_support_queries=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).sum(1));relation={}
                for front in FRONTS:
                    rs=p.partition.frozen.evaluate_scenes([dict(rec,scores=old[front+'_scores'])],THRESHOLD)
                    co=old[front][pairs[:,0]]==old[front][pairs[:,1]];ds=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                    for prefix in ARMS[front]:relation[prefix+'_raw']=rs;relation[prefix+'_decoded']=ds
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=relation)
                p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row)
                rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id'])
            p.common.old.frozen.write(out/fn,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-run','manifest','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
