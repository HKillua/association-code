"""180: matched objective continuation, single-support positions only."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.training.cost_routing import fit_heads,KINDS
from association.evaluation.local_source_evidence import predict_residual
from association.evaluation import history_candidates as h
from tools.analysis import association_context_pilot as p

ARMS={'HARD':('HARD','HO','HC','HL','HT'),'STRUCT':('STRUCT','SO','SC','SL','ST')}
METHODS=tuple(k+'_'+f for arms in ARMS.values() for k in arms for f in ('MEAN','NN'))
PRIMARY=tuple(k+'_NN_vs_'+arms[1]+'_NN' for arms in ARMS.values() for k in arms[2:])+tuple(arms[j]+'_NN_vs_'+arms[i]+'_NN' for arms in ARMS.values() for i,j in [(2,3),(3,4),(2,4)])

def prepare(parent,record,manifest):
    bags=[];population=[]
    for sid in record['fit_training_scene_ids']:
        z=p.common.load_npz(parent/'private'/f'training-{sid}.npz')
        for front in ARMS:
            member=z[front+'__membership'];count=z[front+'__count'];labeled=z[front+'__labeled'];x=z[front+'__features'];y=z[front+'__y'];w=z[front+'__w'];nqueries=single=nodes_presented=0
            for gid,label in enumerate(sorted(set(member.tolist()))):
                nodes=np.flatnonzero(member==label);queries=np.flatnonzero((count[gid]==1)&labeled[nodes[0]])
                if len(nodes)==1:single+=len(queries);continue
                for j in queries:
                    bags.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),front=front,group_index=gid,query_index=int(j),x=x[nodes,j],y=y[nodes,j],w=w[nodes,j]))
                nqueries+=len(queries);nodes_presented+=len(queries)*len(nodes)
            population.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),front=front,decision_queries=nqueries,decision_node_rows=nodes_presented,single_candidate_queries_without_updates=single))
    if not bags:raise ValueError('nonempty fixed T decision bags required')
    return bags,population

def run(a):
    parent=p.common.old.frozen.read(a.parent_run/'run.json');qualification=p.common.old.frozen.read(a.qualification_run/'run.json')
    if parent['status']!='COMPLETED' or parent['optimizer_updates']!=3328 or qualification['status']!='COMPLETED' or qualification['completed_T_scenes']!=32:raise ValueError('completed178/179 required')
    out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-COST-ROUTING-180',plan_version='1.0',mode='cost_routing',methods=METHODS,parent_run=str(a.parent_run),qualification_run=str(a.qualification_run),
        optimizer_updates=0,neural_head_fits=0,analytic_head_fits=0,teacher_forward_calls=0,frontend_optimizer_updates=0,test_opened=False,reference_used_for_prediction=False,
        risk_residual_parameters_per_conditional_head=1505,retained_plus_new_risk_residual_parameters_per_method=3010,additional_risk_residual_parameters_vs178=1505,
        started_at=datetime.now(timezone.utc).isoformat(),**{k:parent[k] for k in ('native_run','manifest','training_scene_ids','scene_ids','fit_training_scene_ids')})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=p.common.old.frozen.read(Path(parent['manifest']));bags,population=prepare(a.parent_run,parent,manifest)
        p.common.old.frozen.write(out/'training-population.json',population);p.common.old.frozen.write(out/'private/training-bags.json.gz',[{k:b[k] for k in ('scene_id','component_id','front','group_index','query_index')} for b in bags])
        models={};folds=[];curves=[];components=sorted({str(manifest['components'][s]) for s in parent['training_scene_ids']})
        for held in components+[None]:
            fold='full' if held is None else 'leave_'+held;ids=np.array([i for i,b in enumerate(bags) if b['component_id']!=held]);rows=[bags[i] for i in ids]
            initial=torch.load(a.parent_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False);before=state['optimizer_updates']
            journal=(out/'training-journal.jsonl').open('a')
            def progress(step,entry):
                journal.write(__import__('json').dumps(dict(fold=fold,step=step,heads=entry))+'\n');journal.flush()
                if step==1 or step%16==0:check(phase='T_fit',last_fold=fold,optimizer_updates=before+3*step)
            try:heads,draws=fit_heads(rows,initial,a.device,check=progress)
            finally:journal.close()
            if any(m['updates']!=256 or m['parameter_count']!=1505 for m in heads.values()):raise ValueError('fixed architecture/update budget')
            torch.save(heads,out/'private'/f'model-{fold}.pt');np.savez_compressed(out/'private'/f'sampling-{fold}.npz',training_bag_indices=ids,local_draws=draws)
            models[fold]=heads;folds.append(dict(fold=fold,held_component=held,training_bags=len(rows),training_scene_ids=list(dict.fromkeys(b['scene_id'] for b in rows)),query_presentations_per_head=int(draws.size),updates_per_head=256))
            curves.extend(dict(fold=fold,kind=k,curve=v['curve']) for k,v in heads.items());check(neural_head_fits=3*len(models),optimizer_updates=3*256*len(models))
        p.common.old.frozen.write(out/'training-folds.json',folds);p.common.old.frozen.write(out/'training-curves.json.gz',curves);check(full_T_models_frozen_before_V=True)
        for split,ids,fn in [('T_core',parent['training_scene_ids'],'T-summary.json'),('V_select',parent['scene_ids'],'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m=p.common.load_npz(Path(parent['native_run'])/'private/native-motion'/split/f'scene-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-{split}-{sid}.npz');old=p.common.load_npz(a.parent_run/'private'/f'raw-{split}-{sid}.npz');policy=p.common.load_npz(a.parent_run/'private'/f'policy-{split}-{sid}.npz');heads=models['leave_'+str(manifest['components'][sid]) if split=='T_core' else 'full'];raw={};parts=dict(pairs=q['pairs'],scores=q['scores']);saved={};groups={}
                for front,arms in ARMS.items():
                    g=p.common.old.groups_from_membership(q[front],len(m['states']));count=policy[front+'__count'];x=policy[front+'__features']
                    for prefix in arms:
                        groups[prefix]=g;parts[prefix]=q[front];parts[prefix+'_scores']=q[front+'_scores'];raw[prefix+'_MEAN']=old[front+'_MEAN']
                    raw[arms[0]+'_NN']=old[arms[0]+'_NN'];raw[arms[1]+'_NN']=old[arms[1]+'_NN']
                    for prefix,kind in zip(arms[2:],KINDS):
                        scores=predict_residual(heads[kind],x.reshape(-1,12),a.device).reshape(m['support'].shape);outputs,weights=h.readouts(m,g,count,scores,old[arms[1]+'_NN']);raw[prefix+'_NN']=outputs['1'];saved[prefix+'__scores']=scores;saved[prefix+'__weights']=weights['1']
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',**parts);np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**saved)
                ev=p.common.evaluation(Path(parent['native_run']),split,sid,m);row,arr,art=p.evaluate(m,ev,groups,{},raw,split,method_names=METHODS)
                original=p.common.old.frozen.read(a.parent_run/'scenes'/f'{split}-{sid}.json.gz');relation={}
                for front,arms in ARMS.items():
                    for prefix in arms:
                        for kind in ('raw','decoded'):relation[prefix+'_'+kind]=original['association'][front+'_'+kind]
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=relation);p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id']);p.common.old.frozen.write(out/fn,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','qualification-run','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
