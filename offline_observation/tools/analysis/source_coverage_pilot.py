"""155: fixed T128 coverage, same10 features/66-coefficient source head."""
from __future__ import annotations
import argparse,sys,time,resource
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import source_risk as risk
from association.evaluation.native_temporal import fit_quadratic,predict
from association.training.evidence_association import load_checkpoint
from tools.analysis.conditional_source_value import teacher_scores
from tools.analysis import static_partition_pilot as partition
from tools.analysis.temporal_source_pilot import physical,old_scores,recover,add_harm,summarize,selection,BASE
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

NEW='WS_T128_Q_HARD';EQUAL='WS_SOURCE_EQUAL';METHODS=(BASE,NEW,EQUAL)

def fit_coverage(rows,held):
    chosen=[r for r in rows if r['component_id']!=held]
    if not chosen:raise ValueError('held component leaves no training')
    return fit_quadratic(np.concatenate([r['x'] for r in chosen]),np.concatenate([r['y'] for r in chosen]),np.concatenate([r['weight'] for r in chosen]))

def role_diagnostic(pack,labels,scores):
    weights=risk.source_weights(scores,pack['available'],pack['group_offsets'],kind='hard');out={k:dict(points=0,wrong=0,excess_sum_m=0.,excess_over_1km=0) for k in ('slot0_best','slot0_not_best')}
    for a,b in zip(pack['group_offsets'][:-1],pack['group_offsets'][1:]):
        use=labels['labeled'][a:b];query=use.any(0);error=np.where(use,labels['errors_m'][a:b],np.inf);best=error.min(0)
        selected=np.sum(weights[a:b]*np.where(use,labels['errors_m'][a:b],0),axis=0);delta=np.where(query,selected-np.where(query,best,0),0)
        notbest=use[0]&(error[0]>best+1e-6)
        for name,mask in [('slot0_best',query&~notbest),('slot0_not_best',query&notbest)]:
            out[name]['points']+=int(mask.sum());out[name]['wrong']+=int((mask&(delta>1e-6)).sum());out[name]['excess_sum_m']+=float(delta[mask].sum());out[name]['excess_over_1km']+=int((mask&(delta>1000)).sum())
    return out

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')};out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-SOURCE-T-COVERAGE-20261030',plan_version='1.0',mode='temporal',methods=METHODS,
        started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,analytic_head_fits=0,teacher_forward_calls=0,test_opened=False,raw_reference_read=False,
        data_run=str(a.data_run),association_checkpoint=str(a.association_checkpoint),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        if 'teacher_forward_calls_increment' in kw:state['teacher_forward_calls']+=kw.pop('teacher_forward_calls_increment')
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('8GiB RSS')
    try:
        if sys.platform.startswith('linux'):
            soft,hard=resource.getrlimit(resource.RLIMIT_AS);limit=min([14*1024**3]+[x for x in (soft,hard) if x!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        torch.set_num_threads(4);manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        prepared=common.old.frozen.read(a.data_run/'run.json')
        if prepared['status']!='COMPLETED' or prepared['completed_T_scenes']!=2100:raise ValueError('complete151 required')
        index=common.old.frozen.read(a.data_run/'private/index.json')['T_core'];expanded=manifest['splits']['T_core'][:128]
        if expanded[:32]!=tids:raise ValueError('fixed prefix nesting')
        teacher,checkpoint=load_checkpoint(a.association_checkpoint,torch.device('cpu'))
        if checkpoint['config']['motion_observation_policy']!='canonical_all':raise ValueError('same old056 observation policy')
        check(training_scene_ids=tids,scene_ids=vids,fit_training_scene_ids=expanded);training=[];train_summary=[]
        for sid in expanded:
            if sid in tids:
                saved=common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz');pack={k:saved[k] for k in ('features','source_states','available','group_offsets')};labels={k:saved[k] for k in ('labeled','y','sample_weight','errors_m')}
            else:
                entry=index[sid];m=common.load_npz(Path(entry['motion']))
                if str(m['state_speed_unit'])!='mps':raise ValueError('151 explicit mps')
                legacy=dict(m,states=m['states'].copy());legacy['states'][...,2]/=1852/3600
                pairs=np.column_stack(np.triu_indices(len(m['states']),1));scores,calls=teacher_scores(teacher,checkpoint['normalizer'],legacy,pairs,check)
                membership,_,_=partition.agglomerate(len(m['states']),pairs,scores,'WS',check);groups=common.old.groups_from_membership(membership,len(m['states']))
                pack=risk.build_source_features(**{k:m[k] for k in source.MOTION_KEYS},groups=groups)
                np.savez_compressed(out/'private'/f'features-T_core-{sid}.npz',**{k:v for k,v in pack.items() if isinstance(v,np.ndarray)},membership=membership,source_state_speed_unit=np.asarray('mps'))
                ev=common.load_npz(Path(entry['evaluation']));common.old._same(m['node_ids'],ev['node_ids'],'newT nodes');labels=source.training_labels(pack,groups,ev)
                np.savez_compressed(out/'private'/f'labels-T_core-{sid}.npz',**{k:v for k,v in labels.items() if isinstance(v,np.ndarray)})
            use=labels['labeled'];training.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),x=pack['features'][use],y=labels['y'][use],weight=labels['sample_weight'][use]))
            roles=role_diagnostic(pack,labels,np.zeros(pack['available'].shape));train_summary.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),labeled_source_query_count=int(use.sum()),training_weight_sum=float(labels['sample_weight'].sum()),role_population={k:v['points'] for k,v in roles.items()},reused133=sid in tids));check(prepared_T_scenes=len(training),phase='T_prepare')
        del teacher;components=sorted({manifest['components'][sid] for sid in tids});models={};saved_models={};folds=[]
        for held in components+[None]:
            fold='full' if held is None else 'leave_'+held;models[fold]=fit_coverage(training,held);state['analytic_head_fits']+=1
            saved_models.update({fold+'__'+k:v for k,v in models[fold].items()});selected=[r for r in training if r['component_id']!=held]
            folds.append(dict(fold=fold,held_component=held,training_scenes=len(selected),labeled_source_queries=sum(len(r['y']) for r in selected)));check(phase='T_fit',fold=fold)
        np.savez_compressed(out/'private/models.npz',**saved_models);common.old.frozen.write(out/'T-fit-population.json',train_summary);common.old.frozen.write(out/'T-fold-index.json',folds);check(full_T_models_frozen_before_V=True)
        for split,ids,filename in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m,g,mean,pack,membership=physical(paths,split,sid);original=old_scores(paths,split,sid);scores={BASE:original,NEW:predict(models['leave_'+manifest['components'][sid] if split=='T_core' else 'full'],pack['features'].reshape(-1,10)).reshape(pack['available'].shape),EQUAL:np.zeros(pack['available'].shape)}
                predictions={n:recover(m,g,pack,s,mean) for n,s in scores.items()};np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**predictions)
                np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**{n+'__scores':v for n,v in scores.items()},membership=membership)
                ev=common.evaluation(paths['native_run'],split,sid,m);row,arr,artifacts=common.evaluate_predictions(m,g,pack,ev,predictions,{},method_names=METHODS,split=split,prediction_speed_unit='kn');add_harm(row,arr);labels=source.training_labels(pack,g,ev)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),selection={n:selection(pack,labels,s,original) for n,s in scores.items()},role_diagnostic={n:role_diagnostic(pack,labels,s) for n,s in scores.items()})
                common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',artifacts);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)},phase=split)
            result=summarize(rows,arrays,METHODS);result.update(status='COMPLETED',idea_id=state['idea_id'],analytic_head_fits=state['analytic_head_fits'],selection={n:{k:sum(r['selection'][n][k] for r in rows) for k in rows[0]['selection'][n]} for n in METHODS})
            common.old.frozen.write(out/filename,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','data-run','association-checkpoint'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
