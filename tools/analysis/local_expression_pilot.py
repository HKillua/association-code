"""159: fixed T128, query-local encoding x quadratic/residual neural scoring."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation.local_source_evidence import local_features,fit_residual,predict_residual
from association.evaluation.native_temporal import fit_quadratic,predict
from tools.analysis.temporal_source_pilot import physical,old_scores,recover,add_harm,summarize,selection,BASE
from tools.analysis.source_coverage_pilot import role_diagnostic,NEW
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source

METHODS=(BASE,NEW,'WS_LOCAL_Q_HARD','WS_T128_RESIDUAL_HARD','WS_LOCAL_RESIDUAL_HARD','WS_SOURCE_EQUAL')

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')};out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-LOCAL-EVIDENCE-EXPRESSION-20261031',plan_version='1.0',mode='temporal',methods=METHODS,started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,analytic_head_fits=0,neural_head_fits=0,teacher_forward_calls=0,test_opened=False,raw_reference_read=False,head_run=str(a.head_run),data_run=str(a.data_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        oldrun=common.old.frozen.read(a.head_run/'run.json')
        if oldrun['status']!='COMPLETED' or oldrun['analytic_head_fits']!=13:raise ValueError('complete155 required')
        ids=oldrun['fit_training_scene_ids'];index=common.old.frozen.read(a.data_run/'private/index.json')['T_core'];oldmodels=common.load_npz(a.head_run/'private/models.npz');training=[]
        if ids!=manifest['splits']['T_core'][:128]:raise ValueError('fixed128')
        check(training_scene_ids=tids,scene_ids=vids,fit_training_scene_ids=ids)
        for sid in ids:
            if sid in tids:
                m,g,mean,pack,membership=physical(paths,'T_core',sid);z=common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz');unit='kn'
            else:
                m=common.load_npz(Path(index[sid]['motion']));pack=common.load_npz(a.head_run/'private'/f'features-T_core-{sid}.npz');z=common.load_npz(a.head_run/'private'/f'labels-T_core-{sid}.npz');unit='mps'
            extra=local_features(m,pack,speed_unit=unit);x=np.concatenate((pack['features'],extra),axis=-1);use=z['labeled']
            np.savez_compressed(out/'private'/f'features-T_core-{sid}.npz',features=x)
            training.append(dict(component=str(manifest['components'][sid]),x=x[use],y=z['y'][use],w=z['sample_weight'][use]));check(prepared_T_scenes=len(training),phase='T_prepare')
        models={};curves=[];components=sorted({manifest['components'][s] for s in tids})
        for held in components+[None]:
            fold='full' if held is None else 'leave_'+held;rows=[r for r in training if r['component']!=held];x=np.concatenate([r['x'] for r in rows]);y=np.concatenate([r['y'] for r in rows]);w=np.concatenate([r['w'] for r in rows]);old={k:oldmodels[fold+'__'+k] for k in ('mean','std','beta')}
            quad=fit_quadratic(x,y,w);state['analytic_head_fits']+=1;netold=fit_residual(x[:,:10],y,w,old,device=a.device);netnew=fit_residual(x,y,w,quad,device=a.device)
            state['optimizer_updates']+=512;state['neural_head_fits']+=2;models[fold]=dict(old=old,quad=quad,netold=netold,netnew=netnew)
            curves.extend([dict(fold=fold,kind=k,training_labels=len(y),**{z:v for z,v in model.items() if z in ('parameter_count','updates','initial_mse','final_mse','curve')}) for k,model in [('old',netold),('local',netnew)]])
            torch.save(models[fold],out/'private'/f'model-{fold}.pt');check(phase='T_fit',last_fold=fold)
        common.old.frozen.write(out/'training-curves.json.gz',curves);check(full_T_models_frozen_before_V=True)
        for split,sceneids,filename in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in sceneids:
                m,g,mean,pack,membership=physical(paths,split,sid);x=np.concatenate((pack['features'],local_features(m,pack,speed_unit='kn')),axis=-1);flat=x.reshape(-1,17);model=models['leave_'+manifest['components'][sid] if split=='T_core' else 'full'];original=old_scores(paths,split,sid)
                scores={BASE:original,NEW:predict(model['old'],flat[:,:10]).reshape(pack['available'].shape),METHODS[2]:predict(model['quad'],flat).reshape(pack['available'].shape),METHODS[3]:predict_residual(model['netold'],flat[:,:10],a.device).reshape(pack['available'].shape),METHODS[4]:predict_residual(model['netnew'],flat,a.device).reshape(pack['available'].shape),METHODS[5]:np.zeros(pack['available'].shape)}
                raw={n:recover(m,g,pack,s,mean) for n,s in scores.items()};prior=common.load_npz(a.head_run/'private'/f'raw-{split}-{sid}.npz');common.old._same(raw[NEW],prior[NEW],'155 exact T128 output')
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**{n+'__scores':v for n,v in scores.items()},membership=membership,features=x)
                ev=common.evaluation(paths['native_run'],split,sid,m);row,arr,art=common.evaluate_predictions(m,g,pack,ev,raw,{},method_names=METHODS,split=split,prediction_speed_unit='kn');add_harm(row,arr);labels=source.training_labels(pack,g,ev)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),selection={n:selection(pack,labels,s,scores[NEW]) for n,s in scores.items()},role_diagnostic={n:role_diagnostic(pack,labels,s) for n,s in scores.items()})
                pure=np.array([len(set(ev['identities'][group].tolist()))==1 for group in g]);strata={}
                for name,group_use in [('pure',pure),('mixed',~pure)]:
                    masked=dict(labels);masked['labeled']=labels['labeled']&group_use[pack['source_group_indices'],None]
                    strata[name]={n:selection(pack,masked,s,scores[NEW]) for n,s in scores.items()}
                row['selection_group_strata']=strata;common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=summarize(rows,arrays,METHODS)
            for right,left in [(METHODS[2],NEW),(METHODS[3],NEW),(METHODS[4],METHODS[2]),(METHODS[4],METHODS[3])]:
                key=right+'_vs_'+left;result['primary_bootstrap'][key]=common.component_bootstrap(rows,arrays,key)
            result.update(status='COMPLETED',idea_id=state['idea_id'],selection={n:{k:sum(r['selection'][n][k] for r in rows) for k in rows[0]['selection'][n]} for n in METHODS});common.old.frozen.write(out/filename,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','head-run','data-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
