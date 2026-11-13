"""162: same nonlinear source scorer, supervision objective x train difficulty."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation.objective_difficulty import ARMS,fit_factorial
from association.evaluation.local_source_evidence import predict_residual
from tools.analysis.temporal_source_pilot import physical,recover,add_harm,summarize,selection,BASE
from tools.analysis.source_coverage_pilot import role_diagnostic,NEW
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source
FROZEN='WS_T128_RESIDUAL_HARD';NAMES=tuple('WS_'+n for n in ARMS);METHODS=(BASE,NEW,FROZEN,*NAMES,'WS_SOURCE_EQUAL')

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')};out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-OBJECTIVE-DIFFICULTY-SEPARATION-20261031',plan_version='1.0',mode='temporal',methods=METHODS,started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,neural_head_fits=0,teacher_forward_calls=0,analytic_head_fits=0,test_opened=False,raw_reference_read=False,head_run=str(a.head_run),expression_run=str(a.expression_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        h=common.old.frozen.read(a.head_run/'run.json');prior=common.old.frozen.read(a.expression_run/'run.json')
        if h['status']!='COMPLETED' or prior['status']!='COMPLETED' or prior['optimizer_updates']!=512*(1+len({manifest['components'][s] for s in tids})):raise ValueError('completed155/159')
        ids=h['fit_training_scene_ids'];rows=[]
        if ids!=manifest['splits']['T_core'][:128]:raise ValueError('fixedT128')
        for sid in ids:
            if sid in tids:p=common.load_npz(paths['source_risk_run']/'private'/f'training-{sid}.npz')
            else:p={**common.load_npz(a.head_run/'private'/f'features-T_core-{sid}.npz'),**common.load_npz(a.head_run/'private'/f'labels-T_core-{sid}.npz')}
            rows.append(dict(component=str(manifest['components'][sid]),pack=p))
        check(training_scene_ids=tids,scene_ids=vids,fit_training_scene_ids=ids,phase='T_fit');models={};originals={};curves=[];population=[]
        for held in sorted({manifest['components'][s] for s in tids})+[None]:
            fold='full' if held is None else 'leave_'+held;old=torch.load(a.expression_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold'];originals[fold]=old
            fitted,account=fit_factorial([r['pack'] for r in rows if r['component']!=held],old,device=a.device);models[fold]=fitted;state['optimizer_updates']+=4*128;state['neural_head_fits']+=4
            torch.save(fitted,out/'private'/f'model-{fold}.pt');population.append(dict(fold=fold,held_component=held,training_scenes=sum(r['component']!=held for r in rows),**account))
            curves.extend([dict(fold=fold,arm=name,**{k:v for k,v in model.items() if k in ('updates','parameter_count','initial_loss','final_loss','curve')}) for name,model in fitted.items()]);check(last_fold=fold)
        common.old.frozen.write(out/'training-curves.json.gz',curves);common.old.frozen.write(out/'T-fold-population.json',population);check(full_T_models_frozen_before_V=True)
        for split,ids,filename in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            records=[];arrays=[]
            for sid in ids:
                m,g,mean,pack,membership=physical(paths,split,sid);fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full';old=common.load_npz(a.expression_run/'private'/f'policy-{split}-{sid}.npz');scores={n:old[n+'__scores'] for n in (BASE,NEW,FROZEN,'WS_SOURCE_EQUAL')}
                for arm,name in zip(ARMS,NAMES):scores[name]=predict_residual(models[fold][arm],pack['features'].reshape(-1,10),a.device).reshape(pack['available'].shape)
                raw={n:recover(m,g,pack,scores[n],mean) for n in METHODS};previous=common.load_npz(a.expression_run/'private'/f'raw-{split}-{sid}.npz')
                for name in (BASE,NEW,FROZEN,'WS_SOURCE_EQUAL'):common.old._same(raw[name],previous[name],'159 frozen '+name)
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**{n+'__scores':v for n,v in scores.items()},membership=membership)
                ev=common.evaluation(paths['native_run'],split,sid,m);row,arr,art=common.evaluate_predictions(m,g,pack,ev,raw,{},method_names=METHODS,split=split,prediction_speed_unit='kn');add_harm(row,arr);labels=source.training_labels(pack,g,ev)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),selection={n:selection(pack,labels,s,scores[FROZEN]) for n,s in scores.items()},role_diagnostic={n:role_diagnostic(pack,labels,s) for n,s in scores.items()})
                pure=np.array([len(set(ev['identities'][group].tolist()))==1 for group in g]);row['selection_group_strata']={}
                for tag,use in [('pure',pure),('mixed',~pure)]:
                    label=dict(labels,labeled=labels['labeled']&use[pack['source_group_indices'],None]);row['selection_group_strata'][tag]={n:selection(pack,label,s,scores[FROZEN]) for n,s in scores.items()}
                common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);records.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(records)})
            result=summarize(records,arrays,METHODS)
            for right,left in [(n,FROZEN) for n in NAMES]+[(NAMES[1],NAMES[0]),(NAMES[2],NAMES[0]),(NAMES[3],NAMES[1]),(NAMES[3],NAMES[2])]:
                key=right+'_vs_'+left;result['primary_bootstrap'][key]=common.component_bootstrap(records,arrays,key)
            result.update(status='COMPLETED',idea_id=state['idea_id'],selection={n:{k:sum(r['selection'][n][k] for r in records) for k in records[0]['selection'][n]} for n in METHODS});common.old.frozen.write(out/filename,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),records)
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','head-run','expression-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
