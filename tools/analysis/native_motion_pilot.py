"""183/184/185 independent fixed-scope native-motion interventions."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import native_compatibility as nc, native_motion_interventions as ni, history_candidates as h
from association.evaluation.local_source_evidence import predict_residual
from association.training import native_motion_correction as correction
from tools.analysis import association_context_pilot as p

def design(kind):
    extra={'feature':('F0','F1'),'bridge':('B','E','CL','CH','CM'),'persistence':('PH','PC','PD','PG')}[kind]
    arms={front:(front,short+'O')+tuple(short+k for k in extra) for front,short in [('HARD','H'),('STRUCT','S')]};methods=tuple(k+'_'+f for values in arms.values() for k in values for f in ('MEAN','NN'))
    primary=[]
    for values in arms.values():
        primary += [v+'_NN_vs_'+values[1]+'_NN' for v in values[2:]]
        pairs={'feature':[(2,3)],'bridge':[(2,4),(2,5),(2,6),(4,5),(5,6),(4,6),(3,6)],'persistence':[(3,4),(3,5),(2,4),(2,5),(4,5)]}[kind]
        primary += [values[b]+'_NN_vs_'+values[a]+'_NN' for a,b in pairs]
    return arms,methods,tuple(primary)

def training(a,parent,out,manifest,check):
    from association.data.archive import MTADArchive
    from association.data.native import _record
    from association.data.source_events import source_events
    from tools.analysis.native_cv_state_control import pack_native_events
    dr=p.common.old.frozen.read(Path(parent['data_run'])/'run.json');index=p.common.old.frozen.read(Path(parent['data_run'])/'private/index.json')['T_core'];records=[];population=[]
    with MTADArchive(dr['archive']) as archive:
        for i,sid in enumerate(parent['fit_training_scene_ids']):
            z=p.common.load_npz(a.parent_run/'private'/f'training-{sid}.npz')
            if sid in parent['training_scene_ids']:
                m=p.common.load_npz(Path(parent['native_run'])/'private/native-motion/T_core'/f'scene-{sid}.npz');events=p.common.load_npz(Path(parent['native_run'])/'private/native-events/T_core'/f'scene-{sid}.npz')
            else:
                m=p.common.load_npz(Path(index[sid]['motion']));native=_record(archive.load_scene(sid),manifest['components'][sid],'T_core',128,split_unit=manifest.get('split_unit'));events=pack_native_events(dict(native,grid=m['query_seconds']),source_events(native));np.savez_compressed(out/'private'/f'events-T_core-{sid}.npz',**events);archive._label_cache.pop(sid,None)
            p.common.old._same(events['node_ids'],m['node_ids'],'T native nodes');p.common.old._same(np.diff(events['event_offsets']),m['native_count'],'T native counts');p.common.old._same(events['query_seconds'],m['query_seconds'],'T native query');features={}
            for front in ('HARD','STRUCT'):
                groups=[np.flatnonzero(z[front+'__membership']==g) for g in sorted(set(z[front+'__membership'].tolist()))];extra,_=nc.descriptors(m,events,groups,speed_unit='kn');count=z[front+'__count'];use=np.zeros(m['support'].shape,bool)
                for gid,nodes in enumerate(groups):
                    if len(nodes)>1:use[nodes]=z[front+'__labeled'][nodes]&(count[gid]==1)
                x=np.concatenate([z[front+'__features'],extra],axis=-1);features[front]=extra;records.append(dict(scene_id=sid,component_id=str(manifest['components'][sid]),x=x[use],y=z[front+'__y'][use],w=.5*z[front+'__w'][use]));population.append(dict(scene_id=sid,front=front,rows=int(use.sum()),weight=float(.5*z[front+'__w'][use].sum())))
            np.savez_compressed(out/'private'/f'training-native-{sid}.npz',**features);check(phase='T_prepare',prepared_T_scenes=i+1)
    p.common.old.frozen.write(out/'training-population.json',population);models={};folds=[];curves=[];components=sorted({str(manifest['components'][sid]) for sid in parent['training_scene_ids']})
    for held in components+[None]:
        fold='full' if held is None else 'leave_'+held;rows=[v for v in records if v['component_id']!=held];x=np.concatenate([v['x'] for v in rows]);y=np.concatenate([v['y'] for v in rows]);w=np.concatenate([v['w'] for v in rows]);old=torch.load(a.parent_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False);oldscore=predict_residual(old,x[:,:12],a.device);before=len(models)*512
        heads,draws=correction.fit(x,y,w,oldscore,a.device,progress=lambda step:check(phase='T_fit',last_fold=fold,optimizer_updates=before+2*step) if step%16==0 else None);torch.save(heads,out/'private'/f'model-{fold}.pt');np.savez_compressed(out/'private'/f'sampling-{fold}.npz',draws=draws);models[fold]=heads
        folds.append(dict(fold=fold,held_component=held,training_scene_ids=list(dict.fromkeys(v['scene_id'] for v in rows)),rows=len(y),weight=float(w.sum()),updates_per_head=256));curves.extend(dict(fold=fold,kind=k,curve=v['curve']) for k,v in heads.items());check(neural_head_fits=2*len(models),optimizer_updates=512*len(models))
    p.common.old.frozen.write(out/'training-folds.json',folds);p.common.old.frozen.write(out/'training-curves.json.gz',curves);return models

def run(a):
    parent=p.common.old.frozen.read(a.parent_run/'run.json');qualification=p.common.old.frozen.read(a.qualification_run/'run.json');assert parent['status']==qualification['status']=='COMPLETED';out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic();arms,methods,primary=design(a.kind);number={'feature':183,'bridge':184,'persistence':185}[a.kind]
    state=dict(status='RUNNING',idea_id={183:'IDEA-NATIVE-MOTION-FEATURES-183',184:'IDEA-ENDPOINT-KINEMATICS-184',185:'IDEA-ONE-SIDED-PERSISTENCE-185'}[number],plan_version='1.0',mode='native_motion_'+a.kind,kind=a.kind,methods=methods,parent_run=str(a.parent_run),qualification_run=str(a.qualification_run),bridge_run=str(a.bridge_run),optimizer_updates=0,neural_head_fits=0,analytic_head_fits=0,native_likelihood_fits=0,teacher_forward_calls=0,frontend_optimizer_updates=0,test_opened=False,reference_used_for_prediction=False,started_at=datetime.now(timezone.utc).isoformat(),**{k:parent[k] for k in ('native_run','manifest','training_scene_ids','scene_ids','fit_training_scene_ids')})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=p.common.old.frozen.read(Path(parent['manifest']));models=training(a,parent,out,manifest,check) if a.kind=='feature' else None
        check(all_rules_models_frozen_before_V=True,additional_risk_parameters=1921 if models else 0,risk_residual_parameters=3426 if models else 1505)
        for split,ids,filename in [('T_core',parent['training_scene_ids'],'T-summary.json'),('V_select',parent['scene_ids'],'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                n=Path(parent['native_run'])/'private';m=p.common.load_npz(n/'native-motion'/split/f'scene-{sid}.npz');events=p.common.load_npz(n/'native-events'/split/f'scene-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-{split}-{sid}.npz');old=p.common.load_npz(a.parent_run/'private'/f'raw-{split}-{sid}.npz');policy=p.common.load_npz(a.parent_run/'private'/f'policy-{split}-{sid}.npz');raw={};parts=dict(pairs=q['pairs'],scores=q['scores']);saved={};groups={};diagnostics={}
                braw=p.common.load_npz(a.bridge_run/'private'/f'raw-{split}-{sid}.npz') if a.kind=='bridge' else None
                for front,values in arms.items():
                    g=p.common.old.groups_from_membership(q[front],len(m['states']));count=policy[front+'__count'];base=old[values[1]+'_NN']
                    for prefix in values:groups[prefix]=g;parts[prefix]=q[front];parts[prefix+'_scores']=q[front+'_scores'];raw[prefix+'_MEAN']=old[front+'_MEAN']
                    raw[front+'_NN']=old[front+'_NN'];raw[values[1]+'_NN']=base
                    if a.kind=='feature':
                        extra,_=nc.descriptors(m,events,g,speed_unit='kn');x=np.concatenate([policy[front+'__features'],extra],axis=-1);fold='leave_'+str(manifest['components'][sid]) if split=='T_core' else 'full';original=policy[front+'__scores'];saved[front+'__features']=x
                        for prefix,kind in zip(values[2:],('OLDINFO','NATIVE')):
                            score=original+correction.predict(models[fold][kind],x.reshape(-1,25),a.device).reshape(original.shape);outputs,w=h.readouts(m,g,count,score,base);raw[prefix+'_NN']=outputs['1'];saved[prefix+'__scores']=score;saved[prefix+'__weights']=w['1']
                        flags=dict(eligible=count==1);diag={}
                    elif a.kind=='bridge':
                        short='H' if front=='HARD' else 'S';raw[values[2]+'_NN']=braw[short+'B_NN'];raw[values[3]+'_NN']=braw[short+'E_NN'];outputs,flags,diag=ni.bridge(m,events,g,count,policy[front+'__scores'],base,raw[values[2]+'_NN'])
                        for prefix,key in zip(values[4:],('CL','CH','CM')):raw[prefix+'_NN']=outputs[key]
                    else:
                        outputs,flags,diag=ni.persistence(m,events,g,count,policy[values[1]+'__weights'],base)
                        for prefix,key in zip(values[2:],('PH','PC','PD','PG')):raw[prefix+'_NN']=outputs[key]
                        state['native_likelihood_fits']+=diag['native_likelihood_fits']
                    saved[front+'__eligible']=flags['eligible'];diagnostics[front]=diag
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',**parts);np.savez_compressed(out/'private'/f'policy-{split}-{sid}.npz',**saved);p.common.old.frozen.write(out/'private'/f'diagnostics-{split}-{sid}.json.gz',diagnostics)
                ev=p.common.evaluation(Path(parent['native_run']),split,sid,m);row,arr,artifact=p.evaluate(m,ev,groups,{},raw,split,method_names=methods);original=p.common.old.frozen.read(a.parent_run/'scenes'/f'{split}-{sid}.json.gz');relation={}
                for front,values in arms.items():
                    for prefix in values:
                        for kind in ('raw','decoded'):relation[prefix+'_'+kind]=original['association'][front+'_'+kind]
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=relation);p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',artifact);p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,methods,primary);result.update(status='COMPLETED',idea_id=state['idea_id']);p.common.old.frozen.write(out/filename,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.kind=='feature' else 0)
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--kind',choices=('feature','bridge','persistence'),required=True)
    for k in ('parent-run','qualification-run','bridge-run','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
