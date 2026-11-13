"""190: T-only endpoint-state x velocity factorial with geometry control."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.native_motion_interventions import fit_endpoint_models,fitted_endpoint_controls
from tools.analysis import association_context_pilot as p
from tools.analysis.drift_memory_pilot import ARMS as OLD_ARMS
ARMS={front:old[:6]+(short+'GL',short+'MC',short+'FL',short+'FM') for (front,old),short in zip(OLD_ARMS.items(),('H','S'))}
METHODS=tuple(k+'_'+s for values in ARMS.values() for k in values for s in ('MEAN','NN'))
PRIMARY=tuple(values[b]+'_NN_vs_'+values[a]+'_NN' for values in ARMS.values() for a,b in [(1,6),(6,7),(6,8),(7,9),(8,9),(1,9),(1,8),(1,7),(5,9)])

def run(a):
    old=p.common.old.frozen;parent=old.read(a.parent_run/'run.json');ablation=old.read(a.ablation_run/'run.json');assert parent['status']==ablation['status']=='COMPLETED' and ablation['parent_run']==str(a.parent_run) and ablation['training_scene_ids']==parent['training_scene_ids'] and len(parent['training_scene_ids'])==32 and ablation['evaluation_splits']==['T_core']
    out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic();manifest=old.read(Path(parent['manifest']))
    record=dict(status='RUNNING',idea_id='IDEA-FITTED-ENDPOINT-FACTORIAL-190',plan_version='1.0',mode='fitted_endpoint_factorial',parent_run=str(a.parent_run),ablation_run=str(a.ablation_run),native_run=parent['native_run'],manifest=parent['manifest'],methods=METHODS,training_scene_ids=parent['training_scene_ids'],scene_ids=[],evaluation_splits=['T_core'],V_scenes=0,optimizer_updates=0,neural_head_fits=0,analytic_head_fits=0,native_likelihood_fits=0,native_OLS_fits=0,native_mean_computations=0,teacher_forward_calls=0,reference_used_for_prediction=False,test_opened=False,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        record.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=old.peak_rss_bytes());old.write(out/'run.json',record)
        if record['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        rows=[];arrays=[];bindings=[];diagnostic_rows=[]
        for sid in parent['training_scene_ids']:
            n=Path(parent['native_run'])/'private';m=p.common.load_npz(n/'native-motion/T_core'/f'scene-{sid}.npz');events=p.common.load_npz(n/'native-events/T_core'/f'scene-{sid}.npz');ev=p.common.load_npz(n/'native-evaluation/T_core'/f'scene-{sid}.npz');q=p.common.load_npz(a.parent_run/'private'/f'partitions-T_core-{sid}.npz');op=p.common.load_npz(a.parent_run/'private'/f'policy-T_core-{sid}.npz');raw=p.common.load_npz(a.ablation_run/'private'/f'raw-T_core-{sid}.npz');p.common.old._same(m['node_ids'],events['node_ids'],'native nodes');p.common.old._same(m['query_seconds'],events['query_seconds'],'native query')
            models,binding=fit_endpoint_models(m,events);bindings.append(dict(scene_id=sid,**binding));record['native_OLS_fits']+=binding['native_OLS_fits'];record['native_mean_computations']+=binding['native_mean_computations'];np.savez_compressed(out/'private'/f'fitted-models-T_core-{sid}.npz',**{key:np.stack([v[key] for v in models]) for key in ('times','native','fitted','velocity','mu','report_known','mean_known')})
            parts=dict(pairs=q['pairs'],scores=q['scores']);groups={};policy={};diags={}
            for front,values in ARMS.items():
                g=p.common.old.groups_from_membership(q[front],len(m['states']));short='H' if front=='HARD' else 'S';outputs,flags,diag=fitted_endpoint_controls(m,models,g,op[front+'__count'],op[short+'O__weights'],raw[short+'MC_NN'])
                for key in ('GL','FL','FM'):raw[short+key+'_NN']=outputs[key];raw[short+key+'_MEAN']=raw[front+'_MEAN']
                for prefix in values:groups[prefix]=g;parts[prefix]=q[front];parts[prefix+'_scores']=q[front+'_scores']
                policy[front+'__eligible']=flags['eligible'];policy[front+'__joint']=flags['joint'];diags[front]=diag
            diagnostic_rows.append(dict(scene_id=sid,heads=diags));np.savez_compressed(out/'private'/f'raw-T_core-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'partitions-T_core-{sid}.npz',**parts);np.savez_compressed(out/'private'/f'policy-T_core-{sid}.npz',**policy);old.write(out/'private'/f'diagnostics-T_core-{sid}.json.gz',diags)
            row,arr,artifact=p.evaluate(m,ev,groups,{},raw,'T_core',method_names=METHODS);previous=old.read(a.ablation_run/'scenes'/f'T_core-{sid}.json.gz');relation={}
            for front,values in ARMS.items():
                for prefix in values:
                    for kind in ('raw','decoded'):relation[prefix+'_'+kind]=previous['association'][front+'_'+kind]
            row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=relation);p.common.save_artifacts(out/'private'/f'states-T_core-{sid}.npz',artifact);old.write(out/'scenes'/f'T_core-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(completed_T_scenes=len(rows))
        result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=record['idea_id']);old.write(out/'T-summary.json',result);old.write(out/'T-scene-summary.json.gz',rows);old.write(out/'endpoint-input-accounting.json',dict(status='COMPLETED',bindings=bindings,scene_diagnostics=diagnostic_rows,scope='native-only full T code/cache binding, working fits and joint domain counts; not physical truth'))
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:check(status='FAILED',exit_code=1,error=f'{type(error).__name__}: {error}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('parent-run','ablation-run','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
