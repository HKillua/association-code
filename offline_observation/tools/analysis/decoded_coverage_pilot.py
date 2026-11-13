"""176: shared-grid objective and coverage ablation on frozen 175 scores."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.analysis import association_context_pilot as p
from tools.analysis.decoded_coverage_calibration import coverage,binary,aggregate,choose
THRESHOLD=p.partition.THRESHOLD
PREFIXES=('WS','HARD','STRUCT','HR','HD','HC','SR','SD','SC')
METHODS=tuple(k+'_'+f for k in PREFIXES for f in ('MEAN','NN'))
PAIRS=(('HARD','HR'),('HR','HD'),('HD','HC'),('STRUCT','SR'),('SR','SD'),('SD','SC'),
       ('HARD','HC'),('STRUCT','HC'),('WS','HC'),('HARD','SC'),('STRUCT','SC'),('WS','SC'))
PRIMARY=tuple(b+'_NN_vs_'+a+'_NN' for a,b in PAIRS)


def run(a):
    parent=p.common.old.frozen.read(a.parent_run/'run.json')
    if parent['status']!='COMPLETED' or parent['optimizer_updates']!=1024:raise ValueError('completed175 required')
    paths={k:Path(parent[k]) for k in ('native_run','value_run','partition_run','source_risk_run')};out=a.output.resolve()
    out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-DECODED-COVERAGE-OPERATING-POINT-176',plan_version='1.0',mode='decoded_coverage',methods=METHODS,
               parent_run=str(a.parent_run),expression_run=parent['expression_run'],optimizer_updates=0,teacher_training_calls=0,
               reference_used_for_prediction=False,test_opened=False,started_at=datetime.now(timezone.utc).isoformat(),**{k:str(v) for k,v in paths.items()})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=p.common.old.frozen.read(a.manifest)
        tids,vids=p.common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        if tids!=parent['training_scene_ids'] or vids!=parent['scene_ids']:raise ValueError('original175 population required')
        check(training_scene_ids=tids,scene_ids=vids,phase='T_calibration')
        oldsum=p.common.old.frozen.read(a.parent_run/'T-summary.json')['all_components'];limits={k:oldsum['association']['WS_'+k]['mstsa']['paper_metrics']['m_pfa'] for k in ('raw','decoded')}
        training=[]
        for sid in tids:
            q=p.common.load_npz(a.parent_run/'private'/f'partitions-T_core-{sid}.npz')
            m=p.common.load_npz(paths['native_run']/'private/native-motion/T_core'/f'scene-{sid}.npz',('sources',))
            ev=p.common.load_npz(paths['native_run']/'private/native-evaluation/T_core'/f'scene-{sid}.npz');pairs=q['pairs']
            training.append(dict(sid=sid,q=q,ev=ev,cross=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]],truth=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]]))
        curves={};chosen={};floors={};grids={};matrices={c['sid']:{} for c in training}
        for head,letter in (('HARD','H'),('STRUCT','S')):
            own=parent['thresholds'][head];grid=np.unique(np.r_[np.quantile(np.concatenate([c['q'][head+'_uncalibrated_scores'][c['cross']] for c in training]),np.linspace(0,1,33)),THRESHOLD,own]);grids[head]=grid.tolist();curve=[]
            expected=oldsum['methods'][head+'_NN']['position'];floor=dict(points=expected['intersection_count'],targets=expected['covered_target_trajectory_count']);floors[head]=floor
            for i,theta in enumerate(grid):
                rows=[]
                for c in training:
                    q=c['q'];ev=c['ev'];pairs=q['pairs'];raw=q[head+'_uncalibrated_scores'];scores=raw+THRESHOLD-theta
                    member=p.partition.agglomerate(len(ev['identities']),pairs,scores,'WS')[0]
                    matrices[c['sid']].setdefault(head,[]).append(member)
                    if theta==own:p.common.old._same(member,q[head],'original175 T partition')
                    row=dict(scene_id=c['sid'],raw=binary(c['truth'][c['cross']],(scores>=THRESHOLD)[c['cross']]),
                             decoded=binary(c['truth'][c['cross']],(member[pairs[:,0]]==member[pairs[:,1]])[c['cross']]),
                             coverage=coverage(member,ev['identities'],ev['target_ids'],ev['target_valid']))
                    rows.append(row)
                curve.append(dict(threshold=float(theta),metrics=aggregate(rows),scenes=rows));check(calibrating_head=head,candidates_completed=i+1)
            anchor=next(r for r in curve if r['threshold']==own)
            if anchor['metrics']['coverage']!=floor:raise ValueError('T predicted coverage not bound to175')
            if choose([anchor],limits,floor,'C') is None:raise ValueError('original anchor should be feasible')
            curves[head]=curve
            for policy in ('R','D','C'):chosen[letter+policy]=choose(curve,limits,floor,policy)
        p.common.old.frozen.write(out/'T-calibration.json',dict(limits=limits,floors=floors,grids=grids,curves=curves,chosen=chosen,T_scenes=tids,scope='T only, same raw and decoded PFA budgets; max objective then lower decoded PFA then higher threshold'))
        for sid,v in matrices.items():np.savez_compressed(out/'private'/f'calibration-T_core-{sid}.npz',**{k:np.asarray(x,np.int32) for k,x in v.items()})
        if any(x is None for x in chosen.values()):check(status='COMPLETED',exit_code=0,phase='T_NOT_QUALIFIED',completed_V_scenes=0);return
        thresholds={k:parent['thresholds'][k] for k in ('WS','HARD','STRUCT')};thresholds.update({k:v['threshold'] for k,v in chosen.items()})
        source={k:k for k in ('WS','HARD','STRUCT')};source.update({k:'HARD' if k[0]=='H' else 'STRUCT' for k in chosen})
        check(thresholds=thresholds,calibration_frozen_before_V=True,full_T_models_frozen_before_V=True)
        del training,matrices
        for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m,_,_,_,_=p.physical(paths,split,sid);old=p.common.load_npz(a.parent_run/'private'/f'partitions-{split}-{sid}.npz');pairs=old['pairs']
                rawscores={k:old[source[k]+'_uncalibrated_scores'] for k in PREFIXES};scores={k:v+THRESHOLD-thresholds[k] for k,v in rawscores.items()}
                memberships={k:p.partition.agglomerate(len(m['states']),pairs,v,'WS')[0] for k,v in scores.items()}
                for k in ('WS','HARD','STRUCT'):
                    p.common.old._same(memberships[k],old[k],'frozen175 membership');p.common.old._same(scores[k],old[k+'_scores'],'frozen175 scores')
                fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full';model=torch.load(Path(parent['expression_run'])/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold']
                groups,packs,raw=p.predictions(m,memberships,model,a.device);oldraw=p.common.load_npz(a.parent_run/'private'/f'raw-{split}-{sid}.npz')
                for k in ('WS','HARD','STRUCT'):
                    for suffix in ('MEAN','NN'):p.common.old._same(raw[k+'_'+suffix],oldraw[k+'_'+suffix],'frozen175 physical output')
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',pairs=pairs,scores=scores['WS'],**memberships,**{k+'_scores':v for k,v in scores.items()},**{k+'_uncalibrated_scores':v for k,v in rawscores.items()})
                ev=p.common.evaluation(paths['native_run'],split,sid,m);row,arr,art=p.evaluate(m,ev,groups,packs,raw,split,method_names=METHODS)
                rec=dict(scene_id=sid,component_id=str(manifest['components'][sid]),node_ids=m['node_ids'],sources=m['sources'],pairs=pairs,labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]],joint_support_queries=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).sum(1));association={}
                for k in PREFIXES:
                    association[k+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=scores[k])],THRESHOLD)
                    co=memberships[k][pairs[:,0]]==memberships[k][pairs[:,1]];association[k+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=association);p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art)
                p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr)
                check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id']);p.common.old.frozen.write(out/fn,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0)
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    q=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','manifest','output'):q.add_argument('--'+k,type=Path,required=True)
    q.add_argument('--device',required=True);q.add_argument('--time-policy',choices=('advisory',),required=True);run(q.parse_args())
