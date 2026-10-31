"""170: same T-only raw false-acceptance budget for three frozen scorers."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from tools.analysis import association_context_pilot as p

THRESHOLD=p.partition.THRESHOLD
METHODS=tuple(k+'_'+s for k in ('WS','WSCAL','OWNCAL','CTXCAL') for s in ('MEAN','NN'))
PRIMARY=('WSCAL_NN_vs_WS_NN','OWNCAL_NN_vs_WS_NN','CTXCAL_NN_vs_WS_NN','CTXCAL_NN_vs_WSCAL_NN','CTXCAL_NN_vs_OWNCAL_NN')

def operating_point(records,key,threshold):
    pca=[];pfa=[]
    for r in records:
        if r['split']!='T_core':raise ValueError('T-only calibration')
        y=r['labels'];scores=r[key];accepted=scores>=threshold;positive=int(y.sum());count=int(accepted.sum())
        if not positive or not count:return None
        tp=int((y&accepted).sum());pca.append(tp/positive);pfa.append(1-tp/count)
    return dict(m_pca=float(np.mean(pca)),m_pfa=float(np.mean(pfa)))

def calibrate(records,key,limit):
    # The grid and tie rule are fixed before any V predictions or labels.
    grid=np.unique(np.r_[THRESHOLD,np.quantile(np.concatenate([r[key] for r in records]),np.linspace(0,1,257))])
    curve=[];qualified=[]
    for threshold in grid:
        value=operating_point(records,key,float(threshold));row=dict(threshold=float(threshold),metrics=value);curve.append(row)
        if value is not None and value['m_pfa']<=limit+1e-12:qualified.append(row)
    if not qualified:return None,curve
    best=max(qualified,key=lambda r:(r['metrics']['m_pca'],-r['metrics']['m_pfa'],r['threshold']))
    return best,curve

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')}
    out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-ASSOCIATION-OPERATING-POINT-20261031',plan_version='1.0',mode='calibration',methods=METHODS,optimizer_updates=0,
        started_at=datetime.now(timezone.utc).isoformat(),test_opened=False,V_used_for_calibration=False,T_identity_used_for_calibration=True,
        expression_run=str(a.expression_run),front_run=str(a.front_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=p.common.old.frozen.read(paths['manifest']);tids,vids=p.common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        parent=p.common.old.frozen.read(a.front_run/'run.json')
        if parent['status']!='COMPLETED' or parent['optimizer_updates']!=2000 or parent['training_scene_ids']!=tids or parent['scene_ids']!=vids:raise ValueError('completed original168 required')
        check(training_scene_ids=tids,scene_ids=vids);records=[]
        for sid in tids:
            q=p.common.load_npz(a.front_run/'private'/f'partitions-T_core-{sid}.npz');ev=p.common.load_npz(paths['native_run']/'private/native-evaluation/T_core'/f'scene-{sid}.npz');m=p.common.load_npz(paths['native_run']/'private/native-motion/T_core'/f'scene-{sid}.npz',('sources',))
            pairs=q['pairs'];cross=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]]
            records.append(dict(split='T_core',labels=(ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]])[cross],**{k:q[k+'_scores'][cross] for k in ('WS','OWN','CTX')}))
        limit=operating_point(records,'WS',THRESHOLD)['m_pfa'];chosen={};curves={}
        for k in ('WS','OWN','CTX'):chosen[k],curves[k]=calibrate(records,k,limit)
        p.common.old.frozen.write(out/'T-calibration.json',dict(limit=limit,chosen=chosen,curves=curves,T_scenes=tids,rule='same raw MSTSA macro PFA budget, max raw macro PCA; tie lower PFA then higher threshold'))
        if any(v is None for v in chosen.values()):check(status='COMPLETED',exit_code=0,phase='T_NOT_QUALIFIED',completed_V_scenes=0);return
        # Independently rescan the stored grid before any V read; never select with V.
        for k,best in chosen.items():
            eligible=[r for r in curves[k] if r['metrics'] is not None and r['metrics']['m_pfa']<=limit+1e-12]
            assert best==max(eligible,key=lambda r:(r['metrics']['m_pca'],-r['metrics']['m_pfa'],r['threshold']))
        check(calibration_frozen_before_V=True,thresholds={k:v['threshold'] for k,v in chosen.items()})
        for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m,g,mean,pack,ws=p.physical(paths,split,sid);q=p.common.load_npz(a.front_run/'private'/f'partitions-{split}-{sid}.npz');pairs=q['pairs']
                scores={'WS':q['WS_scores']};scores.update({k+'CAL':q[k+'_scores']+(THRESHOLD-chosen[k]['threshold']) for k in chosen})
                memberships={k:p.partition.agglomerate(len(m['states']),pairs,v,'WS')[0] for k,v in scores.items()};p.common.old._same(memberships['WS'],ws,'old WS partition')
                fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full';model=torch.load(a.expression_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold']
                groups,packs,raw=p.predictions(m,memberships,model,a.device)
                p.common.old._same(raw['WS_NN'],p.common.load_npz(a.front_run/'private'/f'raw-{split}-{sid}.npz',('WS_NN',))['WS_NN'],'original168/159 baseline')
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',pairs=pairs,scores=scores['WS'],**memberships,**{k+'_scores':v for k,v in scores.items()})
                ev=p.common.evaluation(paths['native_run'],split,sid,m);row,arr,art=p.evaluate(m,ev,groups,packs,raw,split,method_names=METHODS)
                rec=dict(scene_id=sid,component_id=str(manifest['components'][sid]),node_ids=m['node_ids'],sources=m['sources'],pairs=pairs,labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]],joint_support_queries=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).sum(1));association={}
                for k in scores:
                    association[k+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=scores[k])],THRESHOLD)
                    co=memberships[k][pairs[:,0]]==memberships[k][pairs[:,1]];association[k+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=association)
                p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id']);p.common.old.frozen.write(out/fn,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0)
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','expression-run','front-run'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
