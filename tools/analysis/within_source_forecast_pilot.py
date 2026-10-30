"""156: forecast selection only within a source; fixed anonymous nearest source."""
from __future__ import annotations
import argparse,sys,time,json,resource
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.native_forecast_selection import native_forecasts,KINDS
from association.evaluation.native_temporal import unsupported_candidates
from tools.analysis.temporal_source_pilot import physical,old_scores,recover,add_harm,summarize,BASE
from tools.analysis import observation_common as common

METHODS=(BASE,'WS_ZERO_NEAREST_GLOBAL','WS_ZERO_NEAREST_HOLD','WS_ZERO_NEAREST_LOCAL8','WS_ZERO_NEAREST_CV','WS_ZERO_NEAREST_UNIFORM')

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')};out=paths['output'];out.mkdir(exist_ok=False)
    (out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-WITHIN-SOURCE-FORECAST-20261030',plan_version='1.0',mode='zero',started_at=datetime.now(timezone.utc).isoformat(),
        methods=METHODS,analytic_head_fits=0,optimizer_updates=0,test_opened=False,raw_reference_read=False,**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>6*1024**3:raise MemoryError('6GiB RSS')
    try:
        if sys.platform.startswith('linux'):
            soft,hard=resource.getrlimit(resource.RLIMIT_AS);limit=min([12*1024**3]+[x for x in (soft,hard) if x!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest);check(training_scene_ids=tids,scene_ids=vids)
        for split,ids,filename in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m,g,mean,pack,membership=physical(paths,split,sid);base=recover(m,g,pack,old_scores(paths,split,sid),mean);cache=common.load_npz(paths['native_run']/'private/native-events'/split/f'scene-{sid}.npz')
                for key in ('node_ids','sources','query_seconds'):common.old._same(cache[key],m[key],'native binding '+key)
                forecasts,audit=native_forecasts(cache,m['states']);predictions={BASE:base}
                for k,name in zip(KINDS,METHODS[1:]):
                    changed=m['states'].copy();changed[...,:2]=forecasts[k]
                    predictions[name],_=unsupported_candidates(dict(m,states=changed),g,base,{},'nearest')
                previous=common.load_npz(a.candidate_run/'private'/f'raw-{split}-{sid}.npz')
                common.old._same(predictions[BASE],previous[BASE],'149 original Qhard')
                common.old._same(predictions[METHODS[1]],previous['WS_Q_ZERO_NEAREST_OLS'],'149 nearest global binding')
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**predictions)
                ev=common.evaluation(paths['native_run'],split,sid,m);row,arr,artifacts=common.evaluate_predictions(m,g,pack,ev,predictions,{},method_names=METHODS,split=split,prediction_speed_unit='kn');add_harm(row,arr)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),forecast_audit=dict(nodes=len(audit),native_validation_folds=sum(v['validation_folds'] for v in audit),
                    native_ols_fits=sum(v['native_ols_fits'] for v in audit),
                    node_mean_horizon_mismatch_seconds=[v['mean_absolute_horizon_mismatch_seconds'] for v in audit],
                    outside_node_queries=sum(int((v['expert_mass'].sum(-1)>0).sum()) for v in audit),no_validation_queries=sum(v['no_validation_queries'] for v in audit),
                    horizon_exceeds_all_validation_windows=sum(v['horizon_exceeds_all_validation_windows'] for v in audit),latitude_saturations=sum(v['latitude_saturations'] for v in audit),expert_mass_sum=np.sum([v['expert_mass'].sum(0) for v in audit],axis=0).tolist()))
                np.savez_compressed(out/'private'/f'native-forecast-{split}-{sid}.npz',**{k:v for k,v in forecasts.items()},expert_mass=np.stack([v['expert_mass'] for v in audit]) if audit else np.zeros((0,64,3)))
                common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',artifacts);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=summarize(rows,arrays,METHODS);key=METHODS[4]+'_vs_'+METHODS[1];result['primary_bootstrap'][key]=common.component_bootstrap(rows,arrays,key)
            result.update(status='COMPLETED',idea_id=state['idea_id'],analytic_head_fits=0)
            common.old.frozen.write(out/filename,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--candidate-run',type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
