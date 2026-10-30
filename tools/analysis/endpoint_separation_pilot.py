"""160: native endpoint x velocity interventions, anonymous nearest readout."""
from __future__ import annotations
import argparse,sys,time,resource
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.robust_native_endpoint import native_endpoint_states
from association.evaluation.native_temporal import unsupported_candidates
from tools.analysis.temporal_source_pilot import physical,old_scores,recover,add_harm,summarize,BASE
from tools.analysis import observation_common as common
METHODS=(BASE,'WS_ZERO_NEAREST_GLOBAL','WS_ZERO_NEAREST_HOLD','WS_ZERO_NEAREST_MEDIANV','WS_ZERO_ENDPOINT_ONLY','WS_ZERO_VELOCITY_ONLY','WS_ZERO_ENDPOINT_JOINT')

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')};out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-ROBUST-ENDPOINT-SEPARATION-20261031',plan_version='1.0',mode='zero',methods=METHODS,started_at=datetime.now(timezone.utc).isoformat(),analytic_head_fits=0,optimizer_updates=0,test_opened=False,raw_reference_read=False,endpoint_control_run=str(a.endpoint_control_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>6*1024**3:raise MemoryError('6GiB RSS')
    try:
        if sys.platform.startswith('linux'):
            soft,hard=resource.getrlimit(resource.RLIMIT_AS);limit=min([12*1024**3]+[x for x in (soft,hard) if x!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        previous=common.old.frozen.read(a.endpoint_control_run/'run.json')
        if previous['status']!='COMPLETED' or previous['exit_code']!=0:raise ValueError('completed158 required')
        manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest);check(training_scene_ids=tids,scene_ids=vids)
        for split,ids,filename in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m,g,mean,pack,membership=physical(paths,split,sid);base=recover(m,g,pack,old_scores(paths,split,sid),mean);old=common.load_npz(a.endpoint_control_run/'private'/f'raw-{split}-{sid}.npz');common.old._same(base,old[BASE],'158 base');raw={n:old[n].copy() for n in METHODS[:4]}
                cache=common.load_npz(paths['native_run']/'private/native-events'/split/f'scene-{sid}.npz')
                for k in ('node_ids','query_seconds','sources'):common.old._same(cache[k],m[k],'native '+k)
                forecasts,account,details=native_endpoint_states(cache,m['states'])
                for kind,name in zip(('endpoint','velocity','joint'),METHODS[4:]):
                    changed=m['states'].copy();changed[...,:2]=forecasts[kind];raw[name],_=unsupported_candidates(dict(m,states=changed),g,base,{},'nearest')
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw);np.savez_compressed(out/'private'/f'endpoint-{split}-{sid}.npz',**forecasts);common.old.frozen.write(out/'private'/f'fits-{split}-{sid}.json.gz',details)
                ev=common.evaluation(paths['native_run'],split,sid,m);row,arr,art=common.evaluate_predictions(m,g,pack,ev,raw,{},method_names=METHODS,split=split,prediction_speed_unit='kn');add_harm(row,arr);row.update(scene_id=sid,component_id=str(manifest['components'][sid]),endpoint_accounting=account)
                key=lambda r,l:arr['pairs'][r+'_vs_'+l]['all']['delta']
                interaction=key(METHODS[6],METHODS[4])-key(METHODS[5],METHODS[1]);row['endpoint_velocity_interaction_m']=float(interaction.mean())
                common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr);check(**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=summarize(rows,arrays,METHODS)
            for name in METHODS[4:]:
                key=name+'_vs_'+METHODS[1];result['primary_bootstrap'][key]=common.component_bootstrap(rows,arrays,key)
            result.update(status='COMPLETED',idea_id=state['idea_id']);common.old.frozen.write(out/filename,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','endpoint-control-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
