"""182: T-only native kinematic conditions; fixed existing outputs, no training."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation import native_compatibility as nc
from association.evaluation.history_candidates import source_balanced
from tools.analysis.history_tail_qualification import describe
from tools.analysis.decoded_coverage_closeout import read,load
from tools.analysis.observation_common import old

def analyze(m,ev,groups,raw,policy,front,arm,bridge,x,details):
    registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set();rows=[]
    for gid,nodes in enumerate(groups):
        identity=str(ev['identities'][nodes[0]])
        if identity not in registry or identity in claimed:continue
        claimed.add(identity);ti=registry[identity];valid=ev['target_valid'][ti,:,:2].all(1);pure=len(set(ev['identities'][nodes].tolist()))==1
        base=nc.distance(raw[front+'_NN'][gid],ev['target_states'][ti]);o=nc.distance(raw[arm+'_NN'][gid],ev['target_states'][ti]);b=nc.distance(bridge[gid],ev['target_states'][ti]);errors=nc.distance(m['states'][nodes],ev['target_states'][ti])
        for j in np.flatnonzero(valid):
            count=int(policy[front+'__count'][gid,j]);w=policy[arm+'__weights'][nodes,j];has=count<=1
            selected=w@x[nodes,j] if has else np.zeros(len(nc.NAMES));best=nodes[errors[:,j]==errors[:,j].min()];oracle=source_balanced(best,m['sources'])@x[best,j]
            candidates=[(float(policy[front+'__scores'][own,j].min()),a) for own,a in details[gid] if a['bracket'][j]];eligible=bool(candidates)
            picked=[a for score,a in candidates if score==min(k[0] for k in candidates)] if candidates else []
            row=dict(group_index=gid,target_index=ti,query_index=int(j),count=count,pure=pure,eligible=eligible,
                before=all(a['before'][j] for _,a in details[gid]),after=all(a['after'][j] for _,a in details[gid]),
                old_delta_m=float(o[j]-base[j]),bridge_delta_m=float(b[j]-o[j]),old_error_m=float(o[j]),oracle_excess_m=float(o[j]-errors[:,j].min()),
                selected={name:float(v) for name,v in zip(nc.NAMES,selected)},oracle={name:float(v) for name,v in zip(nc.NAMES,oracle)})
            row['bridge_evidence']={key:float(np.mean([a[key][j] for a in picked])) if picked else None for key in ('gap','bracket_report_gap','bracket_report_known','age','report_secant_gap')};rows.append(row)
    return rows

def summarize(rows):
    out={}
    scopes={'all':lambda r:True,'pure_one':lambda r:r['pure'] and r['count']==1,'one':lambda r:r['count']==1,
      'zero_bracket':lambda r:r['count']==0 and r['eligible'],'zero_no_bracket':lambda r:r['count']==0 and not r['eligible'],
      'zero_before':lambda r:r['count']==0 and r['before'],'zero_after':lambda r:r['count']==0 and r['after'],'multi':lambda r:r['count']>1}
    for scope,keep in scopes.items():
        selected=[r for r in rows if keep(r)];cohorts={}
        for label,cond in [('all',lambda r:True),('old_harm',lambda r:r['old_delta_m']>1000),('old_rescued',lambda r:r['old_delta_m']< -1000),('bridge_harm',lambda r:r['bridge_delta_m']>1000),('bridge_rescued',lambda r:r['bridge_delta_m']< -1000)]:
            use=[r for r in selected if cond(r)]
            cohorts[label]=dict(points=len(use),pure_points=sum(r['pure'] for r in use),old_error_m=describe([r['old_error_m'] for r in use]),old_delta_m=describe([r['old_delta_m'] for r in use]),bridge_delta_m=describe([r['bridge_delta_m'] for r in use]),
              selected={name:describe([r['selected'][name] for r in use]) for name in nc.NAMES},
              selected_minus_oracle={name:describe([r['selected'][name]-r['oracle'][name] for r in use]) for name in nc.NAMES},
              bridge_evidence={name:describe([r['bridge_evidence'][name] for r in use if r['bridge_evidence'][name] is not None]) for name in ('gap','bracket_report_gap','bracket_report_known','age','report_secant_gap')})
        out[scope]=cohorts
    return out

def run(a):
    parent=read(a.parent_run/'run.json');br=read(a.bridge_run/'run.json');assert parent['status']==br['status']=='COMPLETED';out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-NATIVE-COMPATIBILITY-QUALIFICATION-182',plan_version='1.0',parent_run=str(a.parent_run),bridge_run=str(a.bridge_run),native_run=parent['native_run'],training_scene_ids=parent['training_scene_ids'],V_scenes=0,optimizer_updates=0,new_predictions=0,test_opened=False,canonical_input_speed_unit='kn',started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=old.frozen.peak_rss_bytes());old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        allrows={k:[] for k in ('HARD','STRUCT')}
        for i,sid in enumerate(parent['training_scene_ids']):
            n=Path(parent['native_run'])/'private';m=load(n/'native-motion/T_core'/f'scene-{sid}.npz');events=load(n/'native-events/T_core'/f'scene-{sid}.npz');ev=load(n/'native-evaluation/T_core'/f'scene-{sid}.npz');q=load(a.parent_run/'private'/f'partitions-T_core-{sid}.npz');raw=load(a.parent_run/'private'/f'raw-T_core-{sid}.npz');policy=load(a.parent_run/'private'/f'policy-T_core-{sid}.npz');bridge=load(a.bridge_run/'private'/f'raw-T_core-{sid}.npz');old._same(m['node_ids'],events['node_ids'],'native nodes');public={}
            for front,arm,barm in [('HARD','HO','HB'),('STRUCT','SO','SB')]:
                groups=[np.flatnonzero(q[front]==g) for g in sorted(set(q[front].tolist()))];x,details=nc.descriptors(m,events,groups,speed_unit='kn');rows=analyze(m,ev,groups,raw,policy,front,arm,bridge[barm+'_NN'],x,details)
                for r in rows:r['scene_id']=sid
                allrows[front].extend(rows);old.frozen.write(out/'private'/f'{front}-{sid}.json.gz',rows);public[front]=summarize(rows)
            old.frozen.write(out/'scenes'/f'T_core-{sid}.json.gz',public);check(completed_T_scenes=i+1)
        expected=read(a.parent_run/'history-candidate-accounting.json')['splits']['T_core']['physics'];bexpected=read(a.bridge_run/'history-followup-accounting.json')['splits']['T_core']['physics'];result={}
        for front,arm,barm in [('HARD','HO','HB'),('STRUCT','SO','SB')]:
            summary=summarize(allrows[front]);e=expected[arm+'_vs_'+front]['strata']['all'];assert summary['all']['all']['points']==e['old']['points'];assert summary['all']['old_harm']['points']==e['new_harm_over_1km'];assert summary['all']['bridge_harm']['points']==bexpected[barm+'_NN_vs_'+arm+'_NN']['all']['new_harm_over_1km'];result[front]=summary
        old.frozen.write(out/'summary.json',dict(status='COMPLETED',heads=result,feature_names=nc.NAMES,scope='T-only descriptive native compatibility and propagation qualification; no fit, deployment, causal proof or V tuning'))
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','bridge-run','output'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
