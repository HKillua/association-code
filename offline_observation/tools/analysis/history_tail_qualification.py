"""179: T-only opportunity, score and native-age attribution; no deployment."""
from __future__ import annotations
import argparse,sys,time,json
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation.source_risk import _position_distance as distance
from tools.analysis.decoded_coverage_closeout import read,load
from tools.analysis.observation_common import old

def describe(v):
    v=np.asarray(v,float);v=v[np.isfinite(v)]
    return dict(finite=len(v),mean=float(v.mean()) if len(v) else None,quantiles=np.quantile(v,[0,.25,.5,.75,.95,1]).tolist() if len(v) else None)

def analyze(m,events,ev,member,base,candidate,policy,front,model):
    offsets=events['event_offsets'];times=events['event_times_seconds'];starts=times[offsets[:-1]];ends=times[offsets[1:]-1]
    registry={str(t):i for i,t in enumerate(ev['target_ids'])};claimed=set();records=[];unknown=duplicate=0
    scores=policy[front+'__scores'];features=policy[front+'__features'];count=policy[front+'__count'];weights=policy[('HO' if front=='HARD' else 'SO')+'__weights']
    for gid,label in enumerate(sorted(set(member.tolist()))):
        nodes=np.flatnonzero(member==label);identity=str(ev['identities'][nodes[0]])
        if identity not in registry:unknown+=1;continue
        if identity in claimed:duplicate+=1;continue
        claimed.add(identity);t=registry[identity];valid=ev['target_valid'][t,:,:2].all(-1);pure=len(set(ev['identities'][nodes].tolist()))==1
        errors=distance(m['states'][nodes],ev['target_states'][t]);errbase=distance(base[gid],ev['target_states'][t]);errnew=distance(candidate[gid],ev['target_states'][t]);nearestbase=distance(m['states'][nodes],base[gid]);best=errors.min(0)
        for j in np.flatnonzero(valid):
            mass=weights[nodes,j];selected=mass>0;low=count[gid,j]<=1;chosen=nodes[selected];w=mass[selected]
            ordered=np.sort(scores[nodes,j]);visible=nodes[m['support'][nodes,j]]
            if low and not selected.any():raise ValueError('178 open action missing')
            age=float(w@m['nearest_seconds'][chosen,j]) if low else np.nan;span=float(w@m['span_seconds'][chosen]) if low else np.nan
            before=bool(np.all(m['query_seconds'][j]<starts[nodes]));after=bool(np.all(m['query_seconds'][j]>ends[nodes]));outside=np.maximum(starts[chosen]-m['query_seconds'][j],0)+np.maximum(m['query_seconds'][j]-ends[chosen],0)
            z=np.abs((features[chosen,j]-model['mean'])/model['std']) if low else np.empty((0,12))
            records.append(dict(group_index=gid,target_index=t,query_index=int(j),pure=pure,count=int(count[gid,j]),before=before,after=after,
                old_error_m=float(errbase[j]),new_error_m=float(errnew[j]),delta_m=float(errnew[j]-errbase[j]),oracle_error_m=float(best[j]),oracle_avoids_harm=bool(best[j]-errbase[j]<=1000),base_has_node=bool(nearestbase[:,j].min()<1e-6),
                age_seconds=age,span_seconds=span,age_span_ratio=float(w@(m['nearest_seconds'][chosen,j]/np.maximum(m['span_seconds'][chosen],1))) if low else np.nan,
                extrapolation_seconds=float(w@outside) if low else np.nan,fit_rmse_m=float(w@m['fit_rmse_m'][chosen]) if low else np.nan,native_count=float(w@m['native_count'][chosen]) if low else np.nan,
                predicted_risk=float(w@scores[chosen,j]) if low else np.nan,top_margin=float(ordered[1]-ordered[0]) if len(ordered)>1 else np.nan,
                supported_risk_advantage=float(scores[visible,j].min()-ordered[0]) if len(visible) else np.nan,selected_feature_clip_fraction=float(w@np.mean(z>8,axis=1)) if low else np.nan,change_m=float(distance(base[gid,j],candidate[gid,j]))))
    return records,dict(unknown_groups=unknown,duplicate_groups=duplicate,claimed_targets=len(claimed),total_targets=len(registry))

def summarize(rows):
    result={};fields=('age_seconds','span_seconds','age_span_ratio','extrapolation_seconds','fit_rmse_m','native_count','predicted_risk','top_margin','supported_risk_advantage','selected_feature_clip_fraction','change_m')
    for scope in ('all','zero','one','multi','pure_zero','pure_one','mixed_zero','mixed_one','before','after','inside_time_envelope'):
        def keep(r):
            return scope=='all' or (scope=='zero' and r['count']==0) or (scope=='one' and r['count']==1) or (scope=='multi' and r['count']>1) or (scope.startswith('pure_') and r['pure'] and r['count']==(0 if scope.endswith('zero') else 1)) or (scope.startswith('mixed_') and not r['pure'] and r['count']==(0 if scope.endswith('zero') else 1)) or (scope=='before' and r['before']) or (scope=='after' and r['after']) or (scope=='inside_time_envelope' and not r['before'] and not r['after'])
        selected=[r for r in rows if keep(r)];groups={}
        for label,condition in [('all',lambda r:True),('harm',lambda r:r['delta_m']>1000),('rescued',lambda r:r['delta_m']< -1000),('other',lambda r:abs(r['delta_m'])<=1000)]:
            use=[r for r in selected if condition(r)];groups[label]=dict(points=len(use),mean_delta_m=float(np.mean([r['delta_m'] for r in use])) if use else None,oracle_avoids_harm=sum(r['oracle_avoids_harm'] for r in use),base_has_node=sum(r['base_has_node'] for r in use),candidate_excess_over_1km=sum(r['new_error_m']-r['oracle_error_m']>1000 for r in use),**{f:describe([r[f] for r in use]) for f in fields})
        result[scope]=groups
    return result

def run(a):
    parent=read(a.parent_run/'run.json');assert parent['status']=='COMPLETED';out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-HISTORY-TAIL-QUALIFICATION-179',plan_version='1.0',parent_run=str(a.parent_run),native_run=parent['native_run'],training_scene_ids=parent['training_scene_ids'],V_scenes=0,optimizer_updates=0,new_predictions=0,test_opened=False,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=old.frozen.peak_rss_bytes());old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        original=Path(parent['parent_run']);manifest=read(Path(parent['manifest']));allrows={k:[] for k in ('HARD','STRUCT')}
        for sid in parent['training_scene_ids']:
            m=load(Path(parent['native_run'])/'private/native-motion/T_core'/f'scene-{sid}.npz');events=load(Path(parent['native_run'])/'private/native-events/T_core'/f'scene-{sid}.npz');ev=load(Path(parent['native_run'])/'private/native-evaluation/T_core'/f'scene-{sid}.npz');q=load(a.parent_run/'private'/f'partitions-T_core-{sid}.npz');raw=load(a.parent_run/'private'/f'raw-T_core-{sid}.npz');policy=load(a.parent_run/'private'/f'policy-T_core-{sid}.npz');prior=load(original/'private'/f'raw-T_core-{sid}.npz');model=torch.load(a.parent_run/'private'/f'model-leave_{manifest["components"][sid]}.pt',map_location='cpu',weights_only=False)
            old._same(m['node_ids'],events['node_ids'],'native events/motion nodes');public={}
            for front,arm in [('HARD','HO'),('STRUCT','SO')]:
                old._same(raw[front+'_NN'],prior[front+'_NN'],'original175 output');rows,pop=analyze(m,events,ev,q[front],raw[front+'_NN'],raw[arm+'_NN'],policy,front,model)
                for r in rows:r.update(scene_id=sid,component_id=str(manifest['components'][sid]))
                # Nonfinite descriptors only mark inapplicable multi-support fields; private only.
                old.frozen.write(out/'private'/f'{front}-{sid}.json.gz',[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items()} for r in rows]);allrows[front].extend(rows);public[front]=dict(population=pop,summary=summarize(rows))
            old.frozen.write(out/'scenes'/f'T_core-{sid}.json.gz',public);check(completed_T_scenes=len(list((out/'scenes').glob('*.gz'))))
        expected=read(a.parent_run/'history-candidate-accounting.json')['splits']['T_core']['physics'];result={}
        for front,arm in [('HARD','HO'),('STRUCT','SO')]:
            rows=allrows[front];summary=summarize(rows);assert summary['all']['harm']['points']==expected[arm+'_vs_'+front]['strata']['all']['new_harm_over_1km'];assert len(rows)==expected[arm+'_vs_'+front]['strata']['all']['old']['points'];assert summary['multi']['all']['mean_delta_m']==0
            examples={}
            for key,condition in [('pure_one',lambda r:r['pure'] and r['count']==1),('pure_zero',lambda r:r['pure'] and r['count']==0),('mixed_zero',lambda r:not r['pure'] and r['count']==0)]:
                examples[key]=[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in r.items() if k not in ('group_index','target_index')} for r in sorted((r for r in rows if condition(r) and r['delta_m']>1000),key=lambda r:r['delta_m'],reverse=True)[:5]]
            result[front]=dict(summary=summary,examples=examples)
        old.frozen.write(out/'summary.json',dict(status='COMPLETED',heads=result,scope='T-only descriptive saved-output qualification; no prediction, causal proof or V tuning'))
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('parent-run','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
