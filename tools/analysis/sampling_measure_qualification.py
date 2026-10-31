"""192: fixed T native sampling/mean-measure qualification, no new predictions."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation.native_sampling import velocity_measures,FIELDS
from tools.analysis.history_tail_qualification import describe
from tools.analysis.decoded_coverage_closeout import read,load
from tools.analysis.observation_common import old

MORE=('count_mean_OLS_gap_mps','time_mean_OLS_gap_mps','weighted_node_gap_times_absdt_m')

def summarize(rows,fields):
    return dict(points=len(rows),**{key:describe([row[key] for row in rows]) for key in fields})

def join_conditions(m,models,rows,diag,policy,front):
    lookup={(v['group_index'],v['query_index']):v for v in diag['decisions']};short='H' if front=='HARD' else 'S';result=[]
    for row in rows:
        gid,j=row['group_index'],row['query_index'];action=lookup[gid,j];nodes=np.asarray(action['selected_nodes'],int);mass=policy[short+'O__weights'][nodes,j];assert len(nodes) and np.isclose(mass.sum(),1.)
        v={key:float(sum(w*models[node]['stats'][key] for w,node in zip(mass,nodes))) for key in FIELDS};v['known_mass']=float(sum(w*models[node]['known'] for w,node in zip(mass,nodes)))
        np.testing.assert_allclose(v['count_mean_speed_mps'],row['mean_speed_mps'],rtol=1e-12,atol=1e-10);np.testing.assert_allclose(v['report_spread_mps'],row['report_spread_mps'],rtol=1e-12,atol=1e-10);assert np.isclose(v['native_count'],row['native_count']) and np.isclose(v['span_seconds'],row['span_seconds'])
        for kind in ('count','time'):v[kind+'_mean_OLS_gap_mps']=float(sum(w*np.linalg.norm(models[node][kind+'_mean']-m['ols_velocity_mps'][node]) for w,node in zip(mass,nodes)))
        np.testing.assert_allclose(v['count_mean_OLS_gap_mps'],row['mean_ols_gap_mps'],rtol=1e-12,atol=1e-10)
        v['weighted_node_gap_times_absdt_m']=float(sum(w*models[node]['stats']['mean_measure_gap_mps']*abs(m['query_seconds'][j]-models[node]['times'][0 if action['before'] else -1]) for w,node in zip(mass,nodes)))
        v.update(scene_id=row.get('scene_id'),group_index=gid,query_index=j,pure=row['pure'],before=row['before'],PG_delta_m=row['PG_delta_m'],group_nodes=row['group_nodes']);result.append(v)
    return result

def cohorts(rows):
    result={}
    for name,keep in [('all',lambda r:True),('pure',lambda r:r['pure']),('mixed',lambda r:not r['pure']),('before',lambda r:r['before']),('after',lambda r:not r['before']),('PG_harm',lambda r:r['PG_delta_m']>1000),('PG_rescued',lambda r:r['PG_delta_m']< -1000)]:
        selected=[r for r in rows if keep(r)];result[name]=dict(**summarize(selected,FIELDS+MORE+('known_mass',)),distinct_group_count=len({(v['scene_id'],v['group_index']) for v in selected}),distinct_scene_count=len({v['scene_id'] for v in selected}))
    return result

def run(a):
    previous=read(a.qualification_run/'run.json');parent=read(Path(previous['parent_run'])/'run.json');assert previous['status']=='COMPLETED' and previous['evaluation_splits']==['T_core'] and previous['completed_T_scenes']==32 and previous['V_scenes']==0
    out=a.output;out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic();r=dict(status='RUNNING',idea_id='IDEA-SAMPLING-MEASURE-QUALIFICATION-192',plan_version='1.0',qualification_run=str(a.qualification_run),parent_run=previous['parent_run'],native_run=previous['native_run'],training_scene_ids=previous['training_scene_ids'],scene_ids=[],evaluation_splits=['T_core'],V_scenes=0,new_predictions=0,optimizer_updates=0,neural_head_fits=0,native_likelihood_fits=0,teacher_forward_calls=0,test_opened=False,native_fragments=0,native_events=0,independent_integral_checks=0,started_at=datetime.now(timezone.utc).isoformat())
    def check(**kwargs):
        r.update(kwargs,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=old.frozen.peak_rss_bytes());old.frozen.write(out/'run.json',r)
        if r['peak_rss_bytes']>4*1024**3:raise MemoryError('RSS4GiB')
    try:
        nodes_all=[];allrows={k:[] for k in ('HARD','STRUCT')}
        for sid in r['training_scene_ids']:
            n=Path(r['native_run'])/'private';m=load(n/'native-motion/T_core'/f'scene-{sid}.npz');events=load(n/'native-events/T_core'/f'scene-{sid}.npz');old._same(m['node_ids'],events['node_ids'],'native nodes');models=[]
            for node in range(len(m['states'])):
                left,right=events['event_offsets'][node:node+2];t=events['event_times_seconds'][left:right];model=velocity_measures(t,events['event_values'][left:right],events['event_valid'][left:right]);model['times']=t[[0,-1]];models.append(model);nodes_all.append(model['stats']);r['native_fragments']+=1;r['native_events']+=len(t);r['independent_integral_checks']+=int(model['known'])
            np.savez_compressed(out/'private'/f'measures-{sid}.npz',count_mean=np.array([v['count_mean'] for v in models]),time_mean=np.array([v['time_mean'] for v in models]),known=np.array([v['known'] for v in models]),stats=np.array([[v['stats'][k] for k in FIELDS] for v in models]))
            diag=read(a.qualification_run/'private'/f'diagnostics-T_core-{sid}.json.gz');policy=load(Path(r['parent_run'])/'private'/f'policy-T_core-{sid}.npz');pub={}
            for front in allrows:
                rows=read(a.qualification_run/'private'/f'conditions-{front}-{sid}.json.gz');joined=join_conditions(m,models,rows,diag[front],policy,front)
                for row in joined:row['scene_id']=sid
                allrows[front].extend(joined);old.frozen.write(out/'private'/f'conditions-{front}-{sid}.json.gz',joined);pub[front]=cohorts(joined)
            old.frozen.write(out/'scenes'/f'T_core-{sid}.json.gz',dict(nodes=summarize([v['stats'] for v in models],FIELDS),heads=pub));check(completed_T_scenes=len(list((out/'scenes').glob('*.json.gz'))))
        assert r['native_fragments']==2310 and r['native_events']==580384 and r['independent_integral_checks']<=r['native_fragments'];heads={k:cohorts(v) for k,v in allrows.items()}
        for front,points,harms,rescued in [('HARD',451,34,3),('STRUCT',344,1,33)]:assert heads[front]['all']['points']==points and heads[front]['PG_harm']['points']==harms and heads[front]['PG_rescued']['points']==rescued
        old.frozen.write(out/'summary.json',dict(status='COMPLETED',nodes=summarize(nodes_all,FIELDS),heads=heads,scope='fixed T native inputs and original PG error-cohort descriptions; nominal gap not deployed prediction or physical risk'))
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('qualification-run','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
