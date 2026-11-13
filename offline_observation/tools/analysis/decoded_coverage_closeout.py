"""176 saved-output audit and candidate counts; inherited saved-physics decomposition, no tuning."""
import argparse,json,gzip,time,resource,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.analysis.audit_shared_control_arrays import position

def read(p):
    if str(p).endswith('.gz'):
        with gzip.open(p,'rt') as f:return json.load(f)
    return json.loads(p.read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def metric(y,d):
    return dict(points=len(y),tp=int((y&d).sum()),fp=int((~y&d).sum()),fn=int((y&~d).sum()),tn=int((~y&~d).sum()))
def stats(v):
    return dict(points=len(v),mean=float(np.mean(v)) if len(v) else None,rmse=float(np.sqrt(np.mean(v*v))) if len(v) else None,p99=float(np.quantile(v,.99)) if len(v) else None)


def verify_calibration(run,record):
    cal=read(run/'T-calibration.json');parent=Path(record['parent_run']);previous=read(parent/'run.json');old=read(parent/'T-summary.json')['all_components'];native=Path(record['native_run'])
    limits={k:old['association']['WS_'+k]['mstsa']['paper_metrics']['m_pfa'] for k in ('raw','decoded')};assert limits==cal['limits']
    prepared={};checked=0;maxdiff=0.
    for sid in cal['T_scenes']:
        q=load(parent/'private'/f'partitions-T_core-{sid}.npz');parts=load(run/'private'/f'calibration-T_core-{sid}.npz');m=load(native/'private/native-motion/T_core'/f'scene-{sid}.npz');ev=load(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
        pairs=q['pairs'];mask=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]];y=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]];prepared[sid]=(q,parts,ev,pairs,mask,y)
    for head,letter in (('HARD','H'),('STRUCT','S')):
        expected=old['methods'][head+'_NN']['position'];floor=dict(points=expected['intersection_count'],targets=expected['covered_target_trajectory_count']);assert floor==cal['floors'][head]
        grid=np.unique(np.r_[np.quantile(np.concatenate([d[0][head+'_uncalibrated_scores'][d[4]] for d in prepared.values()]),np.linspace(0,1,33)),1.7320103112387182,previous['thresholds'][head]])
        assert np.array_equal(grid,cal['grids'][head]);recomputed=[]
        for i,theta in enumerate(grid):
            values={k:{f:[] for f in ('pca','pfa')} for k in ('raw','decoded')};points=targets=0
            stored=cal['curves'][head][i];assert stored['threshold']==theta
            for sid,data in prepared.items():
                q,parts,ev,pairs,mask,y=data;member=parts[head][i];d=member[pairs[:,0]]==member[pairs[:,1]];raw=q[head+'_uncalibrated_scores']+1.7320103112387182-theta>=1.7320103112387182
                public=next(r for r in stored['scenes'] if r['scene_id']==sid)
                for kind,prediction in [('raw',raw),('decoded',d)]:
                    yy=y[mask];dd=prediction[mask];tp=int((yy&dd).sum());fp=int((~yy&dd).sum());fn=int((yy&~dd).sum());positive=tp+fn;accepted=tp+fp
                    for field,v in dict(tp=tp,fp=fp,fn=fn,positive=positive,accepted=accepted).items():assert public[kind][field]==v
                    values[kind]['pca'].append(tp/positive if positive else None);values[kind]['pfa'].append(fp/accepted if accepted else None)
                claimed=set()
                for group in sorted(set(member.tolist())):
                    nodes=np.flatnonzero(member==group);claimed.add(str(ev['identities'][nodes[0]]))
                tv=ev['target_valid'][...,:2].all(-1);ownpoints=owntargets=0
                for t,identity in enumerate(ev['target_ids']):
                    if str(identity) in claimed:ownpoints+=int(tv[t].sum());owntargets+=int(tv[t].any())
                assert public['coverage']==dict(points=ownpoints,targets=owntargets);points+=ownpoints;targets+=owntargets
            metrics={k:{f:None if any(v is None for v in vs) else float(sum(vs)/len(vs)) for f,vs in fields.items()} for k,fields in values.items()};metrics['coverage']=dict(points=points,targets=targets)
            for k in ('raw','decoded'):
                for f in ('pca','pfa'):
                    a,b=metrics[k][f],stored['metrics'][k][f];assert (a is None)==(b is None)
                    if a is not None:maxdiff=max(maxdiff,abs(a-b));assert abs(a-b)<1e-12
            assert metrics['coverage']==stored['metrics']['coverage'];recomputed.append(dict(threshold=float(theta),metrics=metrics));checked+=1
        for policy in ('R','D','C'):
            eligible=[r for r in recomputed if all(r['metrics'][k][f] is not None for k in ('raw','decoded') for f in ('pca','pfa')) and all(r['metrics'][k]['pfa']<=limits[k]+1e-12 for k in ('raw','decoded')) and (policy!='C' or all(r['metrics']['coverage'][k]>=floor[k] for k in floor))]
            best=max(eligible,key=lambda r:(r['metrics']['raw' if policy=='R' else 'decoded']['pca'],-r['metrics']['decoded']['pfa'],r['threshold']))
            assert best['threshold']==cal['chosen'][letter+policy]['threshold']
            final=read(run/'T-summary.json')['all_components']['methods'][letter+policy+'_NN']['position']
            assert final['intersection_count']==best['metrics']['coverage']['points'];assert final['covered_target_trajectory_count']==best['metrics']['coverage']['targets']
    return dict(scope='independent full T grid/candidate counts/coverage/selection and selected physical population; cached candidate partitions, not independent WS solver',candidate_rows_checked=checked,scene_candidate_rows_checked=checked*len(cal['T_scenes']),max_metric_abs_difference=maxdiff,chosen_thresholds={k:v['threshold'] for k,v in cal['chosen'].items()})


def main(run,dest):
    assert not dest.exists();started=time.monotonic();record=read(run/'run.json');assert record['status']=='COMPLETED'
    prefixes=list(dict.fromkeys(k.rsplit('_',1)[0] for k in record['methods']));result=dict(status='COMPLETED',exit_code=0,run=str(run),scope='posthoc saved-array support/purity/anchor/edge transition and descriptive component association intervals; no tuning/fit/prediction or causal certification',splits={})
    result['calibration_verification']=verify_calibration(run,record)
    comparisons=[('HARD','HR'),('HR','HD'),('HD','HC'),('STRUCT','SR'),('SR','SD'),('SD','SC'),('HARD','HC'),('STRUCT','HC'),('WS','HC'),('HARD','SC'),('STRUCT','SC'),('WS','SC')]
    for split,ids,fn,rf in [('T_core',record['training_scene_ids'],'T-summary.json','T-scene-summary.json.gz'),('V_select',record['scene_ids'],'summary.json','scene-summary.json.gz')]:
        rows=read(run/rf);summary=read(run/fn);cs=sorted(set(r['component_id'] for r in rows));draw=np.random.default_rng(20).integers(len(cs),size=(2000,len(cs)))
        unit=dict(association_intervals={},edges={},physics={},search={},teacher_binding_max_abs=max((r.get('teacher_binding_max_abs',0.) for r in rows),default=0.))
        for left,p in comparisons:
            for kind in ('raw','decoded'):
                key=p+'_'+kind+'_vs_'+left+'_'+kind;unit['association_intervals'][key]={}
                for f,field in [('m_pca','pca'),('m_pfa','pfa')]:
                    vals=[]
                    for r in rows:
                        a=r['association'][left+'_'+kind]['mstsa']['paper_metrics'][f];b=r['association'][p+'_'+kind]['mstsa']['paper_metrics'][f]
                        vals.append(None if a is None or b is None else b-a)
                    if any(v is None for v in vals):unit['association_intervals'][key][f]=dict(point_pp=None,percentile95_pp=None,reason='undefined scene denominator; no silent deletion');continue
                    sums=np.array([sum(v for v,r in zip(vals,rows) if r['component_id']==c) for c in cs]);count=np.array([sum(r['component_id']==c for r in rows) for c in cs]);sim=sums[draw].sum(1)/count[draw].sum(1)
                    unit['association_intervals'][key][f]=dict(point_pp=float(np.mean(vals)*100),percentile95_pp=(np.quantile(sim,[.025,.975])*100).tolist(),components=len(cs),resamples=2000,seed=20,independent_test=False)
            unit['physics'][p+'_vs_'+left]=dict(strata={},examples=[]);unit['edges'][p+'_vs_'+left]={}
        for p in rows[0]['search']:
            unit['search'][p]={k:sum(r['search'][p].get(k,0) for r in rows) for k in ['move_count','anchor_rejections','support_rejections']}
        pool={p+'_vs_'+left:{} for left,p in comparisons}
        for sid,row in zip(ids,rows):
            m=load(Path(record['native_run'])/'private/native-motion'/split/f'scene-{sid}.npz');ev=load(Path(record['native_run'])/'private/native-evaluation'/split/f'scene-{sid}.npz');z=load(run/'private'/f'states-{split}-{sid}.npz');q=load(run/'private'/f'partitions-{split}-{sid}.npz')
            pairs=q['pairs'];cross=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]];y=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]];joint=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).any(1)
            nt=len(ev['target_ids']);tv=ev['target_valid'][...,:2].all(-1);arms={}
            for p in prefixes:
                name=p+'_NN';a={k[len(name)+1:]:v for k,v in z.items() if k.startswith(name+'_')};groups=[a['group_node_indices'][b:e] for b,e in zip(a['group_offsets'][:-1],a['group_offsets'][1:])]
                gi=a['aligned_group_indices'][:nt];valid=a['aligned_prediction_valid'][:nt,:,:2].all(-1);pure=np.zeros(nt,bool);anchor=np.full(nt,-1,int);count=np.zeros((nt,64),int)
                for t,g in enumerate(gi):
                    if g<0:continue
                    members=groups[g];anchor[t]=members[0];pure[t]=len(set(ev['identities'][members].tolist()))==1
                    count[t]=sum(m['support'][members[m['sources'][members]==s]].any(0) for s in np.unique(m['sources'][members]))
                arms[p]=dict(error=position(a['aligned_prediction'][:nt],ev['target_states'],tv&valid),valid=valid,pure=pure,anchor=anchor,count=count)
            for left,p in comparisons:
                for reg,mask in [('cross',cross),('cross_no_joint',cross&~joint),('cross_joint',cross&joint)]:
                    for kind in ('raw','decoded'):
                        old=q[left+'_scores']>=1.7320103112387182 if kind=='raw' else q[left][pairs[:,0]]==q[left][pairs[:,1]]
                        new=q[p+'_scores']>=1.7320103112387182 if kind=='raw' else q[p][pairs[:,0]]==q[p][pairs[:,1]]
                        key=reg+'_'+kind;v=unit['edges'][p+'_vs_'+left].setdefault(key,{k:0 for k in ['true_rescued','true_lost','false_removed','false_added']})
                        for k,sel in [('true_rescued',y&~old&new),('true_lost',y&old&~new),('false_removed',~y&old&~new),('false_added',~y&~old&new)]:v[k]+=int((mask&sel).sum())
                a,b=arms[left],arms[p];use=tv&a['valid']&b['valid'];delta=b['error']-a['error'];bad=use&(delta>1000);same=a['anchor']==b['anchor']
                strata=dict(all=np.ones_like(use),same_anchor=np.broadcast_to(same[:,None],use.shape),changed_anchor=np.broadcast_to(~same[:,None],use.shape),new_zero=b['count']==0,new_one=b['count']==1,new_multi=b['count']>=2,pure_to_pure=np.broadcast_to((a['pure']&b['pure'])[:,None],use.shape))
                for k,mask in strata.items():
                    v=pool[p+'_vs_'+left].setdefault(k,dict(old=[],new=[],delta=[]));sel=use&mask
                    for f,item in [('old',a['error']),('new',b['error']),('delta',delta)]:v[f].append(item[sel])
                for t,j in np.argwhere(bad):
                    unit['physics'][p+'_vs_'+left]['examples'].append(dict(scene_id=sid,query_index=int(j),old_error_m=float(a['error'][t,j]),new_error_m=float(b['error'][t,j]),increase_m=float(delta[t,j]),old_sources=int(a['count'][t,j]),new_sources=int(b['count'][t,j]),same_anchor=bool(same[t]),pure_to_pure=bool(a['pure'][t]&b['pure'][t])))
        for left,p in comparisons:
            for k,v in pool[p+'_vs_'+left].items():
                a,b,d=[np.concatenate(v[f]) for f in ['old','new','delta']]
                unit['physics'][p+'_vs_'+left]['strata'][k]=dict(old=stats(a),new=stats(b),mean_delta_m=float(d.mean()) if len(d) else None,new_harm_over_1km=int((d>1000).sum()),max_harm_m=float(max(0,d.max())) if len(d) else None)
            expected=summary['all_components']['pairwise'][p+'_NN_vs_'+left+'_NN'];actual=unit['physics'][p+'_vs_'+left]['strata']['all'];assert actual['new_harm_over_1km']==expected['actual_harm']['new_harm_over_1km'];assert abs(actual['mean_delta_m']-expected['strata']['all']['candidate_minus_A0']['mean_error_change'])<1e-6
            # Full counts above; only anonymous five worst examples retained publicly.
            examples=unit['physics'][p+'_vs_'+left]['examples'];unit['physics'][p+'_vs_'+left]['new_harm_example_count']=len(examples);unit['physics'][p+'_vs_'+left]['examples']=sorted(examples,key=lambda x:x['increase_m'],reverse=True)[:5]
        result['splits'][split]=unit
    result.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],peak_rss_bytes=result['peak_rss_bytes'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    from tools.analysis.association_context_closeout import closeout
    closeout(a.run)
    main(a.run,a.run/'decoded-coverage-squeeze.json')
