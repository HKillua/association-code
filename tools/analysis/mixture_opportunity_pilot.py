"""134: T-only reference mixture opportunity and frozen OOF mistake diagnosis."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from association.evaluation import source_risk as risk
from association.evaluation import mixture_opportunity as mix
from association.evaluation.fusion_controls import ordinary_fusion
from tools.analysis import source_risk_pilot as previous
old=previous.old
IDEA='IDEA-MIXTURE-OPPORTUNITY-20261030'
HEADS=risk.HEADS
GRID=4097
RAM,AS=6*1024**3,12*1024**3
BINS={'score_margin': [0,.05,.1,.25,.5,1,np.inf], 'source_distance_m':[0,250,500,1000,3000,10000,np.inf],
      'selected_nearest_seconds':[0,60,300,900,1800,np.inf], 'selected_fit_rmse_m':[0,100,500,1000,3000,np.inf],
      'selected_within_source_rms_m':[0,100,500,1000,3000,np.inf], 'selected_native_count':[0,4,8,16,32,np.inf],
      'selected_supported_member_count':[0,1,2,4,np.inf]}


def describe(x):
    x=np.asarray(x,dtype=np.float64)
    return dict(count=len(x),sum=float(x.sum()),mean=float(x.mean()) if len(x) else None,
        median=float(np.median(x)) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,
        p99=float(np.quantile(x,.99)) if len(x) else None,min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None)


def head_statistics(error, best, score, endpoint, selection_regret):
    excess=error-best; harm=np.maximum(excess,0.)
    weight=(score==score.min(1,keepdims=True)).astype(float);weight/=weight.sum(1,keepdims=True)
    credit=(weight*(endpoint==endpoint.min(1,keepdims=True))).sum(1)
    top=np.sort(harm)[::-1][:math.ceil(.01*len(harm))]
    return dict(query_count=len(error),hard_error_m=describe(error),signed_excess_m=describe(excess),
        selection_regret_m=describe(selection_regret),positive_harm_sum_m=float(harm.sum()),negative_excess_benefit_sum_m=float(np.maximum(-excess,0.).sum()),
        score_selection_credit_sum=float(credit.sum()),score_selection_accuracy=float(credit.mean()) if len(credit) else None,
        imperfect_selection_query_count=int((credit<1).sum()),score_tie_query_count=int(((weight>0).sum(1)>1).sum()),
        positive_harm_gt1000m_count=int((excess>1000).sum()),top1pct_query_count=len(top),top1pct_positive_harm_sum_m=float(top.sum()),
        top1pct_harm_share=float(top.sum()/harm.sum()) if harm.sum() else None)


def bin_masks(values, edges):
    values=np.asarray(values,float).copy()
    for boundary in edges:
        if np.isfinite(boundary):
            values[np.isclose(values,boundary,rtol=1e-12,atol=1e-9)]=boundary
    return [(f'{lo:g}..{hi:g}',(values>=lo)&(values<hi)) for lo,hi in zip(edges[:-1],edges[1:])]


def direction_cos(source_positions, truth):
    # Project each source unit vector onto the reference tangent plane.
    # Direction is undefined for zero error or antipodal degeneracy.
    def unit(x):
        lat,lon=np.deg2rad(x[...,0]),np.deg2rad(x[...,1])
        return np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],axis=-1)
    ref=unit(truth);u=unit(source_positions)
    tangent=u-(u*ref[:,None]).sum(-1)[...,None]*ref[:,None]
    norm=np.linalg.norm(tangent,axis=-1);valid=(norm>1e-12).all(1)
    result=np.full(len(truth),np.nan)
    result[valid]=np.clip((tangent[valid,0]*tangent[valid,1]).sum(-1)/(norm[valid,0]*norm[valid,1]),-1,1)
    return result


def summarize(data):
    endpoint=data['endpoint_errors_m']; n=len(endpoint);finite=np.isfinite(data['gap_bound_m'])
    out=dict(query_count=n,oracle=dict(best_single_error_m=describe(data['best_single_error_m']),grid_error_m=describe(data['grid_error_m']),
        gain_m=describe(data['gain_m']),alpha=describe(data['alpha']),interior_alpha_001_099_count=int(((data['alpha']>.01)&(data['alpha']<.99)).sum()),
        endpoint_alpha_count=int(((data['alpha']==0)|(data['alpha']==1)).sum()),
        chosen_alpha_counts_only=True,endpoint_attains_grid_min_count=int(np.isclose(data['best_single_error_m'],data['grid_error_m'],rtol=1e-10,atol=1e-7).sum()),
        grid_tie_query_count=int((data['tied_grid_count']>1).sum()),
        strict_positive_gain_interior_gt1e_6m_count=int(((data['alpha']>0)&(data['alpha']<1)&(data['gain_m']>1e-6)).sum()),
        gains_gt_m={str(v):int((data['gain_m']>v).sum()) for v in (1,10,100)},
        gain_sums_gt_m={str(v):float(data['gain_m'][data['gain_m']>v].sum()) for v in (1,10,100)},
        gain_fractions_gt_m={str(v):float((data['gain_m']>v).mean()) if n else None for v in (1,10,100)},
        error_direction_cos=describe(data['error_direction_cos'][np.isfinite(data['error_direction_cos'])]),
        undefined_direction_count=int((~np.isfinite(data['error_direction_cos'])).sum()),
        gap_bound_m=describe(data['gap_bound_m'][finite]),undefined_gap_count=int((~finite).sum()),
        gap_le_m={str(v):int((data['gap_bound_m']<=v).sum()) for v in (.1,1,10)}),heads={},
        role_best_credit={},group_purity={})
    for role in np.unique(data['role_indices']):
        ties=endpoint==endpoint.min(1,keepdims=True);credit=ties/ties.sum(1,keepdims=True)
        out['role_best_credit']['role_'+str(role)]=float((credit*(data['role_indices']==role)).sum())
    for pure,name in ((True,'pure'),(False,'mixed')):
        use=data['group_pure']==pure
        out['group_purity'][name]=dict(query_count=int(use.sum()),gain_m=describe(data['gain_m'][use]))
    for head in HEADS:
        scores=data['oof_'+head+'_scores'];error=data[head+'_hard_error_m'];best=data['best_single_error_m']
        regret=data[head+'_selection_regret_m']; result=head_statistics(error,best,scores,endpoint,regret)
        weight=(scores==scores.min(1,keepdims=True)).astype(float);weight/=weight.sum(1,keepdims=True)
        x=data['features'];selected=lambda k: (weight*np.expm1(x[:,:,k])).sum(1)
        proxies=dict(score_margin=np.abs(np.diff(scores,axis=1)[:,0]),source_distance_m=data['source_distance_m'],
            selected_nearest_seconds=selected(0)*60.,selected_fit_rmse_m=selected(1)*1000.,
            selected_within_source_rms_m=selected(6)*1000.,selected_native_count=selected(2),selected_supported_member_count=selected(4))
        strata={}
        for key,edges in BINS.items():
            strata[key]={}
            # Boundary convention [lower,upper), last [lower,infinity].
            for name,use in bin_masks(proxies[key],edges):
                strata[key][name]=head_statistics(error[use],best[use],scores[use],endpoint[use],regret[use])
        strata['group_purity']={name:head_statistics(error[use],best[use],scores[use],endpoint[use],regret[use])
            for name,use in (('pure',data['group_pure']),('mixed',~data['group_pure']))}
        result['strata']=strata;out['heads'][head]=result
    return out


def joined(items):
    return {k:np.concatenate([d[k] for d in items]) for k in items[0]}


def bootstrap(rows, arrays):
    cs=sorted({r['component_id'] for r in rows});counts=[];sums=[]
    for c in cs:
        selected=[a['gain_m'] for r,a in zip(rows,arrays) if r['component_id']==c]
        counts.append(sum(len(x) for x in selected));sums.append(sum(float(x.sum()) for x in selected))
    count=np.asarray(counts);total=np.asarray(sums);draw=np.random.default_rng(20).integers(0,len(cs),size=(2000,len(cs)))
    den=count[draw].sum(1)
    if not (den>0).all():raise ValueError('empty component bootstrap')
    means=total[draw].sum(1)/den
    return dict(component_count=len(cs),resamples=2000,seed=20,point_estimate_m=float(total.sum()/count.sum()),
        percentile95_m=np.quantile(means,[.025,.975]).tolist(),independent_test=False,estimand='best single minus best4097 grid mixture')


def scene(native,source,sid,component,roles,output):
    pack=previous.load_npz(source/'private'/f'training-{sid}.npz')
    m=previous.load_npz(native/'private/native-motion/T_core'/f'scene-{sid}.npz')
    ev=previous.load_npz(native/'private/native-evaluation/T_core'/f'scene-{sid}.npz')
    for key in ('node_ids','query_seconds'):
        old._same(pack[key],m[key],'133/127 '+key);old._same(m[key],ev[key],'native eval '+key)
    if str(pack['component_id'])!=component:raise ValueError('component binding differs')
    groups=old.groups_from_membership(pack['membership'],len(m['states']))
    moments=mix.source_position_moments(states=m['states'],support=m['support'],groups=groups,pack=pack)
    ids={s:i for i,s in enumerate(ev['target_ids'].tolist())}
    groupids=[];qs=[];sources=[];targets=[];anchors=[];pure=[]
    scope=dict(total_group_queries=len(groups)*len(m['query_seconds']),source_support_zero=0,source_support_one=0,source_support_multi=0,
        complete_target_reference_valid_query_count=int(ev['target_valid'][...,:2].all(-1).sum()),labeled_group_query_count=0,duplicate_anchor_group_count=0,unknown_anchor_group_count=0,
        first_anchor_target_count=0,target_without_anchor_count=0,first_anchor_reference_invalid_query_count=0,
        first_anchor_reference_valid_zero_source_count=0,first_anchor_reference_valid_one_source_count=0,
        first_anchor_reference_valid_multi_source_count=0,mixed_identity_group_count=0,
        duplicate_group_queries=0,unknown_group_queries=0,reference_valid_points_without_anchor=0)
    claimed=set()
    for gid,g in enumerate(groups):
        start,stop=pack['group_offsets'][gid:gid+2];availability=pack['available'][start:stop];counts=availability.sum(0)
        if counts.max(initial=0)>2:raise ValueError('134 diagnostic requires actualK2, not implicit K>=3 oracle')
        for name,mask in (('zero',counts==0),('one',counts==1),('multi',counts>=2)):scope['source_support_'+name]+=int(mask.sum())
        identity=ev['identities'][g[0]]
        scope['mixed_identity_group_count']+=int(len(set(ev['identities'][g].tolist()))>1)
        if identity not in ids:
            scope['unknown_anchor_group_count']+=1;scope['unknown_group_queries']+=len(m['query_seconds'])
        elif identity in claimed:
            scope['duplicate_anchor_group_count']+=1;scope['duplicate_group_queries']+=len(m['query_seconds'])
        else:
            claimed.add(identity);scope['first_anchor_target_count']+=1
            valid=ev['target_valid'][ids[identity],:,:2].all(-1)
            scope['first_anchor_reference_invalid_query_count']+=int((~valid).sum())
            for name,mask in (('zero',counts==0),('one',counts==1),('multi',counts>=2)):
                scope['first_anchor_reference_valid_'+name+'_source_count']+=int((valid&mask).sum())
        use=np.flatnonzero(pack['labeled'][start:stop].sum(0)==2)
        for q in use:
            sr=np.flatnonzero(availability[:,q])+start
            if len(sr)!=2:raise ValueError('labels/support mismatch')
            groupids.append(gid);qs.append(q);sources.append(sr);anchors.append(m['states'][g[0],q,:2]);
            targets.append(ev['target_states'][ids[ev['identities'][g[0]]],q,:2]);pure.append(len(set(ev['identities'][g].tolist()))==1)
    gi=np.asarray(groupids,np.int64);qi=np.asarray(qs,np.int64);sr=np.asarray(sources,np.int64).reshape(-1,2);n=len(gi)
    expected=int(pack['labeled'].sum()/2)
    if n!=expected:raise ValueError('full original labeled T population required')
    scope['labeled_group_query_count']=n
    scope['target_without_anchor_count']=len(ids)-len(claimed)
    scope['reference_valid_points_without_anchor']=int(sum(ev['target_valid'][i,:,:2].all(-1).sum() for identity,i in ids.items() if identity not in claimed))
    pairs=moments[sr,qi[:,None]];anchor=np.asarray(anchors,float).reshape(-1,2);truth=np.asarray(targets,float).reshape(-1,2)
    data=mix.grid_mixture_oracle(pairs,anchor,truth,grid_size=GRID,chunk_size=128)
    data={k:v for k,v in data.items() if isinstance(v,np.ndarray)}
    expected_errors=pack['errors_m'][sr,qi[:,None]]
    if not np.allclose(data['endpoint_errors_m'],expected_errors,rtol=1e-9,atol=1e-5):raise ValueError('133 endpoint error binding failed')
    feature=pack['features'][sr,qi[:,None]]
    role_by_row=np.empty(len(moments),np.int64)
    for row in range(len(moments)):
        name=str(m['sources'][np.flatnonzero(pack['node_to_source_row']==row)[0]])
        if name not in roles:roles[name]=len(roles)
        role_by_row[row]=roles[name]
    data.update(error_direction_cos=direction_cos(pack['source_states'][sr,qi[:,None],:2],truth),features=feature,role_indices=role_by_row[sr],group_pure=np.asarray(pure,bool),
        source_distance_m=risk._position_distance(pack['source_states'][sr[:,0],qi],pack['source_states'][sr[:,1],qi]))
    mean=ordinary_fusion(m['states'],m['nearest_seconds'],m['support'],groups)['predictions']['mean_supported'][np.asarray([g[0] for g in groups],np.int64)]
    original_oof=previous.load_npz(source/'private'/f'oof-{sid}.npz')
    for head in HEADS:
        score=original_oof[head+'__scores'];policy=risk.recover_source_policy(states=m['states'],support=m['support'],groups=groups,pack=pack,scores=score,kind='hard',mean_states=mean)
        hard=policy['group_states'][gi,qi,:2];two_scores=score[sr,qi[:,None]]
        w=(two_scores==two_scores.min(1,keepdims=True)).astype(float);w/=w.sum(1,keepdims=True)
        rebuilt=mix.mixture_positions(pairs,w[:,0],anchor)
        if n and risk._position_distance(rebuilt,hard).max()>1e-5:raise ValueError('moment reconstruction differs from133 actual hard')
        data['oof_'+head+'_scores']=two_scores;data[head+'_hard_position']=hard
        data[head+'_hard_error_m']=risk._position_distance(hard,truth)
        data[head+'_selection_regret_m']=(w*data['endpoint_errors_m']).sum(1)-data['best_single_error_m']
    private=dict(data,moment_pairs=pairs,anchors=anchor,truth=truth,group_indices=gi,query_indices=qi,source_rows=sr,
        source_errors_m=expected_errors,node_ids=m['node_ids'],query_seconds=m['query_seconds'],membership=pack['membership'])
    np.savez_compressed(output/'private'/f'scene-{sid}.npz',**private)
    return dict(scene_id=sid,component_id=component,scope=scope,**summarize(data)),data


def run(args):
    native,source,output=(Path(getattr(args,k)).resolve() for k in ('native_run','source_run','output'))
    if any(output==p or output in p.parents or p in output.parents for p in (native,source)):raise ValueError('output overlaps immutable input')
    if not 1<=args.cpu_threads<=4:raise ValueError('CPU1..4')
    output.mkdir(parents=True,exist_ok=False);(output/'private').mkdir();(output/'scenes').mkdir()
    start=time.monotonic();state=dict(status='RUNNING',idea_id=IDEA,plan_version='1.0',source_run=str(source),native_run=str(native),
        manifest=str(Path(args.manifest).resolve()),started_at=datetime.now(timezone.utc).isoformat(),completed_scene_count=0,
        grid_size=GRID,reference_oracle_only=True,new_fits=0,neural_updates=0,teacher_forwards=0,V_arrays_opened=False,test_opened=False,
        archive_read=False,cpu_threads=args.cpu_threads,time_policy='advisory')
    def check(**extra):
        state.update(extra,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=old.frozen.peak_rss_bytes());old.frozen.write(output/'run.json',state)
        if state['peak_rss_bytes']>RAM:raise MemoryError('6GiB RSS exceeded')
        if state['elapsed_seconds']>1800 and not (output/'time_advisory.json').exists():old.frozen.write(output/'time_advisory.json',dict(action='continue fixedT32/4097 grid only',elapsed_seconds=state['elapsed_seconds']))
    try:
        check(phase='binding')
        if sys.platform.startswith('linux'):
            soft,hard=resource.getrlimit(resource.RLIMIT_AS);limit=min([AS]+[v for v in (soft,hard) if v!=resource.RLIM_INFINITY]);resource.setrlimit(resource.RLIMIT_AS,(limit,hard))
        run133=old.frozen.read(source/'run.json');run127=old.frozen.read(native/'run.json');marker=old.frozen.read(native/'native-cache.json');manifest=old.frozen.read(args.manifest)
        tids=run133['training_scene_ids']
        if (run133.get('status')!='COMPLETED' or run133.get('exit_code')!=0 or run133.get('idea_id')!=previous.IDEA
                or run133.get('completed_training_scene_count')!=32 or run133.get('analytic_head_fits')!=39
                or Path(run133['native_run']).resolve()!=native or run127.get('status')!='COMPLETED' or run127.get('exit_code')!=0
                or marker.get('status')!='CACHE_COMPLETED' or marker['scene_ids']['T_core']!=tids
                or tids!=manifest['splits']['T_core'][:32] or len(tids)!=32 or len({manifest['components'][s] for s in tids})!=12):
            raise ValueError('completed133 and original127 T32 required')
        check(scene_ids=tids,phase='T_reference_diagnostic');rows=[];arrays=[];roles={}
        for sid in tids:
            row,data=scene(native,source,sid,str(manifest['components'][sid]),roles,output)
            rows.append(row);arrays.append(data);old.frozen.write(output/'scenes'/f'scene-{sid}.json.gz',row)
            check(completed_scene_count=len(rows),last_scene_id=sid)
        np.savez_compressed(output/'private'/'roles.npz',source_names=np.asarray(list(roles)),indices=np.asarray(list(roles.values())))
        cs=sorted({r['component_id'] for r in rows});largest=max(cs,key=lambda c:sum(r['component_id']==c for r in rows))
        def aggregate(indices):return summarize(joined([arrays[i] for i in indices]))
        summary=dict(status='COMPLETED',idea_id=IDEA,plan_version='1.0',all_components=aggregate(range(len(rows))),
            by_component={c:aggregate([i for i,r in enumerate(rows) if r['component_id']==c]) for c in cs},
            without_largest_component=aggregate([i for i,r in enumerate(rows) if r['component_id']!=largest]),largest_component_id=largest,
            primary_bootstrap=bootstrap(rows,arrays),scope={k:sum(r['scope'][k] for r in rows) for k in rows[0]['scope']},
            component_scene_counts={c:sum(r['component_id']==c for r in rows) for c in cs},source_role_count=len(roles),
            grid_size=GRID,reference_oracle_only=True,limitations=['T-only diagnostic; reference-selected weights are not deployable predictions',
            'Grid best has stated Lipschitz gap only on nondegenerate continuous paths; never exact global continuous optimum claim',
            'OOF only isolates new133 heads; upstream teacher uses allT','Actual sources=2; no realK>=3 or external physical sensor evidence',
            'Conditional strata are associations; source-role frequencies are not a priority policy'])
        old.frozen.write(output/'scene-summary.json.gz',rows);check(phase='finalizing')
        summary.update(elapsed_seconds=state['elapsed_seconds'],peak_rss_bytes=state['peak_rss_bytes']);old.frozen.write(output/'summary.json',summary)
        check(status='COMPLETED',exit_code=0,phase='completed',finished_at=datetime.now(timezone.utc).isoformat());return summary
    except BaseException as e:
        state.update(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,InterruptedError)) else 'FAILED',exit_code=1,error=f'{type(e).__name__}: {e}',elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
        old.frozen.write(output/'run.json',state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','source-run','manifest','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cpu-threads',type=int,default=4);p.add_argument('--time-policy',choices=('advisory',),default='advisory');args=p.parse_args()
    def interrupted(signum,frame):raise InterruptedError(f'signal{signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:run(args)
    except BaseException:traceback.print_exc();return 1
    print(f'COMPLETED: {args.output}/summary.json',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
