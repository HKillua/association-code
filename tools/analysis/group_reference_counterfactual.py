"""163: fixed references vs fixed inputs; oracle analysis, never a deployment score."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import source_risk as risk
from association.evaluation.native_temporal import predict
from association.evaluation.local_source_evidence import predict_residual
from tools.analysis.temporal_source_pilot import physical
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source
from tools.analysis.audit_source_risk_arrays import distances
NAMES=('quadratic','network')

def rank(e,s,available):
    chosen=available&(s==np.where(available,s,np.inf).min(0));mass=chosen/np.maximum(chosen.sum(0),1)
    selected=(mass*np.where(available,e,0.)).sum(0);best=np.where(available,e,np.inf).min(0)
    return selected,selected-best

def purify(m,g,identities):
    keep=np.zeros(len(m['states']),bool)
    for group in g:
        group=np.asarray(group);keep[group[identities[group]==identities[group[0]]]]=True
    ids=np.flatnonzero(keep);inverse=np.full(len(keep),-1);inverse[ids]=np.arange(len(ids));groups=[inverse[np.asarray(group)[keep[group]]].tolist() for group in g]
    motion={k:m[k][ids] for k in source.MOTION_KEYS};pack=risk.build_source_features(**motion,groups=groups)
    return pack,groups

def scene_counterfactual(m,g,pack,ev,models,original):
    labels=source.training_labels(pack,g,ev);clean,cg=purify(m,g,ev['identities']);shape=(len(g),m['states'].shape[1]);arrays={k:np.zeros(shape,bool) for k in ('eligible','rare','pure','alt_seen','best_flips','clean_multi','clean_one','clean_zero')}
    for name in NAMES:
        for key in ('wrong','alt_correct','clean_wrong'):arrays[name+'_'+key]=np.zeros(shape,bool)
        for key in ('selected_error','clean_selected_error','excess','clean_excess'):arrays[name+'_'+key]=np.full(shape,np.nan)
    original_cpu=predict_residual(models['network'],pack['features'].reshape(-1,10),'cpu').reshape(pack['available'].shape)
    cleaned={'quadratic':predict(models['quadratic'],clean['features'].reshape(-1,10)).reshape(clean['available'].shape),'network':predict_residual(models['network'],clean['features'].reshape(-1,10),'cpu').reshape(clean['available'].shape)}
    registry={str(v):i for i,v in enumerate(ev['target_ids'])};checks=dict(formula_max_difference_m=0.,pure_group_feature_checks=0,cpu_original_choice_changes=0,missing_reference_candidates=0,valid_alternative_query_candidates=0);reference_count=0
    for name in NAMES:
        for bound in ('min','max'):arrays[name+'_reference_excess_'+bound]=np.full(shape,np.nan)
    def error(states,truth):
        e=risk._position_distance(states[...,:2],truth[...,:2]);other=distances(states,truth);np.testing.assert_allclose(e,other,rtol=1e-10,atol=1e-5,equal_nan=True)
        finite=np.isfinite(e)&np.isfinite(other)
        if finite.any():checks['formula_max_difference_m']=max(checks['formula_max_difference_m'],float(np.abs(e-other)[finite].max()))
        return e
    for gid,(group,(a,b),(ca,cb)) in enumerate(zip(g,zip(pack['group_offsets'][:-1],pack['group_offsets'][1:]),zip(clean['group_offsets'][:-1],clean['group_offsets'][1:]))):
        identity=str(ev['identities'][group[0]]);pure=len(set(ev['identities'][group].tolist()))==1;arrays['pure'][gid]=pure
        available=pack['available'][a:b];eligible=labels['labeled'][a:b].any(0)&(available.sum(0)>=2);arrays['eligible'][gid]=eligible
        if not eligible.any():continue
        target=registry[identity];truth=ev['target_states'][target];e=error(pack['source_states'][a:b],truth);use=labels['labeled'][a:b];np.testing.assert_allclose(e[use],labels['errors_m'][a:b][use],rtol=1e-10,atol=1e-6)
        best=np.where(available,e,np.inf).min(0);best_set=available&(e<=best[None,:]+1e-6);arrays['rare'][gid]=eligible&available[0]&(e[0]>best+1e-6)
        for name in NAMES:
            selected,excess=rank(e,original[name][a:b],available);arrays[name+'_wrong'][gid]=eligible&(excess>1e-6);arrays[name+'_selected_error'][gid]=selected;arrays[name+'_excess'][gid]=excess
            arrays[name+'_reference_excess_min'][gid]=excess;arrays[name+'_reference_excess_max'][gid]=excess
        _,cpu_regret=rank(e,original_cpu[a:b],available);checks['cpu_original_choice_changes']+=int((eligible&((cpu_regret>1e-6)!=arrays['network_wrong'][gid])).sum())
        for other_id in sorted(set(map(str,ev['identities'][group].tolist()))-{identity}):
            prefix='ref_'+str(reference_count)+'__';reference_count+=1;arrays[prefix+'group_index']=np.asarray(gid);arrays[prefix+'target_identity']=np.asarray(other_id);arrays[prefix+'source_row_start']=np.asarray(a)
            if other_id not in registry:
                checks['missing_reference_candidates']+=1;arrays[prefix+'target_index']=np.asarray(-1);arrays[prefix+'valid_original_query']=np.zeros(shape[1],bool);continue
            ix=registry[other_id];mask=eligible&ev['target_valid'][ix,:,:2].all(-1);arrays['alt_seen'][gid]|=mask;checks['valid_alternative_query_candidates']+=int(mask.sum());other=error(pack['source_states'][a:b],ev['target_states'][ix]);best_other=np.where(available,other,np.inf).min(0);set_other=available&(other<=best_other[None,:]+1e-6);arrays['best_flips'][gid]|=mask&(set_other!=best_set).any(0)
            arrays[prefix+'target_index']=np.asarray(ix);arrays[prefix+'valid_original_query']=mask;arrays[prefix+'source_errors_m']=other
            for name in NAMES:
                _,regret=rank(other,original[name][a:b],available);arrays[name+'_alt_correct'][gid]|=mask&(regret<=1e-6);arrays[name+'_reference_excess_min'][gid,mask]=np.minimum(arrays[name+'_reference_excess_min'][gid,mask],regret[mask]);arrays[name+'_reference_excess_max'][gid,mask]=np.maximum(arrays[name+'_reference_excess_max'][gid,mask],regret[mask])
        cv=clean['available'][ca:cb];count=cv.sum(0);arrays['clean_multi'][gid]=eligible&(count>=2);arrays['clean_one'][gid]=eligible&(count==1);arrays['clean_zero'][gid]=eligible&(count==0)
        if pure:
            np.testing.assert_allclose(pack['features'][a:b],clean['features'][ca:cb],rtol=0,atol=1e-12);checks['pure_group_feature_checks']+=1
            # Identical pure input reuses its original scores, avoiding CPU/GPU rounding as an intervention.
            for name in NAMES:cleaned[name][ca:cb]=original[name][a:b]
        ce=error(clean['source_states'][ca:cb],truth)
        for name in NAMES:
            selected,excess=rank(ce,cleaned[name][ca:cb],cv);arrays[name+'_clean_wrong'][gid]=arrays['clean_multi'][gid]&(excess>1e-6);arrays[name+'_clean_selected_error'][gid]=selected;arrays[name+'_clean_excess'][gid]=excess
    return arrays,checks

def summarize_arrays(arrays,mask):
    use=mask&arrays['eligible'];common=use&arrays['clean_multi'];out=dict(queries=int(use.sum()),alternative_reference_seen=int((use&arrays['alt_seen']).sum()),reference_best_set_flips=int((use&arrays['best_flips']).sum()),purification_multi=int(common.sum()),purification_one=int((use&arrays['clean_one']).sum()),purification_zero=int((use&arrays['clean_zero']).sum()),methods={})
    for n in NAMES:
        reference_mask=use&arrays['alt_seen'];span=arrays[n+'_reference_excess_max'][reference_mask]-arrays[n+'_reference_excess_min'][reference_mask]
        old=arrays[n+'_wrong'];new=arrays[n+'_clean_wrong'];delta=arrays[n+'_clean_selected_error']-arrays[n+'_selected_error'];out['methods'][n]=dict(official_wrong=int((use&old).sum()),official_excess_sum_m=float(arrays[n+'_excess'][use].sum()),official_excess_over_1km=int((use&(arrays[n+'_excess']>1000)).sum()),reference_excess_range_sum_m=float(span.sum()),official_wrong_with_alternative_correct=int((use&old&arrays[n+'_alt_correct']).sum()),common_original_wrong=int((common&old).sum()),common_clean_wrong=int((common&new).sum()),common_rescued=int((common&old&~new).sum()),common_new_wrong=int((common&~old&new).sum()),common_selected_error_delta_sum_m=float(delta[common].sum()),common_new_harm_over_1km=int((common&(delta>1000)).sum()))
    return out

def aggregate(rows):
    if not rows:return None
    def merge(values):
        return {k:merge([v[k] for v in values]) if isinstance(values[0][k],dict) else sum(v[k] for v in values) for k in values[0]}
    return merge([r['strata'] for r in rows])

def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')};out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();start=time.monotonic();state=dict(status='RUNNING',idea_id='IDEA-GROUP-REFERENCE-COUNTERFACTUAL-20261031',plan_version='1.0',started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,test_opened=False,oracle_diagnostic_only=True,expression_run=str(a.expression_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>6*1024**3:raise MemoryError('RSS6GiB')
    try:
        torch.set_num_threads(4);manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest);check(training_scene_ids=tids,scene_ids=vids)
        previous=common.old.frozen.read(a.expression_run/'run.json')
        if previous['status']!='COMPLETED':raise ValueError('159 completed')
        for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[]
            for sid in ids:
                m,g,mean,pack,membership=physical(paths,split,sid);ev=common.evaluation(paths['native_run'],split,sid,m);fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full';net=torch.load(a.expression_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold'];scores=common.load_npz(a.expression_run/'private'/f'policy-{split}-{sid}.npz');original={'quadratic':scores['WS_T128_Q_HARD__scores'],'network':scores['WS_T128_RESIDUAL_HARD__scores']}
                arrays,checks=scene_counterfactual(m,g,pack,ev,dict(quadratic=net['base'],network=net),original);np.savez_compressed(out/'private'/f'counterfactual-{split}-{sid}.npz',**arrays)
                strata={tag:summarize_arrays(arrays,mask) for tag,mask in [('all',np.ones_like(arrays['eligible'])),('pure',arrays['pure']),('mixed',~arrays['pure']),('rare',arrays['rare']),('common',~arrays['rare'])]};row=dict(scene_id=sid,component_id=str(manifest['components'][sid]),checks=checks,strata=strata)
                old=common.old.frozen.read(a.expression_run/'scenes'/f'{split}-{sid}.json.gz')
                for n,method in [('quadratic','WS_T128_Q_HARD'),('network','WS_T128_RESIDUAL_HARD')]:
                    if strata['all']['methods'][n]['official_wrong']!=old['selection'][method]['wrong']:raise ValueError('159 official error binding')
                    if strata['rare']['methods'][n]['official_wrong']!=old['role_diagnostic'][method]['slot0_not_best']['wrong']:raise ValueError('159 rare binding')
                rows.append(row);check(**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            components=sorted({r['component_id'] for r in rows});largest=max(components,key=lambda c:sum(r['component_id']==c for r in rows));summary=dict(status='COMPLETED',oracle_diagnostic_only=True,all_components=aggregate(rows),by_component={c:aggregate([r for r in rows if r['component_id']==c]) for c in components},without_largest_component=aggregate([r for r in rows if r['component_id']!=largest]),largest_component_id=largest,checks=dict(formula_max_difference_m=max(r['checks']['formula_max_difference_m'] for r in rows),**{k:sum(r['checks'][k] for r in rows) for k in ('pure_group_feature_checks','cpu_original_choice_changes','missing_reference_candidates','valid_alternative_query_candidates')}))
            common.old.frozen.write(out/fn,summary);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','expression-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
