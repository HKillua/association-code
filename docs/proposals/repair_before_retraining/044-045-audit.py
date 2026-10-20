#!/usr/bin/env python3
"""Recompute 044/045 saved-score evidence; no model, dataset or training access.

Run from any directory: .venv/bin/python docs/proposals/repair_before_retraining/044-045-audit.py
The component IDs are receipt metadata (not encoded in score NPZ); identities are
recovered from the complete positive label graph only for grouping diagnostics.
All metrics, threshold sweep and direct-anchor grouping below are independent of
production evaluation functions. Intermediate unsaved score arrays are not audited.
"""
import gzip
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('.json')
DIRS = {k: ROOT/'HandOver/results'/v for k,v in {
    '044': 'EVIDENCE-ASSOC-GEOMETRY-20261021-044',
    '045': 'EVIDENCE-ASSOC-FULL-20261021-045'}.items()}

def read(p):
    return json.load(gzip.open(p)) if p.suffix == '.gz' else json.loads(p.read_text())

def counts(y,p):
    tp,fp,fn,tn = map(int, ((y&p).sum(), (~y&p).sum(), (y&~p).sum(), (~y&~p).sum()))
    d = 2*tp+fp+fn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, f1=2*tp/d if d else 1.,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None)

def auc(y,s):
    # Direct rank sum with average tied ranks, independent of production code.
    order = np.argsort(s, kind='stable'); ordered = s[order]
    ranks = np.arange(1,len(s)+1,dtype=float)
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1])+1]
    for start, stop in zip(starts, np.r_[starts[1:],len(s)]):
        if stop-start > 1: ranks[start:stop] = (start+1+stop)/2
    n = int(y.sum()); m = len(y)-n
    return float((ranks[y[order]].sum()-n*(n+1)/2)/(n*m)) if n and m else None

def load(run,endpoint,comp):
    with np.load(DIRS[run]/f'{endpoint}-scores.npz', allow_pickle=False) as z:
        rows=[]
        for sid in z['scene_ids'].tolist():
            r={k:z[f'{sid}_{k}'].copy() for k in (
                'pairs','labels','scores','node_ids','sources','joint_support_queries')}
            r.update(scene_id=sid, component_id=comp[sid]);r['labels']=r['labels'].astype(bool)
            n=len(r['node_ids']); p=r['pairs']
            assert np.array_equal(p,np.column_stack(np.triu_indices(n,1)))
            assert np.isfinite(r['scores']).all()
            rows.append(r)
    return rows

def sweep(rows):
    # Global descending score order; update only the changed scene's F1.
    score=np.concatenate([r['scores'] for r in rows]); y=np.concatenate([r['labels'] for r in rows])
    idx=np.concatenate([np.full(len(r['labels']),i) for i,r in enumerate(rows)])
    order=np.argsort(-score,kind='stable'); tp=np.zeros(len(rows),int);pred=tp.copy()
    truth=np.array([r['labels'].sum() for r in rows]); f1=(truth==0).astype(float)
    best=float(f1.mean()); threshold=None
    starts=np.r_[0,np.flatnonzero(score[order][1:] != score[order][:-1])+1]
    for start,stop in zip(starts,np.r_[starts[1:],len(score)]):
        group=order[start:stop]; touched=np.unique(idx[group])
        np.add.at(tp,idx[group],y[group]); np.add.at(pred,idx[group],1)
        f1[touched]=2*tp[touched]/(pred[touched]+truth[touched])
        value=float(f1.mean())
        if value>best+1e-12: best=value;threshold=float(score[group[0]])
    return dict(threshold=threshold,macro_f1=best,candidate_count=len(starts)+1)

def grouping(r,threshold):
    n=len(r['node_ids']); pair=r['pairs']; truth=np.eye(n,dtype=bool)
    truth[pair[:,0],pair[:,1]]=r['labels'];truth[pair[:,1],pair[:,0]]=r['labels']
    # Full truth equivalence matrix, including self identity.
    for i in range(n):
        assert np.all(truth[truth[i]] == truth[i])
    edges=np.eye(n,dtype=bool); accept=r['scores']>=threshold
    edges[pair[:,0],pair[:,1]]=accept;edges[pair[:,1],pair[:,0]]=accept
    free=np.ones(n,bool); groups=[];membership=np.empty(n,int)
    for i in range(n):
        if free[i]:
            members=np.flatnonzero(free&edges[i]);free[members]=False
            membership[members]=len(groups);groups.append(members)
    mixed=badanchors=badhelpers=0
    for members in groups:
        wrong=~truth[np.ix_(members,members)]
        mixed+=int(wrong.any());badanchors+=int(wrong.any(1).sum());badhelpers+=int(wrong.sum())
    same=membership[pair[:,0]]==membership[pair[:,1]]
    return dict(scene_id=r['scene_id'],group_count=len(groups),mixed_group_count=mixed,
        anchors_with_wrong_helper=badanchors,directed_wrong_helper_relations=badhelpers,
        membership=membership.tolist(),correct_same_identity_pairs=int((same&r['labels']).sum()),
        split_pairs=int((~same&r['labels']).sum()),merge_pairs=int((same&~r['labels']).sum()),
        correct_different_identity_pairs=int((~same&~r['labels']).sum()))

def metrics(rows,threshold,subset='all'):
    per=[]; ys=[];ss=[]
    for r in rows:
        pair=r['pairs'];same=r['sources'][pair[:,0]]==r['sources'][pair[:,1]]
        mask={'all':np.ones(len(pair),bool),'same_source':same,'cross_source':~same,
              'joint_support':r['joint_support_queries']>0,
              'no_joint_support':r['joint_support_queries']==0}[subset]
        y=r['labels'][mask];s=r['scores'][mask]
        if not len(y): continue
        per.append(dict(scene_id=r['scene_id'],component_id=r['component_id'],pairs=len(y),positives=int(y.sum()),**counts(y,s>=threshold)))
        ys.append(y);ss.append(s)
    y=np.concatenate(ys);s=np.concatenate(ss); c=counts(y,s>=threshold)
    return dict(pairs=len(y),positives=int(y.sum()),scenes=len(per),macro_f1=float(np.mean([r['f1'] for r in per])),
                micro_f1=c.pop('f1'),pooled_auc=auc(y,s),
                m_pca=float(np.mean([r['recall'] for r in per])) if all(r['recall'] is not None for r in per) else None,
                m_pfa=float(np.mean([1-r['precision'] for r in per])) if all(r['precision'] is not None for r in per) else None,
                **c,per_scene=per)

def summarize(rows,threshold):
    overall=metrics(rows,threshold); g=[grouping(r,threshold) for r in rows]
    keys=[k for k in g[0] if k not in ('scene_id','membership')]
    return dict(threshold=threshold,metrics=overall,
        strata={s:metrics(rows,threshold,s) for s in ('same_source','cross_source','joint_support','no_joint_support')},
        grouping={**{k:sum(r[k] for r in g) for k in keys},'per_scene':g})

def comparison(a,b,ta,tb,ma,mb):
    rows=[]
    for r,s,ar,br in zip(a,b,ma['metrics']['per_scene'],mb['metrics']['per_scene']):
        for key in ('pairs','labels','sources','node_ids','joint_support_queries'):
            assert np.array_equal(r[key],s[key])
        y=r['labels'];old=r['scores']>=ta;new=s['scores']>=tb
        fixed=(old!=y)&(new==y);broken=(old==y)&(new!=y)
        rows.append(dict(scene_id=r['scene_id'],component_id=r['component_id'],
            macro_f1_delta=br['f1']-ar['f1'],wrong_to_right=int(fixed.sum()),right_to_wrong=int(broken.sum()),
            corrected_fp=int((fixed&~y).sum()),corrected_fn=int((fixed&y).sum()),
            introduced_fp=int((broken&~y).sum()),introduced_fn=int((broken&y).sum())))
    keys=('wrong_to_right','right_to_wrong','corrected_fp','corrected_fn','introduced_fp','introduced_fn')
    total={k:sum(r[k] for r in rows) for k in keys}; total['net_corrected_edges']=total['wrong_to_right']-total['right_to_wrong']
    components=[]
    for cid in sorted({r['component_id'] for r in rows}):
        group=[r for r in rows if r['component_id']==cid]
        components.append(dict(component_id=cid,scenes=len(group),mean_scene_f1_delta=float(np.mean([r['macro_f1_delta'] for r in group])),
                               net_corrected_edges=sum(r['wrong_to_right']-r['right_to_wrong'] for r in group)))
    delta=np.array([r['macro_f1_delta'] for r in rows]);cd=np.array([r['mean_scene_f1_delta'] for r in components])
    loo=[np.mean([r['macro_f1_delta'] for r in rows if r['component_id']!=c['component_id']]) for c in components]
    return dict(**total,macro_f1_delta=float(delta.mean()),
        scenes=dict(improved=int((delta>1e-12).sum()),worse=int((delta< -1e-12).sum()),tied=int((np.abs(delta)<=1e-12).sum())),
        components=dict(improved=int((cd>1e-12).sum()),worse=int((cd< -1e-12).sum()),tied=int((np.abs(cd)<=1e-12).sum()),
                        equal_component_mean_delta=float(cd.mean()),leave_one_component_out_delta_range=[float(min(loo)),float(max(loo))],per_component=components),
        per_scene=rows)

def main():
    summaries={r:read(DIRS[r]/'summary.json.gz') for r in DIRS}
    baseline={r:read(DIRS[r]/'baseline.json.gz') for r in DIRS}
    comp={r['scene_id']:r['component_id'] for r in summaries['044']['selected_result']['metrics']['per_scene']}
    assert comp=={r['scene_id']:r['component_id'] for r in summaries['045']['selected_result']['metrics']['per_scene']}
    saved={};report=dict(scope='Independent saved-score audit; development V, one seed, no new inference',component_source='Returned metric metadata; absent from NPZ',runs={})
    for run in DIRS:
        result={};s=summaries[run]
        for endpoint in ('baseline','selected','final'):
            rows=load(run,endpoint,comp); saved[(run,endpoint)]=rows
            selected=sweep(rows)
            declared=(baseline[run]['selection'] if endpoint=='baseline' else s['selected']['selection'] if endpoint=='selected' else s['final_selection'])
            assert selected['threshold']==declared['threshold']
            assert abs(selected['macro_f1']-declared['macro_f1'])<1e-12
            result[endpoint]=dict(selection=selected,**summarize(rows,selected['threshold']))
            expected=(baseline[run]['selected_threshold'] if endpoint=='baseline' else s[endpoint+'_result'])
            actual=result[endpoint]
            for key in ('tp','fp','fn','tn','macro_f1','micro_f1'):
                assert abs(actual['metrics'][key]-expected['metrics'][key])<1e-12
            assert abs(actual['metrics']['pooled_auc']-expected['pooled_auc'])<1e-12
            for subset,target in (('same_source',expected['metrics']['itsa']),
                                  ('cross_source',expected['metrics']['mstsa']),
                                  ('joint_support',expected['support']['joint_q_support']),
                                  ('no_joint_support',expected['support']['no_joint_q_support'])):
                for key in ('tp','fp','fn','tn','macro_f1','micro_f1'):
                    assert abs(actual['strata'][subset][key]-target[key])<1e-12
            for key in ('group_count','mixed_group_count','anchors_with_wrong_helper','directed_wrong_helper_relations'):
                assert actual['grouping'][key]==expected['grouping'][key]
            for a,b in zip(actual['grouping']['per_scene'],expected['grouping']['per_scene']):
                assert a['membership']==b['membership']
                for key in ('correct_same_identity_pairs','split_pairs','merge_pairs','correct_different_identity_pairs'):
                    assert a[key]==b['partition'][key]
            result[endpoint]['fixed_ols_threshold']=metrics(rows,s['fixed_ols_score_threshold'])
            fixed_groups=[grouping(r,s['fixed_ols_score_threshold']) for r in rows]
            result[endpoint]['fixed_ols_grouping']={k:sum(r[k] for r in fixed_groups)
                for k in fixed_groups[0] if k not in ('scene_id','membership')}
        h=read(DIRS[run]/'history.json.gz')
        result['history_reported']=[dict(update=x['update'],macro_f1=x['selection']['macro_f1'],threshold=x['selection']['threshold']) for x in h if 'selection' in x]
        result['selected_update']=s['selected']['update'];result['optimizer_updates']=s['optimizer_updates']
        result['config']=read(DIRS[run]/'actual_config.json')
        a=read(DIRS[run]/'data_audit.json.gz')
        result['population']=s['population'];result['normalizer_pairs']=a['normalizer']['pair_count']
        result['timing_seconds']={k:s[k] for k in ('preparation_seconds','training_seconds','validation_seconds','elapsed_seconds')}
        result['all_saved_metrics_match_returned_json']=True
        report['runs'][run]=result
    for a,b in zip(saved[('044','baseline')],saved[('045','baseline')]):
        assert all(np.array_equal(a[k],b[k]) for k in ('pairs','labels','scores','node_ids','sources','joint_support_queries'))
    report['baseline_arrays_identical']=True
    ca,cb=(dict(report['runs'][r]['config']) for r in ('044','045'))
    ca.pop('variant');cb.pop('variant');assert ca==cb
    assert read(DIRS['044']/'data_audit.json.gz')['normalizer']==read(DIRS['045']/'data_audit.json.gz')['normalizer']
    report['configs_identical_except_variant']=True
    report['normalizers_identical']=True
    report['comparisons']={}
    for left,right in ((('044','baseline'),('044','selected')),(('044','baseline'),('045','selected')),(('044','selected'),('045','selected')),(('044','final'),('045','final'))):
        ma=report['runs'][left[0]][left[1]];mb=report['runs'][right[0]][right[1]]
        report['comparisons']['_'.join(right)+'_minus_'+'_'.join(left)]=comparison(saved[left],saved[right],ma['threshold'],mb['threshold'],ma,mb)
    selected_threshold=report['runs']['044']['baseline']['threshold'];fixed=summaries['044']['fixed_ols_score_threshold']
    changed=[]
    for r in saved[('044','baseline')]:
        for i in np.flatnonzero((r['scores']>=selected_threshold)!=(r['scores']>=fixed)):
            changed.append(dict(scene_id=r['scene_id'],pair=r['pairs'][i].tolist(),node_ids=r['node_ids'][r['pairs'][i]].tolist(),label=bool(r['labels'][i]),score=float(r['scores'][i]),distance_inverse_transform_m=float(1000*np.expm1(-r['scores'][i]))))
    report['fixed_threshold_boundary']=dict(selected_threshold=selected_threshold,fixed_threshold=fixed,difference=fixed-selected_threshold,changed=changed,
        difference_in_ulps=float(abs((fixed-selected_threshold)/np.spacing(selected_threshold))),
        note='3 ULP score boundary: keep >= and registered thresholds; do not tune away the difference. Original-distance evaluation can differ by the same boundary pair.')
    report['receipt_corrections']=['candidate_count=29009 is 29008 unique score thresholds plus all-reject, not training step0.',
        'Preparation includes all T_core and V feature caching/normalizer, not V only.',
        f"044 final AUC is {report['runs']['044']['final']['metrics']['pooled_auc']}; selected AUC is {report['runs']['044']['selected']['metrics']['pooled_auc']}."]
    # Keep paired scene/component deltas, but avoid duplicating full evaluation
    # rows and memberships already saved in the server summaries.
    def without_scene_rows(value):
        if isinstance(value,dict):
            return {k:without_scene_rows(v) for k,v in value.items() if k!='per_scene'}
        if isinstance(value,list): return [without_scene_rows(v) for v in value]
        return value
    report['runs']=without_scene_rows(report['runs'])
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for run in DIRS:
        for ep in ('baseline','selected','final'):
            x=report['runs'][run][ep]
            print(run,ep,{k:v for k,v in x['metrics'].items() if k!='per_scene'}, {k:v for k,v in x['grouping'].items() if k!='per_scene'})
    for k,v in report['comparisons'].items():
        print(k,{a:b for a,b in v.items() if a not in ('per_scene','components')}, {a:b for a,b in v['components'].items() if a!='per_component'})
    print('boundary',report['fixed_threshold_boundary'])
    print('Saved',OUT)

if __name__=='__main__': main()
