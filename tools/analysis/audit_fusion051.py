"""Recompute 051 versus 046 from returned arrays, without inference/data ZIP reads.

Run: .venv/bin/python tools/analysis/audit_fusion051.py
Reuses 046's physical-error implementation; tar members are read in memory.
"""
from pathlib import Path
import importlib.util
import gzip
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('audit046', ROOT/'docs/proposals/repair_before_retraining/046-audit.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
DIRS = {'046': ROOT/'HandOver/results/EVIDENCE-FUSION-20261021-046',
        '051': ROOT/'HandOver/results/EVIDENCE-FUSION-SIGN-20261021-051'}
G, O = old.GROUPS
A = old.ATTRS


def load(run, name):
    with gzip.open(DIRS[run]/name, 'rt') as f:
        return json.load(f)


def arrays(run, name):
    old.RESULT = DIRS[run]
    return old.archive(name)


def flat(x, p=''):
    return {k: v for a,b in x.items() for k,v in (flat(b, p+a+'.').items() if isinstance(b,dict) else [(p+a,b)])}


def concise_metrics(e, m):
    return old.metrics(e,m)


def main():
    runs = {r: load(r, 'run.json.gz') for r in DIRS}
    reports = {r: load(r, 'evaluation-step-1000.json.gz') for r in DIRS}
    fs = {r: flat(runs[r]['configuration']) for r in DIRS}
    config_diff = {k:{r:fs[r].get(k,'<missing>') for r in DIRS} for k in sorted(set(fs['046'])|set(fs['051'])) if fs['046'].get(k) != fs['051'].get(k)}
    out = {'experiment':'051 vs 046', 'source':'returned final/step0 arrays and every full-V evaluation; no inference',
           'config_diff':config_diff, 'checks':{}, 'overall':{}, 'by_source':{}, 'routing':{}, 'single_helper':{},
           'per_scene':[], 'per_component':[], 'curve':[], 'training':{}}
    checks = out['checks']
    checks['actual_configuration_matches_run'] = all(json.loads((DIRS[r]/'actual_config.json').read_text()) == runs[r]['configuration'] for r in DIRS)
    executions={}
    for r in DIRS:
        ex=load(r,'execution.json.gz');prep=ex['preparation']
        executions[r]={k:prep[k] for k in ['training_scene_ids','validation_scene_ids','data_config','reference_longitude_policy','observation_members_read','reference_members_read','reference_loading','test_opened','resampled_with_legacy_scene64']}
        assert ex['configuration']==runs[r]['configuration'] and ex['cpu_threads']==4 and not ex['synthetic'] and ex['native_MTAD_opened']
        assert not prep['test_opened']
        del ex,prep
    checks['execution_preparation_same']=executions['046']==executions['051']
    for key in ['training_scene_ids','validation_scene_ids','selection_policy','sampling','modes','training_samples','fixed_distance_threshold_m']:
        checks[key+'_same'] = runs['046'][key] == runs['051'][key]
        assert checks[key+'_same']
    for run in DIRS:
        assert runs[run]['status']=='COMPLETED' and runs[run]['optimizer_updates']==1000 and not runs[run]['test_opened']
    final = {r:arrays(r, 'predictions-final.tar.gz') for r in DIRS}
    assert set(final['046'])==set(final['051'])==set(reports['051']['scene_ids'])
    checks['nonlearned_arrays_same'] = True
    checks['step0_all_arrays_same'] = True
    checks['step0_anchor_exact'] = True
    z0 = arrays('046','predictions-step-0.tar.gz')
    z1 = arrays('051','predictions-step-0.tar.gz')
    for sid in final['046']:
        for k in z0[sid]:
            eq = lambda x,y: np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
            checks['step0_all_arrays_same'] &= eq(z0[sid][k],z1[sid][k])
            if not any(k.endswith(v) for v in ('_learned','_anchor_weights','_zero_helper_counts')):
                checks['nonlearned_arrays_same'] &= eq(final['046'][sid][k],final['051'][sid][k])
        for r,z in [('046',z0),('051',z1)]:
            for g in [G,O]:
                checks['step0_anchor_exact'] &= np.array_equal(z[sid][g+'_learned'],z[sid][g+'_anchor'])
    assert all(checks.values())
    del z0,z1
    cid = {r['scene_id']:r['component_id'] for r in reports['051']['per_scene']}
    rows=[]
    for sid in sorted(final['051']):
        p=final['051'][sid]; v=p['target_valid']; valid=np.stack((v[...,:2].all(-1),v[...,2],v[...,3]),-1)
        membership=p[G+'_membership']; mixed=np.array([len(set(p['identities'][membership==m]))>1 for m in membership])
        regions={k.removeprefix('region_'):value for k,value in p.items() if k.startswith('region_')}
        regions['mixed_group']=np.broadcast_to(mixed[:,None],valid.shape[:2]);regions['pure_group']=~regions['mixed_group']
        es={r:{g:old.errors(final[r][sid][g+'_learned'],p['target']) for g in [G,O]} for r in DIRS}
        rows.append(dict(sid=sid,cid=cid[sid],p=p,v=valid,regions=regions,e=es))
    cat=lambda fn:np.concatenate([fn(r) for r in rows])
    valid=cat(lambda r:r['v']); source=cat(lambda r:r['p']['sources'])
    regs={k:cat(lambda r:r['regions'][k]) for k in rows[0]['regions']}
    errs={r:{g:cat(lambda row:row['e'][r][g]) for g in [G,O]} for r in DIRS}
    out['population']=dict(scenes=len(rows),components=len(set(cid.values())),anchors=len(valid),queries=int(np.prod(valid.shape[:2])),valid_by_attribute=valid.sum((0,1)).tolist())
    md=0.
    for g in [G,O]:
        out['overall'][g]={}
        for reg,rm in regs.items():
            if g==O and reg in ['mixed_group','pure_group']:continue
            mask=valid&rm[...,None]
            result={r:concise_metrics(errs[r][g],mask) for r in DIRS}
            result['paired']=old.compare(errs['051'][g],errs['046'][g],mask)
            out['overall'][g][reg]=result
            for r in DIRS:
                for a in A:
                    for metric in ['mae','rmse','p95']:
                        md=max(md,abs(result[r][a][metric]-reports[r]['overall'][g]['learned']['regions'][reg][a][metric]))
        out['routing'][g]={r:reports[r]['routing'][g] for r in DIRS}
        out['routing'][g]['delta_rejection_percentage_points']=(100*(np.array(reports['051']['routing'][g]['rejection_fraction_by_attribute'])-np.array(reports['046']['routing'][g]['rejection_fraction_by_attribute']))).tolist()
        w0=cat(lambda row:final['046'][row['sid']][g+'_anchor_weights'])
        w1=cat(lambda row:final['051'][row['sid']][g+'_anchor_weights'])
        wd=np.abs(w1-w0)
        out['routing'][g]['weight_movement']={a:dict(mean_absolute=float(wd[...,i].mean()),p95_absolute=float(np.quantile(wd[...,i],.95)),max_absolute=float(wd[...,i].max()),all_rejected_to_active=int(((w0[...,i]==1)&(w1[...,i]<1)).sum()),active_to_all_rejected=int(((w0[...,i]<1)&(w1[...,i]==1)).sum())) for i,a in enumerate(A)}
        out['routing'][g]['prediction_movement']={a:dict(mean=float(e[...,i].mean()),p95=float(np.quantile(e[...,i],.95)),max=float(e[...,i].max())) for i,a in enumerate(A) for e in [cat(lambda row:old.errors(final['051'][row['sid']][g+'_learned'],final['046'][row['sid']][g+'_learned']))]}
        for r in DIRS:
            weights=cat(lambda row:final[r][row['sid']][g+'_anchor_weights'])
            counts=cat(lambda row:row['p'][g+'_helper_counts'])
            rejected=cat(lambda row:final[r][row['sid']][g+'_zero_helper_counts'])
            assert np.array_equal(rejected==counts[:,None,None],weights==1)
            np.testing.assert_allclose(rejected.sum((0,1))/(counts.sum()*64),reports[r]['routing'][g]['rejection_fraction_by_attribute'],rtol=0,atol=0)
            out['routing'][g][r]['eligible_mean_anchor_weight']=weights[counts>0].mean((0,1)).tolist()
    out['max_metric_absolute_difference_vs_reports']=md;assert md<1e-7
    for s in sorted(set(source)):
        out['by_source'][s]={}
        for reg,rm in regs.items():
            mask=valid&rm[...,None]&(source==s)[:,None,None]
            out['by_source'][s][reg]={r:concise_metrics(errs[r][G],mask) for r in DIRS}
            out['by_source'][s][reg]['paired']=old.compare(errs['051'][G],errs['046'][G],mask)
    # Existing saved arrays expose only aggregate helper weights. Exact per-helper
    # decision changes are identifiable only in single-helper groups.
    for g in [G,O]:
        pieces=[]; potentials={r:[] for r in DIRS}
        for row in rows:
            p=row['p']; n,q=p['target'].shape[:2]; m=p[g+'_membership'];counts=p[g+'_helper_counts']
            for i in range(n):
                helpers=np.flatnonzero((m==m[i])&(np.arange(n)!=i))
                if not len(helpers):continue
                he=old.errors(p['native_states'][helpers],p['target'][i]);ae=old.errors(p['native_states'][i],p['target'][i])
                better=(he<ae-1e-8).any(0)
                for r in DIRS:
                    aw=final[r][row['sid']][g+'_anchor_weights'][i]
                    potentials[r].append(np.stack((row['v'][i]&better,row['v'][i]&better&(aw==1)),0))
                if len(helpers)==1:
                    accepted={r:final[r][row['sid']][g+'_anchor_weights'][i]<1 for r in DIRS}
                    pieces.append(dict(d=he[0]-ae,v=row['v'][i],accepted=accepted,same=p['identities'][i]==p['identities'][helpers[0]]))
        out['routing'][g]['better_helper_available']={r:np.stack(potentials[r]).sum((0,2)).tolist() for r in DIRS}
        out['single_helper'][g]={}
        for i,a in enumerate(A):
            v=np.concatenate([x['v'][:,i] for x in pieces]);d=np.concatenate([x['d'][:,i] for x in pieces]);helpful=d < -1e-8;harmful=d > 1e-8
            ac={r:np.concatenate([x['accepted'][r][:,i] for x in pieces]) for r in DIRS}; n=int(v.sum())
            ent=dict(valid_queries=n,helpful_queries=int((v&helpful).sum()),harmful_queries=int((v&harmful).sum()))
            for r in DIRS:
                ent[r]=dict(accepted=int((v&ac[r]).sum()),helpful_rejected=int((v&helpful&~ac[r]).sum()),harmful_accepted=int((v&harmful&ac[r]).sum()),helpful_rejection_rate=float((v&helpful&~ac[r]).sum()/(v&helpful).sum()),harmful_acceptance_rate=float((v&harmful&ac[r]).sum()/(v&harmful).sum()))
            ent['flips']=dict(reject_to_accept=int((v&~ac['046']&ac['051']).sum()),accept_to_reject=int((v&ac['046']&~ac['051']).sum()),helpful_rescued=int((v&helpful&~ac['046']&ac['051']).sum()),helpful_lost=int((v&helpful&ac['046']&~ac['051']).sum()),harmful_blocked=int((v&harmful&ac['046']&~ac['051']).sum()),harmful_added=int((v&harmful&~ac['046']&ac['051']).sum()))
            out['single_helper'][g][a]=ent
    # Paired scene and component effects; 17 component bootstrap is descriptive,
    # conditional on the trained seed, not uncertainty across retraining.
    for row in rows:
        out['per_scene'].append(dict(scene_id=row['sid'],component_id=row['cid'],comparison=old.compare(row['e']['051'][G],row['e']['046'][G],row['v'])))
    sums=[];counts=[]
    for component in sorted(set(cid.values())):
        rs=[r for r in rows if r['cid']==component];v=np.concatenate([r['v'] for r in rs]);e0=np.concatenate([r['e']['046'][G] for r in rs]);e1=np.concatenate([r['e']['051'][G] for r in rs])
        out['per_component'].append(dict(component_id=component,scenes=len(rs),comparison=old.compare(e1,e0,v)))
        sums.append(np.where(v,e1-e0,0).sum((0,1)));counts.append(v.sum((0,1)))
    sums=np.array(sums);counts=np.array(counts);rng=np.random.default_rng(20261021);idx=rng.integers(0,len(sums),(5000,len(sums)))
    boot=sums[idx].sum(1)/counts[idx].sum(1)
    out['component_bootstrap']={a:dict(paired_micro_delta=float(sums[:,i].sum()/counts[:,i].sum()),percentile_95=np.quantile(boot[:,i],[.025,.975]).tolist(),resamples=5000) for i,a in enumerate(A)}
    out['unit_summary']={unit:{a:dict(better=sum(x['comparison'][a]['delta_mae'] < -1e-8 for x in out[unit]),worse=sum(x['comparison'][a]['delta_mae'] > 1e-8 for x in out[unit]),equal_unit_mean_delta=float(np.mean([x['comparison'][a]['delta_mae'] for x in out[unit]])),min_delta=float(min(x['comparison'][a]['delta_mae'] for x in out[unit])),max_delta=float(max(x['comparison'][a]['delta_mae'] for x in out[unit]))) for a in A} for unit in ['per_scene','per_component']}
    for step in range(0,1001,100):
        rr={r:load(r,f'evaluation-step-{step}.json.gz') for r in DIRS}
        out['curve'].append(dict(step=step,**{r:dict(**{a:rr[r]['overall'][G]['learned']['regions']['all'][a]['mae'] for a in A},rejection=rr[r]['routing'][G]['rejection_fraction_by_attribute']) for r in DIRS}))
    for r in DIRS:
        with gzip.open(DIRS[r]/'training.jsonl.gz','rt') as f:train=[json.loads(x) for x in f]
        assert [x['step'] for x in train]==list(range(1,1001))
        keys=[k for k in ['total','original_total','utility','direct','utility_sign','utility_sign_weighted','gradient_norm'] if k in train[0]]
        out['training'][r]=dict(first=train[0],last=train[-1],bins=[dict(start=i+1,end=i+100,**{k:float(np.mean([x[k] for x in train[i:i+100]])) for k in keys}) for i in range(0,1000,100)],by_mode={mode:{k:float(np.mean([x[k] for x in train if x['mode']==mode])) for k in keys} for mode in ['clean','clean_plus_wrong','wrong_only']},fraction_gradient_norm_above_one=float(np.mean([x['gradient_norm']>1 for x in train])))
    out['limits']=['One trained seed and repeatedly used V; no seed-level causal uncertainty.',
       'No individual helper logits/weights or per-attribute training losses/gradients saved; aggregate loss share cannot diagnose gradient dominance.',
       '051 uses original OLS grouping, not 045/048 groups; single-helper correctness is diagnostic, not complete multi-helper identification.',
       'A helper worse on its own can help as part of a mixture; sign error is not a direct proof of fusion harm.']
    dest=ROOT/'docs/proposals/repair_before_retraining/051-audit.json'
    dest.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['checks','population','max_metric_absolute_difference_vs_reports','unit_summary','component_bootstrap','single_helper','curve']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
