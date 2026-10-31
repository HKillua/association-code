"""Independent saved-output/population/J audit for165, allowing coverage changes.

Reuses only NumPy metric/record checks, not grouping or source-risk producers.
Does not certify truth, scoring heads, or causal physical mechanisms.
"""
from __future__ import annotations
import argparse,sys,time,resource
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.analysis import audit_shared_control_arrays as core
from tools.analysis.audit_temporal_source_outputs import load


def audit(run,out,*,extra_scene=None,extra_summary=None):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();checks=core.Checks();result=dict(status='RUNNING',run=str(run))
    try:
        record=core.read(run/'run.json');checks.exact(record['status']=='COMPLETED' and record['exit_code']==0,'terminal science');native=Path(record['native_run']);names=record['methods'];result['splits']={}
        for split,ids,fn,rf in [('T_core',record['training_scene_ids'],'T-summary.json','T-scene-summary.json.gz'),('V_select',record['scene_ids'],'summary.json','scene-summary.json.gz')]:
            summary=core.read(run/fn);rows=core.read(run/rf);items=[];pair_items=[];extra_items=[];checks.exact([r['scene_id'] for r in rows]==ids,'complete order')
            for sid,row in zip(ids,rows):
                z=load(run/'private'/f'states-{split}-{sid}.npz');saved={n:{k[len(n)+1:]:v for k,v in z.items() if k.startswith(n+'_')} for n in names};ev=load(native/'private/native-evaluation'/split/f'scene-{sid}.npz');nt=len(ev['target_ids']);item={}
                for n in names:
                    checks.array(saved[n]['target_ids'],ev['target_ids'],'full native registry');checks.array(saved[n]['aligned_truth'][:nt],ev['target_states'],'native truth');checks.array(saved[n]['aligned_truth_valid'][:nt],ev['target_valid'],'native validity')
                    g=saved[n];groups=[g['group_node_indices'][a:b] for a,b in zip(g['group_offsets'][:-1],g['group_offsets'][1:])];reg={str(v):i for i,v in enumerate(ev['target_ids'])}
                    checks.array(g['group_anchor_target_indices'],np.asarray([reg[str(ev['identities'][v[0]])] for v in groups]),'first-anchor correspondence')
                    single=core.inspect_scene({n:g},dict(methods={n:row['methods'][n]},pairwise={}),[n],checks,expected_split=split)[n];item[n]=single
                pairs={}
                for key,pair in row['pairwise'].items():
                    left,right=pair['baseline_method'],pair['candidate_method'];a,b=saved[left],saved[right];tv=ev['target_valid'][...,:2].all(-1);pa=a['aligned_prediction_valid'][:nt,:,:2].all(-1);pb=b['aligned_prediction_valid'][:nt,:,:2].all(-1);common=pa&pb;use=tv&common
                    ea=core.position(a['aligned_prediction'][:nt],ev['target_states'],tv&pa);eb=core.position(b['aligned_prediction'][:nt],ev['target_states'],tv&pb);values={left:ea[use],right:eb[use]};metrics={n:core.metrics(tv,common,err)[0] for n,err in [(left,ea),(right,eb)]}
                    for side,n in [('A0',left),('candidate',right)]:checks.fields(metrics[n],pair['common_valid_position'][side],core.COUNTS+core.RATES+('mae','mse','rmse'),'scene pair')
                    delta=values[right]-values[left];checks.value(float(delta.mean()) if len(delta) else None,pair['strata']['all']['candidate_minus_A0']['mean_error_change'],'mean_error_change','pair')
                    transitions=dict(gained_target_points=int((tv&pb&~pa).sum()),lost_target_points=int((tv&pa&~pb).sum()),rescued_target_count=int(((tv&pb).any(1)&~(tv&pa).any(1)).sum()),lost_target_count=int(((tv&pa).any(1)&~(tv&pb).any(1)).sum()))
                    checks.fields(transitions,pair['coverage_transitions'],tuple(transitions),'variable coverage')
                    pairs[key]=dict(metrics=metrics,values=values,delta=delta,transitions=transitions)
                if extra_scene is not None:
                    extra_items.append(extra_scene(run,record,split,sid,row,saved,checks))
                else:
                    q=load(run/'private'/f'partitions-{split}-{sid}.npz');scores=q['scores'];edges=q['pairs'];weights=scores-1.7320103112387182;n=len(q['WS']);w=np.zeros((n,n));w[edges[:,0],edges[:,1]]=weights;w+=w.T;labels=q['WS'].copy();J=float(weights[labels[edges[:,0]]==labels[edges[:,1]]].sum());initial=J
                    moves=core.read(run/'private'/f'moves-{split}-{sid}.json')
                    for move in moves:
                        node,old,new=[move[k] for k in ['node','from_label','to_label']];checks.exact(int(labels[node])==old,'move source group');gain=float(w[node,labels==new].sum()-w[node,labels==old].sum());checks.exact(gain>1e-10,'positive true gain');checks.value(gain,move['gain'],'delta','move');labels[node]=new;J+=gain;checks.value(J,move['J'],'delta','trajectory')
                    checks.array(labels[:,None]==labels[None,:],q['REF'][:,None]==q['REF'][None,:],'final relation')
                    final=float(weights[q['REF'][edges[:,0]]==q['REF'][edges[:,1]]].sum());checks.value(initial,row['refinement']['initial_J'],'delta','initial J');checks.value(final,row['refinement']['final_J'],'delta','final J');checks.exact(final>=initial-1e-6 and len(moves)<=4*n,'bounded monotonic J')
                items.append(item);pair_items.append(pairs)
            cs=sorted(set(r['component_id'] for r in rows));largest=summary['largest_component_id'];subsets=[('all_components',list(range(len(rows)))),('without_largest_component',[i for i,r in enumerate(rows) if r['component_id']!=largest])]+[('component:'+c,[i for i,r in enumerate(rows) if r['component_id']==c]) for c in cs]
            for label,indices in subsets:
                sub=summary['by_component'][label.split(':')[1]] if label.startswith('component:') else summary[label]
                for n in names:
                    projected=None if sub is None else dict(sub,methods={n:sub['methods'][n]},pairwise={})
                    core.inspect_summary(projected,indices,items,[dict(r,pairwise={}) for r in rows],[n],checks,label+'/'+n)
                if sub is None:continue
                if extra_summary is not None:extra_summary(sub,indices,extra_items,checks)
                for key,pair in sub['pairwise'].items():
                    for side,n in [('A0',pair['baseline_method']),('candidate',pair['candidate_method'])]:
                        expected=core.pooled([(pair_items[i][key]['metrics'][n],pair_items[i][key]['values'][n]) for i in indices],scene_level=True)
                        checks.fields(expected,pair['common_valid_position'][side],core.COUNTS+core.RATES+core.STATS+('scene_count','defined_mae_scene_count','scene_macro_mae'),'pooled variable pair')
                    for field in pair['coverage_transitions']:checks.value(sum(pair_items[i][key]['transitions'][field] for i in indices),pair['coverage_transitions'][field],field,'pooled transition')
                    d=np.concatenate([pair_items[i][key]['delta'] for i in indices]);checks.value(int((d>1000).sum()),pair['actual_harm']['new_harm_over_1km'],'harm_count','pooled harm');checks.value(float(max(0.,d.max())) if len(d) else 0.,pair['actual_harm']['max_error_increase_m'],'harm_max','pooled harm')
            for key,b in summary['primary_bootstrap'].items():
                sums=np.asarray([sum(pair_items[i][key]['delta'].sum() for i,r in enumerate(rows) if r['component_id']==c) for c in cs]);count=np.asarray([sum(len(pair_items[i][key]['delta']) for i,r in enumerate(rows) if r['component_id']==c) for c in cs]);draw=np.random.default_rng(20).integers(len(cs),size=(2000,len(cs)));interval=np.quantile(sums[draw].sum(1)/count[draw].sum(1),[.025,.975]);checks.value(float(sums.sum()/count.sum()),b['point_estimate_m'],'delta','primary')
                for value,expected in zip(interval,b['percentile95_m']):checks.value(float(value),expected,'bootstrap','primary')
            result['splits'][split]=dict(scenes=len(ids),methods=len(names))
        result.update(status='COMPLETED',exit_code=0,scope='independent saved main position/population/coverage/tails/components/primary CI; score-J for165 or explicit extension; not head inference/physical truth/author validity or all attribute metrics')
    except BaseException as error:result.update(status='FAILED',exit_code=1,error=f'{type(error).__name__}: {error}');raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,checks=vars(checks),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);core_path=out if result['status']=='COMPLETED' else out.with_suffix('.failed.json');core_path.write_text(__import__('json').dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run.resolve(),a.output.resolve())
