"""Reaggregate returned JSON sufficient statistics; no model, raw ZIP, or GPU work.

Run with normal Python from any directory. Raw-array/weight pairing is outside
this local audit; original endpoint summaries supply global P95/P99.
"""
import json,gzip,math
from pathlib import Path
R=Path(__file__).resolve().parents[4]; D=R/'HandOver/results'
paths={'generic':D/'LEVEL1-QUERY-RELATION-20261025-098A','relation':D/'LEVEL1-QUERY-RELATION-20261025-098B','relation_off':D/'LEVEL1-QUERY-RELATION-20261025-098B','finetune':D/'LEVEL1-FROZEN-ADAPTER-20261024-090C'}
def rd(p):
 return json.load(gzip.open(p,'rt') if str(p).endswith('.gz') else open(p))
S={k:rd(p/('relation_off-summary.json.gz' if k=='relation_off' else 'final-summary.json.gz')) for k,p in paths.items()}
L={k:rd(p/('relation_off-scene-summary.json.gz' if k=='relation_off' else 'final-scene-summary.json.gz')) for k,p in paths.items()}
checks=0; maxerr={'mae':0.,'mse':0.,'rmse':0.}
def exact(a,b):
 global checks
 assert a==b,(a,b);checks+=1
for k,rows in L.items():
 s=S[k];exact(len(rows),300);exact([r['scene_id'] for r in rows],s['scene_ids'])
 cohorts=[(rows,s),([r for r in rows if r['component_id']!=s['largest_component']['component_id']],s['remaining_components'])]
 cohorts += [([r for r in rows if r['component_id']==c],v) for c,v in s['components'].items()]
 for rr,ss in cohorts:
  for region,methods in ss['position_weightings'].items():
   for method,weights in methods.items():
    for w,target in weights.items():
     x=[r['position_weightings'][region][method][w] for r in rr]
     n=sum(z['unit_count'] for z in x);exact(n,target['unit_count'])
     if n==0: continue
     for field in ('mae','mse'):
      val=sum(z[field]*z['unit_count'] for z in x if z['unit_count'])/n
      err=abs(val-target[field]);maxerr[field]=max(maxerr[field],err)
      assert math.isclose(val,target[field],rel_tol=1e-11,abs_tol=1e-6),(k,region,method,w,field,val,target[field]);checks+=1
     rms=math.sqrt(sum(z['mse']*z['unit_count'] for z in x if z['unit_count'])/n);maxerr['rmse']=max(maxerr['rmse'],abs(rms-target['rmse']));assert math.isclose(rms,target['rmse'],rel_tol=1e-11,abs_tol=1e-6);checks+=1
base=L['finetune']
for k,rows in L.items():
 for r,b in zip(rows,base):
  exact(r['scene_id'],b['scene_id']);exact(r['component_id'],b['component_id']);exact(r['fragment_count'],b['fragment_count'])
  for method in r['methods']:
   exact(r['methods'][method]['population'],b['methods'][method]['population'])
   if method!='learned_sequence':exact(r['methods'][method],b['methods'][method])
   for region in r['position_weightings']:
    for w in r['position_weightings'][region][method]:exact(r['position_weightings'][region][method][w]['unit_count'],b['position_weightings'][region][method][w]['unit_count'])
# Exact actual configurations, source and reported exposure matching, not raw tensor/JSONL replay.
F={k:rd(p/'fit-run.json') for k,p in paths.items() if k!='relation_off'}
execution={}
for k in ('generic','relation'):
 p=paths[k]; run=rd(p/'run.json');ex=rd(p/'execution.json');summary=rd(p/'summary.json')
 exact(run['configuration'],ex['configuration']);exact(run['configuration'],rd(R/'configs'/Path(ex['arguments']['config']).name))
 exact(run['configuration'],F[k]['configuration']);exact(run['optimizer_updates'],4000);exact(F[k]['exposure'],F['finetune']['exposure']);exact(run['status'],'COMPLETED')
 for kk in ('test_opened','association_recomputed','groups_recomputed','threshold_reselected','validation_selection','default_promoted'):exact(run[kk],False)
 exact(summary['parameter_count'],109954);exact(summary['trainable_parameter_count'],109954)
 execution[k]={key:run[key] for key in ('status','optimizer_updates','source_teacher_updates','total_stage_updates','elapsed_seconds','completed_endpoints','paired_control','budget_control')}
 execution[k]['successful_exposure']={q:F[k]['exposure'][q] for q in ('successful_updates','sampled_anchors','valid_position_points','nonzero_gradient_updates')}
 execution[k]['training_pairing_reported']=F[k]['paired_control']
 execution[k]['gradient_clipping']=F[k]['gradient_clipping']
 execution[k]['training_seconds']=F[k]['training_seconds']
metrics={k:{'all':s['position_weightings']['all']['learned_sequence'],'outside_largest_component':s['remaining_components']['position_weightings']['all']['learned_sequence'],'regions':{reg:v['learned_sequence']['point_equal'] for reg,v in s['position_weightings'].items()}} for k,s in S.items()}
def comparison(new,old):
 n=S[new]['position_weightings']['all']['learned_sequence'];o=S[old]['position_weightings']['all']['learned_sequence']
 delta={x:n['point_equal'][x]-o['point_equal'][x] for x in ('mae','rmse','p95','p99')}
 def counts(a,b):
  d=[x['position_weightings']['all']['learned_sequence']['point_equal']['mae']-y['position_weightings']['all']['learned_sequence']['point_equal']['mae'] for x,y in zip(a,b)]
  return {'better':sum(v < -1e-6 for v in d),'worse':sum(v > 1e-6 for v in d),'same':sum(abs(v)<=1e-6 for v in d)}
 c=sorted(S[new]['components'])
 return {'relative_mae_reduction':1-n['point_equal']['mae']/o['point_equal']['mae'],'delta':delta,'four_mae_weighting_deltas':{w:n[w]['mae']-o[w]['mae'] for w in n},'scenes':counts(L[new],L[old]),'components':counts([S[new]['components'][x] for x in c],[S[old]['components'][x] for x in c]),'outside_largest_delta_mae':S[new]['remaining_components']['position_weightings']['all']['learned_sequence']['point_equal']['mae']-S[old]['remaining_components']['position_weightings']['all']['learned_sequence']['point_equal']['mae']}
comparisons={n+'__minus__'+o:comparison(n,o) for n,o in [('relation','finetune'),('relation','generic'),('relation','relation_off'),('generic','finetune')]}
result={'scope':'Local independent aggregation of returned endpoint/scene sufficient statistics. No original weights/JSONL/NPZ replay or new inference. Global quantiles taken from returned whole-population summary, not reconstructed from scene quantiles.','checks':checks,'max_aggregation_absolute_difference':maxerr,'execution':execution,'population':S['relation']['methods']['learned_sequence']['full_target_metrics']['all']['position'],'groups':S['relation']['methods']['learned_sequence']['population'],'metrics':metrics,'comparisons':comparisons,'decision':'First seed fails frozen budget and generic gates and same-weight bias-benefit gate. C/D and original CPU cached comparison remain to execute; no model-category or full-peer superiority conclusion.','missing_for_raw_audit':['raw initial/final tensors','actual ordered training JSONL samples','full prediction NPZ/membership arrays','native launcher exit/status files (rc0 currently S2L report)','registered CPU joint table for group-level any_multi cohort and exact no_multi predictions']}
# Preserve the bounded 093/094 returned-summary review alongside 098.
a=rd(D/'LEVEL1-COMPLETION-VALUE-TFM-20261024-093-summary/comparison.json.gz'); f=a['comparison']['endpoints'][-1]
b=rd(D/'LEVEL1-COMPLETION-VALUE-TAM-20261024-094-summary/comparison.json.gz'); c=b['comparison']
result['other_receipts']={'093':{'status':a['status'],'summary_elapsed_seconds':a['elapsed_seconds'],'final_updates':f['update'],'full_position':{k:{m:v for m,v in f['full'][k]['all']['position'].items() if m in ['mae','rmse','p99','intersection_count','available_target_count']} for k in ['learned','state_hold']},'scene_wins':f['scene_wins'],'component_wins':f['component_wins'],'limitation':'Both frozen heads remain very weak. A≈B does not prove completion cannot matter; no convergence or single-cause diagnosis.'},'094':{'status':b['status'],'summary_elapsed_seconds':b['elapsed_seconds'],'association':{k:{x:v[x] for x in ['pair_count','macro_f1','micro_f1','micro_precision','micro_recall','tp','fp','fn','tn']} for k,v in c['association'].items()},'grouping':c['grouping_totals'],'margins':c['margins'],'thresholds':c['thresholds'],'paired_edges':{k:v for k,v in c['paired_edges'].items() if k!='per_scene'},'limitation':'Whole validation selection pipelines, distinct margins/thresholds; not pure completion cause. Anchor identity coverage not physical point coverage.'},'096A':{'latest_receipt':'RUNNING claimed','submitted_utc':'2026-10-25T08:09:10Z','unit':'mtad-11e76d7fa8c3.service','gpu_index':0,'note':'Current Handover receipt only, not live health. No final metrics inspected.'}}
Path(__file__).with_name('audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':checks,'maxerr':maxerr,'comparisons':comparisons},ensure_ascii=False,indent=2))
