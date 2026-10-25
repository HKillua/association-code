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

def audit_completed_seed23(existing):
    """Check newly returned CPU table, without rerunning any model or old table."""
    directory = D/'LEVEL1-QUERY-RELATION-20261025-098-seed23-comparison'
    summary = rd(directory/'summary.json.gz')
    rows = rd(directory/'scene-summary.json.gz')
    run = rd(directory/'run.json')
    checks = 0
    max_delta = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0}
    def same(a, b):
        nonlocal checks
        assert a == b, (a, b)
        checks += 1
    same(run['status'], 'COMPLETED')
    for key in ('model_forwards', 'optimizer_updates'):
        same(run[key], 0)
        same(summary[key], 0)
    same(run['test_opened'], False)
    same(summary['test_opened'], False)
    same(summary['seed'], 20261023)
    same(len(rows), 300)
    same([row['scene_ids'][0] for row in rows], summary['scene_ids'])
    same(summary['population'], existing['groups'])
    same(summary['position_population']['covered_target_points'], existing['population']['intersection_count'])
    same(summary['position_population']['available_target_points'], existing['population']['available_target_count'])
    for name, endpoint in existing['metrics'].items():
        same(summary['position_weightings']['all'][name], endpoint['all'])
        same(summary['remaining_components']['position_weightings']['all'][name], endpoint['outside_largest_component'])
        for region, values in endpoint['regions'].items():
            same(summary['position_weightings'][region][name]['point_equal'], values)
    # Reaggregate only the newly returned table's scene sufficient statistics.
    for section in ('position_weightings', 'helper_cohorts', 'group_cohorts'):
        for stratum, methods in summary[section].items():
            for method, weights in methods.items():
                for weight, expected in weights.items():
                    pieces = [row[section][stratum][method][weight] for row in rows]
                    count = sum(piece['unit_count'] for piece in pieces)
                    same(count, expected['unit_count'])
                    if not count:
                        continue
                    for metric in ('mae', 'mse'):
                        value = sum(piece[metric]*piece['unit_count'] for piece in pieces if piece['unit_count'])/count
                        max_delta[metric] = max(max_delta[metric], abs(value-expected[metric]))
                        assert math.isclose(value, expected[metric], rel_tol=1e-11, abs_tol=1e-6)
                        checks += 1
                    value = math.sqrt(sum(piece['mse']*piece['unit_count'] for piece in pieces if piece['unit_count'])/count)
                    max_delta['rmse'] = max(max_delta['rmse'], abs(value-expected['rmse']))
                    assert math.isclose(value, expected['rmse'], rel_tol=1e-11, abs_tol=1e-6)
                    checks += 1
    for key, value in summary['group_relation_population'].items():
        same(sum(row['group_relation_population'][key] for row in rows), value)
    same(all(row['no_multi_prediction_exact'] for row in rows), True)
    same(summary['no_multi_prediction_exact'], True)
    pairing = summary['pairing']
    same(pairing['status'], 'PAIR_VERIFIED')
    same(pairing['generic_relation_initial_parameters_exact'], True)
    same(pairing['generic_relation_parameter_count_equal'], True)
    for arm in ('generic', 'relation'):
        for key in ('actual_successful_batches_exact', 'sampling_exposure_exact',
                    'source_final_parameters_exact', 'initial_sampler_rng_exact',
                    'final_sampler_rng_exact', 'final_parameters_changed'):
            same(pairing['arms'][arm][key], True)
        same(pairing['arms'][arm]['matched_updates'], 4000)
    same(summary['initial_output_source_final_exact'], True)
    # Re-evaluate frozen screen from already independently aggregated figures.
    screen = summary['scientific_screen']
    for key, control, required, majority in (('versus_budget', 'finetune', .03, True), ('versus_generic', 'generic', .01, False)):
        comparison = existing['comparisons']['relation__minus__'+control]
        same(screen[key]['deltas'], comparison['delta'])
        same(screen[key]['relative_mae_reduction'], comparison['relative_mae_reduction'])
        same(screen[key]['required_reduction'], required)
        same(screen[key]['improved_scenes'], comparison['scenes']['better'])
        same(screen[key]['improved_components'], comparison['components']['better'])
        weights_ok = all(value <= 1e-6 for value in comparison['four_mae_weighting_deltas'].values())
        outside_ok = comparison['outside_largest_delta_mae'] <= 1e-6
        passed = comparison['relative_mae_reduction'] >= required and comparison['delta']['mae'] < -1e-6 and comparison['delta']['rmse'] <= 1e-6 and comparison['delta']['p99'] <= 1e-6 and weights_ok and outside_ok
        if majority:
            passed = passed and comparison['scenes']['better'] > 150 and comparison['components']['better'] > 19
        same(screen[key]['all_mae_weightings_nonworse'], weights_ok)
        same(screen[key]['outside_largest_component_nonworse'], outside_ok)
        same(screen[key]['passed'], passed)
    intervention = screen['fixed_weight_bias_intervention']
    same(intervention['delta_mae_m'], existing['comparisons']['relation__minus__relation_off']['delta']['mae'])
    multi_points = summary['group_cohorts']['any_multi_query']['relation']['point_equal']['unit_count']
    same(intervention['groups_with_multi_query_valid_points'], multi_points)
    same(intervention['passed'], intervention['delta_mae_m'] < -1e-6 and multi_points > 0 and summary['no_multi_prediction_exact'])
    same(screen['seed_screen_passed'], False)
    same(screen['outcome'], 'CURRENT_RELATION_RECIPE_NOT_ACCEPTED')
    existing['seed23_cpu_table'] = {
        'scope': 'New formal server-side raw-artifact pairing and cached comparison reviewed; local scene-summary reaggregation and original frozen gate recomputation only.',
        'source_directory': str(directory.relative_to(R)),
        'run': run,
        'checks': checks,
        'max_aggregation_absolute_difference': max_delta,
        'endpoint_aggregates_equal_prior_local_audit': True,
        'pairing_report': {key: value for key, value in pairing.items() if key not in ('budget_control', 'arms')},
        'arm_pairing_report': {name: {key: value for key, value in arm.items() if key != 'helper_support_exposure'} for name, arm in pairing['arms'].items()},
        'initial_output_source_final_exact': summary['initial_output_source_final_exact'],
        'no_multi_prediction_exact': summary['no_multi_prediction_exact'],
        'group_relation_population': summary['group_relation_population'],
        'group_cohorts_relation': {key: value['relation']['point_equal'] for key, value in summary['group_cohorts'].items()},
        'scientific_screen': screen,
        'bias_intervention_point_error_differences': summary['relative_harm']['all']['relation__minus__relation_off'],
        'code_checked': ['tools/analysis/compare_query_relation.py:verify_pairing', 'tools/analysis/compare_query_relation.py:scene_pair', 'tools/analysis/compare_query_relation.py:scientific_screen'],
        'evidence_boundary': 'Server comparator loads actual checkpoints/ordered successful sample journals and cached input/truth/membership/prediction arrays before returning PAIR_VERIFIED. Local audit verifies returned summaries and checked code, but does not possess or independently replay raw tensors/JSONL/NPZ; no new GPU, model forward, or original ZIP read.',
        's2l_interpretation_corrections': ['Generic does not beat the strong budget control; no stronger-budget-baseline signal.', 'A final fixed-weight bias switch does not rule out effects mediated through training, so the B-A training difference cannot be declared unrelated to all relationship-bias effects.']}
    existing['missing_for_raw_audit'] = [
        'raw initial/final tensors remain server-side; formal server verifier completed',
        'actual ordered training JSONL remains server-side; formal server verifier completed',
        'full prediction NPZ/membership arrays remain server-side; formal server verifier completed',
        'native launcher exit/status files (exit0 currently S2L report)']
    existing['decision'] = 'First seed fails frozen budget, generic and fixed-weight bias-benefit gates, confirmed by completed seed23 CPU table. Close expansion of this relation recipe; preserve running C/D and remaining original CPU tables. No model-category or full-peer superiority conclusion.'
    return existing

result = audit_completed_seed23(result)
Path(__file__).with_name('audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':checks,'maxerr':maxerr,'comparisons':comparisons},ensure_ascii=False,indent=2))
