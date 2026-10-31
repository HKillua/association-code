"""165: same signed evidence, split/move refinement, real regrouped physics."""
from __future__ import annotations
import argparse,sys,time
from itertools import combinations
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.evaluation import source_risk as risk
from association.evaluation.partition_refinement import refine_partition
from association.evaluation.local_source_evidence import predict_residual
from association.evaluation.fusion_controls import ordinary_fusion
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source
from tools.analysis import static_partition_pilot as partition
from tools.analysis.temporal_source_pilot import physical,recover,add_harm,summarize
from tools.analysis.source_coverage_pilot import role_diagnostic

METHODS=tuple(p+'_'+f for p in ('WS','CL','REF') for f in ('MEAN','NN'))
PRIMARY=('REF_NN_vs_WS_NN','REF_MEAN_vs_WS_MEAN','CL_NN_vs_WS_NN','CL_MEAN_vs_WS_MEAN','WS_NN_vs_WS_MEAN','CL_NN_vs_CL_MEAN','REF_NN_vs_REF_MEAN')

def predict_partitions(motion, pairs, scores, original, model, old_nn, device):
    n=len(motion['states']);ws,_,_=partition.agglomerate(n,pairs,scores,'WS')
    common.old._same(ws,original,'original WS from original scores')
    cl,_,_=partition.agglomerate(n,pairs,scores,'CL')
    ref,stats,trace=refine_partition(n,pairs,scores,ws,threshold=partition.THRESHOLD)
    memberships={'WS':ws,'CL':cl,'REF':ref};packs={};groups={};raw={};scores_by={}
    for name,membership in memberships.items():
        g=common.old.groups_from_membership(membership,n);groups[name]=g
        mean=ordinary_fusion(motion['states'],motion['nearest_seconds'],motion['support'],g)['predictions']['mean_supported'][[v[0] for v in g]]
        pack=risk.build_source_features(**{k:motion[k] for k in source.MOTION_KEYS},groups=g);packs[name]=pack
        new_scores=predict_residual(model,pack['features'].reshape(-1,10),device).reshape(pack['available'].shape)
        scores_by[name]=new_scores
        raw[name+'_MEAN']=mean;raw[name+'_NN']=recover(motion,g,pack,new_scores,mean)
    # Preserve the real frozen159 deployment as the strong control.
    if old_nn is not None:
        common.old._same(raw['WS_NN'],old_nn,'frozen159 WS NN physical output')
    return memberships,groups,packs,scores_by,raw,stats,trace


def evaluate(m,ev,groups,packs,raw,split):
    outputs={n:emit_group_outputs(v,groups[n.split('_')[0]],node_count=len(m['states'])) for n,v in raw.items()}
    reports,art=evaluate_fusion_methods(outputs,prediction_speed_units={n:'kn' for n in raw},node_identities=ev['identities'],
        target_ids=ev['target_ids'],target_states=ev['target_states'],target_valid=ev['target_valid'],support=m['support'],
        nearest_seconds=m['nearest_seconds'],target_speed_unit='mps',split=split)
    fragment=dict(query_seconds=m['query_seconds'],identities=ev['identities'],support=m['support'],nearest_seconds=m['nearest_seconds'])
    regions=common.old.fixed_regions(fragment,ev['target_ids']);methods={};arr={}
    for n in METHODS:methods[n],arr[n]=common.qc.describe(reports[n],art[n],fragment,regions)
    pairs={};pa={}
    for left,right in combinations(METHODS,2):
        key=right+'_vs_'+left;pairs[key],pa[key],_=common.old.compare_pair(art[left],art[right],regions)
        pairs[key].update(baseline_method=left,candidate_method=right,side_labels=dict(A0=left,candidate=right))
    row=dict(methods=methods,pairwise=pairs);arrays=dict(methods=arr,pairs=pa);add_harm(row,arrays)
    return row,arrays,art


def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')}
    out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();start=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-TRUTH-FREE-PARTITION-REFINEMENT-20261031',plan_version='1.0',methods=METHODS,
        started_at=datetime.now(timezone.utc).isoformat(),optimizer_updates=0,test_opened=False,reference_used_for_prediction=False,
        expression_run=str(a.expression_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=common.old.frozen.read(paths['manifest']);tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        score_path=Path(common.old.frozen.read(paths['partition_run']/'run.json')['source_association'])/'final-scores.npz'
        check(training_scene_ids=tids,scene_ids=vids,association_score_cache=str(score_path))
        with np.load(score_path,allow_pickle=False) as stored:
            for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
                rows=[];arrays=[]
                for sid in ids:
                    m,g,mean,pack,membership=physical(paths,split,sid)
                    if split=='T_core':
                        pairs=common.load_npz(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz',('pairs',))['pairs']
                        scores=common.load_npz(paths['value_run']/'private'/f'T-features-{sid}.npz',('scores',))['scores']
                    else:
                        pairs,scores=stored[sid+'_pairs'],stored[sid+'_scores'];common.old._same(m['node_ids'],stored[sid+'_node_ids'],'V raw score nodes')
                    fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full'
                    model=torch.load(a.expression_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold']
                    oldraw=common.load_npz(a.expression_run/'private'/f'raw-{split}-{sid}.npz',('WS_T128_RESIDUAL_HARD',))['WS_T128_RESIDUAL_HARD']
                    began=time.monotonic();memberships,groups,packs,nn,raw,stats,trace=predict_partitions(m,pairs,scores,membership,model,oldraw,a.device)
                    stats['prediction_seconds']=time.monotonic()-began
                    np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                    np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',pairs=pairs,scores=scores,**memberships)
                    common.old.frozen.write(out/'private'/f'moves-{split}-{sid}.json',trace)
                    ev=common.evaluation(paths['native_run'],split,sid,m)
                    row,arr,art=evaluate(m,ev,groups,packs,raw,split)
                    record=dict(scene_id=sid,component_id=str(manifest['components'][sid]),node_ids=m['node_ids'],sources=m['sources'],pairs=pairs,
                        labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]],scores=scores,joint_support_queries=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).sum(1))
                    relation={'raw':partition.frozen.evaluate_scenes([record],partition.THRESHOLD),'decoded':{}}
                    diagnostics={}
                    for p in memberships:
                        co=memberships[p][pairs[:,0]]==memberships[p][pairs[:,1]]
                        relation['decoded'][p]=partition.frozen.evaluate_scenes([dict(record,scores=co.astype(float))],.5)
                        labels=source.training_labels(packs[p],groups[p],ev)
                        diagnostics[p]=dict(source.score_diagnostic(packs[p],labels,nn[p]),support_counts=source.support_counts(packs[p]),role=role_diagnostic(packs[p],labels,nn[p]))
                    row.update(scene_id=sid,component_id=str(manifest['components'][sid]),refinement=stats,association=relation,source_diagnostics=diagnostics)
                    common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row)
                    rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
                result=common.summarize(rows,arrays,method_names=METHODS,primary_pairs=PRIMARY)
                for target,indices in [(result['all_components'],list(range(len(rows)))),(result['without_largest_component'],[i for i,r in enumerate(rows) if r['component_id']!=result['largest_component_id']])]+[(result['by_component'][c],[i for i,r in enumerate(rows) if r['component_id']==c]) for c in result['by_component']]:
                    if target is None:continue
                    for pair in target['pairwise']:
                        hs=[rows[i]['pairwise'][pair]['actual_harm'] for i in indices]
                        target['pairwise'][pair]['actual_harm']=dict(new_harm_over_1km=sum(h['new_harm_over_1km'] for h in hs),max_error_increase_m=max(h['max_error_increase_m'] or 0. for h in hs))
                    target['association']={}
                    for label in ['raw','WS','CL','REF']:
                        target['association'][label]={scope:partition.frozen._aggregate([partition.frozen.metric_rows(rows[i]['association']['raw'] if label=='raw' else rows[i]['association']['decoded'][label])[scope] for i in indices]) for scope in partition.frozen.SCOPES}
                    target['refinement']=dict(changed_scenes=sum(rows[i]['refinement']['move_count']>0 for i in indices),moves=sum(rows[i]['refinement']['move_count'] for i in indices),budget_exhausted_scenes=sum(not rows[i]['refinement']['locally_optimal'] for i in indices),J_gain=sum(rows[i]['refinement']['final_J']-rows[i]['refinement']['initial_J'] for i in indices))
                result.update(status='COMPLETED',idea_id=state['idea_id']);common.old.frozen.write(out/fn,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','expression-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
