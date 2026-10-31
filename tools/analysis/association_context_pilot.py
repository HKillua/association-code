"""168/169: controlled front-end context or anchor/support-aware regrouping."""
from __future__ import annotations
import argparse,sys,time
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from association.models.association_context import EdgeResidual,context_cache,edge_inputs,fit_context_normalizer,normalize
from association.models.evidence_association import features_from_motion
from association.training.evidence_association import load_checkpoint,normalized_features
from association.training.sampling import BalancedScenePairSampler
from association.evaluation.partition_refinement import refine_partition
from association.evaluation import source_risk as risk
from association.evaluation.local_source_evidence import predict_residual
from association.evaluation.fusion_controls import ordinary_fusion
from tools.analysis import observation_common as common
from tools.analysis import source_risk_pilot as source
from tools.analysis import static_partition_pilot as partition
from tools.analysis.temporal_source_pilot import physical,recover,add_harm
from tools.analysis.truth_free_partition_pilot import evaluate


def prepare_edges(teacher,normalizer,m,pairs,baseline=None):
    hidden=np.empty((len(pairs),64),np.float32);scores=np.empty(len(pairs),np.float64)
    with torch.inference_mode():
        for start in range(0,len(pairs),4096):
            stop=min(start+4096,len(pairs));features,base=features_from_motion(m,pairs[start:stop])
            x=torch.from_numpy(normalized_features(features.astype(np.float32),normalizer))
            hidden[start:stop]=teacher.network[:-1](x).numpy()
            scores[start:stop]=teacher(x,torch.as_tensor(base,dtype=torch.float64)).numpy()
    gap=0. if baseline is None or not len(pairs) else float(np.max(np.abs(scores-baseline)))
    if gap>1e-4:raise ValueError(f'frozen056 teacher binding gap {gap}')
    scores=scores if baseline is None else np.asarray(baseline,np.float64)
    cache=context_cache(hidden,pairs,scores,len(m['states']),partition.THRESHOLD)
    cache.update(scores=scores,teacher_binding_max_abs=gap)
    return cache


def train_heads(caches,device,check,steps=1000):
    normalizers={k:fit_context_normalizer(caches,k) for k in ('OWN','CTX')}
    torch.manual_seed(20);heads={'OWN':EdgeResidual().to(device),'CTX':EdgeResidual().to(device)}
    heads['CTX'].load_state_dict(heads['OWN'].state_dict())
    opts={k:torch.optim.Adam(v.parameters(),lr=.001) for k,v in heads.items()}
    sampler=BalancedScenePairSampler(caches,20);curve=[]
    for step in range(steps):
        batch=sampler.sample(256);losses={}
        for k,head in heads.items():
            x=np.concatenate([normalize(edge_inputs(caches[i],idx,k),normalizers[k]) for i,idx in batch])
            y=np.concatenate([caches[i]['labels'][idx] for i,idx in batch])
            base=np.concatenate([caches[i]['scores'][idx] for i,idx in batch])
            opts[k].zero_grad(set_to_none=True)
            logits=torch.as_tensor(base,device=device)+head(torch.as_tensor(x,device=device)).double()
            loss=F.binary_cross_entropy_with_logits(logits,torch.as_tensor(y,device=device,dtype=torch.float64))
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite BCE')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            opts[k].step();losses[k]=dict(bce=float(loss.detach().cpu()),unclipped_gradient_norm=float(norm))
        sampler.record_completed_batch(batch)
        if step==0 or (step+1)%100==0:
            curve.append(dict(step=step+1,heads=losses));check(phase='T_fit',optimizer_updates=2*(step+1))
    for head in heads.values():head.eval();head.requires_grad_(False)
    return heads,normalizers,curve,sampler.coverage()


def corrected_scores(cache,heads,normalizers,device):
    scores={'WS':cache['scores']}
    with torch.inference_mode():
        for k,head in heads.items():
            scores[k]=cache['scores'].copy()
            for start in range(0,len(cache['pairs']),4096):
                idx=np.arange(start,min(start+4096,len(cache['pairs'])))
                x=normalize(edge_inputs(cache,idx,k),normalizers[k])
                scores[k][idx]+=head(torch.as_tensor(x,device=device)).cpu().numpy().astype(float)
    return scores


def predictions(m,memberships,model,device):
    groups={};packs={};raw={}
    for k,membership in memberships.items():
        g=common.old.groups_from_membership(membership,len(m['states']));groups[k]=g
        mean=ordinary_fusion(m['states'],m['nearest_seconds'],m['support'],g)['predictions']['mean_supported'][[x[0] for x in g]]
        pack=risk.build_source_features(**{key:m[key] for key in source.MOTION_KEYS},groups=g);packs[k]=pack
        scores=predict_residual(model,pack['features'].reshape(-1,10),device).reshape(pack['available'].shape)
        raw[k+'_MEAN']=mean;raw[k+'_NN']=recover(m,g,pack,scores,mean)
    return groups,packs,raw


def finish(rows,arrays,methods,primary):
    result=common.summarize(rows,arrays,method_names=methods,primary_pairs=primary)
    subsets=[(result['all_components'],range(len(rows))),
             (result['without_largest_component'],[i for i,r in enumerate(rows) if r['component_id']!=result['largest_component_id']])]
    subsets += [(sub,[i for i,r in enumerate(rows) if r['component_id']==c]) for c,sub in result['by_component'].items()]
    for sub,indices in subsets:
        if sub is None:continue
        for key in sub['pairwise']:
            hs=[rows[i]['pairwise'][key]['actual_harm'] for i in indices]
            sub['pairwise'][key]['actual_harm']=dict(new_harm_over_1km=sum(h['new_harm_over_1km'] for h in hs),max_error_increase_m=max(h['max_error_increase_m'] or 0. for h in hs))
        sub['association']={}
        for p in rows[0]['association']:
            sub['association'][p]={scope:partition.frozen._aggregate([partition.frozen.metric_rows(rows[i]['association'][p])[scope] for i in indices]) for scope in partition.frozen.SCOPES}
        sub['search']={p:dict(changed_scenes=sum(rows[i]['search'][p]['move_count']>0 for i in indices),
            moves=sum(rows[i]['search'][p]['move_count'] for i in indices),
            J_gain=sum(rows[i]['search'][p]['final_J']-rows[i]['search'][p]['initial_J'] for i in indices),
            budget_exhausted=sum(not rows[i]['search'][p]['locally_optimal'] for i in indices)) for p in rows[0]['search']}
    return result


def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')}
    out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    partitions=('WS','OWN','CTX') if a.mode=='front' else ('WS','REF','ANCHOR','SUPPORT')
    methods=tuple(p+'_'+f for p in partitions for f in ('MEAN','NN'))
    primary=tuple(p+'_NN_vs_WS_NN' for p in partitions[1:])+(('CTX_NN_vs_OWN_NN',) if a.mode=='front' else ('SUPPORT_NN_vs_ANCHOR_NN',))
    state=dict(status='RUNNING',idea_id='IDEA-ASSOCIATION-EDGE-CONTEXT-20261031' if a.mode=='front' else 'IDEA-ANCHOR-QUERY-SUPPORT-20261031',
        plan_version='1.0',mode=a.mode,methods=methods,started_at=datetime.now(timezone.utc).isoformat(),
        optimizer_updates=0,test_opened=False,reference_used_for_prediction=False,expression_run=str(a.expression_run),
        refinement_run=str(a.refinement_run),data_run=str(a.data_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=common.old.frozen.peak_rss_bytes());common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=common.old.frozen.read(paths['manifest'])
        tids,vids=common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        score_path=Path(common.old.frozen.read(paths['partition_run']/'run.json')['source_association'])/'final-scores.npz'
        check(training_scene_ids=tids,scene_ids=vids,association_score_cache=str(score_path))
        def original(m,split,sid,stored):
            if split=='T_core':
                pairs=common.load_npz(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz',('pairs',))['pairs']
                scores=common.load_npz(paths['value_run']/'private'/f'T-features-{sid}.npz',('scores',))['scores']
            else:
                pairs,scores=stored[sid+'_pairs'],stored[sid+'_scores']
                common.old._same(m['node_ids'],stored[sid+'_node_ids'],'original V nodes')
            return pairs,scores
        if a.mode=='front':
            teacher,checkpoint=load_checkpoint(score_path.parent/'final.pt',torch.device('cpu'));teacher.eval()
            if checkpoint['config']['variant']!='full_evidence' or checkpoint['config']['motion_observation_policy']!='canonical_all':raise ValueError('full canonical056 required')
            prepared=common.old.frozen.read(a.data_run/'run.json')
            if prepared['status']!='COMPLETED' or prepared['completed_T_scenes']!=2100:raise ValueError('complete151 required')
            index=common.old.frozen.read(a.data_run/'private/index.json')['T_core'];expanded=manifest['splits']['T_core'][:128]
            if expanded[:32]!=tids:raise ValueError('T prefix mismatch')
            training=[];train_rows=[]
            for sid in expanded:
                if sid in tids:
                    m=common.load_npz(paths['native_run']/'private/native-motion/T_core'/f'scene-{sid}.npz');pairs,base=original(m,'T_core',sid,None)
                else:
                    m=common.load_npz(Path(index[sid]['motion']))
                    if str(m['state_speed_unit'])!='mps':raise ValueError('151 explicit mps')
                    m['states']=m['states'].copy();m['states'][...,2]/=1852/3600
                    pairs=np.column_stack(np.triu_indices(len(m['states']),1));base=None
                cache=prepare_edges(teacher,checkpoint['normalizer'],m,pairs,base)
                np.savez_compressed(out/'private'/f'front-cache-T_core-{sid}.npz',**cache,node_ids=m['node_ids'])
                ev=common.evaluation(paths['native_run'],'T_core',sid,m) if sid in tids else common.load_npz(Path(index[sid]['evaluation']))
                common.old._same(m['node_ids'],ev['node_ids'],'T feature/label nodes')
                labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]]
                cache.update(scene_id=sid,pairs=pairs,labels=labels);training.append(cache)
                train_rows.append(dict(scene_id=sid,pairs=len(pairs),positive=int(labels.sum()),teacher_binding_max_abs=cache['teacher_binding_max_abs']))
                check(phase='T_prepare',prepared_T_scenes=len(training))
            check(fit_training_scene_ids=expanded,head_parameters=6209,training_pair_count=sum(len(c['pairs']) for c in training))
            heads,norms,curve,sampling=train_heads(training,a.device,check)
            torch.save(dict(heads={k:v.state_dict() for k,v in heads.items()},normalizers=norms),out/'private/heads.pt')
            common.old.frozen.write(out/'training-population.json',train_rows);common.old.frozen.write(out/'training-curve.json',curve);common.old.frozen.write(out/'training-sampling.json',sampling)
            check(full_T_models_frozen_before_V=True);del training
        with np.load(score_path,allow_pickle=False) as stored:
            for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
                rows=[];arrays=[]
                for sid in ids:
                    m,g,mean,pack,ws=physical(paths,split,sid);pairs,base=original(m,split,sid,stored)
                    if a.mode=='front':m.update(common.load_npz(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz',('endpoint_velocity_mps',)))
                    bound,_,_=partition.agglomerate(len(m['states']),pairs,base,'WS');common.old._same(bound,ws,'WS original partition')
                    memberships={'WS':ws};search={};traces={}
                    if a.mode=='front':
                        cache=prepare_edges(teacher,checkpoint['normalizer'],m,pairs,base)
                        scores=corrected_scores(cache,heads,norms,a.device)
                        for k in partitions[1:]:memberships[k],_,_=partition.agglomerate(len(m['states']),pairs,scores[k],'WS')
                    else:
                        scores={k:base for k in partitions}
                        for k,policy in [('REF','free'),('ANCHOR','anchors'),('SUPPORT','support')]:
                            memberships[k],search[k],traces[k]=refine_partition(len(m['states']),pairs,base,ws,threshold=partition.THRESHOLD,move_policy=policy,support=m['support'],sources=m['sources'])
                        old=common.load_npz(a.refinement_run/'private'/f'partitions-{split}-{sid}.npz',('REF',))['REF'];common.old._same(memberships['REF'],old,'original165 REF')
                    fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full'
                    model=torch.load(a.expression_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold']
                    groups,packs,raw=predictions(m,memberships,model,a.device)
                    oldraw=common.load_npz(a.expression_run/'private'/f'raw-{split}-{sid}.npz',('WS_T128_RESIDUAL_HARD',))['WS_T128_RESIDUAL_HARD'];common.old._same(raw['WS_NN'],oldraw,'frozen159 physical baseline')
                    np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                    np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',pairs=pairs,scores=base,**memberships,**{k+'_scores':v for k,v in scores.items()})
                    common.old.frozen.write(out/'private'/f'moves-{split}-{sid}.json',traces)
                    ev=common.evaluation(paths['native_run'],split,sid,m)
                    row,arr,art=evaluate(m,ev,groups,packs,raw,split,method_names=methods)
                    rec=dict(scene_id=sid,component_id=str(manifest['components'][sid]),node_ids=m['node_ids'],sources=m['sources'],pairs=pairs,labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]],joint_support_queries=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).sum(1))
                    relation={}
                    for k in partitions:
                        relation[k+'_raw']=partition.frozen.evaluate_scenes([dict(rec,scores=scores[k])],partition.THRESHOLD)
                        co=memberships[k][pairs[:,0]]==memberships[k][pairs[:,1]]
                        relation[k+'_decoded']=partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                    row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search=search,association=relation)
                    if a.mode=='front':row['teacher_binding_max_abs']=cache['teacher_binding_max_abs']
                    common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row)
                    rows.append(row);arrays.append(arr);check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
                result=finish(rows,arrays,methods,primary);result.update(status='COMPLETED',idea_id=state['idea_id'])
                common.old.frozen.write(out/fn,result);common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0,finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','expression-run','refinement-run','data-run'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mode',choices=('front','support'),required=True);p.add_argument('--device',required=True)
    p.add_argument('--time-policy',choices=('advisory',),required=True);run(p.parse_args())
