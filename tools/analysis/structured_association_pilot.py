"""173: matched full-scene BCE vs BCE+partial FY, unknown-k WS deployment."""
from __future__ import annotations
import argparse,copy,sys,time
from pathlib import Path
from datetime import datetime,timezone
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch.nn import functional as F
from association.models.association_context import EdgeResidual,edge_inputs,normalize
from association.training.structured_forest import partial_fy_loss
from association.training.evidence_association import load_checkpoint
from tools.analysis import association_context_pilot as p
from tools.analysis.association_operating_point_pilot import calibrate,operating_point,THRESHOLD

METHODS=tuple(k+'_'+f for k in ('WS','CTX','BCE','STRUCT') for f in ('MEAN','NN'))
PRIMARY=('CTX_NN_vs_WS_NN','BCE_NN_vs_CTX_NN','STRUCT_NN_vs_BCE_NN','STRUCT_NN_vs_CTX_NN','STRUCT_NN_vs_WS_NN')


def balanced_bce(logits,labels):
    losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none');positive=labels.bool()
    if positive.any() and (~positive).any():return (losses[positive].mean()+losses[~positive].mean())/2
    return losses.mean()


def train(caches,initial,normalizer,device,check,epochs=4):
    heads={k:EdgeResidual().to(device) for k in ('BCE','STRUCT')}
    for h in heads.values():h.load_state_dict(initial)
    opts={k:torch.optim.Adam(h.parameters(),lr=.001) for k,h in heads.items()}
    rng=np.random.default_rng(20);noise_rng=np.random.default_rng(20);curve=[];draws=[]
    counts=np.zeros(len(caches),int);presented=0
    for epoch in range(epochs):
        for i in rng.permutation(len(caches)):
            c=caches[i];idx=np.arange(len(c['pairs']));x=torch.as_tensor(normalize(edge_inputs(c,idx,'CTX'),normalizer),device=device)
            y=torch.as_tensor(c['labels'],device=device,dtype=torch.float64);base=torch.as_tensor(c['scores'],device=device)
            z=noise_rng.normal(0.,.1,(4,len(idx)));losses={}
            for key,h in heads.items():
                opts[key].zero_grad(set_to_none=True);logits=base+h(x).double();bce=balanced_bce(logits,y)
                fy=partial_fy_loss(logits,c['pairs'],c['identities'],z) if key=='STRUCT' else logits.sum()*0
                loss=bce+fy
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(h.parameters(),1.,error_if_nonfinite=True);opts[key].step()
                losses[key]=dict(loss=float(loss.detach().cpu()),bce=float(bce.detach().cpu()),fy=float(fy.detach().cpu()),unclipped_gradient_norm=float(norm))
            counts[i]+=1;presented+=len(idx);draws.append(c['scene_id']);curve.append(dict(step=len(draws),epoch=epoch+1,scene_id=c['scene_id'],pairs=len(idx),heads=losses))
            if len(draws)==1 or len(draws)%16==0:check(phase='T_fit',optimizer_updates=2*len(draws))
    for h in heads.values():h.eval();h.requires_grad_(False)
    return heads,curve,dict(scene_draws=draws,epochs=epochs,per_scene_passes=counts.tolist(),unique_pairs=sum(len(c['pairs']) for c in caches),pair_presentations_per_head=presented,updates_per_head=len(draws))


def predict(cache,heads,normalizer,device):
    scores={'WS':cache['scores']}
    with torch.inference_mode():
        for key,h in heads.items():
            scores[key]=cache['scores'].copy()
            for start in range(0,len(cache['pairs']),4096):
                idx=np.arange(start,min(start+4096,len(cache['pairs'])));x=normalize(edge_inputs(cache,idx,'CTX'),normalizer)
                scores[key][idx]+=h(torch.as_tensor(x,device=device)).cpu().numpy().astype(float)
    return scores


def run(a):
    paths={k:Path(getattr(a,k)).resolve() for k in ('native_run','value_run','partition_run','source_risk_run','manifest','output')}
    out=paths['output'];out.mkdir(exist_ok=False);(out/'private').mkdir();(out/'scenes').mkdir();began=time.monotonic()
    state=dict(status='RUNNING',idea_id='IDEA-STRUCTURED-ASSOCIATION-20261031-173',plan_version='2.0',mode='structured',methods=METHODS,
        optimizer_updates=0,started_at=datetime.now(timezone.utc).isoformat(),test_opened=False,reference_used_for_prediction=False,
        expression_run=str(a.expression_run),front_run=str(a.front_run),calibration_run=str(a.calibration_run),data_run=str(a.data_run),
        qualification_run=str(a.qualification_run),**{k:str(v) for k,v in paths.items() if k!='output'})
    def check(**kw):
        state.update(kw,elapsed_seconds=time.monotonic()-began,peak_rss_bytes=p.common.old.frozen.peak_rss_bytes());p.common.old.frozen.write(out/'run.json',state)
        if state['peak_rss_bytes']>8*1024**3:raise MemoryError('RSS8GiB')
    try:
        torch.set_num_threads(4);manifest=p.common.old.frozen.read(paths['manifest']);tids,vids=p.common.metadata(paths['native_run'],paths['value_run'],paths['partition_run'],paths['source_risk_run'],manifest)
        qualification=p.common.old.frozen.read(a.qualification_run/'run.json');qsummary=p.common.old.frozen.read(a.qualification_run/'summary.json')
        if qualification['status']!='COMPLETED' or not qsummary['qualified'] or qualification['training_scene_ids']!=tids:raise ValueError('completed172 required')
        parent=p.common.old.frozen.read(a.front_run/'run.json');calparent=p.common.old.frozen.read(a.calibration_run/'run.json')
        if parent['status']!='COMPLETED' or parent['optimizer_updates']!=2000 or calparent['status']!='COMPLETED':raise ValueError('completed168/170 required')
        trainids=manifest['splits']['T_core'][:128]
        if trainids!=parent['fit_training_scene_ids'] or trainids[:32]!=tids:raise ValueError('fixed T128 required')
        index=p.common.old.frozen.read(a.data_run/'private/index.json')['T_core'];training=[];population=[]
        for sid in trainids:
            cache=p.common.load_npz(a.front_run/'private'/f'front-cache-T_core-{sid}.npz')
            ev=p.common.load_npz(paths['native_run']/'private/native-evaluation/T_core'/f'scene-{sid}.npz') if sid in tids else p.common.load_npz(Path(index[sid]['evaluation']))
            p.common.old._same(cache['node_ids'],ev['node_ids'],'T cache labels nodes');pairs=cache['pairs'];identities=ev['identities']
            labels=identities[pairs[:,0]]==identities[pairs[:,1]];cache.update(scene_id=sid,identities=identities,labels=labels)
            training.append(cache);population.append(dict(scene_id=sid,nodes=len(identities),classes=len(np.unique(identities)),pairs=len(pairs),positive=int(labels.sum())))
        if sum(len(c['pairs']) for c in training)!=775705:raise ValueError('original T128 pair population changed')
        original=torch.load(a.front_run/'private/heads.pt',map_location='cpu',weights_only=False);normalizer=original['normalizers']['CTX'];initial=copy.deepcopy(original['heads']['CTX'])
        check(training_scene_ids=tids,scene_ids=vids,fit_training_scene_ids=trainids,head_parameters=6209,training_pair_count=775705,teacher_training_calls=0)
        heads,curve,sampling=train(training,initial,normalizer,a.device,check)
        if sampling['updates_per_head']!=512 or sampling['pair_presentations_per_head']!=3102820 or sampling['per_scene_passes']!=[4]*128:raise ValueError('fixed complete exposure mismatch')
        p.common.old.frozen.write(out/'training-population.json',population);p.common.old.frozen.write(out/'training-curve.json',curve);p.common.old.frozen.write(out/'training-sampling.json',sampling)
        torch.save(dict(initial=initial,heads={k:h.state_dict() for k,h in heads.items()},normalizer=normalizer),out/'private/heads.pt')
        frozen=EdgeResidual().to(a.device);frozen.load_state_dict(initial);frozen.eval();frozen.requires_grad_(False);heads={'CTX':frozen,**heads}
        records=[]
        for cache in training[:32]:
            sid=cache['scene_id'];m=p.common.load_npz(paths['native_run']/'private/native-motion/T_core'/f'scene-{sid}.npz',('sources',));pairs=cache['pairs'];cross=m['sources'][pairs[:,0]]!=m['sources'][pairs[:,1]]
            scores=predict(cache,heads,normalizer,a.device);old=p.common.load_npz(a.front_run/'private'/f'partitions-T_core-{sid}.npz')
            p.common.old._same(scores['CTX'],old['CTX_scores'],'original168 frozen CTX scores')
            records.append(dict(split='T_core',labels=cache['labels'][cross],**{k:v[cross] for k,v in scores.items()}))
        limit=operating_point(records,'WS',THRESHOLD)['m_pfa'];chosen={};curves={}
        for key in ('WS','CTX','BCE','STRUCT'):chosen[key],curves[key]=calibrate(records,key,limit)
        p.common.old.frozen.write(out/'T-calibration.json',dict(limit=limit,chosen=chosen,curves=curves,T_scenes=tids))
        if any(v is None for v in chosen.values()):check(status='COMPLETED',exit_code=0,phase='T_NOT_QUALIFIED',completed_V_scenes=0);return
        for k in ('WS','CTX'):
            if abs(chosen[k]['threshold']-calparent['thresholds'][k])>1e-5:raise ValueError('original170 operating point not bound')
        check(optimizer_updates=1024,full_T_models_frozen_before_V=True,calibration_frozen_before_V=True,thresholds={k:v['threshold'] for k,v in chosen.items()})
        del training
        score_path=Path(p.common.old.frozen.read(paths['partition_run']/'run.json')['source_association'])/'final-scores.npz'
        teacher,checkpoint=load_checkpoint(score_path.parent/'final.pt',torch.device('cpu'));teacher.eval()
        for split,ids,fn in [('T_core',tids,'T-summary.json'),('V_select',vids,'summary.json')]:
            rows=[];arrays=[]
            for sid in ids:
                m,g,mean,pack,ws=p.physical(paths,split,sid);old=p.common.load_npz(a.front_run/'private'/f'partitions-{split}-{sid}.npz');pairs=old['pairs']
                if split=='T_core':cache=p.common.load_npz(a.front_run/'private'/f'front-cache-T_core-{sid}.npz')
                else:
                    m.update(p.common.load_npz(paths['native_run']/'private/native-motion'/split/f'scene-{sid}.npz',('endpoint_velocity_mps',)))
                    cache=p.prepare_edges(teacher,checkpoint['normalizer'],m,pairs,old['WS_scores'])
                rawscores=predict(cache,heads,normalizer,a.device);scores={k:v+THRESHOLD-chosen[k]['threshold'] for k,v in rawscores.items()}
                memberships={k:p.partition.agglomerate(len(m['states']),pairs,v,'WS')[0] for k,v in scores.items()}
                oldcal=p.common.load_npz(a.calibration_run/'private'/f'partitions-{split}-{sid}.npz')
                p.common.old._same(memberships['WS'],oldcal['WS'],'original WS partition');p.common.old._same(memberships['CTX'],oldcal['CTXCAL'],'original170 CTX partition')
                fold='leave_'+manifest['components'][sid] if split=='T_core' else 'full';model=torch.load(a.expression_run/'private'/f'model-{fold}.pt',map_location='cpu',weights_only=False)['netold']
                groups,packs,raw=p.predictions(m,memberships,model,a.device)
                oldraw=p.common.load_npz(a.calibration_run/'private'/f'raw-{split}-{sid}.npz')
                for key in ('WS','CTX'):p.common.old._same(raw[key+'_NN'],oldraw[('WS' if key=='WS' else 'CTXCAL')+'_NN'],'frozen170 physical output')
                # All deployments saved before this split's evaluation labels enter.
                np.savez_compressed(out/'private'/f'raw-{split}-{sid}.npz',**raw)
                np.savez_compressed(out/'private'/f'partitions-{split}-{sid}.npz',pairs=pairs,scores=scores['WS'],**memberships,**{k+'_scores':v for k,v in scores.items()},**{k+'_uncalibrated_scores':v for k,v in rawscores.items()})
                ev=p.common.evaluation(paths['native_run'],split,sid,m);row,arr,art=p.evaluate(m,ev,groups,packs,raw,split,method_names=METHODS)
                rec=dict(scene_id=sid,component_id=str(manifest['components'][sid]),node_ids=m['node_ids'],sources=m['sources'],pairs=pairs,labels=ev['identities'][pairs[:,0]]==ev['identities'][pairs[:,1]],joint_support_queries=(m['support'][pairs[:,0]]&m['support'][pairs[:,1]]).sum(1));association={}
                for key in scores:
                    association[key+'_raw']=p.partition.frozen.evaluate_scenes([dict(rec,scores=scores[key])],THRESHOLD)
                    co=memberships[key][pairs[:,0]]==memberships[key][pairs[:,1]];association[key+'_decoded']=p.partition.frozen.evaluate_scenes([dict(rec,scores=co.astype(float))],.5)
                row.update(scene_id=sid,component_id=str(manifest['components'][sid]),search={},association=association)
                p.common.save_artifacts(out/'private'/f'states-{split}-{sid}.npz',art);p.common.old.frozen.write(out/'scenes'/f'{split}-{sid}.json.gz',row);rows.append(row);arrays.append(arr)
                check(phase=split,**{'completed_T_scenes' if split=='T_core' else 'completed_V_scenes':len(rows)})
            result=p.finish(rows,arrays,METHODS,PRIMARY);result.update(status='COMPLETED',idea_id=state['idea_id']);p.common.old.frozen.write(out/fn,result);p.common.old.frozen.write(out/('T-scene-summary.json.gz' if split=='T_core' else 'scene-summary.json.gz'),rows)
        check(status='COMPLETED',exit_code=0,finished_at=datetime.now(timezone.utc).isoformat(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else 0)
    except BaseException as e:check(status='FAILED',exit_code=1,error=f'{type(e).__name__}: {e}');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('native-run','value-run','partition-run','source-risk-run','manifest','output','expression-run','front-run','calibration-run','qualification-run','data-run'):parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--time-policy',choices=('advisory',),required=True);run(parser.parse_args())
