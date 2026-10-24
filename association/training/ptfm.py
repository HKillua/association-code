"""074C bounded training-only PTFM reference adaptation."""
from __future__ import annotations
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from association.models.ptfm import (PTFM,SCOPE,ROUTING_SEMANTICS,LOSS_SEMANTICS,ptfm_inputs,objectives,
    RESIDUAL_SCOPE,RESIDUAL_ROUTING_SEMANTICS,RESIDUAL_LOSS_SEMANTICS,POSITION_OBJECTIVES)
from association.training.evidence_fusion import TRAIN_MODES,prepare_evidence_scenes,training_members
from association.utils.runtime import read_json,write_json


OBJECTIVE_SCOPE='ptfm-prior095-objective-control-v1'
POSITION_OBJECTIVE_SCOPE='ptfm-prior095-position-objective-v1'

def objective_mode(value):
    return value.get('objective_mode','adaptive')


def position_objective(value):
    return value.get('position_objective')

def configuration(value):
    residual=value.get('model',{}).get('position_output')=='prior095_residual'
    scope,route,_=semantics(value)
    model=dict(l2=1e-4,normalization='T_exposure_visible_local_zscore',helper_missing='masked_neutral_carrier')
    if residual:model['position_output']='prior095_residual'
    keys={'scope','routing_semantics','data','motion_observation_policy','train_splits','training','model'}
    if objective_mode(value)!='adaptive':keys.add('objective_mode')
    if position_objective(value) is not None:
        from association.evaluation.ptfm_position_objectives import CALIBRATION
        keys.update(('position_objective','position_loss_normalization'))
        if value.get('position_loss_normalization')!=CALIBRATION:
            raise ValueError('091 frozen T-only loss normalization differs')
    if (set(value)!=keys
            or value['scope']!=scope or value['routing_semantics']!=route
            or value['data']!=dict(length=64,max_observations=128,time_unit_seconds=20.,reference_longitude_policy='shortest_arc')
            or value['motion_observation_policy']!='canonical_all' or value['train_splits']!=['T_core']
            or value['model']!=model):
        raise ValueError('PTFM frozen architecture/data/normalization contract differs')
    t=value['training']
    if set(t)!={'seed','updates','batch_size','learning_rate','gradient_clip','advisory_seconds'}:
        raise ValueError('PTFM unexpected training settings')
    for k in ('seed','updates','batch_size'):
        if type(t[k]) is not int or t[k]<1: raise ValueError('PTFM positive integer '+k+' required')
    for k in ('learning_rate','gradient_clip','advisory_seconds'):
        if isinstance(t[k],bool) or not isinstance(t[k],(int,float)) or not math.isfinite(t[k]) or t[k]<=0:
            raise ValueError('PTFM positive finite '+k+' required')
    return value


def semantics(config):
    mode=objective_mode(config)
    if position_objective(config) is not None:
        if (position_objective(config) not in POSITION_OBJECTIVES or mode!='position-only'
                or config.get('model',{}).get('position_output')!='prior095_residual'):
            raise ValueError('091 requires residual position-only PTFM and declared position objective')
        return (POSITION_OBJECTIVE_SCOPE,RESIDUAL_ROUTING_SEMANTICS,
                'ptfm-prior095-position-only-T-initial-mean-calibrated-'+position_objective(config)+'-v1')
    if mode!='adaptive':
        if mode not in ('fixed-equal','position-only') or config.get('model',{}).get('position_output')!='prior095_residual':
            raise ValueError('086 requires residual PTFM and a fixed objective mode')
        return OBJECTIVE_SCOPE,RESIDUAL_ROUTING_SEMANTICS,RESIDUAL_LOSS_SEMANTICS.replace('-adaptive-v1','-'+mode+'-v1')
    if config.get('model',{}).get('position_output')=='prior095_residual':
        return RESIDUAL_SCOPE,RESIDUAL_ROUTING_SEMANTICS,RESIDUAL_LOSS_SEMANTICS
    return SCOPE,ROUTING_SEMANTICS,LOSS_SEMANTICS


def position_mode(config):
    return config.get('model',{}).get('position_output','direct')


def fit_normalizer(train,config,check=None):
    """Replay planned T samples with an independent RNG, before any optimization.

    Inputs only; neither targets/masks nor V affect statistics. Missing helpers
    are excluded rather than counted as duplicate measurements. Actual training
    restarts the same sampler seed, verifies its terminal RNG, and (when paired)
    checks every actual successful sample against the completed072C control.
    """
    rng=np.random.default_rng(config['training']['seed']); count=0; mean=np.zeros(4); m2=np.zeros(4)
    for step in range(config['training']['updates']):
        if check: check()
        mode=TRAIN_MODES[step%len(TRAIN_MODES)]
        for _ in range(config['training']['batch_size']):
            scene=train[int(rng.integers(len(train)))]; anchor=int(rng.integers(len(scene['identities'])))
            members=training_members(scene,anchor,mode)
            a,b,c,_=ptfm_inputs(scene['motion'],members)
            values=np.concatenate((a.numpy(),b.numpy()[c[:,0].numpy().astype(bool)]))
            n=len(values); average=values.mean(0); within=((values-average)**2).sum(0)
            delta=average-mean; total=count+n
            m2+=within+delta**2*count*n/total;mean+=delta*n/total;count=total
    return dict(mean=mean.tolist(),scale=np.maximum(np.sqrt(m2/count),1e-6).tolist(),count=count,
        split='T_core',truth_used=False,missing_helper_rows_counted=False,
        scope='planned identical scene/anchor/member exposure; visible local native units only',
        training_updates=config['training']['updates'],sampled_anchors=config['training']['updates']*config['training']['batch_size'],
        planned_sampler_final_rng=rng.bit_generator.state)


def _paired_reference(path,config,scene_ids):
    from tools.analysis.compare_position_benchmarks import _sequence_sampling
    from association.training.sequence_position import load_checkpoint as sequence_checkpoint
    path=Path(path).resolve();run,rows,exposure=_sequence_sampling(path)
    expected=run['configuration']
    if (any(expected[k]!=config[k] for k in ('data','training','train_splits','motion_observation_policy'))
            or expected.get('position_loss')!='linear' or expected['model']['backbone']!='transformer'
            or run['training_scene_ids']!=scene_ids):
        raise ValueError('PTFM paired control must have the same actual training protocol and ordered scenes')
    with torch.random.fork_rng(devices=[]):
        for filename,step in (('step-0.pt',0),('final.pt',config['training']['updates'])):
            model,saved=sequence_checkpoint(path/filename);del model
            if saved['configuration']!=expected or saved['actual_updates']!=step:
                raise ValueError('PTFM paired control endpoint binding differs')
    return dict(path=str(path),rows=rows,exposure=exposure)


def _paired_ptfm_reference(path,config,scene_ids,check=None):
    """Bind the completed direct run before any residual-model optimization.

    This reads both endpoints and every successful source batch. Neither a
    selected checkpoint nor an excerpted exposure log is an admissible source.
    """
    if position_mode(config)!='prior095_residual':
        raise ValueError('direct PTFM pairing is only defined for residual PTFM')
    path=Path(path).resolve();run=read_json(path/'run.json')
    direct=copy.deepcopy(config)
    direct.pop('position_objective',None);direct.pop('position_loss_normalization',None)
    control=objective_mode(config)!='adaptive'
    if control:
        direct.pop('objective_mode');direct.update(scope=RESIDUAL_SCOPE,routing_semantics=RESIDUAL_ROUTING_SEMANTICS)
    else:
        direct.update(scope=SCOPE,routing_semantics=ROUTING_SEMANTICS);direct['model'].pop('position_output')
    expected_scope,_,expected_loss=semantics(direct)
    if (run.get('status')!='COMPLETED' or run.get('configuration')!=direct
            or run.get('scope')!=expected_scope or run.get('loss_semantics')!=expected_loss
            or run.get('optimizer_updates')!=config['training']['updates']
            or run.get('training_scene_ids')!=scene_ids or run.get('test_opened') is not False
            or run.get('final_checkpoint')!='final.pt' or run.get('normalizer_sampler_replay_exact') is not True):
        raise ValueError('PTFM paired direct run/configuration/endpoint is not completed and equivalent')
    normalizer=read_json(path/'normalizer.json')
    if run.get('normalizer')!=normalizer:raise ValueError('PTFM paired direct normalizer/run differs')
    endpoints={}
    with torch.random.fork_rng(devices=[]):
        for filename,step in (('step-0.pt',0),('final.pt',config['training']['updates'])):
            if check:check()
            model,saved=load_checkpoint(path/filename);del model
            if (saved['configuration']!=direct or saved['actual_updates']!=step
                    or saved['normalizer']!=normalizer):
                raise ValueError('PTFM paired direct checkpoint/configuration/normalizer differs')
            endpoints[step]=saved
    audit=dict(successful_updates=0,sampled_anchors=0,valid_position_points=0,task_valid_counts=[0,0,0],
               scenes={},anchors={})
    rows=[];batch_size=config['training']['batch_size']
    with (path/'training.jsonl').open() as handle:
        for step,line in enumerate(handle,1):
            if check:check()
            row=json.loads(line);batch=row.get('batch',[])
            if (row.get('step')!=step or row.get('mode')!=TRAIN_MODES[(step-1)%len(TRAIN_MODES)]
                    or len(batch)!=batch_size):raise ValueError('PTFM paired direct successful-batch sequence differs')
            for sample in batch:
                counts=sample.get('task_valid_counts')
                if (sample.get('scene_id') not in scene_ids or not isinstance(counts,list) or len(counts)!=3
                        or any(type(n) is not int or n<0 or n>config['data']['length'] for n in counts)
                        or sample.get('valid_position_points')!=counts[0]):
                    raise ValueError('PTFM paired direct task mask/exposure is malformed')
                for i,count in enumerate(counts):audit['task_valid_counts'][i]+=count
                for key,field in (('scenes','scene_id'),('anchors','node_id')):
                    name=sample[field];audit[key][name]=audit[key].get(name,0)+1
            audit['successful_updates']+=1;audit['sampled_anchors']+=len(batch);rows.append(row)
    audit['valid_position_points']=audit['task_valid_counts'][0]
    if (len(rows)!=config['training']['updates'] or
            any(run.get('exposure',{}).get(key)!=value for key,value in audit.items())
            or run.get('trained_scene_count')!=len(audit['scenes'])):
        raise ValueError('PTFM paired direct actual exposure counters differ')
    return dict(path=str(path),rows=rows,normalizer=normalizer,initial=endpoints[0],exposure=run['exposure'])


def _paired_position_reference(path,config,scene_ids,ptfm_reference,check=None):
    """091 also binds completed086B, independently of its076B initial source."""
    if position_objective(config) is None:raise ValueError('091 source binding requires new position objective')
    path=Path(path).resolve();run=read_json(path/'run.json')
    expected=copy.deepcopy(config);expected.pop('position_objective');expected.pop('position_loss_normalization')
    expected['scope']=OBJECTIVE_SCOPE
    if (run.get('status')!='COMPLETED' or run.get('configuration')!=expected
            or run.get('optimizer_updates')!=config['training']['updates'] or run.get('test_opened') is not False
            or run.get('training_scene_ids')!=scene_ids or run.get('normalizer')!=ptfm_reference['normalizer']
            or run.get('exposure')!=ptfm_reference['exposure'] or run.get('final_checkpoint')!='final.pt'
            or run.get('normalizer_sampler_replay_exact') is not True):
        raise ValueError('091 requires completed086B with exact frozen protocol/normalizer/exposure')
    with (path/'training.jsonl').open() as handle:rows=[json.loads(line) for line in handle]
    actual=[{key:row.get(key) for key in ('step','mode','batch')} for row in rows]
    original=[{key:row.get(key) for key in ('step','mode','batch')} for row in ptfm_reference['rows']]
    if actual!=original:raise ValueError('091 source086B actual successful batches differ from076B')
    saved={}
    with torch.random.fork_rng(devices=[]):
        for filename,step in (('step-0.pt',0),('final.pt',config['training']['updates'])):
            if check:check()
            model,payload=load_checkpoint(path/filename)
            if payload['configuration']!=expected or payload['actual_updates']!=step or payload['normalizer']!=run['normalizer']:
                raise ValueError('091 source086B actual checkpoint differs')
            if step==0:
                _check_initial_pair(model,ptfm_reference['initial'],exact=True)
                old=ptfm_reference['initial']
                if payload['numpy_rng']!=old['numpy_rng'] or not torch.equal(payload['torch_rng'],old['torch_rng']):
                    raise ValueError('091 source086B initial RNG differs')
            saved[step]=payload;del model
    return dict(path=str(path),configuration=expected,rows=rows,initial=saved[0],final=saved[config['training']['updates']])


def _check_initial_pair(model,saved,*,exact=False):
    current=model.state_dict();reference=saved['model']
    if set(current)!=set(reference):raise ValueError('PTFM paired initial parameter names differ')
    for name,value in current.items():
        expected=reference[name]
        if not exact and name in ('output.weight','output.bias'):
            expected=expected.clone();expected[:2].zero_()
        if not torch.equal(value.cpu(),expected):
            raise ValueError('PTFM paired initial parameters differ/'+name)


def _save(path,model,optimizer,step,config,rng,normalizer):
    scope,route,loss=semantics(config)
    extra={}
    if position_objective(config) is not None:
        extra['loss_calibration']=getattr(model,'position_loss_calibration',None)
    torch.save(dict(scope=scope,routing_semantics=route,loss_semantics=loss,
        configuration=config,normalizer=normalizer,step=step,actual_updates=step,model=model.state_dict(),
        optimizer=optimizer.state_dict(),numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,**extra),path)


def load_checkpoint(path,*,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    config=configuration(saved['configuration']);scope,route,loss=semantics(config)
    if (saved.get('scope')!=scope or saved.get('routing_semantics')!=route
            or saved.get('loss_semantics')!=loss): raise ValueError('PTFM checkpoint semantics differ')
    step=saved.get('step')
    if (type(step) is not int or not 0<=step<=config['training']['updates'] or type(saved.get('actual_updates')) is not int
            or saved['actual_updates']!=step or any(k not in saved for k in ('optimizer','numpy_rng','torch_rng','cuda_rng'))):
        raise ValueError('PTFM checkpoint actual updates/RNG differ')
    norm=saved['normalizer']
    if (norm.get('training_updates')!=config['training']['updates'] or norm.get('sampled_anchors')!=config['training']['updates']*config['training']['batch_size']
            or type(norm.get('count')) is not int or norm['count']<1 or norm.get('missing_helper_rows_counted') is not False
            or not isinstance(norm.get('planned_sampler_final_rng'),dict)):
        raise ValueError('PTFM normalization exposure/configuration differs')
    if step==config['training']['updates'] and saved['numpy_rng']!=norm['planned_sampler_final_rng']:
        raise ValueError('PTFM final checkpoint sampling/normalization exposure differs')
    model=PTFM(norm,position_mode=position_mode(config)).to(device);model.load_state_dict(saved['model'],strict=True)
    if position_objective(config) is not None:
        from association.evaluation.ptfm_position_objectives import validate_calibration
        model.position_loss_calibration=validate_calibration(saved.get('loss_calibration'),config)
    if (not torch.equal(model.mean.cpu(),torch.tensor(norm['mean'],dtype=torch.float64))
            or not torch.equal(model.scale.cpu(),torch.tensor(norm['scale'],dtype=torch.float64))
            or not all(torch.isfinite(p).all() for p in model.parameters())):
        raise ValueError('PTFM normalizer buffers or model parameters differ')
    groups=saved['optimizer'].get('param_groups',[]);entries=saved['optimizer'].get('state',{})
    if (len(groups)!=1 or groups[0].get('lr')!=config['training']['learning_rate']
            or len(groups[0].get('params',[]))!=len(list(model.parameters()))):
        raise ValueError('PTFM optimizer/configuration differs')
    if (step==0 and entries) or (step>0 and (set(entries)!=set(groups[0]['params']) or
            any(float(entry.get('step',-1))!=step for entry in entries.values()))):
        raise ValueError('PTFM actual optimizer step differs')
    return model,saved


def train_ptfm(train_raw,config,output,*,device='cpu',check=None,paired_reference_fit=None,paired_ptfm_fit=None,
               paired_position_fit=None):
    config=copy.deepcopy(configuration(config));train_raw=list(train_raw)
    ids=[str(s['scene_id']) for s in train_raw]
    if not ids or len(set(ids))!=len(ids) or any(s.get('split')!='T_core' for s in train_raw):
        raise ValueError('PTFM fitting accepts distinct T_core scenes only')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    t=config['training'];rng=np.random.default_rng(t['seed']);torch.manual_seed(t['seed'])
    audit=dict(successful_updates=0,sampled_anchors=0,valid_position_points=0,task_valid_counts=[0,0,0],
               helper_absent_queries=0,scenes={},anchors={})
    scope,route,loss_semantics=semantics(config)
    state=dict(scope=scope,status='RUNNING',configuration=config,optimizer_updates=0,test_opened=False,
        training_scene_ids=ids,exposure=audit,device=str(device),loss_semantics=loss_semantics,
        gradient_clipping=dict(clipped_updates=0),selection_policy='fixed final; no V selection',
        interpretation='PTFM task adaptation with full-state/adaptive-task objective; not author reproduction')
    state['paired_control']=dict(fit=str(Path(paired_reference_fit).resolve()) if paired_reference_fit else None,
        status='PENDING' if paired_reference_fit else 'NOT_REQUESTED',matched_updates=0,matched_anchors=0,
        interpretation='Actual scene/anchor/member exposure only; model, state objectives and initialization differ.')
    state['paired_ptfm_control']=dict(fit=str(Path(paired_ptfm_fit).resolve()) if paired_ptfm_fit else None,
        status='PENDING' if paired_ptfm_fit else 'NOT_REQUESTED',matched_updates=0,matched_anchors=0,
        interpretation='Only position output parameterization and its first two zero-initialized output rows differ.')
    if objective_mode(config)!='adaptive':
        state['interpretation']='PTFM fixed objective strategy control, not author reproduction or sigma-only causality'
        state['paired_ptfm_control']['interpretation']='All initial parameters, normalizer and actual samples match076B; only objective strategy changes.'
    if position_objective(config) is not None:
        state['interpretation']='091 calibrated position-only objective strategy bridge; not pure geometry/gradient causality or author reproduction'
        state['paired_position_control']=dict(fit=str(Path(paired_position_fit).resolve()) if paired_position_fit else None,status='PENDING')
    elif paired_position_fit is not None:
        raise ValueError('paired086B source is only accepted by091')
    write_json(output/'run.json',state);model=optimizer=norm=None;windows=[];window=[]
    try:
        reference=_paired_reference(paired_reference_fit,config,ids) if paired_reference_fit else None
        if position_mode(config)=='prior095_residual' and paired_ptfm_fit is None:
            raise ValueError('PTFM residual training requires the completed direct PTFM paired control')
        ptfm_reference=_paired_ptfm_reference(paired_ptfm_fit,config,ids,check) if paired_ptfm_fit else None
        position_reference=None
        if position_objective(config) is not None:
            if paired_position_fit is None or ptfm_reference is None:
                raise ValueError('091 requires both completed076B and086B controls')
            position_reference=_paired_position_reference(paired_position_fit,config,ids,ptfm_reference,check)
        train=prepare_evidence_scenes(train_raw,check=check,motion_observation_policy='canonical_all')
        if any(not len(s['identities']) for s in train):raise ValueError('PTFM cannot drop empty train scenes')
        norm=fit_normalizer(train,config,check);write_json(output/'normalizer.json',norm)
        model=PTFM(norm,position_mode=position_mode(config)).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'])
        if ptfm_reference is not None:
            if norm!=ptfm_reference['normalizer']:raise ValueError('PTFM paired normalizer differs')
            _check_initial_pair(model,ptfm_reference['initial'],exact=objective_mode(config)!='adaptive')
            if rng.bit_generator.state!=ptfm_reference['initial']['numpy_rng']:
                raise ValueError('PTFM paired initial sampler RNG differs')
            if not torch.equal(torch.get_rng_state(),ptfm_reference['initial']['torch_rng']):
                raise ValueError('PTFM paired initial torch RNG differs')
            state['paired_ptfm_control'].update(initial_parameters_exact=objective_mode(config)!='adaptive',
                initial_parameters_exact_except_position_rows=True,
                position_rows_zero=True,normalizer_exact=True,initial_sampler_rng_exact=True,initial_torch_rng_exact=True)
        state['parameter_count']=sum(p.numel() for p in model.parameters());state['normalizer']=norm
        calibration=probe=None
        if position_reference is not None:
            from association.evaluation.ptfm_position_objectives import calibrate_initial_inputs,fixed_training_probe,PROBE_UPDATES
            initial_rng=torch.get_rng_state().clone();sampler_state=copy.deepcopy(rng.bit_generator.state)
            calibration=calibrate_initial_inputs(train,position_reference['rows'],config,norm,paired_position_fit,check)
            if not torch.equal(initial_rng,torch.get_rng_state()) or sampler_state!=rng.bit_generator.state:
                raise ValueError('091 calibration changed original training RNG')
            model.position_loss_calibration=calibration;state['loss_calibration']=calibration
            write_json(output/'loss-calibration.json',calibration)
            plan=position_reference['rows'][:min(PROBE_UPDATES,t['updates'])]
            write_json(output/'fixed-training-probe-plan.json',dict(source_fit=position_reference['path'],
                split='T_core',selection=False,source_updates=len(plan),
                rows=[{k:r[k] for k in ('step','mode','batch')} for r in plan]))
        _save(output/'step-0.pt',model,optimizer,0,config,rng,norm)
        if position_reference is not None:
            initial_rng=torch.get_rng_state().clone();sampler_state=copy.deepcopy(rng.bit_generator.state)
            probe=dict(scope='091-fixed-training-exposure-probe',selection=False,test_opened=False,endpoints={})
            probe['endpoints']['initial']=fixed_training_probe(model,train,position_reference['rows'],config,
                source_fit=output,label='initial',check=check)
            with torch.random.fork_rng(devices=[]):
                old_model,_=load_checkpoint(Path(paired_position_fit)/'final.pt')
                probe['endpoints']['source086B_final']=fixed_training_probe(old_model,train,position_reference['rows'],config,
                    source_fit=paired_position_fit,label='source086B_final',check=check)
                del old_model
            if not torch.equal(initial_rng,torch.get_rng_state()) or sampler_state!=rng.bit_generator.state:
                raise ValueError('091 calibration/probe changed original training RNG')
            write_json(output/'fixed-training-probe.json',probe)
        fit_started=time.monotonic();cuda=torch.device(device).type=='cuda'
        if cuda:torch.cuda.reset_peak_memory_stats(device)
        for step in range(1,t['updates']+1):
            if check:check()
            model.train();optimizer.zero_grad(set_to_none=True);mode=TRAIN_MODES[(step-1)%len(TRAIN_MODES)]
            batch=[];losses=[];diags=[];absent=0
            for _ in range(t['batch_size']):
                scene=train[int(rng.integers(len(train)))];anchor=int(rng.integers(len(scene['identities'])))
                members=training_members(scene,anchor,mode);result=model(*ptfm_inputs(scene['motion'],members,device=device,
                    include_prior=position_mode(config)=='prior095_residual'))
                target=torch.as_tensor(scene['target'][anchor],dtype=torch.float64,device=device)
                valid=torch.as_tensor(scene['valid'][anchor],dtype=torch.bool,device=device)
                extra={} if calibration is None else dict(position_objective=position_objective(config),
                    position_loss_scale=calibration['alpha'][position_objective(config)])
                loss,diag=objectives(model,result,target,valid,mode=objective_mode(config),**extra);(loss/t['batch_size']).backward()
                losses.append(float(loss.detach()));diags.append(diag)
                absent+=int((~result['helper_present']).sum())
                eligible=~scene['motion']['support'][anchor]&scene['motion']['support'][members[1:]].any(0)
                batch.append(dict(scene_id=str(scene['scene_id']),anchor_index=anchor,node_id=str(scene['node_ids'][anchor]),
                    members=[int(x) for x in members],valid_position_points=diag['task_valid_counts'][0],
                    eligible_query_points=int(eligible.sum()),eligible_valid_position_points=int((eligible&valid[:,:2].all(-1).cpu().numpy()).sum()),
                    task_valid_counts=diag['task_valid_counts']))
            if reference is not None:
                old=reference['rows'][step-1]
                common=[{key:value for key,value in sample.items() if key!='task_valid_counts'} for sample in batch]
                if old['step']!=step or old['mode']!=mode or old['batch']!=common:
                    raise ValueError('PTFM actual batch differs from completed paired control before optimizer update')
            if ptfm_reference is not None:
                old=ptfm_reference['rows'][step-1]
                if old['step']!=step or old['mode']!=mode or old['batch']!=batch:
                    raise ValueError('PTFM actual batch/task masks differ from direct PTFM before optimizer update')
            penalty=config['model']['l2']*model.l2();penalty.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();state['optimizer_updates']=step
            audit['successful_updates']+=1;audit['sampled_anchors']+=len(batch);audit['helper_absent_queries']+=absent
            if reference is not None:
                state['paired_control'].update(matched_updates=step,matched_anchors=audit['sampled_anchors'])
            if ptfm_reference is not None:
                state['paired_ptfm_control'].update(matched_updates=step,matched_anchors=audit['sampled_anchors'])
            for i in range(3):audit['task_valid_counts'][i]+=sum(d['task_valid_counts'][i] for d in diags)
            audit['valid_position_points']=audit['task_valid_counts'][0]
            for sample in batch:
                for key,field in (('scenes','scene_id'),('anchors','node_id')):
                    name=sample[field];audit[key][name]=audit[key].get(name,0)+1
            state['gradient_clipping']['clipped_updates']+=int(grad>t['gradient_clip'])
            if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('PTFM nonfinite updated parameter')
            row=dict(step=step,mode=mode,batch=batch,loss=float(np.mean(losses)),l2=float(penalty.detach()),
                task_losses=np.mean([d['task_losses'] for d in diags],axis=0).tolist(),
                task_sigmas=np.mean([d['task_sigmas'] for d in diags],axis=0).tolist(),gradient_norm=float(grad),
                gradient_clipped=bool(grad>t['gradient_clip']),gradient_clip_scale=min(1.,t['gradient_clip']/(float(grad)+1e-6)))
            if objective_mode(config)!='adaptive':
                for field in ('actual_task_coefficients','actual_task_regularizers','actual_task_contributions'):
                    row[field]=np.mean([d[field] for d in diags],axis=0).tolist()
                row['objective_mode']=objective_mode(config)
            if calibration is not None:
                row.update(position_objective=position_objective(config),position_loss_scale=calibration['alpha'][position_objective(config)],
                    raw_position_sample_means={key:float(np.mean([d['raw_position_sample_means'][key] for d in diags]))
                        for key in ('normalized_squared','radial-linear','geodesic-linear')})
            with (output/'training.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            window.append(row)
            if step%100==0 or step==t['updates']:
                windows.append(dict(first_step=window[0]['step'],last_step=step,loss_mean=float(np.mean([r['loss'] for r in window])),
                    task_losses_mean=np.mean([r['task_losses'] for r in window],0).tolist(),
                    gradient_norm_median=float(np.median([r['gradient_norm'] for r in window])),
                    clipped_updates=sum(r['gradient_clipped'] for r in window)))
                if objective_mode(config)!='adaptive':
                    for field in ('actual_task_coefficients','actual_task_regularizers','actual_task_contributions','task_sigmas'):
                        windows[-1][field+'_mean']=np.mean([r[field] for r in window],0).tolist()
                    windows[-1]['l2_mean']=float(np.mean([r['l2'] for r in window]))
                if calibration is not None:
                    windows[-1].update(position_objective=position_objective(config),
                        position_loss_scale=calibration['alpha'][position_objective(config)],
                        raw_position_sample_means={key:float(np.mean([r['raw_position_sample_means'][key] for r in window]))
                            for key in ('normalized_squared','radial-linear','geodesic-linear')})
                window=[];write_json(output/'training-windows.json',dict(windows=windows,selection=False))
            state.update(elapsed_seconds=time.monotonic()-started,training_seconds=time.monotonic()-fit_started,
                         peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device) if cuda else 0)
            write_json(output/'run.json',state)
        if rng.bit_generator.state!=norm['planned_sampler_final_rng']:
            raise ValueError('PTFM actual sampling differs from planned normalization exposure')
        state['normalizer_sampler_replay_exact']=True
        if reference is not None:
            for key in ('successful_updates','sampled_anchors','valid_position_points','scenes','anchors'):
                if audit[key]!=reference['exposure'][key]:raise ValueError('PTFM paired exposure counter differs/'+key)
            state['paired_control'].update(status='PAIR_VERIFIED',actual_successful_batches_exact=True)
        if ptfm_reference is not None:
            if audit!=ptfm_reference['exposure']:raise ValueError('PTFM paired final task/exposure audit differs')
            state['paired_ptfm_control'].update(status='PAIR_VERIFIED',actual_successful_batches_exact=True,
                all_task_counts_exact=True,full_exposure_exact=True)
        _save(output/'final.pt',model,optimizer,t['updates'],config,rng,norm)
        if position_reference is not None:
            probe['endpoints']['final']=fixed_training_probe(model,train,position_reference['rows'],config,
                source_fit=output,label='final',check=check)
            write_json(output/'fixed-training-probe.json',probe)
            state['paired_position_control'].update(status='PAIR_VERIFIED',matched_updates=t['updates'],
                matched_anchors=audit['sampled_anchors'],initial_parameters_exact=True,
                actual_successful_batches_exact=True,full_exposure_exact=True)
            state['fixed_training_probe']='fixed-training-probe.json'
        state.update(status='COMPLETED',final_checkpoint='final.pt',trained_scene_count=len(audit['scenes']))
        write_json(output/'run.json',state);return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json',state)
        if model is not None:_save(output/'interrupted.pt',model,optimizer,state['optimizer_updates'],config,rng,norm)
        raise
