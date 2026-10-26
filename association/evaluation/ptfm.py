"""074C CPU full-population PTFM evaluation; all groups precede truth alignment."""
from pathlib import Path
import time
import numpy as np
import torch
from association.models.ptfm import ptfm_inputs,native_states
from association.evaluation.sequence_position import _prior,_metric,WEIGHTINGS
from association.evaluation.frozen_fusion_transfer import _population,_changes
from association.evaluation.learned_group_oracle import _helper_evidence,_visible_regions,_quantiles
from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.support_control import support_switch_prediction
from association.evaluation.system_attribution import ATTRIBUTES,_same,assert_metrics,pool_metrics
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion
from association.utils.runtime import read_json,write_json,write_json_gzip
from tools.analysis.compare_association_duration import ARMS,_load_arrays,_replay
from tools.analysis.frozen_fusion_transfer import frozen_motion,groups_from_cache
from tools.experiments.level1_system_evaluate import evaluate_unique_population

METHODS=(*ARMS,'support_switch','fixed_prior095','ptfm_adapted')
REGIONS=('all','anchor_inside_support','anchor_outside_helper_supported','predicted_group_no_support',
         'no_predicted_group','supported_helper_present','supported_helper_absent')


def _predict(model,motion,groups,base,check):
    output=base.copy();weights=[];presence=[];local=[];origins=[];bases=[];residuals=[]
    residual_mode=getattr(model,'position_mode','direct')=='prior095_residual'
    with torch.inference_mode():
        for group in groups:
            if check:check()
            args=ptfm_inputs(motion,group,device='cpu',**({'include_prior':True} if residual_mode else {}));result=model(*args);states=result['states']
            if (states.shape!=args[0].shape or states.dtype!=torch.float64 or states.device.type!='cpu'
                    or not torch.isfinite(states).all()):raise ValueError('PTFM all states must be finite CPU physical4')
            w=result['task_weights']
            if (w.shape!=(len(states),3) or not torch.isfinite(w).all() or (w<0).any()
                    or not torch.allclose(w.sum(-1),torch.ones(len(states),dtype=w.dtype),atol=1e-6,rtol=0)):
                raise ValueError('PTFM task weights must be finite probabilities, not helper weights')
            _same(result['helper_present'].numpy(),args[2][:,0].bool().numpy(),'PTFM visible helper mask')
            extra={}
            if residual_mode:
                extra=dict(position_base=result['base_states'],position_residual_deg=result['position_residual_deg'])
                _same(result['base_states'].numpy(),args[4].numpy(),'PTFM physical prior input')
                if result['position_residual_deg'].shape!=(len(states),2) or not torch.isfinite(result['position_residual_deg']).all():
                    raise ValueError('PTFM position residual must be finite Qx2 native degree')
                bases.append(result['base_states'].numpy());residuals.append(result['position_residual_deg'].numpy())
            _same(states.numpy(),native_states(result['local_states'],args[3],**extra).numpy(),'PTFM physical inverse')
            output[group[0]]=states.numpy();weights.append(w.numpy());presence.append(result['helper_present'].numpy())
            local.append(result['local_states'].numpy());origins.append(args[3].numpy())
    extra=dict(group_base_states=np.asarray(bases),group_position_residual_deg=np.asarray(residuals)) if residual_mode else {}
    return output,np.asarray(weights),np.asarray(presence),np.asarray(local),np.asarray(origins),extra


def _weighted(arrays,methods=METHODS):
    result={}
    for region in REGIONS:
        result[region]={}
        for method in methods:
            points=[];targets=[];scenes=[];scene_targets=[]
            for data in arrays:
                error=data[method+'_position_error'];mask=data['region_'+region]&np.isfinite(error)
                values=error[mask];obs=np.stack((values,values*values),-1);points.append(obs)
                target=[np.array([error[i,use].mean(),np.square(error[i,use]).mean()]) for i,use in enumerate(mask) if use.any()]
                targets.extend(target)
                if len(values):scenes.append(obs.mean(0));scene_targets.append(np.mean(target,0))
            units=(np.concatenate(points),targets,scenes,scene_targets)
            result[region][method]={key:_metric(value) for key,value in zip(WEIGHTINGS,units)}
            result[region][method]['point_equal'].update(_quantiles(units[0][:,0]))
    return result


def _pool(rows,arrays):
    result=dict(scene_count=len(rows),scene_ids=[r['scene_id'] for r in rows],methods={},position_weightings=_weighted(arrays))
    for method in METHODS:
        values=[r['methods'][method] for r in rows]
        result['methods'][method]=dict(population={k:sum(v['population'][k] for v in values) for k in values[0]['population']},
            full_target_metrics=pool_metrics([v['full_target_metrics'] for v in values]),
            paper_fusion_metrics=aggregate_paper_fusion([v['paper_fusion_metrics'] for v in values]),
            attribute_quantiles={attr:{region:_quantiles(np.concatenate([d[method+'_'+attr+'_error'][
                d['region_'+region]&np.isfinite(d[method+'_'+attr+'_error'])] for d in arrays])) for region in REGIONS} for attr in ATTRIBUTES})
    result['pairwise']={}
    for control in METHODS[:-1]:
        result['pairwise']['ptfm_adapted__minus__'+control]={}
        for region in REGIONS:
            result['pairwise']['ptfm_adapted__minus__'+control][region]={}
            for attr in ATTRIBUTES:
                diffs=[]
                for d in arrays:
                    a,b=d['ptfm_adapted_'+attr+'_error'],d[control+'_'+attr+'_error']
                    _same(np.isfinite(a),np.isfinite(b),'PTFM common full-state mask')
                    diffs.append((a-b)[d['region_'+region]&np.isfinite(a)])
                result['pairwise']['ptfm_adapted__minus__'+control][region][attr]=_changes(np.concatenate(diffs))
    result['helper_availability']={scope:{key:sum(r['helper_availability'][scope][key] for r in rows)
        for key in rows[0]['helper_availability'][scope]} for scope in ('all_output_groups','unique_selected_groups')}
    return result


def evaluate_ptfm(model,validation_raw,source056,metadata,output,check=None,*,initial=False):
    source056,output=Path(source056).resolve(),Path(output).resolve();protected=source056.parent
    if output==protected or output in protected.parents or protected in output.parents:raise ValueError('PTFM must preserve source')
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('PTFM evaluation requires CPU')
    ids=[str(s['scene_id']) for s in validation_raw]
    if not ids or ids!=metadata['scene_ids'] or len(set(ids))!=len(ids) or set(metadata['component_ids'])!=set(ids):
        raise ValueError('PTFM complete V population/order/components required')
    residual_mode=getattr(model,'position_mode','direct')=='prior095_residual'
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    state=dict(status='RUNNING',scope='076B-ptfm-residual-evaluation' if residual_mode else '074C-ptfm-evaluation',device='cpu',initial=initial,
        source_system=str(source056),scene_ids=ids,total_scenes=len(ids),completed_scenes=0,test_opened=False,
        groups_recomputed=False,scores_recomputed=False,threshold_reselected=False,model_forwards=0,optimizer_updates=0)
    write_json(output/'run.json',state)
    try:
        for name in ('predictions','system-predictions','scenes'):(output/name).mkdir()
        model.eval();rows=[];arrays=[]
        for raw,sid in zip(validation_raw,ids):
            if check:check()
            fragment,saved=_load_arrays(source056,sid);replay=_replay(fragment,saved,metadata['expected'][sid])
            scene=frozen_motion(raw,fragment,replay[1]['learned_assoc_anchor']);groups=groups_from_cache(fragment,'learned_assoc')
            component=str(metadata['component_ids'][sid])
            if str(raw['component_id'])!=component:raise ValueError('PTFM component changed')
            predictions={arm:fragment['learned_assoc_'+arm] for arm in ARMS}
            predictions['support_switch'],switch=support_switch_prediction(predictions['anchor'],predictions['mean_supported'],fragment['support'],groups)
            predictions['fixed_prior095']=_prior(predictions['anchor'],predictions['mean_supported'],switch)
            predictions['ptfm_adapted'],weights,present,local,origins,extra=_predict(model,scene['motion'],groups,predictions['anchor'],check)
            if residual_mode:
                anchors=np.asarray([g[0] for g in groups],np.int64)
                _same(extra['group_base_states'][...,:2],predictions['fixed_prior095'][anchors,...,:2],'PTFM fixed prior095 position')
                if initial:
                    _same(extra['group_position_residual_deg'],np.zeros_like(extra['group_position_residual_deg']),'PTFM initial zero residual')
                    _same(predictions['ptfm_adapted'][anchors,...,:2],predictions['fixed_prior095'][anchors,...,:2],'PTFM initial exact physical position')
            reports,artifacts=evaluate_unique_population(scene,{'learned_assoc':groups},{'learned_assoc':predictions})
            artifacts={k.removeprefix('learned_assoc_'):v for k,v in artifacts.items()};base=artifacts['anchor']
            for method,artifact in artifacts.items():
                if set(artifact)!=set(base):raise ValueError('PTFM output artifact schema changed')
                for key in base:
                    if key not in ('aligned_prediction','group_states'):_same(artifact[key],base[key],'PTFM fixed population/'+key)
            for arm in ARMS:
                old=replay[1]['learned_assoc_'+arm]
                for key in old:_same(artifacts[arm][key],old[key],'PTFM ordinary replay/'+arm+'/'+key)
                assert_metrics(reports['learned_assoc_'+arm]['full_target_metrics'],metadata['expected'][sid]['learned_assoc_'+arm]['full_target_metrics'],'PTFM ordinary')
            t=len(base['target_ids']);truth=base['aligned_truth'][:t];common=base['aligned_truth_valid'][:t]&base['aligned_prediction_valid'][:t]
            anchors=np.asarray([g[0] for g in groups],np.int64);selected=base['aligned_group_indices'][:t]
            _,helpers,_=_helper_evidence(fragment,groups);visible=_visible_regions(fragment,anchors,selected,helpers)[0]
            has=np.zeros(common.shape[:2],bool);has[selected>=0]=present[selected[selected>=0]]
            regions=dict(all=np.ones(common.shape[:2],bool),**{k:visible[k] for k in REGIONS[1:5]},
                supported_helper_present=has,supported_helper_absent=(selected[:,None]>=0)&~has)
            data={'region_'+k:v for k,v in regions.items()};methods={}
            for method in METHODS:
                error=attribute_errors(artifacts[method]['aligned_prediction'][:t],truth)
                for attr,columns in ATTRIBUTES.items():
                    valid=common[...,list(columns)].all(-1)
                    if not np.isfinite(error[attr][valid]).all():raise ValueError('PTFM nonfinite valid error')
                    data[method+'_'+attr+'_error']=np.where(valid,error[attr],np.nan)
                methods[method]=dict(population=_population(artifacts[method]),
                    full_target_metrics=reports['learned_assoc_'+method]['full_target_metrics'],
                    paper_fusion_metrics=reports['learned_assoc_'+method]['paper_fusion_metrics'])
            used=np.zeros(len(groups),bool);used[selected[selected>=0]]=True
            availability={}
            for scope,use in (('all_output_groups',np.ones(len(groups),bool)),('unique_selected_groups',used)):
                availability[scope]=dict(query_count=int(present[use].size),supported_helper_present=int(present[use].sum()),
                    supported_helper_absent=int((~present[use]).sum()),negative_raw_speed=int((local[use,:,2]<0).sum()),
                    raw_latitude_outside_domain=int((np.abs(local[use,:,0]+origins[use,0,None])>90).sum()))
            row=dict(scene_id=sid,component_id=component,fragment_count=len(fragment['node_ids']),methods=methods,helper_availability=availability)
            row['position_weightings']=_weighted([data]);write_json_gzip(output/'scenes'/f'scene-{sid}.json.gz',row)
            payload={k:fragment[k] for k in ('node_ids','identities','target_identity_ids','native_states','support','nearest_seconds',
                'exact_observation','query_seconds','learned_assoc_group_offsets','learned_assoc_group_node_indices')}
            payload.update(group_anchor_node_indices=anchors,group_helper_present=present,group_task_weights=weights,group_local_states=local,group_origins=origins,
                **{m+'_group_states':a['group_states'] for m,a in artifacts.items()},**data,**extra)
            np.savez_compressed(output/'predictions'/f'states-{sid}.npz',**payload)
            np.savez_compressed(output/'system-predictions'/f'states-{sid}.npz',**{m+'_'+k:v for m,a in artifacts.items() for k,v in a.items()})
            rows.append(row);arrays.append(data);state.update(completed_scenes=len(rows),model_forwards=state['model_forwards']+len(groups),elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json',state)
        if sum(r['fragment_count'] for r in rows)!=metadata['fragment_count']:raise ValueError('PTFM fragment population changed')
        result=_pool(rows,arrays)
        for arm in ARMS:assert_metrics(result['methods'][arm]['full_target_metrics'],metadata['pooled']['learned_assoc_'+arm],'PTFM pooled baseline')
        components={}
        for c in sorted(set(str(v) for v in metadata['component_ids'].values())):
            idx=[i for i,r in enumerate(rows) if r['component_id']==c];components[c]=_pool([rows[i] for i in idx],[arrays[i] for i in idx])
        largest=min(components,key=lambda c:(-components[c]['position_weightings']['all']['anchor']['point_equal']['unit_count'],c))
        rest=[i for i,r in enumerate(rows) if r['component_id']!=largest]
        result.update(scope=state['scope'],status='COMPLETED',device='cpu',source_system=str(source056),method_names=list(METHODS),initial=initial,
            components=components,component_count=len(components),largest_component_id=largest,
            remaining_components=_pool([rows[i] for i in rest],[arrays[i] for i in rest]) if rest else None,
            test_opened=False,optimizer_updates=0,groups_recomputed=False,scores_recomputed=False,threshold_reselected=False,
            model_forwards=state['model_forwards'],fragment_count=metadata['fragment_count'],ordinary_source_replay_exact=True,
            interpretation=('PTFM physical-prior position residual with original full-state heads/loss; no champion-attribute composition; not author reproduction' if residual_mode else 'PTFM task-adapted full-state output, not published-result reproduction; all outputs including missing helper precede truth alignment'),
            normalization='T-only local native-unit zscore; full-state inverse; course circular',
            position_mode='prior095_residual' if residual_mode else 'direct',
            initial_position_prior_exact=bool(initial and residual_mode),
            limitations=['Reused V and shared identities; one seed/fixed budget is not convergence or independent generalization.',
                'Anchor/helper pooled roles and Q64 native completion differ from original aligned dual-radar windows.',
                '208->128 projection, Eq21 mean(-log task weight), native speed absolute value and latitude reflection are explicit reconstruction choices.'])
        write_json_gzip(output/'scene-summary.json.gz',rows);write_json(output/'summary.json',result)
        state.update(status='COMPLETED',elapsed_seconds=time.monotonic()-started);write_json(output/'run.json',state);return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json',state);raise


def compare_endpoints(initial,final):
    initial,final=Path(initial),Path(final);a,b=[read_json(p/'summary.json') for p in (initial,final)]
    if (a.get('status')!='COMPLETED' or b.get('status')!='COMPLETED' or a['scene_ids']!=b['scene_ids']
            or a['source_system']!=b['source_system'] or a.get('initial') is not True or b.get('initial') is not False):
        raise ValueError('PTFM paired completed CPU endpoints required')
    values={attr:[] for attr in ATTRIBUTES}
    for sid in a['scene_ids']:
        with np.load(initial/'system-predictions'/f'states-{sid}.npz',allow_pickle=False) as old,np.load(final/'system-predictions'/f'states-{sid}.npz',allow_pickle=False) as new:
            if set(old.files)!=set(new.files):raise ValueError('PTFM endpoint schema changed')
            for key in old.files:
                if key not in ('ptfm_adapted_group_states','ptfm_adapted_aligned_prediction'):_same(old[key],new[key],'PTFM endpoint fixed population/control')
        with np.load(initial/'predictions'/f'states-{sid}.npz',allow_pickle=False) as old,np.load(final/'predictions'/f'states-{sid}.npz',allow_pickle=False) as new:
            for attr in ATTRIBUTES:
                x,y=old['ptfm_adapted_'+attr+'_error'],new['ptfm_adapted_'+attr+'_error']
                _same(np.isfinite(x),np.isfinite(y),'PTFM endpoint valid masks');values[attr].append((y-x)[np.isfinite(x)])
    return dict(status='COMPLETED',population_and_masks_exact=True,model_forwards=0,test_opened=False,
                final_minus_initial={a:_changes(np.concatenate(v)) for a,v in values.items()})


def _compare_control(current,control,*,control_method='learned_sequence',initial=False):
    """Actual full-state cache comparison against the paired072C Transformer.

    The caller binds the completed training/checkpoints. This step checks every
    output group and shared native input before matching the complete targets.
    """
    from association.evaluation.support_gate_objectives import _read,_artifact
    from association.evaluation.physical import _check_valid_states
    from association.evaluation.position_sources import INPUT_KEYS
    current,control=Path(current),Path(control)
    left,right=[read_json(path/'summary.json') for path in (current,control)]
    if (any(v.get('status')!='COMPLETED' or v.get('device')!='cpu' or v.get('test_opened') is not False
            or v.get('initial') is not initial for v in (left,right))
            or left['scene_ids']!=right['scene_ids'] or left['source_system']!=right['source_system']):
        raise ValueError('PTFM/Transformer comparison requires completed CPU endpoints with one source/population')
    components={sid:str(c) for c,group in left['components'].items() for sid in group['scene_ids']}
    other={sid:str(c) for c,group in right['components'].items() for sid in group['scene_ids']}
    if components!=other:raise ValueError('PTFM/Transformer component registry differs')
    control_alias='learned_sequence' if control_method=='learned_sequence' else 'ptfm_control'
    methods=('ptfm_adapted',control_alias);rows=[];arrays=[]
    def pooled(selected):
        stats={};differences={}
        for region in REGIONS:
            stats[region]={};differences[region]={}
            for attr in ATTRIBUTES:
                values={method:np.concatenate([d[method+'_'+attr+'_error'][d['region_'+region]&np.isfinite(d[method+'_'+attr+'_error'])]
                    for d in selected]) for method in methods}
                stats[region][attr]={method:dict(_metric(np.stack((v,v*v),-1)),**_quantiles(v)) for method,v in values.items()}
                delta=values[methods[0]]-values[methods[1]]
                differences[region][attr]=dict(_changes(delta),improvement_tail=_quantiles(-delta[delta<0]),
                    absolute_change_tail=_quantiles(np.abs(delta)),
                    harm_above_native_threshold={str(k):int((delta>k).sum()) for k in (10,100,1000,10000)})
        difference_key='ptfm_minus_transformer' if control_method=='learned_sequence' else 'residual_minus_ptfm'
        return dict(metrics=stats,**{difference_key:differences},position_weightings=_weighted(selected,methods))
    for sid in left['scene_ids']:
        fragments=[_read(path/'predictions'/f'states-{sid}.npz') for path in (current,control)]
        systems=[_read(path/'system-predictions'/f'states-{sid}.npz') for path in (current,control)]
        for key in INPUT_KEYS:_same(fragments[0][key],fragments[1][key],'PTFM paired native input/'+key)
        for name in METHODS[:-1]:
            a,b=[_artifact(s,name) for s in systems]
            if set(a)!=set(b) or not a:raise ValueError('PTFM paired ordinary artifact schema differs')
            for key in a:_same(a[key],b[key],'PTFM paired ordinary full output/'+name+'/'+key)
        artifacts=[_artifact(system,method) for system,method in zip(systems,('ptfm_adapted',control_method))]
        base=_artifact(systems[0],'anchor');n=len(base['target_ids'])
        data={key:fragments[0][key] for key in fragments[0] if key.startswith('region_')}
        for name in (REGIONS if control_method=='ptfm_adapted' else REGIONS[:5]):_same(data['region_'+name],fragments[1]['region_'+name],'PTFM paired support stratum/'+name)
        for artifact,fragment,method,stored_method in zip(artifacts,fragments,methods,('ptfm_adapted',control_method)):
            if set(artifact)!=set(base):raise ValueError('PTFM paired model full artifact schema differs')
            for key in base:
                if key not in ('aligned_prediction','group_states'):_same(artifact[key],base[key],'PTFM paired fixed population/'+key)
            states=artifact['group_states']
            _check_valid_states(states,np.ones(states.shape,bool),'PTFM paired all output groups')
            _same(states,fragment[stored_method+'_group_states'],'PTFM paired cached group output')
            selected=base['aligned_group_indices'];chosen=selected>=0
            _same(artifact['aligned_prediction'][chosen],states[selected[chosen]],'PTFM paired group-to-target mapping')
            errors=attribute_errors(artifact['aligned_prediction'][:n],base['aligned_truth'][:n])
            valid=base['aligned_truth_valid'][:n]&base['aligned_prediction_valid'][:n]
            for attr,columns in ATTRIBUTES.items():
                mask=valid[...,list(columns)].all(-1)
                if not np.isfinite(errors[attr][mask]).all():raise ValueError('PTFM comparison nonfinite valid error')
                data[method+'_'+attr+'_error']=np.where(mask,errors[attr],np.nan)
                if stored_method=='ptfm_adapted' or attr=='position':
                    _same(data[method+'_'+attr+'_error'],fragment[stored_method+'_'+attr+'_error'],'PTFM cached/recomputed error')
        for attr in ATTRIBUTES:_same(np.isfinite(data['ptfm_adapted_'+attr+'_error']),np.isfinite(data[control_alias+'_'+attr+'_error']),'PTFM paired physical mask')
        rows.append(dict(scene_id=sid,component_id=components[sid],population=_population(base),**pooled([data])));arrays.append(data)
    result=pooled(arrays)
    for method,summary,stored_method in zip(methods,(left,right),('ptfm_adapted',control_method)):
        for attr in ATTRIBUTES:
            actual=result['metrics']['all'][attr][method];expected=summary['methods'][stored_method]['full_target_metrics']['all'][attr]
            if actual['unit_count']!=expected['intersection_count'] or any(not np.isclose(actual[k],expected[k],rtol=1e-10,atol=1e-9) for k in ('mae','rmse')):
                raise ValueError('PTFM paired actual arrays disagree with endpoint summary/'+method+'/'+attr)
    groups={c:pooled([a for r,a in zip(rows,arrays) if r['component_id']==c]) for c in sorted(set(components.values()))}
    largest=left['largest_component_id'];rest=[a for r,a in zip(rows,arrays) if r['component_id']!=largest]
    result.update(status='COMPLETED',current=str(current),control=str(control),scene_ids=left['scene_ids'],scene_count=len(rows),
        scenes=rows,components=groups,largest_component_id=largest,remaining_components=pooled(rest) if rest else None,
        population={key:sum(r['population'][key] for r in rows) for key in rows[0]['population']},
        full_group_population_controls_and_masks_exact=True,model_forwards=0,optimizer_updates=0,test_opened=False,
        interpretation=('Matched actual training samples; full-state multi-task PTFM adaptation versus position-only Transformer with ordinary Mean speed/course. Not a backbone-only or author-reproduction comparison.' if control_method=='learned_sequence' else 'Same full-state architecture/objectives and actual T exposure; physical position initialization and residual parameterization changed. Other final attributes may change through shared training; no champion-attribute composition.'))
    return result


def compare_sequence_control(current,control):
    return _compare_control(current,control)

def compare_ptfm_control(current,control,*,initial=False):
    result=_compare_control(current,control,control_method='ptfm_adapted',initial=initial)
    if initial:
        # Same CPU architecture/parameters for non-position outputs. Preserve the
        # predeclared physical replay contract; do not relax based on outcomes.
        from association.evaluation.evidence_fusion_combo import REPLAY_TOLERANCE
        tolerance={attr:REPLAY_TOLERANCE[key] for attr,key in (('speed','speed'),('heading','heading_deg'))}
        maxima={'speed':0.,'heading':0.}
        for sid in result['scene_ids']:
            with np.load(Path(current)/'system-predictions'/f'states-{sid}.npz') as new,np.load(Path(control)/'system-predictions'/f'states-{sid}.npz') as old:
                a,b=new['ptfm_adapted_group_states'],old['ptfm_adapted_group_states']
                errors=attribute_errors(a,b)
                for attr in maxima:
                    maxima[attr]=max(maxima[attr],float(errors[attr].max(initial=0)))
                    if (errors[attr]>tolerance[attr]).any():raise ValueError('PTFM non-position CPU initial outputs changed/'+attr)
        result['nonposition_initial_replay']={'maximum_native_difference':maxima,'absolute_tolerance':tolerance,'all_output_groups_verified':True}
    return result


def compare_objective_control(current,control,*,initial=False):
    """086 same-architecture objective comparison; initial ALL states replay."""
    result=compare_ptfm_control(current,control,initial=initial)
    if initial:
        for sid in result['scene_ids']:
            with np.load(Path(current)/'system-predictions'/f'states-{sid}.npz') as new, np.load(Path(control)/'system-predictions'/f'states-{sid}.npz') as old:
                _same(new['ptfm_adapted_group_states'],old['ptfm_adapted_group_states'],'086 all initial CPU group states')
        result['all_initial_group_states_exact']=True
    result['objective_control_minus_reference']=result.pop('residual_minus_ptfm')
    result['interpretation']='Same full-state architecture, all initial parameters and actual samples; only objective strategy changes. Constant-vs-adaptive is a strategy comparison, not sigma-only causality. No champion attribute composition.'
    return result
