"""091 CPU cache closeout: exact exposure/calibration/probe binding, no forwards."""
from __future__ import annotations
import argparse,copy,gzip,json,math,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from association.models.ptfm import POSITION_OBJECTIVES,RESIDUAL_SCOPE
from association.training.ptfm import configuration,load_checkpoint,position_objective,OBJECTIVE_SCOPE,_paired_reference
from association.evaluation.ptfm import compare_objective_control,compare_sequence_control,compare_endpoints
from association.evaluation.ptfm_objective_audit import bind_source as bind_ptfm_metadata
from association.evaluation.ptfm_position_objectives import validate_calibration,ERRORS,REGIONS,PROBE_UPDATES
from association.training.evidence_fusion import TRAIN_MODES
from association.utils.runtime import read_json,write_json,write_json_gzip
from tools.analysis.ptfm_objective_compare import bind_arm as bind086,audit_objective_rows,scientific_screen
from tools.experiments.ptfm_fit import bind_paired_control
ROOT=Path(__file__).resolve().parents[2]


def _close(a,b,label):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('091 numeric audit differs/'+label)


def audit_calibrated_rows(path,fit,calibration):
    training,rows=audit_objective_rows(path,fit,'position-only')
    kind=position_objective(fit['configuration']);alpha=calibration['alpha'][kind]
    for row in rows:
        raw=row.get('raw_position_sample_means',{})
        if (row.get('position_objective')!=kind or row.get('position_loss_scale')!=alpha or set(raw)!=set(ERRORS)
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in raw.values())):
            raise ValueError('091 calibrated training diagnostics differ')
        _close(row['task_losses'][0],alpha*raw[kind],'actual calibrated position loss')
    windows=read_json(path/'fit/training-windows.json')
    if windows.get('selection') is not False or len(windows.get('windows',[]))!=math.ceil(len(rows)/100):
        raise ValueError('091 fixed training windows differ')
    for window,start in zip(windows['windows'],range(0,len(rows),100)):
        part=rows[start:start+100]
        if window.get('position_objective')!=kind or window.get('position_loss_scale')!=alpha:
            raise ValueError('091 window objective/scale differs')
        for key in ERRORS:_close(window['raw_position_sample_means'][key],float(np.mean([r['raw_position_sample_means'][key] for r in part])),'window raw loss')
    training.update(position_objective=kind,position_loss_scale=alpha,calibrated_sample_mean_objective_verified=True,
        fixed_T_endpoint_comparison='fit/fixed-training-probe.json; first96 original updates, training exposures only')
    return training,rows


def _audit_totals(value,samples,global_n,label):
    points=sum(s['valid_position_points'] for s in samples);n=len(samples)
    if (value.get('sample_count')!=n or value.get('valid_point_count')!=points
            or value.get('zero_supervision_samples')!=sum(s['valid_position_points']==0 for s in samples)
            or value.get('global_sample_denominator')!=global_n):
        raise ValueError('091 fixed T probe denominators differ/'+label)
    for key in ERRORS:
        total=sum(s['sample_means'][key] for s in samples)
        point_sum=sum(s['sample_means'][key]*s['valid_position_points'] for s in samples)
        _close(value['sample_mean_sums'][key],total,label+'/sample sum')
        _close(value['point_error_sums'][key],point_sum,label+'/point sum')
        _close(value['global_sample_mean_contributions'][key],total/global_n,label+'/global mean')
        if n:_close(value['conditional_sample_means'][key],total/n,label+'/conditional mean')
        elif value['conditional_sample_means'][key] is not None:raise ValueError('091 empty mode mean must be undefined')
    square=value['point_geodesic_square_sum_km2']
    if not isinstance(square,(int,float)) or not math.isfinite(square) or square<0:raise ValueError('091 invalid fixed T square sum')
    if points:
        _close(value['point_mae_m'],1000*value['point_error_sums']['geodesic-linear']/points,label+'/MAE')
        _close(value['point_rmse_m'],1000*math.sqrt(square/points),label+'/RMSE')
        if value['point_rmse_m']+1e-8<value['point_mae_m']:raise ValueError('091 probe RMSE smaller than MAE')
    elif value['point_mae_m'] is not None or value['point_rmse_m'] is not None:raise ValueError('091 empty physical metric must be undefined')


def audit_probe(path,fit,source_rows,source_fit,l2):
    config=fit['configuration'];chosen=source_rows[:min(PROBE_UPDATES,config['training']['updates'])]
    plan=read_json(path/'fit/fixed-training-probe-plan.json')
    if (plan.get('split')!='T_core' or plan.get('selection') is not False or plan.get('source_updates')!=len(chosen)
            or Path(plan.get('source_fit','')).resolve()!=source_fit
            or plan.get('rows')!=[{k:r[k] for k in ('step','mode','batch')} for r in chosen]):
        raise ValueError('091 fixed T probe plan differs from original086B prefix')
    probe=read_json(path/'fit/fixed-training-probe.json');labels=('initial','source086B_final','final')
    if (probe.get('scope')!='091-fixed-training-exposure-probe' or probe.get('selection') is not False
            or probe.get('test_opened') is not False or set(probe.get('endpoints',{}))!=set(labels)):
        raise ValueError('091 three fixed T endpoints required')
    expected=[dict(step=row['step'],mode=row['mode'],scene_id=s['scene_id'],node_id=s['node_id'],
        anchor_index=s['anchor_index'],valid_position_points=s['valid_position_points']) for row in chosen for s in row['batch']]
    n=len(expected)
    for label,endpoint in probe['endpoints'].items():
        if (endpoint.get('label')!=label or endpoint.get('split')!='T_core' or endpoint.get('truth_used') is not True
                or endpoint.get('validation_used') is not False or endpoint.get('selection') is not False
                or endpoint.get('device')!='cpu' or endpoint.get('optimizer_updates')!=0
                or endpoint.get('model_forwards')!=n or endpoint.get('sample_count')!=n
                or endpoint.get('source_updates')!=len(chosen)
                or Path(endpoint.get('source_fit','')).resolve()!=(source_fit if label=='source086B_final' else path/'fit')):
            raise ValueError('091 fixed T endpoint provenance differs/'+label)
        samples=endpoint.get('samples',[])
        if len(samples)!=n or [{k:s.get(k) for k in expected[0]} for s in samples]!=expected:
            raise ValueError('091 fixed T endpoint actual exposure differs/'+label)
        for sample in samples:
            means=sample.get('sample_means',{})
            if set(means)!=set(ERRORS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in means.values()):
                raise ValueError('091 invalid fixed T sample errors')
            if not sample['valid_position_points'] and any(means.values()):raise ValueError('091 unsupervised sample must contribute zero')
        _close(endpoint['L2_penalty'],l2[label],label+'/checkpoint L2')
        _audit_totals(endpoint['metrics'],samples,n,label)
        if set(endpoint.get('by_mode',{}))!=set(TRAIN_MODES) or set(endpoint.get('by_original_region',{}))!=set(REGIONS):
            raise ValueError('091 probe mode/support regions differ')
        for mode in TRAIN_MODES:_audit_totals(endpoint['by_mode'][mode],[s for s in samples if s['mode']==mode],n,label+'/'+mode)
        regions=endpoint['by_original_region']
        if (any(v['sample_count']!=n or v['global_sample_denominator']!=n for v in regions.values())
                or sum(v['valid_point_count'] for v in regions.values())!=endpoint['metrics']['valid_point_count']):
            raise ValueError('091 original support partition denominators differ')
        for key in ERRORS:
            for field in ('point_error_sums','sample_mean_sums','global_sample_mean_contributions'):
                _close(sum(v[field][key] for v in regions.values()),endpoint['metrics'][field][key],label+'/support partition '+field)
        _close(sum(v['point_geodesic_square_sum_km2'] for v in regions.values()),endpoint['metrics']['point_geodesic_square_sum_km2'],label+'/support square partition')
    # Masks and memberships are common across endpoints; their errors may differ.
    first=probe['endpoints']['initial']
    for endpoint in probe['endpoints'].values():
        for region in REGIONS:
            for key in ('sample_count','valid_point_count','zero_supervision_samples','global_sample_denominator'):
                if endpoint['by_original_region'][region][key]!=first['by_original_region'][region][key]:
                    raise ValueError('091 fixed T endpoint region population changed')
    return {label:{k:v for k,v in endpoint.items() if k!='samples'} for label,endpoint in probe['endpoints'].items()}


def bind_arm(path,kind,*,expected_config=None,expected_manifest=None):
    path=Path(path).resolve();run,summary,fit=[read_json(path/name) for name in ('run.json','summary.json','fit/run.json')]
    config=configuration(run['configuration']);steps=config['training']['updates']
    expected_config=expected_config if expected_config is not None else read_json(ROOT/f'configs/ptfm-{kind}-level1.json')
    manifest=expected_manifest if expected_manifest is not None else read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if config!=expected_config or position_objective(config)!=kind or run.get('scope')!='091-ptfm-position-objective' or summary.get('scope')!=run['scope']:
        raise ValueError('091 frozen production objective/configuration differs')
    metadata=bind_ptfm_metadata(path,config,manifest)
    residual=Path(run['paired_residual']).resolve();position=Path(run['paired_position_only']).resolve();paired=Path(run['paired_control']).resolve()
    source=Path(run['source_system']).resolve();bind_paired_control(paired,source)
    strong=_paired_reference(paired/'fit',config,manifest['splits']['T_core'])
    oldconfig=copy.deepcopy(config);oldconfig.pop('position_objective');oldconfig.pop('position_loss_normalization');oldconfig['scope']=OBJECTIVE_SCOPE
    old=bind086(position,'position-only',expected_config=oldconfig,expected_manifest=manifest)
    for key,expected in (('source_system',source),('paired_residual',residual),('paired_control',paired)):
        if Path(old['run'].get(key,'')).resolve()!=expected:raise ValueError('091 source086B control path differs/'+key)
    residualconfig=copy.deepcopy(oldconfig);residualconfig.pop('objective_mode');residualconfig['scope']=RESIDUAL_SCOPE
    bind_ptfm_metadata(residual,residualconfig,manifest)
    rows=[json.loads(line) for line in (path/'fit/training.jsonl').read_text().splitlines()]
    if [{k:r.get(k) for k in ('step','mode','batch')} for r in rows]!=[{k:r[k] for k in ('step','mode','batch')} for r in old['rows']]:
        raise ValueError('091 actual successful batches differ from086B')
    if (fit['exposure']!=old['fit']['exposure'] or fit['normalizer']!=old['fit']['normalizer']
            or fit['training_scene_ids']!=manifest['splits']['T_core']):raise ValueError('091 full T exposure/normalizer differs')
    for new,original in zip(rows,strong['rows']):
        batch=[{k:v for k,v in sample.items() if k!='task_valid_counts'} for sample in new['batch']]
        if batch!=original['batch']:raise ValueError('091 actual batch differs from072C')
    calibration=validate_calibration(read_json(path/'fit/loss-calibration.json'),config)
    counts=[s['valid_position_points'] for r in rows for s in r['batch']]
    if (calibration!=fit.get('loss_calibration') or calibration!=summary.get('loss_calibration')
            or Path(calibration['source_fit']).resolve()!=position/'fit'
            or calibration['valid_position_points']!=sum(counts)
            or calibration['zero_supervision_samples']!=sum(v==0 for v in counts)):
        raise ValueError('091 calibration original exposure binding differs')
    training,rows=audit_calibrated_rows(path,fit,calibration)
    initial=None;l2={}
    with torch.random.fork_rng(devices=[]):
        for label,filename,step in (('initial','step-0.pt',0),('final','final.pt',steps),('source086B_final','final.pt',steps)):
            checkpoint=(position if label=='source086B_final' else path)/'fit'/filename
            model,saved=load_checkpoint(checkpoint);l2[label]=float(config['model']['l2']*model.l2().detach());del model
            if saved['actual_updates']!=step or saved['normalizer']!=fit['normalizer'] or saved['configuration']!=(oldconfig if label=='source086B_final' else config):
                raise ValueError('091 actual checkpoint configuration/endpoint differs')
            if label!='source086B_final' and saved.get('loss_calibration')!=calibration:raise ValueError('091 checkpoint calibration differs')
            if label=='initial':
                initial=saved;original=old['initial']
                if (set(saved['model'])!=set(original['model']) or any(not torch.equal(v,original['model'][k]) for k,v in saved['model'].items())
                        or saved['numpy_rng']!=original['numpy_rng'] or not torch.equal(saved['torch_rng'],original['torch_rng'])):
                    raise ValueError('091 initial parameters/RNG differ from086B')
    probe=audit_probe(path,fit,old['rows'],position/'fit',l2)
    return dict(path=path,run=run,summary=summary,fit=fit,initial=initial,rows=rows,training=training,
        calibration=calibration,probe=probe,metadata=metadata)


def run(args,*,expected_configs=None,expected_manifest=None):
    if args.cpu_threads!=4:raise ValueError('091 cache comparison requires CPU4')
    torch.set_num_threads(4);output=Path(args.output).resolve()
    paths=[Path(args.radial_linear).resolve(),Path(args.geodesic_linear).resolve()]
    if output.exists():raise FileExistsError('091 no overwrite/retry')
    for path in paths:
        record=read_json(path/'run.json')
        protected=[path,Path(record['source_system']).resolve().parent]+[Path(record[k]).resolve() for k in ('paired_control','paired_residual','paired_position_only')]
        if any(output==p or p in output.parents or output in p.parents for p in protected):raise ValueError('091 output overlaps source')
    output.mkdir(parents=True);state=dict(scope='091-paired-position-objective-comparison',status='RUNNING',model_forwards=0,optimizer_updates=0,test_opened=False)
    write_json(output/'run.json',state)
    try:
        arms=[bind_arm(path,kind,expected_config=expected_configs.get(kind) if expected_configs is not None else None,
            expected_manifest=expected_manifest) for path,kind in zip(paths,POSITION_OBJECTIVES)]
        a,b=arms
        for key in ('paired_residual','paired_position_only','paired_control','source_system'):
            if Path(a['run'][key]).resolve()!=Path(b['run'][key]).resolve():raise ValueError('091 common source differs')
        ca,cb=[copy.deepcopy(arm['run']['configuration']) for arm in arms];ca.pop('position_objective');cb.pop('position_objective')
        if ca!=cb or a['calibration']!=b['calibration'] or a['fit']['exposure']!=b['fit']['exposure']:
            raise ValueError('091 paired recipe/calibration differs beyond objective')
        for label in ('initial','source086B_final'):
            first,second=[{k:v for k,v in arm['probe'][label].items() if k!='source_fit'} for arm in arms]
            if first!=second:raise ValueError('091 common fixed T endpoint probe differs')
        initial=compare_objective_control(paths[1]/'initial',paths[0]/'initial',initial=True)
        result=compare_objective_control(paths[1]/'final',paths[0]/'final')
        closeouts={}
        for arm,kind in zip(arms,POSITION_OBJECTIVES):
            path=arm['path'];record=arm['run']
            comparisons={}
            for name,key in (('versus076B','paired_residual'),('versus086B','paired_position_only')):
                comparisons[name]=compare_objective_control(path/'final',Path(record[key])/'final')
            comparisons['versus072C']=compare_sequence_control(path/'final',Path(record['paired_control'])/'final')
            screens=dict(versus_fixed_prior095=scientific_screen(arm['metadata']['endpoints']['final']['summary'],
                'ptfm_adapted','fixed_prior095',arm['metadata']['endpoints']['final']['rows']),
                versus_transformer=scientific_screen(comparisons['versus072C'],'ptfm_adapted','learned_sequence'))
            if screens!=arm['summary'].get('scientific_screens') or screens!=read_json(path/'scientific-screen.json'):
                raise ValueError('091 original screen differs from actual full caches')
            closeouts[kind]=dict(training=arm['training'],fixed_T_probe=arm['probe'],scientific_screens=screens,
                own_endpoints=compare_endpoints(path/'initial',path/'final'),**comparisons)
        result.update(scope=state['scope'],direction='geodesic-linear minus radial-linear',
            initial_replay=initial['all_initial_group_states_exact'],paired_successful_updates=len(a['rows']),
            sampled_anchors=a['fit']['exposure']['sampled_anchors'],matched_actual_samples=True,
            same_initial_parameters_and_RNG=True,loss_calibration=a['calibration'],arms=closeouts,
            mode_semantics={'ptfm_adapted':'geodesic-linear','ptfm_control':'radial-linear'},
            caveat='Initial mean matching does not equate gradients/Adam/clipping/L2; fixed T probe is training exposure, not held-out generalization or author reproduction.')
        write_json_gzip(output/'comparison.json.gz',result)
        compact={k:v for k,v in result.items() if k not in ('scenes','arms')}
        compact['arms']={kind:{name:({k:v for k,v in value.items() if k!='scenes'} if isinstance(value,dict) else value)
            for name,value in closeout.items()} for kind,closeout in closeouts.items()}
        write_json(output/'summary.json',compact);state.update(status='COMPLETED',scene_count=result['scene_count'])
        write_json(output/'run.json',state);return result
    except BaseException as error:
        state.update(status='FAILED',error=f'{type(error).__name__}: {error}');write_json(output/'run.json',state);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('radial-linear','geodesic-linear','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--cpu-threads',type=int,default=4);args=parser.parse_args(argv);run(args);print('COMPLETED',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
