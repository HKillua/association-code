"""086 CPU-only comparison of two completed, paired objective-control runs."""
from __future__ import annotations
import argparse,gzip,json,math,sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
import numpy as np
from association.training.ptfm import load_checkpoint,_paired_ptfm_reference,objective_mode
from association.evaluation.ptfm import compare_objective_control
from association.evaluation.ptfm_objective_audit import audit_training_log
from association.utils.runtime import read_json,write_json,write_json_gzip

ROOT=Path(__file__).resolve().parents[2]


def scientific_screen(summary,candidate,control,scene_rows=None):
    """Frozen resource screen; qualification of a baseline never depends on score."""
    eps=1e-6;table=summary['position_weightings']['all'];new,old=table[candidate],table[control]
    delta={key:new['point_equal'][key]-old['point_equal'][key] for key in ('mae','rmse','p95','p99')}
    reduction=1-new['point_equal']['mae']/old['point_equal']['mae'] if old['point_equal']['mae']>0 else 0.
    nonpoint=all(new[key]['mae']<=old[key]['mae']+eps for key in old if key!='point_equal')
    def improves(row):
        tab=row['position_weightings']['all'];return tab[candidate]['point_equal']['mae']<tab[control]['point_equal']['mae']-eps
    rows=summary.get('scenes') if scene_rows is None else scene_rows
    if rows is None:raise ValueError('086 screen needs actual scene rows')
    scenes=sum(improves(row) for row in rows);components=sum(improves(row) for row in summary['components'].values())
    rest=summary['remaining_components'];outside=bool(rest and rest['position_weightings']['all'][candidate]['point_equal']['mae']<=rest['position_weightings']['all'][control]['point_equal']['mae']+eps)
    return dict(worth_matched_replication=bool(reduction>=.03 and delta['rmse']<=eps and delta['p99']<=eps and nonpoint and outside and scenes>len(rows)/2 and components>len(summary['components'])/2),
        relative_mae_reduction=reduction,required_mae_reduction=.03,deltas=delta,numerical_guard_m=eps,
        nonpoint_mae_nonworse=nonpoint,outside_largest_component_nonworse=outside,
        improved_scenes=scenes,total_scenes=len(rows),improved_components=components,total_components=len(summary['components']),
        interpretation='Single-seed resource screen, not significance, baseline validity or automatic method promotion.')


def _jsonl(path):
    with Path(path).open() as handle:return [json.loads(line) for line in handle]


def audit_objective_rows(path,fit,mode):
    """Recompute fixed coefficients from sample masks, never mean*mean terms."""
    training,rows=audit_training_log(dict(path=path,fit=fit))
    enabled=np.asarray([True,mode=='fixed-equal',mode=='fixed-equal'])
    optimized_counts=np.zeros(3,np.int64)
    for row in rows:
        counts=np.asarray([sample['task_valid_counts'] for sample in row['batch']],np.int64)
        fraction=(counts>0).mean(0)*enabled
        coefficients=fraction/3.
        regularizers=fraction*math.log(3.)
        # The data coefficient is constant for each active task; invalid sample
        # losses already contribute zero to the logged batch mean.
        contributions=np.asarray(row['task_losses'])*enabled/3.+regularizers
        expected=dict(actual_task_coefficients=coefficients,actual_task_regularizers=regularizers,
            actual_task_contributions=contributions)
        for key,value in expected.items():
            actual=np.asarray(row.get(key),np.float64)
            if actual.shape!=(3,) or not np.isfinite(actual).all() or not np.allclose(actual,value,rtol=1e-12,atol=1e-12):
                raise ValueError('086 actual objective diagnostic differs/'+key)
        if row.get('objective_mode')!=mode or not np.isclose(contributions.sum(),row['loss'],rtol=1e-12,atol=1e-12):
            raise ValueError('086 task contributions do not reproduce optimized data objective')
        optimized_counts+=counts.sum(0)*enabled
    for window,start in zip(training['fixed_100_step_windows'],range(0,len(rows),100)):
        subset=rows[start:start+100]
        for key in ('actual_task_coefficients','actual_task_regularizers','actual_task_contributions'):
            window[key+'_mean']=np.mean([r[key] for r in subset],axis=0).tolist()
        window['regularized_objective_mean']=float(np.mean([r['loss']+r['l2'] for r in subset]))
    training.update(actual_fixed_objective_verified=True,optimized_task_valid_counts=optimized_counts.tolist(),
        task_sigmas_semantics='Network output diagnostic, not the optimized constant coefficient.',
        per_sample_weighted_task_terms='Fixed coefficients and mask counts identify the batch-mean contributions; per-sample values are not saved.')
    return training,rows


def bind_arm(path,mode,*,expected_config=None,expected_manifest=None):
    """Production defaults are frozen; explicit fixture expectations are test-only."""
    path=Path(path).resolve();run,summary,fit=[read_json(path/name) for name in ('run.json','summary.json','fit/run.json')]
    config=run['configuration'];steps=config['training']['updates']
    expected_config=expected_config if expected_config is not None else read_json(ROOT/f'configs/ptfm-{mode}-level1.json')
    manifest=expected_manifest if expected_manifest is not None else read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    if config!=expected_config or fit.get('training_scene_ids')!=manifest['splits']['T_core']:
        raise ValueError('086 frozen production configuration/T population differs')
    if objective_mode(config)!=mode or run.get('scope')!='086-ptfm-objective-control':raise ValueError('086 arm/mode differs')
    for item in (run,summary,fit):
        if item.get('status')!='COMPLETED' or item.get('configuration')!=config or item.get('optimizer_updates')!=steps or item.get('test_opened') is not False:
            raise ValueError('086 complete fixed endpoint required')
    if run.get('completed_endpoints')!=['initial','final'] or run.get('validation_selection') is not False:
        raise ValueError('086 initial/final selection binding differs')
    residual=Path(run['paired_residual']).resolve()
    reference=_paired_ptfm_reference(residual/'fit',config,fit['training_scene_ids'])
    rows=_jsonl(path/'fit/training.jsonl')
    if len(rows)!=steps or [{k:r[k] for k in ('step','mode','batch')} for r in rows]!=[{k:r[k] for k in ('step','mode','batch')} for r in reference['rows']]:
        raise ValueError('086 actual successful batches differ from076B')
    if fit.get('normalizer')!=reference['normalizer'] or fit.get('exposure')!=reference['exposure']:
        raise ValueError('086 normalizer/actual exposure differs')
    training,rows=audit_objective_rows(path,fit,mode)
    initial=None
    for filename,step in (('step-0.pt',0),('final.pt',steps)):
        with torch.random.fork_rng(devices=[]):model,saved=load_checkpoint(path/'fit'/filename)
        del model
        if saved['configuration']!=config or saved['actual_updates']!=step or saved['normalizer']!=reference['normalizer']:
            raise ValueError('086 checkpoint endpoint binding differs')
        if step==0:
            initial=saved;old=reference['initial']
            if set(saved['model'])!=set(old['model']) or any(not torch.equal(value,old['model'][key]) for key,value in saved['model'].items()) or saved['numpy_rng']!=old['numpy_rng'] or not torch.equal(saved['torch_rng'],old['torch_rng']):
                raise ValueError('086 initial parameters/RNG differ from076B')
        label='initial' if step==0 else 'final'
        end=read_json(path/label/'summary.json');record=read_json(path/label/'run.json')
        for item in (end,record):
            cp=item.get('checkpoint',{})
            if item.get('status')!='COMPLETED' or item.get('configuration')!=config or item.get('initial') is not (step==0) or item.get('device')!='cpu' or item.get('test_opened') is not False or Path(item.get('source_system','')).resolve()!=Path(run['source_system']).resolve() or cp.get('step')!=step or cp.get('actual_updates')!=step or Path(cp.get('path','')).resolve()!=path/'fit'/filename or cp.get('configured_final_updates')!=steps or cp.get('routing_semantics')!=config['routing_semantics'] or item.get('optimizer_updates')!=0 or any(item.get(key) is not False for key in ('groups_recomputed','scores_recomputed','threshold_reselected')):
                raise ValueError('086 CPU endpoint provenance differs')
        if summary.get('endpoints',{}).get(label)!=end['checkpoint']:raise ValueError('086 endpoint summary mismatch')
        ids=manifest['splits']['V_select']
        if (end.get('scene_ids')!=ids or record.get('scene_ids')!=ids or end.get('scene_count')!=len(ids)
                or record.get('completed_scenes')!=len(ids) or record.get('total_scenes')!=len(ids)):
            raise ValueError('086 complete frozen V population differs')
        components={sid:str(c) for c,group in end['components'].items() for sid in group['scene_ids']}
        if components!={sid:str(manifest['components'][sid]) for sid in ids}:
            raise ValueError('086 component registry differs')
        for folder in ('predictions','system-predictions'):
            if {p.name for p in (path/label/folder).glob('*.npz')}!={f'states-{sid}.npz' for sid in ids}:
                raise ValueError('086 full V cache inventory differs')
    return dict(path=path,run=run,fit=fit,rows=rows,initial=initial,reference=reference,training=training)


def run(args,*,expected_configs=None,expected_manifest=None):
    if args.cpu_threads!=4:raise ValueError('086 cache comparison requires CPU4')
    torch.set_num_threads(4);output=Path(args.output).resolve()
    paths=[Path(args.fixed_equal).resolve(),Path(args.position_only).resolve()]
    if output.exists():raise FileExistsError('086 no overwrite/retry')
    for path in paths:
        run_record=read_json(path/'run.json')
        for protected in (path,Path(run_record['paired_residual']).resolve(),Path(run_record['paired_control']).resolve(),Path(run_record['source_system']).resolve().parent):
            if output==protected or protected in output.parents or output in protected.parents:raise ValueError('086 output overlaps source')
    output.mkdir(parents=True);state=dict(status='RUNNING',model_forwards=0,optimizer_updates=0,test_opened=False)
    write_json(output/'run.json',state)
    try:
        a,b=[bind_arm(path,mode,expected_config=expected_configs.get(mode) if expected_configs is not None else None,
            expected_manifest=expected_manifest) for path,mode in zip(paths,('fixed-equal','position-only'))]
        for key in ('paired_residual','paired_control','source_system'):
            if Path(a['run'][key]).resolve()!=Path(b['run'][key]).resolve():raise ValueError('086 common source differs')
        ca,cb=[dict(arm['run']['configuration']) for arm in (a,b)];ca.pop('objective_mode');cb.pop('objective_mode')
        if ca!=cb or a['fit']['exposure']!=b['fit']['exposure']:raise ValueError('086 paired recipe differs beyond objective')
        initial=compare_objective_control(paths[1]/'initial',paths[0]/'initial',initial=True)
        result=compare_objective_control(paths[1]/'final',paths[0]/'final')
        result.update(scope='086-paired-objective-comparison',direction='position-only minus fixed-equal',
            initial_replay=initial['all_initial_group_states_exact'],paired_successful_updates=len(a['rows']),
            sampled_anchors=a['fit']['exposure']['sampled_anchors'],same_initial_parameters_and_RNG=True,
            matched_actual_samples=True,mode_semantics={'ptfm_adapted':'position-only','ptfm_control':'fixed-equal'},
            training={objective_mode(arm['run']['configuration']):arm['training'] for arm in (a,b)},
            caveat='Auxiliary heads in position-only have no data supervision; shared latent changes may still alter their outputs.')
        write_json_gzip(output/'comparison.json.gz',result)
        write_json(output/'summary.json',{key:value for key,value in result.items() if key!='scenes'})
        state.update(status='COMPLETED',scene_count=result['scene_count']);write_json(output/'run.json',state);return result
    except BaseException as error:
        state.update(status='FAILED',error=f'{type(error).__name__}: {error}');write_json(output/'run.json',state);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fixed-equal','position-only','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--cpu-threads',type=int,default=4);args=parser.parse_args(argv);run(args);print('COMPLETED',flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
