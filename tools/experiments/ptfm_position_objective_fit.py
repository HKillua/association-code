"""091 fixed4000 calibrated radial/geodesic PTFM, paired to076B/086B/072C."""
from __future__ import annotations
import argparse,gzip,json,signal,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from association.data.archive import validate_manifest
from association.data.dataset import load_timed_dataset
from association.training.ptfm import configuration,train_ptfm,position_objective
from association.evaluation.ptfm import evaluate_ptfm,compare_endpoints,compare_sequence_control,compare_objective_control
from association.models.ptfm import POSITION_OBJECTIVES
from association.utils.runtime import read_json,write_json,write_json_gzip
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.analysis.ptfm_objective_compare import bind_arm as bind086,scientific_screen
from tools.experiments.ptfm_fit import endpoint_model,bind_paired_control
from tools.experiments.ptfm_objective_fit import bind_ptfm_control
ROOT=Path(__file__).resolve().parents[2]


def production_contract(config,manifest,alignment):
    configuration(config);validate_manifest(manifest)
    kind=position_objective(config)
    if kind not in POSITION_OBJECTIVES or config!=read_json(ROOT/f'configs/ptfm-{kind}-level1.json'):
        raise ValueError('091 frozen4000 configuration differs')
    if manifest!=read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):raise ValueError('091 full Level1 manifest differs')
    if alignment!=read_json(ROOT/'configs/data/mtad-level1-alignment.json'):raise ValueError('091 alignment differs')


def bind_position_control(path,source,paired,residual,manifest):
    """086B original full logs, actual endpoints and complete cached V population."""
    bound=bind086(path,'position-only',expected_config=read_json(ROOT/'configs/ptfm-position-only-level1.json'),
        expected_manifest=manifest)
    for key,expected in (('source_system',source),('paired_control',paired),('paired_residual',residual)):
        if Path(bound['run'].get(key,'')).resolve()!=Path(expected).resolve():
            raise ValueError('091 source086B common source differs/'+key)
    return dict(status='BOUND',path=str(Path(path).resolve()),source_system=str(Path(source).resolve()),
        configuration=bound['run']['configuration'],successful_updates=len(bound['rows']),
        sampled_anchors=bound['fit']['exposure']['sampled_anchors'],scene_ids=manifest['splits']['V_select'],
        initial_and_final_actual_checkpoints_verified=True,actual_successful_batches_match076B=True,
        normalizer_exact=True,complete_cached_V_population_verified=True)


def run(args):
    source,output,paired,ptfm,position=[Path(v).resolve() for v in
        (args.source_system,args.output,args.paired_control,args.paired_residual,args.paired_position_only)]
    for protected in (source.parent,paired,ptfm,position):
        if output==protected or protected in output.parents or output in protected.parents:
            raise ValueError('091 output must preserve source')
    if output.exists():raise FileExistsError('091 no overwrite/retry')
    if args.cpu_threads!=4 or args.time_policy!='advisory':raise ValueError('091 fixed CPU4/advisory')
    torch.set_num_threads(4)
    if args.device=='cuda:0' and not torch.cuda.is_available():raise ValueError('091 no CUDA fallback')
    config,manifest,alignment=[read_json(p) for p in (args.config,args.manifest,args.alignment)]
    production_contract(config,manifest,alignment);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    state=dict(scope='091-ptfm-position-objective',status='RUNNING',configuration=config,source_system=str(source),
        paired_control=str(paired),paired_residual=str(ptfm),paired_position_only=str(position),
        optimizer_updates=0,test_opened=False,validation_selection=False,completed_endpoints=[],
        old066_acceptance='REJECTED',cpu_threads=4,train_device=args.device,inference_device='cpu',advisory_reached=False)
    write_json(output/'run.json',state)
    def check():
        if time.monotonic()-started>=config['training']['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached']=True;write_json(output/'run.json',state)
            print('ADVISORY:091 continues fixed attempt, no retry, scan or extra updates',flush=True)
    def phase(label):
        state['phase']=label;state['elapsed_seconds']=time.monotonic()-started;write_json(output/'run.json',state)
        print('091 phase: '+label,flush=True);check()
    try:
        phase('bind056_076B_086B_072C');_bind_source(source);bind_paired_control(paired,source)
        binding=bind_ptfm_control(ptfm,source,manifest,config)
        if Path(binding['paired_control'] or '').resolve()!=paired:raise ValueError('091 source076B paired072C source differs')
        binding086=bind_position_control(position,source,paired,ptfm,manifest)
        write_json(output/'source-binding.json',dict(residual076B=binding,position086B=binding086,paired072C=str(paired)))
        metadata=_load_metadata(source,manifest,1000)
        metadata['component_ids']={sid:str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('prepareT2100V300')
        dataset=load_timed_dataset(args.archive,manifest,train_splits=['T_core'],data_config=config['data'],
            alignment_by_scene=alignment,check=check,motion_observation_policy='canonical_all')
        for key,split in (('train','T_core'),('validation','V_select')):
            if [str(s['scene_id']) for s in dataset[key]]!=manifest['splits'][split]:raise ValueError('091 prepared population differs')
        write_json(output/'preparation.json',dataset['report']);phase('T_calibration_fixed_probe_and_fit4000')
        training=train_ptfm(dataset.pop('train'),config,output/'fit',device=args.device,check=check,
            paired_reference_fit=paired/'fit',paired_ptfm_fit=ptfm/'fit',paired_position_fit=position/'fit')
        state['optimizer_updates']=training['optimizer_updates']
        if torch.cuda.is_initialized():torch.cuda.empty_cache()
        endpoints={}
        for label,filename,step in (('initial','step-0.pt',0),('final','final.pt',config['training']['updates'])):
            phase('CPU_'+label);model=endpoint_model(output/'fit'/filename,step,config,training)
            report=evaluate_ptfm(model,dataset['validation'],source,metadata,output/label,check,initial=label=='initial')
            provenance=dict(path=str(output/'fit'/filename),step=step,actual_updates=step,
                configured_final_updates=config['training']['updates'],routing_semantics=config['routing_semantics'])
            report.update(configuration=config,checkpoint=provenance);write_json(output/label/'summary.json',report)
            record=read_json(output/label/'run.json');record.update(configuration=config,checkpoint=provenance)
            write_json(output/label/'run.json',record);endpoints[label]=provenance;state['completed_endpoints'].append(label);del model
            if label=='initial':
                phase('initial076B_086B_replay')
                for name,control in (('ptfm',ptfm),('position-only',position)):
                    write_json_gzip(output/f'initial-{name}-comparison.json.gz',
                        compare_objective_control(output/'initial',control/'initial',initial=True))
        phase('compare_endpoints');write_json(output/'endpoint-comparison.json',compare_endpoints(output/'initial',output/'final'))
        for name,control in (('ptfm',ptfm),('position-only',position)):
            phase('compare_'+name)
            write_json_gzip(output/f'{name}-comparison.json.gz',compare_objective_control(output/'final',control/'final'))
        phase('compare072C');transformer_compare=compare_sequence_control(output/'final',paired/'final')
        write_json_gzip(output/'transformer-comparison.json.gz',transformer_compare)
        with gzip.open(output/'final/scene-summary.json.gz','rt') as handle:scene_rows=json.load(handle)
        screens=dict(versus_fixed_prior095=scientific_screen(read_json(output/'final/summary.json'),'ptfm_adapted','fixed_prior095',scene_rows),
            versus_transformer=scientific_screen(transformer_compare,'ptfm_adapted','learned_sequence'))
        write_json(output/'scientific-screen.json',screens)
        summary=dict(scientific_screens=screens,status='COMPLETED',scope=state['scope'],configuration=config,source_system=str(source),
            optimizer_updates=training['optimizer_updates'],test_opened=False,endpoints=endpoints,
            interpretation='T-initial-mean-calibrated position-only objective strategy bridge. Initial mean matching is not gradient/Adam/clipping/L2 equivalence or author reproduction. Auxiliary heads remain unqualified.',
            normalizer=training['normalizer'],loss_calibration=training['loss_calibration'],
            fixed_training_probe='fit/fixed-training-probe.json',paired_control=training['paired_control'],
            paired_ptfm=training['paired_ptfm_control'],paired_position_only=training['paired_position_control'],
            initial_comparison='initial-position-only-comparison.json.gz',ptfm_comparison='ptfm-comparison.json.gz',
            position_only_comparison='position-only-comparison.json.gz',transformer_comparison='transformer-comparison.json.gz')
        write_json(output/'summary.json',summary);state.update(status='COMPLETED',phase='completed',elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json',state);return summary
    except BaseException as error:
        if (output/'fit/run.json').exists():state['optimizer_updates']=read_json(output/'fit/run.json').get('optimizer_updates',0)
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json',state);raise
    finally:write_json(output/'execution.json',dict(arguments=vars(args),configuration=config,test_opened=False,old066_acceptance='REJECTED'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','alignment','config','source-system','paired-control','paired-residual','paired-position-only','output'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--archive',nargs='+',required=True);parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--cpu-threads',type=int,default=4);parser.add_argument('--time-policy',choices=('advisory',),default='advisory')
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('091 interrupted by signal')
    signal.signal(signal.SIGTERM,stop);result=run(args);print(result['status'],flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
