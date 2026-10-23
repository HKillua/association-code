"""076B fixed4000 PTFM position-prior residual adaptation; other tasks unchanged."""
from __future__ import annotations
import argparse,signal,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from association.data.archive import validate_manifest
from association.data.dataset import load_timed_dataset
from association.training.ptfm import configuration,train_ptfm
from association.evaluation.ptfm import evaluate_ptfm,compare_endpoints,compare_sequence_control,compare_ptfm_control
from association.utils.runtime import read_json,write_json,write_json_gzip
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.ptfm_fit import endpoint_model,bind_paired_control
ROOT=Path(__file__).resolve().parents[2]


def production_contract(config,manifest,alignment):
    configuration(config);validate_manifest(manifest)
    if config!=read_json(ROOT/'configs/ptfm-residual-level1.json'):raise ValueError('076B frozen4000 configuration differs')
    if manifest!=read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):raise ValueError('076B full Level1 manifest differs')
    if alignment!=read_json(ROOT/'configs/data/mtad-level1-alignment.json'):raise ValueError('076B alignment differs')


def bind_ptfm_control(path,source,manifest,config):
    """Bind both original074C CPU endpoints and actual fixed checkpoints, no selected."""
    path,source=Path(path).resolve(),Path(source).resolve()
    expected=read_json(ROOT/'configs/ptfm-adapted-level1.json');steps=expected['training']['updates']
    for key in ('data','training','train_splits','motion_observation_policy'):
        if config[key]!=expected[key]:raise ValueError('076B/074C fixed training protocol differs/'+key)
    run,summary,fit=[read_json(path/name) for name in ('run.json','summary.json','fit/run.json')]
    for record in (run,summary,fit):
        if (record.get('status')!='COMPLETED' or record.get('configuration')!=expected
                or record.get('optimizer_updates')!=steps or record.get('test_opened') is not False):
            raise ValueError('076B requires original074C completed frozen training')
    if (run.get('scope')!='074C-ptfm-task-adapted' or summary.get('scope')!=run['scope']
            or run.get('validation_selection') is not False or run.get('completed_endpoints')!=['initial','final']
            or fit.get('final_checkpoint')!='final.pt' or fit.get('training_scene_ids')!=manifest['splits']['T_core']
            or any(Path(record.get('source_system','')).resolve()!=source for record in (run,summary))):
        raise ValueError('076B original074C source/selection/population differs')
    if fit.get('normalizer')!=summary.get('normalizer') or fit.get('normalizer_sampler_replay_exact') is not True:
        raise ValueError('076B original074C normalizer binding differs')
    for label,filename,step in (('initial','step-0.pt',0),('final','final.pt',steps)):
        with torch.random.fork_rng(devices=[]):model=endpoint_model(path/'fit'/filename,step,expected,fit)
        del model
        end=read_json(path/label/'summary.json')
        for record in (read_json(path/label/'run.json'),end):
            checkpoint=record.get('checkpoint',{})
            if (record.get('status')!='COMPLETED' or record.get('configuration')!=expected
                    or record.get('device')!='cpu' or record.get('optimizer_updates')!=0
                    or record.get('initial') is not (label=='initial') or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('scores_recomputed') is not False
                    or record.get('threshold_reselected') is not False or Path(record.get('source_system','')).resolve()!=source
                    or checkpoint.get('step')!=step or checkpoint.get('actual_updates')!=step
                    or checkpoint.get('configured_final_updates')!=steps
                    or checkpoint.get('routing_semantics')!=expected['routing_semantics']
                    or Path(checkpoint.get('path','')).resolve()!=path/'fit'/filename):
                raise ValueError('076B original074C CPU endpoint binding differs/'+label)
        if end.get('scene_ids')!=manifest['splits']['V_select']:
            raise ValueError('076B original074C validation population differs')
        components={sid:str(c) for c,group in end['components'].items() for sid in group['scene_ids']}
        if components!={sid:str(manifest['components'][sid]) for sid in end['scene_ids']}:
            raise ValueError('076B original074C component registry differs')
        for directory in ('predictions','system-predictions'):
            if {p.name for p in (path/label/directory).glob('*.npz')}!={f'states-{sid}.npz' for sid in end['scene_ids']}:
                raise ValueError('076B original074C endpoint inventory differs')
        if summary.get('endpoints',{}).get(label)!=end['checkpoint']:
            raise ValueError('076B original074C checkpoint summary differs')
    return dict(status='BOUND',path=str(path),source_system=str(source),scene_ids=manifest['splits']['V_select'],
                initial_and_final_actual_checkpoints_verified=True,configuration=expected,paired_control=run.get('paired_control'))


def run(args):
    source,output,paired,ptfm=[Path(v).resolve() for v in (args.source_system,args.output,args.paired_control,args.paired_ptfm)]
    for protected in (source.parent,paired,ptfm):
        if output==protected or protected in output.parents or output in protected.parents:raise ValueError('076B output must preserve source')
    if output.exists():raise FileExistsError('076B no overwrite/retry')
    if args.cpu_threads!=4 or args.time_policy!='advisory':raise ValueError('076B fixed CPU4/advisory')
    torch.set_num_threads(4)
    if args.device=='cuda:0' and not torch.cuda.is_available():raise ValueError('076B no CUDA fallback')
    config,manifest,alignment=[read_json(p) for p in (args.config,args.manifest,args.alignment)]
    production_contract(config,manifest,alignment);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    state=dict(scope='076B-ptfm-prior095-residual',status='RUNNING',configuration=config,source_system=str(source),
        paired_control=str(paired),paired_ptfm=str(ptfm),optimizer_updates=0,test_opened=False,validation_selection=False,
        completed_endpoints=[],old066_acceptance='REJECTED',cpu_threads=4,train_device=args.device,inference_device='cpu',advisory_reached=False)
    write_json(output/'run.json',state)
    def check():
        if time.monotonic()-started>=config['training']['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached']=True;write_json(output/'run.json',state)
            print('ADVISORY:076B continues fixed attempt, no retry or extra updates',flush=True)
    def phase(label):
        state['phase']=label;state['elapsed_seconds']=time.monotonic()-started;write_json(output/'run.json',state)
        print('076B phase: '+label,flush=True);check()
    try:
        phase('bind056_074C_072C');_bind_source(source);bind_paired_control(paired,source)
        binding=bind_ptfm_control(ptfm,source,manifest,config)
        if Path(binding['paired_control'] or '').resolve()!=paired:raise ValueError('076B original074C paired072C source differs')
        write_json(output/'source-binding.json',binding)
        metadata=_load_metadata(source,manifest,1000);metadata['component_ids']={sid:str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('prepareT2100V300')
        dataset=load_timed_dataset(args.archive,manifest,train_splits=['T_core'],data_config=config['data'],
            alignment_by_scene=alignment,check=check,motion_observation_policy='canonical_all')
        for key,split in (('train','T_core'),('validation','V_select')):
            if [str(s['scene_id']) for s in dataset[key]]!=manifest['splits'][split]:raise ValueError('076B prepared population differs')
        write_json(output/'preparation.json',dataset['report']);phase('fit4000')
        training=train_ptfm(dataset.pop('train'),config,output/'fit',device=args.device,check=check,
            paired_reference_fit=paired/'fit',paired_ptfm_fit=ptfm/'fit')
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
                phase('initial074C_nonposition_replay')
                write_json_gzip(output/'initial-ptfm-comparison.json.gz',compare_ptfm_control(output/'initial',ptfm/'initial',initial=True))
        phase('compare_endpoints');write_json(output/'endpoint-comparison.json',compare_endpoints(output/'initial',output/'final'))
        phase('compare074C');write_json_gzip(output/'ptfm-comparison.json.gz',compare_ptfm_control(output/'final',ptfm/'final'))
        phase('compare072C');write_json_gzip(output/'transformer-comparison.json.gz',compare_sequence_control(output/'final',paired/'final'))
        summary=dict(status='COMPLETED',scope=state['scope'],configuration=config,source_system=str(source),
            optimizer_updates=training['optimizer_updates'],test_opened=False,endpoints=endpoints,
            interpretation='PTFM task adaptation with physical position prior and zero initialized residual; other outputs may change through shared training, no attribute composition; not author reproduction',
            normalizer=training['normalizer'],paired_control=training['paired_control'],paired_ptfm=training['paired_ptfm_control'],
            initial_comparison='initial-ptfm-comparison.json.gz',ptfm_comparison='ptfm-comparison.json.gz',transformer_comparison='transformer-comparison.json.gz')
        write_json(output/'summary.json',summary);state.update(status='COMPLETED',phase='completed',elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json',state);return summary
    except BaseException as error:
        if (output/'fit/run.json').exists():state['optimizer_updates']=read_json(output/'fit/run.json').get('optimizer_updates',0)
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json',state);raise
    finally:write_json(output/'execution.json',dict(arguments=vars(args),configuration=config,test_opened=False,old066_acceptance='REJECTED'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','alignment','config','source-system','paired-control','paired-ptfm','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--archive',nargs='+',required=True);parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--cpu-threads',type=int,default=4);parser.add_argument('--time-policy',choices=('advisory',),default='advisory')
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('076B interrupted by signal')
    signal.signal(signal.SIGTERM,stop);result=run(args);print(result['status'],flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
