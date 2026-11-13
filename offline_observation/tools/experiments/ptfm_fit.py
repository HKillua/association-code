"""074C one fixed4000 PTFM adaptation fit followed by CPU initial/final V300."""
from __future__ import annotations
import argparse,signal,sys,time
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from association.data.archive import validate_manifest
from association.data.dataset import load_timed_dataset
from association.training.ptfm import configuration,train_ptfm,load_checkpoint
from association.evaluation.ptfm import evaluate_ptfm,compare_endpoints,compare_sequence_control
from association.utils.runtime import read_json,write_json,write_json_gzip
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
ROOT=Path(__file__).resolve().parents[2]


def production_contract(config,manifest,alignment):
    configuration(config);validate_manifest(manifest)
    if config!=read_json(ROOT/'configs/ptfm-adapted-level1.json'):raise ValueError('074C frozen4000 configuration differs')
    if manifest!=read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):raise ValueError('074C full Level1 manifest differs')
    if alignment!=read_json(ROOT/'configs/data/mtad-level1-alignment.json'):raise ValueError('074C alignment differs')


def endpoint_model(path,step,config,training):
    if (training.get('status')!='COMPLETED' or training.get('configuration')!=config
            or training.get('optimizer_updates')!=config['training']['updates'] or step not in (0,config['training']['updates'])):
        raise ValueError('074C endpoint requires completed fixed training')
    model,payload=load_checkpoint(path,device='cpu')
    if (payload['configuration']!=config or payload['step']!=step or payload['actual_updates']!=step
            or payload['normalizer']!=training.get('normalizer')):
        raise ValueError('074C checkpoint step/configuration mismatch')
    return model


def bind_paired_control(path,source):
    from association.evaluation.position_sources import _metadata
    bound=_metadata(path,'072C_transformer4000')
    if Path(bound['source_system']).resolve()!=source:raise ValueError('074C control must bind the same056 final1000 source')
    return bound


def run(args):
    source,output,paired=Path(args.source_system).resolve(),Path(args.output).resolve(),Path(args.paired_control).resolve()
    for protected in (source.parent,paired):
        if output==protected or protected in output.parents or output in protected.parents:raise ValueError('074C output must preserve source')
    if output.exists():raise FileExistsError('074C no overwrite/retry')
    if args.cpu_threads!=4 or args.time_policy!='advisory':raise ValueError('074C fixed CPU4/advisory')
    torch.set_num_threads(4)
    if args.device=='cuda:0' and not torch.cuda.is_available():raise ValueError('074C no CUDA fallback')
    config,manifest,alignment=[read_json(p) for p in (args.config,args.manifest,args.alignment)]
    production_contract(config,manifest,alignment);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    state=dict(scope='074C-ptfm-task-adapted',status='RUNNING',configuration=config,source_system=str(source),
        optimizer_updates=0,test_opened=False,validation_selection=False,completed_endpoints=[],
        old066_acceptance='REJECTED',cpu_threads=4,train_device=args.device,inference_device='cpu',advisory_reached=False)
    write_json(output/'run.json',state)
    def check():
        if time.monotonic()-started>=config['training']['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached']=True;write_json(output/'run.json',state)
            print('ADVISORY:074C continues the same fixed attempt, no retry or extra updates',flush=True)
    def phase(label):
        state['phase']=label;state['elapsed_seconds']=time.monotonic()-started;write_json(output/'run.json',state)
        print('074C phase: '+label,flush=True);check()
    try:
        phase('bind056_and072C');_bind_source(source);bind_paired_control(paired,source);metadata=_load_metadata(source,manifest,1000)
        state['paired_control']=str(paired)
        metadata['component_ids']={sid:str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('prepareT2100V300')
        dataset=load_timed_dataset(args.archive,manifest,train_splits=['T_core'],data_config=config['data'],
            alignment_by_scene=alignment,check=check,motion_observation_policy='canonical_all')
        for key,split in (('train','T_core'),('validation','V_select')):
            if [str(s['scene_id']) for s in dataset[key]]!=manifest['splits'][split]:raise ValueError('074C prepared population differs')
        write_json(output/'preparation.json',dataset['report']);phase('fit4000')
        training=train_ptfm(dataset.pop('train'),config,output/'fit',device=args.device,check=check,paired_reference_fit=paired/'fit')
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
        phase('compare_endpoints');write_json(output/'endpoint-comparison.json',compare_endpoints(output/'initial',output/'final'))
        phase('compare072C');write_json_gzip(output/'transformer-comparison.json.gz',compare_sequence_control(output/'final',paired/'final'))
        summary=dict(status='COMPLETED',scope=state['scope'],configuration=config,source_system=str(source),
            optimizer_updates=training['optimizer_updates'],test_opened=False,endpoints=endpoints,
            interpretation='Full-state task-adapted PTFM with uncertainty task loss; neither author reproduction nor convergence evidence',
            normalizer=training['normalizer'],paired_control=training['paired_control'],transformer_comparison='transformer-comparison.json.gz')
        write_json(output/'summary.json',summary);state.update(status='COMPLETED',phase='completed',elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json',state);return summary
    except BaseException as error:
        if (output/'fit/run.json').exists():state['optimizer_updates']=read_json(output/'fit/run.json').get('optimizer_updates',0)
        state.update(status='INTERRUPTED' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',error=f'{type(error).__name__}: {error}')
        write_json(output/'run.json',state);raise
    finally:write_json(output/'execution.json',dict(arguments=vars(args),configuration=config,test_opened=False,old066_acceptance='REJECTED'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','alignment','config','source-system','paired-control','output'):parser.add_argument('--'+k,required=True)
    parser.add_argument('--archive',nargs='+',required=True);parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--cpu-threads',type=int,default=4);parser.add_argument('--time-policy',choices=('advisory',),default='advisory')
    args=parser.parse_args(argv)
    def stop(signum,frame):raise KeyboardInterrupt('074C interrupted by signal')
    signal.signal(signal.SIGTERM,stop);result=run(args);print(result['status'],flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
