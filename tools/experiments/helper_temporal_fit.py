"""087 local temporal implementation: one frozen paired4000-update arm.
No V selection, source changes or automatic retry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from association.data.dataset import load_timed_dataset
from association.evaluation.helper_temporal import evaluate_helper_temporal
from association.evaluation.sequence_position import compare_endpoints
from association.training.helper_temporal import configuration, train_helper_temporal
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.sequence_position_fit import (ROOT, bind_linear_control, endpoint_model,
    production_contract as linear_production_contract)

CONFIGS = {mode: f'helper-temporal-{mode}-level1.json' for mode in ('shuffled','preserved')}


def frozen_configuration(config):
    configuration(config)
    if config != read_json(ROOT/'configs'/CONFIGS[config['model']['correspondence']]):
        raise ValueError('087 requires the exact frozen temporal configuration')
    return config


def production_contract(config,manifest,alignment):
    frozen_configuration(config)
    original=read_json(ROOT/'configs/sequence-position-transformer-extended-level1.json')
    linear_production_contract(original,manifest,alignment,profile='extended4000')


def bind_reference(path,config,source,manifest):
    from copy import deepcopy
    from association.models.sequence_position import SCOPE, ROUTING_SEMANTICS
    linear=deepcopy(config);linear.update(scope=SCOPE,routing_semantics=ROUTING_SEMANTICS)
    linear['model'].pop('correspondence');linear['model']['backbone']='transformer'
    return bind_linear_control(path,linear,source,manifest)


def run(args):
    source, reference, output = [Path(p).resolve() for p in (args.source_system, args.paired_control, args.output)]
    for protected in (source.parent, reference, Path(args.pre_pool_control).resolve()):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('087 output overlaps immutable source/control')
    if output.exists():
        raise FileExistsError('087 output exists; no reuse/retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('087 fixes CPU4 and advisory wall time')
    torch.set_num_threads(4)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; no fallback')
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    output.mkdir(parents=True)
    started = time.monotonic()
    experiment = '087'
    state = dict(scope='087-helper-temporal-position', status='RUNNING', phase='source_binding',
        configuration=config, correspondence=config['model']['correspondence'], source_system=str(source),
        paired_control=str(reference), train_device=args.device, inference_device='cpu', cpu_threads=4,
        optimizer_updates=0, completed_endpoints=[], validation_selection=False, association_recomputed=False,
        groups_recomputed=False, threshold_reselected=False, test_opened=False, retries=0,
        old066_acceptance='REJECTED', advisory_reached=False, time_policy='advisory')
    write_json(output/'run.json', state)
    preparation = None

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed)
            write_json(output/'run.json', state)
            print('ADVISORY: declared wall time exceeded; continuing single attempt', flush=True)

    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print(experiment+' phase: '+name, flush=True)
        check()

    try:
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        bind_reference(reference, config, source, manifest)
        from tools.analysis.compare_helper_sequence import helper_metadata
        mode,pre=helper_metadata(args.pre_pool_control)
        if mode!='pre_pool' or pre['source_system']!=str(source): raise ValueError('087 requires original078B pre_pool source')
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'], data_config=config['data'],
            alignment_by_scene=alignment, check=check, motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('087 complete '+split+' population/order differs')
        preparation = dataset['report']
        write_json(output/'preparation.json', preparation)
        phase('fit4000')
        training = train_helper_temporal(dataset.pop('train'), config, output/'fit', device=args.device,
            check=check, paired_reference_fit=reference/'fit')
        state['optimizer_updates'] = training['optimizer_updates']
        write_json(output/'run.json', state)
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        endpoints = {}
        for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', config['training']['updates'])):
            phase('evaluate_cpu_'+label)
            model = endpoint_model(output/'fit'/filename, step, config, training)
            evaluated = evaluate_helper_temporal(model, dataset['validation'], source, metadata,
                output/label, check=check, initial=(label == 'initial'), route_seed=config['training']['seed']+87100)
            provenance = dict(path=str(output/'fit'/filename), step=step, actual_updates=step,
                configured_final_updates=config['training']['updates'], routing_semantics=config['routing_semantics'],
                position_loss=config['position_loss'], correspondence=config['model']['correspondence'])
            evaluated.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'summary.json', evaluated)
            record = read_json(output/label/'run.json')
            record.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'run.json', record)
            endpoints[label] = dict(directory=str(output/label), checkpoint=provenance)
            state['completed_endpoints'].append(label)
            del model
        phase('paired_endpoint_comparison')
        write_json(output/'endpoint-comparison.json', compare_endpoints(output/'initial', output/'final'))
        summary = dict(scope=state['scope'], status='COMPLETED', configuration=config,
            optimizer_updates=training['optimizer_updates'], endpoints=endpoints,
            endpoint_comparison='endpoint-comparison.json', source_system=str(source), paired_control=str(reference), pre_pool_control=str(Path(args.pre_pool_control).resolve()),
            training_pairing=training['paired_control'], inference_device='cpu', test_opened=False,
            scientific_status='single-seed temporal correspondence screen; beating shuffled is insufficient without strong072C gain',
            limitations=['Same reused V300 and shared identities; not independent generalization.',
                'Both temporal arms have109042 parameters; reference072C has105154. Shuffling creates artificial temporal switches.',
                'Learned feature-pooling weights are not physical confidence or causal attribution.',
                'All four physical outputs retained; speed/heading are frozen Mean, not learned here.',
                'Training synthetic groups differ from frozen056 deployment groups.'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        # Binding readers require completed train/evaluation. Failure in the
        # subsequent cached comparison still marks this outer attempt FAILED.
        from tools.analysis.compare_helper_temporal import run as compare_cached
        mode=config['model']['correspondence']
        comparison=compare_cached(argparse.Namespace(
            shuffled_run=str(output) if mode=='shuffled' else None,
            preserved_run=str(output) if mode=='preserved' else None,
            linear_run=str(reference),pre_pool_run=args.pre_pool_control,
            output=str(output/'reference-comparison'),cpu_threads=4))
        summary.update(reference_comparison='reference-comparison/summary.json.gz',
            scientific_screen=comparison['scientific_screen'][mode])
        write_json(output/'summary.json',summary)
        state.update(phase='completed',reference_comparison=summary['reference_comparison'])
        write_json(output/'run.json',state)
        return summary
    except BaseException as error:
        if (output/'fit/run.json').exists():
            state['optimizer_updates'] = read_json(output/'fit/run.json').get('optimizer_updates', 0)
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            preparation_report=str(output/'preparation.json') if preparation is not None else None,
            test_opened=False, model_forward='one new training run and CPU step0/final full-group evaluation'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'alignment', 'config', 'source-system', 'paired-control', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    parser.add_argument('--pre-pool-control', required=True)
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f'cancelled by signal {signum}')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        result = run(args)
        print(f'{result["status"]}: {Path(args.output).resolve()}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
