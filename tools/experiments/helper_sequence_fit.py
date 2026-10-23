"""078 one fixed helper-aggregation run, paired to original072C sampling.

No checkpoint selection, threshold changes, association reruns or automatic retry.
All three arms use seed20261023,4000 updates and unchanged full T2100/V300.
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
from association.evaluation.helper_sequence_position import evaluate_helper_sequence
from association.evaluation.sequence_position import compare_endpoints
from association.training.helper_sequence import configuration, train_helper_sequence
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.sequence_position_fit import (ROOT, bind_linear_control, endpoint_model,
    production_contract as linear_production_contract)

CONFIGS = {mode: f'helper-sequence-{mode.replace("_", "-")}-level1.json'
           for mode in ('post_pool', 'pre_pool', 'weighted_pool')}


def production_contract(config, manifest, alignment):
    configuration(config)
    aggregation = config['model']['aggregation']
    if config != read_json(ROOT/'configs'/CONFIGS[aggregation]):
        raise ValueError('078 requires the frozen aggregation-specific4000-update configuration')
    linear = read_json(ROOT/'configs/sequence-position-transformer-extended-level1.json')
    linear_production_contract(linear, manifest, alignment, profile='extended4000')


def bind_reference(path, config, source, manifest):
    # The new representation differs, but every inherited layer, ordinary
    # configuration and successful training sample has a specific old control.
    from copy import deepcopy
    from association.models.sequence_position import SCOPE, ROUTING_SEMANTICS
    linear = deepcopy(config)
    linear.update(scope=SCOPE, routing_semantics=ROUTING_SEMANTICS)
    linear['model'].pop('aggregation')
    linear['model']['backbone'] = 'transformer'
    return bind_linear_control(path, linear, source, manifest)


def run(args):
    source, reference, output = [Path(p).resolve() for p in (args.source_system, args.paired_control, args.output)]
    for protected in (source.parent, reference):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('078 output overlaps immutable source/control')
    if output.exists():
        raise FileExistsError('078 output exists; no reuse/retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('078 fixes CPU4 and advisory wall time')
    torch.set_num_threads(4)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; no fallback')
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    output.mkdir(parents=True)
    started = time.monotonic()
    state = dict(scope='078-helper-representation-screen', status='RUNNING', phase='source_binding',
        configuration=config, aggregation=config['model']['aggregation'], source_system=str(source),
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
        print('078 phase: '+name, flush=True)
        check()

    try:
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        bind_reference(reference, config, source, manifest)
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'], data_config=config['data'],
            alignment_by_scene=alignment, check=check, motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('078 complete '+split+' population/order differs')
        preparation = dataset['report']
        write_json(output/'preparation.json', preparation)
        phase('fit4000')
        training = train_helper_sequence(dataset.pop('train'), config, output/'fit', device=args.device,
            check=check, paired_reference_fit=reference/'fit')
        state['optimizer_updates'] = training['optimizer_updates']
        write_json(output/'run.json', state)
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        endpoints = {}
        for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', config['training']['updates'])):
            phase('evaluate_cpu_'+label)
            model = endpoint_model(output/'fit'/filename, step, config, training)
            evaluated = evaluate_helper_sequence(model, dataset['validation'], source, metadata,
                output/label, check=check, initial=(label == 'initial'))
            provenance = dict(path=str(output/'fit'/filename), step=step, actual_updates=step,
                configured_final_updates=config['training']['updates'], routing_semantics=config['routing_semantics'],
                position_loss=config['position_loss'], aggregation=config['model']['aggregation'])
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
            endpoint_comparison='endpoint-comparison.json', source_system=str(source), paired_control=str(reference),
            training_pairing=training['paired_control'], inference_device='cpu', test_opened=False,
            scientific_status='single-seed helper-representation screening; no automatic method promotion',
            limitations=['Same reused V300 and shared identities; not independent generalization.',
                'A/B equal parameter count is not equal FLOPs; C adds16 scoring parameters.',
                'Learned feature-pooling weights are not physical confidence or causal attribution.',
                'All four physical outputs retained; speed/heading are frozen Mean, not learned here.',
                'Training synthetic groups differ from frozen056 deployment groups.'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
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
