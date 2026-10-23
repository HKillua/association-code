"""Frozen helper-aggregation run, paired to a same-seed linear reference.

No checkpoint selection, threshold changes, association reruns or automatic retry.
078 screens three arms at seed20261023;081 repeats only B/C at seed20261021.
Both profiles retain4000 updates and unchanged full T2100/V300.
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
REPLICA_CONFIGS = {mode: f'helper-sequence-{mode.replace("_", "-")}-replica-level1.json'
                   for mode in ('pre_pool', 'weighted_pool')}
PROFILE_CONFIGS = {'screen078': CONFIGS, 'replica081': REPLICA_CONFIGS}
PROFILE_REFERENCES = {
    'screen078': ('extended4000', 'sequence-position-transformer-extended-level1.json', '072C_transformer4000'),
    'replica081': ('replica4000', 'sequence-position-transformer-extended-replica-level1.json', '073C_transformer4000_seed21'),
}


def frozen_configuration(config, profile='screen078'):
    """Explicit profiles reject other seeds, post-pool replicas and mixed tables."""
    configuration(config)
    modes = PROFILE_CONFIGS.get(profile, {})
    filename = modes.get(config['model']['aggregation'])
    if filename is None or config != read_json(ROOT/'configs'/filename):
        raise ValueError('helper run requires the frozen aggregation/profile configuration')
    return config


def production_contract(config, manifest, alignment, profile='screen078'):
    frozen_configuration(config, profile)
    linear_profile, filename, _ = PROFILE_REFERENCES[profile]
    linear = read_json(ROOT/'configs'/filename)
    linear_production_contract(linear, manifest, alignment, profile=linear_profile)


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
    profile = getattr(args, 'profile', 'screen078')
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
    production_contract(config, manifest, alignment, profile=profile)
    output.mkdir(parents=True)
    started = time.monotonic()
    experiment = '081' if profile == 'replica081' else '078'
    state = dict(scope=experiment+'-helper-representation-'+('replication' if profile == 'replica081' else 'screen'),
        profile=profile, status='RUNNING', phase='source_binding',
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
        print(experiment+' phase: '+name, flush=True)
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
        summary = dict(scope=state['scope'], profile=profile, status='COMPLETED', configuration=config,
            optimizer_updates=training['optimizer_updates'], endpoints=endpoints,
            endpoint_comparison='endpoint-comparison.json', source_system=str(source), paired_control=str(reference),
            training_pairing=training['paired_control'], inference_device='cpu', test_opened=False,
            scientific_status=('matched second-seed B/C tail-signal check; no automatic method promotion'
                if profile == 'replica081' else 'single-seed helper-representation screening; no automatic method promotion'),
            limitations=['Same reused V300 and shared identities; not independent generalization.',
                'post_pool/pre_pool have107602 parameters; weighted_pool has107618; this does not imply equal FLOPs.',
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
    parser.add_argument('--profile', choices=tuple(PROFILE_CONFIGS), default='screen078',
                        help='explicit frozen seed/control; replica081 permits only pre_pool and weighted_pool')
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
