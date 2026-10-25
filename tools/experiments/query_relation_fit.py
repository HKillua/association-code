"""098: one fixed source-final set-fusion run, full V endpoints, no retry.

Four frozen configurations pair generic/relation with the two existing seeds.
Source-only preflight reads completed sources and makes no CUDA/model forward.
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
from association.evaluation.query_relation import evaluate_query_relation
from association.training.query_relation import (configuration, source_configuration,
    train_query_relation, load_checkpoint)
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.analysis.compare_frozen_position_adapter import adapter_metadata, verify_pairing as verify_budget_pairing
from tools.analysis.compare_position_benchmarks import _metadata as source_metadata
from tools.experiments.sequence_position_fit import ROOT, bind_linear_control, production_contract as original_contract

SCOPE = '098-source-final-query-relation-position'
CONFIGS = {
    (20261023, 'generic'): 'query-relation-generic-level1.json',
    (20261023, 'relation'): 'query-relation-biased-level1.json',
    (20261021, 'generic'): 'query-relation-generic-replica-level1.json',
    (20261021, 'relation'): 'query-relation-biased-replica-level1.json',
}
EXPERIMENTS = {key: '098'+arm for key, arm in zip(CONFIGS, 'ABCD')}
PROFILES = {
    20261023: ('extended4000', '072C_transformer4000', '090'),
    20261021: ('replica4000', '073C_transformer4000_seed21', '097'),
}


def frozen_configuration(config):
    configuration(config)
    key = (config['training']['seed'], config['model']['mode'])
    if key not in CONFIGS or config != read_json(ROOT/'configs'/CONFIGS[key]):
        raise ValueError('098 requires one of four frozen seed/mode configurations')
    return config


def production_contract(config, manifest, alignment):
    frozen_configuration(config)
    original_contract(source_configuration(config), manifest, alignment,
        profile=PROFILES[config['training']['seed']][0])


def check_source(args):
    """Fail closed on incomplete/failed sources; never substitute or open a test."""
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('098 fixes CPU4 and advisory wall time')
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    source, reference, budget = [Path(p).resolve() for p in
        (args.source_system, args.paired_control, args.budget_control)]
    for path in (source, reference, budget):
        if read_json(path/'run.json').get('status') != 'COMPLETED':
            raise ValueError('098 requires completed source: '+str(path))
    bind_linear_control(reference, source_configuration(config), source, manifest)
    _bind_source(source)
    metadata = _load_metadata(source, manifest, 1000)
    metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
    _, source_arm, budget_profile = PROFILES[config['training']['seed']]
    original = source_metadata(reference, source_arm)
    name, control = adapter_metadata(budget, profile=budget_profile)
    if (name != 'finetune' or control['paired_control'] != reference
            or any(control[key] != original[key] for key in ('source_system', 'scene_ids', 'components'))):
        raise ValueError('098 budget baseline must be same-seed, same-source full fine-tuning')
    pairing = verify_budget_pairing({'linear': original, 'finetune': control})
    return dict(metadata=metadata, budget_pairing=pairing, source=str(source), reference=str(reference),
        budget=str(budget), config=config, manifest=manifest, alignment=alignment)


def run(args):
    started = time.monotonic()
    source, reference, budget, output = [Path(p).resolve() for p in
        (args.source_system, args.paired_control, args.budget_control, args.output)]
    for protected in (source.parent, reference, budget):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('098 output overlaps immutable input')
    if output.exists():
        raise FileExistsError('098 output already exists; no reuse/retry')
    torch.set_num_threads(args.cpu_threads)
    binding = check_source(args)
    config, manifest, alignment = [binding[key] for key in ('config', 'manifest', 'alignment')]
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('098 requested CUDA unavailable; no fallback')
    output.mkdir(parents=True, exist_ok=False)
    mode, seed = config['model']['mode'], config['training']['seed']
    state = dict(scope=SCOPE, status='RUNNING', phase='source_binding', configuration=config,
        source_system=str(source), paired_control=str(reference), budget_control=str(budget), mode=mode,
        experiment_id=EXPERIMENTS[(seed, mode)], train_device=args.device, inference_device='cpu', cpu_threads=4,
        optimizer_updates=0, completed_endpoints=[], validation_selection=False, association_recomputed=False,
        groups_recomputed=False, threshold_reselected=False, test_opened=False, retries=0,
        old066_acceptance='REJECTED', advisory_reached=False, time_policy='advisory', default_promoted=False,
        source_teacher_updates=config['training']['updates'], additional_updates=config['training']['updates'],
        total_stage_updates=2*config['training']['updates'])
    write_json(output/'run.json', state)
    write_json(output/'source-binding.json', dict(source_system=str(source), paired_control=str(reference),
        budget_control=str(budget), budget_pairing=binding['budget_pairing']))
    preparation = None

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed)
            write_json(output/'run.json', state)
            print('ADVISORY: 098 fixed attempt continues; no added updates or retries', flush=True)

    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print(state['experiment_id']+' phase: '+name, flush=True)
        check()

    try:
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'], data_config=config['data'],
            alignment_by_scene=alignment, check=check, motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('098 complete '+split+' population/order differs')
        preparation = dataset['report']
        write_json(output/'preparation.json', preparation)
        phase('fit4000')
        training = train_query_relation(dataset.pop('train'), config, output/'fit',
            paired_reference_fit=reference/'fit', device=args.device, check=check)
        state['optimizer_updates'] = training['optimizer_updates']
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        endpoints = {}
        plan = [('initial', 'step-0.pt', 0, mode == 'relation'),
                ('final', 'final.pt', config['training']['updates'], mode == 'relation')]
        if mode == 'relation':
            plan.append(('relation_off', 'final.pt', config['training']['updates'], False))
        for label, filename, step, enabled in plan:
            phase('evaluate_cpu_'+label)
            model, saved = load_checkpoint(output/'fit'/filename, device='cpu')
            if saved['configuration'] != config or saved['step'] != step or saved['actual_updates'] != step:
                raise ValueError('098 endpoint checkpoint/configuration/update mismatch')
            evaluated = evaluate_query_relation(model, dataset['validation'], source, binding['metadata'],
                output/label, source_teacher_checkpoint=reference/'fit/final.pt', check=check,
                initial=label == 'initial', relation_enabled=enabled)
            provenance = dict(path=str(output/'fit'/filename), step=step, actual_updates=step,
                configured_final_updates=config['training']['updates'], source_teacher_checkpoint=str(reference/'fit/final.pt'),
                source_teacher_updates=config['training']['updates'], routing_semantics=config['routing_semantics'],
                position_loss=config['position_loss'], mode=mode, relation_enabled=enabled)
            evaluated.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'summary.json', evaluated)
            record = read_json(output/label/'run.json')
            record.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'run.json', record)
            endpoints[label] = dict(directory=str(output/label), checkpoint=provenance)
            state['completed_endpoints'].append(label)
            write_json(output/'run.json', state)
            del model
        summary = dict(scope=SCOPE, status='COMPLETED', configuration=config,
            experiment_id=state['experiment_id'], optimizer_updates=training['optimizer_updates'],
            source_system=str(source), paired_control=str(reference), budget_control=str(budget), endpoints=endpoints,
            source_teacher_updates=config['training']['updates'], additional_updates=config['training']['updates'],
            total_stage_updates=2*config['training']['updates'], training_pairing=training['paired_control'],
            parameter_count=training['parameter_count'], trainable_parameter_count=training['trainable_parameter_count'],
            model_forwards=sum(read_json(output/label/'run.json')['model_forwards'] for label in endpoints),
            source_teacher_forwards=0, inference_device='cpu', test_opened=False, default_promoted=False,
            scientific_status='Fixed two-seed candidate/control; only the complete paired tables decide continuation.',
            limitations=['Reused development V300 and shared identities; not independent generalization.',
                'Position only; speed/heading remain Mean. Predicted groups remain fixed056.',
                'Generic set attention is a task baseline, not a published-author protocol reproduction.',
                'Compressed-feature consistency bias is not a physical distance or calibrated confidence.',
                'Fresh Adam and batch replay are not uninterrupted8000 optimization or proof of convergence.'])
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
            test_opened=False, operation='One4000-update fit; full CPU endpoints; relation mode adds same-weights bias-off diagnostic.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'alignment', 'config', 'source-system', 'paired-control', 'budget-control'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--archive', nargs='+')
    parser.add_argument('--output')
    parser.add_argument('--check-source-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    if not args.check_source_only and (not args.output or not args.archive):
        parser.error('--archive and --output are required for the fixed run')

    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f'cancelled by signal {signum}')

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        if args.check_source_only:
            torch.set_num_threads(4)
            check_source(args)
            print('READY_DEPENDENCIES: completed same-seed teacher and budget baseline verified', flush=True)
        else:
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
