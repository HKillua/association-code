"""099: one scratch DeepAF P/V fit, three full CPU endpoints, no launch/retry.

The runner validates fixed Level1 sources before opening archives or output.
Seed23 has two predeclared learning rates. Seed21 requires the cached, complete
seed23 selection evidence and uses only its selected rate. Within each run the
training budget and initial/mid/final endpoints are fixed; V never changes fit.
This entry point does not claim GPUs, start services, retry, or open T_test.
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
from association.evaluation.deepaf_peer import evaluate_deepaf_peer
from association.training.deepaf_peer import (
    LOSS_SEMANTICS, ROUTING_SEMANTICS, configuration, source_configuration,
    train_deepaf_peer, load_checkpoint,
)
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.analysis.compare_frozen_position_adapter import adapter_metadata, verify_pairing as verify_budget_pairing
from association.evaluation.position_sources import _metadata as source_metadata
from tools.experiments.sequence_position_fit import ROOT, bind_linear_control, production_contract as original_contract

SCOPE = '099-deepaf-pv-peer-feasibility'
CONFIGS = {
    (20261023, .001): 'deepaf-peer-level1-lr1e3.json',
    (20261023, .0003): 'deepaf-peer-level1-lr3e4.json',
    (20261021, .001): 'deepaf-peer-replica-lr1e3.json',
    (20261021, .0003): 'deepaf-peer-replica-lr3e4.json',
}
EXPERIMENTS = {(20261023, .001): '099A', (20261023, .0003): '099B',
               (20261021, .001): '099C', (20261021, .0003): '099C'}
PROFILES = {20261023: ('extended4000', '072C_transformer4000', '090'),
            20261021: ('replica4000', '073C_transformer4000_seed21', '097')}


def frozen_configuration(config):
    configuration(config)
    key = (config['training']['seed'], config['training']['learning_rate'])
    if key not in CONFIGS or config != read_json(ROOT/'configs'/CONFIGS[key]):
        raise ValueError('099 requires one predeclared seed/learning-rate configuration')
    return config


def production_contract(config, manifest, alignment):
    frozen_configuration(config)
    original_contract(source_configuration(config), manifest, alignment,
        profile=PROFILES[config['training']['seed']][0])


def _selection(args, config):
    path = getattr(args, 'selection_record', None)
    if config['training']['seed'] == 20261023:
        if path is not None:
            raise ValueError('099 seed23 arms cannot use a previous selection record')
        return None
    if path is None:
        raise ValueError('099C requires completed seed23 learning-rate selection evidence')
    selected = read_json(path)
    if (selected.get('scope') != '099-deepaf-peer-lr-selection'
            or selected.get('status') != 'COMPLETED'
            or selected.get('seed') != 20261023
            or selected.get('optimizer_updates') != 0 or selected.get('model_forwards') != 0
            or selected.get('test_opened') is not False
            or selected.get('peer_quality_certified') is not False):
        raise ValueError('099C requires a completed cached-only seed23 selection')
    runs = selected.get('source_runs', {})
    if set(runs) != {'peer_a', 'peer_b'}:
        raise ValueError('099C selection must bind both predeclared seed23 arms')
    # Local import avoids the comparison -> frozen_configuration import cycle.
    from tools.analysis.compare_deepaf_peer import select_learning_rate
    actual = select_learning_rate(runs['peer_a'], runs['peer_b'])
    for key in ('selection_rule', 'selected_learning_rate', 'selected', 'candidates',
                'source_system', 'scene_ids', 'source_runs'):
        if selected.get(key) != actual.get(key):
            raise ValueError('099C selection evidence differs from complete cached results: '+key)
    if actual['selected']['learning_rate'] != config['training']['learning_rate']:
        raise ValueError('099C configuration must use the predeclared seed23 winner')
    return actual


def check_source(args):
    """Read-only source/selection preflight; no archive, output, or CUDA forward."""
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('099 fixes CPU4 and advisory wall time')
    config, manifest, alignment = [read_json(path) for path in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    selection = _selection(args, config)
    source, reference, budget = [Path(path).resolve() for path in
        (args.source_system, args.paired_control, args.budget_control)]
    for path in (source, reference, budget):
        if read_json(path/'run.json').get('status') != 'COMPLETED':
            raise ValueError('099 requires a completed immutable source: '+str(path))
    bind_linear_control(reference, source_configuration(config), source, manifest)
    _bind_source(source)
    metadata = _load_metadata(source, manifest, 1000)
    metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
    _, arm, profile = PROFILES[config['training']['seed']]
    original = source_metadata(reference, arm)
    name, control = adapter_metadata(budget, profile=profile)
    if (name != 'finetune' or control['paired_control'] != reference
            or any(control[key] != original[key] for key in ('source_system', 'scene_ids', 'components'))):
        raise ValueError('099 budget baseline must be same-seed same-source full fine-tuning')
    pairing = verify_budget_pairing({'linear': original, 'finetune': control})
    if selection is not None and selection['source_system'] != str(source):
        raise ValueError('099C selection must use the same frozen056 source system')
    return dict(metadata=metadata, budget_pairing=pairing, source=str(source), reference=str(reference),
        budget=str(budget), config=config, manifest=manifest, alignment=alignment,
        learning_rate_selection=selection)


def run(args):
    started = time.monotonic()
    source, reference, budget, output = [Path(path).resolve() for path in
        (args.source_system, args.paired_control, args.budget_control, args.output)]
    for protected in (source.parent, reference, budget):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('099 output overlaps an immutable input')
    if output.exists():
        raise FileExistsError('099 output already exists; no reuse/retry')
    torch.set_num_threads(args.cpu_threads)
    binding = check_source(args)
    config, manifest, alignment = [binding[key] for key in ('config', 'manifest', 'alignment')]
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('099 requested CUDA unavailable; no CPU fallback')
    output.mkdir(parents=True, exist_ok=False)
    training = config['training']
    selection_path = getattr(args, 'selection_record', None)
    state = dict(scope=SCOPE, status='RUNNING', phase='source_binding', configuration=config,
        experiment_id=EXPERIMENTS[(training['seed'], training['learning_rate'])],
        source_system=str(source), paired_control=str(reference), budget_control=str(budget),
        train_device=args.device, inference_device='cpu', cpu_threads=4,
        source_sampling_updates=training['source_updates'], total_training_updates=training['updates'],
        optimizer_updates=0, completed_endpoints=[],
        validation_selection=binding['learning_rate_selection'] is not None,
        validation_checkpoint_selection=False,
        validation_learning_rate_selection=binding['learning_rate_selection'] is not None,
        selection_record=str(Path(selection_path).resolve()) if selection_path else None,
        learning_rate_selection=binding['learning_rate_selection'],
        source_weights_inherited=False, association_recomputed=False, groups_recomputed=False,
        threshold_reselected=False, test_opened=False, retries=0, old066_acceptance='REJECTED',
        advisory_reached=False, time_policy='advisory', default_promoted=False)
    write_json(output/'run.json', state)
    write_json(output/'source-binding.json', dict(source_system=str(source), paired_control=str(reference),
        budget_control=str(budget), budget_pairing=binding['budget_pairing'],
        learning_rate_selection=binding['learning_rate_selection']))
    preparation = None

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= training['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed)
            write_json(output/'run.json', state)
            print('ADVISORY: 099 fixed attempt continues; no extra updates or retries', flush=True)

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
                raise ValueError('099 complete '+split+' population/order differs')
        preparation = dataset['report']
        write_json(output/'preparation.json', preparation)
        phase('scratch_fit_two_source_cycles')
        fit = train_deepaf_peer(dataset.pop('train'), config, output/'fit',
            paired_reference_fit=reference/'fit', device=args.device, check=check)
        state['optimizer_updates'] = fit['optimizer_updates']
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        endpoints = {}
        plan = [('initial', 'step-0.pt', 0),
                ('mid', f'step-{training["source_updates"]}.pt', training['source_updates']),
                ('final', 'final.pt', training['updates'])]
        for label, filename, step in plan:
            phase('evaluate_cpu_'+label)
            model, saved = load_checkpoint(output/'fit'/filename, device='cpu')
            if (saved['configuration'] != config or saved['step'] != step
                    or saved['actual_updates'] != step or saved['checkpoint_kind'] != 'endpoint'):
                raise ValueError('099 actual checkpoint configuration/update/kind differs')
            evaluated = evaluate_deepaf_peer(model, dataset['validation'], source, binding['metadata'],
                output/label, check=check, initial=label == 'initial')
            provenance = dict(path=str(output/'fit'/filename), step=step, actual_updates=step,
                configured_final_updates=training['updates'], source_sampling_fit=str(reference/'fit'),
                source_sampling_updates=training['source_updates'], source_weights_inherited=False,
                routing_semantics=ROUTING_SEMANTICS, loss_semantics=LOSS_SEMANTICS)
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
            experiment_id=state['experiment_id'], optimizer_updates=fit['optimizer_updates'],
            source_system=str(source), paired_control=str(reference), budget_control=str(budget),
            endpoints=endpoints, completed_endpoints=list(state['completed_endpoints']),
            source_sampling_updates=training['source_updates'], total_training_updates=training['updates'],
            source_weights_inherited=False, training_pairing=fit['replay_verification'],
            training_diagnostics=fit['training_diagnostics'],
            parameter_count=fit['parameter_count'], trainable_parameter_count=fit['trainable_parameter_count'],
            model_forwards=sum(read_json(output/label/'run.json')['model_forwards'] for label in endpoints),
            inference_device='cpu',
            validation_selection=state['validation_selection'],
            validation_checkpoint_selection=False,
            validation_learning_rate_selection=state['validation_learning_rate_selection'],
            test_opened=False, default_promoted=False,
            selection_record=state['selection_record'], learning_rate_selection=state['learning_rate_selection'],
            scientific_status='Complete fixed-budget published-method adaptation; peer quality needs scientific review.',
            limitations=['Reused development V300 and shared identities; no independent generalization claim.',
                'Fixed056 groups; displayed speed/course remain Mean while V is internal to the DeepAF recurrence.',
                'Two-cycle sample exposure matches source-plus-budget controls; optimizer trajectories and compute differ.',
                'Position/velocity MSE, detached closed-loop history and TBPTT1 are disclosed task-adaptation choices.',
                'Two predeclared learning rates are selected only on seed23 final8000 full-population position MAE.'])
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
            test_opened=False, operation='Scratch two-cycle P/V fit; fixed T probes; full CPU initial/mid/final endpoints.'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'alignment', 'config', 'source-system', 'paired-control', 'budget-control'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--selection-record')
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
            print('READY_DEPENDENCIES: fixed source, budget and applicable learning-rate selection verified', flush=True)
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
