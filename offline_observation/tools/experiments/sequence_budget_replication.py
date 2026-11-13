"""097A: one second-seed added-budget sensitivity experiment from final073C.

Same original samples, fresh Adam, 4000 added updates, position only. This is
not adapter replication, optimizer continuation or a convergence demonstration.
Source-only exit75 waits without opening active weights, GPU or output.
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
from association.evaluation.frozen_position_adapter import evaluate_frozen_position_adapter
from association.training.frozen_position_adapter import (configuration, source_configuration,
    train_frozen_position_adapter, load_checkpoint)
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.sequence_position_fit import ROOT, bind_linear_control, production_contract as original_contract

CONFIG_FILE = 'sequence-budget-replication-level1.json'
SCOPE = '097-sequence-added-budget-seed-sensitivity'
SOURCE_ARM = '073C_transformer4000_seed21'
WAITING = 75


class WaitingDependencies(Exception):
    pass


def frozen_configuration(config):
    configuration(config)
    if config != read_json(ROOT/'configs'/CONFIG_FILE):
        raise ValueError('097A requires the declared full-finetune seed20261021 configuration')
    return config


def production_contract(config, manifest, alignment):
    frozen_configuration(config)
    original_contract(source_configuration(config), manifest, alignment, profile='replica4000')


def _terminal(path, label):
    path = Path(path)
    if not path.is_file():
        raise WaitingDependencies(label+' has no run.json yet')
    state = read_json(path)
    if state.get('status') in ('FAILED', 'INTERRUPTED'):
        raise ValueError(label+' failed; no automatic retry or source substitution')
    if state.get('status') != 'COMPLETED':
        raise WaitingDependencies(label+' is not completed')
    return state


def check_source(args):
    """Inspect terminal records before reading any checkpoint; no output or CUDA."""
    reference, source = Path(args.paired_control).resolve(), Path(args.source_system).resolve()
    control = _terminal(reference/'run.json', '073C')
    if control.get('profile') != 'replica4000' or control.get('scope') != '073-generic-sequence-position-replica4000':
        raise ValueError('097 requires the original073C replica4000 source identity')
    for name in ('fit', 'initial', 'final'):
        _terminal(reference/name/'run.json', '073C/'+name)
    _terminal(source/'run.json', '056/system')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('097 fixes CPU4/advisory')
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    bind_linear_control(reference, source_configuration(config), source, manifest)
    _bind_source(source)
    metadata = _load_metadata(source, manifest, 1000)
    metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
    from tools.analysis.compare_frozen_position_adapter import _metadata as source_metadata
    source_metadata(reference, SOURCE_ARM)
    return dict(source=str(source), reference=str(reference), metadata=metadata,
        source_identity='073C', seed=config['training']['seed'], source_updates=config['training']['updates'])


def run(args):
    started = time.monotonic()
    source, reference, output = [Path(p).resolve() for p in (args.source_system, args.paired_control, args.output)]
    for protected in (source.parent, reference):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('097 output overlaps immutable source/control')
    if output.exists(): raise FileExistsError('097 output exists; no reuse/retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory': raise ValueError('097 fixes CPU4/advisory')
    torch.set_num_threads(4)
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    binding = check_source(args)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('097 requested CUDA unavailable; no fallback')
    output.mkdir(parents=True, exist_ok=False)
    mode = config['model']['mode']
    state = dict(scope=SCOPE, status='RUNNING', phase='source_binding',
        configuration=config, source_system=str(source), paired_control=str(reference), mode=mode,
        train_device=args.device, inference_device='cpu', cpu_threads=4, optimizer_updates=0,
        completed_endpoints=[], validation_selection=False, association_recomputed=False,
        groups_recomputed=False, threshold_reselected=False, test_opened=False, retries=0,
        old066_acceptance='REJECTED', advisory_reached=False, time_policy='advisory',
        experiment_id='097A', source_identity='073C', reference_seed=20261021,
        source_teacher_updates=config['training']['updates'], additional_updates=config['training']['updates'],
        total_stage_updates=2*config['training']['updates'],
        optimizer_initialization='fresh Adam; original sampler replay; source optimizer never resumed',
        default_promoted=False)
    write_json(output/'run.json', state); preparation = None

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed); write_json(output/'run.json', state)
            print('ADVISORY: 097 declared wall time exceeded; continuing same attempt', flush=True)

    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started); write_json(output/'run.json', state)
        print('097 phase: '+name, flush=True); check()

    try:
        metadata = binding['metadata']
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'], data_config=config['data'],
            alignment_by_scene=alignment, check=check, motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('097 complete '+split+' population/order differs')
        preparation = dataset['report']; write_json(output/'preparation.json', preparation)
        phase('additional_fit4000')
        training = train_frozen_position_adapter(dataset.pop('train'), config, output/'fit', device=args.device,
            check=check, paired_reference_fit=reference/'fit')
        state['optimizer_updates'] = training['optimizer_updates']; write_json(output/'run.json', state)
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
        endpoints = {}
        for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', config['training']['updates'])):
            phase('evaluate_cpu_'+label)
            model, saved = load_checkpoint(output/'fit'/filename, device='cpu')
            if saved['configuration'] != config or saved['step'] != step:
                raise ValueError('097 actual endpoint configuration/step mismatch')
            evaluated = evaluate_frozen_position_adapter(model, dataset['validation'], source, metadata, output/label,
                source_teacher_checkpoint=reference/'fit/final.pt', check=check, initial=label == 'initial')
            provenance = dict(path=str(output/'fit'/filename), step=step, actual_updates=step,
                configured_final_updates=config['training']['updates'], source_teacher_checkpoint=str(reference/'fit/final.pt'),
                source_teacher_updates=config['training']['updates'], routing_semantics=config['routing_semantics'],
                position_loss='linear', mode=mode)
            evaluated.update(checkpoint=provenance, configuration=config); write_json(output/label/'summary.json', evaluated)
            record = read_json(output/label/'run.json'); record.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'run.json', record)
            endpoints[label] = dict(directory=str(output/label), checkpoint=provenance)
            state['completed_endpoints'].append(label); del model
        summary = dict(scope=state['scope'], status='COMPLETED', configuration=config,
            optimizer_updates=training['optimizer_updates'], endpoints=endpoints, source_system=str(source),
            paired_control=str(reference), training_pairing=training['paired_control'], inference_device='cpu', test_opened=False,
            source_teacher_updates=config['training']['updates'], additional_updates=config['training']['updates'],
            total_stage_updates=2*config['training']['updates'], parameter_count=training['parameter_count'],
            trainable_parameter_count=training['trainable_parameter_count'],
            experiment_id='097A', reference_seed=20261021, source_identity='073C',
            scientific_status='Second-seed added-budget characterization; source Adam not resumed; no candidate promotion.',
            limitations=['Reused V300 and correlated identities; single seed is not independent generalization.',
                'Source073C4000 plus added4000 with fresh Adam and original sample replay, not optimizer continuation.',
                'Support is visible observation span, not accuracy or noise-free evidence.',
                'All helpers remain available; speed/course remain ordinary Mean.',
                'Synthetic T groups differ from frozen056 deployment groups.'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='cached_source_final_comparison', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        from tools.analysis.compare_frozen_position_adapter import run as compare_cached
        comparison = compare_cached(argparse.Namespace(linear_run=str(reference),
            all_run=str(output) if mode == 'all_queries' else None,
            missing_run=str(output) if mode == 'anchor_unsupported' else None,
            finetune_run=str(output) if mode == 'full_finetune' else None,
            output=str(output/'reference-comparison'), cpu_threads=4, profile='097'))
        summary.update(reference_comparison='reference-comparison/summary.json.gz',
            budget_characterization=comparison['budget_characterization'], default_promoted=False)
        write_json(output/'summary.json', summary)
        state.update(phase='completed', reference_comparison=summary['reference_comparison'], elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state); return summary
    except BaseException as error:
        if (output/'fit/run.json').exists(): state['optimizer_updates'] = read_json(output/'fit/run.json').get('optimizer_updates', 0)
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state); raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            preparation_report=str(output/'preparation.json') if preparation is not None else None,
            test_opened=False, operation='one additional-budget fit; source-final initial/final CPU endpoints; cached comparison'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'alignment', 'config', 'source-system', 'paired-control'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--output')
    parser.add_argument('--archive', nargs='+')
    parser.add_argument('--check-source-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN); raise KeyboardInterrupt(f'cancelled by signal {signum}')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        if args.check_source_only:
            check_source(args)
            print('READY_DEPENDENCIES: completed073C seed20261021 source verified', flush=True)
            return 0
        if not args.output or not args.archive:
            raise ValueError('training requires --output and --archive')
        result = run(args); print(f'{result["status"]}: {Path(args.output).resolve()}', flush=True); return 0
    except WaitingDependencies as error:
        print('WAITING_DEPENDENCIES: '+str(error), flush=True); return WAITING
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__': raise SystemExit(main())
