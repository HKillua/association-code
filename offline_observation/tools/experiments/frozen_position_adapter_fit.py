"""090 one additional-budget source-final adapter/full-finetune attempt."""
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

CONFIGS = {mode: f'frozen-position-adapter-{mode.replace("_", "-")}-level1.json'
           for mode in ('all_queries', 'anchor_unsupported', 'full_finetune')}


def frozen_configuration(config):
    configuration(config)
    if config != read_json(ROOT/'configs'/CONFIGS[config['model']['mode']]):
        raise ValueError('090 requires exact declared mode-specific production configuration')
    return config


def production_contract(config, manifest, alignment):
    frozen_configuration(config)
    original_contract(source_configuration(config), manifest, alignment, profile='extended4000')


def run(args):
    source, reference, output = [Path(p).resolve() for p in (args.source_system, args.paired_control, args.output)]
    for protected in (source.parent, reference):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('090 output overlaps immutable source/control')
    if output.exists(): raise FileExistsError('090 output exists; no reuse/retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory': raise ValueError('090 fixes CPU4/advisory')
    torch.set_num_threads(4)
    if args.device == 'cuda:0' and not torch.cuda.is_available(): raise ValueError('090 requested CUDA unavailable; no fallback')
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment); output.mkdir(parents=True)
    started = time.monotonic(); mode = config['model']['mode']
    state = dict(scope='090-source-final-position', status='RUNNING', phase='source_binding',
        configuration=config, source_system=str(source), paired_control=str(reference), mode=mode,
        train_device=args.device, inference_device='cpu', cpu_threads=4, optimizer_updates=0,
        completed_endpoints=[], validation_selection=False, association_recomputed=False,
        groups_recomputed=False, threshold_reselected=False, test_opened=False, retries=0,
        old066_acceptance='REJECTED', advisory_reached=False, time_policy='advisory')
    write_json(output/'run.json', state); preparation = None

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed); write_json(output/'run.json', state)
            print('ADVISORY: 090 declared wall time exceeded; continuing same attempt', flush=True)

    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started); write_json(output/'run.json', state)
        print('090 phase: '+name, flush=True); check()

    try:
        _bind_source(source); metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        bind_linear_control(reference, source_configuration(config), source, manifest)
        # Check the complete source checkpoint/log/cache inventory before allocating a new fit.
        from tools.analysis.compare_frozen_position_adapter import _metadata as source_metadata
        source_metadata(reference, '072C_transformer4000')
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'], data_config=config['data'],
            alignment_by_scene=alignment, check=check, motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('090 complete '+split+' population/order differs')
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
                raise ValueError('090 actual endpoint configuration/step mismatch')
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
            scientific_status='Additional-budget adapter or full-model fine-tuning; source Adam not resumed.',
            limitations=['Reused V300 and correlated identities; single seed is not independent generalization.',
                'Existing 4000-update teacher plus 4000 added updates; not equal total budget to original072C.',
                'Support is visible observation span, not accuracy or noise-free evidence.',
                'All helpers remain available; speed/course remain ordinary Mean.',
                'Synthetic T groups differ from frozen056 deployment groups.'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='cached_source_final_comparison', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        from tools.analysis.compare_frozen_position_adapter import run as compare_cached, NAMES
        comparison = compare_cached(argparse.Namespace(linear_run=str(reference),
            all_run=str(output) if mode == 'all_queries' else None,
            missing_run=str(output) if mode == 'anchor_unsupported' else None,
            finetune_run=str(output) if mode == 'full_finetune' else None,
            output=str(output/'reference-comparison'), cpu_threads=4))
        summary.update(reference_comparison='reference-comparison/summary.json.gz',
            scientific_screen=comparison['scientific_screen'][NAMES[mode]])
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
    for key in ('manifest', 'alignment', 'config', 'source-system', 'paired-control', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN); raise KeyboardInterrupt(f'cancelled by signal {signum}')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        result = run(args); print(f'{result["status"]}: {Path(args.output).resolve()}', flush=True); return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__': raise SystemExit(main())
