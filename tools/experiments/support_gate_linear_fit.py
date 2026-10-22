"""070: paired linear-position loss fit, reusing068's verified CPU initialization.

One from-scratch1000-update attempt; only the outer loss transform changes.
No resume, original two-scene preflight, initial V forward, or V-based selection.
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
from association.evaluation.support_gate import evaluate_support_gate
from association.evaluation.support_gate_linear import bind_linear_source, compare_linear_run
from association.training.support_gate import configuration, loss_semantics, train_support_gate
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.support_gate_fit import endpoint_model, production_contract as original_contract

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, alignment):
    configuration(config)
    if config != read_json(ROOT/'configs/support-gate-linear-level1.json'):
        raise ValueError('070 requires the frozen linear-loss1000-update configuration')
    original = dict(config)
    if original.pop('position_loss', None) != 'linear':
        raise ValueError('070 changes only position_loss to linear')
    original_contract(original, manifest, alignment)


def run(args):
    source = Path(args.source_system).resolve()
    baseline = Path(args.source_068).resolve()
    output = Path(args.output).resolve()
    for protected in (source.parent, baseline):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('070 output must be separate from immutable056 and068 runs')
    if output.exists():
        raise FileExistsError('output already exists; no reuse or retry')
    if args.cpu_threads != 4:
        raise ValueError('070 fixes CPU threads to4')
    if getattr(args, 'time_policy', 'advisory') != 'advisory':
        raise ValueError('070 wall time is advisory only; no undeclared hard cutoff')
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; no fallback')
    config, manifest, alignment = (read_json(path) for path in (args.config, args.manifest, args.alignment))
    production_contract(config, manifest, alignment)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope='070-paired-linear-position-pilot', status='RUNNING', phase='source_binding',
        configuration=config, loss_semantics=loss_semantics(config), source_system=str(source),
        source_068=str(baseline), train_device=args.device, inference_device='cpu', cpu_threads=4,
        optimizer_updates=0, test_opened=False, old066_acceptance='REJECTED', retries=0,
        association_recomputed=False, threshold_reselected=False, validation_groups_recomputed=False,
        validation_selection=False, advisory_reached=False, completed_endpoints=[],
        initial_evaluation_reused=False, raw_preflight_repeated=False, time_policy='advisory')
    write_json(output/'run.json', state)
    preparation = None

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed)
            write_json(output/'run.json', state)
            print('ADVISORY: declared wall time exceeded; continuing this single attempt', flush=True)

    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print('070 phase: '+name, flush=True)
        check()

    try:
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('bind068_cached_controls')
        binding = bind_linear_source(baseline, source, config, manifest, check=check)
        write_json(output/'baseline-binding.json', binding)
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
            data_config=config['data'], alignment_by_scene=alignment, check=check,
            motion_observation_policy='canonical_all')
        preparation = dataset['report']
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('prepared full '+split+' population/order differs')
        write_json(output/'preparation.json', preparation)
        phase('fit1000_paired_linear')
        training = train_support_gate(dataset.pop('train'), config, output/'fit', device=args.device,
            check=check, paired_reference_fit=baseline/'fit')
        state['optimizer_updates'] = training['optimizer_updates']
        state['initial_evaluation_reused'] = True
        write_json(output/'run.json', state)
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        phase('evaluate_cpu_final')
        model = endpoint_model(output/'fit/final.pt', 1000, config, training)
        evaluated = evaluate_support_gate(model, dataset['validation'], source, metadata,
            output/'final', check=check)
        provenance = dict(path=str(output/'fit/final.pt'), step=1000, actual_updates=1000,
            configured_final_updates=1000, routing_semantics=config['routing_semantics'],
            loss_semantics=loss_semantics(config))
        evaluated.update(scope='070-linear-position-gate-evaluation', checkpoint=provenance,
            configuration=config, position_loss='linear')
        write_json(output/'final/summary.json', evaluated)
        evaluated_run = read_json(output/'final/run.json')
        evaluated_run.update(scope=evaluated['scope'], checkpoint=provenance,
            configuration=config, position_loss='linear')
        write_json(output/'final/run.json', evaluated_run)
        state['completed_endpoints'].append('final')
        del model
        phase('paired_cached_comparisons')
        comparisons = compare_linear_run(baseline, output/'final', binding,
            output=output/'comparison', check=check)
        summary = dict(scope=state['scope'], status='COMPLETED', configuration=config,
            loss_semantics=loss_semantics(config), optimizer_updates=training['optimizer_updates'],
            source_system=str(source), source_068=str(baseline), inference_device='cpu', test_opened=False,
            initial_evaluation=dict(directory=str(baseline/'initial'), reused=True, model_forwards=0,
                                    initialization_verified_by='fit/run.json'),
            final_evaluation=dict(directory=str(output/'final'), checkpoint=provenance),
            paired_comparison_directory=str(output/'comparison'), comparison_status=comparisons['status'],
            scientific_status='single-seed loss intervention on reused V; no automatic method promotion',
            limitations=['loss changes sample-amplitude weighting and fixed-optimizer dynamics together',
                'T synthetic groups and per-anchor normalization still differ from deployment groups/population',
                'unusual reference/input values retained; linear loss may amplify their influence',
                'shared identities, reused development set, and offline complete-fragment inputs',
                'original066 remains REJECTED; no retrospective change of its acceptance'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        if (output/'fit/run.json').exists():
            state['optimizer_updates'] = read_json(output/'fit/run.json').get('optimizer_updates', 0)
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            preparation_report=str(output/'preparation.json') if preparation is not None else None,
            test_opened=False, raw_preflight_repeated=False, initial_evaluation_forward=False,
            model_forward='one new training run and one final CPU V evaluation only',
            source_checkpoint='068 step-0 for exact initialization; final for source update validation; never resumed'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'alignment', 'config', 'source-system', 'source-068', 'output'):
        parser.add_argument('--'+flag, required=True)
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
