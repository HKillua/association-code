"""Fixed-budget generic LSTM/Transformer position recovery runs.

One fixed from-scratch run, then independent CPU step0/final full V evaluation.
The default071 pilot uses1000 updates; extended4000 and replica4000 use fresh4000.
No V-based training selection, association inference, regrouping, test access,
old fusion checkpoint, source modification or automatic retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from association.data.archive import validate_manifest
from association.data.dataset import load_timed_dataset
from association.evaluation.sequence_position import evaluate_sequence_position, compare_endpoints
from association.models.sequence_position import SCOPE, RELATIVE_HARM_LOSS
from association.training.sequence_position import configuration, load_checkpoint, train_sequence_position
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, alignment, profile='pilot1000'):
    configuration(config)
    backbone = config['model']['backbone']
    suffixes = {'pilot1000': 'level1', 'extended4000': 'extended-level1',
                'replica4000': 'extended-replica-level1', 'harm4000': 'relative-harm-level1',
                'harm-replica4000': 'relative-harm-replica-level1',
                'helper-dropout4000': 'helper-dropout-level1'}
    if profile not in suffixes:
        raise ValueError('unknown sequence training profile')
    if profile in ('replica4000', 'harm4000', 'harm-replica4000', 'helper-dropout4000') and backbone != 'transformer':
        raise ValueError('replica and relative-harm profiles declare the Transformer backbone only')
    if config != read_json(ROOT/f'configs/sequence-position-{backbone}-{suffixes[profile]}.json'):
        updates = 1000 if profile == 'pilot1000' else 4000
        raise ValueError(f'{profile} requires the frozen backbone-specific {updates}-update configuration')
    validate_manifest(manifest)
    if manifest != read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):
        raise ValueError('sequence training requires unchanged full Level1 manifest')
    if alignment != read_json(ROOT/'configs/data/mtad-level1-alignment.json'):
        raise ValueError('sequence training requires unchanged Level1 alignment')


def endpoint_model(path, expected_step, config, training):
    if (training.get('status') != 'COMPLETED' or training.get('configuration') != config
            or training.get('optimizer_updates') != config['training']['updates']
            or expected_step not in (0, config['training']['updates'])):
        raise ValueError('sequence endpoints require completed fixed-budget training')
    model, checkpoint = load_checkpoint(path, device='cpu')
    if (checkpoint.get('configuration') != config or checkpoint.get('step') != expected_step
            or checkpoint.get('actual_updates') != expected_step):
        raise ValueError('sequence checkpoint step/configuration/actual_updates mismatch')
    return model


def bind_linear_control(path, config, source, manifest):
    """Bind074 to its original same-seed full CPU control before new training."""
    path = Path(path).resolve()
    source = Path(source).resolve()
    expected = deepcopy(config)
    expected.update(scope=SCOPE, position_loss='linear')
    expected.pop('augmentation', None)
    steps = expected['training']['updates']
    run, summary, fit = [read_json(path/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    for record in (run, summary, fit):
        if (record.get('status') != 'COMPLETED' or record.get('configuration') != expected
                or record.get('optimizer_updates') != steps or record.get('test_opened') is not False):
            raise ValueError('relative-harm control must be the completed same-seed linear run')
    if (run.get('source_system') != str(source) or summary.get('source_system') != str(source)
            or run.get('completed_endpoints') != ['initial', 'final']
            or run.get('validation_selection') is not False
            or fit.get('training_scene_ids') != manifest['splits']['T_core']):
        raise ValueError('relative-harm control source056, training population or endpoints differ')
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', steps, 'final.pt')):
        for record in (read_json(path/label/'run.json'), read_json(path/label/'summary.json')):
            checkpoint = record.get('checkpoint', {})
            if (record.get('status') != 'COMPLETED' or record.get('configuration') != expected
                    or record.get('source_system') != str(source) or record.get('device') != 'cpu'
                    or record.get('optimizer_updates') != 0 or record.get('test_opened') is not False
                    or record.get('groups_recomputed') is not False or record.get('threshold_reselected') is not False
                    or checkpoint.get('step') != step or checkpoint.get('actual_updates') != step
                    or checkpoint.get('configured_final_updates') != steps
                    or Path(checkpoint.get('path', '')).resolve() != path/'fit'/filename):
                raise ValueError('relative-harm control CPU output does not bind the frozen endpoint')
        if read_json(path/label/'summary.json')['scene_ids'] != manifest['splits']['V_select']:
            raise ValueError('relative-harm control full validation population differs')
    return path


def run(args):
    profile = getattr(args, 'profile', 'pilot1000')
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    protected = source.parent
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('sequence output must be separate from immutable056 source')
    if output.exists():
        raise FileExistsError('output already exists; no reuse/retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('sequence training fixes CPU4 and advisory wall time')
    torch.set_num_threads(4)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; no fallback')
    config, manifest, alignment = [read_json(path) for path in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment, profile=profile)
    paired_control = getattr(args, 'paired_control', None)
    harm_objective = config['position_loss'] == RELATIVE_HARM_LOSS
    dropout = profile == 'helper-dropout4000'
    if (harm_objective or dropout) != (paired_control is not None):
        raise ValueError('relative-harm and helper-dropout profiles require --paired-control')
    if paired_control is not None:
        paired_control = Path(paired_control).resolve()
        if output == paired_control or output in paired_control.parents or paired_control in output.parents:
            raise ValueError('relative-harm output must be separate from the immutable linear control')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    scope = {'pilot1000': '071-generic-sequence-position-pilot',
             'extended4000': '072-generic-sequence-position-extended4000',
             'replica4000': '073-generic-sequence-position-replica4000',
             'harm4000': '074-transformer-relative-harm4000',
             'harm-replica4000': '074-transformer-relative-harm-replica4000',
             'helper-dropout4000': '083-transformer-helper-fragment-dropout4000'}[profile]
    state = dict(scope=scope, profile=profile, status='RUNNING', phase='source_binding',
        configuration=config, backbone=config['model']['backbone'], source_system=str(source),
        train_device=args.device, inference_device='cpu', cpu_threads=4, optimizer_updates=0,
        test_opened=False, retries=0, old066_acceptance='REJECTED', completed_endpoints=[],
        validation_selection=False, association_recomputed=False, groups_recomputed=False,
        threshold_reselected=False, advisory_reached=False, time_policy='advisory')
    if paired_control is not None:
        state['paired_control'] = str(paired_control)
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
        print(f'{profile} phase: '+name, flush=True)
        check()

    try:
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        if paired_control is not None:
            bind_linear_control(paired_control, config, source, manifest)
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
            data_config=config['data'], alignment_by_scene=alignment, check=check,
            motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('sequence prepared full '+split+' population/order differs')
        preparation = dataset['report']
        write_json(output/'preparation.json', preparation)
        phase(f'fit{config["training"]["updates"]}')
        training = train_sequence_position(dataset.pop('train'), config, output/'fit', device=args.device, check=check,
            paired_reference_fit=paired_control/'fit' if paired_control is not None else None)
        state['optimizer_updates'] = training['optimizer_updates']
        write_json(output/'run.json', state)
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        # No validation is passed to training. Both endpoints are evaluated only
        # after the frozen budget, so neither changes samples, LR or endpoint.
        endpoints = {}
        for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', config['training']['updates'])):
            phase('evaluate_cpu_'+label)
            model = endpoint_model(output/'fit'/filename, step, config, training)
            evaluated = evaluate_sequence_position(model, dataset['validation'], source, metadata,
                output/label, check=check, initial=(label == 'initial'))
            provenance = dict(path=str(output/'fit'/filename), step=step, actual_updates=step,
                configured_final_updates=config['training']['updates'], routing_semantics=config['routing_semantics'],
                position_loss=config['position_loss'], backbone=config['model']['backbone'])
            evaluated.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'summary.json', evaluated)
            record = read_json(output/label/'run.json')
            record.update(checkpoint=provenance, configuration=config)
            write_json(output/label/'run.json', record)
            endpoints[label] = dict(directory=str(output/label), checkpoint=provenance)
            state['completed_endpoints'].append(label)
            del model
        phase('paired_endpoint_comparison')
        comparison = compare_endpoints(output/'initial', output/'final')
        write_json(output/'endpoint-comparison.json', comparison)
        summary = dict(scope=state['scope'], profile=profile, status='COMPLETED', configuration=config,
            optimizer_updates=training['optimizer_updates'], endpoints=endpoints,
            endpoint_comparison='endpoint-comparison.json', source_system=str(source), inference_device='cpu',
            test_opened=False, scientific_status='bounded generic position-only pilot; no automatic method promotion',
            limitations=[f'{config["training"]["updates"]} updates and one seed do not prove competitive baseline convergence.',
                'Shared physical warm start; not original LSTM/PTFM/TMTAFF reproduction.',
                'Full-Q two-dimensional residual is unrestricted by the gate opportunity mask.',
                'Offline canonical inputs, reused V300, shared identities and synthetic training groups.',
                'Speed and heading remain Mean controls, not learned outputs.'])
        if paired_control is not None:
            summary.update(paired_control=str(paired_control), training_pairing=training['paired_control'])
            if harm_objective:
                summary['limitations'].append('Relative harm is against fixed .95 error, not an absolute-tail guarantee; optimizer/clip scaling changes are part of the intervention.')
            if dropout:
                summary.update(augmentation=config['augmentation'], augmentation_exposure=training['augmentation_exposure'],
                    augmentation_applied_only_to_training=True)
                summary['limitations'].append('Whole-helper removal changes information, training physical base and wrong-helper contamination jointly; it is not an isolated causal test of missing-input robustness. Validation uses full frozen groups.')
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
            test_opened=False, old_fusion_checkpoints_loaded=False,
            model_forward='one new training run and CPU step0/final evaluation'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest', 'alignment', 'config', 'source-system', 'output'):
        parser.add_argument('--'+flag, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    parser.add_argument('--profile', choices=('pilot1000', 'extended4000', 'replica4000', 'harm4000', 'harm-replica4000', 'helper-dropout4000'), default='pilot1000',
                        help='explicit fresh budget/seed; extended profiles are not resumed pilots')
    parser.add_argument('--paired-control', help='074/083: completed same-seed072C/073C run root; never resumed')
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
