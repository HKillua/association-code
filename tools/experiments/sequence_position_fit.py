"""Fixed-budget generic LSTM/Transformer position recovery runs.

One fixed from-scratch run, then independent CPU step0/final full V evaluation.
The default 071 pilot uses 1000 updates; extended4000 uses a fresh 4000.
No V-based training selection, association inference, regrouping, test access,
old fusion checkpoint, source modification or automatic retry.
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

from association.data.archive import validate_manifest
from association.data.dataset import load_timed_dataset
from association.evaluation.sequence_position import evaluate_sequence_position, compare_endpoints
from association.training.sequence_position import configuration, load_checkpoint, train_sequence_position
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, alignment, profile='pilot1000'):
    configuration(config)
    backbone = config['model']['backbone']
    suffixes = {'pilot1000': 'level1', 'extended4000': 'extended-level1'}
    if profile not in suffixes:
        raise ValueError('unknown sequence training profile')
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
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    scope = ('071-generic-sequence-position-pilot' if profile == 'pilot1000'
             else '072-generic-sequence-position-extended4000')
    state = dict(scope=scope, profile=profile, status='RUNNING', phase='source_binding',
        configuration=config, backbone=config['model']['backbone'], source_system=str(source),
        train_device=args.device, inference_device='cpu', cpu_threads=4, optimizer_updates=0,
        test_opened=False, retries=0, old066_acceptance='REJECTED', completed_endpoints=[],
        validation_selection=False, association_recomputed=False, groups_recomputed=False,
        threshold_reselected=False, advisory_reached=False, time_policy='advisory')
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
        training = train_sequence_position(dataset.pop('train'), config, output/'fit', device=args.device, check=check)
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
    parser.add_argument('--profile', choices=('pilot1000', 'extended4000'), default='pilot1000',
                        help='explicit fresh training budget; extended4000 is not a resumed pilot')
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
