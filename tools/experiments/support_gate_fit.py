"""068: two-scene preflight, one fixed position-gate fit, CPU endpoint evaluation.

No resume, retry, V selection, association inference, old fusion checkpoint, or
test payload access. GPU is used only for the newly authorized training run.
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
from association.evaluation.support_gate import evaluate_support_gate, compare_endpoints
from association.evaluation.support_gate_preflight import run_preflight
from association.training.support_gate import configuration, load_checkpoint, train_support_gate
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, alignment):
    """Bind actual settings and scene lists, not file names or content hashes."""
    configuration(config)
    if config != read_json(ROOT/'configs/support-gate-level1.json'):
        raise ValueError('068 requires the frozen 1000-update configuration')
    validate_manifest(manifest)
    if manifest != read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):
        raise ValueError('068 requires the unchanged full Level1 scene manifest')
    if alignment != read_json(ROOT/'configs/data/mtad-level1-alignment.json'):
        raise ValueError('068 requires the unchanged Level1 alignment')


def endpoint_model(path, expected_step, config, training_record):
    if (training_record.get('status') != 'COMPLETED'
            or training_record.get('optimizer_updates') != config['training']['updates']
            or training_record.get('configuration') != config):
        raise ValueError('endpoint evaluation requires completed fixed-budget training')
    model, checkpoint = load_checkpoint(path, device='cpu')
    if checkpoint['step'] != expected_step or checkpoint['configuration'] != config:
        raise ValueError('endpoint checkpoint step/configuration mismatch')
    return model


def run(args):
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    protected = source.parent
    if output == protected or output in protected.parents or protected in output.parents:
        raise ValueError('output must be separate from the immutable056 source run')
    if output.exists():
        raise FileExistsError('output already exists; no reuse or retry')
    if args.cpu_threads != 4:
        raise ValueError('068 fixes CPU threads to4')
    torch.set_num_threads(args.cpu_threads)
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('requested CUDA unavailable; no fallback')
    config, manifest, alignment = (read_json(path) for path in (args.config, args.manifest, args.alignment))
    production_contract(config, manifest, alignment)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope='068-support-prior-position-pilot', status='RUNNING', phase='source_binding',
        configuration=config, source_system=str(source), train_device=args.device,
        inference_device='cpu', cpu_threads=args.cpu_threads, optimizer_updates=0,
        test_opened=False, old066_acceptance='REJECTED', retries=0,
        association_recomputed=False, threshold_reselected=False, validation_groups_recomputed=False,
        validation_selection=False, advisory_reached=False, completed_endpoints=[])
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
        print('068 phase: '+name, flush=True)
        check()

    try:
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('preflight3035_2535')
        preflight = run_preflight(args.archive, manifest, alignment, source, output/'preflight', check=check)
        if preflight.get('training_gate_passed') is not True:
            raise ValueError('formal two-scene preflight did not authorize the training stage')
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
            data_config=config['data'], alignment_by_scene=alignment, check=check,
            motion_observation_policy='canonical_all')
        preparation = dataset['report']
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('prepared full '+split+' population/order differs')
        write_json(output/'preparation.json', preparation)
        phase('fit1000')
        training = train_support_gate(dataset.pop('train'), config, output/'fit', device=args.device, check=check)
        state['optimizer_updates'] = training['optimizer_updates']
        # V is not passed to the trainer. Both saved endpoints are evaluated
        # after training; neither can alter the fixed endpoint or training data.
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        endpoints = {}
        for label, filename, step in (('initial', 'step-0.pt', 0), ('final', 'final.pt', 1000)):
            phase('evaluate_cpu_'+label)
            model = endpoint_model(output/'fit'/filename, step, config, training)
            endpoints[label] = evaluate_support_gate(model, dataset['validation'], source, metadata,
                output/label, check=check)
            provenance = dict(path=str(output/'fit'/filename), step=step,
                actual_updates=step, configured_final_updates=config['training']['updates'],
                routing_semantics=config['routing_semantics'])
            endpoints[label]['checkpoint'] = provenance
            write_json(output/label/'summary.json', endpoints[label])
            evaluation_run = read_json(output/label/'run.json')
            evaluation_run['checkpoint'] = provenance
            write_json(output/label/'run.json', evaluation_run)
            state['completed_endpoints'].append(label)
            del model
        phase('paired_endpoint_comparison')
        paired = compare_endpoints(output/'initial', output/'final')
        write_json(output/'endpoint-comparison.json', paired)
        summary = dict(scope=state['scope'], status='COMPLETED', configuration=config,
            preflight=preflight, optimizer_updates=training['optimizer_updates'],
            endpoints={label: {'directory': str(output/label), 'summary': str(output/label/'summary.json')}
                       for label in endpoints}, endpoint_comparison=paired,
            source_system=str(source), inference_device='cpu', test_opened=False,
            scientific_status='single-seed repeatedly used development set; no automatic method promotion',
            limitations=['new restricted group position head, not an isolated ReLU/prior ablation',
                'raw-scene/cache reproduction does not establish physical truth of unusual input values',
                'original066 remains REJECTED; this new CPU protocol does not revalidate old GPU runs',
                'shared identities across splits and offline complete-fragment inputs'])
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        fit = output/'fit/run.json'
        if fit.exists():
            state['optimizer_updates'] = read_json(fit).get('optimizer_updates', 0)
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            preparation_report=str(output/'preparation.json') if preparation is not None else None,
            test_opened=False, model_forward='new gate only, CPU evaluations after one training run',
            old_model_checkpoints_loaded=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--alignment', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-system', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--output', required=True)
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
