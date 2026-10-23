"""072A: T-only finite scalar fit, then one frozen-population CPU V evaluation."""
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
from association.evaluation.support_gate_replication import _fit, compare_cached_endpoints
from association.models.support_gate import ConstantSupportGate
from association.training.constant_gate import collect_training_points, search_constant
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.learned_group_oracle import _bind_source
from tools.experiments.support_gate_replication import production_contract as source_contract

ROOT = Path(__file__).resolve().parents[2]


def run(args):
    source071B, output = Path(args.source071B).resolve(), Path(args.output).resolve()
    source_run = read_json(source071B/'run.json')
    source = Path(source_run['source_system']).resolve()
    for protected in (source071B, source.parent):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('072 output must be separate from immutable inputs')
    if output.exists():
        raise FileExistsError('output already exists; no overwrite/retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('072A fixes CPU4 and advisory-only time')
    torch.set_num_threads(4)
    config, manifest, alignment = map(read_json, (args.config, args.manifest, args.alignment))
    if config != read_json(ROOT/'configs/constant-gate-level1.json'):
        raise ValueError('072A requires the explicit1001-point training-only grid')
    source_contract(source_run['configuration'], manifest, alignment)
    if (source_run.get('status') != 'COMPLETED' or source_run.get('optimizer_updates') != 1000
            or source_run['configuration'].get('position_loss') != 'linear'
            or source_run.get('test_opened') is not False
            or source_run.get('completed_endpoints') != ['initial', 'final']):
        raise ValueError('072A requires the completed071B linear source')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope=config['scope'], status='RUNNING', configuration=config,
        source071B=str(source071B), source_system=str(source), test_opened=False,
        optimizer_updates=0, train_only=True, validation_selection=False,
        completed_endpoints=[], advisory_reached=False, old066_acceptance='REJECTED')
    write_json(output/'run.json', state)

    def check():
        if time.monotonic()-started >= config['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output/'run.json', state)
            print('ADVISORY: continuing fixed scalar search/evaluation, without retry', flush=True)

    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print('072A phase: '+name, flush=True)
        check()

    try:
        phase('bind_actual_successful_training_samples')
        _bind_source(source)
        record, rows, _, _, exposure = _fit(source071B/'fit', 'linear')
        if record['configuration'] != source_run['configuration']:
            raise ValueError('071B actual fit and enclosing configuration differ')
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('prepare_T2100_V300')
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'],
            data_config=source_run['configuration']['data'], alignment_by_scene=alignment,
            check=check, motion_observation_policy='canonical_all')
        for key, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(scene['scene_id']) for scene in dataset[key]] != manifest['splits'][split]:
                raise ValueError('072A complete split/order changed')
        write_json(output/'preparation.json', dataset['report'])
        phase('reconstruct_recorded_T_samples')
        points = collect_training_points(dataset.pop('train'), rows, record, check=check)
        for key in ('sampled_anchors', 'valid_position_points', 'eligible_valid_position_points'):
            if points[key] != exposure[key]:
                raise ValueError('072A reconstructed training population differs: '+key)
        phase('training_only_grid_search')
        fit = search_constant(points, check=check)
        del points
        fit.update(status='COMPLETED', configuration=config, source071B=str(source071B),
            source_system=str(source), source_training_log=str(source071B/'fit/training.jsonl'),
            training_scene_ids=record['training_scene_ids'], test_opened=False,
            sampling_replay_verified=True, successful_updates_replayed=len(rows))
        write_json(output/'fit-sampling.json', [{key: row[key] for key in ('step', 'mode', 'batch')} for row in rows])
        # Freeze the selected scalar on disk before any V evaluation or comparison.
        write_json(output/'fit.json', fit)
        phase('evaluate_cpu_final')
        model = ConstantSupportGate(fit['selected_logit'])
        evaluate_support_gate(model, dataset['validation'], source, metadata, output/'final', check=check)
        for filename in ('run.json', 'summary.json'):
            endpoint = read_json(output/'final'/filename)
            endpoint.update(configuration=config, fit_source=str(output/'fit.json'),
                            selected_logit=fit['selected_logit'], selected_gate=fit['selected_gate'])
            write_json(output/'final'/filename, endpoint)
        state['completed_endpoints'] = ['final']
        phase('cached_population_comparisons')
        for label, endpoint in (('versus_fixed095', 'initial'), ('versus_learned071B', 'final')):
            compare_cached_endpoints(source071B/endpoint, output/'final', metadata['component_ids'],
                                     output/label, check=check)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started,
                     selected_gate=fit['selected_gate'], selected_logit=fit['selected_logit'])
        write_json(output/'summary.json', dict(state, fit_source=str(output/'fit.json'),
            scientific_status='constant control fitted only on original071B successful T samples; not validation-tuned'))
        write_json(output/'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            test_opened=False, optimizer_updates=0, training_forward='scalar physical search only'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source071B', 'manifest', 'alignment', 'config', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)

    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f'cancelled by signal {signum}')

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        result = run(args)
        print(result['status']+': '+str(Path(args.output).resolve()), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
