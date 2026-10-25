"""100 fixed real-Level1 fit and final complete V; no simulation qualification."""
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
from association.evaluation.native_query import evaluate
from association.evaluation.position_sources import _sequence_sampling
from association.training.frozen_position_adapter import load_checkpoint as load_base
from association.training.native_query import fit, load_checkpoint, ARMS
from association.utils.runtime import read_json, write_json
from tools.analysis.compare_association_duration import _load_metadata
from tools.analysis.compare_frozen_position_adapter import adapter_metadata
from tools.analysis.learned_group_oracle import _bind_source

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, alignment):
    if config.get('arm') not in ARMS or config != read_json(ROOT/'configs'/f'native-query-{config["arm"]}-level1.json'):
        raise ValueError('100 requires its declared arm configuration')
    if (manifest != read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
            or alignment != read_json(ROOT/'configs/data/mtad-level1-alignment.json')):
        raise ValueError('100 requires original complete Level1 split and alignment')


def bind_base(path, source, manifest, config):
    """Use actual completed090C final, not original072C or a matching filename."""
    _, metadata = adapter_metadata(path, profile='090')
    if metadata['source_system'] != str(source) or metadata['scene_ids'] != manifest['splits']['V_select']:
        raise ValueError('090C source056 or complete V differs')
    expected = read_json(ROOT/'configs/frozen-position-adapter-full-finetune-level1.json')
    record, rows, _ = _sequence_sampling(path/'fit')
    model, saved = load_base(path/'fit/final.pt')
    if (record['configuration'] != expected or saved['configuration'] != expected
            or saved['step'] != 4000 or record['training_scene_ids'] != manifest['splits']['T_core']
            or len(rows) != config['training']['updates']):
        raise ValueError('100 requires the completed090C actual strong final and all successful batches')
    return model.teacher, saved['source_teacher']['configuration']['model'], rows


def run(args):
    source, strong, output = [Path(p).resolve() for p in (args.source_system, args.strong_run, args.output)]
    if any(output == p or output in p.parents or p in output.parents for p in (source.parent, strong)):
        raise ValueError('new output overlaps immutable source')
    if output.exists(): raise FileExistsError('100 does not reuse an existing attempt')
    if args.cpu_threads != 4 or args.time_policy != 'advisory': raise ValueError('100 fixes CPU4/advisory')
    torch.set_num_threads(4)
    config, manifest, alignment = [read_json(p) for p in (args.config, args.manifest, args.alignment)]
    production_contract(config, manifest, alignment)
    _bind_source(source)
    base, base_config, replay = bind_base(strong, source, manifest, config)
    metadata = _load_metadata(source, manifest, 1000)
    metadata['component_ids'] = {s: str(manifest['components'][s]) for s in metadata['scene_ids']}
    if args.device == 'cuda:0' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no silent CPU fit')
    started = time.monotonic(); output.mkdir(parents=True)
    state = dict(scope=config['scope'], status='RUNNING', phase='prepare', configuration=config,
        source_system=str(source), strong_run=str(strong), optimizer_updates=0, test_opened=False,
        time_policy='advisory', advisory_reached=False, retries=0, validation_selection=False,
        groups_recomputed=False, threshold_reselected=False, completed_endpoints=[])
    write_json(output/'run.json', state)

    def check():
        elapsed = time.monotonic()-started
        if elapsed >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state.update(advisory_reached=True, elapsed_seconds=elapsed)
            write_json(output/'run.json', state)
            print('ADVISORY: 100 exceeded suggested time; same fixed attempt continues', flush=True)

    try:
        dataset = load_timed_dataset(args.archive, manifest, train_splits=['T_core'], data_config=config['data'],
            alignment_by_scene=alignment, check=check, motion_observation_policy='canonical_all',
            retain_native_events=config['arm'] != 'budget')
        for name, split in (('train', 'T_core'), ('validation', 'V_select')):
            if [str(s['scene_id']) for s in dataset[name]] != manifest['splits'][split]:
                raise ValueError('complete population/order differs: '+split)
        write_json(output/'preparation.json', dataset['report'])
        state['phase'] = 'fit'; write_json(output/'run.json', state)
        trained = fit(dataset.pop('train'), config, base, base_config, replay, output/'fit',
            source=dict(run=str(strong), checkpoint=str(strong/'fit/final.pt'), source_system=str(source),
                        source_total_updates=8000), device=args.device, check=check)
        del base
        if torch.cuda.is_initialized(): torch.cuda.empty_cache()
        state.update(phase='final_full_V', optimizer_updates=trained['optimizer_updates'])
        write_json(output/'run.json', state)
        model, saved = load_checkpoint(output/'fit/final.pt')
        if saved['step'] != config['training']['updates'] or saved['configuration'] != config:
            raise ValueError('fixed final checkpoint differs')
        evaluated = evaluate(model, dataset['validation'], source, metadata, output/'final', check=check)
        provenance = dict(path=str(output/'fit/final.pt'), actual_updates=saved['step'], step=saved['step'],
                          configured_final_updates=config['training']['updates'], strong_source=str(strong/'fit/final.pt'))
        evaluated.update(configuration=config, checkpoint=provenance)
        write_json(output/'final/summary.json', evaluated)
        endpoint = read_json(output/'final/run.json'); endpoint.update(configuration=config, checkpoint=provenance)
        write_json(output/'final/run.json', endpoint)
        state.update(status='COMPLETED', phase='completed', completed_endpoints=['final'], elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        summary = dict(state, parameter_count=trained['parameter_count'], trainable_parameter_count=trained['trainable_parameter_count'],
            first100=trained['first100'], initial_endpoint_reused=str(strong/'final'),
            fixed_base_verified=True, actual_successful_batches_matched=trained['actual_successful_batches_matched'],
            limitations=['Single seed and repeatedly used V; not independent identity generalization.',
                'T1P strong provenance-aware grid and second seed still required before native-representation claims.',
                'Position only; fixed056 grouping, ordinary speed/course; no domain-peer victory claim.'])
        write_json(output/'summary.json', summary)
        return summary
    except BaseException as error:
        if (output/'fit/run.json').exists(): state['optimizer_updates'] = read_json(output/'fit/run.json')['optimizer_updates']
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state); raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config, test_opened=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'alignment', 'source-system', 'strong-run', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError(f'cancelled by signal {signum}')
    old = signal.signal(signal.SIGTERM, cancel)
    try:
        run(args); return 0
    except (KeyboardInterrupt, InterruptedError) as error:
        print(str(error), file=sys.stderr); return 130
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr); return 1
    finally: signal.signal(signal.SIGTERM, old)


if __name__ == '__main__': raise SystemExit(main())
