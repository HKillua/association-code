"""074D: native128 TSA final1000 fit and OLS/TSA full-system CPU evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.association import select_threshold, evaluate_scenes
from association.evaluation.tsactfer_level1 import verify_native_source, evaluate_system
from association.utils.runtime import read_json, write_json, save_records, load_records
from baselines.tsactfer_level1 import (SCOPE, configuration, prepare_native_dataset,
                                     train_native, final_model, score_scenes)
from tools.analysis.learned_group_oracle import _bind_source
from tools.analysis.compare_association_duration import _load_metadata

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest):
    configuration(config)
    if config != read_json(ROOT/'configs/tsactfer-level1.json'):
        raise ValueError('074D requires its frozen native128 final1000 configuration')
    if manifest != read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):
        raise ValueError('074D requires frozen complete Level1 T2100/V300 manifest')


def run(args):
    source, output = Path(args.source_system).resolve(), Path(args.output).resolve()
    if output == source.parent or source.parent in output.parents or output in source.parent.parents:
        raise ValueError('074D output must not overlap immutable056')
    if output.exists():
        raise FileExistsError('074D output exists; no overwrite or automatic retry')
    if args.cpu_threads != 4 or args.time_policy != 'advisory':
        raise ValueError('074D fixes CPU4 and advisory-only budget')
    torch.set_num_threads(4)
    config, manifest = read_json(args.config), read_json(args.manifest)
    production_contract(config, manifest)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope=SCOPE, status='RUNNING', configuration=config,
        source_system=str(source), optimizer_updates=0, test_opened=False, retries=0,
        advisory_reached=False, endpoint='final', checkpoint_selection=False,
        train_device=args.device, inference_device='cpu', old066_acceptance='REJECTED')
    write_json(output/'run.json', state)
    def check():
        if time.monotonic()-started >= config['training']['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output/'run.json', state)
            print('ADVISORY: continue this fixed TSA attempt; no retry or extra calls', flush=True)
    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print('074D phase: '+name, flush=True)
        check()
    try:
        phase('bind056')
        _bind_source(source)
        metadata = _load_metadata(source, manifest, 1000)
        metadata['component_ids'] = {sid: str(manifest['components'][sid]) for sid in metadata['scene_ids']}
        phase('stream_native_T2100_V300')
        train, validation, normalizer, preparation = prepare_native_dataset(args.archive, manifest, config,
            verify_validation=lambda scene: verify_native_source(scene, source), check=check)
        write_json(output/'data_audit.json', preparation)
        phase('fit_fixed1000')
        if args.device == 'cuda:0':
            torch.cuda.reset_peak_memory_stats()
        fit = train_native(train, normalizer, config, output/'fit', device=args.device, check=check)
        del train
        state['optimizer_updates'] = fit['optimizer_updates']
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        phase('score_actual_final_on_cpu')
        model, snapshot = final_model(output/'fit', config, normalizer)
        records = score_scenes(model, validation, 'cpu', config['training']['inference_batch_tracks'], check)
        del model, validation
        if [str(row['scene_id']) for row in records] != manifest['splits']['V_select']:
            raise ValueError('074D final score population differs')
        selection = select_threshold(records)
        binding = dict(scope=SCOPE, split='V_select', seed=config['seed'], endpoint='final',
            step=snapshot['step'], checkpoint=str(output/'fit/final.pt'), configuration=config,
            normalizer=normalizer, selection=selection)
        save_records(output/'final-scores', records, binding)
        cached = load_records(output/'final-scores', binding)
        for original, saved in zip(records, cached):
            for key in ('node_ids', 'pairs', 'labels', 'sources', 'identities', 'intervals', 'scores'):
                if not np.array_equal(np.asarray(original[key]), np.asarray(saved[key])):
                    raise ValueError('074D final score cache roundtrip changed '+key)
        endpoint = dict(status='COMPLETED', configuration=config, checkpoint=binding['checkpoint'],
            step=snapshot['step'], actual_updates=snapshot['actual_updates'], normalizer=normalizer,
            selection=selection, metrics=evaluate_scenes(cached, selection['threshold']),
            scene_ids=manifest['splits']['V_select'], test_opened=False,
            score_binding=binding, inference_device='cpu', scores_from_actual_final_checkpoint=True)
        write_json(output/'final-endpoint.json', endpoint)
        phase('ols_tsa_full_system')
        result = evaluate_system(cached, selection['threshold'], source, metadata, output/'system', check=check)
        if result['association']['tsa']['metrics'] != endpoint['metrics']:
            raise ValueError('074D final system association differs from bound final scores')
        summary = dict(status='COMPLETED', scope=SCOPE, configuration=config, source_system=str(source),
            optimizer_updates=fit['optimizer_updates'], nonzero_lr_updates=fit['exposure']['nonzero_lr_updates'],
            task_calls=fit['exposure']['task_calls'], parameter_count=fit['parameter_count'],
            final_endpoint=str(output/'final-endpoint.json'), system_summary=str(output/'system/summary.json'),
            endpoint='final', selection=selection, association=endpoint['metrics'], test_opened=False,
            fit_elapsed_seconds=fit['elapsed_seconds'], elapsed_seconds=time.monotonic()-started,
            limitation='One fixed native128 offline adaptation pilot, not author online method or convergence; comparisons are reused V development evidence.')
        write_json(output/'summary.json', summary)
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        if (output/'fit/run.json').exists():
            state['optimizer_updates'] = read_json(output/'fit/run.json').get('optimizer_updates', 0)
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            test_opened=False, resumed=False, reference_source='immutable056 cache only, after predictions'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-system', 'manifest', 'config', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('SIGTERM: preserve074D attempt without retry')
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
