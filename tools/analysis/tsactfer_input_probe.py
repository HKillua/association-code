"""079: one frozen074D forward probe on32 preselected T_core scenes only."""
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

from association.evaluation.tsactfer_input_probe import (
    DEFAULT, SCOPE, select_roster, bind_source, prepare_inputs, probe)
from association.utils.runtime import read_json, write_json, write_json_gzip

ROOT = Path(__file__).resolve().parents[2]


def production_contract(config, manifest, source):
    if config != DEFAULT:
        raise ValueError('079 requires the frozen bounded probe configuration')
    if manifest != read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):
        raise ValueError('079 requires the existing complete Level1 manifest')
    if source.name != 'LEVEL1-TSA-NATIVE-20261023-074D-attempt-1':
        raise ValueError('079 requires immutable completed074D attempt-1')


def run(args):
    source, output = Path(args.source_074d).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('079 output overlaps immutable source')
    if output.exists():
        raise FileExistsError('079 output exists; no overwrite or retry')
    if args.cpu_threads != 4:
        raise ValueError('079 fixes CPU4')
    config, manifest = read_json(args.config), read_json(args.manifest)
    production_contract(config, manifest, source)
    tsa_config = read_json(ROOT/'configs/tsactfer-level1.json')
    roster = select_roster(manifest, config)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope=SCOPE, status='RUNNING', configuration=config, source_074d=str(source),
        scene_ids=roster, device=args.device, optimizer_updates=0, model_forward=True,
        test_opened=False, validation_observations_read=False, reference_read=False,
        threshold_selected=False, groups_recomputed=False, retries=0, advisory_reached=False,
        old066_acceptance='REJECTED')
    write_json(output/'run.json', state)
    def check():
        if time.monotonic()-started >= config['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output/'run.json', state)
            print('ADVISORY: continue fixed bounded probe, no added scenes or retry', flush=True)
    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print('079 phase: '+name, flush=True)
        check()
    try:
        phase('bind_actual_final_and_training_exposure')
        model, normalizer, binding = bind_source(source, manifest, tsa_config)
        write_json(output/'source-verification.json', binding)
        phase('read_selected_T_core_native_inputs')
        scenes, data_audit = prepare_inputs(args.archive, manifest, roster, normalizer, tsa_config, check)
        write_json(output/'data-audit.json', data_audit)
        phase('frozen_full_and_training_crop_forward')
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        summary, saved = probe(model, scenes, tsa_config, config, device=args.device, check=check)
        (output/'scores').mkdir()
        for task, item in saved.items():
            np.savez_compressed(output/'scores'/f'{task.lower()}.npz', **item['arrays'])
            write_json_gzip(output/'scores'/f'{task.lower()}-views.json.gz', item['rows'])
        summary.update(status='COMPLETED', configuration=config, scene_ids=roster,
            elapsed_seconds=time.monotonic()-started, device=args.device,
            peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
            reference_read=False, validation_observations_read=False, test_opened=False)
        write_json(output/'summary.json', summary)
        lines = ['# 079 frozen T_core input qualification', '',
            'One fixed final checkpoint; no fit, regrouping, threshold selection or V/test observations.', '',
            'These are balanced T-only diagnostic populations. AP is not comparable to075 full-candidate AP.', '',
            '| Head | Domain | Input | Positive / negative | AUC | AP |',
            '|---|---|---|---:|---:|---:|']
        for task, item in summary['tasks'].items():
            for name, condition in item['conditions'].items():
                domain, mode = name.split('__'); metric = condition['pooled']
                lines.append(f"| {task} | {domain} | {mode} | {metric['positive_count']} / {metric['negative_count']} | {metric['roc_auc']} | {metric['average_precision']} |")
        lines.extend(['', summary['limitation'], '',
            'Interpretation: crop/full differences are paired representation effects. Same/cross-scene differences may include geography, timing and candidate composition; they do not establish a shortcut or a causal training-negative failure. Empty or too-small eligibility subsets are non-identifiable, not successful. No follow-on training is automatically authorized by this report.'])
        (output/'report.md').write_text('\n'.join(lines)+'\n')
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            selected_training_scene_ids=roster, normalizer_refit=False, reference_read=False,
            validation_observations_read=False, test_opened=False, training=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-074d', 'manifest', 'config', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('SIGTERM: preserve079 attempt without retry')
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
