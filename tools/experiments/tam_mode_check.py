"""Inspect the selected TAM on fixed T_core/V inputs, with zero optimizer updates.

Reads only the selected scenes' native measurements and labels, never reference
CSV files or T_test. Eight independent repetitions per TAM mode are fixed in
this diagnostic. Wall time is advisory and cannot terminate the computation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'checkpoint', 'output', 'device'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--expected-selected-update', type=int,
                        help='optional explicit selected-update expectation (035 handover uses 600)')
    parser.add_argument('--advisory-seconds', type=float, default=900.)
    args = parser.parse_args(argv)
    started = time.monotonic()
    output, owns_output = Path(args.output), False
    previous_signal = signal.getsignal(signal.SIGTERM)
    run = dict(status='RUNNING', mode='same_input_zero_update_tam_inspection',
        optimizer_updates=0, test_opened=False, reference_read=False,
        threshold_reselected=False, margin_recalibrated=False, normalizer_refit=False,
        checkpoint=str(Path(args.checkpoint).resolve()), archive=str(Path(args.archive).resolve()),
        manifest=str(Path(args.manifest).resolve()), device=args.device,
        repeats=8, seed=2035, cpu_threads=args.cpu_threads,
        expected_selected_update=args.expected_selected_update,
        advisory_seconds=args.advisory_seconds, time_policy='advisory',
        advisory_exceeded=False, stage='loading')

    def write_run(status, **extra):
        run.update(status=status, elapsed_seconds=time.monotonic()-started, **extra)
        temporary = output/'run.json.tmp'
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(output/'run.json')

    def check():
        if not run['advisory_exceeded'] and time.monotonic()-started > args.advisory_seconds:
            run['advisory_exceeded'] = True
            run['advisory_message'] = 'advisory time exceeded; original fixed diagnostic continues without retry'
            write_run('RUNNING')

    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('TAM mode check interrupted by SIGTERM')

    try:
        import math
        if (args.cpu_threads < 1 or not math.isfinite(args.advisory_seconds) or args.advisory_seconds <= 0
                or (args.expected_selected_update is not None and args.expected_selected_update < 1)):
            raise ValueError('positive cpu-threads and finite positive advisory-seconds required')
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        signal.signal(signal.SIGTERM, terminated)
        write_run('RUNNING')
        import numpy as np
        import torch
        from association.utils import runtime
        from association.training.tmtaff import load_complete_snapshot
        from association.evaluation.tam_modes import prepare_fixed_samples, inspect_tam_modes
        torch.set_num_threads(args.cpu_threads)
        device = runtime.device_from_string(args.device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        manifest = runtime.read_json(args.manifest)
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if (args.expected_selected_update is not None
                and (saved.get('selected') or {}).get('update') != args.expected_selected_update):
            raise ValueError('selected snapshot update differs from explicit expected-selected-update')
        check()
        run.update(stage='preparation', device=str(device),
            source_selection={key: value for key, value in saved.get('selected', {}).items() if key != 'states'},
            source_cursor=saved.get('cursor'))
        write_run('RUNNING')
        samples, preparation = prepare_fixed_samples(args.archive, manifest, saved, check=check)
        runtime.write_json(output/'preparation.json', preparation)
        run.update(stage='inspection', selected_scene_ids=[s['metadata']['scene_id'] for s in samples],
                   batch_tracks=saved['config']['batch_tracks'])
        write_run('RUNNING')
        devices = [device.index] if device.type == 'cuda' else []
        # Snapshot construction initializes modules before loading weights; its
        # random draws must not leak into the caller's random stream either.
        with torch.random.fork_rng(devices=devices):
            loaded = load_complete_snapshot(args.checkpoint, device=device, selected=True)
            if loaded['normalizer'] != saved['normalizer'] or loaded['selection'] != run['source_selection']:
                raise ValueError('selected loader changed normalizer or selection metadata')
            report, arrays = inspect_tam_modes(loaded['model'], samples, device=device,
                batch_tracks=saved['config']['batch_tracks'], repeats=8, seed=2035, check=check)
        report.update(source_selection=run['source_selection'], normalizer=saved['normalizer'],
                      preparation=preparation, raw_arrays='distances.npz')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        run['stage'] = 'result_io'
        check()
        np.savez_compressed(output/'distances.npz', **arrays)
        runtime.write_json(output/'summary.json', report)
        check()
        run['stage'] = 'completed'
        write_run('COMPLETED')
        print(f'COMPLETED zero-update TAM mode check: {output.resolve()}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        interrupted = isinstance(error, (InterruptedError, KeyboardInterrupt))
        if owns_output:
            summary = output/'summary.json'
            if summary.exists():
                summary.replace(output/'summary.incomplete.json')
            write_run('INTERRUPTED' if interrupted else 'FAILED',
                failure_type=type(error).__name__, failure=str(error))
        print(f'{"INTERRUPTED" if interrupted else "FAILED"} {type(error).__name__}: {error}',
              file=sys.stderr, flush=True)
        return 130 if interrupted else 1
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    raise SystemExit(main())
