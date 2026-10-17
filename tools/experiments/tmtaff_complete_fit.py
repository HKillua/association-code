"""Inspect or fit the explicitly configured TMTAFF native-time adaptation.

This executable grants no experiment/GPU budget. A fit requires a separately
approved configuration, explicit device and per-scene reference clock mapping.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from association import runtime
from association.fusion import PairFusionConfig
from association.models import ModelConfig
from association.timed_dataset import load_timed_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'manifest', 'alignment', 'config', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', help='required for fit; no automatic CPU/GPU fallback')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--resume', help='an INTERRUPTED run; output must be a new directory')
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    started = time.monotonic()
    previous_signal = signal.getsignal(signal.SIGTERM)

    def terminated(signum, frame):
        raise InterruptedError('SIGTERM: preparation/fit interrupted, not completed')

    signal.signal(signal.SIGTERM, terminated)
    try:
        if args.cpu_threads < 1:
            raise ValueError('--cpu-threads must be positive')
        if args.inspect_only and args.resume:
            raise ValueError('--resume applies to fit, not inspect-only')
        if not args.inspect_only and not args.device:
            raise ValueError('--device must be explicit for fit')
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}')
        config = runtime.read_json(args.config)
        if (set(config) != {'scope', 'model', 'fusion', 'training', 'data', 'train_splits'}
                or config['scope'] != 'TMTAFF-time-adapted-v1'):
            raise ValueError('Explicit TMTAFF-time-adapted-v1 configuration required')
        model, fusion = ModelConfig(**config['model']), PairFusionConfig(**config['fusion'])
        if (model.length, model.d_model) != (fusion.length, fusion.d_model):
            raise ValueError('Model and fusion length/dimension must agree')
        if set(config['data']) != {'max_observations', 'time_unit_seconds'}:
            raise ValueError('data requires explicit max_observations/time_unit_seconds')
        data_config = {**config['data'], 'length': model.length}
        check = None
        if not args.inspect_only:
            from association.complete_training import _configuration
            training = _configuration(config['training'])
            spent = 0.
            if args.resume:
                prior = runtime.read_json(Path(args.resume) / 'run.json')
                if prior.get('status') != 'INTERRUPTED':
                    raise ValueError('Only an INTERRUPTED run may resume')
                spent = float(prior['elapsed_seconds'])
            def check():
                if spent + time.monotonic() - started >= training['max_hours'] * 3600:
                    raise TimeoutError('Cumulative budget exhausted during input preparation')
            check()
        import torch
        torch.set_num_threads(args.cpu_threads)
        dataset = load_timed_dataset(args.archive, runtime.read_json(args.manifest),
            train_splits=config['train_splits'], data_config=data_config,
            alignment_by_scene=runtime.read_json(args.alignment), check=check)
        preparation = {'status': 'INSPECTED' if args.inspect_only else 'PREPARED',
                       'report_phase': 'input preparation before model construction',
                       'configuration': config, 'cpu_threads': args.cpu_threads,
                       'dataset': dataset['report'], 'elapsed_seconds': time.monotonic()-started,
                       'model_constructed': False, 'gpu_training_started': False}
        if args.inspect_only:
            output.mkdir(parents=True, exist_ok=False)
            runtime.write_json(output / 'preparation.json', preparation)
            print(f'INSPECTED native-time inputs: {output.resolve()}', flush=True)
        else:
            from association.complete_training import train_complete
            try:
                result = train_complete(dataset['train'], dataset['validation'], model, fusion,
                    training, output, device=args.device, resume_from=args.resume,
                    initial_elapsed_seconds=time.monotonic()-started)
            finally:
                if output.is_dir():
                    runtime.write_json(output / 'preparation.json', preparation)
            print(f"{result['status']} TMTAFF-time-adapted-v1: {output.resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    raise SystemExit(main())
