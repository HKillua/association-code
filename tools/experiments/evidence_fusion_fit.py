"""Train fixed-motion utility fusion, or run a bounded synthetic software check.

No job launch, automatic retry, threshold search, test access or hard timeout.
Native fit requires explicit frozen manifest, clock alignment and configuration.
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
from association.models.cooperative import CooperativeConfig
from association.training.evidence_fusion import train_evidence_fusion
from association.utils import runtime
from tools.experiments.cooperative_fit import synthetic_scenes


def configuration(value):
    if (set(value) != {'scope', 'data', 'train_splits', 'model', 'training'}
            or value['scope'] != 'native-evidence-fusion-v1'
            or value['train_splits'] != ['T_core']
            or value['data'] != dict(length=64, max_observations=128, time_unit_seconds=20.,
                                     reference_longitude_policy='shortest_arc')
            or set(value['model']) != {'exact_observation_lock'}
            or not isinstance(value['model']['exact_observation_lock'], bool)):
        raise ValueError('explicit native-evidence-fusion-v1 frozen-data configuration required')
    training = value['training']
    integers = ('seed', 'updates', 'batch_size', 'evaluate_every')
    positive = ('learning_rate', 'gradient_clip', 'advisory_seconds')
    if set(training) != {*integers, *positive}:
        raise ValueError('unexpected training settings')
    for key in integers:
        if isinstance(training[key], bool) or not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(key+' must be a positive integer')
    for key in positive:
        number = training[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 < number < float('inf'):
            raise ValueError(key+' must be finite and positive')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('fit', 'check'), required=True)
    parser.add_argument('--archive')
    parser.add_argument('--manifest')
    parser.add_argument('--alignment')
    parser.add_argument('--config')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    owns_output = False
    config = preparation = None

    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f'cancelled by signal {signum}')

    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        if args.cpu_threads < 1:
            raise ValueError('--cpu-threads must be positive')
        torch.set_num_threads(args.cpu_threads)
        if args.device == 'cuda:0' and not torch.cuda.is_available():
            raise ValueError('requested CUDA is unavailable; no CPU fallback')
        if Path(args.output).exists():
            raise FileExistsError('output directory already exists')
        owns_output = True
        paths = (args.archive, args.manifest, args.alignment, args.config)
        if args.mode == 'fit':
            if not all(paths):
                raise ValueError('fit requires archive, manifest, alignment and config')
            config = configuration(runtime.read_json(args.config))
            dataset = load_timed_dataset(args.archive, runtime.read_json(args.manifest),
                train_splits=config['train_splits'], data_config=config['data'],
                alignment_by_scene=runtime.read_json(args.alignment))
            train, validation, preparation = dataset['train'], dataset['validation'], dataset['report']
        else:
            if any(paths):
                raise ValueError('synthetic check must not accept native paths or real configuration')
            config = configuration(runtime.read_json(Path(__file__).resolve().parents[2]
                                                     /'configs/evidence-fusion.json'))
            config['training'].update(updates=3, batch_size=2, evaluate_every=3, advisory_seconds=300.)
            train, validation = synthetic_scenes(CooperativeConfig(length=64, max_observations=4))
            preparation = None
        try:
            report = train_evidence_fusion(train, validation, config, args.output, device=args.device,
                                          initial_elapsed_seconds=time.monotonic()-started)
        finally:
            if Path(args.output).is_dir():
                runtime.write_json(Path(args.output)/'execution.json', dict(command=sys.argv if argv is None else argv,
                    synthetic=args.mode == 'check', native_MTAD_opened=args.mode == 'fit',
                    cpu_threads=args.cpu_threads, preparation=preparation, configuration=config))
        print(f'{report["status"]}: {Path(args.output).resolve()}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        if owns_output:
            destination = Path(args.output)
            destination.mkdir(parents=True, exist_ok=True)
            if not (destination/'run.json').exists():
                runtime.write_json(destination/'run.json', dict(
                    status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                    phase='preparation', optimizer_updates=0,
                    error=f'{type(error).__name__}: {error}', test_opened=False,
                    elapsed_seconds=time.monotonic()-started))
            if not (destination/'execution.json').exists():
                runtime.write_json(destination/'execution.json', dict(
                    command=sys.argv if argv is None else argv, synthetic=args.mode == 'check',
                    native_MTAD_requested=args.mode == 'fit', configuration=config,
                    preparation=preparation, cpu_threads=args.cpu_threads))
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
