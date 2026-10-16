"""Repository-local CLI. CUDA requests fail rather than silently using CPU."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from .config import load_config
from .data import validate_manifest
from .runtime import device_from_string, read_json


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Create immutable split and training/validation coverage audit')
    prepare.add_argument('--archive', required=True)
    prepare.add_argument('--config', default='configs/pilot.json')
    prepare.add_argument('--output', required=True)
    for name in ('train-reference', 'run-pilot', 'profile'):
        command = sub.add_parser(name)
        command.add_argument('--archive', required=True)
        manifest = command.add_mutually_exclusive_group(required=True)
        manifest.add_argument('--prepared', help='Directory containing manifest.json and completed audit.json')
        manifest.add_argument('--manifest', help='Explicit frozen manifest.json')
        command.add_argument('--config', default='configs/pilot.json')
        command.add_argument('--output', required=True)
        command.add_argument('--device', required=True, help='Explicit cuda:0 on server; cpu for local checks')
        command.add_argument('--cpu-threads', type=int, default=4)
        if name in ('train-reference', 'profile'):
            command.add_argument('--role', choices=['A0', 'A_full'], required=True)
            command.add_argument('--branch', choices=['ce', 'contrastive'], required=True)
            command.add_argument('--seed', type=int, required=True)
        if name == 'run-pilot':
            command.add_argument('--seeds', nargs='+', type=int, help='Nonempty prefix of declared config seeds')
            command.add_argument('--reference-root', help='Reuse a complete verified reference tree; missing fits fail')
        if name == 'profile':
            command.add_argument('--steps', type=int, default=200)
            command.add_argument('--max-minutes', type=float, default=30)
    smoke = sub.add_parser('smoke', help='Synthetic end-to-end functional check; never a research result')
    smoke.add_argument('--output', required=True)
    smoke.add_argument('--device', required=True)
    smoke.add_argument('--config', default='configs/smoke.json')
    smoke.add_argument('--cpu-threads', type=int, default=1)
    return parser


def _manifest(args, config):
    if args.prepared:
        directory = Path(args.prepared)
        audit = read_json(directory/'audit.json')
        if audit.get('status') != 'COMPLETED':
            raise ValueError('Preparation audit is not COMPLETED')
        manifest = read_json(directory/'manifest.json')
        if audit.get('length') != config['model']['length']:
            raise ValueError('Prepared registration length differs from config.model.length')
        if audit.get('manifest_sha256') != manifest.get('manifest_sha256'):
            raise ValueError('Preparation audit and manifest do not match')
    else:
        manifest = read_json(args.manifest)
    validate_manifest(manifest)
    return manifest


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'prepare':
            from .dataset import prepare
            prepare(args.archive, config, args.output)
        else:
            if args.cpu_threads < 1:
                raise ValueError('--cpu-threads must be positive')
            torch.set_num_threads(args.cpu_threads)
            device = device_from_string(args.device)
            if args.command == 'smoke':
                from .dataset import make_synthetic_archive, prepare
                from .pipeline import run_pilot
                if config['protocol'] != 'synthetic-smoke-only-not-research-result':
                    raise ValueError('smoke requires an explicitly synthetic smoke config')
                root = Path(args.output)
                root.mkdir(parents=True, exist_ok=False)
                archive = make_synthetic_archive(root/'fixture.zip')
                manifest = prepare(archive, config, root/'prepared', synthetic=True)
                run_pilot(archive, manifest, config, device, root/'pilot')
            else:
                manifest = _manifest(args, config)
                if args.command == 'run-pilot':
                    from .pipeline import run_pilot
                    run_pilot(args.archive, manifest, config, device, args.output,
                              reference_root=args.reference_root, seeds=args.seeds)
                else:
                    if args.seed not in config['seeds']:
                        raise ValueError('--seed must be declared in config.seeds')
                    if args.command == 'train-reference':
                        from .reference import train_reference
                        train_reference(args.archive, manifest, config, args.role, args.branch,
                                        args.seed, device, args.output)
                    else:
                        from .profile import profile_reference
                        profile_reference(args.archive, manifest, config, args.role, args.branch,
                                          args.seed, device, args.output, steps=args.steps,
                                          max_minutes=args.max_minutes)
        print(f'COMPLETED {Path(args.output).resolve()}', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        print(f'FAILED {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
