#!/usr/bin/env python3
"""089 fixed CPU replay of T-only physical priors; no checkpoints or inference."""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from association.evaluation.helper_prior_replay import (
    LIMITS, audit_inputs, bind_sources, iter_training_inputs, read_json, require, same)


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def report(summary):
    lines = ['# 089 T-only physical-prior replay', '', 'Status: '+summary['status'], '',
        'Full and dropped refer to physical fixed.95 starting states, not trained network outputs.']
    if summary.get('global_metrics'):
        value = summary['global_metrics']
        lines += ['', '| Population/weight | Full | Dropped |', '|---|---:|---:|',
            f"| Repeated valid-query MAE m | {value['point_mae_full_m']} | {value['point_mae_dropped_m']} |",
            f"| Repeated valid-query RMSE m | {value['point_rmse_full_m']} | {value['point_rmse_dropped_m']} |",
            f"| Original sample-equal linear loss (km) | {value['sample_equal_linear_loss_full']} | {value['sample_equal_linear_loss_dropped']} |", '',
            f"{summary['successful_sample_count']} successful sample exposures, "
            f"{summary['zero_supervision_sample_count']} zero-supervision exposures retained in the denominator; "
            f"coefficient mass {value['coefficient_mass']}.", '',
            'The joint mode × original-support-region × original/retained helper-count table is in summary.json. '
            'Every cell uses the global sample denominator; conditional MAEs must not be added.', '',
            'Per-scene source/denominator records are in scene-summary.json.gz. No new V comparison is performed.']
    if summary.get('error'): lines += ['', summary['error']]
    lines += ['', '## Limits', '']+['- '+v for v in LIMITS]
    return '\n'.join(lines)+'\n'


def run(args, *, expected_config=None, expected_manifest=None, expected_alignment=None):
    require(args.cpu_threads == 4 and args.time_policy == 'advisory', '089 fixes CPU4 and advisory time')
    output = Path(args.output).resolve()
    roots = [Path(args.dropout_run).resolve(), Path(args.linear_run).resolve()]
    protected = [*roots, *(Path(p).resolve() for p in args.archive)]
    require(all(output != p and output not in p.parents and p not in output.parents for p in protected),
            'output must not overlap immutable sources')
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError('089 output exists; no overwrite or automatic retry')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', cpu_only=True, cpu_threads=4, model_forwards=0, optimizer_updates=0,
        checkpoints_opened=False, raw_archive_split='T_core only', validation_payloads_read=False,
        test_opened=False, old066_acceptance='REJECTED', completed_scenes=0, advisory_reached=False)
    summary = dict(status='RUNNING', limitations=LIMITS)
    write(output/'run.json', state)
    write(output/'execution.json', dict(arguments=vars(args), scope='089-helper-prior-replay',
        no_checkpoints=True, no_models=True, no_forward=True, no_optimizer=True, raw_archive_split='T_core only'))
    try:
        import torch
        torch.set_num_threads(4)
        manifest, alignment = read_json(args.manifest), read_json(args.alignment)
        expected_config = expected_config or read_json(ROOT/'configs/sequence-position-transformer-helper-dropout-level1.json')
        expected_manifest = expected_manifest or read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
        expected_alignment = expected_alignment or read_json(ROOT/'configs/data/mtad-level1-alignment.json')
        require(same(manifest, expected_manifest) and same(alignment, expected_alignment),
                '089 must use frozen Level1 manifest/alignment')
        bound = bind_sources(*roots, manifest, alignment, expected_config, archives=args.archive)
        write(output/'source-verification.json', bound['verification'])

        def check():
            if time.monotonic()-started >= 3600 and not state['advisory_reached']:
                state['advisory_reached'] = True
                print('ADVISORY: continuing the same fixed089 T-only replay.', flush=True)
                write(output/'run.json', state)

        def on_scene(row):
            state['completed_scenes'] += 1
            state['last_scene_id'] = row['scene_id']
            state['elapsed_seconds'] = time.monotonic()-started
            if state['completed_scenes'] % 25 == 0: write(output/'run.json', state)

        summary = audit_inputs(bound, iter_training_inputs(args.archive, manifest, alignment, expected_config,
                                                         bound=bound, check=check),
                               on_scene=on_scene)
        per_scene = summary.pop('per_scene')
        with gzip.open(output/'scene-summary.json.gz', 'wt', encoding='utf-8') as handle:
            json.dump(per_scene, handle, ensure_ascii=False, allow_nan=False)
        summary['source_verification'] = 'source-verification.json'
        summary['scene_summary'] = 'scene-summary.json.gz'
        exposure = bound['verification']['augmentation_exposure']
        summary['point_frequency_exposure'] = {key: {population: dict(counts=counts,
            fractions={k: v/sum(counts.values()) if sum(counts.values()) else None for k, v in counts.items()})
            for population, counts in exposure[key].items()} for key in ('original_support', 'retained_support')}
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write(output/'summary.json', summary); write(output/'run.json', state)
        (output/'report.md').write_text(report(summary), encoding='utf-8')
        return summary
    except BaseException as error:
        text = f'{type(error).__name__}: {error}'
        state.update(status='FAILED', error=text, elapsed_seconds=time.monotonic()-started)
        summary.update(status='FAILED', error=text)
        write(output/'run.json', state); write(output/'summary.json', summary)
        (output/'report.md').write_text(report(summary), encoding='utf-8')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dropout-run', 'linear-run', 'manifest', 'alignment', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--archive', nargs='+', required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--time-policy', choices=('advisory',), default='advisory')
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(f"{result['status']} 089 T-only replay: {args.output}")
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'FAILED {type(error).__name__}: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
