#!/usr/bin/env python3
"""084 read-only audit of 078B/081A successful training exposure and V descriptions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from association.evaluation.helper_exposure_audit import (
    BUCKETS, LIMITATIONS, audit_fit, read_json, validation_description)


def _protocol():
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    configs = [read_json(ROOT/'configs'/name) for name in (
        'helper-sequence-pre-pool-level1.json', 'helper-sequence-pre-pool-replica-level1.json')]
    return (manifest['splits']['T_core'], manifest['splits']['V_select'],
            {c['training']['seed']: c for c in configs})


def _write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def _percent(values):
    return ' / '.join('undefined' if values[k] is None else f'{values[k]*100:.4f}%' for k in BUCKETS)


def report(summary):
    lines = ['# 084 helper exposure audit', '', 'Status: '+summary['status'], '',
        'Read-only accounting of completed successful updates. This is not a model result or a causal diagnosis.', '',
        'Buckets: zero / one / two-or-more supported helpers at each query; support is an interval, not an exact observation.', '']
    for fit in summary['fits']:
        actual = fit['actual_exposure']
        unique = fit['unique_scene_anchor_mode']
        lines += [f"## Training seed {fit['seed']}", '',
            f"{fit['successful_updates']} successful updates; {actual['sample_count']} samples; "
            f"{actual['zero_supervision_sample_count']} zero-supervision samples retained in the batch denominator. "
            f"{fit['observed_training_scene_count']}/{fit['legal_training_scene_count']} legal scenes observed.", '',
            '| Weighting/population | Zero / one / two-or-more |', '|---|---|',
            '| All query point exposure | '+_percent(actual['point_exposure']['all_queries']['fractions'])+' |',
            '| Valid query point exposure | '+_percent(actual['point_exposure']['valid_position_queries']['fractions'])+' |',
            '| Actual sample-equal loss coefficients | '+_percent(actual['sample_equal_loss_coefficients']['shares'])+' |',
            '| Scene-equal mean of sample coefficients | '+_percent(actual['scene_equal']['sample_equal_loss_coefficients']['shares'])+' |',
            '| Unique scene-anchor-mode sample coefficients | '+_percent(unique['sample_equal_loss_coefficients']['shares'])+' |', '',
            f"Actual coefficient mass = {actual['sample_equal_loss_coefficients']['coefficient_mass']:.8f}; "
            'a value below one reflects zero-supervision samples, not missing log entries.',
            f"{unique['sample_count']} distinct scene-anchor-mode combinations; "
            f"{fit['repeated_scene_anchor_mode_exposures']} repeated exposures. No deduplication across modes.", '',
            '| Corruption mode | Actual samples | Valid points | Sample-equal coefficients |', '|---|---:|---:|---|']
        for mode, counts in actual['modes'].items():
            lines.append(f"| {mode} | {counts['sample_count']} | {counts['point_exposure']['valid_position_queries']['total']} | "
                         +_percent(counts['sample_equal_loss_coefficients']['shares'])+' |')
        lines.append('')
    if summary.get('validation'):
        lines += ['## V_select descriptions only', '',
            '| Population | Query count | Zero / one / two-or-more |', '|---|---:|---|']
        for name, value in summary['validation']['helper_population'].items():
            lines.append(f"| {name} | {value['total']} | "+_percent(value['fractions'])+' |')
        lines += ['', 'The four saved V error-weighting metrics are retained in summary.json. They are physical-error '
                  'summaries, not recoverable per-anchor training-loss coefficients.', '']
    lines += ['## Interpretation boundaries', '']+[f'- {s}' for s in LIMITATIONS]
    if summary.get('error'):
        lines += ['', 'Audit failed; completed source audits are retained as partial evidence only.', '', summary['error']]
    return '\n'.join(lines)+'\n'


def run(args):
    if type(args.cpu_threads) is not int or args.cpu_threads != 4:
        raise ValueError('084 fixes a CPU4 resource ceiling; execution is single-process stdlib only')
    fits = [Path(p).resolve() for p in args.fit]
    validation = Path(args.validation_summary).resolve()
    if not validation.exists() and validation.name.endswith('.json'):
        validation = validation.with_suffix('.json.gz')
    output = Path(args.output).resolve()
    if output.exists() or Path(args.output).is_symlink():
        raise FileExistsError('084 output exists; no overwrite or automatic retry')
    if not 1 <= len(fits) <= 2 or len(set(fits)) != len(fits):
        raise ValueError('084 accepts one pre_pool source per seed, at most 078B and081A')
    for protected in [*(path.parent for path in fits), validation.parent]:
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('084 output overlaps an immutable input')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', completed_fits=0, cpu_only=True, cpu_threads=4, workers=1,
                 model_forwards=0, optimizer_updates=0,
                 checkpoints_opened=False, raw_archives_read=False, test_opened=False,
                 old066_acceptance='REJECTED', inputs=dict(fits=[str(p) for p in fits], validation_summary=str(validation)))
    summary = dict(status='RUNNING', fits=[], validation=None, limitations=LIMITATIONS)
    _write(output/'run.json', state)
    try:
        training_ids, validation_ids, configs = _protocol()
        seeds = set()
        for path in fits:
            record = read_json(path/'run.json')
            seed = record.get('configuration', {}).get('training', {}).get('seed')
            if type(seed) is not int or seed not in configs or seed in seeds:
                raise ValueError('084 requires one frozen078B/081A pre_pool source per declared seed')
            current = audit_fit(path, allowed_training_scene_ids=training_ids, expected_configuration=configs[seed])
            summary['fits'].append(current)
            seeds.add(seed)
            state['completed_fits'] = len(summary['fits'])
            _write(output/'summary.json', summary)
            _write(output/'run.json', state)
        summary['validation'] = validation_description(validation, allowed_validation_scene_ids=validation_ids)
        summary['status'] = 'COMPLETED'
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        _write(output/'summary.json', summary)
        (output/'report.md').write_text(report(summary), encoding='utf-8')
        _write(output/'run.json', state)
        return summary
    except BaseException as error:
        text = f'{type(error).__name__}: {error}'
        summary.update(status='FAILED', error=text)
        state.update(status='FAILED', error=text, elapsed_seconds=time.monotonic()-started)
        _write(output/'summary.json', summary)
        (output/'report.md').write_text(report(summary), encoding='utf-8')
        _write(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', action='append', required=True, help='Completed078B/081A fit directory; repeat once for the other seed')
    parser.add_argument('--validation-summary', required=True, help='081 comparison summary.json or summary.json.gz')
    parser.add_argument('--output', required=True, help='New non-overlapping directory; never overwritten')
    parser.add_argument('--cpu-threads', type=int, default=4, help='Fixed CPU4 ceiling; no model or worker pool is loaded')
    result = run(parser.parse_args(argv))
    print(result['status'], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
