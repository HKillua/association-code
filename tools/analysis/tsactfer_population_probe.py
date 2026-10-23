"""080: cached TSA population comparison, no model/archive/threshold changes.

Reuse074D full validation scores and079 controlled training scores. The bounded
validation sample follows the same eligibility/capping rules, not the same data
or a paired T/V random draw. Reweighted AP is an algebraic reference only.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from association.evaluation.tsactfer_input_probe import DEFAULT as PROBE079, select_roster
from association.evaluation.tsactfer_population_probe import DEFAULT, SCOPE, analyze_population
from association.utils.runtime import read_json, write_json, write_json_gzip
from tools.analysis.tsactfer_score_diagnostics import bind_source

ROOT = Path(__file__).resolve().parents[2]
SOURCE079 = 'LEVEL1-TSA-INPUT-PROBE-20261024-079-attempt-1'


def production_contract(config, manifest, source_074d, source_079):
    if config != DEFAULT or manifest != read_json(ROOT/'configs/data/mtad-level1-scene3000.json'):
        raise ValueError('080 requires the fixed config and unchanged Level1 manifest')
    if (source_074d.name != 'LEVEL1-TSA-NATIVE-20261023-074D-attempt-1'
            or source_079.name != SOURCE079):
        raise ValueError('080 requires immutable completed074D and079 attempt-1 sources')


def bind_probe(source, source_074d, manifest):
    """Only JSON/gzip/NPZ consistency; never load the named checkpoint."""
    source, source_074d = Path(source), Path(source_074d)
    run, summary = read_json(source/'run.json'), read_json(source/'summary.json')
    binding = read_json(source/'source-verification.json')
    audit, execution = read_json(source/'data-audit.json'), read_json(source/'execution.json')
    roster = select_roster(manifest, PROBE079)
    for item in (run, summary):
        if (item.get('scope') != PROBE079['scope'] or item.get('configuration') != PROBE079
                or item.get('status') != 'COMPLETED' or item.get('scene_ids') != roster
                or item.get('optimizer_updates') != 0 or item.get('test_opened') is not False
                or item.get('reference_read') is not False
                or item.get('validation_observations_read') is not False
                or item.get('threshold_selected') is not False or item.get('groups_recomputed') is not False):
            raise ValueError('080 source079 completed population/zero-training contract differs')
    if (run.get('source_074d') != str(source_074d) or run.get('phase') != 'completed'
            or run.get('retries') != 0 or run.get('old066_acceptance') != 'REJECTED'
            or summary.get('frozen_weights_unchanged') is not True or summary.get('model_training') is not False
            or binding.get('source') != str(source_074d)
            or binding.get('checkpoint') != str(source_074d/'fit/final.pt')
            or binding.get('endpoint_steps') != 1000 or binding.get('source_final_bound') is not True
            or binding.get('fit_exposure_verified') is not True or binding.get('normalizer_refit') is not False
            or binding.get('validation_arrays_read') is not False):
        raise ValueError('080 source079 frozen final binding differs')
    if (execution.get('configuration') != PROBE079
            or execution.get('selected_training_scene_ids') != roster
            or execution.get('arguments', {}).get('source_074d') != str(source_074d)
            or execution.get('normalizer_refit') is not False or execution.get('training') is not False
            or any(execution.get(key) is not False for key in ('reference_read', 'validation_observations_read', 'test_opened'))
            or audit.get('scene_ids') != roster or any(audit.get(key) is not False for key in ('reference_read', 'validation_read', 'test_opened'))):
        raise ValueError('080 source079 execution/read audit differs')
    reads = audit['archive_members_read']
    if (len(reads) != 2*len(roster)
            or sorted((str(row['scene_id']), row['kind']) for row in reads)
            != sorted((sid, kind) for sid in roster for kind in ('labels', 'measurements'))):
        raise ValueError('080 source079 must have exactly labels/measurements for32 T scenes')
    expected_files = {task.lower()+suffix for task in ('ITSA', 'MSTSA')
                      for suffix in ('.npz', '-views.json.gz')}
    if {p.name for p in (source/'scores').iterdir()} != expected_files:
        raise ValueError('080 source079 full score/view inventory required')
    saved = {}
    for task in ('ITSA', 'MSTSA'):
        with gzip.open(source/'scores'/f'{task.lower()}-views.json.gz', 'rt') as handle:
            import json
            rows = json.load(handle)
        if any(str(row['scene_id']) not in roster or
               str(row['component_id']) != str(manifest['components'][str(row['scene_id'])]) for row in rows):
            raise ValueError('080 source079 view scene/component differs')
        with np.load(source/'scores'/f'{task.lower()}.npz', allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        saved[task] = dict(rows=rows, arrays=arrays)
    return summary, saved, dict(source_079=str(source), source_074d=str(source_074d),
        scene_ids=roster, source_status='COMPLETED', checkpoints_opened=False,
        archive_members_opened=0, source_read_audit_verified=True,
        limitation='JSON/NPZ consistency; no independent checkpoint or source-observation replay')


def render_report(summary):
    lines = ['# 080：TSA缓存人口对照', '',
        '仅使用074D既有V评分与079既有T评分；零GPU、训练、前向或压缩卷读取。',
        '固定原阈值，不重新关联/分组。T/V为相同抽样规则，不是逐样本配对，也非独立身份泛化。',
        '固定先验0.5重加权AP只调整同一分数总体的类别权重，不能当部署AP或校准。', '',
        '完整数字、逐场/组件与079平均margin分解见summary.json；抽样原对应见selected-views.json.gz。',
        '判读须区分：同一V的平衡/完整人口差异、类先验的代数影响，以及同规则T/V仍有的组成/共享身份/设备差异。',
        '079裁窗使用正例上下文，不能把裁窗分数直接当可部署新模型。', '',
        '本项不会自动触发TSA重训；若无清楚的单因素假设，暂停该配方，继续078方法主线。']
    # A compact table is generated recursively only from complete metric records.
    lines += ['', '| Population | n+/n- | AUC | AP | AP at prior .5 | FPR at original threshold |',
              '|---|---:|---:|---:|---:|---:|']
    def fmt(x):
        return 'null' if x is None else f'{x:.6g}' if isinstance(x, (float, int)) else str(x)
    def visit(value, path):
        if not isinstance(value, dict):
            return
        if all(k in value for k in ('positive_count', 'negative_count', 'roc_auc', 'average_precision')):
            # Include pooled/top-level results; detailed scene/component tables stay in JSON.
            if not any(k in path for k in ('scenes', 'components', 'scene_macro', 'component_macro', 'sampling')):
                standardized = value.get('average_precision_prior05', value.get('ap_prior_05'))
                lines.append('| '+' / '.join(path)+' | '+f"{value['positive_count']}/{value['negative_count']}"+
                    ' | '+ ' | '.join(fmt(x) for x in (value['roc_auc'], value['average_precision'], standardized,
                                                       value.get('false_positive_rate')))+' |')
            return
        for key, item in value.items():
            visit(item, [*path, key])
    visit(summary, [])
    return '\n'.join(lines)+'\n'


def run(args):
    source_074d, source_079, output = [Path(value).resolve() for value in
        (args.source_074d, args.source_079, args.output)]
    for source in (source_074d, source_079):
        if output == source or output in source.parents or source in output.parents:
            raise ValueError('080 output overlaps immutable source')
    if output.exists():
        raise FileExistsError('080 output exists; no retry/overwrite')
    if args.cpu_threads != 4:
        raise ValueError('080 requires CPU4')
    config, manifest = read_json(args.config), read_json(args.manifest)
    production_contract(config, manifest, source_074d, source_079)
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope=SCOPE, status='RUNNING', phase='bind_cached_sources', configuration=config,
        source_074d=str(source_074d), source_079=str(source_079), model_forwards=0, optimizer_updates=0,
        checkpoints_opened=False, raw_archives_read=False, threshold_reselected=False, groups_recomputed=False,
        test_opened=False, reference_read=False, retries=0, old066_acceptance='REJECTED', advisory_reached=False)
    write_json(output/'run.json', state)
    def check():
        if time.monotonic()-started >= config['advisory_seconds'] and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output/'run.json', state)
            print('ADVISORY: fixed080 cache analysis continues, no additional samples or retries', flush=True)
    try:
        records, threshold, _, _, source_audit = bind_source(source_074d, check)
        old_summary, saved, probe_audit = bind_probe(source_079, source_074d, manifest)
        write_json(output/'source-verification.json', dict(source074D=source_audit, source079=probe_audit))
        state.update(phase='analyze_fixed_populations', frozen_threshold=threshold)
        write_json(output/'run.json', state)
        check()
        summary, selections = analyze_population(records, manifest, old_summary, saved, threshold, config)
        summary.update(status='COMPLETED', scope=SCOPE, configuration=config,
            threshold_reselected=False, model_forwards=0, optimizer_updates=0, test_opened=False,
            checkpoints_opened=False, raw_archives_read=False)
        check()
        write_json(output/'summary.json', summary)
        write_json_gzip(output/'selected-views.json.gz', selections)
        (output/'report.md').write_text(render_report(summary))
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return summary
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise
    finally:
        write_json(output/'execution.json', dict(arguments=vars(args), configuration=config,
            model_forwards=0, checkpoints_opened=False, raw_archives_read=False, test_opened=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-074d', 'source-079', 'manifest', 'config', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args(argv)
    def cancel(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('SIGTERM: preserve080 without retry')
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
