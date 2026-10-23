"""075: fixed074D scores and successful training logs, CPU only, no new scores.

No checkpoint deserialization, archive access, model inference, grouping or
threshold selection. Source reports are checked against actual cached arrays.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-074d', required=True)
    result.add_argument('--output', required=True)
    result.add_argument('--cpu-threads', type=int, default=4)
    return result


if __name__ == '__main__':
    _args = parser().parse_args()
    if _args.cpu_threads != 4:
        raise SystemExit('075 fixes CPU4')
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_name] = '4'

import numpy as np
import torch

from association.evaluation.association import evaluate_scenes
from association.evaluation.tsactfer_level1 import scope_metrics
from association.evaluation.tsactfer_score_diagnostics import analyze_scores
from association.evaluation.tsactfer_training_diagnostics import analyze_training
from association.utils.runtime import read_json, write_json, write_json_gzip, load_records

ROOT = Path(__file__).resolve().parents[2]
SOURCE_TASK = 'LEVEL1-TSA-NATIVE-20261023-074D-attempt-1'
SCOPE = '075-tsa-frozen-score-diagnostics-v1'
ADVISORY_SECONDS = 3600


def assert_same(actual, expected, label):
    """Freeze integer populations exactly; float aggregation tolerance 1e-12."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(label+' report keys differ')
        for key in expected:
            assert_same(actual[key], expected[key], label+'/'+str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(label+' report population differs')
        for i, (left, right) in enumerate(zip(actual, expected)):
            assert_same(left, right, label+'/'+str(i))
    elif isinstance(expected, float):
        if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                or not np.isfinite(actual) or not np.isclose(actual, expected, rtol=0., atol=1e-12)):
            raise ValueError(label+' numeric report differs')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(label+' value differs')


def bind_source(source, check=None):
    """Read only known074D score/log reports, never the named checkpoint file."""
    source = Path(source).resolve()
    if source.name != SOURCE_TASK:
        raise ValueError('075 requires the completed074D attempt-1 directory')
    config = read_json(ROOT/'configs/tsactfer-level1.json')
    manifest = read_json(ROOT/'configs/data/mtad-level1-scene3000.json')
    ids, train_ids = manifest['splits']['V_select'], manifest['splits']['T_core']
    run, summary = read_json(source/'run.json'), read_json(source/'summary.json')
    endpoint = read_json(source/'final-endpoint.json')
    steps = config['training']['updates']
    for record in (run, summary):
        if (record.get('status') != 'COMPLETED' or record.get('scope') != config['scope']
                or record.get('configuration') != config or record.get('optimizer_updates') != steps
                or record.get('test_opened') is not False or record.get('endpoint') != 'final'):
            raise ValueError('075 requires completed074D fixed final protocol')
    if (run.get('phase') != 'completed' or run.get('checkpoint_selection') is not False
            or run.get('old066_acceptance') != 'REJECTED' or run.get('retries') != 0
            or run.get('inference_device') != 'cpu'
            or summary.get('source_system') != run.get('source_system')
            or summary.get('final_endpoint') != str(source/'final-endpoint.json')
            or summary.get('system_summary') != str(source/'system/summary.json')):
        raise ValueError('075 source endpoint paths/selection/history differ')
    if (endpoint.get('status') != 'COMPLETED' or endpoint.get('configuration') != config
            or endpoint.get('actual_updates') != steps or endpoint.get('step') != steps
            or endpoint.get('checkpoint') != str(source/'fit/final.pt')
            or endpoint.get('inference_device') != 'cpu' or endpoint.get('test_opened') is not False
            or endpoint.get('scores_from_actual_final_checkpoint') is not True
            or endpoint.get('scene_ids') != ids or summary.get('selection') != endpoint.get('selection')):
        raise ValueError('075 final endpoint population/configuration differs')
    normalizer = endpoint['normalizer']
    if (normalizer.get('training_scene_ids') != train_ids
            or normalizer.get('training_splits') != ['T_core']
            or set(train_ids) & set(ids)):
        raise ValueError('075 normalization must be frozen T_core only')
    fit = read_json(source/'fit/run.json')
    if (fit.get('normalizer') != normalizer or fit.get('configuration') != config
            or fit.get('optimizer_updates') != steps
            or summary.get('nonzero_lr_updates') != fit['exposure']['nonzero_lr_updates']
            or summary.get('task_calls') != fit['exposure']['task_calls']):
        raise ValueError('075 fit/final normalization or successful calls differ')
    selection = endpoint['selection']
    binding = dict(scope=config['scope'], split='V_select', seed=config['seed'], endpoint='final',
        step=steps, checkpoint=str(source/'fit/final.pt'), configuration=config,
        normalizer=normalizer, selection=selection)
    index = read_json(source/'final-scores/index.json')
    if (index.get('schema') != 1 or index.get('binding') != binding
            or endpoint.get('score_binding') != binding):
        raise ValueError('075 final score binding differs')
    entries = index['records']
    files = [item['file'] for item in entries]
    if (len(set(files)) != len(files)
            or {p.name for p in (source/'final-scores').glob('*.npz')} != set(files)
            or [str(item['scene_id']) for item in entries] != ids):
        raise ValueError('075 complete score inventory/order required')
    records = load_records(source/'final-scores', binding)
    for row, sid in zip(records, ids):
        if check:
            check()
        n = len(row['node_ids'])
        if (str(row['component_id']) != str(manifest['components'][sid])
                or not np.array_equal(row['pairs'], np.column_stack(np.triu_indices(n, 1)))):
            raise ValueError('075 original all-pair/component population differs')
    threshold = selection['threshold']
    actual = evaluate_scenes(records, threshold)
    assert_same(actual, endpoint['metrics'], '075 endpoint edge metrics')
    assert_same(actual, summary['association'], '075 run edge metrics')
    # Selection is NOT rerun. Confirm only the score at the already fixed point.
    assert_same(actual['macro_f1'], selection['macro_f1'], '075 frozen selection score')
    system_run = read_json(source/'system/run.json')
    system = read_json(source/'system/summary.json')
    if (system_run.get('status') != 'COMPLETED' or system.get('status') != 'COMPLETED'
            or system.get('scene_ids') != ids or system.get('scene_count') != len(ids)
            or system_run.get('completed_scenes') != len(ids)
            or system_run.get('total_scenes') != len(ids) or system.get('endpoint') != 'final'
            or system.get('test_opened') is not False or system_run.get('test_opened') is not False
            or system_run.get('threshold_reselected') is not False or system.get('threshold') != threshold):
        raise ValueError('075 completed full-system source differs')
    assert_same(actual, system['association']['tsa']['metrics'], '075 system edge metrics')
    qualified = scope_metrics(records, threshold)
    assert_same(qualified, system['association']['tsa']['paper_scope'], '075 paper time eligibility')
    fixed_scopes = {'all': actual, 'itsa_all': actual['itsa'], 'mstsa_all': actual['mstsa'],
                    'itsa_eligible': qualified['itsa']['metrics'],
                    'mstsa_eligible': qualified['mstsa']['metrics']}
    verification = dict(source_run=str(source), configuration=config, scene_count=len(ids),
        node_count=sum(len(r['node_ids']) for r in records), pair_count=actual['pair_count'],
        component_count=len({r['component_id'] for r in records}), threshold=threshold,
        source_fixed_final_steps=steps, source_metrics_reproduced=True,
        source_time_eligibility_reproduced=True, normalizer_T_only=True,
        fixed_scope_metrics={name: {key: row[key] for key in
            ('pair_count', 'tp', 'fp', 'fn', 'tn', 'macro_f1')} for name, row in fixed_scopes.items()},
        checkpoints_opened=False, source_status='COMPLETED',
        limitation='JSON/NPZ/log consistency only; checkpoint bytes deliberately not reloaded')
    return records, threshold, config, train_ids, verification


def render_report(scores, training, verified):
    """Concise interpretation guide; all numerical tables come from cached data."""
    lines = ['# 075：074D固定评分与训练曝光诊断', '',
        '固定final与原阈值，只读评分/日志；零GPU、零模型前向、零训练。074D原成绩不变。', '',
        f"完整 {verified['scene_count']} 场、{verified['node_count']} 片段、{verified['pair_count']} 配对；阈值 {verified['threshold']}。",
        '原全量边指标与时间资格子集已重算一致。未加载权重，不能把缓存一致性写成独立模型重放。', '',
        '分数不是概率。ROC-AUC/AP描述排序，AP依赖正例比例；不同头/资格的AP不能脱离类别比例直接排名。',
        '逐场与共享身份组件的汇总分别报告定义数量，不把所有配对当独立样本。', '',
        '## 排序与类别比例', '',
        '| 范围 | 配对数 | 正例数 | 正例比例 | pooled AUC | pooled AP | AP/正例比例 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(value):
        return 'null' if value is None else f'{value:.6g}'
    for scope, metric in scores['pooled'].items():
        lines.append(f"| {scope} | {metric['pair_count']} | {metric['positive_count']} | "
            f"{fmt(metric['prevalence'])} | {fmt(metric['roc_auc'])} | "
            f"{fmt(metric['average_precision'])} | {fmt(metric['ap_over_prevalence'])} |")
    lines.extend(['', 'eligible/ineligible均含时间资格；itsa_all/mstsa_all只按同源/异源，不能互换。', '',
        '| 范围 | 逐场AUC均值（定义场数） | 逐场F1 | FP | FN | FPR=FP/负例 | FDR=FP/接受数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for scope, metric in scores['pooled'].items():
        macro = scores['scene_macro'][scope]
        auc = macro['roc_auc']
        lines.append(f"| {scope} | {fmt(auc['mean'])} ({auc['defined_unit_count']}) | "
            f"{fmt(macro['f1']['mean'])} | {metric['fp']} | {metric['fn']} | "
            f"{fmt(metric['false_positive_rate'])} | {fmt(metric['false_discovery_rate'])} |")
    lines.extend(['', '| 范围 | 正分数P05 / P50 | 负分数P50 / P95 |',
        '| --- | ---: | ---: |'])
    for scope, metric in scores['pooled'].items():
        positive, negative = metric['scores']['positive'], metric['scores']['negative']
        lines.append(f"| {scope} | {fmt(positive['p05'])} / {fmt(positive['p50'])} | "
            f"{fmt(negative['p50'])} / {fmt(negative['p95'])} |")
    lines.extend(['', '## 实际训练曝光', '',
        '| 头 | 调用数 | 首/末100调用loss | 同场负比较占比 | 裁剪调用占比 | 裁窗token保留率 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for head, record in training['heads'].items():
        metric, windows = record['summary'], record['windows']
        lines.append(f"| {head} | {metric['successful_optimizer_calls']} | "
            f"{fmt(windows[0]['loss']['mean'])} / {fmt(windows[-1]['loss']['mean'])} | "
            f"{fmt(metric['negative_inventory']['same_scene_fraction'])} | "
            f"{fmt(metric['clipped_call_fraction'])} | {fmt(metric['crop']['token_retention_fraction'])} |")
    lines.extend(['', '同场负比较以有向InfoNCE分母计数；token保留率相对于已cap128的裁窗前输入，不是原始全部观测。', '',
        '## 误报最多的场景（前五，完整前二十见JSON）', '',
        '| 场景 | 片段数 | FP | 全部FP份额 |', '| --- | ---: | ---: | ---: |'])
    for row in scores['top_fp_scenes'][:5]:
        lines.append(f"| {row['scene_id']} | {row['node_count']} | {row['fp']} | {fmt(row['fp_share'])} |")
    lines.extend(['', '## 阅读产物', '',
        '- `score-summary.json.gz`：分头／资格内外的分数分布、排序、原阈值错误、组件和场景规模分层。',
        '- `scene-summary.json.gz`：逐场同口径结果与定义分母。',
        '- `training-audit.json`：两头真实loss／学习率／裁剪、同场与跨场有向负比较、曝光闭合。',
        '- `source-verification.json`：完整final来源、人口及原指标复现。', '',
        '## 判读边界', '',
        '若头内排序较好而原阈值误报高，分数标度/工作点是值得再查的解释；不等于已证明换阈值能改善最终分组。',
        '若同场排序也差，仅调整阈值无法创造排序信息；下一步再区分训练充分性与输入裁窗失配。',
        '两类问题可能共存。本轮不选择新阈值、不生成新分组或新的完整系统排名，不自动给出方法PASS/FAIL。',
        '同场负比较占比不等于地理或运动困难度；训练loss下降和梯度裁剪不是机制因果证明。',
        'V反复用于开发、单seed及共享身份限制仍在；T_test封存，066保持REJECTED。', ''])
    return '\n'.join(lines)


def run(args):
    source, output = Path(args.source_074d).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('075 output cannot overlap immutable074D')
    if output.exists():
        raise FileExistsError('075 output exists; no overwrite or automatic retry')
    if args.cpu_threads != 4:
        raise ValueError('075 fixes CPU4')
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(scope=SCOPE, status='RUNNING', source_run=str(source), model_forwards=0,
        optimizer_updates=0, checkpoints_opened=False, raw_archives_read=False,
        threshold_reselected=False, groups_recomputed=False, test_opened=False,
        retries=0, advisory_reached=False, old066_acceptance='REJECTED')
    write_json(output/'run.json', state)
    def check():
        if time.monotonic()-started >= ADVISORY_SECONDS and not state['advisory_reached']:
            state['advisory_reached'] = True
            write_json(output/'run.json', state)
            print('ADVISORY: continue this one fixed CPU audit, no new work', flush=True)
    def phase(name):
        state.update(phase=name, elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        print('075 phase: '+name, flush=True)
        check()
    try:
        phase('bind_completed074D_and_replay_fixed_metrics')
        records, threshold, config, train_ids, verified = bind_source(source, check)
        write_json(output/'source-verification.json', verified)
        phase('fixed_score_rank_and_error_diagnostics')
        scores = analyze_scores(records, threshold, check)
        for scope, expected in verified['fixed_scope_metrics'].items():
            for key in ('pair_count', 'tp', 'fp', 'fn', 'tn'):
                assert_same(scores['pooled'][scope][key], expected[key], '075 diagnostic '+scope+'/'+key)
            assert_same(scores['scene_macro'][scope]['f1']['mean'], expected['macro_f1'],
                        '075 diagnostic '+scope+'/macro_f1')
        phase('successful_training_log_exposure')
        training = analyze_training(source/'fit', config, train_ids, check)
        write_json(output/'training-audit.json', training)
        write_json_gzip(output/'scene-summary.json.gz', scores['scenes'])
        write_json_gzip(output/'score-summary.json.gz', {k:v for k,v in scores.items() if k != 'scenes'})
        (output/'report.md').write_text(render_report(scores, training, verified), encoding='utf-8')
        state.update(status='COMPLETED', phase='completed', elapsed_seconds=time.monotonic()-started,
            completed_scenes=verified['scene_count'], pair_count=verified['pair_count'])
        write_json(output/'run.json', state)
        return state
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    args = parser().parse_args(argv)
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('SIGTERM: preserve075 partial output; no retry')
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        print(run(args)['status'])
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
