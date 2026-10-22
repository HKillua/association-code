"""069: read 068 cached endpoints, training records and two reference excerpts.

No models, checkpoints, forward passes, original archives or test payloads.
The fixed source population is validated before any interpretation is produced.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source-068', required=True)
    result.add_argument('--output', required=True)
    result.add_argument('--cpu-threads', type=int, default=4)
    return result


if __name__ == '__main__':
    _args = parser().parse_args()
    if _args.cpu_threads != 4:
        raise SystemExit('069 fixes CPU budget to four threads')
    for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_key] = '4'

import numpy as np

from association.evaluation.support_gate_objectives import load_scene, summarize, verify_source_summaries
from association.evaluation.gate_reference_audit import audit_reference
from association.evaluation.gate_training_audit import audit_training

ROOT = Path(__file__).resolve().parents[2]
TASK = 'LEVEL1-SUPPORT-GATE-20261023-068-attempt-1'
METHODS = ['anchor', 'mean_supported', 'attribute_hybrid', 'support_switch', 'learned_gate']


def read(path):
    path = Path(path)
    if not path.exists():
        path = path.with_suffix(path.suffix + '.gz')
    with (gzip.open(path, 'rt', encoding='utf-8') if path.suffix == '.gz'
          else path.open(encoding='utf-8')) as stream:
        return json.load(stream)


def write(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, default=convert,
                         indent=None if path.suffix == '.gz' else 2) + '\n'
    if path.suffix == '.gz':
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            stream.write(payload)
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)


def bind_source(source):
    """Bind existing records and inventory, never deserialize a checkpoint."""
    if source.name != TASK:
        raise ValueError('069 requires the original068 attempt-1 directory')
    config = read(ROOT/'configs/support-gate-level1.json')
    manifest = read(ROOT/'configs/data/mtad-level1-scene3000.json')
    ids = manifest['splits']['V_select']
    components = {sid: str(manifest['components'][sid]) for sid in ids}
    run, summary = read(source/'run.json'), read(source/'summary.json')
    for record in (run, summary):
        if (record.get('status') != 'COMPLETED'
                or record.get('scope') != '068-support-prior-position-pilot'
                or record.get('configuration') != config or record.get('optimizer_updates') != 1000
                or record.get('test_opened') is not False or record.get('inference_device') != 'cpu'):
            raise ValueError('068 completed fixed-1000 CPU endpoint protocol required')
    parent = Path(run.get('source_system', ''))
    if (summary.get('source_system') != str(parent) or parent.name != 'system'
            or parent.parent.name != 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
            or run.get('completed_endpoints') != ['initial', 'final']
            or run.get('old066_acceptance') != 'REJECTED' or run.get('retries') != 0
            or any(run.get(key) is not False for key in ('association_recomputed',
                'threshold_reselected', 'validation_groups_recomputed', 'validation_selection'))):
        raise ValueError('068 immutable056 provenance/selection boundary differs')
    preflight = read(source/'preflight/report.json')
    if (preflight.get('status') != 'PASS' or preflight.get('requested_scene_ids') != ['3035', '2535']
            or preflight.get('training_gate_passed') is not True
            or preflight.get('data_contract_reproduced') is not True
            or preflight.get('anomalies_removed') is not False):
        raise ValueError('068 original two-scene preflight required')
    endpoints = {}
    expected_inventory = {f'states-{sid}.npz' for sid in ids}
    for label, step, filename in (('initial', 0, 'step-0.pt'), ('final', 1000, 'final.pt')):
        endpoint = source/label
        report, record = read(endpoint/'summary.json'), read(endpoint/'run.json')
        checkpoint = dict(path=str(source/'fit'/filename), step=step, actual_updates=step,
            configured_final_updates=1000, routing_semantics=config['routing_semantics'])
        for item in (report, record):
            if (item.get('status') != 'COMPLETED' or item.get('device') != 'cpu'
                    or item.get('source_system') != str(parent) or item.get('scene_ids') != ids
                    or item.get('checkpoint') != checkpoint or item.get('test_opened') is not False
                    or item.get('optimizer_updates') != 0
                    or any(item.get(key) is not False for key in
                           ('groups_recomputed', 'scores_recomputed', 'threshold_reselected'))):
                raise ValueError('068 endpoint provenance differs: ' + label)
        if (report.get('method_names') != METHODS or report.get('scene_count') != 300
                or report.get('component_count') != 38 or report.get('fragment_count') != 24599
                or report.get('dataset_level') != 'Level1' or report.get('split') != 'V_select'
                or report.get('ordinary_source_replay_exact') is not True
                or report.get('fixed_population_verified') is not True
                or record.get('completed_scenes') != 300 or record.get('total_scenes') != 300
                or {sid: cid for cid, part in report['components'].items() for sid in part['scene_ids']} != components):
            raise ValueError('068 endpoint full V300 population differs: ' + label)
        for directory in ('predictions', 'system-predictions'):
            if {path.name for path in (endpoint/directory).glob('*.npz')} != expected_inventory:
                raise ValueError('068 missing/extra cached scenes: ' + str(endpoint/directory))
        endpoints[label] = report
    fit = read(source/'fit/run.json')
    if (fit.get('configuration') != config or fit.get('training_scene_ids') != manifest['splits']['T_core']
            or fit.get('test_opened') is not False):
        raise ValueError('068 training config/T2100 registry differs')
    return dict(scene_ids=ids, components=components, endpoints=endpoints, source_run=run)


def render_report(objectives, training, reference):
    """Filled from verified arrays only, with no automatic model recommendation."""
    lines = ['# 069：068缓存目标与参考段核对', '',
        '本轮仅复算已保存输出，未训练、前向、打开checkpoint或原数据卷。068负结果和066 REJECTED不变。', '',
        '## 同人口的目标与米误差', '',
        'log目标为log1p(米误差/1000)。scene-target平均不等于随机anchor与人工组的训练目标；共享身份组件不是独立重复。', '',
        '| 方法 | 点数 | 点等权log目标 | MAE(m) | RMSE(m) |',
        '| --- | ---: | ---: | ---: | ---: |']
    methods = objectives['regions']['all']['methods']
    for name, metric in methods.items():
        value = metric['point_equal']
        lines.append(f"| {name} | {value['unit_count']} | {value['log1p']:.8f} | {value['mae']:.6f} | {value['rmse']:.6f} |")
    lines.extend(['', '| final−initial加权口径 | log目标变化 | MAE变化(m) | RMSE变化(m) |',
                  '| --- | ---: | ---: | ---: |'])
    for name, value in objectives['regions']['all']['paired']['final__minus__initial'].items():
        lines.append(f"| {name} | {value['log1p']['mean_difference']:.8f} | {value['mae']['mean_difference']:.6f} | {value['rmse_difference']:.6f} |")
    lines.extend(['', '负值表示下降。四口径、共同支撑分层、逐场和38组件详见objective-summary.json.gz与scene-summary.json.gz。', '',
        '## 训练记录', '',
        f"成功更新 {training['logged_successful_updates']}，实际访问 {training['sampled_training_scene_count']}/{training['declared_training_scene_count']} 个训练场景、{training['unique_sampled_fragment_count']} 个不同片段；日志曝光与原fit/run逐项一致。", '',
        '| 模式 | 更新数 | 平均log loss | 梯度norm中位数（clip前） | norm大于clip的更新数 |',
        '| --- | ---: | ---: | ---: | ---: |'])
    for mode, item in training['overall'].items():
        lines.append(f"| {mode} | {item['log_loss']['count']} | {item['log_loss']['mean']:.8f} | {item['preclip_gradient_norm']['p50']:.6f} | {item['preclip_norm_greater_than_configured_clip_count']} |")
    lines.extend(['',
        'training-audit.json记录成功更新的曝光闭合、三模式和固定100步窗口。loss窗口使用不同随机batch，不构成独立收敛证明；clip前梯度norm不等于实际更新幅度。未保存线性train loss，不能从平均log loss反推。', '',
        '## 固定reference段及canonical输入', '',
        '| 场景／目标 | 括号内有效Q数 | 对全人口final−initial MAE的贡献(m) | 对全人口final−规则 MAE的贡献(m) |',
        '| --- | ---: | ---: | ---: |'])
    for scene in reference['scenes']:
        value = scene['named_bracket_union_contributions']
        lines.append(f"| {scene['scene_id']}/{scene['target_id']} | {scene['named_bracket_valid_position_points']} | {value['initial']['contribution_to_full_population_mae_change_m']:.6f} | {value['support_switch']['contribution_to_full_population_mae_change_m']:.6f} |")
    lines.extend(['', '| 固定四括号并集 | 净差(m累计) | 总改善(m累计) | 总伤害(m累计) | 平方误差变化(m²累计) | 补集MAE差(m) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for base, value in reference['named_bracket_union_contributions'].items():
        lines.append(f"| final−{base} | {value['signed_error_change_sum_m']:.6f} | {value['gross_improvement_sum_m']:.6f} | {value['gross_harm_sum_m']:.6f} | {value['signed_squared_error_change_sum_m2']:.6f} | {value['complement_sensitivity']['mae_change_m']:.6f} |")
    lines.extend(['',
        'reference-audit.json.gz记录两场原始相邻reference段对应的Q、完整truth重建、固定节点canonical序列与误差贡献。主指标保留全部点；补集只是敏感性说明，不是清洗后成绩。四个固定括号并未穷尽所有可疑段：没有解释全部差异，也不能反推其它reference正确。几何跳变证据不自动判定哪一来源错误，也不能代表全库异常率。', '',
        '## 下一步判读', '',
        '- 若log目标改善、米误差变差，支持目标权衡假设；仍需排除参考段集中影响，不能证明loss是唯一原因。',
        '- 若log目标也变差，先检查训练组与部署组的差异；训练随机batch曲线不能替代这一判断。',
        '- 若已知reference段主导误差变化，先做预定T_core小样本源语义核查，不从V选择删除规则。',
        '- 本报告不选新模型、不改loss、不授权新训练。两场特选案例、反复使用V与单seed限制保留。', ''])
    return '\n'.join(lines)


def run(args):
    source, output = Path(args.source_068).resolve(), Path(args.output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('069 output must be separate from immutable source068')
    if args.cpu_threads != 4:
        raise ValueError('069 fixes CPU threads to4')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    state = dict(scope='069-support-gate-cache-diagnostics', status='RUNNING', source_068=str(source),
        cpu_threads=4, gpu_used=False, checkpoint_loaded=False, model_forwards=0, optimizer_updates=0,
        raw_archives_opened=False, test_opened=False, groups_recomputed=False, threshold_reselected=False,
        old066_acceptance='REJECTED', retries=0, completed_scenes=0, advisory_reached=False)
    write(output/'run.json', state)
    try:
        binding = bind_source(source)
        training = audit_training(source/'fit', expected_updates=1000)
        write(output/'training-audit.json', training)
        rows = []
        for sid in binding['scene_ids']:
            rows.append(load_scene(source/'initial', source/'final', sid))
            state.update(completed_scenes=len(rows), last_scene_id=sid,
                         elapsed_seconds=time.monotonic()-start)
            if state['elapsed_seconds'] >= 3600 and not state['advisory_reached']:
                state['advisory_reached'] = True
                print('ADVISORY: one hour exceeded; continuing this single read-only attempt', flush=True)
            write(output/'run.json', state)
            print(f'069 cached scene {len(rows)}/300: {sid}', flush=True)
        verify_source_summaries(rows, binding['endpoints']['initial'], binding['endpoints']['final'])
        objectives = summarize(rows, binding['components'])
        reference = audit_reference(source, rows)
        scene_rows = objectives.pop('per_scene')
        write(output/'scene-summary.json.gz', scene_rows)
        write(output/'objective-summary.json.gz', objectives)
        write(output/'reference-audit.json.gz', reference)
        (output/'report.md').write_text(render_report(objectives, training, reference), encoding='utf-8')
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-start,
                     source_metrics_reproduced=True, scientific_status='development diagnostics; no method promotion')
        write(output/'summary.json', dict(state, objective_report='objective-summary.json.gz',
            training_report='training-audit.json', reference_report='reference-audit.json.gz',
            scene_report='scene-summary.json.gz'))
        write(output/'run.json', state)
        return state
    except (Exception, KeyboardInterrupt) as error:
        state.update(status='INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
            error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-start)
        write(output/'run.json', state)
        raise


def main(args):
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'cancelled by signal {signum}')
    previous = signal.signal(signal.SIGTERM, cancel)
    try:
        result = run(args)
        print(f"{result['status']}: {Path(args.output).resolve()}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main(_args))
