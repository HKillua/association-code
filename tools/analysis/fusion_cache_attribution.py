"""067: inspect frozen 056/057/066 arrays; never load models or raw archives."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('source-066', 'source-system', 'source-fusion', 'manifest', 'output'):
        result.add_argument('--'+name, required=True)
    result.add_argument('--cpu-threads', type=int, default=4)
    return result


# Set library thread budgets before NumPy/runtime imports on the real CLI path.
if __name__ == '__main__':
    _cli_args = parser().parse_args()
    if _cli_args.cpu_threads < 1:
        raise SystemExit('positive CPU budget required')
    for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_name] = str(_cli_args.cpu_threads)

import numpy as np

from association.evaluation.fusion_cache_audit import audit_source, aggregate as aggregate_numeric
from association.evaluation.fusion_opportunity import analyze_scene, aggregate as aggregate_opportunities
from association.evaluation.frozen_fusion_transfer import METHODS, LEARNED_METHODS, summarize_scene
from association.evaluation.system_attribution import _same, assert_metrics, pool_metrics
from association.utils.runtime import read_json_or_gzip as read, write_json, write_json_gzip

TASK066 = 'LEVEL1-FROZEN-FUSION-20261022-066-attempt-1'
TASK056 = 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
TASK057 = 'LEVEL1-FUSION-20261022-057-attempt-1'
ORDINARY = METHODS[:3]


def load_arrays(root, directory, sid):
    with np.load(root/directory/f'states-{sid}.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prefixed(data, prefix):
    return {key[len(prefix):]: value for key, value in data.items() if key.startswith(prefix)}


def exact_subset(current, original, label):
    if not set(original) <= set(current):
        raise ValueError(label + ': original fields missing from cached input')
    for key, value in original.items():
        _same(current[key], value, label + '/' + key)


def _inventory(root, ids):
    expected = {f'states-{sid}.npz' for sid in ids}
    for directory in ('predictions', 'system-predictions'):
        if {p.name for p in (root/directory).glob('*.npz')} != expected:
            raise ValueError('missing or extra V cached scenes: ' + str(root/directory))


def bind_sources(source, system, fusion, manifest):
    """Bind completed caches via records and arrays, without checkpoint access."""
    if source.name != TASK066 or system.name != 'system' or system.parent.name != TASK056 or fusion.name != TASK057:
        raise ValueError('067 requires original 056/057/066 attempt-1 cache directories')
    ids = manifest.get('splits', {}).get('V_select', [])
    train = manifest.get('splits', {}).get('T_core', [])
    test = manifest.get('splits', {}).get('T_test', [])
    if (manifest.get('dataset_level') != 'Level1' or manifest.get('split_unit') != 'scene'
            or (len(train), len(ids), len(test)) != (2100, 300, 600)
            or any(not isinstance(sid, str) or not sid.isdigit() for sid in ids)
            or len(set(train + ids + test)) != 3000):
        raise ValueError('fixed disjoint Level1 T2100/V300/test600 manifest required; test metadata only')
    run, summary, audit = read(source/'run.json'), read(source/'unvalidated-summary.json'), read(source/'replay-audit.json')
    if (source/'summary.json').exists():
        raise ValueError('066 source must retain rejected, unvalidated status')
    for record in (run, summary):
        if (record.get('status') != 'FAILED' or record.get('acceptance') != 'REJECTED'
                or record.get('analysis_complete') is not True or record.get('dataset_level') != 'Level1'
                or record.get('split') != 'V_select' or record.get('test_opened') is not False
                or record.get('utility_replay_domain') != 'output'
                or Path(record.get('source_system', '')).resolve() != system
                or Path(record.get('source_fusion', '')).resolve() != fusion):
            raise ValueError('066 completed rejected cache/source binding required')
    if (run.get('completed_scenes') != 300 or run.get('optimizer_updates') != 0
            or run.get('gpu_used') is not False or summary.get('scene_ids') != ids
            or summary.get('scene_count') != 300 or summary.get('fragment_count') != 24599
            or summary.get('component_count') != 38 or summary.get('association_endpoint_step') != 1000
            or summary.get('ordinary_source_replay_exact') is not True
            or summary.get('support_control_analytical_verified') is not True
            or audit.get('total_replays') != 900 or audit.get('rejected_replays') != 47
            or audit != summary.get('source_replay_audit')):
        raise ValueError('066 full population or immutable audit differs')
    source056 = read(system/'run.json')
    if (source056.get('status') != 'COMPLETED' or source056.get('association_endpoint') != 'final'
            or source056.get('test_opened') is not False or source056.get('completed_scenes') != 300
            or source056.get('split') != 'V_select' or source056.get('dataset_level') != 'Level1'):
        raise ValueError('original056 completed final V300 source required')
    _inventory(source, ids)
    _inventory(system, ids)
    components, arm_systems = None, {}
    if set(summary.get('fusion_sources', {})) != set(LEARNED_METHODS):
        raise ValueError('all three original057 arms required')
    for arm in LEARNED_METHODS:
        meta = summary['fusion_sources'][arm]
        root = fusion/arm.replace('_', '-')/'system'
        record = read(root/'run.json')
        if (Path(meta.get('system', '')).resolve() != root or meta.get('step') != 1000
                or meta.get('scene_ids') != ids or meta.get('objective') != arm
                or meta.get('frozen_parameters_verified') is not True or meta.get('test_opened') is not False
                or record.get('status') != 'COMPLETED' or record.get('fusion_checkpoint_step') != 1000
                or record.get('completed_scenes') != 300 or record.get('test_opened') is not False
                or record.get('split') != 'V_select' or record.get('dataset_level') != 'Level1'
                or Path(record.get('inputs', {}).get('fusion_checkpoint', '')).resolve() != root.parent/'fit/final.pt'):
            raise ValueError('original057 cache endpoint/source differs: ' + arm)
        mapping = meta.get('component_ids', {})
        if set(mapping) != set(ids) or (components is not None and mapping != components):
            raise ValueError('057/066 component registry differs')
        components = mapping
        arm_systems[arm] = root
        _inventory(root, ids)
    if len(set(components.values())) != 38:
        raise ValueError('066 shared-component population differs')
    return dict(summary=summary, audit=audit, scene_ids=ids, components=components, arm_systems=arm_systems)


def _verify_scene(fragment, saved, original, original_system, expected, sid, component):
    exact_subset(fragment, original, '066/056 fragment')
    artifacts = {arm: prefixed(saved, arm+'_') for arm in METHODS}
    for arm in ORDINARY:
        old = prefixed(original_system, 'learned_assoc_'+arm+'_')
        if set(artifacts[arm]) != set(old):
            raise ValueError('066/056 ordinary artifact fields differ')
        exact_subset(artifacts[arm], old, '066/056 '+arm)
    for arm in METHODS:
        states = fragment['output_'+arm]
        if states.shape != fragment['native_states'].shape or not np.isfinite(states).all():
            raise ValueError('complete finite cached output required: '+arm)
        anchors = artifacts[arm]['group_node_indices'][artifacts[arm]['group_offsets'][:-1]]
        _same(states[anchors], artifacts[arm]['group_states'], '066 fragment/group '+arm)
    routes = {arm: prefixed(fragment, arm+'_routing_') for arm in LEARNED_METHODS}
    check_fragment = dict(fragment, direct_only_prediction=fragment['output_direct_only'])
    row, _ = summarize_scene(check_fragment, artifacts, routes, sid, component)
    if expected.get('scene_id') != sid or expected.get('component_id') != component:
        raise ValueError('066 per-scene metadata differs')
    for arm in METHODS:
        assert_metrics(row['methods'][arm]['full_target_metrics'],
                       expected['methods'][arm]['full_target_metrics'], '066 cached metrics '+arm)
        if row['methods'][arm]['population'] != expected['methods'][arm]['population']:
            raise ValueError('066 population summary differs')
    return artifacts, routes, row


def render_report(numeric, opportunity):
    """Evidence front page; no automatic causal verdict or model selection."""
    def fmt(value):
        return '不可识别' if value is None else f'{value:.6g}'
    lines = ['# 067 缓存归因', '',
        '**067分析完成不代表066验收通过。066保持FAILED／REJECTED；方法观察仍为开发集诊断。**', '',
        '## A：原OLS组数值影响', '',
        f"复核{numeric['replay_count']}项，复现{numeric['rejected_replay_count']}项拒绝。",
        '下表按拒绝回放项计数；映射到有效点、改变状态、改变真值误差是不同概念，均不表示工程显著性。', '',
        '| 臂 | 拒绝 | 触及最终有效点 | 状态改变 | 真值误差改变 | 速度分解超舍入界点 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm, item in numeric['by_arm'].items():
        effect = item['rejected_replays_with_unique_valid_effect']
        lines.append(f"| {arm} | {item['rejected_replay_count']} | {item['rejected_replays_with_unique_valid_mapped_violation']} | {effect['state_changed']} | {effect['truth_error_changed']} | {item['speed_algebra']['exceeds_arithmetic_bound_count']} |")
    lines.extend(['', '| 臂／属性 | 旧MAE | 当前MAE | MAE差 | RMSE差 | 状态差最大值 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |'])
    for arm, item in numeric['by_arm'].items():
        for attr, metric in item['metrics']['unique_selected_groups'].items():
            lines.append(f"| {arm}/{attr} | {fmt(metric['previous_error']['mean'])} | {fmt(metric['current_error']['mean'])} | {fmt(metric['mae_change'])} | {fmt(metric['rmse_change'])} | {fmt(metric['state_difference']['maximum'])} |")
    lines.extend(['', '误差使用原OLS组唯一采用目标的共同有效真值点；状态差覆盖所采用输出的有效预测点，两者分母不同。位置米、速度原生单位、航向度。',
        '非输出／重复组映射及P95/P99详见numeric-impact.json.gz；四类映射可在同一拒绝回放中重叠，不得加总为拒绝项总数。', ''])
    special = [dict(arm=arm, **row) for arm, item in numeric['by_arm'].items()
               for row in item['largest_speed_examples'] if row['scene_id'] == '2535']
    for row in special:
        lines.append(f"scene2535/{row['arm']}：最大速度差{fmt(row['state_difference'])}，映射={row['category']}，权重分解={fmt(row['weight_decomposition'])}，残差={fmt(row['residual'])}，累计舍入界={fmt(row['arithmetic_bound'])}。")
    lines.extend(['完整scene2535证据单列scene-2535-numeric.json.gz；缺席上述Top10不等于该场未分析。', '',
        '## B：学习组位置缺口', '',
        '唯一目标评价；全部输出组（含重复）另见JSON，不能混用分母。正缺口表示HQ误差高于简单规则。', ''])
    for region in ('all', 'anchor_outside_helper_supported'):
        item = opportunity['scopes']['unique_targets'][region]
        lines.extend([f"### {region}：{item['point_count']}点", '',
            '| 输出 | MAE米 | RMSE米 |', '| --- | ---: | ---: |'])
        for name, values in item['errors'].items():
            lines.append(f"| {name} | {fmt(values['mae'])} | {fmt(values['rmse'])} |")
        lines.extend(['', '| 类别 | 点数 | 规则改善累计m | 规则伤害累计m | HQ改善累计m | HQ伤害累计m | HQ−规则累计m | 平方误差缺口m² | 零helper质量缺口m |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for name, values in item['classes'].items():
            rd, md = values['rule_vs_anchor'], values['model_vs_anchor']
            zero = values['routing_gap_contributions']['zero_mass_with_helper']['model_minus_rule_sum_m']
            lines.append(f"| {name} | {values['point_count']} | {fmt(rd['improvement_sum_m'])} | {fmt(rd['harm_sum_m'])} | {fmt(md['improvement_sum_m'])} | {fmt(md['harm_sum_m'])} | {fmt(values['model_minus_rule']['signed_delta']['sum'])} | {fmt(values['model_minus_rule_squared_error_m2']['sum'])} | {fmt(zero)} |")
        mechanism = item['mechanism_evidence']
        lines.extend(['', f"有helper且零权重／状态不变：{mechanism['zero_helper_mass_unchanged_count']}点；存在优于Anchor的有支撑helper：{mechanism['beneficial_supported_helper_exists_count']}点；含有支撑错身份helper：{mechanism['supported_wrong_identity_present_count']}点。以上是可重叠边际数，不能相除推断归因比例。", ''])
        lines.extend(['| 非互斥证据组合 | 点数 | HQ−规则净缺口m | 正缺口累计m | 负缺口累计m | 平方缺口m² |',
                      '| --- | ---: | ---: | ---: | ---: | ---: |'])
        for name, values in mechanism['nonexclusive_gap_contributions'].items():
            lines.append(f"| {name} | {values['point_count']} | {fmt(values['model_minus_rule_sum_m'])} | {fmt(values['gross_positive_gap_m'])} | {fmt(values['gross_negative_gap_m'])} | {fmt(values['model_minus_rule_squared_sum_m2'])} |")
        lines.extend(['', f"{mechanism['multiple_flag_point_count']}点具有多个标记；各行不得相加作为已解释缺口，更不是因果贡献比例。", ''])
        for kind, ranked in opportunity['top_contributors'][region].items():
            lines.extend([f"按{kind}缺口贡献的前5条(scene,target)，完整固定Top20见JSON：", '',
                '| 方向 | scene | target | 贡献 |', '| --- | --- | --- | ---: |'])
            for direction, records in ranked['scene_targets'].items():
                for record in records[:5]:
                    lines.append(f"| {direction} | {record['scene_id']} | {record['target_id']} | {fmt(record['contribution'])} |")
            lines.append('')
    lines.extend(['## 决策边界', '',
        '| 证据 | 可考虑的单因素后续 | 不能直接推出 |', '| --- | --- | --- |',
        '| 有利helper被拒绝且状态未变 | 监督或激活的最小消融 | 已证实ReLU死区 |',
        '| 正权重、朝有利方向修正但幅度有限 | 幅度目标或权重分配 | 仅凭小权重就证明不足 |',
        '| 辅助误差／分歧大 | 可靠性特征或风险约束 | support等于准确测量 |',
        '| 错身份标记与主要缺口重合 | 可区分的关联决策对照 | 错分组是唯一原因 |',
        '| 多机制重叠或参考缺失 | 先补可区分对照／记不可识别 | 强行给错误分配唯一原因 |', '',
        '本地收到实际结果后，按误差贡献与竞争解释提出最多一个可证伪修改；此报告不授权训练、不生成新选择规则。',
        '速度收益不因位置失败被否定；原OLS重放数值差不能界定学习分组输出的跨设备误差。',
        '场景及共享组件分布见JSON；反复使用V和单seed限制保留，组件不等于独立训练重复。'])
    return '\n'.join(lines)+'\n'


def run(args):
    source, system, fusion, output = (Path(getattr(args, key)).resolve() for key in
                                    ('source_066', 'source_system', 'source_fusion', 'output'))
    for protected in (source, system.parent, fusion):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('067 output must be separate from all immutable input runs')
    if type(args.cpu_threads) is not int or args.cpu_threads < 1:
        raise ValueError('positive CPU budget required')
    if any(os.environ.get(key) != str(args.cpu_threads)
           for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('set OMP/MKL/OPENBLAS thread limits before importing the cache entry')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', scope='067-readonly-fusion-cache-attribution', dataset_level='Level1',
        split='V_select', source_066=str(source), source_system=str(system), source_fusion=str(fusion),
        source_acceptance='REJECTED', method_result_accepted=False, checkpoint_loaded=False,
        model_forwards=0, optimizer_updates=0, archive_members_read=0, cached_reference_used=True,
        gpu_used=False, test_opened=False, groups_recomputed=False, scores_recomputed=False,
        threshold_reselected=False, cpu_threads=args.cpu_threads, completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        meta = bind_sources(source, system, fusion, read(args.manifest))
        (output/'scenes').mkdir()
        a_rows, a_arrays, b_rows, b_arrays, checked = [], [], [], [], []
        for sid in meta['scene_ids']:
            fragment = load_arrays(source, 'predictions', sid)
            saved = load_arrays(source, 'system-predictions', sid)
            original = load_arrays(system, 'predictions', sid)
            original_system = load_arrays(system, 'system-predictions', sid)
            expected = read(source/'scenes'/f'scene-{sid}.json.gz')
            component = meta['components'][sid]
            artifacts, routes, row = _verify_scene(fragment, saved, original, original_system, expected, sid, component)
            checked.append(row)
            numeric_rows = []
            for arm in LEARNED_METHODS:
                old = load_arrays(meta['arm_systems'][arm], 'predictions', sid)
                old_system = load_arrays(meta['arm_systems'][arm], 'system-predictions', sid)
                exact_subset(old, {key: value for key, value in original.items()
                                   if key.startswith(('ols_', 'region_'))}, '057/056 ordinary fragment')
                exact_subset(old_system, {key: value for key, value in original_system.items()
                                          if key.startswith('ols_')}, '057/056 ordinary system')
                item, values = audit_source(fragment, old, prefixed(old_system, 'ols_learned_'), arm, sid, component)
                if item['rejected_output_replay'] != (not expected['source057_replay'][arm]['passed']):
                    raise ValueError('067 numeric refusal differs from original066: '+sid+'/'+arm)
                a_rows.append(item)
                a_arrays.append(values)
                numeric_rows.append(item)
            b_row, b_values = analyze_scene(fragment, artifacts, routes, sid, component)
            b_rows.append(b_row)
            b_arrays.append(b_values)
            write_json_gzip(output/'scenes'/f'scene-{sid}.json.gz', dict(numeric=numeric_rows, opportunities=b_row))
            if sid == '2535':
                write_json_gzip(output/'scene-2535-numeric.json.gz', numeric_rows)
            state.update(completed_scenes=len(b_rows), last_scene_id=sid, elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        for arm in METHODS:
            pooled = pool_metrics([r['methods'][arm]['full_target_metrics'] for r in checked])
            assert_metrics(pooled, meta['summary']['methods'][arm]['full_target_metrics'], '066 full population '+arm)
        actual = {(r['scene_id'], r['arm']) for r in a_rows if r['rejected_output_replay']}
        expected = {(r['scene_id'], r['arm']) for r in meta['audit']['rejected']}
        if actual != expected:
            raise ValueError('067 full numerical rejection ledger differs from066')
        numeric, opportunity = aggregate_numeric(a_rows, a_arrays), aggregate_opportunities(b_rows, b_arrays)
        write_json_gzip(output/'numeric-impact.json.gz', numeric)
        write_json_gzip(output/'fusion-opportunities.json.gz', opportunity)
        (output/'report.md').write_text(render_report(numeric, opportunity), encoding='utf-8')
        state.update(status='COMPLETED', analysis_complete=True, source_status_unchanged=True,
            total_scenes=len(b_rows), replay_count=len(a_rows), rejected_source_replays=len(actual),
            elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return dict(run=state, numeric=numeric, opportunities=opportunity)
    except Exception as error:
        state.update(status='FAILED', analysis_complete=False, error=f'{type(error).__name__}: {error}',
                     elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


if __name__ == '__main__':
    run(_cli_args)
