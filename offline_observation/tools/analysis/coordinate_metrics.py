"""Read existing090C arrays; no model, training, archive or rematching.

Eq28 uses raw truth-minus-prediction in degrees. Eq29's printed expression
uses truth-minus-ME, not residual-minus-ME: keep it a named diagnostic, never
silently substitute the conventional residual SD or claim author equivalence.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from association.evaluation.physical import _check_valid_states
from association.evaluation.paper_fusion_metrics import paper_fusion_scene, aggregate_paper_fusion
from association.evaluation.association import _aggregate as aggregate_edges
from association.utils.runtime import read_json_or_gzip as read, write_json, write_json_gzip

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('learned_sequence', 'anchor', 'mean_supported')
REGIONS = ('all', 'anchor_inside_support', 'anchor_outside_helper_supported',
           'predicted_group_no_support', 'no_predicted_group')
AXES = {'longitude_raw': (1, False), 'latitude_raw': (0, False), 'longitude_shortest_arc': (1, True)}
TARGET_METRICS = ('ME', 'MAE', 'absolute_ME', 'residual_SD', 'printed_Eq29_diagnostic')
PAPER = dict(unit='1e-4 degree', source='TMTAFF Table3 p23, Eq28-31 p18',
    longitude=dict(mME=6.24, mME_plus_minus=.13, mESD=2.64, mESD_plus_minus=.07),
    latitude=dict(mME=5.88, mME_plus_minus=.11, mESD=2.39, mESD_plus_minus=.08),
    speed=dict(unit='kn', mME=1.03, mME_plus_minus=.06, mESD=.47, mESD_plus_minus=.05),
    heading=dict(unit='degree', mME=.84, mME_plus_minus=.05, mESD=.42, mESD_plus_minus=.04),
    comparable_test_population=False, author_ESD_implementation_verified=False)
ASSOCIATION_PAPER = {
    'itsa': {'m_pca': (94.58, .21), 'mw_pca': (91.71, .24), 'm_pfa': (3.59, .07), 'mw_pfa': (6.48, .09)},
    'mstsa': {'m_pca': (93.91, .26), 'mw_pca': (91.38, .21), 'm_pfa': (4.47, .15), 'mw_pfa': (6.87, .18)},
}


def point_stats(values, *, signed=True):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        out = dict(count=0, ME=None, MAE=None, RMSE=None, abs_P95=None, abs_P99=None)
    else:
        out = dict(count=len(values), ME=float(values.mean()), MAE=float(np.abs(values).mean()),
                   RMSE=float(np.sqrt(np.mean(values*values))),
                   abs_P95=float(np.quantile(np.abs(values), .95)),
                   abs_P99=float(np.quantile(np.abs(values), .99)))
    if not signed: out.pop('ME')  # A radial distance has no signed coordinate bias.
    return out


def coordinate_scene(truth, prediction, target_valid, prediction_valid, region, *, wrap=False):
    """One coordinate [targets,query], including every declared target row.

    Values are reported in 1e-4 degrees. A target's SD needs >=2 valid points;
    no zero imputation. Conditional means and their denominators are explicit.
    Strict full-population means additionally require no missed true points.
    """
    mask = target_valid & prediction_valid & region
    raw = truth - prediction
    errors = ((raw+180.) % 360.-180.) if wrap else raw
    errors = errors*10000.
    n = mask.sum(1)
    target_metrics = {key: [] for key in TARGET_METRICS}
    for row in np.flatnonzero(n):
        e = errors[row, mask[row]]
        bias = float(e.mean())
        target_metrics['ME'].append(bias)
        target_metrics['MAE'].append(float(np.abs(e).mean()))
        target_metrics['absolute_ME'].append(abs(bias))
        if len(e) > 1:
            target_metrics['residual_SD'].append(float(np.std(e, ddof=1)))
            if not wrap:
                # Literal diagnostic only: epsilon denotes the true feature in the PDF.
                actual = truth[row, mask[row]]*10000.
                target_metrics['printed_Eq29_diagnostic'].append(
                    float(np.sqrt(np.sum((actual-bias)**2)/(len(e)-1))))
    target_count = len(truth)
    available = target_valid & region
    missing = available & ~prediction_valid
    means = {key: float(np.mean(v)) if v else None for key, v in target_metrics.items()}
    defined = {key: len(v) for key, v in target_metrics.items()}
    complete = {key: means[key] if target_count and defined[key] == target_count and not missing.any()
                else None for key in TARGET_METRICS}
    values = errors[mask]
    row = dict(declared_targets=target_count, available_target_points=int(available.sum()),
        evaluated_points=int(mask.sum()), missing_prediction_points=int(missing.sum()),
        targets_with_reference=int(available.any(1).sum()), targets_with_evaluation=int((n > 0).sum()),
        targets_with_two_points=int((n > 1).sum()),
        raw_difference_over_180_count=int((np.abs(raw[mask]) > 180.).sum()),
        point_equal=point_stats(values), target_mean=means,
        defined_targets=defined, strict_complete_target_mean=complete)
    return row, values


def pool_coordinates(rows, values):
    """Mean of within-scene target means, NOT a pooled mean of all targets."""
    counts = ('declared_targets', 'available_target_points', 'evaluated_points',
              'missing_prediction_points', 'targets_with_reference', 'targets_with_evaluation',
              'targets_with_two_points', 'raw_difference_over_180_count')
    out = {key: sum(row[key] for row in rows) for key in counts}
    out.update(scene_count=len(rows), point_equal=point_stats(np.concatenate(values)),
               scene_equal_target_mean={}, defined_scenes={}, strict_complete_population={})
    for key in TARGET_METRICS:
        v = [row['target_mean'][key] for row in rows if row['target_mean'][key] is not None]
        out['scene_equal_target_mean'][key] = float(np.mean(v)) if v else None
        out['defined_scenes'][key] = len(v)
        strict = [row['strict_complete_target_mean'][key] for row in rows]
        out['strict_complete_population'][key] = float(np.mean(strict)) if strict and all(
            value is not None for value in strict) else None
    out['coverage'] = out['evaluated_points']/out['available_target_points'] if out['available_target_points'] else None
    return out


def _same(left, right, name):
    if left.shape != right.shape or not (np.array_equal(left, right, equal_nan=True)
            if left.dtype.kind in 'fc' else np.array_equal(left, right)):
        raise ValueError('cached population mismatch: '+name)


def _artifact(saved, method):
    keys = ('aligned_prediction', 'aligned_truth', 'aligned_prediction_valid', 'aligned_truth_valid',
            'target_ids', 'aligned_target_ids', 'aligned_group_indices', 'group_anchor_target_indices')
    return {key: saved[method+'_'+key] for key in keys}


def evaluate_scene(endpoint, sid, *, paper_attributes=False,
                   prediction_speed_unit='native', target_speed_unit='native'):
    """Only cached arrays. The first T rows retain the original target assignment."""
    with np.load(endpoint/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as source:
        artifacts = {method: _artifact(source, method) for method in METHODS}
    base = artifacts['learned_sequence']; t = len(base['target_ids'])
    shape = base['aligned_truth'].shape
    if (len(shape) != 3 or shape[-1] != 4 or shape[0] < t or t == 0
            or len(set(base['target_ids'])) != t):
        raise ValueError('invalid full target rows')
    _same(base['aligned_target_ids'][:t], base['target_ids'], 'target ordering')
    if np.any(base['aligned_target_ids'][t:] != '') or base['aligned_truth_valid'][t:].any():
        raise ValueError('duplicate predictions must have no independent truth')
    for method, artifact in artifacts.items():
        for key in base:
            if key != 'aligned_prediction': _same(artifact[key], base[key], method+'/'+key)
        for value, valid in (('aligned_prediction', 'aligned_prediction_valid'), ('aligned_truth', 'aligned_truth_valid')):
            if artifact[value].shape != shape or artifact[valid].shape != shape or artifact[valid].dtype != bool:
                raise ValueError('invalid physical cache shape/mask')
            _check_valid_states(artifact[value], artifact[valid], method+'/'+value)
    with np.load(endpoint/'predictions'/f'states-{sid}.npz', allow_pickle=False) as source:
        regions = {name: source['region_'+name] for name in REGIONS}
        old_errors = {method: source[method+'_position_error'] for method in METHODS}
    if any(value.shape != (t, shape[1]) or value.dtype != bool for value in regions.values()):
        raise ValueError('cached region masks have wrong population')
    if not regions['all'].all() or not np.all(sum(regions[name].astype(int) for name in REGIONS[1:]) == 1):
        raise ValueError('support regions must partition every declared target query')
    truth = base['aligned_truth'][:t]; tv = base['aligned_truth_valid'][:t]
    pv = base['aligned_prediction_valid'][:t]
    position_mask = tv[..., :2].all(-1) & pv[..., :2].all(-1)
    available = tv[..., :2].all(-1)
    matched = int((base['aligned_group_indices'][:t] >= 0).sum())
    groups = len(base['group_anchor_target_indices'])
    if groups-matched != shape[0]-t:
        raise ValueError('duplicate group/target accounting mismatch')
    row = dict(scene_id=sid, population=dict(targets=t, groups=groups, duplicate_groups=groups-matched,
        available_position_points=int(available.sum()), evaluated_position_points=int(position_mask.sum()),
        missing_prediction_points=int((available & ~position_mask).sum()),
        reference_unavailable_targets=int((~available.any(1)).sum()),
        covered_targets=int(position_mask.any(1).sum())), methods={})
    if paper_attributes:
        # This cache is a position-only learner. Speed/course must still be the
        # original Mean control before declaring its inherited observation units.
        _same(base['aligned_prediction'][..., 2:],
              artifacts['mean_supported']['aligned_prediction'][..., 2:], '090C Mean speed/heading')
        row['paper_fusion_metrics'] = {}
    values = {}
    for method, artifact in artifacts.items():
        prediction = artifact['aligned_prediction'][:t]
        if paper_attributes:
            row['paper_fusion_metrics'][method] = paper_fusion_scene(
                prediction, truth, tv, prediction_valid=pv,
                prediction_speed_unit=prediction_speed_unit, target_speed_unit=target_speed_unit)
        distance = attribute_errors(prediction, truth)['position']
        old = old_errors[method]
        if old.shape != position_mask.shape or not np.array_equal(np.isfinite(old), position_mask) or not np.allclose(
                distance[position_mask], old[position_mask], rtol=1e-10, atol=1e-9):
            raise ValueError('old pointwise position errors do not replay: '+method)
        row['methods'][method] = {}; values[method] = {}
        for name, region in regions.items():
            stats = {}; vectors = {}
            for axis, (column, wrap) in AXES.items():
                stats[axis], vectors[axis] = coordinate_scene(truth[..., column], prediction[..., column],
                    tv[..., column], pv[..., column], region, wrap=wrap)
            vectors['position_m'] = distance[position_mask & region]
            stats['position_m'] = point_stats(vectors['position_m'], signed=False)
            row['methods'][method][name] = stats; values[method][name] = vectors
    return row, values


def aggregate(rows, vectors):
    out = dict(scene_count=len(rows), scene_ids=[row['scene_id'] for row in rows],
        population={key: sum(row['population'][key] for row in rows) for key in rows[0]['population']}, methods={})
    if any('paper_fusion_metrics' in row for row in rows):
        if not all('paper_fusion_metrics' in row for row in rows):
            raise ValueError('cannot pool incomplete paper attribute reports')
        out['paper_fusion_metrics'] = {method: aggregate_paper_fusion([
            row['paper_fusion_metrics'][method] for row in rows]) for method in METHODS}
    for method in METHODS:
        out['methods'][method] = {}
        for region in REGIONS:
            out['methods'][method][region] = {
                axis: pool_coordinates([row['methods'][method][region][axis] for row in rows],
                                        [data[method][region][axis] for data in vectors]) for axis in AXES}
            out['methods'][method][region]['position_m'] = point_stats(np.concatenate([
                data[method][region]['position_m'] for data in vectors]), signed=False)
    return out


def bind_source(source, manifest):
    root_run, root_summary, fit = [read(source/name) for name in ('run.json', 'summary.json', 'fit/run.json')]
    expected = read(ROOT/'configs/frozen-position-adapter-full-finetune-level1.json')
    for doc in (root_run, root_summary, fit):
        if (doc.get('status') != 'COMPLETED' or doc.get('configuration') != expected
                or doc.get('optimizer_updates') != 4000 or doc.get('test_opened') is not False):
            raise ValueError('source must be the completed090C full-finetune endpoint')
    endpoint = source/'final'; summary = read(endpoint/'summary.json')
    for doc in (read(endpoint/'run.json'), summary):
        checkpoint = doc.get('checkpoint', {})
        if (doc.get('status') != 'COMPLETED' or doc.get('initial') is not False
                or doc.get('configuration') != expected or doc.get('optimizer_updates') != 0
                or doc.get('test_opened') is not False or doc.get('groups_recomputed') is not False
                or doc.get('threshold_reselected') is not False or checkpoint.get('step') != 4000
                or checkpoint.get('actual_updates') != 4000
                or Path(checkpoint.get('path', '')).resolve() != source/'fit/final.pt'
                or doc.get('source_system') != root_run.get('source_system')):
            raise ValueError('final cache is not bound to the completed fixed090C endpoint')
    ids = manifest['splits']['V_select']
    if manifest.get('dataset_level') != 'Level1' or len(ids) != 300 or summary['scene_ids'] != ids:
        raise ValueError('requires unchanged full V300 order')
    components = {sid: str(cid) for cid, item in summary['components'].items() for sid in item['scene_ids']}
    if components != {sid: str(manifest['components'][sid]) for sid in ids}:
        raise ValueError('shared component registry differs')
    for sub in ('predictions', 'system-predictions'):
        if {p.name for p in (endpoint/sub).glob('*.npz')} != {f'states-{sid}.npz' for sid in ids}:
            raise ValueError('missing or extra scene cache: '+sub)
    return summary, components


def association_report(source_system, manifest):
    """Recompute only saved per-scene counts, never scores or thresholds."""
    path = Path(source_system).parent/'association'/'summary.json'
    saved = read(path)
    metrics = saved['selected_result']['metrics']
    if (saved.get('status') != 'COMPLETED' or saved.get('test_opened') is not False
            or saved['selected']['update'] != 1000
            or metrics['threshold'] != saved['selected']['selection']['threshold']
            or not np.isclose(metrics['threshold'], 1.7320103112387182, rtol=0, atol=1e-14)):
        raise ValueError('association report requires original fixed056 selected1000')
    groups = {'all': metrics, 'itsa': metrics['itsa'], 'mstsa': metrics['mstsa']}
    result = dict(source=str(path), threshold=metrics['threshold'], optimizer_updates=0,
                  threshold_reselected=False, unit='percent', variants={},
                  paper_reference=ASSOCIATION_PAPER, comparable_test_population=False,
                  note='056 association reused by all three fusion methods; not a new fusion contribution')
    expected_ids = manifest['splits']['V_select']
    for name, group in groups.items():
        if [str(row['scene_id']) for row in group['per_scene']] != expected_ids:
            raise ValueError('association scene population/order differs')
        rows = []
        for row in group['per_scene']:
            tp, fp, fn, tn = (row[k] for k in ('tp','fp','fn','tn'))
            if any(not isinstance(v, int) or v < 0 for v in (tp,fp,fn,tn)) or tp+fp+fn+tn != row['pair_count']:
                raise ValueError('invalid saved confusion counts')
            rows.append({**row, 'positive_count': tp+fn,
                'precision': tp/(tp+fp) if tp+fp else None,
                'recall': tp/(tp+fn) if tp+fn else None,
                'p_fa': fp/(tp+fp) if tp+fp else None,
                'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.})
        actual = aggregate_edges(rows)
        for key in ('tp', 'fp', 'fn', 'tn', 'pair_count'):
            if actual[key] != group[key]: raise ValueError('association count mismatch: '+key)
        for key in ('m_pca', 'mw_pca', 'm_pfa', 'mw_pfa'):
            a, b = actual['paper_metrics'][key], group['paper_metrics'][key]
            if (a is None) != (b is None) or (a is not None and not np.isclose(a, b, rtol=1e-12, atol=1e-14)):
                raise ValueError('association saved paper metric mismatch: '+key)
        result['variants'][name] = dict(counts={key: actual[key] for key in ('tp','fp','fn','tn','pair_count')},
            values={key: None if actual['paper_metrics'][key] is None else 100*actual['paper_metrics'][key]
                    for key in ('m_pca','mw_pca','m_pfa','mw_pfa')},
            denominators=actual['paper_metrics'])
    if result['variants']['all']['counts']['pair_count'] != 4439092:
        raise ValueError('056 full candidate population differs')
    return result


def paper_tables(summary):
    fmt = lambda x: 'null' if x is None else f'{x:.6f}'
    names = dict(learned_sequence='090C＋Mean速度/航向', anchor='Anchor', mean_supported='Mean')
    lines = ['', '## 完整论文维度：本项目实测与作者报告分源', '',
        '以下融合主表为目标时间→场景内目标→场景等权。每属性使用自己的有效mask；',
        '速度预测按kn、reference按m/s解释后分别换算到kn，此前提来自Level1八场核验，未声称全包单位逐条已证实。',
        '090C只学习位置；速度/航向逐点核实继承Mean。仅修正事后速度比较，不修改网络输入或训练。', '',
        '| 来源 | 属性 | 单位 | mME（接近0为佳） | 残差SD／作者mESD* | MAE | 可评点 | 覆盖率 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        for attribute, row in summary['paper_fusion_metrics'][method]['attributes'].items():
            stat = row['scene_equal_target_mean']
            lines.append('| '+names[method]+' | '+attribute+' | '+row['unit']+' | '
                         +' | '.join(fmt(stat[k]) for k in ('ME','residual_SD','MAE'))
                         +f" | {row['evaluated_points']} | {fmt(row['coverage'])} |")
    for attribute in ('longitude','latitude','speed','heading'):
        p = PAPER[attribute]; unit = p.get('unit', PAPER['unit'])
        lines.append(f"| TMTAFF论文Table3 | {attribute} | {unit} | {p['mME']} ± {p['mME_plus_minus']}"
                     +f" | {p['mESD']} ± {p['mESD_plus_minus']} | 未报告 | 未公开对应人口 | 未对齐 |")
    lines += ['', '*本项目是标准残差样本SD；作者ESD实现未核实，印刷Eq29诊断另存JSON，不把两列当已完全等价。',
        'ME为有符号偏差，不按数值越小越好。航向主表按原始度数差；既有圆周物理误差另列。',
        '新paper_fusion_metrics的严格完整人口列要求所有声明query都有真值和预测，较101旧coordinate严格列更严；条件均值不受此区别影响。', '',
        '| 任务／来源 | mPCA % | mwPCA % | mPFA % | mwPFA % |', '|---|---:|---:|---:|---:|']
    for kind in ('itsa','mstsa'):
        keys = ('m_pca','mw_pca','m_pfa','mw_pfa')
        row = summary['association']['variants'][kind]['values']
        lines.append('| '+kind.upper()+' 本项目056（上述三种融合共用） | '+' | '.join(fmt(row[k]) for k in keys)+' |')
        p = ASSOCIATION_PAPER[kind]
        lines.append('| '+kind.upper()+' TMTAFF论文 | '+' | '.join(f'{p[k][0]} ± {p[k][1]}' for k in keys)+' |')
    lines += ['', '本项目为开发V300；作者论文行仅外部参照。同单位/同指标不等于同测试人口，不据此宣布胜负。', '']
    return '\n'.join(lines)


def report(summary):
    fmt = lambda x: 'null' if x is None else f'{x:.6f}'
    names = dict(learned_sequence='090C', anchor='Anchor（OLS单轨）', mean_supported='Mean（普通融合）')
    title = '# 102：090C完整论文指标对齐' if 'paper_fusion_metrics' in summary else '# 101：090C已有预测，经纬度指标对齐'
    lines = [title, '',
        '仅CPU缓存后处理，零训练/前向。原分组、目标对应和所有缺失/重复保持。', '',
        '以下经纬坐标表单位为 **10⁻⁴度**。先目标时间统计，再场景内目标等权，最后场景等权。',
        '下表为有可评点的条件统计；严格完整人口统计及各级分母见JSON。', '',
        '| 本项目V300 | 坐标 | mME Eq28字面 | 目标绝对ME均值 | 坐标MAE层级均值 | 残差样本SD层级均值 |',
        '|---|---|---:|---:|---:|---:|']
    for method in METHODS:
        for axis in ('longitude_raw', 'latitude_raw'):
            stat = summary['methods'][method]['all'][axis]['scene_equal_target_mean']
            lines.append('| '+names[method]+' | '+axis+' | '+' | '.join(fmt(stat[k]) for k in
                ('ME', 'absolute_ME', 'MAE', 'residual_SD'))+' |')
    lines += ['', '## 论文表3外部参考（不作同测试集排名）', '',
        '| 坐标 | 作者mME | 作者mESD |', '|---|---:|---:|',
        '| 经度 | 6.24 ± 0.13 | 2.64 ± 0.07 |', '| 纬度 | 5.88 ± 0.11 | 2.39 ± 0.08 |', '',
        'ME有正负抵消，不能按越小越好或与论文算胜负比例。我们的残差SD定义为std(truth−prediction, ddof=1)，',
        '不是论文印刷Eq29的sqrt(sum((truth−ME)^2)/(n−1))；后者另存printed_Eq29_diagnostic，不能据其判作者实际表现。',
        '论文Level/名单/时间窗/缺失与重复规则未完全还原；当前为开发V300而非封存测试。', '',
        '## 原米制指标与覆盖复核', '', '| 方法 | MAE m | RMSE m | P95 m | P99 m |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        stat = summary['methods'][method]['all']['position_m']
        lines.append('| '+names[method]+' | '+' | '.join(fmt(stat[k]) for k in ('MAE', 'RMSE', 'abs_P95', 'abs_P99'))+' |')
    lines += ['', f"完整人口：{summary['population']}。", '',
        '## 090C误差出现在哪里', '',
        '| 区域 | 可评位置点 | 位置MAE m | 经度MAE层级均值 10⁻⁴度 | 纬度MAE层级均值 10⁻⁴度 |',
        '|---|---:|---:|---:|---:|']
    for region in REGIONS[1:]:
        stat = summary['methods']['learned_sequence'][region]
        lines.append(f"| {region} | {stat['position_m']['count']} | {fmt(stat['position_m']['MAE'])} | "
                     + ' | '.join(fmt(stat[key]['scene_equal_target_mean']['MAE']) for key in
                                  ('longitude_raw', 'latitude_raw'))+' |')
    lines += ['',
        '逐场/组件/最大组件外、支撑区/机会区/无支撑区、经度最短弧与原始差、偏差/绝对误差/尾部见summary和scene-summary。', '']
    if 'paper_fusion_metrics' in summary:
        lines.append(paper_tables(summary))
        lines += ['## 原圆周航向误差（度，与raw主表分列）', '', '| 方法 | MAE | RMSE |', '|---|---:|---:|']
        for method in METHODS:
            values = summary['original_circular_heading'][method]
            lines.append('| '+names[method]+' | '+fmt(values['mae'])+' | '+fmt(values['rmse'])+' |')
    else:
        lines.append('速度和航向未在本次补算；当前位置模型沿用Mean属性，不宣称四属性学习创新。')
    return '\n'.join(lines)


def run(args):
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    paper_attributes = getattr(args, 'paper_attributes', False)
    prediction_unit = getattr(args, 'prediction_speed_unit', None)
    target_unit = getattr(args, 'target_speed_unit', None)
    if paper_attributes and (prediction_unit != 'kn' or target_unit != 'mps'):
        raise ValueError('090C paper report requires explicit kn prediction and mps reference declarations')
    if not paper_attributes and (prediction_unit or target_unit):
        raise ValueError('speed declarations require --paper-attributes')
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('output must not overlap read-only source')
    source_system = None
    if paper_attributes:
        source_system = Path(read(source/'run.json')['source_system']).resolve()
        protected = source_system.parent
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('output must not modify056 source')
    if output.exists(): raise FileExistsError('new output required; no automatic retry')
    started = time.monotonic(); output.mkdir(parents=True)
    state = dict(status='RUNNING', scope='090C-paper-metrics-102' if paper_attributes else '090C-coordinate-cache-101',
        source_run=str(source), device='cpu',
        optimizer_updates=0, model_forwards=0, checkpoint_loaded=False, archive_read=False, test_opened=False,
        groups_recomputed=False, threshold_reselected=False, completed_scenes=0, time_policy='advisory')
    write_json(output/'run.json', state)
    try:
        manifest = read(args.manifest)
        original, components = bind_source(source, manifest)
        association = None
        if paper_attributes:
            association = association_report(source_system, manifest)
            state['speed_unit_contract'] = dict(prediction=prediction_unit, target=target_unit, report='kn',
                basis='Level1 first8 T_core numerical audit; reference mps interpretation, not full-package proof')
        rows, vectors = [], []
        for sid in original['scene_ids']:
            row, values = evaluate_scene(source/'final', sid, paper_attributes=paper_attributes,
                prediction_speed_unit=prediction_unit or 'native', target_speed_unit=target_unit or 'native')
            row['component_id'] = components[sid]; rows.append(row); vectors.append(values)
            state.update(completed_scenes=len(rows), elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', state)
        result = aggregate(rows, vectors)
        if paper_attributes:
            for method in METHODS:
                for attribute, axis in (('longitude','longitude_raw'),('latitude','latitude_raw')):
                    old = result['methods'][method]['all'][axis]
                    new = result['paper_fusion_metrics'][method]['attributes'][attribute]
                    if new['evaluated_points'] != old['evaluated_points']:
                        raise ValueError('paper coordinate population changed')
                    for key in TARGET_METRICS:
                        a, b = new['scene_equal_target_mean'][key], old['scene_equal_target_mean'][key]
                        if (a is None) != (b is None) or (a is not None and not np.isclose(a,b,rtol=1e-10,atol=1e-8)):
                            raise ValueError('paper coordinate statistic changed: '+key)
            result.update(association=association, source_system=str(source_system),
                speed_unit_contract=state['speed_unit_contract'], coordinate_report_reproduced=True,
                attribute_sources=dict(learned_sequence='learned position; unchanged Mean speed/heading',
                                       anchor='visible OLS/interpolation control', mean_supported='visible ordinary fusion'),
                original_circular_heading={method: original['methods'][method]['full_target_metrics']['all']['heading']
                                           for method in METHODS})
        # Establish the original population/error before reporting new coordinate metrics.
        for method in METHODS:
            actual = result['methods'][method]['all']['position_m']
            old = original['methods'][method]['full_target_metrics']['all']['position']
            if actual['count'] != old['intersection_count'] or not np.isclose(actual['MAE'], old['mae'], rtol=1e-10, atol=1e-9):
                raise ValueError('original aggregate MAE/population not reproduced: '+method)
        pop = result['population']
        if (pop['targets'], pop['groups'], pop['duplicate_groups'], pop['available_position_points'],
                pop['evaluated_position_points']) != (9100, 9353, 610, 454952, 442263):
            raise ValueError('090C declared complete target/group/position population differs')
        component_results = {}
        for cid in sorted(set(components.values())):
            chosen = [i for i, row in enumerate(rows) if row['component_id'] == cid]
            component_results[cid] = aggregate([rows[i] for i in chosen], [vectors[i] for i in chosen])
        largest = max(component_results, key=lambda c: component_results[c]['population']['evaluated_position_points'])
        others = [i for i, row in enumerate(rows) if row['component_id'] != largest]
        result.update(status='COMPLETED', source_run=str(source), endpoint='final', paper_reference=PAPER,
            coordinate_unit='1e-4 degree', coordinate_scale=10000, sign='truth minus prediction',
            interpretation='Conditional point/target/scene denominators explicit; strict incomplete populations null. No ranking against external paper.',
            weighting='scene_equal_target_mean: each target over its valid times, targets within scene, then equal scenes',
            components=component_results, largest_component_id=largest,
            outside_largest_component=aggregate([rows[i] for i in others], [vectors[i] for i in others]),
            original_position_errors_reproduced=True, test_opened=False,
            worst_position_scenes=sorted([dict(scene_id=row['scene_id'], component_id=row['component_id'],
                **row['methods']['learned_sequence']['all']['position_m']) for row in rows],
                key=lambda r: r['RMSE']**2*r['count'] if r['count'] else 0, reverse=True)[:10])
        write_json_gzip(output/'summary.json.gz', result)
        write_json_gzip(output/'scene-summary.json.gz', rows)
        (output/'report.md').write_text(report(result), encoding='utf-8')
        state.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        return result
    except BaseException as error:
        state.update(status='INTERRUPTED' if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 'FAILED',
                     error=f'{type(error).__name__}: {error}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--paper-attributes', action='store_true',
                        help='102: include four-attribute paper metrics and original056 association counts')
    parser.add_argument('--prediction-speed-unit', choices=('kn','mps'))
    parser.add_argument('--target-speed-unit', choices=('kn','mps'))
    args = parser.parse_args()
    def stop(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError('requested stop; no retry')
    previous = signal.signal(signal.SIGTERM, stop)
    try: run(args)
    finally: signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
