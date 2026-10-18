"""One-off T_core endpoint analysis. No archives, reference, V metrics, or models.

Run with association-code/.venv/bin/python; writes JSON and Markdown beside this
file. Native gap quantiles are explicitly reused from the completed prior audit,
not reconstructed from endpoints. All other statistics use T_core metadata only.
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import gzip
import json
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/results/frozen_input_audit_20261018/native_sampling/attempt2'
OUT = Path(__file__).resolve().parent
HORIZONS = [60, 300, 600, 1260, 1800, 3600]
Q = 64


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {'count': 0}
    assert np.isfinite(a).all()
    probs = [0, .01, .05, .25, .5, .75, .9, .95, .99, 1]
    names = ['min', 'p01', 'p05', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max']
    return {'count': len(a), **dict(zip(names, map(float, np.quantile(a, probs))))}


def union_intervals(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def fraction(n, d):
    return n / d if d else None


def threshold_counts(values):
    a = np.asarray(values)
    return {str(h): {'count_le': int((a <= h).sum()), 'denominator': len(a),
                    'fraction_le': fraction(int((a <= h).sum()), len(a))}
            for h in HORIZONS}


def main():
    started = time.monotonic()
    assert union_intervals([(0, 10), (5, 15), (20, 25)]) == [[0, 15], [20, 25]]
    assert union_intervals([(0, 10), (10, 20)]) == [[0, 20]]
    saved = json.loads((SOURCE / 'results.json').read_text())['splits']['T_core']
    split_rows = Counter()
    scenes = []
    with gzip.open(SOURCE / 'scene_metadata.jsonl.gz', 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            split_rows[row['split']] += 1
            if row['split'] == 'T_core':
                scenes.append(row)
            else:
                assert row['split'] == 'V_select', 'Unexpected split; do not analyze test data'
    assert len({s['scene_id'] for s in scenes}) == len(scenes) == 512

    counts = Counter()
    track_durations, scene_spans, grid_spacings = [], [], []
    duration_by_source = defaultdict(list)
    mean_native_gap = defaultdict(list)
    native_gap_denominators = Counter()
    identity_spans, identity_union_durations, identity_holes = [], [], []
    identity_support_fractions = []
    head_gaps, tail_gaps = [], []
    head_by_source, tail_by_source = defaultdict(list), defaultdict(list)
    pair_counts = defaultdict(Counter)
    pair_gaps = defaultdict(list)
    query_counts = Counter()
    identity_query_counts = Counter()
    scene_query_fractions = defaultdict(list)
    query_distances = defaultdict(list)
    query_classes = ('self_support', 'other_same_identity_support',
                     'identity_internal_union_gap', 'outside_identity_envelope')

    for scene in scenes:
        tracks = scene['tracks']
        assert len(tracks) == scene['valid_tracks']
        assert len({t['node_id'] for t in tracks}) == len(tracks)
        by_identity = defaultdict(list)
        starts = np.array([t['support_start'] for t in tracks], dtype=float)
        stops = np.array([t['support_stop'] for t in tracks], dtype=float)
        assert np.isfinite(starts).all() and np.isfinite(stops).all() and (stops > starts).all()
        grid = np.linspace(float(starts.min()), float(stops.max()), Q)
        scene_spans.append(float(stops.max() - starts.min()))
        grid_spacings.append(float(grid[1] - grid[0]))
        scene_categories = Counter()
        counts['scenes'] += 1
        for i, track in enumerate(tracks):
            assert track['canonical_index'] == i
            by_identity[track['identity_id']].append(track)
            duration = track['support_stop'] - track['support_start']
            points = track['native_points']
            assert points >= 2
            counts['tracks'] += 1
            counts['canonical_points'] += points
            track_durations.append(duration)
            duration_by_source[track['source']].append(duration)
            mean_native_gap[track['source']].append(duration / (points - 1))
            native_gap_denominators[track['source']] += points - 1
            own = (grid >= track['support_start']) & (grid <= track['support_stop'])
            assert int(own.sum()) == track['query_support_points']
            counts['query_support_counts_matched'] += 1
            counts['zero_query_support_tracks'] += int(not own.any())
            counts['one_query_support_tracks'] += int(own.sum() == 1)

        assert {r['identity_id']: r['valid_tracks'] for r in scene['identities']} == {
            key: len(group) for key, group in by_identity.items()}
        positive_count = sum(len(group) * (len(group) - 1) // 2 for group in by_identity.values())
        assert positive_count == scene['positive_pairs']
        counts['positive_pairs'] += positive_count
        counts['scene_identities'] += len(by_identity)
        for group in by_identity.values():
            merged = union_intervals([(t['support_start'], t['support_stop']) for t in group])
            lo, hi = merged[0][0], merged[-1][1]
            holes = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
            covered = sum(stop - start for start, stop in merged)
            identity_spans.append(hi - lo)
            identity_union_durations.append(covered)
            identity_support_fractions.append(covered / (hi - lo))
            identity_holes.extend(holes)
            counts['identities_with_internal_union_gap'] += int(bool(holes))
            counts['identity_union_intervals'] += len(merged)
            counts['identity_internal_union_gaps'] += len(holes)
            union = np.zeros(Q, dtype=bool)
            for start, stop in merged:
                union |= (grid >= start) & (grid <= stop)
            inside = (grid >= lo) & (grid <= hi)
            identity_query_counts['some_fragment_support'] += int(union.sum())
            identity_query_counts['internal_union_gap'] += int((inside & ~union).sum())
            identity_query_counts['outside_identity_envelope'] += int((~inside).sum())

            for track in group:
                start, stop = track['support_start'], track['support_stop']
                head, tail = start - lo, hi - stop
                head_gaps.append(head); tail_gaps.append(tail)
                head_by_source[track['source']].append(head)
                tail_by_source[track['source']].append(tail)
                own = (grid >= start) & (grid <= stop)
                masks = (own, ~own & union, inside & ~union, ~inside)
                assert np.all(sum(mask.astype(int) for mask in masks) == 1)
                own_distance = np.maximum(np.maximum(start - grid, grid - stop), 0)
                for name, mask in zip(query_classes, masks):
                    n = int(mask.sum())
                    query_counts[name] += n
                    scene_categories[name] += n
                    if name != 'self_support':
                        query_distances[name].extend(own_distance[mask].tolist())
            for left, right in combinations(group, 2):
                relation = 'same_source' if left['source'] == right['source'] else 'cross_source'
                overlap = min(left['support_stop'], right['support_stop']) - max(
                    left['support_start'], right['support_start'])
                category = 'strict_overlap' if overlap > 0 else 'touching' if overlap == 0 else 'disjoint'
                for label in ('all', relation):
                    pair_counts[label][category] += 1
                    if overlap < 0:
                        pair_gaps[label].append(-overlap)
        for name in query_classes:
            scene_query_fractions[name].append(scene_categories[name] / (Q * len(tracks)))

    assert counts['tracks'] == saved['counts']['valid_tracks'] == 54792
    assert counts['canonical_points'] == saved['counts']['canonical_points'] == 2938785
    assert counts['positive_pairs'] == saved['counts']['positive_pairs'] == 110733
    assert counts['scene_identities'] == saved['counts']['scene_identities'] == 12650
    assert sum(pair_counts['all'].values()) == counts['positive_pairs']
    assert sum(query_counts.values()) == Q * counts['tracks']
    assert sum(identity_query_counts.values()) == Q * counts['scene_identities']
    assert sum(native_gap_denominators.values()) == saved['caps']['full']['retained_delta_seconds']['count']
    for source, denominator in native_gap_denominators.items():
        assert denominator == saved['native_delta_seconds_by_source'][source]['count']
    for field, values in [('duration_seconds', track_durations), ('query_grid_step_seconds', grid_spacings)]:
        recomputed = summary(values)
        for key, value in saved[field].items():
            assert np.isclose(recomputed[key], value, rtol=0, atol=1e-8), (field, key)

    geometry_total = Q * counts['tracks']
    outside_self = geometry_total - query_counts['self_support']
    report = {
        'status': 'COMPLETED', 'split': 'T_core',
        'scope': 'Endpoint-only training analysis; no ZIP/reference/model/T_test reads; V rows skipped before track analysis.',
        'sources': {'metadata': str(SOURCE / 'scene_metadata.jsonl.gz'),
                    'prior_audit': str(SOURCE / 'results.json')},
        'split_rows_seen': dict(split_rows), 'excluded_split_rows': {'V_select': split_rows['V_select']},
        'counts': dict(counts),
        'seconds': {
            'scene_span': summary(scene_spans), 'scene_query64_spacing': summary(grid_spacings),
            'track_support_duration': summary(track_durations),
            'track_support_duration_by_source': {k: summary(v) for k, v in duration_by_source.items()},
            'per_track_mean_native_gap_by_source': {k: summary(v) for k, v in mean_native_gap.items()},
            'identity_envelope_duration': summary(identity_spans),
            'identity_union_support_duration': summary(identity_union_durations),
            'identity_internal_union_gap_duration': summary(identity_holes),
            'track_head_to_identity_envelope': summary(head_gaps),
            'track_tail_to_identity_envelope': summary(tail_gaps),
            'track_head_to_identity_envelope_positive_only': summary([v for v in head_gaps if v > 0]),
            'track_tail_to_identity_envelope_positive_only': summary([v for v in tail_gaps if v > 0]),
            'track_head_by_source': {k: summary(v) for k, v in head_by_source.items()},
            'track_tail_by_source': {k: summary(v) for k, v in tail_by_source.items()},
        },
        'identity_union_support_fraction_of_envelope': summary(identity_support_fractions),
        'same_identity_unordered_pairs': {
            label: {'counts': dict(pair_counts[label]), 'denominator': sum(pair_counts[label].values()),
                    'disjoint_gap_seconds': summary(pair_gaps[label])}
            for label in ('all', 'same_source', 'cross_source')},
        'full_scene_q64_fragment_geometry': {
            'denominator': geometry_total, 'outside_self_denominator': outside_self,
            'categories': {key: {'count': query_counts[key],
                                'fraction_of_all': query_counts[key] / geometry_total,
                                'fraction_of_outside_self': None if key == 'self_support' else query_counts[key] / outside_self,
                                'scene_macro_fraction': float(np.mean(scene_query_fractions[key]))}
                           for key in query_classes},
            'outside_self_distance_to_own_span_seconds': {k: summary(v) for k, v in query_distances.items()},
            'outside_self_distance_thresholds_seconds': {k: threshold_counts(v) for k, v in query_distances.items()},
        },
        'full_scene_q64_identity_geometry': {
            'denominator': Q * counts['scene_identities'],
            'categories': {k: {'count': v, 'fraction': v / (Q * counts['scene_identities'])}
                           for k, v in identity_query_counts.items()}},
        'threshold_descriptions': {
            'track_support_duration': 'Fraction with full native endpoint span <= h; NOT actual sliding-window retention.',
            'disjoint_same_identity_pair_gap': 'Necessary, not sufficient, for both disjoint supports to intersect one h-span window; not actual pair coverage.',
            'identity_internal_union_gap': 'Endpoint union internal hole <= h; does not assert reference truth or sample-level observation.',
            'track_head_tail': 'Difference to same-identity observed envelope; not legal missing-target label.',
        },
        'thresholds_seconds': {
            'track_support_duration': threshold_counts(track_durations),
            'disjoint_same_identity_pair_gap': threshold_counts(pair_gaps['all']),
            'identity_internal_union_gap': threshold_counts(identity_holes),
            'track_head_to_identity_envelope': threshold_counts(head_gaps),
            'track_tail_to_identity_envelope': threshold_counts(tail_gaps),
        },
        'reused_prior_native_gap_audit': {
            'provenance': 'Verbatim saved T_core quantiles from prior raw-point audit; only counts rechecked from metadata.',
            'native_gap_denominator_by_source_recomputed': dict(native_gap_denominators),
            'native_delta_seconds': saved['caps']['full']['retained_delta_seconds'],
            'native_delta_seconds_by_source': saved['native_delta_seconds_by_source'],
            'native_per_track_max_gap_seconds': saved['caps']['full']['per_track_max_gap_seconds'],
            'cap128_retained_delta_seconds': saved['caps']['128']['retained_delta_seconds'],
            'cap128_counts': saved['caps']['128']['counts'],
        },
        'limitations': [
            'Support means inclusive first-to-last fragment interval, not an actual native sample at each query; internal sampling gaps remain.',
            'Identity envelope and union use input fragments only, not reference lifespan; outside-envelope queries may have no target truth.',
            'Same-identity grouping uses existing labels for analysis; does not grant oracle identities to the model.',
            'No per-point timestamps/reference coverage in metadata: cannot compute actual per-window native counts, reference-supervised horizons, maneuvers, or proposed window coverage.',
            'Unordered-pair counts weight identities with more fragments quadratically; fragment-query counts weight identities by fragment count; not the scene-uniform balanced training sampler.',
            '20 seconds encoding unit is not sensor period; raw gap median 21 seconds is not evidence for a 21-minute window.',
            'Endpoint quantiles cannot select an optimal window, prove long extrapolation caused poor metrics, estimate speed, or change the approved protocol.',
        ],
        'verification': 'All training counts, every track Q64 support count, duration/query-spacing prior quantiles, native-gap denominators, and category partitions matched.',
        'elapsed_seconds': time.monotonic() - started,
    }
    (OUT / 'train_metadata_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    write_markdown(report)
    print(json.dumps({'status': report['status'], 'counts': report['counts'],
                      'geometry': report['full_scene_q64_fragment_geometry']['categories'],
                      'elapsed_seconds': report['elapsed_seconds']}, ensure_ascii=False, indent=2))


def write_markdown(r):
    def val(x):
        return f'{x:.2f}'

    def pct(x):
        return f'{100*x:.2f}%'

    s = r['seconds']; g = r['full_scene_q64_fragment_geometry']; c = r['counts']
    lines = [
        '# T_core 时间任务尺度：已有端点元数据复算', '',
        '2026-10-19；只做分析，没有更改训练任务、配置或执行代码，没有新增 GPU。', '',
        '**结论：1260 秒（21 分钟）只能视为“64 个 query 间隔 20 秒”的工程候选。下面数据能说明全场景与局部恢复是不同任务，不能证明 21 分钟最佳，也不能凭这些分位数直接批准训练窗口。**', '',
        '## 来源与分母', '',
        f'- 复用既有 `scene_metadata.jsonl.gz`：只分析 T_core **{c["scenes"]} 场景、{c["tracks"]} 条有效片段、{c["scene_identities"]} 个场景内身份、{c["positive_pairs"]} 对同身份无序片段对**。64 条 V_select 场景记录仅识别划分后跳过，不使用其分数或分位数选窗；没有 T_test。',
        f'- 端点来自已有规范化有效轨迹（已排序、去重、按既有规则隔离无效轨迹），原生点共 {c["canonical_points"]}；不是重新读 ZIP，也不是把原始 CSV 所有行计为有效输入。',
        '- 本次重算：场景/轨迹/身份跨度、同身份区间并集、区间空档、片段首尾差及全场景 Q64 几何类别。逐点原生间隔的分位数由此前完整审计结果原样引用，仅用点数减轨数复核分母；没有伪造缺失的时间序列。',
        '- **支撑是片段真实首末时刻之间的闭区间，不代表区间内每个 query 都恰好有原生观测；区间内部原生采样间隔仍然存在。身份包络也是输入包络，不是 reference 的有效生命周期。**', '',
        '## 1. 三种时间尺度', '',
        '| 指标（秒） | 分母 | 中位数 | P90 | P95 | 最大值 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for label, key in [('场景观测包络', 'scene_span'), ('全场景 Q64 相邻 query 间隔', 'scene_query64_spacing'),
                       ('单片段首末支撑跨度', 'track_support_duration'), ('同身份观测包络', 'identity_envelope_duration'),
                       ('同身份支撑并集内部空档', 'identity_internal_union_gap_duration'),
                       ('片段首部相对身份包络的差（含0）', 'track_head_to_identity_envelope'),
                       ('片段尾部相对身份包络的差（含0）', 'track_tail_to_identity_envelope')]:
        d = s[key]
        lines.append(f'| {label} | {d["count"]} | {val(d["median"])} | {val(d["p90"])} | {val(d["p95"])} | {val(d["max"])} |')
    lines += ['', f'有内部并集空档的身份：**{c["identities_with_internal_union_gap"]}/{c["scene_identities"]}（{pct(c["identities_with_internal_union_gap"]/c["scene_identities"])}）**，合计 {c["identity_internal_union_gaps"]} 段空档。这里合并了同身份所有片段后再算相邻并集区间的间距，因此不把两条远端片段之间其实已有第三条片段覆盖的区域当作“所有输入都没有支撑”。空档也不保证有 reference 真值。', '',
                  '片段首尾差表示：该片段相对于“同身份所有输入首末包络”少覆盖多少。它不是人为遮挡长度，也不能直接叫可监督的首尾缺失；不同片段的真实可用范围可能本来就不一样。', '',
                  '## 2. 原生采样间隔（引用既有完整训练审计）', '',
                  '| 序列 | 间隔数 | 中位数（秒） | P90 | P95 | P99 | 最大值 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    prior = r['reused_prior_native_gap_audit']
    for label, d in [('全部未 cap 原生间隔', prior['native_delta_seconds']),
                     *[(f'源 {k}', v) for k, v in prior['native_delta_seconds_by_source'].items()],
                     ('cap128 后保留点间隔', prior['cap128_retained_delta_seconds'])]:
        lines.append(f'| {label} | {d["count"]} | {val(d["median"])} | {val(d["p90"])} | {val(d["p95"])} | {val(d["p99"])} | {val(d["max"])} |')
    lines += ['', '逐点原生间隔分母 = 2,938,785 点 − 54,792 轨 = **2,883,993**，已按源复核。约 20 秒处于较密源的常见采样尺度附近，但两源非均匀采样且差异明显；既有 `time_unit_seconds=20` 仅编码缩放，不构成必须每20秒查询的要求。**原生间隔中位数21秒与候选窗口21分钟没有推导关系。**', '',
              '## 3. 同身份片段之间到底隔多远', '',
              '| 同身份无序 pair | 全部 pair | 严格重叠 | 仅端点相接 | 不重叠 | 不重叠间距中位数/P90/P95（秒） |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    for label, item in r['same_identity_unordered_pairs'].items():
        d = item['disjoint_gap_seconds']; pc = item['counts']
        gap = '/'.join(val(d[k]) for k in ('median', 'p90', 'p95')) if d['count'] else '无'
        lines.append(f'| {label} | {item["denominator"]} | {pc.get("strict_overlap",0)} | {pc.get("touching",0)} | {pc.get("disjoint",0)} | {gap} |')
    lines += ['', '这个表把每个身份的所有片段两两配对，片段多的身份权重更大；其中可能包含两端之间已有其他同身份片段支撑的 pair，不能拿 pair 间距替代真实的联合观测空档。上节“并集内部空档”才对应所有同身份片段首末区间都未覆盖的内部时间。', '',
              '## 4. 全场景 Q64：自身不覆盖，并不都等于远期真值预测', '',
              f'严格按已有 `linspace(场景最早观测, 场景最晚观测, 64)` 重建；全部 {c["tracks"]} 轨的 query 支撑计数逐轨与原审计一致。分母是 **{g["denominator"]:,} 个 query×fragment 几何位置**，不是 reference 合法监督点；同身份真值仅用于事后几何分类，不是给模型提供 oracle 输入。', '',
              '| 几何类别 | query×fragment 数 | 占全部 | 占自身区间外 | 场景等权平均占全部 |',
              '| --- | ---: | ---: | ---: | ---: |']
    labels = {'self_support': '自身首末区间内', 'other_same_identity_support': '自身区间外、同身份其他片段支撑内',
              'identity_internal_union_gap': '同身份包络内、所有片段支撑并集之外',
              'outside_identity_envelope': '同身份所有观测包络之外'}
    for key, item in g['categories'].items():
        other = '—' if item['fraction_of_outside_self'] is None else pct(item['fraction_of_outside_self'])
        lines.append(f'| {labels[key]} | {item["count"]:,} | {pct(item["fraction_of_all"])} | {other} | {pct(item["scene_macro_fraction"])} |')
    lines += ['', '第二类合并了该身份在场景内的**全部**片段，是使用真实身份标签得到的oracle潜在辅助支撑上界。它不等于teacher实际采样的单轨或两轨已经获得这些观测，也不等于预测分组实际提供的helper支撑，且是在人工端点遮挡之前计算。仍不保证同一query有原生点、属性完整、reference真值可用、噪声可消除或关联可判对。第三类是内部支撑空档；第四类可能完全没有该目标 reference，**不能将它计为模型必须恢复的真实缺失**。这些训练几何比例不能和此前3个固定V场景的合法监督比例混为一个分母。', '',
              '| 自身区间外类别 | 到自身区间最近距离中位数（秒） | P90 | P95 |',
              '| --- | ---: | ---: | ---: |']
    for key, d in g['outside_self_distance_to_own_span_seconds'].items():
        lines.append(f'| {labels[key]} | {val(d["median"])} | {val(d["p90"])} | {val(d["p95"])} |')
    lines += ['', '全人口 query×fragment 会让片段多的身份贡献更多位置；场景等权列只消除了场景轨数权重，也不是实际正负均衡 pair 采样权重。JSON另保留 query×identity 计数，避免把两种人口混淆。', '',
              '## 5. 窗口候选阈值的描述性比较', '',
              '| 时间跨度 H（秒） | 固定64点时相邻间隔 | 片段完整跨度≤H | 不重叠同身份pair间距≤H | 并集内部空档≤H |',
              '| --- | ---: | ---: | ---: | ---: |']
    t = r['thresholds_seconds']
    for h in HORIZONS:
        vals = [pct(t[key][str(h)]['fraction_le']) for key in ('track_support_duration', 'disjoint_same_identity_pair_gap', 'identity_internal_union_gap')]
        lines.append(f'| {h} | {h/63:.2f}秒 | {vals[0]} | {vals[1]} | {vals[2]} |')
    lines += ['', f'三列分母分别是 {c["tracks"]} 条片段、{r["same_identity_unordered_pairs"]["all"]["disjoint_gap_seconds"]["count"]} 对不重叠同身份 pair、{c["identity_internal_union_gaps"]} 个并集内部空档。',
              '“片段跨度≤H”不是实际滑窗保留率：跨越固定窗边界仍可能被切分。“pair间距≤H”只是一个窗可能碰到两端支撑的必要条件，窗口起点、需要多少观测上下文、首尾预测规则都影响实际可用性。并集空档≤H不等于窗口内能带足两侧观测；也没有核验其 reference。**本表不是窗口实验或最优窗长选择。**', '',
              '## 6. 能得出的判断与还缺的证据', '',
              '- 当前全场景多数接近2小时，却仅64个query；单片段覆盖、同身份全部输入的oracle联合覆盖、实际训练/预测组可见覆盖和reference生命周期是不同范围。46.42%仅表示潜在辅助支撑，不能推出单轨self、某个训练正对或某个预测组已经获得支撑；其余包络外query也不一定有真值。不能把所有片段外位置统称远期真值预测，亦不能反过来称其中大部分已经有实际helper。',
              '- 21分钟候选的直接来源仍然只有 `63×20秒`。原始采样和支撑统计可帮助讨论局部上下文、可拼接空档与细粒度query成本，不足以宣称论文标准或最佳方案。当前不自动设置任何训练窗口。',
              '- 判断新任务是否合理，必须先决定希望恢复的是“目标观测包络内中断和相邻延续”，还是整场景任意远端目标；还要定义窗外上下文与跨窗身份计分。它们是研究任务定义，不能由端点分位数代替决定。',
              '- 已有元数据不能给出逐窗真实点数、先切窗再cap的保留率、reference合法监督距离、属性缺失、机动变化或训练速度。本轮不为补齐这些缺口重扫ZIP、读reference或跑新实验。',
              '- 017继续原协议；本报告不改变baseline/候选的时间窗、采样、预算、损失或选模。', '',
              '## 复算', '',
              '命令：', '',
              '```sh',
              'python docs/results/time_task_analysis20261019/train_metadata_analysis.py',
              '```', '',
              f'运行完成，{r["elapsed_seconds"]:.3f}秒。训练人口、每轨Q64支撑计数、既有跨度/间隔分位数、所有类别分割与逐源原生间隔分母校验通过。', '',
              f'- 元数据：`{r["sources"]["metadata"]}`',
              f'- 既有逐点统计：`{r["sources"]["prior_audit"]}`',
              '- 完整数值与限制：`train_metadata_analysis.json`；脚本：`train_metadata_analysis.py`。', '']
    (OUT / 'train_metadata_analysis.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
