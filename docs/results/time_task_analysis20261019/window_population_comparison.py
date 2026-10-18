"""T_core interval-only window populations; no ZIP, reference, V, or model access.

Run with Python >=3.10. This is a proposal comparison, not a windowed loader.
Only the first 512 metadata records are parsed; each must declare T_core.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import heapq
from itertools import combinations
import json
import math
from pathlib import Path
import time


SOURCE = Path(__file__).resolve().parents[3] / 'docs/results/frozen_input_audit_20261018/native_sampling/attempt2/scene_metadata.jsonl.gz'
STRATA = ('same_source__overlap', 'same_source__nonoverlap',
          'cross_source__overlap', 'cross_source__nonoverlap')
SCHEMES = ((300, 150), (600, 300), (1260, 640))


def choose2(count):
    return count*(count-1)//2


def windows(tracks, width, step):
    if not tracks:
        return []
    first = 20*math.floor(min(t['support_start'] for t in tracks)/20)
    last = max(t['support_stop'] for t in tracks)
    result = [(first, first+width)]
    while result[-1][1] < last:
        start = first+len(result)*step
        result.append((start, start+width))
    return result


def membership(tracks, bounds):
    # Closed intervals: endpoint-only intersection counts as inclusion.
    masks = [0]*len(tracks)
    populations = []
    for index, (start, stop) in enumerate(bounds):
        members = [i for i, t in enumerate(tracks)
                   if t['support_start'] <= stop and t['support_stop'] >= start]
        for i in members:
            masks[i] |= 1 << index
        populations.append(members)
    if tracks and any(mask == 0 for mask in masks):
        raise AssertionError('a valid original track was omitted')
    return masks, populations


def pair_stratum(a, b):
    source = 'same_source' if a['source'] == b['source'] else 'cross_source'
    overlap = max(a['support_start'], b['support_start']) <= min(a['support_stop'], b['support_stop'])
    return source + ('__overlap' if overlap else '__nonoverlap')


def pair_counts(tracks):
    """All/positive/same-source counts plus overlap sweep; no all-pair array."""
    by_id, by_source, by_id_source = Counter(), Counter(), Counter()
    for t in tracks:
        by_id[t['identity_id']] += 1
        by_source[t['source']] += 1
        by_id_source[(t['identity_id'], t['source'])] += 1
    all_total = choose2(len(tracks))
    positive = sum(map(choose2, by_id.values()))
    same = sum(map(choose2, by_source.values()))
    positive_same = sum(map(choose2, by_id_source.values()))
    heap, active_id, active_source, active_id_source = [], Counter(), Counter(), Counter()
    overlap_all = overlap_positive = overlap_same = overlap_positive_same = 0
    for serial, t in sorted(enumerate(tracks), key=lambda x: (x[1]['support_start'], x[0])):
        start, stop, identity, source = t['support_start'], t['support_stop'], t['identity_id'], t['source']
        while heap and heap[0][0] < start:  # End == start is an overlap.
            _, _, old_id, old_source = heapq.heappop(heap)
            active_id[old_id] -= 1
            active_source[old_source] -= 1
            active_id_source[(old_id, old_source)] -= 1
        overlap_all += len(heap)
        overlap_positive += active_id[identity]
        overlap_same += active_source[source]
        overlap_positive_same += active_id_source[(identity, source)]
        heapq.heappush(heap, (stop, serial, identity, source))
        active_id[identity] += 1
        active_source[source] += 1
        active_id_source[(identity, source)] += 1
    totals = dict(zip(STRATA, (overlap_same, same-overlap_same,
        overlap_all-overlap_same, all_total-same-overlap_all+overlap_same)))
    positives = dict(zip(STRATA, (overlap_positive_same, positive_same-overlap_positive_same,
        overlap_positive-overlap_positive_same,
        positive-positive_same-overlap_positive+overlap_positive_same)))
    negatives = {key: totals[key]-positives[key] for key in STRATA}
    assert all(value >= 0 for group in (positives, negatives) for value in group.values())
    return {'positive': positives, 'negative': negatives}


def distribution(values):
    data = sorted(values)
    if not data:
        return {'count': 0}
    def quantile(q):
        index = (len(data)-1)*q
        lo, hi = math.floor(index), math.ceil(index)
        return data[lo]+(data[hi]-data[lo])*(index-lo)
    return dict(count=len(data), min=data[0], p05=quantile(.05), median=quantile(.5),
                p95=quantile(.95), max=data[-1], mean=sum(data)/len(data))


def self_check():
    def track(a, b, identity='A', source='9001'):
        return dict(support_start=a, support_stop=b, identity_id=identity, source=source)
    assert windows([track(100, 200)], 300, 150) == [(100, 400)]
    assert windows([track(0, 300)], 300, 150) == [(0, 300)]
    tracks = [track(0, 1), track(300, 301)]
    masks, _ = membership(tracks, windows(tracks, 300, 150))
    assert masks == [1, 3] and (masks[0] & masks[1]).bit_count() == 1
    tracks = [track(0, 1), track(301, 302)]
    masks, _ = membership(tracks, windows(tracks, 300, 150))
    assert masks == [1, 2] and masks[0] & masks[1] == 0
    tracks = [track(0, 600), track(300, 301)]
    assert membership(tracks, windows(tracks, 300, 150))[0] == [7, 7]
    assert windows([track(-1, 1)], 300, 150) == [(-20, 280)]
    assert pair_stratum(track(0, 5), track(5, 10)).endswith('__overlap')
    tracks = [track(0, 1), track(1000, 1001)]
    _, groups = membership(tracks, windows(tracks, 300, 150))
    assert any(not group for group in groups)
    fixture = [track(0, 5), track(5, 10, source='9002'), track(11, 12),
               track(2, 7, identity='B'), track(20, 21, identity='B', source='9002')]
    brute = {kind: Counter({key: 0 for key in STRATA}) for kind in ('positive', 'negative')}
    # Only this five-track hand fixture is exhaustively enumerated.
    for a, b in combinations(fixture, 2):
        kind = 'positive' if a['identity_id'] == b['identity_id'] else 'negative'
        brute[kind][pair_stratum(a, b)] += 1
    assert pair_counts(fixture) == brute
    return {'status': 'PASS', 'hand_checks': 9,
            'cases': ['one window', 'first covering final window', 'closed endpoint intersection',
                      'non-cooccurring positive', 'repeated windows', 'negative start floor',
                      'touching intervals overlap', 'empty window accounting', 'sweep vs five-track enumeration']}


def read_training_metadata(path):
    scenes = []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        # Do not request the next record after the known 512 T_core rows.
        for index in range(512):
            line = next(stream)
            scene = json.loads(line)
            if scene['split'] != 'T_core':
                raise ValueError(f'row {index} is not T_core; stop without processing it')
            tracks = scene['tracks']
            assert len(tracks) == scene['valid_tracks']
            assert len({t['node_id'] for t in tracks}) == len(tracks)
            for t in tracks:
                assert math.isfinite(t['support_start']) and math.isfinite(t['support_stop'])
                assert t['support_stop'] >= t['support_start'] and t['native_points'] >= 2
            scenes.append(scene)
    assert len({scene['scene_id'] for scene in scenes}) == 512
    return scenes


def compare(scenes):
    original = {kind: Counter({key: 0 for key in STRATA}) for kind in ('positive', 'negative')}
    for scene in scenes:
        counts = pair_counts(scene['tracks'])
        assert sum(counts['positive'].values()) == scene['positive_pairs']
        assert sum(counts['negative'].values()) == scene['negative_pairs']
        for kind in original:
            original[kind].update(counts[kind])
    track_count = sum(len(scene['tracks']) for scene in scenes)
    result = dict(original=dict(scenes=len(scenes), valid_tracks=track_count,
        positive_pairs=sum(original['positive'].values()), negative_pairs=sum(original['negative'].values()),
        pair_strata=original), schemes=[])
    for width, step in SCHEMES:
        repeated = {kind: Counter({key: 0 for key in STRATA}) for kind in ('positive', 'negative')}
        retained = Counter({key: 0 for key in STRATA})
        positive_multiplicity = Counter()
        track_multiplicities, ntracks, positives, negatives, scene_counts, details = [], [], [], [], [], []
        scheduled = empty = 0
        for scene in scenes:
            tracks = scene['tracks']
            scene_retained_positive = 0
            bounds = windows(tracks, width, step)
            masks, populations = membership(tracks, bounds)
            scheduled += len(bounds)
            track_multiplicities.extend(mask.bit_count() for mask in masks)
            by_identity = defaultdict(list)
            for index, track in enumerate(tracks):
                by_identity[track['identity_id']].append(index)
            for indices in by_identity.values():
                for i, j in combinations(indices, 2):
                    multiplicity = (masks[i] & masks[j]).bit_count()
                    positive_multiplicity[multiplicity] += 1
                    if multiplicity:
                        scene_retained_positive += 1
                        retained[pair_stratum(tracks[i], tracks[j])] += 1
            rows = []
            for (start, stop), indices in zip(bounds, populations):
                if not indices:
                    empty += 1
                    continue
                counts = pair_counts([tracks[i] for i in indices])
                for kind in repeated:
                    repeated[kind].update(counts[kind])
                pos, neg = sum(counts['positive'].values()), sum(counts['negative'].values())
                ntracks.append(len(indices)); positives.append(pos); negatives.append(neg)
                rows.append([start, stop, len(indices), pos, neg])
            scene_counts.append(len(rows))
            details.append(dict(scene_id=scene['scene_id'], original_tracks=len(tracks),
                scheduled_windows=len(bounds), empty_windows=len(bounds)-len(rows),
                original_positive_pairs=scene['positive_pairs'],
                positive_pairs_co_window_at_least_once=scene_retained_positive,
                positive_pair_retention_fraction=(scene_retained_positive/scene['positive_pairs']
                                                  if scene['positive_pairs'] else None), windows=rows))
        original_pos = sum(original['positive'].values())
        kept = sum(retained.values())
        assert sum(positive_multiplicity.values()) == original_pos
        assert sum(k*v for k, v in positive_multiplicity.items()) == sum(repeated['positive'].values())
        assert len(track_multiplicities) == track_count and min(track_multiplicities) >= 1
        assert sum(track_multiplicities) == sum(ntracks)
        retention = {}
        scene_retention = [item['positive_pair_retention_fraction'] for item in details
                           if item['positive_pair_retention_fraction'] is not None]
        for key in STRATA:
            count = original['positive'][key]
            retention[key] = dict(original=count, co_window_at_least_once=retained[key],
                excluded_from_local_task=count-retained[key],
                retained_fraction=retained[key]/count if count else None,
                repeated_window_presentations=repeated['positive'][key])
        result['schemes'].append(dict(width_seconds=width, step_seconds=step,
            scheduled_windows=scheduled, nonempty_windows=len(ntracks), empty_windows=empty,
            window_count_per_scene=distribution(scene_counts),
            original_tracks_with_window=track_count, track_copies=sum(ntracks),
            track_copy_multiplier=sum(ntracks)/track_count,
            per_original_track_window_count=distribution(track_multiplicities),
            per_window_tracks=distribution(ntracks), per_window_positive_pairs=distribution(positives),
            per_window_negative_pairs=distribution(negatives),
            windows_with_no_positive_pairs=sum(value == 0 for value in positives),
            windows_with_no_negative_pairs=sum(value == 0 for value in negatives),
            repeated_positive_pairs=sum(positives), repeated_negative_pairs=sum(negatives),
            repeated_pair_strata=repeated, unique_original_positive_pairs=original_pos,
            positive_pairs_co_window_at_least_once=kept, positive_pairs_excluded=original_pos-kept,
            positive_pair_retention_fraction=kept/original_pos,
            positive_pair_exclusion_fraction=1-kept/original_pos,
            positive_pair_retention_scene_equal_mean=sum(scene_retention)/len(scene_retention),
            positive_pair_retention_scene_equal_population=len(scene_retention),
            positive_pair_retention_per_scene=distribution(scene_retention),
            positive_retention_by_stratum=retention,
            positive_pair_window_multiplicity={str(k): v for k, v in sorted(positive_multiplicity.items())},
            window_row_columns=['start', 'stop', 'track_count', 'positive_pairs', 'negative_pairs'],
            scenes=details))
    return result


def render(result, json_path):
    base = result['original']
    lines = ['# 共同窗口的人口变化：仅T_core端点区间离线比较', '',
        '2026-10-19。讨论材料，不是已批准时间协议或训练配置。只读取既有元数据前512条T_core场景，未读取后续V记录；未开原ZIP/reference/T_test，未调用模型、训练或GPU。', '',
        f"原人口：{base['scenes']}场景、{base['valid_tracks']:,}条有效片段、{base['positive_pairs']:,}正对、{base['negative_pairs']:,}负对。三套均核实每条原轨至少进入一窗。", '',
        '起点为20×floor(场景最早端点/20)，末窗为第一个覆盖最晚端点的窗；闭区间相交即纳入，端点恰落窗边也算。空人口窗单独计数后不构造样本。窗口内正负对含跨窗重复；“保留正对”按原scene+原片段对去重。', '',
        '| 窗长/步长 | 非空窗 | 轨副本/原轨 | 每窗轨数p50/p95 | 正对副本 | 负对副本 | 原正对至少共窗（pair加权） | 场景等权共窗率 | 原正对不共窗 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in result['schemes']:
        lines.append(f"| {s['width_seconds']}/{s['step_seconds']}秒 | {s['nonempty_windows']:,} | {s['track_copy_multiplier']:.2f}× | {s['per_window_tracks']['median']:g}/{s['per_window_tracks']['p95']:g} | {s['repeated_positive_pairs']:,} | {s['repeated_negative_pairs']:,} | {s['positive_pairs_co_window_at_least_once']:,}（{s['positive_pair_retention_fraction']:.2%}） | {s['positive_pair_retention_scene_equal_mean']:.2%} | {s['positive_pairs_excluded']:,}（{s['positive_pair_exclusion_fraction']:.2%}） |")
    lines += ['', 'pair加权率按全部原正对合并计算；场景等权率先算每个场景的共窗正对/原正对，再对512个正对非空场景取均值。两者均是结构覆盖，后者更贴近原场景均匀选择的权重，但仍不是有限训练中的实际抽样频率。']
    lines += ['', '## 哪些原正对不再属于局部任务', '',
        '下表分层由原始端点区间是否相交和原source字段定义，包含接触于同一端点的重叠。比例分母为该层原正对；不共窗不是新负标签，也不证明这些原正对不合理。', '',
        '| 原正对分层 | 原数量 | 300/150共窗率 | 600/300共窗率 | 1260/640共窗率 |',
        '|---|---:|---:|---:|---:|']
    labels = {'same_source__overlap': '同源·原重叠', 'same_source__nonoverlap': '同源·原不重叠',
              'cross_source__overlap': '跨源·原重叠', 'cross_source__nonoverlap': '跨源·原不重叠'}
    for key in STRATA:
        rates = [s['positive_retention_by_stratum'][key]['retained_fraction'] for s in result['schemes']]
        lines.append('| '+labels[key]+f" | {base['pair_strata']['positive'][key]:,} | "+' | '.join(f'{value:.2%}' if value is not None else '无原样本' for value in rates)+' |')
    lines += ['', '## 每窗候选规模与重复', '',
        '| 窗长/步长 | 每窗正对p50/p95 | 每窗负对p50/p95 | 无正对窗 | 无负对窗 | 空人口窗 | 每原轨入窗p50/p95/max |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in result['schemes']:
        p,n,t=s['per_window_positive_pairs'],s['per_window_negative_pairs'],s['per_original_track_window_count']
        lines.append(f"| {s['width_seconds']}/{s['step_seconds']}秒 | {p['median']:g}/{p['p95']:g} | {n['median']:g}/{n['p95']:g} | {s['windows_with_no_positive_pairs']:,} | {s['windows_with_no_negative_pairs']:,} | {s['empty_windows']:,} | {t['median']:g}/{t['p95']:g}/{t['max']:g} |")
    lines += ['', '## 可以和不能据此判断什么', '',
        '- 这些计数衡量任务人口、跨窗重复和原正对接续范围的变化。局部窗保留全部原轨，不等于保留全部原正对，也不等于有限训练更新遍历全部轨迹。',
        '- 端点区间相交不证明窗内有真实观测；元数据没有逐点时间，因此不能计算窗内实际点数、补入窗外上下文数量、cap后的覆盖或reference监督覆盖。',
        '- 轨副本倍率和窗口数不是GPU小时倍率。固定batch/更新数的训练成本与遍历所有窗口的终评成本结构不同；两级场景→窗口采样下的实际呈现频率还取决于候选资格与采样，不能用本表推算正式训练费用。',
        '- 不按任何模型分数挑窗口。300/600秒仅比较端点人口，未设定这两套的query数/间隔；不能默认它们继承1260秒方案的64点×20秒。',
        '- 原重叠正对100%共窗可由窗口覆盖和闭区间规则解释；其他层的排除率直接反映接续任务被缩窄，不能写成模型性能改善。', '',
        '## 复算与边界检查', '',
        '全候选数与正对数按身份/来源组合计数；重叠分层使用端点扫线；仅在同身份组内枚举原正对，未生成全负N²数组。9个手工边界检查通过，包括首末窗、边界接触、负起点、空窗和五轨计数对照。', '',
        f"聚合计数和分层副本计数在[{json_path.name}]({json_path.name})，逐场景逐窗明细保存在本地研究证据包；来源文件未修改。",
        '', '```bash', f'python3 {Path(__file__).resolve()}', '```', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=SOURCE)
    args = parser.parse_args()
    started = time.perf_counter()
    checks = self_check()
    scenes = read_training_metadata(args.metadata)
    result = compare(scenes)
    result.update(status='COMPLETED', scope='T_core interval-only proposal population comparison',
        source_metadata=str(args.metadata.resolve()), parsed_metadata_records=512,
        source_record_policy='first 512 records only, all asserted T_core; no next record requested',
        original_zip_opened=False, reference_opened=False, validation_records_parsed=0, test_opened=False,
        optimizer_updates=0, gpu_used=False, self_checks=checks,
        closed_interval_inclusion=True, elapsed_cpu_wall_seconds=time.perf_counter()-started,
        limits=['no pointwise times', 'no reference availability', 'not GPU cost', 'not a training configuration'])
    # Keep Git evidence compact; detailed rows remain in the local research evidence.
    for scheme in result['schemes']:
        scheme.pop('scenes')
    result['detail_policy'] = 'Aggregate output; per-window rows omitted from Git report.'
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    output.with_suffix('.md').write_text(render(result, output), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_cpu_wall_seconds', 'parsed_metadata_records')}, ensure_ascii=False))
    print(json.dumps([{k:s[k] for k in ('width_seconds','step_seconds','nonempty_windows','track_copy_multiplier','positive_pair_retention_fraction')} for s in result['schemes']], ensure_ascii=False))


if __name__ == '__main__':
    main()
