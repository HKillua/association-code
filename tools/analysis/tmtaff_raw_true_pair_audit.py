"""Reconstruct C90-R1 inputs and supervision independently from eight raw T scenes."""

import argparse
from collections import defaultdict
import csv
import io
from itertools import combinations
import json
from pathlib import Path, PurePosixPath
import re
import resource
import time

import numpy as np
import torch

from association.data.archive import open_mtad_zip


SCENES = ('0', '3', '7', '9', '10', '11', '14', '17')
ATTRS = ('lat', 'lon', 'vel', 'cou')
CAP = 128
LENGTH = 64
TIME_UNIT = 20.
KNOT_TO_MPS = 1852. / 3600.
RUN_107 = 'LEVEL1-TMTAFF-UNIT-BASELINE-A-20261027-107-attempt-1'
RUN_127 = 'LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1'
RUN_153 = 'LEVEL1-TMTAFF-T-PHYSICAL-PROBE-20261030-153-attempt-1'
RUN_C90 = 'LEVEL1-TMTAFF-TRUE-PAIR-QUALIFICATION-20261112-R1-attempt-1'
MEMBERS = {
    'labels': ('关联表', '关联结果', ('t_s', 't_e', 'mmsi', 'source', 'batch')),
    'measurements': ('量测场景', '场景', ('batch', 'source', 'time', *ATTRS)),
    'reference': ('真实场景', '场景', ('MMSI', 'time', *ATTRS)),
}


def require(condition, description):
    if not condition:
        raise ValueError(description)


def exact(actual, expected, description):
    actual, expected = np.asarray(actual), np.asarray(expected)
    same = np.array_equal(actual, expected, equal_nan=actual.dtype.kind == 'f')
    require(same, description)


def locate_members(archive):
    requested = {(folder, f'{prefix}-{sid}.csv'): (kind, sid)
                 for kind, (folder, prefix, _) in MEMBERS.items() for sid in SCENES}
    found = {}
    for info in archive.infolist():
        filename = info.filename
        if not info.flag_bits & 0x800:
            try:
                filename = filename.encode('cp437').decode('gbk')
            except UnicodeError:
                continue
        path = PurePosixPath(filename)
        key = requested.get(path.parts[-2:])
        if key is not None:
            require(key not in found and bool(re.search(r'(?<![a-z0-9])level1(?![a-z0-9])',
                                                       '/'.join(path.parts[:-2]), re.I)),
                    'raw member ambiguity or unexpected dataset level')
            found[key] = info
    require(len(found) == len(requested), 'a requested raw member is absent')
    return found


def read_rows(archive, members, kind, sid):
    payload = archive.read(members[kind, sid])
    try:
        content = payload.decode('utf-8-sig')
    except UnicodeDecodeError:
        content = payload.decode('gb18030')
    reader = csv.reader(io.StringIO(content, newline=''), strict=True)
    fields = MEMBERS[kind][2]
    exact(next(reader), fields, f'{sid} {kind} CSV header')
    result = [tuple(value.strip() for value in row) for row in reader]
    require(all(len(row) == len(fields) for row in result), f'{sid} {kind} CSV row length')
    return result


def number(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def physical(values):
    lat, lon, speed, heading = np.moveaxis(values, -1, 0)
    return np.isfinite(values) & np.stack((abs(lat) <= 90, abs(lon) <= 180,
                                          speed >= 0, (heading >= 0) & (heading <= 360)), axis=-1)


def canonicalize(labels, observations):
    label_map, track_map = defaultdict(set), defaultdict(list)
    targets = {}
    for start, stop, label, source, batch in labels:
        require(source and batch and len(label.split('-')) == 3 and all(label.split('-'))
                and np.isfinite([number(start), number(stop)]).all()
                and number(start) <= number(stop), 'invalid raw label')
        label_map[source, batch].add(label)
        targets.setdefault(label.rsplit('-', 1)[0], None)
    for batch, source, *row in observations:
        require(source and batch, 'empty source or batch')
        track_map[source, batch].append(tuple(map(number, row)))
    tracks = []
    for key in sorted(label_map.keys() | track_map.keys()):
        rows, identities = track_map[key], label_map[key]
        if len(rows) < 2 or len(identities) != 1:
            continue
        points = np.asarray(rows)
        if not np.isfinite(points[:, 0]).all() or not physical(points[:, 1:]).all():
            continue
        points = np.unique(points, axis=0)
        if len(points) < 2 or (np.diff(points[:, 0]) <= 0).any():
            continue
        points[:, 4] %= 360.
        tracks.append((*key, next(iter(identities)).rsplit('-', 1)[0], points))
    return tracks, tuple(targets)


def cap_indices(count):
    size = min(count, CAP)
    return np.arange(size, dtype=np.int64) * (count - 1) // (size - 1)


def packed_input(tracks, grid, normalizer):
    pack = np.zeros((len(tracks), CAP + LENGTH, 5), dtype=np.float64)
    native = np.zeros(pack.shape[:2], dtype=bool)
    supported = np.zeros((len(tracks), LENGTH), dtype=bool)
    mean, std = np.asarray(normalizer['mean']), np.asarray(normalizer['std'])
    origin = grid[0]
    for index, (_, _, _, points) in enumerate(tracks):
        selected = points[cap_indices(len(points))]
        times, values = selected[:, 0], selected[:, 1:].copy()
        values[:, 2] *= KNOT_TO_MPS
        count = len(times)
        native[index, :count] = True
        pack[index, :count, :4] = (values - mean) / std
        pack[index, :count, 4] = (times - origin) / TIME_UNIT
        pack[index, CAP:, 4] = (grid - origin) / TIME_UNIT
        pack[index, CAP:, :4] = np.nan
        inside = (times[0] <= grid) & (grid <= times[-1])
        supported[index] = inside
        queries = grid[inside]
        right = np.searchsorted(times, queries)
        left = np.maximum(0, right - 1)
        fraction = np.divide(queries - times[left], times[right] - times[left],
                             out=np.zeros_like(queries), where=right != left)
        delta = values[right] - values[left]
        for column in (1, 3):
            wrapped = (delta[:, column] + 180.) % 360. - 180.
            delta[:, column] = np.where(abs(wrapped) == 180., np.copysign(180., delta[:, column]), wrapped)
        interpolated = values[left] + fraction[:, None] * delta
        interpolated[:, 1] = (interpolated[:, 1] + 180.) % 360. - 180.
        interpolated[:, 3] %= 360.
        hit = times[right] == queries
        interpolated[hit] = values[right[hit]]
        pack[index, CAP:, :4][inside] = (interpolated - mean) / std
    return pack.astype(np.float32), native, supported


def reference_targets(rows, registry, grid, offset, max_gap):
    grouped = defaultdict(lambda: defaultdict(list))
    for identity, raw_time, *state in rows:
        timestamp = number(raw_time) - offset
        if identity in registry and np.isfinite(timestamp):
            grouped[identity][timestamp].append(tuple(map(number, state)))
    report = defaultdict(int)
    targets = np.full((len(registry), LENGTH, 4), np.nan)
    for i, identity in enumerate(registry):
        groups = grouped[identity]
        times = np.asarray(sorted(groups))
        if not len(times):
            report['no_reference_queries'] += LENGTH
            continue
        values = np.full((len(times), 4), np.nan)
        for j, timestamp in enumerate(times):
            samples = np.asarray(groups[timestamp])
            valid = physical(samples).all(axis=0) & (samples == samples[0]).all(axis=0)
            values[j, valid] = samples[0, valid]
        values[values[:, 3] == 360., 3] = 0.
        for q, query in enumerate(grid):
            right = int(np.searchsorted(times, query))
            if right < len(times) and times[right] == query:
                report['exact_queries'] += 1
                targets[i, q] = values[right]
            elif right == 0 or right == len(times):
                report['outside_reference_queries'] += 1
            else:
                left = right - 1
                gap = times[right] - times[left]
                if gap > max_gap:
                    report['gap_blocked_queries'] += 1
                    continue
                report['interpolated_queries'] += 1
                fraction = (query - times[left]) / gap
                a, b = values[left], values[right]
                state = (1. - fraction) * a + fraction * b
                if abs(b[1] - a[1]) > 180.:
                    delta = (b[1] - a[1] + 180.) % 360. - 180.
                    state[1] = (a[1] + fraction * delta + 180.) % 360. - 180.
                delta = (b[3] - a[3] + 180.) % 360. - 180.
                state[3] = np.nan if abs(delta) == 180. else (a[3] + fraction * delta) % 360.
                targets[i, q] = state
    return targets, np.isfinite(targets), dict(report)


def geodesic_speed_ratio(points):
    """Use raw speed over spherical secant speed; discard tiny distances."""
    dt = np.diff(points[:, 0])
    lat, lon = np.deg2rad(points[:, 1:3].T)
    chord = np.sin(np.diff(lat) / 2.) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2.) ** 2
    distance = 2. * 6371000. * np.arcsin(np.sqrt(np.clip(chord, 0., 1.)))
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = points[:-1, 3] / (distance / dt)
    valid = (dt > 0.) & (distance > .1) & np.isfinite(ratio) & (ratio >= 0.)
    return ratio[valid]


def audit_scene(sid, archive, members, runs, normalizer, saved, policy, c90, pair_slice):
    # Use production adapter only as the comparison path on already-saved inputs.
    from tools.analysis.tmtaff_true_pair_qualification import timed_scene
    from association.data.timed import pack_timed_scene

    labels = read_rows(archive, members, 'labels', sid)
    observations = read_rows(archive, members, 'measurements', sid)
    reference = read_rows(archive, members, 'reference', sid)
    tracks, targets = canonicalize(labels, observations)
    sources = np.asarray([row[0] for row in tracks])
    identities = np.asarray([row[2] for row in tracks])
    nodes = np.asarray([f'{sid}:{source}:{batch}' for source, batch, _, _ in tracks])
    intervals = np.array([[points[0, 0], points[-1, 0]] for _, _, _, points in tracks])
    grid = np.linspace(intervals[:, 0].min(), intervals[:, 1].max(), LENGTH)
    pairs = np.asarray([(a, b) for a, b in combinations(range(len(nodes)), 2)
                        if sources[a] != sources[b] and identities[a] == identities[b]][:2])
    require(pairs.shape == (2, 2), f'{sid} no exact first-two eligible raw pairs')
    offset, max_gap = (policy[name] for name in ('time_offset_seconds', 'max_gap_seconds'))
    target_set = set(targets)
    minimum = min(number(row[1]) for row in reference if row[0] in target_set and np.isfinite(number(row[1])))
    require(offset == minimum and max_gap == 600.
            and all(saved['reference_alignment'][key] == policy[key] for key in policy)
            and saved['reference_alignment']['longitude_policy'] == 'shortest_arc', f'{sid} raw clock contract')
    pack, native_mask, supported = packed_input(tracks, grid, normalizer)
    require(saved['input_adapter'] == 'native_time_v1' and saved['split'] == 'T_core'
            and saved['max_observations'] == CAP and saved['time_unit_seconds'] == TIME_UNIT
            and saved['time_origin_seconds'] == grid[0]
            and saved['physical_units'] == normalizer['physical_units'], f'{sid} 107 physical input contract')
    for key, actual in dict(node_ids=nodes, sources=sources, identities=identities, intervals=intervals,
                            grid=grid, observed_counts=supported.sum(1),
                            native_observed_counts=native_mask.sum(1), target_ids=targets).items():
        exact(actual, saved[key], f'{sid} 107 {key}')
    file = f'scene-{sid}.npz'
    parent = runs / RUN_127 / 'private'
    points = np.concatenate([row[3] for row in tracks])
    sizes = [len(row[3]) for row in tracks]
    with np.load(parent / 'native-events/T_core' / file, allow_pickle=False) as events:
        for key, actual in dict(node_ids=nodes, sources=sources, query_seconds=grid,
                                event_offsets=np.r_[0, np.cumsum(sizes)], event_times_seconds=points[:, 0],
                                event_values=points[:, 1:], event_valid=np.ones((len(points), 4), dtype=bool),
                                event_node_ids=np.repeat(nodes, sizes),
                                event_canonical_indices=np.concatenate([np.arange(size) for size in sizes])).items():
            exact(actual, events[key], f'{sid} 127 raw {key}')
        reference_scene = timed_scene(events, identities, sid)
        production_pack, production_mask = pack_timed_scene(reference_scene, normalizer)
        exact(pack, production_pack, f'{sid} C90 normalized raw pack')
        exact(native_mask, production_mask, f'{sid} C90 normalized raw mask')
        exact(supported, reference_scene['observed'], f'{sid} C90 raw observation support')
    with np.load(runs / RUN_153 / 'private' / f'predictions-{sid}.npz', allow_pickle=False) as old:
        exact(pairs, old['pairs'], f'{sid} 153 same-id cross-source pair ordering')
    chosen = np.array([targets.index(identities[pair[0]]) for pair in pairs])
    chosen_ids = tuple(dict.fromkeys(targets[index] for index in chosen))
    truth, valid, clock = reference_targets(reference, chosen_ids, grid, offset, max_gap)
    pair_targets = [chosen_ids.index(targets[index]) for index in chosen]
    truth, valid = truth[pair_targets], valid[pair_targets]
    with np.load(parent / 'native-evaluation/T_core' / file, allow_pickle=False) as saved_ref:
        for key, actual in dict(node_ids=nodes, identities=identities, query_seconds=grid, target_ids=targets).items():
            exact(actual, saved_ref[key], f'{sid} 127 evaluation {key}')
        exact(valid, saved_ref['target_valid'][chosen], f'{sid} 127 reference mask')
        exact(truth, saved_ref['target_states'][chosen], f'{sid} 127 reference states')
    support = supported[pairs].sum(axis=1)
    for key, actual in dict(query_seconds=np.broadcast_to(grid, (2, LENGTH)), target_states=truth,
                            target_valid=valid, selected_support=support).items():
        exact(actual, c90[key][pair_slice], f'{sid} C90 exact contract {key}')
    position = valid[..., :2].all(-1)
    per_pair = [dict(valid=mask.sum(0).tolist(), support_all=np.bincount(num, minlength=3).tolist(),
                     support_valid_position=np.bincount(num[ok], minlength=3).tolist())
                for mask, num, ok in zip(valid, support, position)]
    ratios = defaultdict(list)
    for source, _, _, track in tracks:
        ratios[source].append(geodesic_speed_ratio(track))
    references = defaultdict(list)
    for identity, raw_time, *state in reference:
        if identity in target_set:
            references[identity].append(tuple(map(number, [raw_time, *state])))
    for values in references.values():
        values = np.asarray(sorted(values, key=lambda row: row[0]))
        times, occurrences = np.unique(values[:, 0], return_counts=True)
        bad_times = times[occurrences > 1]
        invalid = ~np.isfinite(values[:, 0]) | np.isin(values[:, 0], bad_times) | ~physical(values[:, 1:]).all(axis=1)
        values[invalid, 1:] = np.nan
        ratios['reference'].append(geodesic_speed_ratio(values))
    return dict(scene_id=sid, labels=len(labels), measurements=len(observations), reference_rows=len(reference),
                canonical_nodes=len(nodes), canonical_points=len(points), selected_points=int(native_mask.sum()),
                pair_nodes=len(set(pairs.ravel())), distinct_pair_targets=len(chosen_ids),
                clock_offset_nonzero=bool(offset), target_query_clock=clock, pairs=per_pair), ratios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, nargs=3, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((args.metadata / 'mtad-level1-scene3000.json').read_text())
    require(tuple(map(str, manifest['splits']['T_core'][:8])) == SCENES, 'T_core order differs')
    alignment = json.loads((args.metadata / 'mtad-level1-alignment.json').read_text())
    checkpoint = torch.load(args.runs / RUN_107 / 'tcm/checkpoint.pt', map_location='cpu',
                            weights_only=True, mmap=True)
    rows = {str(row['scene_id']): row for row in checkpoint['data_contract'][0] if str(row['scene_id']) in SCENES}
    normalizer = checkpoint['normalizer']
    require(normalizer['feature_order'] == list(ATTRS) and normalizer['training_splits'] == ['T_core']
            and list(map(str, normalizer['training_scene_ids'])) == list(map(str, manifest['splits']['T_core']))
            and normalizer['physical_units']['policy'] == 'measurement_kn_reference_mps_v1'
            and normalizer['physical_units']['measurement_to_state_factor'] == KNOT_TO_MPS
            and normalizer['physical_units']['reference_speed_unit'] == 'mps', '107 normalizer contract')
    scenes, ratios = [], defaultdict(list)
    with open_mtad_zip(args.archive) as archive, np.load(args.runs / RUN_C90 / 'private/contract.npz',
                                                        allow_pickle=False) as contract:
        members = locate_members(archive)
        c90 = {name: contract[name] for name in ('query_seconds', 'target_states', 'target_valid', 'selected_support')}
        require(c90['target_valid'].shape == (16, LENGTH, 4), 'C90 contract population differs')
        for index, sid in enumerate(SCENES):
            scene, local_ratios = audit_scene(sid, archive, members, args.runs, normalizer, rows[sid], alignment[sid],
                                              c90, slice(2 * index, 2 * index + 2))
            scenes.append(scene)
            for source, values in local_ratios.items():
                ratios[source].extend(values)
    pair_rows = [pair for scene in scenes for pair in scene['pairs']]
    feature_valid = np.sum([pair['valid'] for pair in pair_rows], axis=0).tolist()
    support_counts = np.sum([pair['support_valid_position'] for pair in pair_rows], axis=0).tolist()
    require(feature_valid == [732, 732, 729, 732] and support_counts == [132, 299, 301]
            and sum(scene['pair_nodes'] for scene in scenes) == 27, 'frozen C90 denominator differs')
    stats = {source: {'intervals': len(values), 'median_raw_to_secant_mps': float(np.median(values))}
             for source, chunks in sorted(ratios.items()) if len(values := np.concatenate(chunks))}
    result = dict(status='PASSED', scene_ids=list(SCENES), pair_count=len(pair_rows), pair_queries=len(pair_rows)*LENGTH,
                  feature_valid=dict(zip(ATTRS, feature_valid)),
                  support_position_counts=support_counts, raw_archive_read=True, raw_member_count=len(members),
                  validation_opened=False, test_opened=False, c90_contract_exact=True,
                  equality='array_equal with corresponding NaNs; no numerical tolerance',
                  checkpoints={'normalizer': RUN_107, 'native_cache': RUN_127, 'pairs': RUN_153, 'supervision': RUN_C90},
                  speed={'measurement_kn_to_mps': KNOT_TO_MPS, 'reference_conversion': 'none',
                         'raw_speed_to_geometric_mps': stats}, scenes=scenes,
                  elapsed_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    args.output.parent.mkdir(mode=0o700, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'feature_valid', 'support_position_counts',
                                                 'elapsed_seconds', 'peak_rss_bytes')}))


if __name__ == '__main__':
    main()
