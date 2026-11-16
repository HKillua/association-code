"""C149: measure the original C74 B course branches on the frozen fit roster."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.motion_evidence import scene_motion_evidence
from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.mixture_opportunity import _distance
from association.evaluation.physical import _coverage
from tools.analysis.association_partition_disagreement import DATA, PREP
from tools.analysis.conditional_trajectory_pilot import mean_state
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive
from tools.analysis.motion_fallback_diagnostic import support_tags
from tools.analysis.motion_readout_evaluation import COVERAGE_KEYS, POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import readouts
from tools.analysis.offline_association_tail import read_motion, require, scene_split, summary_state
from tools.analysis.same_source_swap_pilot import canonical_membership, group_members, peak_rss_bytes, read, write


C139 = 'LEVEL1-OFFLINE-PARTITION-FIT-QUALIFICATION-20261114-attempt-1'
SPLIT = 'T_core'
GAP_SECONDS = 120.
MIN_DIRECTION_SPEED = .5
TAIL_DEG = 150.
RSS_LIMIT = 8 * 1024**3
BRANCHES = ('derivative', 'enabled_low_speed', 'outside_with_support', 'fewer_than_two_supported')
VECTOR_KEYS = ('error', 'branch', 'span', 'visible', 'far', 'interval_count', 'nonreuse_count',
               'cross_source_pair', 'same_identity_other_member', 'pure')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_inputs(runs, manifest_path):
    manifest = read(manifest_path)
    prep, data, partition = (runs / name for name in (PREP, DATA, C139))
    for parent in (prep, data, partition):
        status = read(parent / 'run.json')
        require(status['status'] == 'COMPLETED' and status.get('exit_code', 0) == 0,
                f'frozen parent {parent.name} did not complete')
    require(read(data / 'run.json')['exit_code'] == 0, '151 scientific exit was not zero')
    launcher_status = []
    for parent in (prep, partition):
        path = runs / parent.name.replace('-attempt-1', '-launch-attempt-1') / 'status.json'
        outcome = read(path)
        require(outcome['status'] == 'COMPLETED' and outcome['returncode'] == 0,
                f'{parent.name} launcher did not exit successfully')
        launcher_status.append(path)
    roles = read(prep / 'frozen-split.json')
    require(roles == scene_split(manifest['splits'][SPLIT], manifest['components']),
            'frozen C92 component roles differ')
    ids = roles['fit']
    components = {sid: str(manifest['components'][sid]) for sid in ids}
    require(len(ids) == len(components) == 1748 and len(set(components.values())) == 76,
            'C92 fit roster differs')
    digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    freeze = read(partition / 'freeze.json')
    require(freeze['scene_count'] == 1748 and freeze['component_count'] == 76
            and freeze['scene_order_sha256'] == digest
            and freeze['all_candidates_and_native_evidence_before_identity']
            and freeze['old_056_fit_scores_reused'] == len(ids) and not freeze['other_splits_opened'],
            'C139 blind WS roster differs')
    original_index = read(data / 'private/index.json')[SPLIT]
    index = {}
    for sid in ids:
        row = original_index[sid]
        require(str(row['component_id']) == components[sid], '151 component differs')
        index[sid] = {key: row[key] for key in ('motion', 'evaluation', 'node_count')}
        for key in ('motion', 'evaluation'):
            expected = data / 'private' / key / SPLIT / f'scene-{sid}.npz'
            require(Path(index[sid][key]).resolve() == expected.resolve(), f'151 {key} file binding differs')
        require((partition / 'private/blind' / f'{sid}.json.gz').is_file(), 'C139 blind WS file missing')
    metadata = (manifest_path, prep/'run.json', prep/'frozen-split.json', data/'run.json', data/'private/index.json',
                partition/'run.json', partition/'freeze.json', *launcher_status)
    return ids, components, index, partition, read(data / 'run.json')['archive'], digest, metadata


def native_inputs(archive, sid, motion, stored_states_mps):
    tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid,
                               motion['node_ids'], motion['sources'])
    query = motion['query_seconds']
    require(query.shape == (64,) and np.isfinite(query).all() and (np.diff(query) > 0.).all(),
            '151 Q64 binding differs')
    lengths = [len(row['canonical_times']) for row in tracks]
    np.testing.assert_array_equal(lengths, motion['native_count'])
    if not tracks:
        require(stored_states_mps.shape == (0, 64, 4), 'empty 151 motion shape differs')
        return tracks, (query, np.empty(0), np.empty((0, 2)), np.zeros(1, int))
    times = [row['canonical_times'] for row in tracks]
    np.testing.assert_array_equal(np.array([t[-1]-t[0] for t in times]), motion['span_seconds'])
    expected_support = np.array([(query >= t[0]) & (query <= t[-1]) for t in times])
    np.testing.assert_array_equal(expected_support, motion['support'])
    for t, expected in zip(times, motion['nearest_seconds'], strict=True):
        i = np.searchsorted(t, query)
        age = np.minimum(abs(query-t[i.clip(0, len(t)-1)]), abs(query-t[(i-1).clip(0, len(t)-1)]))
        np.testing.assert_array_equal(age, expected)
    seconds = np.concatenate(times)
    require(seconds.min() >= query[0]-1e-8 and seconds.max() <= query[-1]+1e-8,
            'native physical times escape the original Q64')
    rebuilt = scene_motion_evidence(dict(tracks=tracks, node_ids=motion['node_ids'], sources=motion['sources'],
                                         grid=query), motion_observation_policy='canonical_all')
    rebuilt['states'][..., 2] *= KNOT_TO_MPS
    np.testing.assert_array_equal(rebuilt['states'], stored_states_mps, err_msg=f'151 source states, scene {sid}')
    for key in ('sources', 'query_seconds', 'support', 'nearest_seconds', 'native_count', 'span_seconds',
                'fit_rmse_m', 'ols_velocity_mps', 'endpoint_velocity_mps'):
        np.testing.assert_array_equal(rebuilt[key], motion[key], err_msg=f'151 {key}, scene {sid}')
    return tracks, (query, seconds, np.concatenate([row['canonical_values'][:, :2] for row in tracks]),
                    np.r_[0, np.cumsum(lengths)])


def different_source_disjoint(left, right, sources):
    """Find two witnesses with four unique report keys and different source labels."""
    if len(set(sources.tolist())) < 2:
        return False
    for i in range(len(left)-1):
        a, b = left[i], right[i]
        if np.any((sources[i+1:] != sources[i]) & (left[i+1:] != a) & (right[i+1:] != a)
                  & (left[i+1:] != b) & (right[i+1:] != b)):
            return True
    return False


def native_group_evidence(query, tracks, position, active, fallback, support, source_ids):
    """Count short visible native intervals and conservatively exclude reused report operands."""
    members = len(tracks)
    times = [row['canonical_times'] for row in tracks]
    lengths = np.array([len(t) for t in times])
    coordinates = [row['canonical_values'][:, :2] for row in tracks]
    raw = np.concatenate([np.column_stack((t, xy)) for t, xy in zip(times, coordinates, strict=True)])
    _, codes = np.unique(raw, axis=0, return_inverse=True)
    codes = codes.reshape(-1)
    ids = np.split(codes, np.cumsum(lengths)[:-1])
    fit_used = np.zeros(int(codes.max())+1, bool)
    brackets, all_edges = [], []
    for member, (t, xy, key) in enumerate(zip(times, coordinates, ids, strict=True)):
        right = np.searchsorted(t, query).clip(1, len(t)-1)
        bracket = np.stack((key[right-1], key[right]), axis=-1)
        brackets.append(bracket)
        in_fit = active & support[member]
        fit_used[bracket[in_fit].ravel()] = True
        dt = np.diff(t)
        good = (dt <= GAP_SECONDS) & (_distance(xy[:-1], xy[1:]) >= MIN_DIRECTION_SPEED * dt)
        all_edges.append(np.column_stack((t[:-1][good], t[1:][good], key[:-1][good],
                                           key[1:][good], np.full(int(good.sum()), member))))
    brackets = np.asarray(brackets)
    # The historical circular longitude Mean can take the anchor's position.
    # If that anchor is outside support, its OLS longitude used all its positions.
    ang = np.deg2rad(position[..., 1])
    n = support.sum(axis=0).clip(1)
    resultant = np.hypot((np.sin(ang)*support).sum(axis=0)/n,
                         (np.cos(ang)*support).sum(axis=0)/n)
    anchor_lon = active & support.any(axis=0) & (resultant <= 1e-12)
    fit_used[brackets[0, anchor_lon].ravel()] = True
    if np.any(anchor_lon & ~support[0]):
        fit_used[ids[0]] = True

    edges = np.concatenate(all_edges)
    edges = edges[np.argsort(edges[:, 0], kind='stable')]
    source_codes = np.unique(source_ids, return_inverse=True)[1]
    count = np.zeros(len(query), np.int64)
    unused_count = np.zeros_like(count)
    cross = np.zeros(len(query), bool)
    unused_members = np.zeros((members, len(query)), bool)
    starts = edges[:, 0]
    lower = np.searchsorted(starts, query-GAP_SECONDS)
    upper = np.searchsorted(starts, query+GAP_SECONDS, side='right')
    for qi in np.flatnonzero(upper > lower):
        candidates = edges[lower[qi]:upper[qi]]
        candidates = candidates[candidates[:, 1] <= query[qi]+GAP_SECONDS]
        left, right, providers = candidates[:, 2:].astype(np.int64).T
        count[qi] = len(np.unique(np.column_stack((left, right)), axis=0))
        keep = ~fit_used[left] & ~fit_used[right]
        if fallback[qi]:
            # Keep both bracket operands for each supported contributor and the
            # anchor, including anchor-only and circular cancellation fallbacks.
            a_used = brackets[np.r_[True, support[1:, qi]], qi].ravel()
            keep &= ~(np.isin(left, a_used) | np.isin(right, a_used))
        if np.any(keep):
            a, b, p = left[keep], right[keep], providers[keep]
            unused_count[qi] = len(np.unique(np.column_stack((a, b)), axis=0))
            unused_members[p, qi] = True
            cross[qi] = different_source_disjoint(a, b, source_codes[p])
    return count, unused_count, cross, unused_members


def freeze_scene(motion, tracks, native, ws):
    nodes, support, query = motion['node_ids'], motion['support'], native[0]
    require(np.asarray(ws).shape == (len(nodes),), 'C139 WS node count changed')
    membership = canonical_membership(np.asarray(ws), len(nodes))
    groups = group_members(membership)
    baseline = mean_state(motion['states'], support, motion['nearest_seconds'], membership)
    baseline[..., 2] *= KNOT_TO_MPS
    tags = support_tags(query, native[1], native[3], groups, support, motion['sources'])
    if groups:
        arms, masks = readouts(native, membership, baseline, support)
        b, active, low = arms['B'], masks['active'], masks['B_low_speed']
        np.testing.assert_array_equal(active, tags['cause'] == 0)
        np.testing.assert_array_equal(low, active & (b[..., 2] < .5))
    else:
        b, active, low = baseline, tags['visible'], tags['visible']
    fallback = ~active | low
    np.testing.assert_array_equal(b[..., :2], baseline[..., :2])
    np.testing.assert_array_equal(b[~active], baseline[~active])
    np.testing.assert_array_equal(b[..., 3][fallback], baseline[..., 3][fallback])
    before = np.isin(tags['cause'], (1, 4))
    after = np.isin(tags['cause'], (2, 5))
    span = np.where(before, 1, np.where(after, 2, 0)).astype(np.uint8)
    branch = np.where(active, low, np.where(tags['eligible'][:, None] >= 2, 2, 3)).astype(np.uint8)
    far = np.isin(tags['cause'], (1, 2)) & ~tags['visible'] & (
        tags['distance']/((query[-1]-query[0])/(len(query)-1)) > 4.)
    count = np.zeros(branch.shape, np.int64)
    unused = np.zeros_like(count)
    cross = np.zeros(branch.shape, bool)
    suppliers = np.zeros(support.shape, bool)
    for gi, members in enumerate(groups):
        count[gi], unused[gi], cross[gi], suppliers[members] = native_group_evidence(
            query, [tracks[member] for member in members], motion['states'][members, :, :2],
            active[gi], fallback[gi], support[members], motion['sources'][members])
    require(np.all(unused <= count) and np.all(~cross | (unused >= 2)), 'invalid interval counts')
    return dict(membership=membership, node_ids=nodes, query=query, support=support,
                nearest=motion['nearest_seconds'], states=b, branch=branch, span=span, visible=tags['visible'],
                far=far, interval_count=count, nonreuse_count=unused,
                cross_source_pair=cross, unused_members=suppliers)


def statistics(vectors):
    values = vectors['error']
    tail = np.abs(values) > TAIL_DEG
    magnitude = np.abs(values)
    total = float(magnitude.sum())
    total_tail = int(tail.sum())

    def region(mask):
        absolute_sum = float(magnitude[mask].sum())
        tails = mask & tail
        n_tail = int(tails.sum())
        return dict(_stats(values[mask]), absolute_error_sum=absolute_sum,
                    absolute_error_share=absolute_sum/total if total else None,
                    tail_gt150=n_tail, tail_share=n_tail/total_tail if total_tail else None,
                    tail_incidence=n_tail/int(mask.sum()) if mask.any() else None,
                    native_candidate_points=int((mask & (vectors['interval_count'] > 0)).sum()),
                    native_candidate_query_incidences=int(vectors['interval_count'][mask].sum()),
                    unused_candidate_points=int((mask & (vectors['nonreuse_count'] > 0)).sum()),
                    unused_candidate_query_incidences=int(vectors['nonreuse_count'][mask].sum()),
                    cross_source_disjoint_points=int((mask & vectors['cross_source_pair']).sum()),
                    tail_with_compatible_distinct_member=int((tails & vectors['same_identity_other_member']).sum()))

    masks = {name: vectors['branch'] == i for i, name in enumerate(BRANCHES)}
    return dict(all=region(np.ones(len(values), bool)), branches={k: region(v) for k, v in masks.items()},
                span={name: region(vectors['span'] == i) for i, name in enumerate(('inside', 'before', 'after'))},
                native_support=region(vectors['visible']), internal_gap=region(~vectors['visible'] & (vectors['span'] == 0)),
                far_rule_domain=region(vectors['far']), pure=region(vectors['pure']), mixed=region(~vectors['pure']))


def measure_scene(saved, row):
    with np.load(row['evaluation'], allow_pickle=False) as ev:
        np.testing.assert_array_equal(saved['node_ids'], ev['node_ids'])
        np.testing.assert_array_equal(saved['query'], ev['query_seconds'])
        identities, ids, truth, valid = (ev[k] for k in ('identities', 'target_ids', 'target_states', 'target_valid'))
    groups = group_members(saved['membership'])
    outputs = emit_group_outputs(saved['states'], [members.tolist() for members in groups],
                                 node_count=len(identities))
    reports, details = evaluate_fusion_methods(
        {'B': outputs}, prediction_speed_units={'B': 'mps'}, node_identities=identities,
        target_ids=ids, target_states=truth, target_valid=valid, support=saved['support'],
        nearest_seconds=saved['nearest'], target_speed_unit='mps', split=SPLIT)
    report, arrays = reports['B'], details['B']
    n = len(ids)
    t, q = np.nonzero(arrays['aligned_truth_valid'][:n, :, 3] & arrays['aligned_prediction_valid'][:n, :, 3])
    g = arrays['aligned_group_indices'][t]
    require((g >= 0).all(), 'course scoring requires an original claimed target')
    actual = _wrap(arrays['aligned_prediction'][t, q, 3]-arrays['aligned_truth'][t, q, 3])
    pure = np.empty(len(groups), bool)
    compatible = np.zeros(saved['branch'].shape, bool)
    for i, members in enumerate(groups):
        same = identities[members] == identities[members[0]]
        pure[i] = same.all()
        same[0] = False
        compatible[i] = saved['unused_members'][members[same]].any(axis=0)
    by_point = dict(error=actual, same_identity_other_member=compatible[g, q], pure=pure[g],
                    **{key: saved[key][g, q] for key in VECTOR_KEYS[1:-2]})
    population = {key: int(report[key]) for key in POPULATION_KEYS}
    all_coverage = report['full_target_metrics']['all']
    for column, attr in enumerate(('latitude', 'longitude')):
        _, all_coverage[attr] = _coverage(arrays['aligned_truth_valid'][..., column],
                                         arrays['aligned_prediction_valid'][..., column])
    coverage = {attr: {key: int(all_coverage[attr][key]) for key in COVERAGE_KEYS}
                for attr in ('latitude', 'longitude', 'speed', 'heading', 'position')}
    require(coverage['heading']['intersection_count'] == len(actual), 'course intersection counts disagree')
    selected_groups = arrays['aligned_group_indices'][:n]
    matched = selected_groups >= 0
    assigned_branches = saved['branch'][selected_groups[matched]]
    missing = ~arrays['aligned_truth_valid'][:n, :, 3][matched]
    blind = {}
    for i, name in enumerate(BRANCHES):
        branch = saved['branch'] == i
        blind[name] = dict(group_query_count=int(branch.sum()),
                           claimed_queries_without_heading_reference=int(((assigned_branches == i) & missing).sum()),
                           native_candidate_points=int((branch & (saved['interval_count'] > 0)).sum()),
                           unused_candidate_points=int((branch & (saved['nonreuse_count'] > 0)).sum()),
                           cross_source_disjoint_points=int((branch & saved['cross_source_pair']).sum()))
    return dict(population=population, coverage=coverage, branch_counts=blind), by_point


def aggregate(rows, vectors):
    pool = {key: np.concatenate([item[key] for item in vectors]) for key in VECTOR_KEYS}
    return dict(scene_count=len(rows), component_count=len({row['component'] for row in rows}),
                population=dict(sum((Counter(row['population']) for row in rows), Counter())),
                coverages={key: dict(sum((Counter(row['coverage'][key]) for row in rows), Counter()))
                           for key in ('latitude', 'longitude', 'speed', 'heading', 'position')},
                branch_counts={key: dict(sum((Counter(row['branch_counts'][key]) for row in rows), Counter()))
                               for key in BRANCHES},
                **statistics(pool))


def run(args, checkpoint):
    ids, components, index, parent, paths, digest, metadata = bind_inputs(args.runs, args.manifest)
    private = args.output / 'private'
    private.mkdir(mode=0o700)
    frozen_hashes = {}
    with MTADArchive(paths) as archive:
        validate_measurement_archive(args.runs, archive)
        for number, sid in enumerate(ids, 1):
            motion_path = index[sid]['motion']
            ws_path = parent / 'private/blind' / f'{sid}.json.gz'
            motion = read_motion(index[sid])
            ws = read(ws_path)['ws']
            with np.load(motion_path, allow_pickle=False) as stored:
                tracks, native = native_inputs(archive, sid, motion, stored['states'])
            arrays = freeze_scene(motion, tracks, native, ws)
            path = private / f'{sid}.npz'
            np.savez_compressed(path, **arrays)
            frozen_hashes[sid] = dict(prediction_sha256=file_hash(path), motion_sha256=file_hash(motion_path),
                                      ws_sha256=file_hash(ws_path))
            if number % 25 == 0:
                checkpoint('freeze', completed_scenes=number)
        require([(r['scene_id'], r['kind']) for r in archive.members_read]
                == [(sid, 'measurements') for sid in ids], 'only declared fit measurements may be read')
    require(len(list(private.glob('*.npz'))) == len(ids), 'blind scene freeze incomplete')
    freeze_path = args.output / 'freeze.json'
    write(freeze_path, dict(fit_scene_count=len(ids), fit_component_count=76,
          scene_order_sha256=digest, archive_measurement_reads=len(ids), frozen_before_truth=True,
          reference_reads_before_freeze=0, association='C139 historical 056 WS',
          estimates='C74 B on C151 supported Mean position', angular_tail_gt_degrees=TAIL_DEG,
          canonical_archive_to_151_state_and_motion_parity=True,
          metadata_sha256={str(path): file_hash(path) for path in metadata}, scene_hashes=frozen_hashes))
    frozen_hashes = read(freeze_path)['scene_hashes']
    rows, arrays, groups = [], [], defaultdict(list)
    checkpoint('evaluate', completed_scenes=0)
    for number, sid in enumerate(ids, 1):
        path = private / f'{sid}.npz'
        require(file_hash(path) == frozen_hashes[sid]['prediction_sha256'], 'frozen B prediction digest changed')
        with np.load(path, allow_pickle=False) as saved:
            row, vector = measure_scene(saved, index[sid])
        row.update(scene_id=sid, component=components[sid], stats=statistics(vector))
        groups[components[sid]].append(len(rows))
        rows.append(row)
        arrays.append(vector)
        if number % 25 == 0:
            checkpoint('evaluate', completed_scenes=number)
    largest = min(groups, key=lambda key: (-len(groups[key]), key))
    without = [i for c, seq in groups.items() if c != largest for i in seq]
    global_stats = aggregate(rows, arrays)
    without_stats = aggregate([rows[i] for i in without], [arrays[i] for i in without])
    by_component = {key: aggregate([rows[i] for i in selected], [arrays[i] for i in selected])
                    for key, selected in groups.items()}

    def concentrated(pool):
        low = pool['branches']['enabled_low_speed']
        return max(low['absolute_error_share'] or 0., low['tail_share'] or 0.) >= .20

    qualified_components = [key for key, value in by_component.items() if key != largest and
                            value['branches']['enabled_low_speed']['tail_with_compatible_distinct_member'] >= 10]
    write(args.output / 'scenes.json.gz', rows)
    write(args.output / 'summary.json', dict(full=global_stats, components=by_component,
          largest_scene_component=largest, without_largest=without_stats,
          low_speed_concentrated_full_and_without_largest=concentrated(global_stats) and concentrated(without_stats),
          qualified_components_without_largest=qualified_components,
          native_opportunity_gate=len(qualified_components) >= 2, study='fit-side descriptive diagnosis'))
    checkpoint('done', status='COMPLETED', exit_code=0, completed_scenes=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, split='C92 fit', optimizer_updates=0, model_forwards=0,
                 raw_reference_or_label_reads=0, other_split_data_opened=False,
                 started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(phase, **fields):
        state.update(fields, phase=phase)
        if state['status'] != 'RUNNING':
            state['finished_at'] = datetime.now(timezone.utc).isoformat()
        summary_state(args.output, start, **state)
        if state['status'] == 'RUNNING' and peak_rss_bytes() > RSS_LIMIT:
            raise MemoryError('C149 8 GiB resident checkpoint exceeded')

    try:
        checkpoint('bind')
        run(args, checkpoint)
    except BaseException as exc:
        checkpoint('failed', status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
