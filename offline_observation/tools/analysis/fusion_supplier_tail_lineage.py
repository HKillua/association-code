"""C124: separate missing supplier support from the frozen position tail."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from association.data.reference_states import iter_reference_states
from tools.analysis import fusion_supplier_train_qualification as c121
from tools.analysis import raw_reference_speed_origin as trace
from tools.analysis.association_registered_holdout_fusion import _distance, attribute_stats
from tools.analysis.disjoint_partition_motion_evidence import CROSS_MAX_BRACKET_S
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive
from tools.analysis.offline_association_tail import read_motion, summary_state, write
from tools.analysis.reference_interval_fingerprint import geometry
from tools.analysis.same_source_swap_pilot import read


C121 = 'LEVEL1-OFFLINE-SUPPLIER-TRAIN-QUALIFICATION-20261113-attempt-1'
ARMS = ('KEEP', 'NEW_NEAREST', 'OLD_ENDPOINT')


def freeze_scene(motion, saved, old, tracks):
    group_nodes = c121.c104.c98.group_nodes(saved['current'])
    support = motion['support']
    predictions = np.repeat(saved['KEEP'][None, ...], len(ARMS), axis=0)
    if tracks is None:
        assert not saved['active_new'].any()
        return predictions, group_nodes
    q = motion['query_seconds']
    for gi, nodes in enumerate(group_nodes):
        retained = nodes[old[nodes] == old[nodes[0]]]
        for point in np.flatnonzero(~saved['old_active'][nodes].any(0) & saved['active_new'][nodes].any(0)):
            donors = nodes[saved['active_new'][nodes, point]]
            selected = donors[np.argmin(motion['nearest_seconds'][donors, point])]
            predictions[1, gi, point, :2] = motion['states'][selected, point, :2]
            times = [tracks[i]['canonical_times'] for i in retained]
            choice = [int(np.argmin(abs(t-q[point]))) for t in times]
            nearest = int(np.argmin([abs(t[j]-q[point]) for t, j in zip(times, choice, strict=True)]))
            track = tracks[retained[nearest]]
            predictions[2, gi, point, :2] = track['canonical_values'][choice[nearest], :2]
    for arm in predictions:
        np.testing.assert_array_equal(arm[..., 2:], saved['MEAN'][..., 2:])
    return predictions, group_nodes


def classify(sid, motion, truth, group_nodes, predictions, saved):
    ids, targets = truth['identities'], truth['target_ids']
    reference, support = truth['target_states'], motion['support']
    valid = truth['target_valid'][..., :2].all(-1)
    first = {}
    for gi, nodes in enumerate(group_nodes):
        first.setdefault(ids[nodes[0]], gi)
    rows, views = [], {}
    for ti, target in enumerate(targets):
        if target not in first:
            continue
        gi = first[target]
        nodes = group_nodes[gi]
        same = np.flatnonzero(ids == target)
        own = np.intersect1d(same, nodes)
        other = np.setdiff1d(same, nodes)
        for point in np.flatnonzero(valid[ti]):
            p = predictions[:, gi, point, :2]
            expected = reference[ti, point, :2]
            errors = _distance(p, expected)
            active = nodes[support[nodes, point]]
            old_active = bool(saved['old_active'][nodes, point].any())
            new_active = bool(saved['active_new'][nodes, point].any())
            same_active = same[support[same, point]]
            group_same = own[support[own, point]]
            node_states = motion['states'][:, point, :2]
            lower = [float(_distance(node_states[take], expected).min()) if len(take) else None
                     for take in (group_same, same_active)]
            row = dict(scene=sid, target=ti, q=int(point), error_m=errors.tolist(), group_active=len(active),
                       group_same_active=len(group_same), other_same_active=int(support[other, point].sum()),
                       wrong_active=int((ids[active] != target).sum()), old_active=old_active,
                       new_active=new_active, best_group_same_m=lower[0], best_scene_same_m=lower[1])
            rows.append(row)
            views[ti, int(point)] = (target, expected, p[0], nodes, gi)
    return rows, views


def raw_trace(raw, policy, query, cached, baseline):
    times, values = raw[:2]
    shifted = times-policy['time_offset_seconds']
    aligned, bound = trace.bracket(shifted, values, query, policy['max_gap_seconds'])
    np.testing.assert_allclose(aligned[:2], cached, rtol=0., atol=2e-10)
    # Locate the interval that actually interpolates Q. An exact Q has no such interval.
    if bound['status'] != 'interior':
        return dict(reference_bracket=bound, position_jump=False, reverse_return=False,
                    path_m=None, secant_mps=None, overshoot_mps=None, direction_deg=None)
    left, right = (i-1 for i in bound['ordinals'])
    i = np.searchsorted(shifted, query)
    assert i == right
    span, direction, _ = geometry(values[[left], :2], values[[right], :2])
    speed = float(span[0]/(shifted[right]-shifted[left]))
    endpoints = values[[left, right], 2]
    excess = float(speed-endpoints.max()) if np.isfinite(endpoints).all() else None
    reverse = False
    if excess is not None and excess > 10:
        for start, stop in ((left-1, right), (left, right+1)):
            if start < 0 or stop >= len(shifted) or not np.isfinite(values[start:stop+1, :3]).all():
                continue
            seq = values[start:stop+1]
            d, initial, arrival = geometry(seq[:-1, :2], seq[1:, :2])
            net = float(_distance(seq[0, :2], seq[-1, :2]))
            jump = d/np.diff(shifted[start:stop+1])-np.maximum(seq[:-1, 2], seq[1:, 2])
            angle = abs((initial[1]-arrival[0]+180) % 360-180)
            reverse |= bool((jump > 10).all() and angle >= 120 and net <= d.sum()/2)
    _, offset, _ = geometry(cached[None], baseline[None])
    relative = float(abs((offset[0]-direction[0]+180) % 360-180))
    return dict(reference_bracket=bound, position_jump=excess is not None and excess > 10,
                reverse_return=reverse, path_m=float(span[0]), secant_mps=speed,
                overshoot_mps=excess, direction_deg=relative)


def native_trace(tracks, motion, saved, view, point):
    _, expected, _, nodes, _ = view
    q = motion['query_seconds'][point]
    rows = []
    for node in nodes:
        track = tracks[node]
        times = track['canonical_times']
        value = track['canonical_values']
        state, info = trace.bracket(times, value, q, CROSS_MAX_BRACKET_S)
        cached = motion['states'][node, point, :2]
        i = int(np.argmin(abs(times-q)))
        rows.append(dict(source=str(motion['sources'][node]),
            was_active=bool(saved['old_active'][node, point]), new_active=bool(saved['active_new'][node, point]),
            age_seconds=float(abs(times[i]-q)), native_bracket=info['status'],
            native_to_cached_m=float(_distance(state[:2], cached)) if np.isfinite(state[:2]).all() else None,
            cached_error_m=float(_distance(cached, expected)), raw_nearest_error_m=float(_distance(value[i, :2], expected))))
    return rows


def results(rows):
    errors = np.asarray([r['error_m'] for r in rows], dtype=float).reshape(-1, len(ARMS))
    return {arm: dict(attribute_stats(errors[:, i]),
                      harms_over_250=int((errors[:, i]-errors[:, 0] > 250).sum()),
                      helps=int((errors[:, 0]-errors[:, i] > 1e-6).sum())) for i, arm in enumerate(ARMS)}


def run(args):
    runs, out = args.runs, args.output
    manifest = read(args.manifest)
    base = runs/C121
    assert read(base/'run.json')['status'] == 'COMPLETED'
    prior = read(base/'summary.json')
    original_points = read(base/'points.json')
    lookup = {(r['scene'], r['target'], r['q']): r for r in original_points}
    roles = read(runs/c121.c104.c98.PREP/'frozen-split.json')
    scenes = roles['held']
    assert len(scenes) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    index = read(runs/c121.c104.c98.DATA/'private/index.json')['T_core']
    data = read(runs/c121.c104.c98.DATA/'run.json')
    policies = read(Path(data['alignment']))
    changed = set(read(base/'freeze.json')['native_measurement_scenes'])
    assert len(changed) == 7 and changed <= set(scenes)
    tracks_by_scene, frozen = {}, {}
    with MTADArchive(data['archive']) as archive:
        validate_measurement_archive(runs, archive)
        for sid in scenes:
            motion = read_motion(index[sid])
            with np.load(base/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
                 np.load(runs/c121.c104.C98/'private/held'/f'{sid}.npz', allow_pickle=False) as old:
                tracks = (_canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
                          if sid in changed else None)
                if tracks is not None:
                    tracks_by_scene[sid] = tracks
                np.testing.assert_array_equal(saved['node_ids'], motion['node_ids'])
                np.testing.assert_array_equal(saved['query'], motion['query_seconds'])
                frozen[sid] = freeze_scene(motion, saved, old['WS'], tracks)
        assert len(archive.members_read) == 7 and {r['scene_id'] for r in archive.members_read} == changed
    write(out/'predictions-frozen.json', dict(scenes=len(frozen), measurement_scenes=sorted(changed, key=int),
                                              all_predictions_before_evaluation=True))
    rows = []
    for sid in scenes:
        with np.load(base/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(index[sid]['evaluation'], allow_pickle=False) as truth:
            for a, b in (('node_ids', 'node_ids'), ('query', 'query_seconds')):
                np.testing.assert_array_equal(saved[a], truth[b])
            pred, group_nodes = frozen[sid]
            local, _ = classify(sid, read_motion(index[sid]), truth, group_nodes, pred, saved)
        rows.extend(local)
    population = results(rows)
    for key, value in prior['metrics']['position']['KEEP'].items():
        np.testing.assert_allclose(population['KEEP'][key], value, atol=2e-8, rtol=0.)
    threshold = population['KEEP']['p99']
    tail = [r for r in rows if r['error_m'][0] >= threshold]
    original = [r for r in rows if r['new_active']]
    assert len(original) == len(lookup) == 341
    for row in original:
        old = lookup[row['scene'], row['target'], row['q']]
        np.testing.assert_allclose(row['error_m'][0], old['error_m'][1], atol=2e-8, rtol=0.)
    only_new = [r for r in original if not r['old_active']]
    improved = [r for r in original if lookup[r['scene'], r['target'], r['q']]['error_m'][0] < r['error_m'][0]-1e-6]
    assert len(only_new) == 53 and len(improved) == 2
    by_scene = Counter(r['scene'] for r in tail)
    ordered = sorted(by_scene, key=lambda sid: (-by_scene[sid], int(sid)))
    raw_scenes = sorted(set(ordered[:3]) | changed, key=int)
    write(out/'tail-freeze.json', dict(threshold_m=threshold, tail_count=len(tail), tail_scenes=ordered,
                                      raw_reference_scenes=raw_scenes, after_predictions=True))
    check = defaultdict(dict)
    for row in (*original, *tail):
        if row['scene'] in raw_scenes:
            check[row['scene']][row['target'], row['q']] = row
    point_rows, tail_raw = [], []
    for reference in iter_reference_states(data['archive'], manifest, split='T_core', scene_ids=raw_scenes):
        sid = reference['scene_id']
        raw_tracks = defaultdict(list)
        for item in reference['rows']:
            raw_tracks[item['identity_id']].append(item)
        motion = read_motion(index[sid])
        with np.load(base/'private'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(index[sid]['evaluation'], allow_pickle=False) as truth:
            pred, nodes = frozen[sid]
            _, views = classify(sid, motion, truth, nodes, pred, saved)
            relevant = {views[key][0] for key in check[sid]}
            tracks = {identity: trace.raw_track(raw_tracks[identity]) for identity in relevant}
            for key, row in check[sid].items():
                view = views[key]
                identity, cached, baseline, _, _ = view
                point = key[1]
                evidence = raw_trace(tracks[identity], policies[sid], motion['query_seconds'][point], cached, baseline)
                if row['error_m'][0] >= threshold:
                    tail_raw.append(dict(row, **evidence))
                if row['new_active']:
                    point_rows.append(dict(row, **evidence, native=native_trace(tracks_by_scene[sid], motion, saved, view, point),
                                           old_mean_error_m=lookup[(sid, *key)]['error_m'][0]))
    assert len(point_rows) == len(original)
    tail_component = Counter(manifest['components'][r['scene']] for r in tail)
    cohort = defaultdict(list)
    for r in only_new:
        cohort[manifest['components'][r['scene']]].append(r)
    tail_summary = dict(count=len(tail), scenes=dict(by_scene), components=dict(tail_component),
        no_group_support=sum(not r['group_active'] for r in tail), any_wrong=sum(bool(r['wrong_active']) for r in tail),
        other_same_support=sum(bool(r['other_same_active']) for r in tail),
        no_group_but_other_same=sum(not r['group_active'] and bool(r['other_same_active']) for r in tail),
        no_scene_same=sum(not (r['group_same_active'] or r['other_same_active']) for r in tail),
        any_new=sum(r['new_active'] for r in tail),
        still_tail_even_best_same=sum(r['best_scene_same_m'] is not None and r['best_scene_same_m'] >= threshold for r in tail),
        best_same_below_tail=sum(r['best_scene_same_m'] is not None and r['best_scene_same_m'] < threshold for r in tail),
        raw_checked=len(tail_raw), raw_bracketed=sum(r['reference_bracket']['status'] == 'interior' for r in tail_raw),
        raw_position_jump=sum(r['position_jump'] for r in tail_raw), raw_reverse_return=sum(r['reverse_return'] for r in tail_raw))
    write(out/'summary.json', dict(population=population, only_new=results(only_new),
          only_new_by_component={key: results(val) for key, val in cohort.items()},
          helpful=[(r['scene'], r['target'], r['q']) for r in improved], tail=tail_summary))
    write(out/'new-supplier-points.json', point_rows)
    write(out/'tail-points.json', tail)
    write(out/'tail-reference-points.json', tail_raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forwards=0, test_opened=False)


if __name__ == '__main__':
    main()
