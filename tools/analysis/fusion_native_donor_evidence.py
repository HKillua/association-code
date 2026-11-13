"""C133: test native contradictions on fixed, fully protected C130 borrowing actions."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import fusion_supplier_matched_activation as c130
from tools.analysis.disjoint_partition_motion_evidence import _coordinates
from tools.analysis.held_fusion_population import _canonical_tracks, validate_measurement_archive
from tools.analysis.offline_association_tail import read_motion, summary_state, write


C127 = c130.C127
C130 = 'LEVEL1-OFFLINE-SUPPLIER-MATCHED-ACTIVATION-TRAJECTORY-20261113-attempt-1'
C130_FIRST = 'LEVEL1-OFFLINE-SUPPLIER-MATCHED-ACTIVATION-20261113-attempt-1'
ARMS = ('J', 'SAME_SOURCE', 'NATIVE_ID', 'NATIVE_POSITION', 'JOINT', 'MATCHED_SOURCE')
SHORT_S = 120.
WITNESS_S = 600.
SEPARATION_M = 1000.
HARM_M = 250.


def bracket(t, q, excluded=()):
    usable = np.array([j for j in range(len(t)) if j not in excluded], dtype=int) if excluded else np.arange(len(t))
    use = t[usable]
    hi = np.searchsorted(use, q)
    if hi < len(use) and use[hi] == q:
        return (int(usable[hi]),)
    if 0 < hi < len(use) and use[hi] - use[hi-1] <= SHORT_S:
        return int(usable[hi-1]), int(usable[hi])
    return ()


def at(t, xy, q, indices):
    if len(indices) == 1:
        return xy[indices[0]]
    a, b = indices
    return xy[a] + (xy[b]-xy[a])*((q-t[a])/(t[b]-t[a]))


def endpoint(t, xy, before, query):
    indices = (-1, -2, -3) if before else (0, 1, 2)
    a, b = indices[:2]
    if len(t) < 2 or abs(t[a]-t[b]) > SHORT_S or abs(query-t[a]) > SHORT_S:
        return None, None
    estimate = at(t, xy, query, (a, b))
    error = None
    if len(t) >= 3 and abs(t[a]-t[indices[2]]) <= SHORT_S:
        c = indices[2]
        error = float(np.linalg.norm(at(t, xy, t[b], (a, c))-xy[b]))
    return estimate, error


def evidence(times, points, group, donor, query, old_q, new_q):
    dt, dp = times[donor], points[donor]
    native_bracket = bracket(dt, query)
    outer = bracket(dt, query, native_bracket) if native_bracket else ()
    local = float(np.linalg.norm(at(dt, dp, query, outer)-new_q)) if outer else None
    witnesses = {}
    bridges, predictions = [], []
    for node in group:
        t, xy = times[node], points[node]
        for j in np.flatnonzero(abs(t-query) <= WITNESS_S):
            when = t[j]
            use = bracket(dt, when, native_bracket)
            if use:
                distance = float(np.linalg.norm(at(dt, dp, when, use)-xy[j]))
                witnesses[when] = min(witnesses.get(when, np.inf), distance)
        if t[-1] < dt[0] or dt[-1] < t[0]:
            before = t[-1] < dt[0]
            a, b = (t[-1], dt[0]) if before else (dt[-1], t[0])
            if b-a <= SHORT_S:
                center = (a+b)/2
                left, _ = endpoint(t, xy, before, center)
                right, _ = endpoint(dt, dp, not before, center)
                if left is not None and right is not None:
                    bridges.append((abs(center-query), float(np.linalg.norm(left-right))))
        assert query < t[0] or query > t[-1], 'recipient unexpectedly has native support'
        before = query > t[-1]
        prediction, check = endpoint(t, xy, before, query)
        if check is not None and check <= HARM_M:
            excess = np.linalg.norm(new_q-prediction)-np.linalg.norm(old_q-prediction)
            predictions.append((abs(query-(t[-1] if before else t[0])), int(node), float(excess)))
    selected = sorted(witnesses, key=lambda when: (abs(when-query), when))[:2]
    shared = [witnesses[when] for when in selected]
    bridge = min(bridges)[1] if bridges else None
    continuity = min(predictions)[2] if predictions else None
    identity_reject = min(shared) > SEPARATION_M if len(shared) == 2 else (bridge is not None and bridge > SEPARATION_M)
    position_reject = (local is not None and local > HARM_M) or (continuity is not None and continuity > HARM_M)
    return dict(witness=shared, bridge_m=bridge, donor_bracket_s=float(dt[native_bracket[-1]]-dt[native_bracket[0]])
        if native_bracket else None, donor_leave_bracket_m=local, receiver_excess_m=continuity,
        identity_reject=bool(identity_reject), position_reject=bool(position_reject))


def freeze_rows(rows, groups, motion, base, tracks):
    if not rows:
        return []
    lengths = [len(track['canonical_times']) for track in tracks]
    np.testing.assert_array_equal(lengths, motion['native_count'])
    offsets = np.r_[0, np.cumsum(lengths)]
    _, xy, scale = _coordinates(np.concatenate([track['canonical_values'][:, :2] for track in tracks]), offsets)
    all_points = np.split(xy, offsets[1:-1])
    times = [track['canonical_times'] for track in tracks]
    for row in rows:
        group, q, donor = (row[key] for key in ('group', 'q', 'donor'))
        geo = motion['states'][donor, q, :2]
        old = base[group, q, :2]
        offset = old-geo
        offset[1] = (offset[1]+180.) % 360.-180.
        dt, dp = times[donor], all_points[donor]
        when = motion['query_seconds'][q]
        # Position of the stored interpolant in the same local chart as native points.
        hi = np.searchsorted(dt, when).clip(1, len(dt)-1)
        new = at(dt, dp, when, (hi-1, hi))
        row['evidence'] = evidence(times, all_points, groups[group], donor, when, new+offset*scale, new)
    return rows


def choose(rows, old, j_order):
    count = len(old)
    # Saved J members retain their original rank ahead of unselected backfill slots.
    def id_bad(i):
        return rows[i]['evidence']['identity_reject']

    def pos_bad(i):
        return rows[i]['evidence']['position_reject']

    def both(i):
        return id_bad(i) or pos_bad(i)

    same = sum(rows[i]['same_source'] for i in old)
    ranked = sorted(j_order, key=both)
    source_matched = [i for i in ranked if rows[i]['same_source']][:same]
    source_matched += [i for i in ranked if not rows[i]['same_source']][:count-same]
    return dict(J=old, SAME_SOURCE=sorted(j_order, key=lambda i: not rows[i]['same_source'])[:count],
                NATIVE_ID=sorted(j_order, key=id_bad)[:count], NATIVE_POSITION=sorted(j_order, key=pos_bad)[:count],
                JOINT=ranked[:count], MATCHED_SOURCE=source_matched)


def evaluate(sid, motion, base, truth, groups, frozen, reach, threshold):
    np.testing.assert_array_equal(base['node_ids'], truth['node_ids'])
    np.testing.assert_array_equal(base['query'], truth['query_seconds'])
    ids = truth['identities']
    first = {}
    for gi, group in enumerate(groups):
        first.setdefault(ids[group[0]], gi)
    targets = {first[x]: ti for ti, x in enumerate(truth['target_ids']) if x in first}
    valid = truth['target_valid'].copy()
    g = np.array([first.get(x, -1) for x in truth['target_ids']])
    valid[g < 0] = False
    p = base['KEEP']
    rows, old = frozen['rows'], frozen['selected']
    changes, details, arrays = {}, [], []
    for arm, selection in old.items():
        states = p.copy()
        count = Counter(planned=len(selection))
        for i in selection:
            row = rows[i]
            receiver, q, donor = (row[k] for k in ('group', 'q', 'donor'))
            assert not motion['support'][groups[receiver], q].any() and motion['support'][donor, q]
            states[receiver, q, :2] = motion['states'][donor, q, :2]
            count['identity_flag_actions'] += row['evidence']['identity_reject']
            count['position_flag_actions'] += row['evidence']['position_reject']
            target = targets.get(receiver)
            if target is None or not valid[target, q, :2].all():
                continue
            ref = truth['target_states'][target, q, :2]
            before = float(c130.DISTANCE(p[receiver, q, :2], ref))
            after = float(c130.DISTANCE(states[receiver, q, :2], ref))
            identity = truth['target_ids'][target]
            true = bool(ids[donor] == identity)
            tail = before >= threshold
            harm = not tail and after-before > HARM_M
            possible = reach[receiver]['first']
            true_pool = bool(np.any(motion['support'][possible, q] & (ids[possible] == identity)))
            count.update(evaluated=1, true=int(true), wrong=int(not true), no_true_pool=int(not true_pool),
                true_tail_exit=int(true and tail and after < threshold), non_tail_harm250=int(harm),
                true_non_tail_harm250=int(harm and true), new_tail=int(not tail and after >= threshold),
                tail_exit=int(tail and after < threshold), position_changed=int(np.any(states[receiver, q, :2] != p[receiver, q, :2])))
            details.append(dict(scene=sid, arm=arm, target=target, q=q, true=true, true_pool=true_pool,
                tail=tail, harm=harm, same_source=row['same_source'], before_m=before, after_m=after, **row['evidence']))
        np.testing.assert_array_equal(states[..., 2:], p[..., 2:])
        protected = frozen['protected']
        np.testing.assert_array_equal(states[protected], p[protected])
        changes[arm] = count
        position = c130.DISTANCE(states[g, :, :2], truth['target_states'][..., :2]) if len(p) else np.zeros(valid.shape[:2])
        arrays.append(position[valid[..., :2].all(-1)])
    return np.array(arrays), details, dict(scene=sid, counts=changes, coverage=valid.sum((0, 1)).tolist())


def run(args):
    runs, out = args.runs, args.output
    c127, read = c130.c127, c130.c127.read
    c98 = c127.c98
    prior, supplier = runs/C130, runs/c127.c124.C121
    assert read(prior/'run.json')['status'] == 'COMPLETED'
    scenes = read(runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(scenes) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    data = read(runs/c98.DATA/'private/index.json')['T_core']
    thresholds = read(runs/c98.C95/'summary.json')['fixed_thresholds']
    private = out/'private'
    private.mkdir()
    with MTADArchive(read(runs/c98.DATA/'run.json')['archive']) as archive:
        validate_measurement_archive(runs, archive)
        for sid in scenes:
            original = read(prior/'private'/f'{sid}.json')
            rows = original['rows']
            j_order = []
            protected = sorted({r['donor_group'] for r in read(runs/C130_FIRST/'private'/f'{sid}.json')['rows']})
            if rows:
                motion = read_motion(data[sid])
                tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, motion['node_ids'], motion['sources'])
                with np.load(supplier/'private'/f'{sid}.npz', allow_pickle=False) as base, \
                     np.load(runs/c98.C95/'private/held'/f'{sid}.npz', allow_pickle=False) as scores:
                    original_selection = np.load(runs/C127/'private'/f'{sid}-selected.npy', allow_pickle=False)[0]
                    groups, all_rows = c130.visible_candidates(motion, base, scores,
                        read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                        read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'), thresholds, original_selection)
                    fixed = [r for r in all_rows if r['group'] not in protected]
                    assert [(r['group'], r['q'], r['donor']) for r in fixed] == [
                            (r['group'], r['q'], r['donor']) for r in rows]
                    j_order = sorted(range(len(fixed)), key=lambda i: -fixed[i]['gain'])
                    assert j_order[:len(original['selected']['J'])] == original['selected']['J']
                    freeze_rows(rows, groups, motion, base['KEEP'], tracks)
            write(private/f'{sid}.json', dict(rows=rows, protected=protected,
                  selected=choose(rows, original['selected']['J'], j_order)))
        native = [row['scene_id'] for row in archive.members_read]
        assert len(set(native)) == len(native) == 85 and set(native) <= set(scenes)
    write(out/'freeze.json', dict(scenes=len(scenes), native_measurement_scenes=native, evaluation_before_freeze=False))
    old_stats = read(prior/'summary.json')
    threshold = old_stats['baseline_threshold_m']
    all_errors, events, scene_rows, totals = [], [], [], defaultdict(Counter)
    components = defaultdict(lambda: defaultdict(Counter))
    for sid in scenes:
        frozen = read(private/f'{sid}.json')
        component = manifest['components'][sid]
        motion = read_motion(data[sid])
        with np.load(supplier/'private'/f'{sid}.npz', allow_pickle=False) as base, \
             np.load(data[sid]['evaluation'], allow_pickle=False) as truth, \
             np.load(runs/c98.C95/'private/held'/f'{sid}.npz', allow_pickle=False) as scores:
            groups, _, reach = c127.frozen_scene(motion, base, scores,
                read(runs/c127.c104.C98/'private/held'/f'{sid}.json.gz'),
                read(runs/c127.c107.C104/'private/held'/f'{sid}-choices.json'), thresholds)
            error, rows, scene = evaluate(sid, motion, base, truth, groups, frozen, reach, threshold)
        for arm, count in scene['counts'].items():
            totals[arm].update(count)
            components[component][arm].update(count)
        assert len({v['planned'] for v in scene['counts'].values()}) == 1
        all_errors.append(error)
        events.extend(rows)
        scene_rows.append(dict(scene, component=component))
    errors = np.concatenate(all_errors, axis=1)
    metrics = {arm: c130.STATS(errors[i]) for i, arm in enumerate(ARMS)}
    for k, v in old_stats['metrics']['J'].items():
        np.testing.assert_allclose(metrics['J'][k], v, rtol=0., atol=2e-8)
    for k in ('planned', 'evaluated', 'true', 'wrong', 'non_tail_harm250', 'true_tail_exit'):
        assert totals['J'][k] == old_stats['totals']['J'][k]
    coverage = dict(zip(('lat', 'lon', 'speed', 'heading'), np.sum([s['coverage'] for s in scene_rows], axis=0).tolist(), strict=True))
    assert coverage == old_stats['coverage']
    write(out/'summary.json', dict(scenes=len(scenes), coverage=coverage, metrics=metrics, totals=totals,
         components=components, baseline_threshold_m=threshold, opportunity_slots=old_stats['opportunity_slots']))
    write(out/'scenes.json', scene_rows)
    write(out/'events.json', events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
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
