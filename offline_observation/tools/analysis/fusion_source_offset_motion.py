"""C142: isolate translation-invariant motion under an identical position anchor."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from association.data.archive import MTADArchive
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion, paper_fusion_scene
from association.evaluation.physical import EARTH_RADIUS_METRES
from tools.analysis import fusion_interval_smoothing as c140
from tools.analysis.offline_association_tail import read_motion, summary_state, write


C140 = 'LEVEL1-OFFLINE-INTERVAL-SMOOTHING-20261114-attempt-1'
ARMS = ('MEAN', 'EQUAL', 'ANCHOR_ABSOLUTE', 'ANCHOR_RELATIVE')
REGIONS = (*c140.REGIONS[1:], 'active', 'active_boundary', 'active_anchor', 'active_other')
GAP_S = 120.
MIN_BLOCK_S = 20.
MIN_BLOCK_POINTS = 3
NEAR_M = 1000.
ASSIGN_EPS_M = 1.e-6
HARM_M = 250.
DISTANCE = c140.c130.DISTANCE
read = c140.read


def interpolation(times, q):
    """Return local report indices and a time fraction inside a continuous segment."""
    if not len(times) or not times[0] <= q <= times[-1]:
        return None
    j = np.searchsorted(times, q)
    if times[j] == q:
        return j, j, 0.
    return j-1, j, (q-times[j-1])/(times[j]-times[j-1])


def source_values(parts, times, xy, queries):
    values, witnesses = [], []
    for part in parts:
        brackets = [interpolation(times[part], q) for q in queries]
        if any(row is None for row in brackets):
            continue
        interpolated = [(1.-u)*xy[part[a]]+u*xy[part[b]] for a, b, u in brackets]
        values.append(interpolated)
        witnesses.extend(int(part[k]) for a, b, _ in brackets for k in (a, b))
    if not values:
        return None
    return np.mean(values, axis=0), sorted(set(witnesses))


def joint_interval(parts, times):
    """Longest interval with one spanning segment for each source, ties go earlier."""
    domains = [[(times[p[0]], times[p[-1]]) for p in segments if len(p) >= 2]
               for segments in parts]
    candidates = []
    for left in sorted({a for domains_i in domains for a, _ in domains_i}):
        right = min(max((b for a, b in domains_i if a <= left), default=-np.inf)
                    for domains_i in domains)
        if right-left >= MIN_BLOCK_S:
            candidates.append((right-left, -left, right))
    if not candidates:
        return None
    _, start, end = max(candidates)
    return -start, end


def verify_block(parts, times, xy):
    for segments in parts:
        source_times = np.concatenate([times[p] for p in segments])
        if len(np.unique(source_times)) < MIN_BLOCK_POINTS:
            return 'block_reports', None
    interval = joint_interval(parts, times)
    if interval is None:
        return 'block_time', None
    sampled = [source_values(segments, times, xy, interval) for segments in parts]
    assert all(row is not None for row in sampled)
    vectors = np.stack([values[1]-values[0] for values, _ in sampled])
    difference = np.linalg.norm(vectors[:, None]-vectors[None, :], axis=-1).max()
    return None, dict(interval=list(interval), max_disagreement_m=float(difference),
                      witness_rows=sorted({i for _, rows in sampled for i in rows}))


def active_segment_sets(segments, times):
    events = {}
    for i, part in enumerate(segments):
        if len(part) >= 2:
            events.setdefault(times[part[0]], (set(), set()))[0].add(i)
            events.setdefault(times[part[-1]], (set(), set()))[1].add(i)
    active = set()
    boundaries = sorted(events)
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        starts, ends = events[left]
        active.difference_update(ends)
        active.update(starts)
        yield left, right, sorted(active)


def qualify(times, geo, xy, sources, lengths, query, multiple):
    """Use native interval membership and disjoint raw rows to freeze each query window."""
    segments, segment_source = [], []
    for rows in np.split(np.arange(len(times)), np.cumsum(lengths)[:-1]):
        for part in np.split(rows, np.flatnonzero(np.diff(times[rows]) > GAP_S)+1):
            segments.append(part)
            segment_source.append(sources[part[0]])
    counts, windows = Counter(), []
    source_codes = np.unique(sources, return_inverse=True)[1]
    occupied = np.zeros(len(query), bool)
    for left, right, active in active_segment_sets(segments, times):
        counts['candidate_windows'] += 1
        by_source = {s: [i for i in active if segment_source[i] == s]
                     for s in sorted({segment_source[i] for i in active})}
        if len(by_source) < 2:
            counts['one_or_zero_source'] += 1
            continue
        counts['multisource_windows'] += 1
        indices = np.flatnonzero((query >= left) & (query <= right) & multiple & ~occupied)
        # A shared window boundary belongs to the earlier multisource window even if it fails.
        occupied[indices] = True
        part_rows = {i: segments[i][(times[segments[i]] >= left) & (times[segments[i]] <= right)]
                     for i in active}
        if min(len(part) for part in part_rows.values()) < MIN_BLOCK_POINTS:
            counts['segment_reports'] += 1
            continue
        midpoint = (left+right)/2.
        parts = [[part_rows[i] for i in group] for group in by_source.values()]
        block_result, halves = [], []
        for before in (True, False):
            block = [[part[(times[part] <= midpoint) == before] for part in group] for group in parts]
            reason, result = verify_block(block, times, xy)
            block_result.append((reason, result))
            halves.append(np.concatenate([p for group in block for p in group]))
        reasons = [reason for reason, _ in block_result if reason is not None]
        if reasons:
            counts[reasons[0]] += 1
            continue
        keys = [set(map(tuple, np.column_stack((source_codes[h], times[h], geo[h])))) for h in halves]
        assert not keys[0] & keys[1]
        counts['two_block_windows'] += 1
        for group in parts:
            spans = np.array([(times[p[0]], times[p[-1]]) for p in group])
            bracketed = ((query[indices, None] >= spans[:, 0]) & (query[indices, None] <= spans[:, 1])).any(1)
            indices = indices[bracketed]
        selected = None
        for q in indices[np.argsort(np.abs(query[indices]-midpoint), kind='stable')]:
            evaluated = [source_values(group, times, xy, [query[q]]) for group in parts]
            if all(v is not None for v in evaluated):
                selected = q, evaluated
                break
        if selected is None:
            counts['no_query_or_anchor'] += 1
            continue
        anchor_i, evaluated = selected
        anchor = np.mean([v[0][0] for v in evaluated], axis=0)
        window_rows = np.concatenate(list(part_rows.values()))
        row_segments = np.repeat(np.arange(len(part_rows)), [len(part) for part in part_rows.values()])
        counts.update(anchored_windows=1, anchored_group_q=len(indices), anchored_rows=len(window_rows))
        windows.append(dict(bounds=[float(left), float(right)], queries=indices.tolist(), anchor_i=int(anchor_i),
                            anchor=anchor.tolist(), rows=window_rows.tolist(), row_segments=row_segments.tolist(),
                            anchor_witness_rows=sorted({i for _, rows in evaluated for i in rows}),
                            blocks=[dict(result, row_count=len(h)) for (_, result), h in
                                    zip(block_result, halves, strict=True)], source_count=len(by_source)))
    return windows, counts


def weighted_problem(indices, a, b, target, weight, n, segments, means):
    diagonal = np.bincount(indices, weights=weight*a*a, minlength=n)
    diagonal += np.bincount(indices+1, weights=weight*b*b, minlength=n)
    off = np.bincount(indices, weights=weight*a*b, minlength=n-1)
    normal = np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1)
    rhs = np.column_stack([np.bincount(indices, weights=weight*a*y, minlength=n)
                           + np.bincount(indices+1, weights=weight*b*y, minlength=n) for y in target.T])
    if means is not None:
        cross = np.zeros_like(means)
        np.add.at(cross, (segments, indices), weight*a)
        np.add.at(cross, (segments, indices+1), weight*b)
        mass = np.bincount(segments, weights=weight, minlength=len(means))
        sums = np.column_stack([np.bincount(segments, weights=weight*y, minlength=len(means))
                                for y in target.T])
        term = cross.T@means
        normal += means.T@(mass[:, None]*means)-term-term.T
        rhs -= means.T@sums
    return normal, rhs


def solve_window(times, xy, segments, query, anchor_t, anchor, *, relative):
    start, end = times.min(), times.max()
    knots = np.linspace(0., end-start, min(257, int(np.ceil((end-start)/c140.KNOT_S))+1))
    indices, a, b = c140.basis(knots, times-start)
    n = len(knots)
    change = np.zeros((n-2, n))
    r = np.arange(n-2)
    steps = c140.KNOT_S/np.diff(knots)
    change[r, r], change[r, r+1], change[r, r+2] = steps[:-1], -steps[:-1]-steps[1:], steps[1:]
    penalty = change.T@change
    m, u, v = c140.basis(knots, np.array([anchor_t-start]))
    (pivot, other, ratio) = (int(m[0]), int(m[0]+1), -v[0]/u[0]) if u[0] >= v[0] else (
        int(m[0]+1), int(m[0]), -u[0]/v[0])
    keep = np.arange(n) != pivot
    index_other = int(keep[:other].sum())
    target = xy-anchor
    means = None
    if relative:
        counts = np.bincount(segments)
        means = np.zeros((len(counts), n))
        np.add.at(means, (segments, indices), a)
        np.add.at(means, (segments, indices+1), b)
        means /= counts[:, None]
        centers = np.column_stack([np.bincount(segments, weights=y)/counts for y in target.T])
        target = target-centers[segments]
    state = np.zeros((n, 2))
    weight = np.ones(len(times))
    objectives, conditions = [], []
    for iteration in range(c140.STEPS):
        normal, rhs = weighted_problem(indices, a, b, target, weight, n, segments, means)
        normal += penalty
        reduced = normal[np.ix_(keep, keep)]
        reduced[index_other] += ratio*normal[pivot, keep]
        reduced[:, index_other] += ratio*normal[keep, pivot]
        reduced[index_other, index_other] += ratio*ratio*normal[pivot, pivot]
        right = rhs[keep]
        right[index_other] += ratio*rhs[pivot]
        if iteration in (0, c140.STEPS-1):
            spectrum = np.linalg.eigvalsh(reduced)
            if spectrum[0] <= 1.e-12*spectrum[-1] or spectrum[0] <= 0.:
                raise np.linalg.LinAlgError('rank')
            conditions.append(float(spectrum[-1]/spectrum[0]))
        state[keep] = np.linalg.solve(reduced, right)
        state[pivot] = ratio*state[other]
        model = a[:, None]*state[indices]+b[:, None]*state[indices+1]
        if means is not None:
            model -= (means@state)[segments]
        error = np.linalg.norm(model-target, axis=1)
        huber = np.where(error <= c140.HUBER_M, error**2, c140.HUBER_M*(2.*error-c140.HUBER_M))
        objectives.append(float(huber.sum()+np.sum((change@state)**2)))
        weight = c140.HUBER_M/np.maximum(c140.HUBER_M, error)
    anchor_error = float(np.linalg.norm(u[0]*state[m[0]]+v[0]*state[m[0]+1]))
    if anchor_error > 1.e-7 or not np.isfinite(objectives).all():
        raise np.linalg.LinAlgError('nonfinite or hard anchor')
    if any(curr > prev+1.e-8*max(1., prev) for prev, curr in zip(objectives[:-1], objectives[1:], strict=True)):
        raise np.linalg.LinAlgError('objective increase')
    j, h, k = c140.basis(knots, query-start)
    return h[:, None]*state[j]+k[:, None]*state[j+1]+anchor, dict(
        knots=n, objectives=objectives, conditions=conditions, anchor_error_m=anchor_error,
        final_next_weight_sum=float(weight.sum()))


def input_paths(runs, sid):
    return (runs/C140/'private'/f'{sid}.npz',
            runs/c140.c127.c124.C121/'private'/f'{sid}.npz')


def qualification(args, scenes, data, components):
    prepared = args.output/'private'/'qualified'
    prepared.mkdir(mode=0o700, parents=True)
    results = []
    with MTADArchive(read(args.runs/c140.c127.c98.DATA/'run.json')['archive']) as archive:
        c140.c133.validate_measurement_archive(args.runs, archive)
        for sid in scenes:
            motion = read_motion(data[sid])
            tracks = c140.c133._canonical_tracks(archive._rows(sid, 'measurements'), sid,
                                                motion['node_ids'], motion['sources'])
            lengths = np.array([len(t['canonical_times']) for t in tracks])
            np.testing.assert_array_equal(lengths, motion['native_count'])
            baseline, original = input_paths(args.runs, sid)
            groups, public = [], Counter()
            with np.load(baseline, allow_pickle=False) as base, np.load(original, allow_pickle=False) as prior:
                partition, flags = base['partition'], base['flags']
                np.testing.assert_array_equal(partition, prior['current'])
                np.testing.assert_array_equal(motion['query_seconds'], prior['query'])
                np.testing.assert_array_equal(motion['node_ids'], prior['node_ids'])
                np.testing.assert_array_equal(base['estimates'][0], prior['MEAN'])
                for gi, members in enumerate(c140.c127.c98.group_nodes(partition)):
                    counts = Counter(groups=1, native_rows=int(lengths[members].sum()))
                    if len(np.unique(motion['sources'][members])) > 1:
                        counts['multisource_groups'] += 1
                        span, geo, xy, sources, origin, scale, _, group_flags = c140.group_inputs(
                            members, tracks, motion)
                        np.testing.assert_array_equal(group_flags, flags[gi])
                        windows, detail = qualify(span, geo, xy, sources, lengths[members],
                                                  motion['query_seconds'], flags[gi, :, 3])
                        counts.update(detail)
                        if windows:
                            groups.append(dict(group=gi, times=span.tolist(), xy=xy.tolist(), origin=origin.tolist(),
                                               scale=scale.tolist(), windows=windows))
                    public.update(counts)
            write(prepared/f'{sid}.json', groups)
            results.append(dict(scene=sid, component=components[sid], counts=dict(public)))
        assert [row['scene_id'] for row in archive.members_read] == scenes
        assert all(row['kind'] == 'measurements' for row in archive.members_read)
    total = sum((Counter(row['counts']) for row in results), Counter())
    component_size = Counter(components.values())
    largest = min(component_size, key=lambda k: (-component_size[k], k))
    qualifying = [row for row in results if row['counts'].get('anchored_group_q', 0)]
    others = [row for row in qualifying if row['component'] != largest]
    active_components = {row['component'] for row in qualifying}
    gate = dict(components=len(active_components), qualifying_scenes=len(qualifying),
                largest=largest, qualifying_scenes_without_largest=len(others),
                pass_gate=len(active_components) >= 2 and len(others) >= 2)
    by_component = {c: dict(sum((Counter(r['counts']) for r in results if r['component'] == c), Counter()))
                    for c in sorted(set(components.values()))}
    write(args.output/'qualification.json', dict(scene_count=len(scenes), scene_rows=results, total=dict(total),
          by_component=by_component, gate=gate, native_archive_reads=len(scenes), evaluations_before_gate=0))
    return gate


def predict(args, scenes, data, gate):
    output = args.output/'private'/'predictions'
    output.mkdir(mode=0o700)
    rows = []
    numerical = []
    for sid in scenes:
        baseline, _ = input_paths(args.runs, sid)
        with np.load(baseline, allow_pickle=False) as base:
            partition, flags = base['partition'], base['flags']
            stored = base['estimates'][:2]
        motion = read_motion(data[sid])
        query = motion['query_seconds']
        estimates = np.concatenate((stored, stored[[1, 1]]), axis=0)
        active = np.zeros(flags.shape[:2]+(4,), bool)
        audit = []
        counts = Counter()
        for group in read(args.output/'private'/'qualified'/f'{sid}.json'):
            gi = group['group']
            times, xy, origin, scale = (np.asarray(group[k]) for k in ('times', 'xy', 'origin', 'scale'))
            for window in group['windows']:
                q, native, anchor = (np.asarray(window[k]) for k in ('queries', 'rows', 'anchor'))
                t_anchor = query[window['anchor_i']]
                assert not active[gi, q, 0].any() and flags[gi, q, 3].all()
                assert times[native].min() <= query[q].min() <= query[q].max() <= times[native].max()
                solves, checks = [], []
                try:
                    for relative in (False, True):
                        fit, check = solve_window(times[native], xy[native], np.asarray(window['row_segments']),
                                                  query[q], t_anchor, anchor, relative=relative)
                        np.testing.assert_allclose(fit[q == window['anchor_i']][0], anchor, rtol=0., atol=1.e-7)
                        predicted = fit/scale+origin
                        predicted[:, 1] = (predicted[:, 1]+180.) % 360.-180.
                        if not np.isfinite(predicted).all() or (np.abs(predicted[:, 0]) > 90.).any():
                            raise np.linalg.LinAlgError('illegal geographic position')
                        solves.append(predicted)
                        checks.append(check)
                except np.linalg.LinAlgError as error:
                    counts[f'fallback_{error}'] += 1
                    continue
                estimates[2:, gi, q, :2] = np.stack(solves)
                active[gi, q, 0] = True
                a, b = window['bounds']
                active[gi, q, 1] = np.minimum(query[q]-a, b-query[q]) <= c140.KNOT_S
                active[gi, window['anchor_i'], 2] = True
                active[gi, q, 3] = ~active[gi, q, 1:3].any(1)
                counts.update(active_windows=1, active_group_q=len(q))
                audit.append(dict(group=gi, window=window['bounds'], source_count=window['source_count'],
                    rows=len(native), queries=len(q), block_disagreement_m=[b['max_disagreement_m'] for b in window['blocks']],
                    absolute=checks[0], relative=checks[1]))
        np.testing.assert_array_equal(estimates[2:, ~active[..., 0]], np.repeat(stored[1:2, ~active[..., 0]], 2, 0))
        np.testing.assert_array_equal(estimates[2:, ..., 2:], np.repeat(stored[1:2, ..., 2:], 2, 0))
        assert not (active[..., 0] & (flags[..., 0] | flags[..., 1])).any()
        np.savez_compressed(output/f'{sid}.npz', partition=partition, estimates=estimates,
                            flags=np.concatenate((flags, active), axis=-1))
        write(output/f'{sid}-solver.json', audit)
        numerical.extend(audit)
        rows.append(dict(scene=sid, counts=dict(counts)))
    numerics = {}
    for arm in ('absolute', 'relative'):
        changes = [(w[arm]['objectives'][-2]-w[arm]['objectives'][-1])/max(w[arm]['objectives'][-2], 1.)
                   for w in numerical]
        numerics[arm] = dict(final_objective_relative_improvement=metrics(changes, False),
            max_rank_condition=max((max(w[arm]['conditions']) for w in numerical), default=None),
            max_anchor_error_m=max((w[arm]['anchor_error_m'] for w in numerical), default=None),
            windows=len(changes))
    disagreements = np.array([w['block_disagreement_m'] for w in numerical]).reshape(-1, 2)
    write(args.output/'freeze.json', dict(scenes=scenes, gate=gate, references_before_freeze=0,
          enabled=sum(row['counts'].get('active_group_q', 0) for row in rows),
          anchor_shared=True, scene_rows=rows, native_inputs='canonical_all', numerics=numerics,
          block_disagreement_m=[metrics(v, False) for v in disagreements.T],
          numerical_policy='fixed knots, squared velocity change, exact hard anchor, five Huber IRLS steps'))


def wrapped(deg):
    return (deg+180.) % 360.-180.


def near_pairs(position, prediction, valid, active):
    """Record deidentified true-near pair geometry and assignment margins at identical Q."""
    output = []
    for q in range(len(position[0])):
        targets = np.flatnonzero(valid[:, q])
        i, j = np.triu_indices(len(targets), 1)
        first, second = targets[i], targets[j]
        a, b = position[first, q], position[second, q]
        gap = DISTANCE(a, b)
        near = gap <= NEAR_M
        first, second, a, b, gap = (x[near] for x in (first, second, a, b, gap))
        pred_a, pred_b = prediction[:, first, q], prediction[:, second, q]
        real_delta = b-a
        pred_delta = pred_b-pred_a
        delta_lat = pred_delta[..., 0]-real_delta[..., 0]
        delta_lon = wrapped(wrapped(pred_delta[..., 1])-wrapped(real_delta[..., 1]))
        scale = EARTH_RADIUS_METRES*np.pi/180.
        directed_error = scale*np.hypot(delta_lat, delta_lon*np.cos(np.deg2rad((a[:, 0]+b[:, 0])/2.)))
        margin = DISTANCE(pred_a, b)+DISTANCE(pred_b, a)-DISTANCE(pred_a, a)-DISTANCE(pred_b, b)
        acting = active[first, q].astype(int)+active[second, q]
        output.append(np.column_stack((gap, acting, directed_error.T, margin.T)))
    return np.concatenate(output)


def evaluate_scene(motion, saved, truth):
    np.testing.assert_array_equal(motion['node_ids'], truth['node_ids'])
    np.testing.assert_array_equal(motion['query_seconds'], truth['query_seconds'])
    groups = c140.c127.c98.group_nodes(saved['partition'])
    identities = truth['identities']
    representatives = {}
    for gi, group in enumerate(groups):
        representatives.setdefault(identities[group[0]], gi)
    matched = np.array([representatives.get(t, -1) for t in truth['target_ids']])
    keep = matched >= 0
    targets, target_valid = truth['target_states'], truth['target_valid']
    reference, valid = targets[keep], target_valid[keep]
    position = reference[..., :2]
    prediction = saved['estimates'][:, matched[keep]]
    flags = saved['flags'][matched[keep]]
    signed = prediction-reference
    signed[..., 1] = wrapped(signed[..., 1])
    signed[..., 2] = prediction[..., 2]/(3600./1852.)-reference[..., 2]
    signed[..., 3] = wrapped(signed[..., 3])
    signed = np.where(valid[None], signed, np.nan)
    legal = valid[..., :2].all(axis=-1)
    distance = np.where(legal[None], DISTANCE(prediction[..., :2], position), np.nan)
    any_valid = valid.any(axis=-1)
    points = np.concatenate((signed, distance[..., None]), axis=-1)[:, any_valid].transpose(1, 0, 2)
    near = near_pairs(position, prediction[..., :2], legal, flags[..., len(c140.REGIONS)-1]) if keep.any() else (
        np.empty((0, 2+2*len(ARMS))))
    paper = {}
    for i, arm in enumerate(ARMS):
        aligned = np.zeros_like(targets)
        aligned[keep] = prediction[i]
        available = np.broadcast_to(keep[:, None, None], targets.shape)
        paper[arm] = paper_fusion_scene(aligned, targets, target_valid, prediction_valid=available,
                                       prediction_speed_unit='kn', target_speed_unit='mps')
    return points, flags[any_valid], near, dict(targets=len(matched), matched=int(keep.sum()), groups=len(groups),
        duplicate_groups=len(groups)-len(representatives), unmatched=int((~keep).sum())), paper


def metrics(values, signed):
    values = np.asarray(values)
    valid = values[np.isfinite(values)]
    absolute = np.abs(valid)
    record = c140.c127.c107.attribute_stats(absolute)
    record.update(p95=float(np.quantile(absolute, .95)) if len(valid) else None,
                  rmse=float(np.sqrt(np.mean(valid**2))) if len(valid) else None)
    if signed:
        record.update(me=float(valid.mean()) if len(valid) else None,
                      residual_sd=float(valid.std()) if len(valid) else None)
    return record


def point_statistics(points, flags, tail):
    out = {}
    for key, use in [('all', np.ones(len(points), bool)), *zip(REGIONS, flags.T, strict=True)]:
        x = points[use]
        z = x[np.isfinite(x[..., 4]).all(1), :, 4]
        previous = z[:, 0] < tail
        outcomes, gains = {}, {}
        for i, arm in enumerate(ARMS):
            outcomes[arm] = {name: metrics(x[:, i, j], j < 4) for j, name in enumerate(c140.FEATURES)}
            outcomes[arm]['old_non_tail_harm250_vs_mean'] = int((previous & (z[:, i]-z[:, 0] > HARM_M)).sum())
            for k in range(i):
                delta = z[:, i]-z[:, k]
                gains[f'{arm}/{ARMS[k]}'] = dict(harm250=int((delta > HARM_M).sum()),
                    old_non_tail_harm250=int((previous & (delta > HARM_M)).sum()),
                    benefit250=int((delta < -HARM_M).sum()), net_delta_m=float(delta.sum()),
                    new_tail=int(((z[:, k] < tail) & (z[:, i] >= tail)).sum()),
                    exit_tail=int(((z[:, k] >= tail) & (z[:, i] < tail)).sum()))
        out[key] = dict(metrics=outcomes, changes=gains)
    return out


def near_statistics(rows):
    n = len(ARMS)
    result = {}
    for key, use in (('all', np.ones(len(rows), bool)), ('active', rows[:, 1] > 0),
                     ('both_active', rows[:, 1] == 2)):
        current = rows[use]
        error, margin = current[:, 2:2+n], current[:, 2+n:]
        result[key] = dict(pairs=len(current), arms={arm: dict(directed_error=metrics(error[:, i], False),
            swapped_better=int((margin[:, i] < -ASSIGN_EPS_M).sum()),
            ambiguous=int((np.abs(margin[:, i]) <= ASSIGN_EPS_M).sum()),
            direct_better=int((margin[:, i] > ASSIGN_EPS_M).sum())) for i, arm in enumerate(ARMS)},
            relative_transitions={ARMS[i]: dict(
                direct_to_swap=int(((margin[:, i] > ASSIGN_EPS_M) & (margin[:, 3] < -ASSIGN_EPS_M)).sum()),
                direct_to_not_direct=int(((margin[:, i] > ASSIGN_EPS_M) & (margin[:, 3] <= ASSIGN_EPS_M)).sum()),
                not_direct_to_direct=int(((margin[:, i] <= ASSIGN_EPS_M) & (margin[:, 3] > ASSIGN_EPS_M)).sum()))
                for i in (1, 2)})
    return result


def decision(result, components, leave_largest, near):
    outcome = {}
    relative = ARMS[-1]
    for control in ARMS[1:3]:
        def better(row):
            ra, ca = (row['all']['metrics'][arm] for arm in (relative, control))
            rp, cp = (row['active']['metrics'][arm]['position']['p99'] for arm in (relative, control))
            return dict(full_mae=ra['position']['mae'] < ca['position']['mae'],
                full_p99=ra['position']['p99'] < ca['position']['p99'],
                active_p99=rp is not None and cp is not None and rp < cp,
                old_non_tail_harm250=ra['old_non_tail_harm250_vs_mean'] < ca['old_non_tail_harm250_vs_mean'])
        checks = better(result)
        checks['leave_largest'] = better(leave_largest)
        changes = result['all']['changes'][f'{relative}/{control}']
        checks['more_benefits_than_harms'] = changes['benefit250'] > changes['harm250']
        checks['net_benefit_in_two_components'] = sum(
            row['all']['changes'][f'{relative}/{control}']['net_delta_m'] < 0. for row in components.values()) >= 2
        near_active = near['active']
        ra, ca = (near_active['arms'][arm] for arm in (relative, control))
        checks['near_direction'] = (ra['directed_error']['mae'] <= ca['directed_error']['mae']
                                    if near_active['pairs'] else None)
        checks['near_swap'] = ra['swapped_better'] <= ca['swapped_better'] if near_active['pairs'] else None
        checks['near_no_new_reversal'] = near_active['relative_transitions'][control]['direct_to_swap'] == 0
        outcome[control] = checks
    promote = all(all(check is not False for key, check in values.items() if key != 'leave_largest')
                  and all(values['leave_largest'].values()) for values in outcome.values())
    return dict(promote=promote, comparisons=outcome)


def evaluate(args, scenes, data, components, gate):
    observations, markers, pair_rows, scene_ids, pair_scene_ids, counts = [], [], [], [], [], []
    for i, sid in enumerate(scenes):
        motion = read_motion(data[sid])
        with np.load(args.output/'private'/'predictions'/f'{sid}.npz', allow_pickle=False) as saved, \
             np.load(data[sid]['evaluation'], allow_pickle=False) as truth:
            points, flags, pairs, coverage, paper = evaluate_scene(motion, saved, truth)
        observations.append(points)
        markers.append(flags)
        pair_rows.append(pairs)
        scene_ids.extend([i]*len(points))
        pair_scene_ids.extend([i]*len(pairs))
        counts.append(dict(scene=sid, component=components[sid], coverage=coverage, paper=paper))
    points, flags, pairs = map(np.concatenate, (observations, markers, pair_rows))
    scene_index, pair_scene_index = np.asarray(scene_ids), np.asarray(pair_scene_ids, dtype=int)
    comp = np.array([components[sid] for sid in scenes])
    valid = np.isfinite(points)
    np.testing.assert_array_equal(valid, np.repeat(valid[:, :1], len(ARMS), axis=1))
    np.testing.assert_array_equal(points[..., 2:4], np.repeat(points[:, :1, 2:4], len(ARMS), axis=1))
    np.testing.assert_array_equal(valid[:, 0].sum(0), [123051, 123056, 123044, 122934, 123051])
    with np.load(args.runs/C140/'public-errors.npz', allow_pickle=False) as prior:
        np.testing.assert_array_equal(prior['scene_ids'], scenes)
        np.testing.assert_array_equal(prior['scene_index'], scene_index)
        np.testing.assert_array_equal(prior['flags'], flags[:, :len(c140.REGIONS)-1])
        np.testing.assert_array_equal(prior['errors'][:, :2], np.abs(points[:, :2]))
    tail = float(np.nanquantile(points[:, 0, 4], .99))
    result = point_statistics(points, flags, tail)
    near = near_statistics(pairs)
    by_component, by_leaveout = {}, {}
    for c in sorted(set(comp)):
        selected, selected_pairs = comp[scene_index] == c, comp[pair_scene_index] == c
        by_component[c] = dict(points=point_statistics(points[selected], flags[selected], tail),
                               near_pairs=near_statistics(pairs[selected_pairs]))
        by_leaveout[c] = dict(points=point_statistics(points[~selected], flags[~selected], tail),
                              near_pairs=near_statistics(pairs[~selected_pairs]))
    for i, row in enumerate(counts):
        selected = scene_index == i
        row.update(points=point_statistics(points[selected], flags[selected], tail),
                   near_pairs=near_statistics(pairs[pair_scene_index == i]))
    write(args.output/'summary.json', dict(scenes=len(scenes), points=result, near_pairs=near,
          coverage=dict(zip(c140.FEATURES, valid[:, 0].sum(0).tolist(), strict=True)),
          target_coverage=dict(sum((Counter(r['coverage']) for r in counts), Counter())),
          point_residual_definition='prediction minus reference; longitude and heading wrapped; SD ddof=0',
          paper={arm: aggregate_paper_fusion([r['paper'][arm] for r in counts]) for arm in ARMS},
          mean_baseline_tail_m=tail, components=by_component, leave_one_component_out=by_leaveout,
          decision=decision(result, {c: v['points'] for c, v in by_component.items()},
                            by_leaveout[gate['largest']]['points'], near)))
    write(args.output/'scenes.json', counts)
    np.savez_compressed(args.output/'public-errors.npz', errors=points, flags=flags, scene_index=scene_index,
        scene_ids=scenes, components=comp, arms=ARMS, features=c140.FEATURES, regions=REGIONS, pairs=pairs,
        pair_scene_index=pair_scene_index, signed_first_four=True, pair_fields=('true_gap_m', 'active_targets',
        *(f'{arm}_directed_error_m' for arm in ARMS), *(f'{arm}_swap_margin_m' for arm in ARMS)))


def run(args, start):
    c98 = c140.c127.c98
    scenes = read(args.runs/c98.PREP/'frozen-split.json')['held']
    manifest = read(args.manifest)
    assert len(scenes) == len(set(scenes)) == 177 and set(scenes) <= set(manifest['splits']['T_core'])
    components = {sid: manifest['components'][sid] for sid in scenes}
    assert len(set(components.values())) == 4
    data = read(args.runs/c98.DATA/'private/index.json')['T_core']
    assert read(args.runs/C140/'run.json')['status'] == 'COMPLETED'
    assert read(args.runs/C140/'freeze.json')['archive_measurements'] == len(scenes)
    summary_state(args.output, start, status='RUNNING', phase='qualification')
    gate = qualification(args, scenes, data, components)
    if gate['pass_gate']:
        summary_state(args.output, start, status='RUNNING', phase='fit')
        predict(args, scenes, data, gate)
        summary_state(args.output, start, status='RUNNING', phase='evaluate')
        evaluate(args, scenes, data, components, gate)
    summary_state(args.output, start, status='COMPLETED', qualification_pass=gate['pass_gate'],
                  evaluated=gate['pass_gate'], native_archive_reads=len(scenes), optimizer_updates=0,
                  model_forwards=0, test_opened=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'manifest', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    try:
        run(args, start)
    except Exception as error:
        summary_state(args.output, start, status='FAILED', error=repr(error))
        raise


if __name__ == '__main__':
    main()
