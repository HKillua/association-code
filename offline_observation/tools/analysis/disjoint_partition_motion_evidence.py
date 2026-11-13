"""Label-blind native motion evidence for the frozen C78 partition pools."""
from __future__ import annotations

import numpy as np

from association.evaluation.mixture_opportunity import _distance as haversine
from association.evaluation.native_motion import _short_delta
from association.evaluation.physical import EARTH_RADIUS_METRES


SAME_MAX_GAP_S = 600.
ENDPOINT_MAX_WINDOW_S = 120.
ENDPOINT_MIN_POINTS = 3
ENDPOINT_MAX_POINTS = 5
CROSS_MIN_OVERLAP_S = 120.
CROSS_MAX_BRACKET_S = 120.
CROSS_TIMES = 5


def _validate_pairs(pairs, sources):
    pairs, sources = np.asarray(pairs), np.asarray(sources)
    n = len(sources)
    if (n == 0 or sources.shape != (n,) or pairs.dtype.kind not in 'iu'
            or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))):
        raise ValueError('one source per node and the original complete ordered pair population required')
    return pairs, sources


def _native_arrays(native, n):
    _, times, latlon, offsets = map(np.asarray, native)
    if (times.ndim != 1 or latlon.shape != (len(times), 2) or offsets.shape != (n+1,)
            or offsets.dtype.kind not in 'iu' or offsets[0] != 0 or offsets[-1] != len(times)
            or not np.isfinite(times).all() or not np.isfinite(latlon).all()
            or (np.abs(latlon) > [90., 180.]).any()):
        raise ValueError('native time, position and offset population changed')
    offsets = offsets.astype(np.int64, copy=False)
    if (np.diff(offsets) < 2).any() or any((np.diff(times[a:b]) <= 0).any()
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)):
        raise ValueError('at least two strictly increasing native positions per fragment required')
    return times, latlon, offsets


def _coordinates(latlon, offsets):
    """Unwrap each fragment into a fixed scene equirectangular metre chart."""
    heads = latlon[offsets[:-1]]
    ref_lat = float(np.median(heads[:, 0]))
    radians = np.deg2rad(heads[:, 1])
    mean_cos, mean_sin = np.cos(radians).mean(), np.sin(radians).mean()
    ref_lon = (float(np.rad2deg(np.arctan2(mean_sin, mean_cos)))
               if np.hypot(mean_cos, mean_sin) > 1e-12 else float(heads[0, 1]))
    unwrapped = latlon.copy().astype(np.float64, copy=False)
    for a, b in zip(offsets[:-1], offsets[1:], strict=True):
        longitude = latlon[a:b, 1]
        first = ref_lon+_short_delta(longitude[0]-ref_lon)
        unwrapped[a:b, 1] = first+np.r_[0., np.cumsum(_short_delta(np.diff(longitude)))]
    scale = (EARTH_RADIUS_METRES*np.pi/180.)*np.array([1., np.cos(np.deg2rad(ref_lat))])
    xy = (unwrapped - np.array([ref_lat, ref_lon]))*scale
    return unwrapped, xy, scale


def _end_slopes(times, xy, offsets):
    """Return anchored fit and adjacent secant in forward physical seconds."""
    n = len(offsets)-1
    ols = np.full((n, 2, 2), np.nan)
    secant = np.empty_like(ols)
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        t, v = times[a:b], xy[a:b]
        for side, end, neighbor, order in (
                (0, 0, 1, np.arange(min(ENDPOINT_MAX_POINTS, len(t)))),
                (1, len(t)-1, len(t)-2, np.arange(max(0, len(t)-ENDPOINT_MAX_POINTS), len(t)))):
            dt = t[order]-t[end]
            secant[i, side] = (v[neighbor]-v[end])/(t[neighbor]-t[end])
            inside = np.abs(dt) <= ENDPOINT_MAX_WINDOW_S
            if np.count_nonzero(inside) >= ENDPOINT_MIN_POINTS:
                dt = dt[inside]
                ols[i, side] = (dt @ (v[order[inside]]-v[end]))/(dt @ dt)
    return ols, secant


def _sample(nodes, queries, times, latlon, offsets):
    """Interpolate positions and return the largest bracket per pair side."""
    result = np.empty((*queries.shape, 2), np.float64)
    span = np.empty(len(nodes), np.float64)
    order = np.argsort(nodes, kind='stable')
    cuts = np.r_[0, np.cumsum(np.bincount(nodes, minlength=len(offsets)-1))]
    for node in np.flatnonzero(np.diff(cuts)):
        a, b = offsets[node:node+2]
        t, values = times[a:b], latlon[a:b]
        take = order[cuts[node]:cuts[node+1]]
        q = queries[take]
        upper = np.searchsorted(t, q).clip(0, len(t)-1)
        lower = (upper-1).clip(0, len(t)-1)
        span[take] = np.where(t[upper] == q, 0., t[upper]-t[lower]).max(axis=1)
        result[take, :, 0] = np.interp(q, t, values[:, 0])
        result[take, :, 1] = np.interp(q, t, values[:, 1])
    return result, span


def scene_motion_evidence(pairs, sources, native):
    """Return fixed motion fields with NaN for ineligible original pairs."""
    pairs, sources = _validate_pairs(pairs, sources)
    times, latlon, offsets = _native_arrays(native, len(sources))
    positions, xy, scale = _coordinates(latlon, offsets)
    left, right = pairs.T
    same = sources[left] == sources[right]
    first, last = offsets[:-1], offsets[1:]-1
    t0, t1 = times[first], times[last]
    earlier_is_left = t1[left] < t0[right]
    early = np.where(earlier_is_left, left, right)
    later = np.where(earlier_is_left, right, left)
    gap = t0[later]-t1[early]
    fields = {name: np.full(len(pairs), np.nan) for name in
              ('same_hold_m', 'same_secant_m', 'same_ols_m', 'same_gap_s', 'same_slope_difference_mps',
               'cross_abs_m', 'cross_relative_mps', 'cross_activity_mps', 'cross_overlap_s')}

    short = same & (gap > 0.) & (gap <= SAME_MAX_GAP_S)
    ols, secant = _end_slopes(times, xy, offsets)
    same_ok = short & np.isfinite(ols[early, 1, 0]) & np.isfinite(ols[later, 0, 0])
    if same_ok.any():
        a, b = early[same_ok], later[same_ok]
        interval = gap[same_ok]
        displacement = xy[first[b]]-xy[last[a]]
        fields['same_hold_m'][same_ok] = np.linalg.norm(displacement, axis=1)
        for name, velocity in (('same_secant_m', secant), ('same_ols_m', ols)):
            ahead = np.linalg.norm(displacement-interval[:, None]*velocity[a, 1], axis=1)
            back = np.linalg.norm(displacement-interval[:, None]*velocity[b, 0], axis=1)
            fields[name][same_ok] = .5*(ahead+back)
        fields['same_gap_s'][same_ok] = interval
        fields['same_slope_difference_mps'][same_ok] = np.linalg.norm(ols[a, 1]-ols[b, 0], axis=1)

    start = np.maximum(t0[left], t0[right])
    end = np.minimum(t1[left], t1[right])
    common = ~same & (end-start >= CROSS_MIN_OVERLAP_S)
    rows = np.flatnonzero(common)
    cross_count = 0
    if len(rows):
        query = np.linspace(start[rows], end[rows], CROSS_TIMES, axis=1)
        first_pos, first_span = _sample(left[rows], query, times, positions, offsets)
        second_pos, second_span = _sample(right[rows], query, times, positions, offsets)
        eligible = np.maximum(first_span, second_span) <= CROSS_MAX_BRACKET_S
        row = rows[eligible]
        a, b = first_pos[eligible], second_pos[eligible]
        dt = np.diff(query[eligible], axis=1)[..., None]
        da, db = np.diff(a, axis=1)*scale/dt, np.diff(b, axis=1)*scale/dt
        fields['cross_abs_m'][row] = np.median(haversine(a, b), axis=1)
        fields['cross_relative_mps'][row] = np.median(np.linalg.norm(da-db, axis=-1), axis=1)
        fields['cross_activity_mps'][row] = np.median(
            .5*(np.linalg.norm(da, axis=-1)+np.linalg.norm(db, axis=-1)), axis=1)
        fields['cross_overlap_s'][row] = end[row]-start[row]
        cross_count = len(row)
    if any(np.isinf(values).any() for values in fields.values()):
        raise ValueError('native motion evidence is nonfinite')
    metadata = dict(original_pairs=len(pairs), same_pairs=int(same.sum()), cross_pairs=int((~same).sum()),
                    same_separated=int((same & (gap > 0.)).sum()), same_short_gap=int(short.sum()),
                    same_eligible=int(same_ok.sum()), cross_overlap_eligible=len(rows), cross_eligible=cross_count)
    return fields, metadata


def _role_advantage(roles, costs):
    """Compare each entry with the cheapest distinct edge for its role."""
    entry = np.arange(len(roles))
    order = np.lexsort((entry, costs, roles))
    cuts = np.r_[0, np.flatnonzero(np.diff(roles[order]))+1, len(roles)]
    sizes = np.diff(cuts)
    counts = np.repeat(sizes, sizes)
    ranked_group = np.repeat(cuts[:-1], sizes)
    best = order[ranked_group]
    second = order[ranked_group+(counts >= 2)]
    alternatives = costs[np.where(order == best, second, best)]
    original = costs[order]
    denom = alternatives+original
    ranked = np.divide(alternatives-original, denom, out=np.zeros_like(original),
                       where=(counts >= 2) & (denom > 0.))
    value = np.empty(len(roles), np.float64)
    sufficient = np.empty(len(roles), bool)
    value[order], sufficient[order] = ranked, counts >= 2
    return value, sufficient


def competition_weights(pairs, sources, native, cost, relation):
    """Signed minimum of role-wise margins, using all eligible scene pairs."""
    pairs, sources = _validate_pairs(pairs, sources)
    cost = np.asarray(cost, np.float64)
    if (relation not in ('same', 'cross') or cost.shape != (len(pairs),)
            or np.isinf(cost).any() or (cost < 0.).any()):
        raise ValueError('same/cross relation and nonnegative per-pair costs or NaN required')
    take = np.flatnonzero(np.isfinite(cost))
    result = np.zeros(len(pairs), np.float64)
    comparable = 0
    if len(take):
        left, right = pairs[take].T
        same = sources[left] == sources[right]
        if not (same if relation == 'same' else ~same).all():
            raise ValueError('finite costs must belong to the declared source relation')
        if relation == 'same':
            times, _, offsets = _native_arrays(native, len(sources))
            t0, t1 = times[offsets[:-1]], times[offsets[1:]-1]
            left_out = t1[left] < t0[right]
            right_out = t1[right] < t0[left]
            if not (left_out ^ right_out).all():
                raise ValueError('same-source continuation requires strictly disjoint times')
            role_left = 2*left+left_out
            role_right = 2*right+right_out
        else:
            unique, code = np.unique(sources, return_inverse=True)
            role_left = len(unique)*left+code[right]
            role_right = len(unique)*right+code[left]
        values, enough = _role_advantage(np.r_[role_left, role_right], np.tile(cost[take], 2))
        both = enough[:len(take)] & enough[len(take):]
        result[take[both]] = np.minimum(values[:len(take)][both], values[len(take):][both])
        comparable = int(both.sum())
    return result, dict(eligible=len(take), comparable=comparable,
                        positive=int((result > 0.).sum()), negative=int((result < 0.).sum()),
                        neutral_comparable=comparable-int(np.count_nonzero(result)))


def self_check():
    latlon = lambda lon: np.column_stack((np.zeros(len(lon)), ((np.asarray(lon)+180.) % 360.)-180.))
    time = np.arange(0., 121., 30.)
    pairs = np.array([[0, 1]])
    position = latlon(np.r_[179.9+.002*time, 180.2+.002*time])
    offsets = np.array([0, 5, 10])
    fields, meta = scene_motion_evidence(pairs, ['a', 'b'],
                                        (None, np.tile(time, 2), position, offsets))
    assert meta['cross_eligible'] == 1 and fields['cross_abs_m'][0] > 30000.
    np.testing.assert_allclose(fields['cross_relative_mps'], 0., atol=1e-9)
    assert fields['cross_activity_mps'][0] > 200.
    bad_time = np.array([0., 150., 180., 210., 240.])
    fields, meta = scene_motion_evidence(pairs, ['a', 'b'],
                                        (None, np.tile(bad_time, 2), position, offsets))
    assert meta['cross_eligible'] == 0 and np.isnan(fields['cross_relative_mps']).all()

    a, b = time, time+150.
    position = latlon(179.9+.002*np.r_[a, b])
    native = None, np.r_[a, b], position, offsets
    fields, meta = scene_motion_evidence(pairs, ['a', 'a'], native)
    assert meta['same_eligible'] == 1 and fields['same_gap_s'][0] == 30.
    assert fields['same_hold_m'][0] > 6000.
    for name in ('same_secant_m', 'same_ols_m', 'same_slope_difference_mps'):
        np.testing.assert_allclose(fields[name], 0., atol=5e-8)

    pairs = np.column_stack(np.triu_indices(4, 1))
    costs = np.array([np.nan, 1., 3., 4., 2., np.nan])
    expected = np.array([0., .5, -.5, -.6, .2, 0.])
    weights, meta = competition_weights(pairs, ['a', 'a', 'b', 'b'], None, costs, 'cross')
    np.testing.assert_allclose(weights, expected)
    assert meta['comparable'] == 4 and meta['positive'] == meta['negative'] == 2
    staggered = None, np.r_[a, a, b, b], np.tile(position, (2, 1)), np.arange(0, 21, 5)
    weights, _ = competition_weights(pairs, ['a']*4, staggered, costs, 'same')
    np.testing.assert_allclose(weights, expected)
    weights, meta = competition_weights(pairs, ['a', 'a', 'b', 'b'], None, np.where(
        np.isfinite(costs), 0., np.nan), 'cross')
    assert not weights.any() and meta['neutral_comparable'] == 4
    weights, meta = competition_weights(pairs, ['a', 'a', 'b', 'b'], None,
                                        np.array([np.nan, 0., np.nan, 1., np.nan, np.nan]), 'cross')
    assert not weights.any() and meta['comparable'] == 0
    return dict(dateline_shift_invariant=True, local_ols=True, strict_interpolation=True,
                directional_and_source_reciprocity=True, insufficient_roles_are_neutral=True)


if __name__ == '__main__':
    print(self_check())
