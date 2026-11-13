"""Label-blind external anchor evidence for ordered supplier pairs."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from tools.analysis.disjoint_partition_motion_evidence import (
    CROSS_MAX_BRACKET_S,
    CROSS_MIN_OVERLAP_S,
    CROSS_TIMES,
    _coordinates,
    _native_arrays,
    _validate_pairs,
)


MIN_ANCHORS = 3
MIN_GOOD_TIMES = 3
MIN_ANCHOR_CONDITION = .1


def _mutual_anchor_groups(pairs, source_codes, membership, margin):
    """Return unique-best positive old links, one per two-node WS group."""
    n = len(source_codes)
    dense = np.full((n, n), -np.inf)
    left, right = pairs.T
    dense[left, right] = margin
    dense[right, left] = margin
    best = np.full((n, int(source_codes.max()) + 1), -1, np.int64)
    for code in np.unique(source_codes):
        alternatives = np.flatnonzero(source_codes == code)
        values = dense[:, alternatives]
        index = values.argmax(axis=1)
        max_score = values[np.arange(n), index]
        unique = ((values == max_score[:, None]).sum(axis=1) == 1) & (max_score > 0)
        best[unique, code] = alternatives[index[unique]]

    groups = defaultdict(list)
    for node, group in enumerate(membership):
        groups[group].append(node)
    anchors = defaultdict(list)
    for group, members in groups.items():
        if len(members) != 2:
            continue
        a, b = sorted(members, key=lambda node: source_codes[node])
        sa, sb = int(source_codes[a]), int(source_codes[b])
        if sa != sb and best[a, sb] == b and best[b, sa] == a:
            anchors[sa, sb].append((group, a, b))
    return anchors


def _sample_at(nodes, query, times, xy, offsets):
    """Return per-time interpolation and strict native bracket eligibility."""
    position = np.empty((len(nodes), len(query), 2), np.float64)
    valid = np.empty((len(nodes), len(query)), bool)
    for row, node in enumerate(nodes):
        start, end = offsets[node:node+2]
        t = times[start:end]
        values = xy[start:end]
        upper = np.searchsorted(t, query).clip(0, len(t)-1)
        lower = (upper-1).clip(0, len(t)-1)
        width = np.where(t[upper] == query, 0., t[upper]-t[lower])
        valid[row] = (query >= t[0]) & (query <= t[-1]) & (width <= CROSS_MAX_BRACKET_S)
        position[row, :, 0] = np.interp(query, t, values[:, 0])
        position[row, :, 1] = np.interp(query, t, values[:, 1])
    return position, valid


def _condition(positions):
    singular = np.linalg.svd(positions-positions.mean(axis=0), compute_uv=False)
    return float(singular[-1]/singular[0]) if singular[0] else 0.


def _score_at(a, b, anchors_a, anchors_b):
    displacement = a-b
    translation = np.median(anchors_a-anchors_b, axis=0)
    range_error = np.linalg.norm(a-anchors_a, axis=1)-np.linalg.norm(b-anchors_b, axis=1)
    return (np.linalg.norm(displacement), np.linalg.norm(displacement-translation),
            np.sqrt(np.mean(range_error**2)))


def external_geometry(pairs, rows, sources, ws, score, native):
    """Compare position, anchor translation and anchor ranges on identical support.

    `pairs` is the original complete upper triangle. `rows` indexes its requested
    recipient/supplier pairs, and `score` holds full raw 056 margins. The result
    is ordered like `rows`. Scores are in metres and undefined scores are NaN.
    Anchor diagnostics retain all five query slots, even when the pair falls
    short of the minimum number of geometrically qualified times.
    """
    pairs, sources = _validate_pairs(pairs, sources)
    rows = np.asarray(rows)
    membership = np.asarray(ws)
    margin = np.asarray(score, np.float64)
    n = len(sources)
    if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or (rows < 0).any()
            or (rows >= len(pairs)).any() or membership.shape != (n,)
            or margin.shape != (len(pairs),) or not np.isfinite(margin).all()):
        raise ValueError('ordered pair indices, original WS membership and finite 056 margins required')
    times, latlon, offsets = _native_arrays(native, n)
    _, xy, _ = _coordinates(latlon, offsets)
    _, codes = np.unique(sources, return_inverse=True)
    anchors = _mutual_anchor_groups(pairs, codes, membership, margin)
    starts = times[offsets[:-1]]
    ends = times[offsets[1:]-1]

    size = len(rows)
    result = {key: np.full(size, np.nan) for key in ('RAW', 'TRAN', 'GEO')}
    result.update(observed=np.zeros(size, bool), available=np.zeros(size, bool),
                  anchor_pool=np.zeros(size, np.int64), good_times=np.zeros(size, np.int64),
                  anchor_counts=np.zeros((size, CROSS_TIMES), np.int64),
                  anchor_conditions=np.full((size, CROSS_TIMES), np.nan))
    for i, nodes in enumerate(pairs[rows]):
        a, b = sorted(nodes, key=lambda node: codes[node])
        sa, sb = int(codes[a]), int(codes[b])
        if sa == sb:
            continue
        candidates = [(left, right) for group, left, right in anchors.get((sa, sb), ())
                      if group not in (membership[a], membership[b])]
        result['anchor_pool'][i] = len(candidates)
        start, end = max(starts[a], starts[b]), min(ends[a], ends[b])
        if end-start < CROSS_MIN_OVERLAP_S:
            continue
        query = np.linspace(start, end, CROSS_TIMES)
        positions, observed = _sample_at((a, b), query, times, xy, offsets)
        if not observed.all():
            continue
        result['observed'][i] = True
        if not candidates:
            continue

        anchor_a, anchor_b = np.asarray(candidates, np.int64).T
        positions_a, valid_a = _sample_at(anchor_a, query, times, xy, offsets)
        positions_b, valid_b = _sample_at(anchor_b, query, times, xy, offsets)
        joint_valid = valid_a & valid_b
        result['anchor_counts'][i] = joint_valid.sum(axis=0)
        scores = []
        for q in np.flatnonzero(result['anchor_counts'][i] >= MIN_ANCHORS):
            take = joint_valid[:, q]
            aa, bb = positions_a[take, q], positions_b[take, q]
            condition = min(_condition(aa), _condition(bb))
            result['anchor_conditions'][i, q] = condition
            if condition >= MIN_ANCHOR_CONDITION:
                scores.append(_score_at(positions[0, q], positions[1, q], aa, bb))
        result['good_times'][i] = len(scores)
        if len(scores) >= MIN_GOOD_TIMES:
            raw, translated, geometry = np.median(scores, axis=0)
            result['RAW'][i], result['TRAN'][i], result['GEO'][i] = raw, translated, geometry
            result['available'][i] = True
    return result


def _self_check():
    # The first two anchors are on the mirror axis. The third separates aliases.
    a = np.array([10., 4.])
    reflected = np.array([10., -4.])
    anchors = np.array([[0., 0.], [20., 0.], [0., 15.]])
    shift = np.array([30., -10.])
    raw, translated, geometry = _score_at(a, a+shift, anchors, anchors+shift)
    assert raw > 0 and translated == geometry == 0
    _, _, alias = _score_at(a, reflected+shift, anchors, anchors+shift)
    assert alias > 0 and _condition(anchors) >= MIN_ANCHOR_CONDITION

    nodes = np.vstack(([.001, .0004], [.004, -.0006],
                          anchors*.0001, anchors*.0001+shift*.0001))
    ws = np.array([0, 0, 1, 2, 3, 1, 2, 3])
    sources = np.array(['a', 'b', 'a', 'a', 'a', 'b', 'b', 'b'])
    pairs = np.column_stack(np.triu_indices(len(nodes), 1))
    margin = np.where(ws[pairs[:, 0]] == ws[pairs[:, 1]], 1., -1.)
    query = np.linspace(0., 120., CROSS_TIMES)
    native = (None, np.tile(query, len(nodes)), np.repeat(nodes, CROSS_TIMES, axis=0),
              np.arange(len(nodes)+1)*CROSS_TIMES)
    output = external_geometry(pairs, np.array([0]), sources, ws, margin, native)
    assert output['observed'][0] and output['available'][0]
    assert output['anchor_pool'][0] == 3 and output['good_times'][0] == CROSS_TIMES
    assert output['RAW'][0] > 0
    np.testing.assert_allclose([output['TRAN'][0], output['GEO'][0]], 0., atol=1e-7)

    # The best anchor link loses uniqueness when a competing supplier ties.
    competing = np.flatnonzero((pairs == [2, 6]).all(axis=1))[0]
    margin[competing] = 1.
    output = external_geometry(pairs, np.array([0]), sources, ws, margin, native)
    assert output['observed'][0] and not output['available'][0]
    assert output['anchor_pool'][0] == 1 and np.isnan(output['GEO'][0])
    print('external geometry self-check passed')


if __name__ == '__main__':
    _self_check()
