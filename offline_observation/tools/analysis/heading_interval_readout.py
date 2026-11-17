"""Prediction-only C150 readout on a frozen C149 scene."""

from __future__ import annotations

import numpy as np

from tools.analysis import heading_fallback_fit as c149


NOT_LOW_SPEED, NO_CANDIDATE, DEGENERATE, ENABLED = range(4)
NO_SIDE, ACROSS_QUERY, BEFORE_QUERY, AFTER_QUERY = range(4)
COURSE_VECTOR_EPSILON = 1e-12
LOW_SPEED_BRANCH = c149.BRANCHES.index('enabled_low_speed')


def initial_bearing(interval_keys):
    """Return start-to-end spherical bearings; undefined directions are NaN."""
    keys = np.asarray(interval_keys, dtype=np.float64)
    if keys.ndim != 2 or keys.shape[1] != 6 or not np.isfinite(keys).all():
        raise ValueError('finite interval keys in (t0,t1,lat0,lon0,lat1,lon1) order required')
    coordinates = keys[:, 2:]
    if (np.abs(coordinates) > [90., 180., 90., 180.]).any():
        raise ValueError('interval coordinates outside the geographical bounds')
    phi0, phi1 = np.deg2rad(keys[:, [2, 4]]).T
    delta = np.deg2rad((keys[:, 5] - keys[:, 3] + 180.) % 360. - 180.)
    east = np.cos(phi1) * np.sin(delta)
    north = np.cos(phi0) * np.sin(phi1) - np.sin(phi0) * np.cos(phi1) * np.cos(delta)
    heading = np.mod(np.rad2deg(np.arctan2(east, north)), 360.)
    heading[heading == 360.] = 0.
    heading[np.hypot(east, north) <= COURSE_VECTOR_EPSILON] = np.nan
    return heading


def _group_intervals(query, members, tracks, positions, active, support):
    """Expand the C149 endpoint exclusions with geometry and every original provider."""
    times = [track['canonical_times'] for track in tracks]
    coordinates = [track['canonical_values'][:, :2] for track in tracks]
    rows = np.concatenate([np.column_stack((t, xy)) for t, xy in zip(times, coordinates, strict=True)])
    points, codes = np.unique(rows, axis=0, return_inverse=True)
    ids = np.split(codes.reshape(-1), np.cumsum([len(t) for t in times])[:-1])
    fit_used = np.zeros(len(points), dtype=bool)
    brackets = np.empty((len(members), len(query), 2), dtype=np.int64)
    pairs, provider, native_row = [], [], []
    for member, (t, xy, keys) in enumerate(zip(times, coordinates, ids, strict=True)):
        right = np.searchsorted(t, query).clip(1, len(t)-1)
        brackets[member] = np.column_stack((keys[right-1], keys[right]))
        fit_used[brackets[member, active & support[member]].ravel()] = True
        dt = np.diff(t)
        observed = np.flatnonzero((dt > 0.) & (dt <= c149.GAP_SECONDS)
                                  & (c149._distance(xy[:-1], xy[1:]) >= c149.MIN_DIRECTION_SPEED * dt))
        pairs.append(np.column_stack((keys[observed], keys[observed+1])))
        provider.append(np.full(len(observed), member, dtype=np.int64))
        native_row.append(observed)

    # Mirror C149's conservative longitude fallback operand exclusion,
    # including the full extrapolating anchor when its support is absent.
    angles = np.deg2rad(positions[..., 1])
    count = support.sum(axis=0).clip(1)
    resultant = np.hypot((np.sin(angles)*support).sum(axis=0)/count,
                         (np.cos(angles)*support).sum(axis=0)/count)
    anchor_lon = active & support.any(axis=0) & (resultant <= 1e-12)
    fit_used[brackets[0, anchor_lon].ravel()] = True
    if (anchor_lon & ~support[0]).any():
        fit_used[ids[0]] = True

    pairs, inverse = np.unique(np.concatenate(pairs), axis=0, return_inverse=True)
    a, b = points[pairs[:, 0]], points[pairs[:, 1]]
    geometry = np.column_stack((a[:, 0], b[:, 0], a[:, 1:], b[:, 1:]))
    # np.unique's pair order sorts by the endpoint codes, whose order is
    # lexicographic in (time, lat, lon); starts are sorted for searchsorted.
    return geometry, pairs, inverse.reshape(-1), np.concatenate(provider), np.concatenate(native_row), fit_used, brackets


def _winner(candidates, geometry, query):
    if len(candidates) == 1:
        return candidates[0]
    keys = geometry[candidates]
    age = np.abs((keys[:, 0]-query) + (keys[:, 1]-query)) / 2.
    # Exact geometry duplicates have already been merged. The stable
    # provider table separately orders their copies by member and native row.
    return candidates[np.lexsort((*keys.T[::-1], age))[0]]


def _low_speed_pool(query, members, tracks, positions, support, sources, baseline, group):
    active = baseline['branch'][group] <= LOW_SPEED_BRANCH
    low = active & (baseline['branch'][group] == LOW_SPEED_BRANCH)
    geometry, pairs, inverse, provider, rows, reused, brackets = _group_intervals(
        query, members, tracks, positions, active, support)
    starts, ends = geometry[:, 0], geometry[:, 1]
    lower = np.searchsorted(starts, query-c149.GAP_SECONDS)
    upper = np.searchsorted(starts, query+c149.GAP_SECONDS, side='right')
    fit_clear = ~reused[pairs].any(axis=1)
    provider_order = np.argsort(inverse, kind='stable')
    provider_offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(geometry)))]
    source_codes = np.unique(sources, return_inverse=True)[1]
    pools = []
    picks = np.full(len(query), -1, dtype=np.int64)
    for qi in np.flatnonzero(low):
        possible = np.arange(lower[qi], upper[qi])
        possible = possible[ends[possible] <= query[qi]+c149.GAP_SECONDS]
        c149.require(len(possible) == baseline['interval_count'][group, qi],
                     'C150 low-speed candidate count differs from C149')
        # Every low-speed slot is a fallback; the C149 anchor is included
        # even if it has no support or merely may be used on resultant zero.
        a_used = brackets[np.r_[True, support[1:, qi]], qi].ravel()
        clear = possible[fit_clear[possible]]
        clear = clear[~np.isin(pairs[clear], a_used).any(axis=1)]
        c149.require(len(clear) == baseline['nonreuse_count'][group, qi],
                     'C150 low-speed nonreuse pool differs from C149')
        if len(clear):
            supplier_rows = np.concatenate([provider_order[provider_offsets[k]:provider_offsets[k+1]]
                                            for k in clear])
            supply = np.zeros(len(members), bool)
            supply[provider[supplier_rows]] = True
            np.testing.assert_array_equal(supply, baseline['unused_members'][members, qi])
            supplier_pairs = pairs[inverse[supplier_rows]]
            cross = c149.different_source_disjoint(supplier_pairs[:, 0], supplier_pairs[:, 1],
                                                   source_codes[provider[supplier_rows]])
            c149.require(cross == baseline['cross_source_pair'][group, qi],
                         'C150 low-speed disjoint cross-source evidence differs from C149')
            pools.append((qi, clear))
            picks[qi] = _winner(clear, geometry, query[qi])
    if not pools:
        return None
    selected_keys = np.unique(np.concatenate([pool for _, pool in pools]))
    retained = np.zeros(len(geometry), bool)
    retained[selected_keys] = True
    copies = provider_order[retained[inverse[provider_order]]]
    copy_counts = np.diff(provider_offsets)[selected_keys]
    slots = np.full(len(geometry), -1, np.int64)
    slots[selected_keys] = np.arange(len(selected_keys))
    picks[picks >= 0] = slots[picks[picks >= 0]]
    return (geometry[selected_keys], members[provider[copies]], rows[copies], copy_counts,
            [(qi, slots[pool]) for qi, pool in pools], picks)


def freeze_scene(motion, tracks, native, baseline_saved):
    """Attest the C149 archive replay and replace only the original low-speed angle.

    Return the original arrays and the new states, reasons, chosen interval
    metadata, and two CSR arrays: one for each low-speed Q's unique keys,
    another for each key's full provider provenance. Native row indices refer
    to canonical per-member rows, with the chronological left endpoint stored.
    A chosen degenerate key retains its provenance even though its angle stays B.
    """
    reconstructed = c149.freeze_scene(motion, tracks, native, baseline_saved['membership'])
    baseline = dict(baseline_saved)
    if set(reconstructed) != set(baseline):
        raise ValueError('C149 scene schema differs')
    for key, value in reconstructed.items():
        np.testing.assert_array_equal(value, baseline[key], err_msg=f'C149 archived {key} differs')
    query = baseline['query']
    groups = c149.group_members(baseline['membership'])
    branch = baseline['branch']
    low = branch == LOW_SPEED_BRANCH
    chosen = np.full(branch.shape, -1, dtype=np.int64)
    per_query = np.zeros(branch.shape, dtype=np.int64)
    geometries, candidate_groups, copies_per_key, nodes, native_rows, used_keys = [], [], [], [], [], []
    catalog_size = 0
    sources = motion['sources']
    for gi, members in enumerate(groups):
        if not low[gi].any():
            continue
        result = _low_speed_pool(query, members, [tracks[n] for n in members],
                                 motion['states'][members, :, :2], baseline['support'][members],
                                 sources[members], baseline, gi)
        if result is None:
            continue
        geometry, source_nodes, row, counts, query_pools, pick = result
        geometries.append(geometry)
        candidate_groups.append(np.full(len(geometry), gi, dtype=np.int64))
        copies_per_key.append(counts)
        nodes.append(source_nodes)
        native_rows.append(row)
        for qi, pool in query_pools:
            per_query[gi, qi] = len(pool)
            used_keys.append(pool + catalog_size)
        has_candidate = pick >= 0
        chosen[gi, has_candidate] = pick[has_candidate] + catalog_size
        catalog_size += len(geometry)

    geometry = np.concatenate(geometries) if geometries else np.empty((0, 6), dtype=np.float64)
    empty = np.empty(0, dtype=np.int64)
    provider_nodes = np.concatenate(nodes) if nodes else empty
    provider_rows = np.concatenate(native_rows) if native_rows else empty
    provider_offsets = np.r_[0, np.cumsum(np.concatenate(copies_per_key) if copies_per_key else empty)]
    pool_offsets = np.r_[0, np.cumsum(per_query.ravel())]
    np.testing.assert_array_equal(per_query, np.where(low, baseline['nonreuse_count'], 0))
    selected = chosen >= 0
    np.testing.assert_array_equal(selected, per_query > 0)

    selected_geometry = np.full((*branch.shape, 6), np.nan)
    selected_geometry[selected] = geometry[chosen[selected]]
    selected_node = np.full(branch.shape, -1, dtype=np.int64)
    selected_row = selected_node.copy()
    # The stable provider list is in original member then native row order.
    representative = provider_offsets[chosen[selected]]
    selected_node[selected] = provider_nodes[representative]
    selected_row[selected] = provider_rows[representative]
    t0, t1 = selected_geometry[..., 0], selected_geometry[..., 1]
    dt = t1 - t0
    midpoint_age = np.abs((t0-query) + (t1-query)) / 2.
    max_age = np.maximum(np.abs(t0-query), np.abs(t1-query))
    distance = np.full(branch.shape, np.nan)
    distance[selected] = c149._distance(selected_geometry[selected][:, 2:4], selected_geometry[selected][:, 4:6])
    side = np.where(t1 < query, BEFORE_QUERY, np.where(t0 > query, AFTER_QUERY, ACROSS_QUERY))
    side = np.where(selected, side, NO_SIDE).astype(np.uint8)
    angles = np.full(branch.shape, np.nan)
    angles[selected] = initial_bearing(selected_geometry[selected])
    enabled = selected & np.isfinite(angles)
    reason = np.where(low, NO_CANDIDATE, NOT_LOW_SPEED).astype(np.uint8)
    reason[selected] = DEGENERATE
    reason[enabled] = ENABLED
    states = baseline['states'].copy()
    states[..., 3][enabled] = angles[enabled]
    np.testing.assert_array_equal(states[..., :3], baseline['states'][..., :3])
    np.testing.assert_array_equal(states[~enabled], baseline['states'][~enabled])
    return dict(baseline, new_states=states, enabled=enabled, fallback_reason=reason, sources=sources,
                selected_interval_index=chosen, selected_interval=selected_geometry,
                selected_provider_node=selected_node, selected_provider_native_row=selected_row,
                selected_midpoint_age_seconds=midpoint_age, selected_max_age_seconds=max_age,
                selected_duration_seconds=dt, selected_distance_metres=distance, selected_time_side=side,
                candidate_keys=geometry, candidate_groups=np.concatenate(candidate_groups) if candidate_groups else empty,
                candidate_provider_offsets=provider_offsets, candidate_provider_nodes=provider_nodes,
                candidate_provider_native_rows=provider_rows, pool_query_offsets=pool_offsets,
                pool_candidate_indices=np.concatenate(used_keys) if used_keys else empty)
