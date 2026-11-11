"""C78 blind, bounded single-scope partition proposals on fixed WS."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from association.evaluation.mixture_opportunity import _distance as haversine
from association.evaluation.native_motion import _short_delta
from association.evaluation.physical import EARTH_RADIUS_METRES
from tools.analysis.same_source_swap_pilot import THRESHOLD, canonical_membership, dense_edges, group_members


SINGLE_LIMIT = 16
PAIR_LIMIT = 16
PAIR_DEGREE = 3
CUT_LIMIT = 3
ARMS = ('SCORE_TREE', 'NATIVE_TREE')


@dataclass(frozen=True)
class Action:
    scope: tuple[int, ...]
    parts: tuple[tuple[int, ...], ...]
    family: str

    @property
    def key(self):
        return self.scope, self.parts


def _xyz(latlon):
    lat, lon = np.deg2rad(latlon).T
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))


def _angles(first, second):
    return np.arctan2(np.linalg.norm(np.cross(first, second), axis=-1),
                      np.clip(np.sum(first*second, axis=-1), -1., 1.))


def _outward_motion(endpoint, inward, interval):
    angle = _angles(endpoint, inward)
    if (angle >= np.pi-1e-10).any():
        raise ValueError('antipodal adjacent native positions do not define an endpoint course')
    tangent = np.cross(endpoint, np.cross(endpoint, inward))
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    direction = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-15)
    speed = np.where(norm[:, 0] > 1e-15, angle / interval, 0.)
    return direction, speed


def _advance(endpoint, direction, speed, gap):
    angle = np.remainder(speed*gap, 2.*np.pi)[:, None]
    return np.cos(angle)*endpoint + np.sin(angle)*direction


def _sample_positions(nodes, queries, times, latlon, offsets, longitude):
    result = np.empty((*queries.shape, 2), np.float64)
    order = np.argsort(nodes, kind='stable')
    counts = np.r_[0, np.cumsum(np.bincount(nodes, minlength=len(offsets)-1))]
    for node in np.flatnonzero(np.diff(counts)):
        a, b = offsets[node:node+2]
        take = order[counts[node]:counts[node+1]]
        result[take, :, 0] = np.interp(queries[take], times[a:b], latlon[a:b, 0])
        result[take, :, 1] = np.interp(queries[take], times[a:b], longitude[a:b])
    return result


def native_pair_distances(pairs, times, latlon, offsets):
    """Native-time pair distances in metres, without measured speed/course."""
    starts = offsets[:-1]
    ends = offsets[1:]-1
    first, last = times[starts], times[ends]
    a, b = pairs.T
    common_start = np.maximum(first[a], first[b])
    common_end = np.minimum(last[a], last[b])
    overlap = common_end > common_start
    instant = common_end == common_start
    outside = common_end < common_start
    distances = np.empty(len(pairs), np.float64)

    if (overlap | instant).any():
        longitude = np.empty(len(times), np.float64)
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            raw = latlon[start:stop, 1]
            longitude[start:stop] = raw[0] + np.r_[0., np.cumsum(_short_delta(np.diff(raw)))]
        for region, count in ((overlap, 5), (instant, 1)):
            if not region.any():
                continue
            start, end = common_start[region, None], common_end[region, None]
            query = np.linspace(start, end, count, axis=1)[:, :, 0]
            first_xy = _sample_positions(a[region], query, times, latlon, offsets, longitude)
            second_xy = _sample_positions(b[region], query, times, latlon, offsets, longitude)
            distances[region] = np.median(haversine(first_xy, second_xy), axis=1)

    if outside.any():
        xyz_first, xyz_last = _xyz(latlon[starts]), _xyz(latlon[ends])
        before_dir, before_speed = _outward_motion(xyz_first, _xyz(latlon[starts+1]),
                                                   times[starts+1]-first)
        after_dir, after_speed = _outward_motion(xyz_last, _xyz(latlon[ends-1]),
                                                last-times[ends-1])
        left, right = a[outside], b[outside]
        left_is_earlier = last[left] < first[right]
        earlier = np.where(left_is_earlier, left, right)
        later = np.where(left_is_earlier, right, left)
        gap = first[later]-last[earlier]
        ahead = _advance(xyz_last[earlier], after_dir[earlier], after_speed[earlier], gap)
        back = _advance(xyz_first[later], before_dir[later], before_speed[later], gap)
        distances[outside] = (.5*EARTH_RADIUS_METRES
                              * (_angles(ahead, xyz_first[later]) + _angles(back, xyz_last[earlier])))
        longest_gap = float(gap.max())
    else:
        longest_gap = None
    if not np.isfinite(distances).all():
        raise ValueError('nonfinite native pair distance')
    return distances, dict(pair_count=len(pairs), overlap_count=int(overlap.sum()),
                           touching_count=int(instant.sum()), gap_count=int(outside.sum()),
                           longest_gap_seconds=longest_gap)


def _rank_pair_scopes(scopes, weights):
    left, right = scopes.T
    ranks = np.lexsort((right, left, weights))
    degree = Counter()
    selected = []
    for row in ranks:
        a, b = int(left[row]), int(right[row])
        if degree[a] >= PAIR_DEGREE or degree[b] >= PAIR_DEGREE:
            continue
        selected.append((a, b))
        degree[a] += 1
        degree[b] += 1
        if len(selected) == PAIR_LIMIT:
            break
    return tuple(selected)


def _common_pair_scopes(membership, pairs, scores, native_distances, group_count):
    group_a, group_b = membership[pairs].T
    different = group_a != group_b
    lo = np.minimum(group_a[different], group_b[different])
    hi = np.maximum(group_a[different], group_b[different])
    score_cost = np.full((group_count, group_count), np.inf)
    native_cost = np.full_like(score_cost, np.inf)
    np.minimum.at(score_cost, (lo, hi), -scores[different])
    np.minimum.at(native_cost, (lo, hi), native_distances[different])
    left, right = np.triu_indices(group_count, 1)
    scopes = np.column_stack((left, right))
    return (_rank_pair_scopes(scopes, score_cost[left, right]),
            _rank_pair_scopes(scopes, native_cost[left, right]))


def _parts(*groups):
    return tuple(sorted(tuple(map(int, np.sort(group))) for group in groups))


def _weakest_cuts(nodes, cost):
    """Kruskal tree with node-pair ties and up to three weakest-edge cuts."""
    size = len(nodes)
    left, right = np.triu_indices(size, 1)
    weights = cost[nodes[left], nodes[right]]
    ranks = np.lexsort((right, left, weights))
    parent = list(range(size))
    adjacency = [[] for _ in nodes]
    tree = []

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row in ranks:
        a, b = int(left[row]), int(right[row])
        ra, rb = root(a), root(b)
        if ra == rb:
            continue
        parent[rb] = ra
        adjacency[a].append(b)
        adjacency[b].append(a)
        tree.append((float(weights[row]), a, b))
        if len(tree) == size-1:
            break
    if len(tree) != size-1:
        raise ValueError('the complete scope graph must admit a spanning tree')

    cut_order = sorted(tree, key=lambda edge: (-edge[0], edge[1], edge[2]))[:CUT_LIMIT]
    for _, a, b in cut_order:
        seen = np.zeros(size, bool)
        seen[0] = True
        pending = [0]
        while pending:
            node = pending.pop()
            for neighbor in adjacency[node]:
                if seen[neighbor] or (node == a and neighbor == b) or (node == b and neighbor == a):
                    continue
                seen[neighbor] = True
                pending.append(neighbor)
        yield _parts(nodes[seen], nodes[~seen])


def propose(membership, pairs, scores, native):
    """Freeze both blind proposal pools on exactly the same scopes and cut budget."""
    n = len(membership)
    pairs = np.asarray(pairs)
    scores = np.asarray(scores, np.float64)
    if (n == 0 or not np.array_equal(canonical_membership(membership, n), membership)
            or pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))
            or scores.shape != (len(pairs),) or not np.isfinite(scores).all()):
        raise ValueError('nonempty canonical WS and frozen complete scored pairs required')
    _, times, latlon, offsets = map(np.asarray, native)
    if (offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
            or offsets[-1] != len(times) or (np.diff(offsets) < 2).any()
            or times.ndim != 1 or latlon.shape != (len(times), 2)
            or not np.isfinite(times).all() or not np.isfinite(latlon).all()
            or (np.abs(latlon) > [90., 180.]).any()
            or any((np.diff(times[a:b]) <= 0).any()
                   for a, b in zip(offsets[:-1], offsets[1:], strict=True))):
        raise ValueError('finite strictly increasing native positions are required')
    membership = np.asarray(membership)
    groups = group_members(membership)
    single_ranking = sorted((i for i, g in enumerate(groups) if len(g) >= 2),
                            key=lambda i: (-len(groups[i]), i))[:SINGLE_LIMIT]
    distances, native_meta = native_pair_distances(pairs, times, latlon, offsets)
    score_pairs, native_pairs = _common_pair_scopes(membership, pairs, scores, distances, len(groups))
    common_pairs = tuple(sorted(set(score_pairs) | set(native_pairs)))
    scopes = tuple((i,) for i in sorted(single_ranking)) + common_pairs
    score_matrix = dense_edges(pairs, -scores, n)
    native_matrix = dense_edges(pairs, distances, n)

    contexts = []
    controls = []
    for scope in scopes:
        original = _parts(*(groups[g] for g in scope))
        nodes = np.sort(np.concatenate([groups[g] for g in scope]))
        contexts.append((scope, original, nodes))
        if len(scope) == 1:
            local = -score_matrix[np.ix_(nodes, nodes)]-THRESHOLD
            np.fill_diagonal(local, 0.)
            peel = int(np.argmin(local.sum(axis=1)))
            controls.append(Action(scope, _parts(nodes[peel:peel+1], np.delete(nodes, peel)), 'peel'))
        else:
            controls.append(Action(scope, _parts(nodes), 'merge'))

    arms, arm_meta = {}, {}
    for arm, cost in zip(ARMS, (score_matrix, native_matrix), strict=True):
        unique = {action.key: action for action in controls}
        slots, no_op, duplicates = 0, 0, 0
        for scope, original, nodes in contexts:
            for parts in _weakest_cuts(nodes, cost):
                slots += 1
                if parts == original:
                    no_op += 1
                    continue
                action = Action(scope, parts, 'split' if len(scope) == 1 else 'recut')
                if action.key in unique:
                    duplicates += 1
                else:
                    unique[action.key] = action
        arms[arm] = tuple(unique[key] for key in sorted(unique))
        arm_meta[arm] = dict(tree_cut_slots=slots, no_op_cuts=no_op, duplicate_cuts=duplicates,
                             shared_control_slots=len(controls), unique_actions=len(unique),
                             action_family_counts=dict(Counter(action.family for action in unique.values())))
    if arm_meta[ARMS[0]]['tree_cut_slots'] != arm_meta[ARMS[1]]['tree_cut_slots']:
        raise AssertionError('candidate arms must have equal raw tree cut slots')
    metadata = dict(original_group_count=len(groups), original_node_count=n,
                    single_scope_ranking=single_ranking, score_pair_scope_ranking=score_pairs,
                    native_pair_scope_ranking=native_pairs, common_pair_scopes=common_pairs,
                    native_pairs=native_meta, arms=arm_meta)
    return arms, metadata


def self_check():
    # Spherical continuation across the date line and exactly touching intervals.
    positions = np.array([[0, 179.], [0, -179.], [0, -177.], [0, -175.],
                          [0, -179.], [0, -178.]])
    time = np.array([0., 2., 4., 6., 2., 3.])
    pairs = np.column_stack(np.triu_indices(3, 1))
    distance, info = native_pair_distances(pairs, time, positions, np.array([0, 2, 4, 6]))
    np.testing.assert_allclose(distance, 0., atol=1e-6)
    assert info['gap_count'] == 2 and info['touching_count'] == 1
    # Date line overlap uses shortest-arc interpolation, including the middle time.
    positions[2:4] = [[0., -180.], [0., -179.]]
    time[2:4] = [1., 2.]
    distance, info = native_pair_distances(pairs, time, positions, np.array([0, 2, 4, 6]))
    np.testing.assert_allclose(distance, 0., atol=1e-6)
    assert info['overlap_count'] == 1 and info['touching_count'] == 2

    # Equal edge weights choose the original node-pair order both for tree and cuts.
    assert tuple(_weakest_cuts(np.arange(4), np.zeros((4, 4)))) == (
        ((0, 2, 3), (1,)), ((0, 1, 3), (2,)), ((0, 1, 2), (3,)))
    ranks = _rank_pair_scopes(np.column_stack(np.triu_indices(20, 1)), np.zeros(190))
    assert len(ranks) == PAIR_LIMIT and max(Counter(n for pair in ranks for n in pair).values()) == PAIR_DEGREE

    nodes = np.arange(6)
    membership = np.array([0, 0, 0, 1, 1, 2])
    times = np.tile([0., 1.], len(nodes))
    lon = np.column_stack((np.repeat(0., len(times)), np.repeat(nodes, 2)*.01))
    pairs = np.column_stack(np.triu_indices(len(nodes), 1))
    arms, meta = propose(membership, pairs, np.full(len(pairs), THRESHOLD+1.),
                         (None, times, lon, np.arange(0, len(times)+1, 2)))
    assert meta['arms'][ARMS[0]]['tree_cut_slots'] == meta['arms'][ARMS[1]]['tree_cut_slots']
    for actions in arms.values():
        assert len({action.key for action in actions}) == len(actions)
        assert sum(action.family == 'peel' for action in actions) == 2
        assert sum(action.family == 'merge' for action in actions) == 3
        for action in actions:
            assert action.parts and all(action.parts)
            assert sorted(node for part in action.parts for node in part) == list(
                np.flatnonzero(np.isin(membership, action.scope)))
    return dict(dateline_and_boundary=True, stable_tree_and_scope_caps=True, legal_actions=True)


if __name__ == '__main__':
    print(self_check())
