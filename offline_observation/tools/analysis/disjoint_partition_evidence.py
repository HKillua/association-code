"""Fixed, label-blind evidence for the original C78 partition catalogue."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from tools.analysis.disjoint_partition_pilot import Action, changed_edges
from tools.analysis.same_source_swap_pilot import EPSILON, THRESHOLD, dense_edges


LINEAR_MODES = ('OLD_J', 'BOUNDED_J', 'RELATION_BALANCED', 'NODE_BALANCED')
MODES = (*LINEAR_MODES, 'NODE_TAIL', 'NODE_BALANCED_SUPPORT')
RELATIONS = ('same', 'cross')


def _incidence(left, right, values, n):
    return (np.bincount(left, weights=values, minlength=n)
            + np.bincount(right, weights=values, minlength=n))


def scene_evidence(pairs, scores, sources):
    """Build fixed signed edge weights and per-relation absolute node exposure."""
    pairs, scores, sources = np.asarray(pairs), np.asarray(scores, np.float64), np.asarray(sources)
    n = len(sources)
    if (sources.shape != (n,) or n == 0 or pairs.dtype.kind not in 'iu'
            or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))
            or scores.shape != (len(pairs),) or not np.isfinite(scores).all()):
        raise ValueError('original complete scored pairs and one source per node required')
    margin = scores-THRESHOLD
    nonzero = np.abs(margin[margin != 0.])
    scale = float(np.median(nonzero)) if len(nonzero) else 1.
    with np.errstate(over='ignore'):
        bounded = np.tanh(margin/scale)
    left, right = pairs.T
    cross = sources[left] != sources[right]
    by_relation, by_node = np.zeros_like(bounded), np.zeros_like(bounded)
    degree = np.zeros((len(RELATIONS), n))
    active_nodes = np.zeros(len(RELATIONS), np.int64)
    metadata = dict(nonzero_margin_count=len(nonzero), bounded_margin_scale=scale, relations={})
    for relation, (name, mask) in enumerate(zip(RELATIONS, (~cross, cross), strict=True)):
        i, j, b = left[mask], right[mask], bounded[mask]
        mass = float(np.abs(b).sum())
        d = _incidence(i, j, np.abs(b), n)
        degree[relation] = d
        active_nodes[relation] = np.count_nonzero(d)
        if mass:
            by_relation[mask] = b/mass
            part_i = np.divide(b, d[i], out=np.zeros_like(b), where=d[i] > 0)
            part_j = np.divide(b, d[j], out=np.zeros_like(b), where=d[j] > 0)
            u = .5*(part_i+part_j)
            node_mass = float(np.abs(u).sum())
            if not np.isclose(node_mass, .5*active_nodes[relation], rtol=1e-12, atol=1e-12):
                raise ValueError('node exposure normalization does not conserve mass')
            by_node[mask] = u/node_mass
        else:
            node_mass = 0.
        metadata['relations'][name] = dict(pair_count=int(mask.sum()), bounded_abs_mass=mass,
            node_abs_mass_before_normalization=node_mass, nonzero_degree_nodes=int(active_nodes[relation]))
    weights = {mode: dense_edges(pairs, values, n) for mode, values in
               zip(LINEAR_MODES, (margin, bounded, by_relation, by_node), strict=True)}
    evidence = dict(weights=weights, pairs=pairs, bounded=bounded, cross=cross, sources=sources,
                    relation_degree=degree, relation_active_nodes=active_nodes)
    return evidence, metadata


def _baseline_disagreement(evidence, membership):
    left, right = evidence['pairs'].T
    b = evidence['bounded']
    same_group = membership[left] == membership[right]
    disagreement = np.maximum(0., np.where(same_group, -b, b))
    cross = evidence['cross']
    return np.asarray([_incidence(left[mask], right[mask], disagreement[mask], len(membership))
                       for mask in (~cross, cross)])


def partition_scores(membership, evidence):
    """Recompute each whole-partition objective for the choice audit."""
    membership = np.asarray(membership)
    left, right = evidence['pairs'].T
    together = membership[left] == membership[right]
    result = {mode: float(matrix[left[together], right[together]].sum())
              for mode, matrix in evidence['weights'].items()}
    wrong = _baseline_disagreement(evidence, membership)
    tail = 0.
    for w, d in zip(wrong, evidence['relation_degree'], strict=True):
        active = d > 0.
        if active.any():
            tail -= float(np.mean((w[active]/d[active])**2))
    result['NODE_TAIL'] = tail
    result['NODE_BALANCED_SUPPORT'] = result['NODE_BALANCED']
    return result


def _tail_gain(evidence, disagreement, left, right, moved):
    """Use primitive changed edges to update the incident squared disagreement."""
    signed = evidence['weights']['BOUNDED_J'][left, right]*moved
    cross = evidence['sources'][left] != evidence['sources'][right]
    degree = evidence['relation_degree']
    value = 0.
    for relation, mask in enumerate((~cross, cross)):
        d = degree[relation]
        active = d > 0.
        if not active.any():
            continue
        change = _incidence(left[mask], right[mask], signed[mask], len(d))[active]
        old = disagreement[relation, active]/d[active]
        new = (disagreement[relation, active]+change)/d[active]
        value += float(np.mean(old**2-new**2))
    return value


def _uncovered(parts, support):
    return sum(len(nodes)*int(np.count_nonzero(~support[list(nodes)].any(axis=0))) for nodes in parts)


def _choices(record, actions):
    observed = [tuple(choice) for choice in record['choices']]
    masks = [sum(1 << int(group) for group in action.scope) for action in actions]
    expected = [()] + [(i,) for i in range(len(actions))]
    expected.extend((i, j) for i, j in combinations(range(len(actions)), 2) if not masks[i] & masks[j])
    if observed != expected:
        raise ValueError('frozen C78 WS, one-action and compatible two-action order changed')
    return observed


def _select(gain, feasible):
    non_ws = np.flatnonzero(feasible[1:])+1
    if len(non_ws):
        forced = int(non_ws[np.argmax(gain[non_ws])])
        return (forced if gain[forced] > EPSILON else 0), forced
    return 0, 0


def score_catalog(record, membership, groups, support, evidence):
    """Score every original C78 choice and select without identities or references."""
    membership, support = np.asarray(membership), np.asarray(support)
    if (support.ndim != 2 or support.shape[0] != len(membership)
            or support.shape[1] == 0 or support.dtype != bool):
        raise ValueError('fixed boolean node-query support required')
    actions = [Action(tuple(scope), tuple(tuple(part) for part in parts), family)
               for scope, parts, family in record['actions']]
    choices = _choices(record, actions)
    disagreement = _baseline_disagreement(evidence, membership)
    primitive = {mode: [] for mode in (*LINEAR_MODES, 'NODE_TAIL')}
    uncovered = []
    for action in actions:
        left, right, added, removed = changed_edges(action, membership, groups)
        for mode, matrix in evidence['weights'].items():
            primitive[mode].append(float(matrix[left[added], right[added]].sum()
                                         - matrix[left[removed], right[removed]].sum()))
        changed = added | removed
        direction = np.where(removed[changed], 1., -1.)
        primitive['NODE_TAIL'].append(_tail_gain(evidence, disagreement,
                                                left[changed], right[changed], direction))
        original = [groups[group] for group in action.scope]
        uncovered.append(_uncovered(action.parts, support)-_uncovered(original, support))
    gains = {mode: [sum(scores[i] for i in choice) for choice in choices] for mode, scores in primitive.items()}
    gains['NODE_BALANCED_SUPPORT'] = gains['NODE_BALANCED']
    uncovered_delta = [sum(uncovered[i] for i in choice) for choice in choices]
    np.testing.assert_allclose(primitive['OLD_J'], record['primitive_J_gain'], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(gains['OLD_J'], record['J_gain'], rtol=1e-12, atol=1e-10)
    universal = np.ones(len(choices), dtype=bool)
    support_gate = np.asarray(uncovered_delta) <= 0
    accepted, forced, forced_unrestricted = {}, {}, {}
    for mode in MODES:
        values = np.asarray(gains[mode], np.float64)
        feasible = support_gate if mode == 'NODE_BALANCED_SUPPORT' else universal
        accepted[mode], forced[mode], forced_unrestricted[mode] = [], [], []
        for limit in (len(actions)+1, len(choices)):
            pick, top = _select(values[:limit], feasible[:limit])
            accepted[mode].append(pick)
            forced[mode].append(top)
            forced_unrestricted[mode].append(_select(values[:limit], universal[:limit])[1])
    if accepted['OLD_J'] != list(record['selected_J']):
        raise ValueError('original C78 J selection changed')
    return dict(gains=gains, uncovered_delta=uncovered_delta, accepted=accepted, forced=forced,
                forced_unrestricted=forced_unrestricted)


def self_check():
    from tools.analysis.disjoint_partition_pilot import catalog, materialize
    from tools.analysis.same_source_swap_pilot import group_members

    evidence, meta = scene_evidence(np.empty((0, 2), np.int64), [], ['a'])
    assert meta['bounded_margin_scale'] == 1.
    assert all(row['nonzero_degree_nodes'] == 0 for row in meta['relations'].values())
    assert all(not matrix.any() for matrix in evidence['weights'].values())

    pairs = np.column_stack(np.triu_indices(4, 1))
    scores = THRESHOLD + np.array([.8, .2, -.1, .3, -.2, -.7])
    _, meta = scene_evidence(pairs, scores, ['a']*4)
    assert meta['relations']['cross']['bounded_abs_mass'] == 0.
    evidence, meta = scene_evidence(pairs, scores, ['a', 'b', 'a', 'b'])
    for mask in (~evidence['cross'], evidence['cross']):
        left, right = pairs[mask].T
        for mode in ('RELATION_BALANCED', 'NODE_BALANCED'):
            np.testing.assert_allclose(np.abs(evidence['weights'][mode][left, right]).sum(), 1.)
    membership = np.arange(4)
    actions = (Action((0, 1), ((0, 1),), 'merge'), Action((2, 3), ((2, 3),), 'merge'))
    record = catalog(membership, actions, evidence['weights']['OLD_J'])
    support = np.ones((4, 3), bool)
    result = score_catalog(record, membership, group_members(membership), support, evidence)

    def direct_cost(partition, power):
        weights = evidence['weights']['BOUNDED_J']
        src = evidence['sources']
        out = 0.
        for same_relation in (True, False):
            loss = []
            for i in range(len(partition)):
                neighbors = [j for j in range(len(partition)) if j != i and (src[i] == src[j]) == same_relation]
                d = sum(abs(weights[i, j]) for j in neighbors)
                if d:
                    wrong = sum(abs(weights[i, j]) for j in neighbors
                                if (partition[i] == partition[j]) != (weights[i, j] > 0.))
                    loss.append((wrong/d)**power)
            out += sum(loss)/len(loss) if loss else 0.
        return out

    for row, choice in enumerate(record['choices']):
        after = materialize(membership, [actions[i] for i in choice])
        for mode, power in (('NODE_BALANCED', 1), ('NODE_TAIL', 2)):
            np.testing.assert_allclose(result['gains'][mode][row],
                                       direct_cost(membership, power)-direct_cost(after, power),
                                       rtol=1e-12, atol=1e-12)
    negative, _ = scene_evidence(np.array([[0, 1]]), [THRESHOLD-1.], ['a', 'b'])
    split = (Action((0,), ((0,), (1,)), 'split'),)
    merge = (Action((0, 1), ((0, 1),), 'merge'),)
    support = np.array([[True, True], [False, False]])
    for action, membership in ((split, np.array([0, 0])), (merge, np.array([0, 1]))):
        old = catalog(membership, action, negative['weights']['OLD_J'])
        result = score_catalog(old, membership, group_members(membership), support, negative)
        assert result['accepted']['NODE_BALANCED_SUPPORT'] == [0, 0]
        assert result['forced_unrestricted']['NODE_BALANCED_SUPPORT'] == [1, 1]
        assert result['accepted']['NODE_BALANCED'] == ([1, 1] if action == split else [0, 0])
        assert result['forced']['NODE_BALANCED_SUPPORT'] == ([0, 0] if action == split else [1, 1])
    return dict(empty_relation=True, normalization=True, direct_p1_p2_and_disjoint=True,
                unchanged_old_J=True, abstention_and_support=True)


if __name__ == '__main__':
    print(self_check())
