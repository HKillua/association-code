"""055A prediction-only graph replay and within-parent clique refinement."""
from __future__ import annotations

import numpy as np

from association.evaluation.association import predict
from association.evaluation.fusion_controls import rms_haversine_scores
from association.evaluation.grouping import _partition, group_predicted_tracks


def refine_groups(node_count, groups, pairs, scores, *, threshold):
    """First-fit cliques within each parent, then globally sort by first node.

    No identities/reference inputs. Every pair uses the original >= decision;
    neither thresholds nor ordering are optimized. Missing/extra/reordered
    candidates are refused: this experiment requires the complete i<j graph.
    """
    if (isinstance(node_count, (bool, np.bool_))
            or not isinstance(node_count, (int, np.integer)) or node_count < 0):
        raise ValueError('nonnegative integer node count required')
    parents = _partition(groups, node_count)
    pairs, scores = np.asarray(pairs), np.asarray(scores)
    expected_pairs = np.column_stack(np.triu_indices(node_count, 1))
    if (pairs.dtype.kind not in 'iu' or not np.array_equal(pairs, expected_pairs)
            or scores.shape != (len(expected_pairs),) or scores.dtype.kind not in 'iuf'):
        raise ValueError('complete ordered i<j candidate graph required')
    if threshold is not None and (isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))):
        raise ValueError('finite numeric threshold or None required')
    decisions = predict(scores, threshold)
    adjacency = np.eye(node_count, dtype=bool)
    adjacency[pairs[:, 0], pairs[:, 1]] = decisions
    adjacency[pairs[:, 1], pairs[:, 0]] = decisions
    result = []
    for parent in parents:
        children = []
        for node in parent:
            for child in children:
                if adjacency[node, child].all():
                    child.append(node)
                    break
            else:
                children.append([node])
        result.extend(children)
    result.sort(key=lambda members: members[0])
    return _partition(result, node_count)


def reconstruct_graph(fragment, threshold):
    """Recover exact053A pure-OLS candidates/scores and replay its partition.

    Only saved motion states and group arrays are consumed. Reference values,
    identities and source labels are unnecessary for graph construction.
    """
    states = np.asarray(fragment['native_states'])
    if states.ndim != 3 or states.shape[1:] != (64, 4) or not np.isfinite(states).all():
        raise ValueError('complete finite Q64 native physical states required')
    n = len(states)
    pairs = np.column_stack(np.triu_indices(n, 1))
    scores = np.empty(len(pairs), np.float64)
    for start in range(0, len(pairs), 4096):
        negative_rms = rms_haversine_scores(states[..., :2], pairs[start:start+4096])
        scores[start:start+4096] = -np.log1p(-negative_rms / 1000.)
    old = group_predicted_tracks(n, pairs, scores, threshold=threshold,
                                grouping_policy='anchor_direct', node_order='input')
    offsets = np.r_[0, np.cumsum([len(g) for g in old], dtype=np.int64)]
    flattened = np.asarray([node for group in old for node in group], np.int64)
    membership = np.empty(n, np.int64)
    for group_id, members in enumerate(old):
        membership[members] = group_id
    for key, value in (('ols_group_offsets', offsets), ('ols_group_node_indices', flattened),
                       ('ols_membership', membership)):
        if not np.array_equal(fragment[key], value):
            raise ValueError(f'reconstructed OLS graph does not replay original {key}')
    refined = refine_groups(n, old, pairs, scores, threshold=threshold)
    return pairs, scores, old, refined
