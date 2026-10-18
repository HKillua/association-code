"""Explicit anchor grouping and recursive TFM inference, without reference data.

The fixed-order, direct-anchor rule and FC-memory feedback implement the stated
Algorithm 1 adaptation. They are explicit call arguments, not a claim that the
paper resolves its recursive-feedback ambiguity. Identity grouping is a separate
evaluation helper; predicted grouping and fusion never read identities/targets.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from association.data.archive import DataError
from association.models.fusion import PairFusion
from association.evaluation.association import predict


def _array(value):
    """Grouping is discrete; detach model scores before CPU indexing."""
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _input_order(node_order):
    if node_order != 'input':
        raise DataError("node_order must explicitly be 'input'")


def group_predicted_tracks(node_count, pairs, scores, *, threshold,
                            grouping_policy, node_order):
    """Partition all input nodes by accepted direct edges from each anchor.

    ``pairs[E,2]`` indexes the fixed input node order. Accept ``scores >=
    threshold``, including ties. Explicit ``threshold=None`` means all reject,
    matching the existing selection/metric contract. Missing edges do not connect. Iterate
    the first unassigned node as anchor, take its still-unassigned accepted
    neighbors in input order, and mark that whole group assigned. Do not extend
    a group through a neighbor's edges or reassign an already consumed node.

    Duplicate undirected candidates (including reversed duplicates), self-edges
    and nonfinite scores are rejected instead of choosing an aggregation rule.
    The function accepts arrays or score/index tensors, and returns index lists.
    """
    if grouping_policy != 'anchor_direct':
        raise DataError("grouping_policy must explicitly be 'anchor_direct'")
    _input_order(node_order)
    if isinstance(node_count, (bool, np.bool_)) or not isinstance(node_count, (int, np.integer)) or node_count < 0:
        raise DataError('node_count must be a nonnegative integer')
    if threshold is not None and (isinstance(threshold, (bool, np.bool_))
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not math.isfinite(float(threshold))):
        raise DataError('threshold must be an explicit finite number or None (all reject)')
    candidates, values = _array(pairs), _array(scores)
    if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.dtype.kind not in 'iu':
        raise DataError('pairs must have integer shape [E,2]')
    if isinstance(pairs, (list, tuple)) and any(
            isinstance(index, (bool, np.bool_)) for pair in pairs for index in pair):
        raise DataError('pair endpoints cannot be boolean')
    if values.shape != (len(candidates),) or values.dtype.kind not in 'iuf' or not np.isfinite(values).all():
        raise DataError('scores must be finite numeric values with shape [E]')
    if ((candidates < 0) | (candidates >= node_count)).any():
        raise DataError('pair endpoint outside the input nodes')
    if (candidates[:, 0] == candidates[:, 1]).any():
        raise DataError('self-edges are not association candidates')

    accepted = predict(values, threshold)
    neighbors = [[] for _ in range(node_count)]
    seen = set()
    for (left, right), decision in zip(candidates, accepted):
        left, right = int(left), int(right)
        key = (min(left, right), max(left, right))
        if key in seen:
            raise DataError('duplicate undirected candidate edge')
        seen.add(key)
        if decision:
            neighbors[left].append(right)
            neighbors[right].append(left)
    assigned = np.zeros(node_count, dtype=bool)
    groups = []
    for anchor in range(node_count):
        if assigned[anchor]:
            continue
        members = [anchor] + sorted(node for node in neighbors[anchor] if not assigned[node])
        assigned[members] = True
        groups.append(members)
    return groups


def group_tracks_by_identity(identity_ids, *, node_order):
    """Group true identities for external evaluation only, in input order.

    This helper never calls the prediction or fusion paths. Supplying its groups
    to fusion is an explicit oracle comparison, not predicted association.
    """
    _input_order(node_order)
    identities = np.asarray(identity_ids, dtype=object)
    if identities.ndim != 1:
        raise DataError('identity_ids must be a one-dimensional sequence')
    groups = {}
    for index, identity in enumerate(identities):
        if not isinstance(identity, str) or not identity:
            raise DataError('each evaluation identity must be a nonempty string')
        groups.setdefault(str(identity), []).append(index)
    return list(groups.values())


def _partition(groups, node_count):
    """Validate fixed-order groups without sorting or silently repairing them."""
    if not isinstance(groups, (list, tuple)):
        raise DataError('groups must be a sequence of node-index sequences')
    copied = []
    seen = set()
    last_anchor = -1
    for group in groups:
        members = _array(group)
        if members.ndim != 1 or not len(members) or members.dtype.kind not in 'iu':
            raise DataError('every group must contain a nonempty integer index sequence')
        if isinstance(group, (list, tuple)) and any(isinstance(i, (bool, np.bool_)) for i in group):
            raise DataError('group indices cannot be boolean')
        if ((members < 0) | (members >= node_count)).any():
            raise DataError('group index outside the completed tracks')
        if (members[1:] <= members[:-1]).any() or int(members[0]) <= last_anchor:
            raise DataError('groups and their members must follow fixed input-node order')
        current = [int(index) for index in members]
        if any(index in seen for index in current):
            raise DataError('each node must appear in exactly one group')
        seen.update(current)
        copied.append(current)
        last_anchor = current[0]
    if len(seen) != node_count:
        raise DataError('groups must cover every input node, including singletons')
    return copied


@torch.inference_mode()
def fuse_track_groups(fusion: PairFusion, completed_tracks: Tensor, groups, *,
                      feedback_policy, singleton_policy):
    """Fuse each full partition group using the explicit FC-memory recurrence.

    ``fusion`` must already be in eval mode; this function does not change model
    modes or weights. Completed tracks are ``[N,L,d]`` TCM outputs in fixed input
    node order. At every step the accumulated FC memory is the *left* input and
    the next original completed fragment is the *right* input. The final step's
    decoder states are the group's physical output. No state or true identity
    is fed back into the model.

    Return one record per group, preserving mixed-identity predicted groups and
    singletons. A singleton has ``status='singleton_missing'`` and both tensor
    outputs ``None``; the caller must handle its observed single-source baseline.
    It is never self-fused or silently excluded. No reference, label, threshold,
    normalization, or file access is part of this function.
    """
    if feedback_policy != 'fc_memory':
        raise DataError("feedback_policy must explicitly be 'fc_memory'")
    if singleton_policy != 'missing':
        raise DataError("singleton_policy must explicitly be 'missing'")
    if not isinstance(fusion, PairFusion):
        raise DataError('fusion must be a PairFusion model')
    if any(module.training for module in fusion.modules()):
        raise DataError('PairFusion must already be in eval mode for inference')
    if not isinstance(completed_tracks, Tensor):
        raise DataError('completed_tracks must be a tensor')
    # Reuse the model's feature/device/dtype contract even for all-singleton or
    # empty scenes, where no call to PairFusion.forward would otherwise occur.
    fusion._validate_features(completed_tracks)
    partition = _partition(groups, len(completed_tracks))
    records = []
    for members in partition:
        record = {'node_indices': members, 'anchor': members[0],
                  'fusion_steps': len(members) - 1, 'states': None, 'fused_features': None,
                  'status': 'singleton_missing' if len(members) == 1 else 'fused'}
        if len(members) > 1:
            memory = completed_tracks[members[0]:members[0] + 1]
            for index in members[1:]:
                result = fusion(memory, completed_tracks[index:index + 1])
                memory = result['fused_features']
            record['states'] = result['states'][0].clone()
            record['fused_features'] = memory[0].clone()
        records.append(record)
    return records
