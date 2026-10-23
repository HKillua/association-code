"""078 helper aggregation on the unchanged056 output groups and target masks.

Feature-pooling weights are diagnostics, never physical fusion probabilities.
Every group is predicted before references are used to count selected valid rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from association.evaluation.sequence_position import METHODS, _predict, evaluate_sequence_position
from association.evaluation.learned_group_oracle import _quantiles
from association.evaluation.system_attribution import _same
from association.models.helper_sequence import helper_sequence_inputs
from association.utils.runtime import read_json, write_json, write_json_gzip

SCOPE = '078-helper-sequence-position-evaluation'
BUCKETS = ('zero', 'one', 'two_or_more')
POPULATIONS = ('all_output_group_queries', 'all_output_group_valid_position_queries',
               'unique_selected_valid_position_queries')


def _helper_predict(model, motion, groups, predictions, check, *, initial=False):
    weights, counts, offsets, nodes = [], [], [0], []
    q = motion['states'].shape[1]

    def observe(group, args, result):
        support = args[-1]
        weight = result['helper_weights']
        h = len(group)-1
        if (support.shape != (h, q) or support.dtype != torch.bool or support.device.type != 'cpu'
                or weight.shape != (h, q) or weight.device.type != 'cpu'
                or not torch.isfinite(weight).all() or (weight < 0).any()
                or (weight[~support] != 0).any()):
            raise ValueError('078 feature helper weights/masks are invalid')
        total = weight.sum(0)
        present = support.any(0)
        if (not torch.allclose(total[present], torch.ones_like(total[present]), rtol=1e-6, atol=1e-6)
                or (total[~present] != 0).any()):
            raise ValueError('078 feature helper weights must sum to one when supported, zero otherwise')
        counts.append(support.sum(0).numpy())
        weights.append(weight.detach().numpy())
        # Inputs canonicalize helper order. Cache matching original node indices.
        ordered = sorted(int(x) for x in group[1:])
        nodes.extend(ordered)
        offsets.append(offsets[-1]+h)

    output, residual = _predict(model, motion, groups, predictions, check, initial=initial,
        input_builder=helper_sequence_inputs, observer=observe)
    return output, residual, dict(
        learned_sequence_helper_feature_weights=np.concatenate(weights, axis=0) if weights else np.empty((0, q)),
        group_supported_helper_count=np.asarray(counts, np.int64),
        group_helper_offsets=np.asarray(offsets, np.int64),
        group_helper_node_indices=np.asarray(nodes, np.int64))


def helper_population(fragment, system):
    """Separate every output query from unique selected, valid target queries."""
    counts = np.asarray(fragment['group_supported_helper_count'])
    offsets = np.asarray(fragment['group_helper_offsets'])
    anchors = np.asarray(fragment['group_anchor_node_indices'])
    if (counts.ndim != 2 or len(counts) != len(anchors) or (counts < 0).any()
            or offsets.shape != (len(anchors)+1,) or offsets[0] != 0 or (np.diff(offsets) < 0).any()):
        raise ValueError('078 helper-count population shape differs')
    partition_offsets = fragment['learned_assoc_group_offsets']
    partition_nodes = fragment['learned_assoc_group_node_indices']
    expected = np.asarray([fragment['support'][partition_nodes[a+1:b]].sum(0)
                           for a, b in zip(partition_offsets[:-1], partition_offsets[1:])])
    _same(counts, expected, '078 cached supported helper counts')
    _same(np.diff(offsets), np.diff(partition_offsets)-1, '078 cached helper group lengths')
    expected_nodes = np.concatenate([np.sort(partition_nodes[a+1:b])
        for a, b in zip(partition_offsets[:-1], partition_offsets[1:])])
    _same(fragment['group_helper_node_indices'], expected_nodes, '078 cached helper node correspondence')
    weights = np.asarray(fragment['learned_sequence_helper_feature_weights'])
    support = fragment['support'][expected_nodes]
    if (weights.shape != support.shape or not np.isfinite(weights).all()
            or (weights < 0).any() or (weights[~support] != 0).any()):
        raise ValueError('078 cached feature helper weights differ from visible support')
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        total = weights[start:end].sum(0)
        if not np.allclose(total, (counts[index] > 0).astype(float), rtol=1e-6, atol=1e-6):
            raise ValueError('078 cached helper weight normalization differs')
    t = len(system['anchor_target_ids'])
    selected = system['anchor_aligned_group_indices'][:t]
    valid = (system['anchor_aligned_truth_valid'][:t, :, :2].all(-1)
             & system['anchor_aligned_prediction_valid'][:t, :, :2].all(-1))
    if ((selected < 0)[:, None] & valid).any():
        raise ValueError('078 valid targets require an actual selected output group')
    target_counts = np.zeros(valid.shape, np.int64)
    chosen = selected >= 0
    target_counts[chosen] = counts[selected[chosen]]
    def bucket(value, use):
        return {name: int((use & predicate).sum()) for name, predicate in zip(
            BUCKETS, (value == 0, value == 1, value >= 2))}
    all_counts = bucket(counts, np.ones(counts.shape, bool))
    group_reference = system['anchor_aligned_truth_valid'][:t][system['anchor_group_anchor_target_indices'], :, :2].all(-1)
    group_valid = group_reference & system['anchor_group_valid'][..., :2].all(-1)
    unique_counts = bucket(target_counts, valid)
    if sum(unique_counts.values()) != int(valid.sum()):
        raise ValueError('078 helper buckets lost valid unique-target points')
    return dict(all_output_group_queries=dict(total=int(counts.size), counts=all_counts),
        all_output_group_valid_position_queries=dict(total=int(group_valid.sum()), counts=bucket(counts, group_valid)),
        unique_selected_valid_position_queries=dict(total=int(valid.sum()), counts=unique_counts),
        semantics='Supported helper count at query; all output groups include duplicates/unreferenced groups. Support is interval inclusion, not measurement correctness.')


def helper_error_values(fragment, system):
    t = len(system['anchor_target_ids'])
    selected = system['anchor_aligned_group_indices'][:t]
    count = np.zeros(system['anchor_aligned_truth_valid'][:t].shape[:2], np.int64)
    exists = selected >= 0
    count[exists] = fragment['group_supported_helper_count'][selected[exists]]
    return {bucket: {method: fragment[method+'_position_error'][exists[:, None] & predicate
        & np.isfinite(fragment[method+'_position_error'])] for method in METHODS}
        for bucket, predicate in zip(BUCKETS, (count == 0, count == 1, count >= 2))}


def error_summary(values):
    result = _quantiles(values)
    result.update(count=int(len(values)), mae=float(np.mean(values)) if len(values) else None,
        mse=float(np.mean(np.square(values))) if len(values) else None,
        rmse=float(np.sqrt(np.mean(np.square(values)))) if len(values) else None)
    return result


def evaluate_helper_sequence(model, validation_raw, source056, metadata, output, check=None, *, initial=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError('078 output already exists; no reuse or retry')
    try:
        result = evaluate_sequence_position(model, validation_raw, source056, metadata, output, check,
            initial=initial, predictor=_helper_predict, evaluation_scope=SCOPE)
        rows, errors = [], []
        for sid in result['scene_ids']:
            if check:
                check()
            with np.load(output/'predictions'/f'states-{sid}.npz', allow_pickle=False) as fragment, \
                    np.load(output/'system-predictions'/f'states-{sid}.npz', allow_pickle=False) as system:
                values = helper_error_values(fragment, system)
                errors.append(values)
                rows.append(dict(scene_id=sid, **helper_population(fragment, system),
                    position_error={bucket: {method: error_summary(value) for method, value in methods.items()}
                                    for bucket, methods in values.items()}))
        pooled = {name: dict(total=sum(row[name]['total'] for row in rows), counts={
            bucket: sum(row[name]['counts'][bucket] for row in rows) for bucket in BUCKETS})
            for name in POPULATIONS}
        result.update(scope=SCOPE, aggregation=model.aggregation, helper_population=pooled,
            helper_position_strata={bucket: {method: error_summary(np.concatenate([
                item[bucket][method] for item in errors])) for method in METHODS} for bucket in BUCKETS},
            interpretation='Helper representation screening; learned feature weights are not physical confidence or attribution.')
        write_json_gzip(output/'helper-population.json.gz', rows)
        write_json(output/'summary.json', result)
        return result
    except BaseException as error:
        # Shared evaluation can finish before the additional cache checks fail.
        # Preserve files but do not leave an accepted endpoint in that case.
        if (output/'run.json').exists():
            state = read_json(output/'run.json')
            state.update(status='FAILED', error=f'{type(error).__name__}: {error}')
            write_json(output/'run.json', state)
        raise
