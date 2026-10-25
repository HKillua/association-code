"""098 fixed-group CPU endpoints, source-final equality and bias-off diagnostic."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from association.evaluation.frozen_position_adapter import exact
from association.evaluation.helper_sequence_position import evaluate_helper_sequence
from association.evaluation.sequence_position import _tensor
from association.evaluation.system_attribution import _same
from association.models.helper_sequence import helper_sequence_inputs
from association.models.sequence_position import tangent_position
from association.utils.runtime import read_json, write_json

SCOPE = '098-query-relation-position-evaluation'


def evaluate_query_relation(model, validation_raw, source056, metadata, output, *,
                            source_teacher_checkpoint, check=None, initial=False,
                            relation_enabled=None):
    """Predict every original output group before any reference-based selection.

    The existing completed teacher CPU caches establish initial output equality;
    no teacher forward is repeated. Source/checkpoint/data binding is owned by
    the runner and rechecked by the independent cached comparison.
    """
    output = Path(output)
    # Refusing a duplicate invocation must not rewrite a previously completed
    # endpoint in the failure handler below.
    if output.exists():
        raise FileExistsError('098 output already exists; no reuse or retry')
    enabled = model.mode == 'relation' if relation_enabled is None else relation_enabled
    if type(enabled) is not bool or (enabled and model.mode != 'relation'):
        raise ValueError('098 relation switch must agree with the declared model')
    reference_final = Path(source_teacher_checkpoint).resolve().parents[1]/'final'
    scene_cursor = 0

    def predict(current, motion, groups, predictions, callback, *, initial=False):
        nonlocal scene_cursor
        sid = metadata['scene_ids'][scene_cursor]
        scene_cursor += 1
        with np.load(reference_final/'predictions'/f'states-{sid}.npz', allow_pickle=False) as source:
            teacher = source['learned_sequence_group_states'].copy()
            source_anchors = source['group_anchor_node_indices'].copy()
        q = motion['states'].shape[1]
        exact(source_anchors, np.asarray([group[0] for group in groups], np.int64),
              '098 source final group representatives')
        if teacher.shape != (len(groups), q, 4) or not np.isfinite(teacher).all():
            raise ValueError('098 source final cache must cover every finite group state')
        output_states = predictions['fixed_prior095'].copy()
        residuals, weights, counts, nodes, offsets = [], [], [], [], [0]
        with torch.inference_mode():
            for index, group in enumerate(groups):
                if callback:
                    callback()
                args = helper_sequence_inputs(motion, group, device='cpu')
                result = current(*args, relation_enabled=enabled)
                states = _tensor(result['states'], (q, 4), '098 states', physical=True)
                residual = _tensor(result['residual_m'], (q, 2), '098 residual', physical=True)
                _same(args[0].numpy(), predictions['fixed_prior095'][group[0]], '098 fixed physical prior')
                exact(args[2].numpy(), predictions['mean_supported'][group[0]], '098 ordinary Mean input')
                exact(states[:, 2:], predictions['mean_supported'][group[0], :, 2:], '098 Mean speed/course')
                exact(states, tangent_position(args[0], result['residual_m']).numpy(), '098 output geometry')
                if initial:
                    exact(states, teacher[index], '098 step0 equals source FINAL cache')
                support, weight = args[-1], result['helper_weights']
                if (weight.shape != support.shape or not torch.isfinite(weight).all()
                        or (weight < 0).any() or (weight[~support] != 0).any()
                        or not torch.allclose(weight.sum(0).to(torch.float64),
                            support.any(0).to(torch.float64), rtol=1e-6, atol=1e-6)):
                    raise ValueError('098 helper weights must match support and empty-set normalization')
                count = support.sum(0)
                if not torch.equal(result['helper_support_count'], count):
                    raise ValueError('098 helper support count differs')
                output_states[group[0]] = states
                residuals.append(residual)
                weights.append(weight.detach().numpy())
                counts.append(count.numpy())
                nodes.extend(sorted(int(x) for x in group[1:]))
                offsets.append(len(nodes))
        counts = np.asarray(counts, np.int64).reshape(len(groups), q)
        payload = dict(
            learned_sequence_helper_feature_weights=np.concatenate(weights, axis=0)
                if weights else np.empty((0, q)),
            group_supported_helper_count=counts,
            group_helper_offsets=np.asarray(offsets, np.int64),
            group_helper_node_indices=np.asarray(nodes, np.int64),
            group_any_multi_helper=(counts >= 2).any(-1),
            group_source_teacher_states=teacher)
        return output_states, np.asarray(residuals), payload

    try:
        result = evaluate_helper_sequence(model, validation_raw, source056, metadata, output, check,
            initial=initial, predictor=predict, evaluation_scope=SCOPE)
        if scene_cursor != len(metadata['scene_ids']):
            raise ValueError('098 incomplete scene traversal')
        result.update(mode=model.mode, relation_enabled=enabled,
            source_teacher_checkpoint=str(Path(source_teacher_checkpoint).resolve()),
            source_teacher_forwards=0, initial_output='cached same-seed Transformer final',
            initial_source_cache_exact=bool(initial),
            interpretation='Query-conditioned set fusion; fixed visible-motion relation bias is not calibrated confidence.')
        write_json(Path(output)/'summary.json', result)
        record = read_json(Path(output)/'run.json')
        record.update(mode=model.mode, relation_enabled=enabled, source_teacher_forwards=0,
                      initial_source_cache_exact=bool(initial))
        write_json(Path(output)/'run.json', record)
        return result
    except BaseException as error:
        # The shared evaluator may already have completed before 098-specific
        # checks/metadata fail. An endpoint must not retain a false terminal.
        path = Path(output)/'run.json'
        if path.exists():
            record = read_json(path)
            record.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                          error=f'{type(error).__name__}: {error}')
            write_json(path, record)
        raise
