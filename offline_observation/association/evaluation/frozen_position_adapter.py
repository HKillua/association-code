"""090 source-final endpoints, explicit teacher-origin increments, full population."""
from __future__ import annotations

import numpy as np
import torch

from association.evaluation.sequence_position import _tensor, evaluate_sequence_position
from association.evaluation.system_attribution import _same
from association.models.sequence_position import sequence_inputs, tangent_position
from association.training.sequence_position import load_checkpoint as load_teacher


def exact(left, right, label):
    left, right = np.asarray(left), np.asarray(right)
    if (left.dtype != right.dtype or left.shape != right.shape
            or not np.array_equal(left, right, equal_nan=left.dtype.kind in 'fc')):
        raise ValueError('090 exact source/invariance check failed: '+label)


def evaluate_frozen_position_adapter(model, validation_raw, source056, metadata, output, *,
                                    source_teacher_checkpoint, check=None, initial=False):
    original, _ = load_teacher(source_teacher_checkpoint, device='cpu'); original.eval()

    def predict(current, motion, groups, predictions, callback, *, initial=False):
        q = motion['states'].shape[1]; output_states = predictions['fixed_prior095'].copy()
        residuals, teachers, teacher_residuals, masks, raw = [], [], [], [], []
        with torch.inference_mode():
            for group in groups:
                if callback: callback()
                args = sequence_inputs(motion, group, device='cpu')
                support = torch.as_tensor(motion['support'][group[0]], dtype=torch.bool)
                reference = original(*args)
                result = current(*args, support)
                states = _tensor(result['states'], (q, 4), '090 states', physical=True)
                teacher = _tensor(reference['states'], (q, 4), '090 source teacher', physical=True)
                residual = _tensor(result['residual_m'], (q, 2), '090 actual residual', physical=True)
                _same(args[0].numpy(), predictions['fixed_prior095'][group[0]], '090 original fixed095 input')
                _same(args[2].numpy(), predictions['mean_supported'][group[0]], '090 original Mean input')
                exact(states[:, 2:], predictions['mean_supported'][group[0], :, 2:], 'unchanged Mean speed/course')
                if current.mode == 'full_finetune':
                    rebuilt = tangent_position(args[0], result['residual_m'])
                else:
                    exact(result['teacher_states'].numpy(), teacher, 'frozen teacher forward')
                    allowed = np.ones(q, bool) if current.mode == 'all_queries' else ~support.numpy()
                    exact(result['update_allowed'].numpy(), allowed, 'visible route')
                    raw.append(result['raw_increment_m'].numpy())
                    exact(residual, raw[-1]*allowed[:, None], 'visible mask applied to raw increment')
                    rebuilt = tangent_position(reference['states'], result['residual_m'])
                    if current.mode == 'anchor_unsupported':
                        exact(states[support.numpy()], teacher[support.numpy()], 'protected source-teacher output')
                exact(states, rebuilt.numpy(), 'two-dimensional geometric output')
                if initial:
                    exact(states, teacher, 'step0 equals source FINAL')
                    if current.mode != 'full_finetune': exact(residual, np.zeros_like(residual), 'zero new head')
                output_states[group[0]] = states; residuals.append(residual)
                teachers.append(teacher); teacher_residuals.append(reference['residual_m'].numpy()); masks.append(support.numpy())
        payload = dict(group_source_teacher_states=np.asarray(teachers),
            group_source_teacher_residual_m=np.asarray(teacher_residuals), group_anchor_support=np.asarray(masks))
        if current.mode != 'full_finetune': payload['group_adapter_raw_increment_m'] = np.asarray(raw)
        return output_states, np.asarray(residuals), payload

    is_adapter = model.mode != 'full_finetune'
    result = evaluate_sequence_position(model, validation_raw, source056, metadata, output,
        check=check, initial=initial, predictor=predict, evaluation_scope='090-source-final-position-evaluation',
        residual_key='group_teacher_increment_m' if is_adapter else 'group_tangent_residual_m')
    result.update(residual_origin='frozen source-final teacher' if is_adapter else 'fixed095 physical prior',
        source_teacher_checkpoint=str(source_teacher_checkpoint), source_teacher_forwards=result['model_forwards'],
        initial_output='source072C final checkpoint', mode=model.mode,
        interpretation='090 additional-budget visible-scope adapter or full-model fine-tuning control')
    return result
