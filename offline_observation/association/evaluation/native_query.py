"""100 full frozen056 population, with native events and immutable strong b."""
import numpy as np
import torch

from association.data.source_events import source_events
from association.data.motion_evidence import scene_motion_evidence
from association.evaluation.sequence_position import evaluate_sequence_position, _tensor
from association.evaluation.system_attribution import _same
from association.models.sequence_position import sequence_inputs, tangent_position


def evaluate(model, validation, source056, metadata, output, *, check=None):
    validation = list(validation)
    for scene in validation:
        motion = scene_motion_evidence(scene, motion_observation_policy='canonical_all')
        if model.arm != 'budget' and 'native_events' not in motion:
            motion['native_events'] = source_events(scene)
        scene['motion'] = motion
        scene['motion_observation_policy'] = 'canonical_all'

    def predict(current, motion, groups, predictions, callback, **kwargs):
        q = len(motion['query_seconds'])
        states = predictions['fixed_prior095'].copy()
        residuals, bases = [], []
        with torch.inference_mode():
            for group in groups:
                if callback: callback()
                result = current(motion, group)
                actual = _tensor(result['states'], (q, 4), '100 positions', physical=True)
                base = _tensor(result['frozen_base_states'], (q, 4), '100 frozen b', physical=True)
                residual = _tensor(result['residual_m'], (q, 2), '100 residual')
                origin = sequence_inputs(motion, group)[0] if current.arm == 'budget' else result['base_states']
                _same(actual, tangent_position(origin, result['residual_m']).numpy(), '100 physical reconstruction')
                _same(actual[:, 2:], predictions['mean_supported'][group[0], :, 2:], '100 unchanged speed/course')
                states[group[0]] = actual
                residuals.append(residual); bases.append(base)
        return states, np.asarray(residuals), dict(group_frozen_base_states=np.asarray(bases))

    result = evaluate_sequence_position(model, validation, source056, metadata, output, check=check,
        predictor=predict, evaluation_scope='100-native-query-fusion', residual_key='group_native_increment_m')
    result.update(arm=model.arm, residual_origin='fixed095' if model.arm == 'budget' else 'frozen090C',
                  scientific_scope='Single-seed native pooling screen; not QuITE++ or a domain peer reproduction.')
    return result
