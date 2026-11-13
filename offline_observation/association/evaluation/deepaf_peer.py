"""099: complete fixed056 population for the disclosed DeepAF P/V adaptation.

The model sees only observable group evidence. References are joined by the
shared evaluator after every group (including duplicates) has a prediction.
Scene-local batches avoid one Python rollout per group; their size is fixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from association.evaluation.sequence_position import evaluate_sequence_position
from association.evaluation.system_attribution import _same
from association.models.deepaf_peer import deepaf_inputs, deepaf_batch_inputs, local_to_positions
from association.utils.runtime import read_json, write_json

SCOPE = '099-deepaf-pv-position-evaluation'
EVALUATION_BATCH_SIZE = 8


def _array(value, shape, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all()):
        raise ValueError('099 expected finite CPU tensor: ' + name)
    return value.detach().numpy()


def evaluate_deepaf_peer(model, validation_raw, source056, metadata, output, *,
                        check=None, initial=False):
    """Full V endpoints; initial is a random model, not the old zero-residual head."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('099 output already exists; no reuse/retry')
    chart_excursions = 0

    def predict(current, motion, groups, predictions, callback, *, initial=False):
        nonlocal chart_excursions
        q = motion['states'].shape[1]
        output_states = predictions['fixed_prior095'].copy()
        collected = {name: [] for name in ('residual_m', 'positions_m', 'base_positions_m',
            'previous_positions_m', 'p_velocity_mps', 'v_velocity_mps', 'origin_latlon',
            'delta_seconds', 'chart_excursion_count')}
        with torch.inference_mode():
            for start in range(0, len(groups), EVALUATION_BATCH_SIZE):
                if callback:
                    callback()
                chosen = groups[start:start + EVALUATION_BATCH_SIZE]
                args = deepaf_batch_inputs([deepaf_inputs(motion, group, device='cpu')
                                           for group in chosen])
                result = current.forward_batch(args, cache=False)
                n = len(chosen)
                states = _array(result['states'], (n, q, 4), 'states')
                if result['states'].dtype != torch.float64 or (np.abs(states[..., 0]) > 90.).any():
                    raise ValueError('099 physical states must use finite float64 geographic coordinates')
                base = _array(result['base_states'], (n, q, 4), 'base states')
                mean = _array(result['mean_states'], (n, q, 4), 'Mean states')
                for i, group in enumerate(chosen):
                    anchor = group[0]
                    _same(base[i], predictions['fixed_prior095'][anchor], '099 visible initialization base')
                    _same(mean[i], predictions['mean_supported'][anchor], '099 ordinary Mean input')
                    _same(states[i, :, 2:], mean[i, :, 2:], '099 fixed Mean speed/heading')
                    output_states[anchor] = states[i]
                for key in collected:
                    shape = ((n,) if key == 'chart_excursion_count' else
                             ((n, 2) if key == 'origin_latlon' else ((n, q) if key == 'delta_seconds' else (n, q, 2))))
                    collected[key].append(_array(result[key], shape, key))
                chart_excursions += int(result['chart_excursion_count'].sum())
                reconstructed = local_to_positions(result['positions_m'], result['origin_latlon'][:, None])
                if not torch.allclose(reconstructed, result['states'][..., :2], rtol=0., atol=1e-10):
                    raise ValueError('099 saved local coordinates disagree with geographic output')
                expected = (result['previous_positions_m'] +
                            result['delta_seconds'][..., None] * result['p_velocity_mps'])
                if not torch.allclose(expected, result['positions_m'], rtol=1e-12, atol=1e-8):
                    raise ValueError('099 closed-loop position recurrence differs')
                if not torch.allclose(result['base_positions_m'] + result['residual_m'],
                                      result['positions_m'], rtol=1e-12, atol=1e-8):
                    raise ValueError('099 saved physical offset differs from the rollout')
        arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
        residual = arrays.pop('residual_m')
        return output_states, residual, {'group_deepaf_' + key: value for key, value in arrays.items()}

    try:
        result = evaluate_sequence_position(model, validation_raw, source056, metadata, output,
            check=check, initial=initial, predictor=predict, evaluation_scope=SCOPE,
            residual_key='group_deepaf_local_offset_m')
        result.update(scope=SCOPE, published_method='DeepAF', adaptation='fixed056 P/V position fusion',
            method_source='https://doi.org/10.3390/aerospace12030194',
            inference_batch_size=EVALUATION_BATCH_SIZE,
            chart_excursion_count=chart_excursions,
            initial_output='random direct-linear P/V heads; visible OLS bootstrap',
            interpretation='Published DeepAF P/V core adapted to fixed groups and scene queries; not the original radar tracking experiment.',
            limitations=[
                'Repeated development V300 and shared identities; not independent generalization.',
                'Position-only comparison: reported speed/heading remain ordinary Mean, not the internal V network.',
                'Fixed056 group candidates replace radar gating, initiation and termination.',
                'Visible spherical coordinates, OLS bootstrap, fixed scale10, history5, dropout0 and TBPTT1 are disclosed implementation choices.',
                'Core mechanism and training checks do not by themselves establish convergence or peer superiority.'])
        write_json(output/'summary.json', result)
        state = read_json(output/'run.json')
        state.update(published_method='DeepAF', adaptation=result['adaptation'],
                     inference_batch_size=EVALUATION_BATCH_SIZE)
        write_json(output/'run.json', state)
        return result
    except BaseException as error:
        path = output/'run.json'
        if path.exists():
            state = read_json(path)
            state.update(status='INTERRUPTED' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'FAILED',
                         error=f'{type(error).__name__}: {error}')
            write_json(path, state)
        raise
