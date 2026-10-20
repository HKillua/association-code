"""One-pass, target-free localization of cooperative query-time variation."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from association.data.cooperative import normalize_cooperative_scene
from association.evaluation.cooperative import _validate_normalizer
from association.evaluation.physical import EARTH_RADIUS_METRES
from association.evaluation.tmtaff import _inference_device
from association.models.timed import continuous_time_encoding
from association.training.common import stable_evaluation_parameters


_ATTRIBUTES = ('latitude', 'longitude', 'speed', 'heading')


def _array(value):
    return value.detach().cpu().double().numpy()


def _centered_rms(value):
    value = _array(value)
    return float(np.sqrt(np.mean(np.square(value-value.mean(axis=0, keepdims=True)))))


def _arc(value):
    return np.remainder(value+180., 360.)-180.


def _angular_rms(value):
    radians = np.deg2rad(value)
    sine, cosine = np.sin(radians).mean(), np.cos(radians).mean()
    center = (float(value[0]) if np.hypot(sine, cosine) < 1e-12
              else np.rad2deg(np.arctan2(sine, cosine)))
    return float(np.sqrt(np.mean(np.square(_arc(value-center)))))


def _physical_rms(value):
    value = _array(value)
    return dict(latitude_rms_deg=float(value[:, 0].std()),
                longitude_circular_rms_deg=_angular_rms(value[:, 1]),
                speed_rms=float(value[:, 2].std()),
                heading_circular_rms_deg=_angular_rms(value[:, 3]))


def _physical_difference(left, right):
    left, right = _array(left), _array(right)
    delta = left-right
    delta[:, 1] = _arc(delta[:, 1])
    delta[:, 3] = _arc(delta[:, 3])
    lat1, lat2 = np.deg2rad(left[:, 0]), np.deg2rad(right[:, 0])
    haversine = (np.sin((lat1-lat2)/2.)**2
                 + np.cos(lat1)*np.cos(lat2)*np.sin(np.deg2rad(delta[:, 1])/2.)**2)
    haversine = np.clip(haversine, 0., 1.)
    distances = 2.*EARTH_RADIUS_METRES*np.arctan2(np.sqrt(haversine), np.sqrt(1.-haversine))
    result = {f'{name}_rms': float(np.sqrt(np.mean(delta[:, column]**2)))
              for column, name in enumerate(_ATTRIBUTES)}
    result['position_rms_metres'] = float(np.sqrt(np.mean(distances**2)))
    return result


def _physical_from_residual(model, residual, position_base, heading_base):
    """Mirror the active v2 physical map, without another neural forward."""
    latitude = (position_base[:, 0]+model.state_std[0]*residual[:, 0]).clamp(-90., 90.)
    longitude = torch.remainder(position_base[:, 1]+model.state_std[1]*residual[:, 1]+180., 360.)-180.
    speed = (model.speed_reference*torch.exp(model.log_speed_std*residual[:, 2])
             if model.config.speed_output_policy == 'scaled_log'
             else F.softplus(residual[:, 2]+model.speed_bias))
    heading = torch.remainder(model.state_std[3]*residual[:, 3]+heading_base, 360.)
    return torch.stack((latitude, longitude, speed, heading), dim=-1)


def _aggregate(rows):
    if not rows:
        return {}
    result = {}
    for key in rows[0]['metrics']:
        values = [row['metrics'][key] for row in rows if row['metrics'][key] is not None]
        result[key] = dict(count=len(values), mean=float(np.mean(values)) if values else None,
                           median=float(np.median(values)) if values else None,
                           min=float(min(values)) if values else None,
                           max=float(max(values)) if values else None)
    return result


@stable_evaluation_parameters
@torch.inference_mode()
def evaluate_cooperative_dynamics(model, normalizer, scenes, device='cpu', progress=None):
    """Observe one ordinary self decode per fragment, without reading targets.

    Centering always occurs within a fragment along its complete query axis.
    These are activation/output amplitudes, not errors or gradient attribution.
    All temporary hooks and every module's entry train/eval flag are restored.
    """
    if (getattr(model, 'input_adapter', None) != 'native_time_v1'
            or model.config.model_variant != 'support_residual_v2'):
        raise ValueError('cooperative dynamics requires native support_residual_v2')
    scenes = list(scenes)
    if not scenes or len({str(scene['scene_id']) for scene in scenes}) != len(scenes):
        raise ValueError('dynamics requires nonempty distinct validation scenes')
    if any(scene.get('split') != 'V_select' for scene in scenes):
        raise ValueError('dynamics requires V_select; T_test is sealed')
    _validate_normalizer(model, normalizer)
    parameter = next(model.parameters())
    requested = torch.device(device)
    if requested.type == 'mps' and requested.index is None and parameter.device.type == 'mps':
        requested = parameter.device
    device = _inference_device(requested, parameter.device, parameter.device)
    if parameter.dtype != torch.float32:
        raise ValueError('cooperative dynamics requires the original FP32 forward')
    modes = [(module, module.training) for module in model.modules()]
    handles, captured = [], {}
    started = time.monotonic()
    rows, scene_rows = [], []
    total = sum(len(scene['node_ids']) for scene in scenes)

    def capture(name, value):
        captured.setdefault(name, []).append(value.detach().clone())

    def pre_decoder(module, inputs):
        capture('decoder_input', inputs[0])
        capture('decoder_memory', inputs[1])

    def output_hook(name):
        def hook(module, inputs, output):
            capture(name, output)
        return hook

    def notify(completed_scenes, scene_id=None):
        if progress is not None:
            progress(dict(stage='cooperative_dynamics', completed_scenes=completed_scenes,
                          total_scenes=len(scenes), completed_fragments=len(rows),
                          total_fragments=total, scene_id=scene_id,
                          elapsed_seconds=time.monotonic()-started))

    try:
        model.eval()
        handles.append(model.decoder.register_forward_pre_hook(pre_decoder))
        for index, layer in enumerate(model.decoder.layers):
            handles.append(layer.register_forward_hook(output_hook(f'decoder_layer_{index+1}')))
        handles.append(model.decoder.register_forward_hook(output_hook('decoder_output')))
        notify(0)
        for scene_index, scene in enumerate(scenes):
            # This input adapter validates native clocks/states, not references.
            values, observed = normalize_cooperative_scene(scene, normalizer, model.config)
            scene_start = len(rows)
            for node in range(len(values)):
                captured.clear()
                encoded = model.encode(
                    torch.as_tensor(values[node:node+1], device=device, dtype=parameter.dtype),
                    torch.as_tensor(observed[node:node+1], device=device))
                prediction = model.decode_group(encoded, [0], routing='plain')
                expected = ['decoder_input', 'decoder_memory', 'decoder_output'] + [
                    f'decoder_layer_{index+1}' for index in range(len(model.decoder.layers))]
                if set(captured) != set(expected) or any(len(captured[key]) != 1 for key in expected):
                    raise RuntimeError('each decoder hook must fire exactly once per fragment')
                if not all(torch.isfinite(captured[key][0]).all() for key in expected):
                    raise FloatingPointError('nonfinite cooperative dynamics activation')
                hidden = captured['decoder_output'][0][0]
                q, d = model.config.length, model.config.d_model
                if hidden.shape != (q, d) or prediction.shape != (q, 4):
                    raise ValueError('invalid cooperative self-decode shape')
                query_pe = continuous_time_encoding(encoded['queries'], d)
                support = model.support_features(encoded)
                support_projection = model.support_projection(support)
                query = captured['decoder_input'][0]
                query_error = float((query-(query_pe+support_projection)).abs().max().item())
                if not torch.equal(query, query_pe+support_projection):
                    raise RuntimeError('captured decoder input differs from PE plus support')
                residual = ((hidden[:, None, :].expand(-1, 4, -1)
                             * model.state_projection.weight[None]).sum(-1)
                            + model.state_projection.bias)
                position_base = model.position_base(encoded, [0])[0]
                heading_base = (model.heading_base(encoded, [0])[0]
                                if model.config.heading_output_policy == 'observation_residual'
                                else model.state_mean[3].expand(q))
                reconstructed = _physical_from_residual(model, residual, position_base, heading_base)
                reconstruction_error = float((reconstructed-prediction).abs().max().item())
                if not torch.equal(reconstructed, prediction):
                    raise RuntimeError('manual self head/physical reconstruction differs from forward')
                constant_residual = residual.mean(0, keepdim=True).expand_as(residual)
                constant_prediction = _physical_from_residual(
                    model, constant_residual, position_base, heading_base)
                zero_prediction = _physical_from_residual(
                    model, torch.zeros_like(residual), position_base, heading_base)
                visible = encoded['observed'][0]
                metrics = dict(
                    query_pe_centered_rms=_centered_rms(query_pe[0]),
                    support_features_centered_rms=_centered_rms(support[0]),
                    support_projection_centered_rms=_centered_rms(support_projection[0]),
                    decoder_input_centered_rms=_centered_rms(query[0]),
                    native_memory_centered_rms=_centered_rms(encoded['memory'][0, visible]),
                    decoder_memory_centered_rms=_centered_rms(captured['decoder_memory'][0][0]),
                    final_hidden_centered_rms=_centered_rms(hidden),
                    query_reconstruction_max_abs=query_error,
                    physical_reconstruction_max_abs=reconstruction_error)
                for index in range(len(model.decoder.layers)):
                    name = f'decoder_layer_{index+1}'
                    metrics[f'{name}_centered_rms'] = _centered_rms(captured[name][0][0])
                h = _array(hidden)
                centered = h-h.mean(0, keepdims=True)
                hidden_energy = float(np.mean(np.square(centered).sum(1)))
                weights = _array(model.state_projection.weight)
                residual_std = _array(residual).std(axis=0)
                for column, attribute in enumerate(_ATTRIBUTES):
                    norm = float(np.linalg.norm(weights[column]))
                    projected = float(np.sqrt(np.mean(np.square(centered@weights[column]))))
                    denominator = norm*np.sqrt(hidden_energy)
                    metrics[f'{attribute}_head_row_norm'] = norm
                    metrics[f'{attribute}_residual_std'] = float(residual_std[column])
                    metrics[f'{attribute}_head_projected_rms'] = projected
                    metrics[f'{attribute}_head_normalized_dynamic_projection'] = (
                        projected/denominator if denominator > 0 else None)
                for prefix, states in (('output', prediction), ('zero_residual_base', zero_prediction)):
                    metrics.update({f'{prefix}_{key}': value for key, value in _physical_rms(states).items()})
                metrics.update({f'remove_query_residual_{key}': value for key, value in
                                _physical_difference(prediction, constant_prediction).items()})
                if any(value is not None and not np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError('nonfinite cooperative dynamics statistic')
                native_times = encoded['times'][0, visible]
                queries = encoded['queries'][0]
                rows.append(dict(scene_id=str(scene['scene_id']), node_index=node,
                                 node_id=str(scene['node_ids'][node]), source=str(scene['sources'][node]),
                                 native_count=int(visible.sum().item()), query_count=q,
                                 before_support_query_count=int((queries < native_times.min()).sum().item()),
                                 inside_support_query_count=int(((queries >= native_times.min())
                                                                 & (queries <= native_times.max())).sum().item()),
                                 after_support_query_count=int((queries > native_times.max()).sum().item()),
                                 metrics=metrics))
            scene_rows.append(dict(scene_id=str(scene['scene_id']), fragment_count=len(rows)-scene_start))
            notify(scene_index+1, str(scene['scene_id']))
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return dict(scene_count=len(scenes), fragment_count=len(rows), completion_forward_count=len(rows),
                per_scene=scene_rows, per_fragment=rows, aggregate=_aggregate(rows),
                by_source={source: _aggregate([row for row in rows if row['source'] == source])
                           for source in sorted({row['source'] for row in rows})},
                decoder_layer_count=len(model.decoder.layers),
                definitions=dict(
                    centered_rms='sqrt(mean over tokens and features of (x-mean over tokens x)^2); within fragment only',
                    native_memory='visible native tokens only; decoder memory additionally includes constant anchor-role embedding',
                    residual_std='population std across all Q of manual unscaled state_projection residual, before physical mapping',
                    head_projected_rms='sqrt(mean_Q((H-mean_Q H) @ head_row)^2), accumulated in float64',
                    head_normalized_dynamic_projection='head_projected_rms / (L2(head_row)*sqrt(mean_Q(sum_D(centered_hidden^2)))); null if denominator is zero',
                    angular_rms='RMS shortest signed degree arc to circular mean; first value is center if resultant length <1e-12',
                    physical_units='latitude/longitude/heading in degrees; speed in native dataset units; position distance in metres',
                    zero_residual_base='visible native position and heading base (or saved absolute heading mean), speed at zero residual',
                    remove_query_residual='RMS output difference after replacing all four pre-physical residuals by their full-Q per-fragment mean; no target mask; not an online method',
                    aggregate='equal-fragment summaries of within-fragment statistics; null values excluded with count retained'),
                checks=dict(single_encode_and_self_decode_per_fragment=True,
                            decoder_hooks_once_per_fragment=True,
                            input_and_physical_reconstruction_exact=True,
                            reference_accessed=False, optimizer_updates=0),
                scope='frozen V_select self-forward amplitudes only; no reference, accuracy, gradient or downstream inference',
                elapsed_seconds=time.monotonic()-started, test_opened=False)
