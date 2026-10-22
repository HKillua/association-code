"""Bounded first-scene replay evidence; never changes acceptance tolerances.

The caller supplies the unhooked first forward. Only the second forward is
traced. No references, identities, truth-selected rows or model updates enter.
"""
from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
import torch

from association.evaluation.evidence_fusion_combo import REPLAY_TOLERANCE, predict_partition
from association.models.evidence_fusion import fusion_inputs

MOTION_KEYS = frozenset(('states', 'support', 'nearest_seconds', 'native_count',
    'span_seconds', 'fit_rmse_m', 'ols_velocity_mps', 'endpoint_velocity_mps',
    'sources', 'query_seconds', 'exact_observation'))


def array_difference(current, previous, *, atol, rtol):
    """Elementwise np.isclose audit, with finite JSON and distinct extremal points.

    Nonfinite values always fail. Max absolute difference and worst tolerance
    ratio are different statistics; the latter can fail away from the former.
    """
    if any(not np.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError('finite nonnegative tolerances required')
    current, previous = np.asarray(current), np.asarray(previous)
    if any(a.dtype.kind not in 'buif' for a in (current, previous)):
        raise ValueError('numeric or boolean arrays required')
    result = dict(passed=False, shape_equal=current.shape == previous.shape,
        current_shape=list(current.shape), previous_shape=list(previous.shape),
        current_dtype=str(current.dtype), previous_dtype=str(previous.dtype),
        dtype_equal=current.dtype == previous.dtype,
        current_nonfinite_count=int((~np.isfinite(current)).sum()),
        previous_nonfinite_count=int((~np.isfinite(previous)).sum()),
        atol=float(atol), rtol=float(rtol), exceed_count=None,
        max_absolute_difference=None, worst_tolerance_ratio=None, first_violations=[])
    if not result['shape_equal']:
        return result
    finite = np.isfinite(current) & np.isfinite(previous)
    # Preserve floating dtype arithmetic used by np.isclose. Integers/bools need
    # floating subtraction to avoid wraparound and unsupported boolean subtract.
    a = current if current.dtype.kind == 'f' else current.astype(np.float64)
    b = previous if previous.dtype.kind == 'f' else previous.astype(np.float64)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        close = np.isclose(current, previous, atol=atol, rtol=rtol) & finite
        difference = np.abs(a - b)
        bound = atol + rtol * np.abs(b)
        ratio = difference.astype(np.float64) / bound.astype(np.float64)
    ratio = np.where((difference == 0) & (bound == 0), 0., ratio)

    def number(value):
        return float(value) if np.isfinite(value) else None

    def kind(value):
        return ('finite' if np.isfinite(value) else 'nan' if np.isnan(value)
                else 'positive_infinity' if value > 0 else 'negative_infinity')

    def point(flat):
        index = np.unravel_index(int(flat), current.shape)
        c, p, d, t, r = (x[index] for x in (current, previous, difference, bound, ratio))
        return dict(index=list(map(int, index)), current=number(c), previous=number(p),
            current_kind=kind(c), previous_kind=kind(p), absolute_difference=number(d),
            tolerance_bound=number(t), tolerance_ratio=number(r),
            ratio_kind=('nonfinite_input' if not finite[index] else
                        'zero_bound_nonzero_difference' if t == 0 and d > 0 else
                        'overflow' if not np.isfinite(r) else 'finite'))

    result.update(passed=bool(close.all()), exceed_count=int((~close).sum()),
        first_violations=[point(i) for i in np.flatnonzero(~close)[:3]])
    if finite.any():
        result['max_absolute_difference'] = point(np.argmax(np.where(finite, difference, -1)))
        result['worst_tolerance_ratio'] = point(np.argmax(np.where(finite, ratio, -1)))
    return result


def runtime_environment(model=None):
    """Read backend/precision/thread context without changing process settings."""
    return dict(python=platform.python_version(), platform=platform.platform(),
        numpy=np.__version__, torch=torch.__version__, torch_build=torch.__config__.show(),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        model_training=None if model is None else model.training,
        model_device=None if model is None else str(next(model.parameters()).device),
        thread_environment={key: os.environ.get(key) for key in (
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')})


def _save(path, arrays):
    for key, value in arrays.items():
        if np.asarray(value).dtype.hasobject:
            raise ValueError('object arrays are not permitted: ' + key)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def _motion(motion):
    if set(motion) != MOTION_KEYS:
        raise ValueError('motion must contain exactly the visible motion-evidence fields')
    if any(np.asarray(value).dtype.hasobject for value in motion.values()):
        raise ValueError('object arrays are not permitted in motion evidence')


def _comparison(predictions, routing, old_predictions, old_routing, motion):
    tol = dict(atol=REPLAY_TOLERANCE['anchor_weight_atol'],
               rtol=REPLAY_TOLERANCE['anchor_weight_rtol'])
    result = {key: array_difference(value, old_routing[key],
        **(tol if key in ('utility', 'helper_weights', 'anchor_weights') else dict(atol=0., rtol=0.)))
        for key, value in routing.items()}
    result['predictions_exact_diagnostic_only'] = {
        key: array_difference(value, old_predictions[key], atol=0., rtol=0.)
        for key, value in predictions.items()}
    for key in ('utility', 'helper_weights', 'anchor_weights'):
        record = result[key]
        points = [record['max_absolute_difference'], record['worst_tolerance_ratio'],
                  *record['first_violations']]
        for point in points:
            if point is None or len(point['index']) != 3:
                continue
            row, query, attribute = point['index']
            point.update(helper_anchor=(row if key == 'anchor_weights'
                                        else int(routing['helper_anchor'][row])),
                helper_node=(None if key == 'anchor_weights'
                             else int(routing['helper_node'][row])),
                query_seconds=float(motion['query_seconds'][query]),
                attribute=('position', 'speed', 'heading')[attribute],
                location_mapping='current routing, not inferred historical ordering')
    return result


def _probe_comparison(motion, probe, calls, device):
    if probe is None:
        return dict(status='unavailable', reason='No original probe motion/script supplied; no external pickle read.')
    _motion(probe)
    fields = {}
    for key in sorted(MOTION_KEYS):
        a, b = np.asarray(motion[key]), np.asarray(probe[key])
        fields[key] = (array_difference(a, b, atol=0., rtol=0.) if a.dtype.kind in 'buif'
                       and b.dtype.kind in 'buif' else dict(passed=np.array_equal(a, b),
                           current_dtype=str(a.dtype), previous_dtype=str(b.dtype)))
    inputs = []
    for members in calls:
        a, b = fusion_inputs(motion, members, device=device), fusion_inputs(probe, members, device=device)
        inputs.append(dict(members=members, **{key: array_difference(x.cpu().numpy(), y.cpu().numpy(),
            atol=0., rtol=0.) for key, x, y in zip(('states', 'features', 'exact_observation'), a, b)}))
    return dict(status='compared', fields=fields, inputs=inputs,
        limitation='Compares supplied arrays only; original probe construction and causality remain unproved.')


def diagnose_replay(model, motion, groups, predictions, routing, old_fragment, output_dir,
                    *, probe_motion=None):
    """Persist unhooked evidence, then perform exactly one traced partition call.

    Existing files are never overwritten. Errors propagate after hook removal
    and partial trace preservation. The caller writes the returned JSON report.
    Optional probe motion may be a dict or an internal deferred loader; loading
    happens only after the first forward has been safely persisted.
    """
    _motion(motion)
    if model.training:
        raise ValueError('frozen model must already be in eval mode')
    output = Path(output_dir)
    if not output.is_dir():
        raise ValueError('diagnostic output directory must already exist')
    first_path, trace_path = output/'first-forward.npz', output/'trace-second-forward.npz'
    if first_path.exists() or trace_path.exists():
        raise FileExistsError('diagnostic evidence already exists')
    calls = [[int(anchor)] + [int(node) for node in group if node != anchor]
             for group in groups for anchor in group]
    old_routing = {key: old_fragment['ols_routing_' + key] for key in routing}
    old_predictions = {key: old_fragment['ols_' + key] for key in predictions}
    first = {prefix + key: np.asarray(value).copy() for prefix, data in (
        ('motion_', motion), ('prediction_', predictions), ('routing_', routing),
        ('old_ols_', old_predictions), ('old_ols_routing_', old_routing)) for key, value in data.items()}
    first.update(group_offsets=np.cumsum([0] + list(map(len, groups)), dtype=np.int64),
                 group_node_indices=np.asarray([node for group in groups for node in group], np.int64))
    _save(first_path, first)
    if callable(probe_motion):
        probe_motion = probe_motion()
        if not isinstance(probe_motion, dict):
            raise ValueError('deferred probe loader must return a motion dictionary')
    if probe_motion is not None:
        _motion(probe_motion)
        _save(output/'probe-motion.npz', {'motion_' + k: v for k, v in probe_motion.items()})
    trace, call_info, handles = {}, [], []
    device = next(model.parameters()).device

    def before_forward(module, args):
        index = len(call_info)
        if index >= len(calls):
            raise ValueError('unexpected extra model forward')
        prefix = f'call_{index:05d}_'
        trace[prefix + 'members'] = np.asarray(calls[index], np.int64)
        tensors = {}
        for key, value in zip(('states', 'features', 'exact_observation'), args):
            trace[prefix + key] = value.detach().cpu().numpy().copy()
            tensors[key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                strides=list(value.stride()), contiguous=value.is_contiguous(), device=str(value.device))
        call_info.append(dict(index=index, members=calls[index], tensors=tensors))

    def layer_hook(name):
        def capture(module, args, value):
            trace[f'call_{len(call_info)-1:05d}_' + name] = value.detach().cpu().numpy().copy()
        return capture

    try:
        handles.append(model.register_forward_pre_hook(before_forward))
        handles.append(model.utility.register_forward_hook(layer_hook('utility')))
        for name, layer in model.utility.named_children():
            handles.append(layer.register_forward_hook(layer_hook('utility_layer_' + name)))
        repeated, repeated_routing = predict_partition(model, motion, groups, device)
        if len(call_info) != len(calls):
            raise ValueError('model forward count differs from original partition traversal')
        trace.update({'repeat_prediction_' + k: v for k, v in repeated.items()})
        trace.update({'repeat_routing_' + k: v for k, v in repeated_routing.items()})
    finally:
        for handle in handles:
            handle.remove()
        _save(trace_path, trace)
    return dict(scope='first-scene-replay-diagnostic', model_updates=0,
        extra_partition_forwards=1, environment=runtime_environment(model),
        first_snapshot=first_path.name, trace_snapshot=trace_path.name,
        trace_scope='Second forward only; first-forward tensors were not intercepted. '
                    'First inputs can be reconstructed from the saved complete motion.',
        calls=call_info, tolerance=dict(REPLAY_TOLERANCE),
        first_vs_old=_comparison(predictions, routing, old_predictions, old_routing, motion),
        repeat_vs_first=_comparison(repeated, repeated_routing, predictions, routing, motion),
        repeat_vs_old=_comparison(repeated, repeated_routing, old_predictions, old_routing, motion),
        historical_057_features_available=False, failed_062_arrays_available=False,
        probe_input_comparison=_probe_comparison(motion, probe_motion, calls, device),
        limitation='Numerical diagnostics, not a tolerance override, acceptance result, '
                   'old failed-job tensor reconstruction or causal attribution.')
