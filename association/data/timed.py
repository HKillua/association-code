"""Actual-time observations and an explicit physical output grid.

Canonical observations are selected together with their real timestamps. The
query support values are deterministic interpolation of those selected points,
not reference truth or new sensor measurements. They never become observation
tokens in the packed model mask. No archive, split, reference or model is read.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from association.data.archive import DataError
from association.data.native import ALLOWED_SPLITS, SELECTION_POLICY, uniform_observation_indices

ADAPTER = 'native_time_v1'
INTERPOLATION = 'linear_lat_vel_shortest_lon_cou_signed_half_turn'
STATE_FIELDS = ['lat', 'lon', 'vel', 'cou']


def _settings(length, max_observations, time_unit_seconds):
    for name, value in (('length', length), ('max_observations', max_observations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
            raise DataError(f'{name} must be an explicit integer >= 2')
    if (isinstance(time_unit_seconds, (bool, np.bool_)) or not isinstance(time_unit_seconds, Real)
            or not np.isfinite(time_unit_seconds) or time_unit_seconds <= 0):
        raise DataError('time_unit_seconds must be explicit, finite and positive')
    return dict(length=int(length), max_observations=int(max_observations),
                time_unit_seconds=float(time_unit_seconds), selection_policy=SELECTION_POLICY,
                observation_interpolation=INTERPOLATION)


def _role(scene, training=False):
    allowed = ('T_core', 'T_head') if training else ALLOWED_SPLITS
    if scene.get('split') not in allowed:
        raise DataError('Timed inputs require explicit training/development roles; T_test is sealed')


def _physical(values):
    if (not np.isfinite(values).all() or (np.abs(values[..., 0]) > 90).any()
            or (np.abs(values[..., 1]) > 180).any() or (values[..., 2] < 0).any()
            or (values[..., 3] < 0).any() or (values[..., 3] > 360).any()):
        raise DataError('Observations must have finite legal lat/lon/native speed/course')


def _grid(value, length):
    grid = np.asarray(value)
    if (grid.shape != (length,) or grid.dtype.kind not in 'iuf'
            or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any()):
        raise DataError('query_times must be a finite strictly increasing numeric grid of declared length')
    return grid.astype(np.float64, copy=True)


def _interpolate(times, values, query):
    """Interpolate strictly inside support; copy exact observations unchanged."""
    output = np.full((len(query), 4), np.nan)
    support = (query >= times[0]) & (query <= times[-1])
    exact_count = interpolated_count = half_turns = 0
    half_turn_by_field = {'lon': 0, 'cou': 0}
    max_bracket = 0.0
    for index in np.flatnonzero(support):
        timestamp = query[index]
        right = int(np.searchsorted(times, timestamp))
        if right < len(times) and times[right] == timestamp:
            output[index] = values[right]
            exact_count += 1
            continue
        left = right - 1
        gap = float(times[right] - times[left])
        fraction = float((timestamp - times[left]) / gap)
        output[index] = values[left] + fraction * (values[right] - values[left])
        for column, field in ((1, 'lon'), (3, 'cou')):
            raw_delta = values[right, column] - values[left, column]
            delta = (raw_delta + 180.0) % 360.0 - 180.0
            if abs(delta) == 180.0:
                delta = 180.0 if raw_delta > 0 else -180.0
                half_turns += 1
                half_turn_by_field[field] += 1
            angle = values[left, column] + fraction * delta
            output[index, column] = ((angle + 180.0) % 360.0 - 180.0
                                     if column == 1 else angle % 360.0)
        interpolated_count += 1
        max_bracket = max(max_bracket, gap)
    return output, support, dict(exact_native_query_points=exact_count,
        interpolated_query_points=interpolated_count,
        max_interpolation_bracket_seconds=max_bracket,
        ambiguous_half_turn_count=half_turns, ambiguous_half_turn_by_field=half_turn_by_field)


def prepare_timed_scene(native_scene, *, length, max_observations,
                        time_unit_seconds, query_times=None):
    """Keep every valid fragment, select native points, and define one output grid.

    Nonempty scenes use linspace(min canonical start, max canonical end, length).
    An empty scene needs caller-supplied query_times, e.g. from declared label
    bounds; no range is invented. Explicit query_times on nonempty scenes must
    equal the canonical grid. Heading/longitude half turns use the sign of b-a,
    an audited engineering convention with no claimed truth-supported direction.
    """
    _role(native_scene)
    config = _settings(length, max_observations, time_unit_seconds)
    length, maximum = config['length'], config['max_observations']
    tracks = native_scene['tracks']
    n = len(tracks)
    for key in ('node_ids', 'identities', 'sources'):
        if np.asarray(native_scene[key]).shape != (n,):
            raise DataError(f'{key} must preserve the native fragment order')
    native_values = np.zeros((n, maximum, 4), dtype=np.float64)
    native_times = np.zeros((n, maximum), dtype=np.float64)
    native_observed = np.zeros((n, maximum), dtype=bool)
    audits, starts, stops = [], [], []
    for node, track in enumerate(tracks):
        if any(track[field] != native_scene[key][node] for field, key in
               (('node_id', 'node_ids'), ('identity_id', 'identities'), ('source', 'sources'))):
            raise DataError('Canonical track order differs from native candidate metadata')
        times = np.asarray(track['canonical_times'], dtype=np.float64)
        values = np.asarray(track['canonical_values'], dtype=np.float64)
        if (times.ndim != 1 or len(times) < 2 or values.shape != (len(times), 4)
                or not np.isfinite(times).all() or (np.diff(times) <= 0).any()):
            raise DataError('Each canonical fragment needs at least two strictly increasing real times')
        _physical(values)
        selected = uniform_observation_indices(len(times), maximum)
        size = len(selected)
        native_values[node, :size] = values[selected]
        native_times[node, :size] = times[selected]
        native_observed[node, :size] = True
        starts.append(float(times[0])); stops.append(float(times[-1]))
        audits.append(dict(node_id=track['node_id'], canonical_points=len(times), selected_points=size,
            omitted_points=len(times)-size, selected_indices=selected.tolist(), both_endpoints_retained=True,
            canonical_max_gap_seconds=float(np.diff(times).max()),
            selected_max_gap_seconds=float(np.diff(times[selected]).max())))
    if n:
        grid = _grid(np.linspace(min(starts), max(stops), length), length)
        if query_times is not None and not np.array_equal(_grid(query_times, length), grid):
            raise DataError('Nonempty scenes require the canonical full-scene linspace query grid')
        origin, grid_policy = min(starts), 'linspace min canonical start to max canonical end'
    else:
        if query_times is None:
            raise DataError('Empty native scene requires explicit query_times; no physical range can be inferred')
        grid = _grid(query_times, length)
        origin, grid_policy = float(grid[0]), 'caller supplied empty-scene query_times'
    values = np.full((n, length, 4), np.nan)
    observed = np.zeros((n, length), dtype=bool)
    for node, audit in enumerate(audits):
        size = audit['selected_points']
        values[node], observed[node], interpolation = _interpolate(
            native_times[node, :size], native_values[node, :size], grid)
        audit.update(interpolation, query_support_points=int(observed[node].sum()))
    canonical_count = sum(row['canonical_points'] for row in audits)
    selected_count = int(native_observed.sum())
    result = dict(native_scene)
    result.update(input_adapter=ADAPTER, adapter_config=config, **{key: config[key] for key in
        ('length', 'max_observations', 'time_unit_seconds', 'observation_interpolation')},
        time_origin_seconds=float(origin), native_values=native_values, native_times=native_times,
        native_observed=native_observed, grid=grid, values=values, observed=observed,
        observation_times=np.where(observed, grid[None, :], np.nan),
        timed_input_audit=dict(canonical_points=canonical_count, selected_native_points=selected_count,
            omitted_native_points=canonical_count-selected_count, fragments_preserved=n,
            fragments_without_query_support=int((~observed.any(axis=1)).sum()),
            fragments_with_one_query_support=int((observed.sum(axis=1) == 1).sum()),
            canonical_definition='existing native_inputs canonical valid observations, not untouched CSV rows',
            selection_policy=SELECTION_POLICY, grid_policy=grid_policy,
            observation_interpolation=INTERPOLATION,
            interpolation_input='same selected native observations; omitted points are not used',
            query_support_semantics='physical interpolation within measured endpoints; not new measurements or truth',
            interpolation_gap_policy='no gap filtering; measured gaps are reported, not claimed repaired',
            half_turn_policy='sign of raw b-a chooses +/-180; direction is not established by truth',
            ambiguous_half_turn_count=sum(row['ambiguous_half_turn_count'] for row in audits),
            half_turn_count_unit='query interpolation channel evaluations at an exact 180 degree difference',
            reference_read=False, per_track=audits))
    validate_timed_scene(result)
    return result


def validate_timed_scene(scene):
    """Validate the lightweight physical input contract; no references or hashes."""
    _role(scene)
    if scene.get('input_adapter') != ADAPTER:
        raise DataError('Expected explicit native_time_v1 input adapter')
    config = _settings(scene['length'], scene['max_observations'], scene['time_unit_seconds'])
    if scene.get('adapter_config') != config or scene.get('observation_interpolation') != INTERPOLATION:
        raise DataError('Timed input settings differ from the declared adapter configuration')
    length, maximum = config['length'], config['max_observations']
    native = np.asarray(scene['native_values'])
    times, mask = np.asarray(scene['native_times']), np.asarray(scene['native_observed'])
    if (native.ndim != 3 or native.shape[1:] != (maximum, 4) or native.dtype.kind not in 'iuf'
            or times.shape != native.shape[:2] or times.dtype.kind not in 'iuf'
            or mask.shape != times.shape or mask.dtype != bool):
        raise DataError('Invalid native values/times/prefix-mask shapes or types')
    count = len(native)
    sizes = mask.sum(axis=1)
    if (sizes < 2).any() or not np.array_equal(mask, np.arange(maximum)[None, :] < sizes[:, None]):
        raise DataError('Every fragment requires a valid native prefix with at least two points')
    if not np.isfinite(times[mask]).all():
        raise DataError('Native observed times must be finite')
    _physical(native[mask])
    for node, size in enumerate(sizes):
        if (np.diff(times[node, :size]) <= 0).any():
            raise DataError('Native real times must be strictly increasing')
    if (times[~mask] != 0).any() or (native[~mask] != 0).any():
        raise DataError('Native padding must be safe zeros')
    grid = _grid(scene['grid'], length)
    origin = scene['time_origin_seconds']
    if (isinstance(origin, (bool, np.bool_)) or not isinstance(origin, Real) or not np.isfinite(origin)):
        raise DataError('time_origin_seconds must be finite')
    if count:
        starts, stops = times[:, 0], times[np.arange(count), sizes-1]
        if origin != starts.min() or not np.array_equal(grid, np.linspace(starts.min(), stops.max(), length)):
            raise DataError('Grid and origin must span every canonical fragment endpoint')
        expected_support = (grid[None, :] >= starts[:, None]) & (grid[None, :] <= stops[:, None])
        expected_intervals = np.column_stack([starts, stops])
    else:
        if origin != grid[0]:
            raise DataError('Empty scene origin must be the explicit first query time')
        expected_support = np.zeros((0, length), dtype=bool)
        expected_intervals = np.empty((0, 2))
    for key in ('node_ids', 'identities', 'sources'):
        if np.asarray(scene[key]).shape != (count,):
            raise DataError(f'{key} must retain every native fragment')
    if not np.array_equal(np.asarray(scene['intervals']), expected_intervals):
        raise DataError('Original intervals must match retained real endpoints')
    observed, values = np.asarray(scene['observed']), np.asarray(scene['values'])
    actual = np.asarray(scene['observation_times'])
    if (observed.dtype != bool or not np.array_equal(observed, expected_support)
            or values.shape != (count, length, 4) or values.dtype.kind not in 'iuf'
            or actual.shape != observed.shape or actual.dtype.kind not in 'iuf'):
        raise DataError('Invalid physical query values, observed support or time array')
    _physical(values[observed])
    if not np.array_equal(actual[observed], np.broadcast_to(grid, observed.shape)[observed]):
        raise DataError('Query support values must be registered at the actual query time')


def fit_timed_normalizer(scenes):
    """Fit four-state population moments on selected training native points only."""
    scenes = list(scenes)
    if not scenes:
        raise DataError('Training native scenes are required')
    count, mean, m2, ids = 0, np.zeros(4), np.zeros(4), []
    config = None
    for scene in scenes:
        _role(scene, training=True)
        validate_timed_scene(scene)
        if config is not None and config != scene['adapter_config']:
            raise DataError('Training scenes must share length, cap, time unit and interpolation policy')
        config = scene['adapter_config']
        ids.append(str(scene['scene_id']))
        points = np.asarray(scene['native_values'], dtype=np.float64)[scene['native_observed']]
        if not len(points):
            continue
        local_mean = points.mean(axis=0)
        local_m2 = ((points - local_mean) ** 2).sum(axis=0)
        delta, combined = local_mean-mean, count+len(points)
        m2 += local_m2 + delta**2 * (count*len(points)/combined)
        mean += delta * (len(points)/combined)
        count = combined
    if len(set(ids)) != len(ids):
        raise DataError('Training scenes must not be repeated')
    if not count:
        raise DataError('Training scenes contain no native observed points')
    std = np.sqrt(np.maximum(m2/count, 0))
    constant = std < 1e-6
    std[constant] = 1.0
    return dict(mean=mean.tolist(), std=std.tolist(), feature_order=STATE_FIELDS.copy(),
        observed_point_count=count, training_scene_ids=ids,
        training_splits=list(dict.fromkeys(s['split'] for s in scenes)),
        constant_features=np.flatnonzero(constant).tolist(), input_adapter=ADAPTER,
        adapter_config=dict(config), policy='training selected native four-state population moments only; std<1e-6 replaced by 1')


def pack_timed_scene(scene, normalizer):
    """Return [N,M+L,5] values and mask for TimedAssociationModel.

    Native tokens are normalized four-state values plus scene-relative physical
    tau. Query tokens carry their requested tau and in-support interpolated four
    states, with NaN states outside support; the entire query observation mask
    is false. The model must interpret the explicit M/L layout, not treat query
    states as additional sensor tokens. Native padding is exactly zero.
    """
    validate_timed_scene(scene)
    if (normalizer.get('input_adapter') != ADAPTER
            or normalizer.get('adapter_config') != scene['adapter_config']
            or not normalizer.get('training_splits')
            or any(s not in ('T_core', 'T_head') for s in normalizer['training_splits'])
            or normalizer.get('feature_order') != STATE_FIELDS):
        raise DataError('Use matching frozen timed training moments, including cap and time unit')
    if scene['split'] == 'V_select' and str(scene['scene_id']) in normalizer.get('training_scene_ids', []):
        raise DataError('Validation scene was included in training normalization')
    mean, std = (np.asarray(normalizer[key], dtype=np.float64) for key in ('mean', 'std'))
    if (mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise DataError('Four finite training means and positive scales are required')
    native_mask, support = scene['native_observed'], scene['observed']
    maximum, length = scene['max_observations'], scene['length']
    result = np.zeros((len(native_mask), maximum+length, 5), dtype=np.float64)
    result[:, :maximum, :4][native_mask] = (scene['native_values'][native_mask]-mean)/std
    result[:, :maximum, 4][native_mask] = (
        scene['native_times'][native_mask]-scene['time_origin_seconds'])/scene['time_unit_seconds']
    result[:, maximum:, :4] = np.nan
    result[:, maximum:, :4][support] = (scene['values'][support]-mean)/std
    result[:, maximum:, 4] = (scene['grid']-scene['time_origin_seconds'])/scene['time_unit_seconds']
    finite = np.concatenate([native_mask, support], axis=1)
    if (not np.isfinite(result[..., :4][finite]).all()
            or not np.isfinite(result[..., 4]).all()
            or (np.abs(result[..., :4][finite]) > np.finfo(np.float32).max).any()
            or (np.abs(result[..., 4]) > np.finfo(np.float32).max).any()):
        raise DataError('Timed normalization or relative time exceeds finite float32 range')
    result = result.astype(np.float32)
    for node in range(len(native_mask)):
        if (np.diff(result[node, :maximum, 4][native_mask[node]]) <= 0).any():
            raise DataError('time_unit_seconds loses distinct native timestamps in float32')
    if len(result) and (np.diff(result[0, maximum:, 4]) <= 0).any():
        raise DataError('time_unit_seconds loses distinct query timestamps in float32')
    if len(result):
        sizes = native_mask.sum(axis=1)
        first = result[:, 0, 4]
        last = result[np.arange(len(result)), sizes-1, 4]
        query_tau = result[:, maximum:, 4]
        cast_support = (query_tau >= first[:, None]) & (query_tau <= last[:, None])
        if not np.array_equal(cast_support, support):
            raise DataError('precision-boundary mismatch: float32 tau changes physical query support')
    return result, np.concatenate([native_mask.copy(), np.zeros((len(native_mask), length), bool)], axis=1)
