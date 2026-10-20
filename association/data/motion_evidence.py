"""Fixed native motion and support evidence, independent of reference targets.

The shared prediction function accepts only visible measurements. The split
loader attaches identity labels for external training/evaluation but never reads
reference CSVs. Q64/cap128/time-unit20 follow the current frozen data protocol.
"""
from __future__ import annotations

import numpy as np

from association.data.archive import DataError, MTADArchive, validate_manifest
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.evaluation.native_motion import _short_delta, predict_native_positions


DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)
EARTH_RADIUS_M = 6371000.


def complete_native_attributes(native_values, native_times, native_observed, query, positions):
    """Only visible native attributes enter predictions; no truth is accepted.

    MTAD canonicalization quarantines any track with an invalid attribute. Check
    those attributes again explicitly; a true observation bit alone is not proof
    that speed or course is legal. There is no invented value for missing course.
    """
    values, times, visible = map(np.asarray, (native_values, native_times, native_observed))
    query, positions = np.asarray(query), np.asarray(positions)
    if (values.ndim != 3 or values.shape[-1] != 4 or times.shape != values.shape[:2]
            or visible.shape != times.shape or visible.dtype != bool
            or positions.shape != (len(values), len(query), 2)):
        raise ValueError('native state, time, observation and position shapes differ')
    selected = values[visible]
    if (not np.isfinite(selected).all() or (selected[:, 2] < 0).any()
            or (selected[:, 3] < 0).any() or (selected[:, 3] > 360).any()):
        raise ValueError('visible native speed/course must independently be legal')
    result = np.empty((len(values), len(query), 4), np.float64)
    result[..., :2] = positions
    for node in range(len(values)):
        t, state = times[node, visible[node]], values[node, visible[node]]
        if len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
            raise ValueError('each anchor needs two increasing visible native times')
        right = np.searchsorted(t, query).clip(1, len(t)-1)
        left = right-1
        fraction = ((query-t[left])/(t[right]-t[left])).clip(0, 1)
        result[node, :, 2] = state[left, 2]+fraction*(state[right, 2]-state[left, 2])
        result[node, :, 3] = (state[left, 3]+fraction*_short_delta(state[right, 3]-state[left, 3])) % 360.
    if not np.isfinite(result).all():
        raise ValueError('complete predictions must be finite on every retained query')
    return result


def build_motion_evidence(native_values, native_times, native_observed, query_seconds, sources):
    """Build fixed float64 states and native-only reliability descriptions.

    Position uses interpolation inside support and measured-endpoint-anchored
    all-native OLS extrapolation outside. Speed/course use interpolation/hold.

    ``fit_rmse_m`` is the RMS Euclidean length of residuals to each fragment's
    all-native OLS line *including fitted intercept*. Longitude is cumulatively
    unwrapped along shortest signed arcs. Degrees are converted to a fixed local
    north/east plane with Earth radius 6,371,000 m and cos(mean visible latitude)
    for east. Both velocity arrays use that same plane and actual seconds; their
    last axis is [north, east]. These are diagnostics, not calibrated covariance.

    Exact observations mean equality of query and visible native timestamp.
    Interpolated support is reported separately and is not an exact observation.
    No source value changes the prediction; sources are retained for same-source
    evidence and evaluation. No identity, label or target can enter this API.
    """
    values = np.asarray(native_values)
    times = np.asarray(native_times)
    visible = np.asarray(native_observed)
    source_values = np.asarray(sources, dtype=str)
    if values.ndim != 3 or values.shape[-1] != 4 or source_values.shape != (len(values),):
        raise ValueError('motion evidence requires native [N,M,4] and one source per fragment')
    control = predict_native_positions(values, times, visible, query_seconds)
    states = complete_native_attributes(values, times, visible, query_seconds,
                                        control['predictions']['native_ols'])
    sizes = control['native_point_counts']
    span = np.empty(len(values), dtype=np.float64)
    residual_rms = np.empty(len(values), dtype=np.float64)
    scale = np.empty((len(values), 2), dtype=np.float64)
    for node, size in enumerate(sizes):
        t = np.asarray(times[node, :size], dtype=np.float64)
        xy = np.asarray(values[node, :size, :2], dtype=np.float64).copy()
        xy[:, 1] = xy[0, 1] + np.r_[0., np.cumsum(_short_delta(np.diff(xy[:, 1])))]
        centred_t = t-t.mean()
        fitted = xy.mean(0) + centred_t[:, None]*control['ols_slope_degrees_per_second'][node]
        scale[node] = EARTH_RADIUS_M*np.pi/180.*np.array(
            [1., np.cos(np.deg2rad(xy[:, 0].mean()))])
        residual = (xy-fitted)*scale[node]
        residual_rms[node] = np.sqrt(np.mean(np.sum(residual*residual, axis=-1)))
        span[node] = t[-1]-t[0]
    return dict(states=states, support=control['regions']['inside_support'],
        nearest_seconds=control['nearest_seconds'], native_count=sizes, span_seconds=span,
        fit_rmse_m=residual_rms,
        ols_velocity_mps=control['ols_slope_degrees_per_second']*scale,
        endpoint_velocity_mps=control['endpoint_slope_degrees_per_second']*scale,
        sources=source_values.copy(), query_seconds=np.asarray(query_seconds, dtype=np.float64).copy(),
        exact_observation=control['regions']['exact_observation'])


def prepare_motion_scenes(archive_path, manifest, split, check=None):
    """Prepare one frozen T_core or V_select split, without reading references.

    Process one scene at a time; retain only compact motion evidence and original
    candidate metadata. Empty scenes survive and use their declared label bounds
    for the query grid. Labels never enter nonempty scene grids or predictions.
    """
    if split not in ('T_core', 'V_select'):
        raise DataError('motion evidence permits only explicit T_core or V_select; T_test is sealed')
    validate_manifest(manifest)
    scenes = []
    with MTADArchive(archive_path) as archive:
        for scene_id in manifest['splits'][split]:
            if check:
                check()
            native = _record(archive.load_scene(scene_id), manifest['components'][scene_id],
                             split, DATA['max_observations'])
            query = None
            if not native['tracks']:
                labels = archive.read_labels(scene_id)
                if not labels:
                    raise DataError(f'Scene {scene_id}: empty scene has no label bounds for query grid')
                start = min(row['t_s'] for row in labels)
                stop = max(row['t_e'] for row in labels)
                if stop <= start:
                    raise DataError(f'Scene {scene_id}: empty scene requires positive-duration label bounds')
                query = np.linspace(start, stop, DATA['length'])
            prepared = prepare_timed_scene(native, **DATA, query_times=query)
            motion = build_motion_evidence(prepared['native_values'], prepared['native_times'],
                prepared['native_observed'], prepared['grid'], prepared['sources'])
            scenes.append(dict(
                **{key: prepared[key] for key in (
                    'scene_id', 'component_id', 'split', 'node_ids', 'identities', 'sources',
                    'pairs', 'labels')}, motion=motion))
            if check:
                check()
    return scenes
