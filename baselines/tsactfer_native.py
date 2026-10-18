"""Opt-in TSA-cTFER packing directly from native_inputs records.

This adapter does not change the existing fit entry point or the 011 protocol.
The explicit point cap selects from full canonical observations, independently
of any earlier optional native view cap. Every fragment and candidate survives;
omitted canonical points are audited. No scene64 registration is involved.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from association.data.archive import DataError
from association.data.native import SELECTION_POLICY, uniform_observation_indices
from .tsactfer_data import (build_positive_pools as _build_positive_pools,
                           fit_normalizer as _fit_normalizer,
                           pack_scene as _pack_scene)


TRAINING_SPLITS = ('T_core', 'T_head')
ALLOWED_SPLITS = (*TRAINING_SPLITS, 'V_select')


def _split(scene, training=False):
    allowed = TRAINING_SPLITS if training else ALLOWED_SPLITS
    if scene.get('split') not in allowed:
        purpose = 'training T_core/T_head' if training else 'T_core/T_head/V_select'
        raise DataError(f'Native TSA requires {purpose}; T_test is sealed')


def _cap(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
        raise DataError('Native TSA max_points must be an explicit integer >= 2')
    return int(value)


def _view(scene, max_points, width=None):
    """Temporary actual-time padding for reuse of existing numerical utilities."""
    _split(scene)
    cap = _cap(max_points)
    width = cap if width is None else width
    tracks = scene['tracks']
    n = len(tracks)
    identities, sources = np.asarray(scene['identities']), np.asarray(scene['sources'])
    intervals = np.asarray(scene['intervals'], dtype=np.float64)
    if (len(scene['node_ids']) != n or identities.shape != (n,) or sources.shape != (n,)
            or intervals.shape != (n, 2)):
        raise DataError('Native track metadata do not have matching node counts')
    values = np.zeros((n, width, 4), dtype=np.float64)
    observed = np.zeros((n, width), dtype=bool)
    times = np.full((n, width), np.nan, dtype=np.float64)
    audits, starts = [], []
    for node, track in enumerate(tracks):
        if (track['node_id'] != scene['node_ids'][node] or track['identity_id'] != identities[node]
                or track['source'] != sources[node]):
            raise DataError('Native track order differs from candidate metadata')
        raw_times = np.asarray(track['canonical_times'], dtype=np.float64)
        raw_values = np.asarray(track['canonical_values'], dtype=np.float64)
        if (raw_times.ndim != 1 or len(raw_times) < 2 or raw_values.shape != (len(raw_times), 4)
                or not np.isfinite(raw_times).all() or not np.isfinite(raw_values).all()
                or not np.all(np.diff(raw_times) > 0)):
            raise DataError('Native canonical observations must be finite with increasing real times')
        if not np.array_equal(intervals[node], raw_times[[0, -1]]):
            raise DataError('Native interval must match canonical endpoints')
        indices = uniform_observation_indices(len(raw_times), cap)
        size = len(indices)
        values[node, :size] = raw_values[indices]
        times[node, :size] = raw_times[indices]
        observed[node, :size] = True
        starts.append(float(raw_times[0]))
        audits.append({'node_id': track['node_id'], 'canonical_points': len(raw_times),
                       'sampled_points': size, 'omitted_points': len(raw_times) - size,
                       'selected_indices': indices.tolist(), 'both_endpoints_retained': True})
    origin = min(starts) if starts else 0.0
    view = dict(scene)
    # pack_scene uses only grid[0] as origin. This is not a resampling grid and
    # is removed from the public result along with temporary observation_times.
    view.update(values=values, observed=observed, observation_times=times,
                grid=np.asarray([origin], dtype=np.float64))
    return view, audits


def fit_native_normalizer(training_scenes, *, max_points):
    """Fit existing lon/lat moments only on capped T_core/T_head observations."""
    cap = _cap(max_points)
    scenes = list(training_scenes)
    for scene in scenes:
        _split(scene, training=True)
    ids = [str(scene['scene_id']) for scene in scenes]
    if len(ids) != len(set(ids)):
        raise DataError('Normalizer training scenes must not be repeated')
    views = [_view(scene, cap)[0] for scene in scenes]
    result = _fit_normalizer(views)
    result.update(policy='training selected native lon/lat; population std; std<1e-6 replaced by 1',
                  training_splits=list(dict.fromkeys(scene['split'] for scene in scenes)),
                  native_max_points=cap, native_selection_policy=SELECTION_POLICY)
    return result


def pack_native_scene(scene, normalizer, *, max_points, length=128,
                      time_bin_seconds=20.0, time_bins=4096):
    """Pack native points with a frozen training normalizer and fixed second bins.

    A vocabulary overflow fails explicitly; the adapter never clips timestamps
    or rescales a scene to fit. Non-numerical candidate metadata remain shallow
    references, as in tsactfer_data.pack_scene, and are not modified.
    """
    _split(scene)
    cap = _cap(max_points)
    if (isinstance(length, (bool, np.bool_)) or not isinstance(length, Integral)
            or length < cap):
        raise DataError('packing length must be an integer >= max_points')
    if (isinstance(time_bins, (bool, np.bool_)) or not isinstance(time_bins, Integral)
            or time_bins < 1):
        raise DataError('time_bins must be a positive integer')
    if (isinstance(time_bin_seconds, (bool, np.bool_)) or not isinstance(time_bin_seconds, Real)
            or not np.isfinite(time_bin_seconds) or time_bin_seconds <= 0):
        raise DataError('time_bin_seconds must be finite and positive')
    if (normalizer.get('native_max_points') != cap
            or normalizer.get('native_selection_policy') != SELECTION_POLICY
            or not normalizer.get('training_splits')
            or any(split not in TRAINING_SPLITS for split in normalizer['training_splits'])):
        raise DataError('Use frozen native training moments fitted with the same point cap')
    if (scene['split'] == 'V_select'
            and str(scene['scene_id']) in normalizer.get('training_scene_ids', [])):
        raise DataError('Validation scene was included in normalizer training')
    view, audits = _view(scene, cap, int(length))
    actual = view['observation_times'][view['observed']]
    bins = np.floor((actual - view['grid'][0]) / float(time_bin_seconds))
    outside = (~np.isfinite(bins)) | (bins < 0) | (bins >= int(time_bins))
    if outside.any():
        raise DataError(f"Scene {scene['scene_id']}: {int(outside.sum())} timestamps exceed the "
                        'fixed vocabulary; change the declared vocabulary, not the physical clock')
    result = _pack_scene(view, normalizer, length=int(length),
                         time_bin_seconds=float(time_bin_seconds), time_bins=int(time_bins))
    result.pop('grid')
    result.pop('observation_times')
    canonical_count = sum(row['canonical_points'] for row in audits)
    sampled_count = sum(row['sampled_points'] for row in audits)
    result['packing_audit'].update(
        source_layout='native actual-time observations; temporary index padding, no scene64',
        selection_policy=SELECTION_POLICY, max_points=cap,
        selection_source='full canonical arrays; independent of incoming optional view cap',
        canonical_points=canonical_count, sampled_points=sampled_count,
        discarded_observations=canonical_count - sampled_count,
        discarded_at_padding=0, capped_fragments=sum(row['omitted_points'] > 0 for row in audits),
        fragments_preserved=len(audits), point_retention_fraction=(
            sampled_count / canonical_count if canonical_count else None),
        origin_policy='minimum canonical start across all valid scene fragments',
        origin_has_observations=bool(audits), per_track=audits)
    return result


def build_native_positive_pools(training_scenes):
    """Reuse the existing ITSA/MSTSA eligibility on unaltered native intervals."""
    scenes = list(training_scenes)
    for scene in scenes:
        _split(scene, training=True)
    return _build_positive_pools(scenes)
