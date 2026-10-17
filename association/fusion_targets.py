"""Explicit, feature-masked alignment of already-read reference supervision.

This module never opens data or estimates clock offsets. Its outputs are targets,
not observed inputs; native velocity units are preserved.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

import numpy as np

from .data import DataError
from .reference_states import ALLOWED_SPLITS, STATE_FIELDS


def _numeric(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, float, np.integer, np.floating)))


def align_reference_states(reference_scene, identity_id, query_times, *,
                           time_offset_seconds, max_gap_seconds):
    """Return ``states[L,4]``, ``valid[L,4]`` and an alignment ``report``.

    The result is a dictionary. States are float64 in lat/lon/native-velocity/
    course order; invalid entries are NaN. Query times must be a finite, strictly
    increasing one-dimensional numeric sequence (an empty sequence is allowed).
    Reference time is explicitly transformed as raw time minus the supplied
    offset. A zero maximum gap permits exact matches only.

    Invalid values and duplicate conflicts invalidate only their own feature at
    that reference time. Such times remain on the interpolation axis, so they
    cannot be skipped to bridge a bad point. Duplicate equality is checked before
    canonicalizing course: simultaneous 0 and 360 are a numeric conflict, as in
    the observation contract. A single valid 360 is mapped to 0 in this copy.

    Report issues/counts concern the requested identity, except reference_rows
    and source_issue_count, which describe the supplied reader scene. Unlocatable
    timestamps are reported and never used. The reader object is never modified.
    """
    if not isinstance(reference_scene, Mapping):
        raise DataError('reference_scene must be an already-read reference scene')
    if reference_scene.get('split') not in ALLOWED_SPLITS:
        raise DataError('Reference alignment permits only training/development splits; T_test is sealed')
    if tuple(reference_scene.get('state_fields', ())) != STATE_FIELDS:
        raise DataError('Reference state fields must be lat/lon/vel/cou in reader order')
    rows = reference_scene.get('rows')
    if not isinstance(rows, (list, tuple)):
        raise DataError('Reference scene must contain the reader rows')
    if not isinstance(identity_id, str) or not identity_id:
        raise DataError('identity_id must be a nonempty string')
    for name, value in (('time_offset_seconds', time_offset_seconds),
                        ('max_gap_seconds', max_gap_seconds)):
        if not _numeric(value) or not math.isfinite(float(value)):
            raise DataError(f'{name} must be an explicit finite number')
    offset, max_gap = float(time_offset_seconds), float(max_gap_seconds)
    if max_gap < 0:
        raise DataError('max_gap_seconds must be nonnegative')
    query = np.asarray(query_times)
    if query.ndim != 1 or query.dtype.kind not in 'iuf':
        raise DataError('query_times must be a one-dimensional numeric sequence')
    query = query.astype(np.float64, copy=True)
    if not np.isfinite(query).all() or (query[1:] <= query[:-1]).any():
        raise DataError('query_times must be finite and strictly increasing')

    states = np.full((len(query), len(STATE_FIELDS)), np.nan, dtype=np.float64)
    valid = np.zeros(states.shape, dtype=bool)
    report = {
        'scene_id': reference_scene.get('scene_id'), 'split': reference_scene['split'],
        'identity_id': identity_id, 'state_fields': list(STATE_FIELDS),
        'time_offset_seconds': offset, 'max_gap_seconds': max_gap,
        'time_policy': 'aligned_reference_time = raw_reference_time - time_offset_seconds',
        'velocity_units': 'native; no conversion', 'reference_rows': len(rows),
        'source_issue_count': len(reference_scene.get('issues', [])),
        'identity_rows': 0, 'invalid_identity_rows': 0, 'unlocatable_time_rows': 0,
        'reference_times': 0, 'duplicate_time_groups': 0, 'duplicate_extra_rows': 0,
        'invalid_feature_rows': dict.fromkeys(STATE_FIELDS, 0),
        'conflicting_duplicate_features': dict.fromkeys(STATE_FIELDS, 0),
        'course_360_reference_times': 0, 'query_count': len(query),
        'exact_queries': 0, 'interior_queries': 0, 'outside_support_queries': 0,
        'no_reference_time_queries': 0, 'gap_exceeded_queries': 0,
        'ambiguous_heading_queries': 0, 'issues': [],
    }
    groups = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise DataError('Reference rows must retain the reader mapping structure')
        if row.get('identity_id') != identity_id:
            continue
        report['identity_rows'] += 1
        line = row.get('line')
        parts = identity_id.split('-')
        if len(parts) != 2 or not all(parts):
            report['invalid_identity_rows'] += 1
            report['issues'].append({'code': 'invalid_identity', 'lines': [line], 'field': 'MMSI'})
            continue
        source_state = row.get('state')
        if not isinstance(source_state, (list, tuple, np.ndarray)) or len(source_state) != 4:
            raise DataError('Reference row state must retain four reader fields')
        values = np.full(4, np.nan, dtype=np.float64)
        usable = np.zeros(4, dtype=bool)
        for column, (field, value) in enumerate(zip(STATE_FIELDS, source_state)):
            problem = None
            if not _numeric(value):
                problem = 'nonnumeric'
            elif not math.isfinite(float(value)):
                problem = 'nonfinite'
            elif ((field == 'lat' and not -90 <= value <= 90)
                  or (field == 'lon' and not -180 <= value <= 180)
                  or (field == 'vel' and value < 0)
                  or (field == 'cou' and not 0 <= value <= 360)):
                problem = 'out_of_range'
            if problem:
                report['invalid_feature_rows'][field] += 1
                report['issues'].append({'code': problem, 'lines': [line], 'field': field})
            else:
                values[column] = float(value)
                usable[column] = True
        raw_time = row.get('time')
        if not _numeric(raw_time) or not math.isfinite(float(raw_time)):
            code = 'nonnumeric_time' if not _numeric(raw_time) else 'nonfinite_time'
            report['unlocatable_time_rows'] += 1
            report['issues'].append({'code': code, 'lines': [line], 'field': 'time'})
            continue
        timestamp = float(raw_time) - offset
        if not math.isfinite(timestamp):
            report['unlocatable_time_rows'] += 1
            report['issues'].append({'code': 'nonfinite_aligned_time', 'lines': [line], 'field': 'time'})
            continue
        groups[timestamp].append((line, values, usable))

    times = np.asarray(sorted(groups), dtype=np.float64)
    reference_values = np.full((len(times), 4), np.nan, dtype=np.float64)
    reference_valid = np.zeros(reference_values.shape, dtype=bool)
    for index, timestamp in enumerate(times):
        duplicates = groups[float(timestamp)]
        lines = [entry[0] for entry in duplicates]
        if len(duplicates) > 1:
            report['duplicate_time_groups'] += 1
            report['duplicate_extra_rows'] += len(duplicates) - 1
            report['issues'].append({'code': 'duplicate_time', 'time': float(timestamp), 'lines': lines})
        for column, field in enumerate(STATE_FIELDS):
            if not all(entry[2][column] for entry in duplicates):
                continue
            value = duplicates[0][1][column]
            if any(entry[1][column] != value for entry in duplicates[1:]):
                report['conflicting_duplicate_features'][field] += 1
                report['issues'].append({'code': 'duplicate_conflict', 'field': field,
                                         'time': float(timestamp), 'lines': lines})
                continue
            if field == 'cou' and value == 360:
                value = 0.0
                report['course_360_reference_times'] += 1
            reference_values[index, column] = value
            reference_valid[index, column] = True

    report['missing_identity'] = report['identity_rows'] == 0
    report['reference_times'] = len(times)
    for index, timestamp in enumerate(query):
        if not len(times):
            report['no_reference_time_queries'] += 1
            continue
        right = int(np.searchsorted(times, timestamp))
        if right < len(times) and times[right] == timestamp:
            report['exact_queries'] += 1
            states[index] = reference_values[right]
            valid[index] = reference_valid[right]
            continue
        if right == 0 or right == len(times):
            report['outside_support_queries'] += 1
            continue
        report['interior_queries'] += 1
        left = right - 1
        gap = float(times[right]) - float(times[left])
        if gap > max_gap:
            report['gap_exceeded_queries'] += 1
            continue
        fraction = (float(timestamp) - float(times[left])) / gap
        usable = reference_valid[left] & reference_valid[right]
        for column in range(4):
            if not usable[column]:
                continue
            a, b = reference_values[left, column], reference_values[right, column]
            if column == 3:
                delta = (b - a + 180.0) % 360.0 - 180.0
                if abs(delta) == 180.0:
                    report['ambiguous_heading_queries'] += 1
                    report['issues'].append({'code': 'ambiguous_heading_arc', 'field': 'cou',
                                             'query_index': index, 'query_time': float(timestamp),
                                             'reference_times': [float(times[left]), float(times[right])]})
                    continue
                value = (a + fraction * delta) % 360.0
            else:
                value = (1.0 - fraction) * a + fraction * b
            states[index, column] = value
            valid[index, column] = True
    report['valid_per_feature'] = dict(zip(STATE_FIELDS, valid.sum(axis=0).tolist()))
    report['fully_valid_queries'] = int(valid.all(axis=1).sum())
    report['unavailable_queries'] = int((~valid.any(axis=1)).sum())
    return {'states': states, 'valid': valid, 'report': report}
