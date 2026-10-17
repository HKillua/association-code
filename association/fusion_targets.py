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
    return {'states': states, 'valid': valid, 'query_times': query.copy(), 'report': report}


def build_fusion_targets(prepared_scene, reference_scene, pair_indices, *,
                         split, nominal_grid, time_offset_seconds, max_gap_seconds):
    """Assemble supervision for caller-selected, same-identity candidate rows.

    This does not sample or filter pairs. Repeated row indices keep their order
    and multiplicity, including when supervision is entirely unavailable. The
    explicit nominal grid must equal ``prepared_scene['grid']``; actual token
    observation times are not a substitute. Outputs are copied targets, never
    additions to the model's observed input. Split membership is supplied by the
    caller/reader; this function does not open or infer a split manifest.
    """
    if not isinstance(prepared_scene, Mapping) or not isinstance(reference_scene, Mapping):
        raise DataError('Fusion targets require prepared and already-read reference scenes')
    if split not in ALLOWED_SPLITS or reference_scene.get('split') != split:
        raise DataError('Fusion targets require matching training/development splits; T_test is sealed')
    if 'split' in prepared_scene and prepared_scene['split'] != split:
        raise DataError('Prepared scene split differs from the explicit split')
    scene_id = prepared_scene.get('scene_id')
    if not isinstance(scene_id, str) or not scene_id or reference_scene.get('scene_id') != scene_id:
        raise DataError('Prepared and reference scene_id must match')
    if tuple(reference_scene.get('state_fields', ())) != STATE_FIELDS:
        raise DataError('Reference state fields must be lat/lon/vel/cou in reader order')
    if not isinstance(reference_scene.get('rows'), (list, tuple)):
        raise DataError('Reference scene must contain the reader rows')
    for name, value in (('time_offset_seconds', time_offset_seconds),
                        ('max_gap_seconds', max_gap_seconds)):
        if not _numeric(value) or not math.isfinite(float(value)):
            raise DataError(f'{name} must be an explicit finite number')
    if max_gap_seconds < 0:
        raise DataError('max_gap_seconds must be nonnegative')

    values = np.asarray(prepared_scene.get('values'))
    observed = np.asarray(prepared_scene.get('observed'))
    if values.ndim != 3 or values.shape[2] != 4 or values.dtype.kind not in 'iuf':
        raise DataError('Prepared values must have numeric shape [N,L,4]')
    if observed.dtype.kind != 'b' or observed.shape != values.shape[:2]:
        raise DataError('Prepared observed must have bool shape [N,L] matching values')
    node_count, length = values.shape[:2]
    query = np.asarray(nominal_grid)
    prepared_grid = np.asarray(prepared_scene.get('grid'))
    if (query.ndim != 1 or query.dtype.kind not in 'iuf' or len(query) != length
            or prepared_grid.shape != query.shape or prepared_grid.dtype.kind not in 'iuf'):
        raise DataError('nominal_grid and prepared grid must have numeric shape [L]')
    query = query.astype(np.float64, copy=True)
    if (not np.isfinite(query).all() or (query[1:] <= query[:-1]).any()
            or not np.array_equal(query, prepared_grid)):
        raise DataError('nominal_grid must be finite, strictly increasing and equal prepared grid')

    candidates = np.asarray(prepared_scene.get('pairs'))
    labels = np.asarray(prepared_scene.get('labels'))
    identities = np.asarray(prepared_scene.get('identities'))
    if candidates.ndim != 2 or candidates.shape[1] != 2 or candidates.dtype.kind not in 'iu':
        raise DataError('Prepared candidate pairs must have integer shape [E,2]')
    if labels.shape != (len(candidates),) or labels.dtype.kind not in 'iuf':
        raise DataError('Prepared labels must have numeric shape [E] matching candidate pairs')
    if identities.shape != (node_count,):
        raise DataError('Prepared identities must match the node count')
    selected = np.asarray(pair_indices)
    # Empty Python lists have float dtype, but contain no noninteger index.
    if selected.ndim != 1 or (selected.size and selected.dtype.kind not in 'iu'):
        raise DataError('pair_indices must be a one-dimensional integer sequence')
    if isinstance(pair_indices, (list, tuple)) and any(isinstance(i, (bool, np.bool_)) for i in pair_indices):
        raise DataError('pair_indices cannot contain boolean indices')
    if ((selected < 0) | (selected >= len(candidates))).any():
        raise DataError('pair_indices are outside the prepared candidate rows')
    selected = selected.astype(np.int64, copy=True)
    pairs = candidates[selected].copy()
    if ((pairs < 0) | (pairs >= node_count)).any():
        raise DataError('Selected pair endpoints are outside the prepared nodes')
    if (pairs[:, 0] == pairs[:, 1]).any():
        raise DataError('Fusion targets cannot supervise self-pairs')
    if not (labels[selected] == 1).all():
        raise DataError('Every selected fusion pair must have label 1')
    identity_ids = []
    for left, right in pairs:
        identity = identities[left]
        if not isinstance(identity, str) or len(identity.split('-')) != 2 or not all(identity.split('-')):
            raise DataError('Selected pair identity must use the A-B form')
        if identity != identities[right]:
            raise DataError('Selected positive label contradicts the endpoint identities')
        identity_ids.append(str(identity))

    aligned = {}
    for identity in identity_ids:
        if identity not in aligned:
            aligned[identity] = align_reference_states(
                reference_scene, identity, query, time_offset_seconds=time_offset_seconds,
                max_gap_seconds=max_gap_seconds)
    states = np.full((len(pairs), length, 4), np.nan, dtype=np.float64)
    valid = np.zeros(states.shape, dtype=bool)
    for index, identity in enumerate(identity_ids):
        states[index] = aligned[identity]['states']
        valid[index] = aligned[identity]['valid']
    per_pair = valid.sum(axis=1)
    report = {
        'scene_id': scene_id, 'split': split, 'state_fields': list(STATE_FIELDS),
        'time_offset_seconds': float(time_offset_seconds), 'max_gap_seconds': float(max_gap_seconds),
        'query_count': length, 'pair_count': len(pairs), 'unique_identity_count': len(aligned),
        'repeated_selection_count': len(selected) - len(set(selected.tolist())),
        'valid_per_feature': dict(zip(STATE_FIELDS, valid.sum(axis=(0, 1)).tolist())),
        'valid_per_pair_feature': per_pair.tolist(),
        'fully_valid_queries_per_pair': valid.all(axis=2).sum(axis=1).tolist(),
        'unavailable_pairs': int((~valid.any(axis=(1, 2))).sum()),
        'missing_identity_ids': [identity for identity, item in aligned.items()
                                 if item['report']['missing_identity']],
        'per_identity': {identity: item['report'] for identity, item in aligned.items()},
        'coverage_counts_include_repeated_selections': True,
    }
    return {'pairs': pairs, 'identity_ids': identity_ids, 'query_times': query,
            'states': states, 'valid': valid, 'report': report}


def build_scene_fusion_targets(prepared_scene, reference_scene, *, target_identity_ids,
                               time_offset_seconds, max_gap_seconds):
    """Build every explicitly declared scene target, including unavailable ones.

    The caller supplies the complete task population from its frozen labels,
    not only successfully associated pairs or identities found in reference CSVs.
    Extra reference identities do not silently enlarge the task. All input node
    identities must be present. The returned mapping fits the complete trainer's
    ``fusion_targets`` field; no observation field is modified or synthesized.
    Clock offset and maximum interpolation gap are mandatory, with no inference.
    """
    if not isinstance(prepared_scene, Mapping) or not isinstance(reference_scene, Mapping):
        raise DataError('Already-read observation and reference scenes are required')
    split = prepared_scene.get('split')
    if split not in ALLOWED_SPLITS or reference_scene.get('split') != split:
        raise DataError('Matching training/development splits required; T_test is sealed')
    if not prepared_scene.get('scene_id') or prepared_scene['scene_id'] != reference_scene.get('scene_id'):
        raise DataError('Observation and reference scene IDs must match')
    if isinstance(target_identity_ids, (str, bytes)):
        raise DataError('target_identity_ids must be an explicit complete sequence')
    identities = list(target_identity_ids)
    if (any(not isinstance(identity, str) or len(identity.split('-')) != 2
            or not all(identity.split('-')) for identity in identities)
            or len(identities) != len(set(identities))):
        raise DataError('target identities must be distinct nonempty A-B identities')
    if not set(prepared_scene['identities']).issubset(identities):
        raise DataError('Every input identity requires a target row, including unavailable references')
    # Validate query and alignment settings even for a truly empty scene.
    if not identities:
        align_reference_states(reference_scene, '__empty__-__validation__', prepared_scene['grid'],
                               time_offset_seconds=time_offset_seconds, max_gap_seconds=max_gap_seconds)
    return {identity: align_reference_states(reference_scene, identity, prepared_scene['grid'],
                        time_offset_seconds=time_offset_seconds, max_gap_seconds=max_gap_seconds)
            for identity in identities}
