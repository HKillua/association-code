"""Native observation views without registration or model-time assumptions.

Read only an existing training/development split. All valid original fragments,
their identities and unordered candidates survive observation thinning. Times
and states are selected together; no interpolation, time warping, padding,
normalization, reference access, or shared output grid is performed here.

"Canonical" means the existing MTADArchive.load_scene result: sorted finite
points, exact duplicates removed, course 360 mapped to 0, and existing whole-node
quarantine applied. It does not mean untouched CSV rows. The canonical arrays
remain in each record even when a smaller observation view is requested.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral

import numpy as np

from .data import DataError, MTADArchive
from .dataset import validate_archive


ALLOWED_SPLITS = ('T_core', 'T_head', 'V_select')
SELECTION_POLICY = 'uniform-canonical-indices-floor-with-both-endpoints-v1'


def _max_points(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 2:
        raise DataError('max_points must be None or an integer >= 2')
    return int(value)


def uniform_observation_indices(point_count, max_points=None):
    """Select original indices, keeping both endpoints and their actual times.

    None means retain every canonical point. If capped at M, the index at k is
    floor(k*(N-1)/(M-1)). This is uniform in index, NOT uniform in physical time;
    no new state or timestamp is synthesized. N must describe a valid fragment.
    """
    cap = _max_points(max_points)
    if (isinstance(point_count, (bool, np.bool_)) or not isinstance(point_count, Integral)
            or point_count < 2):
        raise DataError('A native valid fragment requires at least two distinct times')
    count = int(point_count)
    if cap is None or count <= cap:
        return np.arange(count, dtype=np.int64)
    # Integer arithmetic also avoids floating rounding at the last endpoint.
    return np.asarray([k * (count - 1) // (cap - 1) for k in range(cap)], dtype=np.int64)


def _record(scene, component_id, split, max_points):
    tracks, selection_audits = [], []
    for track in scene.tracks:
        canonical_times = np.asarray(track.times, dtype=np.float64).copy()
        canonical_values = np.asarray(track.values, dtype=np.float64).copy()
        indices = uniform_observation_indices(len(canonical_times), max_points)
        times = canonical_times[indices].copy()
        values = canonical_values[indices].copy()
        selection = {
            'node_id': track.node_id, 'canonical_points': len(canonical_times),
            'selected_points': len(indices), 'omitted_points': len(canonical_times) - len(indices),
            'point_retention_fraction': len(indices) / len(canonical_times),
            'support_start': float(times[0]), 'support_stop': float(times[-1]),
            'both_endpoints_retained': True,
            'canonical_max_gap_seconds': float(np.diff(canonical_times).max()),
            'selected_max_gap_seconds': float(np.diff(times).max()),
        }
        tracks.append({
            'node_id': track.node_id, 'source': track.source, 'batch': track.batch,
            'identity_id': track.identity_id, 'ancestor_id': track.ancestor_id,
            'canonical_times': canonical_times, 'canonical_values': canonical_values,
            'times': times, 'values': values, 'selected_indices': indices,
        })
        selection_audits.append(selection)

    n = len(tracks)
    pairs = np.column_stack(np.triu_indices(n, 1)).astype(np.int64)
    identities = np.asarray([track['identity_id'] for track in tracks], dtype=str)
    labels = (identities[pairs[:, 0]] == identities[pairs[:, 1]]).astype(np.float32)
    audit = deepcopy(scene.audit)
    canonical_count = sum(item['canonical_points'] for item in selection_audits)
    selected_count = sum(item['selected_points'] for item in selection_audits)
    audit['native_inputs'] = {
        'selection_policy': SELECTION_POLICY, 'max_points': max_points,
        'canonical_definition': 'existing load_scene valid, sorted, deduplicated observations',
        'time_semantics': 'original measurement seconds; no time shift or quantization',
        'state_fields': ['lat', 'lon', 'vel', 'cou'],
        'valid_fragments': n, 'fragments_preserved': n, 'fragments_omitted_by_selection': 0,
        'canonical_points': canonical_count, 'selected_points': selected_count,
        'omitted_points': canonical_count - selected_count,
        'point_retention_fraction': selected_count / canonical_count if canonical_count else None,
        'capped_fragments': sum(item['omitted_points'] > 0 for item in selection_audits),
        'candidate_pairs': len(pairs), 'positive_pairs': int(labels.sum()),
        'interpolated_points': 0, 'invented_points': 0, 'coordinate_extrapolation': False,
        'per_track': selection_audits,
    }
    return {
        'scene_id': str(scene.scene_id), 'component_id': str(component_id), 'split': split,
        'tracks': tracks, 'node_ids': [track['node_id'] for track in tracks],
        'identities': identities, 'sources': [track['source'] for track in tracks],
        'intervals': np.asarray([[track['times'][0], track['times'][-1]] for track in tracks],
                                dtype=np.float64).reshape(n, 2),
        'pairs': pairs, 'labels': labels, 'audit': audit,
    }


def _request(split, max_points):
    if split not in ALLOWED_SPLITS:
        raise DataError('Native inputs allow only T_core/T_head/V_select; T_test is sealed')
    return _max_points(max_points)


def load_native_split_from_archive(archive, manifest, split, *, max_points=None, check=None):
    """Reuse an open archive and frozen manifest; read only this split's CSVs.

    Quarantined fragments retain their original diagnostics in each scene audit.
    Selected views and full canonical arrays are independent copies. Reference
    states are not read. No manifest or cache is created, and no hash is a gate.
    Optional check is called before each scene read and after the selected split.
    """
    cap = _request(split, max_points)
    validate_archive(archive, manifest)
    records = []
    for sid in manifest['splits'][split]:
        if check:
            check()
        records.append(_record(archive.load_scene(sid), manifest['components'][sid], split, cap))
    if check:
        check()
    return records


def load_native_split(archive_path, manifest, split, *, max_points=None, check=None):
    """Open the archive for an explicitly allowed existing split, without writes."""
    cap = _request(split, max_points)
    if check:
        check()
    with MTADArchive(archive_path) as archive:
        return load_native_split_from_archive(archive, manifest, split, max_points=cap, check=check)
