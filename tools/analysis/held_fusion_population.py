"""Component-disjoint Level1 V readout inputs with a separate evaluation boundary."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from association.data.legacy_dataset import validate_archive
from association.data.motion_evidence import scene_motion_evidence
from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_controls import ordinary_fusion
from tools.analysis.same_source_swap_pilot import canonical_membership, group_members, read


SPLIT = 'V_select'
PARENTS = {
    'association': 'LEVEL1-ASSOCIATION-20261022-056-attempt-1',
    'partition': 'LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1',
    'native': 'LEVEL1-NATIVE-CV-STATE-20261029-127-attempt-1',
    'mean': 'LEVEL1-SOURCE-RISK-FULL-BENCHMARK-20261030-147-attempt-1',
    'full_data': 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1',
}
HELD_IDS = (
    '38', '152', '203', '472', '524', '626', '670', '765', '929', '932', '1052', '1175',
    '1252', '1272', '1300', '1328', '1393', '1531', '1595', '1934', '2252', '2425', '2665',
    '2731', '2850', '2889', '3039', '3067', '3155', '3272', '3302', '3331', '3355', '3427',
    '3658', '3675', '3723', '3768', '3862', '4039', '4133', '4231', '4424', '4469', '4554',
    '4788', '4846', '4894',
)


def _population_inputs(runs):
    roots = {key: runs / name for key, name in PARENTS.items()}
    records = {key: read(roots[key] / 'run.json') for key in ('partition', 'native', 'mean', 'full_data')}
    if any(row['status'] != 'COMPLETED' or row['exit_code'] != 0 for row in records.values()):
        raise ValueError('completed frozen 114/127/147/151 inputs required')
    manifest = read(Path(records['full_data']['manifest']))
    marker = read(roots['native'] / 'native-cache.json')
    original_v = marker['scene_ids'][SPLIT]
    original_t = marker['scene_ids']['T_core']
    validation = manifest['splits'][SPLIT]
    source_system = str(roots['association'] / 'system')
    archive_paths = marker['archive_paths']
    if (marker['status'] != 'CACHE_COMPLETED' or marker['native_motion_policy'] != 'canonical_all'
            or len(original_v) != 32 or original_t != manifest['splits']['T_core'][:32]
            or original_v != records['native']['scene_ids'] or len(validation) != 300
            or len(set(validation)) != 300 or records['mean']['scene_ids'] != validation
            or records['full_data']['completed_V_scenes'] != 300
            or records['full_data']['state_speed_unit'] != 'mps'
            or records['full_data']['native_run'] != str(roots['native'])
            or any(records[key]['source_system'] != source_system for key in ('native', 'mean', 'full_data'))
            or any(records[key]['partition_run'] != str(roots['partition']) for key in ('native', 'mean'))
            or records['partition']['source_association'] != str(roots['association'] / 'association')
            or any(read(Path(records[key]['manifest'])) != manifest for key in ('partition', 'mean'))
            or archive_paths != records['full_data']['archive'] or not archive_paths
            or len(set(archive_paths)) != len(archive_paths)
            or not all(Path(path).is_file() for path in archive_paths)):
        raise ValueError('original V population, component disjointness or frozen input lineage differs')
    return validation, manifest['components'], archive_paths, original_v + original_t


def load_population(runs):
    validation, components, archive_paths, original = _population_inputs(runs)
    seen = {components[sid] for sid in original}
    held = [sid for sid in validation if components[sid] not in seen]
    held_components = {sid: components[sid] for sid in held}
    if tuple(held) != HELD_IDS or len(set(held_components.values())) != 23:
        raise ValueError('original V48 development population differs')
    return held, held_components, archive_paths


def load_full_population(runs):
    validation, components, archive_paths, _ = _population_inputs(runs)
    return validation, {sid: components[sid] for sid in validation}, archive_paths


def validate_measurement_archive(runs, archive):
    """Check only archive index and provenance, without reading any CSV member."""
    full = read(runs / PARENTS['full_data'] / 'run.json')
    if [str(path.resolve()) for path in archive.paths] != full['archive']:
        raise ValueError('expected exact ordered archive parts recorded by 151')
    validate_archive(archive, read(Path(full['manifest'])))


def _require_scene(sid, scene_ids):
    if sid not in scene_ids:
        raise ValueError(f'scene {sid} is outside the declared validation population')


def _canonical_tracks(rows, sid, nodes, sources):
    registry = {}
    for node, source in zip(nodes.tolist(), sources.tolist(), strict=True):
        prefix = f'{sid}:{source}:'
        if not source or not node.startswith(prefix) or len(node) <= len(prefix):
            raise ValueError('151 node registry does not bind its source and batch')
        registry[(source, node[len(prefix):])] = node
    grouped = defaultdict(list)
    for row in rows:
        key = row['source'], row['batch']
        if key in registry:
            grouped[registry[key]].append(tuple(float(row[name]) for name in ('time', 'lat', 'lon', 'vel', 'cou')))
    tracks = []
    for node, source in zip(nodes.tolist(), sources.tolist(), strict=True):
        values = np.asarray(sorted(set(grouped[node])), np.float64)
        if (values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all()
                or (np.diff(values[:, 0]) <= 0).any()
                or (np.abs(values[:, 1:3]) > [90., 180.]).any()
                or (values[:, 3:] < 0.).any() or (values[:, 4] > 360.).any()):
            raise ValueError('retained 151 node no longer has valid canonical measurement rows')
        values[:, 4] %= 360.
        tracks.append(dict(node_id=node, source=source, canonical_times=values[:, 0],
                           canonical_values=values[:, 1:]))
    return tracks


def load_scene(runs, sid, archive, *, scene_ids=HELD_IDS):
    """Load measurement-only native events, frozen WS, and the original physical Mean."""
    _require_scene(sid, scene_ids)
    full = runs / PARENTS['full_data'] / 'private/motion' / SPLIT / f'scene-{sid}.npz'
    mean = runs / PARENTS['mean'] / 'private'
    with np.load(full, allow_pickle=False) as saved:
        keys = ('node_ids', 'sources', 'query_seconds', 'support', 'nearest_seconds', 'native_count', 'states')
        physical = {key: saved[key] for key in keys}
        if str(saved['state_speed_unit']) != 'mps':
            raise ValueError('151 states require explicit m/s')
    nodes, sources, query, support, nearest, counts, states = (physical[key] for key in keys)
    if (nodes.ndim != 1 or len(nodes) == 0 or len(np.unique(nodes)) != len(nodes)
            or sources.shape != nodes.shape or query.shape != (64,) or not np.isfinite(query).all()
            or (np.diff(query) <= 0).any() or support.shape != (len(nodes), 64) or support.dtype != bool
            or nearest.shape != support.shape or counts.shape != nodes.shape or states.shape != (*support.shape, 4)):
        raise ValueError('151 frozen native motion schema differs')
    with np.load(mean / 'native-motion' / f'scene-{sid}.npz', allow_pickle=False) as saved:
        for key in keys[:-1]:
            np.testing.assert_array_equal(saved[key], physical[key], err_msg=f'147 and 151 {key}')
        legacy_motion = saved['states']
    mps = legacy_motion.copy()
    mps[..., 2] *= KNOT_TO_MPS
    np.testing.assert_array_equal(mps, states, err_msg='147 raw knots to 151 m/s binding')

    partition = read(runs / PARENTS['partition'] / 'private' / f'scene-{sid}.json.gz')
    original = np.asarray(partition['memberships']['WS'])
    membership = canonical_membership(original, len(nodes))
    groups = [g.tolist() for g in group_members(membership)]
    anchors = [g[0] for g in groups]
    with np.load(mean / f'states-{sid}.npz', allow_pickle=False) as saved:
        for key, expected in (
                ('group_node_indices', np.concatenate(groups)),
                ('group_offsets', np.r_[0, np.cumsum([len(g) for g in groups])]),
                ('node_group_indices', membership)):
            np.testing.assert_array_equal(saved['WS_MEAN_' + key], expected, err_msg=f'114/147 WS {key}')
        baseline = saved['WS_MEAN_group_states']
        valid = saved['WS_MEAN_group_valid']
        if (str(saved['WS_MEAN_state_speed_unit']) != 'mps' or str(saved['WS_MEAN_split']) != SPLIT
                or baseline.shape != (len(groups), 64, 4) or valid.shape != baseline.shape or not valid.all()):
            raise ValueError('147 frozen complete WS_MEAN in m/s required')
    expected = ordinary_fusion(legacy_motion, nearest, support, groups)['predictions']['mean_supported'][anchors]
    expected[..., 2] *= KNOT_TO_MPS
    np.testing.assert_array_equal(baseline, expected, err_msg='147 WS_MEAN rebuild in original speed units')
    mps_mean = ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported'][anchors]
    np.testing.assert_array_equal(baseline[..., [0, 1, 3]], mps_mean[..., [0, 1, 3]])
    np.testing.assert_allclose(baseline[..., 2], mps_mean[..., 2], atol=1e-13, rtol=1e-13)

    before = len(archive.members_read)
    tracks = _canonical_tracks(archive._rows(sid, 'measurements'), sid, nodes, sources)
    if [(row['scene_id'], row['kind']) for row in archive.members_read[before:]] != [(sid, 'measurements')]:
        raise ValueError('held input must open only the declared measurement member')
    rebuilt = scene_motion_evidence(dict(tracks=tracks, node_ids=nodes, sources=sources, grid=query),
                                    motion_observation_policy='canonical_all')
    for key, expected in (('states', legacy_motion), ('support', support), ('nearest_seconds', nearest),
                          ('native_count', counts)):
        np.testing.assert_array_equal(rebuilt[key], expected, err_msg=f'measurements-only canonical {key}')
    times = np.concatenate([track['canonical_times'] for track in tracks])
    values = np.concatenate([track['canonical_values'] for track in tracks])
    offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
    native = query, times, values[:, :2], offsets
    return dict(nodes=nodes, sources=sources, query=query, support=support, nearest=nearest,
                membership=membership, native=native, values=values, baseline=baseline,
                native_motion=legacy_motion)


def read_evaluation(runs, sid, nodes, query, *, scene_ids=HELD_IDS):
    """Read the full 151 target registry only after every V prediction is saved."""
    _require_scene(sid, scene_ids)
    path = runs / PARENTS['full_data'] / 'private/evaluation' / SPLIT / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['node_ids'], nodes)
        np.testing.assert_array_equal(saved['query_seconds'], query)
        result = {key: saved[key] for key in ('identities', 'target_ids', 'target_states', 'target_valid')}
    if (result['identities'].shape != nodes.shape
            or result['target_states'].shape != (len(result['target_ids']), len(query), 4)
            or result['target_valid'].shape != result['target_states'].shape
            or result['target_valid'].dtype != bool):
        raise ValueError('151 evaluation registry differs from frozen population or query')
    return result
