"""Frozen C85 development inputs from existing Level1 T128 artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.analysis.same_source_swap_pilot import MAX_NODES, canonical_membership, read
from tools.analysis.static_partition_pilot import agglomerate


PARENTS = {
    'full_data': 'LEVEL1-FUSION-PEER-FULL-DATA-20261030-151-attempt-1',
    'saved_ws': 'LEVEL1-SOURCE-T-COVERAGE-20261030-155-attempt-1',
    'scores': 'LEVEL1-ASSOCIATION-CONTEXT-20261031-168-attempt-1',
    'events': 'LEVEL1-NATIVE-MOTION-FEATURES-20261031-183-attempt-1',
}
HELD_IDS = (
    '84', '91', '97', '98', '143', '147', '215', '225', '243',
    '253', '262', '264', '268', '276', '279', '290', '291', '295',
)


def load_population(runs):
    roots = {key: runs / value for key, value in PARENTS.items()}
    records = {key: read(root / 'run.json') for key, root in roots.items()}
    if any(record['status'] != 'COMPLETED' or record['exit_code'] != 0
           for record in records.values()):
        raise ValueError('completed frozen 151/155/168/183 inputs required')
    front = records['scores']
    manifest_path = Path(front['manifest'])
    manifest = read(manifest_path)
    t = manifest['splits']['T_core']
    component = manifest['components']
    seen = {str(component[sid]) for sid in t[:32]}
    held = [sid for sid in t[32:128] if str(component[sid]) not in seen]
    components = {sid: str(component[sid]) for sid in held}
    score_file = (runs / 'LEVEL1-ASSOCIATION-20261022-056-attempt-1'
                  / 'association/final-scores.npz')
    if (len(t) != 2100 or len(seen) != 12 or tuple(held) != HELD_IDS
            or len(set(components.values())) != 16
            or records['full_data']['completed_T_scenes'] != len(t)
            or Path(front['data_run']).resolve() != roots['full_data'].resolve()
            or Path(front['association_score_cache']).resolve() != score_file.resolve()
            or any(record['training_scene_ids'] != t[:32] or record['fit_training_scene_ids'] != t[:128]
                   for key, record in records.items() if key != 'full_data')
            or any(read(Path(record['manifest'])) != manifest for record in records.values())):
        raise ValueError('frozen 056 provenance or held T18 manifest population differs')
    return held, components


def _private(runs, key):
    return runs / PARENTS[key] / 'private'


def _require_held(sid):
    if sid not in HELD_IDS:
        raise ValueError(f'scene {sid} is outside the frozen T18 development population')


def _load_native(path, nodes, sources, query, native_count):
    with np.load(path, allow_pickle=False) as archive:
        offsets = archive['event_offsets']
        times = archive['event_times_seconds']
        latlon = archive['event_values'][:, :2]
        valid = archive['event_valid'][:, :2]
        if (not np.array_equal(archive['node_ids'], nodes)
                or not np.array_equal(archive['sources'], sources)
                or not np.array_equal(archive['query_seconds'], query)
                or offsets.shape != (len(nodes) + 1,) or offsets.dtype.kind not in 'iu'
                or times.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(times)
                or latlon.shape != (len(times), 2) or valid.shape != latlon.shape
                or valid.dtype != bool or not valid.all()):
            raise ValueError('183 events and 151 motion population differ')
        count = np.diff(offsets)
        if ((count < 2).any() or not np.array_equal(count, native_count)
                or not np.array_equal(archive['event_node_ids'], np.repeat(nodes, count))
                or not np.array_equal(archive['event_canonical_indices'],
                                      np.arange(len(times)) - np.repeat(offsets[:-1], count))):
            raise ValueError('canonical event count, node order or index differs')
    if (not np.isfinite(times).all() or not np.isfinite(latlon).all()
            or (np.abs(latlon) > [90., 180.]).any()
            or times.min() < query[0] - 1e-8 or times.max() > query[-1] + 1e-8
            or any((np.diff(times[left:right]) <= 0).any()
                   for left, right in zip(offsets[:-1], offsets[1:], strict=True))):
        raise ValueError('finite native positions with strictly increasing physical seconds required')
    return query, times, latlon, offsets


def load_scene(runs, sid):
    """Load only motion, scores, and WS membership, without evaluation arrays."""
    _require_held(sid)
    motion_path = _private(runs, 'full_data') / 'motion/T_core' / f'scene-{sid}.npz'
    score_path = _private(runs, 'scores') / f'front-cache-T_core-{sid}.npz'
    ws_path = _private(runs, 'saved_ws') / f'features-T_core-{sid}.npz'
    event_path = _private(runs, 'events') / f'events-T_core-{sid}.npz'
    with np.load(motion_path, allow_pickle=False) as motion:
        nodes, sources, support, query, count = (
            motion[k] for k in ('node_ids', 'sources', 'support', 'query_seconds', 'native_count'))
    n = len(nodes)
    if (not 0 < n <= MAX_NODES or nodes.shape != (n,) or len(np.unique(nodes)) != n
            or sources.shape != (n,) or np.any(sources == '') or support.shape != (n, 64)
            or support.dtype != bool or query.shape != (64,) or not np.isfinite(query).all()
            or (np.diff(query) <= 0).any()):
        raise ValueError(f'invalid original node or query support in {sid}')
    with np.load(score_path, allow_pickle=False) as saved:
        pairs, scores = saved['pairs'], saved['scores']
        if (not np.array_equal(saved['node_ids'], nodes) or pairs.dtype.kind not in 'iu'
                or not np.array_equal(pairs, np.column_stack(np.triu_indices(n, 1)))
                or scores.shape != (len(pairs),) or scores.dtype != np.float64
                or not np.isfinite(scores).all()):
            raise ValueError(f'invalid 168 frozen 056 score or pair order in {sid}')
    with np.load(ws_path, allow_pickle=False) as original:
        ws = original['membership']
    reconstructed, _, _ = agglomerate(n, pairs, scores, 'WS')
    if not (np.array_equal(ws, canonical_membership(ws, n))
            and np.array_equal(ws, reconstructed)):
        raise ValueError(f'155 WS differs from the original decoder on frozen 056 scores in {sid}')
    native = _load_native(event_path, nodes, sources, query, count)
    return nodes, sources, support, pairs, scores, ws, native


def load_identities(runs, sid, nodes):
    """Load frozen 151 identities only after the caller has saved every selection."""
    _require_held(sid)
    path = _private(runs, 'full_data') / 'evaluation/T_core' / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as evaluation:
        if not np.array_equal(evaluation['node_ids'], nodes):
            raise ValueError(f'151 evaluation node order differs in {sid}')
        identities = evaluation['identities']
    if identities.shape != nodes.shape or np.any(identities == ''):
        raise ValueError(f'invalid 151 identity vector in {sid}')
    return identities
