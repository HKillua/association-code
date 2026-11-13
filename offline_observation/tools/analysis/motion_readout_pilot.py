"""C74: compare two motion readouts while retaining the old WS position output."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from tools.analysis.conditional_trajectory_pilot import (
    KNOT_COUNT, RIDGE, design, load_native, scene_arrays, state_output,
)
from tools.analysis.same_source_swap_pilot import (
    MAX_AS, PARENTS, canonical_membership, group_members, peak_rss_bytes, read, write,
)

SPLIT = 'V_select'
PARTITION_NAME = 'LEVEL1-STATIC-PARTITION-PILOT-20261029-114-attempt-1'
METHODS = ('A', 'B', 'C')


def load_population(runs):
    parent = runs / PARENTS['native']
    original = read(parent / 'run.json')
    marker = read(parent / 'native-cache.json')
    partition = runs / PARTITION_NAME
    prior = read(partition / 'run.json')
    manifest = read(Path(read(runs / PARENTS['base'] / 'run.json')['manifest']))
    ids = original['scene_ids']
    components = {sid: str(manifest['components'][sid]) for sid in ids}
    if (original['status'] != 'COMPLETED' or original['exit_code'] != 0
            or prior['status'] != 'COMPLETED' or prior['exit_code'] != 0
            or marker['status'] != 'CACHE_COMPLETED' or marker['scene_ids'][SPLIT] != ids
            or Path(original['partition_run']).resolve() != partition.resolve()
            or len(ids) != 32 or len(components) != 32 or len(set(components.values())) != 9
            or not set(ids).issubset(manifest['splits'][SPLIT])):
        raise ValueError('original WS and 127 V32 cache binding required')
    return ids, components


def load_scene(runs, sid):
    parent = runs / PARENTS['native'] / 'private'
    with np.load(parent / 'native-motion' / SPLIT / f'scene-{sid}.npz', allow_pickle=False) as saved:
        nodes, sources, support, nearest, query = (saved[key] for key in
            ('node_ids', 'sources', 'support', 'nearest_seconds', 'query_seconds'))
    native = load_native(runs, sid, nodes, sources, split=SPLIT)
    if (not np.array_equal(query, native[0]) or len(np.unique(nodes)) != len(nodes)
            or support.shape != (len(nodes), 64) or support.dtype.kind != 'b'
            or nearest.shape != support.shape or not np.isfinite(nearest).all() or (nearest < 0).any()):
        raise ValueError('original Q64 support and native event binding required')
    partition = read(runs / PARTITION_NAME / 'private' / f'scene-{sid}.json.gz')
    original = np.asarray(partition['memberships']['WS'])
    membership = canonical_membership(original, len(nodes))
    groups = group_members(membership)
    with np.load(parent / f'states-{sid}.npz', allow_pickle=False) as saved:
        if (not np.array_equal(saved['WS_MEAN_group_node_indices'], np.concatenate(groups))
                or not np.array_equal(saved['WS_MEAN_group_offsets'], np.r_[0, np.cumsum([len(g) for g in groups])])):
            raise ValueError('historical Mean group order differs from original WS')
        baseline = saved['WS_MEAN_group_states'].astype(np.float64)
        valid = saved['WS_MEAN_group_valid']
    if (baseline.shape != (len(groups), 64, 4) or valid.shape != baseline.shape
            or not valid.all() or not np.isfinite(baseline).all()):
        raise ValueError('historical Mean requires complete finite group states')
    baseline[..., 2] *= KNOT_TO_MPS
    return nodes, sources, support, nearest, membership, native, baseline


def readouts(native, membership, baseline, support):
    query, times, _, offsets = native
    duration = query[-1] - query[0]
    x, interval = design((query-query[0])/duration)
    stats, _, _, _, origin, scale = scene_arrays(native, check_holdout_origin=False)
    group_native, _ = state_output(stats, native, origin, scale, membership, baseline)
    fixed = baseline.copy()
    original = baseline.copy()
    raw = baseline.copy()
    masks = {key: np.zeros(baseline.shape[:2], bool) for key in
             ('active', 'internal_gap', 'near_endpoint', 'B_low_speed', 'C_low_speed')}
    for index, group in enumerate(group_members(membership)):
        start = times[offsets[group]].min()
        end = times[offsets[group+1]-1].max()
        within = (query >= start) & (query <= end)
        visible = support[group].any(axis=0)
        training = within & visible
        if training.sum() < 2:
            continue
        weight = len(group) / int(training.sum())
        features = x[training]
        positions = baseline[index, training, :2] - origin
        positions[:, 1] = (positions[:, 1] + 180.) % 360. - 180.
        xy = positions * scale
        beta = np.linalg.solve(weight * (features.T @ features) + RIDGE, weight * (features.T @ xy))
        velocity = np.diff(beta, axis=0)[interval] * ((KNOT_COUNT-1)/duration)
        speed = np.linalg.norm(velocity, axis=-1)
        course = np.rad2deg(np.arctan2(velocity[:, 1], velocity[:, 0])) % 360.
        low = speed < .5
        fixed[index, within, 2] = speed[within]
        fixed[index, within & ~low, 3] = course[within & ~low]
        raw[index, within, 2:] = group_native[index, within, 2:]
        masks['active'][index] = within
        masks['internal_gap'][index] = within & ~visible
        masks['near_endpoint'][index] = within & (np.minimum(query-start, end-query) <= query[1]-query[0])
        masks['B_low_speed'][index] = within & low
        masks['C_low_speed'][index] = within & (group_native[index, :, 2] < .5)
    if not np.isfinite(fixed).all() or not np.isfinite(raw).all():
        raise FloatingPointError('nonfinite motion readout')
    return dict(A=original, B=fixed, C=raw), masks


def read_reference(runs, sid, nodes, query):
    path = runs / PARENTS['native'] / 'private/native-evaluation' / SPLIT / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(nodes, saved['node_ids']) or not np.array_equal(query, saved['query_seconds']):
            raise ValueError('reference order and Q64 changed')
        return {key: saved[key] for key in ('identities', 'target_ids', 'target_states', 'target_valid')}


def run(runs, output):
    from tools.analysis.motion_readout_evaluation import evaluate_scene, summarize

    output.mkdir(exist_ok=False)
    (output/'private').mkdir()
    began = time.monotonic()
    record = dict(status='RUNNING', split=SPLIT, stages=[], optimizer_updates=0,
                  affinity=sorted(os.sched_getaffinity(0)), started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(**fields):
        record.update(fields, elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', record)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        checkpoint(phase='predictions', scene_ids=ids)
        event_count = node_count = group_count = 0
        for sid in ids:
            nodes, _, support, _, membership, native, baseline = load_scene(runs, sid)
            arms, masks = readouts(native, membership, baseline, support)
            np.savez_compressed(output/'private'/f'{sid}.npz', membership=membership, **arms, **masks)
            node_count += len(nodes)
            group_count += len(baseline)
            event_count += len(native[1])
        if (node_count, group_count, event_count) != (3851, 1605, 1095780):
            raise ValueError('original V32 node/group/event totals changed')
        record['stages'].append('all_readouts_saved_before_identity_and_reference')
        checkpoint(phase='evaluation', node_count=node_count, group_count=group_count, event_count=event_count)
        rows, error_arrays = [], []
        for sid in ids:
            nodes, _, support, nearest, membership, native, _ = load_scene(runs, sid)
            query = native[0]
            with np.load(output/'private'/f'{sid}.npz', allow_pickle=False) as saved:
                if not np.array_equal(saved['membership'], membership):
                    raise ValueError('saved WS differs from original')
                predictions = {key: saved[key] for key in METHODS}
                masks = {key: saved[key] for key in ('active','internal_gap','near_endpoint','B_low_speed','C_low_speed')}
            ev = read_reference(runs, sid, nodes, query)
            groups = [g.tolist() for g in group_members(membership)]
            row, arrays = evaluate_scene(sid, components[sid], nodes, groups, support, nearest, ev,
                                         predictions, masks, query)
            rows.append(row)
            error_arrays.append(arrays)
        write(output/'scenes.json.gz', rows)
        write(output/'summary.json', summarize(rows, error_arrays))
        record['stages'].append('fixed_V32_evaluation_complete')
        checkpoint(status='COMPLETED', exit_code=0, completed_scenes=len(rows),
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve())
