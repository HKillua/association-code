"""C70: fixed-pool, native-time holdout scoring and full-scene state comparison."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import time

import numpy as np

from association.data.physical_units import KNOT_TO_MPS
from association.evaluation.fusion_controls import ordinary_fusion
from tools.analysis.same_source_swap_pilot import (
    EPSILON, MAX_AS, PARENTS, canonical_membership, group_members, load_population,
    load_scene, peak_rss_bytes, read, swapped, write,
)

C71_NAME = 'LEVEL1-SAME-SOURCE-SWAP-20261111-attempt-1'
C71_AUDIT = 'LEVEL1-SAME-SOURCE-SWAP-20261111-audit-attempt-1'
SCORE_EPS = 1e-3
KNOT_COUNT = 5
FOLDS = ((.20, .30), (.45, .55), (.70, .80))
SCALE = 6_371_000. * np.pi / 180.
DIFFERENCES = np.diff(np.eye(KNOT_COUNT), n=2, axis=0)
RIDGE = .01 * DIFFERENCES.T @ DIFFERENCES + 1e-9 * np.eye(KNOT_COUNT)
ARMS = ('WS_MEAN', 'WS_FRAGMENT', 'WS_GROUP', 'J_GROUP', 'HOLDOUT_GROUP', 'J_ALL_GROUP')


def load_native(runs, sid, node_ids, sources, *, split='T_core'):
    if split not in ('T_core', 'V_select'):
        raise ValueError('native events require a declared development split')
    path = runs / PARENTS['native'] / 'private/native-events' / split / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as cache:
        if not np.array_equal(cache['node_ids'], node_ids) or not np.array_equal(cache['sources'], sources):
            raise ValueError('native event nodes or sources changed')
        query = cache['query_seconds']
        times = cache['event_times_seconds']
        latlon = cache['event_values'][:, :2]
        offsets = cache['event_offsets']
        n = len(node_ids)
        if (query.shape != (64,) or (np.diff(query) <= 0).any()
                or offsets.shape != (n+1,) or offsets[0] != 0 or offsets[-1] != len(times)
                or (np.diff(offsets) < 2).any() or latlon.shape != (len(times), 2)
                or not cache['event_valid'][:, :2].all()
                or not np.array_equal(cache['event_node_ids'], np.repeat(node_ids, np.diff(offsets)))):
            raise ValueError('frozen canonical native event lineage differs')
        indices = np.arange(len(times)) - np.repeat(offsets[:-1], np.diff(offsets))
        if not np.array_equal(cache['event_canonical_indices'], indices):
            raise ValueError('native canonical event order differs')
    if (not np.isfinite(query).all() or not np.isfinite(times).all() or not np.isfinite(latlon).all()
            or (np.abs(latlon) > [90., 180.]).any()
            or times.min() < query[0] - 1e-8 or times.max() > query[-1] + 1e-8):
        raise ValueError('native finite positions and query support required')
    for left, right in zip(offsets[:-1], offsets[1:], strict=True):
        if (np.diff(times[left:right]) <= 0).any():
            raise ValueError('strict per-fragment native order required')
    return query, times, latlon, offsets


def design(fraction):
    t = np.asarray(fraction).clip(0., 1.) * (KNOT_COUNT-1)
    interval = np.minimum(t.astype(np.int64), KNOT_COUNT-2)
    progress = t - interval
    basis = np.zeros((len(t), KNOT_COUNT))
    basis[np.arange(len(t)), interval] = 1. - progress
    basis[np.arange(len(t)), interval+1] = progress
    return basis, interval


@dataclass
class Statistics:
    gram: np.ndarray
    target: np.ndarray
    squared: np.ndarray

    def group_sum(self, membership):
        count = int(membership.max())+1
        totals = []
        for x in (self.gram, self.target, self.squared):
            y = np.zeros((count, *x.shape[1:]))
            np.add.at(y, membership, x)
            totals.append(y)
        return Statistics(*totals)

    def swap_in(self, group, incoming, outgoing, nodes):
        return Statistics(self.gram[group] + nodes.gram[incoming] - nodes.gram[outgoing],
                          self.target[group] + nodes.target[incoming] - nodes.target[outgoing],
                          self.squared[group] + nodes.squared[incoming] - nodes.squared[outgoing])


def node_statistics(basis, position, offsets, use):
    gram, rhs, squared = [], [], []
    for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
        selected = use[start:stop]
        x, y = basis[start:stop][selected], position[start:stop][selected]
        weight = 1. / (stop-start)
        gram.append((x.T @ x) * weight)
        rhs.append((x.T @ y) * weight)
        squared.append(np.sum(y*y) * weight)
    return Statistics(np.stack(gram), np.stack(rhs), np.array(squared))


def coefficients(stats):
    return np.linalg.solve(stats.gram + RIDGE, stats.target)


def squared_error(stats, beta):
    residual = stats.squared - 2.*np.sum(beta*stats.target, axis=(-2,-1))
    residual += np.sum(beta*(stats.gram @ beta), axis=(-2,-1))
    if not np.isfinite(residual).all() or np.any(residual < -1e-7 * np.maximum(1., stats.squared)):
        raise FloatingPointError('invalid squared holdout residual')
    return np.maximum(residual, 0.)


def best(gain, mask, epsilon):
    eligible = np.flatnonzero(mask & (gain > epsilon))
    return int(eligible[np.argmax(gain[eligible])]) if len(eligible) else -1


def holdout_scores(fraction, basis, position, offsets, membership, actions):
    left, right = actions.T
    a, b = membership[left], membership[right]
    comparable = np.ones(len(actions), dtype=bool)
    has_test = np.zeros(len(actions), dtype=bool)
    gain = np.zeros(len(actions))
    sizes = np.bincount(membership)
    records = []
    for lo, hi in FOLDS:
        test = (fraction >= lo) & (fraction < hi)
        side = np.column_stack((fraction[offsets[:-1]] < lo,
                                fraction[offsets[1:]-1] >= hi,
                                np.add.reduceat(test.astype(np.int64), offsets[:-1]) > 0))
        grouped = np.zeros((len(sizes), 3), dtype=np.int64)
        np.add.at(grouped, membership, side)
        swapped_a = grouped[a] + side[right] - side[left]
        swapped_b = grouped[b] + side[left] - side[right]
        has_test |= (grouped[a, 2] + grouped[b, 2]) > 0
        for counts in (grouped[a], grouped[b], swapped_a, swapped_b):
            comparable &= (counts[:, 2] == 0) | ((counts[:, 0] > 0) & (counts[:, 1] > 0))
        train = node_statistics(basis, position, offsets, ~test)
        heldout = node_statistics(basis, position, offsets, test)
        records.append((train, heldout))
    comparable &= has_test
    use = np.flatnonzero(comparable)
    for train, heldout in records:
        t_group, h_group = train.group_sum(membership), heldout.group_sum(membership)
        base = squared_error(h_group, coefficients(t_group))
        for dest, enter, leave in ((a, right, left), (b, left, right)):
            group, incoming, outgoing = dest[use], enter[use], leave[use]
            beta = coefficients(t_group.swap_in(group, incoming, outgoing, train))
            test = h_group.swap_in(group, incoming, outgoing, heldout)
            gain[use] += base[group] - squared_error(test, beta)
    return comparable, gain, int((~has_test).sum())


def motion_fields(runs, sid, query, nodes, sources):
    path = runs / PARENTS['native'] / 'private/native-motion/T_core' / f'scene-{sid}.npz'
    with np.load(path, allow_pickle=False) as arr:
        if (not np.array_equal(arr['query_seconds'], query) or not np.array_equal(arr['node_ids'], nodes)
                or not np.array_equal(arr['sources'], sources)):
            raise ValueError('native event and motion query binding failed')
        states, support, nearest = (arr[k] for k in ('states', 'support', 'nearest_seconds'))
    states = states.astype(np.float64)
    states[..., 2] *= KNOT_TO_MPS
    return states, support, nearest


def scene_arrays(native, *, check_holdout_origin=True):
    query, times, latlon, offsets = native
    duration = query[-1]-query[0]
    origin = latlon[np.argmin(times)]
    scale = SCALE * np.array([1., np.cos(np.deg2rad(origin[0]))])
    if scale[1] < 1e-9:
        raise ValueError('unusable local east scale')
    delta = latlon-origin
    delta[:, 1] = (delta[:, 1]+180.) % 360.-180.
    fraction = (times-query[0])/duration
    if check_holdout_origin and any(lo <= fraction.min() < hi for lo, hi in FOLDS):
        raise ValueError('coordinate origin enters a holdout block')
    basis, _ = design(fraction)
    xy = delta*scale
    stats = node_statistics(basis, xy, offsets, np.ones(len(times), bool))
    return stats, fraction, basis, xy, origin, scale


def mean_state(states, support, nearest, membership):
    groups = [g.tolist() for g in group_members(membership)]
    anchors = [g[0] for g in groups]
    return ordinary_fusion(states, nearest, support, groups)['predictions']['mean_supported'][anchors]


def state_output(stats, native, origin, scale, membership, fallback):
    query, times, _, offsets = native
    duration = query[-1]-query[0]
    basis, segment = design((query-query[0])/duration)
    beta = coefficients(stats.group_sum(membership))
    angles = basis @ beta / scale + origin
    angles[..., 1] = (angles[..., 1]+180.) % 360.-180.
    velocity = np.diff(beta, axis=1)[:, segment] * ((KNOT_COUNT-1)/duration)
    speed = np.linalg.norm(velocity, axis=-1)
    course = np.rad2deg(np.arctan2(velocity[..., 1], velocity[..., 0])) % 360.
    groups = group_members(membership)
    starts = np.array([times[offsets[g]].min() for g in groups])
    stops = np.array([times[offsets[g+1]-1].max() for g in groups])
    inside = (query[None] >= starts[:, None]) & (query[None] <= stops[:, None])
    course = np.where(speed < .5, fallback[..., 3], course)
    modeled = np.concatenate((angles, speed[..., None], course[..., None]), axis=-1)
    output = np.where(inside[..., None], modeled, fallback)
    if not np.isfinite(output).all() or (np.abs(output[..., 0]) > 90.).any():
        raise FloatingPointError('nonfinite or illegal state output')
    return output, dict(inside=int(inside.sum()), outside=int((~inside).sum()),
                        low_speed=int((inside & (speed < .5)).sum()))


def qualify_population(runs):
    scenes, components = load_population(runs)
    c71 = read(runs / C71_NAME / 'run.json')
    audit = read(runs / C71_AUDIT / 'audit.json')
    if (c71['status'] != 'COMPLETED' or c71['exit_code'] != 0 or c71['scene_ids'] != scenes
            or audit['status'] != 'COMPLETED' or audit['exit_code'] != 0 or audit['totals']['scenes'] != len(scenes)):
        raise ValueError('audited C71 frozen pool required')
    return scenes, components


def prediction_scene(runs, sid):
    nodes, sources, _, _, _, partitions = load_scene(runs, sid)
    native = load_native(runs, sid, nodes, sources)
    query, _, _, offsets = native
    states, support, nearest = motion_fields(runs, sid, query, nodes, sources)
    ws = partitions['WS']
    stats, fraction, basis, xy, origin, scale = scene_arrays(native)
    with np.load(runs / C71_NAME / 'private' / f'{sid}.npz', allow_pickle=False) as arr:
        keep = arr['WS_keep']
        actions, j_gain = arr['WS_actions'][keep], arr['WS_gain'][keep]
        original_choice = int(arr['WS_selected'][0])
        all_positions = np.flatnonzero(keep)
    comparable, score_gain, no_test = holdout_scores(fraction, basis, xy, offsets, ws, actions)
    chosen = np.array([best(j_gain, comparable, EPSILON), best(score_gain, comparable, SCORE_EPS),
                       best(j_gain, np.ones(len(actions), bool), EPSILON)])
    if (all_positions[chosen[2]] if chosen[2] >= 0 else -1) != original_choice:
        raise ValueError('unrestricted J selection differs from C71')
    result = dict(actions=actions, j_gain=j_gain, comparable=comparable, score_gain=score_gain, chosen=chosen)
    diagnostics = dict(actions=len(actions), comparable=int(comparable.sum()), no_test=no_test,
                       lacked_bracket=int((~comparable).sum())-no_test,
                       score_within_zero_floor=int((comparable & (np.abs(score_gain) <= SCORE_EPS)).sum()),
                       state_readout={})
    base_mean = mean_state(states, support, nearest, ws)
    singleton = np.arange(len(nodes), dtype=np.int64)
    fragment, diagnostics['state_readout']['WS_FRAGMENT'] = state_output(stats, native, origin, scale, singleton, states)
    result['WS_MEAN_membership'], result['WS_MEAN_states'] = ws, base_mean
    result['WS_FRAGMENT_membership'] = ws
    result['WS_FRAGMENT_states'] = mean_state(fragment, support, nearest, ws)
    for arm, pick in zip(ARMS[2:], np.r_[-1, chosen], strict=True):
        membership = canonical_membership(swapped(ws, actions[pick]), len(nodes)) if pick >= 0 else ws
        mean = base_mean if pick < 0 else mean_state(states, support, nearest, membership)
        output, diagnostics['state_readout'][arm] = state_output(stats, native, origin, scale, membership, mean)
        result[f'{arm}_membership'], result[f'{arm}_states'] = membership, output
    return result, diagnostics


def evaluation_scene(runs, output, sid, component, diagnostics):
    from tools.analysis.conditional_trajectory_evaluation import evaluate_scene

    nodes, sources, support, pairs, _, _ = load_scene(runs, sid)
    native = load_native(runs, sid, nodes, sources)
    query = native[0]
    motion, _, nearest = motion_fields(runs, sid, query, nodes, sources)
    reference_path = runs / PARENTS['native'] / 'private/native-evaluation/T_core' / f'scene-{sid}.npz'
    with np.load(reference_path, allow_pickle=False) as arr:
        if not np.array_equal(arr['node_ids'], nodes) or not np.array_equal(arr['query_seconds'], query):
            raise ValueError('evaluation node or query binding failed')
        identities = arr['identities']
    _, raw_membership = np.unique(identities, return_inverse=True)
    membership = canonical_membership(raw_membership, len(nodes))
    mean = mean_state(motion, support, nearest, membership)
    stats, _, _, _, origin, scale = scene_arrays(native)
    smoothed, diagnostics['state_readout']['TRUE_GROUP'] = state_output(stats, native, origin, scale, membership, mean)
    np.savez_compressed(output / 'private' / f'{sid}-truth-control.npz',
                        membership=membership, mean_states=mean, group_states=smoothed)
    with np.load(output / 'private' / f'{sid}.npz', allow_pickle=False) as arr:
        predictions = {arm: (arr[f'{arm}_membership'], arr[f'{arm}_states']) for arm in ARMS}
        comparable, chosen = arr['comparable'], arr['chosen']
        diagnostics['selected_score_gain'] = [float(arr['score_gain'][pick]) if pick >= 0 and comparable[pick] else None
                                               for pick in chosen]
    predictions.update(TRUE_MEAN=(membership, mean), TRUE_GROUP=(membership, smoothed))
    with np.load(runs / C71_NAME / 'private' / f'{sid}.npz', allow_pickle=False) as arr:
        keep = arr['WS_keep']
    with np.load(runs / C71_NAME / 'private' / f'{sid}-truth.npz', allow_pickle=False) as arr:
        delta = arr['WS_tp_delta'][keep]
    good = (delta >= 0).all(axis=1) & (delta > 0).any(axis=1)
    diagnostics.update(all_joint_actions=int(good.sum()), comparable_joint_actions=int((good & comparable).sum()),
                       selected_tp_delta=[delta[pick].tolist() if pick >= 0 else [0, 0] for pick in chosen])
    with np.load(reference_path, allow_pickle=False) as arr:
        ev = {key: arr[key] for key in ('target_ids', 'target_states', 'target_valid')}
    ev['identities'] = identities
    row, arrays = evaluate_scene(sid, component, nodes, sources, pairs, support, nearest, ev, predictions)
    row['scoring'] = diagnostics
    return row, arrays


def run(runs, output):
    from tools.analysis.conditional_trajectory_evaluation import summarize

    output.mkdir(exist_ok=False)
    (output/'private').mkdir()
    began = time.monotonic()
    report = dict(status='RUNNING', split='T_core', stages=[], started_at=datetime.now(timezone.utc).isoformat(),
                  optimizer_updates=0, affinity=sorted(os.sched_getaffinity(0)), folds=FOLDS)

    def checkpoint(**fields):
        report.update(fields, elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', report)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        scenes, components = qualify_population(runs)
        checkpoint(scene_ids=scenes, phase='prediction')
        diagnostics = {}
        for sid in scenes:
            payload, diagnostics[sid] = prediction_scene(runs, sid)
            np.savez_compressed(output/'private'/f'{sid}.npz', **payload)
        if sum(row['actions'] for row in diagnostics.values()) != 58975:
            raise ValueError('C71 original action total mismatch')
        report['stages'].append('all_predicted_partitions_and_states_saved')
        checkpoint(phase='evaluation', prediction_scenes=len(scenes))
        rows, arrays = [], []
        for sid in scenes:
            row, metrics = evaluation_scene(runs, output, sid, components[sid], diagnostics[sid])
            rows.append(row)
            arrays.append(metrics)
        write(output/'scenes.json.gz', rows)
        write(output/'summary.json', summarize(rows, arrays))
        report['stages'].append('T_identity_and_reference_evaluation_complete')
        checkpoint(status='COMPLETED', exit_code=0, completed_scenes=len(rows), phase='done',
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve())
