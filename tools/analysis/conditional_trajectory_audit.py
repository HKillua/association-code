"""Identity-blind C70 audit of decisions and native-position state mathematics.

Original input loaders are shared. All comparability flags are independently
checked; bounded, prespecified candidates are refitted by augmented least
squares, not production sufficient statistics. Every predicted state's direct
fit and fallback are checked. No identity/reference or physical metric is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np


C71_NAME = 'LEVEL1-SAME-SOURCE-SWAP-20261111-attempt-1'
C71_AUDIT_NAME = 'LEVEL1-SAME-SOURCE-SWAP-20261111-audit-attempt-1'
ARMS = ('WS_MEAN', 'WS_FRAGMENT', 'WS_GROUP', 'J_GROUP', 'HOLDOUT_GROUP', 'J_ALL_GROUP')
FOLDS = ((.20, .30), (.45, .55), (.70, .80))
J_EPS = 1e-10
SCORE_EPS = 1e-3
EXPECTED_SCENES = 32
EXPECTED_COMPONENTS = 12
EXPECTED_ACTIONS = 58975
SCALE = 6371000. * np.pi / 180.
MEMORY_BYTES = 8 * 1024**3
SCORE_ATOL = 1e-3
SCORE_RTOL = 1e-7
STATE_ATOL = 1e-6
STATE_RTOL = 1e-8


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def peak_rss_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))


def groups_of(membership):
    member = np.asarray(membership)
    require(member.ndim == 1 and member.dtype.kind in 'iu' and (member >= 0).all(), 'integer partition')
    groups = sorted((np.flatnonzero(member == label) for label in np.unique(member)), key=lambda nodes: int(nodes[0]))
    labels = np.empty(len(member), np.int64)
    for index, nodes in enumerate(groups):
        labels[nodes] = index
    return labels, groups


def exchanged(membership, action):
    modified = np.asarray(membership).copy()
    if action is not None:
        u, v = action
        modified[u], modified[v] = membership[v], membership[u]
    return groups_of(modified)[0]


def basis(fraction):
    """Five fixed linear hats; direct distances, independent of interval writes."""
    fraction = np.asarray(fraction).clip(0., 1.)
    return np.maximum(1. - np.abs(fraction[:, None] - np.linspace(0., 1., 5)) * 4., 0.)


class DirectNative:
    def __init__(self, native):
        self.query, self.times, self.latlon, self.offsets = (np.asarray(value) for value in native)
        require(self.query.shape == (64,) and np.all(np.diff(self.query) > 0), 'fixed increasing Q64')
        require(self.offsets.ndim == 1 and self.offsets.dtype.kind in 'iu'
                and self.offsets[0] == 0 and self.offsets[-1] == len(self.times)
                and (np.diff(self.offsets) >= 2).all(), 'native fragment offsets')
        require(self.latlon.shape == (len(self.times), 2) and np.isfinite(self.times).all()
                and np.isfinite(self.latlon).all(), 'finite native position events')
        self.duration = float(self.query[-1] - self.query[0])
        self.fraction = (self.times - self.query[0]) / self.duration
        require((self.fraction >= -1e-9).all() and (self.fraction <= 1 + 1e-9).all(), 'events in fixed query window')
        self.origin_index = int(np.argmin(self.times))
        self.origin = self.latlon[self.origin_index]
        self.scale = SCALE * np.array([1., np.cos(np.deg2rad(self.origin[0]))])
        require(self.scale[1] > 1e-9, 'defined east chart scale')
        difference = self.latlon - self.origin
        difference[:, 1] = (difference[:, 1] + 180.) % 360. - 180.
        self.position = difference * self.scale
        self.design = basis(self.fraction)
        self.node = np.repeat(np.arange(len(self.offsets)-1), np.diff(self.offsets))
        self.weight = 1. / np.diff(self.offsets)[self.node]
        self.penalty = np.vstack((.1 * np.diff(np.eye(5), n=2, axis=0), np.sqrt(1e-9) * np.eye(5)))
        self.holdout = [(self.fraction >= lo) & (self.fraction < hi) for lo, hi in FOLDS]
        # A chart using a held-out position would change under label-free holdout
        # perturbation. The fixed earliest event must remain visible in all folds.
        require(not any(mask[self.origin_index] for mask in self.holdout), 'chart origin outside every holdout window')
        for start, stop in zip(self.offsets[:-1], self.offsets[1:]):
            require((np.diff(self.times[start:stop]) > 0).all(), 'strict native per-fragment times')

    def event_indices(self, nodes):
        return np.concatenate([np.arange(self.offsets[node], self.offsets[node+1]) for node in nodes])

    def fit(self, rows):
        root_weight = np.sqrt(self.weight[rows])[:, None]
        matrix = np.vstack((self.design[rows] * root_weight, self.penalty))
        target = np.vstack((self.position[rows] * root_weight, np.zeros((len(self.penalty), 2))))
        beta, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        require(rank == 5 and np.isfinite(beta).all(), 'finite full-rank independent ridge fit')
        return beta

    def comparable(self, membership, actions):
        """All actions, using native event counts instead of fit statistics."""
        labels, groups = groups_of(membership)
        a, b = labels[actions[:, 0]], labels[actions[:, 1]]
        u, v = actions.T
        result, has_test = np.ones(len(actions), bool), np.zeros(len(actions), bool)
        for (lo, hi), test in zip(FOLDS, self.holdout):
            counts = np.column_stack([np.bincount(self.node, weights=mask, minlength=len(labels)).astype(np.int64)
                                      for mask in (self.fraction < lo, self.fraction >= hi, test)])
            total = np.array([counts[nodes].sum(axis=0) for nodes in groups])
            sides = (total[a], total[b], total[a] - counts[u] + counts[v], total[b] - counts[v] + counts[u])
            for count in sides:
                result &= (count[:, 2] == 0) | ((count[:, 0] > 0) & (count[:, 1] > 0))
            has_test |= (total[a, 2] + total[b, 2]) > 0
        return result & has_test, ~has_test

    def group_loss(self, nodes):
        rows = self.event_indices(nodes)
        total, mass = 0., 0.
        for heldout in self.holdout:
            test = rows[heldout[rows]]
            if not len(test):
                continue
            train = rows[~heldout[rows]]
            predicted = self.design[test] @ self.fit(train)
            total += float(np.sum(self.weight[test] * np.sum((predicted - self.position[test])**2, axis=1)))
            mass += float(self.weight[test].sum())
        return total, mass

    def gain(self, membership, action):
        labels, groups = groups_of(membership)
        u, v = action
        a, b = groups[labels[u]], groups[labels[v]]
        original = [self.group_loss(group) for group in (a, b)]
        swapped = [self.group_loss(np.sort(np.r_[a[a != u], v])),
                   self.group_loss(np.sort(np.r_[b[b != v], u]))]
        np.testing.assert_allclose(sum(x[1] for x in original), sum(x[1] for x in swapped), rtol=0., atol=1e-10)
        return float(sum(x[0] for x in original) - sum(x[0] for x in swapped))

    def output(self, membership, fallback):
        _, groups = groups_of(membership)
        fraction = (self.query - self.query[0]) / self.duration
        query_design = basis(fraction)
        interval = np.minimum(np.floor(fraction.clip(0., 1.) * 4.).astype(int), 3)
        output = np.asarray(fallback).copy()
        require(output.shape == (len(groups), 64, 4), 'complete fallback output shape')
        for index, nodes in enumerate(groups):
            rows = self.event_indices(nodes)
            beta = self.fit(rows)
            angles = query_design @ beta / self.scale + self.origin
            angles[:, 1] = (angles[:, 1] + 180.) % 360. - 180.
            velocity = (beta[interval+1] - beta[interval]) * (4. / self.duration)
            speed = np.sqrt(np.sum(velocity**2, axis=1))
            course = np.rad2deg(np.arctan2(velocity[:, 1], velocity[:, 0])) % 360.
            course[speed < .5] = fallback[index, speed < .5, 3]
            fitted = np.column_stack((angles, speed, course))
            inside = (self.query >= self.times[rows].min()) & (self.query <= self.times[rows].max())
            output[index, inside] = fitted[inside]
        require(np.isfinite(output).all(), 'finite full output')
        return output


def mean_supported(states, support, membership):
    """Independent supported-member means, with original first-node fallbacks."""
    _, groups = groups_of(membership)
    result = np.empty((len(groups), states.shape[1], 4))
    for index, group in enumerate(groups):
        for query in range(states.shape[1]):
            selected = group[support[group, query]]
            fallback = states[group[0], query]
            if not len(selected):
                result[index, query] = fallback
                continue
            values = states[selected, query]
            result[index, query, [0, 2]] = values[:, [0, 2]].mean(axis=0)
            for column in (1, 3):
                angle = np.deg2rad(values[:, column])
                sine, cosine = float(np.sin(angle).mean()), float(np.cos(angle).mean())
                if np.hypot(sine, cosine) <= 1e-12:
                    value = fallback[column]
                else:
                    value = np.rad2deg(np.arctan2(sine, cosine))
                    value = (value + 180.) % 360. - 180. if column == 1 else value % 360.
                result[index, query, column] = value
    return result


def choose(gain, allowed, epsilon):
    options = np.flatnonzero(allowed & (gain > epsilon))
    return int(options[np.argmax(gain[options])]) if len(options) else -1


def checked_states(actual, expected):
    require(actual.shape == expected.shape and np.isfinite(actual).all(), 'finite complete saved states')
    error = actual - expected
    error[..., 1] = (error[..., 1] + 180.) % 360. - 180.
    error[..., 3] = (error[..., 3] + 180.) % 360. - 180.
    tolerance = STATE_ATOL + STATE_RTOL * np.abs(expected)
    require((np.abs(error) <= tolerance).all(), 'independent full state reconstruction')
    return np.max(np.abs(error), axis=(0, 1)).tolist()


def audit(runs, science, output, *, source=None, swap=None):
    runs, science, output = (Path(path).resolve() for path in (runs, science, output))
    require(output != runs and output != science and science not in output.parents, 'fresh independent audit output')
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='RUNNING', exit_code=None, started_at=datetime.now(timezone.utc).isoformat(),
                  runs=str(runs), science=str(science), scenes=[],
                  totals=dict(scenes=0, actions=0, comparability_checks=0, selected_decisions=0,
                              comparable_refits=0, sampled_ineligible_checks=0, state_values=0),
                  max_sampled_score_difference=0., max_state_difference_by_column=[0., 0., 0., 0.],
                  tolerance=dict(score_atol=SCORE_ATOL, score_rtol=SCORE_RTOL,
                                 state_atol=STATE_ATOL, state_rtol=STATE_RTOL, score_epsilon=SCORE_EPS))

    def checkpoint():
        report.update(elapsed_seconds=time.monotonic()-began, peak_rss_bytes=peak_rss_bytes())
        with (output/'audit.json').open('w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')

    try:
        checkpoint()
        if source is None:
            from tools.analysis import conditional_trajectory_pilot as source
        if swap is None:
            from tools.analysis import same_source_swap_pilot as swap
        require(source.C71_NAME == C71_NAME and source.C71_AUDIT == C71_AUDIT_NAME and tuple(source.ARMS) == ARMS
                and tuple(source.FOLDS) == FOLDS and source.SCORE_EPS == SCORE_EPS
                and source.EPSILON == J_EPS, 'frozen source constants')
        require(source.KNOT_COUNT == 5, 'five fixed basis knots')
        penalty = .01 * np.diff(np.eye(5), n=2, axis=0).T @ np.diff(np.eye(5), n=2, axis=0) + 1e-9*np.eye(5)
        np.testing.assert_array_equal(source.RIDGE, penalty)
        record = read_json(science/'run.json')
        require(record['status'] == 'COMPLETED' and record['exit_code'] == 0, 'completed science run')
        ids, components = swap.load_population(runs)
        require(len(ids) == len(set(ids)) == EXPECTED_SCENES, 'original T32 population')
        require(len({components[sid] for sid in ids}) == EXPECTED_COMPONENTS, 'original T12 components')
        require(record['scene_ids'] == ids, 'saved scene order binding')
        prior = read_json(runs/C71_NAME/'run.json')
        prior_audit = read_json(runs/C71_AUDIT_NAME/'audit.json')
        require(prior['status'] == 'COMPLETED' and prior['exit_code'] == 0 and prior['scene_ids'] == ids,
                'completed original C71 pool and scene binding')
        require(prior_audit['status'] == 'COMPLETED' and prior_audit['exit_code'] == 0
                and prior_audit['totals']['scenes'] == len(ids), 'completed independent C71 audit')
        for sid in ids:
            nodes, sources, frozen_support, pairs, _, memberships = swap.load_scene(runs, sid)
            ws, groups = groups_of(memberships['WS'])
            native = source.load_native(runs, sid, nodes, sources)
            direct = DirectNative(native)
            c71 = read_arrays(runs/C71_NAME/'private'/f'{sid}.npz')
            arrays = read_arrays(science/'private'/f'{sid}.npz')
            require(set(arrays) == {'actions', 'j_gain', 'comparable', 'score_gain', 'chosen'}
                    | {f'{arm}_{field}' for arm in ARMS for field in ('membership', 'states')}, 'complete saved policy schema')
            keep_indices = np.flatnonzero(c71['WS_keep'])
            actions = c71['WS_actions'][keep_indices]
            j_gain = c71['WS_gain'][keep_indices]
            np.testing.assert_array_equal(arrays['actions'], actions)
            np.testing.assert_array_equal(arrays['j_gain'], j_gain)
            require(actions.shape == (len(actions), 2) and actions.dtype.kind in 'iu', 'integer action pairs')
            require(len(np.unique(actions, axis=0)) == len(actions), 'unique action pairs')
            require(np.all(actions[:, 0] < actions[:, 1]), 'ordered action nodes')
            u, v = actions.T
            require(np.all(sources[u] == sources[v]) and np.all(ws[u] != ws[v]), 'same-source cross-group actions')
            anchors = np.array([group[0] for group in groups])
            require(np.all(u > anchors[ws[u]]) and np.all(v > anchors[ws[v]])
                    and np.all(v > anchors[ws[u]]) and np.all(u > anchors[ws[v]]), 'original two-group anchors fixed')
            comparable, no_test = direct.comparable(ws, actions)
            require(arrays['comparable'].dtype == bool, 'boolean saved comparability')
            np.testing.assert_array_equal(arrays['comparable'], comparable)
            score_gain, chosen = arrays['score_gain'], arrays['chosen']
            require(score_gain.shape == j_gain.shape and np.isfinite(score_gain).all(), 'finite complete score vector')
            require(np.all(score_gain[~comparable] == 0.), 'incomparable gain placeholder exactly zero')
            expected = np.array([choose(j_gain, comparable, J_EPS), choose(score_gain, comparable, SCORE_EPS),
                                 choose(j_gain, np.ones(len(actions), bool), J_EPS)])
            require(chosen.shape == (3,) and chosen.dtype.kind in 'iu', 'three integer selections')
            np.testing.assert_array_equal(chosen, expected)
            require((int(keep_indices[chosen[2]]) if chosen[2] >= 0 else -1) == int(c71['WS_selected'][0]),
                    'original unrestricted J choice preserved')
            sampled = {int(index) for index in chosen if index >= 0}
            for mask in (comparable, ~comparable):
                indices = np.flatnonzero(mask)
                if len(indices):
                    sampled.update((int(indices[0]), int(indices[-1])))
            score_checks = []
            for index in sorted(sampled):
                if comparable[index]:
                    recomputed = direct.gain(ws, actions[index])
                    np.testing.assert_allclose(score_gain[index], recomputed, rtol=SCORE_RTOL, atol=SCORE_ATOL)
                    error = abs(float(score_gain[index])-recomputed)
                    report['max_sampled_score_difference'] = max(report['max_sampled_score_difference'], error)
                    report['totals']['comparable_refits'] += 1
                    score_checks.append(dict(comparable=True, selected_roles=[int(k) for k in np.flatnonzero(chosen == index)],
                        recorded_gain=float(score_gain[index]), direct_gain=recomputed, absolute_difference=error,
                        reliably_positive=recomputed > SCORE_ATOL + SCORE_RTOL*abs(recomputed)))
                else:
                    report['totals']['sampled_ineligible_checks'] += 1
                    score_checks.append(dict(comparable=False, selected_roles=[int(k) for k in np.flatnonzero(chosen == index)]))
            motion_path = runs/swap.PARENTS['native']/'private/native-motion/T_core'/f'scene-{sid}.npz'
            with np.load(motion_path, allow_pickle=False) as motion:
                for key, value in (('node_ids', nodes), ('sources', sources), ('query_seconds', native[0]), ('support', frozen_support)):
                    np.testing.assert_array_equal(motion[key], value)
                states = motion['states'].astype(np.float64)
                support = motion['support']
            states[..., 2] *= 1852./3600.
            require(states.shape == (len(nodes), 64, 4) and np.isfinite(states).all(), 'finite native prediction input')
            base_mean = mean_supported(states, support, ws)
            fragment = direct.output(np.arange(len(nodes)), states)
            outputs = {'WS_MEAN': (ws, base_mean), 'WS_FRAGMENT': (ws, mean_supported(fragment, support, ws))}
            for arm, selected in zip(ARMS[2:], np.r_[-1, chosen]):
                membership = exchanged(ws, actions[selected] if selected >= 0 else None)
                _, new_groups = groups_of(membership)
                require(len(new_groups) == len(groups), 'output group count preserved')
                for before, after in zip(groups, new_groups):
                    require(before[0] == after[0] and len(before) == len(after)
                            and Counter(sources[before].tolist()) == Counter(sources[after].tolist()), 'output anchor/capacity/source invariants')
                fallback = mean_supported(states, support, membership)
                outputs[arm] = (membership, direct.output(membership, fallback))
            for arm, (membership, predicted) in outputs.items():
                np.testing.assert_array_equal(arrays[f'{arm}_membership'], membership)
                error = checked_states(arrays[f'{arm}_states'], predicted)
                report['max_state_difference_by_column'] = np.maximum(report['max_state_difference_by_column'], error).tolist()
                report['totals']['state_values'] += predicted.size
            report['scenes'].append(dict(scene_id=sid, component_id=str(components[sid]), actions=len(actions),
                comparable=int(comparable.sum()), no_test=int(no_test.sum()), sampled_candidates=len(sampled),
                sampled_score_checks=score_checks, output_arms_checked=list(ARMS)))
            report['totals']['scenes'] += 1
            report['totals']['actions'] += len(actions)
            report['totals']['comparability_checks'] += len(actions)
            report['totals']['selected_decisions'] += 3
            checkpoint()
            require(peak_rss_bytes() <= MEMORY_BYTES, 'RSS below 8GiB')
        require(report['totals']['actions'] == EXPECTED_ACTIONS, 'fixed complete C71 WS keep pool')
        report.update(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat(),
            scope='shared original population/scene/native loaders; all saved actions bound to audited C71; all comparability flags and exact stored-score selections independently checked; direct augmented least-squares refits only for union of selected actions plus first/last comparable actions, first/last incomparable actions checked structurally; all six complete predicted state outputs independently reconstructed with explicit support/low-speed/outside-extent fallback; no identities/reference, true-group outputs, physical errors or public evaluation aggregates read or certified; no claim of independently refitting every candidate or verifying no earlier process-level leakage')
    except BaseException as error:
        report.update(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}',
                      finished_at=datetime.now(timezone.utc).isoformat())
        raise
    finally:
        checkpoint()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'science', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    from tools.analysis import conditional_trajectory_pilot as source
    from tools.analysis import same_source_swap_pilot as swap
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    audit(args.runs, args.science, args.output, source=source, swap=swap)
