"""Independent T32 audit of frozen same-source swap candidates and choices.

The source loaders are shared. Enumeration, anchor constraints, objective
algebra, identity-count deltas and explicit selected-pair accounting are
recomputed here, without production candidate or evaluation functions.
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


THRESHOLD = 1.7320103112387182
EPSILON = 1e-10
STARTS = ('WS', 'HARD', 'STRUCT')
EXPECTED_SCENES = 32
EXPECTED_COMPONENTS = 12
ATOL = 1e-8
RTOL = 1e-10
MEMORY_BYTES = 8 * 1024**3


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
    scale = 1 if sys.platform == 'darwin' else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale)


def canonical_groups(membership):
    membership = np.asarray(membership)
    require(membership.ndim == 1 and membership.dtype.kind in 'iu', 'integer membership')
    groups = [np.flatnonzero(membership == value) for value in np.unique(membership)]
    groups.sort(key=lambda nodes: int(nodes[0]))
    labels = np.empty(len(membership), np.int64)
    for gid, nodes in enumerate(groups):
        labels[nodes] = gid
    return labels, groups


def enumerate_actions(sources, membership, pairs, scores):
    """All-pair enumeration and direct changed-edge sums; no identities accepted."""
    sources = np.asarray(sources)
    labels, groups = canonical_groups(membership)
    n = len(labels)
    pairs, scores = np.asarray(pairs), np.asarray(scores)
    require(n > 0 and sources.shape == (n,), 'one source and group for every node')
    require(pairs.shape == (n * (n - 1) // 2, 2) and pairs.dtype.kind in 'iu', 'complete integer pairs')
    require(scores.shape == (len(pairs),) and np.isfinite(scores).all(), 'finite full scores')
    require(np.all((pairs >= 0) & (pairs < n)) and np.all(pairs[:, 0] < pairs[:, 1]), 'unordered nonself pairs')
    require(len(np.unique(pairs, axis=0)) == len(pairs), 'unique complete pairs')
    sizes = np.array([len(nodes) for nodes in groups], np.int64)
    anchors = np.array([nodes[0] for nodes in groups], np.int64)
    u, v = np.triu_indices(n, 1)
    legal = ((sources[u] == sources[v]) & (labels[u] != labels[v])
             & ~((sizes[labels[u]] == 1) & (sizes[labels[v]] == 1)))
    actions = np.column_stack((u[legal], v[legal])).astype(np.int64, copy=False)
    u, v = actions.T
    left, right = labels[u], labels[v]
    keep = ((u != anchors[left]) & (v != anchors[right])
            & (v > anchors[left]) & (u > anchors[right]))
    weights = np.zeros((n, n), np.float64)
    weights[pairs[:, 0], pairs[:, 1]] = scores - THRESHOLD
    weights += weights.T
    gain = np.empty(len(actions), np.float64)
    # Sum only edges that change, avoiding the production per-group-sum formula.
    # The u-v edge remains between groups and never enters either remaining set.
    for index, (a, b) in enumerate(actions):
        remaining_a = groups[labels[a]][groups[labels[a]] != a]
        remaining_b = groups[labels[b]][groups[labels[b]] != b]
        gain[index] = ((weights[b, remaining_a] - weights[a, remaining_a]).sum()
                       + (weights[a, remaining_b] - weights[b, remaining_b]).sum())
    require(np.isfinite(gain).all(), 'finite independently reconstructed gains')
    return actions, keep, gain


def identity_deltas(actions, sources, membership, identities):
    """Exact integer ΔTP via group/identity and group/source/identity counts."""
    actions, sources, identities = np.asarray(actions), np.asarray(sources), np.asarray(identities)
    labels, _ = canonical_groups(membership)
    require(identities.shape == sources.shape == labels.shape, 'one identity per source-bound node')
    source_values, identity_values = sources.tolist(), identities.tolist()
    total = Counter(zip(labels.tolist(), identity_values))
    within_source = Counter(zip(labels.tolist(), source_values, identity_values))
    result = np.empty((len(actions), 2), np.int64)
    for index, (u, v) in enumerate(actions):
        a, b = int(labels[u]), int(labels[v])
        yu, yv, source = identity_values[u], identity_values[v], source_values[u]
        require(sources[u] == sources[v] and a != b, 'same-source cross-group truth accounting')
        self_correction = 2 - 2 * int(yu == yv)
        all_delta = total[a, yv] - total[a, yu] + total[b, yu] - total[b, yv] + self_correction
        itsa_delta = (within_source[a, source, yv] - within_source[a, source, yu]
                      + within_source[b, source, yu] - within_source[b, source, yv] + self_correction)
        result[index] = itsa_delta, all_delta - itsa_delta
    same = identities[actions[:, 0]] == identities[actions[:, 1]]
    require(not result[same].any(), 'same-identity swap must preserve all true edge counts')
    return result


def select(gain, allowed):
    """Original first maximum in lex action order; positive improvement only."""
    gain, allowed = np.asarray(gain), np.asarray(allowed)
    indices = np.flatnonzero(allowed)
    if not len(indices):
        return -1
    chosen = int(indices[np.argmax(gain[indices])])
    return chosen if gain[chosen] > EPSILON else -1


def full_counts(membership, pairs, sources, identities):
    prediction = membership[pairs[:, 0]] == membership[pairs[:, 1]]
    truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    same_source = sources[pairs[:, 0]] == sources[pairs[:, 1]]
    result = {}
    for name, scope in (('all', np.ones(len(pairs), bool)), ('itsa', same_source), ('mstsa', ~same_source)):
        p, t = prediction[scope], truth[scope]
        tp, fp = int((p & t).sum()), int((p & ~t).sum())
        fn, tn = int((~p & t).sum()), int((~p & ~t).sum())
        result[name] = dict(pair_count=len(p), predicted_positive=tp + fp,
                            truth_positive=tp + fn, tp=tp, fp=fp, fn=fn, tn=tn,
                            pca=tp / (tp + fn) if tp + fn else None,
                            pfa=fp / (tp + fp) if tp + fp else None)
    return result


def check_public_counts(published, expected):
    """Bind every published per-scene association metric to full-pair counts."""
    values = {key: expected[key] for key in ('pair_count', 'tp', 'fp', 'fn', 'tn')}
    values['positive_count'] = expected['truth_positive']
    for key, value in values.items():
        require(type(published[key]) is int and published[key] == value, 'published count: ' + key)
    tp, fp, fn = (expected[key] for key in ('tp', 'fp', 'fn'))
    precision = tp / (tp + fp) if tp + fp else None
    values = dict(f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.,
                  precision=precision, recall=expected['pca'], p_ca=expected['pca'], p_fa=expected['pfa'])
    for key, value in values.items():
        if value is None:
            require(published[key] is None, 'undefined published metric: ' + key)
        else:
            np.testing.assert_allclose(published[key], value, rtol=RTOL, atol=ATOL)


def full_selected_check(membership, sources, pairs, scores, identities,
                        actions, gain, delta, selected, anchor_preserving):
    before = np.asarray(membership)
    after = before.copy()
    if selected >= 0:
        u, v = actions[selected]
        after[u], after[v] = before[v], before[u]
    _, groups = canonical_groups(before)
    _, new_groups = canonical_groups(after)
    require(len(groups) == len(new_groups), 'selected group count unchanged')
    for gid, nodes in enumerate(groups):
        after_nodes = np.flatnonzero(after == before[nodes[0]])
        require(len(after_nodes) == len(nodes), 'selected original-group size unchanged')
        require(Counter(sources[after_nodes].tolist()) == Counter(sources[nodes].tolist()),
                'selected original-group per-source counts unchanged')
        if anchor_preserving:
            require(after_nodes[0] == nodes[0], 'selected original-group anchor unchanged')
    expected = np.zeros(2, np.int64) if selected < 0 else delta[selected]
    old_counts, new_counts = (full_counts(m, pairs, sources, identities) for m in (before, after))
    for scope, d in (('all', int(expected.sum())), ('itsa', int(expected[0])), ('mstsa', int(expected[1]))):
        a, b = old_counts[scope], new_counts[scope]
        require(a['predicted_positive'] == b['predicted_positive'], 'fixed predicted positives: ' + scope)
        require(b['tp'] - a['tp'] == d and b['fp'] - a['fp'] == -d
                and b['fn'] - a['fn'] == -d and b['tn'] - a['tn'] == d,
                'full selected contingency deltas: ' + scope)
    edge_weights = scores - THRESHOLD
    old_j = float(edge_weights[before[pairs[:, 0]] == before[pairs[:, 1]]].sum())
    new_j = float(edge_weights[after[pairs[:, 0]] == after[pairs[:, 1]]].sum())
    expected_gain = 0. if selected < 0 else float(gain[selected])
    np.testing.assert_allclose(new_j - old_j, expected_gain, rtol=RTOL, atol=ATOL)
    old_prediction = before[pairs[:, 0]] == before[pairs[:, 1]]
    new_prediction = after[pairs[:, 0]] == after[pairs[:, 1]]
    truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    same = sources[pairs[:, 0]] == sources[pairs[:, 1]]
    transitions = {}
    for scope, mask in (('itsa', same), ('mstsa', ~same)):
        transitions[scope] = dict(
            true_added=int((mask & truth & new_prediction & ~old_prediction).sum()),
            true_removed=int((mask & truth & old_prediction & ~new_prediction).sum()),
            false_added=int((mask & ~truth & new_prediction & ~old_prediction).sum()),
            false_removed=int((mask & ~truth & old_prediction & ~new_prediction).sum()))
    return dict(changed=selected >= 0, full_pair_gain=new_j - old_j,
                tp_delta_itsa=int(expected[0]), tp_delta_mstsa=int(expected[1]),
                jointly_improving=bool((expected >= 0).all() and (expected > 0).any()),
                before=old_counts, after=new_counts, transitions=transitions)


def check_public_choice(published, expected):
    require(type(published['changed']) is bool and published['changed'] == expected['changed'],
            'published changed flag')
    require(type(published['joint_improvement']) is bool
            and published['joint_improvement'] == expected['jointly_improving'], 'published joint flag')
    np.testing.assert_allclose(published['gain'], expected['full_pair_gain'], rtol=RTOL, atol=ATOL)
    require(set(published['metrics']) == {'itsa', 'mstsa'}, 'published two metric scopes')
    for scope in ('itsa', 'mstsa'):
        check_public_counts(published['metrics'][scope], expected['after'][scope])
        require(published['transitions'][scope] == expected['transitions'][scope], 'published edge transitions')


def audit(runs, science, output, *, source=None):
    runs, science, output = map(lambda p: Path(p).resolve(), (runs, science, output))
    require(output != science and science not in output.parents and output != runs,
            'fresh audit output must not overwrite science or input root')
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = dict(status='RUNNING', exit_code=None, runs=str(runs), science=str(science),
                  started_at=datetime.now(timezone.utc).isoformat(), scenes=[],
                  totals=dict(scenes=0, start_partitions=0, actions=0, anchor_preserving_actions=0,
                              objective_checks=0, exact_truth_delta_checks=0, selected_full_pair_checks=0,
                              oracle_full_pair_checks=0, public_scene_start_checks=0),
                  tolerance=dict(objective_atol=ATOL, objective_rtol=RTOL, selection_epsilon=EPSILON),
                  max_independent_gain_difference=0., max_selected_optimum_gap=0.)

    def checkpoint():
        report.update(elapsed_seconds=time.monotonic() - began, peak_rss_bytes=peak_rss_bytes())
        with (output / 'audit.json').open('w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')

    try:
        checkpoint()
        if source is None:
            from tools.analysis import same_source_swap_pilot as source
        require(source.THRESHOLD == THRESHOLD and source.EPSILON == EPSILON
                and tuple(source.STARTS) == STARTS, 'frozen threshold, epsilon and starts')
        record = read_json(science / 'run.json')
        require(record['status'] == 'COMPLETED' and record['exit_code'] == 0, 'terminal completed science')
        ids, components = source.load_population(runs)
        require(len(ids) == len(set(ids)) == EXPECTED_SCENES and all(isinstance(s, str) for s in ids), 'original T32 scenes')
        require(len({str(components[sid]) for sid in ids}) == EXPECTED_COMPONENTS, 'original T12 components')
        if 'scene_ids' in record:
            require(record['scene_ids'] == ids, 'science scene order binding')
        public_rows = read_json(science / 'scenes.json')
        require(isinstance(public_rows, list) and [row['scene_id'] for row in public_rows] == ids,
                'published scene population and order')
        public_by_scene = {row['scene_id']: row for row in public_rows}
        for sid in ids:
            node_ids, sources, support, pairs, scores, memberships = source.load_scene(runs, sid)
            node_ids, sources, support = map(np.asarray, (node_ids, sources, support))
            pairs, scores = np.asarray(pairs), np.asarray(scores, np.float64)
            require(node_ids.ndim == 1 and len(set(node_ids.tolist())) == len(node_ids), 'unique node binding')
            require(sources.shape == node_ids.shape and support.ndim == 2 and support.shape[0] == len(node_ids), 'source/support/node population')
            require(set(memberships) == set(STARTS), 'three frozen starting partitions')
            saved = read_arrays(science / 'private' / f'{sid}.npz')
            require(set(saved) == {start + '_' + name for start in STARTS for name in ('actions', 'gain', 'keep', 'selected')},
                    'complete frozen action artifact')
            expected = {}
            scene_report = dict(scene_id=sid, component_id=str(components[sid]), node_count=len(node_ids), starts={})
            published = public_by_scene[sid]
            require(str(published['component_id']) == str(components[sid]), 'published component binding')
            # Verify the entire truth-free pool and both choices before opening identities.
            for start in STARTS:
                membership = np.asarray(memberships[start])
                require(membership.shape == node_ids.shape, 'complete baseline partition')
                actions, keep, gain = enumerate_actions(sources, membership, pairs, scores)
                a, g, k, chosen = (saved[start + '_' + name] for name in ('actions', 'gain', 'keep', 'selected'))
                require(a.dtype.kind in 'iu' and k.dtype.kind == 'b' and chosen.dtype.kind in 'iu', 'action/keep/selection dtypes')
                require(g.shape == gain.shape and np.isfinite(g).all() and chosen.shape == (2,), 'gain and selection shapes')
                np.testing.assert_array_equal(a, actions)
                np.testing.assert_array_equal(k, keep)
                np.testing.assert_allclose(g, gain, rtol=RTOL, atol=ATOL)
                if len(g):
                    report['max_independent_gain_difference'] = max(report['max_independent_gain_difference'], float(np.abs(g - gain).max()))
                np.testing.assert_array_equal(chosen, np.array([select(g, keep), select(g, np.ones(len(g), bool))]))
                for selected, allowed in zip(chosen, (keep, np.ones(len(g), bool))):
                    optimum = max(0., float(gain[allowed].max())) if allowed.any() else 0.
                    actual = 0. if selected < 0 else float(gain[selected])
                    gap = max(0., optimum - actual)
                    require(gap <= EPSILON + ATOL + RTOL * max(abs(optimum), abs(actual)), 'independent near-optimal selection')
                    report['max_selected_optimum_gap'] = max(report['max_selected_optimum_gap'], gap)
                expected[start] = (actions, keep, gain, chosen)
                report['totals']['start_partitions'] += 1
                report['totals']['actions'] += len(actions)
                report['totals']['anchor_preserving_actions'] += int(keep.sum())
                report['totals']['objective_checks'] += len(gain)
            identities = np.asarray(source.load_identities(runs, sid, node_ids))
            require(identities.shape == node_ids.shape, 'T identity binding')
            truth_saved = read_arrays(science / 'private' / f'{sid}-truth.npz')
            require(set(truth_saved) == {start + '_tp_delta' for start in STARTS}, 'complete frozen truth artifact')
            for start in STARTS:
                actions, keep, gain, chosen = expected[start]
                delta = identity_deltas(actions, sources, memberships[start], identities)
                actual_delta = truth_saved[start + '_tp_delta']
                require(actual_delta.dtype.kind in 'iu', 'integer true edge deltas')
                np.testing.assert_array_equal(actual_delta, delta)
                report['totals']['exact_truth_delta_checks'] += len(delta) * 2
                item = dict(actions=len(actions), anchor_preserving_actions=int(keep.sum()),
                            same_identity_actions=int((identities[actions[:, 0]] == identities[actions[:, 1]]).sum()),
                            joint_improving_actions={}, selected={}, oracle={})
                membership = np.asarray(memberships[start])
                baseline = full_counts(membership, pairs, sources, identities)
                for scope in ('itsa', 'mstsa'):
                    check_public_counts(published[start]['baseline'][scope], baseline[scope])
                good = (delta >= 0).all(axis=1) & (delta > 0).any(axis=1)
                for mode, selected in zip(('keep', 'all'), chosen):
                    allowed = keep if mode == 'keep' else np.ones(len(actions), bool)
                    candidates = np.flatnonzero(allowed & good)
                    oracle = int(candidates[np.argmax(delta[candidates].sum(axis=1))]) if len(candidates) else -1
                    arm = published[start][mode]
                    require(type(arm['actions']) is int and arm['actions'] == int(allowed.sum()),
                            'published legal action count')
                    require(type(arm['joint_improving_actions']) is int
                            and arm['joint_improving_actions'] == len(candidates), 'published joint opportunity count')
                    item['joint_improving_actions'][mode] = len(candidates)
                    item['selected'][mode] = full_selected_check(membership, sources, pairs, scores,
                                                               identities, actions, gain, delta, int(selected), mode == 'keep')
                    item['oracle'][mode] = full_selected_check(membership, sources, pairs, scores,
                                                             identities, actions, gain, delta, oracle, mode == 'keep')
                    check_public_choice(arm['selected'], item['selected'][mode])
                    check_public_choice(arm['oracle'], item['oracle'][mode])
                    report['totals']['selected_full_pair_checks'] += 1
                    report['totals']['oracle_full_pair_checks'] += 1
                report['totals']['public_scene_start_checks'] += 1
                scene_report['starts'][start] = item
            report['scenes'].append(scene_report)
            report['totals']['scenes'] += 1
            checkpoint()
            if peak_rss_bytes() > MEMORY_BYTES:
                raise MemoryError('RSS exceeded 8GiB')
        report.update(status='COMPLETED', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat(),
                      scope='shared read-only original T loaders; independent complete action enumeration, original-group anchors, changed-edge signed gains and Counter identity-count TP deltas; exact scientific tie policy plus tolerance-checked independent optimum; full selected/oracle pair contingencies and positive-count invariants; published scenes.json baseline/selected/oracle metrics, edge transitions, choices and opportunity counts bound to these independent results; summary.json aggregates and other descriptive scene fields are not checked; action counts are not deduplicated partitions; no reference, physical states, training, V or test')
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
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    from tools.analysis import same_source_swap_pilot as source
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    audit(args.runs, args.science, args.output, source=source)
