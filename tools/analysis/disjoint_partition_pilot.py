"""C78 finite real partition candidates with at most two disjoint WS edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.evaluation.association import _counts, _aggregate
from tools.analysis.conditional_trajectory_pilot import load_native
from tools.analysis.disjoint_partition_proposals import Action, propose
from tools.analysis.offline_multi_scope_bound import Scope, exact_matching
from tools.analysis.offline_partition_scope_bound import _edge_counts
from tools.analysis.same_source_swap_pilot import (
    EPSILON, MAX_AS, PARENTS, THRESHOLD, canonical_membership, dense_edges, group_members,
    load_identities, load_population, load_scene, peak_rss_bytes, read, write,
)

ARMS = ('SCORE_TREE', 'NATIVE_TREE')
EDGE_AXES = ('tp_same', 'tp_cross', 'fp_same', 'fp_cross')
SCOPES = ('same', 'cross')
MAX_EDITS = 192


def materialize(membership, edits):
    result = membership.copy()
    fresh = int(membership.max()) + 1
    for action in edits:
        for part in action.parts:
            result[list(part)] = fresh
            fresh += 1
    return canonical_membership(result, len(result))


def changed_edges(action, membership, groups):
    nodes = np.concatenate([groups[g] for g in action.scope])
    part = {node: i for i, group in enumerate(action.parts) for node in group}
    if len(part) != len(nodes) or set(part) != set(nodes):
        raise ValueError('edit must partition exactly its original-group union')
    labels = np.asarray([part[n] for n in nodes])
    i, j = np.triu_indices(len(nodes), 1)
    left, right = nodes[i], nodes[j]
    before = membership[left] == membership[right]
    after = labels[i] == labels[j]
    return left, right, after & ~before, before & ~after


def catalog(membership, actions, weights):
    if len(actions) > MAX_EDITS:
        raise ValueError('scene edit budget exceeded')
    groups = group_members(membership)
    gain = []
    for action in actions:
        left, right, added, removed = changed_edges(action, membership, groups)
        gain.append(float(weights[left[added], right[added]].sum()
                          - weights[left[removed], right[removed]].sum()))
    masks = [sum(1 << int(g) for g in a.scope) for a in actions]
    choices = [()] + [(i,) for i in range(len(actions))]
    choices.extend((i, j) for i, j in combinations(range(len(actions)), 2) if not masks[i] & masks[j])
    total = [sum(gain[i] for i in c) for c in choices]
    selected = []
    for count in (len(actions) + 1, len(choices)):
        best = int(np.argmax(total[:count]))
        selected.append(best if total[best] > EPSILON else 0)
    return dict(actions=[[a.scope, a.parts, a.family] for a in actions], choices=choices,
                primitive_J_gain=gain, J_gain=total, selected_J=selected)


def by_axes(values):
    return {axis: int(value) for axis, value in zip(EDGE_AXES, values, strict=True)}


def changed_counts(left, right, added, removed, identities, sources):
    truth = identities[left] == identities[right]
    same = sources[left] == sources[right]
    axes = (truth & same, truth & ~same, ~truth & same, ~truth & ~same)
    return np.asarray([[np.count_nonzero(mask & moved) for mask in axes]
                       for moved in (added, removed)], dtype=np.int64)


def anchor_population(counts):
    covered = int(np.count_nonzero(counts))
    return dict(represented=covered, missed=len(counts)-covered,
                duplicates=int(counts.sum())-covered, groups=int(counts.sum()))


def anchor_change(original, modified):
    return dict(lost=int(np.count_nonzero((original > 0) & (modified == 0))),
                recovered=int(np.count_nonzero((original == 0) & (modified > 0))),
                **anchor_population(modified))


def support_state(membership, support, sources):
    group_sources = []
    for nodes in group_members(membership):
        votes = [support[nodes[sources[nodes] == source]].any(axis=0) for source in np.unique(sources[nodes])]
        group_sources.append(np.minimum(2, np.sum(votes, axis=0)))
    state = np.asarray(group_sources)
    return state[membership], np.bincount(state.ravel(), minlength=3)


def selected_output(index, choices, actions, counts, record, membership, identities, sources, support, pairs,
                    weights, base_prediction, base_anchors, base_support):
    choice = choices[index]
    changed = materialize(membership, (actions[i] for i in choice))
    prediction = changed[pairs[:, 0]] == changed[pairs[:, 1]]
    added = prediction & ~base_prediction
    removed = base_prediction & ~prediction
    transitions = changed_counts(*pairs.T, added, removed, identities, sources)
    if not np.array_equal(transitions, counts[index]):
        raise ValueError('composed local and independent full-edge transitions disagree')
    full_j = weights[pairs[added, 0], pairs[added, 1]].sum() - weights[pairs[removed, 0], pairs[removed, 1]].sum()
    if not np.isclose(full_j, record['J_gain'][index], atol=1e-9, rtol=1e-10):
        raise ValueError('composed old J and full-edge J disagree')
    anchors = np.bincount([identities[nodes[0]] for nodes in group_members(changed)], minlength=len(base_anchors))
    node_source_state, group_source_state = support_state(changed, support, sources)
    before, before_groups = base_support
    support_transitions = np.bincount((3 * before + node_source_state).ravel(), minlength=9).reshape(3, 3)
    truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    same = sources[pairs[:, 0]] == sources[pairs[:, 1]]
    metrics = {name: dict(_counts(truth[mask], prediction[mask]), pair_count=int(mask.sum()),
                          positive_count=int(np.count_nonzero(truth & mask)))
               for name, mask in zip(SCOPES, (same, ~same), strict=True)}
    return dict(edits=len(choice), J_gain=float(full_j), metrics=metrics,
                edge_added=by_axes(transitions[0]), edge_removed=by_axes(transitions[1]),
                edge_net=by_axes(transitions[0]-transitions[1]), anchors=anchor_change(base_anchors, anchors),
                node_query_source_support_transition=support_transitions.tolist(),
                group_query_source_support_before=before_groups.tolist(),
                group_query_source_support_after=group_source_state.tolist())


def evaluate_catalog(record, membership, identity_codes, sources, support, pairs, weights):
    actions = [Action(tuple(s), tuple(tuple(p) for p in parts), family)
               for s, parts, family in record['actions']]
    choices = [tuple(c) for c in record['choices']]
    groups = group_members(membership)
    original_anchors = np.bincount([identity_codes[nodes[0]] for nodes in groups],
                                   minlength=int(identity_codes.max())+1)
    original_duplicates = len(groups) - np.count_nonzero(original_anchors)
    anchor_changes = [([identity_codes[groups[g][0]] for g in a.scope],
                       [identity_codes[p[0]] for p in a.parts]) for a in actions]
    primitive = [changed_counts(*changed_edges(a, membership, groups), identity_codes, sources)
                 for a in actions]
    counts = np.asarray([sum((primitive[i] for i in c), np.zeros((2, 4), dtype=np.int64))
                         for c in choices])
    net = counts[:, 0] - counts[:, 1]
    merit = net[:, :2].sum(axis=1) - net[:, 2:].sum(axis=1)
    joint = np.all(net[:, :2] >= 0, axis=1) & np.all(net[:, 2:] <= 0, axis=1) & (merit > 0)
    protected = np.zeros(len(choices), dtype=bool)
    for i in np.flatnonzero(joint):
        anchors = original_anchors.copy()
        for edit in choices[i]:
            old, new = anchor_changes[edit]
            np.add.at(anchors, old, -1)
            np.add.at(anchors, new, 1)
        protected[i] = (np.all(anchors[original_anchors > 0] > 0)
                        and anchors.sum()-np.count_nonzero(anchors) <= original_duplicates)
    before = membership[pairs[:, 0]] == membership[pairs[:, 1]]
    base_support = support_state(membership, support, sources)
    checked = {}

    def output(i):
        if i not in checked:
            checked[i] = selected_output(i, choices, actions, counts, record, membership, identity_codes, sources,
                                         support, pairs, weights, before, original_anchors, base_support)
        return checked[i]

    audits = [0]
    for count in (1, 2):
        eligible = [i for i, choice in enumerate(choices) if len(choice) == count][:4]
        audits.extend(eligible)
    for index in audits:
        output(index)

    result = {}
    for budget, count in ((1, len(actions)+1), (2, len(choices))):
        selected = record['selected_J'][budget-1]
        joint_options = np.flatnonzero(joint[:count])
        protected_options = np.flatnonzero(protected[:count])
        positive = np.asarray(record['J_gain'][:count]) > EPSILON
        joint_winner = int(joint_options[np.argmax(merit[joint_options])]) if len(joint_options) else 0
        safe_winner = int(protected_options[np.argmax(merit[protected_options])]) if len(protected_options) else 0
        result[str(budget)] = dict(
            candidates=count, joint_candidates=len(joint_options), protected_candidates=len(protected_options),
            positive_J_joint=int(np.count_nonzero(positive & joint[:count])),
            positive_J_protected=int(np.count_nonzero(positive & protected[:count])),
            J_joint=bool(joint[selected]), J_protected=bool(protected[selected]), J=output(selected),
            oracle_joint=output(joint_winner), oracle_protected=output(safe_winner),
        )
        if safe_winner and (result[str(budget)]['oracle_protected']['anchors']['lost'] != 0
                            or result[str(budget)]['oracle_protected']['anchors']['duplicates'] > original_duplicates):
            raise ValueError('protected oracle anchor invariant failed')
    return result, len(checked)


def scope_ceiling(membership, sources, identities, pairs, chosen_scopes):
    within, between, baseline = _edge_counts(membership, sources, identities, pairs)
    options = []
    for i, groups in enumerate(chosen_scopes):
        counts = [int(x) for x in within[list(groups)].sum(axis=0)]
        counts.extend(map(int, between[tuple(groups)] if len(groups) == 2 else (0, 0)))
        options.append(Scope('recut' if len(groups) == 2 else 'split', i, tuple(groups), 0, tuple(counts)))
    best = {}
    for budget in (1, 2):
        axes = []
        for axis in (None, 3):
            ranked = [(sum(s.counts) if axis is None else s.counts[axis], s) for s in options]
            ranked = sorted((x for x in ranked if x[0] > 0), key=lambda x: (-x[0], x[1].canonical))
            choices, _ = exact_matching(ranked, budget)
            axes.append([sum(s.counts[j] for s in choices) for j in range(4)])
        best[str(budget)] = dict(fixed_sum_vector=axes[0], cross_fn_marginal=axes[1][3])
    return best, baseline


def summarize(rows):
    result = dict(scene_count=len(rows), nodes=sum(r['nodes'] for r in rows), arms={})
    for arm in ARMS:
        result['arms'][arm] = {}
        for k in ('1', '2'):
            entries = [r['arms'][arm][k] for r in rows]
            layer = {key: sum(row[key] for row in entries) for key in
                     ('candidates', 'joint_candidates', 'protected_candidates', 'J_joint', 'J_protected')}
            for case in ('joint', 'protected'):
                has_candidate = f'{case}_candidates'
                has_positive = f'positive_J_{case}'
                accepted = f'J_{case}'
                layer[case+'_scenes'] = sum(row[has_candidate] > 0 for row in entries)
                layer[case+'_all_nonpositive_J_scenes'] = sum(
                    row[has_candidate] > 0 and row[has_positive] == 0 for row in entries)
                layer[case+'_outscored_scenes'] = sum(
                    row[has_positive] > 0 and not row[accepted] for row in entries)
            for method in ('J', 'oracle_joint', 'oracle_protected'):
                items = [r[method] for r in entries]
                layer[method] = dict(
                    metrics={s: _aggregate([x['metrics'][s] for x in items]) for s in SCOPES},
                    **{edge: {axis: sum(x[edge][axis] for x in items) for axis in EDGE_AXES}
                       for edge in ('edge_net', 'edge_added', 'edge_removed')},
                    anchors={key: sum(x['anchors'][key] for x in items) for key in
                             ('lost', 'recovered', 'represented', 'missed', 'duplicates', 'groups')},
                    scenes_changed=sum(x['edits'] > 0 for x in items),
                    two_edits=sum(x['edits'] == 2 for x in items),
                )
            result['arms'][arm][k] = layer
    result['shortlist_ceiling'] = {k: {
        'fixed_sum_vector': [sum(r['shortlist_ceiling'][k]['fixed_sum_vector'][i] for r in rows) for i in range(4)],
        'cross_fn_marginal': sum(r['shortlist_ceiling'][k]['cross_fn_marginal'] for r in rows),
    } for k in ('1', '2')}
    return result


def run(runs, output):
    output.mkdir(exist_ok=False)
    private = output / 'private'
    private.mkdir()
    start = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, split='T_core', optimizer_updates=0, gpu_used=False,
                 inputs={key: str(runs / name) for key, name in PARENTS.items()}, stages=[],
                 time_policy='advisory', affinity=sorted(os.sched_getaffinity(0)),
                 started_at=datetime.now(timezone.utc).isoformat(), completed_scenes=0)

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        checkpoint(phase='freeze', scene_ids=ids)
        for sid in ids:
            nodes, sources, _, pairs, scores, memberships = load_scene(runs, sid)
            old = memberships['WS']
            arms, meta = propose(old, pairs, scores, load_native(runs, sid, nodes, sources))
            weights = dense_edges(pairs, scores.astype(np.float64)-THRESHOLD, len(nodes))
            write(private/f'candidates-{sid}.json.gz',
                  dict(proposals=meta, arms={name: catalog(old, arms[name], weights) for name in ARMS}))
        state['stages'].append('all_real_single_pair_candidates_and_old_J_choices_frozen_before_identity')
        checkpoint(phase='evaluate')
        rows, verified = [], 0
        for sid in ids:
            nodes, sources, support, pairs, scores, memberships = load_scene(runs, sid)
            old = memberships['WS']
            identities = np.unique(load_identities(runs, sid, nodes), return_inverse=True)[1]
            weights = dense_edges(pairs, scores.astype(np.float64)-THRESHOLD, len(nodes))
            frozen = read(private/f'candidates-{sid}.json.gz')
            sets = sorted({tuple(a[0]) for arm in ARMS for a in frozen['arms'][arm]['actions']})
            ceiling, baseline = scope_ceiling(old, sources, identities, pairs, sets)
            arms = {}
            for name in ARMS:
                arms[name], checks = evaluate_catalog(frozen['arms'][name], old, identities, sources,
                                                      support, pairs, weights)
                verified += checks
            rows.append(dict(scene_id=sid, component_id=components[sid], nodes=len(nodes), baseline=baseline,
                             arms=arms, shortlist_ceiling=ceiling, proposals=frozen['proposals']))
            checkpoint(completed_scenes=len(rows))
        totals = [sum(row['baseline'][s][kind] for row in rows) for kind in ('fp', 'fn') for s in SCOPES]
        if sum(r['nodes'] for r in rows) != 2310 or totals != [298, 523, 124, 385]:
            raise ValueError('original C75 population or WS pair totals changed')
        groups = {c: [r for r in rows if r['component_id'] == c] for c in sorted(set(components.values()))}
        largest = min(groups, key=lambda c: (-sum(r['nodes'] for r in groups[c]), c))
        write(output/'scenes.json', rows)
        write(output/'summary.json', dict(
            all_scenes=summarize(rows), components={key: summarize(group) for key, group in groups.items()},
            without_largest=summarize([r for r in rows if r['component_id'] != largest]),
            largest_component=largest, independently_recounted_outputs=verified,
            graph_definition='complete old upper triangle, source equality; not paper-eligibility masks',
            ceiling_axes=['fp_same_touch', 'fp_cross_touch', 'fn_same_touch', 'fn_cross_touch'],
            ceiling_definition='posthoc relaxed bound on original scopes represented in either finite arm',
        ))
        state['stages'].append('signed_transition_anchor_and_complete_pair_recount')
        checkpoint(status='COMPLETED', exit_code=0, phase='done', independently_recounted_outputs=verified,
                   completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


def self_check():
    membership = np.repeat(np.arange(4), 2)
    pairs = np.column_stack(np.triu_indices(len(membership), 1))
    sources = np.tile([0, 1], 4)
    identities = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    actions = (Action((0, 1), ((0, 1, 2, 3),), 'merge'), Action((2, 3), ((4, 5, 6, 7),), 'merge'),
               Action((0, 2), ((0, 1, 4, 5),), 'merge'))
    weights = dense_edges(pairs, np.ones(len(pairs)), len(membership))
    cat = catalog(membership, actions, weights)
    assert cat['choices'] == [(), (0,), (1,), (2,), (0, 1)]
    result, audits = evaluate_catalog(cat, membership, identities, sources, np.ones((8, 64), bool), pairs, weights)
    assert audits == 5 and result['2']['J']['edits'] == 2 and result['2']['J_protected']
    assert result['2']['oracle_protected']['edge_net'] == dict(tp_same=4, tp_cross=4, fp_same=0, fp_cross=0)
    assert result['1']['joint_candidates'] == 2 and result['2']['joint_candidates'] == 3
    ceiling, _ = scope_ceiling(membership, sources, identities, pairs, [a.scope for a in actions])
    assert ceiling['2']['cross_fn_marginal'] == 4
    aggregate = summarize([dict(nodes=8, arms={arm: result for arm in ARMS}, shortlist_ceiling=ceiling)])
    assert aggregate['arms']['NATIVE_TREE']['2']['J']['metrics']['cross']['tp'] == 8
    shifted_membership = np.repeat(np.arange(70), 2)
    shifted_actions = (Action((65, 66), ((130, 131, 132, 133),), 'merge'),
                       Action((66, 67), ((132, 133, 134, 135),), 'merge'))
    assert catalog(shifted_membership, shifted_actions, np.ones((140, 140)))['choices'] == [(), (0,), (1,)]
    print('C78 self-check passed: disjoint composition, edge counts, anchors, J and scope ceiling.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.runs is not None and args.output is not None and args.time_policy == 'advisory':
        run(args.runs.resolve(), args.output.resolve())
    else:
        parser.error('--runs, --output and --time-policy advisory are required')
