"""C85: validate split-only motion evidence and original-query support on held T components."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from tools.analysis.disjoint_partition_evidence import _choices, _select
from tools.analysis.disjoint_partition_motion_rerank import MODES as C82_MODES, edge_scores, qualify_edges
from tools.analysis.disjoint_partition_pilot import (
    ARMS, Action, catalog, changed_edges, materialize, qualify_catalog, selected_output, support_state,
)
from tools.analysis.disjoint_partition_proposals import propose
from tools.analysis.disjoint_partition_rerank import positive_rank, summarize
from tools.analysis.held_partition_population import PARENTS, load_identities, load_population, load_scene
from tools.analysis.same_source_swap_pilot import EPSILON, MAX_AS, dense_edges, group_members, peak_rss_bytes, read, write

MODES = ('OLD_J', 'ABS', 'CROSS_REL', 'REL_SPLIT', 'ABS_KEEP_Q', 'REL_SPLIT_KEEP_Q')
GUARDED = ('ABS_KEEP_Q', 'REL_SPLIT_KEEP_Q')


def asymmetric_weights(ws, pairs, sources, motion):
    old, absolute, relative = (motion[C82_MODES.index(mode)] for mode in ('OLD_J', 'ABS', 'CROSS_REL'))
    left, right = pairs.T
    old_cross_link = (ws[left] == ws[right]) & (sources[left] != sources[right])
    split = np.where(old_cross_link, relative, absolute)
    return np.stack((old, absolute, relative, split, absolute, split))


def lost_queries(action, support, before):
    """Count individual originally supported node-query slots that become unsupported."""
    lost = 0
    for part in action.parts:
        nodes = list(part)
        lost += int(np.count_nonzero(before[nodes] & ~support[nodes].any(axis=0)))
    return lost


def score_catalog(record, ws, support, edge, pair_index, mutable):
    groups = group_members(ws)
    actions = [Action(tuple(scope), tuple(map(tuple, parts)), family)
               for scope, parts, family in record['actions']]
    choices = _choices(record, actions)
    original_support = np.empty_like(support)
    for nodes in groups:
        original_support[nodes] = support[nodes].any(axis=0)
    primitive = np.zeros((len(MODES), len(actions)+1))
    primitive_lost = np.zeros(len(actions)+1, dtype=np.int64)
    for index, action in enumerate(actions):
        left, right, added, removed = changed_edges(action, ws, groups)
        add = pair_index[left[added], right[added]]
        remove = pair_index[left[removed], right[removed]]
        mutable[add] = True
        mutable[remove] = True
        primitive[:, index] = edge[:, add].sum(axis=1)-edge[:, remove].sum(axis=1)
        primitive_lost[index] = lost_queries(action, support, original_support)
    indices = np.full((len(choices), 2), len(actions), dtype=np.int64)
    for index, choice in enumerate(choices):
        indices[index, :len(choice)] = choice
    gains = primitive[:, indices].sum(axis=2)
    lost = primitive_lost[indices].sum(axis=1)
    np.testing.assert_allclose(primitive[0, :-1], record['primitive_J_gain'], atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(gains[0], record['J_gain'], atol=1e-9, rtol=1e-10)
    gains[0] = record['J_gain']
    every = np.ones(len(choices), bool)
    accepted, forced = zip(*(_select(row, lost == 0 if mode in GUARDED else every)
                             for mode, row in zip(MODES, gains, strict=True)), strict=True)
    unrestricted = [_select(row, every)[1] for row in gains]
    if accepted[0] != record['selected_J'][1]:
        raise ValueError('C78 old-J tie/order parity failed')
    return dict(gains=gains, lost=lost, accepted=np.asarray(accepted), forced=np.asarray(forced),
                unrestricted=np.asarray(unrestricted))


def evaluate_arm(record, saved, ws, identities, sources, support, pairs, edge):
    actions, choices, anchors, _, counts, joint, protected, _ = qualify_catalog(record, ws, identities, sources)
    old_weights = dense_edges(pairs, edge[0], len(ws))
    before = ws[pairs[:, 0]] == ws[pairs[:, 1]]
    base_support = support_state(ws, support, sources)
    gains, lost = saved['gains'], saved['lost']
    checked = {}

    def output(index):
        if index not in checked:
            state = materialize(ws, (actions[i] for i in choices[index]))
            after = state[pairs[:, 0]] == state[pairs[:, 1]]
            full_delta = edge[:, after & ~before].sum(axis=1) - edge[:, before & ~after].sum(axis=1)
            np.testing.assert_allclose(full_delta, gains[:, index], atol=1e-9, rtol=1e-10)
            selected = selected_output(index, choices, actions, counts, record, ws, identities,
                                       sources, support, pairs, old_weights, before, anchors, base_support)
            transitions = np.asarray(selected['node_query_source_support_transition'])
            if int(transitions[1:, 0].sum()) != lost[index]:
                raise ValueError('strict support count differs in full selected partition')
            checked[index] = selected, transitions
        return checked[index]

    modes = {}
    for i, mode in enumerate(MODES):
        selected, forced, unrestricted = (int(saved[name][i]) for name in ('accepted', 'forced', 'unrestricted'))
        result, transition = output(selected)
        output(forced)
        output(unrestricted)
        positive = gains[i] > EPSILON
        qualified = lost == 0 if mode in GUARDED else np.ones(len(choices), bool)
        if not qualified[selected] or not qualified[forced]:
            raise ValueError('selected choice violates the original-support constraint')
        delta = result['edge_net']
        simultaneous = (bool(protected[selected]) and delta['tp_same']+delta['tp_cross'] > 0
                        and delta['fp_same']+delta['fp_cross'] < 0)
        modes[mode] = dict(
            selected=result, choice=selected, score_gain=float(gains[i, selected]),
            same_as_old=bool(selected == saved['accepted'][0]),
            joint=bool(joint[selected]), protected=bool(protected[selected]),
            positive_protected=int((positive & protected).sum()),
            eligible_positive_protected=int((positive & protected & qualified).sum()),
            nonws_top_protected=bool(protected[forced]), nonws_top_unrestricted_protected=bool(protected[unrestricted]),
            best_protected_rank=positive_rank(gains[i], protected),
            uncovered_delta=int(transition[:, 0].sum()-transition[0].sum()),
            lost_supported_node_queries=int(transition[1:, 0].sum()),
            recovered_node_queries=int(transition[0, 1:].sum()),
            two_to_one_source_node_queries=int(transition[2, 1]),
            simultaneous_protected=simultaneous, strict_simultaneous_protected=simultaneous and bool(lost[selected] == 0),
        )
    baseline, _ = output(0)
    return dict(candidates=len(choices), protected_candidates=int(protected.sum()),
                strict_protected_candidates=int((protected & (lost == 0)).sum()),
                strict_candidates=int((lost == 0).sum()),
                baseline=baseline, modes=modes, independent_outputs=len(checked))


def summarize_held(rows):
    result = summarize(rows, modes=MODES)
    for arm in ARMS:
        entries = [row['arms'][arm] for row in rows]
        totals = result['arms'][arm]
        for name in ('strict_protected_candidates', 'strict_candidates', 'independent_outputs'):
            totals[name] = sum(r[name] for r in entries)
        totals['strict_opportunity_scenes'] = sum(r['strict_protected_candidates'] > 0 for r in entries)
        for mode in MODES:
            choices = [r['modes'][mode] for r in entries]
            for key in ('lost_supported_node_queries', 'recovered_node_queries', 'two_to_one_source_node_queries',
                        'simultaneous_protected', 'strict_simultaneous_protected'):
                totals['modes'][mode][key] = sum(q[key] for q in choices)
        totals['strict_split_protection_gain_scenes'] = sum(
            r['modes']['REL_SPLIT_KEEP_Q']['protected'] and not r['modes']['ABS_KEEP_Q']['protected'] for r in entries)
        totals['strict_split_protection_loss_scenes'] = sum(
            r['modes']['ABS_KEEP_Q']['protected'] and not r['modes']['REL_SPLIT_KEEP_Q']['protected'] for r in entries)
    evidence = {}
    for name, source in rows[0]['edge_qualification'].items():
        group = [row['edge_qualification'][name] for row in rows]
        metrics = {}
        for key in source['auc']:
            values = [r['auc'][key] for r in group if r['auc'][key] is not None]
            metrics[key] = dict(scenes=len(values), scene_equal_mean=float(np.mean(values)) if values else None)
        evidence[name] = dict(pairs=sum(r['pairs'] for r in group), positives=sum(r['positives'] for r in group),
                              auc=metrics)
    return dict(result, edge_qualification=evidence)


def run(runs, output):
    if output.parent != runs:
        raise ValueError('independent output must use the existing runs directory')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    start = time.monotonic()
    state = dict(status='RUNNING', split='T_core', completed_scenes=0, stages=[], argv=sys.argv,
                 inputs={k: str(runs/v) for k, v in PARENTS.items()}, time_policy='advisory',
                 started_at=datetime.now(timezone.utc).isoformat(), affinity=sorted(os.sched_getaffinity(0)),
                 optimizer_updates=0, gpu_used=False, reference_read=False)

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        checkpoint(phase='blind', scene_ids=ids)
        metadata = {}
        for sid in ids:
            nodes, sources, support, pairs, scores, ws, native = load_scene(runs, sid)
            candidates, proposal = propose(ws, pairs, scores, native)
            motion, fields, meta = edge_scores(pairs, sources, scores, native)
            edge = asymmetric_weights(ws, pairs, sources, motion)
            weights = dense_edges(pairs, edge[0], len(ws))
            pair_index = dense_edges(pairs, np.arange(len(pairs)), len(ws))
            mutable = np.zeros(len(pairs), bool)
            frozen = {arm: catalog(ws, candidates[arm], weights) for arm in ARMS}
            arrays = {}
            for arm in ARMS:
                local = score_catalog(frozen[arm], ws, support, edge, pair_index, mutable)
                arrays.update({arm+'_'+name: value for name, value in local.items()})
            write(private/f'catalogue-{sid}.json.gz', frozen)
            np.savez_compressed(private/f'scores-{sid}.npz', edge=edge, mutable=mutable, **fields, **arrays)
            metadata[sid] = dict(proposal=proposal, motion=meta, nodes=len(nodes), pairs=len(pairs),
                                 native_events=len(native[1]), ws_groups=len(group_members(ws)))
        population = {k: sum(meta[k] for meta in metadata.values())
                      for k in ('nodes', 'pairs', 'native_events', 'ws_groups')}
        if tuple(population.values()) != (461, 13151, 116539, 177):
            raise ValueError('preflight T18 population counts changed')
        state['stages'].append('all_choices_and_support_eligibility_frozen_before_identity')
        checkpoint(phase='truth_evaluation', frozen_scenes=len(ids), population=population)
        rows = []
        for sid in ids:
            nodes, sources, support, pairs, _, ws, _ = load_scene(runs, sid)
            identities = np.unique(load_identities(runs, sid, nodes), return_inverse=True)[1]
            before = ws[pairs[:, 0]] == ws[pairs[:, 1]]
            records = read(private/f'catalogue-{sid}.json.gz')
            with np.load(private/f'scores-{sid}.npz', allow_pickle=False) as cache:
                arms = {}
                for arm in ARMS:
                    keys = ('gains', 'lost', 'accepted', 'forced', 'unrestricted')
                    saved = {k: cache[arm+'_'+k] for k in keys}
                    arms[arm] = evaluate_arm(records[arm], saved, ws, identities, sources, support, pairs, cache['edge'])
                qualification = qualify_edges(cache, pairs, identities, before)
            rows.append(dict(scene_id=sid, component_id=components[sid], arms=arms,
                             edge_qualification=qualification, **metadata[sid]))
            checkpoint(completed_scenes=len(rows))
        groups = {key: [r for r in rows if r['component_id'] == key] for key in sorted(set(components.values()))}
        largest = min(groups, key=lambda c: (-sum(r['nodes'] for r in groups[c]), c))
        write(output/'scenes.json', rows)
        write(output/'summary.json', dict(primary=('NATIVE_TREE', 'REL_SPLIT_KEEP_Q', 'ABS_KEEP_Q'),
            population=population, all_scenes=summarize_held(rows), largest_component=largest,
            by_component={key: summarize_held(group) for key, group in groups.items()},
            without_largest=summarize_held([r for r in rows if r['component_id'] != largest])))
        state['stages'].append('full_edge_and_node_query_support_recount')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, finished_at=datetime.now(timezone.utc).isoformat())
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
