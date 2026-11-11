"""C80: rerank the exact two frozen C78 pools without using identities."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.evaluation.association import _aggregate
from tools.analysis.disjoint_partition_evidence import MODES, partition_scores, scene_evidence, score_catalog
from tools.analysis.disjoint_partition_pilot import (
    ARMS, EDGE_AXES, SCOPES, materialize, qualify_catalog, selected_output, support_state,
)
from tools.analysis.same_source_swap_pilot import (
    EPSILON, MAX_AS, PARENTS, group_members, load_identities, load_population, load_scene, peak_rss_bytes, read, write,
)

PRIMARY = 'NODE_TAIL'
SELECTIONS = ('accepted', 'forced', 'forced_unrestricted')
HISTORICAL_POOL = 'LEVEL1-DISJOINT-PARTITIONS-20261111-attempt-1'
HISTORY_STAGES = ['all_real_single_pair_candidates_and_old_J_choices_frozen_before_identity',
                  'signed_transition_anchor_and_complete_pair_recount']


def bind_pool(runs, root, ids):
    old = read(root/'run.json')
    if (root.name != HISTORICAL_POOL or old['status'] != 'COMPLETED' or old['exit_code'] != 0
            or old['split'] != 'T_core' or old['scene_ids'] != ids
            or old['completed_scenes'] != len(ids) or old['stages'] != HISTORY_STAGES
            or old['inputs'] != {name: str(runs/path) for name, path in PARENTS.items()}
            or {path.name for path in (root/'private').glob('candidates-*.json.gz')}
            != {f'candidates-{sid}.json.gz' for sid in ids}):
        raise ValueError('original completed C78 pool and same input population required')


def save_scores(path, records):
    data = {}
    for arm, record in records.items():
        data[arm+'_gains'] = np.asarray([record['gains'][mode] for mode in MODES])
        data[arm+'_uncovered_delta'] = np.asarray(record['uncovered_delta'], np.int64)
        for key in SELECTIONS:
            data[arm+'_'+key] = np.asarray([record[key][mode] for mode in MODES], np.int64)
    np.savez_compressed(path, **data)


def positive_rank(gains, mask):
    feasible = np.flatnonzero(mask)
    if not len(feasible):
        return None
    best = feasible[np.argmax(gains[feasible])]
    ranked_ahead = (gains[1:] > gains[best]) | ((gains[1:] == gains[best]) & (np.arange(1, len(gains)) < best))
    return int(ranked_ahead.sum())+1


def evaluate_arm(arm, candidate, scored, historical, membership, identities, sources, support, pairs, evidence):
    actions, choices, anchors, _, counts, joint, protected, _ = qualify_catalog(
        candidate, membership, identities, sources)
    old_gain = scored[arm+'_gains']
    uncovered = scored[arm+'_uncovered_delta']
    old_choice = int(scored[arm+'_accepted'][0, 1])
    if (old_gain.shape != (len(MODES), len(choices)) or uncovered.shape != (len(choices),)
            or old_choice != candidate['selected_J'][1]
            or len(choices) != historical['candidates'] or int(joint.sum()) != historical['joint_candidates']
            or int(protected.sum()) != historical['protected_candidates']):
        raise ValueError('C78 original pool and qualification counts differ')
    before = membership[pairs[:, 0]] == membership[pairs[:, 1]]
    source_before = support_state(membership, support, sources)
    baseline_uncovered = int(np.count_nonzero(source_before[0] == 0))
    base_scores = partition_scores(membership, evidence)
    checked = {}

    def output(index):
        if index not in checked:
            change = materialize(membership, (actions[i] for i in choices[index]))
            whole = partition_scores(change, evidence)
            np.testing.assert_allclose([whole[mode]-base_scores[mode] for mode in MODES], old_gain[:, index],
                                       rtol=1e-9, atol=1e-9, err_msg='whole-partition score delta differs')
            candidate_uncovered = int(np.count_nonzero(support_state(change, support, sources)[0] == 0))
            if candidate_uncovered-baseline_uncovered != uncovered[index]:
                raise ValueError('full node×Q uncovered count differs from disjoint composition')
            checked[index] = selected_output(index, choices, actions, counts, candidate, membership, identities,
                                             sources, support, pairs, evidence['weights']['OLD_J'], before,
                                             anchors, source_before)
        return checked[index]

    baseline = output(0)
    old_output = output(old_choice)
    if old_output != historical['J'] or bool(protected[old_choice]) != historical['J_protected']:
        raise ValueError('complete old J selection result differs from published C78')
    result = {}
    for mi, mode in enumerate(MODES):
        gains = old_gain[mi]
        accepted, forced, unrestricted = (int(scored[arm+'_'+key][mi, 1]) for key in SELECTIONS)
        any_positive = gains > EPSILON
        support_eligible = uncovered <= 0 if mode == 'NODE_BALANCED_SUPPORT' else np.ones(len(choices), bool)
        predicted = output(accepted)
        result[mode] = dict(
            selected=predicted, choice=accepted, score_gain=float(gains[accepted]),
            same_as_old=accepted == old_choice, joint=bool(joint[accepted]), protected=bool(protected[accepted]),
            positive_choices=int(any_positive.sum()), positive_protected=int((protected & any_positive).sum()),
            eligible_positive_protected=int((protected & any_positive & support_eligible).sum()),
            nonws_top_protected=bool(protected[forced]), nonws_top_unrestricted_protected=bool(protected[unrestricted]),
            nonws_top_score=float(gains[forced]) if forced else None,
            best_protected_rank=positive_rank(gains, protected), uncovered_delta=int(uncovered[accepted]),
        )
        output(forced)
        output(unrestricted)
    return dict(candidates=len(choices), joint_candidates=int(joint.sum()), protected_candidates=int(protected.sum()),
                baseline=baseline, modes=result, independently_checked_outputs=len(checked))


def combine_outputs(items):
    return dict(
        metrics={s: _aggregate([row['metrics'][s] for row in items]) for s in SCOPES},
        **{edge: {key: sum(row[edge][key] for row in items) for key in EDGE_AXES}
           for edge in ('edge_added', 'edge_removed', 'edge_net')},
        anchors={key: sum(row['anchors'][key] for row in items)
                 for key in ('lost', 'recovered', 'represented', 'missed', 'duplicates', 'groups')},
        support_transition=np.asarray([row['node_query_source_support_transition'] for row in items]).sum(axis=0).tolist(),
    )


def summarize(rows):
    result = dict(scene_count=len(rows), components=len({r['component_id'] for r in rows}), arms={})
    for arm in ARMS:
        entries = [r['arms'][arm] for r in rows]
        result['arms'][arm] = dict(candidates=sum(r['candidates'] for r in entries),
                                   opportunity_scenes=sum(r['protected_candidates'] > 0 for r in entries),
                                   protected_candidates=sum(r['protected_candidates'] for r in entries),
                                   baseline=combine_outputs([r['baseline'] for r in entries]), modes={})
        for mode in MODES:
            details = [r['modes'][mode] for r in entries]
            selected = [r['selected'] for r in details]
            axis_by_scene = np.asarray([[r['edge_net'][a] for a in EDGE_AXES] for r in selected])
            harmful = (axis_by_scene[:, :2] < 0).any(axis=1) | (axis_by_scene[:, 2:] > 0).any(axis=1)
            ranks = [r['best_protected_rank'] for r in details if r['best_protected_rank'] is not None]
            old = [r['modes']['OLD_J'] for r in entries]
            flags = [r['protected'] for r in details]
            result['arms'][arm]['modes'][mode] = dict(
                accepted_scenes=sum(r['choice'] > 0 for r in details), two_edits=sum(r['edits'] == 2 for r in selected),
                protected_scenes=sum(flags), joint_scenes=sum(r['joint'] for r in details),
                protected_gain_vs_old=sum(a and not b['protected'] for a,b in zip(flags,old,strict=True)),
                protected_loss_vs_old=sum(not a and b['protected'] for a,b in zip(flags,old,strict=True)),
                same_choice_as_old=sum(r['same_as_old'] for r in details),
                edge_harm_scenes=int(harmful.sum()), identity_loss_scenes=sum(r['anchors']['lost'] > 0 for r in selected),
                duplicate_increase_scenes=sum(a['anchors']['duplicates'] > b['baseline']['anchors']['duplicates']
                                              for a,b in zip(selected,entries,strict=True)),
                all_protected_nonpositive_scenes=sum(r['protected_candidates'] > 0 and q['positive_protected'] == 0
                                                     for r,q in zip(entries,details,strict=True)),
                all_positive_protected_support_blocked_scenes=sum(
                    q['positive_protected'] > 0 and q['eligible_positive_protected'] == 0 for q in details),
                positive_protected_outscored_scenes=sum(q['eligible_positive_protected'] > 0 and not q['protected']
                                                        for q in details),
                forced_top_protected_scenes=sum(r['nonws_top_protected'] for r in details),
                forced_unrestricted_protected_scenes=sum(r['nonws_top_unrestricted_protected'] for r in details),
                best_protected_rank_median=float(np.median(ranks)) if ranks else None,
                max_scene_original_true_edges_removed=max(r['edge_removed']['tp_same']+r['edge_removed']['tp_cross']
                                                          for r in selected),
                max_scene_new_false_edges_added=max(r['edge_added']['fp_same']+r['edge_added']['fp_cross']
                                                    for r in selected),
                uncovered_delta=sum(r['uncovered_delta'] for r in details),
                selected=combine_outputs(selected),
            )
    return result


def run(runs, c78, output):
    if output == runs or output == c78 or c78 in output.parents or output in c78.parents:
        raise ValueError('output must be independent of frozen inputs')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    start = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, runs=str(runs), c78=str(c78), output=str(output),
                 time_policy='advisory', optimizer_updates=0, gpu_used=False, split='T_core',
                 stages=[], completed_scenes=0, affinity=sorted(os.sched_getaffinity(0)),
                 started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        bind_pool(runs, c78, ids)
        checkpoint(phase='blind_scores', scene_ids=ids)
        metadata = []
        for sid in ids:
            _, sources, support, pairs, scores, starts = load_scene(runs, sid)
            membership = starts['WS']
            orig = read(c78/'private'/f'candidates-{sid}.json.gz')['arms']
            evidence, meta = scene_evidence(pairs, scores, sources)
            records = {arm: score_catalog(orig[arm], membership, group_members(membership), support, evidence)
                       for arm in ARMS}
            save_scores(private/f'rerank-{sid}.npz', records)
            metadata.append(meta)
        state['stages'].append('every_frozen_c78_candidate_scored_and_selections_saved_before_identity')
        checkpoint(phase='truth_evaluation', frozen_scenes=len(metadata))
        original_rows = read(c78/'scenes.json')
        if [r['scene_id'] for r in original_rows] != ids:
            raise ValueError('published C78 scene order changed')
        rows = []
        for sid, prior, meta in zip(ids, original_rows, metadata, strict=True):
            nodes, sources, support, pairs, scores, starts = load_scene(runs, sid)
            membership = starts['WS']
            orig = read(c78/'private'/f'candidates-{sid}.json.gz')['arms']
            identities = np.unique(load_identities(runs, sid, nodes), return_inverse=True)[1]
            evidence, _ = scene_evidence(pairs, scores, sources)
            with np.load(private/f'rerank-{sid}.npz', allow_pickle=False) as scored:
                arms = {arm: evaluate_arm(arm, orig[arm], scored, prior['arms'][arm]['2'], membership,
                                          identities, sources, support, pairs, evidence) for arm in ARMS}
            rows.append(dict(scene_id=sid, component_id=components[sid], nodes=len(nodes), evidence=meta, arms=arms))
            checkpoint(completed_scenes=len(rows))
        by_component = {c: [r for r in rows if r['component_id'] == c] for c in sorted(set(components.values()))}
        largest = min(by_component, key=lambda c: (-sum(r['nodes'] for r in by_component[c]), c))
        if sum(r['nodes'] for r in rows) != 2310 or largest != read(c78/'summary.json')['largest_component']:
            raise ValueError('published C78 node population changed')
        all_results = summarize(rows)
        for arm, count in zip(ARMS, (101757, 124523), strict=True):
            if all_results['arms'][arm]['candidates'] != count or all_results['arms'][arm]['opportunity_scenes'] != 20:
                raise ValueError('published C78 candidate or opportunity population changed')
        total_checks = sum(r['arms'][a]['independently_checked_outputs'] for r in rows for a in ARMS)
        write(output/'scenes.json', rows)
        write(output/'summary.json', dict(primary=PRIMARY, all_scenes=all_results, largest_component=largest,
            components={k: summarize(group) for k,group in by_component.items()},
            without_largest=summarize([r for r in rows if r['component_id'] != largest]),
            independently_recomputed_selected_or_forced=total_checks,
            edge_population='complete original source-stratified pairs, no author qualification subgraph',
            scope='original repeatedly used T32, fixed C78 catalogue, max two disjoint edits'))
        state['stages'].append('old_j_original_truth_counts_and_whole_partition_scores_verified')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, independently_recomputed=total_checks,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


def self_check():
    from tempfile import TemporaryDirectory
    from tools.analysis.disjoint_partition_pilot import Action, catalog, evaluate_catalog
    from tools.analysis.same_source_swap_pilot import THRESHOLD

    membership = np.repeat(np.arange(4), 2)
    pairs = np.column_stack(np.triu_indices(len(membership), 1))
    sources = np.tile(['x','y'], 4)
    scores = THRESHOLD + np.where(pairs[:, 0]//4 == pairs[:, 1]//4, .5, -.5)
    identities = np.repeat([0,1], 4)
    support = np.ones((8,64), bool)
    actions = (Action((0,1), ((0,1,2,3),), 'merge'), Action((2,3), ((4,5,6,7),), 'merge'))
    evidence, _ = scene_evidence(pairs, scores, sources)
    weights = evidence['weights']['OLD_J']
    record = catalog(membership, actions, weights)
    historical, _ = evaluate_catalog(record, membership, identities, sources, support, pairs, weights)
    result = score_catalog(record, membership, group_members(membership), support, evidence)
    with TemporaryDirectory() as folder:
        path = Path(folder)/'rank.npz'
        save_scores(path, {arm: result for arm in ARMS})
        with np.load(path, allow_pickle=False) as scored:
            a = {arm: evaluate_arm(arm, record, scored, historical['2'], membership, identities,
                                   sources, support, pairs, evidence) for arm in ARMS}
        aggregate = summarize([dict(component_id='synthetic', arms=a)])
        assert aggregate['arms']['SCORE_TREE']['modes'][PRIMARY]['protected_scenes'] == 1
        out = Path(folder)/'summary.json'
        write(out, aggregate)
        assert read(out) == aggregate
    print('C80 self-check passed: fixed catalogue, old parity, full objective, protected outcomes and serialization.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'c78', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--time-policy', choices=('advisory',))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.runs is not None and args.c78 is not None and args.output is not None and args.time_policy == 'advisory':
        run(args.runs.resolve(), args.c78.resolve(), args.output.resolve())
    else:
        parser.error('--runs, --c78, --output and --time-policy advisory are required')
