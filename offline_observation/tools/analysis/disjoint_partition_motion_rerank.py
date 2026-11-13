"""C82: score frozen C78 partitions with separate native motion evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from tools.analysis.conditional_trajectory_pilot import load_native
from tools.analysis.disjoint_partition_motion_evidence import competition_weights, scene_motion_evidence
from tools.analysis.disjoint_partition_pilot import (
    ARMS, EDGE_AXES, Action, changed_edges, materialize, qualify_catalog, selected_output, support_state,
)
from tools.analysis.disjoint_partition_rerank import bind_pool, positive_rank, summarize
from tools.analysis.same_source_swap_pilot import (
    EPSILON, MAX_AS, THRESHOLD, dense_edges, group_members, load_identities, load_population, load_scene,
    peak_rss_bytes, read, write,
)

MODES = ('OLD_J', 'ABS', 'SAME_OLS', 'CROSS_REL', 'BOTH')
MOTION_FIELDS = {'same': ('same_hold_m', 'same_secant_m', 'same_ols_m'),
                 'cross': ('cross_abs_m', 'cross_relative_mps')}


def edge_scores(pairs, sources, scores, native):
    fields, meta = scene_motion_evidence(pairs, sources, native)
    margin = np.asarray(scores, np.float64) - THRESHOLD
    nonzero = np.abs(margin[margin != 0])
    scale = float(np.median(nonzero)) if len(nonzero) else 1.
    weights = {}
    for relation, names in MOTION_FIELDS.items():
        for name in names:
            if not np.array_equal(np.isfinite(fields[name]), np.isfinite(fields[names[0]])):
                raise ValueError('motion contrasts must use identical pair support')
            if name != 'same_secant_m':
                weights[name], meta[name+'_competition'] = competition_weights(
                    pairs, sources, native, fields[name], relation)
    hold, ols, absolute, relative = (
        weights[name] for name in ('same_hold_m', 'same_ols_m', 'cross_abs_m', 'cross_relative_mps'))
    edge = np.asarray([margin, margin+scale*(hold+absolute), margin+scale*(ols+absolute),
                       margin+scale*(hold+relative), margin+scale*(ols+relative)])
    if not np.isfinite(edge).all():
        raise ValueError('nonfinite combined score')
    return edge, fields, dict(meta, scale=scale)


def score_arm(record, membership, edge, pair_index, mutable):
    groups = group_members(membership)
    edits = [Action(tuple(s), tuple(tuple(p) for p in parts), family)
             for s, parts, family in record['actions']]
    primitive = np.zeros((len(MODES), len(edits)+1))
    for i, action in enumerate(edits):
        left, right, added, removed = changed_edges(action, membership, groups)
        add_idx = pair_index[left[added], right[added]]
        rem_idx = pair_index[left[removed], right[removed]]
        mutable[add_idx] = True
        mutable[rem_idx] = True
        primitive[:, i] = edge[:, add_idx].sum(axis=1)-edge[:, rem_idx].sum(axis=1)
    np.testing.assert_allclose(primitive[0, :-1], record['primitive_J_gain'], atol=1e-9, rtol=1e-10)
    indices = np.full((len(record['choices']), 2), len(edits), dtype=np.int64)
    for i, choice in enumerate(record['choices']):
        indices[i, :len(choice)] = choice
    gains = primitive[:, indices].sum(axis=2)
    np.testing.assert_allclose(gains[0], record['J_gain'], atol=1e-9, rtol=1e-10)
    # Preserve the historical arithmetic and tie order exactly.
    gains[0] = record['J_gain']
    accepted = np.argmax(gains, axis=1)
    accepted = np.where(gains[np.arange(len(MODES)), accepted] > EPSILON, accepted, 0)
    forced = np.argmax(gains[:, 1:], axis=1)+1 if gains.shape[1] > 1 else np.zeros(len(MODES), int)
    if accepted[0] != record['selected_J'][1]:
        raise ValueError('historical old J choice differs')
    return gains, accepted, forced


def evaluate_arm(record, historical, scored, arm, membership, identities, sources, support, pairs):
    actions, choices, anchors, _, counts, joint, protected, _ = qualify_catalog(
        record, membership, identities, sources)
    if (len(choices), int(joint.sum()), int(protected.sum())) != (
            historical['candidates'], historical['joint_candidates'], historical['protected_candidates']):
        raise ValueError('historical C78 qualification counts differ')
    edge = scored['edge']
    old_weights = dense_edges(pairs, edge[0], len(membership))
    before = membership[pairs[:, 0]] == membership[pairs[:, 1]]
    base_support = support_state(membership, support, sources)
    gains, accepted, forced = (scored[arm+'_'+name] for name in ('gains', 'accepted', 'forced'))
    checked = {}

    def output(index):
        if index not in checked:
            state = materialize(membership, (actions[i] for i in choices[index]))
            after = state[pairs[:, 0]] == state[pairs[:, 1]]
            full_delta = edge[:, after & ~before].sum(axis=1)-edge[:, before & ~after].sum(axis=1)
            np.testing.assert_allclose(full_delta, gains[:, index], atol=1e-9, rtol=1e-10)
            checked[index] = selected_output(index, choices, actions, counts, record, membership, identities,
                                             sources, support, pairs, old_weights, before, anchors, base_support)
        return checked[index]

    baseline = output(0)
    if output(accepted[0]) != historical['J'] or bool(protected[accepted[0]]) != historical['J_protected']:
        raise ValueError('full old J output differs from C78')
    result = {}
    for i, mode in enumerate(MODES):
        index, nonws = int(accepted[i]), int(forced[i])
        positive = gains[i] > EPSILON
        selected = output(index)
        output(nonws)
        result[mode] = dict(
            selected=selected, choice=index, score_gain=float(gains[i, index]), same_as_old=bool(index == accepted[0]),
            joint=bool(joint[index]), protected=bool(protected[index]),
            positive_choices=int(positive.sum()), positive_protected=int((protected & positive).sum()),
            eligible_positive_protected=int((protected & positive).sum()),
            nonws_top_protected=bool(protected[nonws]), nonws_top_unrestricted_protected=bool(protected[nonws]),
            nonws_top_score=float(gains[i, nonws]) if nonws else None,
            best_protected_rank=positive_rank(gains[i], protected),
        )
        transition = np.asarray(selected['node_query_source_support_transition'])
        result[mode]['uncovered_delta'] = int(transition[:, 0].sum()-transition[0].sum())
    return dict(candidates=len(choices), joint_candidates=int(joint.sum()), protected_candidates=int(protected.sum()),
                baseline=baseline, modes=result, independently_checked_outputs=len(checked))


def auc(scores, truth):
    positives = int(truth.sum())
    negatives = len(truth)-positives
    if not positives or not negatives:
        return None
    _, inverse, sizes = np.unique(scores, return_inverse=True, return_counts=True)
    true = np.bincount(inverse[truth], minlength=len(sizes))
    false = sizes-true
    less = np.cumsum(false)-false
    return float(np.sum(true*(less+.5*false))/(positives*negatives))


def qualify_edges(scored, pairs, identities, before):
    truth = identities[pairs[:, 0]] == identities[pairs[:, 1]]
    mutable = scored['mutable']
    rows = {}
    for relation, fields in MOTION_FIELDS.items():
        supported = np.isfinite(scored[fields[0]])
        for name, direction in (('all', mutable), ('add', mutable & ~before), ('remove', mutable & before)):
            mask = supported & direction
            values = {'OLD_J': scored['edge'][0, mask]}
            values.update({field: -scored[field][mask] for field in fields})
            rows[f'{relation}_{name}'] = dict(pairs=int(mask.sum()), positives=int(truth[mask].sum()),
                                              auc={k: auc(v, truth[mask]) for k, v in values.items()})
    return rows


def aggregate(rows):
    result = summarize(rows, modes=MODES)
    stats = {}
    for name, source in rows[0]['qualification'].items():
        entries = [row['qualification'][name] for row in rows]
        metrics = {}
        for key in source['auc']:
            defined = [entry['auc'][key] for entry in entries if entry['auc'][key] is not None]
            metrics[key] = dict(scene_equal_mean=float(np.mean(defined)) if defined else None, scenes=len(defined))
        stats[name] = dict(pairs=sum(row['pairs'] for row in entries),
                           positives=sum(row['positives'] for row in entries), auc=metrics)
    return dict(result, motion_edge_qualification=stats)


def bind_frozen(root, runs, c78, ids):
    original = read(root/'run.json')
    if (root.parent != runs or root.name != 'LEVEL1-DISJOINT-MOTION-EVIDENCE-20261112-attempt-1'
            or original['status'] != 'FAILED' or original['exit_code'] != 1
            or original['error'] != 'TypeError: Object of type bool_ is not JSON serializable'
            or original['output'] != str(root) or original['runs'] != str(runs) or original['c78'] != str(c78)
            or original['split'] != 'T_core' or original['scene_ids'] != ids
            or original['frozen_scenes'] != len(ids) or original['completed_scenes'] != len(ids)
            or original['stages'] != ['all_motion_scores_and_candidate_choices_frozen_before_identity']
            or {f.name for f in (root/'private').glob('motion-*.npz')} != {f'motion-{sid}.npz' for sid in ids}):
        raise ValueError('original C82 publication failure with complete immutable scores required')
    return root/'private'


def saved_metadata(scored):
    return {
        'source': 'original frozen score arrays',
        'finite_counts': {name: int(np.isfinite(scored[name]).sum())
                          for fields in MOTION_FIELDS.values() for name in fields},
        'changed_weights_vs_old': {mode: int(np.count_nonzero(scored['edge'][i] != scored['edge'][0]))
                                  for i, mode in enumerate(MODES) if i},
    }


def run(runs, c78, output, frozen=None):
    if output.parent != runs or output in (c78, frozen):
        raise ValueError('new independent output must be under the existing runs directory')
    output.mkdir(exist_ok=False)
    private = output/'private'
    if frozen is None:
        private.mkdir()
    start = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, runs=str(runs), c78=str(c78), output=str(output),
                 time_policy='advisory', optimizer_updates=0, gpu_used=False, split='T_core',
                 stages=[], completed_scenes=0, affinity=sorted(os.sched_getaffinity(0)),
                 frozen_from=str(frozen) if frozen is not None else None,
                 started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-start, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components = load_population(runs)
        bind_pool(runs, c78, ids)
        if frozen is not None:
            private = bind_frozen(frozen, runs, c78, ids)
            metadata = [None] * len(ids)
            state['stages'].append('bind_original_pre_identity_scores_without_rescoring')
        else:
            metadata = []
            checkpoint(phase='blind_scores', scene_ids=ids)
            for sid in ids:
                nodes, sources, _, pairs, scores, starts = load_scene(runs, sid)
                edge, fields, meta = edge_scores(pairs, sources, scores, load_native(runs, sid, nodes, sources))
                mutable = np.zeros(len(pairs), bool)
                lookup = dense_edges(pairs, np.arange(len(pairs)), len(nodes))
                saved = dict(edge=edge, mutable=mutable, **fields)
                originals = read(c78/'private'/f'candidates-{sid}.json.gz')['arms']
                for arm in ARMS:
                    result = score_arm(originals[arm], starts['WS'], edge, lookup, mutable)
                    saved.update({arm+'_'+key: value for key, value in zip(('gains', 'accepted', 'forced'), result,
                                                                           strict=True)})
                np.savez_compressed(private/f'motion-{sid}.npz', **saved)
                metadata.append(meta)
            state['stages'].append('all_motion_scores_and_candidate_choices_frozen_before_identity')
        checkpoint(phase='truth_evaluation', scene_ids=ids, frozen_scenes=len(metadata))
        historical = read(c78/'scenes.json')
        if [row['scene_id'] for row in historical] != ids:
            raise ValueError('C78 scene order differs')
        rows = []
        for sid, prior, meta in zip(ids, historical, metadata, strict=True):
            nodes, sources, support, pairs, _, starts = load_scene(runs, sid)
            membership = starts['WS']
            before = membership[pairs[:, 0]] == membership[pairs[:, 1]]
            identities = np.unique(load_identities(runs, sid, nodes), return_inverse=True)[1]
            originals = read(c78/'private'/f'candidates-{sid}.json.gz')['arms']
            with np.load(private/f'motion-{sid}.npz', allow_pickle=False) as cache:
                scored = {key: cache[key] for key in cache.files}
            if meta is None:
                meta = saved_metadata(scored)
            arms = {arm: evaluate_arm(originals[arm], prior['arms'][arm]['2'], scored, arm, membership, identities,
                                      sources, support, pairs) for arm in ARMS}
            rows.append(dict(scene_id=sid, component_id=components[sid], nodes=len(nodes), evidence=meta,
                             qualification=qualify_edges(scored, pairs, identities, before), arms=arms))
            checkpoint(completed_scenes=len(rows))
        by_component = {c: [r for r in rows if r['component_id'] == c] for c in sorted(set(components.values()))}
        largest = min(by_component, key=lambda c: (-sum(r['nodes'] for r in by_component[c]), c))
        if len(by_component) != 12 or sum(r['nodes'] for r in rows) != 2310:
            raise ValueError('fixed T32 component or node population differs')
        if largest != read(c78/'summary.json')['largest_component']:
            raise ValueError('fixed C78 largest component differs')
        all_results = aggregate(rows)
        for arm, count in zip(ARMS, (101757, 124523), strict=True):
            arm_totals = all_results['arms'][arm]
            if arm_totals['candidates'] != count or arm_totals['opportunity_scenes'] != 20:
                raise ValueError('fixed C78 candidate or protected opportunity population differs')
        checks = sum(r['arms'][a]['independently_checked_outputs'] for r in rows for a in ARMS)
        write(output/'scenes.json', rows)
        write(output/'summary.json', dict(primary='BOTH', all_scenes=all_results, largest_component=largest,
            components={k: aggregate(group) for k, group in by_component.items()},
            without_largest=aggregate([r for r in rows if r['component_id'] != largest]),
            independently_recomputed_selected_or_forced=checks,
            edge_population='complete source-stratified pair graph; no author qualification subgraph',
            qualification_population='distinct edges that can be flipped in either original C78 pool on common support'))
        state['stages'].append('old_j_full_recount_and_supported_edge_qualification')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, independently_recomputed=checks,
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        checkpoint(status='FAILED', exit_code=1, error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'c78', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.c78.resolve(), args.output.resolve(),
        args.frozen.resolve() if args.frozen is not None else None)
