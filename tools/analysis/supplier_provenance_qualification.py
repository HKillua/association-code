"""C87: classify borrowed support on held components and replay frozen C85 edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from tools.analysis import asymmetric_partition_validation as c85
from tools.analysis import held_partition_population as held
from tools.analysis import same_source_swap_pilot as original
from tools.analysis.conditional_trajectory_pilot import load_native
from tools.analysis.disjoint_partition_evidence import _choices, _select
from tools.analysis.disjoint_partition_motion_rerank import auc, bind_frozen
from tools.analysis.disjoint_partition_pilot import (
    Action, EDGE_AXES, materialize, qualify_catalog, selected_output, support_state,
)
from tools.analysis.supplier_external_geometry import external_geometry

C78 = 'LEVEL1-DISJOINT-PARTITIONS-20261111-attempt-1'
C82 = 'LEVEL1-DISJOINT-MOTION-EVIDENCE-20261112-attempt-1'
C85 = 'LEVEL1-ASYMMETRIC-LINKS-20261112-attempt-1'
FEATURES = ('J_FULL', 'J_COMMON', 'NATIVE_ABS', 'NATIVE_REL')
CONTEXT = ('RAW', 'TRAN', 'GEO')
REPLAY = ('NO_GATE', 'STRICT_Q', 'KEEP_ORIGINAL', *FEATURES)
RELATIONS = ('same', 'cross')
MAX_TRUE_REJECT_PERCENT = 5
MIN_TRUE_CALIBRATION = 20


def scene_inputs(runs, sid, training):
    if training:
        nodes, sources, support, pairs, scores, memberships = original.load_scene(runs, sid)
        return nodes, sources, support, pairs, scores, memberships['WS'], load_native(runs, sid, nodes, sources)
    return held.load_scene(runs, sid)


def read_truth(runs, sid, nodes, training):
    load = original.load_identities if training else held.load_identities
    return np.unique(load(runs, sid, nodes), return_inverse=True)[1]


def freeze_scene(scene, archive):
    nodes, sources, support, pairs, scores, ws, native = scene
    left, right = pairs.T
    grouped = np.flatnonzero(ws[left] == ws[right])
    exposure = np.count_nonzero(support[left[grouped]] ^ support[right[grouped]], axis=1)
    chosen = exposure > 0
    rows, exposure = grouped[chosen], exposure[chosen]
    supplier_pairs = pairs[rows]
    same = sources[left[rows]] == sources[right[rows]]
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['edge'][0], scores-original.THRESHOLD)
        j = saved['edge'][0, rows]
        absolute = -np.where(same, saved['same_hold_m'][rows], saved['cross_abs_m'][rows])
        relative = -np.where(same, saved['same_hold_m'][rows], saved['cross_relative_mps'][rows])
    common = np.isfinite(absolute)
    np.testing.assert_array_equal(common, np.isfinite(relative))
    geo = external_geometry(pairs, rows, sources, ws, scores-original.THRESHOLD, native)
    np.testing.assert_array_equal(geo['observed'], (~same) & common)
    return dict(nodes=nodes, sources=sources, support=support, ws=ws, pairs=pairs, scores=scores,
                supplier_pairs=supplier_pairs, same=same, exposure=exposure,
                J_FULL=j, J_COMMON=np.where(common, j, np.nan), NATIVE_ABS=absolute, NATIVE_REL=relative,
                **geo)


def read_scene(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def pair_truth(data, identities):
    left, right = data['supplier_pairs'].T
    return identities[left] == identities[right]


def relation_masks(data):
    same = data['same']
    return dict(same=same, cross=~same)


def thresholds(calibration):
    result = {}
    for key in FEATURES:
        result[key] = {}
        for rel in RELATIONS:
            truth_scores = np.concatenate([
                data[key][relation_masks(data)[rel] & truth & np.isfinite(data[key])]
                for data, truth in calibration])
            budget = len(truth_scores)*MAX_TRUE_REJECT_PERCENT//100
            cutoff = (float(np.partition(truth_scores, budget)[budget])
                      if len(truth_scores) >= MIN_TRUE_CALIBRATION else None)
            result[key][rel] = dict(cutoff=cutoff, finite_true_pairs=len(truth_scores),
                                    max_true_pairs_rejected=budget)
    return result


def reject(data, feature, limits):
    values = data[feature]
    refused = np.zeros(len(values), bool)
    for rel, mask in relation_masks(data).items():
        cutoff = limits[feature][rel]['cutoff']
        if cutoff is not None:
            refused |= mask & np.isfinite(values) & (values < cutoff)
    return refused


def provider_matrix(data, keep):
    return original.dense_edges(data['supplier_pairs'], keep, len(data['ws']))


def supported(adjacency, support):
    return (adjacency.astype(np.int32) @ support.astype(np.int32)) > 0


def protected_choices(actions, choices, adjacency, support):
    needed = ~support & supported(adjacency, support)
    permitted = []
    for action in actions:
        survives = True
        for part in action.parts:
            nodes = list(part)
            remains = supported(adjacency[np.ix_(nodes, nodes)], support[nodes])
            if (needed[nodes] & ~remains).any():
                survives = False
                break
        permitted.append(survives)
    return np.asarray([all(permitted[edit] for edit in choice) for choice in choices])


def replay_choices(data, record, frozen, limits):
    actions = [Action(tuple(scope), tuple(map(tuple, parts)), family)
               for scope, parts, family in record['actions']]
    choices = _choices(record, actions)
    gains = frozen['NATIVE_TREE_gains'][c85.MODES.index('ABS')]
    allowed = [np.ones(len(choices), bool), frozen['NATIVE_TREE_lost'] == 0]
    rejected = [reject(data, key, limits) for key in FEATURES]
    for refused in (np.zeros(len(data['supplier_pairs']), bool), *rejected):
        allowed.append(protected_choices(actions, choices, provider_matrix(data, ~refused), data['support']))
    allowed = np.stack(allowed)
    selected = [_select(gains, keep)[0] for keep in allowed]
    for mode, index in (('ABS', 0), ('ABS_KEEP_Q', 1)):
        if selected[index] != frozen['NATIVE_TREE_accepted'][c85.MODES.index(mode)]:
            raise ValueError('C85 control did not replay exactly')
    return dict(allowed=allowed, selected=np.asarray(selected), rejected=np.stack(rejected), gains=gains)


def count_pairs(scores, truth, exposure, refused, mask):
    eligible = mask & np.isfinite(scores)
    selected = mask & refused
    return dict(
        truth=int((mask & truth).sum()), other=int((mask & ~truth).sum()),
        eligible_truth=int((eligible & truth).sum()), eligible_other=int((eligible & ~truth).sum()),
        reject_truth=int((selected & truth).sum()), reject_other=int((selected & ~truth).sum()),
        eligible_truth_q=int(exposure[eligible & truth].sum()), eligible_other_q=int(exposure[eligible & ~truth].sum()),
        reject_truth_q=int(exposure[selected & truth].sum()), reject_other_q=int(exposure[selected & ~truth].sum()),
        auc=auc(scores[eligible], truth[eligible]))


def score_scene(data, identities, replay):
    truth = pair_truth(data, identities)
    exposure = data['exposure']
    support = data['support']
    truth_before = ~support & supported(provider_matrix(data, truth), support)
    any_before = ~support & supported(provider_matrix(data, np.ones(len(truth), bool)), support)
    feature_rows, slot_rows = {}, {}
    for key, refused in zip(FEATURES, replay['rejected'], strict=True):
        feature_rows[key] = {name: count_pairs(data[key], truth, exposure, refused, mask)
                             for name, mask in relation_masks(data).items()}
        truth_after = supported(provider_matrix(data, truth & ~refused), support)
        any_after = supported(provider_matrix(data, ~refused), support)
        false_before = any_before & ~truth_before
        slot_rows[key] = dict(
            true_slots=int(truth_before.sum()), false_only_slots=int(false_before.sum()),
            true_slots_lost=int((truth_before & ~truth_after).sum()),
            true_slots_lost_but_other_survives=int((truth_before & ~truth_after & any_after).sum()),
            false_only_slots_released=int((false_before & ~any_after).sum()))
    available = data['available']
    context = {key: count_pairs(-data[key], truth, exposure, np.zeros(len(truth), bool), available)
               for key in CONTEXT}
    return dict(features=feature_rows, slots=slot_rows, geometry=context,
                cross_suppliers=int((~data['same']).sum()), cross_observed=int(data['observed'].sum()),
                cross_external=int(available.sum()),
                cross_three_anchors_three_times=int(((data['anchor_counts'] >= 3).sum(axis=1) >= 3).sum()),
                external_anchors_max=int(data['anchor_counts'].max(initial=0)))


def score_replay(data, identities, record, replay):
    ws, sources, support, pairs = (data[key] for key in ('ws', 'sources', 'support', 'pairs'))
    actions, choices, anchors, _, counts, _, protected, _ = qualify_catalog(record, ws, identities, sources)
    net = counts[:, 0]-counts[:, 1]
    good = protected & (net[:, :2].sum(axis=1) > 0) & (net[:, 2:].sum(axis=1) < 0)
    same_truth = identities[:, None] == identities[None, :]
    before_members = ws[:, None] == ws[None, :]
    truth_before = supported(before_members & same_truth, support)
    before = before_members[pairs[:, 0], pairs[:, 1]]
    old = original.dense_edges(pairs, data['scores']-original.THRESHOLD, len(ws))
    support_before = support_state(ws, support, sources)
    rows = {}
    for key, index, permitted in zip(REPLAY, replay['selected'], replay['allowed'], strict=True):
        index = int(index)
        state = materialize(ws, (actions[j] for j in choices[index]))
        state_members = state[:, None] == state[None, :]
        truth_after = supported(state_members & same_truth, support)
        group_after = supported(state_members, support)
        missing = truth_before & ~truth_after
        outcome = selected_output(index, choices, actions, counts, record, ws, identities, sources, support,
                                  pairs, old, before, anchors, support_before)
        rows[key] = dict(choice=index, permitted=int(permitted.sum()),
                         simultaneous_candidates_permitted=int((permitted & good).sum()),
                         simultaneous_positive_candidates=int((permitted & good & (replay['gains'] > original.EPSILON)).sum()),
                         protected=bool(protected[index]), simultaneous=bool(good[index]),
                         simultaneous_true_support_safe=bool(good[index] and not missing.any()),
                         same_identity_support_lost=int(missing.sum()),
                         same_identity_loss_hidden=int((missing & group_after).sum()),
                         old_support_lost=int((supported(before_members, support) & ~group_after).sum()),
                         anchor_loss=outcome['anchors']['lost'], duplicates=outcome['anchors']['duplicates'],
                         **{axis: outcome['edge_net'][axis] for axis in EDGE_AXES})
    return rows


def sum_counts(rows, *, mean_key=None):
    result = {key: sum(row[key] for row in rows) for key in rows[0] if key != mean_key}
    if mean_key is not None:
        values = [row[mean_key] for row in rows if row[mean_key] is not None]
        result.update(auc_scene_mean=float(np.mean(values)) if values else None, auc_scenes=len(values))
    return result


def aggregate(rows):
    return dict(
        scenes=len(rows), components=len({row['component'] for row in rows}),
        **{key: sum(row[key] for row in rows) for key in
           ('cross_suppliers', 'cross_observed', 'cross_external', 'cross_three_anchors_three_times')},
        external_anchors_max=max(row['external_anchors_max'] for row in rows),
        features={key: {rel: sum_counts([r['features'][key][rel] for r in rows], mean_key='auc')
                        for rel in RELATIONS} for key in FEATURES},
        slots={key: sum_counts([r['slots'][key] for r in rows]) for key in FEATURES},
        geometry={key: sum_counts([r['geometry'][key] for r in rows], mean_key='auc') for key in CONTEXT},
        replay={key: sum_counts([{k: (int(bool(v)) if k == 'choice' else v)
                                  for k, v in r['replay'][key].items()} for r in rows]) for key in REPLAY})


def run(runs, output):
    if output.parent != runs:
        raise ValueError('use a fresh child of the frozen runs directory')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, split='T_core', optimizer_updates=0, gpu_used=False,
                 reference_read=False, stages=[], started_at=datetime.now(timezone.utc).isoformat(),
                 affinity=sorted(os.sched_getaffinity(0)), time_policy='advisory',
                 inputs=dict(c78=C78, c82=C82, c85=C85))

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=original.peak_rss_bytes())
        original.write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (original.MAX_AS, original.MAX_AS))
        t32, comps32 = original.load_population(runs)
        t18, comps18 = held.load_population(runs)
        if set(comps32.values()) & set(comps18.values()):
            raise ValueError('supplier calibration and evaluation share a component')
        parent = original.read(runs/C85/'run.json')
        if (parent['status'] != 'COMPLETED' or parent['exit_code'] != 0
                or parent['scene_ids'] != t18 or parent['frozen_scenes'] != len(t18)):
            raise ValueError('completed C85 frozen holdout required')
        archive32 = bind_frozen(runs/C82, runs, runs/C78, t32)
        archive18 = runs/C85/'private'
        checkpoint(phase='blind', calibration_ids=t32, held_ids=t18)
        for ids, training, path, name in ((t32, True, archive32, 'motion'), (t18, False, archive18, 'scores')):
            population = np.zeros(3, dtype=np.int64)
            for sid in ids:
                data = freeze_scene(scene_inputs(runs, sid, training), path/f'{name}-{sid}.npz')
                group_available = supported(provider_matrix(data, np.ones(len(data['same']), bool)), data['support'])
                population += [(group_available & ~data['support']).sum(), data['same'].sum(), (~data['same']).sum()]
                np.savez_compressed(private/f'features-{sid}.npz', **data)
            expected = (36739, 923, 1559) if training else (8597, 164, 308)
            np.testing.assert_array_equal(population, expected)
        state['stages'].append('all_supplier_features_saved_before_identity')
        checkpoint(phase='calibration')
        calibration = []
        for sid in t32:
            data = read_scene(private/f'features-{sid}.npz')
            calibration.append((data, pair_truth(data, read_truth(runs, sid, data['nodes'], True))))
        limits = thresholds(calibration)
        original.write(output/'calibration.json', limits)
        for sid in t18:
            data = read_scene(private/f'features-{sid}.npz')
            record = original.read(archive18/f'catalogue-{sid}.json.gz')['NATIVE_TREE']
            with np.load(archive18/f'scores-{sid}.npz', allow_pickle=False) as source:
                replay = replay_choices(data, record, source, limits)
            np.savez_compressed(private/f'choices-{sid}.npz', **replay)
        state['stages'].append('all_held_replay_choices_saved_before_held_identity')
        checkpoint(phase='held_evaluation')
        rows = []
        for sid in t18:
            data = read_scene(private/f'features-{sid}.npz')
            replay = read_scene(private/f'choices-{sid}.npz')
            identities = read_truth(runs, sid, data['nodes'], False)
            record = original.read(archive18/f'catalogue-{sid}.json.gz')['NATIVE_TREE']
            rows.append(dict(scene_id=sid, component=comps18[sid], nodes=len(data['nodes']),
                             **score_scene(data, identities, replay),
                             replay=score_replay(data, identities, record, replay)))
        grouped = {c: [row for row in rows if row['component'] == c] for c in sorted(set(comps18.values()))}
        largest = min(grouped, key=lambda c: (-sum(r['nodes'] for r in grouped[c]), c))
        original.write(output/'scenes.json', rows)
        original.write(output/'summary.json', dict(
            calibration=limits, held=aggregate(rows), largest_component=largest,
            without_largest=aggregate([r for r in rows if r['component'] != largest]),
            by_component={c: aggregate(group) for c, group in grouped.items()}))
        state['stages'].append('held_identity_and_frozen_selection_evaluation_complete')
        checkpoint(status='COMPLETED', phase='done', exit_code=0, completed_scenes=len(rows),
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
