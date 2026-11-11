"""C88: validate external translation as a supplier gate on frozen new components."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

from association.data.archive import MTADArchive
from tools.analysis import held_fusion_population as v_population
from tools.analysis import same_source_swap_pilot as original
from tools.analysis import supplier_provenance_qualification as c87
from tools.analysis.disjoint_partition_evidence import _choices, _select
from tools.analysis.disjoint_partition_motion_rerank import MODES as C82_MODES, edge_scores, score_arm
from tools.analysis.disjoint_partition_pilot import Action, catalog, materialize, qualify_catalog, selected_output, support_state
from tools.analysis.disjoint_partition_proposals import propose
from tools.analysis.disjoint_partition_rerank import combine_outputs
from tools.analysis.static_partition_pilot import agglomerate
from tools.analysis.supplier_external_geometry import external_geometry

C87 = 'LEVEL1-SUPPLIER-PROVENANCE-20261112-attempt-1'
C83 = 'LEVEL1-HELD-SPEED-READOUT-20261112-attempt-1'
V16 = ('38', '152', '203', '472', '670', '929', '932', '1328', '1393', '1531', '1595', '2252',
       '2731', '2850', '3272', '3331')
MODES = ('WS', 'ABS', 'J_GATE', 'RAW_J', 'TRAN_J', 'TRAN_KEEP')
PAIR_MODES = ('J_GATE', 'RAW_J', 'TRAN_J', 'TRAN_KEEP')
ABS_INDEX = C82_MODES.index('ABS')
SUPPORT_KEYS = ('same_ship_before', 'same_ship_after', 'same_ship_lost', 'same_ship_recovered',
                'same_ship_loss_hidden', 'any_support_lost', 'wrong_only_before', 'wrong_only_after',
                'wrong_only_released', 'wrong_only_corrected', 'wrong_only_introduced')


def bind_population(runs):
    v48, components, archive_paths = v_population.load_population(runs)
    original32, _ = original.load_population(runs)
    c83, c87_run = (original.read(runs/key/'run.json') for key in (C83, C87))
    if (any(row['status'] != 'COMPLETED' or row['exit_code'] != 0 for row in (c83, c87_run))
            or c83['scene_ids'] != v48 or c83['frozen_prediction_scenes'] != 48
            or c87_run['calibration_ids'] != original32 or c87_run['completed_scenes'] != 18
            or c87_run['stages'][0] != 'all_supplier_features_saved_before_identity'):
        raise ValueError('original completed C83 V48 and C87 supplier cohorts required')
    full = original.read(runs/v_population.PARENTS['full_data']/'run.json')
    manifest = original.read(Path(full['manifest']))
    component_map = manifest['components']
    excluded = {component_map[sid] for sid in original32+c87_run['held_ids']}
    v_ids = [sid for sid in v48 if components[sid] not in excluded]
    if tuple(v_ids) != V16 or len({components[sid] for sid in v_ids}) != 12:
        raise ValueError('original component-excluded V16 differs')
    return original32, v_ids, {sid: components[sid] for sid in v_ids}, archive_paths, manifest


def freeze_scene(runs, sid, archive, score_cache, private):
    scene = v_population.load_scene(runs, sid, archive)
    nodes, sources, support, ws, native = (scene[key] for key in ('nodes', 'sources', 'support', 'membership', 'native'))
    with np.load(runs/C83/'private'/f'{sid}.npz', allow_pickle=False) as frozen:
        for key in ('nodes', 'query', 'support', 'membership'):
            np.testing.assert_array_equal(scene[key], frozen[key], err_msg=f'C83 frozen {sid} {key}')
    pairs, scores = (score_cache[f'{sid}_{key}'] for key in ('pairs', 'scores'))
    np.testing.assert_array_equal(score_cache[f'{sid}_node_ids'], nodes)
    np.testing.assert_array_equal(score_cache[f'{sid}_sources'], sources)
    np.testing.assert_array_equal(pairs, np.column_stack(np.triu_indices(len(ws), 1)))
    if scores.dtype != np.float64 or scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError('056 frozen V score schema changed')
    np.testing.assert_array_equal(agglomerate(len(ws), pairs, scores, 'WS')[0], ws)

    left, right = pairs.T
    grouped = np.flatnonzero(ws[left] == ws[right])
    exposure = np.count_nonzero(support[left[grouped]] ^ support[right[grouped]], axis=1)
    rows, exposure = grouped[exposure > 0], exposure[exposure > 0]
    margin = scores-original.THRESHOLD
    geo = external_geometry(pairs, rows, sources, ws, margin, native)
    motion, fields, _ = edge_scores(pairs, sources, scores, native)
    np.testing.assert_array_equal(geo['observed'], np.isfinite(fields['cross_abs_m'][rows]))
    proposals, _ = propose(ws, pairs, scores, native)
    record = catalog(ws, proposals['NATIVE_TREE'], original.dense_edges(pairs, margin, len(ws)))
    index = original.dense_edges(pairs, np.arange(len(pairs)), len(ws))
    gains, _, _ = score_arm(record, ws, motion, index, np.zeros(len(pairs), bool))
    data = dict(nodes=nodes, sources=sources, support=support, ws=ws, pairs=pairs, query=native[0],
                supplier_pairs=pairs[rows], same=sources[left[rows]] == sources[right[rows]],
                exposure=exposure, J_FULL=margin[rows], margin=margin,
                abs_edge=motion[ABS_INDEX], gains=gains[ABS_INDEX], **geo)
    original.write(private/f'catalogue-{sid}.json.gz', record)
    np.savez_compressed(private/f'features-{sid}.npz', **data)
    return len(nodes), len(original.group_members(ws)), len(native[1])


def fit_calibration(runs, ids):
    path = runs/C87/'private'
    features = [c87.read_scene(path/f'features-{sid}.npz') for sid in ids]
    if (sum(int(row['available'].sum()) for row in features),
            sum(bool(row['available'].any()) for row in features)) != (849, 14):
        raise ValueError('original C87 T32 geometry population differs')
    masks = []
    for sid, data in zip(ids, features, strict=True):
        identity = c87.read_truth(runs, sid, data['nodes'], True)
        masks.append(c87.pair_truth(data, identity) & data['available'])
    true_count = sum(int(mask.sum()) for mask in masks)
    budget = true_count*c87.MAX_TRUE_REJECT_PERCENT//100
    result = dict(J_FULL=original.read(runs/C87/'calibration.json')['J_FULL'],
                  context_calibration_true_pairs=true_count, context_true_reject_budget=budget, context={})
    for key in ('RAW', 'TRAN'):
        true_scores = np.concatenate([-row[key][mask] for row, mask in zip(features, masks, strict=True)])
        result['context'][key] = (float(np.partition(true_scores, budget)[budget])
                                  if true_count >= c87.MIN_TRUE_CALIBRATION else None)
    return result


def rejected_by_mode(data, limits):
    base = c87.reject(data, 'J_FULL', limits)
    available = data['available'] & (limits['context']['TRAN'] is not None)
    reject = {}
    for key in ('RAW', 'TRAN'):
        cutoff = limits['context'][key]
        reject[key] = (-data[key] < cutoff) & available if cutoff is not None else available.copy()
    return np.stack((base, np.where(available, reject['RAW'], base),
                     np.where(available, reject['TRAN'], base), reject['TRAN']))


def freeze_choices(data, record, limits):
    actions = [Action(tuple(scope), tuple(map(tuple, parts)), family) for scope, parts, family in record['actions']]
    choices = _choices(record, actions)
    rejected = rejected_by_mode(data, limits)
    permits = [np.ones(len(choices), bool), np.ones(len(choices), bool)]
    permits.extend(c87.protected_choices(actions, choices, c87.provider_matrix(data, ~mask), data['support'])
                   for mask in rejected)
    permits = np.stack(permits)
    selected = np.asarray([0, *(_select(data['gains'], gate)[0] for gate in permits[1:])])
    membership = np.stack([materialize(data['ws'], (actions[j] for j in choices[i])) for i in selected])
    return dict(rejected=rejected, permits=permits, selected=selected, membership=membership)


def read_identities(runs, sid, data):
    evaluation = runs/v_population.PARENTS['full_data']/'private/evaluation'/v_population.SPLIT/f'scene-{sid}.npz'
    with np.load(evaluation, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['node_ids'], data['nodes'])
        np.testing.assert_array_equal(saved['query_seconds'], data['query'])
        identities = saved['identities']
    if identities.shape != data['nodes'].shape:
        raise ValueError('151 identity population differs')
    return np.unique(identities, return_inverse=True)[1]


def support_outcome(true_before, any_before, true_after, any_after):
    lost = true_before & ~true_after
    wrong_before = any_before & ~true_before
    wrong_after = any_after & ~true_after
    masks = (true_before, true_after, lost, true_after & ~true_before, lost & any_after,
             any_before & ~any_after, wrong_before, wrong_after, wrong_before & ~any_after,
             wrong_before & true_after, wrong_after & ~wrong_before)
    return dict(zip(SUPPORT_KEYS, (int(mask.sum()) for mask in masks), strict=True))


def evaluate(data, record, frozen, identity):
    ws, sources, support, pairs = (data[key] for key in ('ws', 'sources', 'support', 'pairs'))
    actions, choices, anchors, _, counts, _, protected, _ = qualify_catalog(record, ws, identity, sources)
    net = counts[:, 0]-counts[:, 1]
    good = protected & (net[:, :2].sum(axis=1) > 0) & (net[:, 2:].sum(axis=1) < 0)
    same_identity = identity[:, None] == identity[None, :]
    old_members = ws[:, None] == ws[None, :]
    true_before = c87.supported(old_members & same_identity, support)
    any_before = c87.supported(old_members, support)
    old_pred = old_members[pairs[:, 0], pairs[:, 1]]
    old_edge = original.dense_edges(pairs, data['margin'], len(ws))
    source_before = support_state(ws, support, sources)
    modes = {}
    for key, index, keep, state in zip(MODES, frozen['selected'], frozen['permits'],
                                        frozen['membership'], strict=True):
        index = int(index)
        np.testing.assert_array_equal(state, materialize(ws, (actions[j] for j in choices[index])))
        if not keep[index]:
            raise ValueError('selected partition violates frozen supplier gate')
        new_members = state[:, None] == state[None, :]
        new_pred = new_members[pairs[:, 0], pairs[:, 1]]
        signed = new_pred.astype(np.int8)-old_pred.astype(np.int8)
        np.testing.assert_allclose(data['abs_edge'] @ signed, data['gains'][index], rtol=1e-10, atol=1e-9)
        outcome = selected_output(index, choices, actions, counts, record, ws, identity, sources,
                                  support, pairs, old_edge, old_pred, anchors, source_before)
        slots = support_outcome(true_before, any_before, c87.supported(new_members & same_identity, support),
                                c87.supported(new_members, support))
        modes[key] = dict(choice=index, protected=bool(protected[index]), simultaneous=bool(good[index]),
                          simultaneous_safe=bool(good[index] and slots['same_ship_lost'] == 0),
                          permitted=int(keep.sum()), simultaneous_permitted=int((keep & good).sum()),
                          simultaneous_positive=int((keep & good & (data['gains'] > original.EPSILON)).sum()),
                          **slots, output=outcome)
    truth, exposure = c87.pair_truth(data, identity), data['exposure']
    available = data['available']
    context = {}
    refusals = dict(zip(PAIR_MODES, frozen['rejected'], strict=True))
    for key, scores, rejected in (('J', data['J_FULL'], refusals['J_GATE']),
                                   ('RAW', -data['RAW'], refusals['RAW_J']),
                                   ('TRAN', -data['TRAN'], refusals['TRAN_J'])):
        context[key] = c87.count_pairs(scores, truth, exposure, rejected, available)
    cross = ~data['same']
    fallback = {key: c87.count_pairs(data['J_FULL'], truth, exposure, refused, cross & ~available)
                for key, refused in refusals.items()}
    return dict(modes=modes, candidates=len(choices), context=context, fallback=fallback,
                cross_suppliers=int(cross.sum()), cross_observed=int(data['observed'].sum()),
                cross_external=int(available.sum()), anchor_scenes=int(available.any()))


def aggregate(rows):
    return dict(
        scenes=len(rows), components=len({row['component'] for row in rows}),
        **{key: sum(row[key] for row in rows) for key in
           ('candidates', 'cross_suppliers', 'cross_observed', 'cross_external', 'anchor_scenes')},
        context={key: c87.sum_counts([row['context'][key] for row in rows], mean_key='auc')
                 for key in ('J', 'RAW', 'TRAN')},
        fallback={key: c87.sum_counts([row['fallback'][key] for row in rows], mean_key='auc')
                  for key in PAIR_MODES},
        modes={key: aggregate_mode([row['modes'][key] for row in rows]) for key in MODES},
    )


def aggregate_mode(rows):
    counters = ('protected', 'simultaneous', 'simultaneous_safe', 'permitted', 'simultaneous_permitted',
                'simultaneous_positive', *SUPPORT_KEYS)
    return dict(edits_selected=sum(row['choice'] > 0 for row in rows),
                **{key: sum(row[key] for row in rows) for key in counters},
                **combine_outputs([row['output'] for row in rows]))


def run(runs, output):
    if output.parent != runs:
        raise ValueError('new child of the frozen runs directory required')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', stages=[], argv=sys.argv, split=v_population.SPLIT,
                 inputs=dict(c83=C83, c87=C87, **v_population.PARENTS), completed_scenes=0,
                 time_policy='advisory', optimizer_updates=0, gpu_used=False, physical_reference_read=False,
                 raw_label_scenes=0, raw_reference_scenes=0, test_opened=False,
                 started_at=datetime.now(timezone.utc).isoformat(), affinity=sorted(os.sched_getaffinity(0)))

    def checkpoint(**fields):
        state.update(fields, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=original.peak_rss_bytes())
        original.write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (original.MAX_AS, original.MAX_AS))
        t_ids, ids, components, archive_paths, manifest = bind_population(runs)
        score_path = runs/v_population.PARENTS['association']/'association/final-scores.npz'
        checkpoint(phase='blind', calibration_ids=t_ids, scene_ids=ids)
        population = np.zeros(3, np.int64)
        with MTADArchive(archive_paths) as archive, np.load(score_path, allow_pickle=False) as scores:
            v_population.validate_measurement_archive(runs, archive)
            if scores['scene_ids'].tolist() != manifest['splits'][v_population.SPLIT]:
                raise ValueError('056 final score manifest order differs')
            for sid in ids:
                population += freeze_scene(runs, sid, archive, scores, private)
            opened = [(item['scene_id'], item['kind']) for item in archive.members_read]
            if opened != [(sid, 'measurements') for sid in ids]:
                raise ValueError('measurement-only archive read did not match V16')
        np.testing.assert_array_equal(population, (283, 112, 75659))
        state['stages'].append('all_v16_features_and_candidate_catalogues_frozen_before_identity')
        checkpoint(phase='calibrate', nodes=int(population[0]), groups=int(population[1]),
                   events=int(population[2]), raw_measurement_scenes=len(ids))

        limits = fit_calibration(runs, t_ids)
        original.write(output/'calibration.json', limits)
        for sid in ids:
            data = c87.read_scene(private/f'features-{sid}.npz')
            record = original.read(private/f'catalogue-{sid}.json.gz')
            np.savez_compressed(private/f'choices-{sid}.npz', **freeze_choices(data, record, limits))
        state['stages'].append('all_v16_choices_frozen_before_v16_identity')
        checkpoint(phase='evaluate')

        rows = []
        for sid in ids:
            data = c87.read_scene(private/f'features-{sid}.npz')
            record = original.read(private/f'catalogue-{sid}.json.gz')
            frozen = c87.read_scene(private/f'choices-{sid}.npz')
            rows.append(dict(scene_id=sid, component=components[sid], nodes=len(data['nodes']),
                             **evaluate(data, record, frozen, read_identities(runs, sid, data))))
            checkpoint(completed_scenes=len(rows))
        by_component = {key: [row for row in rows if row['component'] == key] for key in sorted(set(components.values()))}
        largest = min(by_component, key=lambda key: (-sum(row['nodes'] for row in by_component[key]), key))
        original.write(output/'scenes.json', rows)
        original.write(output/'summary.json', dict(
            primary=('TRAN_J', 'RAW_J'), calibration=limits, all_scenes=aggregate(rows), largest_component=largest,
            without_largest=aggregate([row for row in rows if row['component'] != largest]),
            by_component={key: aggregate(group) for key, group in by_component.items()}))
        state['stages'].append('all_v16_identity_and_full_partition_recounts_complete')
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
