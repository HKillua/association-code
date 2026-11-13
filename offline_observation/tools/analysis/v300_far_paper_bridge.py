"""C89: extend the fixed far readout to V300 and pair against historical 113."""
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
from association.evaluation.artifacts import load_scene_prediction
from association.evaluation.fusion_benchmark import evaluate_fusion_methods
from association.evaluation.group_prediction import emit_group_outputs
from association.evaluation.mixture_opportunity import _distance
from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion, paper_fusion_scene
from tools.analysis.far_supported_speed_readout import build_readouts
from tools.analysis.held_fusion_population import (
    HELD_IDS, SPLIT, load_full_population, load_scene, read_evaluation, validate_measurement_archive,
)
from tools.analysis.motion_fallback_diagnostic import INVARIANTS, freeze_scene, require
from tools.analysis.motion_readout_evaluation import POPULATION_KEYS, _stats, _wrap
from tools.analysis.motion_readout_pilot import readouts
from tools.analysis.offline_endpoint_lineage import endpoint_arrays, freeze_endpoints
from tools.analysis.same_source_swap_pilot import MAX_AS, group_members, peak_rss_bytes, read, write

METHODS = ('MEAN', 'B', 'FAR')
PEER = 'HISTORICAL_TCM_TFM_113'
C83 = 'LEVEL1-HELD-SPEED-READOUT-20261112-attempt-1'
RUN113 = 'LEVEL1-TMTAFF-FIXED-EVAL-REPAIR-20261029-113-attempt-1'


def prepare(scene):
    nodes, sources, support, nearest, members, native, baseline = (
        scene[k] for k in ('nodes', 'sources', 'support', 'nearest', 'membership', 'native', 'baseline'))
    query, times, _, offsets = native
    outputs, masks = readouts(native, members, baseline, support)
    c74 = dict(outputs, **masks, membership=members)
    legacy = scene['native_motion'][..., 2:]
    tags = freeze_scene((nodes, sources, support, nearest, members, native, baseline), c74, legacy)
    lineage = freeze_endpoints(tags, query, endpoint_arrays(times, scene['values'], offsets),
                               sources, baseline, legacy)
    base = c74['B']
    fixed = dict(B=base, NEAR=base, active=c74['active'], membership=members,
                 switch=np.zeros(base.shape[:2], bool))
    far = build_readouts(fixed, c74, tags, lineage)
    return dict(MEAN=baseline, B=base, FAR=far['SUPPORTED_HOLD'], far=far['far'], nodes=nodes,
                support=support, nearest=nearest, membership=members, query=query)


def verify_peer_inputs(runs, ids):
    root = runs/RUN113
    source = read(root/'run.json')
    index = read(root/'predictions/index.json')
    require(source['status'] == 'COMPLETED' and source['scenes'] == len(ids)
            and source['source_verification']['aligned_reference_exact']
            and source['source_verification']['fixed_groups_exact']
            and index['test_opened'] is False and [row['scene_id'] for row in index['scenes']] == ids,
            'original completed 113 V300 artifacts required')
    entries = {}
    for row in index['scenes']:
        filename = row['metadata']
        require(Path(filename).name == filename, 'invalid 113 metadata filename')
        entries[row['scene_id']] = root/'predictions'/filename
    return entries


def align_peer(path, sid, component, frozen, ev):
    saved = load_scene_prediction(path)
    scene = saved['scene']
    old = saved['methods']['predicted']
    require(scene['scene_id'] == sid and scene['component_id'] == component and scene['split'] == SPLIT
            and old['correspondence_policy'] == 'anchor_identity_first'
            and scene['physical_units']['state_speed_unit'] == 'mps'
            and scene['physical_units']['reference_speed_unit'] == 'mps',
            '113 split, scene, grouping or units differ')
    np.testing.assert_array_equal(scene['node_ids'], frozen['nodes'])
    np.testing.assert_array_equal(scene['grid'], frozen['query'])
    np.testing.assert_array_equal(scene['identities'], ev['identities'])
    targets = old['target_ids']
    wanted = ev['target_ids']
    index = {target: i for i, target in enumerate(targets.tolist())}
    if len(index) != len(targets) or set(index) != set(wanted.tolist()):
        return None, dict(reason='different_target_population', old_targets=len(targets), current_targets=len(wanted))
    order = np.asarray([index[target] for target in wanted.tolist()], dtype=np.int64)
    target_valid = ev['target_valid']
    prior_valid = old['aligned_truth_valid'][order]
    mask_diff = np.count_nonzero(prior_valid != target_valid, axis=(0, 1)).tolist()
    reference_diff = np.count_nonzero(prior_valid & target_valid
                                      & (old['aligned_truth'][order] != ev['target_states']), axis=(0, 1)).tolist()
    if any(mask_diff) or any(reference_diff):
        return None, dict(reason='different_reference_mask_or_value', mask_diff=mask_diff, value_diff=reference_diff)
    return dict(prediction=old['aligned_prediction'][order], valid=old['aligned_prediction_valid'][order],
                groups=len(old['group_states']), matched=int((old['aligned_group_indices'][order] >= 0).sum()),
                duplicates=len(old['aligned_group_indices'])-len(targets)), None


def point_errors(prediction, target, mask):
    return dict(position=_distance(prediction[..., :2][mask[..., :2].all(axis=-1)],
                                   target[..., :2][mask[..., :2].all(axis=-1)]),
                speed=(prediction[..., 2]-target[..., 2])[mask[..., 2]],
                heading=_wrap(prediction[..., 3][mask[..., 3]] % 360.-target[..., 3][mask[..., 3]] % 360.))


def metric_vectors(predictions, valid, ev, common=None):
    target, target_valid = ev['target_states'], ev['target_valid']
    paper, errors = {}, {}
    for method, prediction in predictions.items():
        mask = valid[method] if common is None else common
        paper[method] = paper_fusion_scene(prediction, target, target_valid, prediction_valid=mask,
                                           prediction_speed_unit='mps', target_speed_unit='mps')
        errors[method] = point_errors(prediction, target, mask & target_valid)
    return paper, errors


def evaluate(sid, component, frozen, ev, peer_path):
    groups = [g.tolist() for g in group_members(frozen['membership'])]
    reports, artifacts = evaluate_fusion_methods(
        {method: emit_group_outputs(frozen[method], groups, node_count=len(frozen['nodes'])) for method in METHODS},
        prediction_speed_units={method: 'mps' for method in METHODS}, node_identities=ev['identities'],
        target_ids=ev['target_ids'], target_states=ev['target_states'], target_valid=ev['target_valid'],
        support=frozen['support'], nearest_seconds=frozen['nearest'], target_speed_unit='mps', split=SPLIT)
    base = artifacts['B']
    for method in ('MEAN', 'FAR'):
        for key in (*INVARIANTS, 'aligned_truth'):
            np.testing.assert_array_equal(base[key], artifacts[method][key])
        require(all(reports[method][key] == reports['B'][key] for key in POPULATION_KEYS),
                'fixed WS target coverage changed')
    n = len(ev['target_ids'])
    predictions = {method: a['aligned_prediction'][:n] for method, a in artifacts.items()}
    valid = {method: a['aligned_prediction_valid'][:n] for method, a in artifacts.items()}
    peer, unpaired = align_peer(peer_path, sid, component, frozen, ev)
    if peer is not None:
        predictions[PEER], valid[PEER] = peer['prediction'], peer['valid']
    paper = {method: reports[method]['paper_fusion_metrics'] for method in METHODS}
    own_errors = {method: point_errors(predictions[method], ev['target_states'], valid[method] & ev['target_valid'])
                  for method in METHODS}
    common_paper, common_errors, peer_coverage = {}, {}, None
    if peer is not None:
        paper[PEER], peer_errors = metric_vectors({PEER: predictions[PEER]}, valid, ev)
        paper[PEER] = paper[PEER][PEER]
        own_errors.update(peer_errors)
        common = valid['B'] & valid[PEER] & ev['target_valid']
        common_paper, common_errors = metric_vectors(predictions, valid, ev, common)
        peer_coverage = {k: peer[k] for k in ('groups', 'matched', 'duplicates')}
    idx, slot = np.nonzero(valid['B'][..., 2] & ev['target_valid'][..., 2])
    gid = base['aligned_group_indices'][idx]
    require((gid >= 0).all(), 'valid fixed WS speed must map to an original group')
    far = frozen['far'][gid, slot]
    row = dict(scene=sid, component=component, target_count=n,
               ws_coverage={key: reports['B'][key] for key in POPULATION_KEYS},
               ws_unmatched=len(reports['B']['unmatched_target_ids']), peer_coverage=peer_coverage,
               peer_alignment_issue=unpaired, own_paper=paper, paired_paper=common_paper,
               far_group_queries=int(frozen['far'].sum()))
    return row, dict(own=own_errors, paired=common_errors,
                     far={method: own_errors[method]['speed'][far] for method in METHODS})


def pooled(rows, vectors):
    result = dict(scenes=len(rows), components=len({r['component'] for r in rows}),
                  peer_paired_scenes=sum(r['peer_alignment_issue'] is None for r in rows),
                  targets=sum(r['target_count'] for r in rows), ws_unmatched=sum(r['ws_unmatched'] for r in rows),
                  far_group_queries=sum(r['far_group_queries'] for r in rows),
                  ws_coverage={k: sum(r['ws_coverage'][k] for r in rows) for k in POPULATION_KEYS},
                  peer_coverage={k: sum(r['peer_coverage'][k] for r in rows if r['peer_coverage'] is not None)
                                 for k in ('groups', 'matched', 'duplicates')})
    for region, field in (('own', 'own_paper'), ('paired', 'paired_paper')):
        result[field] = {}
        point = {}
        for method in (*METHODS, PEER):
            reports = [r[field][method] for r in rows if method in r[field]]
            if reports:
                result[field][method] = aggregate_paper_fusion(reports)
                sample = [v[region][method] for v in vectors if method in v[region]]
                point[method] = {name: _stats(np.concatenate([x[name] for x in sample])) for name in sample[0]}
        result[region+'_point'] = point
    far = {method: np.concatenate([v['far'][method] for v in vectors]) for method in METHODS}
    result['far_speed'] = {method: _stats(error) for method, error in far.items()}
    deltas = np.abs(far['FAR'])-np.abs(far['B'])
    result['far_paired_changes'] = dict(improved=int((deltas < 0).sum()), worsened=int((deltas > 0).sum()),
                                       max_harm=float(deltas.max(initial=0.)))
    return result


def run(runs, output):
    require(output.parent == runs, 'fresh output under frozen runs required')
    output.mkdir(exist_ok=False)
    private = output/'private'
    private.mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', argv=sys.argv, stages=[], split=SPLIT, raw_measurement_scenes=0,
                 raw_label_scenes=0, raw_reference_scenes=0, test_opened=False, optimizer_updates=0,
                 affinity=sorted(os.sched_getaffinity(0)), time_policy='advisory',
                 started_at=datetime.now(timezone.utc).isoformat())

    def checkpoint(**kwargs):
        state.update(kwargs, elapsed_seconds=time.monotonic()-started, peak_rss_bytes=peak_rss_bytes())
        write(output/'run.json', state)

    try:
        resource.setrlimit(resource.RLIMIT_AS, (MAX_AS, MAX_AS))
        ids, components, archives = load_full_population(runs)
        prior = read(runs/C83/'run.json')
        require(prior['status'] == 'COMPLETED' and tuple(prior['scene_ids']) == HELD_IDS,
                'original completed C83 held output required')
        peers = verify_peer_inputs(runs, ids)
        checkpoint(phase='freeze_predictions', scene_ids=ids)
        population = set(ids)
        with MTADArchive(archives) as archive:
            validate_measurement_archive(runs, archive)
            for i, sid in enumerate(ids, 1):
                frozen = prepare(load_scene(runs, sid, archive, scene_ids=population))
                if sid in HELD_IDS:
                    with np.load(runs/C83/'private'/f'{sid}.npz', allow_pickle=False) as c83:
                        for key in ('B', 'FAR', 'far', 'membership', 'query', 'support', 'nodes'):
                            np.testing.assert_array_equal(frozen[key], c83[key])
                np.savez_compressed(private/f'{sid}.npz', **frozen)
                if i % 10 == 0:
                    checkpoint(raw_measurement_scenes=i, frozen_scenes=i)
        state['stages'].append('all_own_predictions_frozen_before_reference_and_peer')
        checkpoint(phase='evaluate', held_c83_replayed=len(HELD_IDS))
        rows, vectors = [], []
        for i, sid in enumerate(ids, 1):
            with np.load(private/f'{sid}.npz', allow_pickle=False) as saved:
                frozen = {key: saved[key] for key in saved.files}
            ev = read_evaluation(runs, sid, frozen['nodes'], frozen['query'], scene_ids=population)
            row, vector = evaluate(sid, components[sid], frozen, ev, peers[sid])
            rows.append(row)
            vectors.append(vector)
            if i % 10 == 0:
                checkpoint(completed_scenes=i)

        def subset(mask):
            take = [i for i, row in enumerate(rows) if mask(row)]
            return pooled([rows[i] for i in take], [vectors[i] for i in take])

        reference_points = {c: sum(row['own_paper']['B']['attributes']['latitude']['available_target_points']
                                   for row in rows if row['component'] == c) for c in set(components.values())}
        largest = min(reference_points, key=lambda c: (-reference_points[c], c))
        all_scenes = pooled(rows, vectors)
        component_summary = {c: subset(lambda row: row['component'] == c) for c in sorted(reference_points)}
        leave = subset(lambda row: row['component'] != largest)
        write(output/'scenes.json.gz', [dict(row, far_speed={k: _stats(v) for k, v in vector['far'].items()})
                                       for row, vector in zip(rows, vectors, strict=True)])
        write(output/'summary.json', dict(
            all_scenes=all_scenes, held_v48=subset(lambda row: row['scene'] in HELD_IDS),
            remaining_v252=subset(lambda row: row['scene'] not in HELD_IDS), by_component=component_summary,
            without_largest=dict(component=largest, result=leave),
            peer_alignment_issues=[dict(scene=row['scene'], **row['peer_alignment_issue'])
                                   for row in rows if row['peer_alignment_issue'] is not None],
            far_mae_improving_components=sum(
                r['far_speed']['B']['count'] > 0 and r['far_speed']['FAR']['mae'] < r['far_speed']['B']['mae']
                for r in component_summary.values())))
        state['stages'].append('own_and_verified_peer_four_attribute_evaluation_complete')
        checkpoint(status='COMPLETED', exit_code=0, phase='done', peer_paired_scenes=all_scenes['peer_paired_scenes'],
                   finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        checkpoint(status='FAILED', phase=state.get('phase'), exit_code=1, error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    run(args.runs.resolve(), args.output.resolve())
