"""C126: pair saved T128+FAR and historical 113 outputs on V300."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from association.evaluation.paper_fusion_metrics import aggregate_paper_fusion, paper_fusion_scene
from tools.analysis.held_fusion_population import SPLIT, load_full_population, read_evaluation
from tools.analysis.motion_fallback_diagnostic import require
from tools.analysis.motion_readout_evaluation import _stats
from tools.analysis.same_source_swap_pilot import group_members, peak_rss_bytes, read, write
from tools.analysis.v300_far_paper_bridge import PEER, align_peer, point_errors, verify_peer_inputs


C89 = 'LEVEL1-V300-FAR-PAPER-BRIDGE-20261112-attempt-1'
C94 = 'LEVEL1-V300-POSITION-FAR-20261112-attempt-1'
ARMS = ('FAR', 'T128_FAR', PEER)


def summary_state(output, start, **fields):
    write(output/'run.json', dict(elapsed_seconds=time.monotonic()-start,
                                 peak_rss_bytes=peak_rss_bytes(), **fields))


def anchor_outputs(saved, ev):
    """Apply the frozen group order and first-anchor correspondence, without rematching."""
    targets = ev['target_ids'].tolist()
    target_ordinals = {target: i for i, target in enumerate(targets)}
    require(len(targets) == len(target_ordinals), '151 has duplicate declared targets')
    membership = group_members(saved['membership'])
    anchors = {}
    for gid, members in enumerate(membership):
        target = target_ordinals[ev['identities'][members[0]]]
        anchors.setdefault(target, gid)
    target_rows = np.asarray(list(anchors), dtype=int)
    group_rows = np.asarray(list(anchors.values()), dtype=int)
    prediction = {}
    mask = np.zeros_like(ev['target_valid'])
    mask[target_rows] = True
    for arm in ARMS[:2]:
        states = np.asarray(saved[arm])
        require(states.shape == (len(membership), len(saved['query']), 4)
                and np.isfinite(states).all(), 'invalid frozen C94 outputs')
        values = np.zeros_like(ev['target_states'])
        values[target_rows] = states[group_rows]
        prediction[arm] = values
    np.testing.assert_array_equal(prediction['FAR'][..., 2:], prediction['T128_FAR'][..., 2:])
    return prediction, mask, dict(groups=len(membership), matched=len(anchors),
                                  duplicates=len(membership)-len(anchors))


def metrics(predictions, masks, ev):
    reports = {}
    for arm, prediction in predictions.items():
        mask = masks[arm]
        reports[arm] = paper_fusion_scene(prediction, ev['target_states'], ev['target_valid'],
                            prediction_valid=mask, prediction_speed_unit='mps', target_speed_unit='mps')
    return reports


def errors(prediction, truth, mask):
    result = point_errors(prediction, truth, mask)
    for name, column in (('longitude', 1), ('latitude', 0)):
        valid = mask[..., column]
        result[name] = (prediction[..., column][valid]-truth[..., column][valid])*10000.
    return result


def read_scene(runs, sid, component, peer_path, old_c89, old_c94, population):
    with np.load(runs/C94/'private'/f'{sid}.npz', allow_pickle=False) as source:
        saved = {key: source[key] for key in ('FAR', 'T128_FAR', 'nodes', 'query', 'membership')}
    ev = read_evaluation(runs, sid, saved['nodes'], saved['query'], scene_ids=population)
    predictions, mask, ws_count = anchor_outputs(saved, ev)
    peer, issue = align_peer(peer_path, sid, component, saved, ev)
    require(peer is not None, f'historical 113 alignment issue at {sid}: {issue}')
    predictions[PEER] = peer['prediction']
    own_masks = {'FAR': mask, 'T128_FAR': mask, PEER: peer['valid']}
    common = mask & peer['valid'] & ev['target_valid']
    paired_masks = dict.fromkeys(ARMS, common)
    own, paired = metrics(predictions, own_masks, ev), metrics(predictions, paired_masks, ev)
    for arm in ('FAR', PEER):
        require(own[arm] == old_c89['own_paper'][arm] and paired[arm] == old_c89['paired_paper'][arm],
                f'C89 metric parity failed at {sid}/{arm}')
    require(own['T128_FAR'] == old_c94['paper']['T128_FAR'], f'C94 metric parity failed at {sid}')
    peer_count = {key: peer[key] for key in ('groups', 'matched', 'duplicates')}
    require(peer_count == old_c89['peer_coverage'] and all(ws_count[k] == old_c94['population'][field]
            for k, field in (('groups', 'predicted_group_count'), ('matched', 'anchor_matched_target_count'),
                             ('duplicates', 'duplicate_prediction_group_count'))),
            f'coverage parity failed at {sid}')
    vectors = {scope: {arm: errors(prediction, ev['target_states'], arm_masks[arm] & ev['target_valid'])
                        for arm, prediction in predictions.items()}
               for scope, arm_masks in (('own', own_masks), ('paired', paired_masks))}
    own_count = {'FAR': ws_count, 'T128_FAR': ws_count, PEER: peer_count}
    return dict(scene=sid, component=component, target_count=len(ev['target_ids']),
                population=own_count, own=own, paired=paired), vectors


def pool(rows, values):
    return {scope: {arm: dict(paper=aggregate_paper_fusion(row[scope][arm] for row in rows),
                point={attribute: _stats(np.concatenate([v[scope][arm][attribute] for v in values]))
                       for attribute in values[0][scope][arm]}) for arm in ARMS}
            for scope in ('own', 'paired')}


def run(runs, output, start):
    ids, components, _ = load_full_population(runs)
    for parent in (C89, C94):
        run = read(runs/parent/'run.json')
        require(run['status'] == 'COMPLETED' and run['exit_code'] == 0
                and run['scene_ids'] == ids and not run['test_opened'], f'parent {parent} changed')
    peer = verify_peer_inputs(runs, ids)
    original = [read(runs/parent/'scenes.json.gz') for parent in (C89, C94)]
    require(all([row['scene'] for row in records] == ids for records in original), 'scene order changed')
    rows, values = [], []
    for i, (sid, old_c89, old_c94) in enumerate(zip(ids, *original, strict=True), 1):
        row, vectors = read_scene(runs, sid, components[sid], peer[sid], old_c89, old_c94, ids)
        rows.append(row)
        values.append(vectors)
        if i % 50 == 0:
            summary_state(output, start, status='RUNNING', processed_scenes=i)
    overall = pool(rows, values)
    coverage = {arm: {field: sum(row['population'][arm][field] for row in rows)
                     for field in rows[0]['population'][arm]} for arm in ARMS}
    write(output/'summary.json', dict(scenes=len(rows), components=len(set(components.values())),
                targets=sum(row['target_count'] for row in rows), population=coverage, **overall))
    write(output/'scenes.json.gz', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-policy', choices=('advisory',), required=True)
    args = parser.parse_args()
    require(args.output.parent == args.runs, 'output must be a fresh child of runs')
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    try:
        run(args.runs, args.output, start)
    except Exception as exc:
        summary_state(args.output, start, status='FAILED', error=repr(exc))
        raise
    summary_state(args.output, start, status='COMPLETED', optimizer_updates=0, model_forwards=0,
                  raw_archive_reads=0, test_opened=False, processed_scenes=300)


if __name__ == '__main__':
    main()
