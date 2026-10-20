"""Fixed, zero-training leave-two-nodes-out anonymous translation control.

This is a full-fragment offline diagnostic. Estimation uses positions, source
IDs and native support only; identities enter after all prediction scores exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from association.data.archive import validate_manifest
from association.evaluation.association import (compare_predictions, evaluate_scenes,
    select_threshold)
from association.evaluation.grouping import group_predicted_tracks
from tools.analysis.motion_association_control import (_auc_summary, _subset, _write,
    rms_haversine_scores)

BASES = ('native_endpoint', 'native_ols')
ARMS = ('uncalibrated', 'once', 'three')
METRES_PER_DEGREE = 6371000. * np.pi / 180.


def _weighted_median(values, weights):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1]/2, side='left')]])


def _translation_data(positions, sources, support):
    """Per-edge moments in a local tangent approximation, using joint support.

    Each pair has its own latitude scale. Thus removing two nodes removes all
    dependence on their positions, including any projection-origin dependence.
    Geographic haversine, not this approximation, supplies final edge scores.
    """
    n = len(sources)
    if positions.shape != (n, 64, 2) or support.shape != (n, 64):
        raise ValueError('positions/support must be [N,64,2]/[N,64]')
    if support.dtype != bool or not np.isfinite(positions).all():
        raise ValueError('finite positions and boolean native support required')
    if not set(sources.tolist()).issubset({'9001', '9002'}):
        raise ValueError('this fixed control requires sources 9001 and 9002')
    a, b = np.where(sources == '9001')[0], np.where(sources == '9002')[0]
    edges = np.column_stack((np.repeat(a, len(b)), np.tile(b, len(a))))
    if not len(edges):
        return dict(edges=edges, means=np.empty((0, 2)), cos_lat=np.empty(0),
                    variance=np.empty(0))
    common = support[edges[:, 0]] & support[edges[:, 1]]
    usable = common.any(axis=1)
    edges, common = edges[usable], common[usable]
    first, second = positions[edges[:, 0]], positions[edges[:, 1]]
    delta = first-second
    delta[..., 1] = (delta[..., 1]+180.) % 360.-180.
    count = common.sum(axis=1)
    means = (delta*common[..., None]).sum(axis=1)/count[:, None]
    latitude = ((first[..., 0]+second[..., 0])/2*common).sum(axis=1)/count
    cos_lat = np.cos(np.deg2rad(latitude))
    centered = delta-means[:, None]
    variance = ((centered[..., 0]**2 + (centered[..., 1]*cos_lat[:, None])**2)
                *common).sum(axis=1)/count*METRES_PER_DEGREE**2
    return dict(edges=edges, means=means, cos_lat=cos_lat, variance=variance)


def estimate_translation(data, excluded):
    """Three fixed updates; no labels, reference, global bijection or tuning.

    Every available node votes for its nearest opposite-source neighbour under
    the current translation. Many nodes may choose the same neighbour. Each
    source carries half of total pre-trim vote mass. Votes whose RMS exceeds
    median + 3*MAD are trimmed (1 metre MAD floor only for numeric degeneracy),
    followed by coordinate-wise weighted medians of pair mean offsets.
    """
    edges = data['edges']
    keep = ~np.isin(edges, excluded).any(axis=1)
    edges, means = edges[keep], data['means'][keep]
    cos_lat, variance = data['cos_lat'][keep], data['variance'][keep]
    if not len(edges):
        return np.zeros((3, 2)), np.zeros(3, dtype=int), np.zeros(3, dtype=int), True
    shifts, vote_counts, retained_counts = [], [], []
    shift = np.zeros(2)
    for _ in range(3):
        residual = means-shift
        cost = np.sqrt(np.maximum(0., variance+METRES_PER_DEGREE**2 *
                       (residual[:, 0]**2+(residual[:, 1]*cos_lat)**2)))
        chosen, weights = [], []
        for column in (0, 1):
            nodes = np.unique(edges[:, column])
            for node in nodes:
                candidates = np.flatnonzero(edges[:, column] == node)
                chosen.append(candidates[np.argmin(cost[candidates])])
                weights.append(.5/len(nodes))
        chosen, weights = np.asarray(chosen), np.asarray(weights)
        residuals = cost[chosen]
        median = np.median(residuals)
        mad = np.median(np.abs(residuals-median))
        retained = residuals <= median + 3*max(float(mad), 1.)
        indices, weights = chosen[retained], weights[retained]
        shift = np.array([_weighted_median(means[indices, axis], weights) for axis in (0, 1)])
        shifts.append(shift)
        vote_counts.append(len(chosen))
        retained_counts.append(int(retained.sum()))
    return np.asarray(shifts), np.asarray(vote_counts), np.asarray(retained_counts), False


def calibrate_scene(positions, sources, support):
    """Score every pair; each cross-source pair has an independently held-out fit."""
    positions, sources = np.asarray(positions, np.float64), np.asarray(sources)
    support = np.asarray(support)
    pairs = np.column_stack(np.triu_indices(len(sources), 1))
    data = _translation_data(positions, sources, support)
    baseline = rms_haversine_scores(positions, pairs)
    scores = {arm: baseline.copy() for arm in ARMS}
    shifts = np.zeros((len(pairs), 2, 2))
    fallback = np.zeros(len(pairs), bool)
    attempted = sources[pairs[:, 0]] != sources[pairs[:, 1]]
    vote_counts, retained_counts = np.zeros((len(pairs), 2), int), np.zeros((len(pairs), 2), int)
    latitude_clips = np.zeros(2, int)
    for index in np.flatnonzero(attempted):
        pair = pairs[index]
        updates, votes, retained, missing = estimate_translation(data, pair)
        shifts[index] = updates[[0, 2]]
        vote_counts[index], retained_counts[index] = votes[[0, 2]], retained[[0, 2]]
        fallback[index] = missing
        for arm_index, arm in enumerate(('once', 'three')):
            corrected = positions[pair].copy()
            source_two = sources[pair] == '9002'
            corrected[source_two] += shifts[index, arm_index]
            latitude_clips[arm_index] += int(np.count_nonzero(np.abs(corrected[..., 0]) > 90.))
            corrected[..., 0] = np.clip(corrected[..., 0], -90., 90.)
            corrected[..., 1] = (corrected[..., 1]+180.) % 360.-180.
            scores[arm][index] = rms_haversine_scores(corrected, np.array([[0, 1]]))[0]
    # Same-source edges retain exact baseline scores: a common sensor shift is
    # deliberately not allowed to improve or distort their within-source metric.
    return dict(pairs=pairs, scores=scores, shifts_degrees=shifts,
                attempted=attempted, fallback=fallback, vote_counts=vote_counts,
                retained_counts=retained_counts, latitude_clips=latitude_clips,
                eligible_calibration_edges=len(data['edges']))


def _summary(values):
    values = np.asarray(values)
    return dict(count=int(values.size), mean=float(np.mean(values)) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                p95=float(np.quantile(values, .95)) if values.size else None,
                maximum=float(np.max(values)) if values.size else None)


def load_records(predictions_dir, manifest):
    validate_manifest(manifest)
    directory = Path(predictions_dir)
    scene_ids = manifest['splits']['V_select']
    provenance = json.loads((directory.parent/'run.json').read_text())
    if (provenance['status'] != 'COMPLETED' or provenance['scene_ids'] != scene_ids
            or provenance['data'] != dict(length=64, max_observations=128, time_unit_seconds=20.)
            or provenance['inference_uses_reference'] or provenance['inference_uses_identity']
            or provenance['test_opened']):
        raise ValueError('requires completed native-only complete V cache with frozen settings')
    index = json.loads((directory/'index.json').read_text())
    if not scene_ids or [row['scene_id'] for row in index] != scene_ids:
        raise ValueError('prediction index must exactly match complete V_select')
    records = {f'{base}_{arm}': [] for base in BASES for arm in ARMS}
    payload, diagnostics = {'scene_ids': np.asarray(scene_ids)}, []
    for item in index:
        sid, filename = item['scene_id'], Path(item['path'])
        if filename.name != str(filename):
            raise ValueError('index must reference direct cache child files')
        with np.load(directory/filename, allow_pickle=False) as cache:
            if str(cache['scene_id'].item()) != sid:
                raise ValueError('cache scene mismatch')
            node_ids, sources = cache['node_ids'], cache['sources']
            n = len(node_ids)
            support, query = cache['region_inside_support'], cache['query_seconds']
            if (n != item['fragment_count'] or len(np.unique(node_ids)) != n or sources.shape != (n,)
                    or query.shape != (64,) or not np.isfinite(query).all() or not np.all(np.diff(query) > 0)):
                raise ValueError('cache population/query metadata mismatch')
            results = {base: calibrate_scene(cache[base], sources, support) for base in BASES}
            # Identity labels are read only after all inference paths finish.
            identities = cache['identities']
            if identities.shape != (n,):
                raise ValueError('identity population mismatch')
        pairs = results[BASES[0]]['pairs']
        labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
        common = dict(scene_id=sid, component_id=manifest['components'][sid], pairs=pairs,
                      labels=labels, sources=sources, node_ids=node_ids,
                      joint_support_queries=(support[pairs[:, 0]] & support[pairs[:, 1]]).sum(axis=1))
        for key in ('pairs', 'labels', 'sources', 'node_ids', 'joint_support_queries'):
            payload[f'{sid}_{key}'] = common[key]
        for base, result in results.items():
            for arm in ARMS:
                name = f'{base}_{arm}'
                records[name].append(dict(common, scores=result['scores'][arm]))
                payload[f'{sid}_{name}_scores'] = result['scores'][arm]
            for key in ('shifts_degrees', 'attempted', 'fallback', 'vote_counts', 'retained_counts'):
                payload[f'{sid}_{base}_{key}'] = result[key]
            cross = result['attempted']
            # Magnitude for reporting only. Mean latitude of scored nodes does
            # not enter the leave-two-out transformation estimator.
            shifts = result['shifts_degrees'][cross].copy()
            with np.load(directory/filename, allow_pickle=False) as cache:
                lat = cache[base][..., 0].mean(axis=1)
            report_cos = np.cos(np.deg2rad(lat[pairs[cross]].mean(axis=1)))
            shift_norm = METRES_PER_DEGREE*np.sqrt(shifts[..., 0]**2 +
                                                 (shifts[..., 1]*report_cos[:, None])**2)
            diagnostics.append(dict(scene_id=sid, base=base, pairs=len(pairs),
                calibration_attempts=int(cross.sum()), no_estimate_fallbacks=int(result['fallback'].sum()),
                eligible_calibration_edges=result['eligible_calibration_edges'],
                latitude_clips=result['latitude_clips'].tolist(),
                shift_magnitude_m={arm: _summary(shift_norm[:, k]) for k, arm in enumerate(('once', 'three'))}))
    payload['component_ids'] = np.asarray([manifest['components'][sid] for sid in scene_ids])
    return records, payload, diagnostics


def _group_report(record, threshold):
    """Repository anchor_direct/input partition, not every accepted-edge star."""
    n = len(record['sources'])
    groups = group_predicted_tracks(n, record['pairs'], record['scores'],
        threshold=threshold, grouping_policy='anchor_direct', node_order='input')
    truth = np.eye(n, dtype=bool)
    pairs = record['pairs']
    truth[pairs[:, 0], pairs[:, 1]] = record['labels']
    truth[pairs[:, 1], pairs[:, 0]] = record['labels']
    links = wrong = mixed_groups = mixed_anchors = anchors_with_helpers = 0
    for group in groups:
        errors = ~truth[np.ix_(group, group)]
        links += len(group)*(len(group)-1)
        wrong += int(errors.sum())
        mixed_groups += int(errors.any())
        mixed_anchors += int(errors.any(axis=1).sum())
        anchors_with_helpers += len(group) if len(group) > 1 else 0
    return dict(scene_id=record['scene_id'], grouping_policy='anchor_direct', node_order='input',
        group_count=len(groups), singleton_count=sum(len(group) == 1 for group in groups),
        mixed_identity_groups=mixed_groups, groups=groups,
        helper_links=links, wrong_helper_links=wrong,
        wrong_helper_fraction=wrong/links if links else None,
        anchors_with_helpers=anchors_with_helpers, mixed_anchors=mixed_anchors,
        mixed_fraction_of_all_anchors=mixed_anchors/n if n else None)


def run(predictions_dir, manifest_path, output_dir, command=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='RUNNING', command=command, device='cpu', optimizer_updates=0,
        predictions_dir=str(Path(predictions_dir).resolve()), manifest=str(Path(manifest_path).resolve()),
        inference_uses_identity=False, inference_uses_reference=False, position_targets_read=False,
        test_opened=False, training_data_read=False)
    _write(output/'run.json', state)
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        records, payload, diagnostics = load_records(predictions_dir, manifest)
        selections = {name: select_threshold(rows) for name, rows in records.items()}
        methods = {}
        for name, rows in records.items():
            base = next(base for base in BASES if name.startswith(base+'_'))
            baseline = records[f'{base}_uncalibrated']
            fixed = selections[f'{base}_uncalibrated']['threshold']
            own = selections[name]['threshold']
            methods[name] = dict(selection=selections[name], auc=_auc_summary(rows),
                subgroup_auc={kind: _auc_summary(_subset(rows, kind)) for kind in
                              ('itsa', 'mstsa', 'joint_q_support', 'no_joint_q_support')},
                own_threshold=evaluate_scenes(rows, own), fixed_uncalibrated_threshold=evaluate_scenes(rows, fixed),
                changes_vs_uncalibrated={key: compare_predictions(baseline, rows, fixed, threshold)
                    for key, threshold in [('own_threshold', own), ('fixed_uncalibrated_threshold', fixed)]},
                grouping={key: [_group_report(row, threshold) for row in rows]
                    for key, threshold in [('own_threshold', own), ('fixed_uncalibrated_threshold', fixed)]})
        base_rows = next(iter(records.values()))
        population = dict(scenes=len(base_rows), components=len({r['component_id'] for r in base_rows}),
            nodes=sum(len(r['node_ids']) for r in base_rows), pairs=sum(len(r['pairs']) for r in base_rows),
            positives=sum(int(r['labels'].sum()) for r in base_rows))
        report = dict(**state, population=population, scene_ids=manifest['splits']['V_select'],
            algorithm='source 9001 anchor; all cross-source pairs with >=1 joint native-support Q; '
                      'per scored cross-source edge exclude both nodes; initialize zero translation; '
                      'each available node votes nearest cross-source neighbour using RMS local tangent distance; '
                      'source vote mass .5 each; trim above unweighted vote median+3*max(MAD,1m); '
                      'coordinate weighted medians of pair mean geographic offsets; save updates 1 and 3',
            scoring='negative RMS haversine at all Q64; same-source scores exactly unchanged; all pairs retained',
            fallback='zero translation if no cross-source jointly supported pair remains after two-node exclusion',
            selection='each arm exact scene-macro F1 on same full V; also fixed uncalibrated base threshold',
            limitations=['reused V development evaluation; not held-out performance or established innovation',
                'full-fragment offline geometry control, no learned component or one-to-one global matching',
                'source 9001 fixes a coordinate anchor, not known absolute truth; shared absolute bias is unidentified',
                'excluded nodes are not excluded identities; other fragments of the same identity may remain',
                'pair-specific leave-two-out translations do not define one consistent scene coordinate transform',
                'one or few contributing nodes can still yield an estimate; fallback rate is not a confidence guarantee',
                'source-level translation cannot directly repair same-source errors; graph closure may amplify cross-source errors',
                'fitting uses local tangent approximation; final scoring uses geographic haversine'],
            methods=methods, calibration_diagnostics=diagnostics)
        np.savez_compressed(output/'scores.npz', **payload)
        report.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        _write(output/'summary.json', report)
        state.update(status='COMPLETED', population=population, elapsed_seconds=report['elapsed_seconds'])
        _write(output/'run.json', state)
        return report
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        _write(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('predictions-dir', 'manifest', 'output-dir'):
        parser.add_argument('--'+option, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    result = run(args.predictions_dir, args.manifest, args.output_dir, command)
    print(json.dumps(dict(status=result['status'], population=result['population'],
        elapsed_seconds=result['elapsed_seconds'], macro_f1={name: row['own_threshold']['macro_f1']
        for name, row in result['methods'].items()})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
