"""R4: four fixed ordinary fusion controls on complete frozen V, with no training.

Unit: every original fragment is an anchor, evaluated against its own identity's
cached reference on Q64. Groups are fixed oracle identities (diagnostic only) or
existing anchor_direct at the already selected native_OLS threshold. Both use
uncalibrated native OLS positions and native speed/course interpolation + hold.
Arms, fixed before looking at results: own anchor; nearest real observation at
each query (ties follow input order); equal mean of the full group; equal mean
of members whose measured endpoint span contains the query, else own anchor.
Latitude/speed use arithmetic means, longitude/course circular means; ambiguous
zero resultant falls back to the anchor and is counted. No source preference,
reference-based member selection, threshold search, filtering or T_test reads.

Run: .venv/bin/python tools/analysis/fixed_group_fusion_control.py
  --predictions-dir runs/NATIVE-MOTION-20261020-R2-local/predictions
  --association-dir runs/MOTION-ASSOCIATION-20261020-local
  --manifest /tmp/mtad-r1-r2-20261020-manifest.json
  --archive /Users/a1/Downloads/数据集level2.zip
  --output-dir runs/FIXED-GROUP-FUSION-20261020-local
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

from association.data.archive import MTADArchive, validate_manifest
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.evaluation.native_motion import _short_delta
from association.evaluation.physical import evaluate_aligned_fusion, group_membership_diagnostics
from tools.analysis.motion_association_control import rms_haversine_scores

ARMS = ('anchor', 'nearest_observation', 'mean_all', 'mean_supported')
GROUPINGS = ('oracle_identity', 'predicted_anchor_direct')
DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def complete_native_attributes(native_values, native_times, native_observed, query, positions):
    """Only visible native attributes enter predictions; no truth is accepted.

    MTAD canonicalization quarantines any track with an invalid attribute. Check
    those attributes again explicitly; a true observation bit alone is not proof
    that speed or course is legal. There is no invented value for missing course.
    """
    values, times, visible = map(np.asarray, (native_values, native_times, native_observed))
    query, positions = np.asarray(query), np.asarray(positions)
    if (values.ndim != 3 or values.shape[-1] != 4 or times.shape != values.shape[:2]
            or visible.shape != times.shape or visible.dtype != bool
            or positions.shape != (len(values), len(query), 2)):
        raise ValueError('native state, time, observation and position shapes differ')
    selected = values[visible]
    if (not np.isfinite(selected).all() or (selected[:, 2] < 0).any()
            or (selected[:, 3] < 0).any() or (selected[:, 3] > 360).any()):
        raise ValueError('visible native speed/course must independently be legal')
    result = np.empty((len(values), len(query), 4), np.float64)
    result[..., :2] = positions
    for node in range(len(values)):
        t, state = times[node, visible[node]], values[node, visible[node]]
        if len(t) < 2 or not np.isfinite(t).all() or (np.diff(t) <= 0).any():
            raise ValueError('each anchor needs two increasing visible native times')
        right = np.searchsorted(t, query).clip(1, len(t)-1)
        left = right-1
        fraction = ((query-t[left])/(t[right]-t[left])).clip(0, 1)
        result[node, :, 2] = state[left, 2]+fraction*(state[right, 2]-state[left, 2])
        result[node, :, 3] = (state[left, 3]+fraction*_short_delta(state[right, 3]-state[left, 3])) % 360.
    if not np.isfinite(result).all():
        raise ValueError('complete predictions must be finite on every retained query')
    return result


def ordinary_fusion(states, nearest_seconds, support, groups):
    """Prediction-only four-arm calculation: groups are supplied, never inferred from truth."""
    states = np.asarray(states, np.float64)
    nearest, support = np.asarray(nearest_seconds), np.asarray(support)
    n, q, width = states.shape
    if (width != 4 or nearest.shape != (n, q) or support.shape != (n, q)
            or support.dtype != bool or not np.isfinite(states).all()
            or not np.isfinite(nearest).all() or (nearest < 0).any()):
        raise ValueError('invalid complete states/visible-support arrays')
    flat = [node for group in groups for node in group]
    if sorted(flat) != list(range(n)) or any(group != sorted(group) or not group for group in groups):
        raise ValueError('fixed groups must partition every node once in input order')
    predictions = {arm: states.copy() for arm in ARMS}
    selected_nodes = np.empty((n, q), np.int64)
    helper_support = np.zeros((n, q), bool)
    membership = np.empty(n, np.int64)
    audit = {arm: dict(longitude_zero_resultant=0, heading_zero_resultant=0,
                      no_supported_member_fallback=0) for arm in ARMS}
    for gid, group in enumerate(groups):
        members = np.asarray(group, np.int64)
        membership[members] = gid
        selected = members[np.argmin(nearest[members], axis=0)]
        for anchor in members:
            selected_nodes[anchor] = selected
            predictions['nearest_observation'][anchor] = states[selected, np.arange(q)]
            helper_support[anchor] = support[members[members != anchor]].any(0)
        for arm in ('mean_all', 'mean_supported'):
            weights = np.ones((len(members), q), bool) if arm == 'mean_all' else support[members]
            count = weights.sum(0)
            divisor = np.maximum(count, 1)
            mean = np.zeros((q, 4))
            for column in (0, 2):
                mean[:, column] = (states[members, :, column]*weights).sum(0)/divisor
            ambiguous = {}
            for column in (1, 3):
                radians = np.deg2rad(states[members, :, column])
                sine = (np.sin(radians)*weights).sum(0)/divisor
                cosine = (np.cos(radians)*weights).sum(0)/divisor
                ambiguous[column] = np.hypot(sine, cosine) <= 1e-12
                angle = np.rad2deg(np.arctan2(sine, cosine))
                mean[:, column] = (angle+180.) % 360.-180. if column == 1 else angle % 360.
            for anchor in members:
                output = mean.copy()
                output[count == 0] = states[anchor, count == 0]
                for column, name in ((1, 'longitude'), (3, 'heading')):
                    fallback = ambiguous[column] & (count > 0)
                    output[fallback, column] = states[anchor, fallback, column]
                    audit[arm][name+'_zero_resultant'] += int(fallback.sum())
                audit[arm]['no_supported_member_fallback'] += int((count == 0).sum())
                predictions[arm][anchor] = output
    return dict(predictions=predictions, selected_nodes=selected_nodes,
                helper_support=helper_support, membership=membership, audit=audit)


def attribute_errors(prediction, target):
    """Errors before masking, used only after all predictions have been fixed."""
    lat1, lat2 = np.deg2rad(prediction[..., 0]), np.deg2rad(target[..., 0])
    dlon = np.deg2rad((prediction[..., 1]-target[..., 1]+180.) % 360.-180.)
    a = np.clip(np.sin((lat1-lat2)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0, 1)
    return dict(position=2*6371000.*np.arctan2(np.sqrt(a), np.sqrt(1-a)),
                speed=np.abs(prediction[..., 2]-target[..., 2]),
                heading=np.abs((prediction[..., 3]-target[..., 3]+180.) % 360.-180.))


def metrics(prediction, target, valid, region):
    result = evaluate_aligned_fusion(prediction, target, valid,
        prediction_valid=np.ones(prediction.shape, bool), region_mask=region)
    errors = attribute_errors(prediction, target)
    for name, columns in (('position', [0, 1]), ('speed', [2]), ('heading', [3])):
        mask = valid[..., columns].all(-1) & region
        selected = errors[name][mask]
        result[name]['p95'] = float(np.percentile(selected, 95)) if len(selected) else None
    return {name: result[name] for name in ('position', 'speed', 'heading')}


def run(predictions_dir, association_dir, manifest_path, archive_path, output_dir, command=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'predictions').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', command=command, optimizer_updates=0, device='cpu',
        test_opened=False, calibration='none', new_reference_csv_reads=False,
        cached_reference_used_only_for_evaluation=True, threshold_search=False,
        inference_uses_identity='oracle grouping only; predicted path does not',
        design=__doc__, completed_scenes=0)
    write_json(output/'run.json', state)
    try:
        directory, association_dir = Path(predictions_dir), Path(association_dir)
        manifest = json.loads(Path(manifest_path).read_text())
        validate_manifest(manifest)
        scene_ids = manifest['splits']['V_select']
        provenance = json.loads((directory.parent/'run.json').read_text())
        index = json.loads((directory/'index.json').read_text())
        assoc = json.loads((association_dir/'summary.json').read_text())
        if (not scene_ids or [row['scene_id'] for row in index] != scene_ids
                or provenance['status'] != 'COMPLETED' or provenance['scene_ids'] != scene_ids
                or provenance['data'] != DATA or provenance['test_opened']
                or provenance['inference_uses_reference'] or provenance['inference_uses_identity']
                or assoc['test_opened'] or assoc['status'] != 'COMPLETED' or assoc['scene_ids'] != scene_ids):
            raise ValueError('cache and association run must match the complete frozen V')
        threshold = assoc['methods']['native_ols']['selection']['threshold']
        state['fixed_threshold'] = threshold
        rows, scene_reports, anchor_reports = [], [], []
        with MTADArchive(archive_path) as archive, np.load(association_dir/'scores.npz', allow_pickle=False) as scores:
            for item in index:
                sid = item['scene_id']
                filename = Path(item['path'])
                if filename.name != str(filename):
                    raise ValueError('cache filenames must stay in prediction directory')
                with np.load(directory/filename, allow_pickle=False) as cache:
                    native = _record(archive.load_scene(sid), manifest['components'][sid], 'V_select', 128)
                    scene = prepare_timed_scene(native, **DATA)
                    for key, actual in (('node_ids', scene['node_ids']), ('sources', scene['sources']),
                            ('query_seconds', scene['grid']), ('native_times', scene['native_times']),
                            ('native_observed', scene['native_observed']), ('native_values', scene['native_values'][..., :2])):
                        if not np.array_equal(cache[key], actual):
                            raise ValueError('native/cached population or input mismatch: '+sid+' '+key)
                    n = len(scene['node_ids'])
                    states = complete_native_attributes(scene['native_values'], scene['native_times'],
                        scene['native_observed'], scene['grid'], cache['native_ols'])
                    pairs = np.column_stack(np.triu_indices(n, 1))
                    pair_scores = rms_haversine_scores(states[..., :2], pairs)
                    if not np.array_equal(pairs, scores[sid+'_pairs']) or not np.allclose(
                            pair_scores, scores[sid+'_native_ols_scores'], atol=1e-7, rtol=0):
                        raise ValueError('original fixed association scores changed')
                    predicted_groups = group_predicted_tracks(n, pairs, pair_scores, threshold=threshold,
                        grouping_policy='anchor_direct', node_order='input')
                    support, nearest = cache['region_inside_support'], cache['nearest_seconds']
                    predicted = ordinary_fusion(states, nearest, support, predicted_groups)
                    # Identities enter only after anonymous predictions exist.
                    identities = cache['identities']
                    if not np.array_equal(identities, scene['identities']):
                        raise ValueError('evaluation identity population mismatch')
                    oracle_groups = group_tracks_by_identity(identities, node_order='input')
                    oracle = ordinary_fusion(states, nearest, support, oracle_groups)
                    target, valid = cache['target'], cache['target_valid']
                    regions = {name: cache['region_'+name] for name in
                               ('all', 'inside_support', 'outside_support', 'exact_observation', 'inside_nonobserved')}
                    regions['far_gt_1800s'] = nearest > 1800.
                groups = dict(zip(GROUPINGS, (oracle_groups, predicted_groups)))
                outputs = dict(zip(GROUPINGS, (oracle, predicted)))
                row = dict(target=target, valid=valid, regions=regions, outputs=outputs,
                           sources=np.asarray(scene['sources']))
                local = dict(scene_id=sid, component_id=manifest['components'][sid],
                    fragments=n, query_count=n*64, groups={}, methods={})
                payload = dict(node_ids=np.asarray(scene['node_ids']), identities=identities,
                    sources=np.asarray(scene['sources']), query_seconds=scene['grid'],
                    target=target, target_valid=valid, nearest_seconds=nearest, native_states=states,
                    **{'region_'+key: value for key, value in regions.items()})
                for grouping, result in outputs.items():
                    membership = result['membership']
                    mixed = np.zeros(n, bool)
                    wrong_helpers = np.zeros(n, np.int64)
                    helpers = np.zeros(n, np.int64)
                    for group in groups[grouping]:
                        mixed[group] = len(set(identities[group])) > 1
                        for anchor in group:
                            wrong_helpers[anchor] = (identities[group] != identities[anchor]).sum()
                            helpers[anchor] = len(group)-1
                    result['mixed_group'] = mixed
                    result['own_missing_helper_supported'] = ~support & result['helper_support']
                    selected_wrong = identities[result['selected_nodes']] != identities[:, None]
                    local['groups'][grouping] = dict(diagnostics=group_membership_diagnostics(membership, identities),
                        mixed_group_anchors=int(mixed.sum()), helper_links=int(helpers.sum()),
                        wrong_helper_links=int(wrong_helpers.sum()),
                        nearest_selection_wrong_identity_queries=int(selected_wrong.sum()),
                        own_missing_helper_supported_queries=int(result['own_missing_helper_supported'].sum()),
                        numerical_fallbacks=result['audit'], groups=groups[grouping])
                    local['methods'][grouping] = {arm: metrics(prediction, target, valid, regions['all'])
                                                 for arm, prediction in result['predictions'].items()}
                    payload[grouping+'_membership'] = membership
                    payload[grouping+'_selected_nodes'] = result['selected_nodes']
                    for arm, prediction in result['predictions'].items():
                        payload[grouping+'_'+arm] = prediction
                for node in range(n):
                    take = slice(node, node+1)
                    anchor_reports.append(dict(scene_id=sid, node_id=scene['node_ids'][node],
                        identity=str(identities[node]), source=str(scene['sources'][node]),
                        methods={grouping: {arm: metrics(prediction[take], target[take], valid[take], regions['all'][take])
                            for arm, prediction in result['predictions'].items()} for grouping, result in outputs.items()}))
                np.savez_compressed(output/'predictions'/('states-'+sid+'.npz'), **payload)
                rows.append(row)
                scene_reports.append(local)
                state.update(completed_scenes=len(rows), elapsed_seconds=time.monotonic()-started)
                write_json(output/'run.json', state)
            state['archive_members_read'] = list(archive.members_read)
        target = np.concatenate([row['target'] for row in rows])
        valid = np.concatenate([row['valid'] for row in rows])
        sources = np.concatenate([row['sources'] for row in rows])
        regions = {name: np.concatenate([row['regions'][name] for row in rows]) for name in rows[0]['regions']}
        overall = {}
        for grouping in GROUPINGS:
            extra = dict(mixed_group=np.broadcast_to(np.concatenate([r['outputs'][grouping]['mixed_group'] for r in rows])[:, None], valid.shape[:2]),
                own_missing_helper_supported=np.concatenate([r['outputs'][grouping]['own_missing_helper_supported'] for r in rows]))
            all_regions = dict(regions, **extra, pure_group=~extra['mixed_group'])
            overall[grouping] = {}
            for arm in ARMS:
                prediction = np.concatenate([r['outputs'][grouping]['predictions'][arm] for r in rows])
                overall[grouping][arm] = dict(regions={name: metrics(prediction, target, valid, region)
                    for name, region in all_regions.items()}, by_source={str(source): {name: metrics(
                        prediction[sources == source], target[sources == source], valid[sources == source], region[sources == source])
                        for name, region in regions.items()} for source in sorted(set(sources))})
        report = dict(**state, scene_count=len(rows), component_count=len({r['component_id'] for r in scene_reports}),
            fragment_count=len(target), query_count=len(target)*64, target_valid_by_attribute=valid.sum((0, 1)).tolist(),
            coverage_scope='all retained fragment anchors x Q64, identical masks in every arm; not unique-target population',
            limitations=['V64 is repeatedly exposed development data, 17 related components, not independent confirmation',
                'oracle groups diagnose available helper information, not deployable grouping',
                'no calibration or learned fusion; fixed direct grouping is evaluated, no connected closure',
                '9001/reference reuse prevents an independent denoising claim',
                'support is the native endpoint span, not a guarantee that its interpolation is reliable',
                'each fragment is an anchor; identities with more fragments have greater point-pooled weight'],
            overall=overall, per_scene=scene_reports, per_anchor=anchor_reports)
        report.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        write_json(output/'summary.json', report)
        state.update(status='COMPLETED', fragment_count=len(target), query_count=len(target)*64,
                     elapsed_seconds=report['elapsed_seconds'])
        write_json(output/'run.json', state)
        return report
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.monotonic()-started)
        write_json(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions-dir', 'association-dir', 'manifest', 'archive', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    result = run(args.predictions_dir, args.association_dir, args.manifest, args.archive, args.output_dir, command)
    print(json.dumps(dict(status=result['status'], fragments=result['fragment_count'],
        all_mae={group: {arm: {attribute: result['overall'][group][arm]['regions']['all'][attribute]['mae']
        for attribute in ('position', 'speed', 'heading')} for arm in ARMS} for group in GROUPINGS})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
