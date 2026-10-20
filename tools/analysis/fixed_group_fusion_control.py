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
from association.data.motion_evidence import complete_native_attributes
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.evaluation.physical import group_membership_diagnostics
from association.evaluation.fusion_controls import ordinary_fusion, attribute_errors, metrics, rms_haversine_scores

ARMS = ('anchor', 'nearest_observation', 'mean_all', 'mean_supported')
GROUPINGS = ('oracle_identity', 'predicted_anchor_direct')
DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


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
