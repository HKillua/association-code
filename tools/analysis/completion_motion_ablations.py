"""Two fixed, zero-training native OLS completion diagnostics on complete V.

The fitted-anchor arm tests endpoint error, but may be discontinuous at support
boundaries. The local-half arm tests full-fragment constant-velocity mismatch.
Neither changes a production default or searches windows on validation data.
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
from association.evaluation.association import compare_predictions, evaluate_scenes, select_threshold
from association.evaluation.native_motion import (
    DISTANCE_REGIONS, REGIONS, _short_delta, position_metrics,
    predict_native_positions, summarize_positions,
)
from tools.analysis.motion_association_control import _auc_summary, _subset, rms_haversine_scores


METHODS = ('native_ols', 'ols_fit_anchor', 'ols_local_half')
DATA = dict(length=64, max_observations=128, time_unit_seconds=20.)


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _distances(first, second):
    """Spherical metres, matching the repository physical metric."""
    lat1, lat2 = np.deg2rad(first[..., 0]), np.deg2rad(second[..., 0])
    dlon = np.deg2rad((first[..., 1]-second[..., 1]+180.) % 360.-180.)
    hav = np.clip(np.sin((lat1-lat2)/2)**2
                  + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2, 0., 1.)
    return 2*6371000.*np.arctan2(np.sqrt(hav), np.sqrt(1.-hav))


def _fit(times, positions):
    centred = times-times.mean()
    slope = (centred[:, None]*(positions-positions.mean(0))).sum(0)/(centred @ centred)
    return slope, times.mean(), positions.mean(0)


def predict_ablations(native_values, native_times, native_observed, query_seconds):
    """Accept only native positions/times/prefix masks and query times.

    All arms retain the original exact observations and inside interpolation.
    Outside, fitted-anchor uses the all-native fitted boundary position; local
    half uses separate first/last K=min(N,max(3,ceil(N/2))) point slopes anchored
    to the measured endpoints. Two-point tracks exactly retain native OLS.
    """
    base = predict_native_positions(native_values, native_times, native_observed, query_seconds)
    values = np.asarray(native_values, dtype=np.float64)[..., :2]
    times, query = np.asarray(native_times, dtype=np.float64), np.asarray(query_seconds, dtype=np.float64)
    predictions = {METHODS[0]: base['predictions']['native_ols']}
    predictions.update({name: base['predictions']['native_hold'].copy() for name in METHODS[1:]})
    sizes = base['native_point_counts']
    boundary_offsets = np.zeros((len(sizes), 2), np.float64)
    boundary_clamps = np.zeros((len(sizes), 2), bool)
    local_sizes = np.minimum(sizes, np.maximum(3, (sizes+1)//2))
    local_slopes = np.empty((len(sizes), 2, 2), np.float64)
    for node, size in enumerate(sizes):
        t, xy = times[node, :size], values[node, :size]
        unwrapped = xy.copy()
        unwrapped[:, 1] = xy[0, 1]+np.r_[0., np.cumsum(_short_delta(np.diff(xy[:, 1])))]
        slope, centre_t, centre_xy = _fit(t, unwrapped)
        k = local_sizes[node]
        local_slopes[node, 0] = _fit(t[:k], unwrapped[:k])[0]
        local_slopes[node, 1] = _fit(t[-k:], unwrapped[-k:])[0]
        if size == 2:
            for name in METHODS[1:]:
                predictions[name][node] = predictions['native_ols'][node]
            continue
        for side, region in enumerate(('before_support', 'after_support')):
            endpoint = 0 if side == 0 else size-1
            fitted_boundary = centre_xy+(t[endpoint]-centre_t)*slope
            legal_boundary = fitted_boundary.copy()
            boundary_clamps[node, side] = abs(legal_boundary[0]) > 90.
            legal_boundary[0] = np.clip(legal_boundary[0], -90., 90.)
            legal_boundary[1] = (legal_boundary[1]+180.) % 360.-180.
            boundary_offsets[node, side] = _distances(legal_boundary, xy[endpoint])
            selected = base['regions'][region][node]
            delta = (query[selected]-t[endpoint])[:, None]
            predictions['ols_fit_anchor'][node, selected] = fitted_boundary+delta*slope
            predictions['ols_local_half'][node, selected] = xy[endpoint]+delta*local_slopes[node, side]
    outside = base['regions']['outside_support']
    clamps = {'native_ols': base['latitude_clamp_counts']['native_ols']}
    for name in METHODS[1:]:
        prediction = predictions[name]
        clamps[name] = ((np.abs(prediction[..., 0]) > 90.) & outside).sum(1)
        # The two-point baseline is already clipped, so preserve its counts.
        clamps[name][sizes == 2] = clamps['native_ols'][sizes == 2]
        prediction[..., 0][outside] = np.clip(prediction[..., 0][outside], -90., 90.)
        prediction[..., 1][outside] = (prediction[..., 1][outside]+180.) % 360.-180.
        if not np.isfinite(prediction).all():
            raise FloatingPointError('completion diagnostic produced nonfinite positions')
    return dict(predictions=predictions, regions=base['regions'],
        nearest_seconds=base['nearest_seconds'], native_point_counts=sizes,
        local_fit_point_counts=local_sizes, local_slopes_degrees_per_second=local_slopes,
        latitude_clamp_counts=clamps, fitted_boundary_offsets_metres=boundary_offsets,
        fitted_boundary_latitude_clamped=boundary_clamps)


def _summary(predictions, target, target_valid, regions, nearest_seconds):
    result = summarize_positions(predictions, target, target_valid, regions, nearest_seconds)
    nearest = nearest_seconds
    distances = dict(zip(DISTANCE_REGIONS, (
        nearest <= 300., (nearest > 300.) & (nearest <= 600.),
        (nearest > 600.) & (nearest <= 1800.),
        (nearest > 1800.) & np.isfinite(nearest), ~np.isfinite(nearest))))
    valid = target_valid[..., :2].all(-1)
    for name, prediction in predictions.items():
        for category, partitions in (('regions', regions), ('input_distance_metrics', distances)):
            for key, mask in partitions.items():
                take = valid & mask
                errors = _distances(prediction[take], target[..., :2][take])
                result[name][category][key]['p95'] = float(np.quantile(errors, .95)) if errors.size else None
    return result


def _changes(rows, key):
    result = {}
    for method in METHODS[1:]:
        deltas = [row[key][method]['mae']-row[key]['native_ols']['mae'] for row in rows
                  if row[key]['native_ols']['mae'] is not None]
        delta = np.asarray(deltas)
        result[method] = dict(improved=int((delta < -1e-9).sum()),
            worse=int((delta > 1e-9).sum()), tied=int((np.abs(delta) <= 1e-9).sum()),
            undefined=len(rows)-len(delta), equality_tolerance_metres=1e-9)
    return result


def run(predictions_dir, manifest_path, output_dir, command=None):
    output, directory = Path(output_dir), Path(predictions_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'predictions').mkdir()
    started = time.monotonic()
    state = dict(status='RUNNING', command=command, predictions_dir=str(directory.resolve()),
        manifest=str(Path(manifest_path).resolve()), device='cpu', optimizer_updates=0,
        test_opened=False, training_data_read=False, inference_uses_identity=False,
        inference_uses_reference=False, evaluation_reads_cached_position_targets=True)
    _write(output/'run.json', state)
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        validate_manifest(manifest)
        scene_ids = manifest['splits']['V_select']
        provenance = json.loads((directory.parent/'run.json').read_text())
        index = json.loads((directory/'index.json').read_text())
        if (not scene_ids or provenance['status'] != 'COMPLETED'
                or provenance['scene_ids'] != scene_ids or provenance['data'] != DATA
                or provenance['inference_uses_reference'] or provenance['inference_uses_identity']
                or provenance['test_opened'] or [row['scene_id'] for row in index] != scene_ids):
            raise ValueError('require completed native-only R2 cache and exact full frozen V population/settings')
        records = {name: [] for name in METHODS}
        rows, scenes, tracks, saved_index = [], [], [], []
        score_payload = dict(scene_ids=np.asarray(scene_ids),
            component_ids=np.asarray([manifest['components'][sid] for sid in scene_ids]))
        for item in index:
            sid, filename = item['scene_id'], Path(item['path'])
            if filename.name != str(filename):
                raise ValueError('cache paths must name local prediction files')
            with np.load(directory/filename, allow_pickle=False) as cache:
                if str(cache['scene_id'].item()) != sid:
                    raise ValueError('cached scene_id differs from index')
                query = cache['query_seconds']
                if query.shape != (64,):
                    raise ValueError('require frozen Q64 grid')
                control = predict_ablations(cache['native_values'], cache['native_times'],
                                            cache['native_observed'], query)
                predictions = control['predictions']
                if not np.array_equal(predictions['native_ols'], cache['native_ols']):
                    raise ValueError('recomputed native_ols differs from original cache')
                n = len(predictions['native_ols'])
                node_ids, sources = cache['node_ids'], cache['sources']
                if (n != item['fragment_count'] or node_ids.shape != (n,)
                        or len(np.unique(node_ids)) != n or sources.shape != (n,)
                        or not np.array_equal(control['regions']['inside_support'], cache['region_inside_support'])):
                    raise ValueError('cache population or support mismatch')
                pairs = np.column_stack(np.triu_indices(n, 1))
                scores = {name: rms_haversine_scores(prediction, pairs) for name, prediction in predictions.items()}
                # Prediction and all association scores exist before identities or targets enter.
                identities, target, target_valid = cache['identities'], cache['target'], cache['target_valid']
                if identities.shape != (n,) or target.shape != (n, 64, 4):
                    raise ValueError('evaluation cache population mismatch')
            labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
            support = control['regions']['inside_support']
            common = dict(scene_id=sid, component_id=manifest['components'][sid], pairs=pairs,
                labels=labels, node_ids=node_ids, sources=sources,
                joint_support_queries=(support[pairs[:, 0]] & support[pairs[:, 1]]).sum(1))
            for name in METHODS:
                records[name].append(dict(common, scores=scores[name]))
                score_payload[f'{sid}_{name}_scores'] = scores[name]
            for key in ('pairs', 'labels', 'node_ids', 'sources', 'joint_support_queries'):
                score_payload[f'{sid}_{key}'] = common[key]
            row = dict(predictions=predictions, target=target, target_valid=target_valid,
                regions=control['regions'], nearest_seconds=control['nearest_seconds'], sources=sources)
            rows.append(row)
            methods = _summary(**{k: v for k, v in row.items() if k != 'sources'})
            scenes.append(dict(scene_id=sid, component_id=common['component_id'], fragment_count=n,
                methods=methods, all_position={name: value['regions']['all'] for name, value in methods.items()},
                outside_position={name: value['regions']['outside_support'] for name, value in methods.items()}))
            for node in range(n):
                selected = slice(node, node+1)
                metrics = {name: position_metrics(prediction[selected], target[selected], target_valid[selected],
                    control['regions']['all'][selected]) for name, prediction in predictions.items()}
                outside_metrics = {name: position_metrics(prediction[selected], target[selected], target_valid[selected],
                    control['regions']['outside_support'][selected]) for name, prediction in predictions.items()}
                tracks.append(dict(scene_id=sid, node_id=str(node_ids[node]), source=str(sources[node]),
                    native_point_count=int(control['native_point_counts'][node]),
                    local_fit_point_count=int(control['local_fit_point_counts'][node]),
                    query_support_count=int(support[node].sum()),
                    fitted_boundary_offsets_metres=control['fitted_boundary_offsets_metres'][node].tolist(),
                    fitted_boundary_latitude_clamped=control['fitted_boundary_latitude_clamped'][node].tolist(),
                    latitude_clamp_counts={name: int(value[node]) for name, value in control['latitude_clamp_counts'].items()},
                    methods=metrics, outside_position=outside_metrics))
            saved = 'positions-'+sid+'.npz'
            np.savez_compressed(output/'predictions'/saved, scene_id=np.asarray(sid),
                node_ids=node_ids, sources=sources, identities=identities, query_seconds=query, **predictions)
            saved_index.append(dict(scene_id=sid, path=saved, fragment_count=n))
            state.update(completed_scenes=len(rows), elapsed_seconds=time.monotonic()-started)
            _write(output/'run.json', state)
        pooled = dict(predictions={name: np.concatenate([row['predictions'][name] for row in rows]) for name in METHODS},
            target=np.concatenate([row['target'] for row in rows]),
            target_valid=np.concatenate([row['target_valid'] for row in rows]),
            regions={name: np.concatenate([row['regions'][name] for row in rows]) for name in REGIONS},
            nearest_seconds=np.concatenate([row['nearest_seconds'] for row in rows]))
        sources = np.concatenate([row['sources'] for row in rows])
        by_source = {}
        for source in sorted(set(sources)):
            take = sources == source
            by_source[str(source)] = _summary(
                {name: value[take] for name, value in pooled['predictions'].items()},
                pooled['target'][take], pooled['target_valid'][take],
                {name: value[take] for name, value in pooled['regions'].items()}, pooled['nearest_seconds'][take])
        selections = {name: select_threshold(record) for name, record in records.items()}
        fixed = selections['native_ols']['threshold']
        association = {}
        for name, record in records.items():
            own = selections[name]['threshold']
            association[name] = dict(selection=selections[name], auc=_auc_summary(record),
                own_threshold=evaluate_scenes(record, own), ols_threshold=evaluate_scenes(record, fixed),
                subgroup_auc={kind: _auc_summary(_subset(record, kind)) for kind in
                    ('itsa', 'mstsa', 'joint_q_support', 'no_joint_q_support')})
            if name != 'native_ols':
                association[name]['changes_vs_ols'] = {
                    label: compare_predictions(records['native_ols'], record, fixed, threshold)
                    for label, threshold in (('own_threshold', own), ('ols_threshold', fixed))}
        offsets = np.asarray([row['fitted_boundary_offsets_metres'] for row in tracks]).reshape(-1, 2)
        boundary = {side: dict(median=float(np.median(offsets[:, i])),
            p95=float(np.quantile(offsets[:, i], .95)), max=float(offsets[:, i].max()))
            if len(offsets) else dict(median=None, p95=None, max=None)
            for i, side in enumerate(('before', 'after'))}
        population = dict(scenes=len(rows), components=len(set(manifest['components'][sid] for sid in scene_ids)),
            nodes=len(sources), queries=len(sources)*64, pairs=sum(len(r['pairs']) for r in records['native_ols']),
            positives=sum(int(r['labels'].sum()) for r in records['native_ols']),
            native_points=sum(row['native_point_count'] for row in tracks),
            zero_query_support_tracks=sum(row['query_support_count'] == 0 for row in tracks),
            one_query_support_tracks=sum(row['query_support_count'] == 1 for row in tracks))
        report = dict(**state, population=population, scene_ids=scene_ids, data=DATA,
            method_scope='exactly two predeclared ordinary offline position ablations; no window/parameter search',
            primary_metric='outside-support position MAE; RMSE/p95 and downstream association are complementary',
            limits=['V reused for development reporting and each-arm threshold selection; no independent test claim',
                    'position-only; no speed/heading prediction or full-state completion acceptance',
                    'fitted-anchor may jump at native support boundary; inside observations unchanged; diagnostic only',
                    'association uses all unordered pairs and unweighted Q64 RMS distance; no grouping/fusion'],
            boundary_offset_definition='geodesic distance from measured endpoint to clipped/wrapped fitted-line boundary; '
                'two-point tracks preserve baseline exactly and have zero offset',
            fitted_boundary_offsets_metres=boundary,
            fitted_boundary_latitude_clamp_count=sum(sum(row['fitted_boundary_latitude_clamped']) for row in tracks),
            latitude_clamp_counts={name: sum(row['latitude_clamp_counts'][name] for row in tracks) for name in METHODS},
            overall=_summary(**pooled), by_source=by_source, per_scene=scenes, per_track=tracks,
            changes_vs_ols=dict(scene=_changes(scenes, 'all_position'), track=_changes(tracks, 'methods'),
                scene_outside=_changes(scenes, 'outside_position'), track_outside=_changes(tracks, 'outside_position')),
            association=association)
        np.savez_compressed(output/'scores.npz', **score_payload)
        _write(output/'predictions'/'index.json', saved_index)
        report.update(status='COMPLETED', elapsed_seconds=time.monotonic()-started)
        _write(output/'summary.json', report)
        state.update(status='COMPLETED', population=population, elapsed_seconds=report['elapsed_seconds'])
        _write(output/'run.json', state)
        return report
    except BaseException as exc:
        state.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                     elapsed_seconds=time.monotonic()-started)
        _write(output/'run.json', state)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions-dir', 'manifest', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    command = [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    report = run(args.predictions_dir, args.manifest, args.output_dir, command)
    print(json.dumps(dict(status=report['status'], population=report['population'],
        position_mae={name: row['regions']['all']['mae'] for name, row in report['overall'].items()},
        association_f1={name: row['own_threshold']['macro_f1'] for name, row in report['association'].items()})))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
