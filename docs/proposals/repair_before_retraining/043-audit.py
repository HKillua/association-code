"""Audit returned 043/034 completion arrays only: no inference, ZIP scan or fitting."""
from pathlib import Path
import argparse
import copy
import gzip
import io
import json
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REGIONS = ('all', 'inside_support', 'before_support', 'after_support',
           'exact_observation', 'inside_nonobserved')
ATTRS = ('position', 'speed', 'heading')
COMMON = ('scene_id', 'node_ids', 'identities', 'sources', 'query_seconds',
          'target', 'target_valid', 'nearest_seconds', *('region_'+r for r in REGIONS))


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def arc(x):
    return (x + 180.) % 360. - 180.


def distance(a, b):
    # Cartesian great-circle expression, independent of production haversine.
    def unit(x):
        lat, lon = np.deg2rad(x[..., 0]), np.deg2rad(x[..., 1])
        return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), -1)
    u, v = unit(a.astype(np.float64)), unit(b.astype(np.float64))
    return 6371000.*np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), (u*v).sum(-1))


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(n=x.size, mae=float(x.mean()) if x.size else None,
                rmse=float(np.sqrt(np.square(x).mean())) if x.size else None,
                p50=float(np.quantile(x, .5)) if x.size else None,
                p90=float(np.quantile(x, .9)) if x.size else None,
                p99=float(np.quantile(x, .99)) if x.size else None,
                maximum=float(x.max()) if x.size else None)


def errors(row, base=False, frozen=False):
    p = row['position_base' if base else 'prediction'].astype(float).copy()
    if frozen:
        residual = p[..., :2]-row['position_base'][..., :2]
        residual[..., 1] = arc(residual[..., 1])
        p[..., :2] = row['position_base'][..., :2] + residual.mean(1, keepdims=True)
        p[..., 1] = arc(p[..., 1])
    target = row['target']
    result = {'position': distance(p[..., :2], target[..., :2])}
    if not base:
        result.update(speed=np.abs(p[..., 2]-target[..., 2]),
                      heading=np.abs(arc(p[..., 3]-target[..., 3])))
    return result


def regions(row):
    d = row['nearest_seconds']
    return {**{r: row['region_'+r] for r in REGIONS},
            'le_300s': d <= 300, 'gt_300_le_600s': (d > 300) & (d <= 600),
            'gt_600_le_1800s': (d > 600) & (d <= 1800),
            'gt_1800s': (d > 1800) & np.isfinite(d), 'no_input': ~np.isfinite(d)}


def metrics(row, base=False, frozen=False):
    e, valid = errors(row, base, frozen), row['target_valid']
    masks = dict(position=valid[..., :2].all(-1), speed=valid[..., 2], heading=valid[..., 3])
    return {region: {attr: stats(values[mask & masks[attr]]) for attr, values in e.items()}
            for region, mask in regions(row).items()}


def load(tag):
    prefix = {'034': 'HEADING', '043': 'MOTION'}[tag]
    path = ROOT/'HandOver/results'/f'COOPERATIVE-{prefix}-20261020-{tag}'
    run, report = read(path/'run.json.gz'), read(path/'completion-evaluation.json.gz')
    with tarfile.open(path/'completion-predictions.tar.gz') as tar:
        index = json.load(tar.extractfile('completion-predictions/index.json'))
        rows = []
        for entry in index:
            with np.load(io.BytesIO(tar.extractfile('completion-predictions/'+entry['path']).read()),
                         allow_pickle=False) as saved:
                rows.append({k: saved[k] for k in saved.files})
    contract = run['scene_contract'][1]
    assert len(contract) == len(rows) == len(index) == report['scene_count'] == 64
    assert report['fragment_count'] == report['completion_forward_count'] == 1635
    assert report['test_opened'] is False
    for row, c, ix in zip(rows, contract, index):
        assert str(row['scene_id']) == c['scene_id'] == ix['scene_id'] and c['split'] == 'V_select'
        for key in ('node_ids', 'identities', 'sources'):
            np.testing.assert_array_equal(row[key], c[key])
        np.testing.assert_array_equal(row['query_seconds'], c['grid'])
        n = len(row['node_ids'])
        assert all(row[k].shape == (n, 64, 4) for k in ('prediction', 'position_base', 'target', 'target_valid'))
        assert np.isfinite(row['prediction']).all() and np.isfinite(row['position_base'][..., :2]).all()
        assert np.isnan(row['position_base'][..., 2:]).all()
        assert row['target_valid'].dtype == np.bool_
        assert row['region_all'].all()
        np.testing.assert_array_equal(row['region_inside_support'].astype(int)+row['region_before_support']+
                                      row['region_after_support'], np.ones((n, 64)))
        np.testing.assert_array_equal(row['region_exact_observation'], row['nearest_seconds'] == 0)
    return path, run, report, rows


def pool(rows):
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0] if k not in ('scene_id', 'query_seconds')}


def paired(delta):
    x = np.asarray(delta, dtype=float)
    return dict(improved=int((x < 0).sum()), worsened=int((x > 0).sum()), unchanged=int((x == 0).sum()),
                mean_delta=float(x.mean()), median_delta=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()))


def fixed_support_gate(row):
    """One fixed 600-second output diagnostic; no fitting or sweep.

    Construction uses all saved Q64 predictions, bases and native-support
    distances, never targets, masks, identities or sources. Keep arithmetic
    in float64 before coordinate subtraction and shortest-arc wrapping.
    """
    prediction = row['prediction'].astype(np.float64).copy()
    base = row['position_base'][..., :2].astype(np.float64)
    residual = prediction[..., :2] - base
    residual[..., 1] = arc(residual[..., 1])
    nearest = row['nearest_seconds'].astype(np.float64)
    assert np.all(nearest >= 0) and not np.isnan(nearest).any()
    gate = -np.expm1(-nearest / 600.)
    prediction[..., :2] = base + gate[..., None] * residual
    prediction[..., 1] = arc(prediction[..., 1])
    np.testing.assert_array_equal(prediction[..., 2:], row['prediction'][..., 2:])
    np.testing.assert_array_equal(prediction[..., :2][nearest == 0], base[nearest == 0])
    assert np.isfinite(prediction).all()
    return {**row, 'prediction': prediction}


def fixed_support_gate_analysis(rows):
    gated_rows = [fixed_support_gate(row) for row in rows]
    data = pool(gated_rows)
    scene_records, fragment_records = [], []
    for original, gated in zip(rows, gated_rows):
        values = dict(base=errors(original, base=True)['position'],
                      model=errors(original)['position'], gated=errors(gated)['position'])
        valid = original['target_valid'][..., :2].all(-1)
        scene_records.append(dict(scene_id=str(original['scene_id']),
                                  **{key: stats(value[valid]) for key, value in values.items()}))
        for i, node in enumerate(original['node_ids']):
            fragment_records.append(dict(scene_id=str(original['scene_id']), node_id=str(node),
                source=str(original['sources'][i]),
                **{key: stats(value[i, valid[i]]) for key, value in values.items()}))
    comparisons = {}
    for scope, records in (('scene', scene_records), ('fragment', fragment_records)):
        comparisons[scope] = dict(
            mean_of_unit_mae={key: float(np.mean([r[key]['mae'] for r in records]))
                              for key in ('base', 'model', 'gated')},
            gated_minus_base=paired([r['gated']['mae']-r['base']['mae'] for r in records]),
            gated_minus_model=paired([r['gated']['mae']-r['model']['mae'] for r in records]))
    return dict(
        definition='prediction_xy = base_xy + (1-exp(-nearest_seconds/600)) * '
                   '(prediction_xy-base_xy); longitude residual uses shortest arc. '
                   'Applies to all sources and all Q64; speed/heading unchanged.',
        status='exploratory posthoc full-V output diagnostic, not a trained candidate or held-out result',
        scale_seconds=600., scale_choice='One fixed existing gap600 scale; no sweep or target-based fit.',
        construction_uses_targets=False, construction_uses_target_mask=False,
        construction_uses_source_or_identity=False,
        inference_calls=0, optimizer_updates=0,
        input_dtypes={key: sorted({str(row[key].dtype) for row in rows})
                      for key in ('prediction', 'position_base', 'nearest_seconds')},
        arithmetic='Cast prediction/base/nearest to float64 before subtraction, wrapping and expm1; '
                   'reuse this audit\'s independent unit-vector distance and all original masks.',
        weighting='Pooled metrics weight every valid query equally; per-scene/per-fragment MAE '
                  'first averages valid queries within that unit, then gives units equal weight. '
                  'Fragments are not unique target identities.',
        metrics=metrics(data),
        by_source={str(s): metrics({key: value[data['sources'] == s] for key, value in data.items()})
                   for s in sorted(set(data['sources']))},
        paired=comparisons,
        original_record_reference='Original base043/model043 metrics and sources are in the root '
                                  'per_scene/per_fragment records; join by scene_id/node_id.',
        per_scene=[dict(scene_id=r['scene_id'], gated=r['gated']) for r in scene_records],
        per_fragment=[dict(scene_id=r['scene_id'], node_id=r['node_id'], gated=r['gated'])
                      for r in fragment_records],
        limitations=['600 seconds was not swept, but the diagnostic was motivated by already observed V failures.',
                     'Finite residuals and a continuous distance gate imply zero correction at native observations; '
                     'this output cache does not itself verify between-query continuity.',
                     'A small pooled gain does not establish support protection, learned motion, repeatability or system benefit.'])


def training(paths, runs):
    configs = [copy.deepcopy(read(paths[t]/'actual_config.json')) for t in ('034', '043')]
    assert configs[1]['model'].pop('position_base_policy') == 'native_ols'
    assert configs[0] == configs[1]
    histories = {t: read(paths[t]/'history.json.gz') for t in paths}
    sampling = ('stage', 'routing', 'valid_counts', 'sampled_pairs', 'used_pairs',
                'removed_observations', 'update', 'sampling')
    for a, b in zip(histories['034'], histories['043']):
        assert all(a[k] == b[k] for k in sampling)
    for k in ('scene_contract', 'error_scales', 'cooperative_representation', 'support_time_scaling'):
        assert runs['034'][k] == runs['043'][k]
    result = {}
    for tag, run in runs.items():
        h = histories[tag]
        status = read(paths[tag]/'status.json')
        assert status['status'] == run['status'] == 'COMPLETED'
        assert status['returncode'] == int((paths[tag]/'exit_code.txt').read_text()) == 0
        assert run['completed_updates'] == len(h) == 1000
        assert [r['update'] for r in h] == list(range(1, 1001)) and all(r['stage'] == 'teacher' for r in h)
        assert run['phase'] == 'teacher' and run['completed_routes'] == []
        assert not run['identity_selection_performed'] and not run['test_opened']
        assert run['selection_policy'] == 'predetermined teacher endpoint; no V selection'
        assert run['parameter_count'] == 2190735
        assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in h)
        assert all(abs(sum(r['loss_components'].values())-r['loss']) < 1e-6 for r in h)
        result[tag] = dict(completed_updates=1000, parameters=run['parameter_count'],
                          elapsed_seconds=status['elapsed_seconds'],
                          phases_seconds={k: v['seconds'] for k, v in run['phases'].items()},
                          clipped_steps=sum(r['gradient_norm'] > 1 for r in h),
                          gradient_norm=stats([r['gradient_norm'] for r in h]),
                          mean_update_seconds=float(np.mean([r['seconds'] for r in h])),
                          windows={f'{i+1}-{i+100}': {k: float(np.mean([r['loss_components'][k] for r in h[i:i+100]]))
                                                    for k in ('physical', 'identity_bce')}
                                   for i in range(0, 1000, 100)})
    request = read(paths['043']/'request.json')
    argv = request['argv']
    assert argv[argv.index('--phase')+1] == 'teacher' and '--save-predictions' in argv
    assert argv[argv.index('--device')+1] == 'cuda:0' and argv[argv.index('--cpu-threads')+1] == '4'
    assert argv[argv.index('--time-policy')+1] == request['time_policy'] == 'advisory'
    return dict(config_only_change='model.position_base_policy=native_ols',
                same_sampling_metadata=True, same_reported_fitted_metadata=True,
                full_weights_or_normalizer_not_locally_loaded=True, arms=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths, runs, reports, rows = {}, {}, {}, {}
    for tag in ('034', '043'):
        paths[tag], runs[tag], reports[tag], rows[tag] = load(tag)
    train = training(paths, runs)
    for a, b in zip(rows['034'], rows['043']):
        for key in COMMON:
            np.testing.assert_array_equal(a[key], b[key])
    pooled = {t: pool(v) for t, v in rows.items()}
    assert pooled['043']['prediction'].shape == (1635, 64, 4)
    model = {t: metrics(v) for t, v in pooled.items()}
    base = {t: metrics(v, base=True) for t, v in pooled.items()}
    source_metrics = {t: {str(s): dict(model=metrics({k: v[data['sources'] == s] for k, v in data.items()}),
                                       base=metrics({k: v[data['sources'] == s] for k, v in data.items()}, base=True))
                         for s in sorted(set(data['sources']))} for t, data in pooled.items()}
    report_error = {}
    for tag in rows:
        diffs = []
        for method, actual in (('model', model[tag]), ('position_base', base[tag])):
            for group in ('regions', 'input_distance_metrics'):
                for region, attrs in reports[tag]['methods'][method][group].items():
                    for attr, value in actual[region].items():
                        assert value['n'] == attrs[attr]['intersection_count']
                        for metric in ('mae', 'rmse'):
                            if value[metric] is not None:
                                diffs.append(abs(value[metric]-attrs[attr][metric]))
                                np.testing.assert_allclose(value[metric], attrs[attr][metric], atol=1e-6, rtol=1e-10)
        report_error[tag] = max(diffs)
    scene = []
    fragment = []
    for a, b in zip(rows['034'], rows['043']):
        ma, mb, ba, bb = metrics(a)['all'], metrics(b)['all'], metrics(a, base=True)['all'], metrics(b, base=True)['all']
        scene.append(dict(scene_id=str(b['scene_id']), model034=ma, model043=mb,
                          base034=ba['position'], base043=bb['position']))
        ea, eb, ebase = errors(a)['position'], errors(b)['position'], errors(b, base=True)['position']
        for i, node in enumerate(b['node_ids']):
            valid = b['target_valid'][i, :, :2].all(-1)
            assert valid.any()
            fragment.append(dict(scene_id=str(b['scene_id']), node_id=str(node), source=str(b['sources'][i]),
                                 model034=stats(ea[i, valid]), model043=stats(eb[i, valid]), base043=stats(ebase[i, valid])))
    contrast = {}
    for scope, records in (('scene', scene), ('fragment', fragment)):
        def get(r, key):
            x = r[key]
            return x['position']['mae'] if scope == 'scene' and key.startswith('model') else x['mae']
        contrast[scope] = dict(model043_minus_own_base=paired([get(r, 'model043')-get(r, 'base043') for r in records]),
                               model043_minus034=paired([get(r, 'model043')-get(r, 'model034') for r in records]))
    data = pooled['043']
    displacement = distance(data['prediction'][..., :2], data['position_base'][..., :2])
    residual = data['prediction'][..., :2].astype(float)-data['position_base'][..., :2]
    residual[..., 1] = arc(residual[..., 1])
    constant_residual_metrics = metrics(data, frozen=True)
    centered = residual-residual.mean(1, keepdims=True)
    meters = centered*np.pi/180*6371000
    meters[..., 1] *= np.cos(np.deg2rad(data['position_base'][..., 0]))
    dynamic_rms = np.sqrt(np.square(meters).sum(-1).mean(1))
    result = dict(status='COMPLETED', scope='returned full V 034 and 043 predictions only',
                  inference_calls=0, optimizer_updates=0, raw_archive_members_read=[], test_opened=False,
                  population=dict(scenes=64, components=len({r['component_id'] for r in runs['043']['scene_contract'][1]}),
                                  fragments=1635, queries=104640,
                                  valid_each_attribute=[int(data['target_valid'][..., i].sum()) for i in range(4)],
                                  predictions_without_target=23433),
                  common_arrays_equal=len(COMMON)*64, training=train,
                  reported_metrics_max_abs_error=report_error, model=model, base=base,
                  by_source=source_metrics, paired=contrast, per_scene=scene, per_fragment=fragment,
                  residual=dict(displacement_metres=stats(displacement),
                                per_fragment_centered_rms_metres=stats(dynamic_rms),
                                mean_only_metrics=constant_residual_metrics,
                                definition='Mean latitude and shortest-arc longitude residual over all Q64; no labels/masks. '
                                'Posthoc output ablation, not new inference/training. Heading/speed kept unchanged.'),
                  fixed_support_gate_600s=fixed_support_gate_analysis(rows['043']),
                  limitations=['Single seed full development V, not independent test or significance.',
                               'Loss windows use changing minibatches and learned objective weights; no historical gradient-conflict proof.',
                               'Position_base has no speed/heading, so no invented two-attribute residual-gain claim.',
                               'Final objective weights and checkpoint tensors not independently loaded here.',
                               'Same parameters/config except base does not force identical optimization trajectories.',
                               'GPU initial snapshot does not independently establish all-run exclusive resource use.'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(model043=model['043']['all'], base043=base['043']['all'], paired=contrast,
                         reported_metrics_max_abs_error=report_error), ensure_ascii=False))


if __name__ == '__main__':
    main()
