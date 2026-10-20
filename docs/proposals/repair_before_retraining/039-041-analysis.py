"""Audit saved 034/039–041 predictions and CPU diagnostics; no inference/data scan."""
from pathlib import Path
import argparse
import copy
import gzip
import json
import sys
import tarfile
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.analysis.compare_completion_predictions import COMMON, REGIONS, _completed, _equal

PREFIX = {'034': 'HEADING', '039': 'PRENORM', '040': 'SHALLOW', '041': 'NOBCE'}
ATTRS = ('position', 'speed', 'heading')


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def circular(x):
    return (x + 180.) % 360. - 180.


def distance(a, b):
    # Independent Cartesian great-circle formula, rather than evaluator haversine.
    def unit(x):
        lat, lon = np.deg2rad(x[..., 0]), np.deg2rad(x[..., 1])
        return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), -1)
    a, b = unit(a), unit(b)
    return 6371000. * np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a*b).sum(-1))


def pool(rows):
    keys = ('prediction', 'position_base', 'target', 'target_valid', 'sources',
            'nearest_seconds', *('region_'+k for k in REGIONS))
    return {k: np.concatenate([r[k] for r in rows]) for k in keys}


def errors(row, prediction=None):
    p = np.asarray(row['prediction'] if prediction is None else prediction, dtype=np.float64)
    t = row['target'].astype(np.float64)
    return (distance(p[..., :2], t[..., :2]), abs(p[..., 2]-t[..., 2]),
            abs(circular(p[..., 3]-t[..., 3])))


def masks(row):
    d = row['nearest_seconds']
    return {**{k: row['region_'+k] for k in REGIONS},
            'le_300s': d <= 300, 'gt_300_le_600s': (d > 300) & (d <= 600),
            'gt_600_le_1800s': (d > 600) & (d <= 1800),
            'gt_1800s': (d > 1800) & np.isfinite(d), 'no_input': ~np.isfinite(d)}


def metrics(row, prediction=None):
    es = errors(row, prediction)
    valid = row['target_valid']
    vs = (valid[..., :2].all(-1), valid[..., 2], valid[..., 3])
    out = {}
    for region, mask in masks(row).items():
        out[region] = {}
        for name, error, v in zip(ATTRS, es, vs):
            values = error[mask & v]
            assert np.isfinite(values).all()
            out[region][name] = dict(n=len(values), mae=float(values.mean()) if len(values) else None,
                                     rmse=float(np.sqrt(np.square(values).mean())) if len(values) else None)
    return out


def describe(row, prediction=None):
    return dict(overall=metrics(row, prediction), by_source={
        str(s): metrics({k: v[row['sources'] == s] for k, v in row.items()},
                       None if prediction is None else prediction[row['sources'] == s])
        for s in sorted(set(row['sources']))})


def frozen_query_prediction(row):
    # Freeze residual/latent-log level using all Q64, never truth or validity masks.
    p, base = row['prediction'].astype(np.float64), row['position_base'].astype(np.float64)
    residual = p[..., :2] - base[..., :2]
    residual[..., 1] = circular(residual[..., 1])
    frozen = p.copy()
    frozen[..., :2] = base[..., :2] + residual.mean(1, keepdims=True)
    frozen[..., 1] = circular(frozen[..., 1])
    frozen[..., 2] = np.maximum(np.exp(np.log(p[..., 2]+1e-6).mean(1, keepdims=True))-1e-6, 0.)
    return frozen


def summary(x):
    x = np.asarray(x, dtype=float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()))


def audit_dynamics(source):
    run, report, status = (read(source/n) for n in
        ('run.json.gz', 'candidate-dynamics.json.gz', 'status.json'))
    rows, depth = report['per_fragment'], report['decoder_layer_count']
    assert run['status'] == status['status'] == 'COMPLETED'
    assert int((source/'exit_code.txt').read_text()) == status['returncode'] == 0
    assert run['device'] == 'cpu' and run['cpu_threads'] == 4 and not run['reference_members_read']
    assert run['optimizer_updates'] == 0 and not run['test_opened']
    assert report['fragment_count'] == report['completion_forward_count'] == len(rows) == 86
    assert run['scene_ids'] == ['1671', '2056', '290', '113']
    max_error = 0.
    for key, reported in report['aggregate'].items():
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        actual = summary(values)
        assert actual['n'] == reported['count']
        for field in ('mean', 'median', 'min', 'max'):
            max_error = max(max_error, abs(actual[field]-reported[field]))
            np.testing.assert_allclose(actual[field], reported[field], atol=1e-12, rtol=1e-12)
    for r in rows:
        m = r['metrics']
        assert m['query_reconstruction_max_abs'] == m['physical_reconstruction_max_abs'] == 0
        assert m['final_hidden_centered_rms'] == m[f'decoder_layer_{depth}_centered_rms']
    fields = ['decoder_input_centered_rms',
              *(f'decoder_layer_{i}_centered_rms' for i in range(1, depth+1)),
              'final_hidden_centered_rms', 'remove_query_residual_position_rms_metres',
              'remove_query_residual_speed_rms', 'remove_query_residual_heading_rms']
    return dict(depth=depth, fragments=86, scenes=run['scene_ids'], exit_code=0,
                elapsed_seconds=status['elapsed_seconds'], aggregate_max_error=max_error,
                all={k: summary([r['metrics'][k] for r in rows]) for k in fields},
                by_source={str(s): {k: summary([r['metrics'][k] for r in rows if r['source'] == s])
                                   for k in fields}
                           for s in sorted({r['source'] for r in rows})},
                normalizer=run['candidate_normalizer'])


def audit_training(sources, runs):
    baseline = read(sources['034']/'actual_config.json')
    histories = {tag: read(source/'history.json.gz') for tag, source in sources.items()}
    result = {}
    sampling = ('stage', 'routing', 'valid_counts', 'sampled_pairs', 'used_pairs',
                'removed_observations', 'update', 'sampling')
    for tag, source in sources.items():
        run, h = runs[tag], histories[tag]
        status = read(source/'status.json')
        assert status['status'] == run['status'] == 'COMPLETED'
        assert int((source/'exit_code.txt').read_text()) == status['returncode'] == 0
        assert len(h) == 1000 and [r['update'] for r in h] == list(range(1, 1001))
        assert all(r['stage'] == 'teacher' for r in h)
        assert run['completed_updates'] == 1000 and run['completed_routes'] == []
        assert not run['identity_selection_performed'] and not run['test_opened']
        assert all(r[k] == old[k] for r, old in zip(h, histories['034']) for k in sampling)
        for k in ('scene_contract', 'error_scales', 'cooperative_representation', 'support_time_scaling'):
            assert run[k] == runs['034'][k]
        config = copy.deepcopy(read(source/'actual_config.json'))
        intervention = None
        if tag == '039':
            intervention = ('model.decoder_norm_first', config['model'].pop('decoder_norm_first'))
            assert intervention[1] is True
        elif tag == '040':
            intervention = ('model.decoder_num_layers', config['model'].pop('decoder_num_layers'))
            assert intervention[1] == 2
        elif tag == '041':
            intervention = ('training.teacher_identity_weight', config['training'].pop('teacher_identity_weight'))
            assert intervention[1] == 0.
        assert config == baseline
        assert np.isfinite([r['loss'] for r in h]).all()
        assert np.isfinite([r['gradient_norm'] for r in h]).all()
        assert all(abs(sum(r['loss_components'].values())-r['loss']) < 1e-6 for r in h)
        expected_params = 2190735 - (4*(8*128**2 + 2*128*256 + 15*128 + 256) if tag == '040' else 0)
        assert run['parameter_count'] == expected_params
        if tag != '034':
            weight = 0. if tag == '041' else 1.
            assert all(r['teacher_identity_weight'] == weight and
                       abs(r['identity_bce_raw']*weight-r['loss_components']['identity_bce']) < 1e-9 for r in h)
            request = read(source/'request.json')
            argv = request['argv']
            assert '--phase' in argv and argv[argv.index('--phase')+1] == 'teacher'
            assert '--save-predictions' in argv and request['time_policy'] == 'advisory'
            assert argv[argv.index('--device')+1] == 'cuda:0'
            assert argv[argv.index('--cpu-threads')+1] == '4'
        result[tag] = dict(intervention=intervention, completed_teacher_updates=len(h),
            sampling_metadata_matches034=True, scientific_config_otherwise_matches034=True,
            parameter_count=expected_params, launcher_elapsed_seconds=status['elapsed_seconds'],
            stages_seconds={k: v['seconds'] for k, v in run['phases'].items()},
            gradient_norm=summary([r['gradient_norm'] for r in h]),
            clipped_steps=sum(r['gradient_norm'] > 1 for r in h),
            clipped_step_fraction=sum(r['gradient_norm'] > 1 for r in h)/len(h),
            teacher_identity_weight=run['training'].get('teacher_identity_weight', 1.),
            final_learned_physical_loss_weights=None,
            final_weight_note='Objective log_variance is trained but its final values are not in returned run/history.',
            windows={f'{start+1}-{start+100}': {
                k: float(np.mean([r['loss_components'][k] for r in h[start:start+100]]))
                for k in ('physical', 'identity_bce')} for start in (0, 900)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {tag: ROOT/'HandOver/results'/f'COOPERATIVE-{prefix}-20261020-{tag}'
               for tag, prefix in PREFIX.items()}
    with tempfile.TemporaryDirectory(prefix='mtad039-041-audit-') as scratch:
        runs, reports, rows = {}, {}, {}
        for tag, source in sources.items():
            dst = Path(scratch)/tag
            dst.mkdir()
            for name in ('run.json', 'completion-evaluation.json'):
                (dst/name).write_bytes(gzip.decompress((source/(name+'.gz')).read_bytes()))
            with tarfile.open(source/'completion-predictions.tar.gz') as archive:
                archive.extractall(dst, filter='data')
            _, runs[tag], reports[tag], rows[tag] = _completed(dst)
        training = audit_training(sources, runs)
        for tag in PREFIX:
            for a, b in zip(rows['034'], rows[tag]):
                for key in (*COMMON, *('region_'+k for k in REGIONS)):
                    assert _equal(a[key], b[key]), (tag, key)
        pooled = {tag: pool(items) for tag, items in rows.items()}
        physical, frozen, deltas, reported_error, per_scene = {}, {}, {}, {}, {}
        for tag, data in pooled.items():
            assert data['prediction'].shape == (1635, 64, 4)
            assert np.isfinite(data['prediction']).all()
            physical[tag] = describe(data)
            frozen[tag] = describe(data, frozen_query_prediction(data))
            max_error = 0.
            for group, regions in reports[tag]['methods']['model'].items():
                if group not in ('regions', 'input_distance_metrics'):
                    continue
                for region, values in regions.items():
                    actual = physical[tag]['overall'][region]
                    for attr in ATTRS:
                        assert actual[attr]['n'] == values[attr]['intersection_count']
                        for metric in ('mae', 'rmse'):
                            if actual[attr][metric] is not None:
                                error = abs(actual[attr][metric]-values[attr][metric])
                                max_error = max(max_error, error)
                                np.testing.assert_allclose(actual[attr][metric], values[attr][metric],
                                                           atol=1e-6, rtol=1e-10)
            reported_error[tag] = max_error
            per_scene[tag] = {str(row['scene_id']): metrics(row)['all'] for row in rows[tag]}
            deltas[tag] = {region: {attr: frozen[tag]['overall'][region][attr]['mae'] -
                                            physical[tag]['overall'][region][attr]['mae']
                                   for attr in ('position', 'speed')}
                           for region in REGIONS}
        base_input = pooled['034']['prediction'].copy()
        base_input[..., :2] = pooled['034']['position_base'][..., :2]
        base = metrics(pooled['034'], base_input)
        base_position = {region: values['position'] for region, values in base.items()}
        paired = {}
        for tag in ('039', '040', '041'):
            paired[tag] = {}
            for attr in ATTRS:
                differences = [per_scene[tag][sid][attr]['mae']-per_scene['034'][sid][attr]['mae']
                               for sid in per_scene[tag]]
                paired[tag][attr] = dict(improved_scenes=sum(x < 0 for x in differences),
                    worsened_scenes=sum(x > 0 for x in differences), delta_of_scene_mae=summary(differences))
        dyn_sources = {'034': ROOT/'HandOver/results/CANDIDATE-DYNAMICS-20261020-038',
                       **{tag: sources[tag]/'dynamics' for tag in ('039', '040', '041')}}
        dynamics = {tag: audit_dynamics(source) for tag, source in dyn_sources.items()}
        assert all(d['normalizer'] == dynamics['034']['normalizer'] for d in dynamics.values())
        for d in dynamics.values():
            del d['normalizer']
        old = read(Path(__file__).with_name('034-audit.json'))
        result = dict(status='COMPLETED', scope='saved full development V and returned four-scene diagnostics',
            test_opened=False, model_inference_calls=0, optimizer_updates=0, archive_members_read=[],
            population=dict(scenes=64, fragments=1635, queries=104640,
                            valid_each_attribute=81207, predictions_without_target=23433),
            training=training, population_and_arrays_identical=True,
            returned_physical_metrics_max_abs_error=reported_error,
            physical=physical, position_base=base_position,
            all_query_mean_residual=frozen, mean_residual_minus_original_mae=deltas,
            scene_comparison_to034=paired, per_scene=per_scene, dynamics=dynamics,
            earlier_readonly_native_controls=old['metrics']['native'],
            controls_note='Earlier 034 audit raw-input control reused, not re-read/recomputed here. '
                'Position_base is independently recomputed from every current NPZ. No speed/heading '
                'control is fabricated from position_base zeros.',
            mean_residual_policy='All-Q64 latitude/short-arc-longitude residual means relative to saved '
                'position base, and mean log(predicted speed+1e-6); no target/mask used. Heading unchanged: '
                'full-V heading base was not exported. This posthoc observational ablation is not a '
                'trained alternative. Four-scene CPU diagnostics separately cover true heading residual.',
            limitations=['Single seed, multiple attributes and interventions; no significance claim.',
                'Pre-norm RMS recovery is amplitude, not useful information or predictive accuracy.',
                'Two-layer intervention changes capacity and dropout RNG consumption as well as depth.',
                'Removing BCE changes shared-encoder gradients and global clipping; not isolated gradient conflict.',
                '039 and 041 share a GPU; 040 shared with another process. Wall time is not a controlled speed benchmark.',
                'Dynamic aggregate recomputation verifies returned per-fragment scalars, not model weights or hidden arrays.',
                'All four snapshots are candidate teacher completion endpoints, not completed association/fusion systems.'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({tag: physical[tag]['overall']['all'] for tag in PREFIX}, ensure_ascii=False))


if __name__ == '__main__':
    main()
