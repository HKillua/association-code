"""Audit 034 vs 032 and visible-native controls on the unchanged full V.

Uses returned teacher predictions and V observation/association ZIP members only.
No model inference, checkpoint loading, fitting, reference read or T_test access.
Mean-only residual ablations average all Q64 predictions without targets/masks.
"""
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
from association.data.archive import MTADArchive
from association.data.native import _record
from association.data.timed import prepare_timed_scene
from tools.analysis.compare_completion_predictions import compare, _completed, _metrics


def circular(value):
    return (value + 180.) % 360. - 180.


def circular_mean(values, axis=1, keepdims=True):
    radians = np.deg2rad(values)
    return np.rad2deg(np.arctan2(np.sin(radians).mean(axis, keepdims=keepdims),
                               np.cos(radians).mean(axis, keepdims=keepdims)))


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p90=float(np.percentile(values, 90)) if len(values) else None,
                max=float(values.max()) if len(values) else None)


def distance(a, b):
    la1, lo1, la2, lo2 = [np.deg2rad(v) for v in (a[..., 0], a[..., 1], b[..., 0], b[..., 1])]
    h = np.sin((la1-la2)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo1-lo2)/2)**2
    return 6371000*2*np.arctan2(np.sqrt(np.clip(h, 0, 1)), np.sqrt(np.clip(1-h, 0, 1)))


def compact(metrics):
    return {group: {region: {attr: {key: value for key, value in result.items()
                     if key in ('intersection_count', 'mae', 'rmse', 'count', 'mse')}
                     for attr, result in values.items()
                     if attr in ('position', 'speed', 'heading', 'raw_log_speed_mse')}
                    for region, values in regions.items()}
            for group, regions in metrics.items()}


def pool(rows):
    keys = ('node_ids', 'identities', 'sources', 'prediction', 'target', 'target_valid',
            'nearest_seconds', *[key for key in rows[0] if key.startswith('region_')])
    return {key: np.concatenate([row[key] for row in rows], axis=0) for key in keys}


def describe(rows):
    pooled = pool(rows)
    return dict(overall=compact(_metrics(pooled)), by_source={
        str(source): compact(_metrics({key: value[pooled['sources'] == source]
                                      for key, value in pooled.items()}))
        for source in sorted(set(pooled['sources']))})


def unpack(tag, scratch):
    prefix = 'COOPERATIVE-SPEED' if tag == '032' else 'COOPERATIVE-HEADING'
    source = ROOT/'HandOver/results'/f'{prefix}-20261020-{tag}'
    destination = scratch/tag
    destination.mkdir()
    for name in ('run.json', 'completion-evaluation.json'):
        (destination/name).write_bytes(gzip.decompress((source/(name+'.gz')).read_bytes()))
    with tarfile.open(source/'completion-predictions.tar.gz') as archive:
        archive.extractall(destination, filter='data')
    return source, destination


def heading_base_fp32(scene):
    """034 time/arc arithmetic from raw inputs, without unavailable normalization roundtrip."""
    output = np.empty((len(scene['node_ids']), 64), np.float32)
    query = np.asarray((scene['grid']-scene['time_origin_seconds'])/20., dtype=np.float32)
    for node, mask in enumerate(scene['native_observed']):
        times = np.asarray((scene['native_times'][node, mask]-scene['time_origin_seconds'])/20.,
                           dtype=np.float32)
        headings = np.asarray(scene['native_values'][node, mask, 3], dtype=np.float32)
        for index, time in enumerate(query):
            left = max(0, int(np.searchsorted(times, time, side='right'))-1)
            right = min(len(times)-1, int(np.searchsorted(times, time, side='left')))
            width = times[right]-times[left]
            alpha = (time-times[left])/width if width > 0 else np.float32(0.)
            raw = headings[right]-headings[left]
            delta = (raw+np.float32(180.)) % np.float32(360.)-np.float32(180.)
            if abs(abs(delta)-np.float32(180.)) <= 8*np.finfo(np.float32).eps*360:
                delta = np.float32(180. if raw > 0 else -180.)
            output[node, index] = (headings[left]+alpha*delta) % np.float32(360.)
    return output


def training_checks(sources, runs):
    configs = [json.loads((source/'actual_config.json').read_text()) for source in sources]
    configs = copy.deepcopy(configs)
    configs[1]['model'].pop('heading_output_policy')
    assert configs[0] == configs[1]
    histories = [json.loads(gzip.decompress((source/'history.json.gz').read_bytes()))
                 for source in sources]
    keys = ('stage', 'routing', 'valid_counts', 'sampled_pairs', 'used_pairs',
            'removed_observations', 'update', 'sampling')
    assert all([row['update'] for row in history] == list(range(1, 1001)) for history in histories)
    assert all(a[key] == b[key] for a, b in zip(*histories) for key in keys)
    matching = ('scene_contract', 'error_scales', 'cooperative_representation', 'support_time_scaling')
    assert all(runs[0][key] == runs[1][key] for key in matching)
    return dict(config_equal_except_heading_output_policy=True, completed_teacher_updates=[1000, 1000],
                sampling_and_valid_count_metadata_identical=True,
                matching_contract_and_fitted_metadata=list(matching),
                full_normalizer_buffer_independently_compared=False,
                reason='Returned runs contain scales and log-speed moments but not full state_mean/checkpoint.',
                initial_identity_bce_equal=histories[0][0]['loss_components']['identity_bce'] ==
                                          histories[1][0]['loss_components']['identity_bce'],
                training={tag: dict(
                    gradient_median=float(np.median([row['gradient_norm'] for row in history])),
                    clipped_steps=sum(row['gradient_norm'] > 1 for row in history),
                    first100={key: float(np.mean([row['loss_components'][key] for row in history[:100]]))
                              for key in ('physical', 'identity_bce')},
                    last100={key: float(np.mean([row['loss_components'][key] for row in history[-100:]]))
                             for key in ('physical', 'identity_bce')})
                          for tag, history in zip(('032', '034'), histories)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(prefix='mtad-heading-audit-') as scratch:
        sources, directories = zip(*(unpack(tag, Path(scratch)) for tag in ('032', '034')))
        comparison = compare(*directories, Path(scratch)/'comparison.json')
        _, run032, _, old = _completed(directories[0])
        _, run034, _, new = _completed(directories[1])
        training = training_checks(sources, [run032, run034])
        controls, model_base_controls, precision_deltas = [], [], []
        with MTADArchive(args.archive) as archive:
            for contract, saved in zip(run034['scene_contract'][1], new):
                native = _record(archive.load_scene(contract['scene_id']),
                                 contract['component_id'], 'V_select', 128)
                scene = prepare_timed_scene(native, length=64, max_observations=128,
                                            time_unit_seconds=20., query_times=saved['query_seconds'])
                for key in ('node_ids', 'identities', 'sources'):
                    np.testing.assert_array_equal(scene[key], saved[key])
                np.testing.assert_array_equal(scene['native_observed'].sum(1),
                                              contract['native_observed_counts'])
                control = scene['values'].copy()
                for node, mask in enumerate(scene['native_observed']):
                    times, states = scene['native_times'][node, mask], scene['native_values'][node, mask]
                    control[node, scene['grid'] < times[0]] = states[0]
                    control[node, scene['grid'] > times[-1]] = states[-1]
                if not np.isfinite(control).all():
                    raise ValueError('native control must retain every query')
                controls.append({**saved, 'prediction': control})
                approximate_model_base = control.copy()
                approximate_model_base[..., 3] = heading_base_fp32(scene)
                model_base_controls.append({**saved, 'prediction': approximate_model_base})
                difference = np.abs(circular(approximate_model_base[..., 3]-control[..., 3]))
                precision_deltas.append(dict(scene_id=contract['scene_id'],
                    mean_abs_degrees=float(difference.mean()), max_abs_degrees=float(difference.max()),
                    count_over_one_degree=int((difference > 1).sum()),
                    affected_node_ids=[str(scene['node_ids'][index]) for index in
                                       np.flatnonzero((difference > 1).any(1))]))
            members = list(archive.members_read)

        # Each alternative is constructed before scoring; target/mask never enter it.
        heading_frozen, all_frozen = [], []
        residual_rows, fragment_rows, scene_rows = [], [], []
        for contract, a, b, control, model_control in zip(
                run034['scene_contract'][1], old, new, controls, model_base_controls):
            prediction = b['prediction'].astype(np.float64)
            base = model_control['prediction']
            residual = circular(prediction[..., 3] - base[..., 3])
            residual_mean = circular_mean(residual)
            stable_heading = (base[..., 3] + residual_mean) % 360.
            ablated = prediction.copy()
            ablated[..., 3] = stable_heading
            heading_frozen.append({**b, 'prediction': ablated})
            position_residual = prediction[..., :2] - b['position_base'][..., :2]
            position_residual[..., 1] = circular(position_residual[..., 1])
            frozen = ablated.copy()
            frozen[..., :2] = b['position_base'][..., :2] + position_residual.mean(1, keepdims=True)
            frozen[..., 1] = circular(frozen[..., 1])
            log_speed = np.log(prediction[..., 2] + 1e-6)
            frozen[..., 2] = np.maximum(np.exp(log_speed.mean(1, keepdims=True)) - 1e-6, 0.)
            all_frozen.append({**b, 'prediction': frozen})
            heading_errors = {name: np.abs(circular(row['prediction'][..., 3] - b['target'][..., 3]))
                              for name, row in [('032', a), ('034', b), ('native', control),
                                                ('model_base', model_control),
                                                ('mean_residual', heading_frozen[-1])]}
            scene_metrics = {name: {'regions': {region: attributes
                                for region, attributes in compact(_metrics(row))['regions'].items()
                                if region in ('all', 'inside_support', 'before_support', 'after_support')}}
                             for name, row in [('032', a), ('034', b), ('native', control),
                                               ('model_base', model_control),
                                               ('mean_residual', heading_frozen[-1])]}
            scene_rows.append(dict(scene_id=contract['scene_id'], component_id=contract['component_id'],
                                   fragment_count=len(b['node_ids']), methods=scene_metrics))
            for node, node_id in enumerate(b['node_ids']):
                valid = b['target_valid'][node, :, 3]
                region_errors = {}
                for region in ('all', 'inside_support', 'before_support', 'after_support',
                               'exact_observation', 'inside_nonobserved'):
                    mask = valid & b['region_'+region][node]
                    region_errors[region] = dict(count=int(mask.sum()),
                        **{name: float(error[node, mask].mean()) if mask.any() else None
                           for name, error in heading_errors.items()})
                fragment_rows.append(dict(scene_id=contract['scene_id'], component_id=contract['component_id'],
                                          node_id=str(node_id), source=str(b['sources'][node]),
                                          regions=region_errors))
                r = residual[node]
                centered = circular(r-residual_mean[node, 0])
                pvalid = b['target_valid'][node, :, :2].all(-1)
                svalid = b['target_valid'][node, :, 2]
                # Native base latitude, not target, defines this local diagnostic scale.
                scale = np.array([1., np.cos(np.deg2rad(base[node, :, 0].mean()))])*6371000*np.pi/180.
                required = b['target'][node, :, :2]-b['position_base'][node, :, :2]
                required[..., 1] = circular(required[..., 1])
                residual_rows.append(dict(source=str(b['sources'][node]),
                    residual_abs_mean_all=float(np.abs(r).mean()),
                    residual_circular_mean=float(residual_mean[node, 0]),
                    residual_centered_std_all=float(centered.std()),
                    residual_centered_std_valid=float(centered[valid].std()) if valid.any() else None,
                    heading_output_centered_std_valid=float(circular(prediction[node, valid, 3] -
                        circular_mean(prediction[node:node+1, valid, 3])[0, 0]).std()) if valid.any() else None,
                    predicted_position_residual_std_m=float(np.linalg.norm(
                        np.std(position_residual[node, pvalid]*scale, axis=0))) if pvalid.any() else None,
                    oracle_required_position_residual_std_m=float(np.linalg.norm(
                        np.std(required[pvalid]*scale, axis=0))) if pvalid.any() else None,
                    predicted_log_speed_std=float(log_speed[node, svalid].std()) if svalid.any() else None,
                    target_log_speed_std=float(np.log(b['target'][node, svalid, 2]+1e-6).std())
                                         if svalid.any() else None))

        def wins(rows):
            result = {}
            for region in fragment_rows[0]['regions']:
                valid = [r['regions'][region] for r in rows if r['regions'][region]['count']]
                result[region] = dict(evaluable_fragments=len(valid),
                    **{name: dict(
                        better=sum(row['034'] < row[name]-.001 for row in valid),
                        worse=sum(row['034'] > row[name]+.001 for row in valid),
                        tied=sum(abs(row['034']-row[name]) <= .001 for row in valid),
                        median_mae_delta=float(np.median([row['034']-row[name] for row in valid]))
                                         if valid else None)
                       for name in ('032', 'native', 'model_base', 'mean_residual')})
            return result
        sources_found = sorted(set(row['source'] for row in fragment_rows))
        residual_stats = {}
        for source in ['all', *sources_found]:
            selected = [row for row in residual_rows if source == 'all' or row['source'] == source]
            residual_stats[source] = {key: quantiles([row[key] for row in selected if row[key] is not None])
                                      for key in residual_rows[0] if key != 'source'}
            residual_stats[source]['mean_residual_sign'] = {
                'negative': sum(row['residual_circular_mean'] < -.001 for row in selected),
                'positive': sum(row['residual_circular_mean'] > .001 for row in selected),
                'near_zero': sum(abs(row['residual_circular_mean']) <= .001 for row in selected)}
        scene_comparisons = {}
        for region in ('all', 'inside_support', 'before_support', 'after_support'):
            scene_comparisons[region] = {}
            for arm in ('032', 'native', 'model_base'):
                deltas = [row['methods']['034']['regions'][region]['heading']['mae'] -
                          row['methods'][arm]['regions'][region]['heading']['mae'] for row in scene_rows]
                scene_comparisons[region][arm] = dict(
                    better=sum(delta < -.001 for delta in deltas),
                    worse=sum(delta > .001 for delta in deltas),
                    tied=sum(abs(delta) <= .001 for delta in deltas))
        result = dict(status='COMPLETED', date='2026-10-20',
            scope='034 candidate teacher self completion, not TMTAFF TCM or downstream association/fusion.',
            population={key: comparison[key] for key in ('scene_count', 'fragment_count', 'query_count')},
            training_audit=training, test_opened=False, optimizer_updates=0, inference_calls=0,
            threshold_reselected=False, observation_members_read=members,
            reference_policy='Only returned targets/masks. ZIP reads native observations and original association '
                             'metadata for unchanged V fragments; no reference/T_test or training members.',
            native_control_policy='Raw float64 cap128 inputs; shortest signed course/longitude interpolation '
                                  'inside native support and own endpoint hold outside. No target/mask in predictor.',
            precision_limit='Raw float64 input control, not a bit-identical FP32 normalized/inverted model base. '
                            '034 full state_mean/checkpoint was not returned. Uses exact signed half-turn convention; '
                            '034 adds FP32 0.000343-degree half-turn tolerance. Residual estimates include rounding.',
            model_base_control_policy='Second input-only control approximates 034 by casting raw headings and '
                'scene-relative times to FP32 and applying its 8*eps*360 half-turn tolerance. Omits unavailable '
                'normalization/inversion roundtrip. Residual ablations use this control to avoid attributing '
                'different half-turn choices to learned dynamics. No learned layers executed.',
            float64_vs_fp32_base_difference=precision_deltas,
            mean_ablation_policy='Circular mean heading residual and linear position/log-speed means over all Q64 '
                                 'queries, regardless of target validity; observational ablation, not trained alternative.',
            metrics={'032': describe(old), '034': describe(new), 'native': describe(controls),
                     'model_base_fp32_approx': describe(model_base_controls),
                     '034_heading_mean_residual': describe(heading_frozen),
                     '034_all_mean_residual': describe(all_frozen)},
            heading_fragment_comparisons={'all': wins(fragment_rows),
                **{source: wins([row for row in fragment_rows if row['source'] == source])
                   for source in sources_found}},
            fragment_tie_tolerance_degrees=.001,
            heading_scene_comparisons=scene_comparisons,
            residual_dynamics=residual_stats, per_scene=scene_rows,
            interpretation=['Single seed20 endpoint, 64 scenes nested in 17 identity components; descriptive only.',
                '9001/reference reuse limits observed-point and denoising claims.',
                'Endpoint-vs-zero residual attribution is descriptive, not an additive causal training decomposition.',
                'No observed heading dynamic contribution implies neither no query input nor no neural gradients.',
                'No threshold or alternative method selected; all original queries retained.'])
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        for name, values in result['metrics'].items():
            print(name, json.dumps({region: {key: data[key]['mae'] for key in ('position', 'speed', 'heading')}
                                   for region, data in values['overall']['regions'].items()}))
        print('residual', json.dumps(result['residual_dynamics']['all']))
        print('wins', json.dumps(result['heading_fragment_comparisons']['all']))


if __name__ == '__main__':
    main()
