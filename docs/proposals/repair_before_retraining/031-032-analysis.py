"""Reproduce the saved speed A/B audit and a fixed native-observation control.

No model, fitting, threshold selection, new reference read or test split access.
The control was fixed before scoring: linear speed, signed shortest-arc course,
and endpoint holding outside native support, using the original cap128 tokens.
"""
from pathlib import Path
import argparse
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


def training_summary():
    histories, configs, result = [], [], {}
    for tag in ('031', '032'):
        path = ROOT/'HandOver/results'/f'COOPERATIVE-SPEED-20261020-{tag}'
        history = json.loads(gzip.decompress((path/'history.json.gz').read_bytes()))
        assert [row['update'] for row in history] == list(range(1, 1001))
        histories.append(history)
        configs.append(json.loads((path/'actual_config.json').read_text()))
        result[tag] = dict(history_count=len(history),
            first_loss=history[0]['loss'], first_components=history[0]['loss_components'],
            preclip_gradient_median=float(np.median([row['gradient_norm'] for row in history])),
            clip_steps=sum(row['gradient_norm'] > 1 for row in history),
            windows={name: {key: float(np.mean([row['loss_components'][key] for row in part]))
                            for key in ('physical', 'identity_bce')}
                     for name, part in [('first100', history[:100]), ('last100', history[-100:])]})
    for config in configs:
        config['model'].pop('speed_output_policy')
    assert configs[0] == configs[1]
    keys = ('stage', 'routing', 'valid_counts', 'sampled_pairs', 'used_pairs',
            'removed_observations', 'update', 'sampling')
    assert all(a[key] == b[key] for a, b in zip(*histories) for key in keys)
    assert histories[0][0]['loss_components'] == histories[1][0]['loss_components']
    result['checks'] = dict(configs_equal_except_speed_output=True,
                           per_step_sampling_metadata_equal=True, first_loss_components_equal=True,
                           initial_checkpoint_tensors_independently_compared=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(prefix='mtad-speed-audit-') as scratch:
        directories = []
        for tag in ('031', '032'):
            source = ROOT/'HandOver/results'/f'COOPERATIVE-SPEED-20261020-{tag}'
            destination = Path(scratch)/tag
            destination.mkdir()
            for name in ('run.json', 'completion-evaluation.json'):
                (destination/name).write_bytes(gzip.decompress((source/(name+'.gz')).read_bytes()))
            with tarfile.open(source/'completion-predictions.tar.gz') as pack:
                pack.extractall(destination, filter='data')
            directories.append(destination)
        comparison = compare(*directories, Path(scratch)/'comparison.json')
        _, run, _, saved = _completed(directories[0])
        _, _, _, changed = _completed(directories[1])
        rows = []
        with MTADArchive(args.archive) as archive:
            for contract, data in zip(run['scene_contract'][1], saved):
                native = _record(archive.load_scene(contract['scene_id']),
                                 contract['component_id'], 'V_select', 128)
                scene = prepare_timed_scene(native, length=64, max_observations=128,
                                            time_unit_seconds=20., query_times=data['query_seconds'])
                for name in ('node_ids', 'identities', 'sources'):
                    np.testing.assert_array_equal(scene[name], data[name])
                np.testing.assert_array_equal(scene['native_observed'].sum(1),
                                              contract['native_observed_counts'])
                # prepare_timed_scene interpolates inside support only, with
                # signed half-turn handling. Fill each outside region by its
                # own native endpoint, never by a reference value.
                control = scene['values'].copy()
                for node, keep in enumerate(scene['native_observed']):
                    times = scene['native_times'][node, keep]
                    states = scene['native_values'][node, keep]
                    control[node, scene['grid'] < times[0]] = states[0]
                    control[node, scene['grid'] > times[-1]] = states[-1]
                if not np.isfinite(control).all():
                    raise ValueError('native control did not retain every prediction')
                rows.append({**data, 'prediction': control})
            observation_members = list(archive.members_read)

        keys = ('prediction', 'target', 'target_valid', 'nearest_seconds',
                *[k for k in saved[0] if k.startswith('region_')])
        def pooled(items):
            return {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}
        full = pooled(rows)
        sources = np.concatenate([row['sources'] for row in rows])
        controls = {'overall': _metrics(full), 'by_source': {}}
        for source in sorted(set(sources)):
            controls['by_source'][str(source)] = _metrics({k: v[sources == source] for k, v in full.items()})
        def compact(metrics):
            return {region: {key: value for key, value in data.items()
                              if key in ('position', 'speed', 'heading', 'raw_log_speed_mse')}
                    for region, data in metrics['regions'].items()}

        per_scene = []
        temporal = {name: [] for name in ('A', 'B', 'target')}
        improved = {key: 0 for key in ('speed_scenes', 'speed_fragments', 'heading_scenes',
                                       'inside_position_scenes')}
        for contract, row, a, b, comp in zip(run['scene_contract'][1], rows, saved, changed,
                                            comparison['per_scene']):
            for name, speed in [('A', a['prediction'][..., 2]), ('B', b['prediction'][..., 2]),
                                ('target', a['target'][..., 2])]:
                temporal[name].extend(float(p[m].std()) for p, m in
                                     zip(speed, a['target_valid'][..., 2]) if m.sum() >= 2)
            for i in range(len(a['node_ids'])):
                mask = a['target_valid'][i, :, 2]
                if mask.any():
                    improved['speed_fragments'] += int(np.abs(b['prediction'][i, mask, 2] -
                        a['target'][i, mask, 2]).mean() < np.abs(a['prediction'][i, mask, 2] -
                        a['target'][i, mask, 2]).mean())
            ca, cb = compact(comp['left']), compact(comp['right'])
            for region, attr, key in [('all', 'speed', 'speed_scenes'),
                                     ('all', 'heading', 'heading_scenes'),
                                     ('inside_support', 'position', 'inside_position_scenes')]:
                if ca[region][attr]['mae'] is not None and cb[region][attr]['mae'] is not None:
                    improved[key] += int(cb[region][attr]['mae'] < ca[region][attr]['mae'])
            # Per-scene means and denominators suffice; raw predictions are retained in the S2L.
            def scene_summary(metrics):
                return {region: {attr: {k: values[k] for k in ('intersection_count', 'mae', 'rmse')}
                                  for attr, values in data.items() if attr != 'raw_log_speed_mse'}
                        for region, data in compact(metrics).items()}
            per_scene.append(dict(scene_id=contract['scene_id'], component_id=contract['component_id'],
                fragment_count=len(row['node_ids']), native_control=scene_summary(_metrics(row)),
                A=scene_summary(comp['left']), B=scene_summary(comp['right'])))
        temporal = {name: dict(fragment_count=len(values), median=float(np.median(values)),
                              mean=float(np.mean(values))) for name, values in temporal.items()}
        a, b = pooled(saved), pooled(changed)
        valid = a['target_valid'][..., 2]
        target = a['target'][..., 2][valid]
        errors = {name: data['prediction'][..., 2][valid] - target for name, data in [('A', a), ('B', b)]}
        speed_strata = {}
        # Exploratory bins describe the returned errors only; they select no data or model.
        for name, mask in [('below_40', target < 40), ('at_least_40', target >= 40),
                           ('at_least_200', target >= 200)]:
            speed_strata[name] = dict(count=int(mask.sum()), fraction=float(mask.mean()),
                **{arm: dict(mae=float(np.abs(error[mask]).mean()),
                             squared_error_share=float(np.square(error[mask]).sum()/np.square(error).sum()))
                   for arm, error in errors.items()},
                contribution_to_pooled_mae_change=float((np.abs(errors['B'][mask]) -
                                                        np.abs(errors['A'][mask])).sum()/len(target)))
        result = dict(status='COMPLETED', date='2026-10-20',
            experiment_ids=['COOPERATIVE-SPEED-20261020-031', 'COOPERATIVE-SPEED-20261020-032'],
            scope='Full saved V_select teacher self predictions; single seed20, no independent repeat.',
            population={key: comparison[key] for key in ('scene_count', 'fragment_count', 'query_count')},
            training_audit=training_summary(),
            test_opened=False, optimizer_updates=0, inference_calls=0, threshold_reselected=False,
            observation_control='Original cap128 observations: native linear interpolation for speed; '
                'signed shortest-arc interpolation for course/longitude; endpoint hold outside support. '
                'Labels only assemble original fragments, never select helpers or fit predictions.',
            reference_policy='Only original returned per-attribute targets/masks; none read from ZIP. '
                '9001/reference reuse limits denoising claims. No extreme targets excluded.',
            source_results='HandOver/results/COOPERATIVE-SPEED-20261020-{031,032}/',
            observation_members_read=observation_members,
            A_B_overall=comparison['overall'], A_B_by_source=comparison['by_source'],
            native_control=controls, per_scene=per_scene,
            B_better_counts=improved, within_fragment_speed_std_on_same_valid_queries=temporal,
            exploratory_target_speed_strata=speed_strata,
            interpretation='Descriptive pooled/scene/component evidence; queries and fragments are dependent; '
                'no significance claim or automatic acceptance. A/B changed speed formula only, '
                'not a test of utility fusion, predicted association, or complete TMTAFF superiority.')
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        for method, values in [('A', comparison['overall']['left']),
                               ('B', comparison['overall']['right']), ('native', controls['overall'])]:
            print(method, json.dumps({region: {key: values['regions'][region][key]['mae']
                  for key in ('position', 'speed', 'heading')}
                  for region in ('all', 'inside_support', 'before_support', 'after_support')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
