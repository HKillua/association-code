"""Read-only audit of 016 reports and its 16 returned scene NPZ artifacts.

No model load, inference, training, archive access or threshold selection. Full
64-scene scalar reports are cross-checked against 015; independent coordinate
recomputation is explicitly limited to the four returned case scenes.
"""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.evaluation.artifacts import load_scene_prediction
from tools.analysis.compare_physical_predictions import _validate_pair

SOURCE = ROOT / 'HandOver/results/COOPERATIVE-ARRAYS-20261018-016/files'
PREVIOUS = ROOT / 'HandOver/results/COOPERATIVE-COST-20261018-015/files'
ARMS = ('plain', 'identity', 'utility', 'learned')
METHODS = ('predicted', 'oracle')
REGIONS = ('all', 'observed', 'missing')
FEATURES = ('position', 'speed', 'heading')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def differing_leaves(left, right, path=''):
    assert isinstance(left, dict) == isinstance(right, dict)
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return [row for key in left for row in differing_leaves(left[key], right[key], path+'/'+key)]
    return [] if left == right else [{'path': path, 'original': left, 'common': right}]


def independent_metrics(artifact, region, prediction_valid=None):
    """Use unit-vector atan2 cross/dot geodesics, independent of metric module.

    The production implementation uses a haversine formula. Mask/coverage logic
    is independently written here and does not call evaluate_aligned_fusion.
    """
    pred = artifact['aligned_prediction'].astype(np.float64)
    truth = artifact['aligned_truth'].astype(np.float64)
    pv = artifact['aligned_prediction_valid'] if prediction_valid is None else prediction_valid
    tv = artifact['aligned_truth_valid']
    observed = artifact['aligned_observed']
    region_mask = np.ones(observed.shape, bool) if region == 'all' else observed if region == 'observed' else ~observed
    result = {}
    for feature, columns in (('position', [0, 1]), ('speed', [2]), ('heading', [3])):
        pvalid = pv[..., columns].all(axis=-1) & region_mask
        tvalid = tv[..., columns].all(axis=-1) & region_mask
        overlap = pvalid & tvalid
        count = int(overlap.sum())
        target_count = int(tvalid.sum())
        target_rows = int(tvalid.any(axis=1).sum())
        covered_rows = int(overlap.any(axis=1).sum())
        p, t = pred[overlap], truth[overlap]
        if feature == 'position':
            def unit_vector(x):
                lat, lon = np.deg2rad(x[:, 0]), np.deg2rad(x[:, 1])
                return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))
            p3, t3 = unit_vector(p), unit_vector(t)
            errors = 6_371_000.0 * np.arctan2(np.linalg.norm(np.cross(p3, t3), axis=1), np.einsum('ij,ij->i', p3, t3))
        elif feature == 'speed':
            errors = np.abs(p[:, 2] - t[:, 2])
        else:
            angle = np.abs(np.remainder(p[:, 3], 360.0) - np.remainder(t[:, 3], 360.0))
            errors = np.minimum(angle, 360.0-angle)
        result[feature] = dict(available_target_count=target_count,
            available_prediction_count=int(pvalid.sum()), intersection_count=count,
            missing_prediction_count=target_count-count,
            prediction_without_target_count=int((pvalid & ~tvalid).sum()),
            coverage=count/target_count if target_count else None,
            available_target_trajectory_count=target_rows,
            covered_target_trajectory_count=covered_rows,
            missing_target_trajectory_count=target_rows-covered_rows,
            trajectory_coverage=covered_rows/target_rows if target_rows else None,
            mae=float(np.mean(errors)) if count else None,
            mse=float(np.mean(errors**2)) if count else None,
            rmse=float(np.sqrt(np.mean(errors**2))) if count else None)
    return result


def compare_metrics(recomputed, saved, deltas):
    for feature in FEATURES:
        for key, actual in recomputed[feature].items():
            expected = saved[feature][key]
            if actual is None or expected is None:
                assert actual is expected, (feature, key, actual, expected)
            elif key in ('mae', 'mse', 'rmse'):
                delta = abs(actual-expected)
                deltas[feature][key] = max(deltas[feature][key], delta)
                assert np.isclose(actual, expected, rtol=1e-11, atol=1e-7), (feature,key,actual,expected)
            else:
                assert actual == expected, (feature,key,actual,expected)


def main():
    prior = {arm: read(PREVIOUS / (arm+'-evaluation.json.gz')) for arm in ARMS}
    inspection = read(SOURCE/'inspection.json')
    reports = {}
    report_audit = []
    for report_path in sorted(SOURCE.glob('*/comparison.json')):
        report = read(report_path)
        name = report_path.parent.name
        reports[name] = report
        left_arm, method = name.split('-vs-utility-')
        assert method == report['method'] and report['status'] == 'COMPLETED'
        assert report['test_opened'] is False and report['scene_count'] == len(report['per_scene']) == 64
        assert report['target_count'] == sum(x['target_count'] for x in report['per_scene']) == 377
        for side, arm in (('left', left_arm), ('right', 'utility')):
            assert report['original'][side] == prior[arm]['fusion'][method], (name,side)
            for row, previous in zip(report['per_scene'], prior[arm]['per_scene']):
                assert row['scene_id'] == previous['scene_id']
                assert row['original'][side] == previous['methods'][method]['full_target_metrics'], (name,side,row['scene_id'])
        differences = differing_leaves(report['original'], report['common'])
        assert {r['path'].split('/')[-1] for r in differences} == {'available_prediction_count', 'prediction_without_target_count'}
        assert len(differences) == 36  # 2 sides x 3 regions x 3 quantities x 2 counts.
        scene_differences = [r for s in report['per_scene'] for r in differing_leaves(s['original'], s['common'])]
        assert {r['path'].split('/')[-1] for r in scene_differences} <= {'available_prediction_count', 'prediction_without_target_count'}
        per_region = {region: {key: {'original': report['original']['left'][region]['position'][key],
                                      'common': report['common']['left'][region]['position'][key]}
                        for key in ('available_prediction_count','prediction_without_target_count','intersection_count',
                                    'available_target_count','coverage','trajectory_coverage')}
                        for region in REGIONS}
        report_audit.append(dict(report=name, original_matches_015_aggregate_and_all_64_scenes_exactly=True,
            aggregate_differing_leaf_count=len(differences), differing_fields=sorted({r['path'].split('/')[-1] for r in differences}),
            all_error_statistics_and_reference_denominators_exactly_equal=True,
            per_scene_differing_leaf_count=len(scene_differences), position_counts_by_region=per_region))
    assert len(reports) == 6

    deltas = {feature: {key: 0.0 for key in ('mae','mse','rmse')} for feature in FEATURES}
    cases = []
    pair_validation_count = 0
    metrics_comparisons = 0
    for case_dir in sorted((SOURCE/'cases').iterdir()):
        loaded = {arm: load_scene_prediction(case_dir/arm/(case_dir.name+'.json')) for arm in ARMS}
        base = loaded['utility']
        metadata = base['metadata']
        scene_id, scene_index = metadata['scene_id'], metadata['scene_index']
        case = dict(scene_id=scene_id, scene_index=scene_index,
                    node_count=len(base['scene']['node_ids']),
                    target_count=len(base['methods']['oracle']['target_ids']), arms={})
        for arm in ARMS:
            item = loaded[arm]
            assert item['metadata'] == read(PREVIOUS/(arm+'-predictions-index')/(case_dir.name+'.json'))
            assert item['metadata']['test_opened'] is False
            case['arms'][arm] = {}
            for method in METHODS:
                a, b, n = _validate_pair(item, base, item['metadata'], metadata, method)
                pair_validation_count += 1
                for key in ('aligned_prediction_valid','aligned_group_indices','node_group_indices'):
                    assert np.array_equal(a[key], b[key]), (scene_id,arm,method,key)
                for key in ('aligned_truth','aligned_truth_valid','aligned_observed'):
                    assert np.array_equal(a[key], b[key], equal_nan=True), (scene_id,arm,method,key)
                assert len(a['aligned_prediction']) == n, 'unexpected duplicate rows in selected cases'
                recomputed = {region: independent_metrics(a, region) for region in REGIONS}
                previous = prior[arm]['per_scene'][scene_index]
                assert previous['scene_id'] == scene_id
                for region in REGIONS:
                    compare_metrics(recomputed[region], previous['methods'][method]['full_target_metrics'][region], deltas)
                    metrics_comparisons += 1
                case['arms'][arm][method] = recomputed
                report_names = [arm+'-vs-utility-'+method] if arm != 'utility' else [x+'-vs-utility-'+method for x in ('plain','identity','learned')]
                side = 'right' if arm == 'utility' else 'left'
                common = a['aligned_prediction_valid'] & b['aligned_prediction_valid'] & a['aligned_truth_valid']
                common_recomputed = {region: independent_metrics(a, region, common) for region in REGIONS}
                for name in report_names:
                    row = reports[name]['per_scene'][scene_index]
                    assert row['scene_id'] == scene_id
                    for region in REGIONS:
                        compare_metrics(recomputed[region], row['original'][side][region], deltas)
                        compare_metrics(common_recomputed[region], row['common'][side][region], deltas)
                        metrics_comparisons += 2
        case['oracle_utility_minus_identity_mae_metres'] = (case['arms']['utility']['oracle']['all']['position']['mae']-
                                                         case['arms']['identity']['oracle']['all']['position']['mae'])
        cases.append(case)
    assert len(cases) == 4
    case_selection = inspection['case_selection']
    result = dict(status='VERIFIED_WITH_REPORT_WORDING_CORRECTIONS',
        scope={'full_validation': 'Six returned 64-scene reports compared exactly with original 015 scalars; no claim of local access to all 256 NPZ files.',
               'local_raw_arrays': 'Four scenes x four arms = 16 NPZ independently recomputed, with saved original masks and correspondence; no model, ZIP or T_test accessed.'},
        six_reports=report_audit,
        common_vs_original_correction='Errors, evaluated intersections and reference/trajectory denominators are unchanged, but prediction counts and prediction-without-reference counts are not equal. The common view removes only prediction points lacking truth in these returned reports; retain original counts.',
        local_npz_count=16, case_scene_count=4, loader_pair_validations=pair_validation_count,
        region_metric_comparisons=metrics_comparisons,
        coordinate_formula='Independent unit-vector atan2(norm(cross), dot) spherical distance with R=6371000m; production uses haversine.',
        maximum_absolute_recomputation_difference=deltas,
        all_case_inputs_truth_masks_groups_and_support_equal_across_arms=True,
        all_case_metadata_matches_015_exactly=True,
        cases=cases,
        no_inference_training_threshold_selection_or_test_access=True)
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status','local_npz_count','case_scene_count','loader_pair_validations','region_metric_comparisons','maximum_absolute_recomputation_difference')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
