"""Recompute 014 saved V scores and pool reported metrics; no data/model inference.

Physical coordinates were not returned in 014. Fusion errors below are checked
and reaggregated from scene/group sufficient statistics, not recomputed distances.
Run from any directory with the repository's existing Python environment.
"""
from collections import defaultdict
import gzip
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from association.evaluation.tmtaff import aggregate_fusion_reports
from association.evaluation.grouping import group_predicted_tracks, group_tracks_by_identity
from association.evaluation.physical import group_membership_diagnostics
from association.evaluation.association import evaluate_scenes

SOURCE = ROOT / 'HandOver/results/TMTAFF-TIME-EVAL-20261018-014/files/summary.json.gz'
OUTPUT = Path(__file__).with_suffix('.json')
REGIONS = ('all', 'observed', 'missing')
FEATURES = ('position', 'speed', 'heading')
COUNTS = ('tp', 'fp', 'fn', 'tn')


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'),
                    map(float, np.quantile(values, [0, .05, .25, .5, .75, .95, 1])))) if len(values) else None


def compare_tree(actual, expected, path=''):
    """Count/non-numeric equality is exact; record every floating sum difference."""
    differences = []
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            differences += compare_tree(actual[key], expected[key], path + '/' + str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected)):
            differences += compare_tree(left, right, path + '/' + str(i))
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (path, actual, expected)
        if actual != expected:
            differences.append({'path': path, 'absolute_difference': abs(actual - expected),
                                'relative_difference': abs(actual - expected) / max(abs(expected), 1e-300)})
    else:
        assert actual == expected, (path, actual, expected)
    return differences


def compact(metrics):
    keys = ('scene_count', 'pair_count', 'macro_f1', 'micro_f1', 'micro_precision',
            'micro_recall', 'tp', 'fp', 'fn', 'tn', 'negative_scene_count',
            'negative_scene_false_acceptance_rate')
    result = {key: metrics[key] for key in keys}
    result.update(positive_prevalence=(metrics['tp'] + metrics['fn']) / metrics['pair_count'],
                  acceptance_rate=(metrics['tp'] + metrics['fp']) / metrics['pair_count'],
                  fpr=metrics['fp'] / (metrics['fp'] + metrics['tn']), fdr=metrics['p_fa'])
    return result


def rank_and_scores(records):
    # Reuse the existing tie-aware ranking implementation without running its main.
    spec = importlib.util.spec_from_file_location('compare011_analysis', ROOT / 'docs/results/compare011_interim/analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scores = np.concatenate([r['scores'] for r in records])
    labels = np.concatenate([r['labels'] for r in records]).astype(bool)
    return {**module.rank_metrics(records), 'positive_score_quantiles': quantiles(scores[labels]),
            'negative_score_quantiles': quantiles(scores[~labels])}


def component_bootstrap(rows):
    components = sorted({r['component_id'] for r in rows})
    counts = np.asarray([[sum(r[k] for r in rows if r['component_id'] == cid) for k in COUNTS]
                         for cid in components], dtype=np.int64)
    rng = np.random.default_rng(20261018)
    draws = counts[rng.integers(len(components), size=(5000, len(components)))].sum(axis=1)
    tp, fp, fn, _ = draws.T
    values = {'precision': tp / (tp + fp), 'recall': tp / (tp + fn), 'micro_f1': 2 * tp / (2 * tp + fp + fn)}
    return {'component_count': len(components), 'replicates': 5000, 'seed': 20261018,
            'resampling_unit': 'shared-identity component; retain all its scenes per draw',
            'fixed_threshold': True, 'intervals_95_percentile': {
                key: list(map(float, np.quantile(value, [.025, .975]))) for key, value in values.items()},
            'scope': 'Exploratory sampling variability on the same selected V snapshot; not independent test inference, training-seed uncertainty or reselection.'}


def pooled_group_errors(groups):
    result = {}
    for region in REGIONS:
        result[region] = {}
        for feature in FEATURES:
            rows = [g['metrics'][region][feature] for g in groups]
            n = sum(r['intersection_count'] for r in rows)
            mae = math.fsum(r['mae'] * r['intersection_count'] for r in rows if r['intersection_count']) / n if n else None
            mse = math.fsum(r['mse'] * r['intersection_count'] for r in rows if r['intersection_count']) / n if n else None
            result[region][feature] = {'intersection_count': n, 'mae': mae,
                                      'mse': mse, 'rmse': math.sqrt(mse) if n else None}
    return result


def describe_groups(groups):
    maes = [g['metrics']['all']['position']['mae'] for g in groups
            if g['metrics']['all']['position']['intersection_count']]
    sizes = [len(g['node_indices']) for g in groups]
    valid_groups = [g for g in groups if g['metrics']['all']['position']['intersection_count']]
    correlation = None
    if len(valid_groups) > 1:
        x = np.asarray([len(g['node_indices']) - 1 for g in valid_groups])
        y = np.asarray([g['metrics']['all']['position']['mae'] for g in valid_groups])
        if np.std(x) and np.std(y):
            correlation = float(np.corrcoef(x, y)[0, 1])
    return {'scope': 'Descriptive group errors against each anchor; duplicates repeat targets and are not the full-population metric.',
            'group_count': len(groups), 'size_quantiles': quantiles(sizes),
            'recursive_depth_equals_size_minus_one': True,
            'group_mean_position_error_quantiles_m': quantiles(maes),
            'groups_with_mae_above_10_million_m': sum(x > 1e7 for x in maes),
            'group_mae_vs_recursive_depth_pearson_descriptive': correlation,
            'pooled_errors': pooled_group_errors(groups)}


def main():
    with gzip.open(SOURCE, 'rt') as handle:
        summary = json.load(handle)
    evaluation = summary['evaluation']
    assert not summary['test_opened'] and not evaluation['test_opened']
    records = [{**r, **{k: np.asarray(r[k]) for k in ('scores', 'labels', 'pairs', 'sources')}}
               for r in evaluation['records']]
    threshold = evaluation['threshold']
    association = evaluate_scenes(records, threshold)
    assert association == evaluation['association']
    association_result = {'threshold': threshold, 'complete_saved_metrics_exactly_equal': True,
                          'full': {**compact(association), **rank_and_scores(records)},
                          'per_scene': association['per_scene']}
    for key, same in (('itsa', True), ('mstsa', False)):
        filtered = []
        for r in records:
            left, right = r['pairs'].T
            mask = (r['sources'][left] == r['sources'][right]) == same
            filtered.append({**r, **{k: r[k][mask] for k in ('scores', 'labels', 'pairs')}})
        association_result[key] = {**compact(association[key]), **rank_and_scores(filtered),
                                  'scope': f'Fixed {key.upper()} source subset of full V_select; no fitting'}
    association_result['node_count_strata'] = {}
    for label, lo, hi in (('2_to_15', 2, 15), ('16_to_25', 16, 25), ('26_to_40', 26, 40), ('41_plus', 41, math.inf)):
        selected = [r for r in records if lo <= len(r['sources']) <= hi]
        if selected:
            association_result['node_count_strata'][label] = compact(evaluate_scenes(selected, threshold))
    association_result['component_bootstrap'] = component_bootstrap(association['per_scene'])
    association_result['interpretation'] = [
        'AUC/AP show weak positive ranking information, not random or perfect separation. Positive and negative score distributions overlap heavily.',
        'At the original threshold, high recall coexists with high FPR; the dense scenes have much lower positive prevalence and dominate micro counts.',
        'The original threshold was selected by V_select macro-F1, not a 0.5 training probability cutoff. Balanced training may affect learned scores, but these records cannot establish prior shift as the cause.',
        'The bootstrap conditions on one selected snapshot and threshold. It does not remove validation-selection bias or quantify convergence.'
    ]

    scenes = evaluation['per_scene']
    record_by_scene = {str(r['scene_id']): r for r in records}
    methods = list(evaluation['fusion'])
    reaggregated, float_differences = {}, {}
    for name in methods:
        reaggregated[name] = aggregate_fusion_reports([s['methods'][name] for s in scenes])
        diffs = compare_tree(reaggregated[name], evaluation['fusion'][name])
        float_differences[name] = {'count': len(diffs),
                                  'max_absolute': max(diffs, key=lambda d: d['absolute_difference'], default=None),
                                  'max_relative': max(diffs, key=lambda d: d['relative_difference'], default=None)}
    group_checks, group_reports, scene_distributions = {}, {}, {}
    no_joint_position, common_full_targets = [], []
    for name in ('predicted', 'oracle'):
        groups, diagnostics, unmatched, matched = [], defaultdict(int), 0, 0
        for scene in scenes:
            row = scene['methods'][name]
            record = record_by_scene[str(scene['scene_id'])]
            expected = (group_predicted_tracks(len(record['sources']), record['pairs'], record['scores'],
                        threshold=threshold, grouping_policy='anchor_direct', node_order='input') if name == 'predicted'
                        else group_tracks_by_identity(record['identities'], node_order='input'))
            actual = [g['node_indices'] for g in row['all_predicted_groups']]
            assert actual == expected, (name, scene['scene_id'])
            assert sorted(node for group in actual for node in group) == list(range(len(record['sources'])))
            memberships = [None] * len(record['sources'])
            for index, group in enumerate(actual):
                for node in group:
                    memberships[node] = index
            assert group_membership_diagnostics(memberships, record['identities']) == row['group_diagnostics']
            for key, value in row['group_diagnostics'].items():
                if isinstance(value, int):
                    diagnostics[key] += value
            unmatched += len(row['unmatched_target_ids'])
            matched += row['anchor_matched_target_count']
            unique = [g for g in row['all_predicted_groups'] if not g['duplicate_for_anchor_target']]
            # Original full-target population has extra unmatched rows. Its error
            # numerator comes only from the first group for each anchor identity.
            pooled = pooled_group_errors(unique)
            for region in REGIONS:
                for feature in FEATURES:
                    for key in ('intersection_count', 'mae', 'mse', 'rmse'):
                        compare_tree(pooled[region][feature][key], row['full_target_metrics'][region][feature][key])
            for g in row['all_predicted_groups']:
                groups.append({**g, 'scene_id': scene['scene_id']})
                if not g['duplicate_for_anchor_target'] and not g['metrics']['all']['position']['intersection_count']:
                    no_joint_position.append({'method': name, 'scene_id': scene['scene_id'],
                        'group_index': g['group_index'], 'node_indices': g['node_indices'],
                        'anchor_target_id': g['anchor_target_id'], 'has_position_output': g['has_position_output'],
                        'position': g['metrics']['all']['position']})
        unique = [g for g in groups if not g['duplicate_for_anchor_target']]
        duplicates = [g for g in groups if g['duplicate_for_anchor_target']]
        group_reports[name] = {'all_groups': describe_groups(groups),
            'all_singletons': describe_groups([g for g in groups if len(g['node_indices']) == 1]),
            'all_multi_node': describe_groups([g for g in groups if len(g['node_indices']) > 1]),
            'first_anchor_singletons': describe_groups([g for g in unique if len(g['node_indices']) == 1]),
            'first_anchor_multi_node': describe_groups([g for g in unique if len(g['node_indices']) > 1]),
            'duplicate_anchor_groups_diagnostic_only': describe_groups(duplicates),
            'multi_node_size_bins': {label: describe_groups([g for g in groups if lo <= len(g['node_indices']) <= hi])
                for label, lo, hi in (('2', 2, 2), ('3_to_4', 3, 4), ('5_to_8', 5, 8), ('9_plus', 9, math.inf))}}
        group_checks[name] = {'groups_exactly_reconstructed': True, 'all_nodes_assigned_exactly_once': True,
            'diagnostics_exactly_recomputed': True, 'full_errors_reconstructed_from_first_anchor_groups': True,
            'diagnostics_counts': dict(diagnostics), 'anchor_matched_targets': matched,
            'unmatched_targets': unmatched, 'duplicate_anchor_group_count': len(duplicates),
            'recursive_calls': sum(len(g['node_indices']) - 1 for g in groups)}
        scene_rows = []
        for scene in scenes:
            row = scene['methods'][name]
            position = row['full_target_metrics']['all']['position']
            scene_rows.append({'scene_id': scene['scene_id'], 'position': position,
                'group_count': row['predicted_group_count'], 'target_count': row['target_identity_count'],
                'singleton_count': sum(len(g['node_indices']) == 1 for g in row['all_predicted_groups']),
                'absolute_error_sum_m': position['mae'] * position['intersection_count']})
        total_error = math.fsum(r['absolute_error_sum_m'] for r in scene_rows)
        for row in scene_rows:
            row['share_of_total_absolute_error'] = row['absolute_error_sum_m'] / total_error
        scene_rows.sort(key=lambda r: r['position']['mae'])
        scene_distributions[name] = {'scene_mae_quantiles_m': quantiles([r['position']['mae'] for r in scene_rows]),
            'scenes_with_mae_above_10_million_m': sum(r['position']['mae'] > 1e7 for r in scene_rows),
            'all_scenes_sorted_by_mae_ascending': scene_rows}
    assert sum(r['recursive_calls'] for r in group_checks.values()) == evaluation['fusion_forward_count']

    common_groups = {name: [] for name in ('predicted', 'oracle')}
    for scene in scenes:
        maps = {name: {g['anchor_target_id']: g for g in scene['methods'][name]['all_predicted_groups']
                       if not g['duplicate_for_anchor_target']} for name in ('predicted', 'oracle')}
        for target in sorted(maps['predicted'].keys() & maps['oracle'].keys()):
            pg, og = maps['predicted'][target], maps['oracle'][target]
            p, o = [g['metrics']['all']['position'] for g in (pg, og)]
            # Both multi-node routes emit the complete same 64-query grid. If
            # each covers every available truth point, their position supports
            # coincide by construction; this is not inferred from equal counts alone.
            if (len(pg['node_indices']) > 1 and len(og['node_indices']) > 1
                    and p['available_prediction_count'] == o['available_prediction_count'] == 64
                    and p['intersection_count'] == p['available_target_count'] == o['available_target_count'] == o['intersection_count']):
                for region in REGIONS:
                    for feature in FEATURES:
                        pm, om = pg['metrics'][region][feature], og['metrics'][region][feature]
                        assert (pm['intersection_count'] == pm['available_target_count']
                                == om['available_target_count'] == om['intersection_count'])
                common_groups['predicted'].append(pg)
                common_groups['oracle'].append(og)
                common_full_targets.append({'scene_id': scene['scene_id'], 'target_id': target,
                    'position_points': p['intersection_count'], 'predicted_group_size': len(pg['node_indices']),
                    'oracle_group_size': len(og['node_indices']), 'predicted_mae_m': p['mae'], 'oracle_mae_m': o['mae']})
    common_full_targets.sort(key=lambda r: (str(r['scene_id']), r['target_id']))
    single_source_rows = []
    for scene in scenes:
        p = scene['methods']['single_source:9001']['full_target_metrics']['all']['position']
        single_source_rows.append({'scene_id': scene['scene_id'], 'mae_m': p['mae'],
            'intersection_count': p['intersection_count'], 'absolute_error_sum_m': (p['mae'] or 0) * p['intersection_count']})
    single_source_rows.sort(key=lambda r: r['absolute_error_sum_m'], reverse=True)
    total = math.fsum(r['absolute_error_sum_m'] for r in single_source_rows)
    for row in single_source_rows:
        row['share_absolute_error'] = row['absolute_error_sum_m'] / total

    # Both routes cover all observed reference positions in every scene, so
    # this bounded observed-region comparison does not mix missing-point support.
    for scene in scenes:
        p, q = [scene['methods'][name]['full_target_metrics']['observed']['position']
                for name in ('oracle', 'simple_fusion_oracle')]
        assert (p['intersection_count'] == p['available_target_count']
                == q['available_target_count'] == q['intersection_count'])
    observed_comparison = {name: reaggregated[name]['observed']['position']
                           for name in ('oracle', 'simple_fusion_oracle')}
    observed_comparison['scope'] = ('Observed region only: each route covers every available reference position '
                                    'on the evaluator common grid in each scene; summary reaggregation, not raw distance recomputation.')

    result = {'source': str(SOURCE.relative_to(ROOT)), 'scope': 'V_select only; CPU saved-score and summary arithmetic; no training/inference/data ZIP/T_test reads',
        'limitations': [
            '014 contains no per-query physical predictions/truth/masks. Physical distances and exact numerical coordinate validity cannot be independently recomputed here.',
            'Fusion MAE/MSE/RMSE are reaggregated reported scene/group sufficient statistics. Counts are checked exactly; floating sum tails tolerated at rtol=1e-12.',
            'Group and scene quantiles describe means, not point-error quantiles; point error distributions are unavailable.',
            'Full predicted and oracle error populations differ. Overall subtraction does not estimate the causal cost of association.',
            'A short selected snapshot is not a trained-method benchmark. No threshold or checkpoint was reselected.'],
        'selection': {k: v for k, v in summary['selection'].items() if k != 'validation'},
        'association': association_result,
        'fusion': {'reaggregated': reaggregated, 'floating_sum_differences': float_differences,
            'group_checks': group_checks, 'group_distributions': group_reports,
            'scene_distributions': scene_distributions, 'first_anchor_groups_without_joint_position': no_joint_position,
            'oracle_vs_simple_on_full_observed_position_support': observed_comparison,
            'matched_multi_node_full_position_support': {'target_count': len(common_full_targets),
                'scope': 'Same scene/anchor target, both multi-node groups emit all 64 grid points and cover all available reference positions; no matching or threshold change. Conditional diagnostic, not the full target population.',
                'oracle_lower_group_mae_count': sum(r['oracle_mae_m'] < r['predicted_mae_m'] for r in common_full_targets),
                'pooled': {name: pooled_group_errors(groups) for name, groups in common_groups.items()},
                'targets': common_full_targets},
            'single_source_9001_error_concentration': single_source_rows},
        'receipt_corrections': [
            'The selected fusion snapshot is TFM_FROZEN@2; source_cursor reports eight total updates including JOINT@2, but those JOINT weights were not selected for 014.',
            'Huge oracle multi-node errors establish a non-association-side problem in this selected snapshot. They do not show association has no effect, and global oracle-minus-predicted errors compare different supports.',
            'Heading MAE around 80–90 degrees alone cannot establish random outputs; a biased or nearly constant output can also have this MAE.',
            'Do not quote a universal 4–7-order improvement from mixed-coverage tables. Compare common support and report coverage; simple fusion has no missing-region predictions.',
            'Two selected TFM updates are established; undertraining versus coordinate/decoding/feedback causes cannot be isolated by these aggregate metrics alone.'],
        'execution': {'test_opened': False, 'threshold_reselected': False, 'optimizer_updates': 0,
                      'physical_distance_recomputed': False, 'fusion_recursive_calls_verified': evaluation['fusion_forward_count']}}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'association': association_result['full'],
                      'bootstrap': association_result['component_bootstrap'],
                      'group_checks': group_checks, 'common_targets': len(common_full_targets),
                      'no_joint_position': no_joint_position}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
