"""Audit saved 015 scores and reported scene/group statistics, without inference.

The physical NPZ arrays remain on the server: distances and pointwise support
are NOT independently recomputed here. Labels are reconstructed from 015's saved
scene-contract identities and score-cache pairs, then checked against 014's full
records with the unchanged scene/node order. No threshold is reselected and no
test split or raw ZIP is read.
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

spec = importlib.util.spec_from_file_location(
    'analysis014', ROOT / 'docs/results/tmtaff014_local_review/numeric_analysis.py')
old_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_analysis)
SOURCE = ROOT / 'HandOver/results/COOPERATIVE-COST-20261018-015/files'
ARMS = ('plain', 'identity', 'utility', 'learned')
REGIONS = ('all', 'observed', 'missing')
FEATURES = ('position', 'speed', 'heading')


def read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def without_errors(tree):
    if isinstance(tree, dict):
        return {k: without_errors(v) for k, v in tree.items()
                if k not in ('mae', 'mse', 'rmse')}
    if isinstance(tree, list):
        return [without_errors(v) for v in tree]
    return tree


def describe_joint_groups(groups):
    """Reuse scalar aggregation, not 014's recursive pair-fusion interpretation."""
    result = old_analysis.describe_groups(groups)
    result.pop('recursive_depth_equals_size_minus_one')
    correlation = result.pop('group_mae_vs_recursive_depth_pearson_descriptive')
    # Pearson correlation is unchanged by subtracting one from group size.
    result['group_mae_vs_size_pearson_descriptive'] = correlation
    result['decoder_execution'] = 'One joint decode per group; no recursive pair fusion'
    return result


def pair_comparison(evaluations, records, reference, method, region, feature):
    """Paired component bootstrap of fixed saved mean-error numerators."""
    rows = []
    components = sorted({r['component_id'] for r in records})
    for r, ref, candidate in zip(records, evaluations[reference]['per_scene'],
                                  evaluations['utility']['per_scene']):
        left = ref['methods'][method]['full_target_metrics'][region][feature]
        right = candidate['methods'][method]['full_target_metrics'][region][feature]
        assert without_errors(left) == without_errors(right)
        n = left['intersection_count']
        if n:
            delta = right['mae'] - left['mae']
            rows.append({'scene_id': r['scene_id'], 'component_id': r['component_id'],
                         'intersection_count': n, 'reference_mae': left['mae'],
                         'utility_mae': right['mae'], 'utility_minus_reference_mae': delta,
                         'delta_absolute_error_sum': delta * n})
    counts = np.array([sum(r['intersection_count'] for r in rows if r['component_id'] == cid)
                       for cid in components], dtype=np.float64)
    delta_sums = np.array([math.fsum(r['delta_absolute_error_sum'] for r in rows
                                   if r['component_id'] == cid) for cid in components])
    rng = np.random.default_rng(20261018)
    ids = rng.integers(len(components), size=(5000, len(components)))
    denominators = counts[ids].sum(axis=1)
    valid = denominators > 0
    draws = delta_sums[ids].sum(axis=1)[valid] / denominators[valid]
    component_delta = np.divide(delta_sums, counts, out=np.full_like(counts, np.nan), where=counts > 0)
    rows.sort(key=lambda r: r['delta_absolute_error_sum'])
    baseline = evaluations[reference]['fusion'][method][region][feature]['mae']
    difference = math.fsum(r['delta_absolute_error_sum'] for r in rows) / sum(r['intersection_count'] for r in rows)
    favorable = math.fsum(-r['delta_absolute_error_sum'] for r in rows if r['delta_absolute_error_sum'] < 0)
    adverse = math.fsum(r['delta_absolute_error_sum'] for r in rows if r['delta_absolute_error_sum'] > 0)
    return {
        'direction': 'negative = utility lower error',
        'pooled_mae_difference': difference,
        'relative_mae_change': difference / baseline,
        'eligible_scene_count': len(rows),
        'utility_better_scenes': sum(r['utility_minus_reference_mae'] < 0 for r in rows),
        'utility_worse_scenes': sum(r['utility_minus_reference_mae'] > 0 for r in rows),
        'equal_scenes': sum(r['utility_minus_reference_mae'] == 0 for r in rows),
        'scene_mae_difference_quantiles': old_analysis.quantiles([r['utility_minus_reference_mae'] for r in rows]),
        'eligible_component_count': int((counts > 0).sum()),
        'utility_better_components': int((component_delta < 0).sum()),
        'utility_worse_components': int((component_delta > 0).sum()),
        'positive_improvement_absolute_error_sum': favorable,
        'new_worsening_absolute_error_sum': adverse,
        'largest_5_improvements_share_of_all_favorable_mass':
            sum(max(0, -r['delta_absolute_error_sum']) for r in rows[:5]) / favorable if favorable else None,
        'bootstrap': {'unit': '17 shared-identity components, all scenes retained per draw',
                      'seed': 20261018, 'replicates': 5000, 'nonempty_replicates': int(valid.sum()),
                      'percentile_95_interval': list(map(float, np.quantile(draws, [.025, .975]))),
                      'scope': 'Descriptive uncertainty for one short-trained saved snapshot. No training-seed uncertainty, convergence claim, multiplicity correction or independent test inference.'},
        'largest_5_improvements': rows[:5],
        'largest_5_worsenings': [r for r in reversed(rows) if r['utility_minus_reference_mae'] > 0][:5],
        **({'per_scene': rows} if feature == 'position' else {}),
    }


def main():
    evaluations = {arm: read(SOURCE / f'{arm}-evaluation.json.gz') for arm in ARMS}
    run = read(SOURCE / 'run.json.gz')
    old = read(ROOT / 'HandOver/results/TMTAFF-TIME-EVAL-20261018-014/files/summary.json.gz')['evaluation']
    assert not run['test_opened'] and all(not e['test_opened'] for e in evaluations.values())
    caches = {arm: np.load(SOURCE / f'{arm}-scores.npz', allow_pickle=False) for arm in ARMS}
    for arm in ARMS:
        assert caches[arm].files == caches['plain'].files
        for key in caches['plain'].files:
            assert np.array_equal(caches[arm][key], caches['plain'][key]), (arm, key)
    records = []
    contracts = run['scene_contract'][1]
    assert len(contracts) == len(old['records']) == 64
    for i, old_record in enumerate(old['records']):
        contract = contracts[i]
        metadata = read(SOURCE / 'plain-predictions-index' / f'{i:06d}.json')
        assert metadata['scene_id'] == old_record['scene_id']
        assert metadata['component_id'] == old_record['component_id']
        assert metadata['split'] == 'V_select' and not metadata['test_opened']
        for key in ('scene_id', 'component_id', 'node_ids', 'identities', 'sources'):
            assert contract[key] == old_record[key], (i, key)
        assert contract['split'] == 'V_select'
        for arm in ARMS:
            assert read(SOURCE / f'{arm}-predictions-index' / f'{i:06d}.json') == metadata
            assert evaluations[arm]['per_scene'][i]['scene_id'] == old_record['scene_id']
        assert np.array_equal(caches['plain'][f'pairs_{i}'], old_record['pairs'])
        pairs = caches['plain'][f'pairs_{i}']
        identities = np.asarray(contract['identities'])
        labels = identities[pairs[:, 0]] == identities[pairs[:, 1]]
        assert np.array_equal(labels, old_record['labels'])
        assert int(labels.sum()) == contract['positive_count']
        assert len(pairs) == contract['pair_count']
        records.append({**old_record, 'pairs': pairs, 'labels': labels, 'sources': np.asarray(contract['sources']),
                        'scores': caches['plain'][f'scores_{i}']})
    threshold = evaluations['plain']['threshold']
    assert all(e['threshold'] == threshold and e['selection'] is None for e in evaluations.values())
    association = evaluate_scenes(records, threshold)
    assert all(association == e['association'] for e in evaluations.values())
    scores = np.concatenate([r['scores'] for r in records])
    assert np.isfinite(scores).all()
    unique, counts = np.unique(scores, return_counts=True)
    assert run['identity_selection']['candidate_count'] == len(unique) + 1
    rejected = []
    for record in records:
        for i in np.flatnonzero(record['scores'] < threshold):
            rejected.append({'scene_id': record['scene_id'], 'pair': record['pairs'][i].tolist(),
                             'score': float(record['scores'][i]), 'label': int(record['labels'][i])})
    all_accept = evaluate_scenes(records, float(scores.min()))
    assoc_result = {'threshold': threshold, 'all_four_arms_scores_and_pairs_exactly_equal': True,
                    'all_four_arms_original_metrics_exactly_recomputed': True,
                    'old_014_candidate_arrays_exactly_equal': True,
                    'label_provenance': 'Reconstructed from 015 run.scene_contract validation identities and 015 score-cache pairs; checked against 014 ordered node identities/labels and 015 positive counts; final complete association metrics agree exactly with all 015 originals',
                    'full': {**old_analysis.compact(association), **old_analysis.rank_and_scores(records)},
                    'threshold_candidate_count_explanation': {'pair_count': len(scores),
                        'distinct_scores': len(unique), 'tied_score_group_count': int((counts > 1).sum()),
                        'extra_tied_entries': int((counts - 1).sum()),
                        'threshold_candidates': len(unique) + 1,
                        'formula': 'distinct finite score values + one all-reject threshold',
                        'source': 'association/evaluation/association.py select_threshold; this is not candidate pair count'},
                    'rejected_pairs': rejected,
                    'all_accept_descriptive_fixed_rule': old_analysis.compact(all_accept),
                    'component_bootstrap': old_analysis.component_bootstrap(association['per_scene'])}
    for key, same in (('itsa', True), ('mstsa', False)):
        subset = []
        for r in records:
            left, right = r['pairs'].T
            mask = (r['sources'][left] == r['sources'][right]) == same
            subset.append({**r, **{k: r[k][mask] for k in ('scores', 'labels', 'pairs')}})
        assoc_result[key] = {**old_analysis.compact(association[key]), **old_analysis.rank_and_scores(subset)}

    physical = {}
    for arm, evaluation in evaluations.items():
        results, differences = {}, {}
        for method in evaluation['fusion']:
            results[method] = aggregate_fusion_reports([s['methods'][method] for s in evaluation['per_scene']])
            diffs = old_analysis.compare_tree(results[method], evaluation['fusion'][method])
            differences[method] = {'floating_sum_differences': len(diffs),
                                   'max_absolute': max(diffs, key=lambda r: r['absolute_difference'], default=None)}
            assert without_errors(results[method]) == without_errors(evaluations['plain']['fusion'][method])
        groups = {}
        for method in ('predicted', 'oracle'):
            all_groups, diagnostics = [], defaultdict(int)
            for record, scene, base_scene in zip(records, evaluation['per_scene'], evaluations['plain']['per_scene']):
                row = scene['methods'][method]
                assert without_errors(row) == without_errors(base_scene['methods'][method])
                expected = (group_predicted_tracks(len(record['sources']), record['pairs'], record['scores'],
                            threshold=threshold, grouping_policy='anchor_direct', node_order='input')
                            if method == 'predicted' else group_tracks_by_identity(record['identities'], node_order='input'))
                assert expected == [g['node_indices'] for g in row['all_predicted_groups']]
                assert sorted(n for g in expected for n in g) == list(range(len(record['sources'])))
                membership = [None] * len(record['sources'])
                for k, group in enumerate(expected):
                    for node in group:
                        membership[node] = k
                assert group_membership_diagnostics(membership, record['identities']) == row['group_diagnostics']
                for key, value in row['group_diagnostics'].items():
                    if isinstance(value, int):
                        diagnostics[key] += value
                unique_groups = [g for g in row['all_predicted_groups'] if not g['duplicate_for_anchor_target']]
                pooled = old_analysis.pooled_group_errors(unique_groups)
                for region in REGIONS:
                    for feature in FEATURES:
                        for key in ('intersection_count', 'mae', 'mse', 'rmse'):
                            old_analysis.compare_tree(pooled[region][feature][key], row['full_target_metrics'][region][feature][key])
                all_groups.extend({**g, 'scene_id': scene['scene_id']} for g in row['all_predicted_groups'])
            groups[method] = {'diagnostics': dict(diagnostics),
                              'all': describe_joint_groups(all_groups),
                              'singletons': describe_joint_groups([g for g in all_groups if len(g['node_indices']) == 1]),
                              'multi_node': describe_joint_groups([g for g in all_groups if len(g['node_indices']) > 1])}
        physical[arm] = {'reaggregated': results, 'aggregate_rounding': differences, 'groups': groups}
    for name in ('simple_fusion_oracle', 'single_source:9001', 'single_source:9002'):
        for evaluation in evaluations.values():
            assert evaluation['fusion'][name] == old['fusion'][name]
    comparisons = {method: {region: {feature: {baseline: pair_comparison(
        evaluations, records, baseline, method, region, feature) for baseline in ('plain', 'identity', 'learned')}
        for feature in FEATURES} for region in REGIONS} for method in ('predicted', 'oracle')}
    output = {
        'scope': '015 V_select one-seed cost probe; 2 teacher + 2 utility + 4 arms x 2 adapt updates',
        'no_training_no_threshold_reselection_no_raw_zip_no_test': True,
        'association': assoc_result,
        'physical_audit_limits': [
            'Original physical arrays are retained server-side; this is exact population/group checking and floating-tolerance reaggregation of reported sufficient statistics, not independent coordinate-distance recomputation.',
            'Four arms have identical scene/component metadata, scores, groups, anchors and all reported per-group/per-scene coverage counts. Pointwise physical masks are not independently compared because NPZ is absent.',
            'All oracle outputs cover all 18364 available positions, so the oracle arm comparison covers the full declared reference population. Predicted metrics cover only the first anchor identity in each scene; no alternative correspondence is applied.',
            'Utility position improvements are descriptive short-training signals, not evidence of convergence, unique mechanism, multi-seed robustness or deployment advantage. Speed and heading are audited separately.'
        ],
        'physical': physical, 'utility_minus_controls': comparisons,
        'server_statement_corrections': [
            '29007 is the count of threshold choices (29006 unique scores plus all-reject), not 29007 candidate pairs. All 29008 pairs were scored and evaluated.',
            'The selected threshold is above the minimum score, not below every score; six true negatives are rejected.',
            'All 64 predicted scenes nevertheless form one group each, so the partition merges all 25711 different-identity pairs. The 6 rejected raw edges do not survive as separations under anchor_direct grouping.',
            'Zero negative scenes means negative-only scene false-acceptance rate is unavailable; over-merging itself is measurable here from full-population grouping diagnostics.'
        ],
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'association': assoc_result['full'], 'threshold_candidates': assoc_result['threshold_candidate_count_explanation'],
                      'grouping': physical['plain']['groups']['predicted']['diagnostics'],
                      'output': str(Path(__file__).with_suffix('.json'))}, indent=2))


if __name__ == '__main__':
    main()
