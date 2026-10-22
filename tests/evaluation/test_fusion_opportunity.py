"""067-B cache attribution must preserve population and separate evidence/cause."""
from copy import deepcopy
import unittest

import numpy as np

from association.evaluation.fusion_opportunity import (
    CLASSES, REGIONS, _classify, _points, _summarize, aggregate, analyze_scene,
)
from tests.evaluation import test_frozen_fusion_transfer as fixtures


class FusionOpportunityTests(unittest.TestCase):
    def fixture(self, **kwargs):
        return fixtures.FrozenFusionTransferTests.fixture(**kwargs)

    def test_duplicates_unused_anchors_and_missing_targets_keep_separate_views(self):
        fragment, artifacts, routing = self.fixture()
        row, arrays = analyze_scene(fragment, artifacts, routing, 's', 'c')
        self.assertEqual(row['unique_valid_position_point_count'], 6)
        self.assertEqual(row['duplicate_group_count'], 1)
        group = row['scopes']['all_output_groups']['all']
        unique = row['scopes']['unique_targets']['all']
        self.assertEqual(group['point_count'], 9)
        self.assertEqual(unique['point_count'], 6)
        self.assertEqual(group['mechanism_evidence']['positive_helper_mass_count'], 3)
        self.assertEqual(unique['mechanism_evidence']['positive_helper_mass_count'], 0)
        self.assertEqual(unique['mechanism_evidence']['zero_helper_mass_unchanged_count'], 3)
        self.assertEqual(row['scopes']['unique_targets']['anchor_outside_helper_supported']['point_count'], 1)
        self.assertEqual(len(arrays['unique_targets__target_index']), 6)
        self.assertNotIn(2, arrays['unique_targets__target_index'])  # no output for target c
        self.assertNotIn(1, arrays['unique_targets__group_index'])   # duplicate group excluded

    def test_classes_are_disjoint_with_all_ties_separate(self):
        rule = np.array([-2., 2., -2., 2., 0., -2., 0., 1e-10])
        model = np.array([2., -2., -1., 1., 2., 0., 0., -1e-10])
        np.testing.assert_array_equal(_classify(rule, model), [0, 1, 2, 3, 4, 5, 6, 6])
        f, a, r = self.fixture()
        _, arrays = analyze_scene(f, a, r, 's', 'c')
        example = _points(arrays, 'unique_targets')
        points = {key: np.resize(value, 8) for key, value in example.items()}
        points.update(anchor_error_m=np.full(8, 10.), rule_error_m=10.+rule, model_error_m=10.+model)
        report = _summarize(points, np.ones(8, bool))
        self.assertEqual(sum(v['point_count'] for v in report['classes'].values()), 8)
        self.assertEqual(report['classes'][CLASSES[0]]['model_vs_anchor']['harm_sum_m'], 2.)
        self.assertEqual(report['classes'][CLASSES[1]]['model_vs_anchor']['improvement_sum_m'], 2.)
        self.assertEqual(report['model_minus_rule']['signed_delta']['p99'], np.quantile(model-rule, .99))
        self.assertAlmostEqual(report['model_minus_rule_squared_error_m2']['sum'],
                               (np.square(10.+model)-np.square(10.+rule)).sum())

    def test_nonexclusive_mechanism_gap_contributions_do_not_double_count_union(self):
        f, a, r = self.fixture()
        _, arrays = analyze_scene(f, a, r, 's', 'c')
        points = {key: value[:4].copy() for key, value in _points(arrays, 'unique_targets').items()}
        points.update(anchor_error_m=np.full(4, 10.), rule_error_m=np.array([5., 8., 15., 10.]),
            model_error_m=np.array([10., 9., 20., 9.]), helper_mass=np.array([0., .5, .5, .5]),
            model_state_change_m=np.array([0., 1., 10., 1.]), helper_count=np.ones(4),
            supported_helper_count=np.ones(4), supported_helper_better_than_anchor_count=np.array([1, 1, 0, 0]),
            signed_progress_toward_rule=np.array([0., .5, .2, .5]),
            supported_helper_wrong_identity_count=np.array([0., 0., 1., 0.]),
            group_is_pure=np.array([1., 1., 0., 1.]))
        report = _summarize(points, np.ones(4, bool))
        evidence = report['mechanism_evidence']
        flags = evidence['nonexclusive_gap_contributions']
        expected = dict(beneficial_supported_helper_zero_mass_unchanged=[0],
            positive_mass_partial_progress_rule_improves=[1], supported_wrong_identity=[2],
            pure_group=[0, 1, 3], supported_helpers_all_no_better=[2, 3])
        gap = points['model_error_m']-points['rule_error_m']
        squared = np.square(points['model_error_m'])-np.square(points['rule_error_m'])
        for name, indices in expected.items():
            self.assertEqual(flags[name]['point_count'], len(indices))
            self.assertEqual(flags[name]['model_minus_rule_sum_m'], gap[indices].sum())
            self.assertEqual(flags[name]['gross_positive_gap_m'], gap[indices][gap[indices] > 0].sum())
            self.assertEqual(flags[name]['gross_negative_gap_m'], gap[indices][gap[indices] < 0].sum())
            self.assertEqual(flags[name]['model_minus_rule_squared_sum_m2'], squared[indices].sum())
        self.assertEqual(evidence['multiple_flag_point_count'], 4)
        self.assertGreater(sum(flag['point_count'] for flag in flags.values()), report['point_count'])
        self.assertEqual(evidence['any_flag_union_gap']['point_count'], report['point_count'])
        self.assertEqual(evidence['any_flag_union_gap']['model_minus_rule_sum_m'],
                         report['model_minus_rule']['signed_delta']['sum'])

    def test_positive_weights_can_cancel_and_partial_progress_is_not_a_verdict(self):
        f, a, r = self.fixture(cancellation=True)
        row, arrays = analyze_scene(f, a, r, 's', 'c')
        evidence = row['scopes']['unique_targets']['all']['mechanism_evidence']
        self.assertEqual(evidence['positive_helper_mass_count'], 3)
        self.assertEqual(evidence['positive_helper_mass_unchanged_count'], 3)
        self.assertGreater(evidence['distributions']['supported_helper_local_spread_rms_m']['max'], 0)
        self.assertIn('signed_progress_toward_rule', _points(arrays, 'unique_targets'))
        self.assertNotIn('underweighted_point_count', evidence)
        f, a, r = self.fixture(singleton=True)
        row, _ = analyze_scene(f, a, r, 's', 'c')
        evidence = row['scopes']['unique_targets']['all']['mechanism_evidence']
        self.assertEqual(evidence['helper_present_count'], 0)
        self.assertIsNone(evidence['distributions']['supported_helper_best_error_m']['mean'])

    def test_no_reference_is_unidentifiable_not_a_zero_error(self):
        f, a, r = self.fixture(sparse=True)
        row, arrays = analyze_scene(f, a, r, 's', 'c')
        self.assertEqual(row['unique_valid_position_point_count'], 1)
        self.assertEqual(row['unique_reference_unavailable_query_count'], 5)
        self.assertTrue(np.isfinite(arrays['unique_targets__anchor_error_m']).all())
        self.assertEqual(row['scopes']['unique_targets']['all']['point_count'], 1)
        del f['identities']
        row, _ = analyze_scene(f, a, r, 's', 'c')
        self.assertEqual(row['scopes']['unique_targets']['all']['mechanism_evidence']['identity_unidentifiable_count'], 1)

    def test_supported_helper_errors_are_against_anchor_target_and_identity_posthoc(self):
        f, a, r = self.fixture()
        f['identities'][1] = 'b'  # cached posthoc flag only, no regrouping
        row, arrays = analyze_scene(f, a, r, 's', 'c')
        points = _points(arrays, 'unique_targets')
        opportunity = points['region_anchor_outside_helper_supported']
        self.assertEqual(points['supported_helper_wrong_identity_count'][opportunity].tolist(), [1.])
        self.assertEqual(points['supported_helper_best_error_m'][opportunity].tolist(), [0.])
        self.assertEqual(row['scopes']['unique_targets']['anchor_outside_helper_supported']['mechanism_evidence']['mixed_group_count'], 1)

    def test_pooling_top_contributions_and_components_keep_scene_target_identity(self):
        rows, arrays = [], []
        for scene, sparse in (('2', False), ('1', True)):
            f, a, r = self.fixture(sparse=sparse)
            row, array = analyze_scene(f, a, r, scene, 'shared')
            rows.append(row)
            arrays.append(array)
        report = aggregate(rows, arrays)
        self.assertEqual(report['population']['unique_valid_position_point_count'], 7)
        self.assertEqual(report['component_count'], 1)
        self.assertEqual(report['components']['shared']['scene_count'], 2)
        self.assertEqual(report['scopes']['all_output_groups']['all']['point_count'], 11)
        error = np.concatenate([a['unique_targets__anchor_error_m'] for a in arrays])
        self.assertAlmostEqual(report['scopes']['unique_targets']['all']['errors']['anchor']['distribution']['p95'],
                               np.quantile(error, .95))
        ranks = report['top_contributors']['all']['absolute_error']['scene_targets']
        for direction in ('rule_better', 'model_better'):
            for record in ranks[direction]:
                self.assertEqual(record['component_id'], 'shared')
                self.assertIn(record['scene_id'], ('1', '2'))
                self.assertLessEqual(len(ranks[direction]), 20)
        reversed_report = aggregate(rows[::-1], arrays[::-1])
        self.assertEqual(report['top_contributors'], reversed_report['top_contributors'])
        corrupt = deepcopy(arrays)
        corrupt[0]['unique_targets__model_error_m'][0] += 1
        with self.assertRaisesRegex(ValueError, 'cached points'):
            aggregate(rows, corrupt)

    def test_fixed_population_and_routes_cannot_change(self):
        f, a, r = self.fixture()
        a['helper_quality']['aligned_prediction_valid'][0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, 'fixed'):
            analyze_scene(f, a, r, 's', 'c')
        f, a, r = self.fixture()
        r['helper_quality']['helper_weights'][0, 0, 0] = .4
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            analyze_scene(f, a, r, 's', 'c')


if __name__ == '__main__':
    unittest.main()
