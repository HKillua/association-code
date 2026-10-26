"""Hand-calculated paper reporting: hierarchy, masks, units and unique targets."""
import json
import math
import unittest

import numpy as np

from association.evaluation.paper_fusion_metrics import paper_fusion_scene, aggregate_paper_fusion
from association.evaluation.physical import evaluate_aligned_fusion
from association.evaluation.scene import evaluate_full_scene
from association.evaluation.tmtaff import aggregate_fusion_reports


def longitude_report(errors):
    errors = np.asarray(errors, dtype=float)
    truth = np.zeros((*errors.shape, 4))
    prediction = truth.copy()
    prediction[..., 1] = -errors / 10000.
    valid = np.repeat(np.isfinite(errors)[..., None], 4, axis=-1)
    return paper_fusion_scene(prediction, truth, valid, prediction_valid=valid)


class PaperFusionMetricTests(unittest.TestCase):
    def test_target_then_scene_weighting_and_sample_sd_not_point_pooling(self):
        first = longitude_report([[1, 3, np.nan], [9, np.nan, np.nan]])
        second = longitude_report([[100, 100, 100]])
        row = aggregate_paper_fusion([first, second])['attributes']['longitude']
        self.assertAlmostEqual(first['attributes']['longitude']['target_mean']['ME'], 5.5)
        self.assertAlmostEqual(row['scene_equal_target_mean']['ME'], 52.75)
        self.assertAlmostEqual(row['scene_equal_target_mean']['residual_SD'], math.sqrt(2)/2)
        self.assertEqual(row['defined_targets']['ME'], 3)
        self.assertEqual(row['defined_targets']['residual_SD'], 2)
        self.assertEqual(row['evaluated_points'], 6)
        self.assertIsNone(row['strict_complete_population']['ME'])
        self.assertNotAlmostEqual(row['scene_equal_target_mean']['ME'], 313/6)

    def test_zero_one_point_and_missing_predictions_keep_declared_population(self):
        truth = np.zeros((3, 2, 4)); prediction = truth.copy()
        truth[..., 2] = 10; prediction[..., 2] = 8
        tv = np.ones_like(truth, bool); tv[1, 1] = False; tv[2] = False
        pv = tv.copy(); pv[0, 1] = False
        result = paper_fusion_scene(prediction, truth, tv, prediction_valid=pv)
        row = result['attributes']['speed']
        self.assertEqual(row['declared_targets'], 3)
        self.assertEqual(row['available_target_points'], 3)
        self.assertEqual(row['evaluated_points'], 2)
        self.assertEqual(row['missing_prediction_points'], 1)
        self.assertEqual(row['missing_reference_points'], 3)
        self.assertEqual(row['target_mean']['ME'], 2)
        self.assertIsNone(row['target_mean']['residual_SD'])
        self.assertIsNone(row['strict_complete_target_mean']['ME'])
        self.assertEqual(row['unit'], 'native_speed')
        self.assertEqual(row['paper_unit'], 'kn')
        self.assertFalse(row['paper_unit_verified'])
        json.dumps(aggregate_paper_fusion([result]), allow_nan=False)

    def test_raw_angular_bias_cancellation_is_distinct_from_unchanged_physical_metrics(self):
        truth = np.array([[[0, 179, 10, 359], [0, -179, 10, 1]]], dtype=float)
        prediction = np.array([[[0, -179, 10, 1], [0, 179, 10, 359]]], dtype=float)
        valid = np.ones_like(truth, bool)
        physical = evaluate_aligned_fusion(prediction, truth, valid,
            prediction_valid=valid, region_mask=np.ones((1, 2), bool))
        result = paper_fusion_scene(prediction, truth, valid, prediction_valid=valid)
        heading = result['attributes']['heading']['target_mean']
        self.assertEqual(heading['ME'], 0)
        self.assertEqual(heading['MAE'], 358)
        self.assertAlmostEqual(heading['residual_SD'], 358*math.sqrt(2))
        self.assertEqual(result['attributes']['longitude']['target_mean']['MAE'], 3580000)
        self.assertEqual(physical['heading']['mae'], 2)
        self.assertEqual(physical, evaluate_aligned_fusion(prediction, truth, valid,
            prediction_valid=valid, region_mask=np.ones((1, 2), bool)))
        self.assertFalse(result['metadata']['author_ESD_implementation_verified'])

    def test_literal_eq29_is_separate_and_duplicate_groups_are_not_targets(self):
        literal = longitude_report([[4, 6]])['attributes']['longitude']['target_mean']
        self.assertAlmostEqual(literal['residual_SD'], math.sqrt(2))
        self.assertAlmostEqual(literal['printed_Eq29_diagnostic'], math.sqrt(50))
        truth = np.zeros((1, 2, 4))
        first = truth[0].copy(); first[:, 1] = -.001
        duplicate = truth[0].copy(); duplicate[:, 1] = -1
        report = evaluate_full_scene(
            [{'node_indices': [0], 'states': first}, {'node_indices': [1], 'states': duplicate}],
            node_identities=['A', 'A'], target_identities=['A'], target_states=truth,
            target_valid=np.ones_like(truth, bool), node_observed=np.ones((2, 2), bool),
            split='V_select', grouping_kind='predicted', correspondence_policy='anchor_identity_first')
        row = report['paper_fusion_metrics']['attributes']['longitude']
        self.assertEqual(report['duplicate_prediction_group_count'], 1)
        self.assertEqual(row['declared_targets'], 1)
        self.assertEqual(row['target_mean']['ME'], 10)
        combined = aggregate_fusion_reports([report, report])
        self.assertEqual(combined['paper_fusion_metrics']['attributes']['longitude']
                         ['scene_equal_target_mean']['ME'], 10)
        self.assertEqual(combined['all']['position']['mae'], report['full_target_metrics']['all']['position']['mae'])


if __name__ == '__main__':
    unittest.main()
