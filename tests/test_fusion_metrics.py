"""Hand-computed physical errors/coverage and full-population group diagnostics."""

import json
import math
import unittest

import numpy as np

from association.fusion_metrics import (
    EARTH_RADIUS_METRES, evaluate_aligned_fusion, group_membership_diagnostics,
)


def evaluate(prediction, target, target_valid=None, prediction_valid=None, region=None):
    prediction, target = np.asarray(prediction, dtype=float), np.asarray(target, dtype=float)
    if target_valid is None:
        target_valid = np.ones(target.shape, dtype=bool)
    if prediction_valid is None:
        prediction_valid = np.ones(prediction.shape, dtype=bool)
    if region is None:
        region = np.ones(target.shape[:2], dtype=bool)
    return evaluate_aligned_fusion(prediction, target, target_valid,
                                  prediction_valid=prediction_valid, region_mask=region)


class FusionMetricTests(unittest.TestCase):
    def test_spherical_metres_across_date_line_and_at_antipodes(self):
        prediction = [[[0, -179.5, 1, 0], [1, 0, 1, 0], [0, 180, 1, 0]]]
        target = [[[0, 179.5, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]]
        result = evaluate(prediction, target)["position"]
        degree = EARTH_RADIUS_METRES * math.pi / 180
        errors = np.array([degree, degree, math.pi * EARTH_RADIUS_METRES])
        self.assertAlmostEqual(result["mae"], errors.mean(), places=7)
        self.assertAlmostEqual(result["mse"], np.square(errors).mean(), delta=1)
        self.assertAlmostEqual(result["rmse"], np.sqrt(np.square(errors).mean()), places=7)
        self.assertEqual(result["unit"], "metres")
        self.assertEqual(result["coverage"], 1)

    def test_native_speed_and_shortest_circular_heading_errors(self):
        prediction = [[[0, 0, 800, 359], [0, 0, 400, 1], [0, 0, 600, 721]]]
        target = [[[0, 0, 400, 1], [0, 0, 600, 359], [0, 0, 600, 359]]]
        result = evaluate(prediction, target)
        self.assertEqual(result["speed"]["unit"], "native_speed")
        self.assertEqual(result["speed"]["mae"], 200)
        self.assertAlmostEqual(result["speed"]["mse"], (400**2 + 200**2) / 3)
        self.assertEqual(result["heading"]["mae"], 2)
        self.assertEqual(result["heading"]["mse"], 4)
        self.assertEqual(result["heading"]["rmse"], 2)
        half = evaluate([[[0, 0, 1, 180]]], [[[0, 0, 1, 0]]])
        self.assertEqual(half["heading"]["mae"], 180)

    def test_partial_masks_nan_and_missing_predictions_keep_target_denominator(self):
        target = np.array([[[0, 0, 2, 10], [1, 1, 4, 20],
                            [np.nan, np.nan, 6, np.nan], [np.nan]*4]])
        prediction = np.array([[[0, 0, 2, 10], [np.nan]*4,
                                [np.nan, np.nan, 8, np.nan], [1, 2, 10, 30]]])
        tv, pv = np.isfinite(target), np.isfinite(prediction)
        result = evaluate(prediction, target, tv, pv)
        for quantity in ("position", "heading"):
            row = result[quantity]
            self.assertEqual(row["available_target_count"], 2)
            self.assertEqual(row["intersection_count"], 1)
            self.assertEqual(row["missing_prediction_count"], 1)
            self.assertEqual(row["coverage"], .5)
            self.assertEqual(row["prediction_without_target_count"], 1)
            self.assertEqual(row["mae"], 0)
        speed = result["speed"]
        self.assertEqual(speed["available_target_count"], 3)
        self.assertEqual(speed["available_prediction_count"], 3)
        self.assertEqual(speed["intersection_count"], 2)
        self.assertEqual(speed["coverage"], 2/3)
        self.assertEqual(speed["mae"], 1)
        self.assertEqual(speed["mse"], 2)
        self.assertEqual(speed["rmse"], math.sqrt(2))
        json.dumps(result, allow_nan=False)

    def test_position_requires_both_coordinates_and_does_not_mix_feature_masks(self):
        target = np.array([[[0., 0., 1., 0.], [1., 1., 1., 0.]]])
        prediction = target.copy()
        tv = np.ones_like(target, dtype=bool)
        pv = tv.copy(); pv[0, 0, 1] = False
        prediction[0, 0, 1] = np.nan
        tv[0, 1, 0] = False; target[0, 1, 0] = np.nan
        result = evaluate(prediction, target, tv, pv)
        self.assertEqual(result["position"]["available_target_count"], 1)
        self.assertEqual(result["position"]["intersection_count"], 0)
        self.assertEqual(result["position"]["coverage"], 0)
        self.assertIsNone(result["position"]["mae"])
        self.assertEqual(result["speed"]["coverage"], 1)
        self.assertEqual(result["heading"]["coverage"], 1)

    def test_whole_missed_target_retained_in_point_and_trajectory_coverage(self):
        target = np.zeros((2, 3, 4)); prediction = target.copy()
        pv = np.ones_like(target, dtype=bool); pv[1] = False
        prediction[1] = np.nan
        result = evaluate(prediction, target, prediction_valid=pv)
        for quantity in ("position", "speed", "heading"):
            row = result[quantity]
            self.assertEqual(row["available_target_count"], 6)
            self.assertEqual(row["intersection_count"], 3)
            self.assertEqual(row["missing_prediction_count"], 3)
            self.assertEqual(row["available_target_trajectory_count"], 2)
            self.assertEqual(row["covered_target_trajectory_count"], 1)
            self.assertEqual(row["missing_target_trajectory_count"], 1)
            self.assertEqual(row["trajectory_coverage"], .5)
            self.assertEqual(row["coverage"], .5)

    def test_explicit_regions_do_not_infer_observed_from_values(self):
        target = np.zeros((1, 2, 4)); prediction = target.copy()
        prediction[0, :, 2] = [3, 7]
        observed = np.array([[True, False]])
        first = evaluate(prediction, target, region=observed)
        second = evaluate(prediction, target, region=~observed)
        self.assertEqual(first["speed"]["mae"], 3)
        self.assertEqual(second["speed"]["mae"], 7)
        self.assertEqual(first["region_point_count"], 1)
        self.assertEqual(second["speed"]["available_target_count"], 1)
        with self.assertRaises(TypeError):
            evaluate_aligned_fusion(prediction, target, np.ones_like(target, dtype=bool))

    def test_zero_target_zero_predictions_empty_rows_and_empty_region_are_null(self):
        target = np.full((1, 2, 4), np.nan)
        unavailable = np.zeros(target.shape, dtype=bool)
        no_truth = evaluate(np.zeros_like(target), target, target_valid=unavailable)
        for quantity in ("position", "speed", "heading"):
            self.assertEqual(no_truth[quantity]["available_target_count"], 0)
            self.assertIsNone(no_truth[quantity]["coverage"])
            self.assertIsNone(no_truth[quantity]["mae"])
            self.assertEqual(no_truth[quantity]["prediction_without_target_count"], 2)
        no_pred = evaluate(target, np.zeros_like(target), prediction_valid=unavailable)
        self.assertEqual(no_pred["position"]["coverage"], 0)
        self.assertEqual(no_pred["position"]["missing_prediction_count"], 2)
        self.assertIsNone(no_pred["position"]["mae"])
        for shape in ((0, 2, 4), (2, 0, 4)):
            empty = evaluate(np.empty(shape), np.empty(shape))
            self.assertIsNone(empty["speed"]["mse"])
            self.assertIsNone(empty["position"]["trajectory_coverage"])
            json.dumps(empty, allow_nan=False)
        empty_region = evaluate(np.zeros((1, 2, 4)), np.zeros((1, 2, 4)), region=np.zeros((1, 2), dtype=bool))
        self.assertIsNone(empty_region["heading"]["rmse"])

    def test_illegal_declared_valid_values_shapes_and_masks_are_rejected(self):
        baseline = np.array([[[0., 0., 1., 0.]]]); valid = np.ones_like(baseline, dtype=bool)
        for column, value in ((0, 91), (0, -91), (1, 181), (1, -181),
                              (2, -1), (0, np.nan), (2, np.inf), (3, np.nan)):
            bad = baseline.copy(); bad[0, 0, column] = value
            for side in ("prediction", "target"):
                with self.subTest(column=column, side=side), self.assertRaises(ValueError):
                    evaluate(bad if side == "prediction" else baseline,
                             bad if side == "target" else baseline,
                             region=np.zeros((1, 1), dtype=bool))
        for masks in ((valid.astype(int), valid, np.ones((1,1), bool)),
                      (valid, valid[..., :3], np.ones((1,1), bool)),
                      (valid, valid, np.ones((1,1), int))):
            with self.assertRaises(ValueError):
                evaluate(baseline, baseline, *masks[:2], region=masks[2])
        with self.assertRaises(ValueError):
            evaluate(baseline, np.zeros((2, 1, 4)))
        with self.assertRaises(ValueError):
            evaluate_aligned_fusion(np.array([[['0']*4]]), baseline, valid,
                                    prediction_valid=valid, region_mask=np.ones((1, 1), bool))

    def test_inputs_and_masks_are_not_modified(self):
        values = np.array([[[0., 180., 0., 360.], [np.nan]*4]])
        valid = np.isfinite(values); region = np.array([[True, False]])
        originals = [x.copy() for x in (values, valid, region)]
        result = evaluate(values, values, valid, valid, region)
        result["position"]["coverage"] = -1
        for value, original in zip((values, valid, region), originals):
            np.testing.assert_equal(value, original)


class FusionGroupMetricTests(unittest.TestCase):
    def test_split_and_merge_on_all_nodes_without_matching_label_names(self):
        result = group_membership_diagnostics([10, 20, 20, 20], ['A', 'A', 'B', 'B'])
        self.assertEqual(result['pair_count'], 6)
        self.assertEqual(result['correct_same_identity_pairs'], 1)
        self.assertEqual(result['split_pairs'], 1)
        self.assertEqual(result['merge_pairs'], 2)
        self.assertEqual(result['correct_different_identity_pairs'], 2)
        self.assertEqual(result['split_target_group_count'], 1)
        self.assertEqual(result['merged_predicted_group_count'], 1)
        good = group_membership_diagnostics(['x','x','y','y'], ['A','A','B','B'])
        self.assertEqual(good['split_pairs'], 0)
        self.assertEqual(good['merge_pairs'], 0)

    def test_unassigned_nodes_are_separate_and_stay_in_denominator(self):
        result = group_membership_diagnostics([None,None,3], ['A','A','B'])
        self.assertEqual(result['node_count'], 3)
        self.assertEqual(result['pair_count'], 3)
        self.assertEqual(result['unassigned_node_count'], 2)
        self.assertEqual(result['assigned_node_coverage'], 1/3)
        self.assertEqual(result['predicted_assigned_group_count'], 1)
        self.assertEqual(result['effective_partition_group_count'], 3)
        self.assertEqual(result['split_pairs'], 1)
        self.assertEqual(result['merge_pairs'], 0)

    def test_empty_singleton_and_typed_group_ids(self):
        empty = group_membership_diagnostics([], [])
        self.assertEqual(empty['pair_count'], 0)
        self.assertIsNone(empty['assigned_node_coverage'])
        one = group_membership_diagnostics([None], ['A'])
        self.assertEqual(one['target_group_count'], 1)
        self.assertEqual(one['split_pairs'], 0)
        self.assertEqual(one['assigned_node_coverage'], 0)
        typed = group_membership_diagnostics([1,'1'], ['A','A'])
        self.assertEqual(typed['split_pairs'], 1)
        json.dumps(typed, allow_nan=False)

    def test_invalid_or_unaligned_group_population_is_rejected(self):
        for predicted, truth in (([0], ['A','B']), ([[0]], ['A']),
                                  ([0], [None]), ([np.nan], ['A']),
                                  ([True], ['A']), ([0], [''])):
            with self.subTest(predicted=predicted, truth=truth), self.assertRaises(ValueError):
                group_membership_diagnostics(predicted, truth)


if __name__ == '__main__':
    unittest.main()
