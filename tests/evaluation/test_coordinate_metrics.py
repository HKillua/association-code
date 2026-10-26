"""Small arithmetic/cache checks only; these are not real MTAD results."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from association.evaluation.fusion_controls import attribute_errors
from tools.analysis.coordinate_metrics import (
    METHODS, REGIONS, aggregate, coordinate_scene, evaluate_scene, pool_coordinates, report, paper_tables,
)


def coordinate_errors(errors):
    errors = np.asarray(errors, dtype=float)
    truth = np.full_like(errors, 30.)
    prediction = truth-errors/10000.
    valid = np.ones_like(errors, dtype=bool)
    return coordinate_scene(truth, prediction, valid, valid, valid)


class CoordinateMetricsTests(unittest.TestCase):
    def test_signed_cancellation_does_not_become_zero_absolute_error(self):
        row, values = coordinate_errors([[2., -2.]])
        self.assertAlmostEqual(row['target_mean']['ME'], 0.)
        self.assertAlmostEqual(row['target_mean']['MAE'], 2.)
        self.assertAlmostEqual(row['target_mean']['absolute_ME'], 0.)
        self.assertAlmostEqual(row['target_mean']['residual_SD'], np.sqrt(8.))
        self.assertAlmostEqual(row['point_equal']['RMSE'], 2.)
        # The printed Eq29 diagnostic is truth-based, not the residual SD.
        self.assertAlmostEqual(row['target_mean']['printed_Eq29_diagnostic'], 300000.*np.sqrt(2.))

    def test_equal_scenes_after_equal_targets_not_pooled_targets(self):
        a, av = coordinate_errors([[2., 2.], [4., 4.]])
        b, bv = coordinate_errors([[12., 12.]])
        combined = pool_coordinates([a, b], [av, bv])
        self.assertAlmostEqual(combined['scene_equal_target_mean']['ME'], 7.5)
        self.assertAlmostEqual(combined['point_equal']['ME'], 6.)
        self.assertEqual(combined['declared_targets'], 3)
        self.assertEqual(combined['defined_scenes']['ME'], 2)
        self.assertAlmostEqual(combined['strict_complete_population']['ME'], 7.5)

    def test_sample_sd_is_averaged_per_target_not_pooled_or_rms(self):
        row, values = coordinate_errors([[-1., 1.], [-3., 3.]])
        expected = 2.*np.sqrt(2.)
        self.assertAlmostEqual(row['target_mean']['residual_SD'], expected)
        self.assertNotAlmostEqual(row['target_mean']['residual_SD'], np.std(values, ddof=1))
        self.assertNotAlmostEqual(row['target_mean']['residual_SD'], np.sqrt(10.))

    def test_missing_predictions_truth_and_singleton_sd_remain_undefined(self):
        truth = np.array([[1., 2.], [3., 4.], [np.nan, np.nan]])
        prediction = np.array([[.9999, np.nan], [np.nan, np.nan], [5., 6.]])
        tv, pv = np.isfinite(truth), np.isfinite(prediction)
        row, values = coordinate_scene(truth, prediction, tv, pv, np.ones_like(tv))
        self.assertEqual(row['available_target_points'], 4)
        self.assertEqual(row['evaluated_points'], 1)
        self.assertEqual(row['missing_prediction_points'], 3)
        self.assertEqual(row['targets_with_reference'], 2)
        self.assertEqual(row['targets_with_evaluation'], 1)
        self.assertAlmostEqual(row['target_mean']['ME'], 1.)
        self.assertIsNone(row['target_mean']['residual_SD'])
        self.assertIsNone(row['strict_complete_target_mean']['ME'])
        empty, ev = coordinate_scene(truth, prediction, tv, pv, np.zeros_like(tv))
        pooled = pool_coordinates([row, empty], [values, ev])
        self.assertEqual(pooled['defined_scenes']['ME'], 1)
        self.assertEqual(pooled['defined_scenes']['residual_SD'], 0)
        self.assertIsNone(pooled['strict_complete_population']['ME'])
        self.assertEqual(pooled['coverage'], .25)

    def test_longitude_raw_and_shortest_arc_are_separate_units(self):
        truth, prediction = np.array([[179.5]]), np.array([[-179.5]])
        valid = np.ones_like(truth, dtype=bool)
        raw, _ = coordinate_scene(truth, prediction, valid, valid, valid)
        short, _ = coordinate_scene(truth, prediction, valid, valid, valid, wrap=True)
        self.assertEqual(raw['target_mean']['ME'], 3590000.)
        self.assertEqual(short['target_mean']['ME'], -10000.)
        self.assertEqual(raw['raw_difference_over_180_count'], 1)
        self.assertIsNone(short['target_mean']['printed_Eq29_diagnostic'])

    def test_cached_scene_keeps_missing_and_duplicate_rows_and_coordinate_order(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'system-predictions').mkdir(); (root/'predictions').mkdir()
            truth = np.full((4, 2, 4), np.nan)
            truth[0] = [30., 120., 10., 90.]
            truth[1] = [32., 122., 12., 91.]
            prediction = truth.copy()
            prediction[0, :, :2] -= [1e-4, 2e-4]
            prediction[1] = np.nan
            prediction[2] = [31., 121., 9., 92.]
            prediction[3] = [10., 110., 11., 89.]  # Duplicate A must not gain truth.
            artifact = dict(aligned_truth=truth, aligned_prediction=prediction,
                aligned_truth_valid=np.isfinite(truth), aligned_prediction_valid=np.isfinite(prediction),
                target_ids=np.array(['A', 'B', 'C']), aligned_target_ids=np.array(['A', 'B', 'C', '']),
                aligned_group_indices=np.array([0, -1, 1, 2]), group_anchor_target_indices=np.array([0, 2, 0]))
            np.savez_compressed(root/'system-predictions/states-a.npz', **{
                method+'_'+key: value for method in METHODS for key, value in artifact.items()})
            common = np.isfinite(truth[:3, :, :2]).all(-1) & np.isfinite(prediction[:3, :, :2]).all(-1)
            distances = np.where(common, attribute_errors(prediction[:3], truth[:3])['position'], np.nan)
            masks = {name: np.zeros((3, 2), bool) for name in REGIONS}
            masks['all'][:] = True; masks['anchor_inside_support'][0] = True
            masks['no_predicted_group'][1] = True; masks['predicted_group_no_support'][2] = True
            np.savez_compressed(root/'predictions/states-a.npz', **{
                **{'region_'+name: mask for name, mask in masks.items()},
                **{method+'_position_error': distances for method in METHODS}})
            row, vectors = evaluate_scene(root, 'a')
            self.assertEqual(row['population'], dict(targets=3, groups=3, duplicate_groups=1,
                available_position_points=4, evaluated_position_points=2, missing_prediction_points=2,
                reference_unavailable_targets=1, covered_targets=1))
            result = aggregate([row], [vectors])
            ours = result['methods']['learned_sequence']['all']
            self.assertAlmostEqual(ours['latitude_raw']['scene_equal_target_mean']['ME'], 1.)
            self.assertAlmostEqual(ours['longitude_raw']['scene_equal_target_mean']['ME'], 2.)
            self.assertAlmostEqual(ours['position_m']['MAE'], float(distances[common].mean()))
            self.assertEqual(ours['longitude_raw']['declared_targets'], 3)
            self.assertIsNone(ours['longitude_raw']['strict_complete_population']['ME'])
            self.assertEqual(ours['position_m']['count'], 2)
            text = report(result)
            self.assertIn('6.24 ± 0.13', text)
            self.assertIn('10⁻⁴度', text)
            self.assertIn('零训练/前向', text)
            # Same saved arrays, independently declared speed units. Only the
            # report changes; input/position predictions are never rewritten.
            new_row, new_vectors = evaluate_scene(root, 'a', paper_attributes=True,
                prediction_speed_unit='kn', target_speed_unit='mps')
            new_result = aggregate([new_row], [new_vectors])
            attrs = new_result['paper_fusion_metrics']['learned_sequence']['attributes']
            self.assertEqual(attrs['speed']['unit'], 'kn')
            self.assertAlmostEqual(attrs['speed']['scene_equal_target_mean']['ME'], 10*3600/1852-10)
            self.assertAlmostEqual(attrs['longitude']['scene_equal_target_mean']['ME'], 2.)
            self.assertEqual(new_result['methods'], result['methods'])
            new_result['association'] = {'variants': {
                kind: {'values': dict(m_pca=90., mw_pca=89., m_pfa=10., mw_pfa=11.)}
                for kind in ('itsa','mstsa')}}
            table = paper_tables(new_result)
            self.assertIn('mPCA %', table)
            self.assertIn('speed | kn', table)
            self.assertIn('heading | degree', table)
            self.assertIn('作者ESD实现未核实', table)
            self.assertIn('Mean速度/航向', table)
            with np.load(root/'system-predictions/states-a.npz') as unchanged:
                np.testing.assert_array_equal(unchanged['learned_sequence_aligned_prediction'], prediction)


if __name__ == '__main__':
    unittest.main()
