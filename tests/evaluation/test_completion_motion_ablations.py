"""Fixed native motion ablations isolate boundary anchoring and local slopes."""
import unittest

import numpy as np

from tools.analysis.completion_motion_ablations import METHODS, predict_ablations


class CompletionMotionAblationTests(unittest.TestCase):
    def test_unequal_time_linear_dateline_and_two_point_baseline_are_preserved(self):
        times = np.array([[0., 3., 8., 12., 20., 30.], [11., 12., 0., 0., 0., 0.]])
        mask = np.array([[True]*6, [True, True, False, False, False, False]])
        values = np.zeros((2, 6, 4))
        values[..., 0] = 20.+.001*times
        values[..., 1] = (179.9+.01*times+180.) % 360.-180.
        values[1, 2:] = np.nan
        values[..., 2:] = np.nan  # Speed/heading never enter prediction.
        times[1, 2:] = np.nan
        query = np.array([-10., 0., 3., 10., 20., 30., 40.])
        result = predict_ablations(values, times, mask, query)
        expected = np.column_stack((20.+.001*query, (179.9+.01*query+180.) % 360.-180.))
        for name in METHODS:
            np.testing.assert_allclose(result['predictions'][name],
                                       np.broadcast_to(expected, (2, 7, 2)), atol=1e-12)
        for name in METHODS[1:]:
            np.testing.assert_array_equal(result['predictions'][name][1], result['predictions']['native_ols'][1])
            inside = result['regions']['inside_support']
            np.testing.assert_array_equal(result['predictions'][name][inside],
                                          result['predictions']['native_ols'][inside])
        self.assertEqual(result['local_fit_point_counts'].tolist(), [3, 2])
        np.testing.assert_allclose(result['fitted_boundary_offsets_metres'], 0., atol=1e-8)

    def test_nonlinear_arms_match_independent_lstsq_and_keep_inside_unchanged(self):
        times = np.array([[0., 2., 7., 13., 25., 40., 70.]])
        positions = np.array([[[20., 120.], [20.01, 120.005], [20.02, 120.01],
                               [20.03, 120.02], [20.06, 120.09], [20.13, 120.2], [20.4, 120.5]]])
        query = np.array([-20., 0., 5., 25., 70., 90.])
        result = predict_ablations(positions, times, np.ones((1, 7), bool), query)
        all_fit = np.linalg.lstsq(np.column_stack((times[0], np.ones(7))), positions[0], rcond=None)[0]
        expected_fit = np.column_stack((query[[0, -1]], np.ones(2))) @ all_fit
        np.testing.assert_allclose(result['predictions']['ols_fit_anchor'][0, [0, -1]],
                                   expected_fit, atol=1e-12)
        for side, take, index, dt in ((0, slice(0, 4), 0, -20.), (1, slice(-4, None), -1, 20.)):
            t, xy = times[0, take], positions[0, take]
            fitted = np.linalg.lstsq(np.column_stack((t, np.ones(4))), xy, rcond=None)[0]
            expected = positions[0, index]+dt*fitted[0]
            np.testing.assert_allclose(result['predictions']['ols_local_half'][0, index], expected, atol=1e-12)
        inside = result['regions']['inside_support']
        for name in METHODS[1:]:
            np.testing.assert_array_equal(result['predictions'][name][inside],
                                          result['predictions']['native_ols'][inside])
            self.assertFalse(np.allclose(result['predictions'][name], result['predictions']['native_ols']))
        self.assertEqual(result['local_fit_point_counts'].tolist(), [4])
        self.assertTrue((result['fitted_boundary_offsets_metres'] > 0).all())


if __name__ == '__main__':
    unittest.main()
