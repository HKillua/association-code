"""Native-only motion controls preserve short tracks, masks and data isolation."""
import unittest

import numpy as np

from association.evaluation.native_motion import METHODS, predict_native_positions, position_metrics


class NativeMotionTests(unittest.TestCase):
    def test_constant_velocity_and_no_query_support_short_fragment(self):
        times = np.array([[0., 7., 20., 0.], [11., 12., 0., 0.]])
        visible = np.array([[True, True, True, False], [True, True, False, False]])
        states = np.zeros((2, 4, 4))
        states[..., 0] = 20.+times*.001
        states[..., 1] = 120.+times*.002
        query = np.array([-10., 0., 10., 20., 30., 40.])
        result = predict_native_positions(states, times, visible, query)
        expected = np.column_stack([20.+query*.001, 120.+query*.002])
        for name in METHODS[1:]:
            np.testing.assert_allclose(result['predictions'][name], np.broadcast_to(expected, (2, 6, 2)), atol=1e-12)
        self.assertFalse(result['regions']['inside_support'][1].any())
        self.assertEqual(result['native_point_counts'].tolist(), [3, 2])
        for name in METHODS[1:]:
            np.testing.assert_array_equal(result['predictions'][name][result['regions']['inside_support']],
                                          result['predictions']['native_hold'][result['regions']['inside_support']])
        self.assertGreater(abs(result['predictions']['native_ols'][1, 0, 0]
                               -result['predictions']['native_hold'][1, 0, 0]), .01)
        np.testing.assert_array_equal(result['regions']['all'],
            result['regions']['inside_support'] | result['regions']['before_support'] | result['regions']['after_support'])

    def test_dateline_slopes_exact_endpoints_and_latitude_clamp_are_reported(self):
        times = np.array([[0., 10., 20.], [0., 10., 20.]])
        positions = np.array([[[10., 179.], [10., 180.], [10., -179.]],
                              [[88., 0.], [89., 0.], [90., 0.]]])
        query = np.array([-10., 0., 5., 10., 20., 30.])
        result = predict_native_positions(positions, times, np.ones((2, 3), bool), query)
        for name in METHODS[1:]:
            np.testing.assert_allclose(result['predictions'][name][0, :, 1], [178., 179., 179.5, 180., -179., -178.])
            self.assertEqual(result['latitude_clamp_counts'][name].tolist(), [0, 1])
            self.assertEqual(result['predictions'][name][1, -1, 0], 90.)
        self.assertEqual(result['predictions']['native_hold'][0, 3, 1], 180.)
        # Exact half-turn direction follows the raw difference in both signs.
        half = predict_native_positions([[[0., 0.], [0., 180.]], [[0., 0.], [0., -180.]]],
            [[0., 10.], [0., 10.]], np.ones((2, 2), bool), [0., 5., 10.])
        np.testing.assert_array_equal(half['predictions']['native_hold'][:, 1, 1], [90., -90.])
        np.testing.assert_array_equal(half['native_half_turn_counts'], [1, 1])

    def test_hidden_padding_and_nonposition_states_do_not_change_predictions(self):
        values = np.array([[[20., 120., 5., 45.], [20.1, 120.1, 5., 45.], [0., 0., 0., 0.]]])
        times, visible = np.array([[0., 10., 0.]]), np.array([[True, True, False]])
        first = predict_native_positions(values, times, visible, [-5., 0., 5., 10., 15.])
        changed = values.copy()
        changed[..., 2:] = np.nan
        changed[:, 2] = np.nan
        times[:, 2] = np.nan
        second = predict_native_positions(changed, times, visible, [-5., 0., 5., 10., 15.])
        for name in METHODS:
            np.testing.assert_array_equal(first['predictions'][name], second['predictions'][name])
        target = np.zeros((1, 5, 4))
        target[..., :2] = first['predictions']['native_ols']
        valid = np.zeros_like(target, bool)
        valid[..., :2] = True
        score = position_metrics(first['predictions']['native_ols'], target, valid, first['regions']['all'])
        self.assertEqual(score['mae'], 0.)
        valid[..., 1] = False
        missing = position_metrics(first['predictions']['native_ols'], target, valid, first['regions']['all'])
        self.assertEqual(missing['intersection_count'], 0)
        self.assertIsNone(missing['mae'])


if __name__ == '__main__':
    unittest.main()
