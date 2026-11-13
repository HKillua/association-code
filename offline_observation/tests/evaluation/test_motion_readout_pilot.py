import unittest

import numpy as np

from tools.analysis.conditional_trajectory_pilot import SCALE
from tools.analysis.motion_readout_pilot import readouts


class MotionReadoutTest(unittest.TestCase):
    def test_equal_group_mass_recovers_distinct_slopes_without_moving_positions(self):
        q = np.linspace(0., 63., 64)
        times = np.tile(np.arange(5., 58.), 2)
        native_positions = np.column_stack((3. * times / SCALE, np.zeros(len(times))))
        native = q, times, native_positions, np.array([0, 53, 106])
        membership = np.array([0, 0])
        baseline = np.zeros((1, 64, 4))
        baseline[0, :, 0] = 2. * q / SCALE
        baseline[0, :, 2:] = [10., 123.]
        support = np.tile((q >= 5.) & (q <= 57.), (2, 1))
        support[:, 31] = False

        arms, masks = readouts(native, membership, baseline, support)
        for pred in arms.values():
            np.testing.assert_array_equal(pred[..., :2], baseline[..., :2])
            np.testing.assert_array_equal(pred[:, [0, 63]], baseline[:, [0, 63]])
        np.testing.assert_allclose(arms['B'][0, 31, 2:], [2., 0.], atol=1e-6)
        np.testing.assert_allclose(arms['C'][0, 31, 2:], [3., 0.], atol=1e-6)
        self.assertTrue(masks['internal_gap'][0, 31])
        self.assertTrue(masks['near_endpoint'][0, 5])

    def test_one_supported_query_keeps_a_in_both_arms(self):
        q = np.linspace(0., 63., 64)
        native = q, np.array([3., 50.]), np.zeros((2, 2)), np.array([0, 2])
        baseline = np.tile([0., 0., 6., 355.], (1, 64, 1))
        support = np.zeros((1, 64), dtype=bool)
        support[0, 10] = True
        arms, masks = readouts(native, np.array([0]), baseline, support)
        self.assertFalse(masks['active'].any())
        np.testing.assert_array_equal(arms['B'], baseline)
        np.testing.assert_array_equal(arms['C'], baseline)


if __name__ == '__main__':
    unittest.main()
