import unittest

import numpy as np

from tools.analysis import conditional_trajectory_pilot as pilot


class ConditionalTrajectoryTest(unittest.TestCase):
    def test_swap_gain_matches_direct_weighted_holdout_rows(self):
        ts = np.tile(np.linspace(0., 1., 101), 4)
        offsets = np.arange(5)*101
        basis, _ = pilot.design(ts)
        xy = np.column_stack((np.repeat([0., 100., 100., 0.], 101)+ts, 2.*ts))
        base = np.array([0, 1, 0, 1])
        action = np.array([[2, 3]])
        comparable, score, no_test = pilot.holdout_scores(ts, basis, xy, offsets, base, action)
        self.assertTrue(comparable[0])
        self.assertGreater(score[0], 1000.)
        self.assertEqual(no_test, 0)

        def direct(membership):
            loss = 0.
            for group in (0, 1):
                member = np.repeat(membership == group, 101)
                for lo, hi in pilot.FOLDS:
                    test = (ts >= lo) & (ts < hi) & member
                    train = member & ~test
                    x = np.vstack((basis[train]/np.sqrt(101.), np.linalg.cholesky(pilot.RIDGE).T))
                    y = np.vstack((xy[train]/np.sqrt(101.), np.zeros((5, 2))))
                    fit = np.linalg.lstsq(x, y, rcond=None)[0]
                    loss += np.sum((xy[test]-basis[test] @ fit)**2)/101.
            return loss

        expected = direct(base)-direct(np.array([0, 1, 1, 0]))
        np.testing.assert_allclose(score[0], expected, atol=1e-9)

    def test_heldout_events_without_two_sides_are_ineligible(self):
        ts = np.array([.02, .05, .80, .90, .22, .25, .11, .12])
        basis, _ = pilot.design(ts)
        eligible, gain, no_test = pilot.holdout_scores(
            ts, basis, np.zeros((8, 2)), np.arange(5)*2, np.array([0, 1, 0, 1]), np.array([[2, 3]]))
        self.assertFalse(eligible[0])
        self.assertEqual(gain[0], 0.)
        self.assertEqual(no_test, 0)

    def test_state_readout_uses_mps_slope_and_falls_back_outside_support(self):
        q = np.linspace(0., 60., 64)
        times = np.tile(np.linspace(6., 45., 11), 2)
        xy = np.column_stack((2.*times/pilot.SCALE, np.zeros(len(times))))
        native = (q, times, xy, np.array([0, 11, 22]))
        stats, _, _, _, origin, scale = pilot.scene_arrays(native)
        fallback = np.tile([1., 2., 4., 60.], (1, 64, 1))
        pred, coverage = pilot.state_output(stats, native, origin, scale, np.array([0, 0]), fallback)
        np.testing.assert_array_equal(pred[0, [0, -1]], fallback[0, [0, -1]])
        np.testing.assert_allclose(pred[0, 31, 2:], [2., 0.], atol=1e-5)
        self.assertEqual(coverage['inside']+coverage['outside'], 64)


if __name__ == '__main__':
    unittest.main()
