import math
import unittest

from online_observation.continuous_tracker import POSITION_VARIANCE
from online_observation.rejoin_geometry_increment import score_increment


def _sample(release=(-1., 0.), retain=(3., 0.), innovation=(2., 1.), *, excess=(4., 9.)):
    def arrived(residual):
        return tuple(value + delta for value, delta in zip(residual, innovation, strict=True))

    return dict(pred_delta=dict(release=release, retain=retain),
                cross=dict(release=arrived(release), retain=arrived(retain)),
                self=innovation, self_variance=POSITION_VARIANCE + 16.,
                variance=dict(release=POSITION_VARIANCE + excess[0], retain=POSITION_VARIANCE + excess[1]))


class RejoinGeometryIncrementTests(unittest.TestCase):
    def test_projection_and_nominal_axes_have_different_scales(self):
        sample = _sample()
        score = score_increment(sample, dict(actual=116., masked=100.))
        self.assertEqual(score["increment_m2"], 16.)
        self.assertEqual(score["separation_m"], 4.)
        self.assertEqual(score["projected_m"], 2.)
        self.assertEqual(score["relative_increment"], .5)
        self.assertAlmostEqual(score["reporter_z"], 2 / math.sqrt(POSITION_VARIANCE + 16.))
        self.assertAlmostEqual(score["joint_z"], 8 / math.sqrt((POSITION_VARIANCE + 16.) * (16. + 26.)))
        self.assertEqual(score["release_change_m2"], 1.)
        self.assertEqual(score["retain_change_m2"], 17.)
        self.assertAlmostEqual(score["gaussian_increment"], 17 / (POSITION_VARIANCE + 9.)
                               - 1 / (POSITION_VARIANCE + 4.))

    def test_common_shift_preserves_projection_but_changes_residuals(self):
        baseline = score_increment(_sample(), dict(actual=116., masked=100.))
        shifted = score_increment(_sample(release=(5., 7.), retain=(9., 7.)), dict(actual=216., masked=200.))
        for name in ("increment_m2", "projected_m", "reporter_z", "joint_z"):
            self.assertEqual(shifted[name], baseline[name])
        self.assertNotEqual(shifted["release_change_m2"], baseline["release_change_m2"])

    def test_zero_separation_has_no_deterministic_projection(self):
        score = score_increment(_sample(retain=(-1., 0.)), dict(actual=0., masked=0.))
        self.assertIsNone(score["projected_m"])
        self.assertIsNone(score["reporter_z"])
        self.assertEqual(score["joint_z"], 0.)

    def test_inconsistent_frozen_geometry_fails(self):
        with self.assertRaisesRegex(ValueError, "frozen group geometry"):
            score_increment(_sample(), dict(actual=115., masked=100.))


if __name__ == "__main__":
    unittest.main()
