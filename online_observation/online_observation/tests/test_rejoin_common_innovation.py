"""Verify cancellation, external attribution and balanced group support."""
import unittest

from online_observation.rejoin_common_innovation import compare_common_innovations


def geometry(innovation, axis):
    return dict(actual=innovation, masked=(0.0, 0.0), release=axis, retain=(0.0, 0.0))


class RejoinCommonInnovationTest(unittest.TestCase):
    def test_asynchronous_axis_drift_separates_common_and_differential_terms(self):
        rows = dict(release=geometry((4.0, 3.0), (5.0, 1.0)), retain=geometry((2.0, 1.0), (-1.0, 3.0)))
        output = compare_common_innovations(rows, dict(release=None, retain=None))
        decomposition = output["decomposition"]
        self.assertEqual(decomposition["mean_innovation_m"], (3.0, 2.0))
        self.assertEqual(decomposition["differential_innovation_m"], (1.0, 1.0))
        self.assertEqual(decomposition["mean_axis_m"], (2.0, 2.0))
        self.assertEqual(decomposition["common_increment_m2"], 20.0)
        self.assertEqual(decomposition["differential_increment_m2"], 4.0)
        self.assertEqual(output["raw"]["increment_m2"], 24.0)
        self.assertEqual(output["demeaned"]["increment_m2"], 4.0)
        self.assertEqual(output["reports"]["retain"]["demeaned_residual_m2"], 2.0)
        self.assertEqual(output["demeaned"]["masked_m2"], output["raw"]["masked_m2"])

    def test_a_constant_correction_cancels_for_opposite_axes(self):
        rows = dict(release=geometry((4.0, 3.0), (5.0, 1.0)), retain=geometry((2.0, 1.0), (-5.0, -1.0)))
        output = compare_common_innovations(rows, dict(release=(9.0, 2.0), retain=(5.0, 4.0)))
        self.assertEqual(output["decomposition"]["common_increment_m2"], 0.0)
        self.assertEqual(output["donor_dot_m2"], 53.0)
        self.assertEqual(output["shared_external"]["actual_m2"], output["raw"]["actual_m2"])
        self.assertEqual(output["demeaned"]["increment_m2"], output["raw"]["increment_m2"])
        self.assertNotEqual(output["raw"]["reports"]["release"]["actual_m2"],
                            output["shared_external"]["reports"]["release"]["actual_m2"])

    def test_donor_direction_is_compared_with_equal_magnitude_shrink(self):
        rows = dict(release=geometry((3.0, 4.0), (10.0, 0.0)), retain=geometry((0.0, 0.0), (-10.0, 0.0)))
        output = compare_common_innovations(rows, dict(release=(0.0, 10.0), retain=(4.0, 0.0)))
        report = output["reports"]["release"]
        self.assertEqual(report["donor_applied_m"], (0.0, 5.0))
        self.assertEqual(report["innovation_squared_m2"], 25.0)
        self.assertEqual(report["external_raw_residual_m2"], 45.0)
        self.assertEqual(report["external_raw_reduction_m2"], -20.0)
        self.assertEqual(report["external_applied_residual_m2"], 10.0)
        self.assertEqual(report["external_applied_reduction_m2"], 15.0)
        self.assertEqual(report["shrink_residual_m2"], 0.0)
        self.assertEqual(output["external"]["increment_m2"], 30.0)
        self.assertEqual(output["equal_shrink"]["increment_m2"], 0.0)
        self.assertEqual(output["reports"]["retain"]["donor_applied_m"], (0.0, 0.0))

    def test_missing_donor_keeps_report_level_support_without_one_sided_group_action(self):
        rows = dict(release=geometry((2.0, 1.0), (3.0, 0.0)), retain=geometry((1.0, 2.0), (-3.0, 0.0)))
        output = compare_common_innovations(rows, dict(release=(1.0, 1.0), retain=None))
        self.assertEqual(output["raw"]["support_count"], 2)
        self.assertEqual(output["external"]["support_count"], 1)
        self.assertIsNone(output["external"]["actual_m2"])
        self.assertIsNone(output["external"]["reports"]["retain"])
        self.assertEqual(output["shared_external"]["support_count"], 0)
        self.assertIsNone(output["donor_dot_m2"])
        single = compare_common_innovations(rows | {"retain": None}, dict(release=(1.0, 1.0), retain=None))
        self.assertIsNone(single["decomposition"])
        self.assertIsNone(single["raw"]["increment_m2"])


if __name__ == "__main__":
    unittest.main()
