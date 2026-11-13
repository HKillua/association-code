"""Native report-time priors and opposite innovation controls on a fixed switch."""
import unittest

from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.rejoin_async_innovation import score_report_innovations


A, C = ("left", "1"), ("left", "2")
B, D = ("right", "1"), ("right", "2")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))
UNREGISTERED = dict(translation_east_north_m=None)
FIT = dict(translation_east_north_m=(2.0, 0.0), source_order=("right", "left"))
HISTORIES = {
    A: [(590.0, (0.0, 0.0)), (601.0, (20.0, 0.0)), (610.0, (90.0, 0.0))],
    C: [(590.0, (10.0, 0.0)), (599.0, (999.0, 0.0)), (610.0, (90.0, 0.0))],
    B: [(590.0, (10.0, 0.0)), (599.0, (1.0, 0.0))],
    D: [(590.0, (0.0, 0.0)), (610.0, (9.0, 0.0))],
}
NOVEL = dict(release=(B, 599.0), retain=(D, 610.0))


class RejoinAsyncInnovationTest(unittest.TestCase):
    def test_report_uses_its_own_clock_and_opposite_roles_reverse(self):
        output = score_report_innovations(SWITCH, HISTORIES, NOVEL, 590.0, 620.0, UNREGISTERED)
        first, second = (output["reports"][side] for side in ("release", "retain"))
        self.assertEqual(first["partners"], {"release": A, "retain": C})
        self.assertEqual(second["partners"], {"release": C, "retain": A})
        self.assertEqual(first["rolling"]["prior_stamps"], dict(own=590.0, release=590.0, retain=590.0))
        self.assertEqual(second["rolling"]["prior_stamps"], dict(own=590.0, release=599.0, retain=601.0))
        self.assertEqual(second["onset_origin"]["prior_stamps"], dict(own=590.0, release=590.0, retain=590.0))
        self.assertEqual(second["refreshed_roles"], ("release", "retain"))
        self.assertEqual(first["age_to_stop_s"], 21.0)
        self.assertEqual(first["rolling"]["actual_m2"], 80.0)
        self.assertEqual(first["rolling"]["masked_m2"], -100.0)
        self.assertEqual(first["rolling"]["increment_m2"], 180.0)
        self.assertEqual(second["onset_origin"]["actual_m2"], 80.0)
        self.assertEqual(second["onset_origin"]["masked_m2"], -100.0)
        self.assertEqual(output["onset_origin"]["actual_m2"], 80.0)
        self.assertEqual(output["onset_origin"]["increment_m2"], 180.0)

    def test_same_clock_mask_and_source_registration_leave_increment_unchanged(self):
        output = score_report_innovations(SWITCH, HISTORIES, NOVEL, 590.0, 620.0, FIT)
        first, second = (output["reports"][side] for side in ("release", "retain"))
        self.assertEqual(first["shift_m"], (0.0, 0.0))
        self.assertEqual(second["shift_m"], (2.0, 0.0))
        origin = second["onset_origin"]
        self.assertEqual(origin["actual_m2"], 120.0)
        self.assertEqual(origin["masked_m2"], -60.0)
        self.assertEqual(origin["increment_m2"], origin["unregistered_increment_m2"])
        self.assertEqual(origin["retain_change_m2"] - origin["release_change_m2"], origin["increment_m2"])
        self.assertEqual(origin["common_variance_m2"], origin["variance_m2"]["release"])
        self.assertAlmostEqual(origin["common_increment"], origin["increment_m2"] / origin["common_variance_m2"])

    def test_missing_and_stale_are_separate_and_do_not_turn_into_a_one_report_mean(self):
        stale = HISTORIES | {A: [(530.0, (0.0, 0.0)), (601.0, (20.0, 0.0))]}
        result = score_report_innovations(SWITCH, stale, NOVEL, 590.0, 620.0, UNREGISTERED)
        self.assertEqual(result["reports"]["release"]["rolling"]["status"], "stale_prior")
        self.assertEqual(result["rolling"]["support_count"], 1)
        self.assertIsNone(result["rolling"]["actual_m2"])
        self.assertEqual(result["onset_origin"]["support_count"], 0)

        missing = HISTORIES | {B: [(599.0, (1.0, 0.0))]}
        result = score_report_innovations(SWITCH, missing, NOVEL | {"retain": None}, 590.0, 620.0, UNREGISTERED)
        self.assertEqual(result["reports"]["release"]["rolling"]["status"], "missing_prior")
        self.assertIsNone(result["reports"]["retain"])
        self.assertEqual(result["rolling"]["support_count"], 0)

    def test_rejects_wrong_selected_timestamp(self):
        for stamp, message in ((590.0, "strictly precede"), (620.0, "strictly precede"), (599.5, "native clock")):
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, message):
                score_report_innovations(SWITCH, HISTORIES, NOVEL | {"release": (B, stamp)},
                                         590.0, 620.0, UNREGISTERED)


if __name__ == "__main__":
    unittest.main()
