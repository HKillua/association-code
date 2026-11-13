"""Check causal history membership and the same-clock scoring controls."""
import unittest

from online_observation.hypothesis_gap_diagnostic import Switch
from online_observation.rejoin_async_innovation import score_report_innovations
from online_observation.rejoin_track_history import score_track_history


A, C = ("a", "1"), ("a", "2")
B, D = ("b", "1"), ("b", "2")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))
FIT = dict(translation_east_north_m=(10.0, 1.0), source_order=("a", "b"))


def reports(offset, pulse=0.0, shift=0.0):
    return [(600.0 + index * 10.0 + shift, (offset + (0.0, 0.0, 1.0, 3.0, 7.0 + pulse, 15.0)[index], 0.0))
            for index in range(6)]


HISTORIES = {A: reports(100.0), C: reports(500.0), B: reports(0.0), D: reports(800.0, shift=1.0)}
NOVEL = dict(release=(B, 640.0), retain=(D, 641.0))


class RejoinTrackHistoryTest(unittest.TestCase):
    def test_raw_scores_match_c51_and_preceding_excludes_same_time(self):
        output = score_track_history(SWITCH, HISTORIES, NOVEL, 625.0, FIT)
        c51 = score_report_innovations(SWITCH, HISTORIES, NOVEL, 625.0, 645.0, FIT)
        for side, report in output["reports"].items():
            self.assertEqual(report["rolling"], c51["reports"][side]["rolling"])
        report = output["reports"]["release"]
        previous, last = report["preceding"]
        self.assertEqual((previous["stamp"], last["stamp"]), (620.0, 630.0))
        self.assertEqual((previous["after_onset"], last["after_onset"]), (False, True))
        self.assertEqual(previous["prior_stamps"], dict(own=610.0, release=610.0, retain=610.0))
        self.assertEqual(last["prior_stamps"], dict(own=620.0, release=620.0, retain=620.0))
        self.assertEqual((previous["status"], last["status"]), ("supported", "supported"))

    def test_own_correction_preserves_actual_and_has_exact_projection_decomposition(self):
        output = score_track_history(SWITCH, HISTORIES, NOVEL, 625.0, FIT)
        report = output["reports"]["release"]
        history = report["history"]["own"]
        self.assertEqual(history["status"], "qualified")
        self.assertEqual(history["innovation_stamps"], (620.0, 630.0))
        self.assertEqual(history["prior_stamps"], (610.0, 620.0))
        self.assertLess(history["holdout_loss_m2"], history["holdout_baseline_m2"])
        self.assertEqual(report["own_corrected"]["actual_m2"], report["rolling"]["actual_m2"])
        terms = report["decomposition"]
        self.assertAlmostEqual(terms["persistent_increment_m2"] + terms["remainder_increment_m2"],
                               report["rolling"]["increment_m2"])
        self.assertAlmostEqual(terms["remainder_increment_m2"], report["own_corrected"]["increment_m2"])
        self.assertLess(report["own_corrected_loss_m2"], report["own_loss_m2"])
        changed = HISTORIES | {B: reports(0.0, pulse=800.0)}
        other = score_track_history(SWITCH, changed, NOVEL, 625.0, FIT)["reports"]["release"]
        self.assertEqual(other["history"], report["history"])
        self.assertEqual(other["preceding"], report["preceding"])
        self.assertNotEqual(other["rolling"]["actual_m2"], report["rolling"]["actual_m2"])

    def test_history_uses_consecutive_immediate_reports_and_preserves_abstentions(self):
        changed = HISTORIES | {C: [(580.0, (0.0, 0.0)), (600.0, (0.0, 0.0)), (620.0, (0.0, 0.0)),
                                  (630.0, (0.0, 0.0))], B: HISTORIES[B][1:]}
        output = score_track_history(SWITCH, changed, NOVEL | {"retain": None}, 625.0, FIT)
        report = output["reports"]["release"]
        self.assertEqual(report["history"]["own"]["status"], "immature_history")
        self.assertEqual(report["history"]["retain"]["status"], "no_holdout_gain")
        self.assertEqual(tuple(row["stamp"] for row in report["preceding"]), (620.0, 630.0))
        self.assertEqual(tuple(row["status"] for row in report["preceding"]), ("immature_prior", "supported"))
        self.assertEqual(report["opponents_corrected"]["status"], "unqualified_history")
        self.assertEqual(output["rolling"]["support_count"], 1)
        self.assertIsNone(output["rolling"]["actual_m2"])
        self.assertEqual(output["opponents_corrected"]["support_count"], 0)

    def test_stale_innovation_gap_does_not_select_an_earlier_pair(self):
        changed = HISTORIES | {B: [(-20.0, (0.0, 0.0)), (0.0, (0.0, 0.0)), *HISTORIES[B][2:]]}
        output = score_track_history(SWITCH, changed, NOVEL, 625.0, FIT)
        report = output["reports"]["release"]
        self.assertEqual(report["history"]["own"]["status"], "stale_history")
        self.assertEqual(report["history"]["own"]["innovation_stamps"], (620.0, 630.0))
        self.assertEqual(report["preceding"][0]["status"], "stale_prior")
        self.assertIsNone(report["decomposition"])


if __name__ == "__main__":
    unittest.main()
