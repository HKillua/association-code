"""The reverse witness only uses arrived coordinates on its fixed past clock."""
import unittest

from online_observation.rejoin_witness import score_reverse_witness


REPORTER = ("source-a", "track-1")
RELEASE = ("source-b", "track-2")
RETAIN = ("source-b", "track-3")
SAMPLE = dict(stamp=80.0, reporter=REPORTER, partners={"release": RELEASE, "retain": RETAIN},
              shift_delta=(0.0, 0.0))
HISTORY = {
    REPORTER: [(30.0, (30.0, 0.0)), (50.0, (50.0, 0.0))],
    RELEASE: [(20.0, (20.0, 0.0)), (40.0, (40.0, 0.0)), (60.0, (60.0, 0.0))],
    RETAIN: [(20.0, (20.0, 0.0)), (40.0, (40.0, 0.0)), (70.0, (70.0, 30.0))],
}


class RejoinWitnessTest(unittest.TestCase):
    def test_position_and_interval_use_the_same_strictly_earlier_query(self):
        result = score_reverse_witness(SAMPLE, HISTORY)
        self.assertEqual(result["query"], 50.0)
        self.assertEqual(result["ages_s"], {"reporter": 30.0, "release": 20.0, "retain": 10.0})
        self.assertEqual(result["endpoint_status"], "supported")
        self.assertEqual(result["interval_status"], "supported")
        endpoint, interval = result["endpoint"], result["interval"]
        self.assertEqual(endpoint["margin_m2"], 100.0)
        self.assertEqual(endpoint["provenance"], {REPORTER: (50.0,), RELEASE: (40.0, 60.0), RETAIN: (40.0, 70.0)})
        self.assertEqual(endpoint["late_partner_count"], 2)
        self.assertTrue(endpoint["both_partners_after_query"])
        self.assertEqual(interval["start"], 30.0)
        self.assertEqual(interval["earlier"]["margin_m2"], 0.0)
        self.assertEqual(interval["residual_change_m2"], {"release": 0.0, "retain": 100.0})
        self.assertEqual(interval["margin_increment_m2"], 100.0)
        self.assertEqual(interval["displacement_margin_m2"], 100.0)
        self.assertEqual(interval["available_at"], 70.0)

    def test_current_and_future_coordinates_have_no_effect(self):
        padded = {key: points + [(80.0, (10.0, 50.0)), (100.0, (50.0, 10.0))]
                  for key, points in HISTORY.items()}
        self.assertEqual(score_reverse_witness(SAMPLE, padded), score_reverse_witness(SAMPLE, HISTORY))
        missing = padded | {RELEASE: [(40.0, (40.0, 0.0)), (80.0, (10.0, 50.0))]}
        result = score_reverse_witness(SAMPLE, missing)
        self.assertEqual(result["endpoint_status"], "missing_window")
        self.assertIsNone(result["endpoint"])

    def test_exact_match_is_not_a_distinct_late_bracket(self):
        exact = HISTORY | {RELEASE: [(20.0, (20.0, 0.0)), (50.0, (50.0, 0.0)), (60.0, (60.0, 0.0))]}
        result = score_reverse_witness(SAMPLE, exact)
        self.assertEqual(result["endpoint"]["late_partner_count"], 1)
        self.assertFalse(result["endpoint"]["both_partners_after_query"])
        self.assertEqual(result["endpoint"]["provenance"][RELEASE], (50.0,))

    def test_preserves_endpoint_when_history_is_insufficient_for_interval(self):
        short = HISTORY | {REPORTER: [(50.0, (50.0, 0.0))]}
        result = score_reverse_witness(SAMPLE, short)
        self.assertEqual(result["endpoint_status"], "supported")
        self.assertEqual(result["interval_status"], "missing_window")
        self.assertIsNone(result["interval"])
        wide = HISTORY | {RETAIN: [(10.0, (10.0, 0.0)), (60.0, (60.0, 0.0))]}
        result = score_reverse_witness(SAMPLE, wide)
        self.assertEqual(result["endpoint_status"], "gap_too_wide")
        self.assertIsNone(result["endpoint"])

    def test_shift_changes_position_residual_but_cancels_in_displacement(self):
        shifted = score_reverse_witness(SAMPLE | {"shift_delta": (1.0, 2.0)}, HISTORY)
        self.assertEqual(shifted["endpoint"]["margin_m2"], 60.0)
        self.assertEqual(shifted["interval"]["displacement_margin_m2"], 100.0)


if __name__ == "__main__":
    unittest.main()
