import unittest
from copy import deepcopy

from online_observation.continuous_tracker import FRESH_S, MEMORY_S, LocalTrack
from online_observation.disjoint_report_closure import score_pair
from online_observation.reciprocal_report_downdate import score_removal


A0, A1, B0, B1 = (("a", "0"), ("a", "1"), ("b", "0"), ("b", "1"))


def reports():
    older = dict(stamp=10.0, reporter=A0, partners={"release": B0, "retain": B1},
                 cross={"release": (3.0, 0.0), "retain": (10.0, 0.0)},
                 pred_delta={"release": (1.0, 0.0), "retain": (8.0, 0.0)},
                 age={"release": 2.0, "retain": 1.0}, shift_delta=(0.0, 0.0))
    newer = dict(stamp=12.0, reporter=B0, partners={"release": A0, "retain": A1},
                 cross={"release": (-4.0, 0.0), "retain": (-7.0, 0.0)},
                 pred_delta={"release": (-5.0, 0.0), "retain": (-8.0, 0.0)},
                 age={"release": 2.0, "retain": 2.0}, shift_delta=(0.0, 0.0))
    return older, newer


class ReciprocalReportDowndateTest(unittest.TestCase):
    def test_removal_changes_only_the_later_reciprocal_prediction(self):
        older, newer = reports()
        before = deepcopy(newer)
        baseline = score_pair(older, newer)
        prior = LocalTrack(8.0, (2.0, 3.0), (2.0, 3.0), (2.0, -1.0))
        posterior = prior.updated(older["stamp"], (25.0, 8.0))
        result = score_removal(older, newer, baseline, prior, posterior)

        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["side"], "release")
        self.assertEqual((result["original_age"], result["omitted_age"]), (2.0, 4.0))
        self.assertEqual(result["omitted_prediction"], (10.0, -1.0))
        self.assertEqual(result["original_prediction"], posterior.position(newer["stamp"], True))
        self.assertGreater(result["omitted_variance"], result["original_variance"])
        self.assertFalse(result["omitted"]["reused_older_report"])
        self.assertEqual(result["omitted"]["sides"]["retain"], baseline["sides"]["retain"])
        self.assertEqual(newer, before)

        dx, dy = result["prediction_delta"]
        reciprocal = result["omitted"]["sides"]["release"]
        self.assertAlmostEqual(reciprocal["actual"], (dx - 1.0) ** 2 + dy**2)
        self.assertAlmostEqual(reciprocal["masked"], (dx - 2.0) ** 2 + dy**2)
        self.assertAlmostEqual(reciprocal["earlier_masked"], (dx - 3.0) ** 2 + dy**2)
        self.assertNotEqual(result["omitted"]["actual"], baseline["earlier_masked"])

    def test_zero_innovation_update_still_reduces_variance(self):
        older, newer = reports()
        prior = LocalTrack(8.0, (2.0, 3.0), (2.0, 3.0), (2.0, -1.0))
        posterior = prior.updated(older["stamp"], (6.0, 1.0))
        baseline = score_pair(older, newer)
        result = score_removal(older, newer, baseline, prior, posterior)

        self.assertEqual(result["prediction_delta"], (0.0, 0.0))
        self.assertEqual(result["omitted"], baseline | {"reused_older_report": False})
        self.assertGreater(result["omitted_variance"], result["original_variance"])

    def test_missing_reset_and_stale_prior_are_unscored(self):
        older, newer = reports()
        baseline = score_pair(older, newer)
        posterior = LocalTrack(older["stamp"], (3.0, 0.0), (3.0, 0.0))
        for reason, earlier_stamp in (("no_prior", None), ("reset", older["stamp"] - MEMORY_S - 1),
                                      ("stale", newer["stamp"] - FRESH_S - 1)):
            with self.subTest(reason=reason):
                prior = None if earlier_stamp is None else LocalTrack(earlier_stamp, (0.0, 0.0), (0.0, 0.0))
                result = score_removal(older, newer, baseline, prior, posterior)
                self.assertEqual(result["status"], reason)
                self.assertIsNone(result["omitted"])
                self.assertIsNone(result["omitted_variance"])

        prior = LocalTrack(newer["stamp"] - FRESH_S, (0.0, 0.0), (0.0, 0.0))
        self.assertEqual(score_removal(older, newer, baseline, prior, posterior)["status"], "supported")

    def test_forecast_must_use_the_selected_older_report(self):
        older, newer = reports()
        baseline = score_pair(older, newer)
        prior = LocalTrack(8.0, (0.0, 0.0), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "latest update"):
            score_removal(older, newer, baseline, prior, prior.updated(11.0, (3.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
