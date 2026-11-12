import unittest

from online_observation.async_report_scores import reported_direction, score_batch
from online_observation.continuous_tracker import FRESH_S, LocalTrack
from online_observation.hypothesis_gap_diagnostic import Switch


A, C = ("left", "1"), ("left", "2")
B, D = ("right", "1"), ("right", "2")
UNRELATED = ("left", "3")
SWITCH = Switch(((A, B), (C, D)), ((A, D), (C, B)))


def track(stamp, east, velocity=0.):
    point = east, 0.
    return LocalTrack(stamp, point, point, (velocity, 0.))


class AsyncReportScoresTest(unittest.TestCase):
    def test_equal_covariances_preserve_report_order_and_mask_is_same_time(self):
        priors = {A: track(8., 10.), B: track(8., 0., 1.), C: track(8., 0.), D: track(8., 8.)}
        scores = score_batch(SWITCH, 10., {A: (2., 0.)}, priors, {})
        row, = scores["reports"]
        self.assertEqual((row["release_age"], row["retain_age"], row["age"]), (2., 2., 2.))
        self.assertEqual((row["release_squared"], row["retain_squared"], row["self_squared"]), (0., 36., 64.))
        self.assertEqual((scores["euclidean_actual"], scores["euclidean_masked"]), (36., -60.))
        self.assertEqual((scores["position_actual"], scores["position_masked"]), (-24., -120.))
        self.assertEqual(scores["log_volume"], 0.)
        for arm in ("common", "mahalanobis", "likelihood"):
            self.assertEqual(reported_direction(scores[f"{arm}_actual"]), 1)
            self.assertEqual(reported_direction(scores[f"{arm}_masked"]), -1)

    def test_second_report_does_not_replace_its_partner_forecast(self):
        priors = {A: track(8., 0.), B: track(8., 5.), C: track(8., 0.), D: track(8., 15.)}
        shifts = {"left": (5., 0.), "right": (-2., 0.)}
        single = score_batch(SWITCH, 10., {A: (-2., 0.)}, priors, shifts)
        pair = score_batch(SWITCH, 10., {A: (-2., 0.), B: (500., 0.)}, priors, shifts)
        first, second = pair["reports"]
        self.assertEqual(first, single["reports"][0])
        self.assertEqual((first["release_squared"], first["retain_squared"]), (0., 100.))
        self.assertEqual(second["release_key"], A)
        self.assertEqual(second["release_squared"], (498. - 5.)**2)
        self.assertEqual(pair["euclidean_actual"], 50.)

    def test_likelihood_penalizes_uncertainty_even_when_residuals_are_zero(self):
        priors = {A: track(8., 0.), B: track(8., 0.), C: track(8., 0.), D: track(1., 0.)}
        scores = score_batch(SWITCH, 10., {A: (0., 0.)}, priors, {})
        row, = scores["reports"]
        self.assertGreater(row["retain_variance"], row["release_variance"])
        self.assertEqual(scores["mahalanobis_actual"], 0.)
        self.assertGreater(scores["likelihood_actual"], 0.)
        self.assertEqual(scores["likelihood_actual"], scores["likelihood_masked"])

    def test_quiet_missing_stale_and_same_or_future_prior_have_no_score(self):
        priors = {key: track(10. - FRESH_S, 0.) for key in SWITCH.keys}
        self.assertIsNotNone(score_batch(SWITCH, 10., {A: (0., 0.)}, priors, {}))
        self.assertIsNone(score_batch(SWITCH, 10., {UNRELATED: (0., 0.)}, priors, {}))
        self.assertIsNone(score_batch(SWITCH, 10., {A: (0., 0.)}, {A: priors[A]}, {}))
        for stamp in (9. - FRESH_S, 10., 11.):
            with self.subTest(stamp=stamp):
                self.assertIsNone(score_batch(SWITCH, 10., {A: (0., 0.)}, priors | {D: track(stamp, 0.)}, {}))


if __name__ == "__main__":
    unittest.main()
